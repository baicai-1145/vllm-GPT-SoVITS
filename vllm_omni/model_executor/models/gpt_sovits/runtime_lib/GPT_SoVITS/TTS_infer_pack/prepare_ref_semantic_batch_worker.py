import asyncio
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import torch
import torchaudio

from TTS_infer_pack.prepare_gpu_timeline import sync_timeline_cuda, trace_gpu_batch


REF_AUDIO_MIN_SAMPLES_16K = 48000
REF_AUDIO_MAX_SAMPLES_16K = 160000
DEFAULT_REF_BATCH_BUCKETS = (
    56000,
    64000,
    72000,
    80000,
    88000,
    96000,
    104000,
    112000,
    120000,
    128000,
    136000,
    144000,
    152000,
    160000,
    169600,
)
_RESAMPLE_CACHE_LOCK = threading.Lock()
_RESAMPLE_CACHE: Dict[Tuple[int, int, str], torchaudio.transforms.Resample] = {}
_RESAMPLE_STREAM_CACHE: Dict[str, torch.cuda.Stream] = {}


def _get_resampler(orig_sr: int, target_sr: int, device: str) -> torchaudio.transforms.Resample:
    device_key = str(device)
    key = (int(orig_sr), int(target_sr), device_key)
    with _RESAMPLE_CACHE_LOCK:
        transform = _RESAMPLE_CACHE.get(key)
        if transform is None:
            transform = torchaudio.transforms.Resample(orig_freq=int(orig_sr), new_freq=int(target_sr)).to(device_key)
            _RESAMPLE_CACHE[key] = transform
    return transform


def _get_resample_stream(device: str) -> torch.cuda.Stream:
    device_key = str(device)
    with _RESAMPLE_CACHE_LOCK:
        stream = _RESAMPLE_STREAM_CACHE.get(device_key)
        if stream is None:
            stream = torch.cuda.Stream(device=device_key)
            _RESAMPLE_STREAM_CACHE[device_key] = stream
    return stream


def prepare_prompt_semantic_wav16k(raw_audio: torch.Tensor, raw_sr: int, zero_wav_samples: int) -> torch.Tensor:
    resample_device = os.environ.get("GPTSOVITS_PREPARE_REF_RESAMPLE_DEVICE", "cpu").strip().lower() or "cpu"
    if resample_device not in {"cpu", "cuda"}:
        resample_device = "cpu"
    if resample_device == "cuda" and not torch.cuda.is_available():
        resample_device = "cpu"
    wav_mono = raw_audio
    if wav_mono.dim() == 2 and wav_mono.shape[0] != 1:
        wav_mono = wav_mono.mean(0, keepdim=True)
    if resample_device == "cuda":
        stream = _get_resample_stream(resample_device)
        with torch.cuda.stream(stream):
            wav16k = wav_mono.to(dtype=torch.float32, device=resample_device)
            if raw_sr != 16000:
                wav16k = _get_resampler(int(raw_sr), 16000, resample_device)(wav16k)
            wav16k = wav16k.squeeze(0).contiguous()
        stream.synchronize()
        wav16k = wav16k.detach().to(device="cpu", dtype=torch.float32).contiguous()
    else:
        wav16k = wav_mono.to(dtype=torch.float32, device=resample_device)
        if raw_sr != 16000:
            wav16k = _get_resampler(int(raw_sr), 16000, resample_device)(wav16k)
        wav16k = wav16k.squeeze(0).contiguous()
    if wav16k.shape[0] > REF_AUDIO_MAX_SAMPLES_16K or wav16k.shape[0] < REF_AUDIO_MIN_SAMPLES_16K:
        raise OSError("参考音频在3~10秒范围外，请更换！")
    if zero_wav_samples > 0:
        wav16k = torch.cat(
            [wav16k, torch.zeros(int(zero_wav_samples), dtype=torch.float32, device=wav16k.device)],
            dim=0,
        )
    return wav16k.contiguous()


def conv1d_output_lengths(input_lengths: torch.Tensor, conv1d: torch.nn.Conv1d | None) -> torch.Tensor:
    if conv1d is None:
        return input_lengths.to(dtype=torch.long)
    kernel_size = int(conv1d.kernel_size[0])
    stride = int(conv1d.stride[0])
    padding = int(conv1d.padding[0])
    dilation = int(conv1d.dilation[0])
    output_lengths = torch.div(
        input_lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
        stride,
        rounding_mode="floor",
    ) + 1
    return torch.clamp(output_lengths, min=0).to(dtype=torch.long)


def parse_ref_batch_buckets(raw: str | None) -> List[int]:
    values: List[int] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(max(REF_AUDIO_MIN_SAMPLES_16K, int(item)))
    if not values:
        values = list(DEFAULT_REF_BATCH_BUCKETS)
    values = sorted(set(values))
    return values


def _clamp_pad_ratio(raw: str | None, default: float) -> float:
    try:
        value = float(str(raw if raw is not None else default).strip())
    except Exception:
        value = float(default)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


@dataclass
class RefSemanticTask:
    raw_audio: torch.Tensor
    raw_sr: int
    wav16k: torch.Tensor | None = None
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.perf_counter)
    batch_popped_at: float = 0.0
    done_event: threading.Event = field(default_factory=threading.Event)
    done_loop: asyncio.AbstractEventLoop | None = None
    done_future: asyncio.Future | None = None
    result_prompt_semantic: torch.Tensor | None = None
    error: Exception | None = None
    profile: Dict[str, float] = field(default_factory=dict)


class PrepareRefSemanticBatchWorker:
    def __init__(
        self,
        ssl_model,
        vits_model,
        device,
        is_half: bool,
        zero_wav_samples: int,
        stage_limiter=None,
        batch_window_ms: int = 5,
        max_batch_items: int = 8,
        max_batch_samples: int = 960000,
        shard_index: int = 0,
    ):
        self.ssl_model = ssl_model
        self.vits_model = vits_model
        self.device = device
        self.is_half = bool(is_half)
        self.zero_wav_samples = max(0, int(zero_wav_samples))
        self.stage_limiter = stage_limiter
        self.batch_window_s = max(0.0, float(batch_window_ms) / 1000.0)
        self.max_batch_items = max(1, int(max_batch_items))
        self.max_batch_samples = max(REF_AUDIO_MIN_SAMPLES_16K + self.zero_wav_samples, int(max_batch_samples))
        self.shard_index = int(shard_index)
        model_config = getattr(getattr(self.ssl_model, "model", None), "config", None)
        feat_extract_norm = str(getattr(model_config, "feat_extract_norm", "") or "").strip().lower()
        self.skip_attention_mask = (
            str(
                os.environ.get(
                    "GPTSOVITS_PREPARE_REF_SSL_SKIP_ATTENTION_MASK",
                    "1" if feat_extract_norm == "group" else "0",
                )
            )
            .strip()
            .lower()
            not in {"0", "false", "no", "off"}
        )
        self.use_pinned_h2d = (
            str(os.environ.get("GPTSOVITS_PREPARE_REF_PINNED_H2D", "1")).strip().lower()
            not in {"0", "false", "no", "off"}
            and str(self.device) != "cpu"
            and torch.cuda.is_available()
        )
        self.bucket_upper_bounds = parse_ref_batch_buckets(os.environ.get("GPTSOVITS_PREPARE_REF_BATCH_BUCKETS"))
        self.bucket_merge_distance = max(
            0, int(os.environ.get("GPTSOVITS_PREPARE_REF_BATCH_BUCKET_MERGE_DISTANCE", "0"))
        )
        self.max_pad_ratio = _clamp_pad_ratio(
            os.environ.get("GPTSOVITS_PREPARE_REF_BATCH_MAX_PAD_RATIO"),
            0.20,
        )
        self.pad_to_bucket_upper_bound = (
            str(os.environ.get("GPTSOVITS_PREPARE_REF_PAD_TO_BUCKET_UPPER_BOUND", "0")).strip().lower()
            not in {"0", "false", "no", "off"}
        )

        self.condition = threading.Condition()
        self.pending_tasks_by_bucket: Dict[int, Deque[RefSemanticTask]] = {
            bucket_index: deque() for bucket_index in range(len(self.bucket_upper_bounds))
        }
        self.pending_peak = 0
        self.total_submitted = 0
        self.total_finished = 0
        self.total_batches = 0
        self.active_batch_size = 0
        self.active_batch_anchor_bucket_index = -1
        self.active_batch_peak = 0
        self.active_batch_samples = 0
        self.active_batch_samples_peak = 0
        self.worker_thread = threading.Thread(
            target=self._run_loop,
            name=f"prepare-ref-semantic-batch-worker-{self.shard_index}",
            daemon=True,
        )
        self.worker_thread.start()

    def _pending_count_locked(self) -> int:
        return int(sum(len(queue) for queue in self.pending_tasks_by_bucket.values()))

    def _pending_iter_locked(self):
        for queue in self.pending_tasks_by_bucket.values():
            for task in queue:
                yield task

    def _bucket_index_for_samples(self, sample_count: int) -> int:
        target = max(REF_AUDIO_MIN_SAMPLES_16K, int(sample_count))
        for index, upper_bound in enumerate(self.bucket_upper_bounds):
            if target <= upper_bound:
                return index
        return max(0, len(self.bucket_upper_bounds) - 1)

    def _bucket_index_for_task(self, task: RefSemanticTask) -> int:
        return self._bucket_index_for_samples(self._estimate_task_samples(task))

    def bucket_index_for_inputs(
        self,
        raw_audio: torch.Tensor,
        raw_sr: int,
        *,
        wav16k: torch.Tensor | None = None,
    ) -> int:
        task = RefSemanticTask(raw_audio=raw_audio, raw_sr=int(raw_sr), wav16k=wav16k)
        return self._bucket_index_for_task(task)

    def estimate_input_samples(
        self,
        raw_audio: torch.Tensor,
        raw_sr: int,
        *,
        wav16k: torch.Tensor | None = None,
    ) -> int:
        task = RefSemanticTask(raw_audio=raw_audio, raw_sr=int(raw_sr), wav16k=wav16k)
        return int(self._estimate_task_samples(task))

    def _oldest_non_empty_bucket_index_locked(self) -> int | None:
        selected_bucket = None
        selected_created_at = None
        for bucket_index, queue in self.pending_tasks_by_bucket.items():
            if not queue:
                continue
            created_at = float(queue[0].created_at)
            if selected_created_at is None or created_at < selected_created_at:
                selected_bucket = bucket_index
                selected_created_at = created_at
        return selected_bucket

    def _candidate_bucket_indices(self, anchor_bucket_index: int) -> List[int]:
        start = max(0, int(anchor_bucket_index) - self.bucket_merge_distance)
        end = min(len(self.bucket_upper_bounds) - 1, int(anchor_bucket_index) + self.bucket_merge_distance)
        return list(range(start, end + 1))

    def pending_count(self) -> int:
        with self.condition:
            return self._pending_count_locked()

    def pending_samples(self) -> int:
        with self.condition:
            return int(sum(self._estimate_task_samples(task) for task in self._pending_iter_locked()))

    def outstanding_count(self) -> int:
        with self.condition:
            return int(self._pending_count_locked() + self.active_batch_size)

    def outstanding_samples(self) -> int:
        with self.condition:
            pending_samples = int(sum(self._estimate_task_samples(task) for task in self._pending_iter_locked()))
            return int(pending_samples + self.active_batch_samples)

    def routing_snapshot_for_bucket(self, bucket_index: int) -> Dict[str, int]:
        with self.condition:
            candidate_bucket_indices = self._candidate_bucket_indices(bucket_index)
            pending_count = self._pending_count_locked()
            pending_samples = int(sum(self._estimate_task_samples(task) for task in self._pending_iter_locked()))
            return {
                "mergeable_pending": int(
                    sum(len(self.pending_tasks_by_bucket[candidate_index]) for candidate_index in candidate_bucket_indices)
                ),
                "exact_pending": int(len(self.pending_tasks_by_bucket.get(bucket_index, ()))),
                "pending": int(pending_count),
                "outstanding": int(pending_count + self.active_batch_size),
                "outstanding_samples": int(pending_samples + self.active_batch_samples),
                "active_batch_size": int(self.active_batch_size),
                "active_mergeable": int(
                    self.active_batch_size > 0
                    and self.active_batch_anchor_bucket_index >= 0
                    and self.active_batch_anchor_bucket_index in candidate_bucket_indices
                ),
                "active_batch_anchor_bucket_index": int(self.active_batch_anchor_bucket_index),
            }

    def _estimate_task_samples(self, task: RefSemanticTask) -> int:
        if task.wav16k is not None:
            return int(task.wav16k.shape[-1])
        raw_len = int(task.raw_audio.shape[-1]) if task.raw_audio.dim() > 0 else 0
        base = int(round(raw_len * 16000.0 / max(1, int(task.raw_sr))))
        return max(REF_AUDIO_MIN_SAMPLES_16K, base) + self.zero_wav_samples

    @staticmethod
    def _estimate_batch_pad_ratio(total_samples: int, max_task_samples: int, batch_size: int) -> float:
        padded_total = int(max_task_samples) * int(batch_size)
        if padded_total <= 0:
            return 0.0
        return max(0.0, 1.0 - (float(total_samples) / float(padded_total)))

    def submit(
        self,
        raw_audio: torch.Tensor,
        raw_sr: int,
        *,
        wav16k: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        task = RefSemanticTask(raw_audio=raw_audio, raw_sr=int(raw_sr), wav16k=wav16k)
        with self.condition:
            bucket_index = self._bucket_index_for_task(task)
            self.pending_tasks_by_bucket[bucket_index].append(task)
            self.total_submitted += 1
            pending_count = self._pending_count_locked()
            if pending_count > self.pending_peak:
                self.pending_peak = pending_count
            self.condition.notify_all()
        task.done_event.wait()
        if task.error is not None:
            raise task.error
        assert task.result_prompt_semantic is not None
        return task.result_prompt_semantic, dict(task.profile)

    def prewarm_shape(
        self,
        raw_audio: torch.Tensor,
        raw_sr: int,
        *,
        wav16k: torch.Tensor | None = None,
        batch_size: int = 1,
    ) -> List[Dict[str, float]]:
        warm_batch_size = max(1, int(batch_size))
        tasks = [
            RefSemanticTask(raw_audio=raw_audio, raw_sr=int(raw_sr), wav16k=wav16k)
            for _ in range(warm_batch_size)
        ]
        with self.condition:
            for task in tasks:
                bucket_index = self._bucket_index_for_task(task)
                self.pending_tasks_by_bucket[bucket_index].append(task)
                self.total_submitted += 1
            pending_count = self._pending_count_locked()
            if pending_count > self.pending_peak:
                self.pending_peak = pending_count
            self.condition.notify_all()
        profiles: List[Dict[str, float]] = []
        for task in tasks:
            task.done_event.wait()
            if task.error is not None:
                raise task.error
            profiles.append(dict(task.profile))
        return profiles

    async def submit_async(
        self,
        raw_audio: torch.Tensor,
        raw_sr: int,
        *,
        wav16k: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loop = asyncio.get_running_loop()
        task = RefSemanticTask(
            raw_audio=raw_audio,
            raw_sr=int(raw_sr),
            wav16k=wav16k,
            done_loop=loop,
            done_future=loop.create_future(),
        )
        with self.condition:
            bucket_index = self._bucket_index_for_task(task)
            self.pending_tasks_by_bucket[bucket_index].append(task)
            self.total_submitted += 1
            pending_count = self._pending_count_locked()
            if pending_count > self.pending_peak:
                self.pending_peak = pending_count
            self.condition.notify_all()
        return await task.done_future

    @staticmethod
    def _resolve_done_future(task: RefSemanticTask) -> None:
        if task.done_future is None or task.done_future.done():
            return
        if task.error is not None:
            task.done_future.set_exception(task.error)
            return
        assert task.result_prompt_semantic is not None
        task.done_future.set_result((task.result_prompt_semantic, dict(task.profile)))

    def _notify_task_done(self, task: RefSemanticTask) -> None:
        task.done_event.set()
        if task.done_loop is None or task.done_future is None:
            return
        try:
            task.done_loop.call_soon_threadsafe(self._resolve_done_future, task)
        except RuntimeError:
            pass

    def snapshot(self) -> Dict[str, int]:
        with self.condition:
            pending_count = self._pending_count_locked()
            pending_samples = int(sum(self._estimate_task_samples(task) for task in self._pending_iter_locked()))
            return {
                "shard_index": self.shard_index,
                "pending": pending_count,
                "pending_peak": self.pending_peak,
                "total_submitted": self.total_submitted,
                "total_finished": self.total_finished,
                "total_batches": self.total_batches,
                "active_batch_size": self.active_batch_size,
                "active_batch_anchor_bucket_index": self.active_batch_anchor_bucket_index,
                "active_batch_peak": self.active_batch_peak,
                "active_batch_samples": self.active_batch_samples,
                "active_batch_samples_peak": self.active_batch_samples_peak,
                "outstanding": pending_count + self.active_batch_size,
                "outstanding_samples": int(pending_samples + self.active_batch_samples),
                "batch_window_ms": int(self.batch_window_s * 1000.0),
                "max_batch_items": self.max_batch_items,
                "max_batch_samples": self.max_batch_samples,
                "bucket_merge_distance": int(self.bucket_merge_distance),
                "max_pad_ratio_pct": int(round(self.max_pad_ratio * 100.0)),
                "bucket_upper_bounds": list(self.bucket_upper_bounds),
                "pending_buckets": {
                    str(self.bucket_upper_bounds[bucket_index]): len(queue)
                    for bucket_index, queue in self.pending_tasks_by_bucket.items()
                },
            }

    def _collect_batch(self) -> tuple[List[RefSemanticTask], float]:
        with self.condition:
            while self._pending_count_locked() <= 0:
                self.condition.wait()

            selected_bucket_index = self._oldest_non_empty_bucket_index_locked()
            if selected_bucket_index is None:
                raise RuntimeError("ref semantic pending bucket missing")
            selected_queue = self.pending_tasks_by_bucket[selected_bucket_index]
            first_task = selected_queue.popleft()
            first_task.batch_popped_at = time.perf_counter()
            batch: List[RefSemanticTask] = [first_task]
            batch_samples = self._estimate_task_samples(batch[0])
            batch_max_task_samples = batch_samples
            deadline = time.perf_counter() + self.batch_window_s

            while len(batch) < self.max_batch_items:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                if not any(self.pending_tasks_by_bucket[bucket_index] for bucket_index in self._candidate_bucket_indices(selected_bucket_index)):
                    self.condition.wait(timeout=remaining)
                    continue

                chosen_bucket_index = None
                chosen_task = None
                chosen_samples = 0
                chosen_padded_cost = None
                chosen_pad_ratio = None
                for bucket_index in self._candidate_bucket_indices(selected_bucket_index):
                    bucket_queue = self.pending_tasks_by_bucket[bucket_index]
                    if not bucket_queue:
                        continue
                    candidate_task = bucket_queue[0]
                    candidate_samples = self._estimate_task_samples(candidate_task)
                    if (batch_samples + candidate_samples) > self.max_batch_samples:
                        continue
                    next_max_task_samples = max(batch_max_task_samples, candidate_samples)
                    next_pad_ratio = self._estimate_batch_pad_ratio(
                        batch_samples + candidate_samples,
                        next_max_task_samples,
                        len(batch) + 1,
                    )
                    if next_pad_ratio > self.max_pad_ratio:
                        continue
                    padded_cost = next_max_task_samples * (len(batch) + 1)
                    if (
                        chosen_padded_cost is None
                        or next_pad_ratio < chosen_pad_ratio
                        or (next_pad_ratio == chosen_pad_ratio and padded_cost < chosen_padded_cost)
                    ):
                        chosen_bucket_index = bucket_index
                        chosen_task = candidate_task
                        chosen_samples = candidate_samples
                        chosen_padded_cost = padded_cost
                        chosen_pad_ratio = next_pad_ratio

                if chosen_task is None or chosen_bucket_index is None:
                    break

                popped_task = self.pending_tasks_by_bucket[chosen_bucket_index].popleft()
                popped_task.batch_popped_at = time.perf_counter()
                batch.append(popped_task)
                batch_samples += chosen_samples
                batch_max_task_samples = max(batch_max_task_samples, chosen_samples)

            self.active_batch_size = len(batch)
            self.active_batch_anchor_bucket_index = int(selected_bucket_index)
            self.active_batch_samples = batch_samples
            if self.active_batch_size > self.active_batch_peak:
                self.active_batch_peak = self.active_batch_size
            if self.active_batch_samples > self.active_batch_samples_peak:
                self.active_batch_samples_peak = self.active_batch_samples
            return batch, time.perf_counter()

    def _finalize_batch(self, batch: List[RefSemanticTask]) -> None:
        with self.condition:
            self.active_batch_size = 0
            self.active_batch_anchor_bucket_index = -1
            self.active_batch_samples = 0
            self.total_batches += 1
            self.total_finished += len(batch)

    def _get_hidden_lengths(self, attention_mask: torch.Tensor, hidden_length: int) -> torch.Tensor:
        model = self.ssl_model.model
        if hasattr(model, "_get_feature_vector_attention_mask"):
            feature_mask = model._get_feature_vector_attention_mask(hidden_length, attention_mask)
            return feature_mask.to(dtype=torch.long).sum(dim=1)
        raw_lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
        if hasattr(model, "_get_feat_extract_output_lengths"):
            return model._get_feat_extract_output_lengths(raw_lengths).to(dtype=torch.long)
        return torch.full((attention_mask.shape[0],), int(hidden_length), dtype=torch.long, device=attention_mask.device)

    def _get_hidden_lengths_from_raw_lengths(self, raw_lengths: torch.Tensor, hidden_length: int, device) -> torch.Tensor:
        model = self.ssl_model.model
        raw_lengths = raw_lengths.to(device=device, dtype=torch.long)
        if hasattr(model, "_get_feat_extract_output_lengths"):
            return model._get_feat_extract_output_lengths(raw_lengths).to(dtype=torch.long)
        return torch.full((raw_lengths.shape[0],), int(hidden_length), dtype=torch.long, device=device)

    @torch.inference_mode()
    def _run_batch(self, batch: List[RefSemanticTask], batch_collected_at: float) -> None:
        batch_started = time.perf_counter()
        prepared_start = time.perf_counter()
        prepared_wavs = [
            task.wav16k
            if task.wav16k is not None
            else prepare_prompt_semantic_wav16k(task.raw_audio, int(task.raw_sr), self.zero_wav_samples)
            for task in batch
        ]
        prepared_end = time.perf_counter()
        cpu_prepare_ms = (time.perf_counter() - prepared_start) * 1000.0
        wav_lengths = torch.tensor([int(wav.shape[0]) for wav in prepared_wavs], dtype=torch.long)
        batch_samples = int(wav_lengths.sum().item())
        max_wav_len = int(wav_lengths.max().item())
        pad_target_len = max_wav_len
        if self.pad_to_bucket_upper_bound:
            bucket_target_len = max(
                int(self.bucket_upper_bounds[self._bucket_index_for_task(task)])
                for task in batch
            )
            pad_target_len = max(pad_target_len, int(bucket_target_len))
        padded_batch_samples = int(len(batch) * pad_target_len)
        batch_pad_ratio = 0.0 if padded_batch_samples <= 0 else max(0.0, 1.0 - (batch_samples / padded_batch_samples))

        pack_start = time.perf_counter()
        pin_memory = bool(self.use_pinned_h2d)
        input_values_cpu = torch.zeros((len(batch), pad_target_len), dtype=torch.float32, pin_memory=pin_memory)
        attention_mask_cpu = None
        if not self.skip_attention_mask:
            attention_mask_cpu = torch.zeros((len(batch), pad_target_len), dtype=torch.long, pin_memory=pin_memory)
        for batch_index, wav in enumerate(prepared_wavs):
            wav_len = int(wav.shape[0])
            input_values_cpu[batch_index, :wav_len] = wav
            if attention_mask_cpu is not None:
                attention_mask_cpu[batch_index, :wav_len] = 1
        pack_end = time.perf_counter()
        pack_ms = (pack_end - pack_start) * 1000.0

        limiter_stats = {"wait_ms": 0.0, "peak_inflight": 1, "slots": 0}
        h2d_ms = 0.0
        ssl_forward_ms = 0.0
        hidden_length_ms = 0.0
        extract_latent_ms = 0.0
        gpu_acquired_ts = pack_end
        h2d_start_ts = pack_end
        h2d_end_ts = pack_end
        ssl_start_ts = pack_end
        ssl_end_ts = pack_end
        latent_start_ts = pack_end
        latent_end_ts = pack_end
        if self.stage_limiter is None:
            h2d_start = time.perf_counter()
            h2d_start_ts = h2d_start
            input_values = input_values_cpu.to(self.device, non_blocking=pin_memory)
            attention_mask = None
            if attention_mask_cpu is not None:
                attention_mask = attention_mask_cpu.to(self.device, non_blocking=pin_memory)
            if self.is_half:
                input_values = input_values.half()
            h2d_end_ts = time.perf_counter()
            h2d_ms = (h2d_end_ts - h2d_start) * 1000.0
            gpu_acquired_ts = h2d_start_ts
            with torch.inference_mode():
                ssl_start = time.perf_counter()
                ssl_start_ts = ssl_start
                if attention_mask is None:
                    outputs = self.ssl_model.model(input_values)
                else:
                    outputs = self.ssl_model.model(input_values, attention_mask=attention_mask)
                sync_timeline_cuda(self.device)
                ssl_end_ts = time.perf_counter()
                ssl_forward_ms = (ssl_end_ts - ssl_start) * 1000.0
                hubert_feature = outputs["last_hidden_state"].transpose(1, 2)
                hidden_length_start = time.perf_counter()
                if attention_mask is None:
                    hidden_lengths = self._get_hidden_lengths_from_raw_lengths(
                        wav_lengths,
                        int(hubert_feature.shape[-1]),
                        hubert_feature.device,
                    )
                else:
                    hidden_lengths = self._get_hidden_lengths(attention_mask, int(hubert_feature.shape[-1]))
                hidden_length_ms = (time.perf_counter() - hidden_length_start) * 1000.0
                latent_start = time.perf_counter()
                latent_start_ts = latent_start
                codes = self.vits_model.extract_latent(hubert_feature)
                sync_timeline_cuda(self.device)
                latent_end_ts = time.perf_counter()
                extract_latent_ms = (latent_end_ts - latent_start) * 1000.0
        else:
            with self.stage_limiter.enter() as limiter_stats:
                gpu_acquired_ts = time.perf_counter()
                h2d_start = gpu_acquired_ts
                h2d_start_ts = h2d_start
                input_values = input_values_cpu.to(self.device, non_blocking=pin_memory)
                attention_mask = None
                if attention_mask_cpu is not None:
                    attention_mask = attention_mask_cpu.to(self.device, non_blocking=pin_memory)
                if self.is_half:
                    input_values = input_values.half()
                h2d_end_ts = time.perf_counter()
                h2d_ms = (h2d_end_ts - h2d_start) * 1000.0
                with torch.inference_mode():
                    ssl_start = time.perf_counter()
                    ssl_start_ts = ssl_start
                    if attention_mask is None:
                        outputs = self.ssl_model.model(input_values)
                    else:
                        outputs = self.ssl_model.model(input_values, attention_mask=attention_mask)
                    sync_timeline_cuda(self.device)
                    ssl_end_ts = time.perf_counter()
                    ssl_forward_ms = (ssl_end_ts - ssl_start) * 1000.0
                    hubert_feature = outputs["last_hidden_state"].transpose(1, 2)
                    hidden_length_start = time.perf_counter()
                    if attention_mask is None:
                        hidden_lengths = self._get_hidden_lengths_from_raw_lengths(
                            wav_lengths,
                            int(hubert_feature.shape[-1]),
                            hubert_feature.device,
                        )
                    else:
                        hidden_lengths = self._get_hidden_lengths(attention_mask, int(hubert_feature.shape[-1]))
                    hidden_length_ms = (time.perf_counter() - hidden_length_start) * 1000.0
                    latent_start = time.perf_counter()
                    latent_start_ts = latent_start
                    codes = self.vits_model.extract_latent(hubert_feature)
                    sync_timeline_cuda(self.device)
                    latent_end_ts = time.perf_counter()
                    extract_latent_ms = (latent_end_ts - latent_start) * 1000.0
        forward_ms = float(h2d_ms + ssl_forward_ms + hidden_length_ms + extract_latent_ms)

        code_lengths = conv1d_output_lengths(hidden_lengths.detach().cpu(), getattr(self.vits_model, "ssl_proj", None))
        sync_timeline_cuda(self.device)
        gpu_active_end_ts = time.perf_counter()
        scatter_start = time.perf_counter()
        for batch_index, task in enumerate(batch):
            try:
                code_len = int(code_lengths[batch_index].item())
                task.result_prompt_semantic = codes[batch_index, 0, :code_len].detach().clone()
                worker_queue_wait_ms = max(0.0, (float(task.batch_popped_at) - float(task.created_at)) * 1000.0)
                batch_collect_wait_ms = max(0.0, (float(batch_collected_at) - float(task.batch_popped_at)) * 1000.0)
                stage_limiter_wait_ms = float(limiter_stats["wait_ms"])
                task.profile = {
                    "prompt_semantic_wait_ms": worker_queue_wait_ms
                    + batch_collect_wait_ms
                    + stage_limiter_wait_ms,
                    "prompt_semantic_shard_index": float(self.shard_index),
                    "prompt_semantic_worker_queue_wait_ms": worker_queue_wait_ms,
                    "prompt_semantic_batch_collect_wait_ms": batch_collect_wait_ms,
                    "prompt_semantic_stage_limiter_wait_ms": stage_limiter_wait_ms,
                    "prompt_semantic_batch_dispatch_delay_ms": max(
                        0.0, (float(batch_started) - float(batch_collected_at)) * 1000.0
                    ),
                    "prompt_semantic_cpu_prepare_ms": float(cpu_prepare_ms),
                    "prompt_semantic_pack_ms": float(pack_ms),
                    "prompt_semantic_h2d_ms": float(h2d_ms),
                    "prompt_semantic_ssl_forward_ms": float(ssl_forward_ms),
                    "prompt_semantic_hidden_length_ms": float(hidden_length_ms),
                    "prompt_semantic_extract_latent_ms": float(extract_latent_ms),
                    "prompt_semantic_forward_ms": float(forward_ms),
                    "prompt_semantic_scatter_ms": 0.0,
                    "prompt_semantic_calls": 1.0,
                    "prompt_semantic_stage_slots": float(limiter_stats["slots"]),
                    "prompt_semantic_stage_inflight_peak": float(limiter_stats["peak_inflight"]),
                    "prompt_semantic_batch_size": float(len(batch)),
                    "prompt_semantic_batch_samples": float(batch_samples),
                    "prompt_semantic_padded_batch_samples": float(padded_batch_samples),
                    "prompt_semantic_batch_pad_target_samples": float(pad_target_len),
                    "prompt_semantic_batch_pad_ratio": float(batch_pad_ratio),
                    "prompt_semantic_ssl_skip_attention_mask": 1.0 if attention_mask_cpu is None else 0.0,
                }
            except Exception as exc:  # noqa: PERF203
                task.error = exc
        scatter_ms = (time.perf_counter() - scatter_start) * 1000.0
        batch_finished_ts = time.perf_counter()
        avg_queue_wait_ms = 0.0
        if batch:
            avg_queue_wait_ms = float(
                sum(max(0.0, (float(task.batch_popped_at) - float(task.created_at)) * 1000.0) for task in batch)
                / len(batch)
            )
        notify_start = time.perf_counter()
        for task in batch:
            if task.result_prompt_semantic is not None:
                task.profile["prompt_semantic_scatter_ms"] = float(scatter_ms)
            self._notify_task_done(task)
        notify_end = time.perf_counter()
        notify_ms = (notify_end - notify_start) * 1000.0
        trace_gpu_batch(
            "ref_gpu_batch",
            stage="ref_semantic",
            shard_index=int(self.shard_index),
            batch_size=int(len(batch)),
            batch_samples=int(batch_samples),
            limiter_wait_ms=float(limiter_stats["wait_ms"]),
            queue_wait_ms=float(avg_queue_wait_ms),
            cpu_prepare_ms=float(cpu_prepare_ms),
            pack_ms=float(pack_ms),
            h2d_ms=float(h2d_ms),
            ssl_forward_ms=float(ssl_forward_ms),
            hidden_length_ms=float(hidden_length_ms),
            extract_latent_ms=float(extract_latent_ms),
            scatter_ms=float(scatter_ms),
            notify_ms=float(notify_ms),
            batch_collected_ts=float(batch_collected_at),
            batch_started_ts=batch_started,
            batch_finished_ts=batch_finished_ts,
            notify_end_ts=notify_end,
            gpu_acquired_ts=gpu_acquired_ts,
            gpu_active_start_ts=h2d_start_ts,
            h2d_end_ts=h2d_end_ts,
            ssl_start_ts=ssl_start_ts,
            ssl_end_ts=ssl_end_ts,
            latent_start_ts=latent_start_ts,
            latent_end_ts=latent_end_ts,
            gpu_active_end_ts=gpu_active_end_ts,
            cpu_prepare_end_ts=prepared_end,
            pack_end_ts=pack_end,
        )

    def _run_loop(self) -> None:
        while True:
            batch, batch_collected_at = self._collect_batch()
            try:
                self._run_batch(batch, batch_collected_at)
            except Exception as exc:  # noqa: PERF203
                for task in batch:
                    task.error = exc
                    self._notify_task_done(task)
            finally:
                self._finalize_batch(batch)


class PrepareRefSemanticBatchWorkerPool:
    def __init__(
        self,
        ssl_model,
        vits_model,
        device,
        is_half: bool,
        zero_wav_samples: int,
        stage_limiter=None,
        batch_window_ms: int = 5,
        max_batch_items: int = 8,
        max_batch_samples: int = 960000,
        worker_count: int = 1,
    ):
        self.worker_count = max(1, int(worker_count))
        self.lock = threading.Lock()
        self.runtime_exact_prewarm_enabled = (
            str(os.environ.get("GPTSOVITS_PREPARE_REF_RUNTIME_EXACT_PREWARM", "0")).strip().lower()
            not in {"0", "false", "no", "off"}
        )
        self.runtime_exact_prewarm_lock = threading.Lock()
        self.runtime_exact_prewarm_max_unique = max(
            0,
            int(os.environ.get("GPTSOVITS_PREPARE_REF_RUNTIME_EXACT_PREWARM_MAX_UNIQUE", "4")),
        )
        self.bucket_first_hit_serialization_enabled = (
            str(os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_SERIALIZE_FIRST_HITS", "1")).strip().lower()
            not in {"0", "false", "no", "off"}
        )
        self.bucket_first_hit_required_hits = max(
            0,
            int(os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_SERIALIZE_FIRST_HITS_REQUIRED", "1")),
        )
        raw_bucket_indices = str(
            os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_SERIALIZE_FIRST_HIT_BUCKET_INDICES", "3,4,9")
        ).strip()
        bucket_first_hit_bucket_indices: set[int] = set()
        for item in raw_bucket_indices.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                bucket_first_hit_bucket_indices.add(max(0, int(item)))
            except Exception:
                continue
        self.bucket_first_hit_bucket_indices = bucket_first_hit_bucket_indices
        self.bucket_first_hit_lock = threading.Lock()
        self.bucket_first_hit_states: Dict[int, Dict[str, int]] = {}
        self.bucket_aware_sharding = (
            str(os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_AWARE_SHARDING", "1")).strip().lower()
            not in {"0", "false", "no", "off"}
        )
        self.bucket_aware_max_outstanding_gap = max(
            0,
            int(os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_AWARE_MAX_OUTSTANDING_GAP", "2")),
        )
        self.shards = [
            PrepareRefSemanticBatchWorker(
                ssl_model=ssl_model,
                vits_model=vits_model,
                device=device,
                is_half=is_half,
                zero_wav_samples=zero_wav_samples,
                stage_limiter=stage_limiter,
                batch_window_ms=batch_window_ms,
                max_batch_items=max_batch_items,
                max_batch_samples=max_batch_samples,
                shard_index=index,
            )
            for index in range(self.worker_count)
        ]
        shard0 = self.shards[0]
        max_batch_items = max(1, int(getattr(shard0, "max_batch_items", 1)))
        default_exact_prewarm_batch_sizes = [1]
        if max_batch_items > 2:
            default_exact_prewarm_batch_sizes.append(
                max(2, min(max_batch_items, int(round(float(max_batch_items) * 0.625))))
            )
        raw_batch_sizes = str(
            os.environ.get(
                "GPTSOVITS_PREPARE_REF_RUNTIME_EXACT_PREWARM_BATCH_SIZES",
                ",".join(str(value) for value in default_exact_prewarm_batch_sizes),
            )
        ).strip()
        runtime_exact_prewarm_batch_sizes: List[int] = []
        for item in raw_batch_sizes.split(","):
            item = item.strip()
            if not item:
                continue
            runtime_exact_prewarm_batch_sizes.append(max(1, min(max_batch_items, int(item))))
        self.runtime_exact_prewarm_batch_sizes = sorted(set(runtime_exact_prewarm_batch_sizes)) or [1]
        self.runtime_exact_prewarmed_samples: set[int] = set()
        self.runtime_exact_prewarm_total = 0
        self.runtime_exact_prewarm_total_ms = 0.0
        self.runtime_exact_prewarm_peak_ms = 0.0

    def _bucket_index_for_inputs(
        self,
        raw_audio: torch.Tensor,
        raw_sr: int,
        *,
        wav16k: torch.Tensor | None = None,
    ) -> int:
        return self.shards[0].bucket_index_for_inputs(raw_audio, raw_sr, wav16k=wav16k)

    def _estimate_input_samples(
        self,
        raw_audio: torch.Tensor,
        raw_sr: int,
        *,
        wav16k: torch.Tensor | None = None,
    ) -> int:
        return self.shards[0].estimate_input_samples(raw_audio, raw_sr, wav16k=wav16k)

    def _maybe_runtime_exact_prewarm(
        self,
        raw_audio: torch.Tensor,
        raw_sr: int,
        *,
        wav16k: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        profile = {
            "prompt_semantic_runtime_exact_prewarm_applied": 0.0,
            "prompt_semantic_runtime_exact_prewarm_ms": 0.0,
            "prompt_semantic_runtime_exact_prewarm_target_samples": 0.0,
            "prompt_semantic_runtime_exact_prewarm_batch_sizes": 0.0,
            "prompt_semantic_runtime_exact_prewarm_skipped_capacity": 0.0,
        }
        if not self.runtime_exact_prewarm_enabled or self.runtime_exact_prewarm_max_unique <= 0:
            return profile
        target_samples = int(self._estimate_input_samples(raw_audio, raw_sr, wav16k=wav16k))
        if target_samples <= 0:
            return profile
        with self.runtime_exact_prewarm_lock:
            if target_samples in self.runtime_exact_prewarmed_samples:
                return profile
            if len(self.runtime_exact_prewarmed_samples) >= self.runtime_exact_prewarm_max_unique:
                profile["prompt_semantic_runtime_exact_prewarm_skipped_capacity"] = 1.0
                profile["prompt_semantic_runtime_exact_prewarm_target_samples"] = float(target_samples)
                return profile
            warm_start = time.perf_counter()
            shard = self.shards[0]
            for batch_size in self.runtime_exact_prewarm_batch_sizes:
                shard.prewarm_shape(
                    raw_audio,
                    int(raw_sr),
                    wav16k=wav16k,
                    batch_size=int(batch_size),
                )
            warm_ms = max(0.0, (time.perf_counter() - warm_start) * 1000.0)
            self.runtime_exact_prewarmed_samples.add(int(target_samples))
            self.runtime_exact_prewarm_total += 1
            self.runtime_exact_prewarm_total_ms += float(warm_ms)
            self.runtime_exact_prewarm_peak_ms = max(self.runtime_exact_prewarm_peak_ms, float(warm_ms))
            profile["prompt_semantic_runtime_exact_prewarm_applied"] = 1.0
            profile["prompt_semantic_runtime_exact_prewarm_ms"] = float(warm_ms)
            profile["prompt_semantic_runtime_exact_prewarm_target_samples"] = float(target_samples)
            profile["prompt_semantic_runtime_exact_prewarm_batch_sizes"] = float(
                len(self.runtime_exact_prewarm_batch_sizes)
            )
            return profile

    def _pick_shard(self, bucket_index: int | None = None) -> PrepareRefSemanticBatchWorker:
        with self.lock:
            if self.bucket_aware_sharding and bucket_index is not None and self.worker_count > 1:
                shard_states = [(shard, shard.routing_snapshot_for_bucket(bucket_index)) for shard in self.shards]
                min_outstanding = min(int(state["outstanding"]) for _, state in shard_states)
                preferred_states = [
                    (shard, state)
                    for shard, state in shard_states
                    if int(state["mergeable_pending"]) > 0
                    and int(state["outstanding"]) <= (min_outstanding + self.bucket_aware_max_outstanding_gap)
                ]
                if preferred_states:
                    return min(
                        preferred_states,
                        key=lambda item: (
                            -int(item[1]["mergeable_pending"]),
                            -int(item[1]["exact_pending"]),
                            int(item[1]["outstanding"]),
                            int(item[1]["outstanding_samples"]),
                            0 if int(item[1]["active_mergeable"]) > 0 else 1,
                            item[0].shard_index,
                        ),
                    )[0]
            return min(
                self.shards,
                key=lambda shard: (
                    shard.outstanding_count(),
                    shard.outstanding_samples(),
                    shard.snapshot().get("active_batch_size", 0),
                    shard.shard_index,
                ),
            )

    def _pick_first_hit_serialized_shard(self, bucket_index: int | None) -> PrepareRefSemanticBatchWorker | None:
        if (
            not self.bucket_first_hit_serialization_enabled
            or self.bucket_first_hit_required_hits <= 0
            or bucket_index is None
            or self.worker_count <= 1
            or (self.bucket_first_hit_bucket_indices and int(bucket_index) not in self.bucket_first_hit_bucket_indices)
        ):
            return None
        with self.bucket_first_hit_lock:
            state = self.bucket_first_hit_states.get(int(bucket_index))
            if state is not None and int(state.get("dispatched_hits", 0)) >= self.bucket_first_hit_required_hits:
                return None
            if state is None:
                preferred = min(
                    self.shards,
                    key=lambda shard: (
                        shard.outstanding_count(),
                        shard.outstanding_samples(),
                        shard.snapshot().get("active_batch_size", 0),
                        shard.shard_index,
                    ),
                )
                state = {
                    "reserved_shard_index": int(preferred.shard_index),
                    "dispatched_hits": 0,
                    "completed_hits": 0,
                }
                self.bucket_first_hit_states[int(bucket_index)] = state
            dispatched_hits = int(state.get("dispatched_hits", 0)) + 1
            state["dispatched_hits"] = min(dispatched_hits, int(self.bucket_first_hit_required_hits))
            reserved_shard_index = int(state.get("reserved_shard_index", 0))
        return next(
            (shard for shard in self.shards if int(shard.shard_index) == reserved_shard_index),
            None,
        )

    def _mark_first_hit_bucket_completed(self, bucket_index: int | None) -> None:
        if (
            not self.bucket_first_hit_serialization_enabled
            or self.bucket_first_hit_required_hits <= 0
            or bucket_index is None
        ):
            return
        with self.bucket_first_hit_lock:
            state = self.bucket_first_hit_states.get(int(bucket_index))
            if state is None:
                return
            completed_hits = int(state.get("completed_hits", 0)) + 1
            dispatched_hits = int(state.get("dispatched_hits", 0))
            state["completed_hits"] = min(completed_hits, dispatched_hits, int(self.bucket_first_hit_required_hits))

    def submit(
        self,
        raw_audio: torch.Tensor,
        raw_sr: int,
        *,
        wav16k: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        runtime_exact_prewarm_profile = self._maybe_runtime_exact_prewarm(raw_audio, raw_sr, wav16k=wav16k)
        bucket_index = self._bucket_index_for_inputs(raw_audio, raw_sr, wav16k=wav16k)
        serialized_shard = self._pick_first_hit_serialized_shard(bucket_index)
        shard = serialized_shard or self._pick_shard(bucket_index)
        try:
            result, profile = shard.submit(raw_audio, raw_sr, wav16k=wav16k)
        finally:
            self._mark_first_hit_bucket_completed(bucket_index)
        profile.update(runtime_exact_prewarm_profile)
        profile["prompt_semantic_pool_workers"] = float(self.worker_count)
        profile["prompt_semantic_pool_bucket_index"] = float(bucket_index)
        profile["prompt_semantic_bucket_first_hit_serialized"] = 1.0 if serialized_shard is not None else 0.0
        return result, profile

    def prewarm(
        self,
        plans: List[Dict[str, torch.Tensor | int | float]],
        rounds: int = 1,
    ) -> List[Dict[str, float]]:
        warm_rounds = max(1, int(rounds))
        profiles: List[Dict[str, float]] = []
        for warm_round in range(warm_rounds):
            for shard in self.shards:
                for plan_index, plan in enumerate(plans):
                    plan_profiles = shard.prewarm_shape(
                        plan["raw_audio"],
                        int(plan["raw_sr"]),
                        wav16k=plan.get("wav16k"),
                        batch_size=int(plan.get("batch_size", 1)),
                    )
                    for profile in plan_profiles:
                        shard_profile = dict(profile)
                        shard_profile["prompt_semantic_prewarm_round"] = float(warm_round)
                        shard_profile["prompt_semantic_prewarm_plan_index"] = float(plan_index)
                        shard_profile["prompt_semantic_prewarm_target_samples"] = float(
                            int(plan.get("target_samples", 0))
                        )
                        shard_profile["prompt_semantic_prewarm_target_batch_size"] = float(
                            int(plan.get("batch_size", 1))
                        )
                        profiles.append(shard_profile)
        return profiles

    async def submit_async(
        self,
        raw_audio: torch.Tensor,
        raw_sr: int,
        *,
        wav16k: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        runtime_exact_prewarm_profile = self._maybe_runtime_exact_prewarm(raw_audio, raw_sr, wav16k=wav16k)
        bucket_index = self._bucket_index_for_inputs(raw_audio, raw_sr, wav16k=wav16k)
        serialized_shard = self._pick_first_hit_serialized_shard(bucket_index)
        shard = serialized_shard or self._pick_shard(bucket_index)
        try:
            result, profile = await shard.submit_async(raw_audio, raw_sr, wav16k=wav16k)
        finally:
            self._mark_first_hit_bucket_completed(bucket_index)
        profile.update(runtime_exact_prewarm_profile)
        profile["prompt_semantic_pool_workers"] = float(self.worker_count)
        profile["prompt_semantic_pool_bucket_index"] = float(bucket_index)
        profile["prompt_semantic_bucket_first_hit_serialized"] = 1.0 if serialized_shard is not None else 0.0
        return result, profile

    def snapshot(self) -> Dict[str, int | List[Dict[str, int]]]:
        shard_snapshots = [dict(shard.snapshot()) for shard in self.shards]
        return {
            "worker_count": int(self.worker_count),
            "runtime_exact_prewarm_enabled": int(self.runtime_exact_prewarm_enabled),
            "runtime_exact_prewarm_max_unique": int(self.runtime_exact_prewarm_max_unique),
            "runtime_exact_prewarm_batch_sizes": list(self.runtime_exact_prewarm_batch_sizes),
            "runtime_exact_prewarm_unique_count": int(len(self.runtime_exact_prewarmed_samples)),
            "runtime_exact_prewarm_total": int(self.runtime_exact_prewarm_total),
            "runtime_exact_prewarm_total_ms": float(self.runtime_exact_prewarm_total_ms),
            "runtime_exact_prewarm_peak_ms": float(self.runtime_exact_prewarm_peak_ms),
            "bucket_first_hit_serialization_enabled": int(self.bucket_first_hit_serialization_enabled),
            "bucket_first_hit_required_hits": int(self.bucket_first_hit_required_hits),
            "bucket_first_hit_bucket_indices": sorted(int(value) for value in self.bucket_first_hit_bucket_indices),
            "bucket_first_hit_tracked_buckets": {
                str(bucket_index): dict(state) for bucket_index, state in self.bucket_first_hit_states.items()
            },
            "bucket_aware_sharding": int(self.bucket_aware_sharding),
            "bucket_aware_max_outstanding_gap": int(self.bucket_aware_max_outstanding_gap),
            "pending": int(sum(int(snapshot.get("pending", 0)) for snapshot in shard_snapshots)),
            "pending_peak": int(max((int(snapshot.get("pending_peak", 0)) for snapshot in shard_snapshots), default=0)),
            "outstanding": int(sum(int(snapshot.get("outstanding", 0)) for snapshot in shard_snapshots)),
            "outstanding_samples": int(sum(int(snapshot.get("outstanding_samples", 0)) for snapshot in shard_snapshots)),
            "total_submitted": int(sum(int(snapshot.get("total_submitted", 0)) for snapshot in shard_snapshots)),
            "total_finished": int(sum(int(snapshot.get("total_finished", 0)) for snapshot in shard_snapshots)),
            "total_batches": int(sum(int(snapshot.get("total_batches", 0)) for snapshot in shard_snapshots)),
            "active_batch_size": int(sum(int(snapshot.get("active_batch_size", 0)) for snapshot in shard_snapshots)),
            "active_batch_peak": int(
                max((int(snapshot.get("active_batch_peak", 0)) for snapshot in shard_snapshots), default=0)
            ),
            "active_batch_samples": int(sum(int(snapshot.get("active_batch_samples", 0)) for snapshot in shard_snapshots)),
            "active_batch_samples_peak": int(
                max((int(snapshot.get("active_batch_samples_peak", 0)) for snapshot in shard_snapshots), default=0)
            ),
            "batch_window_ms": int(shard_snapshots[0].get("batch_window_ms", 0)) if shard_snapshots else 0,
            "max_batch_items": int(shard_snapshots[0].get("max_batch_items", 0)) if shard_snapshots else 0,
            "max_batch_samples": int(shard_snapshots[0].get("max_batch_samples", 0)) if shard_snapshots else 0,
            "shards": shard_snapshots,
        }
