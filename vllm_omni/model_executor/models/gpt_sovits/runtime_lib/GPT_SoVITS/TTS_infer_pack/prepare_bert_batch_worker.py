import asyncio
import threading
import time
import uuid
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from TTS_infer_pack.prepare_gpu_timeline import sync_timeline_cuda, trace_gpu_batch


@dataclass
class BertFeatureTask:
    norm_text: str
    word2ph: list[int]
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.perf_counter)
    enqueued_at: float = 0.0
    admission_wait_ms: float = 0.0
    pending_depth_on_enqueue: int = 0
    batch_popped_at: float = 0.0
    done_event: threading.Event = field(default_factory=threading.Event)
    done_loop: asyncio.AbstractEventLoop | None = None
    done_future: asyncio.Future | None = None
    result_feature: torch.Tensor | None = None
    error: Exception | None = None
    profile: dict[str, float] = field(default_factory=dict)


class PrepareBertBatchWorker:
    def __init__(
        self,
        bert_model,
        tokenizer,
        device,
        stage_limiter=None,
        batch_window_ms: int = 5,
        max_batch_items: int = 16,
        max_batch_tokens: int = 4096,
        max_pending_tasks: int = 0,
        admission_poll_ms: int = 1,
        high_pressure_pending_threshold: int = 0,
        high_pressure_batch_window_ms: int | None = None,
        high_pressure_max_batch_items: int | None = None,
        high_pressure_max_batch_tokens: int | None = None,
        shard_index: int = 0,
    ):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.stage_limiter = stage_limiter
        self.batch_window_ms = max(0, int(batch_window_ms))
        self.batch_window_s = float(self.batch_window_ms) / 1000.0
        self.max_batch_items = max(1, int(max_batch_items))
        self.max_batch_tokens = max(16, int(max_batch_tokens))
        self.max_pending_tasks = max(0, int(max_pending_tasks))
        self.admission_poll_s = max(0.0005, float(max(1, int(admission_poll_ms))) / 1000.0)

        self.high_pressure_pending_threshold = max(
            0,
            int(high_pressure_pending_threshold)
            if int(high_pressure_pending_threshold) > 0
            else max(self.max_batch_items * 2, 32),
        )
        hp_window_ms = (
            self.batch_window_ms if high_pressure_batch_window_ms is None else int(high_pressure_batch_window_ms)
        )
        hp_items = self.max_batch_items if high_pressure_max_batch_items is None else int(high_pressure_max_batch_items)
        hp_tokens = (
            self.max_batch_tokens if high_pressure_max_batch_tokens is None else int(high_pressure_max_batch_tokens)
        )
        self.high_pressure_batch_window_ms = max(0, hp_window_ms)
        self.high_pressure_batch_window_s = float(self.high_pressure_batch_window_ms) / 1000.0
        self.high_pressure_max_batch_items = max(self.max_batch_items, hp_items)
        self.high_pressure_max_batch_tokens = max(self.max_batch_tokens, hp_tokens)
        self.shard_index = int(shard_index)

        self.condition = threading.Condition()
        self.pending_tasks: deque[BertFeatureTask] = deque()
        self.pending_peak = 0
        self.total_submitted = 0
        self.total_finished = 0
        self.total_batches = 0
        self.active_batch_size = 0
        self.active_batch_peak = 0
        self.active_batch_tokens = 0
        self.active_batch_tokens_peak = 0
        self.high_pressure_batches = 0
        self.admission_wait_total_ms = 0.0
        self.admission_wait_peak_ms = 0.0
        self.worker_thread = threading.Thread(
            target=self._run_loop,
            name=f"prepare-bert-batch-worker-{self.shard_index}",
            daemon=True,
        )
        self.worker_thread.start()

    def _estimate_task_tokens(self, task: BertFeatureTask) -> int:
        return max(1, len(task.norm_text) + 2)

    def pending_count(self) -> int:
        with self.condition:
            return int(len(self.pending_tasks))

    def pending_tokens(self) -> int:
        with self.condition:
            return int(sum(self._estimate_task_tokens(task) for task in self.pending_tasks))

    def outstanding_count(self) -> int:
        with self.condition:
            return int(len(self.pending_tasks) + self.active_batch_size)

    def outstanding_tokens(self) -> int:
        with self.condition:
            pending_tokens = int(sum(self._estimate_task_tokens(task) for task in self.pending_tasks))
            return int(pending_tokens + self.active_batch_tokens)

    def _can_enqueue_locked(self) -> bool:
        if self.max_pending_tasks <= 0:
            return True
        return (len(self.pending_tasks) + self.active_batch_size) < self.max_pending_tasks

    def _record_enqueue_locked(self, task: BertFeatureTask, admission_wait_ms: float) -> None:
        task.admission_wait_ms = float(max(0.0, admission_wait_ms))
        task.enqueued_at = time.perf_counter()
        task.pending_depth_on_enqueue = int(len(self.pending_tasks))
        self.pending_tasks.append(task)
        self.total_submitted += 1
        self.admission_wait_total_ms += task.admission_wait_ms
        self.admission_wait_peak_ms = max(self.admission_wait_peak_ms, task.admission_wait_ms)
        if len(self.pending_tasks) > self.pending_peak:
            self.pending_peak = len(self.pending_tasks)
        self.condition.notify_all()

    def _record_enqueue_many_locked(self, tasks: Sequence[BertFeatureTask], admission_started: float) -> None:
        if not tasks:
            return
        enqueued_at = time.perf_counter()
        pending_depth = int(len(self.pending_tasks))
        peak_pending = pending_depth
        for task in tasks:
            task.admission_wait_ms = float(max(0.0, (enqueued_at - admission_started) * 1000.0))
            task.enqueued_at = enqueued_at
            task.pending_depth_on_enqueue = pending_depth
            self.pending_tasks.append(task)
            pending_depth += 1
            self.total_submitted += 1
            self.admission_wait_total_ms += task.admission_wait_ms
            self.admission_wait_peak_ms = max(self.admission_wait_peak_ms, task.admission_wait_ms)
            peak_pending = max(peak_pending, pending_depth)
        self.pending_peak = max(self.pending_peak, peak_pending)
        self.condition.notify_all()

    def _enqueue_task(self, task: BertFeatureTask) -> None:
        admission_started = time.perf_counter()
        with self.condition:
            while not self._can_enqueue_locked():
                self.condition.wait(timeout=self.admission_poll_s)
            self._record_enqueue_locked(task, (time.perf_counter() - admission_started) * 1000.0)

    async def _enqueue_task_async(self, task: BertFeatureTask) -> None:
        admission_started = time.perf_counter()
        while True:
            with self.condition:
                if self._can_enqueue_locked():
                    self._record_enqueue_locked(task, (time.perf_counter() - admission_started) * 1000.0)
                    return
            await asyncio.sleep(self.admission_poll_s)

    def submit(self, norm_text: str, word2ph: list[int]) -> tuple[torch.Tensor, dict[str, float]]:
        task = BertFeatureTask(norm_text=str(norm_text), word2ph=list(word2ph))
        self._enqueue_task(task)
        task.done_event.wait()
        if task.error is not None:
            raise task.error
        assert task.result_feature is not None
        return task.result_feature, dict(task.profile)

    async def submit_async(self, norm_text: str, word2ph: list[int]) -> tuple[torch.Tensor, dict[str, float]]:
        loop = asyncio.get_running_loop()
        task = BertFeatureTask(
            norm_text=str(norm_text),
            word2ph=list(word2ph),
            done_loop=loop,
            done_future=loop.create_future(),
        )
        await self._enqueue_task_async(task)
        return await task.done_future

    async def _enqueue_tasks_async(self, tasks: Sequence[BertFeatureTask]) -> None:
        if not tasks:
            return
        remaining = deque(tasks)
        admission_started = time.perf_counter()
        while remaining:
            with self.condition:
                accepted: list[BertFeatureTask] = []
                while remaining and self._can_enqueue_locked():
                    accepted.append(remaining.popleft())
                if accepted:
                    self._record_enqueue_many_locked(accepted, admission_started)
                if not remaining:
                    return
            await asyncio.sleep(self.admission_poll_s)

    async def submit_many_async(
        self,
        items: Sequence[tuple[str, list[int]]],
    ) -> list[tuple[torch.Tensor, dict[str, float]]]:
        if not items:
            return []
        loop = asyncio.get_running_loop()
        tasks = [
            BertFeatureTask(
                norm_text=str(norm_text),
                word2ph=list(word2ph),
                done_loop=loop,
                done_future=loop.create_future(),
            )
            for norm_text, word2ph in items
        ]
        await self._enqueue_tasks_async(tasks)
        return list(await asyncio.gather(*[task.done_future for task in tasks]))

    def snapshot(self) -> dict[str, int]:
        with self.condition:
            return {
                "shard_index": self.shard_index,
                "pending": len(self.pending_tasks),
                "pending_peak": self.pending_peak,
                "outstanding": len(self.pending_tasks) + self.active_batch_size,
                "total_submitted": self.total_submitted,
                "total_finished": self.total_finished,
                "total_batches": self.total_batches,
                "active_batch_size": self.active_batch_size,
                "active_batch_peak": self.active_batch_peak,
                "active_batch_tokens": self.active_batch_tokens,
                "active_batch_tokens_peak": self.active_batch_tokens_peak,
                "pending_tokens": int(sum(self._estimate_task_tokens(task) for task in self.pending_tasks)),
                "outstanding_tokens": int(
                    sum(self._estimate_task_tokens(task) for task in self.pending_tasks) + self.active_batch_tokens
                ),
                "batch_window_ms": int(self.batch_window_s * 1000.0),
                "max_batch_items": self.max_batch_items,
                "max_batch_tokens": self.max_batch_tokens,
                "max_pending_tasks": self.max_pending_tasks,
                "high_pressure_pending_threshold": self.high_pressure_pending_threshold,
                "high_pressure_batch_window_ms": self.high_pressure_batch_window_ms,
                "high_pressure_max_batch_items": self.high_pressure_max_batch_items,
                "high_pressure_max_batch_tokens": self.high_pressure_max_batch_tokens,
                "high_pressure_batches": self.high_pressure_batches,
                "admission_wait_total_ms": self.admission_wait_total_ms,
                "admission_wait_peak_ms": self.admission_wait_peak_ms,
            }

    def _select_batch_policy_locked(self) -> tuple[float, int, int, bool, int]:
        pending_depth = len(self.pending_tasks)
        use_high_pressure = (
            self.high_pressure_pending_threshold > 0 and pending_depth >= self.high_pressure_pending_threshold
        )
        if use_high_pressure:
            return (
                self.high_pressure_batch_window_s,
                self.high_pressure_max_batch_items,
                self.high_pressure_max_batch_tokens,
                True,
                pending_depth,
            )
        return (
            self.batch_window_s,
            self.max_batch_items,
            self.max_batch_tokens,
            False,
            pending_depth,
        )

    def _collect_batch(self) -> tuple[list[BertFeatureTask], dict[str, float]]:
        with self.condition:
            while not self.pending_tasks:
                self.condition.wait()

            collect_started = time.perf_counter()
            batch_window_s, max_batch_items, max_batch_tokens, use_high_pressure, pending_depth_on_collect = (
                self._select_batch_policy_locked()
            )
            first_task = self.pending_tasks.popleft()
            first_task.batch_popped_at = time.perf_counter()
            batch: list[BertFeatureTask] = [first_task]
            batch_tokens = self._estimate_task_tokens(batch[0])
            deadline = time.perf_counter() + batch_window_s

            while len(batch) < max_batch_items:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                if not self.pending_tasks:
                    self.condition.wait(timeout=remaining)
                    continue
                next_task = self.pending_tasks[0]
                next_tokens = self._estimate_task_tokens(next_task)
                if len(batch) >= max_batch_items or (batch_tokens + next_tokens) > max_batch_tokens:
                    break
                popped_task = self.pending_tasks.popleft()
                popped_task.batch_popped_at = time.perf_counter()
                batch.append(popped_task)
                batch_tokens += next_tokens

            self.active_batch_size = len(batch)
            self.active_batch_tokens = batch_tokens
            if self.active_batch_size > self.active_batch_peak:
                self.active_batch_peak = self.active_batch_size
            if self.active_batch_tokens > self.active_batch_tokens_peak:
                self.active_batch_tokens_peak = self.active_batch_tokens
            if use_high_pressure:
                self.high_pressure_batches += 1
            collected_at = time.perf_counter()
            return batch, {
                "collect_wait_ms": (time.perf_counter() - collect_started) * 1000.0,
                "collected_at_ts": float(collected_at),
                "batch_tokens": float(batch_tokens),
                "pending_depth_on_collect": float(pending_depth_on_collect),
                "high_pressure_mode": 1.0 if use_high_pressure else 0.0,
                "batch_window_ms": float(
                    self.high_pressure_batch_window_ms if use_high_pressure else self.batch_window_ms
                ),
            }

    def _finalize_batch(self, batch: list[BertFeatureTask]) -> None:
        with self.condition:
            self.active_batch_size = 0
            self.active_batch_tokens = 0
            self.total_batches += 1
            self.total_finished += len(batch)
            self.condition.notify_all()

    def _run_batch(self, batch: list[BertFeatureTask], batch_meta: dict[str, float]) -> None:
        batch_started = time.perf_counter()
        texts = [task.norm_text for task in batch]
        batch_tokens = int(batch_meta["batch_tokens"])
        batch_collected_ts = float(batch_meta.get("collected_at_ts", batch_started))
        tokenize_start = time.perf_counter()
        inputs_cpu = self.tokenizer(texts, return_tensors="pt", padding=True)
        tokenize_end_ts = time.perf_counter()
        tokenize_ms = (tokenize_end_ts - tokenize_start) * 1000.0
        attention_mask_cpu = inputs_cpu["attention_mask"]
        gpu_acquired_ts = batch_started
        forward_start_ts = batch_started
        forward_end_ts = batch_started

        limiter_stats = {"wait_ms": 0.0, "peak_inflight": 1, "slots": 0}
        if self.stage_limiter is None:
            gpu_acquired_ts = time.perf_counter()
            inputs = {key: value.to(self.device) for key, value in inputs_cpu.items()}
            forward_start = time.perf_counter()
            forward_start_ts = forward_start
            with torch.inference_mode():
                outputs = self.bert_model(**inputs, output_hidden_states=True)
            sync_timeline_cuda(self.device)
            forward_end_ts = time.perf_counter()
            forward_ms = (forward_end_ts - forward_start) * 1000.0
        else:
            with self.stage_limiter.enter() as limiter_stats:
                gpu_acquired_ts = time.perf_counter()
                inputs = {key: value.to(self.device) for key, value in inputs_cpu.items()}
                forward_start = time.perf_counter()
                forward_start_ts = forward_start
                with torch.inference_mode():
                    outputs = self.bert_model(**inputs, output_hidden_states=True)
                sync_timeline_cuda(self.device)
                forward_end_ts = time.perf_counter()
                forward_ms = (forward_end_ts - forward_start) * 1000.0
        hidden = outputs["hidden_states"][-3].detach().cpu()
        gpu_active_end_ts = time.perf_counter()
        scatter_start = time.perf_counter()
        for batch_index, task in enumerate(batch):
            try:
                text_len = len(task.word2ph)
                if text_len != len(task.norm_text):
                    raise AssertionError(
                        f"word2ph/text length mismatch: task={task.task_id} word2ph={text_len} text={len(task.norm_text)}"
                    )
                seq_len = int(attention_mask_cpu[batch_index].sum().item())
                char_features = hidden[batch_index, 1 : seq_len - 1]
                if char_features.shape[0] != text_len:
                    raise AssertionError(
                        f"bert token length mismatch: task={task.task_id} token_len={char_features.shape[0]} text_len={text_len}"
                    )
                phone_level_feature = []
                for char_index, repeat_count in enumerate(task.word2ph):
                    repeat_count = int(repeat_count)
                    if repeat_count <= 0:
                        continue
                    phone_level_feature.append(char_features[char_index].repeat(repeat_count, 1))
                worker_queue_wait_ms = max(0.0, (float(task.batch_popped_at) - float(task.enqueued_at)) * 1000.0)
                batch_collect_wait_ms = max(0.0, (float(batch_collected_ts) - float(task.batch_popped_at)) * 1000.0)
                batch_dispatch_delay_ms = max(0.0, (float(batch_started) - float(batch_collected_ts)) * 1000.0)
                task.result_feature = (
                    torch.cat(phone_level_feature, dim=0).T
                    if phone_level_feature
                    else char_features.new_zeros((char_features.shape[-1], 0))
                )
                task.profile = {
                    "bert_wait_ms": (batch_started - task.created_at) * 1000.0 + float(limiter_stats["wait_ms"]),
                    "bert_shard_index": float(self.shard_index),
                    "bert_admission_wait_ms": float(task.admission_wait_ms),
                    "bert_queue_wait_ms": max(0.0, (batch_started - task.enqueued_at) * 1000.0),
                    "bert_worker_queue_wait_ms": worker_queue_wait_ms,
                    "bert_batch_collect_wait_ms": batch_collect_wait_ms,
                    "bert_batch_dispatch_delay_ms": batch_dispatch_delay_ms,
                    "bert_forward_ms": float(forward_ms),
                    "bert_tokenize_ms": float(tokenize_ms),
                    "bert_scatter_ms": 0.0,
                    "bert_calls": 1.0,
                    "bert_stage_slots": float(limiter_stats["slots"]),
                    "bert_stage_inflight_peak": float(limiter_stats["peak_inflight"]),
                    "bert_batch_size": float(len(batch)),
                    "bert_batch_tokens": float(batch_tokens),
                    "bert_pending_depth_on_enqueue": float(task.pending_depth_on_enqueue),
                    "bert_pending_depth_on_collect": float(batch_meta["pending_depth_on_collect"]),
                    "bert_high_pressure_mode": float(batch_meta["high_pressure_mode"]),
                    "bert_batch_window_ms": float(batch_meta["batch_window_ms"]),
                }
            except Exception as exc:  # noqa: PERF203
                task.error = exc
        scatter_ms = (time.perf_counter() - scatter_start) * 1000.0
        batch_finished_ts = time.perf_counter()
        avg_queue_wait_ms = 0.0
        avg_worker_queue_wait_ms = 0.0
        avg_batch_collect_wait_ms = 0.0
        avg_batch_dispatch_delay_ms = max(0.0, (float(batch_started) - float(batch_collected_ts)) * 1000.0)
        if batch:
            avg_queue_wait_ms = float(
                sum(max(0.0, (batch_started - task.enqueued_at) * 1000.0) for task in batch) / len(batch)
            )
            avg_worker_queue_wait_ms = float(
                sum(max(0.0, (float(task.batch_popped_at) - float(task.enqueued_at)) * 1000.0) for task in batch)
                / len(batch)
            )
            avg_batch_collect_wait_ms = float(
                sum(max(0.0, (float(batch_collected_ts) - float(task.batch_popped_at)) * 1000.0) for task in batch)
                / len(batch)
            )
        notify_start = time.perf_counter()
        for task in batch:
            if task.result_feature is not None:
                task.profile["bert_scatter_ms"] = float(scatter_ms)
            task.done_event.set()
            self._notify_done_future(task)
        notify_end = time.perf_counter()
        notify_ms = (notify_end - notify_start) * 1000.0
        trace_gpu_batch(
            "bert_gpu_batch",
            stage="bert",
            shard_index=int(self.shard_index),
            batch_size=int(len(batch)),
            batch_tokens=int(batch_tokens),
            pending_depth_on_collect=int(batch_meta["pending_depth_on_collect"]),
            high_pressure_mode=int(batch_meta["high_pressure_mode"]),
            batch_window_ms=float(batch_meta["batch_window_ms"]),
            limiter_wait_ms=float(limiter_stats["wait_ms"]),
            queue_wait_ms=float(avg_queue_wait_ms),
            worker_queue_wait_ms=float(avg_worker_queue_wait_ms),
            batch_collect_wait_ms=float(avg_batch_collect_wait_ms),
            batch_dispatch_delay_ms=float(avg_batch_dispatch_delay_ms),
            tokenize_ms=float(tokenize_ms),
            forward_ms=float(forward_ms),
            scatter_ms=float(scatter_ms),
            notify_ms=float(notify_ms),
            batch_collected_ts=batch_collected_ts,
            batch_started_ts=batch_started,
            batch_finished_ts=batch_finished_ts,
            notify_end_ts=notify_end,
            gpu_acquired_ts=gpu_acquired_ts,
            gpu_active_start_ts=gpu_acquired_ts,
            tokenize_end_ts=tokenize_end_ts,
            forward_start_ts=forward_start_ts,
            forward_end_ts=forward_end_ts,
            gpu_active_end_ts=gpu_active_end_ts,
        )

    @staticmethod
    def _resolve_done_future(task: BertFeatureTask) -> None:
        if task.done_future is None or task.done_future.done():
            return
        if task.error is not None:
            task.done_future.set_exception(task.error)
            return
        assert task.result_feature is not None
        task.done_future.set_result((task.result_feature, dict(task.profile)))

    def _notify_done_future(self, task: BertFeatureTask) -> None:
        if task.done_loop is None or task.done_future is None:
            return
        try:
            task.done_loop.call_soon_threadsafe(self._resolve_done_future, task)
        except RuntimeError:
            pass

    def _run_loop(self) -> None:
        while True:
            batch, batch_meta = self._collect_batch()
            try:
                self._run_batch(batch, batch_meta)
            except Exception as exc:  # noqa: PERF203
                for task in batch:
                    task.error = exc
                    task.done_event.set()
                    self._notify_done_future(task)
            finally:
                self._finalize_batch(batch)


class PrepareBertBatchWorkerPool:
    def __init__(
        self,
        bert_model,
        tokenizer,
        device,
        stage_limiter=None,
        batch_window_ms: int = 5,
        max_batch_items: int = 16,
        max_batch_tokens: int = 4096,
        max_pending_tasks: int = 0,
        admission_poll_ms: int = 1,
        high_pressure_pending_threshold: int = 0,
        high_pressure_batch_window_ms: int | None = None,
        high_pressure_max_batch_items: int | None = None,
        high_pressure_max_batch_tokens: int | None = None,
        worker_count: int = 1,
    ):
        self.worker_count = max(1, int(worker_count))
        self.lock = threading.Lock()
        self.shards = [
            PrepareBertBatchWorker(
                bert_model=bert_model,
                tokenizer=tokenizer,
                device=device,
                stage_limiter=stage_limiter,
                batch_window_ms=batch_window_ms,
                max_batch_items=max_batch_items,
                max_batch_tokens=max_batch_tokens,
                max_pending_tasks=max_pending_tasks,
                admission_poll_ms=admission_poll_ms,
                high_pressure_pending_threshold=high_pressure_pending_threshold,
                high_pressure_batch_window_ms=high_pressure_batch_window_ms,
                high_pressure_max_batch_items=high_pressure_max_batch_items,
                high_pressure_max_batch_tokens=high_pressure_max_batch_tokens,
                shard_index=index,
            )
            for index in range(self.worker_count)
        ]

    def _pick_shard(self) -> PrepareBertBatchWorker:
        with self.lock:
            return min(
                self.shards,
                key=lambda shard: (
                    shard.outstanding_count(),
                    shard.outstanding_tokens(),
                    shard.snapshot().get("active_batch_size", 0),
                    shard.shard_index,
                ),
            )

    def submit(self, norm_text: str, word2ph: list[int]) -> tuple[torch.Tensor, dict[str, float]]:
        shard = self._pick_shard()
        result, profile = shard.submit(norm_text, word2ph)
        profile["bert_pool_workers"] = float(self.worker_count)
        return result, profile

    async def submit_async(self, norm_text: str, word2ph: list[int]) -> tuple[torch.Tensor, dict[str, float]]:
        shard = self._pick_shard()
        result, profile = await shard.submit_async(norm_text, word2ph)
        profile["bert_pool_workers"] = float(self.worker_count)
        return result, profile

    async def submit_many_async(
        self,
        items: Sequence[tuple[str, list[int]]],
    ) -> list[tuple[torch.Tensor, dict[str, float]]]:
        if not items:
            return []
        if self.worker_count == 1:
            results = await self.shards[0].submit_many_async(items)
            for _result, profile in results:
                profile["bert_pool_workers"] = float(self.worker_count)
            return results

        shard_items: list[list[tuple[int, str, list[int]]]] = [[] for _ in range(self.worker_count)]
        shard_token_loads: list[int] = [0] * self.worker_count
        for item_index, (norm_text, word2ph) in enumerate(items):
            with self.lock:
                shard_scores = [
                    (
                        shard.outstanding_count(),
                        shard.outstanding_tokens() + shard_token_loads[shard_index],
                        shard_index,
                    )
                    for shard_index, shard in enumerate(self.shards)
                ]
            shard_index = min(shard_scores)[2]
            shard_items[shard_index].append((item_index, str(norm_text), list(word2ph)))
            shard_token_loads[shard_index] += max(1, len(str(norm_text)) + 2)

        shard_tasks = []
        shard_metas: list[list[tuple[int, str, list[int]]]] = []
        for shard_index, batch in enumerate(shard_items):
            if not batch:
                continue
            shard_metas.append(batch)
            shard_tasks.append(
                self.shards[shard_index].submit_many_async([(norm_text, word2ph) for _, norm_text, word2ph in batch])
            )

        gathered = await asyncio.gather(*shard_tasks)
        ordered_results: list[tuple[torch.Tensor, dict[str, float]] | None] = [None] * len(items)
        for batch_meta, batch_results in zip(shard_metas, gathered):
            for (item_index, _norm_text, _word2ph), (result, profile) in zip(batch_meta, batch_results):
                profile["bert_pool_workers"] = float(self.worker_count)
                ordered_results[item_index] = (result, profile)
        return [item for item in ordered_results if item is not None]

    def snapshot(self) -> dict[str, int | list[dict[str, int]]]:
        shard_snapshots = [dict(shard.snapshot()) for shard in self.shards]
        return {
            "worker_count": int(self.worker_count),
            "pending": int(sum(int(snapshot.get("pending", 0)) for snapshot in shard_snapshots)),
            "pending_peak": int(max((int(snapshot.get("pending_peak", 0)) for snapshot in shard_snapshots), default=0)),
            "pending_tokens": int(sum(int(snapshot.get("pending_tokens", 0)) for snapshot in shard_snapshots)),
            "outstanding": int(sum(int(snapshot.get("outstanding", 0)) for snapshot in shard_snapshots)),
            "outstanding_tokens": int(sum(int(snapshot.get("outstanding_tokens", 0)) for snapshot in shard_snapshots)),
            "total_submitted": int(sum(int(snapshot.get("total_submitted", 0)) for snapshot in shard_snapshots)),
            "total_finished": int(sum(int(snapshot.get("total_finished", 0)) for snapshot in shard_snapshots)),
            "total_batches": int(sum(int(snapshot.get("total_batches", 0)) for snapshot in shard_snapshots)),
            "active_batch_size": int(sum(int(snapshot.get("active_batch_size", 0)) for snapshot in shard_snapshots)),
            "active_batch_peak": int(
                max((int(snapshot.get("active_batch_peak", 0)) for snapshot in shard_snapshots), default=0)
            ),
            "active_batch_tokens": int(
                sum(int(snapshot.get("active_batch_tokens", 0)) for snapshot in shard_snapshots)
            ),
            "active_batch_tokens_peak": int(
                max((int(snapshot.get("active_batch_tokens_peak", 0)) for snapshot in shard_snapshots), default=0)
            ),
            "batch_window_ms": int(shard_snapshots[0].get("batch_window_ms", 0)) if shard_snapshots else 0,
            "max_batch_items": int(shard_snapshots[0].get("max_batch_items", 0)) if shard_snapshots else 0,
            "max_batch_tokens": int(shard_snapshots[0].get("max_batch_tokens", 0)) if shard_snapshots else 0,
            "max_pending_tasks": int(shard_snapshots[0].get("max_pending_tasks", 0)) if shard_snapshots else 0,
            "high_pressure_pending_threshold": (
                int(shard_snapshots[0].get("high_pressure_pending_threshold", 0)) if shard_snapshots else 0
            ),
            "high_pressure_batch_window_ms": (
                int(shard_snapshots[0].get("high_pressure_batch_window_ms", 0)) if shard_snapshots else 0
            ),
            "high_pressure_max_batch_items": (
                int(shard_snapshots[0].get("high_pressure_max_batch_items", 0)) if shard_snapshots else 0
            ),
            "high_pressure_max_batch_tokens": (
                int(shard_snapshots[0].get("high_pressure_max_batch_tokens", 0)) if shard_snapshots else 0
            ),
            "high_pressure_batches": int(
                sum(int(snapshot.get("high_pressure_batches", 0)) for snapshot in shard_snapshots)
            ),
            "admission_wait_total_ms": float(
                sum(float(snapshot.get("admission_wait_total_ms", 0.0)) for snapshot in shard_snapshots)
            ),
            "admission_wait_peak_ms": float(
                max((float(snapshot.get("admission_wait_peak_ms", 0.0)) for snapshot in shard_snapshots), default=0.0)
            ),
            "shards": shard_snapshots,
        }
