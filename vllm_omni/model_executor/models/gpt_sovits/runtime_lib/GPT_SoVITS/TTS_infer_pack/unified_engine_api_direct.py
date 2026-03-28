from __future__ import annotations

import asyncio
import os
import queue
import re
import threading
import time
import uuid
from io import BytesIO
from types import SimpleNamespace
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import torch

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import decode_one_step, run_prefill_active_batch
from GPT_SoVITS.TTS_infer_pack.unified_engine_audio import pack_audio, wave_header_chunk
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import DirectTTSExecution, EngineStatus, NormalizedEngineRequest, SchedulerPendingJob


class EngineApiDirectFlow:
    def __init__(self, api: Any) -> None:
        self.api = api
        self._direct_segment_suffix_re = re.compile(r"^(?P<parent>.+)_seg_(?P<index>\d{3,})$")

    def _iter_legacy_direct_tts_bytes(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> Generator[bytes, None, None]:
        payload = normalized.to_payload()
        media_type = normalized.media_type
        request_id = normalized.request_id
        request_start = time.perf_counter()
        chunk_count = 0
        stream_total_bytes = 0
        first_chunk_ms: float | None = None
        self.api._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": backend, "backend_mode": backend, "fallback_reason": fallback_reason},
        )
        try:
            with self.api.direct_tts_lock:
                tts_generator = self.api.tts.run(payload)
                first_chunk = True
                current_media_type = media_type
                for sr, chunk in tts_generator:
                    if first_chunk:
                        first_chunk_ms = max(0.0, (time.perf_counter() - request_start) * 1000.0)
                        self.api._update_request_state(
                            request_id,
                            EngineStatus.STREAMING,
                            {
                                "backend": backend,
                                "backend_mode": backend,
                                "fallback_reason": fallback_reason,
                                "sample_rate": int(sr),
                            },
                        )
                    if first_chunk and media_type == "wav":
                        header = wave_header_chunk(sample_rate=sr)
                        chunk_count += 1
                        stream_total_bytes += len(header)
                        yield header
                        current_media_type = "raw"
                        first_chunk = False
                    elif first_chunk:
                        first_chunk = False
                    packed_chunk = pack_audio(BytesIO(), chunk, sr, current_media_type).getvalue()
                    chunk_count += 1
                    stream_total_bytes += len(packed_chunk)
                    yield packed_chunk
        except Exception as exc:
            self.api._fail_request_state(request_id, str(exc))
            raise
        self.api._complete_request_state(
            request_id,
            dict(
                self.api._build_legacy_direct_profile(
                    backend=backend,
                    fallback_reason=fallback_reason,
                    request_start=request_start,
                    finished_at=time.perf_counter(),
                    audio_bytes=stream_total_bytes,
                    chunk_count=chunk_count,
                    stream_total_bytes=stream_total_bytes,
                    first_chunk_ms=first_chunk_ms,
                ),
                streaming_completed=True,
            ),
        )

    def _should_use_scheduler_backend_for_direct(self, req: dict | NormalizedEngineRequest) -> bool:
        if isinstance(req, NormalizedEngineRequest):
            normalized = req
        else:
            normalized = self.api._normalize_engine_request(
                req,
                request_id=str(req.get("request_id") or f"direct_{uuid.uuid4().hex[:12]}"),
                normalize_streaming=True,
            )
        backend, _ = self.api._select_direct_backend(normalized)
        return backend == "scheduler_v1_direct"

    def _segment_direct_text(self, normalized: dict | NormalizedEngineRequest) -> List[str]:
        payload = normalized.to_payload() if isinstance(normalized, NormalizedEngineRequest) else normalized
        return self.api.tts.text_preprocessor.pre_seg_text(
            str(payload["text"]),
            str(payload["text_lang"]),
            str(payload.get("text_split_method", "cut5")),
        )

    def _should_regroup_direct_segments(
        self,
        normalized: NormalizedEngineRequest,
        segment_texts: List[str],
    ) -> bool:
        if os.environ.get("GPTSOVITS_DIRECT_SEGMENT_REGROUP", "1").strip().lower() in {"0", "false", "no", "off"}:
            return False
        if normalized.response_streaming:
            return False
        if normalized.aux_ref_audio_paths:
            return False
        if str(normalized.text_lang).strip().lower() not in {"zh", "all_zh"}:
            return False
        min_segments = max(2, int(os.environ.get("GPTSOVITS_DIRECT_SEGMENT_REGROUP_MIN_SEGMENTS", "24")))
        if len(segment_texts) < min_segments:
            return False
        avg_chars = sum(len(str(item).strip()) for item in segment_texts) / max(1, len(segment_texts))
        avg_chars_threshold = max(1, int(os.environ.get("GPTSOVITS_DIRECT_SEGMENT_REGROUP_MAX_AVG_CHARS", "24")))
        return avg_chars <= float(avg_chars_threshold)

    def _regroup_direct_segments(
        self,
        normalized: NormalizedEngineRequest,
        segment_texts: List[str],
    ) -> List[str]:
        if not self._should_regroup_direct_segments(normalized, segment_texts):
            return list(segment_texts)

        avg_chars = sum(len(str(item).strip()) for item in segment_texts) / max(1, len(segment_texts))
        target_chars = max(8, int(os.environ.get("GPTSOVITS_DIRECT_SEGMENT_REGROUP_TARGET_CHARS", "48")))
        max_chars = max(target_chars, int(os.environ.get("GPTSOVITS_DIRECT_SEGMENT_REGROUP_MAX_CHARS", "72")))
        max_sentences = max(1, int(os.environ.get("GPTSOVITS_DIRECT_SEGMENT_REGROUP_MAX_SENTENCES", "4")))
        aggressive_enabled = (
            os.environ.get("GPTSOVITS_DIRECT_SEGMENT_REGROUP_AGGRESSIVE", "0").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        if aggressive_enabled:
            if len(segment_texts) >= 80 and avg_chars <= 20.0:
                target_chars = max(target_chars, 96)
                max_chars = max(max_chars, 128)
                max_sentences = max(max_sentences, 8)
            elif len(segment_texts) >= 48 and avg_chars <= 24.0:
                target_chars = max(target_chars, 72)
                max_chars = max(max_chars, 96)
                max_sentences = max(max_sentences, 6)

        regrouped: List[str] = []
        current_parts: List[str] = []
        current_chars = 0

        def _flush() -> None:
            nonlocal current_parts, current_chars
            if not current_parts:
                return
            regrouped.append("\n".join(current_parts))
            current_parts = []
            current_chars = 0

        for raw_segment in segment_texts:
            segment = str(raw_segment).strip()
            if not segment:
                continue
            segment_chars = len(segment)
            if not current_parts:
                current_parts.append(segment)
                current_chars = segment_chars
                if current_chars >= max_chars or len(current_parts) >= max_sentences:
                    _flush()
                continue

            next_chars = current_chars + segment_chars
            should_flush_before_append = current_chars >= target_chars or len(current_parts) >= max_sentences
            if next_chars > max_chars and should_flush_before_append:
                _flush()
                current_parts.append(segment)
                current_chars = segment_chars
            else:
                current_parts.append(segment)
                current_chars = next_chars
            if current_chars >= max_chars or len(current_parts) >= max_sentences:
                _flush()

        _flush()
        return regrouped or list(segment_texts)

    def _extract_direct_parent_request_id(self, request_id: str) -> str | None:
        matched = self._direct_segment_suffix_re.match(str(request_id))
        if not matched:
            return None
        return str(matched.group("parent"))

    @staticmethod
    def _sync_profile_device(device: Any) -> None:
        try:
            device_str = str(device)
            if device_str.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            elif device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            pass

    @staticmethod
    def _weighted_ms_shares(total_ms: float, weights: List[float]) -> List[float]:
        if not weights:
            return []
        normalized_weights = [max(1.0, float(item)) for item in weights]
        total_weight = float(sum(normalized_weights))
        if total_weight <= 0.0:
            return [0.0] * len(normalized_weights)
        return [float(total_ms) * item / total_weight for item in normalized_weights]

    @staticmethod
    def _parse_startup_shape_semantic_multiplier() -> float:
        raw_value = os.environ.get("GPTSOVITS_STARTUP_PREWARM_SHAPE_SEMANTIC_MULTIPLIER", "4.0")
        try:
            return max(1.0, float(raw_value))
        except Exception:
            return 4.0

    @staticmethod
    def _estimate_startup_shape_semantic_len(phone_len: int) -> int:
        multiplier = EngineApiDirectFlow._parse_startup_shape_semantic_multiplier()
        estimated = int(round(float(phone_len) * multiplier))
        return max(32, min(768, estimated))

    def _should_use_local_batched_direct_fastpath(
        self,
        normalized: NormalizedEngineRequest,
        segment_requests: List[NormalizedEngineRequest],
    ) -> bool:
        if os.environ.get("GPTSOVITS_DIRECT_LOCAL_BATCH_FASTPATH", "1").strip().lower() in {"0", "false", "no", "off"}:
            return False
        if normalized.response_streaming:
            return False
        if len(segment_requests) <= 1:
            return False
        if normalized.aux_ref_audio_paths:
            return False
        if bool(getattr(self.api.tts.configs, "use_vocoder", False)):
            return False
        return True

    def _build_local_batched_worker_profiles(
        self,
        *,
        states: List[Any],
        finished_items_by_id: Dict[str, Any],
        audio_lengths_by_id: Dict[str, int],
        batch_prepare_wall_ms: float,
        batch_prefill_ms: float,
        batch_decode_ms: float,
        batch_synth_ms: float,
        batch_request_total_ms: float,
    ) -> List[Dict[str, Any]]:
        prefill_weights = [max(1, int(state.all_phones.shape[-1])) for state in states]
        decode_weights = [
            max(
                1,
                int(finished_items_by_id[state.request_id].semantic_tokens.shape[0]),
                int(finished_items_by_id[state.request_id].finish_idx) + 1,
            )
            for state in states
        ]
        synth_weights = [max(1, int(audio_lengths_by_id.get(state.request_id, 0))) for state in states]
        total_weights = [
            float(prefill_weights[index]) + float(decode_weights[index]) + float(synth_weights[index])
            for index in range(len(states))
        ]
        prefill_shares = self._weighted_ms_shares(batch_prefill_ms, prefill_weights)
        decode_shares = self._weighted_ms_shares(batch_decode_ms, decode_weights)
        synth_shares = self._weighted_ms_shares(batch_synth_ms, synth_weights)
        total_shares = self._weighted_ms_shares(batch_request_total_ms, total_weights)

        profiles: List[Dict[str, Any]] = []
        batch_request_count = len(states)
        decode_batch_wall_ms = float(batch_prefill_ms) + float(batch_decode_ms)
        for index, state in enumerate(states):
            item = finished_items_by_id[state.request_id]
            profile = self.api._build_scheduler_debug_request_profile(
                state=state,
                item=item,
                batch_request_count=batch_request_count,
                prepare_batch_wall_ms=float(batch_prepare_wall_ms),
                decode_batch_wall_ms=decode_batch_wall_ms,
                batch_request_total_ms=float(batch_request_total_ms),
            )
            profile.update(
                {
                    "request_id": state.request_id,
                    "engine_policy_wait_ms": 0.0,
                    "engine_dispatch_wait_ms": 0.0,
                    "decode_admission_wait_ms": 0.0,
                    "queue_wait_ms": 0.0,
                    "prefill_ms": float(prefill_shares[index]),
                    "merge_ms": 0.0,
                    "decode_ms": float(decode_shares[index]),
                    "finalize_wait_ms": 0.0,
                    "synth_ms": float(synth_shares[index]),
                    "worker_total_ms": float(prefill_shares[index] + decode_shares[index] + synth_shares[index]),
                    "worker_other_ms": max(
                        0.0,
                        float(total_shares[index] - prefill_shares[index] - decode_shares[index] - synth_shares[index]),
                    ),
                }
            )
            profiles.append(profile)
        return profiles

    async def _run_local_batched_synth_with_backoff(
        self,
        *,
        normalized: NormalizedEngineRequest,
        states: List[Any],
        finished_items_by_id: Dict[str, Any],
        timing_device: Any,
    ) -> tuple[List[Any], float]:
        max_items = max(1, int(os.environ.get("GPTSOVITS_DIRECT_LOCAL_BATCH_SYNTH_MAX_ITEMS", "12")))
        max_semantic = max(1, int(os.environ.get("GPTSOVITS_DIRECT_LOCAL_BATCH_SYNTH_MAX_SEMANTIC", "3072")))
        ordered_states = sorted(
            states,
            key=lambda state: (
                -int(finished_items_by_id[state.request_id].semantic_tokens.shape[0]),
                -int(state.phones.shape[-1]),
                str(state.request_id),
            ),
        )
        audio_fragments_by_id: Dict[str, Any] = {}
        total_synth_ms = 0.0
        start_index = 0
        shared_synth_runtime_cache: Dict[str, Any] | None = {}

        while start_index < len(ordered_states):
            chunk_size = min(max_items, len(ordered_states) - start_index)
            while chunk_size > 0:
                end_limit = min(len(ordered_states), start_index + chunk_size)
                current_chunk_states = ordered_states[start_index:end_limit]
                semantic_budget = 0
                actual_end = start_index
                for state in current_chunk_states:
                    semantic_budget += int(finished_items_by_id[state.request_id].semantic_tokens.shape[0])
                    if actual_end > start_index and semantic_budget > max_semantic:
                        break
                    actual_end += 1
                    if semantic_budget >= max_semantic:
                        break
                if actual_end <= start_index:
                    actual_end = min(len(ordered_states), start_index + 1)
                current_chunk_states = ordered_states[start_index:actual_end]
                self._sync_profile_device(timing_device)
                synth_started_at = time.perf_counter()
                try:
                    chunk_audio_fragments = self.api.tts.synthesize_audio_requests_local_batched(
                        semantic_tokens_list=[
                            finished_items_by_id[state.request_id].semantic_tokens for state in current_chunk_states
                        ],
                        phones_list=[state.phones for state in current_chunk_states],
                        refer_specs=[[state.refer_spec] for state in current_chunk_states],
                        speeds=[float(normalized.speed_factor)] * len(current_chunk_states),
                        sample_steps_list=[int(normalized.sample_steps)] * len(current_chunk_states),
                        shared_runtime_cache=shared_synth_runtime_cache,
                    )
                    self._sync_profile_device(timing_device)
                except torch.OutOfMemoryError:
                    self._sync_profile_device(timing_device)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if len(current_chunk_states) <= 1:
                        raise
                    chunk_size = max(1, len(current_chunk_states) // 2)
                    continue
                total_synth_ms += max(0.0, (time.perf_counter() - synth_started_at) * 1000.0)
                for state, audio_fragment in zip(current_chunk_states, chunk_audio_fragments):
                    audio_fragments_by_id[state.request_id] = audio_fragment
                start_index = actual_end
                break
        return [audio_fragments_by_id[state.request_id] for state in states], total_synth_ms

    async def _run_direct_tts_via_local_batched_scheduler(
        self,
        normalized: NormalizedEngineRequest,
        *,
        segment_texts: List[str],
        segment_requests: List[NormalizedEngineRequest],
        request_start: float,
        raw_segment_count: int,
        effective_segment_count: int,
    ) -> DirectTTSExecution:
        request_id = normalized.request_id
        media_type = normalized.media_type
        backend = "scheduler_v1_direct_local_batch"
        prepared_items = await self._prepare_shared_segment_scheduler_states(segment_requests)
        prepare_profiles = [prepare_profile for _, prepare_profile in prepared_items]
        states = [state for state, _ in prepared_items]
        batch_prepare_wall_ms = sum(float(item.get("prepare_wall_ms", 0.0)) for item in prepare_profiles)
        self.api._merge_request_state_profile(
            request_id,
            {
                "engine_policy_wait_ms": 0.0,
                "engine_dispatch_wait_ms": 0.0,
                "prepare_aggregate": self.api._aggregate_numeric_dicts([item["prepare_profile"] for item in prepare_profiles]),
            },
        )
        self.api._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": backend, "backend_mode": backend},
        )

        t2s_model = self.api.tts.t2s_model.model
        max_steps = int(self.api.scheduler_worker.max_steps)
        timing_device = getattr(self.api.tts.configs, "device", None)

        self._sync_profile_device(timing_device)
        prefill_started_at = time.perf_counter()
        active_batch, finished_items = run_prefill_active_batch(t2s_model, states, max_steps=max_steps)
        self._sync_profile_device(timing_device)
        batch_prefill_ms = max(0.0, (time.perf_counter() - prefill_started_at) * 1000.0)

        batch_decode_ms = 0.0
        while active_batch is not None:
            self._sync_profile_device(timing_device)
            decode_step_started_at = time.perf_counter()
            active_batch, step_finished = decode_one_step(t2s_model, active_batch, max_steps=max_steps)
            self._sync_profile_device(timing_device)
            batch_decode_ms += max(0.0, (time.perf_counter() - decode_step_started_at) * 1000.0)
            finished_items.extend(step_finished)

        finished_items_by_id = {item.request_id: item for item in finished_items}
        missing_request_ids = [state.request_id for state in states if state.request_id not in finished_items_by_id]
        if missing_request_ids:
            raise RuntimeError(f"local batched decode 未返回完整结果: {missing_request_ids[:4]}")

        self.api._update_request_state(
            request_id,
            EngineStatus.FINALIZING,
            {"backend": backend, "backend_mode": backend},
        )

        semantic_tokens_list = [finished_items_by_id[state.request_id].semantic_tokens for state in states]
        if len(semantic_tokens_list) != len(states):
            raise RuntimeError("local batched direct semantic token 数量不匹配")
        audio_fragments, batch_synth_ms = await self._run_local_batched_synth_with_backoff(
            normalized=normalized,
            states=states,
            finished_items_by_id=finished_items_by_id,
            timing_device=timing_device,
        )

        output_sr = (
            self.api.tts.configs.sampling_rate
            if not self.api.tts.configs.use_vocoder
            else self.api.tts.vocoder_configs["sr"]
        )
        sample_rate: int | None = None
        audio_parts: List[np.ndarray] = []
        audio_lengths_by_id: Dict[str, int] = {}
        silence_chunk: Optional[np.ndarray] = None
        fragment_interval = float(normalized.fragment_interval)
        for state, audio_fragment in zip(states, audio_fragments):
            segment_sample_rate, audio_data = self.api.tts.audio_postprocess(
                audio=[[audio_fragment]],
                sr=int(output_sr),
                batch_index_list=None,
                speed_factor=float(normalized.speed_factor),
                split_bucket=False,
                fragment_interval=0.0,
                super_sampling=bool(normalized.super_sampling),
            )
            if sample_rate is None:
                sample_rate = int(segment_sample_rate)
                silence_samples = int(fragment_interval * float(sample_rate))
                if silence_samples > 0:
                    silence_chunk = np.zeros(silence_samples, dtype=np.int16)
            elif int(segment_sample_rate) != sample_rate:
                raise RuntimeError("local batched direct sample rate mismatch")
            audio_lengths_by_id[state.request_id] = int(audio_data.shape[0])
            audio_parts.append(audio_data)
            if silence_chunk is not None:
                audio_parts.append(silence_chunk.copy())
        if sample_rate is None or not audio_parts:
            raise RuntimeError("local batched direct backend produced no audio")

        pack_started_at = time.perf_counter()
        merged_audio = np.concatenate(audio_parts, axis=0)
        audio_bytes = pack_audio(BytesIO(), merged_audio, sample_rate, media_type).getvalue()
        pack_ms = max(0.0, (time.perf_counter() - pack_started_at) * 1000.0)
        request_total_ms = max(0.0, (time.perf_counter() - request_start) * 1000.0)
        worker_profiles = self._build_local_batched_worker_profiles(
            states=states,
            finished_items_by_id=finished_items_by_id,
            audio_lengths_by_id=audio_lengths_by_id,
            batch_prepare_wall_ms=batch_prepare_wall_ms,
            batch_prefill_ms=batch_prefill_ms,
            batch_decode_ms=batch_decode_ms,
            batch_synth_ms=batch_synth_ms,
            batch_request_total_ms=request_total_ms,
        )
        direct_profile = self.api._build_direct_scheduler_profile(
            backend=backend,
            request_start=request_start,
            response_ready_at=time.perf_counter(),
            audio_bytes=len(audio_bytes),
            sample_rate=int(sample_rate),
            segment_texts=segment_texts,
            prepare_profiles=prepare_profiles,
            worker_profiles=worker_profiles,
            pack_ms=pack_ms,
            response_overhead_ms=0.0,
        )
        direct_profile.update(
            {
                "prepare_ms": float(batch_prepare_wall_ms),
                "prepare_wall_ms": float(batch_prepare_wall_ms),
                "prefill_ms": float(batch_prefill_ms),
                "merge_ms": 0.0,
                "decode_ms": float(batch_decode_ms),
                "synth_ms": float(batch_synth_ms),
                "worker_total_ms": float(batch_prefill_ms + batch_decode_ms + batch_synth_ms),
                "request_total_ms": float(request_total_ms),
                "request_other_ms": max(
                    0.0,
                    float(request_total_ms - batch_prepare_wall_ms - batch_prefill_ms - batch_decode_ms - batch_synth_ms - pack_ms),
                ),
                "raw_segment_count": int(raw_segment_count),
                "effective_segment_count": int(effective_segment_count),
                "segment_regrouped": bool(effective_segment_count != raw_segment_count),
            }
        )
        self.api._complete_request_state(
            request_id,
            dict(
                direct_profile,
                streaming_completed=False,
            ),
        )
        return DirectTTSExecution(
            media_type=media_type,
            streaming=False,
            audio_bytes=audio_bytes,
            request_id=request_id,
        )

    def _build_segment_request(
        self,
        normalized: NormalizedEngineRequest,
        *,
        request_id: str,
        text: str,
    ) -> NormalizedEngineRequest:
        payload = normalized.to_payload()
        payload["request_id"] = request_id
        payload["text"] = text
        payload["streaming_mode"] = False
        payload["return_fragment"] = False
        payload["fixed_length_chunk"] = False
        payload["response_streaming"] = False
        return self.api._normalize_engine_request(payload, error_prefix="segment request 参数非法: ")

    async def _execute_single_segment_scheduler_job(
        self,
        normalized: NormalizedEngineRequest,
        *,
        segment_request: NormalizedEngineRequest,
    ) -> tuple[SchedulerPendingJob, Dict[str, Any]]:
        spec = self.api._build_scheduler_submit_spec(segment_request)
        state, prepare_exec_started_at, prepare_exec_finished_at = await self.api._prepare_state_via_engine_gpu_queue(
            spec=spec,
            prepare_submit_at=time.perf_counter(),
            engine_request_id=None,
        )
        prepare_wall_ms = max(0.0, (prepare_exec_finished_at - prepare_exec_started_at) * 1000.0)
        prepare_profile_total_ms = float(state.prepare_profile.get("wall_total_ms", prepare_wall_ms))
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        await self.api._enqueue_prepared_state_for_dispatch(
            state=state,
            speed_factor=float(normalized.speed_factor),
            sample_steps=int(normalized.sample_steps),
            media_type=normalized.media_type,
            super_sampling=bool(normalized.super_sampling),
            prepare_wall_ms=prepare_wall_ms,
            prepare_profile_total_ms=prepare_profile_total_ms,
            done_loop=loop,
            done_future=done_future,
            engine_request_id=None,
            timeout_sec=normalized.timeout_sec,
        )
        timeout_sec = float(normalized.timeout_sec if normalized.timeout_sec is not None else 30.0)
        job: SchedulerPendingJob = await asyncio.wait_for(done_future, timeout=timeout_sec)
        return job, {
            "request_id": spec.request_id,
            "prepare_wall_ms": prepare_wall_ms,
            "prepare_profile_total_ms": prepare_profile_total_ms,
            "prepare_profile": dict(state.prepare_profile),
        }

    async def _prepare_shared_segment_scheduler_states(
        self,
        segment_requests: List[NormalizedEngineRequest],
    ) -> List[tuple[Any, Dict[str, Any]]]:
        specs = [self.api._build_scheduler_submit_spec(segment_request) for segment_request in segment_requests]
        prepared_results = await self.api.scheduler_worker.prepare_direct_shared_segments_profiled_async(specs)
        prepared_items: List[tuple[Any, Dict[str, Any]]] = []
        for spec, prepared in zip(specs, prepared_results):
            if isinstance(prepared, Exception):
                raise prepared
            state, prepare_exec_started_at, prepare_exec_finished_at = prepared
            prepare_wall_ms = max(0.0, (prepare_exec_finished_at - prepare_exec_started_at) * 1000.0)
            prepare_profile_total_ms = float(state.prepare_profile.get("wall_total_ms", prepare_wall_ms))
            prepared_items.append(
                (
                    state,
                    {
                        "request_id": spec.request_id,
                        "prepare_wall_ms": prepare_wall_ms,
                        "prepare_profile_total_ms": prepare_profile_total_ms,
                        "prepare_profile": dict(state.prepare_profile),
                    },
                )
            )
        return prepared_items

    async def run_direct_tts_shape_prewarm_async(self, req: dict) -> Dict[str, Any]:
        normalized = self.api._normalize_engine_request(
            req,
            request_id=str(req.get("request_id") or f"shape_prewarm_{uuid.uuid4().hex[:12]}"),
            normalize_streaming=True,
            error_prefix="",
        )
        if bool(normalized.response_streaming):
            raise ValueError("shape-only startup prewarm 不支持 streaming 请求")
        backend, fallback_reason = self.api._select_direct_backend(normalized)
        if backend != "scheduler_v1_direct":
            raise ValueError(f"shape-only startup prewarm 仅支持 scheduler_v1_direct，当前为 {backend}")

        raw_segment_texts = self._segment_direct_text(normalized)
        if not raw_segment_texts:
            raise ValueError("shape-only startup prewarm 未得到有效切分结果")
        segment_texts = self._regroup_direct_segments(normalized, raw_segment_texts)
        segment_requests = [
            self._build_segment_request(
                normalized,
                request_id=f"{normalized.request_id}_seg_{segment_index:03d}",
                text=segment_text,
            )
            for segment_index, segment_text in enumerate(segment_texts)
        ]
        prepared_items = await self._prepare_shared_segment_scheduler_states(segment_requests)
        states = [state for state, _ in prepared_items]
        prepare_profiles = [profile for _, profile in prepared_items]
        if not states:
            raise ValueError("shape-only startup prewarm 未得到可用 prepared states")

        t2s_model = self.api.tts.t2s_model.model
        max_steps = int(self.api.scheduler_worker.max_steps)
        timing_device = getattr(self.api.tts.configs, "device", None)

        self._sync_profile_device(timing_device)
        prefill_started_at = time.perf_counter()
        active_batch, finished_items = run_prefill_active_batch(t2s_model, states, max_steps=max_steps)
        self._sync_profile_device(timing_device)
        prefill_ms = max(0.0, (time.perf_counter() - prefill_started_at) * 1000.0)

        extra_decode_steps = max(0, int(os.environ.get("GPTSOVITS_STARTUP_PREWARM_SHAPE_EXTRA_DECODE_STEPS", "0")))
        decode_ms = 0.0
        for _ in range(extra_decode_steps):
            if active_batch is None:
                break
            self._sync_profile_device(timing_device)
            decode_started_at = time.perf_counter()
            active_batch, step_finished = decode_one_step(t2s_model, active_batch, max_steps=max_steps)
            self._sync_profile_device(timing_device)
            decode_ms += max(0.0, (time.perf_counter() - decode_started_at) * 1000.0)
            finished_items.extend(step_finished)

        finished_items_by_id = {item.request_id: item for item in finished_items}
        device = self.api.tts.configs.device
        for state in states:
            if state.request_id in finished_items_by_id:
                continue
            phone_len = int(state.phones.shape[-1])
            semantic_len = self._estimate_startup_shape_semantic_len(phone_len)
            finished_items_by_id[state.request_id] = SimpleNamespace(
                request_id=state.request_id,
                semantic_tokens=torch.zeros((semantic_len,), dtype=torch.long, device=device),
                finish_idx=max(0, semantic_len - 1),
                finish_reason="shape_only",
            )

        _, synth_ms = await self._run_local_batched_synth_with_backoff(
            normalized=normalized,
            states=states,
            finished_items_by_id=finished_items_by_id,
            timing_device=timing_device,
        )

        max_phone_len = max(int(state.phones.shape[-1]) for state in states)
        max_all_phone_len = max(int(state.all_phones.shape[-1]) for state in states)
        return {
            "warmup_backend": "scheduler_v1_direct_shape_only",
            "fallback_reason": fallback_reason,
            "audio_bytes": 0,
            "segment_count": int(len(segment_texts)),
            "raw_segment_count": int(len(raw_segment_texts)),
            "segment_regrouped": bool(len(segment_texts) != len(raw_segment_texts)),
            "shape_prepare_ms": float(sum(float(item.get("prepare_wall_ms", 0.0)) for item in prepare_profiles)),
            "shape_prefill_ms": float(prefill_ms),
            "shape_decode_ms": float(decode_ms),
            "shape_synth_ms": float(synth_ms),
            "shape_max_phone_len": int(max_phone_len),
            "shape_max_all_phone_len": int(max_all_phone_len),
        }

    def _iter_scheduler_direct_tts_bytes(self, normalized: NormalizedEngineRequest) -> Generator[bytes, None, None]:
        request_start = time.perf_counter()
        request_id = normalized.request_id
        media_type = normalized.media_type
        segment_texts = self._segment_direct_text(normalized)
        if not segment_texts:
            raise ValueError("text preprocessing returned no valid segments")
        chunk_queue: queue.Queue[object] = queue.Queue(maxsize=8)
        done_marker = object()

        async def _produce_chunks() -> None:
            self.api._update_request_state(
                request_id,
                EngineStatus.CPU_PREPARING,
                {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct", "segment_count": len(segment_texts)},
            )
            sample_rate: int | None = None
            current_media_type = media_type
            chunk_count = 0
            stream_total_bytes = 0
            first_chunk_ms: float | None = None
            prepare_profiles: List[Dict[str, Any]] = []
            worker_profiles: List[Dict[str, Any]] = []
            try:
                for segment_index, segment_text in enumerate(segment_texts):
                    segment_request = self._build_segment_request(
                        normalized,
                        request_id=f"{request_id}_seg_{segment_index:03d}",
                        text=segment_text,
                    )
                    self.api._update_request_state(
                        request_id,
                        EngineStatus.READY_FOR_PREFILL,
                        {
                            "backend": "scheduler_v1_direct",
                            "backend_mode": "scheduler_v1_direct",
                            "segment_index": segment_index,
                            "segment_count": len(segment_texts),
                        },
                    )
                    job, prepare_profile = await self._execute_single_segment_scheduler_job(
                        normalized,
                        segment_request=segment_request,
                    )
                    prepare_profiles.append(prepare_profile)
                    if job.error is not None:
                        raise RuntimeError(job.error)
                    if job.audio_data is None or job.sample_rate is None or job.result is None:
                        raise RuntimeError(f"{job.request_id} finished without audio result")
                    worker_profiles.append(dict(job.result))
                    if sample_rate is None:
                        sample_rate = int(job.sample_rate)
                        first_chunk_ms = max(0.0, (time.perf_counter() - request_start) * 1000.0)
                        self.api._update_request_state(
                            request_id,
                            EngineStatus.STREAMING,
                            {
                                "backend": "scheduler_v1_direct",
                                "backend_mode": "scheduler_v1_direct",
                                "sample_rate": int(sample_rate),
                            },
                        )
                        if media_type == "wav":
                            header = wave_header_chunk(sample_rate=int(sample_rate))
                            chunk_count += 1
                            stream_total_bytes += len(header)
                            chunk_queue.put(header)
                            current_media_type = "raw"
                    packed_chunk = pack_audio(BytesIO(), job.audio_data, int(job.sample_rate), current_media_type).getvalue()
                    chunk_count += 1
                    stream_total_bytes += len(packed_chunk)
                    chunk_queue.put(packed_chunk)
                    if segment_index + 1 < len(segment_texts):
                        silence_samples = int(float(normalized.fragment_interval) * float(job.sample_rate))
                        if silence_samples > 0:
                            silence_chunk = np.zeros(silence_samples, dtype=np.int16)
                            packed_silence = pack_audio(
                                BytesIO(), silence_chunk, int(job.sample_rate), current_media_type
                            ).getvalue()
                            chunk_count += 1
                            stream_total_bytes += len(packed_silence)
                            chunk_queue.put(packed_silence)
            except Exception as exc:
                self.api._fail_request_state(request_id, str(exc))
                chunk_queue.put(exc)
            else:
                self.api._merge_request_state_profile(
                    request_id,
                    {
                        "prepare_aggregate": self.api._aggregate_numeric_dicts(
                            [item["prepare_profile"] for item in prepare_profiles]
                        ),
                        "engine_policy_wait_ms": sum(
                            float(item.get("engine_policy_wait_ms", 0.0)) for item in worker_profiles
                        ),
                        "engine_dispatch_wait_ms": sum(
                            float(item.get("engine_dispatch_wait_ms", 0.0)) for item in worker_profiles
                        ),
                    },
                )
                direct_profile = self.api._build_direct_scheduler_profile(
                    backend="scheduler_v1_direct",
                    request_start=request_start,
                    response_ready_at=time.perf_counter(),
                    audio_bytes=stream_total_bytes,
                    sample_rate=int(sample_rate or 0),
                    segment_texts=segment_texts,
                    prepare_profiles=prepare_profiles,
                    worker_profiles=worker_profiles,
                    pack_ms=0.0,
                    response_overhead_ms=0.0,
                )
                self.api._complete_request_state(
                    request_id,
                    dict(direct_profile, streaming_completed=True, first_chunk_ms=first_chunk_ms),
                )
            finally:
                chunk_queue.put(done_marker)

        producer_thread = threading.Thread(target=lambda: asyncio.run(_produce_chunks()), daemon=True)
        producer_thread.start()
        while True:
            item = chunk_queue.get()
            if item is done_marker:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def _run_direct_tts_via_scheduler(self, normalized: NormalizedEngineRequest) -> DirectTTSExecution:
        request_start = time.perf_counter()
        request_id = normalized.request_id
        media_type = normalized.media_type
        raw_segment_texts = self._segment_direct_text(normalized)
        if not raw_segment_texts:
            raise ValueError("text preprocessing returned no valid segments")
        segment_texts = self._regroup_direct_segments(normalized, raw_segment_texts)
        raw_segment_count = len(raw_segment_texts)
        effective_segment_count = len(segment_texts)
        if normalized.response_streaming:
            return DirectTTSExecution(
                media_type=media_type,
                streaming=True,
                audio_generator=self._iter_scheduler_direct_tts_bytes(normalized),
                request_id=request_id,
            )
        self.api._update_request_state(
            request_id,
            EngineStatus.CPU_PREPARING,
            {
                "backend": "scheduler_v1_direct",
                "backend_mode": "scheduler_v1_direct",
                "segment_count": int(effective_segment_count),
                "raw_segment_count": int(raw_segment_count),
                "segment_regrouped": bool(effective_segment_count != raw_segment_count),
            },
        )
        segment_requests = [
            self._build_segment_request(
                normalized,
                request_id=f"{request_id}_seg_{segment_index:03d}",
                text=segment_text,
            )
            for segment_index, segment_text in enumerate(segment_texts)
        ]
        prepare_profiles: List[Dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        done_futures: List[asyncio.Future] = []
        self.api._update_request_state(
            request_id,
            EngineStatus.READY_FOR_PREFILL,
            {
                "backend": "scheduler_v1_direct",
                "backend_mode": "scheduler_v1_direct",
                "segment_count": len(segment_requests),
                "raw_segment_count": int(raw_segment_count),
                "segment_regrouped": bool(effective_segment_count != raw_segment_count),
            },
        )
        if self._should_use_local_batched_direct_fastpath(normalized, segment_requests):
            return await self._run_direct_tts_via_local_batched_scheduler(
                normalized,
                segment_texts=segment_texts,
                segment_requests=segment_requests,
                request_start=request_start,
                raw_segment_count=raw_segment_count,
                effective_segment_count=effective_segment_count,
            )
        if len(segment_requests) <= 1:
            prepared_items = await asyncio.gather(
                *[
                    self._execute_single_segment_scheduler_job(
                        normalized,
                        segment_request=segment_request,
                    )
                    for segment_request in segment_requests
                ]
            )
            for job, prepare_profile in prepared_items:
                prepare_profiles.append(prepare_profile)
                done_future = loop.create_future()
                done_future.set_result(job)
                done_futures.append(done_future)
        else:
            prepared_items = await self._prepare_shared_segment_scheduler_states(segment_requests)
            for state, prepare_profile in prepared_items:
                prepare_profiles.append(prepare_profile)
                done_future = loop.create_future()
                done_futures.append(done_future)
                await self.api._enqueue_prepared_state_for_dispatch(
                    state=state,
                    speed_factor=float(normalized.speed_factor),
                    sample_steps=int(normalized.sample_steps),
                    media_type=normalized.media_type,
                    super_sampling=bool(normalized.super_sampling),
                    prepare_wall_ms=float(prepare_profile["prepare_wall_ms"]),
                    prepare_profile_total_ms=float(prepare_profile["prepare_profile_total_ms"]),
                    done_loop=loop,
                    done_future=done_future,
                    engine_request_id=None,
                    timeout_sec=normalized.timeout_sec,
                )
        self.api._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct"},
        )
        timeout_sec = float(normalized.timeout_sec if normalized.timeout_sec is not None else 30.0)
        jobs: List[SchedulerPendingJob] = list(await asyncio.wait_for(asyncio.gather(*done_futures), timeout=timeout_sec))
        for profile_item, job in zip(prepare_profiles, jobs):
            profile_item["engine_policy_wait_ms"] = float(job.engine_policy_wait_ms)
            profile_item["engine_dispatch_wait_ms"] = float(job.engine_dispatch_wait_ms)
        self.api._merge_request_state_profile(
            request_id,
            {
                "engine_policy_wait_ms": sum(float(job.engine_policy_wait_ms) for job in jobs),
                "engine_dispatch_wait_ms": sum(float(job.engine_dispatch_wait_ms) for job in jobs),
                "prepare_aggregate": self.api._aggregate_numeric_dicts([item["prepare_profile"] for item in prepare_profiles]),
            },
        )

        sample_rate: int | None = None
        audio_parts: List[np.ndarray] = []
        worker_profiles: List[Dict[str, Any]] = []
        fragment_interval = float(normalized.fragment_interval)
        silence_chunk: Optional[np.ndarray] = None
        for job in jobs:
            if job.error is not None:
                raise RuntimeError(job.error)
            if job.audio_data is None or job.sample_rate is None or job.result is None:
                raise RuntimeError(f"{job.request_id} finished without audio result")
            if sample_rate is None:
                sample_rate = int(job.sample_rate)
                silence_samples = int(fragment_interval * float(sample_rate))
                if silence_samples > 0:
                    silence_chunk = np.zeros(silence_samples, dtype=np.int16)
            elif int(job.sample_rate) != sample_rate:
                raise RuntimeError("segment sample rate mismatch")
            audio_parts.append(job.audio_data)
            if silence_chunk is not None:
                audio_parts.append(silence_chunk.copy())
            worker_profiles.append(dict(job.result))
        if sample_rate is None or not audio_parts:
            raise RuntimeError("direct scheduler backend produced no audio")
        self.api._update_request_state(
            request_id,
            EngineStatus.FINALIZING,
            {"backend": "scheduler_v1_direct", "backend_mode": "scheduler_v1_direct"},
        )
        merged_audio = np.concatenate(audio_parts, axis=0)
        pack_start = time.perf_counter()
        audio_bytes = pack_audio(BytesIO(), merged_audio, sample_rate, media_type).getvalue()
        pack_ms = max(0.0, (time.perf_counter() - pack_start) * 1000.0)
        direct_profile = self.api._build_direct_scheduler_profile(
            backend="scheduler_v1_direct",
            request_start=request_start,
            response_ready_at=time.perf_counter(),
            audio_bytes=len(audio_bytes),
            sample_rate=int(sample_rate),
            segment_texts=segment_texts,
            prepare_profiles=prepare_profiles,
            worker_profiles=worker_profiles,
            pack_ms=pack_ms,
            response_overhead_ms=0.0,
        )
        self.api._complete_request_state(
            request_id,
            dict(
                direct_profile,
                streaming_completed=False,
                raw_segment_count=int(raw_segment_count),
                effective_segment_count=int(effective_segment_count),
                segment_regrouped=bool(effective_segment_count != raw_segment_count),
            ),
        )
        return DirectTTSExecution(
            media_type=media_type,
            streaming=False,
            audio_bytes=audio_bytes,
            request_id=request_id,
        )

    def _run_legacy_direct_tts_blocking(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> DirectTTSExecution:
        normalized_payload = normalized.to_payload()
        request_id = normalized.request_id
        media_type = normalized.media_type
        request_start = time.perf_counter()
        self.api._update_request_state(
            request_id,
            EngineStatus.ACTIVE_DECODE,
            {"backend": backend, "backend_mode": backend, "fallback_reason": fallback_reason},
        )
        with self.api.direct_tts_lock:
            tts_generator = self.api.tts.run(normalized_payload)
            try:
                sr, audio_data = next(tts_generator)
            except Exception as exc:
                self.api._fail_request_state(request_id, str(exc))
                raise
        self.api._update_request_state(
            request_id,
            EngineStatus.FINALIZING,
            {"backend": backend, "backend_mode": backend, "fallback_reason": fallback_reason},
        )
        pack_start = time.perf_counter()
        packed_audio = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
        pack_ms = max(0.0, (time.perf_counter() - pack_start) * 1000.0)
        self.api._complete_request_state(
            request_id,
            dict(
                self.api._build_legacy_direct_profile(
                    backend=backend,
                    fallback_reason=fallback_reason,
                    request_start=request_start,
                    finished_at=time.perf_counter(),
                    sample_rate=int(sr),
                    audio_bytes=len(packed_audio),
                    pack_ms=pack_ms,
                ),
                streaming_completed=False,
            ),
        )
        return DirectTTSExecution(
            media_type=media_type,
            streaming=False,
            audio_bytes=packed_audio,
            request_id=request_id,
        )

    async def _run_direct_tts_via_legacy_backend(
        self,
        normalized: NormalizedEngineRequest,
        *,
        backend: str,
        fallback_reason: str | None,
    ) -> DirectTTSExecution:
        if normalized.response_streaming:
            return DirectTTSExecution(
                media_type=normalized.media_type,
                streaming=True,
                audio_generator=self._iter_legacy_direct_tts_bytes(
                    normalized,
                    backend=backend,
                    fallback_reason=fallback_reason,
                ),
                request_id=normalized.request_id,
            )
        return await asyncio.to_thread(
            self._run_legacy_direct_tts_blocking,
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    async def run_direct_tts_async(self, req: dict) -> DirectTTSExecution:
        normalized = self.api._normalize_engine_request(
            req,
            request_id=str(req.get("request_id") or f"direct_{uuid.uuid4().hex[:12]}"),
            normalize_streaming=True,
            error_prefix="",
        )
        request_id = normalized.request_id
        media_type = normalized.media_type
        backend, fallback_reason = self.api._select_direct_backend(normalized)
        self.api._register_request_state(
            request_id=request_id,
            api_mode="tts",
            backend=backend,
            media_type=media_type,
            response_streaming=bool(normalized.response_streaming),
            deadline_ts=(time.perf_counter() + float(normalized.timeout_sec) if normalized.timeout_sec is not None else None),
            meta=self.api._build_request_meta(normalized.to_payload()),
        )
        self.api._update_request_state(
            request_id,
            EngineStatus.VALIDATED,
            {
                "request_source": "direct_tts",
                "selected_backend": backend,
                "fallback_reason": fallback_reason,
            },
        )
        if backend == "scheduler_v1_direct":
            try:
                return await self._run_direct_tts_via_scheduler(normalized)
            except Exception as exc:
                self.api._fail_request_state(request_id, str(exc))
                raise
        return await self._run_direct_tts_via_legacy_backend(
            normalized,
            backend=backend,
            fallback_reason=fallback_reason,
        )

    def run_direct_tts_startup_prewarm(self, req: dict) -> DirectTTSExecution:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_direct_tts_async(req))
        raise RuntimeError("startup prewarm direct tts 不能在已有 event loop 内同步执行")

    def run_direct_tts_shape_prewarm(self, req: dict) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_direct_tts_shape_prewarm_async(req))
        raise RuntimeError("shape-only startup prewarm 不能在已有 event loop 内同步执行")

    def run_direct_tts(self, req: dict) -> DirectTTSExecution:
        normalized = self.api._normalize_engine_request(
            req,
            request_id=str(req.get("request_id") or f"direct_{uuid.uuid4().hex[:12]}"),
            normalize_streaming=True,
            error_prefix="",
        )
        request_id = normalized.request_id
        media_type = normalized.media_type
        backend, fallback_reason = self.api._select_direct_backend(normalized)
        if not self.api._has_active_request(request_id):
            self.api._register_request_state(
                request_id=request_id,
                api_mode="tts",
                backend=backend,
                media_type=media_type,
                response_streaming=bool(normalized.response_streaming),
                meta=self.api._build_request_meta(normalized.to_payload()),
            )
        self.api._update_request_state(
            request_id,
            EngineStatus.VALIDATED,
            {
                "request_source": "direct_tts",
                "selected_backend": backend,
                "fallback_reason": fallback_reason,
            },
        )
        if backend != "scheduler_v1_direct":
            if normalized.response_streaming:
                return DirectTTSExecution(
                    media_type=media_type,
                    streaming=True,
                    audio_generator=self._iter_legacy_direct_tts_bytes(
                        normalized,
                        backend=backend,
                        fallback_reason=fallback_reason,
                    ),
                    request_id=request_id,
                )
            return self._run_legacy_direct_tts_blocking(
                normalized,
                backend=backend,
                fallback_reason=fallback_reason,
            )
        if normalized.response_streaming:
            return DirectTTSExecution(
                media_type=media_type,
                streaming=True,
                audio_generator=self._iter_legacy_direct_tts_bytes(
                    normalized,
                    backend="legacy_direct_sync_compat",
                    fallback_reason="sync_direct_compat",
                ),
                request_id=request_id,
            )
        return self._run_legacy_direct_tts_blocking(
            normalized,
            backend="legacy_direct_sync_compat",
            fallback_reason="sync_direct_compat",
        )
