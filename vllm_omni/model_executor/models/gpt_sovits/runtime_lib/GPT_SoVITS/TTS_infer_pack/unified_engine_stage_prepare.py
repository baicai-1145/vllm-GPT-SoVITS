from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineGpuPrepareTask, EngineStatus


class EnginePrepareStageMixin:
    @staticmethod
    def _engine_prepare_batch_wait_env_name(phase: str, suffix: str) -> str:
        phase_key = str(phase or "").strip().upper()
        return f"GPTSOVITS_ENGINE_PREPARE_{phase_key}_{suffix}"

    def _engine_prepare_batch_wait_policy(self, phase: str, batch_max_items: int) -> dict[str, float]:
        wait_ms_raw = os.environ.get(
            self._engine_prepare_batch_wait_env_name(phase, "BATCH_WAIT_MS"),
            os.environ.get("GPTSOVITS_ENGINE_PREPARE_BATCH_WAIT_MS", "0"),
        )
        min_items_raw = os.environ.get(
            self._engine_prepare_batch_wait_env_name(phase, "BATCH_WAIT_MIN_ITEMS"),
            os.environ.get("GPTSOVITS_ENGINE_PREPARE_BATCH_WAIT_MIN_ITEMS"),
        )
        high_pressure_waiting_raw = os.environ.get(
            self._engine_prepare_batch_wait_env_name(phase, "BATCH_WAIT_HIGH_PRESSURE_WAITING"),
            os.environ.get("GPTSOVITS_ENGINE_PREPARE_BATCH_WAIT_HIGH_PRESSURE_WAITING", "0"),
        )
        wait_ms = max(0.0, float(wait_ms_raw or 0.0))
        min_items = max(
            1,
            min(
                max(1, int(batch_max_items)),
                int(min_items_raw) if min_items_raw not in [None, ""] else int(batch_max_items),
            ),
        )
        high_pressure_waiting = max(0, int(high_pressure_waiting_raw or 0))
        return {
            "wait_ms": float(wait_ms),
            "min_items": float(min_items),
            "high_pressure_waiting": float(high_pressure_waiting),
        }

    def _engine_prepare_effective_batch_max_items(self, phase: str, batch_max_items: int) -> int:
        batch_max_items = max(1, int(batch_max_items))
        phase_raw = os.environ.get(self._engine_prepare_batch_wait_env_name(phase, "QUOTA_ITEMS"))
        global_raw = os.environ.get("GPTSOVITS_ENGINE_PREPARE_QUOTA_ITEMS")
        raw = phase_raw if phase_raw not in [None, ""] else global_raw
        if raw in [None, ""]:
            return batch_max_items
        return max(1, min(batch_max_items, int(raw)))

    def _engine_prepare_audio_is_first_round(self) -> bool:
        return (
            int(self.prepare_queue_owner.waiting_count()) > 0
            and int(self.prepare_text_queue_owner.waiting_count()) <= 0
            and int(self.prepare_ref_spec_queue_owner.waiting_count()) <= 0
        )

    def _engine_prepare_audio_effective_batch_max_items(self, batch_max_items: int) -> int:
        batch_max_items = self._engine_prepare_effective_batch_max_items("audio", batch_max_items)
        if not self._engine_prepare_audio_is_first_round():
            return batch_max_items
        raw = os.environ.get("GPTSOVITS_ENGINE_PREPARE_AUDIO_FIRST_BATCH_MAX_ITEMS")
        if raw in [None, ""]:
            return batch_max_items
        return max(1, min(batch_max_items, int(raw)))

    def _engine_prepare_audio_effective_batch_wait_policy(self, batch_max_items: int) -> dict[str, float]:
        effective_batch_max_items = self._engine_prepare_audio_effective_batch_max_items(batch_max_items)
        policy = self._engine_prepare_batch_wait_policy("audio", effective_batch_max_items)
        if not self._engine_prepare_audio_is_first_round():
            return policy
        first_wait_ms = os.environ.get("GPTSOVITS_ENGINE_PREPARE_AUDIO_FIRST_BATCH_WAIT_MS")
        first_min_items = os.environ.get("GPTSOVITS_ENGINE_PREPARE_AUDIO_FIRST_BATCH_WAIT_MIN_ITEMS")
        first_high_pressure_waiting = os.environ.get(
            "GPTSOVITS_ENGINE_PREPARE_AUDIO_FIRST_BATCH_WAIT_HIGH_PRESSURE_WAITING"
        )
        if first_wait_ms not in [None, ""]:
            policy["wait_ms"] = max(0.0, float(first_wait_ms))
        if first_min_items not in [None, ""]:
            policy["min_items"] = float(
                max(1, min(effective_batch_max_items, int(first_min_items)))
            )
        else:
            policy["min_items"] = float(min(int(policy.get("min_items", effective_batch_max_items)), effective_batch_max_items))
        if first_high_pressure_waiting not in [None, ""]:
            policy["high_pressure_waiting"] = float(max(0, int(first_high_pressure_waiting)))
        return policy

    def _schedule_prepare_batch_wait_retry(self, phase: str, delay_s: float) -> None:
        delay_s = max(0.0, float(delay_s))
        if delay_s <= 0.0:
            self.notify_arbiter()
            return
        lock = getattr(self, "_prepare_batch_wait_retry_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._prepare_batch_wait_retry_lock = lock
        deadlines = getattr(self, "_prepare_batch_wait_retry_deadlines", None)
        if deadlines is None:
            deadlines = {}
            self._prepare_batch_wait_retry_deadlines = deadlines
        target_ts = time.perf_counter() + delay_s
        with lock:
            current_ts = float(deadlines.get(phase, 0.0) or 0.0)
            if current_ts > 0.0 and current_ts <= (target_ts + 1e-6):
                return
            deadlines[phase] = float(target_ts)

        def _wake() -> None:
            try:
                self.notify_arbiter()
            finally:
                current_lock = getattr(self, "_prepare_batch_wait_retry_lock", None)
                current_deadlines = getattr(self, "_prepare_batch_wait_retry_deadlines", None)
                if current_lock is None or current_deadlines is None:
                    return
                with current_lock:
                    active_ts = float(current_deadlines.get(phase, 0.0) or 0.0)
                    if abs(active_ts - target_ts) <= 1e-6:
                        current_deadlines.pop(phase, None)

        timer = threading.Timer(delay_s, _wake)
        timer.daemon = True
        timer.start()

    def _should_defer_prepare_batch(
        self,
        *,
        phase: str,
        queue_owner,
        batch_max_items: int,
        timestamp_attr: str = "enqueue_time",
        policy_override: dict[str, float] | None = None,
    ) -> bool:
        policy = dict(policy_override) if policy_override is not None else self._engine_prepare_batch_wait_policy(phase, batch_max_items)
        wait_ms = float(policy.get("wait_ms", 0.0))
        if wait_ms <= 0.0 or batch_max_items <= 1:
            return False
        if self.finalize_queue_owner.has_items() or self.dispatch_queue_owner.has_items():
            return False
        decode_runtime_state = self.snapshot_engine_decode_runtime_state()
        if bool(decode_runtime_state.get("has_work", False)):
            return False
        waiting_count = int(queue_owner.waiting_count())
        if waiting_count <= 0:
            return False
        min_items = max(1, int(policy.get("min_items", batch_max_items) or batch_max_items))
        if waiting_count >= min_items:
            return False
        high_pressure_waiting = max(0, int(policy.get("high_pressure_waiting", 0.0) or 0.0))
        if high_pressure_waiting > 0 and waiting_count < high_pressure_waiting:
            return False
        oldest_age_ms = float(queue_owner.peek_oldest_age_ms(timestamp_attr))
        remaining_ms = max(0.0, wait_ms - oldest_age_ms)
        if remaining_ms <= 0.0:
            return False
        self._schedule_prepare_batch_wait_retry(phase, remaining_ms / 1000.0)
        return True

    @staticmethod
    def _engine_prepare_ref_wav16k_preload_enabled() -> bool:
        return str(os.environ.get("GPTSOVITS_ENGINE_PREPARE_REF_WAV16K_PRELOAD", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _prepare_waiting_total(self) -> int:
        return (
            int(self.prepare_queue_owner.waiting_count())
            + int(self.prepare_text_queue_owner.waiting_count())
            + int(self.prepare_ref_spec_queue_owner.waiting_count())
        )

    @staticmethod
    def _engine_prepare_audio_first_enabled() -> bool:
        return str(os.environ.get("GPTSOVITS_ENGINE_PREPARE_AUDIO_FIRST", "0")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    async def _wait_prepare_queue_admission(self) -> float:
        soft_max = max(0, int(os.environ.get("GPTSOVITS_ENGINE_PREPARE_QUEUE_SOFT_MAX", "0")))
        if soft_max <= 0:
            return 0.0
        poll_s = max(
            0.0005,
            float(max(1, int(os.environ.get("GPTSOVITS_ENGINE_PREPARE_QUEUE_ADMISSION_POLL_MS", "1")))) / 1000.0,
        )
        wait_start = time.perf_counter()
        while self._prepare_waiting_total() >= soft_max:
            await asyncio.sleep(poll_s)
        return max(0.0, (time.perf_counter() - wait_start) * 1000.0)

    async def prepare_state_via_engine_gpu_queue(
        self,
        *,
        spec: Any,
        prepare_submit_at: float,
        engine_request_id: str | None,
    ) -> tuple[T2SRequestState, float, float]:
        prepare_queue_admission_wait_ms = await self._wait_prepare_queue_admission()
        cpu_stage = await self.scheduler_worker.prepare_cpu_stage_profiled_async(spec, prepare_submit_at)
        if engine_request_id not in [None, ""]:
            self.update_request_state(
                str(engine_request_id),
                EngineStatus.GPU_PREPARING,
                {
                    "engine_prepare_queue_admission_wait_ms": float(prepare_queue_admission_wait_ms),
                    "prompt_text_cpu_queue_ms": float(cpu_stage.prompt_cpu_profiled.queue_ms),
                    "prompt_text_cpu_run_ms": float(cpu_stage.prompt_cpu_profiled.run_ms),
                    "text_cpu_queue_ms": float(cpu_stage.target_cpu_profiled.queue_ms),
                    "text_cpu_run_ms": float(cpu_stage.target_cpu_profiled.run_ms),
                },
            )
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        task = EngineGpuPrepareTask(
            request_id=spec.request_id,
            cpu_stage=cpu_stage,
            done_loop=loop,
            done_future=done_future,
            engine_request_id=engine_request_id or spec.request_id,
            enqueue_time=time.perf_counter(),
            phase="audio",
            audio_enqueue_time=time.perf_counter(),
            admission_wait_ms=float(prepare_queue_admission_wait_ms),
        )
        if self._engine_prepare_ref_wav16k_preload_enabled():
            task.ref_audio_prepare_future = self.scheduler_worker.submit_prepare_ref_audio_asset(
                str(cpu_stage.spec.ref_audio_path),
                submit_at=float(task.audio_enqueue_time),
            )
        self.prepare_queue_owner.enqueue(task)
        self.notify_arbiter()
        return await done_future

    def _should_chain_prepare_text_after_audio(self) -> bool:
        if str(os.environ.get("GPTSOVITS_ENGINE_PREPARE_CHAIN_TEXT", "1")).strip().lower() in {"0", "false", "no", "off"}:
            return False
        if self.finalize_queue_owner.has_items() or self.dispatch_queue_owner.has_items():
            return False
        decode_runtime_state = self.snapshot_engine_decode_runtime_state()
        if bool(decode_runtime_state.get("has_work", False)):
            return False
        return True

    def _maybe_apply_ref_spec_to_state(self, task: EngineGpuPrepareTask) -> None:
        if task.state_result is None or task.ref_spec_result is None:
            return
        self.scheduler_worker.apply_ref_spec_result_to_state(task.state_result, task.ref_spec_result)
        if task.engine_request_id not in [None, ""]:
            self.merge_request_state_profile(
                str(task.engine_request_id),
                {
                    "engine_prepare_ref_spec_queue_wait_ms": float(task.ref_spec_queue_wait_ms),
                    "ref_spec_wait_ms": float(task.ref_spec_result[1].get("ref_spec_wait_ms", 0.0)),
                    "ref_spec_ms": float(task.ref_spec_result[1].get("ref_spec_ms", 0.0)),
                    "ref_spec_to_device_ms": float(task.ref_spec_result[1].get("ref_spec_to_device_ms", 0.0)),
                    "ref_spec_main_resample_ms": float(task.ref_spec_result[1].get("ref_spec_main_resample_ms", 0.0)),
                    "ref_spec_norm_ms": float(task.ref_spec_result[1].get("ref_spec_norm_ms", 0.0)),
                    "ref_spec_spectrogram_ms": float(task.ref_spec_result[1].get("ref_spec_spectrogram_ms", 0.0)),
                    "ref_spec_post_resample_ms": float(task.ref_spec_result[1].get("ref_spec_post_resample_ms", 0.0)),
                },
            )

    def _mark_ref_spec_async_failed(
        self,
        task: EngineGpuPrepareTask,
        error: Exception,
        *,
        queue_wait_ms: float,
    ) -> None:
        task.error = str(error)
        task.cancelled = True
        if task.state_result is not None:
            task.state_result.prepare_profile["ref_spec_async_failed"] = 1.0
            task.state_result.prepare_profile["engine_prepare_ref_spec_queue_wait_ms"] = float(queue_wait_ms)
        if task.engine_request_id not in [None, ""]:
            self.merge_request_state_profile(
                str(task.engine_request_id),
                {
                    "ref_spec_async_failed": 1.0,
                    "engine_prepare_ref_spec_queue_wait_ms": float(queue_wait_ms),
                },
            )
        self.fail_request_state(task.engine_request_id or task.request_id, str(error))
        self.fail_engine_jobs([task.request_id], str(error))
        self.notify_arbiter()

    def _run_engine_prepare_audio_once(self, batch_max_items: int) -> bool:
        is_first_round = self._engine_prepare_audio_is_first_round()
        effective_batch_max_items = self._engine_prepare_audio_effective_batch_max_items(batch_max_items)
        effective_wait_policy = self._engine_prepare_audio_effective_batch_wait_policy(batch_max_items)
        if self._should_defer_prepare_batch(
            phase="audio",
            queue_owner=self.prepare_queue_owner,
            batch_max_items=effective_batch_max_items,
            timestamp_attr="enqueue_time",
            policy_override=effective_wait_policy,
        ):
            return False
        tasks = self.prepare_queue_owner.pop_left_many(effective_batch_max_items)
        if not tasks:
            return False
        now = time.perf_counter()
        queue_wait_ms_list = [max(0.0, (now - task.enqueue_time) * 1000.0) for task in tasks]
        for task in tasks:
            task.audio_start_time = float(now)
        batch_results = asyncio.run(
            self.scheduler_worker.prepare_gpu_audio_phases_async(
                [task.cpu_stage for task in tasks],
                [task.ref_audio_prepare_future for task in tasks],
            )
        )
        completed_count = 0
        for task, queue_wait_ms, result in zip(tasks, queue_wait_ms_list, batch_results):
            task.audio_end_time = time.perf_counter()
            if isinstance(result, Exception):
                task.error = str(result)
                self.fail_request_state(task.engine_request_id or task.request_id, str(result))
                self._notify_prepare_error(task, result)
                completed_count += 1
                continue
            task.audio_queue_wait_ms = float(queue_wait_ms)
            task.phase_one = result
            task.phase = "text"
            task.enqueue_time = time.perf_counter()
            task.text_enqueue_time = float(task.enqueue_time)
            task.ref_spec_enqueue_time = float(task.enqueue_time)
            self.prepare_text_queue_owner.enqueue(task)
            self.prepare_ref_spec_queue_owner.enqueue(task)
            if task.engine_request_id not in [None, ""]:
                self.merge_request_state_profile(
                    str(task.engine_request_id),
                    {
                        "engine_prepare_queue_admission_wait_ms": float(task.admission_wait_ms),
                        "engine_prepare_audio_queue_wait_ms": float(queue_wait_ms),
                        "engine_prepare_audio_batch_size": float(len(tasks)),
                        "engine_prepare_audio_batch_limit": float(effective_batch_max_items),
                        "engine_prepare_audio_first_round": float(1.0 if is_first_round else 0.0),
                        "engine_prepare_audio_phase_wall_ms": float(result.get("phase_wall_ms", 0.0)),
                        "engine_prepare_audio_enqueue_ts": float(task.audio_enqueue_time),
                        "engine_prepare_audio_start_ts": float(task.audio_start_time),
                        "engine_prepare_audio_end_ts": float(task.audio_end_time),
                        "engine_prepare_text_enqueue_ts": float(task.text_enqueue_time),
                        "engine_prepare_ref_spec_enqueue_ts": float(task.ref_spec_enqueue_time),
                    },
                )
            completed_count += 1
        self.prepare_queue_owner.mark_completed(completed_count)
        if completed_count > 0 and self._should_chain_prepare_text_after_audio():
            self._run_engine_prepare_text_once(min(batch_max_items, completed_count))
            return True
        if completed_count > 0:
            self.notify_arbiter()
        return True

    def _run_engine_prepare_text_once(self, batch_max_items: int) -> bool:
        batch_max_items = self._engine_prepare_effective_batch_max_items("text", batch_max_items)
        tasks = self.prepare_text_queue_owner.pop_left_many(batch_max_items)
        if not tasks:
            return False
        now = time.perf_counter()
        queue_wait_ms_list = [max(0.0, (now - task.enqueue_time) * 1000.0) for task in tasks]
        for task in tasks:
            task.text_start_time = float(now)
        items = [(task.cpu_stage, task.phase_one) for task in tasks if task.phase_one is not None]
        batch_results = asyncio.run(self.scheduler_worker.prepare_gpu_text_phases_async(items))
        completed_count = 0
        for task, queue_wait_ms, result in zip(tasks, queue_wait_ms_list, batch_results):
            task.text_end_time = time.perf_counter()
            if isinstance(result, Exception):
                task.error = str(result)
                task.cancelled = True
                self.fail_request_state(task.engine_request_id or task.request_id, str(result))
                self._notify_prepare_error(task, result)
                completed_count += 1
                continue
            task.text_queue_wait_ms = float(queue_wait_ms)
            state, prepare_exec_started_at, prepare_exec_finished_at = self.scheduler_worker.build_gpu_prepare_result_from_phases(
                task.cpu_stage,
                task.phase_one or {},
                result,
                extra_profile={
                    "engine_prepare_queue_admission_wait_ms": float(task.admission_wait_ms),
                    "engine_prepare_audio_queue_wait_ms": float(task.audio_queue_wait_ms),
                    "engine_prepare_text_queue_wait_ms": float(task.text_queue_wait_ms),
                    "engine_gpu_prepare_queue_wait_ms": float(task.audio_queue_wait_ms + task.text_queue_wait_ms),
                    "engine_prepare_audio_batch_size": float(len(tasks)),
                    "engine_prepare_text_batch_size": float(len(tasks)),
                    "engine_prepare_audio_phase_mode": 2.0,
                    "engine_prepare_audio_phase_wall_ms": float((task.phase_one or {}).get("phase_wall_ms", 0.0)),
                    "engine_prepare_text_phase_wall_ms": float(result.get("phase_wall_ms", 0.0)),
                    "engine_prepare_text_phase_batch_size": float(len(tasks)),
                    "engine_prepare_audio_enqueue_ts": float(task.audio_enqueue_time),
                    "engine_prepare_audio_start_ts": float(task.audio_start_time),
                    "engine_prepare_audio_end_ts": float(task.audio_end_time),
                    "engine_prepare_text_enqueue_ts": float(task.text_enqueue_time),
                    "engine_prepare_text_start_ts": float(task.text_start_time),
                    "engine_prepare_text_end_ts": float(task.text_end_time),
                    "engine_prepare_ref_spec_enqueue_ts": float(task.ref_spec_enqueue_time),
                },
            )
            task.state_result = state
            self._maybe_apply_ref_spec_to_state(task)
            state.prepare_profile["engine_gpu_prepare_batch_size"] = float(len(tasks))
            if task.engine_request_id not in [None, ""]:
                self.merge_request_state_profile(
                    str(task.engine_request_id),
                    {
                        "engine_prepare_queue_admission_wait_ms": float(task.admission_wait_ms),
                        "engine_prepare_audio_queue_wait_ms": float(task.audio_queue_wait_ms),
                        "engine_prepare_text_queue_wait_ms": float(task.text_queue_wait_ms),
                        "engine_gpu_prepare_queue_wait_ms": float(task.audio_queue_wait_ms + task.text_queue_wait_ms),
                        "engine_gpu_prepare_batch_size": float(len(tasks)),
                    },
                )
            self._notify_prepare_result(task, (state, prepare_exec_started_at, prepare_exec_finished_at))
            completed_count += 1
        self.prepare_text_queue_owner.mark_completed(completed_count)
        return True

    def _run_engine_prepare_ref_spec_once(self, batch_max_items: int) -> bool:
        batch_max_items = self._engine_prepare_effective_batch_max_items("ref_spec", batch_max_items)
        tasks = self.prepare_ref_spec_queue_owner.pop_left_many(batch_max_items)
        if not tasks:
            return False
        now = time.perf_counter()
        runnable_tasks: list[EngineGpuPrepareTask] = []
        queue_wait_ms_list: list[float] = []
        completed_count = 0
        for task in tasks:
            if task.cancelled or task.phase_one is None:
                completed_count += 1
                continue
            task.ref_spec_start_time = float(now)
            runnable_tasks.append(task)
            queue_wait_ms_list.append(max(0.0, (now - task.ref_spec_enqueue_time) * 1000.0))
        if not runnable_tasks:
            self.prepare_ref_spec_queue_owner.mark_completed(completed_count)
            return True
        batch_results = asyncio.run(
            self.scheduler_worker.prepare_ref_spec_stages_async([task.phase_one or {} for task in runnable_tasks])
        )
        for task, queue_wait_ms, result in zip(runnable_tasks, queue_wait_ms_list, batch_results):
            task.ref_spec_end_time = time.perf_counter()
            task.ref_spec_queue_wait_ms = float(queue_wait_ms)
            if isinstance(result, Exception):
                self._mark_ref_spec_async_failed(task, result, queue_wait_ms=float(queue_wait_ms))
                completed_count += 1
                continue
            task.ref_spec_result = result
            self._maybe_apply_ref_spec_to_state(task)
            if task.state_result is not None:
                task.state_result.prepare_profile["engine_prepare_ref_spec_queue_wait_ms"] = float(queue_wait_ms)
                task.state_result.prepare_profile["engine_prepare_ref_spec_enqueue_ts"] = float(task.ref_spec_enqueue_time)
                task.state_result.prepare_profile["engine_prepare_ref_spec_start_ts"] = float(task.ref_spec_start_time)
                task.state_result.prepare_profile["engine_prepare_ref_spec_end_ts"] = float(task.ref_spec_end_time)
            completed_count += 1
        self.prepare_ref_spec_queue_owner.mark_completed(completed_count)
        return True

    def run_engine_prepare_once(self) -> bool:
        prepare_batch_policy = self.scheduler_worker.get_prepare_batch_policy()
        batch_max_items = int(prepare_batch_policy.get("prepare_batch_max_items", 1))
        if self._engine_prepare_audio_first_enabled() and self.prepare_queue_owner.has_items():
            return self._run_engine_prepare_audio_once(batch_max_items)
        audio_age_ms = self.prepare_queue_owner.peek_oldest_age_ms("enqueue_time")
        text_age_ms = self.prepare_text_queue_owner.peek_oldest_age_ms("enqueue_time")
        if self.prepare_text_queue_owner.has_items() and (
            not self.prepare_queue_owner.has_items() or text_age_ms >= audio_age_ms
        ):
            return self._run_engine_prepare_text_once(batch_max_items)
        if self.prepare_queue_owner.has_items():
            return self._run_engine_prepare_audio_once(batch_max_items)
        if self.prepare_ref_spec_queue_owner.has_items():
            return self._run_engine_prepare_ref_spec_once(batch_max_items)
        if self.prepare_text_queue_owner.has_items():
            return self._run_engine_prepare_text_once(batch_max_items)
        if self.prepare_ref_spec_queue_owner.has_items():
            return self._run_engine_prepare_ref_spec_once(batch_max_items)
        return False

    def run_engine_prepare_audio_once(self) -> bool:
        prepare_batch_policy = self.scheduler_worker.get_prepare_batch_policy()
        return self._run_engine_prepare_audio_once(int(prepare_batch_policy.get("prepare_batch_max_items", 1)))

    def run_engine_prepare_text_once(self) -> bool:
        prepare_batch_policy = self.scheduler_worker.get_prepare_batch_policy()
        return self._run_engine_prepare_text_once(int(prepare_batch_policy.get("prepare_batch_max_items", 1)))

    def run_engine_prepare_ref_spec_once(self) -> bool:
        prepare_batch_policy = self.scheduler_worker.get_prepare_batch_policy()
        return self._run_engine_prepare_ref_spec_once(int(prepare_batch_policy.get("prepare_batch_max_items", 1)))
