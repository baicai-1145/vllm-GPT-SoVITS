from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import (
    PreparedTextFeatures,
    SchedulerRequestSpec,
    T2SRequestState,
    build_empty_text_features,
    build_request_state_from_parts,
    normalize_sentence,
)


@dataclass
class ProfiledResult:
    result: Any
    submit_at: float
    started_at: float
    finished_at: float
    profile: Dict[str, float] | None = None

    @property
    def queue_ms(self) -> float:
        return max(0.0, (self.started_at - self.submit_at) * 1000.0)

    @property
    def run_ms(self) -> float:
        return max(0.0, (self.finished_at - self.started_at) * 1000.0)


@dataclass
class PreparedCpuStage:
    spec: SchedulerRequestSpec
    prepare_submit_at: float
    prepare_start: float
    prompt_text: str
    text: str
    prepare_admission_wait_ms: float
    current_inflight: int
    peak_inflight: int
    prompt_cpu_profiled: ProfiledResult
    target_cpu_profiled: ProfiledResult


@dataclass
class PreparedRefAudioAsset:
    raw_audio: Any
    raw_sr: int
    wav16k: Any
    profile: Dict[str, float] = field(default_factory=dict)


class AsyncStageGate:
    def __init__(self, max_inflight: int, poll_ms: int = 1):
        self.max_inflight = max(0, int(max_inflight))
        self.lock = threading.Lock()
        self.poll_s = max(0.0005, float(max(1, int(poll_ms))) / 1000.0)
        self.inflight = 0
        self.peak_inflight = 0
        self.total_entered = 0
        self.total_wait_ms = 0.0
        self.wait_peak_ms = 0.0

    async def acquire(self) -> Dict[str, float]:
        wait_start = time.perf_counter()
        while True:
            with self.lock:
                if self.max_inflight <= 0 or self.inflight < self.max_inflight:
                    self.inflight += 1
                    self.total_entered += 1
                    wait_ms = max(0.0, (time.perf_counter() - wait_start) * 1000.0)
                    self.total_wait_ms += float(wait_ms)
                    self.wait_peak_ms = max(self.wait_peak_ms, float(wait_ms))
                    self.peak_inflight = max(self.peak_inflight, self.inflight)
                    return {
                        "wait_ms": float(wait_ms),
                        "inflight": float(self.inflight),
                        "peak_inflight": float(self.peak_inflight),
                        "max_inflight": float(self.max_inflight),
                    }
            await asyncio.sleep(self.poll_s)

    def release(self) -> None:
        with self.lock:
            self.inflight = max(0, self.inflight - 1)

    def snapshot(self) -> Dict[str, float]:
        with self.lock:
            return {
                "max_inflight": float(self.max_inflight),
                "inflight": float(self.inflight),
                "peak_inflight": float(self.peak_inflight),
                "total_entered": float(self.total_entered),
                "total_wait_ms": float(self.total_wait_ms),
                "wait_peak_ms": float(self.wait_peak_ms),
            }


class RuntimePrepareCoordinatorAdapter:
    def __init__(self, tts: Any, runtime_owner: Any):
        self.tts = tts
        self.runtime_owner = runtime_owner

    def _native(self) -> Any:
        return self.runtime_owner._ensure_prepare_coordinator()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native(), name)

    def snapshot(self) -> Dict[str, Any]:
        coordinator = self._native()
        runtime_state = getattr(self.tts, "snapshot_prepare_runtime_components", lambda: None)()
        g2pw_state = runtime_state.get("g2pw") if isinstance(runtime_state, dict) else None
        g2pw_runtime_workers = None
        if isinstance(g2pw_state, dict):
            worker_count = g2pw_state.get("worker_count")
            try:
                if worker_count is not None:
                    g2pw_runtime_workers = max(1, int(worker_count))
            except Exception:
                g2pw_runtime_workers = None

        def _executor_workers(executor: Any) -> int:
            max_workers = getattr(executor, "_max_workers", 0)
            try:
                return max(0, int(max_workers or 0))
            except Exception:
                return 0

        def _gate_snapshot(gate: Any) -> Dict[str, float]:
            if hasattr(gate, "snapshot"):
                try:
                    return dict(gate.snapshot())
                except Exception:
                    return {}
            return {}

        cache = dict(getattr(coordinator, "ref_audio_asset_cache", {}) or {})
        inflight = dict(getattr(coordinator, "ref_audio_asset_inflight", {}) or {})
        return {
            "inflight": int(getattr(coordinator, "inflight", 0) or 0),
            "peak_inflight": int(getattr(coordinator, "peak_inflight", 0) or 0),
            "max_inflight": int(getattr(getattr(coordinator, "inflight_gate", None), "max_inflight", 0) or 0),
            "text_feature_workers": int(_executor_workers(getattr(coordinator, "text_feature_executor", None))),
            "g2pw_workers": int(_executor_workers(getattr(coordinator, "g2pw_executor", None))),
            "g2pw_runtime_workers": None if g2pw_runtime_workers is None else int(g2pw_runtime_workers),
            "ref_audio_workers": int(_executor_workers(getattr(coordinator, "ref_audio_executor", None))),
            "ref_audio_asset_cache": {
                "ttl_sec": float(getattr(coordinator, "ref_audio_asset_cache_ttl_sec", 0.0) or 0.0),
                "max_entries": int(getattr(coordinator, "ref_audio_asset_cache_max_entries", 0) or 0),
                "cached_entries": int(len(cache)),
                "inflight_entries": int(len(inflight)),
            },
            "prepare_runtime_state": runtime_state,
            "prepare_stage_gates": {
                "text_cpu": _gate_snapshot(getattr(coordinator, "text_cpu_gate", None)),
                "g2pw": _gate_snapshot(getattr(coordinator, "g2pw_gate", None)),
                "text_feature": _gate_snapshot(getattr(coordinator, "text_feature_gate", None)),
                "ref_audio": _gate_snapshot(getattr(coordinator, "ref_audio_gate", None)),
                "ref_load": _gate_snapshot(getattr(coordinator, "ref_load_gate", None)),
                "ref_spec": _gate_snapshot(getattr(coordinator, "ref_spec_gate", None)),
            },
        }

    @staticmethod
    def _phase_one_to_dict(phase_one: Any) -> Dict[str, Any]:
        if isinstance(phase_one, dict):
            return dict(phase_one)
        return {
            "prompt_g2pw_profiled": getattr(phase_one, "prompt_g2pw_profiled"),
            "target_g2pw_profiled": getattr(phase_one, "target_g2pw_profiled"),
            "ref_audio_profiled": getattr(phase_one, "ref_audio_profiled"),
            "ref_spec_result": None,
            "g2pw_pair_ms": float(getattr(phase_one, "g2pw_pair_ms", 0.0) or 0.0),
            "phase_wall_ms": float(getattr(phase_one, "phase_wall_ms", 0.0) or 0.0),
        }

    @staticmethod
    def _phase_two_to_dict(phase_two: Any) -> Dict[str, Any]:
        if isinstance(phase_two, dict):
            return dict(phase_two)
        return {
            "prompt_feature_profiled": getattr(phase_two, "prompt_feature_profiled"),
            "target_feature_profiled": getattr(phase_two, "target_feature_profiled"),
            "phase_wall_ms": float(getattr(phase_two, "phase_wall_ms", 0.0) or 0.0),
        }

    def _phase_one_to_native(self, phase_one: Any) -> tuple[Any, Any]:
        if not isinstance(phase_one, dict):
            return phase_one, None
        with self.runtime_owner._project_root_cwd():
            from vllm_omni.model_executor.models.gpt_sovits.runtime import GPTSoVITSPrepareAudioPhaseData

        return (
            GPTSoVITSPrepareAudioPhaseData(
                prompt_g2pw_profiled=phase_one["prompt_g2pw_profiled"],
                target_g2pw_profiled=phase_one["target_g2pw_profiled"],
                ref_audio_profiled=phase_one["ref_audio_profiled"],
                g2pw_pair_ms=float(phase_one.get("g2pw_pair_ms", 0.0) or 0.0),
                phase_wall_ms=float(phase_one.get("phase_wall_ms", 0.0) or 0.0),
            ),
            phase_one.get("ref_spec_result"),
        )

    def _phase_two_to_native(self, phase_two: Any) -> Any:
        if not isinstance(phase_two, dict):
            return phase_two
        with self.runtime_owner._project_root_cwd():
            from vllm_omni.model_executor.models.gpt_sovits.runtime import GPTSoVITSPrepareTextPhaseData

        return GPTSoVITSPrepareTextPhaseData(
            prompt_feature_profiled=phase_two["prompt_feature_profiled"],
            target_feature_profiled=phase_two["target_feature_profiled"],
            phase_wall_ms=float(phase_two.get("phase_wall_ms", 0.0) or 0.0),
        )

    async def prepare_cpu_stage_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> Any:
        return await self.runtime_owner._prepare_cpu_stage_async(
            self.runtime_owner._ensure_prepare_coordinator(),
            spec,
            prepare_submit_at=prepare_submit_at,
        )

    async def prepare_gpu_audio_phases_async(
        self,
        cpu_stages: list[Any],
        prepared_ref_audio_futures: list[concurrent.futures.Future | None] | None = None,
    ) -> list[Dict[str, Any] | Exception]:
        if not cpu_stages:
            return []
        coordinator = self.runtime_owner._ensure_prepare_coordinator()
        if prepared_ref_audio_futures is not None:
            for cpu_stage, prepared_ref_audio_future in zip(cpu_stages, prepared_ref_audio_futures):
                if prepared_ref_audio_future is not None and hasattr(cpu_stage, "ref_audio_prepare_future"):
                    setattr(cpu_stage, "ref_audio_prepare_future", prepared_ref_audio_future)
        if coordinator.enable_g2pw_audio_batch_merge and len(cpu_stages) > 1:
            phase_ones = await self.runtime_owner._prepare_gpu_audio_phase_batch_async(coordinator, cpu_stages)
            return [self._phase_one_to_dict(phase_one) if not isinstance(phase_one, Exception) else phase_one for phase_one in phase_ones]
        phase_ones = await asyncio.gather(
            *[
                self.runtime_owner._prepare_gpu_audio_phase_async(coordinator, cpu_stage)
                for cpu_stage in cpu_stages
            ],
            return_exceptions=True,
        )
        return [self._phase_one_to_dict(phase_one) if not isinstance(phase_one, Exception) else phase_one for phase_one in phase_ones]

    async def prepare_gpu_text_phases_async(
        self,
        items: list[tuple[Any, Dict[str, Any]]],
    ) -> list[Dict[str, Any] | Exception]:
        if not items:
            return []
        coordinator = self.runtime_owner._ensure_prepare_coordinator()
        with self.runtime_owner._project_root_cwd():
            from vllm_omni.model_executor.models.gpt_sovits.runtime import GPTSoVITSPreparedAudioPhase

        phase_twos = await asyncio.gather(
            *[
                self.runtime_owner._prepare_gpu_text_phase_async(
                    coordinator,
                    GPTSoVITSPreparedAudioPhase(
                        request_id=str(getattr(cpu_stage, "request_id", getattr(getattr(cpu_stage, "spec", None), "request_id", ""))),
                        prepared_cpu_stage=cpu_stage,
                        phase_one=self._phase_one_to_native(phase_one)[0],
                    ),
                )
                for cpu_stage, phase_one in items
            ],
            return_exceptions=True,
        )
        return [self._phase_two_to_dict(phase_two) if not isinstance(phase_two, Exception) else phase_two for phase_two in phase_twos]

    async def prepare_ref_spec_stages_async(
        self,
        phase_ones: list[Dict[str, Any]],
    ) -> list[tuple[tuple[Any, Any], Dict[str, float]] | Exception]:
        if not phase_ones:
            return []
        coordinator = self.runtime_owner._ensure_prepare_coordinator()

        async def _one(phase_one: Dict[str, Any]):
            ref_audio_profiled = phase_one["ref_audio_profiled"]
            raw_audio = ref_audio_profiled.result.raw_audio
            raw_sr = int(ref_audio_profiled.result.raw_sr)
            profiled = await self.runtime_owner._run_ref_spec_stage(coordinator, raw_audio, raw_sr)
            refer_spec_raw, profile = profiled.result
            refer_spec = self.runtime_owner._coerce_refer_spec(refer_spec_raw)
            merged_profile = dict(profile)
            merged_profile["ref_spec_wait_ms"] = float(profiled.queue_ms)
            merged_profile["ref_spec_ms"] = float(profiled.run_ms)
            return (refer_spec.spec_audio, refer_spec.audio_16k), merged_profile

        return list(await asyncio.gather(*[_one(phase_one) for phase_one in phase_ones], return_exceptions=True))

    def apply_ref_spec_result_to_state(
        self,
        state: T2SRequestState,
        ref_spec_result: tuple[tuple[Any, Any], Dict[str, float]],
    ) -> None:
        refer_spec, profile = ref_spec_result
        state.refer_spec = refer_spec
        state.prepare_profile["ref_spec_wait_ms"] = float(profile.get("ref_spec_wait_ms", 0.0))
        state.prepare_profile["ref_spec_ms"] = float(profile.get("ref_spec_ms", 0.0))
        state.prepare_profile["ref_spec_to_device_ms"] = float(profile.get("ref_spec_to_device_ms", 0.0))
        state.prepare_profile["ref_spec_main_resample_ms"] = float(profile.get("ref_spec_main_resample_ms", 0.0))
        state.prepare_profile["ref_spec_norm_ms"] = float(profile.get("ref_spec_norm_ms", 0.0))
        state.prepare_profile["ref_spec_spectrogram_ms"] = float(profile.get("ref_spec_spectrogram_ms", 0.0))
        state.prepare_profile["ref_spec_post_resample_ms"] = float(profile.get("ref_spec_post_resample_ms", 0.0))

    def build_gpu_prepare_result_from_phases(
        self,
        cpu_stage: Any,
        phase_one: Dict[str, Any],
        phase_two: Dict[str, Any],
        extra_profile: Dict[str, float] | None = None,
    ) -> tuple[T2SRequestState, float, float]:
        coordinator = self.runtime_owner._ensure_prepare_coordinator()
        native_phase_one, ref_spec_result = self._phase_one_to_native(phase_one)
        native_phase_two = self._phase_two_to_native(phase_two)
        try:
            state = self.runtime_owner._build_request_state_from_prepare_phases(
                cpu_stage,
                native_phase_one,
                native_phase_two,
                ref_spec_result=ref_spec_result,
                extra_profile=extra_profile,
            )
        finally:
            self.runtime_owner._release_prepare_split_stage_slot(coordinator)
        prepare_exec_finished_at = time.perf_counter()
        return state, float(cpu_stage.prepare_start), float(prepare_exec_finished_at)

    async def prepare_gpu_stage_profiled_async(
        self,
        cpu_stage: Any,
    ) -> tuple[T2SRequestState, float, float]:
        try:
            phase_one = (await self.prepare_gpu_audio_phases_async([cpu_stage]))[0]
            if isinstance(phase_one, Exception):
                raise phase_one
            phase_two = (await self.prepare_gpu_text_phases_async([(cpu_stage, phase_one)]))[0]
            if isinstance(phase_two, Exception):
                raise phase_two
            return self.build_gpu_prepare_result_from_phases(
                cpu_stage,
                phase_one,
                phase_two,
                extra_profile={
                    "engine_prepare_audio_phase_mode": 0.0,
                    "engine_prepare_audio_phase_wall_ms": float(phase_one.get("phase_wall_ms", 0.0)),
                    "engine_prepare_audio_phase_batch_size": 1.0,
                    "engine_prepare_text_phase_wall_ms": float(phase_two.get("phase_wall_ms", 0.0)),
                    "engine_prepare_text_phase_batch_size": 1.0,
                },
            )
        except Exception:
            self.runtime_owner._release_prepare_split_stage_slot(self.runtime_owner._ensure_prepare_coordinator())
            raise

    async def prepare_gpu_stages_profiled_async(
        self,
        cpu_stages: list[Any],
    ) -> list[tuple[T2SRequestState, float, float] | Exception]:
        if not cpu_stages:
            return []
        phase_ones = await self.prepare_gpu_audio_phases_async(cpu_stages)
        items: list[tuple[Any, Dict[str, Any]]] = []
        outputs: list[tuple[T2SRequestState, float, float] | Exception | None] = [None] * len(cpu_stages)
        for index, (cpu_stage, phase_one) in enumerate(zip(cpu_stages, phase_ones)):
            if isinstance(phase_one, Exception):
                outputs[index] = phase_one
                self.runtime_owner._release_prepare_split_stage_slot(self.runtime_owner._ensure_prepare_coordinator())
                continue
            items.append((cpu_stage, phase_one))
        phase_twos = await self.prepare_gpu_text_phases_async(items)
        item_iter = iter(items)
        for index, phase_one in enumerate(phase_ones):
            if isinstance(phase_one, Exception):
                continue
            cpu_stage, phase_one_dict = next(item_iter)
            phase_two = phase_twos.pop(0)
            if isinstance(phase_two, Exception):
                outputs[index] = phase_two
                self.runtime_owner._release_prepare_split_stage_slot(self.runtime_owner._ensure_prepare_coordinator())
                continue
            try:
                outputs[index] = self.build_gpu_prepare_result_from_phases(
                    cpu_stage,
                    phase_one_dict,
                    phase_two,
                    extra_profile={
                        "engine_prepare_audio_phase_mode": 1.0,
                        "engine_prepare_audio_phase_wall_ms": float(phase_one_dict.get("phase_wall_ms", 0.0)),
                        "engine_prepare_audio_phase_batch_size": float(len(cpu_stages)),
                        "engine_prepare_text_phase_wall_ms": float(phase_two.get("phase_wall_ms", 0.0)),
                        "engine_prepare_text_phase_batch_size": float(len(items)),
                    },
                )
            except Exception as exc:  # noqa: PERF203
                outputs[index] = exc
        return [item if item is not None else RuntimeError("prepare batch result missing") for item in outputs]

    async def prepare_state_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> tuple[T2SRequestState, float, float]:
        cpu_stage = await self.prepare_cpu_stage_profiled_async(spec, prepare_submit_at)
        return await self.prepare_gpu_stage_profiled_async(cpu_stage)

    async def prepare_direct_shared_segments_profiled_async(
        self,
        specs: list[SchedulerRequestSpec],
    ) -> list[tuple[T2SRequestState, float, float] | Exception]:
        if not specs:
            return []
        native_shared_prepare = getattr(self.runtime_owner, "_prepare_direct_shared_segments_async", None)
        if callable(native_shared_prepare):
            return await native_shared_prepare(
                self.runtime_owner._ensure_prepare_coordinator(),
                specs,
            )
        return list(
            await asyncio.gather(
                *[
                    self.prepare_state_profiled_async(spec, time.perf_counter())
                    for spec in specs
                ],
                return_exceptions=True,
            )
        )

    def submit_prepare_ref_audio_asset(
        self,
        ref_audio_path: str,
        *,
        submit_at: float | None = None,
    ) -> concurrent.futures.Future:
        return self.runtime_owner.preload_ref_audio_asset(ref_audio_path, submit_at=submit_at)


def build_prepare_coordinator(tts: Any) -> Any:
    runtime_owner = getattr(tts, "_vllm_runtime_owner", None)
    if runtime_owner is not None and callable(getattr(tts, "_vllm_runtime_prepare_coordinator_factory", None)):
        return RuntimePrepareCoordinatorAdapter(tts, runtime_owner)
    return PrepareCoordinator(tts)


class PrepareCoordinator:
    @staticmethod
    def _resolve_prepare_runtime_state(tts: Any) -> dict | None:
        state_provider = getattr(tts, "_vllm_runtime_prepare_state_provider", None)
        if callable(state_provider):
            try:
                runtime_state = state_provider()
            except Exception:
                runtime_state = None
            if isinstance(runtime_state, dict):
                return runtime_state
        snapshot_fn = getattr(tts, "snapshot_prepare_runtime_components", None)
        if not callable(snapshot_fn):
            return None
        try:
            runtime_state = snapshot_fn()
        except Exception:
            return None
        return runtime_state if isinstance(runtime_state, dict) else None

    @staticmethod
    def _detect_g2pw_runtime_workers(tts: Any) -> int | None:
        runtime_state = PrepareCoordinator._resolve_prepare_runtime_state(tts)
        if runtime_state is None:
            return None
        g2pw_state = runtime_state.get("g2pw")
        if not isinstance(g2pw_state, dict):
            return None
        worker_count = g2pw_state.get("worker_count")
        try:
            worker_count = int(worker_count)
        except Exception:
            return None
        return max(1, worker_count)

    def __init__(self, tts: Any):
        self.tts = tts
        self.lock = threading.Lock()
        self.inflight = 0
        self.peak_inflight = 0
        self.ref_audio_asset_cache_ttl_sec = max(
            0.0,
            float(os.environ.get("GPTSOVITS_PREPARE_REF_AUDIO_ASSET_CACHE_TTL_SEC", "15")),
        )
        self.ref_audio_asset_cache_max_entries = max(
            0,
            int(os.environ.get("GPTSOVITS_PREPARE_REF_AUDIO_ASSET_CACHE_MAX_ENTRIES", "4")),
        )
        self.ref_audio_asset_lock = threading.Lock()
        self.ref_audio_asset_inflight: Dict[str, concurrent.futures.Future] = {}
        self.ref_audio_asset_cache: Dict[str, Tuple[PreparedRefAudioAsset, float]] = {}
        self.use_async_text_feature_path = bool(
            getattr(tts, "prepare_bert_batch_worker", None) is not None
            and os.environ.get("GPTSOVITS_PREPARE_TEXT_FEATURE_DIRECT", "0") != "0"
        )
        self.max_inflight = max(0, int(os.environ.get("GPTSOVITS_PREPARE_MAX_INFLIGHT", "0")))
        gate_poll_ms = int(os.environ.get("GPTSOVITS_PREPARE_GATE_POLL_MS", "1"))
        self._inflight_gate = AsyncStageGate(self.max_inflight, poll_ms=gate_poll_ms)
        self.text_feature_workers = 0
        self.text_feature_executor = None
        if not self.use_async_text_feature_path:
            text_feature_default_workers = 16
            self.text_feature_workers = max(
                1,
                int(os.environ.get("GPTSOVITS_PREPARE_TEXT_FEATURE_WORKERS", str(text_feature_default_workers))),
            )
            self.text_feature_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.text_feature_workers,
                thread_name_prefix="prepare-text-feature",
            )
        g2pw_runtime_workers = self._detect_g2pw_runtime_workers(tts)
        self.g2pw_runtime_workers = int(g2pw_runtime_workers) if g2pw_runtime_workers is not None else None
        g2pw_default_workers = (
            int(self.g2pw_runtime_workers)
            if self.g2pw_runtime_workers is not None
            else max(8, int(getattr(tts, "prepare_text_cpu_workers", 8) or 8))
        )
        self.g2pw_workers = max(
            1,
            int(os.environ.get("GPTSOVITS_PREPARE_G2PW_WORKERS", str(g2pw_default_workers))),
        )
        self.enable_g2pw_pair_batch = os.environ.get("GPTSOVITS_PREPARE_G2PW_PAIR_BATCH", "1") != "0"
        self.enable_g2pw_audio_batch_merge = os.environ.get("GPTSOVITS_PREPARE_G2PW_AUDIO_BATCH_MERGE", "0") != "0"
        self.g2pw_audio_batch_merge_group_size = max(
            1,
            int(os.environ.get("GPTSOVITS_PREPARE_G2PW_AUDIO_BATCH_GROUP_SIZE", "8")),
        )
        self.g2pw_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.g2pw_workers,
            thread_name_prefix="prepare-g2pw",
        )
        ref_audio_default_workers = max(1, int(os.environ.get("GPTSOVITS_PREPARE_REF_SLOTS", "4")))
        self.ref_audio_workers = max(
            1,
            int(os.environ.get("GPTSOVITS_PREPARE_REF_ASYNC_WORKERS", str(ref_audio_default_workers))),
        )
        self.ref_audio_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.ref_audio_workers,
            thread_name_prefix="prepare-ref-audio",
        )
        # text CPU stage now relies on the worker's own pending queue for cross-request batching.
        # Keeping this gate coupled to worker_count would cap queue formation and defeat batching.
        text_cpu_gate_default = 0
        g2pw_gate_default = (
            int(self.g2pw_runtime_workers)
            if self.g2pw_runtime_workers is not None
            else max(0, int(self.g2pw_workers))
        )
        text_feature_gate_default = max(0, int(self.text_feature_workers))
        ref_audio_gate_default = max(0, int(self.ref_audio_workers))
        self.text_cpu_gate = AsyncStageGate(
            int(os.environ.get("GPTSOVITS_PREPARE_TEXT_CPU_MAX_INFLIGHT", str(text_cpu_gate_default))),
            poll_ms=gate_poll_ms,
        )
        self.g2pw_gate = AsyncStageGate(
            int(os.environ.get("GPTSOVITS_PREPARE_G2PW_MAX_INFLIGHT", str(g2pw_gate_default))),
            poll_ms=gate_poll_ms,
        )
        self.text_feature_gate = AsyncStageGate(
            int(os.environ.get("GPTSOVITS_PREPARE_TEXT_FEATURE_MAX_INFLIGHT", str(text_feature_gate_default))),
            poll_ms=gate_poll_ms,
        )
        self.ref_audio_gate = AsyncStageGate(
            int(os.environ.get("GPTSOVITS_PREPARE_REF_MAX_INFLIGHT", str(ref_audio_gate_default))),
            poll_ms=gate_poll_ms,
        )
        self.ref_load_gate = AsyncStageGate(
            int(os.environ.get("GPTSOVITS_PREPARE_REF_LOAD_MAX_INFLIGHT", str(ref_audio_gate_default))),
            poll_ms=gate_poll_ms,
        )
        self.ref_spec_gate = AsyncStageGate(
            int(os.environ.get("GPTSOVITS_PREPARE_REF_SPEC_MAX_INFLIGHT", str(ref_audio_gate_default))),
            poll_ms=gate_poll_ms,
        )

    def _mark_enter(self) -> Tuple[int, int]:
        with self.lock:
            self.inflight += 1
            current_inflight = self.inflight
            if current_inflight > self.peak_inflight:
                self.peak_inflight = current_inflight
            return current_inflight, self.peak_inflight

    def _mark_leave(self) -> None:
        with self.lock:
            self.inflight = max(0, self.inflight - 1)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            snapshot: Dict[str, Any] = {
                "inflight": int(self.inflight),
                "peak_inflight": int(self.peak_inflight),
                "max_inflight": int(self.max_inflight),
                "text_feature_workers": int(self.text_feature_workers),
                "g2pw_workers": int(self.g2pw_workers),
                "g2pw_runtime_workers": (
                    None if self.g2pw_runtime_workers is None else int(self.g2pw_runtime_workers)
                ),
                "ref_audio_workers": int(self.ref_audio_workers),
            }
        with self.ref_audio_asset_lock:
            now = time.perf_counter()
            self._prune_ref_audio_asset_cache_locked(now)
            snapshot["ref_audio_asset_cache"] = {
                "ttl_sec": float(self.ref_audio_asset_cache_ttl_sec),
                "max_entries": int(self.ref_audio_asset_cache_max_entries),
                "cached_entries": int(len(self.ref_audio_asset_cache)),
                "inflight_entries": int(len(self.ref_audio_asset_inflight)),
            }
        snapshot["prepare_runtime_state"] = self._resolve_prepare_runtime_state(self.tts)
        snapshot["prepare_stage_gates"] = {
            "text_cpu": self.text_cpu_gate.snapshot(),
            "g2pw": self.g2pw_gate.snapshot(),
            "text_feature": self.text_feature_gate.snapshot(),
            "ref_audio": self.ref_audio_gate.snapshot(),
            "ref_load": self.ref_load_gate.snapshot(),
            "ref_spec": self.ref_spec_gate.snapshot(),
        }
        return snapshot

    @staticmethod
    def _run_profiled(fn, submit_at: float, *args) -> ProfiledResult:
        started_at = time.perf_counter()
        result = fn(*args)
        finished_at = time.perf_counter()
        return ProfiledResult(
            result=result,
            submit_at=float(submit_at),
            started_at=float(started_at),
            finished_at=float(finished_at),
        )

    def _prepare_text_cpu(self, text: str, language: str):
        return self.tts.prepare_text_segments(text, language)

    def _resolve_g2pw_segments(self, prepared_segments):
        profile: Dict[str, float] = {}
        resolved_segments = self.tts.resolve_g2pw_segments(prepared_segments, profile=profile)
        return resolved_segments, profile

    def _resolve_g2pw_segment_batches(self, prepared_segment_batches):
        profiles: List[Dict[str, float]] = [{} for _ in prepared_segment_batches]
        resolved_batches = self.tts.resolve_g2pw_segments_batch(prepared_segment_batches, profiles=profiles)
        return resolved_batches, profiles

    def _load_ref_audio_raw(self, ref_audio_path: str):
        return self.tts._load_ref_audio_raw(ref_audio_path)

    def _prepare_ref_audio_asset(self, ref_audio_path: str) -> PreparedRefAudioAsset:
        load_start = time.perf_counter()
        raw_audio, raw_sr = self._load_ref_audio_raw(ref_audio_path)
        load_ms = (time.perf_counter() - load_start) * 1000.0
        wav16k, cpu_prepare_ms, limiter_stats = self.tts._prepare_prompt_semantic_wav16k_profile(raw_audio, raw_sr)
        return PreparedRefAudioAsset(
            raw_audio=raw_audio,
            raw_sr=int(raw_sr),
            wav16k=wav16k,
            profile={
                "audio_load_ms": float(load_ms),
                "prompt_semantic_cpu_prepare_ms": float(cpu_prepare_ms),
                "prompt_semantic_cpu_prepare_wait_ms": float(limiter_stats.get("wait_ms", 0.0)),
                "prompt_semantic_cpu_prepare_slots": float(limiter_stats.get("slots", 0.0)),
                "prompt_semantic_cpu_prepare_inflight_peak": float(limiter_stats.get("peak_inflight", 0.0)),
                "prepared_ref_audio_cache_hit": 0.0,
                "prepared_ref_audio_cache_age_ms": 0.0,
            },
        )

    @staticmethod
    def _normalize_ref_audio_cache_key(ref_audio_path: str) -> str:
        return os.path.abspath(str(ref_audio_path))

    @staticmethod
    def _clone_prepared_ref_audio_asset(
        asset: PreparedRefAudioAsset,
        *,
        extra_profile: Dict[str, float] | None = None,
    ) -> PreparedRefAudioAsset:
        profile = dict(asset.profile or {})
        if extra_profile:
            profile.update({key: float(value) for key, value in extra_profile.items()})
        return PreparedRefAudioAsset(
            raw_audio=asset.raw_audio,
            raw_sr=int(asset.raw_sr),
            wav16k=asset.wav16k,
            profile=profile,
        )

    def _prune_ref_audio_asset_cache_locked(self, now: float | None = None) -> None:
        now_ts = time.perf_counter() if now is None else float(now)
        ttl_sec = float(self.ref_audio_asset_cache_ttl_sec)
        if ttl_sec <= 0.0 or self.ref_audio_asset_cache_max_entries <= 0:
            self.ref_audio_asset_cache.clear()
            return
        expired_keys = [
            key
            for key, (_, cached_at) in self.ref_audio_asset_cache.items()
            if (now_ts - float(cached_at)) > ttl_sec
        ]
        for key in expired_keys:
            self.ref_audio_asset_cache.pop(key, None)
        overflow = len(self.ref_audio_asset_cache) - int(self.ref_audio_asset_cache_max_entries)
        if overflow > 0:
            oldest_keys = sorted(
                self.ref_audio_asset_cache.items(),
                key=lambda item: float(item[1][1]),
            )[:overflow]
            for key, _ in oldest_keys:
                self.ref_audio_asset_cache.pop(key, None)

    def _build_ref_audio_asset_cache_hit_future(
        self,
        asset: PreparedRefAudioAsset,
        *,
        submit_ts: float,
        cache_age_ms: float,
    ) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        future.set_result(
            ProfiledResult(
                result=self._clone_prepared_ref_audio_asset(
                    asset,
                    extra_profile={
                        "prepared_ref_audio_cache_hit": 1.0,
                        "prepared_ref_audio_cache_age_ms": float(cache_age_ms),
                    },
                ),
                submit_at=float(submit_ts),
                started_at=float(submit_ts),
                finished_at=float(submit_ts),
            )
        )
        return future

    def _build_ref_prompt_semantic_from_raw(self, raw_audio, raw_sr: int):
        load_profile = {"audio_load_ms": 0.0}
        if getattr(self.tts, "prepare_ref_semantic_batch_worker", None) is not None:
            wav16k, local_cpu_prepare_profile = self.tts._prepare_ref_prompt_wav16k_for_worker(raw_audio, raw_sr)
            prompt_semantic, worker_profile = self.tts.prepare_ref_semantic_batch_worker.submit(
                raw_audio,
                raw_sr,
                wav16k=wav16k,
            )
            return {
                "prompt_semantic": prompt_semantic,
                "raw_audio": raw_audio,
                "raw_sr": raw_sr,
                "profile": {
                    **load_profile,
                    "audio_stage_wait_ms": float(worker_profile.get("prompt_semantic_wait_ms", 0.0)),
                    "audio_stage_slots": float(worker_profile.get("prompt_semantic_stage_slots", 0.0)),
                    "audio_stage_inflight_peak": float(worker_profile.get("prompt_semantic_stage_inflight_peak", 0.0)),
                    "prompt_semantic_ms": float(
                        worker_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)
                        + local_cpu_prepare_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)
                        + worker_profile.get("prompt_semantic_forward_ms", 0.0)
                        + worker_profile.get("prompt_semantic_scatter_ms", 0.0)
                    ),
                    **{key: float(value) for key, value in worker_profile.items()},
                    "prompt_semantic_cpu_prepare_ms": float(
                        worker_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)
                        + local_cpu_prepare_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)
                    ),
                    "prompt_semantic_cpu_prepare_wait_ms": float(
                        worker_profile.get("prompt_semantic_cpu_prepare_wait_ms", 0.0)
                        + local_cpu_prepare_profile.get("prompt_semantic_cpu_prepare_wait_ms", 0.0)
                    ),
                    "prompt_semantic_cpu_prepare_slots": float(
                        max(
                            worker_profile.get("prompt_semantic_cpu_prepare_slots", 0.0),
                            local_cpu_prepare_profile.get("prompt_semantic_cpu_prepare_slots", 0.0),
                        )
                    ),
                    "prompt_semantic_cpu_prepare_inflight_peak": float(
                        max(
                            worker_profile.get("prompt_semantic_cpu_prepare_inflight_peak", 0.0),
                            local_cpu_prepare_profile.get("prompt_semantic_cpu_prepare_inflight_peak", 0.0),
                        )
                    ),
                    "ref_spec_wait_ms": 0.0,
                    "ref_spec_ms": 0.0,
                    "bundle_total_ms": float(worker_profile.get("prompt_semantic_wait_ms", 0.0))
                    + float(local_cpu_prepare_profile.get("prompt_semantic_cpu_prepare_ms", 0.0))
                    + float(worker_profile.get("prompt_semantic_cpu_prepare_ms", 0.0))
                    + float(worker_profile.get("prompt_semantic_forward_ms", 0.0))
                    + float(worker_profile.get("prompt_semantic_scatter_ms", 0.0)),
                },
            }
        wav16k, cpu_prepare_ms, limiter_stats = self.tts._prepare_prompt_semantic_wav16k_profile(raw_audio, raw_sr)
        with self.tts.prepare_ref_semantic_stage_limiter.enter() as stage_stats:
            prompt_semantic, runtime_profile = self.tts._extract_prompt_semantic_profile_from_prepared_wav16k(wav16k)
        return {
            "prompt_semantic": prompt_semantic,
            "raw_audio": raw_audio,
            "raw_sr": raw_sr,
            "profile": {
                "audio_load_ms": 0.0,
                "audio_stage_wait_ms": float(stage_stats.get("wait_ms", 0.0)),
                "audio_stage_slots": float(stage_stats.get("slots", 0.0)),
                "audio_stage_inflight_peak": float(stage_stats.get("peak_inflight", 0.0)),
                "prompt_semantic_wait_ms": float(stage_stats.get("wait_ms", 0.0)),
                "prompt_semantic_cpu_prepare_wait_ms": float(limiter_stats.get("wait_ms", 0.0)),
                "prompt_semantic_cpu_prepare_slots": float(limiter_stats.get("slots", 0.0)),
                "prompt_semantic_cpu_prepare_inflight_peak": float(limiter_stats.get("peak_inflight", 0.0)),
                "prompt_semantic_worker_queue_wait_ms": 0.0,
                "prompt_semantic_batch_collect_wait_ms": 0.0,
                "prompt_semantic_stage_limiter_wait_ms": float(stage_stats.get("wait_ms", 0.0)),
                "prompt_semantic_batch_dispatch_delay_ms": 0.0,
                "prompt_semantic_cpu_prepare_ms": float(cpu_prepare_ms),
                "prompt_semantic_pack_ms": 0.0,
                "prompt_semantic_h2d_ms": float(runtime_profile.get("prompt_semantic_h2d_ms", 0.0)),
                "prompt_semantic_ssl_forward_ms": float(runtime_profile.get("prompt_semantic_ssl_forward_ms", 0.0)),
                "prompt_semantic_hidden_length_ms": float(
                    runtime_profile.get("prompt_semantic_hidden_length_ms", 0.0)
                ),
                "prompt_semantic_extract_latent_ms": float(
                    runtime_profile.get("prompt_semantic_extract_latent_ms", 0.0)
                ),
                "prompt_semantic_forward_ms": float(runtime_profile.get("prompt_semantic_forward_ms", 0.0)),
                "prompt_semantic_scatter_ms": 0.0,
                "prompt_semantic_stage_slots": float(stage_stats.get("slots", 0.0)),
                "prompt_semantic_stage_inflight_peak": float(stage_stats.get("peak_inflight", 0.0)),
                "prompt_semantic_batch_size": 1.0,
                "prompt_semantic_batch_samples": 0.0,
                "ref_spec_wait_ms": 0.0,
                "ref_spec_ms": 0.0,
                "bundle_total_ms": float(
                    cpu_prepare_ms + runtime_profile.get("prompt_semantic_forward_ms", 0.0) + stage_stats.get("wait_ms", 0.0)
                ),
            },
        }

    def _extract_ref_spec_from_raw(self, raw_audio, raw_sr: int):
        spec, audio, _, _, profile = self.tts._extract_ref_spec_profile_from_raw(raw_audio, raw_sr)
        return (spec, audio), profile

    @staticmethod
    def _build_empty_text_features_like(reference: PreparedTextFeatures | None = None) -> PreparedTextFeatures:
        feature_dim = 1024
        dtype = None
        if reference is not None:
            try:
                feature_dim = int(reference.bert_features.shape[0])
                dtype = reference.bert_features.dtype
            except Exception:
                pass
        return build_empty_text_features(
            feature_dim=int(feature_dim),
            dtype=(dtype if dtype is not None else None) or __import__("torch").float32,
        )

    def _build_text_features(
        self,
        prepared_segments,
        language: str,
        cpu_run_ms: float,
        base_profile: Dict[str, float] | None = None,
    ) -> PreparedTextFeatures:
        profile: Dict[str, float] = dict(base_profile or {})
        profile["cpu_preprocess_ms"] = float(cpu_run_ms)
        branch_start = time.perf_counter()
        phones, bert_features, norm_text = self.tts.build_text_features_from_segments(prepared_segments, profile=profile)
        total_ms = float(cpu_run_ms + (time.perf_counter() - branch_start) * 1000.0)
        profile["bert_total_ms"] = max(0.0, total_ms - float(cpu_run_ms))
        return PreparedTextFeatures(
            phones=phones,
            bert_features=bert_features,
            norm_text=norm_text,
            profile=profile,
            total_ms=total_ms,
            cpu_preprocess_ms=float(cpu_run_ms),
        )

    async def _run_on_executor(self, executor, fn, *args) -> ProfiledResult:
        loop = asyncio.get_running_loop()
        submit_at = time.perf_counter()
        return await loop.run_in_executor(executor, self._run_profiled, fn, float(submit_at), *args)

    def submit_prepare_ref_audio_asset(
        self,
        ref_audio_path: str,
        *,
        submit_at: float | None = None,
    ) -> concurrent.futures.Future:
        submit_ts = time.perf_counter() if submit_at is None else float(submit_at)
        cache_key = self._normalize_ref_audio_cache_key(ref_audio_path)
        with self.ref_audio_asset_lock:
            now = time.perf_counter()
            self._prune_ref_audio_asset_cache_locked(now)
            cached_item = self.ref_audio_asset_cache.get(cache_key)
            if cached_item is not None:
                cached_asset, cached_at = cached_item
                return self._build_ref_audio_asset_cache_hit_future(
                    cached_asset,
                    submit_ts=float(submit_ts),
                    cache_age_ms=max(0.0, (now - float(cached_at)) * 1000.0),
                )
            inflight_future = self.ref_audio_asset_inflight.get(cache_key)
            if inflight_future is not None and not inflight_future.done():
                return inflight_future
            future = self.ref_audio_executor.submit(
                self._run_profiled,
                self._prepare_ref_audio_asset,
                float(submit_ts),
                ref_audio_path,
            )
            self.ref_audio_asset_inflight[cache_key] = future

        def _finalize(done_future: concurrent.futures.Future) -> None:
            cached_asset: PreparedRefAudioAsset | None = None
            try:
                profiled = done_future.result()
                result = getattr(profiled, "result", None)
                if isinstance(result, PreparedRefAudioAsset):
                    cached_asset = self._clone_prepared_ref_audio_asset(result)
            except Exception:
                cached_asset = None
            finally:
                with self.ref_audio_asset_lock:
                    current_future = self.ref_audio_asset_inflight.get(cache_key)
                    if current_future is done_future:
                        self.ref_audio_asset_inflight.pop(cache_key, None)
                    if (
                        cached_asset is not None
                        and self.ref_audio_asset_cache_ttl_sec > 0.0
                        and self.ref_audio_asset_cache_max_entries > 0
                    ):
                        self.ref_audio_asset_cache[cache_key] = (cached_asset, time.perf_counter())
                        self._prune_ref_audio_asset_cache_locked()

        future.add_done_callback(_finalize)
        return future

    @staticmethod
    def _build_text_cpu_profiled_result(
        submit_at: float,
        result: Any,
        worker_profile: Dict[str, float],
    ) -> ProfiledResult:
        started_at = float(
            submit_at
            + (
                float(worker_profile.get("text_cpu_admission_wait_ms", 0.0))
                + float(worker_profile.get("text_cpu_queue_wait_ms", 0.0))
            )
            / 1000.0
        )
        finished_at = float(started_at + float(worker_profile.get("text_cpu_run_ms", 0.0)) / 1000.0)
        return ProfiledResult(
            result=result,
            submit_at=float(submit_at),
            started_at=started_at,
            finished_at=finished_at,
            profile=dict(worker_profile),
        )

    async def _run_text_cpu_stage(self, text: str, language: str) -> ProfiledResult:
        await self.text_cpu_gate.acquire()
        if text in [None, ""]:
            try:
                submit_at = time.perf_counter()
                return ProfiledResult(result=[], submit_at=submit_at, started_at=submit_at, finished_at=submit_at)
            finally:
                self.text_cpu_gate.release()
        text_cpu_worker = getattr(self.tts, "prepare_text_cpu_worker", None)
        executor = getattr(self.tts, "prepare_text_cpu_executor", None)
        try:
            if text_cpu_worker is not None:
                submit_at = time.perf_counter()
                result, worker_profile = await text_cpu_worker.submit_async(text, language)
                started_at = float(
                    submit_at
                    + (
                        float(worker_profile.get("text_cpu_admission_wait_ms", 0.0))
                        + float(worker_profile.get("text_cpu_queue_wait_ms", 0.0))
                    )
                    / 1000.0
                )
                finished_at = float(started_at + float(worker_profile.get("text_cpu_run_ms", 0.0)) / 1000.0)
                return ProfiledResult(
                    result=result,
                    submit_at=float(submit_at),
                    started_at=started_at,
                    finished_at=finished_at,
                    profile=dict(worker_profile),
                )
            if executor is None:
                submit_at = time.perf_counter()
                return self._run_profiled(self._prepare_text_cpu, submit_at, text, language)
            return await self._run_on_executor(executor, self._prepare_text_cpu, text, language)
        finally:
            self.text_cpu_gate.release()

    async def _run_text_cpu_stage_pair(
        self,
        prompt_text: str,
        prompt_lang: str,
        text: str,
        text_lang: str,
    ) -> tuple[ProfiledResult, ProfiledResult]:
        text_cpu_worker = getattr(self.tts, "prepare_text_cpu_worker", None)
        if (
            text_cpu_worker is None
            or not hasattr(text_cpu_worker, "submit_many_async")
            or int(getattr(self.text_cpu_gate, "max_inflight", 0)) > 0
        ):
            prompt_cpu_task = asyncio.create_task(self._run_text_cpu_stage(prompt_text, prompt_lang))
            target_cpu_task = asyncio.create_task(self._run_text_cpu_stage(text, text_lang))
            return await asyncio.gather(prompt_cpu_task, target_cpu_task)

        items = []
        item_indices = []
        profiled_results: list[ProfiledResult | None] = [None, None]
        for index, (item_text, item_lang) in enumerate(((prompt_text, prompt_lang), (text, text_lang))):
            if item_text in [None, ""]:
                submit_at = time.perf_counter()
                profiled_results[index] = ProfiledResult(
                    result=[],
                    submit_at=submit_at,
                    started_at=submit_at,
                    finished_at=submit_at,
                )
                continue
            items.append((item_text, item_lang))
            item_indices.append(index)

        if items:
            submit_at = time.perf_counter()
            worker_results = await text_cpu_worker.submit_many_async(items)
            for item_index, (result, worker_profile) in zip(item_indices, worker_results):
                profiled_results[item_index] = self._build_text_cpu_profiled_result(
                    submit_at,
                    result,
                    dict(worker_profile),
                )

        assert profiled_results[0] is not None
        assert profiled_results[1] is not None
        return profiled_results[0], profiled_results[1]

    async def _run_text_feature_stage(
        self,
        prepared_segments,
        language: str | None,
        cpu_run_ms: float,
        base_profile: Dict[str, float] | None = None,
    ) -> ProfiledResult:
        if self.text_feature_executor is not None:
            await self.text_feature_gate.acquire()
            try:
                return await self._run_on_executor(
                    self.text_feature_executor,
                    self._build_text_features,
                    prepared_segments,
                    language,
                    cpu_run_ms,
                    base_profile,
                )
            finally:
                self.text_feature_gate.release()

        await self.text_feature_gate.acquire()
        profile: Dict[str, float] = dict(base_profile or {})
        profile["cpu_preprocess_ms"] = float(cpu_run_ms)
        submit_at = time.perf_counter()
        started_at = float(submit_at)
        try:
            result_raw = await self.tts.build_text_features_from_segments_async(
                prepared_segments,
                profile=profile,
            )
            finished_at = time.perf_counter()
            result = PreparedTextFeatures(
                phones=result_raw[0],
                bert_features=result_raw[1],
                norm_text=result_raw[2],
                profile=profile,
                total_ms=float(cpu_run_ms + self._estimate_text_feature_run_ms(profile)),
                cpu_preprocess_ms=float(cpu_run_ms),
            )
            profiled = ProfiledResult(
                result=result,
                submit_at=float(submit_at),
                started_at=started_at,
                finished_at=float(submit_at + self._estimate_text_feature_run_ms(profile) / 1000.0),
            )
            if finished_at > profiled.finished_at:
                result.profile["bert_total_ms"] = max(
                    self._estimate_text_feature_run_ms(profile),
                    (finished_at - submit_at) * 1000.0,
                )
            else:
                result.profile["bert_total_ms"] = self._estimate_text_feature_run_ms(profile)
            return profiled
        finally:
            self.text_feature_gate.release()

    async def _run_g2pw_stage(self, prepared_segments) -> ProfiledResult:
        has_pending = any(bool(getattr(segment, "needs_g2pw", False)) for segment in (prepared_segments or []))
        if not has_pending:
            submit_at = time.perf_counter()
            return ProfiledResult(
                result=prepared_segments,
                submit_at=float(submit_at),
                started_at=float(submit_at),
                finished_at=float(submit_at),
                profile={},
            )
        await self.g2pw_gate.acquire()
        try:
            profiled = await self._run_on_executor(self.g2pw_executor, self._resolve_g2pw_segments, prepared_segments)
            result, stage_profile = profiled.result
            return ProfiledResult(
                result=result,
                submit_at=float(profiled.submit_at),
                started_at=float(profiled.started_at),
                finished_at=float(profiled.finished_at),
                profile=dict(stage_profile),
            )
        finally:
            self.g2pw_gate.release()

    @staticmethod
    def _merge_g2pw_pair_stage_profile(
        profile: Dict[str, float] | None,
        pair_profile: Dict[str, float],
    ) -> Dict[str, float]:
        merged = dict(profile or {})
        for key, value in pair_profile.items():
            merged[key] = float(value)
        return merged

    async def _run_g2pw_pair_stage(self, prompt_segments, target_segments) -> tuple[ProfiledResult, ProfiledResult]:
        pair_submit_at = time.perf_counter()
        prompt_is_empty = len(prompt_segments or []) == 0
        prompt_has_pending = any(bool(getattr(segment, "needs_g2pw", False)) for segment in (prompt_segments or []))
        target_has_pending = any(bool(getattr(segment, "needs_g2pw", False)) for segment in (target_segments or []))
        g2pw_batch_worker = getattr(self.tts, "prepare_g2pw_batch_worker", None)
        if g2pw_batch_worker is not None and (prompt_has_pending or target_has_pending):
            resolved_batches, batch_profiles, worker_profile, submit_at, started_at, finished_at = (
                await g2pw_batch_worker.submit_async([prompt_segments or [], target_segments or []])
            )
            prompt_result, target_result = resolved_batches
            prompt_profile, target_profile = batch_profiles
            pair_finished_at = time.perf_counter()
            pair_compute_ms = max(0.0, (float(finished_at) - float(started_at)) * 1000.0)
            pair_profile = {
                "g2pw_pair_gate_wait_ms": 0.0,
                "g2pw_pair_executor_queue_ms": 0.0,
                "g2pw_pair_compute_ms": float(pair_compute_ms),
                "g2pw_pair_stage_overhead_ms": max(
                    0.0,
                    (pair_finished_at - pair_submit_at) * 1000.0 - float(pair_compute_ms),
                ),
            }
            prompt_profile = self._merge_g2pw_pair_stage_profile(
                {**dict(prompt_profile), **dict(worker_profile)},
                pair_profile,
            )
            target_profile = self._merge_g2pw_pair_stage_profile(
                {**dict(target_profile), **dict(worker_profile)},
                pair_profile,
            )
            if prompt_has_pending:
                prompt_profiled = ProfiledResult(
                    result=prompt_result,
                    submit_at=float(submit_at),
                    started_at=float(started_at),
                    finished_at=float(finished_at),
                    profile=prompt_profile,
                )
            else:
                idle_ts = time.perf_counter()
                prompt_profiled = ProfiledResult(
                    result=prompt_segments,
                    submit_at=float(idle_ts),
                    started_at=float(idle_ts),
                    finished_at=float(idle_ts),
                    profile={},
                )
            if target_has_pending:
                target_profiled = ProfiledResult(
                    result=target_result,
                    submit_at=float(submit_at),
                    started_at=float(started_at),
                    finished_at=float(finished_at),
                    profile=target_profile,
                )
            else:
                idle_ts = time.perf_counter()
                target_profiled = ProfiledResult(
                    result=target_segments,
                    submit_at=float(idle_ts),
                    started_at=float(idle_ts),
                    finished_at=float(idle_ts),
                    profile={},
                )
            return prompt_profiled, target_profiled
        if self.enable_g2pw_pair_batch and (prompt_has_pending or target_has_pending):
            gate_wait_start = time.perf_counter()
            await self.g2pw_gate.acquire()
            gate_acquired_at = time.perf_counter()
            try:
                profiled = await self._run_on_executor(
                    self.g2pw_executor,
                    self._resolve_g2pw_segment_batches,
                    [prompt_segments or [], target_segments or []],
                )
                pair_finished_at = time.perf_counter()
                pair_profile = {
                    "g2pw_pair_gate_wait_ms": max(0.0, (gate_acquired_at - gate_wait_start) * 1000.0),
                    "g2pw_pair_executor_queue_ms": float(profiled.queue_ms),
                    "g2pw_pair_compute_ms": float(profiled.run_ms),
                    "g2pw_pair_stage_overhead_ms": max(
                        0.0,
                        (pair_finished_at - pair_submit_at) * 1000.0
                        - max(0.0, (gate_acquired_at - gate_wait_start) * 1000.0)
                        - float(profiled.queue_ms)
                        - float(profiled.run_ms),
                    ),
                }
                resolved_batches, batch_profiles = profiled.result
                prompt_result, target_result = resolved_batches
                prompt_profile, target_profile = batch_profiles
                if prompt_has_pending:
                    prompt_profiled = ProfiledResult(
                        result=prompt_result,
                        submit_at=float(profiled.submit_at),
                        started_at=float(profiled.started_at),
                        finished_at=float(profiled.finished_at),
                        profile=self._merge_g2pw_pair_stage_profile(prompt_profile, pair_profile),
                    )
                else:
                    submit_at = time.perf_counter()
                    prompt_profiled = ProfiledResult(
                        result=prompt_segments,
                        submit_at=float(submit_at),
                        started_at=float(submit_at),
                        finished_at=float(submit_at),
                        profile={},
                    )
                if target_has_pending:
                    target_profiled = ProfiledResult(
                        result=target_result,
                        submit_at=float(profiled.submit_at),
                        started_at=float(profiled.started_at),
                        finished_at=float(profiled.finished_at),
                        profile=self._merge_g2pw_pair_stage_profile(target_profile, pair_profile),
                    )
                else:
                    submit_at = time.perf_counter()
                    target_profiled = ProfiledResult(
                        result=target_segments,
                        submit_at=float(submit_at),
                        started_at=float(submit_at),
                        finished_at=float(submit_at),
                        profile={},
                    )
                return prompt_profiled, target_profiled
            finally:
                self.g2pw_gate.release()
        target_task = asyncio.create_task(self._run_g2pw_stage(target_segments))
        if not prompt_is_empty:
            prompt_task = asyncio.create_task(self._run_g2pw_stage(prompt_segments))
            return await asyncio.gather(prompt_task, target_task)
        target_profiled = await target_task
        submit_at = time.perf_counter()
        prompt_profiled = ProfiledResult(
            result=prompt_segments,
            submit_at=float(submit_at),
            started_at=float(submit_at),
            finished_at=float(submit_at),
            profile={},
        )
        return prompt_profiled, target_profiled

    async def _run_g2pw_pair_stage_batch(
        self,
        cpu_stages: list[PreparedCpuStage],
    ) -> list[tuple[ProfiledResult, ProfiledResult] | Exception]:
        pair_submit_at = time.perf_counter()
        if not cpu_stages:
            return []

        group_batches: list[list[Any]] = []
        group_request_index: list[tuple[int, str]] = []
        has_pending_pairs: list[tuple[bool, bool]] = []
        idle_prompt_target: list[tuple[Any, Any]] = []
        for index, cpu_stage in enumerate(cpu_stages):
            prompt_segments = cpu_stage.prompt_cpu_profiled.result
            target_segments = cpu_stage.target_cpu_profiled.result
            prompt_has_pending = any(bool(getattr(segment, "needs_g2pw", False)) for segment in (prompt_segments or []))
            target_has_pending = any(bool(getattr(segment, "needs_g2pw", False)) for segment in (target_segments or []))
            has_pending_pairs.append((prompt_has_pending, target_has_pending))
            idle_prompt_target.append((prompt_segments, target_segments))
            if prompt_has_pending:
                group_request_index.append((index, "prompt"))
                group_batches.append(prompt_segments or [])
            if target_has_pending:
                group_request_index.append((index, "target"))
                group_batches.append(target_segments or [])

        if not group_batches:
            profiled_results: list[tuple[ProfiledResult, ProfiledResult]] = []
            for prompt_segments, target_segments in idle_prompt_target:
                idle_ts = time.perf_counter()
                profiled_results.append(
                    (
                        ProfiledResult(
                            result=prompt_segments,
                            submit_at=float(idle_ts),
                            started_at=float(idle_ts),
                            finished_at=float(idle_ts),
                            profile={},
                        ),
                        ProfiledResult(
                            result=target_segments,
                            submit_at=float(idle_ts),
                            started_at=float(idle_ts),
                            finished_at=float(idle_ts),
                            profile={},
                        ),
                    )
                )
            return profiled_results

        gate_wait_start = time.perf_counter()
        await self.g2pw_gate.acquire()
        gate_acquired_at = time.perf_counter()
        try:
            profiled = await self._run_on_executor(
                self.g2pw_executor,
                self._resolve_g2pw_segment_batches,
                group_batches,
            )
        finally:
            self.g2pw_gate.release()

        pair_finished_at = time.perf_counter()
        pair_profile = {
            "g2pw_pair_gate_wait_ms": max(0.0, (gate_acquired_at - gate_wait_start) * 1000.0),
            "g2pw_pair_executor_queue_ms": float(profiled.queue_ms),
            "g2pw_pair_compute_ms": float(profiled.run_ms),
            "g2pw_pair_stage_overhead_ms": max(
                0.0,
                (pair_finished_at - pair_submit_at) * 1000.0
                - max(0.0, (gate_acquired_at - gate_wait_start) * 1000.0)
                - float(profiled.queue_ms)
                - float(profiled.run_ms),
            ),
            "g2pw_pair_audio_batch_merge_size": float(len(cpu_stages)),
        }
        resolved_batches, batch_profiles = profiled.result

        prompt_results: list[ProfiledResult | None] = [None] * len(cpu_stages)
        target_results: list[ProfiledResult | None] = [None] * len(cpu_stages)
        for (request_index, branch), resolved_segments, stage_profile in zip(
            group_request_index,
            resolved_batches,
            batch_profiles,
        ):
            branch_profile = self._merge_g2pw_pair_stage_profile(stage_profile, pair_profile)
            branch_result = ProfiledResult(
                result=resolved_segments,
                submit_at=float(profiled.submit_at),
                started_at=float(profiled.started_at),
                finished_at=float(profiled.finished_at),
                profile=branch_profile,
            )
            if branch == "prompt":
                prompt_results[request_index] = branch_result
            else:
                target_results[request_index] = branch_result

        profiled_results = []
        for index, (prompt_has_pending, target_has_pending) in enumerate(has_pending_pairs):
            prompt_segments, target_segments = idle_prompt_target[index]
            if prompt_has_pending:
                prompt_profiled = prompt_results[index]
            else:
                idle_ts = time.perf_counter()
                prompt_profiled = ProfiledResult(
                    result=prompt_segments,
                    submit_at=float(idle_ts),
                    started_at=float(idle_ts),
                    finished_at=float(idle_ts),
                    profile={},
                )
            if target_has_pending:
                target_profiled = target_results[index]
            else:
                idle_ts = time.perf_counter()
                target_profiled = ProfiledResult(
                    result=target_segments,
                    submit_at=float(idle_ts),
                    started_at=float(idle_ts),
                    finished_at=float(idle_ts),
                    profile={},
                )
            assert prompt_profiled is not None
            assert target_profiled is not None
            profiled_results.append((prompt_profiled, target_profiled))
        return profiled_results

    async def _prepare_gpu_phase_one_batch(
        self,
        cpu_stages: list[PreparedCpuStage],
    ) -> list[Dict[str, Any] | Exception]:
        if not cpu_stages:
            return []
        phase_start = time.perf_counter()
        ref_audio_tasks = [
            asyncio.create_task(self._run_ref_prompt_semantic_stage(str(cpu_stage.spec.ref_audio_path)))
            for cpu_stage in cpu_stages
        ]
        try:
            g2pw_pairs: list[tuple[ProfiledResult, ProfiledResult] | Exception | None] = [None] * len(cpu_stages)
            group_size = max(1, int(self.g2pw_audio_batch_merge_group_size))
            for start_index in range(0, len(cpu_stages), group_size):
                group = cpu_stages[start_index : start_index + group_size]
                group_pairs = await self._run_g2pw_pair_stage_batch(group)
                for offset, group_pair in enumerate(group_pairs):
                    g2pw_pairs[start_index + offset] = group_pair
            g2pw_pair_end = time.perf_counter()
            ref_audio_results = await asyncio.gather(*ref_audio_tasks, return_exceptions=True)
            outputs: list[Dict[str, Any] | Exception] = []
            for cpu_stage, g2pw_pair, ref_audio_profiled in zip(cpu_stages, g2pw_pairs, ref_audio_results):
                if isinstance(g2pw_pair, Exception):
                    outputs.append(g2pw_pair)
                    continue
                if isinstance(ref_audio_profiled, Exception):
                    outputs.append(ref_audio_profiled)
                    continue
                prompt_g2pw_profiled, target_g2pw_profiled = g2pw_pair
                phase_end = max(float(g2pw_pair_end), float(ref_audio_profiled.finished_at))
                outputs.append(
                    {
                        "prompt_g2pw_profiled": prompt_g2pw_profiled,
                        "target_g2pw_profiled": target_g2pw_profiled,
                        "ref_audio_profiled": ref_audio_profiled,
                        "ref_spec_result": None,
                        "g2pw_pair_ms": max(0.0, (g2pw_pair_end - phase_start) * 1000.0),
                        "phase_wall_ms": max(0.0, (phase_end - phase_start) * 1000.0),
                    }
                )
            return outputs
        finally:
            for task in ref_audio_tasks:
                if task.done():
                    continue
                task.cancel()

    @staticmethod
    def _estimate_text_feature_run_ms(profile: Dict[str, float]) -> float:
        return float(
            profile.get("bert_wait_ms", 0.0)
            + profile.get("bert_tokenize_ms", 0.0)
            + profile.get("bert_forward_ms", 0.0)
            + profile.get("bert_scatter_ms", 0.0)
        )

    async def _run_text_feature_pair_stage(
        self,
        prompt_segments,
        target_segments,
        prompt_cpu_run_ms: float,
        target_cpu_run_ms: float,
        prompt_base_profile: Dict[str, float] | None = None,
        target_base_profile: Dict[str, float] | None = None,
    ) -> tuple[ProfiledResult, ProfiledResult]:
        prompt_is_empty = len(prompt_segments or []) == 0
        if self.text_feature_executor is not None:
            target_feature_task = asyncio.create_task(
                self._run_on_executor(
                    self.text_feature_executor,
                    self._build_text_features,
                    target_segments,
                    None,
                    target_cpu_run_ms,
                    target_base_profile,
                )
            )
            if not prompt_is_empty:
                prompt_feature_task = asyncio.create_task(
                    self._run_on_executor(
                        self.text_feature_executor,
                        self._build_text_features,
                        prompt_segments,
                        None,
                        prompt_cpu_run_ms,
                        prompt_base_profile,
                    )
                )
                return await asyncio.gather(prompt_feature_task, target_feature_task)
            target_profiled = await target_feature_task
            submit_at = time.perf_counter()
            prompt_profiled = ProfiledResult(
                result=self._build_empty_text_features_like(target_profiled.result),
                submit_at=float(submit_at),
                started_at=float(submit_at),
                finished_at=float(submit_at),
            )
            return prompt_profiled, target_profiled

        await self.text_feature_gate.acquire()
        target_profile: Dict[str, float] = dict(target_base_profile or {})
        target_profile["cpu_preprocess_ms"] = float(target_cpu_run_ms)
        submit_at = time.perf_counter()
        started_at = float(submit_at)
        try:
            if prompt_is_empty:
                target_result_raw = await self.tts.build_text_features_from_segments_async(
                    target_segments,
                    profile=target_profile,
                )
                prompt_result = self._build_empty_text_features_like(
                    PreparedTextFeatures(
                        phones=target_result_raw[0],
                        bert_features=target_result_raw[1],
                        norm_text=target_result_raw[2],
                        profile=target_profile,
                        total_ms=float(target_cpu_run_ms + self._estimate_text_feature_run_ms(target_profile)),
                        cpu_preprocess_ms=float(target_cpu_run_ms),
                    )
                )
                finished_at = time.perf_counter()
                prompt_profiled = ProfiledResult(
                    result=prompt_result,
                    submit_at=float(submit_at),
                    started_at=float(submit_at),
                    finished_at=float(submit_at),
                )
                target_result = PreparedTextFeatures(
                    phones=target_result_raw[0],
                    bert_features=target_result_raw[1],
                    norm_text=target_result_raw[2],
                    profile=target_profile,
                    total_ms=float(target_cpu_run_ms + self._estimate_text_feature_run_ms(target_profile)),
                    cpu_preprocess_ms=float(target_cpu_run_ms),
                )
                target_profiled = ProfiledResult(
                    result=target_result,
                    submit_at=float(submit_at),
                    started_at=started_at,
                    finished_at=float(submit_at + self._estimate_text_feature_run_ms(target_profile) / 1000.0),
                )
                if finished_at > target_profiled.finished_at:
                    target_result.profile["bert_total_ms"] = max(
                        self._estimate_text_feature_run_ms(target_profile),
                        (finished_at - submit_at) * 1000.0,
                    )
                else:
                    target_result.profile["bert_total_ms"] = self._estimate_text_feature_run_ms(target_profile)
                return prompt_profiled, target_profiled

            prompt_profile: Dict[str, float] = dict(prompt_base_profile or {})
            prompt_profile["cpu_preprocess_ms"] = float(prompt_cpu_run_ms)
            prompt_result_raw, target_result_raw = await self.tts.build_text_feature_pair_from_segments_async(
                prompt_segments,
                target_segments,
                prompt_profile=prompt_profile,
                target_profile=target_profile,
            )
            finished_at = time.perf_counter()

            prompt_result = PreparedTextFeatures(
                phones=prompt_result_raw[0],
                bert_features=prompt_result_raw[1],
                norm_text=prompt_result_raw[2],
                profile=prompt_profile,
                total_ms=float(prompt_cpu_run_ms + self._estimate_text_feature_run_ms(prompt_profile)),
                cpu_preprocess_ms=float(prompt_cpu_run_ms),
            )
            target_result = PreparedTextFeatures(
                phones=target_result_raw[0],
                bert_features=target_result_raw[1],
                norm_text=target_result_raw[2],
                profile=target_profile,
                total_ms=float(target_cpu_run_ms + self._estimate_text_feature_run_ms(target_profile)),
                cpu_preprocess_ms=float(target_cpu_run_ms),
            )
            prompt_profiled = ProfiledResult(
                result=prompt_result,
                submit_at=float(submit_at),
                started_at=started_at,
                finished_at=float(submit_at + self._estimate_text_feature_run_ms(prompt_profile) / 1000.0),
            )
            target_profiled = ProfiledResult(
                result=target_result,
                submit_at=float(submit_at),
                started_at=started_at,
                finished_at=float(submit_at + self._estimate_text_feature_run_ms(target_profile) / 1000.0),
            )
            if finished_at > prompt_profiled.finished_at:
                prompt_result.profile["bert_total_ms"] = max(
                    self._estimate_text_feature_run_ms(prompt_profile),
                    (finished_at - submit_at) * 1000.0,
                )
                target_result.profile["bert_total_ms"] = max(
                    self._estimate_text_feature_run_ms(target_profile),
                    (finished_at - submit_at) * 1000.0,
                )
            else:
                prompt_result.profile["bert_total_ms"] = self._estimate_text_feature_run_ms(prompt_profile)
                target_result.profile["bert_total_ms"] = self._estimate_text_feature_run_ms(target_profile)
            return prompt_profiled, target_profiled
        finally:
            self.text_feature_gate.release()

    async def _run_ref_prompt_semantic_stage(
        self,
        ref_audio_path: str,
        prepared_asset_future: concurrent.futures.Future | None = None,
        prepared_asset: PreparedRefAudioAsset | None = None,
    ) -> ProfiledResult:
        if getattr(self.tts, "prepare_ref_semantic_batch_worker", None) is not None:
            submit_at = time.perf_counter()
            started_at = float(submit_at)
            preload_profiled: ProfiledResult | None = None
            if prepared_asset is not None:
                preload_profiled = ProfiledResult(
                    result=prepared_asset,
                    submit_at=float(submit_at),
                    started_at=float(submit_at),
                    finished_at=float(submit_at),
                )
            elif prepared_asset_future is not None:
                preload_profiled = await asyncio.wrap_future(prepared_asset_future)

            if preload_profiled is None:
                await self.ref_load_gate.acquire()
                try:
                    load_profiled = await self._run_on_executor(self.ref_audio_executor, self._load_ref_audio_raw, ref_audio_path)
                finally:
                    self.ref_load_gate.release()
                raw_audio, raw_sr = load_profiled.result
                wav16k = None
                load_queue_ms = float(load_profiled.queue_ms)
                load_ms = float(load_profiled.run_ms)
                cpu_prepare_wait_ms = 0.0
                cpu_prepare_slots = 0.0
                cpu_prepare_inflight_peak = 0.0
                preload_cpu_prepare_ms = 0.0
            else:
                prepared_result = preload_profiled.result
                assert isinstance(prepared_result, PreparedRefAudioAsset)
                raw_audio = prepared_result.raw_audio
                raw_sr = prepared_result.raw_sr
                wav16k = prepared_result.wav16k
                preload_profile = dict(prepared_result.profile or {})
                load_queue_ms = float(preload_profiled.queue_ms)
                load_ms = float(preload_profile.get("audio_load_ms", 0.0))
                cpu_prepare_wait_ms = float(preload_profile.get("prompt_semantic_cpu_prepare_wait_ms", 0.0))
                cpu_prepare_slots = float(preload_profile.get("prompt_semantic_cpu_prepare_slots", 0.0))
                cpu_prepare_inflight_peak = float(
                    preload_profile.get("prompt_semantic_cpu_prepare_inflight_peak", 0.0)
                )
                preload_cpu_prepare_ms = float(preload_profile.get("prompt_semantic_cpu_prepare_ms", 0.0))
            prompt_semantic_task = asyncio.create_task(
                self.tts.prepare_ref_semantic_batch_worker.submit_async(raw_audio, raw_sr, wav16k=wav16k)
            )
            prompt_semantic, prompt_semantic_profile = await prompt_semantic_task
            limiter_snapshot = (
                self.tts.prepare_ref_semantic_stage_limiter.snapshot()
                if getattr(self.tts, "prepare_ref_semantic_stage_limiter", None) is not None
                else {}
            )
            prompt_semantic_ms = (
                float(prompt_semantic_profile.get("prompt_semantic_cpu_prepare_ms", 0.0))
                + float(prompt_semantic_profile.get("prompt_semantic_forward_ms", 0.0))
                + float(prompt_semantic_profile.get("prompt_semantic_scatter_ms", 0.0))
            )
            finished_at = time.perf_counter()
            result = {
                "prompt_semantic": prompt_semantic,
                "raw_audio": raw_audio,
                "raw_sr": raw_sr,
                "profile": {
                    "audio_load_queue_ms": float(load_queue_ms),
                    "audio_load_ms": float(load_ms),
                    "audio_stage_wait_ms": float(prompt_semantic_profile.get("prompt_semantic_wait_ms", 0.0)),
                    "audio_stage_slots": float(
                        max(
                            float(prompt_semantic_profile.get("prompt_semantic_stage_slots", 0.0)),
                            float(limiter_snapshot.get("slots", 0.0)),
                        )
                    ),
                    "audio_stage_inflight_peak": float(
                        max(
                            float(prompt_semantic_profile.get("prompt_semantic_stage_inflight_peak", 0.0)),
                            float(limiter_snapshot.get("peak_inflight", 0.0)),
                        )
                    ),
                    "prompt_semantic_ms": float(prompt_semantic_ms),
                    "prompt_semantic_wait_ms": float(prompt_semantic_profile.get("prompt_semantic_wait_ms", 0.0)),
                    "prompt_semantic_worker_queue_wait_ms": float(
                        prompt_semantic_profile.get("prompt_semantic_worker_queue_wait_ms", 0.0)
                    ),
                    "prompt_semantic_batch_collect_wait_ms": float(
                        prompt_semantic_profile.get("prompt_semantic_batch_collect_wait_ms", 0.0)
                    ),
                    "prompt_semantic_stage_limiter_wait_ms": float(
                        prompt_semantic_profile.get("prompt_semantic_stage_limiter_wait_ms", 0.0)
                    ),
                    "prompt_semantic_batch_dispatch_delay_ms": float(
                        prompt_semantic_profile.get("prompt_semantic_batch_dispatch_delay_ms", 0.0)
                    ),
                    "prompt_semantic_cpu_prepare_ms": float(
                        prompt_semantic_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)
                    ),
                    "prompt_semantic_preload_cpu_prepare_ms": float(preload_cpu_prepare_ms),
                    "prompt_semantic_cpu_prepare_wait_ms": float(cpu_prepare_wait_ms),
                    "prompt_semantic_cpu_prepare_slots": float(cpu_prepare_slots),
                    "prompt_semantic_cpu_prepare_inflight_peak": float(cpu_prepare_inflight_peak),
                    "prompt_semantic_preload_queue_ms": float(load_queue_ms),
                    "prompt_semantic_pack_ms": float(prompt_semantic_profile.get("prompt_semantic_pack_ms", 0.0)),
                    "prompt_semantic_h2d_ms": float(prompt_semantic_profile.get("prompt_semantic_h2d_ms", 0.0)),
                    "prompt_semantic_ssl_forward_ms": float(
                        prompt_semantic_profile.get("prompt_semantic_ssl_forward_ms", 0.0)
                    ),
                    "prompt_semantic_hidden_length_ms": float(
                        prompt_semantic_profile.get("prompt_semantic_hidden_length_ms", 0.0)
                    ),
                    "prompt_semantic_extract_latent_ms": float(
                        prompt_semantic_profile.get("prompt_semantic_extract_latent_ms", 0.0)
                    ),
                    "prompt_semantic_forward_ms": float(prompt_semantic_profile.get("prompt_semantic_forward_ms", 0.0)),
                    "prompt_semantic_scatter_ms": float(prompt_semantic_profile.get("prompt_semantic_scatter_ms", 0.0)),
                    "prompt_semantic_stage_slots": float(prompt_semantic_profile.get("prompt_semantic_stage_slots", 0.0)),
                    "prompt_semantic_stage_inflight_peak": float(
                        prompt_semantic_profile.get("prompt_semantic_stage_inflight_peak", 0.0)
                    ),
                    "prompt_semantic_batch_size": float(prompt_semantic_profile.get("prompt_semantic_batch_size", 1.0)),
                    "prompt_semantic_batch_samples": float(
                        prompt_semantic_profile.get("prompt_semantic_batch_samples", 0.0)
                    ),
                    "prompt_semantic_padded_batch_samples": float(
                        prompt_semantic_profile.get("prompt_semantic_padded_batch_samples", 0.0)
                    ),
                    "prompt_semantic_batch_pad_ratio": float(
                        prompt_semantic_profile.get("prompt_semantic_batch_pad_ratio", 0.0)
                    ),
                    "prompt_semantic_ssl_skip_attention_mask": float(
                        prompt_semantic_profile.get("prompt_semantic_ssl_skip_attention_mask", 0.0)
                    ),
                    "prompt_semantic_pool_workers": float(
                        prompt_semantic_profile.get("prompt_semantic_pool_workers", 0.0)
                    ),
                    "prompt_semantic_pool_bucket_index": float(
                        prompt_semantic_profile.get("prompt_semantic_pool_bucket_index", 0.0)
                    ),
                    "prompt_semantic_bucket_first_hit_serialized": float(
                        prompt_semantic_profile.get("prompt_semantic_bucket_first_hit_serialized", 0.0)
                    ),
                    "prompt_semantic_runtime_exact_prewarm_applied": float(
                        prompt_semantic_profile.get("prompt_semantic_runtime_exact_prewarm_applied", 0.0)
                    ),
                    "prompt_semantic_runtime_exact_prewarm_ms": float(
                        prompt_semantic_profile.get("prompt_semantic_runtime_exact_prewarm_ms", 0.0)
                    ),
                    "prompt_semantic_runtime_exact_prewarm_target_samples": float(
                        prompt_semantic_profile.get("prompt_semantic_runtime_exact_prewarm_target_samples", 0.0)
                    ),
                    "prompt_semantic_runtime_exact_prewarm_batch_sizes": float(
                        prompt_semantic_profile.get("prompt_semantic_runtime_exact_prewarm_batch_sizes", 0.0)
                    ),
                    "prompt_semantic_runtime_exact_prewarm_skipped_capacity": float(
                        prompt_semantic_profile.get("prompt_semantic_runtime_exact_prewarm_skipped_capacity", 0.0)
                    ),
                    "prompt_semantic_shard_index": float(
                        prompt_semantic_profile.get("prompt_semantic_shard_index", 0.0)
                    ),
                    "bundle_total_ms": float(
                        load_queue_ms
                        + load_ms
                        + preload_cpu_prepare_ms
                        + prompt_semantic_ms
                    ),
                },
            }
            return ProfiledResult(
                result=result,
                submit_at=float(submit_at),
                started_at=started_at,
                finished_at=float(finished_at),
            )

        await self.ref_audio_gate.acquire()
        try:
            preload_profiled: ProfiledResult | None = None
            if prepared_asset is not None:
                submit_at = time.perf_counter()
                preload_profiled = ProfiledResult(
                    result=prepared_asset,
                    submit_at=float(submit_at),
                    started_at=float(submit_at),
                    finished_at=float(submit_at),
                )
            elif prepared_asset_future is not None:
                preload_profiled = await asyncio.wrap_future(prepared_asset_future)

            if preload_profiled is None:
                load_profiled = await self._run_on_executor(self.ref_audio_executor, self._load_ref_audio_raw, ref_audio_path)
                raw_audio, raw_sr = load_profiled.result
                wav16k = None
                load_queue_ms = float(load_profiled.queue_ms)
                load_ms = float(load_profiled.run_ms)
                cpu_prepare_wait_ms = 0.0
                cpu_prepare_slots = 0.0
                cpu_prepare_inflight_peak = 0.0
                preload_cpu_prepare_ms = 0.0
            else:
                prepared_result = preload_profiled.result
                assert isinstance(prepared_result, PreparedRefAudioAsset)
                raw_audio = prepared_result.raw_audio
                raw_sr = prepared_result.raw_sr
                wav16k = prepared_result.wav16k
                preload_profile = dict(prepared_result.profile or {})
                load_queue_ms = float(preload_profiled.queue_ms)
                load_ms = float(preload_profile.get("audio_load_ms", 0.0))
                cpu_prepare_wait_ms = float(preload_profile.get("prompt_semantic_cpu_prepare_wait_ms", 0.0))
                cpu_prepare_slots = float(preload_profile.get("prompt_semantic_cpu_prepare_slots", 0.0))
                cpu_prepare_inflight_peak = float(
                    preload_profile.get("prompt_semantic_cpu_prepare_inflight_peak", 0.0)
                )
                preload_cpu_prepare_ms = float(preload_profile.get("prompt_semantic_cpu_prepare_ms", 0.0))
            submit_at = time.perf_counter()
            started_at = time.perf_counter()
            if wav16k is None:
                result = await asyncio.to_thread(self._build_ref_prompt_semantic_from_raw, raw_audio, raw_sr)
            else:
                with self.tts.prepare_ref_semantic_stage_limiter.enter() as stage_stats:
                    prompt_semantic, runtime_profile = await asyncio.to_thread(
                        self.tts._extract_prompt_semantic_profile_from_prepared_wav16k,
                        wav16k,
                    )
                result = {
                    "prompt_semantic": prompt_semantic,
                    "raw_audio": raw_audio,
                    "raw_sr": raw_sr,
                    "profile": {
                        "audio_load_queue_ms": float(load_queue_ms),
                        "audio_load_ms": float(load_ms),
                        "audio_stage_wait_ms": float(stage_stats.get("wait_ms", 0.0)),
                        "audio_stage_slots": float(stage_stats.get("slots", 0.0)),
                        "audio_stage_inflight_peak": float(stage_stats.get("peak_inflight", 0.0)),
                        "prompt_semantic_wait_ms": float(stage_stats.get("wait_ms", 0.0)),
                        "prompt_semantic_cpu_prepare_wait_ms": float(cpu_prepare_wait_ms),
                        "prompt_semantic_cpu_prepare_slots": float(cpu_prepare_slots),
                        "prompt_semantic_cpu_prepare_inflight_peak": float(cpu_prepare_inflight_peak),
                        "prompt_semantic_worker_queue_wait_ms": 0.0,
                        "prompt_semantic_batch_collect_wait_ms": 0.0,
                        "prompt_semantic_stage_limiter_wait_ms": float(stage_stats.get("wait_ms", 0.0)),
                        "prompt_semantic_batch_dispatch_delay_ms": 0.0,
                        "prompt_semantic_cpu_prepare_ms": 0.0,
                        "prompt_semantic_preload_cpu_prepare_ms": float(preload_cpu_prepare_ms),
                        "prompt_semantic_preload_queue_ms": float(load_queue_ms),
                        "prompt_semantic_pack_ms": 0.0,
                        "prompt_semantic_h2d_ms": float(runtime_profile.get("prompt_semantic_h2d_ms", 0.0)),
                        "prompt_semantic_ssl_forward_ms": float(
                            runtime_profile.get("prompt_semantic_ssl_forward_ms", 0.0)
                        ),
                        "prompt_semantic_hidden_length_ms": float(
                            runtime_profile.get("prompt_semantic_hidden_length_ms", 0.0)
                        ),
                        "prompt_semantic_extract_latent_ms": float(
                            runtime_profile.get("prompt_semantic_extract_latent_ms", 0.0)
                        ),
                        "prompt_semantic_forward_ms": float(runtime_profile.get("prompt_semantic_forward_ms", 0.0)),
                        "prompt_semantic_scatter_ms": 0.0,
                        "prompt_semantic_stage_slots": float(stage_stats.get("slots", 0.0)),
                        "prompt_semantic_stage_inflight_peak": float(stage_stats.get("peak_inflight", 0.0)),
                        "prompt_semantic_batch_size": 1.0,
                        "prompt_semantic_batch_samples": float(wav16k.shape[0]),
                        "bundle_total_ms": float(
                            load_queue_ms
                            + load_ms
                            + preload_cpu_prepare_ms
                            + runtime_profile.get("prompt_semantic_forward_ms", 0.0)
                        ),
                    },
                }
            result.setdefault("profile", {})
            result["profile"]["audio_load_queue_ms"] = float(load_queue_ms)
            result["profile"]["audio_load_ms"] = float(load_ms)
            finished_at = time.perf_counter()
            return ProfiledResult(result=result, submit_at=float(submit_at), started_at=float(started_at), finished_at=float(finished_at))
        finally:
            self.ref_audio_gate.release()

    async def _run_ref_spec_stage(self, raw_audio, raw_sr: int) -> ProfiledResult:
        await self.ref_spec_gate.acquire()
        try:
            return await self._run_on_executor(self.ref_audio_executor, self._extract_ref_spec_from_raw, raw_audio, raw_sr)
        finally:
            self.ref_spec_gate.release()

    def _release_split_stage_slot(self) -> None:
        self._mark_leave()
        self._inflight_gate.release()

    async def prepare_cpu_stage_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> PreparedCpuStage:
        admission_start = time.perf_counter()
        admission_stats = await self._inflight_gate.acquire()
        prepare_admission_wait_ms = max(
            float(admission_stats.get("wait_ms", 0.0)),
            (time.perf_counter() - admission_start) * 1000.0,
        )
        current_inflight, peak_inflight = self._mark_enter()
        prepare_start = time.perf_counter()
        prompt_text = normalize_sentence(spec.prompt_text, spec.prompt_lang)
        text = spec.text.strip("\n")
        try:
            prompt_cpu_profiled, target_cpu_profiled = await self._run_text_cpu_stage_pair(
                prompt_text,
                spec.prompt_lang,
                text,
                spec.text_lang,
            )
            return PreparedCpuStage(
                spec=spec,
                prepare_submit_at=float(prepare_submit_at),
                prepare_start=float(prepare_start),
                prompt_text=prompt_text,
                text=text,
                prepare_admission_wait_ms=float(prepare_admission_wait_ms),
                current_inflight=int(current_inflight),
                peak_inflight=int(peak_inflight),
                prompt_cpu_profiled=prompt_cpu_profiled,
                target_cpu_profiled=target_cpu_profiled,
            )
        except Exception:
            self._release_split_stage_slot()
            raise

    async def _prepare_cpu_stage_with_shared_prompt_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        *,
        prepare_submit_at: float,
        prompt_text: str,
        prompt_cpu_profiled: ProfiledResult,
    ) -> PreparedCpuStage:
        admission_start = time.perf_counter()
        admission_stats = await self._inflight_gate.acquire()
        prepare_admission_wait_ms = max(
            float(admission_stats.get("wait_ms", 0.0)),
            (time.perf_counter() - admission_start) * 1000.0,
        )
        current_inflight, peak_inflight = self._mark_enter()
        prepare_start = time.perf_counter()
        text = spec.text.strip("\n")
        try:
            target_cpu_profiled = await self._run_text_cpu_stage(text, spec.text_lang)
            return PreparedCpuStage(
                spec=spec,
                prepare_submit_at=float(prepare_submit_at),
                prepare_start=float(prepare_start),
                prompt_text=prompt_text,
                text=text,
                prepare_admission_wait_ms=float(prepare_admission_wait_ms),
                current_inflight=int(current_inflight),
                peak_inflight=int(peak_inflight),
                prompt_cpu_profiled=prompt_cpu_profiled,
                target_cpu_profiled=target_cpu_profiled,
            )
        except Exception:
            self._release_split_stage_slot()
            raise

    @staticmethod
    def _merge_ref_spec_profiled_result(
        profiled: ProfiledResult,
    ) -> tuple[tuple[Any, Any], Dict[str, float]]:
        refer_spec, profile = profiled.result
        merged_profile = dict(profile)
        merged_profile["ref_spec_wait_ms"] = float(profiled.queue_ms)
        merged_profile["ref_spec_ms"] = float(profiled.run_ms)
        return refer_spec, merged_profile

    def _build_shared_phase_profile(
        self,
        *,
        prompt_profiled: ProfiledResult,
        target_profiled: ProfiledResult,
        shared_profiled: ProfiledResult,
        shared_ref_spec_result: tuple[tuple[Any, Any], Dict[str, float]] | None,
        phase_kind: str,
    ) -> Dict[str, float]:
        phase_wall_ms = max(
            float(prompt_profiled.run_ms),
            float(target_profiled.run_ms),
            float(shared_profiled.run_ms),
        )
        if phase_kind == "audio" and shared_ref_spec_result is not None:
            phase_wall_ms = max(
                phase_wall_ms,
                float(shared_ref_spec_result[1].get("ref_spec_ms", 0.0)),
            )
        return {
            f"engine_prepare_{phase_kind}_phase_mode": 2.0,
            f"engine_prepare_{phase_kind}_phase_wall_ms": float(phase_wall_ms),
            f"engine_prepare_{phase_kind}_phase_batch_size": 1.0,
            "engine_prepare_shared_prompt_ref_enabled": 1.0,
        }

    def _release_cpu_stage_results(
        self,
        outputs: List[tuple[T2SRequestState, float, float] | Exception | None],
        cpu_stages: List[PreparedCpuStage | Exception | None],
    ) -> List[tuple[T2SRequestState, float, float] | Exception]:
        for index, cpu_stage in enumerate(cpu_stages):
            if not isinstance(cpu_stage, PreparedCpuStage):
                continue
            if outputs[index] is None:
                outputs[index] = RuntimeError("shared prepare result missing")
            self._release_split_stage_slot()
        return [item if item is not None else RuntimeError("shared prepare result missing") for item in outputs]

    async def prepare_direct_shared_segments_profiled_async(
        self,
        specs: List[SchedulerRequestSpec],
    ) -> List[tuple[T2SRequestState, float, float] | Exception]:
        if not specs:
            return []

        shared_spec = specs[0]
        shared_prompt_text = normalize_sentence(shared_spec.prompt_text, shared_spec.prompt_lang)
        shared_prompt_cpu_profiled = await self._run_text_cpu_stage(shared_prompt_text, shared_spec.prompt_lang)
        cpu_stage_results = await asyncio.gather(
            *[
                self._prepare_cpu_stage_with_shared_prompt_profiled_async(
                    spec,
                    prepare_submit_at=time.perf_counter(),
                    prompt_text=shared_prompt_text,
                    prompt_cpu_profiled=shared_prompt_cpu_profiled,
                )
                for spec in specs
            ],
            return_exceptions=True,
        )
        outputs: List[tuple[T2SRequestState, float, float] | Exception | None] = [None] * len(specs)
        runnable: List[tuple[int, PreparedCpuStage]] = []
        for index, cpu_stage in enumerate(cpu_stage_results):
            if isinstance(cpu_stage, Exception):
                outputs[index] = cpu_stage
                continue
            runnable.append((index, cpu_stage))

        if not runnable:
            return [item if item is not None else RuntimeError("shared prepare cpu stage missing") for item in outputs]

        try:
            shared_prompt_g2pw_profiled = await self._run_g2pw_stage(shared_prompt_cpu_profiled.result)
            shared_prompt_feature_profiled = await self._run_text_feature_stage(
                shared_prompt_g2pw_profiled.result,
                shared_spec.prompt_lang,
                shared_prompt_cpu_profiled.run_ms,
                base_profile=dict(shared_prompt_g2pw_profiled.profile or {}),
            )
            shared_ref_audio_profiled = await self._run_ref_prompt_semantic_stage(
                str(shared_spec.ref_audio_path),
                prepared_asset_future=self.submit_prepare_ref_audio_asset(
                    str(shared_spec.ref_audio_path),
                    submit_at=time.perf_counter(),
                ),
            )
            shared_ref_spec_profiled = await self._run_ref_spec_stage(
                shared_ref_audio_profiled.result["raw_audio"],
                int(shared_ref_audio_profiled.result["raw_sr"]),
            )
            shared_ref_spec_result = self._merge_ref_spec_profiled_result(shared_ref_spec_profiled)
        except Exception as exc:
            for index, _ in runnable:
                outputs[index] = exc
            return self._release_cpu_stage_results(outputs, cpu_stage_results)

        target_g2pw_results = await asyncio.gather(
            *[self._run_g2pw_stage(cpu_stage.target_cpu_profiled.result) for _, cpu_stage in runnable],
            return_exceptions=True,
        )
        target_feature_tasks = []
        for target_g2pw_profiled, (_, cpu_stage) in zip(target_g2pw_results, runnable):
            if isinstance(target_g2pw_profiled, Exception):
                target_feature_tasks.append(target_g2pw_profiled)
                continue
            target_feature_tasks.append(
                asyncio.create_task(
                    self._run_text_feature_stage(
                    target_g2pw_profiled.result,
                    cpu_stage.spec.text_lang,
                    cpu_stage.target_cpu_profiled.run_ms,
                    base_profile=dict(target_g2pw_profiled.profile or {}),
                    )
                )
            )
        target_feature_results = []
        for task in target_feature_tasks:
            if isinstance(task, Exception):
                target_feature_results.append(task)
                continue
            target_feature_results.append(task)
        target_feature_results = list(
            await asyncio.gather(
                *[
                    item
                    for item in target_feature_results
                    if not isinstance(item, Exception)
                ],
                return_exceptions=True,
            )
        ) if any(not isinstance(item, Exception) for item in target_feature_results) else []
        merged_target_feature_results: List[ProfiledResult | Exception] = []
        gather_index = 0
        for item in target_feature_tasks:
            if isinstance(item, Exception):
                merged_target_feature_results.append(item)
                continue
            merged_target_feature_results.append(target_feature_results[gather_index])
            gather_index += 1
        target_feature_results = merged_target_feature_results

        for (index, cpu_stage), target_g2pw_profiled, target_feature_profiled in zip(
            runnable,
            target_g2pw_results,
            target_feature_results,
        ):
            try:
                if isinstance(target_g2pw_profiled, Exception):
                    outputs[index] = target_g2pw_profiled
                    self._release_split_stage_slot()
                    continue
                if isinstance(target_feature_profiled, Exception):
                    outputs[index] = target_feature_profiled
                    self._release_split_stage_slot()
                    continue
                phase_one = {
                    "prompt_g2pw_profiled": shared_prompt_g2pw_profiled,
                    "target_g2pw_profiled": target_g2pw_profiled,
                    "ref_audio_profiled": shared_ref_audio_profiled,
                    "ref_spec_result": shared_ref_spec_result,
                    "g2pw_pair_ms": max(
                        float(shared_prompt_g2pw_profiled.run_ms),
                        float(target_g2pw_profiled.run_ms),
                    ),
                    "phase_wall_ms": max(
                        float(shared_prompt_g2pw_profiled.run_ms),
                        float(target_g2pw_profiled.run_ms),
                        float(shared_ref_audio_profiled.run_ms),
                        float(shared_ref_spec_result[1].get("ref_spec_ms", 0.0)),
                    ),
                }
                phase_two = {
                    "prompt_feature_profiled": shared_prompt_feature_profiled,
                    "target_feature_profiled": target_feature_profiled,
                    "phase_wall_ms": max(
                        float(shared_prompt_feature_profiled.run_ms),
                        float(target_feature_profiled.run_ms),
                    ),
                }
                outputs[index] = self.build_gpu_prepare_result_from_phases(
                    cpu_stage,
                    phase_one,
                    phase_two,
                    extra_profile={
                        **self._build_shared_phase_profile(
                            prompt_profiled=shared_prompt_g2pw_profiled,
                            target_profiled=target_g2pw_profiled,
                            shared_profiled=shared_ref_audio_profiled,
                            shared_ref_spec_result=shared_ref_spec_result,
                            phase_kind="audio",
                        ),
                        **self._build_shared_phase_profile(
                            prompt_profiled=shared_prompt_feature_profiled,
                            target_profiled=target_feature_profiled,
                            shared_profiled=shared_prompt_feature_profiled,
                            shared_ref_spec_result=None,
                            phase_kind="text",
                        ),
                        "engine_prepare_audio_phase_batch_size": float(len(runnable)),
                        "engine_prepare_text_phase_batch_size": float(len(runnable)),
                    },
                )
            except Exception as exc:
                outputs[index] = exc
                self._release_split_stage_slot()

        return [item if item is not None else RuntimeError("shared prepare result missing") for item in outputs]

    async def prepare_gpu_stage_profiled_async(
        self,
        cpu_stage: PreparedCpuStage,
    ) -> tuple[T2SRequestState, float, float]:
        try:
            phase_one = await self._prepare_gpu_phase_one(cpu_stage)
            phase_two = await self._prepare_gpu_phase_two(cpu_stage, phase_one)
            return self._build_gpu_prepare_result(
                cpu_stage,
                phase_one,
                phase_two,
                extra_profile={
                    "engine_prepare_audio_phase_mode": 0.0,
                    "engine_prepare_audio_phase_wall_ms": float(phase_one["phase_wall_ms"]),
                    "engine_prepare_audio_phase_batch_size": 1.0,
                    "engine_prepare_text_phase_wall_ms": float(phase_two["phase_wall_ms"]),
                    "engine_prepare_text_phase_batch_size": 1.0,
                },
            )
        finally:
            self._release_split_stage_slot()

    async def _prepare_gpu_phase_one(
        self,
        cpu_stage: PreparedCpuStage,
        prepared_ref_audio_future: concurrent.futures.Future | None = None,
    ) -> Dict[str, Any]:
        phase_start = time.perf_counter()
        g2pw_pair_task = asyncio.create_task(
            self._run_g2pw_pair_stage(
                cpu_stage.prompt_cpu_profiled.result,
                cpu_stage.target_cpu_profiled.result,
            )
        )
        ref_audio_task = asyncio.create_task(
            self._run_ref_prompt_semantic_stage(
                str(cpu_stage.spec.ref_audio_path),
                prepared_asset_future=prepared_ref_audio_future,
            )
        )
        prompt_g2pw_profiled, target_g2pw_profiled = await g2pw_pair_task
        g2pw_pair_end = time.perf_counter()
        ref_audio_profiled = await ref_audio_task
        phase_end = time.perf_counter()
        return {
            "prompt_g2pw_profiled": prompt_g2pw_profiled,
            "target_g2pw_profiled": target_g2pw_profiled,
            "ref_audio_profiled": ref_audio_profiled,
            "ref_spec_result": None,
            "g2pw_pair_ms": max(0.0, (g2pw_pair_end - phase_start) * 1000.0),
            "phase_wall_ms": max(0.0, (phase_end - phase_start) * 1000.0),
        }

    async def _prepare_gpu_phase_two(
        self,
        cpu_stage: PreparedCpuStage,
        phase_one: Dict[str, Any],
    ) -> Dict[str, Any]:
        phase_start = time.perf_counter()
        prompt_g2pw_profiled = phase_one["prompt_g2pw_profiled"]
        target_g2pw_profiled = phase_one["target_g2pw_profiled"]
        prompt_feature_profiled, target_feature_profiled = await self._run_text_feature_pair_stage(
            prompt_g2pw_profiled.result,
            target_g2pw_profiled.result,
            cpu_stage.prompt_cpu_profiled.run_ms,
            cpu_stage.target_cpu_profiled.run_ms,
            prompt_base_profile=dict(prompt_g2pw_profiled.profile or {}),
            target_base_profile=dict(target_g2pw_profiled.profile or {}),
        )
        phase_end = time.perf_counter()
        return {
            "prompt_feature_profiled": prompt_feature_profiled,
            "target_feature_profiled": target_feature_profiled,
            "phase_wall_ms": max(0.0, (phase_end - phase_start) * 1000.0),
        }

    def _build_gpu_prepare_result(
        self,
        cpu_stage: PreparedCpuStage,
        phase_one: Dict[str, Any],
        phase_two: Dict[str, Any],
        extra_profile: Dict[str, float] | None = None,
    ) -> tuple[T2SRequestState, float, float]:
        prompt_g2pw_profiled = phase_one["prompt_g2pw_profiled"]
        target_g2pw_profiled = phase_one["target_g2pw_profiled"]
        ref_audio_profiled = phase_one["ref_audio_profiled"]
        ref_spec_result = phase_one.get("ref_spec_result")
        prompt_feature_profiled = phase_two["prompt_feature_profiled"]
        target_feature_profiled = phase_two["target_feature_profiled"]
        profile_overrides = {
            "executor_queue_ms": max(0.0, (cpu_stage.prepare_start - cpu_stage.prepare_submit_at) * 1000.0),
            "prepare_admission_wait_ms": cpu_stage.prepare_admission_wait_ms,
            "prepare_submit_ts": float(cpu_stage.prepare_submit_at),
            "prepare_cpu_start_ts": float(cpu_stage.prepare_start),
            "prepare_cpu_done_ts": float(
                max(cpu_stage.prompt_cpu_profiled.finished_at, cpu_stage.target_cpu_profiled.finished_at)
            ),
            "prompt_text_cpu_start_ts": float(cpu_stage.prompt_cpu_profiled.started_at),
            "prompt_text_cpu_end_ts": float(cpu_stage.prompt_cpu_profiled.finished_at),
            "text_cpu_start_ts": float(cpu_stage.target_cpu_profiled.started_at),
            "text_cpu_end_ts": float(cpu_stage.target_cpu_profiled.finished_at),
            "executor_run_wall_ms": max(0.0, (time.perf_counter() - cpu_stage.prepare_start) * 1000.0),
            "text_feature_pair_ms": float(phase_two["phase_wall_ms"]),
            "g2pw_pair_ms": float(phase_one["g2pw_pair_ms"]),
            "g2pw_pair_gate_wait_ms": float((target_g2pw_profiled.profile or {}).get("g2pw_pair_gate_wait_ms", 0.0)),
            "g2pw_pair_executor_queue_ms": float(
                (target_g2pw_profiled.profile or {}).get("g2pw_pair_executor_queue_ms", 0.0)
            ),
            "g2pw_pair_compute_ms": float((target_g2pw_profiled.profile or {}).get("g2pw_pair_compute_ms", 0.0)),
            "g2pw_pair_stage_overhead_ms": float(
                (target_g2pw_profiled.profile or {}).get("g2pw_pair_stage_overhead_ms", 0.0)
            ),
            "g2pw_pair_audio_batch_merge_size": float(
                (target_g2pw_profiled.profile or {}).get("g2pw_pair_audio_batch_merge_size", 0.0)
            ),
            "prompt_text_g2pw_queue_ms": prompt_g2pw_profiled.queue_ms,
            "prompt_text_g2pw_run_ms": prompt_g2pw_profiled.run_ms,
            "prompt_text_g2pw_prepare_ms": float((prompt_g2pw_profiled.profile or {}).get("g2pw_prepare_ms", 0.0)),
            "prompt_text_g2pw_predict_ms": float((prompt_g2pw_profiled.profile or {}).get("g2pw_predict_ms", 0.0)),
            "prompt_text_g2pw_post_ms": float((prompt_g2pw_profiled.profile or {}).get("g2pw_post_ms", 0.0)),
            "prompt_text_g2pw_wait_ms": float((prompt_g2pw_profiled.profile or {}).get("g2pw_wait_ms", 0.0)),
            "prompt_text_g2pw_admission_wait_ms": float(
                (prompt_g2pw_profiled.profile or {}).get("g2pw_admission_wait_ms", 0.0)
            ),
            "prompt_text_g2pw_worker_queue_wait_ms": float(
                (prompt_g2pw_profiled.profile or {}).get("g2pw_worker_queue_wait_ms", 0.0)
            ),
            "prompt_text_g2pw_batch_collect_wait_ms": float(
                (prompt_g2pw_profiled.profile or {}).get("g2pw_batch_collect_wait_ms", 0.0)
            ),
            "prompt_text_g2pw_batch_dispatch_delay_ms": float(
                (prompt_g2pw_profiled.profile or {}).get("g2pw_batch_dispatch_delay_ms", 0.0)
            ),
            "prompt_text_g2pw_batch_size": float((prompt_g2pw_profiled.profile or {}).get("g2pw_batch_size", 0.0)),
            "prompt_text_g2pw_batch_groups": float((prompt_g2pw_profiled.profile or {}).get("g2pw_batch_groups", 0.0)),
            "prompt_text_g2pw_batch_chars": float((prompt_g2pw_profiled.profile or {}).get("g2pw_batch_chars", 0.0)),
            "text_g2pw_queue_ms": target_g2pw_profiled.queue_ms,
            "text_g2pw_run_ms": target_g2pw_profiled.run_ms,
            "text_g2pw_prepare_ms": float((target_g2pw_profiled.profile or {}).get("g2pw_prepare_ms", 0.0)),
            "text_g2pw_predict_ms": float((target_g2pw_profiled.profile or {}).get("g2pw_predict_ms", 0.0)),
            "text_g2pw_post_ms": float((target_g2pw_profiled.profile or {}).get("g2pw_post_ms", 0.0)),
            "text_g2pw_wait_ms": float((target_g2pw_profiled.profile or {}).get("g2pw_wait_ms", 0.0)),
            "text_g2pw_admission_wait_ms": float((target_g2pw_profiled.profile or {}).get("g2pw_admission_wait_ms", 0.0)),
            "text_g2pw_worker_queue_wait_ms": float(
                (target_g2pw_profiled.profile or {}).get("g2pw_worker_queue_wait_ms", 0.0)
            ),
            "text_g2pw_batch_collect_wait_ms": float(
                (target_g2pw_profiled.profile or {}).get("g2pw_batch_collect_wait_ms", 0.0)
            ),
            "text_g2pw_batch_dispatch_delay_ms": float(
                (target_g2pw_profiled.profile or {}).get("g2pw_batch_dispatch_delay_ms", 0.0)
            ),
            "text_g2pw_batch_size": float((target_g2pw_profiled.profile or {}).get("g2pw_batch_size", 0.0)),
            "text_g2pw_batch_groups": float((target_g2pw_profiled.profile or {}).get("g2pw_batch_groups", 0.0)),
            "text_g2pw_batch_chars": float((target_g2pw_profiled.profile or {}).get("g2pw_batch_chars", 0.0)),
            "prompt_text_parallel_future_wait_ms": 0.0,
            "prompt_text_parallel_future_executor_queue_ms": 0.0,
            "prompt_text_parallel_future_run_ms": 0.0,
            "prompt_text_parallel_future_finish_after_submit_ms": 0.0,
            "prompt_text_parallel_future_queue_tail_after_target_ms": 0.0,
            "prompt_text_parallel_future_run_tail_after_target_ms": 0.0,
            "prompt_text_cpu_queue_ms": cpu_stage.prompt_cpu_profiled.queue_ms,
            "prompt_text_cpu_run_ms": cpu_stage.prompt_cpu_profiled.run_ms,
            "prompt_text_cpu_admission_wait_ms": float(
                (cpu_stage.prompt_cpu_profiled.profile or {}).get("text_cpu_admission_wait_ms", 0.0)
            ),
            "prompt_text_cpu_backpressure_wait_ms": float(
                (cpu_stage.prompt_cpu_profiled.profile or {}).get("text_cpu_backpressure_wait_ms", 0.0)
            ),
            "prompt_text_cpu_capacity_wait_ms": float(
                (cpu_stage.prompt_cpu_profiled.profile or {}).get("text_cpu_capacity_wait_ms", 0.0)
            ),
            "prompt_text_feature_queue_ms": prompt_feature_profiled.queue_ms,
            "prompt_text_feature_run_ms": prompt_feature_profiled.run_ms,
            "text_cpu_queue_ms": cpu_stage.target_cpu_profiled.queue_ms,
            "text_cpu_run_ms": cpu_stage.target_cpu_profiled.run_ms,
            "text_cpu_admission_wait_ms": float(
                (cpu_stage.target_cpu_profiled.profile or {}).get("text_cpu_admission_wait_ms", 0.0)
            ),
            "text_cpu_backpressure_wait_ms": float(
                (cpu_stage.target_cpu_profiled.profile or {}).get("text_cpu_backpressure_wait_ms", 0.0)
            ),
            "text_cpu_capacity_wait_ms": float(
                (cpu_stage.target_cpu_profiled.profile or {}).get("text_cpu_capacity_wait_ms", 0.0)
            ),
            "text_feature_queue_ms": target_feature_profiled.queue_ms,
            "text_feature_run_ms": target_feature_profiled.run_ms,
            "ref_audio_task_queue_ms": ref_audio_profiled.queue_ms,
            "ref_audio_task_run_ms": ref_audio_profiled.run_ms,
            "worker_prepare_inflight_on_enter": float(cpu_stage.current_inflight),
            "worker_prepare_peak_inflight": float(cpu_stage.peak_inflight),
        }
        if extra_profile:
            profile_overrides.update({key: float(value) for key, value in extra_profile.items()})
        ref_audio_bundle = dict(ref_audio_profiled.result)
        ref_audio_profile = dict(ref_audio_bundle.get("profile", {}))
        if ref_spec_result is not None:
            refer_spec, ref_spec_profiled = ref_spec_result
            ref_audio_bundle["refer_spec"] = refer_spec
            ref_audio_profile.update(
                {
                    "ref_spec_wait_ms": float(ref_spec_profiled.get("ref_spec_wait_ms", 0.0)),
                    "ref_spec_ms": float(ref_spec_profiled.get("ref_spec_ms", 0.0)),
                    "ref_spec_to_device_ms": float(ref_spec_profiled.get("ref_spec_to_device_ms", 0.0)),
                    "ref_spec_main_resample_ms": float(ref_spec_profiled.get("ref_spec_main_resample_ms", 0.0)),
                    "ref_spec_norm_ms": float(ref_spec_profiled.get("ref_spec_norm_ms", 0.0)),
                    "ref_spec_spectrogram_ms": float(ref_spec_profiled.get("ref_spec_spectrogram_ms", 0.0)),
                    "ref_spec_post_resample_ms": float(ref_spec_profiled.get("ref_spec_post_resample_ms", 0.0)),
                }
            )
        else:
            ref_audio_bundle["refer_spec"] = None
            ref_audio_profile.setdefault("ref_spec_wait_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_to_device_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_main_resample_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_norm_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_spectrogram_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_post_resample_ms", 0.0)
        ref_audio_bundle["profile"] = ref_audio_profile
        state = build_request_state_from_parts(
            tts=self.tts,
            spec=cpu_stage.spec,
            prompt_text=cpu_stage.prompt_text,
            text=cpu_stage.text,
            prompt_result=prompt_feature_profiled.result,
            target_result=target_feature_profiled.result,
            ref_audio_bundle=ref_audio_bundle,
            prepare_start=cpu_stage.prepare_start,
            prepare_sync_start=cpu_stage.prepare_start,
            profile_overrides=profile_overrides,
        )
        prepare_exec_finished_at = time.perf_counter()
        state.prepare_profile["executor_run_wall_ms"] = max(0.0, (prepare_exec_finished_at - cpu_stage.prepare_start) * 1000.0)
        return state, cpu_stage.prepare_start, prepare_exec_finished_at

    async def prepare_ref_spec_stages_async(
        self,
        phase_ones: list[Dict[str, Any]],
    ) -> list[tuple[tuple[Any, Any], Dict[str, float]] | Exception]:
        async def _one(phase_one: Dict[str, Any]):
            ref_audio_profiled = phase_one["ref_audio_profiled"]
            raw_audio = ref_audio_profiled.result["raw_audio"]
            raw_sr = int(ref_audio_profiled.result["raw_sr"])
            profiled = await self._run_ref_spec_stage(raw_audio, raw_sr)
            refer_spec, profile = profiled.result
            merged_profile = dict(profile)
            merged_profile["ref_spec_wait_ms"] = float(profiled.queue_ms)
            merged_profile["ref_spec_ms"] = float(profiled.run_ms)
            return refer_spec, merged_profile

        if not phase_ones:
            return []
        return list(await asyncio.gather(*[_one(phase_one) for phase_one in phase_ones], return_exceptions=True))

    def apply_ref_spec_result_to_state(
        self,
        state: T2SRequestState,
        ref_spec_result: tuple[tuple[Any, Any], Dict[str, float]],
    ) -> None:
        refer_spec, profile = ref_spec_result
        state.refer_spec = refer_spec
        state.prepare_profile["ref_spec_wait_ms"] = float(profile.get("ref_spec_wait_ms", 0.0))
        state.prepare_profile["ref_spec_ms"] = float(profile.get("ref_spec_ms", 0.0))
        state.prepare_profile["ref_spec_to_device_ms"] = float(profile.get("ref_spec_to_device_ms", 0.0))
        state.prepare_profile["ref_spec_main_resample_ms"] = float(profile.get("ref_spec_main_resample_ms", 0.0))
        state.prepare_profile["ref_spec_norm_ms"] = float(profile.get("ref_spec_norm_ms", 0.0))
        state.prepare_profile["ref_spec_spectrogram_ms"] = float(profile.get("ref_spec_spectrogram_ms", 0.0))
        state.prepare_profile["ref_spec_post_resample_ms"] = float(profile.get("ref_spec_post_resample_ms", 0.0))

    async def prepare_gpu_stages_profiled_async(
        self,
        cpu_stages: list[PreparedCpuStage],
    ) -> list[tuple[T2SRequestState, float, float] | Exception]:
        if not cpu_stages:
            return []
        if len(cpu_stages) == 1:
            single_stage = cpu_stages[0]
            try:
                return [await self.prepare_gpu_stage_profiled_async(single_stage)]
            except Exception as exc:  # noqa: PERF203
                return [exc]

        phase_one_started_at = time.perf_counter()
        phase_one_results = await asyncio.gather(
            *[self._prepare_gpu_phase_one(cpu_stage) for cpu_stage in cpu_stages],
            return_exceptions=True,
        )
        phase_one_finished_at = time.perf_counter()
        phase_one_wall_ms = max(0.0, (phase_one_finished_at - phase_one_started_at) * 1000.0)

        outputs: list[tuple[T2SRequestState, float, float] | Exception | None] = [None] * len(cpu_stages)
        pending_phase_two: list[tuple[int, PreparedCpuStage, Dict[str, Any]]] = []
        for index, (cpu_stage, phase_one) in enumerate(zip(cpu_stages, phase_one_results)):
            if isinstance(phase_one, Exception):
                outputs[index] = phase_one
                self._release_split_stage_slot()
                continue
            pending_phase_two.append((index, cpu_stage, phase_one))

        phase_two_started_at = time.perf_counter()
        phase_two_results = await asyncio.gather(
            *[self._prepare_gpu_phase_two(cpu_stage, phase_one) for _, cpu_stage, phase_one in pending_phase_two],
            return_exceptions=True,
        )
        phase_two_finished_at = time.perf_counter()
        phase_two_wall_ms = max(0.0, (phase_two_finished_at - phase_two_started_at) * 1000.0)

        for (index, cpu_stage, phase_one), phase_two in zip(pending_phase_two, phase_two_results):
            try:
                if isinstance(phase_two, Exception):
                    outputs[index] = phase_two
                    continue
                outputs[index] = self._build_gpu_prepare_result(
                    cpu_stage,
                    phase_one,
                    phase_two,
                    extra_profile={
                        "engine_prepare_audio_phase_mode": 1.0,
                        "engine_prepare_audio_phase_wall_ms": float(phase_one_wall_ms),
                        "engine_prepare_audio_phase_batch_size": float(len(cpu_stages)),
                        "engine_prepare_text_phase_wall_ms": float(phase_two_wall_ms),
                        "engine_prepare_text_phase_batch_size": float(len(pending_phase_two)),
                    },
                )
            except Exception as exc:  # noqa: PERF203
                outputs[index] = exc
            finally:
                self._release_split_stage_slot()

        return [item if item is not None else RuntimeError("prepare batch result missing") for item in outputs]

    async def prepare_gpu_audio_phases_async(
        self,
        cpu_stages: list[PreparedCpuStage],
        prepared_ref_audio_futures: list[concurrent.futures.Future | None] | None = None,
    ) -> list[Dict[str, Any] | Exception]:
        if not cpu_stages:
            return []
        if self.enable_g2pw_audio_batch_merge and len(cpu_stages) > 1:
            return await self._prepare_gpu_phase_one_batch(cpu_stages)
        if prepared_ref_audio_futures is None:
            prepared_ref_audio_futures = [None] * len(cpu_stages)
        return list(
            await asyncio.gather(
                *[
                    self._prepare_gpu_phase_one(cpu_stage, prepared_ref_audio_future=prepared_ref_audio_future)
                    for cpu_stage, prepared_ref_audio_future in zip(cpu_stages, prepared_ref_audio_futures)
                ],
                return_exceptions=True,
            )
        )

    async def prepare_gpu_text_phases_async(
        self,
        items: list[tuple[PreparedCpuStage, Dict[str, Any]]],
    ) -> list[Dict[str, Any] | Exception]:
        if not items:
            return []
        return list(
            await asyncio.gather(
                *[self._prepare_gpu_phase_two(cpu_stage, phase_one) for cpu_stage, phase_one in items],
                return_exceptions=True,
            )
        )

    def build_gpu_prepare_result_from_phases(
        self,
        cpu_stage: PreparedCpuStage,
        phase_one: Dict[str, Any],
        phase_two: Dict[str, Any],
        extra_profile: Dict[str, float] | None = None,
    ) -> tuple[T2SRequestState, float, float]:
        try:
            return self._build_gpu_prepare_result(cpu_stage, phase_one, phase_two, extra_profile=extra_profile)
        finally:
            self._release_split_stage_slot()

    async def prepare_state_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> tuple[T2SRequestState, float, float]:
        cpu_stage = await self.prepare_cpu_stage_profiled_async(spec, prepare_submit_at)
        return await self.prepare_gpu_stage_profiled_async(cpu_stage)
