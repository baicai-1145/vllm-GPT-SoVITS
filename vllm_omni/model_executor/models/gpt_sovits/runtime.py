from __future__ import annotations

import asyncio
import concurrent.futures
import math
import os
import sys
import threading
import ctypes
import types
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, cast

import numpy as np
import torch
import torch.nn.functional as F

try:
    from vllm.logger import init_logger
except Exception:  # pragma: no cover - fallback for standalone runtime smoke tests
    import logging

    def init_logger(name: str):
        return logging.getLogger(name)

logger = init_logger(__name__)

_DEFAULT_PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "runtime_lib",
    )
)
_DEFAULT_CONFIG_PATH = "GPT_SoVITS/configs/tts_infer.yaml"


def _sync_runtime_device(device: Any) -> None:
    try:
        device_str = str(device)
        if device_str.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    except Exception:
        pass


_PREPARE_REF_AUDIO_MIN_SAMPLES_16K = 3 * 16000
_PREPARE_REF_AUDIO_MAX_SAMPLES_16K = 10 * 16000
_PREPARE_REF_RESAMPLE_TRANSFORMS: dict[tuple[int, int, str], Any] = {}
_PREPARE_REF_RESAMPLE_LOCK = threading.Lock()


def _safe_runtime_component_snapshot(component: Any) -> dict[str, Any] | None:
    if component is None or not hasattr(component, "snapshot"):
        return None
    try:
        snapshot = component.snapshot()
    except Exception:
        return None
    if snapshot is None:
        return None
    try:
        return dict(snapshot)
    except Exception:
        return None


def _safe_runtime_executor_queue_size(executor: Any) -> int:
    work_queue = getattr(executor, "_work_queue", None)
    if work_queue is None or not hasattr(work_queue, "qsize"):
        return 0
    try:
        return max(0, int(work_queue.qsize()))
    except Exception:
        return 0


def _detect_native_g2pw_runtime_state() -> dict[str, Any] | None:
    try:
        from text import chinese2

        g2pw_instance = getattr(chinese2, "g2pw", None)
        g2pw_backend = None if g2pw_instance is None else getattr(g2pw_instance, "_g2pw", None)
        if g2pw_backend is None or not hasattr(g2pw_backend, "snapshot"):
            return None
        snapshot = g2pw_backend.snapshot()
    except Exception:
        return None
    if snapshot is None:
        return None
    try:
        return dict(snapshot)
    except Exception:
        return None


def _build_native_prepare_runtime_state(tts: Any) -> dict[str, Any]:
    text_cpu_worker = getattr(tts, "prepare_text_cpu_worker", None)
    text_cpu_executor = getattr(tts, "prepare_text_cpu_executor", None)
    bert_worker = getattr(tts, "prepare_bert_batch_worker", None)
    ref_semantic_worker = getattr(tts, "prepare_ref_semantic_batch_worker", None)
    g2pw_batch_worker = getattr(tts, "prepare_g2pw_batch_worker", None)
    text_preprocessor = getattr(tts, "text_preprocessor", None)
    text_cpu_admission = None
    admission_builder = getattr(tts, "_build_text_cpu_admission_state", None)
    if callable(admission_builder):
        try:
            built_state = admission_builder()
        except Exception:
            built_state = None
        if isinstance(built_state, dict):
            text_cpu_admission = built_state
    g2pw_runtime = _detect_native_g2pw_runtime_state()
    if g2pw_runtime is None:
        g2pw_batch_state = _safe_runtime_component_snapshot(g2pw_batch_worker)
        if isinstance(g2pw_batch_state, dict):
            worker_count = g2pw_batch_state.get("worker_count")
            try:
                if worker_count is not None:
                    g2pw_runtime = {"worker_count": max(1, int(worker_count))}
            except Exception:
                g2pw_runtime = None
    return {
        "text_cpu": {
            "workers": int(getattr(tts, "prepare_text_cpu_workers", 0) or 0),
            "queue_size": _safe_runtime_executor_queue_size(text_cpu_executor),
            "enabled": bool(text_cpu_worker is not None or text_cpu_executor is not None),
            "worker": _safe_runtime_component_snapshot(text_cpu_worker),
            "admission": text_cpu_admission,
        },
        "bert": {
            "stage_limiter": _safe_runtime_component_snapshot(getattr(tts, "prepare_bert_stage_limiter", None)),
            "batch_worker": _safe_runtime_component_snapshot(bert_worker),
            "batching_enabled": bool(bert_worker is not None),
        },
        "ref_semantic": {
            "stage_limiter": _safe_runtime_component_snapshot(getattr(tts, "prepare_ref_semantic_stage_limiter", None)),
            "batch_worker": _safe_runtime_component_snapshot(ref_semantic_worker),
            "batching_enabled": bool(ref_semantic_worker is not None),
        },
        "ref_spec": {
            "stage_limiter": _safe_runtime_component_snapshot(getattr(tts, "prepare_ref_spec_stage_limiter", None)),
        },
        "ref_audio_limiters": {
            "split": bool(getattr(tts, "prepare_ref_stage_limiters_split", False)),
            "shared_legacy_limiter": bool(
                getattr(tts, "prepare_ref_semantic_stage_limiter", None)
                is getattr(tts, "prepare_ref_spec_stage_limiter", None)
            ),
        },
        "text_preprocessor": _safe_runtime_component_snapshot(text_preprocessor),
        "g2pw": g2pw_runtime,
        "g2pw_batch": _safe_runtime_component_snapshot(g2pw_batch_worker),
    }


def _resolve_native_prepare_runtime_state(tts: Any) -> dict[str, Any]:
    state_provider = getattr(tts, "_vllm_runtime_prepare_state_provider", None)
    if callable(state_provider):
        try:
            provided_state = state_provider()
        except Exception:
            provided_state = None
        if isinstance(provided_state, dict):
            return provided_state
    return _build_native_prepare_runtime_state(tts)


def _get_prepare_ref_resampler(sr0: int, sr1: int, device: str) -> Any:
    key = (int(sr0), int(sr1), str(device))
    with _PREPARE_REF_RESAMPLE_LOCK:
        transform = _PREPARE_REF_RESAMPLE_TRANSFORMS.get(key)
        if transform is not None:
            return transform
        import torchaudio

        transform = torchaudio.transforms.Resample(int(sr0), int(sr1)).to(device)
        _PREPARE_REF_RESAMPLE_TRANSFORMS[key] = transform
        return transform


def _prepare_prompt_semantic_wav16k_native(
    raw_audio: torch.Tensor,
    raw_sr: int,
    *,
    zero_wav_samples: int,
) -> torch.Tensor:
    resample_device = os.environ.get("GPTSOVITS_PREPARE_REF_RESAMPLE_DEVICE", "cpu").strip().lower() or "cpu"
    if resample_device not in {"cpu", "cuda"}:
        resample_device = "cpu"
    if resample_device == "cuda" and not torch.cuda.is_available():
        resample_device = "cpu"

    wav_mono = raw_audio
    if wav_mono.dim() == 2 and wav_mono.shape[0] != 1:
        wav_mono = wav_mono.mean(0, keepdim=True)
    wav16k = wav_mono.to(dtype=torch.float32, device=resample_device)
    if int(raw_sr) != 16000:
        wav16k = _get_prepare_ref_resampler(int(raw_sr), 16000, resample_device)(wav16k)
    wav16k = wav16k.squeeze(0).contiguous()
    if resample_device != "cpu":
        _sync_runtime_device(resample_device)
        wav16k = wav16k.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if wav16k.shape[0] > _PREPARE_REF_AUDIO_MAX_SAMPLES_16K or wav16k.shape[0] < _PREPARE_REF_AUDIO_MIN_SAMPLES_16K:
        raise OSError("参考音频在3~10秒范围外，请更换！")
    if int(zero_wav_samples) > 0:
        wav16k = torch.cat(
            [wav16k, torch.zeros(int(zero_wav_samples), dtype=torch.float32, device=wav16k.device)],
            dim=0,
        )
    return wav16k.contiguous()


def _prepare_prompt_semantic_wav16k_profile_native(
    tts: Any,
    raw_audio: torch.Tensor,
    raw_sr: int,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    sampling_rate = int(getattr(getattr(tts, "configs", None), "sampling_rate", 32000))
    zero_wav_samples = int(sampling_rate * 0.3)
    limiter = getattr(tts, "prepare_ref_audio_cpu_limiter", None)
    if limiter is None:
        cpu_prepare_start = time.perf_counter()
        wav16k = _prepare_prompt_semantic_wav16k_native(
            raw_audio,
            int(raw_sr),
            zero_wav_samples=zero_wav_samples,
        )
        cpu_prepare_ms = (time.perf_counter() - cpu_prepare_start) * 1000.0
        return wav16k, cpu_prepare_ms, {"wait_ms": 0.0, "slots": 0.0, "peak_inflight": 0.0}

    with limiter.enter() as limiter_stats:
        cpu_prepare_start = time.perf_counter()
        wav16k = _prepare_prompt_semantic_wav16k_native(
            raw_audio,
            int(raw_sr),
            zero_wav_samples=zero_wav_samples,
        )
        cpu_prepare_ms = (time.perf_counter() - cpu_prepare_start) * 1000.0
    return wav16k, cpu_prepare_ms, {
        "wait_ms": float(limiter_stats.get("wait_ms", 0.0)),
        "slots": float(limiter_stats.get("slots", 0.0)),
        "peak_inflight": float(limiter_stats.get("peak_inflight", 0.0)),
    }


def _left_pad_hidden(hidden: torch.Tensor, target_len: int) -> torch.Tensor:
    if hidden.shape[0] >= target_len:
        return hidden
    return F.pad(hidden, (0, 0, target_len - hidden.shape[0], 0), value=0)


def _make_pad_mask_left(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be 1-D, got ndim={lengths.ndim}")
    if lengths.numel() == 0:
        return torch.zeros((0, max_len), dtype=torch.bool, device=lengths.device)
    max_len = max(int(max_len), int(lengths.max().item()))
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expanded_lengths = seq_range.unsqueeze(0).repeat(lengths.size(0), 1)
    expanded_lengths -= (max_len - lengths).unsqueeze(-1)
    return expanded_lengths < 0


def _pad_token_sequences(
    token_sequences: Sequence[torch.LongTensor],
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    if not token_sequences:
        raise ValueError("token_sequences 不能为空")
    device = token_sequences[0].device
    max_len = max(int(sequence.shape[0]) for sequence in token_sequences)
    padded = torch.zeros((len(token_sequences), max_len), dtype=token_sequences[0].dtype, device=device)
    mask = torch.zeros((len(token_sequences), max_len), dtype=torch.bool, device=device)
    for row_index, sequence in enumerate(token_sequences):
        seq_len = int(sequence.shape[0])
        padded[row_index, :seq_len] = sequence
        mask[row_index, :seq_len] = True
    return padded, mask


@dataclass(slots=True)
class GPTSoVITSResult:
    sample_rate: int
    audio: np.ndarray


def _clone_transport_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if value is None:
        return torch.empty((0,), dtype=dtype)
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").contiguous().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype).detach().to("cpu").contiguous()


@dataclass(slots=True)
class GPTSoVITSStageTransport:
    request_id: str
    semantic_tokens: torch.Tensor
    phones: torch.Tensor
    prompt_phones: torch.Tensor
    prompt_semantic: torch.Tensor
    refer_audio_spec: torch.Tensor
    refer_audio_16k: torch.Tensor
    raw_audio: torch.Tensor
    raw_sr: int
    speed_factor: float
    sample_steps: int
    super_sampling: bool

    @property
    def semantic_token_count(self) -> int:
        return int(self.semantic_tokens.numel())

    @classmethod
    def empty(
        cls,
        *,
        request_id: str = "",
        speed_factor: float = 1.0,
        sample_steps: int = 32,
        super_sampling: bool = False,
    ) -> "GPTSoVITSStageTransport":
        return cls(
            request_id=str(request_id),
            semantic_tokens=torch.empty((0,), dtype=torch.long),
            phones=torch.empty((0,), dtype=torch.long),
            prompt_phones=torch.empty((0,), dtype=torch.long),
            prompt_semantic=torch.empty((0,), dtype=torch.long),
            refer_audio_spec=torch.empty((0,), dtype=torch.float32),
            refer_audio_16k=torch.empty((0,), dtype=torch.float32),
            raw_audio=torch.empty((0,), dtype=torch.float32),
            raw_sr=0,
            speed_factor=float(speed_factor),
            sample_steps=int(sample_steps),
            super_sampling=bool(super_sampling),
        )

    @classmethod
    def from_state(cls, state: Any, spec: "GPTSoVITSRequestSpec") -> "GPTSoVITSStageTransport":
        refer_spec = getattr(state, "refer_spec", None)
        refer_audio_spec = refer_spec.spec_audio if refer_spec is not None else None
        refer_audio_16k = refer_spec.audio_16k if refer_spec is not None else None
        return cls(
            request_id=str(getattr(state, "request_id", spec.request_id)),
            semantic_tokens=torch.empty((0,), dtype=torch.long),
            phones=_clone_transport_tensor(getattr(state, "phones", None), dtype=torch.long),
            prompt_phones=_clone_transport_tensor(getattr(state, "prompt_phones", None), dtype=torch.long),
            prompt_semantic=_clone_transport_tensor(getattr(state, "prompt_semantic", None), dtype=torch.long),
            refer_audio_spec=_clone_transport_tensor(refer_audio_spec, dtype=torch.float32),
            refer_audio_16k=_clone_transport_tensor(refer_audio_16k, dtype=torch.float32),
            raw_audio=_clone_transport_tensor(getattr(state, "raw_audio", None), dtype=torch.float32),
            raw_sr=int(getattr(state, "raw_sr", 0)),
            speed_factor=float(spec.speed_factor),
            sample_steps=int(spec.sample_steps),
            super_sampling=bool(spec.super_sampling),
        )

    @classmethod
    def from_info(
        cls,
        info: Any,
        *,
        semantic_tokens: torch.Tensor | None = None,
    ) -> "GPTSoVITSStageTransport":
        if isinstance(info, cls):
            base = info
        elif isinstance(info, dict):
            transport = info.get("gpt_sovits_transport")
            if isinstance(transport, cls):
                base = transport
            elif isinstance(transport, dict):
                base = cls(
                    request_id=str(transport.get("request_id", "")),
                    semantic_tokens=_clone_transport_tensor(transport.get("semantic_tokens"), dtype=torch.long),
                    phones=_clone_transport_tensor(transport.get("phones"), dtype=torch.long),
                    prompt_phones=_clone_transport_tensor(transport.get("prompt_phones"), dtype=torch.long),
                    prompt_semantic=_clone_transport_tensor(transport.get("prompt_semantic"), dtype=torch.long),
                    refer_audio_spec=_clone_transport_tensor(transport.get("refer_audio_spec"), dtype=torch.float32),
                    refer_audio_16k=_clone_transport_tensor(transport.get("refer_audio_16k"), dtype=torch.float32),
                    raw_audio=_clone_transport_tensor(transport.get("raw_audio"), dtype=torch.float32),
                    raw_sr=int(transport.get("raw_sr", 0) or 0),
                    speed_factor=float(transport.get("speed_factor", 1.0)),
                    sample_steps=int(transport.get("sample_steps", 32)),
                    super_sampling=bool(transport.get("super_sampling", False)),
                )
            else:
                request_id = str(
                    info.get("gpt_sovits_request_id")
                    or info.get("engine_request_id")
                    or info.get("request_id")
                    or ""
                )
                base = cls(
                    request_id=request_id,
                    semantic_tokens=_clone_transport_tensor(info.get("gpt_sovits_semantic_tokens"), dtype=torch.long),
                    phones=_clone_transport_tensor(info.get("gpt_sovits_phones"), dtype=torch.long),
                    prompt_phones=_clone_transport_tensor(info.get("gpt_sovits_prompt_phones"), dtype=torch.long),
                    prompt_semantic=_clone_transport_tensor(info.get("gpt_sovits_prompt_semantic"), dtype=torch.long),
                    refer_audio_spec=_clone_transport_tensor(info.get("gpt_sovits_refer_audio_spec"), dtype=torch.float32),
                    refer_audio_16k=_clone_transport_tensor(info.get("gpt_sovits_refer_audio_16k"), dtype=torch.float32),
                    raw_audio=_clone_transport_tensor(info.get("gpt_sovits_raw_audio"), dtype=torch.float32),
                    raw_sr=int(info.get("gpt_sovits_raw_sr", 0) or 0),
                    speed_factor=float(info.get("gpt_sovits_speed_factor", 1.0)),
                    sample_steps=int(info.get("gpt_sovits_sample_steps", 32)),
                    super_sampling=bool(info.get("gpt_sovits_super_sampling", False)),
                )
        else:
            base = cls.empty()
        if semantic_tokens is None:
            return base
        return base.with_semantic_tokens(semantic_tokens)

    def with_semantic_tokens(self, semantic_tokens: torch.Tensor) -> "GPTSoVITSStageTransport":
        return GPTSoVITSStageTransport(
            request_id=self.request_id,
            semantic_tokens=_clone_transport_tensor(semantic_tokens, dtype=torch.long),
            phones=self.phones,
            prompt_phones=self.prompt_phones,
            prompt_semantic=self.prompt_semantic,
            refer_audio_spec=self.refer_audio_spec,
            refer_audio_16k=self.refer_audio_16k,
            raw_audio=self.raw_audio,
            raw_sr=int(self.raw_sr),
            speed_factor=float(self.speed_factor),
            sample_steps=int(self.sample_steps),
            super_sampling=bool(self.super_sampling),
        )

    def has_decode_conditioning(self) -> bool:
        required = (
            self.semantic_tokens,
            self.phones,
            self.prompt_phones,
            self.prompt_semantic,
            self.refer_audio_spec,
            self.raw_audio,
        )
        return all(isinstance(value, torch.Tensor) and value.numel() > 0 for value in required)

    def to_transport_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "semantic_tokens": self.semantic_tokens,
            "phones": self.phones,
            "prompt_phones": self.prompt_phones,
            "prompt_semantic": self.prompt_semantic,
            "refer_audio_spec": self.refer_audio_spec,
            "refer_audio_16k": self.refer_audio_16k,
            "raw_audio": self.raw_audio,
            "raw_sr": int(self.raw_sr),
            "speed_factor": float(self.speed_factor),
            "sample_steps": int(self.sample_steps),
            "super_sampling": bool(self.super_sampling),
        }

    def to_additional_information(self) -> dict[str, Any]:
        return {
            "gpt_sovits_transport": self.to_transport_dict(),
            "gpt_sovits_request_id": self.request_id,
            "gpt_sovits_semantic_token_count": self.semantic_token_count,
        }

    def to_model_intermediate_buffer(self) -> dict[str, Any]:
        return self.to_additional_information()


@dataclass(slots=True)
class GPTSoVITSPreparedRequest:
    request_id: str
    state: Any
    transport_info: GPTSoVITSStageTransport | dict[str, Any]


@dataclass(slots=True)
class GPTSoVITSRequestSpec:
    request_id: str
    ref_audio_path: str
    prompt_text: str
    prompt_lang: str
    text: str
    text_lang: str
    text_split_method: str
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    early_stop_num: int
    aux_ref_audio_paths: list[str]
    speed_factor: float = 1.0
    sample_steps: int = 32
    super_sampling: bool = False
    ready_step: int = 0


@dataclass(slots=True)
class GPTSoVITSPreparedCpuStage:
    request_id: str
    spec: GPTSoVITSRequestSpec
    prepare_submit_at: float
    prepare_start: float
    prompt_text: str
    text: str
    prepare_admission_wait_ms: float
    current_inflight: int
    peak_inflight: int
    prompt_cpu_profiled: Any
    target_cpu_profiled: Any
    ref_audio_prepare_future: Any | None = None

    @property
    def cpu_stage(self) -> "GPTSoVITSPreparedCpuStage":
        return self


@dataclass(slots=True)
class GPTSoVITSNativePreparedCpuStage:
    spec: GPTSoVITSRequestSpec
    prepare_submit_at: float
    prepare_start: float
    prompt_text: str
    text: str
    prepare_admission_wait_ms: float
    current_inflight: int
    peak_inflight: int
    prompt_cpu_profiled: Any
    target_cpu_profiled: Any


@dataclass(slots=True)
class GPTSoVITSPrepareAudioPhaseData:
    prompt_g2pw_profiled: Any
    target_g2pw_profiled: Any
    ref_audio_profiled: Any
    g2pw_pair_ms: float = 0.0
    phase_wall_ms: float = 0.0


@dataclass(slots=True)
class GPTSoVITSReferSpec:
    spec_audio: Any
    audio_16k: Any | None = None


@dataclass(slots=True)
class GPTSoVITSPrepareRefSpecResult:
    refer_spec: GPTSoVITSReferSpec
    profile: dict[str, float]


@dataclass(slots=True)
class GPTSoVITSPrepareTextPhaseData:
    prompt_feature_profiled: Any
    target_feature_profiled: Any
    phase_wall_ms: float = 0.0


@dataclass(slots=True)
class GPTSoVITSPreparedRefAudioAsset:
    raw_audio: Any
    raw_sr: int
    wav16k: Any
    profile: dict[str, float]


@dataclass(slots=True)
class GPTSoVITSRefAudioBundle:
    prompt_semantic: Any
    raw_audio: Any
    raw_sr: int
    profile: dict[str, float]
    refer_spec: GPTSoVITSReferSpec | None = None


@dataclass(slots=True)
class GPTSoVITSTextFeatures:
    phones: list[int]
    bert_features: torch.Tensor
    norm_text: str
    profile: dict[str, float]
    total_ms: float
    cpu_preprocess_ms: float


class _GPTSoVITSNoopPrepareGate:
    max_inflight = 0

    async def acquire(self) -> dict[str, float]:
        return {
            "wait_ms": 0.0,
            "inflight": 0.0,
            "peak_inflight": 0.0,
            "max_inflight": 0.0,
        }

    def release(self) -> None:
        return None


_GPTSOVITS_NOOP_PREPARE_GATE = _GPTSoVITSNoopPrepareGate()


class _GPTSoVITSAsyncStageGate:
    def __init__(self, max_inflight: int, poll_ms: int = 1):
        self.max_inflight = max(0, int(max_inflight))
        self.lock = threading.Lock()
        self.poll_s = max(0.0005, float(max(1, int(poll_ms))) / 1000.0)
        self.inflight = 0
        self.peak_inflight = 0
        self.total_entered = 0
        self.total_wait_ms = 0.0
        self.wait_peak_ms = 0.0

    async def acquire(self) -> dict[str, float]:
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

    def snapshot(self) -> dict[str, float]:
        with self.lock:
            return {
                "max_inflight": float(self.max_inflight),
                "inflight": float(self.inflight),
                "peak_inflight": float(self.peak_inflight),
                "total_entered": float(self.total_entered),
                "total_wait_ms": float(self.total_wait_ms),
                "wait_peak_ms": float(self.wait_peak_ms),
            }


@dataclass(slots=True)
class GPTSoVITSPrepareRuntimeCoordinator:
    text_cpu_gate: Any = _GPTSOVITS_NOOP_PREPARE_GATE
    inflight_gate: Any = _GPTSOVITS_NOOP_PREPARE_GATE
    text_feature_gate: Any = _GPTSOVITS_NOOP_PREPARE_GATE
    g2pw_gate: Any = _GPTSOVITS_NOOP_PREPARE_GATE
    ref_audio_gate: Any = _GPTSOVITS_NOOP_PREPARE_GATE
    ref_load_gate: Any = _GPTSOVITS_NOOP_PREPARE_GATE
    ref_spec_gate: Any = _GPTSOVITS_NOOP_PREPARE_GATE
    text_feature_executor: Any | None = None
    g2pw_executor: Any | None = None
    ref_audio_executor: Any | None = None
    enable_g2pw_pair_batch: bool = False
    enable_g2pw_audio_batch_merge: bool = False
    g2pw_audio_batch_merge_group_size: int = 8
    submit_prepare_ref_audio_asset_fn: Any | None = None
    mark_enter_fn: Any | None = None
    release_split_stage_slot_fn: Any | None = None
    tts: Any | None = None
    lock: Any | None = None
    inflight: int = 0
    peak_inflight: int = 0
    ref_audio_asset_cache_ttl_sec: float = 0.0
    ref_audio_asset_cache_max_entries: int = 0
    ref_audio_asset_lock: Any | None = None
    ref_audio_asset_inflight: dict[str, concurrent.futures.Future] = field(default_factory=dict)
    ref_audio_asset_cache: dict[str, tuple[GPTSoVITSPreparedRefAudioAsset, float]] = field(default_factory=dict)
    ref_prompt_semantic_runtime_exact_prewarm_enabled: bool = False
    ref_prompt_semantic_runtime_exact_prewarm_max_unique: int = 0
    ref_prompt_semantic_runtime_exact_prewarm_batch_sizes: tuple[int, ...] = ()
    ref_prompt_semantic_runtime_exact_prewarm_lock: Any | None = None
    ref_prompt_semantic_runtime_exact_prewarmed_samples: set[int] = field(default_factory=set)
    ref_prompt_semantic_runtime_exact_prewarm_inflight_samples: set[int] = field(default_factory=set)
    ref_prompt_semantic_runtime_exact_prewarm_total: int = 0
    ref_prompt_semantic_runtime_exact_prewarm_total_ms: float = 0.0
    ref_prompt_semantic_runtime_exact_prewarm_peak_ms: float = 0.0
    ref_prompt_semantic_bucket_first_hit_serialization_enabled: bool = False
    ref_prompt_semantic_bucket_first_hit_required_hits: int = 0
    ref_prompt_semantic_bucket_first_hit_bucket_indices: tuple[int, ...] = ()
    ref_prompt_semantic_bucket_first_hit_lock: Any | None = None
    ref_prompt_semantic_bucket_first_hit_states: dict[int, dict[str, int]] = field(default_factory=dict)
    ref_prompt_semantic_bucket_aware_sharding: bool = False
    ref_prompt_semantic_bucket_aware_max_outstanding_gap: int = 0

    @staticmethod
    def _resolve_ref_prompt_semantic_runtime_exact_prewarm_batch_sizes(tts: Any) -> tuple[int, ...]:
        worker = getattr(tts, "prepare_ref_semantic_batch_worker", None)
        worker_batch_sizes = getattr(worker, "runtime_exact_prewarm_batch_sizes", None)
        if worker_batch_sizes is not None:
            try:
                normalized = sorted({max(1, int(value)) for value in worker_batch_sizes})
            except Exception:
                normalized = []
            if normalized:
                return tuple(normalized)
        raw_batch_sizes = str(os.environ.get("GPTSOVITS_PREPARE_REF_RUNTIME_EXACT_PREWARM_BATCH_SIZES", "1")).strip()
        normalized = []
        for item in raw_batch_sizes.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                normalized.append(max(1, int(item)))
            except Exception:
                continue
        return tuple(sorted(set(normalized)) or [1])

    @staticmethod
    def _resolve_ref_prompt_semantic_bucket_first_hit_bucket_indices(tts: Any) -> tuple[int, ...]:
        worker = getattr(tts, "prepare_ref_semantic_batch_worker", None)
        worker_bucket_indices = getattr(worker, "bucket_first_hit_bucket_indices", None)
        if worker_bucket_indices is not None:
            try:
                normalized = sorted({max(0, int(value)) for value in worker_bucket_indices})
            except Exception:
                normalized = []
            if normalized:
                return tuple(normalized)
        raw_bucket_indices = str(
            os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_SERIALIZE_FIRST_HIT_BUCKET_INDICES", "3,4,9")
        ).strip()
        normalized = []
        for item in raw_bucket_indices.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                normalized.append(max(0, int(item)))
            except Exception:
                continue
        return tuple(sorted(set(normalized)))

    @classmethod
    def from_runtime_coordinator(cls, coordinator: Any) -> "GPTSoVITSPrepareRuntimeCoordinator":
        if isinstance(coordinator, cls):
            return coordinator
        return cls(
            text_cpu_gate=getattr(coordinator, "text_cpu_gate", _GPTSOVITS_NOOP_PREPARE_GATE),
            inflight_gate=getattr(coordinator, "_inflight_gate", _GPTSOVITS_NOOP_PREPARE_GATE),
            text_feature_gate=getattr(coordinator, "text_feature_gate", _GPTSOVITS_NOOP_PREPARE_GATE),
            g2pw_gate=getattr(coordinator, "g2pw_gate", _GPTSOVITS_NOOP_PREPARE_GATE),
            ref_audio_gate=getattr(coordinator, "ref_audio_gate", _GPTSOVITS_NOOP_PREPARE_GATE),
            ref_load_gate=getattr(coordinator, "ref_load_gate", _GPTSOVITS_NOOP_PREPARE_GATE),
            ref_spec_gate=getattr(coordinator, "ref_spec_gate", _GPTSOVITS_NOOP_PREPARE_GATE),
            text_feature_executor=getattr(coordinator, "text_feature_executor", None),
            g2pw_executor=getattr(coordinator, "g2pw_executor", None),
            ref_audio_executor=getattr(coordinator, "ref_audio_executor", None),
            enable_g2pw_pair_batch=bool(getattr(coordinator, "enable_g2pw_pair_batch", False)),
            enable_g2pw_audio_batch_merge=bool(getattr(coordinator, "enable_g2pw_audio_batch_merge", False)),
            g2pw_audio_batch_merge_group_size=max(
                1,
                int(getattr(coordinator, "g2pw_audio_batch_merge_group_size", 8) or 8),
            ),
            submit_prepare_ref_audio_asset_fn=getattr(coordinator, "submit_prepare_ref_audio_asset", None),
            mark_enter_fn=getattr(coordinator, "_mark_enter", None),
            release_split_stage_slot_fn=getattr(coordinator, "_release_split_stage_slot", None),
            tts=getattr(coordinator, "tts", None),
            lock=getattr(coordinator, "lock", None),
            inflight=int(getattr(coordinator, "inflight", 0) or 0),
            peak_inflight=int(getattr(coordinator, "peak_inflight", 0) or 0),
            ref_audio_asset_cache_ttl_sec=float(getattr(coordinator, "ref_audio_asset_cache_ttl_sec", 0.0) or 0.0),
            ref_audio_asset_cache_max_entries=int(getattr(coordinator, "ref_audio_asset_cache_max_entries", 0) or 0),
            ref_audio_asset_lock=getattr(coordinator, "ref_audio_asset_lock", None),
            ref_audio_asset_inflight=dict(getattr(coordinator, "ref_audio_asset_inflight", {}) or {}),
            ref_audio_asset_cache=dict(getattr(coordinator, "ref_audio_asset_cache", {}) or {}),
            ref_prompt_semantic_runtime_exact_prewarm_enabled=bool(
                getattr(
                    coordinator,
                    "ref_prompt_semantic_runtime_exact_prewarm_enabled",
                    os.environ.get("GPTSOVITS_PREPARE_REF_RUNTIME_EXACT_PREWARM", "0") not in {"0", "false", "False"},
                )
            ),
            ref_prompt_semantic_runtime_exact_prewarm_max_unique=max(
                0,
                int(
                    getattr(
                        coordinator,
                        "ref_prompt_semantic_runtime_exact_prewarm_max_unique",
                        os.environ.get("GPTSOVITS_PREPARE_REF_RUNTIME_EXACT_PREWARM_MAX_UNIQUE", "4"),
                    )
                    or 0
                ),
            ),
            ref_prompt_semantic_runtime_exact_prewarm_batch_sizes=tuple(
                int(value)
                for value in (
                    getattr(coordinator, "ref_prompt_semantic_runtime_exact_prewarm_batch_sizes", ()) or ()
                )
            ),
            ref_prompt_semantic_runtime_exact_prewarm_lock=getattr(
                coordinator,
                "ref_prompt_semantic_runtime_exact_prewarm_lock",
                threading.Lock(),
            ),
            ref_prompt_semantic_runtime_exact_prewarmed_samples=set(
                getattr(coordinator, "ref_prompt_semantic_runtime_exact_prewarmed_samples", set()) or set()
            ),
            ref_prompt_semantic_runtime_exact_prewarm_inflight_samples=set(
                getattr(coordinator, "ref_prompt_semantic_runtime_exact_prewarm_inflight_samples", set()) or set()
            ),
            ref_prompt_semantic_runtime_exact_prewarm_total=int(
                getattr(coordinator, "ref_prompt_semantic_runtime_exact_prewarm_total", 0) or 0
            ),
            ref_prompt_semantic_runtime_exact_prewarm_total_ms=float(
                getattr(coordinator, "ref_prompt_semantic_runtime_exact_prewarm_total_ms", 0.0) or 0.0
            ),
            ref_prompt_semantic_runtime_exact_prewarm_peak_ms=float(
                getattr(coordinator, "ref_prompt_semantic_runtime_exact_prewarm_peak_ms", 0.0) or 0.0
            ),
            ref_prompt_semantic_bucket_first_hit_serialization_enabled=bool(
                getattr(
                    coordinator,
                    "ref_prompt_semantic_bucket_first_hit_serialization_enabled",
                    str(os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_SERIALIZE_FIRST_HITS", "1")).strip().lower()
                    not in {"0", "false", "no", "off"},
                )
            ),
            ref_prompt_semantic_bucket_first_hit_required_hits=max(
                0,
                int(
                    getattr(
                        coordinator,
                        "ref_prompt_semantic_bucket_first_hit_required_hits",
                        os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_SERIALIZE_FIRST_HITS_REQUIRED", "1"),
                    )
                    or 0
                ),
            ),
            ref_prompt_semantic_bucket_first_hit_bucket_indices=tuple(
                int(value)
                for value in (
                    getattr(coordinator, "ref_prompt_semantic_bucket_first_hit_bucket_indices", ()) or ()
                )
            ),
            ref_prompt_semantic_bucket_first_hit_lock=getattr(
                coordinator,
                "ref_prompt_semantic_bucket_first_hit_lock",
                threading.Lock(),
            ),
            ref_prompt_semantic_bucket_first_hit_states={
                int(bucket_index): dict(state)
                for bucket_index, state in (
                    dict(getattr(coordinator, "ref_prompt_semantic_bucket_first_hit_states", {}) or {}).items()
                )
            },
            ref_prompt_semantic_bucket_aware_sharding=bool(
                getattr(
                    coordinator,
                    "ref_prompt_semantic_bucket_aware_sharding",
                    str(os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_AWARE_SHARDING", "1")).strip().lower()
                    not in {"0", "false", "no", "off"},
                )
            ),
            ref_prompt_semantic_bucket_aware_max_outstanding_gap=max(
                0,
                int(
                    getattr(
                        coordinator,
                        "ref_prompt_semantic_bucket_aware_max_outstanding_gap",
                        os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_AWARE_MAX_OUTSTANDING_GAP", "2"),
                    )
                    or 0
                ),
            ),
        )

    @staticmethod
    def _detect_g2pw_runtime_workers(tts: Any) -> int | None:
        runtime_state = _resolve_native_prepare_runtime_state(tts)
        g2pw_state = runtime_state.get("g2pw")
        if not isinstance(g2pw_state, dict):
            return None
        worker_count = g2pw_state.get("worker_count")
        try:
            worker_count = int(worker_count)
        except Exception:
            return None
        return max(1, worker_count)

    @classmethod
    def build_native(cls, tts: Any) -> "GPTSoVITSPrepareRuntimeCoordinator":
        gate_poll_ms = int(os.environ.get("GPTSOVITS_PREPARE_GATE_POLL_MS", "1"))
        use_async_text_feature_path = bool(
            getattr(tts, "prepare_bert_batch_worker", None) is not None
            and os.environ.get("GPTSOVITS_PREPARE_TEXT_FEATURE_DIRECT", "0") != "0"
        )
        text_feature_workers = 0
        text_feature_executor = None
        if not use_async_text_feature_path:
            text_feature_default_workers = 16
            text_feature_workers = max(
                1,
                int(os.environ.get("GPTSOVITS_PREPARE_TEXT_FEATURE_WORKERS", str(text_feature_default_workers))),
            )
            text_feature_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=text_feature_workers,
                thread_name_prefix="prepare-text-feature",
            )
        g2pw_runtime_workers = cls._detect_g2pw_runtime_workers(tts)
        g2pw_default_workers = (
            int(g2pw_runtime_workers)
            if g2pw_runtime_workers is not None
            else max(8, int(getattr(tts, "prepare_text_cpu_workers", 8) or 8))
        )
        g2pw_workers = max(
            1,
            int(os.environ.get("GPTSOVITS_PREPARE_G2PW_WORKERS", str(g2pw_default_workers))),
        )
        ref_audio_default_workers = max(1, int(os.environ.get("GPTSOVITS_PREPARE_REF_SLOTS", "4")))
        ref_audio_workers = max(
            1,
            int(os.environ.get("GPTSOVITS_PREPARE_REF_ASYNC_WORKERS", str(ref_audio_default_workers))),
        )
        text_cpu_gate_default = 0
        g2pw_gate_default = (
            int(g2pw_runtime_workers) if g2pw_runtime_workers is not None else max(0, int(g2pw_workers))
        )
        text_feature_gate_default = max(0, int(text_feature_workers))
        ref_audio_gate_default = max(0, int(ref_audio_workers))
        return cls(
            text_cpu_gate=_GPTSoVITSAsyncStageGate(
                int(os.environ.get("GPTSOVITS_PREPARE_TEXT_CPU_MAX_INFLIGHT", str(text_cpu_gate_default))),
                poll_ms=gate_poll_ms,
            ),
            inflight_gate=_GPTSoVITSAsyncStageGate(
                int(os.environ.get("GPTSOVITS_PREPARE_MAX_INFLIGHT", "0")),
                poll_ms=gate_poll_ms,
            ),
            text_feature_gate=_GPTSoVITSAsyncStageGate(
                int(os.environ.get("GPTSOVITS_PREPARE_TEXT_FEATURE_MAX_INFLIGHT", str(text_feature_gate_default))),
                poll_ms=gate_poll_ms,
            ),
            g2pw_gate=_GPTSoVITSAsyncStageGate(
                int(os.environ.get("GPTSOVITS_PREPARE_G2PW_MAX_INFLIGHT", str(g2pw_gate_default))),
                poll_ms=gate_poll_ms,
            ),
            ref_audio_gate=_GPTSoVITSAsyncStageGate(
                int(os.environ.get("GPTSOVITS_PREPARE_REF_MAX_INFLIGHT", str(ref_audio_gate_default))),
                poll_ms=gate_poll_ms,
            ),
            ref_load_gate=_GPTSoVITSAsyncStageGate(
                int(os.environ.get("GPTSOVITS_PREPARE_REF_LOAD_MAX_INFLIGHT", str(ref_audio_gate_default))),
                poll_ms=gate_poll_ms,
            ),
            ref_spec_gate=_GPTSoVITSAsyncStageGate(
                int(os.environ.get("GPTSOVITS_PREPARE_REF_SPEC_MAX_INFLIGHT", str(ref_audio_gate_default))),
                poll_ms=gate_poll_ms,
            ),
            text_feature_executor=text_feature_executor,
            g2pw_executor=concurrent.futures.ThreadPoolExecutor(
                max_workers=g2pw_workers,
                thread_name_prefix="prepare-g2pw",
            ),
            ref_audio_executor=concurrent.futures.ThreadPoolExecutor(
                max_workers=ref_audio_workers,
                thread_name_prefix="prepare-ref-audio",
            ),
            enable_g2pw_pair_batch=os.environ.get("GPTSOVITS_PREPARE_G2PW_PAIR_BATCH", "1") != "0",
            enable_g2pw_audio_batch_merge=os.environ.get("GPTSOVITS_PREPARE_G2PW_AUDIO_BATCH_MERGE", "0") != "0",
            g2pw_audio_batch_merge_group_size=max(
                1,
                int(os.environ.get("GPTSOVITS_PREPARE_G2PW_AUDIO_BATCH_GROUP_SIZE", "8")),
            ),
            tts=tts,
            lock=threading.Lock(),
            ref_audio_asset_cache_ttl_sec=max(
                0.0,
                float(os.environ.get("GPTSOVITS_PREPARE_REF_AUDIO_ASSET_CACHE_TTL_SEC", "15")),
            ),
            ref_audio_asset_cache_max_entries=max(
                0,
                int(os.environ.get("GPTSOVITS_PREPARE_REF_AUDIO_ASSET_CACHE_MAX_ENTRIES", "4")),
            ),
            ref_audio_asset_lock=threading.Lock(),
            ref_prompt_semantic_runtime_exact_prewarm_enabled=(
                str(os.environ.get("GPTSOVITS_PREPARE_REF_RUNTIME_EXACT_PREWARM", "0")).strip().lower()
                not in {"0", "false", "no", "off"}
            ),
            ref_prompt_semantic_runtime_exact_prewarm_max_unique=max(
                0,
                int(os.environ.get("GPTSOVITS_PREPARE_REF_RUNTIME_EXACT_PREWARM_MAX_UNIQUE", "4")),
            ),
            ref_prompt_semantic_runtime_exact_prewarm_batch_sizes=cls._resolve_ref_prompt_semantic_runtime_exact_prewarm_batch_sizes(tts),
            ref_prompt_semantic_runtime_exact_prewarm_lock=threading.Lock(),
            ref_prompt_semantic_bucket_first_hit_serialization_enabled=(
                str(os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_SERIALIZE_FIRST_HITS", "1")).strip().lower()
                not in {"0", "false", "no", "off"}
            ),
            ref_prompt_semantic_bucket_first_hit_required_hits=max(
                0,
                int(os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_SERIALIZE_FIRST_HITS_REQUIRED", "1")),
            ),
            ref_prompt_semantic_bucket_first_hit_bucket_indices=cls._resolve_ref_prompt_semantic_bucket_first_hit_bucket_indices(
                tts
            ),
            ref_prompt_semantic_bucket_first_hit_lock=threading.Lock(),
            ref_prompt_semantic_bucket_aware_sharding=(
                str(os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_AWARE_SHARDING", "1")).strip().lower()
                not in {"0", "false", "no", "off"}
            ),
            ref_prompt_semantic_bucket_aware_max_outstanding_gap=max(
                0,
                int(os.environ.get("GPTSOVITS_PREPARE_REF_BUCKET_AWARE_MAX_OUTSTANDING_GAP", "2")),
            ),
        )

    @staticmethod
    def _coerce_prepared_ref_audio_asset(asset: Any) -> Any:
        if isinstance(asset, GPTSoVITSPreparedRefAudioAsset):
            return asset
        if all(hasattr(asset, name) for name in ("raw_audio", "raw_sr", "wav16k")):
            return GPTSoVITSPreparedRefAudioAsset(
                raw_audio=getattr(asset, "raw_audio"),
                raw_sr=int(getattr(asset, "raw_sr")),
                wav16k=getattr(asset, "wav16k"),
                profile=dict(getattr(asset, "profile", {}) or {}),
            )
        return asset

    @classmethod
    def _coerce_prepare_profiled_result(cls, profiled: Any) -> Any:
        if isinstance(profiled, GPTSoVITSPrepareProfiledResult):
            native_result = cls._coerce_prepared_ref_audio_asset(profiled.result)
            if native_result is profiled.result:
                return profiled
            return GPTSoVITSPrepareProfiledResult(
                result=native_result,
                submit_at=float(profiled.submit_at),
                started_at=float(profiled.started_at),
                finished_at=float(profiled.finished_at),
                profile=dict(profiled.profile or {}) if profiled.profile is not None else None,
            )
        if all(hasattr(profiled, name) for name in ("result", "submit_at", "started_at", "finished_at")):
            return GPTSoVITSPrepareProfiledResult(
                result=cls._coerce_prepared_ref_audio_asset(getattr(profiled, "result")),
                submit_at=float(getattr(profiled, "submit_at")),
                started_at=float(getattr(profiled, "started_at")),
                finished_at=float(getattr(profiled, "finished_at")),
                profile=dict(getattr(profiled, "profile", {}) or {}) or None,
            )
        return profiled

    @classmethod
    def _wrap_prepare_future(cls, value: Any) -> Any:
        if not (hasattr(value, "add_done_callback") and hasattr(value, "result")):
            return cls._coerce_prepare_profiled_result(value)
        wrapped: concurrent.futures.Future = concurrent.futures.Future()

        def _forward(done_future: Any) -> None:
            try:
                wrapped.set_result(cls._coerce_prepare_profiled_result(done_future.result()))
            except BaseException as exc:  # pragma: no cover - passthrough
                wrapped.set_exception(exc)

        value.add_done_callback(_forward)
        return wrapped

    @staticmethod
    def _normalize_ref_audio_cache_key(ref_audio_path: str) -> str:
        return os.path.abspath(str(ref_audio_path))

    @staticmethod
    def _clone_prepared_ref_audio_asset(
        asset: GPTSoVITSPreparedRefAudioAsset,
        *,
        extra_profile: dict[str, float] | None = None,
    ) -> GPTSoVITSPreparedRefAudioAsset:
        profile = dict(asset.profile or {})
        if extra_profile:
            profile.update({key: float(value) for key, value in extra_profile.items()})
        return GPTSoVITSPreparedRefAudioAsset(
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
        asset: GPTSoVITSPreparedRefAudioAsset,
        *,
        submit_ts: float,
        cache_age_ms: float,
    ) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        future.set_result(
            GPTSoVITSPrepareProfiledResult(
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

    def _prepare_ref_audio_asset_native(
        self,
        ref_audio_path: str,
        *,
        submit_at: float,
    ) -> GPTSoVITSPrepareProfiledResult:
        if self.tts is None:
            raise RuntimeError("GPT-SoVITS prepare coordinator missing native TTS runtime")
        started_at = time.perf_counter()
        load_start = time.perf_counter()
        raw_audio, raw_sr = self.tts._load_ref_audio_raw(ref_audio_path)
        load_ms = (time.perf_counter() - load_start) * 1000.0
        wav16k, cpu_prepare_ms, limiter_stats = _prepare_prompt_semantic_wav16k_profile_native(
            self.tts,
            raw_audio,
            raw_sr,
        )
        finished_at = time.perf_counter()
        return GPTSoVITSPrepareProfiledResult(
            result=GPTSoVITSPreparedRefAudioAsset(
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
            ),
            submit_at=float(submit_at),
            started_at=float(started_at),
            finished_at=float(finished_at),
        )

    def submit_prepare_ref_audio_asset(self, ref_audio_path: str, *, submit_at: float | None = None) -> Any:
        submit_fn = self.submit_prepare_ref_audio_asset_fn
        if callable(submit_fn):
            return self._wrap_prepare_future(submit_fn(ref_audio_path, submit_at=submit_at))

        submit_ts = time.perf_counter() if submit_at is None else float(submit_at)
        ref_audio_path = self._normalize_ref_audio_cache_key(ref_audio_path)
        cache_lock = self.ref_audio_asset_lock
        executor = self.ref_audio_executor
        if cache_lock is None or executor is None:
            raise RuntimeError("GPT-SoVITS prepare coordinator does not provide native ref-audio preload support")

        with cache_lock:
            now = time.perf_counter()
            self._prune_ref_audio_asset_cache_locked(now)
            cached_entry = self.ref_audio_asset_cache.get(ref_audio_path)
            if cached_entry is not None:
                cached_asset, cached_at = cached_entry
                return self._build_ref_audio_asset_cache_hit_future(
                    cached_asset,
                    submit_ts=submit_ts,
                    cache_age_ms=max(0.0, (now - float(cached_at)) * 1000.0),
                )
            inflight_future = self.ref_audio_asset_inflight.get(ref_audio_path)
            if inflight_future is not None:
                return inflight_future
            future = executor.submit(self._prepare_ref_audio_asset_native, ref_audio_path, submit_at=submit_ts)
            self.ref_audio_asset_inflight[ref_audio_path] = future

        def _finalize(done_future: concurrent.futures.Future) -> None:
            with cache_lock:
                self.ref_audio_asset_inflight.pop(ref_audio_path, None)
                try:
                    profiled = done_future.result()
                except Exception:
                    return
                asset = self._coerce_prepared_ref_audio_asset(profiled.result)
                if isinstance(asset, GPTSoVITSPreparedRefAudioAsset):
                    self.ref_audio_asset_cache[ref_audio_path] = (
                        self._clone_prepared_ref_audio_asset(asset),
                        time.perf_counter(),
                    )
                    self._prune_ref_audio_asset_cache_locked()

        future.add_done_callback(_finalize)
        return future

    async def acquire_prepare_admission(self) -> dict[str, float]:
        return await self.inflight_gate.acquire()

    def mark_prepare_enter(self) -> tuple[int, int]:
        mark_enter = self.mark_enter_fn
        if callable(mark_enter):
            current_inflight, peak_inflight = mark_enter()
            return int(current_inflight), int(peak_inflight)
        if self.lock is None:
            return 0, 0
        with self.lock:
            self.inflight += 1
            current_inflight = int(self.inflight)
            self.peak_inflight = max(int(self.peak_inflight), current_inflight)
            return current_inflight, int(self.peak_inflight)

    def release_split_stage_slot(self) -> None:
        release_slot = self.release_split_stage_slot_fn
        if callable(release_slot):
            release_slot()
            return
        if self.lock is not None:
            with self.lock:
                self.inflight = max(0, int(self.inflight) - 1)
        self.inflight_gate.release()


@dataclass(slots=True)
class GPTSoVITSPreparedAudioPhase:
    request_id: str
    prepared_cpu_stage: GPTSoVITSPreparedCpuStage
    phase_one: GPTSoVITSPrepareAudioPhaseData


@dataclass(slots=True)
class GPTSoVITSPreparedRefSpecPhase:
    request_id: str
    prepared_audio_phase: GPTSoVITSPreparedAudioPhase
    ref_spec_result: GPTSoVITSPrepareRefSpecResult


@dataclass(slots=True)
class GPTSoVITSPreparedTextPhase:
    request_id: str
    prepared_audio_phase: GPTSoVITSPreparedAudioPhase
    phase_two: GPTSoVITSPrepareTextPhaseData


@dataclass(slots=True)
class GPTSoVITST2SRequestState:
    request_id: str
    ref_audio_path: Any
    prompt_text: str
    prompt_lang: str
    text: str
    text_lang: str
    norm_prompt_text: str
    norm_text: str
    phones: torch.LongTensor
    prompt_phones: torch.LongTensor
    all_phones: torch.LongTensor
    all_bert_features: torch.Tensor
    prompt_semantic: torch.LongTensor
    refer_spec: GPTSoVITSReferSpec | None
    aux_refer_specs: list[tuple[torch.Tensor, torch.Tensor | None]]
    raw_audio: torch.Tensor
    raw_sr: int
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    early_stop_num: int
    ready_step: int
    prepare_profile: dict[str, float]


@dataclass(slots=True)
class GPTSoVITSActiveBatch:
    request_ids: list[str]
    states: list[Any]
    x: torch.Tensor | None
    x_lens: torch.LongTensor | None
    y_sequences: list[torch.LongTensor]
    prefix_lens: torch.LongTensor
    xy_pos: torch.Tensor
    key_padding_mask: torch.Tensor | None
    prefill_attn_mask: torch.Tensor | None
    decode_attn_mask: torch.Tensor | None
    k_cache: list[torch.Tensor] | None
    v_cache: list[torch.Tensor] | None
    kv_lens: torch.LongTensor | None
    step_indices: torch.LongTensor
    prefill_done: bool
    kv_cache_pooled: bool = False
    kv_cache_capacity: int = 0
    kv_cache_batch_capacity: int = 0


@dataclass(slots=True)
class GPTSoVITSDecodePreparedRequest:
    request_id: str
    semantic_tokens: torch.Tensor
    phones: torch.Tensor
    prompt_phones: torch.Tensor
    prompt_semantic: torch.Tensor
    refer_audio_spec: torch.Tensor
    refer_audio_16k: torch.Tensor
    raw_audio: torch.Tensor
    raw_sr: int
    speed_factor: float
    sample_steps: int
    super_sampling: bool


@dataclass(slots=True)
class GPTSoVITSDecodedAudio:
    request_id: str
    audio_fragment: Any
    output_sr: int
    speed_factor: float
    super_sampling: bool


@dataclass
class GPTSoVITSARSession:
    request_id: str
    active_batch: Any
    transport_info: GPTSoVITSStageTransport | dict[str, Any]
    current_logits: torch.Tensor


@dataclass(slots=True)
class GPTSoVITSARFinishedItem:
    request_id: str
    semantic_tokens: torch.LongTensor
    finish_idx: int
    finish_reason: str


@dataclass(slots=True)
class GPTSoVITSPrepareProfiledResult:
    result: Any
    submit_at: float
    started_at: float
    finished_at: float
    profile: dict[str, float] | None = None

    @property
    def queue_ms(self) -> float:
        return max(0.0, (self.started_at - self.submit_at) * 1000.0)

    @property
    def run_ms(self) -> float:
        return max(0.0, (self.finished_at - self.started_at) * 1000.0)


class GPTSoVITSRuntime:
    """Thin, process-local wrapper around GPT-SoVITS TTS.run()."""

    def __init__(
        self,
        *,
        project_root: str | None = None,
        config_path: str | None = None,
    ) -> None:
        self.project_root = os.path.abspath(project_root or os.environ.get("GPT_SOVITS_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT))
        configured_path = config_path or os.environ.get("GPT_SOVITS_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
        self.config_path = self._resolve_path(configured_path)
        self._init_lock = threading.RLock()
        self._run_lock = threading.RLock()
        self._cwd_lock = threading.RLock()
        self._pipeline: Any | None = None
        self._config: Any | None = None
        self._prepare_coordinator: Any | None = None
        self._native_runtime_ready = False
        self._runtime_configs: Any | None = None
        self._runtime_precision: torch.dtype | None = None
        self._runtime_is_v2pro = False
        self._runtime_t2s_model: Any | None = None
        self._runtime_vits_model: Any | None = None
        self._runtime_bert_tokenizer: Any | None = None
        self._runtime_bert_model: Any | None = None
        self._runtime_cnhuhbert_model: Any | None = None
        self._runtime_vocoder: Any | None = None
        self._runtime_sv_model: Any | None = None
        self._runtime_sr_model: Any | None = None
        self._runtime_sr_model_not_exist = False
        self._runtime_prepare_bert_batch_worker: Any | None = None
        self._runtime_prepare_ref_semantic_batch_worker: Any | None = None
        self._runtime_prepare_g2pw_batch_worker: Any | None = None
        self._runtime_prepare_text_cpu_worker: Any | None = None
        self._runtime_prepare_text_cpu_executor: Any | None = None
        self._runtime_prepare_bert_stage_limiter: Any | None = None
        self._runtime_prepare_ref_semantic_stage_limiter: Any | None = None
        self._last_t2s_scheduler_stats: dict[str, Any] = {}
        self._sampling_ops_cache: tuple[Any, Any] | None = None

    def _resolve_path(self, maybe_relative_path: str) -> str:
        if os.path.isabs(maybe_relative_path):
            return maybe_relative_path
        return os.path.join(self.project_root, maybe_relative_path)

    def _ensure_import_path(self) -> None:
        project_root = self.project_root
        package_root = os.path.join(project_root, "GPT_SoVITS")
        for candidate in (project_root, package_root):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)

    def _ensure_native_runtime_deps(self) -> None:
        if self._native_runtime_ready:
            return
        preload_native_deps = str(os.environ.get("GPTSOVITS_PRELOAD_NATIVE_RUNTIME_DEPS", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not preload_native_deps:
            self._native_runtime_ready = True
            return

        lib_roots = []
        for root in {
            os.environ.get("CONDA_PREFIX"),
            sys.prefix,
            os.path.dirname(sys.executable),
        }:
            if not root:
                continue
            lib_roots.append(root if root.endswith("/lib") else os.path.join(root, "lib"))

        for lib_root in lib_roots:
            if not os.path.isdir(lib_root):
                continue
            for lib_name in ("libstdc++.so.6", "libgcc_s.so.1"):
                lib_path = os.path.join(lib_root, lib_name)
                if os.path.exists(lib_path):
                    try:
                        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    except OSError as exc:
                        logger.debug("Failed to preload %s: %s", lib_path, exc)

        self._native_runtime_ready = True

    @contextmanager
    def _project_root_cwd(self):
        with self._cwd_lock:
            previous_cwd = os.getcwd()
            if previous_cwd != self.project_root:
                os.chdir(self.project_root)
            try:
                yield
            finally:
                if os.getcwd() != previous_cwd:
                    os.chdir(previous_cwd)

    @staticmethod
    @contextmanager
    def _temporary_env_override(name: str, value: str):
        previous = os.environ.get(name)
        os.environ[name] = value
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    @staticmethod
    @contextmanager
    def _temporary_attr_override(obj: Any, name: str, value: Any):
        sentinel = object()
        previous = getattr(obj, name, sentinel)
        setattr(obj, name, value)
        try:
            yield
        finally:
            if previous is sentinel:
                delattr(obj, name)
            else:
                setattr(obj, name, previous)

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        with self._init_lock:
            if self._pipeline is not None:
                return self._pipeline

            self._ensure_import_path()
            self._ensure_native_runtime_deps()
            with self._project_root_cwd():
                from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

                config = TTS_Config(self.config_path)
                with self._temporary_env_override("GPTSOVITS_RUNTIME_SKIP_PREPARE_COMPONENTS", "1"):
                    with self._temporary_attr_override(TTS, "refresh_runtime_components", lambda _self: None):
                        pipeline = TTS(config)
                setattr(pipeline, "runtime_prepare_components_deferred", True)
            self._install_ref_audio_loader_fallback(pipeline)
            self._install_sv_half_safe_patch(pipeline)
            self._refresh_runtime_t2s_kv_cache_pool_state(pipeline)
            self._install_runtime_prepare_components(pipeline)
            self._bind_pipeline_components(pipeline)
            self._config = config
            self._pipeline = pipeline
            logger.info(
                "Initialized GPT-SoVITS runtime from %s using config %s",
                self.project_root,
                self.config_path,
            )
        return self._pipeline

    @staticmethod
    def _runtime_prepare_batching_enabled(env_name: str, default: str = "1") -> bool:
        return str(os.environ.get(env_name, default)).strip().lower() not in {"0", "false", "no", "off"}

    def _refresh_runtime_t2s_kv_cache_pool_state(self, pipeline: Any) -> None:
        t2s_model = getattr(getattr(pipeline, "t2s_model", None), "model", None)
        if t2s_model is None:
            pipeline.t2s_kv_cache_pool_state = None
            return
        with self._project_root_cwd():
            from TTS_infer_pack.t2s_kv_cache_pool import attach_t2s_kv_cache_pool

        pool_state = attach_t2s_kv_cache_pool(t2s_model, getattr(getattr(pipeline, "configs", None), "device", "cpu"))
        pool = getattr(t2s_model, "kv_cache_pool", None)
        pipeline.t2s_kv_cache_pool_state = pool.snapshot() if pool is not None else dict(vars(pool_state))

    def _build_runtime_prepare_bert_batch_worker(self, pipeline: Any) -> Any | None:
        if not self._runtime_prepare_batching_enabled("GPTSOVITS_PREPARE_BERT_BATCHING", "1"):
            return None
        with self._project_root_cwd():
            from TTS_infer_pack.prepare_bert_batch_worker import PrepareBertBatchWorker, PrepareBertBatchWorkerPool

        bert_worker_kwargs = dict(
            bert_model=getattr(pipeline, "bert_model", None),
            tokenizer=getattr(pipeline, "bert_tokenizer", None),
            device=getattr(getattr(pipeline, "configs", None), "device", "cpu"),
            stage_limiter=getattr(pipeline, "prepare_bert_stage_limiter", None),
            batch_window_ms=int(os.environ.get("GPTSOVITS_PREPARE_BERT_BATCH_WINDOW_MS", "5")),
            max_batch_items=int(os.environ.get("GPTSOVITS_PREPARE_BERT_BATCH_MAX_ITEMS", "16")),
            max_batch_tokens=int(os.environ.get("GPTSOVITS_PREPARE_BERT_BATCH_MAX_TOKENS", "4096")),
            max_pending_tasks=int(os.environ.get("GPTSOVITS_PREPARE_BERT_MAX_PENDING_TASKS", "0")),
            admission_poll_ms=int(os.environ.get("GPTSOVITS_PREPARE_BERT_ADMISSION_POLL_MS", "1")),
            high_pressure_pending_threshold=int(
                os.environ.get("GPTSOVITS_PREPARE_BERT_HIGH_PRESSURE_PENDING_THRESHOLD", "0")
            ),
            high_pressure_batch_window_ms=int(
                os.environ.get("GPTSOVITS_PREPARE_BERT_HIGH_PRESSURE_BATCH_WINDOW_MS", "1")
            ),
            high_pressure_max_batch_items=int(
                os.environ.get("GPTSOVITS_PREPARE_BERT_HIGH_PRESSURE_MAX_ITEMS", "32")
            ),
            high_pressure_max_batch_tokens=int(
                os.environ.get("GPTSOVITS_PREPARE_BERT_HIGH_PRESSURE_MAX_TOKENS", "8192")
            ),
        )
        bert_batch_workers = max(1, int(os.environ.get("GPTSOVITS_PREPARE_BERT_BATCH_WORKERS", "2")))
        if bert_batch_workers > 1:
            return PrepareBertBatchWorkerPool(worker_count=int(bert_batch_workers), **bert_worker_kwargs)
        return PrepareBertBatchWorker(**bert_worker_kwargs)

    def _build_runtime_prepare_ref_semantic_batch_worker(self, pipeline: Any) -> Any | None:
        if not self._runtime_prepare_batching_enabled("GPTSOVITS_PREPARE_REF_BATCHING", "1"):
            return None
        with self._project_root_cwd():
            from TTS_infer_pack.prepare_ref_semantic_batch_worker import PrepareRefSemanticBatchWorkerPool

        ref_max_batch_samples = os.environ.get("GPTSOVITS_PREPARE_REF_BATCH_MAX_SAMPLES")
        if ref_max_batch_samples is None:
            ref_max_batch_samples = os.environ.get("GPTSOVITS_PREPARE_REF_BATCH_MAX_FRAMES", "960000")
        ref_batch_workers = max(1, int(os.environ.get("GPTSOVITS_PREPARE_REF_BATCH_WORKERS", "2")))
        return PrepareRefSemanticBatchWorkerPool(
            ssl_model=getattr(pipeline, "cnhuhbert_model", None),
            vits_model=getattr(pipeline, "vits_model", None),
            device=getattr(getattr(pipeline, "configs", None), "device", "cpu"),
            precision=getattr(pipeline, "precision", torch.float32),
            is_half=bool(getattr(getattr(pipeline, "configs", None), "is_half", False)),
            zero_wav_samples=int(getattr(getattr(pipeline, "configs", None), "sampling_rate", 32000) * 0.3),
            stage_limiter=getattr(pipeline, "prepare_ref_semantic_stage_limiter", None),
            batch_window_ms=int(os.environ.get("GPTSOVITS_PREPARE_REF_BATCH_WINDOW_MS", "5")),
            max_batch_items=int(os.environ.get("GPTSOVITS_PREPARE_REF_BATCH_MAX_ITEMS", "8")),
            max_batch_samples=int(ref_max_batch_samples),
            worker_count=int(ref_batch_workers),
        )

    def _build_runtime_text_preprocessor(self, pipeline: Any, bert_batch_worker: Any | None) -> Any:
        with self._project_root_cwd():
            from TTS_infer_pack.TextPreprocessor import TextPreprocessor

        return TextPreprocessor(
            getattr(pipeline, "bert_model", None),
            getattr(pipeline, "bert_tokenizer", None),
            getattr(getattr(pipeline, "configs", None), "device", "cpu"),
            version=str(getattr(getattr(pipeline, "configs", None), "version", "") or ""),
            bert_stage_limiter=getattr(pipeline, "prepare_bert_stage_limiter", None),
            bert_batch_worker=bert_batch_worker,
        )

    def _build_runtime_prepare_g2pw_batch_worker(self, pipeline: Any) -> Any | None:
        if not self._runtime_prepare_batching_enabled("GPTSOVITS_PREPARE_G2PW_BATCHING", "0"):
            return None
        with self._project_root_cwd():
            from TTS_infer_pack.prepare_g2pw_batch_worker import PrepareG2PWBatchWorker, PrepareG2PWBatchWorkerPool

        text_preprocessor = getattr(pipeline, "text_preprocessor", None)
        if text_preprocessor is None:
            return None
        g2pw_worker_kwargs = dict(
            resolve_batch_fn=text_preprocessor.resolve_g2pw_segments_batch,
            batch_window_ms=int(os.environ.get("GPTSOVITS_PREPARE_G2PW_BATCH_WINDOW_MS", "2")),
            max_batch_tasks=int(os.environ.get("GPTSOVITS_PREPARE_G2PW_BATCH_MAX_TASKS", "64")),
            max_batch_groups=int(os.environ.get("GPTSOVITS_PREPARE_G2PW_BATCH_MAX_GROUPS", "128")),
            max_batch_chars=int(os.environ.get("GPTSOVITS_PREPARE_G2PW_BATCH_MAX_CHARS", "4096")),
            max_pending_tasks=int(os.environ.get("GPTSOVITS_PREPARE_G2PW_MAX_PENDING_TASKS", "0")),
            admission_poll_ms=int(os.environ.get("GPTSOVITS_PREPARE_G2PW_ADMISSION_POLL_MS", "1")),
            high_pressure_pending_threshold=int(
                os.environ.get("GPTSOVITS_PREPARE_G2PW_HIGH_PRESSURE_PENDING_THRESHOLD", "32")
            ),
            high_pressure_batch_window_ms=int(
                os.environ.get("GPTSOVITS_PREPARE_G2PW_HIGH_PRESSURE_BATCH_WINDOW_MS", "4")
            ),
            high_pressure_max_batch_tasks=int(
                os.environ.get("GPTSOVITS_PREPARE_G2PW_HIGH_PRESSURE_MAX_TASKS", "128")
            ),
            high_pressure_max_batch_groups=int(
                os.environ.get("GPTSOVITS_PREPARE_G2PW_HIGH_PRESSURE_MAX_GROUPS", "256")
            ),
            high_pressure_max_batch_chars=int(
                os.environ.get("GPTSOVITS_PREPARE_G2PW_HIGH_PRESSURE_MAX_CHARS", "8192")
            ),
        )
        g2pw_batch_workers = max(1, int(os.environ.get("GPTSOVITS_PREPARE_G2PW_BATCH_WORKERS", "2")))
        if g2pw_batch_workers > 1:
            return PrepareG2PWBatchWorkerPool(worker_count=int(g2pw_batch_workers), **g2pw_worker_kwargs)
        return PrepareG2PWBatchWorker(**g2pw_worker_kwargs)

    def _build_runtime_prepare_text_cpu_worker(self, pipeline: Any) -> Any | None:
        if int(getattr(pipeline, "prepare_text_cpu_workers", 0) or 0) <= 0:
            return None
        text_preprocessor = getattr(pipeline, "text_preprocessor", None)
        if text_preprocessor is None:
            return None
        with self._project_root_cwd():
            from TTS_infer_pack.prepare_text_cpu_worker import PrepareTextCpuWorker

        return PrepareTextCpuWorker(
            process_fn=lambda text, language, text_split_method: text_preprocessor.preprocess_text_segments(
                text,
                language,
                getattr(getattr(pipeline, "configs", None), "version", ""),
                text_split_method=text_split_method,
            ),
            batch_process_fn=lambda items: text_preprocessor.preprocess_text_segments_batch(
                list(items),
                getattr(getattr(pipeline, "configs", None), "version", ""),
            ),
            worker_count=int(getattr(pipeline, "prepare_text_cpu_workers", 0)),
            batch_window_ms=int(os.environ.get("GPTSOVITS_PREPARE_TEXT_CPU_BATCH_WINDOW_MS", "2")),
            max_batch_items=int(os.environ.get("GPTSOVITS_PREPARE_TEXT_CPU_BATCH_MAX_ITEMS", "512")),
            high_pressure_pending_threshold=int(
                os.environ.get("GPTSOVITS_PREPARE_TEXT_CPU_HIGH_PRESSURE_PENDING_THRESHOLD", "64")
            ),
            high_pressure_batch_window_ms=int(
                os.environ.get("GPTSOVITS_PREPARE_TEXT_CPU_HIGH_PRESSURE_BATCH_WINDOW_MS", "4")
            ),
            high_pressure_max_batch_items=int(
                os.environ.get("GPTSOVITS_PREPARE_TEXT_CPU_HIGH_PRESSURE_MAX_ITEMS", "512")
            ),
            max_pending_tasks=int(os.environ.get("GPTSOVITS_PREPARE_TEXT_CPU_MAX_PENDING_TASKS", "0")),
            admission_poll_ms=int(os.environ.get("GPTSOVITS_PREPARE_TEXT_CPU_ADMISSION_POLL_MS", "1")),
            admission_controller=getattr(pipeline, "_build_text_cpu_admission_state"),
        )

    def _install_runtime_prepare_components(self, pipeline: Any) -> None:
        runtime_prepare_state_provider = lambda: _build_native_prepare_runtime_state(pipeline)
        runtime_prepare_refresh = lambda: self._refresh_runtime_prepare_components(pipeline)
        runtime_prepare_generation = int(getattr(pipeline, "_vllm_runtime_prepare_generation", 0) or 0) + 1
        bert_batch_worker = self._build_runtime_prepare_bert_batch_worker(pipeline)
        ref_semantic_batch_worker = self._build_runtime_prepare_ref_semantic_batch_worker(pipeline)
        pipeline.prepare_bert_batch_worker = bert_batch_worker
        pipeline.prepare_ref_semantic_batch_worker = ref_semantic_batch_worker
        pipeline.text_preprocessor = self._build_runtime_text_preprocessor(pipeline, bert_batch_worker)
        pipeline.prepare_g2pw_batch_worker = self._build_runtime_prepare_g2pw_batch_worker(pipeline)
        pipeline.prepare_text_cpu_worker = self._build_runtime_prepare_text_cpu_worker(pipeline)
        pipeline._vllm_runtime_prepare_generation = runtime_prepare_generation
        pipeline._vllm_runtime_owner = self
        pipeline._vllm_runtime_prepare_state_provider = runtime_prepare_state_provider
        pipeline._vllm_runtime_prepare_coordinator_factory = lambda: GPTSoVITSPrepareRuntimeCoordinator.build_native(
            pipeline
        )
        pipeline._vllm_runtime_refresh_prepare_components = runtime_prepare_refresh
        pipeline.refresh_runtime_components = runtime_prepare_refresh
        pipeline.snapshot_prepare_runtime_components = runtime_prepare_state_provider
        pipeline._vllm_runtime_owned_prepare_components = True
        if hasattr(pipeline, "_prewarm_g2pw_runtime"):
            pipeline._prewarm_g2pw_runtime()
        if hasattr(pipeline, "_prewarm_prepare_ref_runtime"):
            pipeline._prewarm_prepare_ref_runtime()

    def _refresh_runtime_prepare_components(self, pipeline: Any) -> None:
        with self._init_lock:
            self._refresh_runtime_t2s_kv_cache_pool_state(pipeline)
            self._install_runtime_prepare_components(pipeline)
            setattr(pipeline, "_scheduler_prepare_coordinator", None)
            self._bind_pipeline_components(pipeline)
            self._prepare_coordinator = None

    def _bind_pipeline_components(self, pipeline: Any) -> None:
        self._runtime_configs = getattr(pipeline, "configs", None)
        self._runtime_precision = getattr(pipeline, "precision", None)
        self._runtime_is_v2pro = bool(getattr(pipeline, "is_v2pro", False))
        runtime_t2s = getattr(pipeline, "t2s_model", None)
        self._runtime_t2s_model = getattr(runtime_t2s, "model", None) if runtime_t2s is not None else None
        self._runtime_vits_model = getattr(pipeline, "vits_model", None)
        self._runtime_bert_tokenizer = getattr(pipeline, "bert_tokenizer", None)
        self._runtime_bert_model = getattr(pipeline, "bert_model", None)
        self._runtime_cnhuhbert_model = getattr(pipeline, "cnhuhbert_model", None)
        self._runtime_vocoder = getattr(pipeline, "vocoder", None)
        self._runtime_sv_model = getattr(pipeline, "sv_model", None)
        self._runtime_sr_model = getattr(pipeline, "sr_model", None)
        self._runtime_sr_model_not_exist = bool(getattr(pipeline, "sr_model_not_exist", False))
        self._runtime_prepare_bert_batch_worker = getattr(pipeline, "prepare_bert_batch_worker", None)
        self._runtime_prepare_ref_semantic_batch_worker = getattr(pipeline, "prepare_ref_semantic_batch_worker", None)
        self._runtime_prepare_g2pw_batch_worker = getattr(pipeline, "prepare_g2pw_batch_worker", None)
        self._runtime_prepare_text_cpu_worker = getattr(pipeline, "prepare_text_cpu_worker", None)
        self._runtime_prepare_text_cpu_executor = getattr(pipeline, "prepare_text_cpu_executor", None)
        self._runtime_prepare_bert_stage_limiter = getattr(pipeline, "prepare_bert_stage_limiter", None)
        self._runtime_prepare_ref_semantic_stage_limiter = getattr(pipeline, "prepare_ref_semantic_stage_limiter", None)

    def _refresh_runtime_bound_components(self) -> None:
        pipeline = self._ensure_pipeline()
        self._bind_pipeline_components(pipeline)

    def _get_runtime_configs(self) -> Any:
        if self._runtime_configs is None:
            self._refresh_runtime_bound_components()
        return self._runtime_configs

    def _get_runtime_precision(self) -> torch.dtype:
        if self._runtime_precision is None:
            self._refresh_runtime_bound_components()
        precision = self._runtime_precision
        if precision is None:
            raise RuntimeError("GPT-SoVITS runtime precision 未初始化")
        return precision

    def _is_runtime_v2pro(self) -> bool:
        if self._runtime_vits_model is None and self._pipeline is None:
            self._refresh_runtime_bound_components()
        return bool(self._runtime_is_v2pro)

    def _get_runtime_t2s_model(self) -> Any:
        if self._runtime_t2s_model is None:
            self._refresh_runtime_bound_components()
        model = self._runtime_t2s_model
        if model is None:
            raise RuntimeError("GPT-SoVITS T2S model 未初始化")
        return model

    def _get_runtime_vits_model(self) -> Any:
        if self._runtime_vits_model is None:
            self._refresh_runtime_bound_components()
        model = self._runtime_vits_model
        if model is None:
            raise RuntimeError("GPT-SoVITS VITS model 未初始化")
        return model

    def _get_runtime_bert_tokenizer(self) -> Any:
        if self._runtime_bert_tokenizer is None:
            self._refresh_runtime_bound_components()
        tokenizer = self._runtime_bert_tokenizer
        if tokenizer is None:
            raise RuntimeError("GPT-SoVITS BERT tokenizer 未初始化")
        return tokenizer

    def _get_runtime_bert_model(self) -> Any:
        if self._runtime_bert_model is None:
            self._refresh_runtime_bound_components()
        model = self._runtime_bert_model
        if model is None:
            raise RuntimeError("GPT-SoVITS BERT model 未初始化")
        return model

    def _get_runtime_cnhuhbert_model(self) -> Any:
        if self._runtime_cnhuhbert_model is None:
            self._refresh_runtime_bound_components()
        model = self._runtime_cnhuhbert_model
        if model is None:
            raise RuntimeError("GPT-SoVITS CNHubert model 未初始化")
        return model

    def _get_runtime_vocoder(self) -> Any:
        if self._runtime_vocoder is None:
            self._refresh_runtime_bound_components()
        vocoder = self._runtime_vocoder
        if vocoder is None:
            raise RuntimeError("GPT-SoVITS vocoder 未初始化")
        return vocoder

    def _get_runtime_sv_model(self) -> Any:
        if self._runtime_sv_model is None:
            self._refresh_runtime_bound_components()
        model = self._runtime_sv_model
        if model is None:
            raise RuntimeError("GPT-SoVITS SV model 未初始化")
        return model

    def _get_runtime_sr_model(self) -> tuple[Any | None, bool]:
        if self._pipeline is None and self._runtime_sr_model is None:
            self._refresh_runtime_bound_components()
        return self._runtime_sr_model, bool(self._runtime_sr_model_not_exist)

    def _ensure_runtime_sr_model(self) -> tuple[Any | None, bool]:
        sr_model, sr_model_not_exist = self._get_runtime_sr_model()
        if sr_model is not None or sr_model_not_exist:
            return sr_model, sr_model_not_exist
        with self._project_root_cwd():
            from GPT_SoVITS.TTS_infer_pack.TTS import AP_BWE, DictToAttrRecursive

        configs = self._get_runtime_configs()
        try:
            sr_model = AP_BWE(configs.device, DictToAttrRecursive)
            sr_model_not_exist = False
        except FileNotFoundError:
            sr_model = None
            sr_model_not_exist = True
        self._runtime_sr_model = sr_model
        self._runtime_sr_model_not_exist = bool(sr_model_not_exist)
        if self._pipeline is not None:
            setattr(self._pipeline, "sr_model", sr_model)
            setattr(self._pipeline, "sr_model_not_exist", bool(sr_model_not_exist))
        return sr_model, bool(sr_model_not_exist)

    def _get_runtime_prepare_bert_batch_worker(self) -> Any | None:
        worker = self._runtime_prepare_bert_batch_worker
        if worker is None and self._pipeline is not None:
            worker = getattr(self._pipeline, "prepare_bert_batch_worker", None)
            self._runtime_prepare_bert_batch_worker = worker
        return worker

    def _get_runtime_prepare_ref_semantic_batch_worker(self) -> Any | None:
        worker = self._runtime_prepare_ref_semantic_batch_worker
        if worker is None and self._pipeline is not None:
            worker = getattr(self._pipeline, "prepare_ref_semantic_batch_worker", None)
            self._runtime_prepare_ref_semantic_batch_worker = worker
        return worker

    def _get_runtime_prepare_g2pw_batch_worker(self) -> Any | None:
        worker = self._runtime_prepare_g2pw_batch_worker
        if worker is None and self._pipeline is not None:
            worker = getattr(self._pipeline, "prepare_g2pw_batch_worker", None)
            self._runtime_prepare_g2pw_batch_worker = worker
        return worker

    def _get_runtime_prepare_text_cpu_worker(self) -> Any | None:
        worker = self._runtime_prepare_text_cpu_worker
        if worker is None and self._pipeline is not None:
            worker = getattr(self._pipeline, "prepare_text_cpu_worker", None)
            self._runtime_prepare_text_cpu_worker = worker
        return worker

    def _get_runtime_prepare_text_cpu_executor(self) -> Any | None:
        executor = self._runtime_prepare_text_cpu_executor
        if executor is None and self._pipeline is not None:
            executor = getattr(self._pipeline, "prepare_text_cpu_executor", None)
            self._runtime_prepare_text_cpu_executor = executor
        return executor

    def _get_runtime_prepare_bert_stage_limiter(self) -> Any | None:
        limiter = self._runtime_prepare_bert_stage_limiter
        if limiter is None and self._pipeline is not None:
            limiter = getattr(self._pipeline, "prepare_bert_stage_limiter", None)
            self._runtime_prepare_bert_stage_limiter = limiter
        return limiter

    def _get_runtime_prepare_ref_semantic_stage_limiter(self) -> Any | None:
        limiter = self._runtime_prepare_ref_semantic_stage_limiter
        if limiter is None and self._pipeline is not None:
            limiter = getattr(self._pipeline, "prepare_ref_semantic_stage_limiter", None)
            self._runtime_prepare_ref_semantic_stage_limiter = limiter
        return limiter

    def get_t2s_model(self) -> Any:
        return self._get_runtime_t2s_model()

    def get_last_t2s_scheduler_stats(self) -> dict[str, Any]:
        return dict(self._last_t2s_scheduler_stats)

    def get_semantic_eos_id(self) -> int:
        model = self.get_t2s_model()
        return int(model.EOS)

    def get_semantic_vocab_size(self) -> int:
        model = self.get_t2s_model()
        return int(model.vocab_size)

    @staticmethod
    def _coerce_prepare_coordinator(coordinator: Any) -> GPTSoVITSPrepareRuntimeCoordinator:
        return GPTSoVITSPrepareRuntimeCoordinator.from_runtime_coordinator(coordinator)

    def _ensure_prepare_coordinator(self) -> GPTSoVITSPrepareRuntimeCoordinator:
        if self._prepare_coordinator is not None:
            return self._prepare_coordinator
        with self._init_lock:
            if self._prepare_coordinator is not None:
                return self._prepare_coordinator
            pipeline = self._ensure_pipeline()
            self._prepare_coordinator = GPTSoVITSPrepareRuntimeCoordinator.build_native(pipeline)
        return self._prepare_coordinator

    @staticmethod
    def _ensure_audio_position_encoding(
        model: Any,
        max_position: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        required_len = max_position + 1
        if model.ar_audio_position.pe is not None and model.ar_audio_position.pe.size(1) >= required_len:
            if model.ar_audio_position.pe.dtype != dtype or model.ar_audio_position.pe.device != device:
                model.ar_audio_position.pe = model.ar_audio_position.pe.to(dtype=dtype, device=device)
            return
        model.ar_audio_position.extend_pe(
            torch.zeros(1, required_len, model.ar_audio_position.embedding_dim, device=device, dtype=dtype)
        )

    def _build_prefill_active_batch(
        self,
        model: Any,
        states: Sequence[Any],
    ) -> Any:
        if not states:
            raise ValueError("GPT-SoVITS AR prefill requires at least one state")

        x_items: list[torch.Tensor] = []
        y_pos_items: list[torch.Tensor] = []
        x_lens: list[int] = []
        prefix_lens: list[int] = []
        y_sequences: list[torch.Tensor] = []

        for state in states:
            text_emb = model.ar_text_embedding(state.all_phones.unsqueeze(0))
            bert_proj = model.bert_proj(state.all_bert_features.transpose(0, 1).unsqueeze(0))
            x_pos = model.ar_text_position(text_emb + bert_proj).squeeze(0)
            y_emb = model.ar_audio_embedding(state.prompt_semantic.unsqueeze(0))
            y_pos = model.ar_audio_position(y_emb).squeeze(0)
            x_items.append(x_pos)
            y_pos_items.append(y_pos)
            x_lens.append(int(x_pos.shape[0]))
            prefix_lens.append(int(y_pos.shape[0]))
            y_sequences.append(state.prompt_semantic.clone())

        max_x_len = max(x_lens)
        max_prefix_len = max(prefix_lens)
        x_batch = torch.stack([_left_pad_hidden(item, max_x_len) for item in x_items], dim=0)
        y_pos_batch = torch.stack([_left_pad_hidden(item, max_prefix_len) for item in y_pos_items], dim=0)
        xy_pos = torch.cat([x_batch, y_pos_batch], dim=1)

        device = x_batch.device
        x_lens_tensor = torch.tensor(x_lens, dtype=torch.long, device=device)
        prefix_lens_tensor = torch.tensor(prefix_lens, dtype=torch.long, device=device)
        x_padding_mask = _make_pad_mask_left(x_lens_tensor, max_x_len)
        y_padding_mask = _make_pad_mask_left(prefix_lens_tensor, max_prefix_len)
        key_padding_mask = torch.cat([x_padding_mask, y_padding_mask], dim=1).bool()
        x_mask = F.pad(torch.zeros(max_x_len, max_x_len, dtype=torch.bool, device=device), (0, max_prefix_len), value=True)
        y_mask = F.pad(
            torch.triu(torch.ones(max_prefix_len, max_prefix_len, dtype=torch.bool, device=device), diagonal=1),
            (max_x_len, 0),
            value=False,
        )
        causal_mask = torch.cat([x_mask, y_mask], dim=0).unsqueeze(0)
        attn_mask = causal_mask.logical_or(key_padding_mask.unsqueeze(1)).unsqueeze(1)

        return GPTSoVITSActiveBatch(
            request_ids=[str(state.request_id) for state in states],
            states=list(states),
            x=x_batch,
            x_lens=x_lens_tensor,
            y_sequences=y_sequences,
            prefix_lens=prefix_lens_tensor,
            xy_pos=xy_pos,
            key_padding_mask=key_padding_mask,
            prefill_attn_mask=attn_mask,
            decode_attn_mask=None,
            k_cache=None,
            v_cache=None,
            kv_lens=None,
            step_indices=torch.zeros((len(states),), dtype=torch.long, device=device),
            prefill_done=False,
            kv_cache_pooled=False,
            kv_cache_capacity=0,
            kv_cache_batch_capacity=0,
        )

    def _build_next_xy_pos(
        self,
        model: Any,
        y_sequences: Sequence[torch.LongTensor],
    ) -> torch.Tensor:
        if not y_sequences:
            raise ValueError("GPT-SoVITS AR decode requires at least one semantic history")
        last_tokens = torch.stack([seq[-1:] for seq in y_sequences], dim=0)
        y_emb = model.ar_audio_embedding(last_tokens)
        position_ids = torch.tensor([int(seq.shape[0] - 1) for seq in y_sequences], dtype=torch.long, device=y_emb.device)
        self._ensure_audio_position_encoding(model, int(position_ids.max().item()), y_emb.dtype, y_emb.device)
        pos_emb = model.ar_audio_position.pe[0].index_select(0, position_ids).unsqueeze(1)
        return y_emb * model.ar_audio_position.x_scale + model.ar_audio_position.alpha * pos_emb.to(
            dtype=y_emb.dtype,
            device=y_emb.device,
        )

    @staticmethod
    def _get_kv_pool(model: Any) -> Any | None:
        pool = getattr(model, "kv_cache_pool", None)
        if pool is None:
            return None
        if not getattr(getattr(pool, "state", None), "enabled", False):
            return None
        return pool

    def _set_kv_pool_active_rows(self, model: Any, active_rows: int) -> None:
        pool = self._get_kv_pool(model)
        if pool is None:
            return
        try:
            pool.set_active_rows(active_rows)
        except Exception:
            pass

    def _pack_active_batch_into_pool(self, model: Any, active_batch: Any) -> bool:
        pool = self._get_kv_pool(model)
        if (
            pool is None
            or active_batch.k_cache is None
            or active_batch.v_cache is None
            or active_batch.kv_lens is None
            or active_batch.kv_lens.numel() <= 0
        ):
            return False
        batch_size = int(active_batch.kv_lens.shape[0])
        max_kv_len = int(active_batch.kv_lens.max().item())
        if max_kv_len + 1 > int(pool.max_seq_len):
            try:
                pool.record_fallback(f"pack_decode_headroom_overflow(batch={batch_size},seq={max_kv_len},next={max_kv_len + 1})")
            except Exception:
                pass
            active_batch.kv_cache_pooled = False
            active_batch.kv_cache_capacity = 0
            active_batch.kv_cache_batch_capacity = 0
            self._set_kv_pool_active_rows(model, 0)
            return False
        pooled_views = pool.pack_dynamic_cache_layers(
            k_layers=active_batch.k_cache,
            v_layers=active_batch.v_cache,
            kv_lens=active_batch.kv_lens,
        )
        if pooled_views is None:
            active_batch.kv_cache_pooled = False
            active_batch.kv_cache_capacity = 0
            active_batch.kv_cache_batch_capacity = 0
            self._set_kv_pool_active_rows(model, 0)
            return False
        active_batch.k_cache, active_batch.v_cache = pooled_views
        active_batch.decode_attn_mask = None
        active_batch.kv_cache_pooled = True
        active_batch.kv_cache_capacity = int(pool.max_seq_len)
        active_batch.kv_cache_batch_capacity = int(pool.max_batch_size)
        self._set_kv_pool_active_rows(model, len(active_batch.request_ids))
        return True

    @staticmethod
    def _build_decode_mask_from_kv_lens(
        kv_lens: torch.LongTensor,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        if kv_lens.numel() <= 0:
            return None
        target_len = int(kv_lens.max().item()) + 1
        mask = torch.ones((int(kv_lens.shape[0]), 1, 1, target_len), dtype=torch.bool, device=device)
        for batch_index, kv_len in enumerate(kv_lens.tolist()):
            current_len = kv_len + 1
            mask[batch_index, :, :, -current_len:] = False
        if not mask.any().item():
            return None
        return mask

    @staticmethod
    def _compact_cache_to_kv_lens(cache: torch.Tensor, kv_lens: torch.LongTensor) -> torch.Tensor:
        target_len = int(kv_lens.max().item())
        if cache.shape[1] == target_len and bool(torch.all(kv_lens == target_len).item()):
            return cache
        compacted = cache.new_zeros((cache.shape[0], target_len, cache.shape[2]))
        for batch_index, kv_len in enumerate(kv_lens.tolist()):
            if kv_len <= 0:
                continue
            compacted[batch_index, -kv_len:, :] = cache[batch_index, -kv_len:, :]
        return compacted

    @staticmethod
    def _compact_decode_mask_to_kv_lens(
        decode_attn_mask: torch.Tensor | None,
        kv_lens: torch.LongTensor,
    ) -> torch.Tensor | None:
        target_len = int(kv_lens.max().item()) + 1
        if decode_attn_mask is None:
            return None
        if decode_attn_mask.shape[-1] == target_len and bool(torch.all(kv_lens + 1 == target_len).item()):
            return decode_attn_mask
        compacted = torch.ones(
            (decode_attn_mask.shape[0], 1, 1, target_len),
            dtype=decode_attn_mask.dtype,
            device=decode_attn_mask.device,
        )
        for batch_index, kv_len in enumerate(kv_lens.tolist()):
            current_len = kv_len + 1
            compacted[batch_index, :, :, -current_len:] = decode_attn_mask[batch_index, :, :, -current_len:]
        if not compacted.any().item():
            return None
        return compacted

    @staticmethod
    def _pad_decode_mask_left(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        pad_len = target_len - mask.shape[-1]
        if pad_len <= 0:
            return mask
        return F.pad(mask, (pad_len, 0), value=True)

    def _fit_decode_mask_length(self, mask: torch.Tensor, target_len: int) -> torch.Tensor:
        if mask.shape[-1] > target_len:
            return mask[:, :, :, -target_len:]
        if mask.shape[-1] < target_len:
            return self._pad_decode_mask_left(mask, target_len)
        return mask

    def _materialize_decode_mask_for_active_batch(
        self,
        active_batch: Any,
        target_mask_len: int | None = None,
    ) -> torch.Tensor:
        if active_batch.k_cache is None or active_batch.kv_lens is None:
            raise ValueError("GPT-SoVITS active batch is missing KV cache or kv_lens")
        current_mask_len = active_batch.k_cache[0].shape[1] + 1
        if target_mask_len is None:
            target_mask_len = current_mask_len
        if active_batch.decode_attn_mask is None:
            mask = torch.zeros(
                (len(active_batch.request_ids), 1, 1, current_mask_len),
                dtype=torch.bool,
                device=active_batch.k_cache[0].device,
            )
        else:
            rows: list[torch.Tensor] = []
            for batch_index, kv_len in enumerate(active_batch.kv_lens.tolist()):
                row_len = kv_len + 1
                row_mask = self._fit_decode_mask_length(
                    active_batch.decode_attn_mask[batch_index : batch_index + 1],
                    row_len,
                )
                rows.append(self._pad_decode_mask_left(row_mask, target_mask_len))
            mask = torch.cat(rows, dim=0)
        if target_mask_len != current_mask_len and active_batch.decode_attn_mask is None:
            mask = self._pad_decode_mask_left(mask, target_mask_len)
        return mask

    def _advance_decode_mask(
        self,
        decode_attn_mask: torch.Tensor | None,
        kv_lens: torch.LongTensor,
    ) -> torch.Tensor | None:
        if decode_attn_mask is None:
            return None
        target_len = int(kv_lens.max().item()) + 2
        advanced = torch.zeros(
            (decode_attn_mask.shape[0], 1, 1, target_len),
            dtype=decode_attn_mask.dtype,
            device=decode_attn_mask.device,
        )
        for batch_index, kv_len in enumerate(kv_lens.tolist()):
            current_len = kv_len + 1
            next_mask = F.pad(decode_attn_mask[batch_index : batch_index + 1, :, :, -current_len:], (0, 1), value=False)
            advanced[batch_index : batch_index + 1, :, :, -next_mask.shape[-1] :] = next_mask
        if not advanced.any().item():
            return None
        return advanced

    @staticmethod
    def _pad_cache_left(cache: torch.Tensor, target_len: int) -> torch.Tensor:
        pad_len = target_len - cache.shape[1]
        if pad_len <= 0:
            return cache
        return F.pad(cache, (0, 0, pad_len, 0), value=0)

    @staticmethod
    def _extract_cache_row(
        cache: torch.Tensor,
        batch_index: int,
        kv_len: int,
        *,
        pooled: bool,
    ) -> torch.Tensor:
        if kv_len <= 0:
            return cache[batch_index : batch_index + 1, :0, :]
        if pooled:
            return cache[batch_index : batch_index + 1, :kv_len, :]
        return cache[batch_index : batch_index + 1, -kv_len:, :]

    def _compact_pooled_active_batch(
        self,
        model: Any,
        active_batch: Any,
        keep_indices: Sequence[int],
    ) -> bool:
        pool = self._get_kv_pool(model)
        if pool is None or active_batch.kv_lens is None:
            return False
        pooled_views = pool.compact_rows(keep_indices=keep_indices, kv_lens=active_batch.kv_lens)
        if pooled_views is None:
            active_batch.kv_cache_pooled = False
            active_batch.kv_cache_capacity = 0
            active_batch.kv_cache_batch_capacity = 0
            self._set_kv_pool_active_rows(model, 0)
            return False
        active_batch.k_cache, active_batch.v_cache = pooled_views
        active_batch.decode_attn_mask = None
        active_batch.kv_cache_pooled = True
        active_batch.kv_cache_capacity = int(pool.max_seq_len)
        active_batch.kv_cache_batch_capacity = int(pool.max_batch_size)
        self._set_kv_pool_active_rows(model, len(active_batch.request_ids))
        return True

    def _fallback_pooled_active_batch_to_dynamic_cache(
        self,
        model: Any,
        active_batch: Any,
        *,
        reason: str,
    ) -> None:
        if active_batch.k_cache is None or active_batch.v_cache is None or active_batch.kv_lens is None:
            raise ValueError("GPT-SoVITS pooled KV fallback requires KV cache and kv_lens")
        pool = self._get_kv_pool(model)
        if pool is not None:
            try:
                pool.record_fallback(reason)
            except Exception:
                pass
        target_len = int(active_batch.kv_lens.max().item())
        kv_lens_list = [int(item) for item in active_batch.kv_lens.tolist()]
        unpacked_k_cache: list[torch.Tensor] = []
        unpacked_v_cache: list[torch.Tensor] = []
        for pooled_k, pooled_v in zip(active_batch.k_cache, active_batch.v_cache):
            unpacked_k = pooled_k.new_zeros((pooled_k.shape[0], target_len, pooled_k.shape[2]))
            unpacked_v = pooled_v.new_zeros((pooled_v.shape[0], target_len, pooled_v.shape[2]))
            for batch_index, kv_len in enumerate(kv_lens_list):
                if kv_len <= 0:
                    continue
                unpacked_k[batch_index, -kv_len:, :] = pooled_k[batch_index, :kv_len, :]
                unpacked_v[batch_index, -kv_len:, :] = pooled_v[batch_index, :kv_len, :]
            unpacked_k_cache.append(unpacked_k)
            unpacked_v_cache.append(unpacked_v)
        active_batch.k_cache = unpacked_k_cache
        active_batch.v_cache = unpacked_v_cache
        active_batch.decode_attn_mask = self._build_decode_mask_from_kv_lens(
            active_batch.kv_lens,
            device=active_batch.k_cache[0].device,
        )
        active_batch.kv_cache_pooled = False
        active_batch.kv_cache_capacity = 0
        active_batch.kv_cache_batch_capacity = 0
        self._set_kv_pool_active_rows(model, 0)

    def _get_sampling_ops(self) -> tuple[Any, Any]:
        if self._sampling_ops_cache is not None:
            return self._sampling_ops_cache
        self._ensure_import_path()
        self._ensure_native_runtime_deps()
        with self._project_root_cwd():
            from AR.models.utils import logits_to_probs, multinomial_sample_one_no_sync

        self._sampling_ops_cache = (logits_to_probs, multinomial_sample_one_no_sync)
        return self._sampling_ops_cache

    @staticmethod
    def _accumulate_t2s_stat(stats: dict[str, Any] | None, key: str, elapsed_ms: float) -> None:
        if stats is None:
            return
        stats[key] = float(stats.get(key, 0.0)) + float(elapsed_ms)

    @staticmethod
    def _increment_t2s_stat(stats: dict[str, Any] | None, key: str, delta: int = 1) -> None:
        if stats is None:
            return
        stats[key] = int(stats.get(key, 0)) + int(delta)

    @staticmethod
    def _time_t2s_call(
        fn: Callable[[], Any],
        *,
        pending_cuda_timings: list[tuple[str, torch.cuda.Event | None, torch.cuda.Event | float]] | None,
        stat_key: str,
        device: torch.device | None,
    ) -> Any:
        if pending_cuda_timings is not None and device is not None and device.type == "cuda" and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = fn()
            end_event.record()
            pending_cuda_timings.append((stat_key, start_event, end_event))
            return result

        started = time.perf_counter()
        result = fn()
        if pending_cuda_timings is not None:
            pending_cuda_timings.append((stat_key, None, (time.perf_counter() - started) * 1000.0))
        return result

    @staticmethod
    def _flush_t2s_timing_records(
        stats: dict[str, Any] | None,
        pending_cuda_timings: list[tuple[str, torch.cuda.Event | None, torch.cuda.Event | float]],
        *,
        device: torch.device | None,
    ) -> None:
        if stats is None or not pending_cuda_timings:
            return
        if device is not None and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.current_stream(device).synchronize()
        for stat_key, start_event, end_event_or_elapsed in pending_cuda_timings:
            if start_event is None:
                elapsed_ms = float(end_event_or_elapsed)
            else:
                elapsed_ms = float(start_event.elapsed_time(cast(torch.cuda.Event, end_event_or_elapsed)))
            stats[stat_key] = float(stats.get(stat_key, 0.0)) + elapsed_ms
        pending_cuda_timings.clear()

    @staticmethod
    def _merge_numeric_t2s_stats(stats: dict[str, Any] | None, delta: dict[str, Any] | None) -> None:
        if stats is None or not delta:
            return
        for key, value in delta.items():
            if isinstance(value, bool):
                stats[key] = bool(value)
            elif isinstance(value, int):
                stats[key] = int(stats.get(key, 0)) + int(value)
            elif isinstance(value, float):
                stats[key] = float(stats.get(key, 0.0)) + float(value)
            else:
                stats[key] = value

    @staticmethod
    def _stack_token_sequences_if_same_length(
        token_sequences: Sequence[torch.LongTensor],
    ) -> torch.LongTensor | None:
        if not token_sequences:
            raise ValueError("token_sequences 不能为空")
        target_len = int(token_sequences[0].shape[0])
        for sequence in token_sequences[1:]:
            if int(sequence.shape[0]) != target_len:
                return None
        return torch.stack(list(token_sequences), dim=0)

    @staticmethod
    def _sampling_group_key(
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        trim_eos: bool,
    ) -> tuple[int, float, float, float, bool]:
        return (
            int(top_k),
            float(top_p),
            float(temperature),
            float(repetition_penalty),
            bool(trim_eos),
        )

    @staticmethod
    def _iter_contiguous_sampling_groups(
        sampling_keys: Sequence[tuple[int, float, float, float, bool]],
    ) -> list[tuple[tuple[int, float, float, float, bool], list[int]]]:
        groups: list[tuple[tuple[int, float, float, float, bool], list[int]]] = []
        if not sampling_keys:
            return groups
        current_key = sampling_keys[0]
        current_indices: list[int] = [0]
        for index in range(1, len(sampling_keys)):
            key = sampling_keys[index]
            if key == current_key:
                current_indices.append(index)
                continue
            groups.append((current_key, current_indices))
            current_key = key
            current_indices = [index]
        groups.append((current_key, current_indices))
        return groups

    def _uniform_sampling_group_key(self, active_batch: Any) -> tuple[int, float, float, float, bool] | None:
        if not active_batch.states:
            return None
        if active_batch.step_indices.numel() <= 0:
            return None
        first_step_index = int(active_batch.step_indices[0].item())
        if bool((active_batch.step_indices != first_step_index).any().item()):
            return None
        first_state = active_batch.states[0]
        first_key = self._sampling_group_key(
            top_k=first_state.top_k,
            top_p=first_state.top_p,
            temperature=first_state.temperature,
            repetition_penalty=first_state.repetition_penalty,
            trim_eos=first_step_index < 11,
        )
        for state in active_batch.states[1:]:
            if (
                state.top_k != first_state.top_k
                or state.top_p != first_state.top_p
                or state.temperature != first_state.temperature
                or state.repetition_penalty != first_state.repetition_penalty
            ):
                return None
        return first_key

    def _batched_sample_uniform(
        self,
        logits: torch.Tensor,
        histories: Sequence[torch.LongTensor],
        sampling_key: tuple[int, float, float, float, bool],
        *,
        stats: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits_to_probs, multinomial_sample_one_no_sync = self._get_sampling_ops()
        top_k, top_p, temperature, repetition_penalty, trim_eos = sampling_key
        sample_logits = logits[:, :-1] if trim_eos else logits
        pending_cuda_timings: list[tuple[str, torch.cuda.Event | None, torch.cuda.Event | float]] | None = (
            [] if stats is not None else None
        )
        padded_histories = self._time_t2s_call(
            lambda: self._stack_token_sequences_if_same_length(histories),
            pending_cuda_timings=pending_cuda_timings,
            stat_key="sampling_history_stack_pad_ms",
            device=sample_logits.device,
        )
        history_mask = None
        if padded_histories is None:
            padded_histories, history_mask = self._time_t2s_call(
                lambda: _pad_token_sequences(histories),
                pending_cuda_timings=pending_cuda_timings,
                stat_key="sampling_history_stack_pad_ms",
                device=sample_logits.device,
            )
        probs = self._time_t2s_call(
            lambda: logits_to_probs(
                logits=sample_logits,
                previous_tokens=padded_histories,
                previous_token_mask=history_mask,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
            ),
            pending_cuda_timings=pending_cuda_timings,
            stat_key="sampling_logits_to_probs_ms",
            device=sample_logits.device,
        )
        sampled = self._time_t2s_call(
            lambda: multinomial_sample_one_no_sync(probs),
            pending_cuda_timings=pending_cuda_timings,
            stat_key="sampling_multinomial_ms",
            device=probs.device,
        )
        argmax_tokens = self._time_t2s_call(
            lambda: torch.argmax(sample_logits, dim=-1),
            pending_cuda_timings=pending_cuda_timings,
            stat_key="sampling_argmax_ms",
            device=sample_logits.device,
        )
        if pending_cuda_timings is not None:
            self._flush_t2s_timing_records(stats, pending_cuda_timings, device=sample_logits.device)
        return sampled, argmax_tokens

    def _batched_sample_by_group(
        self,
        logits: torch.Tensor,
        histories: Sequence[torch.LongTensor],
        sampling_keys: Sequence[tuple[int, float, float, float, bool]],
        *,
        stats: dict[str, Any] | None = None,
    ) -> tuple[list[torch.Tensor], list[int]]:
        logits_to_probs, multinomial_sample_one_no_sync = self._get_sampling_ops()
        sampled_list: list[torch.Tensor | None] = [None] * len(histories)
        argmax_list: list[int | None] = [None] * len(histories)
        pending_cuda_timings: list[tuple[str, torch.cuda.Event | None, torch.cuda.Event | float]] | None = (
            [] if stats is not None else None
        )
        for group_key, group_indices in self._iter_contiguous_sampling_groups(sampling_keys):
            top_k, top_p, temperature, repetition_penalty, trim_eos = group_key
            index_tensor = self._time_t2s_call(
                lambda: torch.tensor(group_indices, dtype=torch.long, device=logits.device),
                pending_cuda_timings=pending_cuda_timings,
                stat_key="sampling_group_select_ms",
                device=logits.device,
            )
            group_logits = self._time_t2s_call(
                lambda: torch.index_select(logits, dim=0, index=index_tensor),
                pending_cuda_timings=pending_cuda_timings,
                stat_key="sampling_group_select_ms",
                device=logits.device,
            )
            if trim_eos:
                group_logits = group_logits[:, :-1]
            group_histories = [histories[index] for index in group_indices]
            padded_histories = self._time_t2s_call(
                lambda: self._stack_token_sequences_if_same_length(group_histories),
                pending_cuda_timings=pending_cuda_timings,
                stat_key="sampling_history_stack_pad_ms",
                device=group_logits.device,
            )
            history_mask = None
            if padded_histories is None:
                padded_histories, history_mask = self._time_t2s_call(
                    lambda: _pad_token_sequences(group_histories),
                    pending_cuda_timings=pending_cuda_timings,
                    stat_key="sampling_history_stack_pad_ms",
                    device=group_logits.device,
                )
            probs = self._time_t2s_call(
                lambda: logits_to_probs(
                    logits=group_logits,
                    previous_tokens=padded_histories,
                    previous_token_mask=history_mask,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                ),
                pending_cuda_timings=pending_cuda_timings,
                stat_key="sampling_logits_to_probs_ms",
                device=group_logits.device,
            )
            argmax_tokens = self._time_t2s_call(
                lambda: torch.argmax(group_logits, dim=-1),
                pending_cuda_timings=pending_cuda_timings,
                stat_key="sampling_argmax_ms",
                device=group_logits.device,
            )
            for local_index, global_index in enumerate(group_indices):
                sampled_list[global_index] = self._time_t2s_call(
                    lambda current_index=local_index: multinomial_sample_one_no_sync(probs[current_index : current_index + 1]),
                    pending_cuda_timings=pending_cuda_timings,
                    stat_key="sampling_multinomial_ms",
                    device=probs.device,
                )
                argmax_list[global_index] = int(argmax_tokens[local_index].item())
        if pending_cuda_timings is not None:
            self._flush_t2s_timing_records(stats, pending_cuda_timings, device=logits.device)
        return [item for item in sampled_list if item is not None], [int(item) for item in argmax_list if item is not None]

    def _sample_active_batch_requests(
        self,
        model: Any,
        active_batch: Any,
        logits: torch.Tensor,
        *,
        max_steps: int,
        stats: dict[str, Any] | None = None,
    ) -> tuple[list[GPTSoVITSARFinishedItem], list[int], list[torch.LongTensor]]:
        sampling_started = time.perf_counter()
        finished_items: list[GPTSoVITSARFinishedItem] = []
        keep_indices: list[int] = []
        updated_sequences: list[torch.LongTensor] = []

        if len(active_batch.states) == 1:
            self._increment_t2s_stat(stats, "sampling_single_request_calls")
            state = active_batch.states[0]
            step_index = int(active_batch.step_indices[0].item())
            sampling_key = self._sampling_group_key(
                top_k=state.top_k,
                top_p=state.top_p,
                temperature=state.temperature,
                repetition_penalty=state.repetition_penalty,
                trim_eos=step_index < 11,
            )
            sampled_tensor, argmax_tensor = self._batched_sample_uniform(
                logits=logits,
                histories=active_batch.y_sequences,
                sampling_key=sampling_key,
                stats=stats,
            )
            finish_scan_started = time.perf_counter()
            sampled_token = int(sampled_tensor[0].item())
            argmax_token = int(argmax_tensor[0].item())
            current_history = active_batch.y_sequences[0]
            new_history = torch.cat([current_history, sampled_tensor.view(-1)], dim=0)
            prefix_len = int(active_batch.prefix_lens[0].item())

            if (
                state.early_stop_num == -1
                and step_index + 1 < max_steps
                and sampled_token != model.EOS
                and argmax_token != model.EOS
            ):
                self._accumulate_t2s_stat(stats, "sampling_finish_scan_ms", (time.perf_counter() - finish_scan_started) * 1000.0)
                self._accumulate_t2s_stat(stats, "sampling_total_ms", (time.perf_counter() - sampling_started) * 1000.0)
                return [], [0], [new_history]

            finish_reason: str | None = None
            if state.early_stop_num != -1 and (new_history.shape[0] - prefix_len) > state.early_stop_num:
                finish_reason = "early_stop"
            elif step_index + 1 >= max_steps:
                finish_reason = "max_step"
            elif sampled_token == model.EOS:
                finish_reason = "eos_sample"
            elif argmax_token == model.EOS:
                finish_reason = "eos_argmax"

            if finish_reason is not None:
                finished_items.append(
                    GPTSoVITSARFinishedItem(
                        request_id=str(state.request_id),
                        semantic_tokens=new_history[prefix_len:-1].clone(),
                        finish_idx=step_index,
                        finish_reason=finish_reason,
                    )
                )
                self._accumulate_t2s_stat(stats, "sampling_finish_scan_ms", (time.perf_counter() - finish_scan_started) * 1000.0)
                self._accumulate_t2s_stat(stats, "sampling_total_ms", (time.perf_counter() - sampling_started) * 1000.0)
                return finished_items, [], []
            self._accumulate_t2s_stat(stats, "sampling_finish_scan_ms", (time.perf_counter() - finish_scan_started) * 1000.0)
            self._accumulate_t2s_stat(stats, "sampling_total_ms", (time.perf_counter() - sampling_started) * 1000.0)
            return [], [0], [new_history]

        uniform_sampling_key = self._uniform_sampling_group_key(active_batch)
        sampled_items: list[torch.Tensor]
        argmax_tokens: list[int]
        sampled_token_tensor: torch.Tensor | None = None
        argmax_token_tensor: torch.Tensor | None = None
        if uniform_sampling_key is not None:
            self._increment_t2s_stat(stats, "sampling_uniform_calls")
            sampled_tensor, argmax_tensor = self._batched_sample_uniform(
                logits=logits,
                histories=active_batch.y_sequences,
                sampling_key=uniform_sampling_key,
                stats=stats,
            )
            sampled_token_tensor = sampled_tensor.view(-1)
            argmax_token_tensor = argmax_tensor.view(-1)
            finish_scan_started = time.perf_counter()
            stacked_histories = self._stack_token_sequences_if_same_length(active_batch.y_sequences)
            if (
                all(state.early_stop_num == -1 for state in active_batch.states)
                and int(active_batch.step_indices[0].item()) + 1 < max_steps
                and not bool(sampled_token_tensor.eq(model.EOS).any().item())
                and not bool(argmax_token_tensor.eq(model.EOS).any().item())
            ):
                self._accumulate_t2s_stat(stats, "sampling_finish_scan_ms", (time.perf_counter() - finish_scan_started) * 1000.0)
                self._accumulate_t2s_stat(stats, "sampling_total_ms", (time.perf_counter() - sampling_started) * 1000.0)
                return (
                    [],
                    list(range(len(active_batch.states))),
                    list(torch.cat([stacked_histories, sampled_token_tensor.view(-1, 1)], dim=1).unbind(0))
                    if stacked_histories is not None
                    else [
                        torch.cat([history, sampled_token_tensor[index : index + 1]], dim=0)
                        for index, history in enumerate(active_batch.y_sequences)
                    ],
                )
            self._accumulate_t2s_stat(stats, "sampling_finish_scan_ms", (time.perf_counter() - finish_scan_started) * 1000.0)
            sampled_items = [sampled_tensor[index : index + 1] for index in range(sampled_tensor.shape[0])]
            argmax_tokens = [int(item) for item in argmax_tensor.tolist()]
        else:
            self._increment_t2s_stat(stats, "sampling_grouped_calls")
            sampling_keys = [
                self._sampling_group_key(
                    top_k=state.top_k,
                    top_p=state.top_p,
                    temperature=state.temperature,
                    repetition_penalty=state.repetition_penalty,
                    trim_eos=int(active_batch.step_indices[batch_index].item()) < 11,
                )
                for batch_index, state in enumerate(active_batch.states)
            ]
            sampled_items, argmax_tokens = self._batched_sample_by_group(
                logits=logits,
                histories=active_batch.y_sequences,
                sampling_keys=sampling_keys,
                stats=stats,
            )

        finish_scan_started = time.perf_counter()
        for batch_index, state in enumerate(active_batch.states):
            step_index = int(active_batch.step_indices[batch_index].item())
            current_history = active_batch.y_sequences[batch_index]
            if sampled_token_tensor is not None and argmax_token_tensor is not None:
                sampled = sampled_token_tensor[batch_index : batch_index + 1]
                sampled_token = int(sampled_token_tensor[batch_index].item())
                argmax_token = int(argmax_token_tensor[batch_index].item())
            else:
                sampled = sampled_items[batch_index]
                sampled_token = int(sampled[0, 0].item())
                argmax_token = argmax_tokens[batch_index]
            new_history = torch.cat([current_history, sampled.view(-1)], dim=0)

            finish_reason: str | None = None
            if state.early_stop_num != -1 and (
                new_history.shape[0] - int(active_batch.prefix_lens[batch_index].item())
            ) > state.early_stop_num:
                finish_reason = "early_stop"
            elif step_index + 1 >= max_steps:
                finish_reason = "max_step"
            elif sampled_token == model.EOS:
                finish_reason = "eos_sample"
            elif argmax_token == model.EOS:
                finish_reason = "eos_argmax"

            if finish_reason is not None:
                prefix_len = int(active_batch.prefix_lens[batch_index].item())
                finished_items.append(
                    GPTSoVITSARFinishedItem(
                        request_id=str(state.request_id),
                        semantic_tokens=new_history[prefix_len:-1].clone(),
                        finish_idx=step_index,
                        finish_reason=finish_reason,
                    )
                )
            else:
                keep_indices.append(batch_index)
                updated_sequences.append(new_history)
        self._accumulate_t2s_stat(stats, "sampling_finish_scan_ms", (time.perf_counter() - finish_scan_started) * 1000.0)
        self._accumulate_t2s_stat(stats, "sampling_total_ms", (time.perf_counter() - sampling_started) * 1000.0)
        return finished_items, keep_indices, updated_sequences

    def _decode_active_batch_one_step(
        self,
        model: Any,
        active_batch: Any,
        *,
        max_steps: int,
        stats: dict[str, Any] | None = None,
    ) -> tuple[Any | None, list[GPTSoVITSARFinishedItem]]:
        was_prefill = not active_batch.prefill_done
        stage_started = time.perf_counter()
        timing_device = active_batch.xy_pos.device if isinstance(getattr(active_batch, "xy_pos", None), torch.Tensor) else None
        pending_cuda_timings: list[tuple[str, torch.cuda.Event | None, torch.cuda.Event | float]] | None = (
            [] if stats is not None else None
        )

        def _record_stage(path: str, current_active_batch: Any | None) -> None:
            if stats is None:
                return
            elapsed_ms = (time.perf_counter() - stage_started) * 1000.0
            if was_prefill:
                stats["prefill_calls"] = int(stats.get("prefill_calls", 0)) + 1
                stats["prefill_wall_ms"] = float(stats.get("prefill_wall_ms", 0.0)) + float(elapsed_ms)
            else:
                stats["decode_calls"] = int(stats.get("decode_calls", 0)) + 1
                stats["decode_wall_ms"] = float(stats.get("decode_wall_ms", 0.0)) + float(elapsed_ms)
                if path == "pooled":
                    stats["pooled_decode_calls"] = int(stats.get("pooled_decode_calls", 0)) + 1
                    stats["pooled_decode_wall_ms"] = float(stats.get("pooled_decode_wall_ms", 0.0)) + float(elapsed_ms)
                else:
                    stats["dynamic_decode_calls"] = int(stats.get("dynamic_decode_calls", 0)) + 1
                    stats["dynamic_decode_wall_ms"] = float(stats.get("dynamic_decode_wall_ms", 0.0)) + float(elapsed_ms)
            if current_active_batch is not None:
                stats["max_batch_size_seen"] = max(
                    int(stats.get("max_batch_size_seen", 0)),
                    int(len(getattr(current_active_batch, "request_ids", []) or [])),
                )
                kv_lens = getattr(current_active_batch, "kv_lens", None)
                if kv_lens is not None and getattr(kv_lens, "numel", lambda: 0)() > 0:
                    stats["max_kv_len_seen"] = max(
                        int(stats.get("max_kv_len_seen", 0)),
                        int(kv_lens.max().item()),
                    )

        if was_prefill:
            if active_batch.prefill_attn_mask is None or active_batch.key_padding_mask is None:
                raise ValueError("GPT-SoVITS AR prefill stage is missing masks")
            xy_dec, active_batch.k_cache, active_batch.v_cache = self._time_t2s_call(
                lambda: model.t2s_transformer.process_prompt(
                    active_batch.xy_pos,
                    active_batch.prefill_attn_mask,
                    None,
                ),
                pending_cuda_timings=pending_cuda_timings,
                stat_key="prefill_transformer_ms",
                device=timing_device,
            )
            active_batch.kv_lens = active_batch.x_lens + active_batch.prefix_lens
            if active_batch.k_cache is None or active_batch.v_cache is None or active_batch.kv_lens is None:
                raise ValueError("GPT-SoVITS AR prefill did not produce complete KV cache")
            if not self._pack_active_batch_into_pool(model, active_batch):
                active_batch.decode_attn_mask = F.pad(
                    active_batch.key_padding_mask.unsqueeze(1).unsqueeze(1),
                    (0, 1),
                    value=False,
                )
                active_batch.k_cache = [
                    self._compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.k_cache
                ]
                active_batch.v_cache = [
                    self._compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.v_cache
                ]
                active_batch.decode_attn_mask = self._compact_decode_mask_to_kv_lens(
                    active_batch.decode_attn_mask,
                    active_batch.kv_lens,
                )
            active_batch.x = None
            active_batch.x_lens = None
            active_batch.key_padding_mask = None
            active_batch.prefill_attn_mask = None
            active_batch.prefill_done = True
        else:
            if active_batch.k_cache is None or active_batch.v_cache is None or active_batch.kv_lens is None:
                raise ValueError("GPT-SoVITS AR decode stage is missing KV cache")
            if active_batch.kv_cache_pooled:
                pool = self._get_kv_pool(model)
                if pool is None:
                    raise ValueError("GPT-SoVITS AR pooled KV cache is unavailable")
                next_kv_lens = active_batch.kv_lens + 1
                if int(next_kv_lens.max().item()) > int(pool.max_seq_len):
                    self._fallback_pooled_active_batch_to_dynamic_cache(
                        model,
                        active_batch,
                        reason=(
                            "decode_headroom_overflow"
                            f"(batch={int(next_kv_lens.shape[0])},seq={int(active_batch.kv_lens.max().item())},"
                            f"next={int(next_kv_lens.max().item())})"
                        ),
                    )
                else:
                    batched_decode_attn_mask = self._time_t2s_call(
                        lambda: pool.build_decode_mask(next_kv_lens),
                        pending_cuda_timings=pending_cuda_timings,
                        stat_key="pooled_build_decode_mask_ms",
                        device=timing_device,
                    )
                    xy_dec, active_batch.k_cache, active_batch.v_cache = self._time_t2s_call(
                        lambda: model.decode_next_token_prealloc_runtime(
                            active_batch.xy_pos,
                            active_batch.k_cache,
                            active_batch.v_cache,
                            active_batch.kv_lens,
                            batched_decode_attn_mask,
                        ),
                        pending_cuda_timings=pending_cuda_timings,
                        stat_key="pooled_prealloc_decode_kernel_ms",
                        device=timing_device,
                    )
                    logits = self._time_t2s_call(
                        lambda: model.ar_predict_layer(xy_dec[:, -1]),
                        pending_cuda_timings=pending_cuda_timings,
                        stat_key="ar_predict_layer_ms",
                        device=xy_dec.device,
                    )
                    self._merge_numeric_t2s_stats(
                        stats,
                        getattr(model, "get_last_prealloc_decode_profile", lambda: {})(),
                    )
                    if pending_cuda_timings is not None:
                        self._flush_t2s_timing_records(stats, pending_cuda_timings, device=xy_dec.device)
                    finished_items, keep_indices, updated_sequences = self._sample_active_batch_requests(
                        model,
                        active_batch,
                        logits,
                        max_steps=max_steps,
                        stats=stats,
                    )
                    if len(keep_indices) == 0:
                        self._set_kv_pool_active_rows(model, 0)
                        _record_stage("pooled", None)
                        return None, finished_items
                    if len(keep_indices) == len(active_batch.request_ids):
                        active_batch.y_sequences = updated_sequences
                        active_batch.step_indices = active_batch.step_indices + 1
                        active_batch.kv_lens = active_batch.kv_lens + 1
                        xy_pos_started = time.perf_counter()
                        active_batch.xy_pos = self._build_next_xy_pos(model, active_batch.y_sequences)
                        self._accumulate_t2s_stat(stats, "xy_pos_update_ms", (time.perf_counter() - xy_pos_started) * 1000.0)
                        _record_stage("pooled", active_batch)
                        return active_batch, finished_items
                    device = logits.device
                    keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=device)
                    active_batch.request_ids = [active_batch.request_ids[i] for i in keep_indices]
                    active_batch.states = [active_batch.states[i] for i in keep_indices]
                    active_batch.y_sequences = updated_sequences
                    active_batch.prefix_lens = torch.index_select(active_batch.prefix_lens, dim=0, index=keep_tensor)
                    next_step_indices = torch.index_select(active_batch.step_indices, dim=0, index=keep_tensor)
                    next_kv_lens = torch.index_select(active_batch.kv_lens, dim=0, index=keep_tensor) + 1
                    active_batch.step_indices = next_step_indices + 1
                    active_batch.kv_lens = next_kv_lens
                    pooled_compacted = self._compact_pooled_active_batch(model, active_batch, keep_indices)
                    if not pooled_compacted:
                        active_batch.kv_cache_pooled = False
                    if (not active_batch.kv_cache_pooled) and active_batch.k_cache is not None and active_batch.v_cache is not None:
                        for cache_index in range(len(active_batch.k_cache)):
                            active_batch.k_cache[cache_index] = torch.index_select(
                                active_batch.k_cache[cache_index],
                                dim=0,
                                index=keep_tensor,
                            )
                            active_batch.v_cache[cache_index] = torch.index_select(
                                active_batch.v_cache[cache_index],
                                dim=0,
                                index=keep_tensor,
                            )
                        active_batch.k_cache = [
                            self._compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.k_cache
                        ]
                        active_batch.v_cache = [
                            self._compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.v_cache
                        ]
                        active_batch.decode_attn_mask = self._build_decode_mask_from_kv_lens(
                            active_batch.kv_lens,
                            device=active_batch.k_cache[0].device,
                        )
                    xy_pos_started = time.perf_counter()
                    active_batch.xy_pos = self._build_next_xy_pos(model, active_batch.y_sequences)
                    self._accumulate_t2s_stat(stats, "xy_pos_update_ms", (time.perf_counter() - xy_pos_started) * 1000.0)
                    _record_stage("pooled", active_batch)
                    return active_batch, finished_items
            batched_decode_attn_mask = None
            if active_batch.decode_attn_mask is not None:
                batched_decode_attn_mask = self._time_t2s_call(
                    lambda: self._materialize_decode_mask_for_active_batch(active_batch),
                    pending_cuda_timings=pending_cuda_timings,
                    stat_key="dynamic_materialize_decode_mask_ms",
                    device=timing_device,
                )
                if not batched_decode_attn_mask.any().item():
                    batched_decode_attn_mask = None
            xy_dec, active_batch.k_cache, active_batch.v_cache = self._time_t2s_call(
                lambda: model.t2s_transformer.decode_next_token(
                    active_batch.xy_pos,
                    active_batch.k_cache,
                    active_batch.v_cache,
                    batched_decode_attn_mask,
                ),
                pending_cuda_timings=pending_cuda_timings,
                stat_key="dynamic_decode_kernel_ms",
                device=timing_device,
            )
            active_batch.decode_attn_mask = self._time_t2s_call(
                lambda: self._advance_decode_mask(active_batch.decode_attn_mask, active_batch.kv_lens),
                pending_cuda_timings=pending_cuda_timings,
                stat_key="dynamic_advance_decode_mask_ms",
                device=timing_device,
            )

        logits = self._time_t2s_call(
            lambda: model.ar_predict_layer(xy_dec[:, -1]),
            pending_cuda_timings=pending_cuda_timings,
            stat_key="ar_predict_layer_ms",
            device=xy_dec.device,
        )
        if pending_cuda_timings is not None:
            self._flush_t2s_timing_records(stats, pending_cuda_timings, device=xy_dec.device)
        finished_items, keep_indices, updated_sequences = self._sample_active_batch_requests(
            model,
            active_batch,
            logits,
            max_steps=max_steps,
            stats=stats,
        )

        if len(keep_indices) == 0:
            if active_batch.kv_cache_pooled:
                self._set_kv_pool_active_rows(model, 0)
            _record_stage("prefill" if was_prefill else "dynamic", None)
            return None, finished_items

        if len(keep_indices) == len(active_batch.request_ids):
            active_batch.y_sequences = updated_sequences
            active_batch.step_indices = active_batch.step_indices + 1
            if not was_prefill and active_batch.kv_lens is not None:
                active_batch.kv_lens = active_batch.kv_lens + 1
            xy_pos_started = time.perf_counter()
            active_batch.xy_pos = self._build_next_xy_pos(model, active_batch.y_sequences)
            self._accumulate_t2s_stat(stats, "xy_pos_update_ms", (time.perf_counter() - xy_pos_started) * 1000.0)
            _record_stage("prefill" if was_prefill else "dynamic", active_batch)
            return active_batch, finished_items

        device = logits.device
        keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=device)
        active_batch.request_ids = [active_batch.request_ids[i] for i in keep_indices]
        active_batch.states = [active_batch.states[i] for i in keep_indices]
        active_batch.y_sequences = updated_sequences
        active_batch.prefix_lens = torch.index_select(active_batch.prefix_lens, dim=0, index=keep_tensor)
        next_step_indices = torch.index_select(active_batch.step_indices, dim=0, index=keep_tensor)
        next_kv_lens = None if active_batch.kv_lens is None else torch.index_select(active_batch.kv_lens, dim=0, index=keep_tensor)
        active_batch.step_indices = next_step_indices + 1
        active_batch.kv_lens = next_kv_lens if was_prefill else (None if next_kv_lens is None else next_kv_lens + 1)

        if active_batch.decode_attn_mask is not None:
            active_batch.decode_attn_mask = torch.index_select(active_batch.decode_attn_mask, dim=0, index=keep_tensor)
            if not active_batch.decode_attn_mask.any().item():
                active_batch.decode_attn_mask = None
        if active_batch.kv_cache_pooled:
            pooled_compacted = self._compact_pooled_active_batch(model, active_batch, keep_indices)
            if not pooled_compacted:
                active_batch.kv_cache_pooled = False
        if (not active_batch.kv_cache_pooled) and active_batch.k_cache is not None and active_batch.v_cache is not None:
            for cache_index in range(len(active_batch.k_cache)):
                active_batch.k_cache[cache_index] = torch.index_select(active_batch.k_cache[cache_index], dim=0, index=keep_tensor)
                active_batch.v_cache[cache_index] = torch.index_select(active_batch.v_cache[cache_index], dim=0, index=keep_tensor)
            if active_batch.kv_lens is not None:
                active_batch.k_cache = [
                    self._compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.k_cache
                ]
                active_batch.v_cache = [
                    self._compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.v_cache
                ]
                active_batch.decode_attn_mask = self._compact_decode_mask_to_kv_lens(
                    active_batch.decode_attn_mask,
                    active_batch.kv_lens,
                )

        xy_pos_started = time.perf_counter()
        active_batch.xy_pos = self._build_next_xy_pos(model, active_batch.y_sequences)
        self._accumulate_t2s_stat(stats, "xy_pos_update_ms", (time.perf_counter() - xy_pos_started) * 1000.0)
        _record_stage("prefill" if was_prefill else "dynamic", active_batch)
        return active_batch, finished_items

    def _run_prefill_active_batch(
        self,
        model: Any,
        states: Sequence[Any],
        *,
        max_steps: int,
        stats: dict[str, Any] | None = None,
    ) -> tuple[Any | None, list[GPTSoVITSARFinishedItem]]:
        if not states:
            return None, []
        active_batch = self._build_prefill_active_batch(model, states)
        return self._decode_active_batch_one_step(model, active_batch, max_steps=max_steps, stats=stats)

    def _merge_active_batches(
        self,
        model: Any,
        left_batch: Any | None,
        right_batch: Any | None,
    ) -> Any | None:
        if left_batch is None:
            return right_batch
        if right_batch is None:
            return left_batch
        if not left_batch.prefill_done or not right_batch.prefill_done:
            raise ValueError("Only prefill-complete GPT-SoVITS active batches can be merged")
        if left_batch.k_cache is None or left_batch.v_cache is None or right_batch.k_cache is None or right_batch.v_cache is None:
            raise ValueError("GPT-SoVITS active batch merge is missing KV cache")
        if left_batch.kv_lens is None or right_batch.kv_lens is None:
            raise ValueError("GPT-SoVITS active batch merge is missing kv_lens")

        merged_kv_lens = torch.cat([left_batch.kv_lens, right_batch.kv_lens], dim=0)
        merged_kv_len = int(merged_kv_lens.max().item())
        merged_mask_len = merged_kv_len + 1
        merged_k_cache: list[torch.Tensor] = []
        merged_v_cache: list[torch.Tensor] = []
        left_request_count = len(left_batch.request_ids)
        right_request_count = len(right_batch.request_ids)

        for layer_index in range(len(left_batch.k_cache)):
            layer_device = left_batch.k_cache[layer_index].device
            layer_dtype = left_batch.k_cache[layer_index].dtype
            layer_hidden = int(left_batch.k_cache[layer_index].shape[2])
            merged_layer_k = torch.zeros(
                (left_request_count + right_request_count, merged_kv_len, layer_hidden),
                dtype=layer_dtype,
                device=layer_device,
            )
            merged_layer_v = torch.zeros_like(merged_layer_k)
            for batch_index, kv_len in enumerate(left_batch.kv_lens.tolist()):
                if kv_len <= 0:
                    continue
                merged_layer_k[batch_index : batch_index + 1, -kv_len:, :] = self._extract_cache_row(
                    left_batch.k_cache[layer_index],
                    batch_index,
                    int(kv_len),
                    pooled=bool(left_batch.kv_cache_pooled),
                )
                merged_layer_v[batch_index : batch_index + 1, -kv_len:, :] = self._extract_cache_row(
                    left_batch.v_cache[layer_index],
                    batch_index,
                    int(kv_len),
                    pooled=bool(left_batch.kv_cache_pooled),
                )
            for batch_index, kv_len in enumerate(right_batch.kv_lens.tolist()):
                if kv_len <= 0:
                    continue
                target_index = left_request_count + batch_index
                merged_layer_k[target_index : target_index + 1, -kv_len:, :] = self._extract_cache_row(
                    right_batch.k_cache[layer_index],
                    batch_index,
                    int(kv_len),
                    pooled=bool(right_batch.kv_cache_pooled),
                )
                merged_layer_v[target_index : target_index + 1, -kv_len:, :] = self._extract_cache_row(
                    right_batch.v_cache[layer_index],
                    batch_index,
                    int(kv_len),
                    pooled=bool(right_batch.kv_cache_pooled),
                )
            merged_k_cache.append(merged_layer_k)
            merged_v_cache.append(merged_layer_v)

        merged_decode_attn_mask = torch.ones(
            (left_request_count + right_request_count, 1, 1, merged_mask_len),
            dtype=torch.bool,
            device=merged_k_cache[0].device,
        )
        for batch_index, kv_len in enumerate(merged_kv_lens.tolist()):
            merged_decode_attn_mask[batch_index : batch_index + 1, :, :, -(int(kv_len) + 1) :] = False
        if not merged_decode_attn_mask.any().item():
            merged_decode_attn_mask = None

        merged_batch = GPTSoVITSActiveBatch(
            request_ids=list(left_batch.request_ids) + list(right_batch.request_ids),
            states=list(left_batch.states) + list(right_batch.states),
            x=None,
            x_lens=None,
            y_sequences=list(left_batch.y_sequences) + list(right_batch.y_sequences),
            prefix_lens=torch.cat([left_batch.prefix_lens, right_batch.prefix_lens], dim=0),
            xy_pos=self._build_next_xy_pos(model, list(left_batch.y_sequences) + list(right_batch.y_sequences)),
            key_padding_mask=None,
            prefill_attn_mask=None,
            decode_attn_mask=merged_decode_attn_mask,
            k_cache=merged_k_cache,
            v_cache=merged_v_cache,
            kv_lens=merged_kv_lens,
            step_indices=torch.cat([left_batch.step_indices, right_batch.step_indices], dim=0),
            prefill_done=True,
            kv_cache_pooled=False,
            kv_cache_capacity=0,
            kv_cache_batch_capacity=0,
        )
        self._pack_active_batch_into_pool(model, merged_batch)
        return merged_batch

    def _run_continuous_batch_scheduler(
        self,
        model: Any,
        states: Sequence[Any],
        *,
        max_steps: int,
    ) -> list[GPTSoVITSARFinishedItem]:
        pool = self._get_kv_pool(model)
        pool_snapshot_start = None if pool is None else dict(pool.snapshot())
        stats: dict[str, Any] = {
            "request_count": int(len(states)),
            "max_steps_limit": int(max_steps),
            "scheduler_ticks": 0,
            "prefill_calls": 0,
            "prefill_wall_ms": 0.0,
            "decode_calls": 0,
            "decode_wall_ms": 0.0,
            "pooled_decode_calls": 0,
            "pooled_decode_wall_ms": 0.0,
            "dynamic_decode_calls": 0,
            "dynamic_decode_wall_ms": 0.0,
            "max_batch_size_seen": 0,
            "max_kv_len_seen": 0,
            "pool_snapshot_start": pool_snapshot_start,
            "prefill_transformer_ms": 0.0,
            "pooled_build_decode_mask_ms": 0.0,
            "pooled_prealloc_decode_kernel_ms": 0.0,
            "pooled_prealloc_profiled_decode_calls": 0,
            "pooled_prealloc_profiled_layer_calls": 0,
            "pooled_prealloc_qkv_linear_ms": 0.0,
            "pooled_prealloc_kv_write_ms": 0.0,
            "pooled_prealloc_kv_context_ms": 0.0,
            "pooled_prealloc_sdpa_ms": 0.0,
            "pooled_prealloc_out_proj_ms": 0.0,
            "pooled_prealloc_norm1_ms": 0.0,
            "pooled_prealloc_ffn_ms": 0.0,
            "pooled_prealloc_norm2_ms": 0.0,
            "pooled_prealloc_layer_total_ms": 0.0,
            "dynamic_materialize_decode_mask_ms": 0.0,
            "dynamic_decode_kernel_ms": 0.0,
            "dynamic_advance_decode_mask_ms": 0.0,
            "ar_predict_layer_ms": 0.0,
            "sampling_total_ms": 0.0,
            "sampling_history_stack_pad_ms": 0.0,
            "sampling_logits_to_probs_ms": 0.0,
            "sampling_multinomial_ms": 0.0,
            "sampling_argmax_ms": 0.0,
            "sampling_group_select_ms": 0.0,
            "sampling_finish_scan_ms": 0.0,
            "sampling_single_request_calls": 0,
            "sampling_uniform_calls": 0,
            "sampling_grouped_calls": 0,
            "xy_pos_update_ms": 0.0,
        }
        pending = sorted(states, key=lambda item: (item.ready_step, item.request_id))
        active_batch: Any | None = None
        finished: list[GPTSoVITSARFinishedItem] = []
        current_tick = 0

        while pending or active_batch is not None:
            stats["scheduler_ticks"] = int(stats["scheduler_ticks"]) + 1
            admitted: list[Any] = []
            while pending and pending[0].ready_step <= current_tick:
                admitted.append(pending.pop(0))

            admitted_active_batch, admitted_finished = self._run_prefill_active_batch(
                model,
                admitted,
                max_steps=max_steps,
                stats=stats,
            )
            finished.extend(admitted_finished)
            active_batch = self._merge_active_batches(model, active_batch, admitted_active_batch)

            if active_batch is not None:
                if active_batch.kv_lens is not None and active_batch.kv_lens.numel() > 0:
                    stats["max_kv_len_seen"] = max(
                        int(stats["max_kv_len_seen"]),
                        int(active_batch.kv_lens.max().item()),
                    )
                stats["max_batch_size_seen"] = max(
                    int(stats["max_batch_size_seen"]),
                    int(len(active_batch.request_ids)),
                )
                active_batch, step_finished = self._decode_active_batch_one_step(
                    model,
                    active_batch,
                    max_steps=max_steps,
                    stats=stats,
                )
                finished.extend(step_finished)

            if active_batch is None and pending:
                current_tick = max(current_tick + 1, int(pending[0].ready_step))
                continue

            current_tick += 1

        finished.sort(key=lambda item: item.request_id)
        pool = self._get_kv_pool(model)
        pool_snapshot_end = None if pool is None else dict(pool.snapshot())
        stats["pool_snapshot_end"] = pool_snapshot_end
        if isinstance(pool_snapshot_start, dict) and isinstance(pool_snapshot_end, dict):
            stats["pool_pack_hits_delta"] = int(pool_snapshot_end.get("pack_hits", 0)) - int(pool_snapshot_start.get("pack_hits", 0))
            stats["pool_fallback_count_delta"] = int(pool_snapshot_end.get("fallback_count", 0)) - int(
                pool_snapshot_start.get("fallback_count", 0)
            )
            stats["pool_last_fallback_reason"] = str(pool_snapshot_end.get("last_fallback_reason", ""))
        self._last_t2s_scheduler_stats = stats
        return finished

    @staticmethod
    def _run_awaitable_sync(awaitable: Any) -> Any:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is None or not running_loop.is_running():
            return asyncio.run(awaitable)

        result_holder: dict[str, Any] = {}
        error_holder: dict[str, BaseException] = {}

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result_holder["value"] = loop.run_until_complete(awaitable)
            except BaseException as exc:  # pragma: no cover - defensive
                error_holder["error"] = exc
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                asyncio.set_event_loop(None)
                loop.close()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder.get("value")

    @staticmethod
    def _load_ref_audio_with_soundfile(ref_audio_path: str) -> tuple[torch.Tensor, int]:
        import soundfile as sf

        raw_audio, raw_sr = sf.read(ref_audio_path, dtype="float32")
        raw_audio = np.asarray(raw_audio, dtype=np.float32)
        if raw_audio.ndim == 1:
            raw_audio = raw_audio[None, :]
        else:
            raw_audio = raw_audio.T
        return torch.from_numpy(np.ascontiguousarray(raw_audio)), int(raw_sr)

    def _install_ref_audio_loader_fallback(self, pipeline: Any) -> None:
        if getattr(pipeline, "_vllm_omni_ref_audio_loader_patched", False):
            return

        original_loader = getattr(pipeline, "_load_ref_audio_raw", None)
        if original_loader is None:
            return

        def _load_ref_audio_raw_with_fallback(tts_self: Any, ref_audio_path: str):
            try:
                return original_loader(ref_audio_path)
            except Exception as exc:
                exc_summary = str(exc).splitlines()[0].strip() or exc.__class__.__name__
                logger.warning(
                    "torchaudio failed to load GPT-SoVITS reference audio %s; falling back to soundfile: %s",
                    ref_audio_path,
                    exc_summary,
                )
                try:
                    return self._load_ref_audio_with_soundfile(ref_audio_path)
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"Failed to load GPT-SoVITS reference audio {ref_audio_path} "
                        f"with both torchaudio and soundfile"
                    ) from fallback_exc

        pipeline._load_ref_audio_raw = types.MethodType(_load_ref_audio_raw_with_fallback, pipeline)
        pipeline._vllm_omni_ref_audio_loader_patched = True

    def _install_sv_half_safe_patch(self, pipeline: Any) -> None:
        sv_model = getattr(pipeline, "sv_model", None)
        if sv_model is None or getattr(sv_model, "_vllm_omni_compute_embedding3_patched", False):
            return

        original_compute_embedding3 = getattr(sv_model, "compute_embedding3", None)
        if original_compute_embedding3 is None:
            return

        original_func = getattr(original_compute_embedding3, "__func__", original_compute_embedding3)
        kaldi_module = getattr(original_func, "__globals__", {}).get("Kaldi")
        if kaldi_module is None:
            return

        def _compute_embedding3_float32_fbank(sv_self: Any, wav: torch.Tensor):
            with torch.no_grad():
                wav = wav.float()
                feat = torch.stack(
                    [
                        kaldi_module.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0)
                        for wav0 in wav
                    ]
                )
                model_param = next(sv_self.embedding_model.parameters())
                model_device = model_param.device
                feat = feat.to(device=model_device, dtype=model_param.dtype)
                return sv_self.embedding_model.forward3(feat)

        sv_model.compute_embedding3 = types.MethodType(_compute_embedding3_float32_fbank, sv_model)
        sv_model._vllm_omni_compute_embedding3_patched = True

    def _build_scheduler_request_spec(
        self,
        request: dict[str, Any],
        request_id: str | None = None,
    ) -> GPTSoVITSRequestSpec:
        pipeline = self._ensure_pipeline()
        inputs = self.build_tts_inputs(request)
        return GPTSoVITSRequestSpec(
            request_id=str(request_id or request.get("request_id") or request.get("engine_request_id") or "gpt_sovits"),
            ref_audio_path=str(inputs["ref_audio_path"]),
            prompt_text=str(inputs["prompt_text"]),
            prompt_lang=str(inputs["prompt_lang"]),
            text=str(inputs["text"]),
            text_lang=str(inputs["text_lang"]),
            text_split_method=str(inputs["text_split_method"]),
            top_k=int(inputs["top_k"]),
            top_p=float(inputs["top_p"]),
            temperature=float(inputs["temperature"]),
            repetition_penalty=float(inputs["repetition_penalty"]),
            early_stop_num=int(getattr(pipeline.configs, "hz", 50) * getattr(pipeline.configs, "max_sec", 30)),
            aux_ref_audio_paths=[str(item) for item in list(inputs.get("aux_ref_audio_paths") or [])],
            speed_factor=float(inputs.get("speed_factor", 1.0)),
            sample_steps=int(inputs.get("sample_steps", 32)),
            super_sampling=bool(inputs.get("super_sampling", False)),
            ready_step=int(request.get("ready_step", 0)),
        )

    def build_request_spec(
        self,
        request: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> GPTSoVITSRequestSpec:
        return self._build_scheduler_request_spec(request, request_id=request_id)

    def _state_to_transport_info(self, state: Any, spec: GPTSoVITSRequestSpec) -> GPTSoVITSStageTransport:
        return GPTSoVITSStageTransport.from_state(state, spec)

    @staticmethod
    def _estimate_scheduler_max_steps(states: list[Any]) -> int:
        env_override = os.environ.get("GPT_SOVITS_SCHEDULER_MAX_STEPS")
        max_steps = max(1, int(env_override)) if env_override not in [None, ""] else 1500
        for state in states:
            phones = getattr(state, "all_phones", None)
            bert = getattr(state, "all_bert_features", None)
            early_stop_num = int(getattr(state, "early_stop_num", -1))
            phones_len = int(phones.shape[0]) if isinstance(phones, torch.Tensor) and phones.ndim > 0 else 0
            bert_len = int(bert.shape[-1]) if isinstance(bert, torch.Tensor) and bert.ndim > 0 else 0
            # Semantic length regularly exceeds phone/BERT sequence length, so
            # the scheduler budget must not be derived from those tensors alone.
            heuristic_steps = max(phones_len, bert_len) * 8
            max_steps = max(max_steps, phones_len, bert_len, heuristic_steps)
            if early_stop_num > 0:
                max_steps = max(max_steps, early_stop_num + 1)
        return max_steps

    def preload_ref_audio_asset(self, ref_audio_path: str, *, submit_at: float | None = None) -> Any:
        coordinator = self._ensure_prepare_coordinator()
        return coordinator.submit_prepare_ref_audio_asset(ref_audio_path, submit_at=submit_at)

    def _normalize_prepare_sentence(self, text: str, language: str) -> str:
        with self._project_root_cwd():
            from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import normalize_sentence

        return normalize_sentence(text, language)

    @staticmethod
    def _prepare_result_has_pending_g2pw(prepared_segments: Any) -> bool:
        return any(bool(getattr(segment, "needs_g2pw", False)) for segment in (prepared_segments or []))

    def _get_text_preprocessor_symbol(self, symbol: str) -> Any:
        self._ensure_import_path()
        with self._project_root_cwd():
            from GPT_SoVITS.TTS_infer_pack import TextPreprocessor as _text_preprocessor_module

        return getattr(_text_preprocessor_module, symbol)

    def _get_text_frontend_symbol(self, module_name: str, symbol: str) -> Any:
        self._ensure_import_path()
        with self._project_root_cwd():
            module = __import__(module_name, fromlist=[symbol])
        return getattr(module, symbol)

    def _current_text_frontend_version(self) -> str:
        pipeline = self._ensure_pipeline()
        return str(getattr(getattr(pipeline, "configs", None), "version", "v2"))

    def _preprocess_text_segments_payload(
        self,
        text: str,
        language: str,
        version: str,
    ) -> list[dict[str, object]]:
        preprocess_text_segments_payload = self._get_text_frontend_symbol(
            "GPT_SoVITS.TTS_infer_pack.text_cpu_preprocess",
            "preprocess_text_segments_payload",
        )
        return list(preprocess_text_segments_payload(text, language, version))

    def _payloads_to_prepared_text_segments(self, payloads: list[dict[str, object]]) -> list[Any]:
        prepared_text_segment_cls = self._get_text_preprocessor_symbol("PreparedTextSegment")
        return [
            prepared_text_segment_cls(
                language=str(payload["language"]),
                phones=list(payload["phones"]),
                word2ph=None if payload["word2ph"] is None else list(payload["word2ph"]),
                norm_text=str(payload["norm_text"]),
                needs_g2pw=bool(payload.get("needs_g2pw", False)),
            )
            for payload in payloads
        ]

    def _prepare_text_cpu(self, text: str, language: str, text_split_method: str = "cut1") -> Any:
        version = self._current_text_frontend_version()
        pipeline = self._ensure_pipeline()
        text_preprocessor = getattr(pipeline, "text_preprocessor", None)
        if text_preprocessor is not None:
            return text_preprocessor.preprocess_text_segments(
                text,
                language,
                version,
                text_split_method=text_split_method,
            )
        payloads = self._preprocess_text_segments_payload(text, language, version)
        return self._payloads_to_prepared_text_segments(payloads)

    @classmethod
    def _merge_g2pw_profile(
        cls,
        profile: dict[str, float] | None,
        g2pw_profile: dict[str, float],
    ) -> None:
        cls._accumulate_profile_metric(profile, "g2pw_prepare_ms", g2pw_profile.get("g2pw_prepare_ms", 0.0))
        cls._accumulate_profile_metric(profile, "g2pw_predict_ms", g2pw_profile.get("g2pw_predict_ms", 0.0))
        cls._accumulate_profile_metric(profile, "g2pw_post_ms", g2pw_profile.get("g2pw_post_ms", 0.0))
        cls._accumulate_profile_metric(profile, "g2pw_total_ms", g2pw_profile.get("g2pw_total_ms", 0.0))
        cls._accumulate_profile_metric(
            profile,
            "g2pw_runtime_total_ms",
            g2pw_profile.get("g2pw_runtime_total_ms", 0.0),
        )
        cls._accumulate_profile_metric(
            profile,
            "g2pw_runtime_queue_wait_ms",
            g2pw_profile.get("g2pw_runtime_queue_wait_ms", 0.0),
        )
        cls._accumulate_profile_metric(
            profile,
            "g2pw_runtime_collect_wait_ms",
            g2pw_profile.get("g2pw_runtime_collect_wait_ms", 0.0),
        )
        cls._accumulate_profile_metric(
            profile,
            "g2pw_runtime_run_ms",
            g2pw_profile.get("g2pw_runtime_run_ms", 0.0),
        )
        cls._update_profile_metric_peak(
            profile,
            "g2pw_runtime_batch_rows_peak",
            g2pw_profile.get("g2pw_runtime_batch_rows", 0.0),
        )
        cls._update_profile_metric_peak(
            profile,
            "g2pw_runtime_batch_requests_peak",
            g2pw_profile.get("g2pw_runtime_batch_requests", 0.0),
        )
        cls._update_profile_metric_peak(
            profile,
            "g2pw_runtime_pool_workers",
            g2pw_profile.get("g2pw_runtime_pool_workers", 0.0),
        )

    def _resolve_g2pw_segments(self, prepared_segments: Any) -> tuple[Any, dict[str, float]]:
        profile: dict[str, float] = {}
        prepared_segment_list = list(prepared_segments or [])
        zh_indices = [
            index for index, segment in enumerate(prepared_segment_list) if bool(getattr(segment, "needs_g2pw", False))
        ]
        if not zh_indices:
            return prepared_segment_list, profile

        g2p_segments = self._get_text_frontend_symbol("text.chinese2", "g2p_segments")
        cleaned_text_to_sequence = self._get_text_frontend_symbol("text", "cleaned_text_to_sequence")
        prepared_text_segment_cls = self._get_text_preprocessor_symbol("PreparedTextSegment")
        version = self._current_text_frontend_version()
        normalized_segments = [str(getattr(prepared_segment_list[index], "norm_text", "") or "") for index in zh_indices]
        resolved_segments, g2pw_profile = g2p_segments(normalized_segments, return_profile=True)
        self._merge_g2pw_profile(profile, dict(g2pw_profile or {}))
        for index, (phones, word2ph, norm_text) in zip(zh_indices, resolved_segments):
            prepared_segment_list[index] = prepared_text_segment_cls(
                language=str(getattr(prepared_segment_list[index], "language", "")),
                phones=list(cleaned_text_to_sequence(phones, version)),
                word2ph=None if word2ph is None else list(word2ph),
                norm_text=str(norm_text),
                needs_g2pw=False,
            )
        return prepared_segment_list, profile

    def _resolve_g2pw_segment_batches(
        self,
        prepared_segment_batches: list[Any],
    ) -> tuple[Any, list[dict[str, float]]]:
        prepared_batches = [list(batch or []) for batch in (prepared_segment_batches or [])]
        profiles: list[dict[str, float]] = [{} for _ in prepared_batches]
        if not prepared_batches:
            return prepared_batches, profiles
        zh_indices_batches = [
            [index for index, segment in enumerate(prepared_segments) if bool(getattr(segment, "needs_g2pw", False))]
            for prepared_segments in prepared_batches
        ]
        if not any(zh_indices_batches):
            return prepared_batches, profiles

        g2p_segments_batch = self._get_text_frontend_symbol("text.chinese2", "g2p_segments_batch")
        cleaned_text_to_sequence = self._get_text_frontend_symbol("text", "cleaned_text_to_sequence")
        prepared_text_segment_cls = self._get_text_preprocessor_symbol("PreparedTextSegment")
        version = self._current_text_frontend_version()
        normalized_segment_batches = [
            [str(getattr(prepared_segments[index], "norm_text", "") or "") for index in zh_indices]
            for prepared_segments, zh_indices in zip(prepared_batches, zh_indices_batches)
        ]
        resolved_segment_batches, g2pw_profiles = g2p_segments_batch(
            normalized_segment_batches,
            return_profiles=True,
        )
        resolved_batches: list[Any] = []
        for batch_index, (prepared_segments, zh_indices, resolved_segments) in enumerate(
            zip(prepared_batches, zh_indices_batches, resolved_segment_batches)
        ):
            batch_profile = dict((g2pw_profiles or [])[batch_index] or {})
            self._merge_g2pw_profile(profiles[batch_index], batch_profile)
            batch_result = list(prepared_segments)
            for index, (phones, word2ph, norm_text) in zip(zh_indices, resolved_segments):
                batch_result[index] = prepared_text_segment_cls(
                    language=str(getattr(batch_result[index], "language", "")),
                    phones=list(cleaned_text_to_sequence(phones, version)),
                    word2ph=None if word2ph is None else list(word2ph),
                    norm_text=str(norm_text),
                    needs_g2pw=False,
                )
            resolved_batches.append(batch_result)
        return resolved_batches, profiles

    def _load_ref_audio_raw(self, ref_audio_path: str) -> Any:
        pipeline = self._ensure_pipeline()
        return pipeline._load_ref_audio_raw(ref_audio_path)

    def _extract_ref_spec_native(self, ref_audio_path: str) -> tuple[GPTSoVITSReferSpec, Any, int]:
        raw_audio, raw_sr = self._load_ref_audio_raw(ref_audio_path)
        refer_spec, _profile = self._extract_ref_spec_from_raw(raw_audio, raw_sr)
        return refer_spec, raw_audio, int(raw_sr)

    def _extract_ref_spec_from_raw(self, raw_audio: Any, raw_sr: int) -> tuple[GPTSoVITSReferSpec, dict[str, float]]:
        pipeline = self._ensure_pipeline()
        with self._project_root_cwd():
            if self.project_root not in sys.path:
                sys.path.insert(0, self.project_root)
            from GPT_SoVITS.module.mel_processing import spectrogram_torch

        device = torch.device(getattr(pipeline.configs, "device", "cpu"))
        sampling_rate = int(getattr(pipeline.configs, "sampling_rate", 32000))
        profile = {
            "ref_spec_to_device_ms": 0.0,
            "ref_spec_main_resample_ms": 0.0,
            "ref_spec_norm_ms": 0.0,
            "ref_spec_spectrogram_ms": 0.0,
            "ref_spec_post_resample_ms": 0.0,
        }
        with torch.no_grad():
            to_device_start = time.perf_counter()
            raw_audio_device = raw_audio.to(device).float()
            profile["ref_spec_to_device_ms"] = (time.perf_counter() - to_device_start) * 1000.0

            if raw_audio_device.ndim == 1:
                raw_audio_device = raw_audio_device.unsqueeze(0)
            if int(raw_sr) != sampling_rate:
                resample_start = time.perf_counter()
                audio = raw_audio_device
                if audio.shape[0] == 2:
                    audio = audio.mean(0).unsqueeze(0)
                audio = self._resample_audio(audio, int(raw_sr), sampling_rate, device)
                profile["ref_spec_main_resample_ms"] = (time.perf_counter() - resample_start) * 1000.0
            else:
                audio = raw_audio_device
                if audio.shape[0] == 2:
                    audio = audio.mean(0).unsqueeze(0)

            norm_start = time.perf_counter()
            max_audio = float(audio.abs().max().item()) if audio.numel() > 0 else 0.0
            if max_audio > 1.0:
                audio = audio / min(2.0, max_audio)
            profile["ref_spec_norm_ms"] = (time.perf_counter() - norm_start) * 1000.0

            spec_start = time.perf_counter()
            spec = spectrogram_torch(
                audio,
                int(getattr(pipeline.configs, "filter_length")),
                sampling_rate,
                int(getattr(pipeline.configs, "hop_length")),
                int(getattr(pipeline.configs, "win_length")),
                center=False,
            )
            profile["ref_spec_spectrogram_ms"] = (time.perf_counter() - spec_start) * 1000.0
            if bool(getattr(pipeline.configs, "is_half", False)):
                spec = spec.half()

            audio_16k = None
            if bool(getattr(pipeline, "is_v2pro", False)):
                post_resample_start = time.perf_counter()
                audio_16k = self._resample_audio(audio, sampling_rate, 16000, device)
                profile["ref_spec_post_resample_ms"] = (time.perf_counter() - post_resample_start) * 1000.0
                if bool(getattr(pipeline.configs, "is_half", False)):
                    audio_16k = audio_16k.half()
        return GPTSoVITSReferSpec(spec_audio=spec, audio_16k=audio_16k), profile

    @staticmethod
    def _coerce_refer_spec(value: Any) -> GPTSoVITSReferSpec:
        if isinstance(value, GPTSoVITSReferSpec):
            return value
        if isinstance(value, tuple) and len(value) == 2:
            return GPTSoVITSReferSpec(spec_audio=value[0], audio_16k=value[1])
        raise TypeError(f"Unsupported GPT-SoVITS refer spec value: {type(value)!r}")

    @staticmethod
    def _build_text_cpu_profiled_result(
        submit_at: float,
        result: Any,
        worker_profile: dict[str, float],
    ) -> GPTSoVITSPrepareProfiledResult:
        started_at = float(
            submit_at
            + (
                float(worker_profile.get("text_cpu_admission_wait_ms", 0.0))
                + float(worker_profile.get("text_cpu_queue_wait_ms", 0.0))
            )
            / 1000.0
        )
        finished_at = float(started_at + float(worker_profile.get("text_cpu_run_ms", 0.0)) / 1000.0)
        return GPTSoVITSPrepareProfiledResult(
            result=result,
            submit_at=float(submit_at),
            started_at=started_at,
            finished_at=finished_at,
            profile=dict(worker_profile),
        )

    def _prepare_prompt_semantic_wav16k_profile(
        self,
        raw_audio: torch.Tensor,
        raw_sr: int,
    ) -> tuple[torch.Tensor, float, dict[str, float]]:
        pipeline = self._ensure_pipeline()
        return _prepare_prompt_semantic_wav16k_profile_native(pipeline, raw_audio, raw_sr)

    def _prepare_ref_prompt_wav16k_for_worker(
        self,
        raw_audio: torch.Tensor,
        raw_sr: int,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        self._ensure_pipeline()
        if (
            self._get_runtime_prepare_ref_semantic_batch_worker() is None
            or str(os.environ.get("GPTSOVITS_PREPARE_REF_SUBMIT_PREPARED_WAV16K", "1")).strip().lower()
            in {"0", "false", "no", "off"}
        ):
            return None, {
                "prompt_semantic_cpu_prepare_wait_ms": 0.0,
                "prompt_semantic_cpu_prepare_slots": 0.0,
                "prompt_semantic_cpu_prepare_inflight_peak": 0.0,
                "prompt_semantic_cpu_prepare_ms": 0.0,
            }
        wav16k, cpu_prepare_ms, limiter_stats = self._prepare_prompt_semantic_wav16k_profile(raw_audio, raw_sr)
        return wav16k, {
            "prompt_semantic_cpu_prepare_wait_ms": float(limiter_stats.get("wait_ms", 0.0)),
            "prompt_semantic_cpu_prepare_slots": float(limiter_stats.get("slots", 0.0)),
            "prompt_semantic_cpu_prepare_inflight_peak": float(limiter_stats.get("peak_inflight", 0.0)),
            "prompt_semantic_cpu_prepare_ms": float(cpu_prepare_ms),
        }

    def _build_ref_prompt_semantic_runtime_exact_prewarm_profile(
        self,
        coordinator: Any,
        worker: Any,
        raw_audio: Any,
        raw_sr: int,
        *,
        wav16k: Any | None = None,
    ) -> dict[str, float]:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        profile = {
            "prompt_semantic_runtime_exact_prewarm_applied": 0.0,
            "prompt_semantic_runtime_exact_prewarm_ms": 0.0,
            "prompt_semantic_runtime_exact_prewarm_target_samples": 0.0,
            "prompt_semantic_runtime_exact_prewarm_batch_sizes": 0.0,
            "prompt_semantic_runtime_exact_prewarm_skipped_capacity": 0.0,
        }
        if (
            not getattr(coordinator, "ref_prompt_semantic_runtime_exact_prewarm_enabled", False)
            or int(getattr(coordinator, "ref_prompt_semantic_runtime_exact_prewarm_max_unique", 0) or 0) <= 0
        ):
            return profile
        estimate_fn = getattr(worker, "estimate_runtime_exact_prewarm_target_samples", None)
        run_fn = getattr(worker, "run_runtime_exact_prewarm", None)
        if not callable(estimate_fn) or not callable(run_fn):
            return profile
        target_samples = int(estimate_fn(raw_audio, int(raw_sr), wav16k=wav16k) or 0)
        if target_samples <= 0:
            return profile
        prewarm_lock = getattr(coordinator, "ref_prompt_semantic_runtime_exact_prewarm_lock", None)
        if prewarm_lock is None:
            return profile
        batch_sizes = tuple(
            int(value)
            for value in (
                getattr(coordinator, "ref_prompt_semantic_runtime_exact_prewarm_batch_sizes", ()) or ()
            )
        )
        if not batch_sizes:
            batch_sizes = GPTSoVITSPrepareRuntimeCoordinator._resolve_ref_prompt_semantic_runtime_exact_prewarm_batch_sizes(
                getattr(coordinator, "tts", None)
            )
        with prewarm_lock:
            prewarmed_samples = coordinator.ref_prompt_semantic_runtime_exact_prewarmed_samples
            inflight_samples = coordinator.ref_prompt_semantic_runtime_exact_prewarm_inflight_samples
            if target_samples in prewarmed_samples or target_samples in inflight_samples:
                return profile
            if len(prewarmed_samples) + len(inflight_samples) >= int(
                getattr(coordinator, "ref_prompt_semantic_runtime_exact_prewarm_max_unique", 0) or 0
            ):
                profile["prompt_semantic_runtime_exact_prewarm_skipped_capacity"] = 1.0
                profile["prompt_semantic_runtime_exact_prewarm_target_samples"] = float(target_samples)
                return profile
            inflight_samples.add(int(target_samples))
        try:
            profile = dict(
                run_fn(
                    raw_audio,
                    int(raw_sr),
                    wav16k=wav16k,
                    batch_sizes=list(batch_sizes),
                )
                or {}
            )
        except Exception:
            with prewarm_lock:
                coordinator.ref_prompt_semantic_runtime_exact_prewarm_inflight_samples.discard(int(target_samples))
            raise
        with prewarm_lock:
            coordinator.ref_prompt_semantic_runtime_exact_prewarm_inflight_samples.discard(int(target_samples))
            if float(profile.get("prompt_semantic_runtime_exact_prewarm_applied", 0.0)) > 0.0:
                coordinator.ref_prompt_semantic_runtime_exact_prewarmed_samples.add(int(target_samples))
                coordinator.ref_prompt_semantic_runtime_exact_prewarm_total += 1
                coordinator.ref_prompt_semantic_runtime_exact_prewarm_total_ms += float(
                    profile.get("prompt_semantic_runtime_exact_prewarm_ms", 0.0)
                )
                coordinator.ref_prompt_semantic_runtime_exact_prewarm_peak_ms = max(
                    float(getattr(coordinator, "ref_prompt_semantic_runtime_exact_prewarm_peak_ms", 0.0) or 0.0),
                    float(profile.get("prompt_semantic_runtime_exact_prewarm_ms", 0.0)),
                )
        return {
            "prompt_semantic_runtime_exact_prewarm_applied": float(
                profile.get("prompt_semantic_runtime_exact_prewarm_applied", 0.0)
            ),
            "prompt_semantic_runtime_exact_prewarm_ms": float(
                profile.get("prompt_semantic_runtime_exact_prewarm_ms", 0.0)
            ),
            "prompt_semantic_runtime_exact_prewarm_target_samples": float(
                profile.get("prompt_semantic_runtime_exact_prewarm_target_samples", float(target_samples))
            ),
            "prompt_semantic_runtime_exact_prewarm_batch_sizes": float(
                profile.get("prompt_semantic_runtime_exact_prewarm_batch_sizes", float(len(batch_sizes)))
            ),
            "prompt_semantic_runtime_exact_prewarm_skipped_capacity": float(
                profile.get("prompt_semantic_runtime_exact_prewarm_skipped_capacity", 0.0)
            ),
        }

    def _build_ref_prompt_semantic_worker_routing(
        self,
        coordinator: Any,
        worker: Any,
        raw_audio: Any,
        raw_sr: int,
        *,
        wav16k: Any | None = None,
    ) -> dict[str, Any]:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        bucket_index_fn = getattr(worker, "bucket_index_for_inputs", None)
        if not callable(bucket_index_fn):
            return {
                "bucket_index": None,
                "preferred_shard_index": None,
                "bucket_first_hit_serialized": False,
            }
        bucket_index = int(bucket_index_fn(raw_audio, int(raw_sr), wav16k=wav16k))
        preferred_shard_index = self._select_ref_prompt_semantic_worker_shard_index(
            coordinator,
            worker,
            bucket_index=bucket_index,
        )
        route = {
            "bucket_index": int(bucket_index),
            "preferred_shard_index": preferred_shard_index,
            "bucket_first_hit_serialized": False,
        }
        if (
            not getattr(coordinator, "ref_prompt_semantic_bucket_first_hit_serialization_enabled", False)
            or int(getattr(coordinator, "ref_prompt_semantic_bucket_first_hit_required_hits", 0) or 0) <= 0
        ):
            return route
        bucket_indices = tuple(
            int(value)
            for value in (
                getattr(coordinator, "ref_prompt_semantic_bucket_first_hit_bucket_indices", ()) or ()
            )
        )
        if bucket_indices and int(bucket_index) not in set(bucket_indices):
            return route
        lock = getattr(coordinator, "ref_prompt_semantic_bucket_first_hit_lock", None)
        if lock is None:
            return route
        with lock:
            state = coordinator.ref_prompt_semantic_bucket_first_hit_states.get(int(bucket_index))
            if state is not None and int(state.get("dispatched_hits", 0)) >= int(
                getattr(coordinator, "ref_prompt_semantic_bucket_first_hit_required_hits", 0) or 0
            ):
                return route
            if state is None:
                pick_first_hit_fn = getattr(worker, "pick_runtime_first_hit_shard_index", None)
                if callable(pick_first_hit_fn):
                    reserved_shard_index = int(pick_first_hit_fn())
                else:
                    reserved_shard_index = (
                        int(preferred_shard_index)
                        if preferred_shard_index is not None
                        else 0
                    )
                state = {
                    "reserved_shard_index": int(reserved_shard_index),
                    "dispatched_hits": 0,
                    "completed_hits": 0,
                }
                coordinator.ref_prompt_semantic_bucket_first_hit_states[int(bucket_index)] = state
            state["dispatched_hits"] = min(
                int(state.get("dispatched_hits", 0)) + 1,
                int(getattr(coordinator, "ref_prompt_semantic_bucket_first_hit_required_hits", 0) or 0),
            )
            route["preferred_shard_index"] = int(state.get("reserved_shard_index", 0))
            route["bucket_first_hit_serialized"] = True
        return route

    def _select_ref_prompt_semantic_worker_shard_index(
        self,
        coordinator: Any,
        worker: Any,
        *,
        bucket_index: int | None,
    ) -> int | None:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        snapshots_fn = getattr(worker, "runtime_routing_snapshots_for_bucket", None)
        if not callable(snapshots_fn):
            pick_shard_fn = getattr(worker, "pick_runtime_shard_index", None)
            return int(pick_shard_fn(bucket_index)) if callable(pick_shard_fn) else None
        snapshots_raw = list(snapshots_fn(bucket_index))
        if not snapshots_raw:
            return None
        snapshots = []
        for snapshot in snapshots_raw:
            normalized = {str(key): int(value) for key, value in dict(snapshot).items()}
            if "shard_index" not in normalized:
                continue
            snapshots.append(normalized)
        if not snapshots:
            return None
        if (
            bool(getattr(coordinator, "ref_prompt_semantic_bucket_aware_sharding", False))
            and bucket_index is not None
            and len(snapshots) > 1
        ):
            min_outstanding = min(int(snapshot.get("outstanding", 0)) for snapshot in snapshots)
            allowed_gap = int(getattr(coordinator, "ref_prompt_semantic_bucket_aware_max_outstanding_gap", 0) or 0)
            preferred = [
                snapshot
                for snapshot in snapshots
                if int(snapshot.get("mergeable_pending", 0)) > 0
                and int(snapshot.get("outstanding", 0)) <= (min_outstanding + allowed_gap)
            ]
            if preferred:
                best = min(
                    preferred,
                    key=lambda snapshot: (
                        -int(snapshot.get("mergeable_pending", 0)),
                        -int(snapshot.get("exact_pending", 0)),
                        int(snapshot.get("outstanding", 0)),
                        int(snapshot.get("outstanding_samples", 0)),
                        0 if int(snapshot.get("active_mergeable", 0)) > 0 else 1,
                        int(snapshot.get("shard_index", 0)),
                    ),
                )
                return int(best.get("shard_index", 0))
        fallback = min(
            snapshots,
            key=lambda snapshot: (
                int(snapshot.get("outstanding", snapshot.get("pending", 0))),
                int(snapshot.get("outstanding_samples", snapshot.get("pending_samples", 0))),
                int(snapshot.get("active_batch_size", 0)),
                int(snapshot.get("shard_index", 0)),
            ),
        )
        return int(fallback.get("shard_index", 0))

    def _mark_ref_prompt_semantic_worker_routing_completed(
        self,
        coordinator: Any,
        route: dict[str, Any] | None,
    ) -> None:
        if not route or not bool(route.get("bucket_first_hit_serialized", False)):
            return
        coordinator = self._coerce_prepare_coordinator(coordinator)
        bucket_index = route.get("bucket_index")
        if bucket_index is None:
            return
        lock = getattr(coordinator, "ref_prompt_semantic_bucket_first_hit_lock", None)
        if lock is None:
            return
        with lock:
            state = coordinator.ref_prompt_semantic_bucket_first_hit_states.get(int(bucket_index))
            if state is None:
                return
            completed_hits = int(state.get("completed_hits", 0)) + 1
            dispatched_hits = int(state.get("dispatched_hits", 0))
            required_hits = int(getattr(coordinator, "ref_prompt_semantic_bucket_first_hit_required_hits", 0) or 0)
            state["completed_hits"] = min(completed_hits, dispatched_hits, required_hits)

    @staticmethod
    def _prepare_run_profiled(fn: Any, submit_at: float, *args: Any) -> GPTSoVITSPrepareProfiledResult:
        started_at = time.perf_counter()
        result = fn(*args)
        finished_at = time.perf_counter()
        return GPTSoVITSPrepareProfiledResult(
            result=result,
            submit_at=float(submit_at),
            started_at=float(started_at),
            finished_at=float(finished_at),
        )

    async def _prepare_run_on_executor(self, executor: Any, fn: Any, *args: Any) -> GPTSoVITSPrepareProfiledResult:
        loop = asyncio.get_running_loop()
        submit_at = time.perf_counter()
        return await loop.run_in_executor(executor, self._prepare_run_profiled, fn, float(submit_at), *args)

    @staticmethod
    def _estimate_text_feature_run_ms(profile: dict[str, float]) -> float:
        return float(
            profile.get("bert_wait_ms", 0.0)
            + profile.get("bert_tokenize_ms", 0.0)
            + profile.get("bert_forward_ms", 0.0)
            + profile.get("bert_scatter_ms", 0.0)
        )

    @staticmethod
    def _accumulate_profile_metric(
        profile: dict[str, float] | None,
        key: str,
        amount: float,
    ) -> None:
        if profile is None:
            return
        profile[key] = float(profile.get(key, 0.0)) + float(amount)

    @staticmethod
    def _update_profile_metric_peak(
        profile: dict[str, float] | None,
        key: str,
        value: float,
    ) -> None:
        if profile is None:
            return
        profile[key] = max(float(profile.get(key, 0.0)), float(value))

    @classmethod
    def _merge_bert_worker_profile(
        cls,
        profile: dict[str, float] | None,
        worker_profile: dict[str, float],
    ) -> None:
        cls._accumulate_profile_metric(profile, "bert_wait_ms", worker_profile.get("bert_wait_ms", 0.0))
        cls._accumulate_profile_metric(
            profile,
            "bert_admission_wait_ms",
            worker_profile.get("bert_admission_wait_ms", 0.0),
        )
        cls._accumulate_profile_metric(
            profile,
            "bert_queue_wait_ms",
            worker_profile.get("bert_queue_wait_ms", 0.0),
        )
        cls._accumulate_profile_metric(
            profile,
            "bert_worker_queue_wait_ms",
            worker_profile.get("bert_worker_queue_wait_ms", 0.0),
        )
        cls._accumulate_profile_metric(
            profile,
            "bert_batch_collect_wait_ms",
            worker_profile.get("bert_batch_collect_wait_ms", 0.0),
        )
        cls._accumulate_profile_metric(
            profile,
            "bert_batch_dispatch_delay_ms",
            worker_profile.get("bert_batch_dispatch_delay_ms", 0.0),
        )
        cls._accumulate_profile_metric(
            profile,
            "bert_forward_ms",
            worker_profile.get("bert_forward_ms", 0.0),
        )
        cls._accumulate_profile_metric(
            profile,
            "bert_tokenize_ms",
            worker_profile.get("bert_tokenize_ms", 0.0),
        )
        cls._accumulate_profile_metric(
            profile,
            "bert_scatter_ms",
            worker_profile.get("bert_scatter_ms", 0.0),
        )
        cls._accumulate_profile_metric(profile, "bert_calls", worker_profile.get("bert_calls", 1.0))
        cls._update_profile_metric_peak(
            profile,
            "bert_stage_inflight_peak",
            worker_profile.get("bert_stage_inflight_peak", 0.0),
        )
        cls._update_profile_metric_peak(
            profile,
            "bert_batch_size_peak",
            worker_profile.get("bert_batch_size", 0.0),
        )
        cls._update_profile_metric_peak(
            profile,
            "bert_batch_tokens_peak",
            worker_profile.get("bert_batch_tokens", 0.0),
        )
        cls._update_profile_metric_peak(
            profile,
            "bert_pending_depth_on_enqueue_peak",
            worker_profile.get("bert_pending_depth_on_enqueue", 0.0),
        )
        cls._update_profile_metric_peak(
            profile,
            "bert_pending_depth_on_collect_peak",
            worker_profile.get("bert_pending_depth_on_collect", 0.0),
        )
        cls._update_profile_metric_peak(
            profile,
            "bert_high_pressure_mode_peak",
            worker_profile.get("bert_high_pressure_mode", 0.0),
        )
        if profile is not None:
            profile["bert_stage_slots"] = float(worker_profile.get("bert_stage_slots", 0.0))
            profile["bert_batch_window_ms"] = float(worker_profile.get("bert_batch_window_ms", 0.0))

    @staticmethod
    def _mark_bert_submit_offsets(profile: dict[str, float] | None) -> None:
        if profile is None:
            return
        branch_start_ts = float(profile.get("_branch_start_ts", 0.0) or 0.0)
        if branch_start_ts <= 0.0:
            return
        offset_ms = max(0.0, (time.perf_counter() - branch_start_ts) * 1000.0)
        if "bert_submit_offset_first_ms" not in profile:
            profile["bert_submit_offset_first_ms"] = float(offset_ms)
        profile["bert_submit_offset_last_ms"] = float(offset_ms)

    @staticmethod
    def _make_text_features(
        phones: Any,
        bert_features: torch.Tensor,
        norm_text: str,
        *,
        profile: dict[str, float] | None = None,
        total_ms: float = 0.0,
        cpu_preprocess_ms: float = 0.0,
    ) -> GPTSoVITSTextFeatures:
        return GPTSoVITSTextFeatures(
            phones=[int(item) for item in list(phones or [])],
            bert_features=bert_features,
            norm_text=str(norm_text),
            profile=dict(profile or {}),
            total_ms=float(total_ms),
            cpu_preprocess_ms=float(cpu_preprocess_ms),
        )

    def _build_empty_text_features_like(self, reference: Any | None = None) -> GPTSoVITSTextFeatures:
        feature_dim = 1024
        dtype = None
        if reference is not None:
            try:
                feature_dim = int(reference.bert_features.shape[0])
                dtype = reference.bert_features.dtype
            except Exception:
                pass
        return GPTSoVITSTextFeatures(
            phones=[],
            bert_features=torch.empty(
                (int(feature_dim), 0),
                dtype=((dtype if dtype is not None else None) or torch.float32),
            ),
            norm_text="",
            profile={"cpu_preprocess_ms": 0.0, "bert_total_ms": 0.0},
            total_ms=0.0,
            cpu_preprocess_ms=0.0,
        )

    @staticmethod
    def _empty_bert_feature(phone_count: int, device: Any) -> torch.Tensor:
        return torch.zeros((1024, max(0, int(phone_count))), dtype=torch.float32, device=device)

    def _compute_sync_bert_feature(
        self,
        text: str,
        word2ph: list[int],
        profile: dict[str, float] | None = None,
    ) -> torch.Tensor:
        self._ensure_pipeline()
        worker = self._get_runtime_prepare_bert_batch_worker()
        self._mark_bert_submit_offsets(profile)
        if worker is not None:
            feature, worker_profile = worker.submit(text, list(word2ph))
            self._merge_bert_worker_profile(profile, dict(worker_profile))
            return feature.to(getattr(self._get_runtime_configs(), "device", "cpu"))

        tokenizer = self._get_runtime_bert_tokenizer()
        bert_model = self._get_runtime_bert_model()
        device = getattr(self._get_runtime_configs(), "device", "cpu")
        limiter = self._get_runtime_prepare_bert_stage_limiter()
        limiter_stats = {"wait_ms": 0.0, "inflight": 1, "peak_inflight": 1, "slots": 0}

        def _run_bert_forward() -> torch.Tensor:
            inputs = tokenizer(text, return_tensors="pt")
            tokenize_ms = 0.0
            h2d_start = time.perf_counter()
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            tokenize_ms += (time.perf_counter() - h2d_start) * 1000.0
            forward_start = time.perf_counter()
            with torch.no_grad():
                result = bert_model(**inputs, output_hidden_states=True)
                hidden = torch.cat(result["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
            forward_ms = (time.perf_counter() - forward_start) * 1000.0
            self._accumulate_profile_metric(profile, "bert_tokenize_ms", tokenize_ms)
            self._accumulate_profile_metric(profile, "bert_forward_ms", forward_ms)
            return hidden

        if limiter is None:
            hidden = _run_bert_forward()
        else:
            with limiter.enter() as limiter_stats:
                hidden = _run_bert_forward()
        self._accumulate_profile_metric(profile, "bert_wait_ms", limiter_stats.get("wait_ms", 0.0))
        self._accumulate_profile_metric(profile, "bert_calls", 1.0)
        self._update_profile_metric_peak(profile, "bert_stage_inflight_peak", limiter_stats.get("peak_inflight", 0.0))
        if profile is not None:
            profile["bert_stage_slots"] = float(limiter_stats.get("slots", 0.0))
        if len(word2ph) != len(text):
            raise ValueError("中文文本 word2ph 与文本长度不一致，无法提取 BERT 特征")
        phone_level_feature = [
            hidden[index].repeat(int(phone_count), 1)
            for index, phone_count in enumerate(word2ph)
            if int(phone_count) > 0
        ]
        if not phone_level_feature:
            return self._empty_bert_feature(0, device)
        return torch.cat(phone_level_feature, dim=0).T.to(device)

    def _build_segment_bert_feature(
        self,
        segment: Any,
        profile: dict[str, float] | None = None,
    ) -> torch.Tensor:
        device = getattr(self._get_runtime_configs(), "device", "cpu")
        phones = [int(item) for item in list(getattr(segment, "phones", []) or [])]
        segment_language = str(getattr(segment, "language", "") or "").replace("all_", "")
        if segment_language != "zh":
            return self._empty_bert_feature(len(phones), device)
        word2ph = getattr(segment, "word2ph", None)
        if word2ph is None:
            raise ValueError("中文文本缺少 word2ph，无法提取 BERT 特征")
        norm_text = str(getattr(segment, "norm_text", "") or "")
        return self._compute_sync_bert_feature(norm_text, list(word2ph), profile=profile)

    def _prepare_text_feature_segment_jobs(
        self,
        prepared_segments: Any,
        profile: dict[str, float] | None,
    ) -> dict[str, Any]:
        self._ensure_pipeline()
        worker = self._get_runtime_prepare_bert_batch_worker()
        if worker is None:
            raise RuntimeError("GPT-SoVITS BERT batch worker 未初始化")
        device = getattr(self._get_runtime_configs(), "device", "cpu")
        phones_list: list[list[int]] = []
        bert_list: list[torch.Tensor | None] = []
        norm_text_list: list[str] = []
        pending_items: list[tuple[list[torch.Tensor | None], int, dict[str, float] | None, Any, Any]] = []

        for segment in prepared_segments or []:
            phones = [int(item) for item in list(getattr(segment, "phones", []) or [])]
            norm_text = str(getattr(segment, "norm_text", "") or "")
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            segment_language = str(getattr(segment, "language", "") or "").replace("all_", "")
            if segment_language != "zh":
                bert_list.append(self._empty_bert_feature(len(phones), device))
                continue

            word2ph = getattr(segment, "word2ph", None)
            if word2ph is None:
                raise ValueError("中文文本缺少 word2ph，无法提取 BERT 特征")
            self._mark_bert_submit_offsets(profile)
            bert_list.append(None)
            pending_items.append(
                (
                    bert_list,
                    len(bert_list) - 1,
                    profile,
                    device,
                    worker.submit_async(norm_text, list(word2ph)),
                )
            )

        return {
            "device": device,
            "phones_list": phones_list,
            "bert_list": bert_list,
            "norm_text_list": norm_text_list,
            "pending_items": pending_items,
        }

    async def _finalize_text_feature_segment_jobs(
        self,
        segment_jobs: dict[str, Any],
    ) -> tuple[list[int], torch.Tensor, str]:
        pending_items = list(segment_jobs.get("pending_items", []) or [])
        if pending_items:
            pending_results = await asyncio.gather(*[future for _, _, _, _, future in pending_items])
            for (bert_list, bert_index, profile, device, _), (feature, worker_profile) in zip(
                pending_items,
                pending_results,
            ):
                self._merge_bert_worker_profile(profile, dict(worker_profile))
                bert_list[bert_index] = feature.to(device)

        bert_features = [feature for feature in segment_jobs.get("bert_list", []) if feature is not None]
        if bert_features:
            bert = torch.cat(bert_features, dim=1)
        else:
            bert = self._empty_bert_feature(0, segment_jobs.get("device", "cpu"))
        phones = sum(segment_jobs.get("phones_list", []), [])
        norm_text = "".join(segment_jobs.get("norm_text_list", []))
        return phones, bert, norm_text

    def _build_text_features(
        self,
        prepared_segments: Any,
        language: str | None,
        cpu_run_ms: float,
        base_profile: dict[str, float] | None = None,
    ) -> GPTSoVITSTextFeatures:
        del language
        profile: dict[str, float] = dict(base_profile or {})
        profile["cpu_preprocess_ms"] = float(cpu_run_ms)
        branch_start = time.perf_counter()
        resolved_segments, g2pw_profile = self._resolve_g2pw_segments(prepared_segments)
        self._merge_g2pw_profile(profile, g2pw_profile)
        phones_list: list[list[int]] = []
        bert_list: list[torch.Tensor] = []
        norm_text_list: list[str] = []
        for segment in resolved_segments:
            phones = [int(item) for item in list(getattr(segment, "phones", []) or [])]
            norm_text = str(getattr(segment, "norm_text", "") or "")
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(self._build_segment_bert_feature(segment, profile=profile))
        if bert_list:
            bert_features = torch.cat(bert_list, dim=1)
        else:
            bert_features = self._empty_bert_feature(0, "cpu")
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)
        total_ms = float(cpu_run_ms + (time.perf_counter() - branch_start) * 1000.0)
        profile["bert_total_ms"] = max(0.0, total_ms - float(cpu_run_ms))
        return self._make_text_features(
            phones,
            bert_features,
            norm_text,
            profile=profile,
            total_ms=total_ms,
            cpu_preprocess_ms=cpu_run_ms,
        )

    def _extract_prompt_semantic_profile_from_prepared_wav16k(
        self,
        wav16k: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        configs = self._get_runtime_configs()
        cnhuhbert_model = self._get_runtime_cnhuhbert_model()
        vits_model = self._get_runtime_vits_model()
        with torch.no_grad():
            h2d_start = time.perf_counter()
            wav16k = wav16k.to(configs.device)
            if bool(getattr(configs, "is_half", False)):
                wav16k = wav16k.half()
            h2d_ms = (time.perf_counter() - h2d_start) * 1000.0

            ssl_start = time.perf_counter()
            hubert_feature = cnhuhbert_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            ssl_forward_ms = (time.perf_counter() - ssl_start) * 1000.0

            latent_start = time.perf_counter()
            codes = vits_model.extract_latent(hubert_feature)
            extract_latent_ms = (time.perf_counter() - latent_start) * 1000.0
            prompt_semantic = codes[0, 0].to(configs.device)
        profile = {
            "prompt_semantic_h2d_ms": float(h2d_ms),
            "prompt_semantic_ssl_forward_ms": float(ssl_forward_ms),
            "prompt_semantic_hidden_length_ms": 0.0,
            "prompt_semantic_extract_latent_ms": float(extract_latent_ms),
            "prompt_semantic_forward_ms": float(h2d_ms + ssl_forward_ms + extract_latent_ms),
        }
        return prompt_semantic, profile

    def _build_ref_prompt_semantic_from_raw(self, raw_audio: Any, raw_sr: int) -> GPTSoVITSRefAudioBundle:
        pipeline = self._ensure_pipeline()
        load_profile = {"audio_load_ms": 0.0}
        ref_worker = self._get_runtime_prepare_ref_semantic_batch_worker()
        if ref_worker is not None:
            wav16k, local_cpu_prepare_profile = self._prepare_ref_prompt_wav16k_for_worker(raw_audio, raw_sr)
            coordinator = self._ensure_prepare_coordinator()
            runtime_exact_prewarm_profile = self._build_ref_prompt_semantic_runtime_exact_prewarm_profile(
                coordinator,
                ref_worker,
                raw_audio,
                raw_sr,
                wav16k=wav16k,
            )
            route = self._build_ref_prompt_semantic_worker_routing(
                coordinator,
                ref_worker,
                raw_audio,
                raw_sr,
                wav16k=wav16k,
            )
            try:
                prompt_semantic, worker_profile = ref_worker.submit(
                    raw_audio,
                    raw_sr,
                    wav16k=wav16k,
                    runtime_exact_prewarm_profile=runtime_exact_prewarm_profile,
                    bucket_index=route["bucket_index"],
                    preferred_shard_index=route["preferred_shard_index"],
                    bucket_first_hit_serialized=route["bucket_first_hit_serialized"],
                )
            finally:
                self._mark_ref_prompt_semantic_worker_routing_completed(coordinator, route)
            return GPTSoVITSRefAudioBundle(
                prompt_semantic=prompt_semantic,
                raw_audio=raw_audio,
                raw_sr=int(raw_sr),
                profile={
                    **load_profile,
                    "audio_stage_wait_ms": float(worker_profile.get("prompt_semantic_wait_ms", 0.0)),
                    "audio_stage_slots": float(worker_profile.get("prompt_semantic_stage_slots", 0.0)),
                    "audio_stage_inflight_peak": float(
                        worker_profile.get("prompt_semantic_stage_inflight_peak", 0.0)
                    ),
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
            )

        wav16k, cpu_prepare_ms, limiter_stats = self._prepare_prompt_semantic_wav16k_profile(raw_audio, raw_sr)
        ref_stage_limiter = self._get_runtime_prepare_ref_semantic_stage_limiter()
        if ref_stage_limiter is None:
            stage_stats = {"wait_ms": 0.0, "slots": 0.0, "peak_inflight": 0.0}
            prompt_semantic, runtime_profile = self._extract_prompt_semantic_profile_from_prepared_wav16k(wav16k)
        else:
            with ref_stage_limiter.enter() as stage_stats:
                prompt_semantic, runtime_profile = self._extract_prompt_semantic_profile_from_prepared_wav16k(wav16k)
        return GPTSoVITSRefAudioBundle(
            prompt_semantic=prompt_semantic,
            raw_audio=raw_audio,
            raw_sr=int(raw_sr),
            profile={
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
                    cpu_prepare_ms
                    + runtime_profile.get("prompt_semantic_forward_ms", 0.0)
                    + stage_stats.get("wait_ms", 0.0)
                ),
            },
        )

    @staticmethod
    def _build_ref_prompt_semantic_worker_profiled_result(
        *,
        submit_at: float,
        started_at: float,
        finished_at: float,
        prompt_semantic: torch.Tensor,
        raw_audio: Any,
        raw_sr: int,
        load_queue_ms: float,
        load_ms: float,
        cpu_prepare_wait_ms: float,
        cpu_prepare_slots: float,
        cpu_prepare_inflight_peak: float,
        preload_cpu_prepare_ms: float,
        prompt_semantic_profile: dict[str, float],
        limiter_snapshot: dict[str, float] | None = None,
    ) -> GPTSoVITSPrepareProfiledResult:
        limiter_snapshot = dict(limiter_snapshot or {})
        prompt_semantic_ms = (
            float(prompt_semantic_profile.get("prompt_semantic_cpu_prepare_ms", 0.0))
            + float(prompt_semantic_profile.get("prompt_semantic_forward_ms", 0.0))
            + float(prompt_semantic_profile.get("prompt_semantic_scatter_ms", 0.0))
        )
        result = GPTSoVITSRefAudioBundle(
            prompt_semantic=prompt_semantic,
            raw_audio=raw_audio,
            raw_sr=int(raw_sr),
            profile={
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
                "prompt_semantic_cpu_prepare_ms": float(prompt_semantic_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)),
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
                "prompt_semantic_batch_samples": float(prompt_semantic_profile.get("prompt_semantic_batch_samples", 0.0)),
                "prompt_semantic_padded_batch_samples": float(
                    prompt_semantic_profile.get("prompt_semantic_padded_batch_samples", 0.0)
                ),
                "prompt_semantic_batch_pad_ratio": float(
                    prompt_semantic_profile.get("prompt_semantic_batch_pad_ratio", 0.0)
                ),
                "prompt_semantic_ssl_skip_attention_mask": float(
                    prompt_semantic_profile.get("prompt_semantic_ssl_skip_attention_mask", 0.0)
                ),
                "prompt_semantic_pool_workers": float(prompt_semantic_profile.get("prompt_semantic_pool_workers", 0.0)),
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
                "prompt_semantic_shard_index": float(prompt_semantic_profile.get("prompt_semantic_shard_index", 0.0)),
                "bundle_total_ms": float(load_queue_ms + load_ms + preload_cpu_prepare_ms + prompt_semantic_ms),
            },
        )
        return GPTSoVITSPrepareProfiledResult(
            result=result,
            submit_at=float(submit_at),
            started_at=float(started_at),
            finished_at=float(finished_at),
        )

    async def _run_text_cpu_stage(
        self,
        coordinator: Any,
        text: str,
        language: str,
        text_split_method: str = "cut1",
    ) -> GPTSoVITSPrepareProfiledResult:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        await coordinator.text_cpu_gate.acquire()
        if text in [None, ""]:
            try:
                submit_at = time.perf_counter()
                return GPTSoVITSPrepareProfiledResult(
                    result=[],
                    submit_at=submit_at,
                    started_at=submit_at,
                    finished_at=submit_at,
                )
            finally:
                coordinator.text_cpu_gate.release()

        self._ensure_pipeline()
        text_cpu_worker = self._get_runtime_prepare_text_cpu_worker()
        executor = self._get_runtime_prepare_text_cpu_executor()
        try:
            if text_cpu_worker is not None:
                submit_at = time.perf_counter()
                result, worker_profile = await text_cpu_worker.submit_async(text, language, text_split_method)
                return self._build_text_cpu_profiled_result(submit_at, result, dict(worker_profile))
            if executor is None:
                submit_at = time.perf_counter()
                return self._prepare_run_profiled(self._prepare_text_cpu, submit_at, text, language, text_split_method)
            return await self._prepare_run_on_executor(
                executor,
                self._prepare_text_cpu,
                text,
                language,
                text_split_method,
            )
        finally:
            coordinator.text_cpu_gate.release()

    async def _run_text_cpu_stage_pair(
        self,
        coordinator: Any,
        prompt_text: str,
        prompt_lang: str,
        text: str,
        text_lang: str,
        text_split_method: str = "cut1",
    ) -> tuple[GPTSoVITSPrepareProfiledResult, GPTSoVITSPrepareProfiledResult]:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        self._ensure_pipeline()
        text_cpu_worker = self._get_runtime_prepare_text_cpu_worker()
        if (
            text_cpu_worker is None
            or not hasattr(text_cpu_worker, "submit_many_async")
            or int(getattr(coordinator.text_cpu_gate, "max_inflight", 0)) > 0
        ):
            prompt_cpu_task = asyncio.create_task(
                self._run_text_cpu_stage(coordinator, prompt_text, prompt_lang, text_split_method)
            )
            target_cpu_task = asyncio.create_task(
                self._run_text_cpu_stage(coordinator, text, text_lang, text_split_method)
            )
            return await asyncio.gather(prompt_cpu_task, target_cpu_task)

        items = []
        item_indices = []
        profiled_results: list[GPTSoVITSPrepareProfiledResult | None] = [None, None]
        for index, (item_text, item_lang) in enumerate(((prompt_text, prompt_lang), (text, text_lang))):
            if item_text in [None, ""]:
                submit_at = time.perf_counter()
                profiled_results[index] = GPTSoVITSPrepareProfiledResult(
                    result=[],
                    submit_at=submit_at,
                    started_at=submit_at,
                    finished_at=submit_at,
                )
                continue
            items.append((item_text, item_lang, text_split_method))
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

    async def _prepare_cpu_stage_async(
        self,
        coordinator: Any,
        spec: GPTSoVITSRequestSpec,
        *,
        prepare_submit_at: float,
    ) -> Any:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        admission_start = time.perf_counter()
        admission_stats = await coordinator.acquire_prepare_admission()
        prepare_admission_wait_ms = max(
            float(admission_stats.get("wait_ms", 0.0)),
            (time.perf_counter() - admission_start) * 1000.0,
        )
        current_inflight, peak_inflight = coordinator.mark_prepare_enter()
        prepare_start = time.perf_counter()
        prompt_text = self._normalize_prepare_sentence(spec.prompt_text, spec.prompt_lang)
        text = spec.text.strip("\n")
        try:
            prompt_cpu_profiled, target_cpu_profiled = await self._run_text_cpu_stage_pair(
                coordinator,
                prompt_text,
                spec.prompt_lang,
                text,
                spec.text_lang,
                spec.text_split_method,
            )
            return GPTSoVITSNativePreparedCpuStage(
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
            self._release_prepare_split_stage_slot(coordinator)
            raise

    async def _prepare_cpu_stage_with_shared_prompt_async(
        self,
        coordinator: Any,
        spec: GPTSoVITSRequestSpec,
        *,
        prepare_submit_at: float,
        prompt_text: str,
        prompt_cpu_profiled: GPTSoVITSPrepareProfiledResult,
    ) -> GPTSoVITSNativePreparedCpuStage:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        admission_start = time.perf_counter()
        admission_stats = await coordinator.acquire_prepare_admission()
        prepare_admission_wait_ms = max(
            float(admission_stats.get("wait_ms", 0.0)),
            (time.perf_counter() - admission_start) * 1000.0,
        )
        current_inflight, peak_inflight = coordinator.mark_prepare_enter()
        prepare_start = time.perf_counter()
        text = spec.text.strip("\n")
        try:
            target_cpu_profiled = await self._run_text_cpu_stage(
                coordinator,
                text,
                spec.text_lang,
                spec.text_split_method,
            )
            return GPTSoVITSNativePreparedCpuStage(
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
            self._release_prepare_split_stage_slot(coordinator)
            raise

    def prepare_request_cpu_stage(
        self,
        request: dict[str, Any],
        *,
        request_id: str | None = None,
        preload_ref_audio: bool = True,
    ) -> GPTSoVITSPreparedCpuStage:
        spec = self._build_scheduler_request_spec(request, request_id=request_id)
        return self.prepare_request_spec_cpu_stage(spec, preload_ref_audio=preload_ref_audio)

    def prepare_request_spec_cpu_stage(
        self,
        spec: GPTSoVITSRequestSpec,
        *,
        preload_ref_audio: bool = True,
    ) -> GPTSoVITSPreparedCpuStage:
        coordinator = self._ensure_prepare_coordinator()
        ref_audio_prepare_future = None
        submit_at = time.perf_counter()
        if preload_ref_audio:
            ref_audio_prepare_future = self.preload_ref_audio_asset(str(spec.ref_audio_path), submit_at=submit_at)
        cpu_stage = self._run_awaitable_sync(
            self._prepare_cpu_stage_async(coordinator, spec, prepare_submit_at=submit_at)
        )
        return GPTSoVITSPreparedCpuStage(
            request_id=str(spec.request_id),
            spec=spec,
            prepare_submit_at=float(cpu_stage.prepare_submit_at),
            prepare_start=float(cpu_stage.prepare_start),
            prompt_text=str(cpu_stage.prompt_text),
            text=str(cpu_stage.text),
            prepare_admission_wait_ms=float(cpu_stage.prepare_admission_wait_ms),
            current_inflight=int(cpu_stage.current_inflight),
            peak_inflight=int(cpu_stage.peak_inflight),
            prompt_cpu_profiled=cpu_stage.prompt_cpu_profiled,
            target_cpu_profiled=cpu_stage.target_cpu_profiled,
            ref_audio_prepare_future=ref_audio_prepare_future,
        )

    def prepare_request_cpu_stages(
        self,
        requests: list[dict[str, Any]],
    ) -> list[GPTSoVITSPreparedCpuStage]:
        if not requests:
            return []
        specs = [
            self._build_scheduler_request_spec(
                request,
                request_id=str(request.get("engine_request_id") or f"gpt_sovits_{index}"),
            )
            for index, request in enumerate(requests)
        ]
        return self.prepare_request_spec_cpu_stages(specs)

    def prepare_request_spec_cpu_stages(
        self,
        specs: list[GPTSoVITSRequestSpec],
    ) -> list[GPTSoVITSPreparedCpuStage]:
        if not specs:
            return []
        coordinator = self._ensure_prepare_coordinator()
        submit_ats: list[float] = []
        ref_audio_prepare_futures: list[Any | None] = []
        for spec in specs:
            submit_at = time.perf_counter()
            submit_ats.append(submit_at)
            ref_audio_prepare_futures.append(
                self.preload_ref_audio_asset(str(spec.ref_audio_path), submit_at=submit_at)
            )

        async def _gather_cpu_stages():
            return await asyncio.gather(
                *[
                    self._prepare_cpu_stage_async(coordinator, spec, prepare_submit_at=submit_at)
                    for spec, submit_at in zip(specs, submit_ats)
                ],
                return_exceptions=True,
            )

        cpu_stage_results = self._run_awaitable_sync(_gather_cpu_stages())
        if len(cpu_stage_results) != len(specs):
            raise ValueError("GPT-SoVITS batch prepare CPU stage count mismatch")
        prepared_cpu_stages: list[GPTSoVITSPreparedCpuStage] = []
        for spec, cpu_stage, ref_audio_prepare_future in zip(
            specs,
            cpu_stage_results,
            ref_audio_prepare_futures,
        ):
            if isinstance(cpu_stage, Exception):
                raise cpu_stage
            prepared_cpu_stages.append(
                GPTSoVITSPreparedCpuStage(
                    request_id=str(spec.request_id),
                    spec=spec,
                    prepare_submit_at=float(cpu_stage.prepare_submit_at),
                    prepare_start=float(cpu_stage.prepare_start),
                    prompt_text=str(cpu_stage.prompt_text),
                    text=str(cpu_stage.text),
                    prepare_admission_wait_ms=float(cpu_stage.prepare_admission_wait_ms),
                    current_inflight=int(cpu_stage.current_inflight),
                    peak_inflight=int(cpu_stage.peak_inflight),
                    prompt_cpu_profiled=cpu_stage.prompt_cpu_profiled,
                    target_cpu_profiled=cpu_stage.target_cpu_profiled,
                    ref_audio_prepare_future=ref_audio_prepare_future,
                )
            )
        return prepared_cpu_stages

    @staticmethod
    def _merge_ref_spec_profiled_result(
        profiled: GPTSoVITSPrepareProfiledResult,
    ) -> GPTSoVITSPrepareRefSpecResult:
        refer_spec, profile = profiled.result
        refer_spec_native = GPTSoVITSRuntime._coerce_refer_spec(refer_spec)
        merged_profile = dict(profile)
        merged_profile["ref_spec_wait_ms"] = float(profiled.queue_ms)
        merged_profile["ref_spec_ms"] = float(profiled.run_ms)
        return GPTSoVITSPrepareRefSpecResult(
            refer_spec=refer_spec_native,
            profile=merged_profile,
        )

    @staticmethod
    def _build_shared_prepare_phase_profile(
        *,
        prompt_profiled: GPTSoVITSPrepareProfiledResult,
        target_profiled: GPTSoVITSPrepareProfiledResult,
        shared_profiled: GPTSoVITSPrepareProfiledResult,
        shared_ref_spec_result: GPTSoVITSPrepareRefSpecResult | None,
        phase_kind: str,
    ) -> dict[str, float]:
        phase_wall_ms = max(
            float(prompt_profiled.run_ms),
            float(target_profiled.run_ms),
            float(shared_profiled.run_ms),
        )
        if phase_kind == "audio" and shared_ref_spec_result is not None:
            phase_wall_ms = max(
                phase_wall_ms,
                float(shared_ref_spec_result.profile.get("ref_spec_ms", 0.0)),
            )
        return {
            f"engine_prepare_{phase_kind}_phase_mode": 2.0,
            f"engine_prepare_{phase_kind}_phase_wall_ms": float(phase_wall_ms),
            f"engine_prepare_{phase_kind}_phase_batch_size": 1.0,
            "engine_prepare_shared_prompt_ref_enabled": 1.0,
        }

    def _finalize_shared_prepare_results(
        self,
        coordinator: Any,
        outputs: list[tuple[T2SRequestState, float, float] | Exception | None],
        cpu_stages: list[GPTSoVITSNativePreparedCpuStage | Exception | None],
    ) -> list[tuple[T2SRequestState, float, float] | Exception]:
        for index, cpu_stage in enumerate(cpu_stages):
            if not isinstance(cpu_stage, GPTSoVITSNativePreparedCpuStage):
                continue
            if outputs[index] is None:
                outputs[index] = RuntimeError("shared prepare result missing")
            self._release_prepare_split_stage_slot(coordinator)
        return [item if item is not None else RuntimeError("shared prepare result missing") for item in outputs]

    async def _prepare_direct_shared_segments_async(
        self,
        coordinator: Any,
        specs: list[GPTSoVITSRequestSpec],
    ) -> list[tuple[T2SRequestState, float, float] | Exception]:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        if not specs:
            return []

        shared_spec = specs[0]
        shared_prompt_text = self._normalize_prepare_sentence(shared_spec.prompt_text, shared_spec.prompt_lang)
        shared_prompt_cpu_profiled = await self._run_text_cpu_stage(
            coordinator,
            shared_prompt_text,
            shared_spec.prompt_lang,
        )
        cpu_stage_results = await asyncio.gather(
            *[
                self._prepare_cpu_stage_with_shared_prompt_async(
                    coordinator,
                    spec,
                    prepare_submit_at=time.perf_counter(),
                    prompt_text=shared_prompt_text,
                    prompt_cpu_profiled=shared_prompt_cpu_profiled,
                )
                for spec in specs
            ],
            return_exceptions=True,
        )
        outputs: list[tuple[T2SRequestState, float, float] | Exception | None] = [None] * len(specs)
        runnable: list[tuple[int, GPTSoVITSNativePreparedCpuStage]] = []
        for index, cpu_stage in enumerate(cpu_stage_results):
            if isinstance(cpu_stage, Exception):
                outputs[index] = cpu_stage
                continue
            runnable.append((index, cpu_stage))

        if not runnable:
            return self._finalize_shared_prepare_results(coordinator, outputs, cpu_stage_results)

        try:
            shared_prompt_g2pw_profiled = await self._run_g2pw_stage(coordinator, shared_prompt_cpu_profiled.result)
            shared_prompt_feature_profiled = await self._run_text_feature_stage(
                coordinator,
                shared_prompt_g2pw_profiled.result,
                shared_spec.prompt_lang,
                shared_prompt_cpu_profiled.run_ms,
                base_profile=dict(shared_prompt_g2pw_profiled.profile or {}),
            )
            shared_ref_audio_future = self.preload_ref_audio_asset(
                str(shared_spec.ref_audio_path),
                submit_at=time.perf_counter(),
            )
            shared_ref_audio_profiled = await self._run_ref_prompt_semantic_stage(
                coordinator,
                str(shared_spec.ref_audio_path),
                prepared_asset_future=shared_ref_audio_future,
            )
            shared_ref_spec_profiled = await self._run_ref_spec_stage(
                coordinator,
                shared_ref_audio_profiled.result.raw_audio,
                int(shared_ref_audio_profiled.result.raw_sr),
            )
            shared_ref_spec_result = self._merge_ref_spec_profiled_result(shared_ref_spec_profiled)
        except Exception as exc:
            for index, _ in runnable:
                outputs[index] = exc
            return self._finalize_shared_prepare_results(coordinator, outputs, cpu_stage_results)

        target_g2pw_results = await asyncio.gather(
            *[
                self._run_g2pw_stage(coordinator, cpu_stage.target_cpu_profiled.result)
                for _, cpu_stage in runnable
            ],
            return_exceptions=True,
        )
        target_feature_tasks: list[asyncio.Task[Any] | Exception] = []
        for target_g2pw_profiled, (_, cpu_stage) in zip(target_g2pw_results, runnable):
            if isinstance(target_g2pw_profiled, Exception):
                target_feature_tasks.append(target_g2pw_profiled)
                continue
            target_feature_tasks.append(
                asyncio.create_task(
                    self._run_text_feature_stage(
                        coordinator,
                        target_g2pw_profiled.result,
                        cpu_stage.spec.text_lang,
                        cpu_stage.target_cpu_profiled.run_ms,
                        base_profile=dict(target_g2pw_profiled.profile or {}),
                    )
                )
            )
        gathered_target_features = (
            list(
                await asyncio.gather(
                    *[
                        item
                        for item in target_feature_tasks
                        if not isinstance(item, Exception)
                    ],
                    return_exceptions=True,
                )
            )
            if any(not isinstance(item, Exception) for item in target_feature_tasks)
            else []
        )
        target_feature_results: list[GPTSoVITSPrepareProfiledResult | Exception] = []
        gather_index = 0
        for item in target_feature_tasks:
            if isinstance(item, Exception):
                target_feature_results.append(item)
                continue
            target_feature_results.append(gathered_target_features[gather_index])
            gather_index += 1

        for (index, cpu_stage), target_g2pw_profiled, target_feature_profiled in zip(
            runnable,
            target_g2pw_results,
            target_feature_results,
        ):
            if isinstance(target_g2pw_profiled, Exception):
                outputs[index] = target_g2pw_profiled
                self._release_prepare_split_stage_slot(coordinator)
                continue
            if isinstance(target_feature_profiled, Exception):
                outputs[index] = target_feature_profiled
                self._release_prepare_split_stage_slot(coordinator)
                continue
            phase_one = GPTSoVITSPrepareAudioPhaseData(
                prompt_g2pw_profiled=shared_prompt_g2pw_profiled,
                target_g2pw_profiled=target_g2pw_profiled,
                ref_audio_profiled=shared_ref_audio_profiled,
                g2pw_pair_ms=max(
                    float(shared_prompt_g2pw_profiled.run_ms),
                    float(target_g2pw_profiled.run_ms),
                ),
                phase_wall_ms=max(
                    float(shared_prompt_g2pw_profiled.run_ms),
                    float(target_g2pw_profiled.run_ms),
                    float(shared_ref_audio_profiled.run_ms),
                    float(shared_ref_spec_result.profile.get("ref_spec_ms", 0.0)),
                ),
            )
            phase_two = GPTSoVITSPrepareTextPhaseData(
                prompt_feature_profiled=shared_prompt_feature_profiled,
                target_feature_profiled=target_feature_profiled,
                phase_wall_ms=max(
                    float(shared_prompt_feature_profiled.run_ms),
                    float(target_feature_profiled.run_ms),
                ),
            )
            try:
                state = self._build_request_state_from_prepare_phases(
                    cpu_stage,
                    phase_one,
                    phase_two,
                    ref_spec_result=shared_ref_spec_result,
                    extra_profile={
                        **self._build_shared_prepare_phase_profile(
                            prompt_profiled=shared_prompt_g2pw_profiled,
                            target_profiled=target_g2pw_profiled,
                            shared_profiled=shared_ref_audio_profiled,
                            shared_ref_spec_result=shared_ref_spec_result,
                            phase_kind="audio",
                        ),
                        **self._build_shared_prepare_phase_profile(
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
                outputs[index] = (
                    state,
                    float(cpu_stage.prepare_start),
                    float(time.perf_counter()),
                )
            except Exception as exc:
                outputs[index] = exc
            finally:
                self._release_prepare_split_stage_slot(coordinator)

        return [item if item is not None else RuntimeError("shared prepare result missing") for item in outputs]

    async def _run_text_feature_stage(
        self,
        coordinator: Any,
        prepared_segments: Any,
        language: str | None,
        cpu_run_ms: float,
        base_profile: dict[str, float] | None = None,
    ) -> GPTSoVITSPrepareProfiledResult:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        if coordinator.text_feature_executor is not None:
            await coordinator.text_feature_gate.acquire()
            try:
                return await self._prepare_run_on_executor(
                    coordinator.text_feature_executor,
                    self._build_text_features,
                    prepared_segments,
                    language,
                    cpu_run_ms,
                    base_profile,
                )
            finally:
                coordinator.text_feature_gate.release()

        self._ensure_pipeline()
        prepare_bert_batch_worker = self._get_runtime_prepare_bert_batch_worker()
        await coordinator.text_feature_gate.acquire()
        profile: dict[str, float] = dict(base_profile or {})
        profile["cpu_preprocess_ms"] = float(cpu_run_ms)
        submit_at = time.perf_counter()
        started_at = float(submit_at)
        try:
            if prepare_bert_batch_worker is None:
                result = self._build_text_features(
                    prepared_segments,
                    language,
                    cpu_run_ms,
                    base_profile,
                )
                finished_at = time.perf_counter()
                return GPTSoVITSPrepareProfiledResult(
                    result=result,
                    submit_at=float(submit_at),
                    started_at=started_at,
                    finished_at=float(finished_at),
                )
            segment_jobs = self._prepare_text_feature_segment_jobs(prepared_segments, profile)
            result_raw = await self._finalize_text_feature_segment_jobs(segment_jobs)
            finished_at = time.perf_counter()
            result = self._make_text_features(
                result_raw[0],
                result_raw[1],
                result_raw[2],
                profile=profile,
                total_ms=float(cpu_run_ms + self._estimate_text_feature_run_ms(profile)),
                cpu_preprocess_ms=cpu_run_ms,
            )
            profiled = GPTSoVITSPrepareProfiledResult(
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
            coordinator.text_feature_gate.release()

    async def _run_g2pw_stage(
        self,
        coordinator: Any,
        prepared_segments: Any,
    ) -> GPTSoVITSPrepareProfiledResult:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        has_pending = self._prepare_result_has_pending_g2pw(prepared_segments)
        if not has_pending:
            submit_at = time.perf_counter()
            return GPTSoVITSPrepareProfiledResult(
                result=prepared_segments,
                submit_at=float(submit_at),
                started_at=float(submit_at),
                finished_at=float(submit_at),
                profile={},
            )
        await coordinator.g2pw_gate.acquire()
        try:
            profiled = await self._prepare_run_on_executor(
                coordinator.g2pw_executor,
                self._resolve_g2pw_segments,
                prepared_segments,
            )
            result, stage_profile = profiled.result
            return GPTSoVITSPrepareProfiledResult(
                result=result,
                submit_at=float(profiled.submit_at),
                started_at=float(profiled.started_at),
                finished_at=float(profiled.finished_at),
                profile=dict(stage_profile),
            )
        finally:
            coordinator.g2pw_gate.release()

    @staticmethod
    def _merge_g2pw_pair_stage_profile(
        profile: dict[str, float] | None,
        pair_profile: dict[str, float],
    ) -> dict[str, float]:
        merged = dict(profile or {})
        for key, value in pair_profile.items():
            merged[key] = float(value)
        return merged

    async def _run_g2pw_pair_stage(
        self,
        coordinator: Any,
        prompt_segments: Any,
        target_segments: Any,
    ) -> tuple[GPTSoVITSPrepareProfiledResult, GPTSoVITSPrepareProfiledResult]:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        pair_submit_at = time.perf_counter()
        prompt_is_empty = len(prompt_segments or []) == 0
        prompt_has_pending = self._prepare_result_has_pending_g2pw(prompt_segments)
        target_has_pending = self._prepare_result_has_pending_g2pw(target_segments)
        self._ensure_pipeline()
        g2pw_batch_worker = self._get_runtime_prepare_g2pw_batch_worker()
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
                prompt_profiled = GPTSoVITSPrepareProfiledResult(
                    result=prompt_result,
                    submit_at=float(submit_at),
                    started_at=float(started_at),
                    finished_at=float(finished_at),
                    profile=prompt_profile,
                )
            else:
                idle_ts = time.perf_counter()
                prompt_profiled = GPTSoVITSPrepareProfiledResult(
                    result=prompt_segments,
                    submit_at=float(idle_ts),
                    started_at=float(idle_ts),
                    finished_at=float(idle_ts),
                    profile={},
                )
            if target_has_pending:
                target_profiled = GPTSoVITSPrepareProfiledResult(
                    result=target_result,
                    submit_at=float(submit_at),
                    started_at=float(started_at),
                    finished_at=float(finished_at),
                    profile=target_profile,
                )
            else:
                idle_ts = time.perf_counter()
                target_profiled = GPTSoVITSPrepareProfiledResult(
                    result=target_segments,
                    submit_at=float(idle_ts),
                    started_at=float(idle_ts),
                    finished_at=float(idle_ts),
                    profile={},
                )
            return prompt_profiled, target_profiled

        if coordinator.enable_g2pw_pair_batch and (prompt_has_pending or target_has_pending):
            gate_wait_start = time.perf_counter()
            await coordinator.g2pw_gate.acquire()
            gate_acquired_at = time.perf_counter()
            try:
                profiled = await self._prepare_run_on_executor(
                    coordinator.g2pw_executor,
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
                    prompt_profiled = GPTSoVITSPrepareProfiledResult(
                        result=prompt_result,
                        submit_at=float(profiled.submit_at),
                        started_at=float(profiled.started_at),
                        finished_at=float(profiled.finished_at),
                        profile=self._merge_g2pw_pair_stage_profile(prompt_profile, pair_profile),
                    )
                else:
                    idle_ts = time.perf_counter()
                    prompt_profiled = GPTSoVITSPrepareProfiledResult(
                        result=prompt_segments,
                        submit_at=float(idle_ts),
                        started_at=float(idle_ts),
                        finished_at=float(idle_ts),
                        profile={},
                    )
                if target_has_pending:
                    target_profiled = GPTSoVITSPrepareProfiledResult(
                        result=target_result,
                        submit_at=float(profiled.submit_at),
                        started_at=float(profiled.started_at),
                        finished_at=float(profiled.finished_at),
                        profile=self._merge_g2pw_pair_stage_profile(target_profile, pair_profile),
                    )
                else:
                    idle_ts = time.perf_counter()
                    target_profiled = GPTSoVITSPrepareProfiledResult(
                        result=target_segments,
                        submit_at=float(idle_ts),
                        started_at=float(idle_ts),
                        finished_at=float(idle_ts),
                        profile={},
                    )
                return prompt_profiled, target_profiled
            finally:
                coordinator.g2pw_gate.release()

        target_task = asyncio.create_task(self._run_g2pw_stage(coordinator, target_segments))
        if not prompt_is_empty:
            prompt_task = asyncio.create_task(self._run_g2pw_stage(coordinator, prompt_segments))
            return await asyncio.gather(prompt_task, target_task)
        target_profiled = await target_task
        submit_at = time.perf_counter()
        prompt_profiled = GPTSoVITSPrepareProfiledResult(
            result=prompt_segments,
            submit_at=float(submit_at),
            started_at=float(submit_at),
            finished_at=float(submit_at),
            profile={},
        )
        return prompt_profiled, target_profiled

    async def _run_g2pw_pair_stage_batch(
        self,
        coordinator: Any,
        cpu_stages: list[Any],
    ) -> list[tuple[GPTSoVITSPrepareProfiledResult, GPTSoVITSPrepareProfiledResult] | Exception]:
        coordinator = self._coerce_prepare_coordinator(coordinator)
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
            prompt_has_pending = self._prepare_result_has_pending_g2pw(prompt_segments)
            target_has_pending = self._prepare_result_has_pending_g2pw(target_segments)
            has_pending_pairs.append((prompt_has_pending, target_has_pending))
            idle_prompt_target.append((prompt_segments, target_segments))
            if prompt_has_pending:
                group_request_index.append((index, "prompt"))
                group_batches.append(prompt_segments or [])
            if target_has_pending:
                group_request_index.append((index, "target"))
                group_batches.append(target_segments or [])

        if not group_batches:
            profiled_results: list[tuple[GPTSoVITSPrepareProfiledResult, GPTSoVITSPrepareProfiledResult]] = []
            for prompt_segments, target_segments in idle_prompt_target:
                idle_ts = time.perf_counter()
                profiled_results.append(
                    (
                        GPTSoVITSPrepareProfiledResult(
                            result=prompt_segments,
                            submit_at=float(idle_ts),
                            started_at=float(idle_ts),
                            finished_at=float(idle_ts),
                            profile={},
                        ),
                        GPTSoVITSPrepareProfiledResult(
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
        await coordinator.g2pw_gate.acquire()
        gate_acquired_at = time.perf_counter()
        try:
            profiled = await self._prepare_run_on_executor(
                coordinator.g2pw_executor,
                self._resolve_g2pw_segment_batches,
                group_batches,
            )
        finally:
            coordinator.g2pw_gate.release()

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

        prompt_results: list[GPTSoVITSPrepareProfiledResult | None] = [None] * len(cpu_stages)
        target_results: list[GPTSoVITSPrepareProfiledResult | None] = [None] * len(cpu_stages)
        for (request_index, branch), resolved_segments, stage_profile in zip(
            group_request_index,
            resolved_batches,
            batch_profiles,
        ):
            branch_profile = self._merge_g2pw_pair_stage_profile(stage_profile, pair_profile)
            branch_result = GPTSoVITSPrepareProfiledResult(
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
                prompt_profiled = GPTSoVITSPrepareProfiledResult(
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
                target_profiled = GPTSoVITSPrepareProfiledResult(
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

    async def _run_text_feature_pair_stage(
        self,
        coordinator: Any,
        prompt_segments: Any,
        target_segments: Any,
        prompt_cpu_run_ms: float,
        target_cpu_run_ms: float,
        prompt_base_profile: dict[str, float] | None = None,
        target_base_profile: dict[str, float] | None = None,
    ) -> tuple[GPTSoVITSPrepareProfiledResult, GPTSoVITSPrepareProfiledResult]:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        prompt_is_empty = len(prompt_segments or []) == 0
        pipeline = self._ensure_pipeline()
        if coordinator.text_feature_executor is not None:
            target_feature_task = asyncio.create_task(
                self._prepare_run_on_executor(
                    coordinator.text_feature_executor,
                    self._build_text_features,
                    target_segments,
                    None,
                    target_cpu_run_ms,
                    target_base_profile,
                )
            )
            if not prompt_is_empty:
                prompt_feature_task = asyncio.create_task(
                    self._prepare_run_on_executor(
                        coordinator.text_feature_executor,
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
            prompt_profiled = GPTSoVITSPrepareProfiledResult(
                result=self._build_empty_text_features_like(target_profiled.result),
                submit_at=float(submit_at),
                started_at=float(submit_at),
                finished_at=float(submit_at),
            )
            return prompt_profiled, target_profiled

        prepare_bert_batch_worker = self._get_runtime_prepare_bert_batch_worker()
        await coordinator.text_feature_gate.acquire()
        submit_at = time.perf_counter()
        started_at = float(submit_at)
        try:
            if prepare_bert_batch_worker is None:
                target_result = self._build_text_features(
                    target_segments,
                    None,
                    target_cpu_run_ms,
                    target_base_profile,
                )
                if prompt_is_empty:
                    finished_at = time.perf_counter()
                    prompt_profiled = GPTSoVITSPrepareProfiledResult(
                        result=self._build_empty_text_features_like(target_result),
                        submit_at=float(submit_at),
                        started_at=float(submit_at),
                        finished_at=float(submit_at),
                    )
                    target_profiled = GPTSoVITSPrepareProfiledResult(
                        result=target_result,
                        submit_at=float(submit_at),
                        started_at=started_at,
                        finished_at=float(finished_at),
                    )
                    return prompt_profiled, target_profiled

                prompt_result = self._build_text_features(
                    prompt_segments,
                    None,
                    prompt_cpu_run_ms,
                    prompt_base_profile,
                )
                finished_at = time.perf_counter()
                prompt_profiled = GPTSoVITSPrepareProfiledResult(
                    result=prompt_result,
                    submit_at=float(submit_at),
                    started_at=started_at,
                    finished_at=float(finished_at),
                )
                target_profiled = GPTSoVITSPrepareProfiledResult(
                    result=target_result,
                    submit_at=float(submit_at),
                    started_at=started_at,
                    finished_at=float(finished_at),
                )
                return prompt_profiled, target_profiled

            target_profile: dict[str, float] = dict(target_base_profile or {})
            target_profile["cpu_preprocess_ms"] = float(target_cpu_run_ms)
            if prompt_is_empty:
                target_jobs = self._prepare_text_feature_segment_jobs(target_segments, target_profile)
                target_result_raw = await self._finalize_text_feature_segment_jobs(target_jobs)
                prompt_result = self._build_empty_text_features_like(
                    self._make_text_features(
                        target_result_raw[0],
                        target_result_raw[1],
                        target_result_raw[2],
                        profile=target_profile,
                        total_ms=float(target_cpu_run_ms + self._estimate_text_feature_run_ms(target_profile)),
                        cpu_preprocess_ms=target_cpu_run_ms,
                    )
                )
                finished_at = time.perf_counter()
                prompt_profiled = GPTSoVITSPrepareProfiledResult(
                    result=prompt_result,
                    submit_at=float(submit_at),
                    started_at=float(submit_at),
                    finished_at=float(submit_at),
                )
                target_result = self._make_text_features(
                    target_result_raw[0],
                    target_result_raw[1],
                    target_result_raw[2],
                    profile=target_profile,
                    total_ms=float(target_cpu_run_ms + self._estimate_text_feature_run_ms(target_profile)),
                    cpu_preprocess_ms=target_cpu_run_ms,
                )
                target_profiled = GPTSoVITSPrepareProfiledResult(
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

            prompt_profile: dict[str, float] = dict(prompt_base_profile or {})
            prompt_profile["cpu_preprocess_ms"] = float(prompt_cpu_run_ms)
            prompt_jobs = self._prepare_text_feature_segment_jobs(prompt_segments, prompt_profile)
            target_jobs = self._prepare_text_feature_segment_jobs(target_segments, target_profile)
            pending_jobs = list(prompt_jobs["pending_items"]) + list(target_jobs["pending_items"])
            if pending_jobs:
                pending_results = await asyncio.gather(*[future for _, _, _, _, future in pending_jobs])
                for (bert_list, bert_index, profile, device, _), (feature, worker_profile) in zip(
                    pending_jobs,
                    pending_results,
                ):
                    self._merge_bert_worker_profile(profile, dict(worker_profile))
                    bert_list[bert_index] = feature.to(device)
            prompt_result_raw = await self._finalize_text_feature_segment_jobs({**prompt_jobs, "pending_items": []})
            target_result_raw = await self._finalize_text_feature_segment_jobs({**target_jobs, "pending_items": []})
            finished_at = time.perf_counter()

            prompt_result = self._make_text_features(
                prompt_result_raw[0],
                prompt_result_raw[1],
                prompt_result_raw[2],
                profile=prompt_profile,
                total_ms=float(prompt_cpu_run_ms + self._estimate_text_feature_run_ms(prompt_profile)),
                cpu_preprocess_ms=prompt_cpu_run_ms,
            )
            target_result = self._make_text_features(
                target_result_raw[0],
                target_result_raw[1],
                target_result_raw[2],
                profile=target_profile,
                total_ms=float(target_cpu_run_ms + self._estimate_text_feature_run_ms(target_profile)),
                cpu_preprocess_ms=target_cpu_run_ms,
            )
            prompt_profiled = GPTSoVITSPrepareProfiledResult(
                result=prompt_result,
                submit_at=float(submit_at),
                started_at=started_at,
                finished_at=float(submit_at + self._estimate_text_feature_run_ms(prompt_profile) / 1000.0),
            )
            target_profiled = GPTSoVITSPrepareProfiledResult(
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
            coordinator.text_feature_gate.release()

    async def _run_ref_prompt_semantic_stage(
        self,
        coordinator: Any,
        ref_audio_path: str,
        prepared_asset_future: Any | None = None,
        prepared_asset: Any | None = None,
    ) -> GPTSoVITSPrepareProfiledResult:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        self._ensure_pipeline()
        ref_worker = self._get_runtime_prepare_ref_semantic_batch_worker()
        ref_stage_limiter = self._get_runtime_prepare_ref_semantic_stage_limiter()
        if ref_worker is not None:
            submit_at = time.perf_counter()
            started_at = float(submit_at)
            preload_profiled: Any | None = None
            if prepared_asset is not None:
                preload_profiled = GPTSoVITSPrepareProfiledResult(
                    result=prepared_asset,
                    submit_at=float(submit_at),
                    started_at=float(submit_at),
                    finished_at=float(submit_at),
                )
            elif prepared_asset_future is not None:
                preload_profiled = await asyncio.wrap_future(prepared_asset_future)

            if preload_profiled is None:
                await coordinator.ref_load_gate.acquire()
                try:
                    load_profiled = await self._prepare_run_on_executor(
                        coordinator.ref_audio_executor,
                        self._load_ref_audio_raw,
                        ref_audio_path,
                    )
                finally:
                    coordinator.ref_load_gate.release()
                raw_audio, raw_sr = load_profiled.result
                wav16k, local_cpu_prepare_profile = await asyncio.to_thread(
                    self._prepare_ref_prompt_wav16k_for_worker,
                    raw_audio,
                    raw_sr,
                )
                load_queue_ms = float(load_profiled.queue_ms)
                load_ms = float(load_profiled.run_ms)
                cpu_prepare_wait_ms = float(local_cpu_prepare_profile.get("prompt_semantic_cpu_prepare_wait_ms", 0.0))
                cpu_prepare_slots = float(local_cpu_prepare_profile.get("prompt_semantic_cpu_prepare_slots", 0.0))
                cpu_prepare_inflight_peak = float(
                    local_cpu_prepare_profile.get("prompt_semantic_cpu_prepare_inflight_peak", 0.0)
                )
                preload_cpu_prepare_ms = float(local_cpu_prepare_profile.get("prompt_semantic_cpu_prepare_ms", 0.0))
            else:
                prepared_result = preload_profiled.result
                raw_audio = prepared_result.raw_audio
                raw_sr = prepared_result.raw_sr
                wav16k = prepared_result.wav16k
                preload_profile = dict(getattr(prepared_result, "profile", {}) or {})
                load_queue_ms = float(preload_profiled.queue_ms)
                load_ms = float(preload_profile.get("audio_load_ms", 0.0))
                cpu_prepare_wait_ms = float(preload_profile.get("prompt_semantic_cpu_prepare_wait_ms", 0.0))
                cpu_prepare_slots = float(preload_profile.get("prompt_semantic_cpu_prepare_slots", 0.0))
                cpu_prepare_inflight_peak = float(
                    preload_profile.get("prompt_semantic_cpu_prepare_inflight_peak", 0.0)
                )
                preload_cpu_prepare_ms = float(preload_profile.get("prompt_semantic_cpu_prepare_ms", 0.0))

            route = self._build_ref_prompt_semantic_worker_routing(
                coordinator,
                ref_worker,
                raw_audio,
                raw_sr,
                wav16k=wav16k,
            )
            prompt_semantic_task = asyncio.create_task(
                ref_worker.submit_async(
                    raw_audio,
                    raw_sr,
                    wav16k=wav16k,
                    runtime_exact_prewarm_profile=self._build_ref_prompt_semantic_runtime_exact_prewarm_profile(
                        coordinator,
                        ref_worker,
                        raw_audio,
                        raw_sr,
                        wav16k=wav16k,
                    ),
                    bucket_index=route["bucket_index"],
                    preferred_shard_index=route["preferred_shard_index"],
                    bucket_first_hit_serialized=route["bucket_first_hit_serialized"],
                )
            )
            try:
                prompt_semantic, prompt_semantic_profile = await prompt_semantic_task
            finally:
                self._mark_ref_prompt_semantic_worker_routing_completed(coordinator, route)
            limiter_snapshot = (
                ref_stage_limiter.snapshot()
                if ref_stage_limiter is not None
                else {}
            )
            finished_at = time.perf_counter()
            return self._build_ref_prompt_semantic_worker_profiled_result(
                submit_at=float(submit_at),
                started_at=started_at,
                finished_at=float(finished_at),
                prompt_semantic=prompt_semantic,
                raw_audio=raw_audio,
                raw_sr=int(raw_sr),
                load_queue_ms=float(load_queue_ms),
                load_ms=float(load_ms),
                cpu_prepare_wait_ms=float(cpu_prepare_wait_ms),
                cpu_prepare_slots=float(cpu_prepare_slots),
                cpu_prepare_inflight_peak=float(cpu_prepare_inflight_peak),
                preload_cpu_prepare_ms=float(preload_cpu_prepare_ms),
                prompt_semantic_profile=dict(prompt_semantic_profile),
                limiter_snapshot=limiter_snapshot,
            )

        await coordinator.ref_audio_gate.acquire()
        try:
            preload_profiled: Any | None = None
            if prepared_asset is not None:
                submit_at = time.perf_counter()
                preload_profiled = GPTSoVITSPrepareProfiledResult(
                    result=prepared_asset,
                    submit_at=float(submit_at),
                    started_at=float(submit_at),
                    finished_at=float(submit_at),
                )
            elif prepared_asset_future is not None:
                preload_profiled = await asyncio.wrap_future(prepared_asset_future)

            if preload_profiled is None:
                load_profiled = await self._prepare_run_on_executor(
                    coordinator.ref_audio_executor,
                    self._load_ref_audio_raw,
                    ref_audio_path,
                )
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
                raw_audio = prepared_result.raw_audio
                raw_sr = prepared_result.raw_sr
                wav16k = prepared_result.wav16k
                preload_profile = dict(getattr(prepared_result, "profile", {}) or {})
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
                if ref_stage_limiter is None:
                    stage_stats = {"wait_ms": 0.0, "slots": 0.0, "peak_inflight": 0.0}
                    prompt_semantic, runtime_profile = await asyncio.to_thread(
                        self._extract_prompt_semantic_profile_from_prepared_wav16k,
                        wav16k,
                    )
                else:
                    with ref_stage_limiter.enter() as stage_stats:
                        prompt_semantic, runtime_profile = await asyncio.to_thread(
                            self._extract_prompt_semantic_profile_from_prepared_wav16k,
                            wav16k,
                        )
                result = GPTSoVITSRefAudioBundle(
                    prompt_semantic=prompt_semantic,
                    raw_audio=raw_audio,
                    raw_sr=int(raw_sr),
                    profile={
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
                )
            result.profile.setdefault("audio_load_queue_ms", float(load_queue_ms))
            result.profile.setdefault("audio_load_ms", float(load_ms))
            finished_at = time.perf_counter()
            return GPTSoVITSPrepareProfiledResult(
                result=result,
                submit_at=float(submit_at),
                started_at=float(started_at),
                finished_at=float(finished_at),
            )
        finally:
            coordinator.ref_audio_gate.release()

    async def _run_ref_prompt_semantic_stage_batch(
        self,
        coordinator: Any,
        items: list[tuple[str, Any | None, Any | None]],
    ) -> list[GPTSoVITSPrepareProfiledResult | Exception]:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        if not items:
            return []
        self._ensure_pipeline()
        worker = self._get_runtime_prepare_ref_semantic_batch_worker()
        ref_stage_limiter = self._get_runtime_prepare_ref_semantic_stage_limiter()
        if worker is None:
            return await asyncio.gather(
                *[
                    self._run_ref_prompt_semantic_stage(
                        coordinator,
                        ref_audio_path,
                        prepared_asset_future=prepared_asset_future,
                        prepared_asset=prepared_asset,
                    )
                    for ref_audio_path, prepared_asset_future, prepared_asset in items
                ],
                return_exceptions=True,
            )

        preload_profiled_results: list[GPTSoVITSPrepareProfiledResult | Exception | None] = [None] * len(items)
        preload_future_indices: list[int] = []
        preload_future_awaitables: list[Any] = []
        for index, (_, prepared_asset_future, prepared_asset) in enumerate(items):
            if prepared_asset is not None:
                submit_at = time.perf_counter()
                preload_profiled_results[index] = GPTSoVITSPrepareProfiledResult(
                    result=prepared_asset,
                    submit_at=float(submit_at),
                    started_at=float(submit_at),
                    finished_at=float(submit_at),
                )
            elif prepared_asset_future is not None:
                preload_future_indices.append(index)
                preload_future_awaitables.append(asyncio.wrap_future(prepared_asset_future))

        if preload_future_awaitables:
            awaited_preloads = await asyncio.gather(*preload_future_awaitables, return_exceptions=True)
            for index, preload_profiled in zip(preload_future_indices, awaited_preloads):
                preload_profiled_results[index] = preload_profiled

        results: list[GPTSoVITSPrepareProfiledResult | Exception | None] = [None] * len(items)
        worker_tasks: list[asyncio.Task[Any]] = []
        worker_indices: list[int] = []
        worker_submit_meta: dict[int, dict[str, Any]] = {}
        fallback_tasks: list[asyncio.Task[Any]] = []
        fallback_indices: list[int] = []
        try:
            for index, (ref_audio_path, prepared_asset_future, prepared_asset) in enumerate(items):
                preload_profiled = preload_profiled_results[index]
                if isinstance(preload_profiled, Exception):
                    results[index] = preload_profiled
                    continue
                if preload_profiled is None:
                    fallback_indices.append(index)
                    fallback_tasks.append(
                        asyncio.create_task(
                            self._run_ref_prompt_semantic_stage(
                                coordinator,
                                ref_audio_path,
                                prepared_asset_future=prepared_asset_future,
                                prepared_asset=prepared_asset,
                            )
                        )
                    )
                    continue
                prepared_result = preload_profiled.result
                preload_profile = dict(getattr(prepared_result, "profile", {}) or {})
                raw_audio = prepared_result.raw_audio
                raw_sr = prepared_result.raw_sr
                wav16k = prepared_result.wav16k
                submit_at = time.perf_counter()
                route = self._build_ref_prompt_semantic_worker_routing(
                    coordinator,
                    worker,
                    raw_audio,
                    raw_sr,
                    wav16k=wav16k,
                )
                worker_indices.append(index)
                worker_submit_meta[index] = {
                    "submit_at": float(submit_at),
                    "started_at": float(submit_at),
                    "raw_audio": raw_audio,
                    "raw_sr": int(raw_sr),
                    "route": route,
                    "runtime_exact_prewarm_profile": self._build_ref_prompt_semantic_runtime_exact_prewarm_profile(
                        coordinator,
                        worker,
                        raw_audio,
                        raw_sr,
                        wav16k=wav16k,
                    ),
                    "load_queue_ms": float(preload_profiled.queue_ms),
                    "load_ms": float(preload_profile.get("audio_load_ms", 0.0)),
                    "cpu_prepare_wait_ms": float(preload_profile.get("prompt_semantic_cpu_prepare_wait_ms", 0.0)),
                    "cpu_prepare_slots": float(preload_profile.get("prompt_semantic_cpu_prepare_slots", 0.0)),
                    "cpu_prepare_inflight_peak": float(
                        preload_profile.get("prompt_semantic_cpu_prepare_inflight_peak", 0.0)
                    ),
                    "preload_cpu_prepare_ms": float(preload_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)),
                }
                worker_tasks.append(
                    asyncio.create_task(
                        worker.submit_async(
                            raw_audio,
                            raw_sr,
                            wav16k=wav16k,
                            runtime_exact_prewarm_profile=worker_submit_meta[index]["runtime_exact_prewarm_profile"],
                            bucket_index=route["bucket_index"],
                            preferred_shard_index=route["preferred_shard_index"],
                            bucket_first_hit_serialized=route["bucket_first_hit_serialized"],
                        )
                    )
                )

            if worker_tasks:
                worker_outputs = await asyncio.gather(*worker_tasks, return_exceptions=True)
                limiter_snapshot = (
                    ref_stage_limiter.snapshot()
                    if ref_stage_limiter is not None
                    else {}
                )
                batch_finished_at = time.perf_counter()
                for index, worker_output in zip(worker_indices, worker_outputs):
                    self._mark_ref_prompt_semantic_worker_routing_completed(
                        coordinator,
                        worker_submit_meta[index].get("route"),
                    )
                    if isinstance(worker_output, Exception):
                        results[index] = worker_output
                        continue
                    prompt_semantic, prompt_semantic_profile = worker_output
                    submit_meta = worker_submit_meta[index]
                    results[index] = self._build_ref_prompt_semantic_worker_profiled_result(
                        submit_at=submit_meta["submit_at"],
                        started_at=submit_meta["started_at"],
                        finished_at=float(batch_finished_at),
                        prompt_semantic=prompt_semantic,
                        raw_audio=submit_meta["raw_audio"],
                        raw_sr=submit_meta["raw_sr"],
                        load_queue_ms=submit_meta["load_queue_ms"],
                        load_ms=submit_meta["load_ms"],
                        cpu_prepare_wait_ms=submit_meta["cpu_prepare_wait_ms"],
                        cpu_prepare_slots=submit_meta["cpu_prepare_slots"],
                        cpu_prepare_inflight_peak=submit_meta["cpu_prepare_inflight_peak"],
                        preload_cpu_prepare_ms=submit_meta["preload_cpu_prepare_ms"],
                        prompt_semantic_profile=dict(prompt_semantic_profile),
                        limiter_snapshot=limiter_snapshot,
                    )

            if fallback_tasks:
                fallback_outputs = await asyncio.gather(*fallback_tasks, return_exceptions=True)
                for index, fallback_output in zip(fallback_indices, fallback_outputs):
                    results[index] = fallback_output
        finally:
            for task in worker_tasks:
                if not task.done():
                    task.cancel()
            for task in fallback_tasks:
                if not task.done():
                    task.cancel()

        return [
            result if result is not None else RuntimeError("GPT-SoVITS ref prompt semantic batch result missing")
            for result in results
        ]

    async def _run_ref_spec_stage(
        self,
        coordinator: Any,
        raw_audio: Any,
        raw_sr: int,
    ) -> GPTSoVITSPrepareProfiledResult:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        await coordinator.ref_spec_gate.acquire()
        try:
            return await self._prepare_run_on_executor(
                coordinator.ref_audio_executor,
                self._extract_ref_spec_from_raw,
                raw_audio,
                raw_sr,
            )
        finally:
            coordinator.ref_spec_gate.release()

    async def _prepare_gpu_audio_phase_batch_async(
        self,
        coordinator: Any,
        prepared_cpu_stages: list[GPTSoVITSPreparedCpuStage],
    ) -> list[GPTSoVITSPrepareAudioPhaseData | Exception]:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        if not prepared_cpu_stages:
            return []
        phase_start = time.perf_counter()
        ref_audio_batch_task = asyncio.create_task(
            self._run_ref_prompt_semantic_stage_batch(
                coordinator,
                [
                    (
                        str(prepared_cpu_stage.spec.ref_audio_path),
                        prepared_cpu_stage.ref_audio_prepare_future,
                        None,
                    )
                    for prepared_cpu_stage in prepared_cpu_stages
                ],
            )
        )
        try:
            g2pw_pairs: list[tuple[GPTSoVITSPrepareProfiledResult, GPTSoVITSPrepareProfiledResult] | Exception | None] = [
                None
            ] * len(prepared_cpu_stages)
            group_size = max(1, int(getattr(coordinator, "g2pw_audio_batch_merge_group_size", 8)))
            for start_index in range(0, len(prepared_cpu_stages), group_size):
                group = prepared_cpu_stages[start_index : start_index + group_size]
                group_pairs = await self._run_g2pw_pair_stage_batch(coordinator, group)
                for offset, group_pair in enumerate(group_pairs):
                    g2pw_pairs[start_index + offset] = group_pair
            g2pw_pair_end = time.perf_counter()
            ref_audio_results = await ref_audio_batch_task
            outputs: list[GPTSoVITSPrepareAudioPhaseData | Exception] = []
            for prepared_cpu_stage, g2pw_pair, ref_audio_profiled in zip(
                prepared_cpu_stages, g2pw_pairs, ref_audio_results
            ):
                if isinstance(g2pw_pair, Exception):
                    outputs.append(g2pw_pair)
                    continue
                if isinstance(ref_audio_profiled, Exception):
                    outputs.append(ref_audio_profiled)
                    continue
                assert g2pw_pair is not None
                prompt_g2pw_profiled, target_g2pw_profiled = g2pw_pair
                phase_end = max(float(g2pw_pair_end), float(ref_audio_profiled.finished_at))
                outputs.append(
                    GPTSoVITSPrepareAudioPhaseData(
                        prompt_g2pw_profiled=prompt_g2pw_profiled,
                        target_g2pw_profiled=target_g2pw_profiled,
                        ref_audio_profiled=ref_audio_profiled,
                        g2pw_pair_ms=max(0.0, (g2pw_pair_end - phase_start) * 1000.0),
                        phase_wall_ms=max(0.0, (phase_end - phase_start) * 1000.0),
                    )
                )
            return outputs
        finally:
            if not ref_audio_batch_task.done():
                ref_audio_batch_task.cancel()

    async def _prepare_gpu_audio_phase_async(
        self,
        coordinator: Any,
        prepared_cpu_stage: GPTSoVITSPreparedCpuStage,
    ) -> GPTSoVITSPrepareAudioPhaseData:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        phase_start = time.perf_counter()
        g2pw_pair_task = asyncio.create_task(
            self._run_g2pw_pair_stage(
                coordinator,
                prepared_cpu_stage.prompt_cpu_profiled.result,
                prepared_cpu_stage.target_cpu_profiled.result,
            )
        )
        ref_audio_task = asyncio.create_task(
            self._run_ref_prompt_semantic_stage(
                coordinator,
                str(prepared_cpu_stage.spec.ref_audio_path),
                prepared_asset_future=prepared_cpu_stage.ref_audio_prepare_future,
            )
        )
        prompt_g2pw_profiled, target_g2pw_profiled = await g2pw_pair_task
        g2pw_pair_end = time.perf_counter()
        ref_audio_profiled = await ref_audio_task
        phase_end = time.perf_counter()
        return GPTSoVITSPrepareAudioPhaseData(
            prompt_g2pw_profiled=prompt_g2pw_profiled,
            target_g2pw_profiled=target_g2pw_profiled,
            ref_audio_profiled=ref_audio_profiled,
            g2pw_pair_ms=max(0.0, (g2pw_pair_end - phase_start) * 1000.0),
            phase_wall_ms=max(0.0, (phase_end - phase_start) * 1000.0),
        )

    def prepare_request_gpu_audio_phase(
        self,
        prepared_cpu_stage: GPTSoVITSPreparedCpuStage,
    ) -> GPTSoVITSPreparedAudioPhase:
        coordinator = self._ensure_prepare_coordinator()
        phase_one = self._run_awaitable_sync(
            self._prepare_gpu_audio_phase_async(coordinator, prepared_cpu_stage)
        )
        return GPTSoVITSPreparedAudioPhase(
            request_id=prepared_cpu_stage.request_id,
            prepared_cpu_stage=prepared_cpu_stage,
            phase_one=phase_one,
        )

    def prepare_request_gpu_audio_phases(
        self,
        prepared_cpu_stages: list[GPTSoVITSPreparedCpuStage],
    ) -> list[GPTSoVITSPreparedAudioPhase]:
        if not prepared_cpu_stages:
            return []
        coordinator = self._ensure_prepare_coordinator()

        async def _gather_phase_ones():
            if coordinator.enable_g2pw_audio_batch_merge and len(prepared_cpu_stages) > 1:
                return await self._prepare_gpu_audio_phase_batch_async(coordinator, prepared_cpu_stages)
            return await asyncio.gather(
                *[
                    self._prepare_gpu_audio_phase_async(coordinator, prepared_cpu_stage)
                    for prepared_cpu_stage in prepared_cpu_stages
                ],
                return_exceptions=True,
            )

        phase_one_results = self._run_awaitable_sync(_gather_phase_ones())
        if len(phase_one_results) != len(prepared_cpu_stages):
            raise ValueError("GPT-SoVITS batch prepare audio phase count mismatch")
        prepared_audio_phases: list[GPTSoVITSPreparedAudioPhase] = []
        for prepared_cpu_stage, phase_one in zip(prepared_cpu_stages, phase_one_results):
            if isinstance(phase_one, Exception):
                raise phase_one
            prepared_audio_phases.append(
                GPTSoVITSPreparedAudioPhase(
                    request_id=prepared_cpu_stage.request_id,
                    prepared_cpu_stage=prepared_cpu_stage,
                    phase_one=phase_one,
                )
            )
        return prepared_audio_phases

    async def _prepare_gpu_text_phase_async(
        self,
        coordinator: Any,
        prepared_audio_phase: GPTSoVITSPreparedAudioPhase,
    ) -> GPTSoVITSPrepareTextPhaseData:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        prepared_cpu_stage = prepared_audio_phase.prepared_cpu_stage
        phase_one = prepared_audio_phase.phase_one
        phase_start = time.perf_counter()
        prompt_g2pw_profiled = phase_one.prompt_g2pw_profiled
        target_g2pw_profiled = phase_one.target_g2pw_profiled
        prompt_feature_profiled, target_feature_profiled = await self._run_text_feature_pair_stage(
            coordinator,
            prompt_g2pw_profiled.result,
            target_g2pw_profiled.result,
            prepared_cpu_stage.prompt_cpu_profiled.run_ms,
            prepared_cpu_stage.target_cpu_profiled.run_ms,
            prompt_base_profile=dict(prompt_g2pw_profiled.profile or {}),
            target_base_profile=dict(target_g2pw_profiled.profile or {}),
        )
        phase_end = time.perf_counter()
        return GPTSoVITSPrepareTextPhaseData(
            prompt_feature_profiled=prompt_feature_profiled,
            target_feature_profiled=target_feature_profiled,
            phase_wall_ms=max(0.0, (phase_end - phase_start) * 1000.0),
        )

    def prepare_request_gpu_text_phase(
        self,
        prepared_audio_phase: GPTSoVITSPreparedAudioPhase,
    ) -> GPTSoVITSPreparedTextPhase:
        coordinator = self._ensure_prepare_coordinator()
        phase_two = self._run_awaitable_sync(
            self._prepare_gpu_text_phase_async(coordinator, prepared_audio_phase)
        )
        return GPTSoVITSPreparedTextPhase(
            request_id=prepared_audio_phase.request_id,
            prepared_audio_phase=prepared_audio_phase,
            phase_two=phase_two,
        )

    def prepare_request_gpu_text_phases(
        self,
        prepared_audio_phases: list[GPTSoVITSPreparedAudioPhase],
    ) -> list[GPTSoVITSPreparedTextPhase]:
        if not prepared_audio_phases:
            return []
        coordinator = self._ensure_prepare_coordinator()
        async def _gather_phase_twos():
            return await asyncio.gather(
                *[
                    self._prepare_gpu_text_phase_async(coordinator, prepared_audio_phase)
                    for prepared_audio_phase in prepared_audio_phases
                ],
                return_exceptions=True,
            )
        phase_two_results = self._run_awaitable_sync(_gather_phase_twos())
        if len(phase_two_results) != len(prepared_audio_phases):
            raise ValueError("GPT-SoVITS batch prepare text phase count mismatch")
        prepared_text_phases: list[GPTSoVITSPreparedTextPhase] = []
        for prepared_audio_phase, phase_two in zip(prepared_audio_phases, phase_two_results):
            if isinstance(phase_two, Exception):
                raise phase_two
            prepared_text_phases.append(
                GPTSoVITSPreparedTextPhase(
                    request_id=prepared_audio_phase.request_id,
                    prepared_audio_phase=prepared_audio_phase,
                    phase_two=phase_two,
                )
            )
        return prepared_text_phases

    async def _prepare_ref_spec_phase_async(
        self,
        coordinator: Any,
        prepared_audio_phase: GPTSoVITSPreparedAudioPhase,
    ) -> GPTSoVITSPrepareRefSpecResult:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        ref_audio_profiled = prepared_audio_phase.phase_one.ref_audio_profiled
        raw_audio = ref_audio_profiled.result.raw_audio
        raw_sr = int(ref_audio_profiled.result.raw_sr)
        profiled = await self._run_ref_spec_stage(coordinator, raw_audio, raw_sr)
        refer_spec_raw, profile = profiled.result
        refer_spec = self._coerce_refer_spec(refer_spec_raw)
        merged_profile = dict(profile)
        merged_profile["ref_spec_wait_ms"] = float(profiled.queue_ms)
        merged_profile["ref_spec_ms"] = float(profiled.run_ms)
        return GPTSoVITSPrepareRefSpecResult(
            refer_spec=refer_spec,
            profile=merged_profile,
        )

    def prepare_request_ref_spec_phase(
        self,
        prepared_audio_phase: GPTSoVITSPreparedAudioPhase,
    ) -> GPTSoVITSPreparedRefSpecPhase:
        coordinator = self._ensure_prepare_coordinator()
        ref_spec_result = self._run_awaitable_sync(
            self._prepare_ref_spec_phase_async(coordinator, prepared_audio_phase)
        )
        return GPTSoVITSPreparedRefSpecPhase(
            request_id=prepared_audio_phase.request_id,
            prepared_audio_phase=prepared_audio_phase,
            ref_spec_result=ref_spec_result,
        )

    def prepare_request_ref_spec_phases(
        self,
        prepared_audio_phases: list[GPTSoVITSPreparedAudioPhase],
    ) -> list[GPTSoVITSPreparedRefSpecPhase]:
        if not prepared_audio_phases:
            return []
        coordinator = self._ensure_prepare_coordinator()
        async def _gather_ref_specs():
            return await asyncio.gather(
                *[
                    self._prepare_ref_spec_phase_async(coordinator, prepared_audio_phase)
                    for prepared_audio_phase in prepared_audio_phases
                ],
                return_exceptions=True,
            )
        ref_spec_results = self._run_awaitable_sync(_gather_ref_specs())
        if len(ref_spec_results) != len(prepared_audio_phases):
            raise ValueError("GPT-SoVITS batch prepare ref spec phase count mismatch")
        prepared_ref_spec_phases: list[GPTSoVITSPreparedRefSpecPhase] = []
        for prepared_audio_phase, ref_spec_result in zip(prepared_audio_phases, ref_spec_results):
            if isinstance(ref_spec_result, Exception):
                raise ref_spec_result
            prepared_ref_spec_phases.append(
                GPTSoVITSPreparedRefSpecPhase(
                    request_id=prepared_audio_phase.request_id,
                    prepared_audio_phase=prepared_audio_phase,
                    ref_spec_result=ref_spec_result,
                )
            )
        return prepared_ref_spec_phases

    @staticmethod
    def _profile_value(profiled: Any, key: str, default: float = 0.0) -> float:
        return float((getattr(profiled, "profile", None) or {}).get(key, default))

    def _build_prepare_profile_overrides(
        self,
        cpu_stage: Any,
        phase_one: GPTSoVITSPrepareAudioPhaseData,
        phase_two: GPTSoVITSPrepareTextPhaseData,
        *,
        extra_profile: dict[str, float] | None = None,
    ) -> dict[str, float]:
        prompt_g2pw_profiled = phase_one.prompt_g2pw_profiled
        target_g2pw_profiled = phase_one.target_g2pw_profiled
        ref_audio_profiled = phase_one.ref_audio_profiled
        prompt_feature_profiled = phase_two.prompt_feature_profiled
        target_feature_profiled = phase_two.target_feature_profiled
        profile_overrides = {
            "executor_queue_ms": max(0.0, (cpu_stage.prepare_start - cpu_stage.prepare_submit_at) * 1000.0),
            "prepare_admission_wait_ms": float(cpu_stage.prepare_admission_wait_ms),
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
            "text_feature_pair_ms": float(phase_two.phase_wall_ms),
            "g2pw_pair_ms": float(phase_one.g2pw_pair_ms),
            "g2pw_pair_gate_wait_ms": self._profile_value(target_g2pw_profiled, "g2pw_pair_gate_wait_ms"),
            "g2pw_pair_executor_queue_ms": self._profile_value(target_g2pw_profiled, "g2pw_pair_executor_queue_ms"),
            "g2pw_pair_compute_ms": self._profile_value(target_g2pw_profiled, "g2pw_pair_compute_ms"),
            "g2pw_pair_stage_overhead_ms": self._profile_value(target_g2pw_profiled, "g2pw_pair_stage_overhead_ms"),
            "g2pw_pair_audio_batch_merge_size": self._profile_value(target_g2pw_profiled, "g2pw_pair_audio_batch_merge_size"),
            "prompt_text_g2pw_queue_ms": float(prompt_g2pw_profiled.queue_ms),
            "prompt_text_g2pw_run_ms": float(prompt_g2pw_profiled.run_ms),
            "prompt_text_g2pw_prepare_ms": self._profile_value(prompt_g2pw_profiled, "g2pw_prepare_ms"),
            "prompt_text_g2pw_predict_ms": self._profile_value(prompt_g2pw_profiled, "g2pw_predict_ms"),
            "prompt_text_g2pw_post_ms": self._profile_value(prompt_g2pw_profiled, "g2pw_post_ms"),
            "prompt_text_g2pw_wait_ms": self._profile_value(prompt_g2pw_profiled, "g2pw_wait_ms"),
            "prompt_text_g2pw_admission_wait_ms": self._profile_value(prompt_g2pw_profiled, "g2pw_admission_wait_ms"),
            "prompt_text_g2pw_worker_queue_wait_ms": self._profile_value(prompt_g2pw_profiled, "g2pw_worker_queue_wait_ms"),
            "prompt_text_g2pw_batch_collect_wait_ms": self._profile_value(prompt_g2pw_profiled, "g2pw_batch_collect_wait_ms"),
            "prompt_text_g2pw_batch_dispatch_delay_ms": self._profile_value(prompt_g2pw_profiled, "g2pw_batch_dispatch_delay_ms"),
            "prompt_text_g2pw_batch_size": self._profile_value(prompt_g2pw_profiled, "g2pw_batch_size"),
            "prompt_text_g2pw_batch_groups": self._profile_value(prompt_g2pw_profiled, "g2pw_batch_groups"),
            "prompt_text_g2pw_batch_chars": self._profile_value(prompt_g2pw_profiled, "g2pw_batch_chars"),
            "text_g2pw_queue_ms": float(target_g2pw_profiled.queue_ms),
            "text_g2pw_run_ms": float(target_g2pw_profiled.run_ms),
            "text_g2pw_prepare_ms": self._profile_value(target_g2pw_profiled, "g2pw_prepare_ms"),
            "text_g2pw_predict_ms": self._profile_value(target_g2pw_profiled, "g2pw_predict_ms"),
            "text_g2pw_post_ms": self._profile_value(target_g2pw_profiled, "g2pw_post_ms"),
            "text_g2pw_wait_ms": self._profile_value(target_g2pw_profiled, "g2pw_wait_ms"),
            "text_g2pw_admission_wait_ms": self._profile_value(target_g2pw_profiled, "g2pw_admission_wait_ms"),
            "text_g2pw_worker_queue_wait_ms": self._profile_value(target_g2pw_profiled, "g2pw_worker_queue_wait_ms"),
            "text_g2pw_batch_collect_wait_ms": self._profile_value(target_g2pw_profiled, "g2pw_batch_collect_wait_ms"),
            "text_g2pw_batch_dispatch_delay_ms": self._profile_value(target_g2pw_profiled, "g2pw_batch_dispatch_delay_ms"),
            "text_g2pw_batch_size": self._profile_value(target_g2pw_profiled, "g2pw_batch_size"),
            "text_g2pw_batch_groups": self._profile_value(target_g2pw_profiled, "g2pw_batch_groups"),
            "text_g2pw_batch_chars": self._profile_value(target_g2pw_profiled, "g2pw_batch_chars"),
            "prompt_text_parallel_future_wait_ms": 0.0,
            "prompt_text_parallel_future_executor_queue_ms": 0.0,
            "prompt_text_parallel_future_run_ms": 0.0,
            "prompt_text_parallel_future_finish_after_submit_ms": 0.0,
            "prompt_text_parallel_future_queue_tail_after_target_ms": 0.0,
            "prompt_text_parallel_future_run_tail_after_target_ms": 0.0,
            "prompt_text_cpu_queue_ms": float(cpu_stage.prompt_cpu_profiled.queue_ms),
            "prompt_text_cpu_run_ms": float(cpu_stage.prompt_cpu_profiled.run_ms),
            "prompt_text_cpu_admission_wait_ms": self._profile_value(cpu_stage.prompt_cpu_profiled, "text_cpu_admission_wait_ms"),
            "prompt_text_cpu_backpressure_wait_ms": self._profile_value(cpu_stage.prompt_cpu_profiled, "text_cpu_backpressure_wait_ms"),
            "prompt_text_cpu_capacity_wait_ms": self._profile_value(cpu_stage.prompt_cpu_profiled, "text_cpu_capacity_wait_ms"),
            "prompt_text_feature_queue_ms": float(prompt_feature_profiled.queue_ms),
            "prompt_text_feature_run_ms": float(prompt_feature_profiled.run_ms),
            "text_cpu_queue_ms": float(cpu_stage.target_cpu_profiled.queue_ms),
            "text_cpu_run_ms": float(cpu_stage.target_cpu_profiled.run_ms),
            "text_cpu_admission_wait_ms": self._profile_value(cpu_stage.target_cpu_profiled, "text_cpu_admission_wait_ms"),
            "text_cpu_backpressure_wait_ms": self._profile_value(cpu_stage.target_cpu_profiled, "text_cpu_backpressure_wait_ms"),
            "text_cpu_capacity_wait_ms": self._profile_value(cpu_stage.target_cpu_profiled, "text_cpu_capacity_wait_ms"),
            "text_feature_queue_ms": float(target_feature_profiled.queue_ms),
            "text_feature_run_ms": float(target_feature_profiled.run_ms),
            "ref_audio_task_queue_ms": float(ref_audio_profiled.queue_ms),
            "ref_audio_task_run_ms": float(ref_audio_profiled.run_ms),
            "worker_prepare_inflight_on_enter": float(cpu_stage.current_inflight),
            "worker_prepare_peak_inflight": float(cpu_stage.peak_inflight),
        }
        if extra_profile:
            profile_overrides.update({key: float(value) for key, value in extra_profile.items()})
        return profile_overrides

    def _build_ref_audio_bundle_from_phase(
        self,
        phase_one: GPTSoVITSPrepareAudioPhaseData,
        *,
        ref_spec_result: GPTSoVITSPrepareRefSpecResult | None = None,
    ) -> GPTSoVITSRefAudioBundle:
        ref_audio_profiled = phase_one.ref_audio_profiled
        ref_audio_result = ref_audio_profiled.result
        ref_audio_profile = dict(ref_audio_result.profile or {})
        if ref_spec_result is not None:
            ref_audio_profile.update(
                {
                    "ref_spec_wait_ms": float(ref_spec_result.profile.get("ref_spec_wait_ms", 0.0)),
                    "ref_spec_ms": float(ref_spec_result.profile.get("ref_spec_ms", 0.0)),
                    "ref_spec_to_device_ms": float(ref_spec_result.profile.get("ref_spec_to_device_ms", 0.0)),
                    "ref_spec_main_resample_ms": float(ref_spec_result.profile.get("ref_spec_main_resample_ms", 0.0)),
                    "ref_spec_norm_ms": float(ref_spec_result.profile.get("ref_spec_norm_ms", 0.0)),
                    "ref_spec_spectrogram_ms": float(ref_spec_result.profile.get("ref_spec_spectrogram_ms", 0.0)),
                    "ref_spec_post_resample_ms": float(ref_spec_result.profile.get("ref_spec_post_resample_ms", 0.0)),
                }
            )
        else:
            ref_audio_profile.setdefault("ref_spec_wait_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_to_device_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_main_resample_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_norm_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_spectrogram_ms", 0.0)
            ref_audio_profile.setdefault("ref_spec_post_resample_ms", 0.0)
        return GPTSoVITSRefAudioBundle(
            prompt_semantic=ref_audio_result.prompt_semantic,
            raw_audio=ref_audio_result.raw_audio,
            raw_sr=int(ref_audio_result.raw_sr),
            refer_spec=(None if ref_spec_result is None else self._coerce_refer_spec(ref_spec_result.refer_spec)),
            profile=ref_audio_profile,
        )

    def _build_request_state_native(
        self,
        *,
        pipeline: Any,
        spec: GPTSoVITSRequestSpec,
        prompt_text: str,
        text: str,
        prompt_result: Any,
        target_result: Any,
        ref_audio_bundle: GPTSoVITSRefAudioBundle,
        prepare_start: float,
        prepare_sync_start: float,
        profile_overrides: dict[str, float] | None = None,
    ) -> GPTSoVITST2SRequestState:
        device = pipeline.configs.device
        _sync_runtime_device(device)
        ref_audio_bundle_ms = float(ref_audio_bundle.profile.get("bundle_total_ms", 0.0))
        bundle_profile = ref_audio_bundle.profile
        prompt_semantic = ref_audio_bundle.prompt_semantic.long()
        refer_spec_value = ref_audio_bundle.refer_spec
        if refer_spec_value is None:
            spec_audio, audio_16k = None, None
        else:
            spec_audio = refer_spec_value.spec_audio
            audio_16k = refer_spec_value.audio_16k
        aux_refer_specs: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        for aux_ref_audio_path in list(getattr(spec, "aux_ref_audio_paths", []) or []):
            if aux_ref_audio_path in [None, ""]:
                continue
            if not os.path.exists(str(aux_ref_audio_path)):
                continue
            aux_refer_spec, _aux_raw_audio, _aux_raw_sr = self._extract_ref_spec_native(str(aux_ref_audio_path))
            aux_refer_specs.append((aux_refer_spec.spec_audio, aux_refer_spec.audio_16k))
        raw_audio = ref_audio_bundle.raw_audio
        raw_sr = int(ref_audio_bundle.raw_sr)
        prompt_semantic_ms = float(bundle_profile.get("prompt_semantic_ms", ref_audio_bundle_ms))
        ref_spec_ms = float(bundle_profile.get("ref_spec_ms", 0.0))
        audio_load_ms = float(bundle_profile.get("audio_load_ms", 0.0))

        _sync_runtime_device(device)
        tensorize_start = time.perf_counter()
        phones_tensor = torch.LongTensor(target_result.phones).to(pipeline.configs.device)
        prompt_phones_tensor = torch.LongTensor(prompt_result.phones).to(pipeline.configs.device)
        all_phones = torch.LongTensor(prompt_result.phones + target_result.phones).to(pipeline.configs.device)
        prompt_bert_features = prompt_result.bert_features.to(dtype=pipeline.precision, device=pipeline.configs.device)
        target_bert_features = target_result.bert_features.to(dtype=pipeline.precision, device=pipeline.configs.device)
        all_bert_features = torch.cat([prompt_bert_features, target_bert_features], dim=1)
        _sync_runtime_device(device)
        tensorize_ms = (time.perf_counter() - tensorize_start) * 1000.0

        prepare_profile = {
            "prompt_text_features_ms": float(prompt_result.total_ms),
            "text_features_ms": float(target_result.total_ms),
            "prompt_text_cpu_preprocess_ms": float(prompt_result.cpu_preprocess_ms),
            "text_cpu_preprocess_ms": float(target_result.cpu_preprocess_ms),
            "prompt_text_bert_wait_ms": float(prompt_result.profile.get("bert_wait_ms", 0.0)),
            "prompt_text_bert_admission_wait_ms": float(prompt_result.profile.get("bert_admission_wait_ms", 0.0)),
            "prompt_text_bert_queue_wait_ms": float(prompt_result.profile.get("bert_queue_wait_ms", 0.0)),
            "prompt_text_bert_worker_queue_wait_ms": float(prompt_result.profile.get("bert_worker_queue_wait_ms", 0.0)),
            "prompt_text_bert_submit_offset_first_ms": float(prompt_result.profile.get("bert_submit_offset_first_ms", 0.0)),
            "prompt_text_bert_submit_offset_last_ms": float(prompt_result.profile.get("bert_submit_offset_last_ms", 0.0)),
            "prompt_text_bert_batch_collect_wait_ms": float(prompt_result.profile.get("bert_batch_collect_wait_ms", 0.0)),
            "prompt_text_bert_batch_dispatch_delay_ms": float(
                prompt_result.profile.get("bert_batch_dispatch_delay_ms", 0.0)
            ),
            "prompt_text_bert_forward_ms": float(prompt_result.profile.get("bert_forward_ms", 0.0)),
            "prompt_text_bert_tokenize_ms": float(prompt_result.profile.get("bert_tokenize_ms", 0.0)),
            "prompt_text_bert_scatter_ms": float(prompt_result.profile.get("bert_scatter_ms", 0.0)),
            "prompt_text_bert_calls": float(prompt_result.profile.get("bert_calls", 0.0)),
            "prompt_text_bert_stage_slots": float(prompt_result.profile.get("bert_stage_slots", 0.0)),
            "prompt_text_bert_stage_inflight_peak": float(prompt_result.profile.get("bert_stage_inflight_peak", 0.0)),
            "prompt_text_bert_batch_size_peak": float(prompt_result.profile.get("bert_batch_size_peak", 0.0)),
            "prompt_text_bert_batch_tokens_peak": float(prompt_result.profile.get("bert_batch_tokens_peak", 0.0)),
            "prompt_text_bert_pending_depth_on_enqueue_peak": float(
                prompt_result.profile.get("bert_pending_depth_on_enqueue_peak", 0.0)
            ),
            "prompt_text_bert_pending_depth_on_collect_peak": float(
                prompt_result.profile.get("bert_pending_depth_on_collect_peak", 0.0)
            ),
            "prompt_text_bert_high_pressure_mode_peak": float(
                prompt_result.profile.get("bert_high_pressure_mode_peak", 0.0)
            ),
            "prompt_text_bert_batch_window_ms": float(prompt_result.profile.get("bert_batch_window_ms", 0.0)),
            "prompt_text_g2pw_total_ms": float(prompt_result.profile.get("g2pw_total_ms", 0.0)),
            "prompt_text_g2pw_prepare_ms": float(prompt_result.profile.get("g2pw_prepare_ms", 0.0)),
            "prompt_text_g2pw_predict_ms": float(prompt_result.profile.get("g2pw_predict_ms", 0.0)),
            "prompt_text_g2pw_post_ms": float(prompt_result.profile.get("g2pw_post_ms", 0.0)),
            "prompt_text_g2pw_runtime_total_ms": float(prompt_result.profile.get("g2pw_runtime_total_ms", 0.0)),
            "prompt_text_g2pw_runtime_queue_wait_ms": float(
                prompt_result.profile.get("g2pw_runtime_queue_wait_ms", 0.0)
            ),
            "prompt_text_g2pw_runtime_collect_wait_ms": float(
                prompt_result.profile.get("g2pw_runtime_collect_wait_ms", 0.0)
            ),
            "prompt_text_g2pw_runtime_run_ms": float(prompt_result.profile.get("g2pw_runtime_run_ms", 0.0)),
            "prompt_text_g2pw_runtime_batch_rows_peak": float(
                prompt_result.profile.get("g2pw_runtime_batch_rows_peak", 0.0)
            ),
            "prompt_text_g2pw_runtime_batch_requests_peak": float(
                prompt_result.profile.get("g2pw_runtime_batch_requests_peak", 0.0)
            ),
            "prompt_text_parallel_future_wait_ms": 0.0,
            "prompt_text_parallel_future_executor_queue_ms": 0.0,
            "prompt_text_parallel_future_run_ms": float(prompt_result.total_ms),
            "prompt_text_parallel_future_finish_after_submit_ms": float(prompt_result.total_ms),
            "prompt_text_parallel_future_queue_tail_after_target_ms": 0.0,
            "prompt_text_parallel_future_run_tail_after_target_ms": 0.0,
            "text_bert_wait_ms": float(target_result.profile.get("bert_wait_ms", 0.0)),
            "text_bert_admission_wait_ms": float(target_result.profile.get("bert_admission_wait_ms", 0.0)),
            "text_bert_queue_wait_ms": float(target_result.profile.get("bert_queue_wait_ms", 0.0)),
            "text_bert_worker_queue_wait_ms": float(target_result.profile.get("bert_worker_queue_wait_ms", 0.0)),
            "text_bert_submit_offset_first_ms": float(target_result.profile.get("bert_submit_offset_first_ms", 0.0)),
            "text_bert_submit_offset_last_ms": float(target_result.profile.get("bert_submit_offset_last_ms", 0.0)),
            "text_bert_batch_collect_wait_ms": float(target_result.profile.get("bert_batch_collect_wait_ms", 0.0)),
            "text_bert_batch_dispatch_delay_ms": float(target_result.profile.get("bert_batch_dispatch_delay_ms", 0.0)),
            "text_bert_forward_ms": float(target_result.profile.get("bert_forward_ms", 0.0)),
            "text_bert_tokenize_ms": float(target_result.profile.get("bert_tokenize_ms", 0.0)),
            "text_bert_scatter_ms": float(target_result.profile.get("bert_scatter_ms", 0.0)),
            "text_bert_calls": float(target_result.profile.get("bert_calls", 0.0)),
            "text_bert_stage_slots": float(target_result.profile.get("bert_stage_slots", 0.0)),
            "text_bert_stage_inflight_peak": float(target_result.profile.get("bert_stage_inflight_peak", 0.0)),
            "text_bert_batch_size_peak": float(target_result.profile.get("bert_batch_size_peak", 0.0)),
            "text_bert_batch_tokens_peak": float(target_result.profile.get("bert_batch_tokens_peak", 0.0)),
            "text_bert_pending_depth_on_enqueue_peak": float(
                target_result.profile.get("bert_pending_depth_on_enqueue_peak", 0.0)
            ),
            "text_bert_pending_depth_on_collect_peak": float(
                target_result.profile.get("bert_pending_depth_on_collect_peak", 0.0)
            ),
            "text_bert_high_pressure_mode_peak": float(target_result.profile.get("bert_high_pressure_mode_peak", 0.0)),
            "text_bert_batch_window_ms": float(target_result.profile.get("bert_batch_window_ms", 0.0)),
            "text_g2pw_total_ms": float(target_result.profile.get("g2pw_total_ms", 0.0)),
            "text_g2pw_prepare_ms": float(target_result.profile.get("g2pw_prepare_ms", 0.0)),
            "text_g2pw_predict_ms": float(target_result.profile.get("g2pw_predict_ms", 0.0)),
            "text_g2pw_post_ms": float(target_result.profile.get("g2pw_post_ms", 0.0)),
            "text_g2pw_runtime_total_ms": float(target_result.profile.get("g2pw_runtime_total_ms", 0.0)),
            "text_g2pw_runtime_queue_wait_ms": float(target_result.profile.get("g2pw_runtime_queue_wait_ms", 0.0)),
            "text_g2pw_runtime_collect_wait_ms": float(target_result.profile.get("g2pw_runtime_collect_wait_ms", 0.0)),
            "text_g2pw_runtime_run_ms": float(target_result.profile.get("g2pw_runtime_run_ms", 0.0)),
            "text_g2pw_runtime_batch_rows_peak": float(target_result.profile.get("g2pw_runtime_batch_rows_peak", 0.0)),
            "text_g2pw_runtime_batch_requests_peak": float(
                target_result.profile.get("g2pw_runtime_batch_requests_peak", 0.0)
            ),
            "text_feature_pair_ms": float(max(prompt_result.total_ms, target_result.total_ms)),
            "text_cpu_parallel_workers": float(getattr(pipeline, "prepare_text_cpu_workers", 0)),
            "audio_load_ms": audio_load_ms,
            "audio_stage_wait_ms": float(bundle_profile.get("audio_stage_wait_ms", 0.0)),
            "audio_stage_slots": float(bundle_profile.get("audio_stage_slots", 0.0)),
            "audio_stage_inflight_peak": float(bundle_profile.get("audio_stage_inflight_peak", 0.0)),
            "prompt_semantic_ms": prompt_semantic_ms,
            "prompt_semantic_wait_ms": float(bundle_profile.get("prompt_semantic_wait_ms", 0.0)),
            "prompt_semantic_submit_offset_ms": float(bundle_profile.get("prompt_semantic_submit_offset_ms", 0.0)),
            "prompt_semantic_submit_after_load_ms": float(
                bundle_profile.get("prompt_semantic_submit_after_load_ms", 0.0)
            ),
            "prompt_semantic_cpu_prepare_wait_ms": float(
                bundle_profile.get("prompt_semantic_cpu_prepare_wait_ms", 0.0)
            ),
            "prompt_semantic_cpu_prepare_slots": float(
                bundle_profile.get("prompt_semantic_cpu_prepare_slots", 0.0)
            ),
            "prompt_semantic_cpu_prepare_inflight_peak": float(
                bundle_profile.get("prompt_semantic_cpu_prepare_inflight_peak", 0.0)
            ),
            "prompt_semantic_worker_queue_wait_ms": float(
                bundle_profile.get("prompt_semantic_worker_queue_wait_ms", 0.0)
            ),
            "prompt_semantic_batch_collect_wait_ms": float(
                bundle_profile.get("prompt_semantic_batch_collect_wait_ms", 0.0)
            ),
            "prompt_semantic_stage_limiter_wait_ms": float(
                bundle_profile.get("prompt_semantic_stage_limiter_wait_ms", 0.0)
            ),
            "prompt_semantic_batch_dispatch_delay_ms": float(
                bundle_profile.get("prompt_semantic_batch_dispatch_delay_ms", 0.0)
            ),
            "prompt_semantic_cpu_prepare_ms": float(bundle_profile.get("prompt_semantic_cpu_prepare_ms", 0.0)),
            "prompt_semantic_pack_ms": float(bundle_profile.get("prompt_semantic_pack_ms", 0.0)),
            "prompt_semantic_h2d_ms": float(bundle_profile.get("prompt_semantic_h2d_ms", 0.0)),
            "prompt_semantic_ssl_forward_ms": float(bundle_profile.get("prompt_semantic_ssl_forward_ms", 0.0)),
            "prompt_semantic_hidden_length_ms": float(
                bundle_profile.get("prompt_semantic_hidden_length_ms", 0.0)
            ),
            "prompt_semantic_extract_latent_ms": float(
                bundle_profile.get("prompt_semantic_extract_latent_ms", 0.0)
            ),
            "prompt_semantic_forward_ms": float(bundle_profile.get("prompt_semantic_forward_ms", 0.0)),
            "prompt_semantic_scatter_ms": float(bundle_profile.get("prompt_semantic_scatter_ms", 0.0)),
            "prompt_semantic_stage_slots": float(bundle_profile.get("prompt_semantic_stage_slots", 0.0)),
            "prompt_semantic_stage_inflight_peak": float(
                bundle_profile.get("prompt_semantic_stage_inflight_peak", 0.0)
            ),
            "prompt_semantic_batch_size": float(bundle_profile.get("prompt_semantic_batch_size", 0.0)),
            "prompt_semantic_batch_samples": float(bundle_profile.get("prompt_semantic_batch_samples", 0.0)),
            "prompt_semantic_padded_batch_samples": float(
                bundle_profile.get("prompt_semantic_padded_batch_samples", 0.0)
            ),
            "prompt_semantic_batch_pad_ratio": float(bundle_profile.get("prompt_semantic_batch_pad_ratio", 0.0)),
            "prompt_semantic_pool_bucket_index": float(
                bundle_profile.get("prompt_semantic_pool_bucket_index", 0.0)
            ),
            "prompt_semantic_pool_workers": float(bundle_profile.get("prompt_semantic_pool_workers", 0.0)),
            "prompt_semantic_bucket_first_hit_serialized": float(
                bundle_profile.get("prompt_semantic_bucket_first_hit_serialized", 0.0)
            ),
            "prompt_semantic_runtime_exact_prewarm_applied": float(
                bundle_profile.get("prompt_semantic_runtime_exact_prewarm_applied", 0.0)
            ),
            "prompt_semantic_runtime_exact_prewarm_ms": float(
                bundle_profile.get("prompt_semantic_runtime_exact_prewarm_ms", 0.0)
            ),
            "prompt_semantic_runtime_exact_prewarm_target_samples": float(
                bundle_profile.get("prompt_semantic_runtime_exact_prewarm_target_samples", 0.0)
            ),
            "prompt_semantic_runtime_exact_prewarm_batch_sizes": float(
                bundle_profile.get("prompt_semantic_runtime_exact_prewarm_batch_sizes", 0.0)
            ),
            "prompt_semantic_runtime_exact_prewarm_skipped_capacity": float(
                bundle_profile.get("prompt_semantic_runtime_exact_prewarm_skipped_capacity", 0.0)
            ),
            "prompt_semantic_shard_index": float(bundle_profile.get("prompt_semantic_shard_index", 0.0)),
            "ref_spec_wait_ms": float(bundle_profile.get("ref_spec_wait_ms", 0.0)),
            "ref_spec_ms": ref_spec_ms,
            "ref_spec_to_device_ms": float(bundle_profile.get("ref_spec_to_device_ms", 0.0)),
            "ref_spec_main_resample_ms": float(bundle_profile.get("ref_spec_main_resample_ms", 0.0)),
            "ref_spec_norm_ms": float(bundle_profile.get("ref_spec_norm_ms", 0.0)),
            "ref_spec_spectrogram_ms": float(bundle_profile.get("ref_spec_spectrogram_ms", 0.0)),
            "ref_spec_post_resample_ms": float(bundle_profile.get("ref_spec_post_resample_ms", 0.0)),
            "ref_audio_bundle_ms": ref_audio_bundle_ms,
            "tensorize_ms": tensorize_ms,
            "total_ms": (time.perf_counter() - prepare_sync_start) * 1000.0,
            "wall_total_ms": (time.perf_counter() - prepare_start) * 1000.0,
        }
        if profile_overrides:
            prepare_profile.update({key: float(value) for key, value in profile_overrides.items()})
        return GPTSoVITST2SRequestState(
            request_id=str(spec.request_id),
            ref_audio_path=spec.ref_audio_path,
            prompt_text=str(prompt_text),
            prompt_lang=str(spec.prompt_lang),
            text=str(text),
            text_lang=str(spec.text_lang),
            norm_prompt_text=str(prompt_result.norm_text),
            norm_text=str(target_result.norm_text),
            phones=phones_tensor,
            prompt_phones=prompt_phones_tensor,
            all_phones=all_phones,
            all_bert_features=all_bert_features,
            prompt_semantic=prompt_semantic,
            refer_spec=(
                None
                if spec_audio is None
                else GPTSoVITSReferSpec(spec_audio=spec_audio, audio_16k=audio_16k)
            ),
            aux_refer_specs=aux_refer_specs,
            raw_audio=raw_audio,
            raw_sr=raw_sr,
            top_k=int(spec.top_k),
            top_p=float(spec.top_p),
            temperature=float(spec.temperature),
            repetition_penalty=float(spec.repetition_penalty),
            early_stop_num=int(spec.early_stop_num),
            ready_step=int(spec.ready_step),
            prepare_profile=prepare_profile,
        )

    def _build_request_state_from_prepare_phases(
        self,
        prepared_cpu_stage: GPTSoVITSPreparedCpuStage,
        phase_one: GPTSoVITSPrepareAudioPhaseData,
        phase_two: GPTSoVITSPrepareTextPhaseData,
        *,
        ref_spec_result: GPTSoVITSPrepareRefSpecResult | None = None,
        extra_profile: dict[str, float] | None = None,
    ) -> Any:
        pipeline = self._ensure_pipeline()
        profile_overrides = self._build_prepare_profile_overrides(
            prepared_cpu_stage,
            phase_one,
            phase_two,
            extra_profile=extra_profile,
        )
        ref_audio_bundle = self._build_ref_audio_bundle_from_phase(
            phase_one,
            ref_spec_result=ref_spec_result,
        )
        state = self._build_request_state_native(
            pipeline=pipeline,
            spec=prepared_cpu_stage.spec,
            prompt_text=prepared_cpu_stage.prompt_text,
            text=prepared_cpu_stage.text,
            prompt_result=phase_two.prompt_feature_profiled.result,
            target_result=phase_two.target_feature_profiled.result,
            ref_audio_bundle=ref_audio_bundle,
            prepare_start=prepared_cpu_stage.prepare_start,
            prepare_sync_start=prepared_cpu_stage.prepare_start,
            profile_overrides=profile_overrides,
        )
        prepare_exec_finished_at = time.perf_counter()
        state.prepare_profile["executor_run_wall_ms"] = max(
            0.0,
            (prepare_exec_finished_at - prepared_cpu_stage.prepare_start) * 1000.0,
        )
        return state

    def _release_prepare_split_stage_slot(self, coordinator: Any) -> None:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        coordinator.release_split_stage_slot()

    def build_prepared_request_from_phases(
        self,
        prepared_text_phase: GPTSoVITSPreparedTextPhase,
        prepared_ref_spec_phase: GPTSoVITSPreparedRefSpecPhase | None = None,
        *,
        extra_profile: dict[str, float] | None = None,
    ) -> GPTSoVITSPreparedRequest:
        coordinator = self._ensure_prepare_coordinator()
        prepared_cpu_stage = prepared_text_phase.prepared_audio_phase.prepared_cpu_stage
        try:
            state = self._build_request_state_from_prepare_phases(
                prepared_cpu_stage,
                prepared_text_phase.prepared_audio_phase.phase_one,
                prepared_text_phase.phase_two,
                ref_spec_result=(
                    prepared_ref_spec_phase.ref_spec_result if prepared_ref_spec_phase is not None else None
                ),
                extra_profile=extra_profile,
            )
        finally:
            self._release_prepare_split_stage_slot(coordinator)
        return GPTSoVITSPreparedRequest(
            request_id=prepared_text_phase.request_id,
            state=state,
            transport_info=self._state_to_transport_info(state, prepared_cpu_stage.spec),
        )

    def prepare_request(self, request: dict[str, Any], *, request_id: str | None = None) -> GPTSoVITSPreparedRequest:
        spec = self._build_scheduler_request_spec(request, request_id=request_id)
        return self.prepare_request_spec(spec)

    def prepare_request_spec(self, spec: GPTSoVITSRequestSpec) -> GPTSoVITSPreparedRequest:
        prepared_cpu_stage = self.prepare_request_spec_cpu_stage(spec)
        prepared_audio_phase = self.prepare_request_gpu_audio_phase(prepared_cpu_stage)
        prepared_ref_spec_phase = self.prepare_request_ref_spec_phase(prepared_audio_phase)
        prepared_text_phase = self.prepare_request_gpu_text_phase(prepared_audio_phase)
        return self.build_prepared_request_from_phases(prepared_text_phase, prepared_ref_spec_phase=prepared_ref_spec_phase)

    def prepare_requests(self, requests: list[dict[str, Any]]) -> list[GPTSoVITSPreparedRequest]:
        if not requests:
            return []
        specs = [
            self._build_scheduler_request_spec(
                request,
                request_id=str(request.get("engine_request_id") or f"gpt_sovits_{index}"),
            )
            for index, request in enumerate(requests)
        ]
        return self.prepare_request_specs(specs)

    def prepare_request_specs(self, specs: list[GPTSoVITSRequestSpec]) -> list[GPTSoVITSPreparedRequest]:
        if not specs:
            return []
        prepared_cpu_stages = self.prepare_request_spec_cpu_stages(specs)
        audio_phase_start = time.perf_counter()
        prepared_audio_phases = self.prepare_request_gpu_audio_phases(prepared_cpu_stages)
        audio_phase_wall_ms = max(0.0, (time.perf_counter() - audio_phase_start) * 1000.0)
        prepared_ref_spec_phases = self.prepare_request_ref_spec_phases(prepared_audio_phases)
        text_phase_start = time.perf_counter()
        prepared_text_phases = self.prepare_request_gpu_text_phases(prepared_audio_phases)
        text_phase_wall_ms = max(0.0, (time.perf_counter() - text_phase_start) * 1000.0)
        if len(prepared_text_phases) != len(prepared_ref_spec_phases):
            raise ValueError("GPT-SoVITS batch prepare text/ref-spec phase count mismatch")

        prepared: list[GPTSoVITSPreparedRequest] = []
        extra_profile = {
            "engine_prepare_audio_phase_mode": 1.0,
            "engine_prepare_audio_phase_wall_ms": float(audio_phase_wall_ms),
            "engine_prepare_audio_phase_batch_size": float(len(prepared_audio_phases)),
            "engine_prepare_text_phase_wall_ms": float(text_phase_wall_ms),
            "engine_prepare_text_phase_batch_size": float(len(prepared_text_phases)),
        }
        for prepared_text_phase, prepared_ref_spec_phase in zip(prepared_text_phases, prepared_ref_spec_phases):
            prepared.append(
                self.build_prepared_request_from_phases(
                    prepared_text_phase,
                    prepared_ref_spec_phase=prepared_ref_spec_phase,
                    extra_profile=extra_profile,
                )
            )
        return prepared

    def _build_ar_session_from_prepared(self, prepared: GPTSoVITSPreparedRequest) -> GPTSoVITSARSession:
        pipeline = self._ensure_pipeline()
        model = pipeline.t2s_model.model
        active_batch = self._build_prefill_active_batch(model, [prepared.state])
        if active_batch.prefill_attn_mask is None or active_batch.key_padding_mask is None:
            raise ValueError("GPT-SoVITS AR prefill batch is missing attention masks")

        xy_dec, active_batch.k_cache, active_batch.v_cache = model.t2s_transformer.process_prompt(
            active_batch.xy_pos,
            active_batch.prefill_attn_mask,
            None,
        )
        active_batch.kv_lens = active_batch.x_lens + active_batch.prefix_lens
        if active_batch.k_cache is None or active_batch.v_cache is None or active_batch.kv_lens is None:
            raise ValueError("GPT-SoVITS AR prefill did not produce KV cache")

        packed_into_pool = bool(self._pack_active_batch_into_pool(model, active_batch))
        if not packed_into_pool:
            active_batch.decode_attn_mask = F.pad(
                active_batch.key_padding_mask.unsqueeze(1).unsqueeze(1),
                (0, 1),
                value=False,
            )
            active_batch.k_cache = [
                self._compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.k_cache
            ]
            active_batch.v_cache = [
                self._compact_cache_to_kv_lens(layer, active_batch.kv_lens) for layer in active_batch.v_cache
            ]
            active_batch.decode_attn_mask = self._compact_decode_mask_to_kv_lens(
                active_batch.decode_attn_mask,
                active_batch.kv_lens,
            )

        active_batch.x = None
        active_batch.x_lens = None
        active_batch.key_padding_mask = None
        active_batch.prefill_attn_mask = None
        active_batch.prefill_done = True
        active_batch.xy_pos = self._build_next_xy_pos(model, active_batch.y_sequences)
        current_logits = model.ar_predict_layer(xy_dec[:, -1]).detach()

        return GPTSoVITSARSession(
            request_id=prepared.request_id,
            active_batch=active_batch,
            transport_info=prepared.transport_info,
            current_logits=current_logits,
        )

    def start_ar_session(self, request: dict[str, Any], *, request_id: str | None = None) -> GPTSoVITSARSession:
        spec = self._build_scheduler_request_spec(
            request,
            request_id=str(request_id or request.get("engine_request_id") or f"gpt_sovits_ar_{time.time_ns()}"),
        )
        return self.start_ar_session_from_spec(spec)

    def start_ar_session_from_spec(self, spec: GPTSoVITSRequestSpec) -> GPTSoVITSARSession:
        with self._run_lock:
            with torch.inference_mode(False), torch.no_grad():
                prepared = self.prepare_request_spec(spec)
                return self._build_ar_session_from_prepared(prepared)

    def advance_ar_session(self, session: GPTSoVITSARSession, sampled_token_id: int) -> torch.Tensor:
        pipeline = self._ensure_pipeline()
        token_value = int(sampled_token_id)
        with self._run_lock:
            with torch.inference_mode(False), torch.no_grad():
                model = pipeline.t2s_model.model
                active_batch = session.active_batch
                device = active_batch.y_sequences[0].device
                token_tensor = torch.tensor([token_value], device=device, dtype=torch.long)
                active_batch.y_sequences[0] = torch.cat([active_batch.y_sequences[0], token_tensor], dim=0)
                active_batch.step_indices = active_batch.step_indices + 1
                active_batch.xy_pos = self._build_next_xy_pos(model, active_batch.y_sequences)

                if active_batch.k_cache is None or active_batch.v_cache is None or active_batch.kv_lens is None:
                    raise ValueError("GPT-SoVITS AR decode session is missing KV cache")

                if active_batch.kv_cache_pooled:
                    pool = self._get_kv_pool(model)
                    if pool is None:
                        raise ValueError("GPT-SoVITS AR pooled KV cache is unavailable")
                    batched_decode_attn_mask = pool.build_decode_mask(active_batch.kv_lens + 1)
                    xy_dec, active_batch.k_cache, active_batch.v_cache = model.decode_next_token_prealloc_runtime(
                        active_batch.xy_pos,
                        active_batch.k_cache,
                        active_batch.v_cache,
                        active_batch.kv_lens,
                        batched_decode_attn_mask,
                    )
                else:
                    batched_decode_attn_mask = None
                    if active_batch.decode_attn_mask is not None:
                        batched_decode_attn_mask = self._materialize_decode_mask_for_active_batch(active_batch)
                        if not batched_decode_attn_mask.any().item():
                            batched_decode_attn_mask = None
                    xy_dec, active_batch.k_cache, active_batch.v_cache = model.t2s_transformer.decode_next_token(
                        active_batch.xy_pos,
                        active_batch.k_cache,
                        active_batch.v_cache,
                        batched_decode_attn_mask,
                    )
                    active_batch.decode_attn_mask = self._advance_decode_mask(
                        active_batch.decode_attn_mask, active_batch.kv_lens
                    )

                active_batch.kv_lens = active_batch.kv_lens + 1
                session.current_logits = model.ar_predict_layer(xy_dec[:, -1]).detach()
        return session.current_logits

    def get_ar_session_logits(self, session: GPTSoVITSARSession, *, suppress_eos_until_step: int = 11) -> torch.Tensor:
        logits = session.current_logits.detach().clone()
        active_batch = session.active_batch
        if logits.ndim == 2 and logits.shape[-1] > self.get_semantic_eos_id():
            step_idx = int(active_batch.step_indices[0].item()) if active_batch.step_indices.numel() > 0 else 0
            if step_idx < suppress_eos_until_step:
                logits[..., self.get_semantic_eos_id()] = float("-inf")
        return logits

    @staticmethod
    def _active_batch_semantic_tokens(active_batch: Any) -> torch.Tensor:
        if active_batch is None or not getattr(active_batch, "y_sequences", None):
            return torch.zeros((0,), dtype=torch.long)
        if getattr(active_batch, "prefix_lens", None) is None:
            return torch.zeros((0,), dtype=torch.long)
        prefix_len = int(active_batch.prefix_lens[0].item()) if active_batch.prefix_lens.numel() > 0 else 0
        current = active_batch.y_sequences[0]
        if not isinstance(current, torch.Tensor) or current.numel() <= prefix_len:
            return torch.zeros((0,), dtype=torch.long, device=current.device if isinstance(current, torch.Tensor) else None)
        return current[prefix_len:].detach().to(dtype=torch.long).contiguous()

    def get_ar_session_semantic_tokens(self, session: GPTSoVITSARSession) -> torch.Tensor:
        tokens = self._active_batch_semantic_tokens(session.active_batch)
        if tokens.numel() > 0 and int(tokens[-1].item()) == self.get_semantic_eos_id():
            tokens = tokens[:-1]
        return tokens.to("cpu").contiguous()

    def generate_semantic_tokens(
        self,
        prepared_requests: list[GPTSoVITSPreparedRequest],
        *,
        max_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if not prepared_requests:
            return {}
        pipeline = self._ensure_pipeline()
        states = [item.state for item in prepared_requests]
        max_steps = int(max_steps or self._estimate_scheduler_max_steps(states))
        with self._run_lock:
            with torch.inference_mode(False), torch.no_grad():
                finished_items = self._run_continuous_batch_scheduler(
                    pipeline.t2s_model.model,
                    states,
                    max_steps=max_steps,
                )
        return {
            str(item.request_id): item.semantic_tokens.detach().to("cpu").contiguous().to(dtype=torch.long)
            for item in finished_items
        }

    def prepare_decode_request(
        self,
        semantic_tokens: torch.Tensor,
        transport_info: GPTSoVITSStageTransport | dict[str, Any],
    ) -> GPTSoVITSDecodePreparedRequest:
        pipeline = self._ensure_pipeline()
        device = torch.device(getattr(pipeline.configs, "device", "cpu"))
        transport = GPTSoVITSStageTransport.from_info(
            transport_info,
            semantic_tokens=semantic_tokens,
        )
        request_id = str(transport.request_id or f"gpt_sovits_decode_{time.time_ns()}")
        return GPTSoVITSDecodePreparedRequest(
            request_id=request_id,
            semantic_tokens=transport.semantic_tokens.to(device=device, dtype=torch.long),
            phones=transport.phones.to(device=device, dtype=torch.long),
            prompt_phones=transport.prompt_phones.to(device=device, dtype=torch.long),
            prompt_semantic=transport.prompt_semantic.to(device=device, dtype=torch.long),
            refer_audio_spec=transport.refer_audio_spec.to(dtype=torch.float32),
            refer_audio_16k=transport.refer_audio_16k.to(dtype=torch.float32),
            raw_audio=transport.raw_audio.to(dtype=torch.float32),
            raw_sr=int(transport.raw_sr),
            speed_factor=float(transport.speed_factor),
            sample_steps=int(transport.sample_steps),
            super_sampling=bool(transport.super_sampling),
        )

    def prepare_decode_requests(
        self,
        semantic_tokens_list: list[torch.Tensor],
        transport_infos: list[GPTSoVITSStageTransport | dict[str, Any]],
    ) -> list[GPTSoVITSDecodePreparedRequest]:
        if len(semantic_tokens_list) != len(transport_infos):
            raise ValueError("GPT-SoVITS decode prepare batch input count mismatch")
        return [
            self.prepare_decode_request(semantic_tokens, transport_info)
            for semantic_tokens, transport_info in zip(semantic_tokens_list, transport_infos)
        ]

    @staticmethod
    def _build_refer_spec_from_prepared(
        prepared: GPTSoVITSDecodePreparedRequest,
    ) -> GPTSoVITSReferSpec:
        return GPTSoVITSReferSpec(
            spec_audio=prepared.refer_audio_spec,
            audio_16k=(None if prepared.refer_audio_16k.numel() == 0 else prepared.refer_audio_16k),
        )

    def _resample_audio(self, audio: torch.Tensor, sr0: int, sr1: int, device: torch.device) -> torch.Tensor:
        if int(sr0) == int(sr1):
            return audio
        cache = getattr(self, "_resample_transform_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_resample_transform_cache", cache)
        key = f"{int(sr0)}-{int(sr1)}-{device}"
        transform = cache.get(key)
        if transform is None:
            import torchaudio

            transform = torchaudio.transforms.Resample(int(sr0), int(sr1)).to(device)
            cache[key] = transform
        return transform(audio)

    @staticmethod
    def _norm_vocoder_spec(x: torch.Tensor) -> torch.Tensor:
        spec_min = -12.0
        spec_max = 2.0
        return (x - spec_min) / (spec_max - spec_min) * 2 - 1

    @staticmethod
    def _denorm_vocoder_spec(x: torch.Tensor) -> torch.Tensor:
        spec_min = -12.0
        spec_max = 2.0
        return (x + 1) / 2 * (spec_max - spec_min) + spec_min

    def _compute_vocoder_mel(self, audio: torch.Tensor, *, version: str) -> torch.Tensor:
        with self._project_root_cwd():
            from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch

        if version == "v3":
            kwargs = {
                "n_fft": 1024,
                "win_size": 1024,
                "hop_size": 256,
                "num_mels": 100,
                "sampling_rate": 24000,
                "fmin": 0,
                "fmax": None,
                "center": False,
            }
        else:
            kwargs = {
                "n_fft": 1280,
                "win_size": 1280,
                "hop_size": 320,
                "num_mels": 100,
                "sampling_rate": 32000,
                "fmin": 0,
                "fmax": None,
                "center": False,
            }
        return mel_spectrogram_torch(audio, **kwargs)

    def _decode_prepared_request_vocoder_fragment(
        self,
        pipeline: Any,
        prepared: GPTSoVITSDecodePreparedRequest,
        refer_audio_spec: torch.Tensor,
    ) -> tuple[Any, int]:
        prompt_context = self._build_vocoder_prompt_context(
            pipeline,
            prepared,
            refer_audio_spec,
        )
        return self._decode_vocoder_with_prompt_context(pipeline, prepared, prompt_context)

    @staticmethod
    def _tensor_identity_key(value: torch.Tensor | None) -> tuple[Any, ...] | None:
        if value is None:
            return None
        return (
            str(value.device),
            str(value.dtype),
            tuple(int(item) for item in value.shape),
            int(value.data_ptr()),
        )

    def _build_vocoder_prompt_cache_key(
        self,
        prepared: GPTSoVITSDecodePreparedRequest,
        refer_audio_spec: torch.Tensor,
    ) -> tuple[Any, ...]:
        return (
            self._tensor_identity_key(prepared.prompt_semantic),
            self._tensor_identity_key(prepared.prompt_phones),
            self._tensor_identity_key(refer_audio_spec),
            self._tensor_identity_key(prepared.raw_audio),
            int(prepared.raw_sr),
        )

    @staticmethod
    def _default_vocoder_runtime_config(version: str) -> dict[str, int]:
        normalized = str(version or "").strip()
        if normalized == "v3":
            return {
                "sr": 24000,
                "T_ref": 468,
                "T_chunk": 934,
                "upsample_rate": 256,
                "overlapped_len": 12,
            }
        if normalized == "v4":
            return {
                "sr": 48000,
                "T_ref": 500,
                "T_chunk": 1000,
                "upsample_rate": 480,
                "overlapped_len": 12,
            }
        return {
            "sr": 0,
            "T_ref": 0,
            "T_chunk": 0,
            "upsample_rate": 1,
            "overlapped_len": 0,
        }

    def _resolve_vocoder_runtime_config(self, pipeline: Any) -> dict[str, int]:
        version = str(getattr(getattr(pipeline, "configs", None), "version", "") or "")
        if not version:
            version = str(getattr(self._get_runtime_configs(), "version", "") or "")
        config = self._default_vocoder_runtime_config(version)
        configured = dict(getattr(pipeline, "vocoder_configs", {}) or {})
        for key, default_value in config.items():
            value = configured.get(key, default_value)
            if value is None:
                value = default_value
            config[key] = int(value)
        return config

    @staticmethod
    def _run_vocoder_module(vocoder: Any, pred_spec: torch.Tensor) -> torch.Tensor:
        return vocoder(pred_spec)

    def _build_vocoder_prompt_context(
        self,
        pipeline: Any,
        prepared: GPTSoVITSDecodePreparedRequest,
        refer_audio_spec: torch.Tensor,
    ) -> dict[str, Any]:
        self._bind_pipeline_components(pipeline)
        vocoder_config = self._resolve_vocoder_runtime_config(pipeline)
        configs = self._get_runtime_configs()
        precision = self._get_runtime_precision()
        device = torch.device(configs.device)
        prompt_semantic_tokens = prepared.prompt_semantic.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.long)
        prompt_phones = prepared.prompt_phones.unsqueeze(0).to(device=device, dtype=torch.long)
        refer_audio_spec = refer_audio_spec.to(dtype=precision, device=device)

        fea_ref, ge = self._run_vits_vocoder_feature_decode(
            pipeline.vits_model,
            codes=prompt_semantic_tokens,
            text=prompt_phones,
            refer_audio_spec=refer_audio_spec,
        )
        ref_audio = prepared.raw_audio.to(device=device, dtype=torch.float32)
        if ref_audio.ndim == 1:
            ref_audio = ref_audio.unsqueeze(0)
        if ref_audio.shape[0] == 2:
            ref_audio = ref_audio.mean(0).unsqueeze(0)

        target_sr = 24000 if configs.version == "v3" else 32000
        if int(prepared.raw_sr) != target_sr:
            ref_audio = self._resample_audio(ref_audio, int(prepared.raw_sr), target_sr, device)

        mel2 = self._compute_vocoder_mel(ref_audio, version=str(configs.version))
        mel2 = self._norm_vocoder_spec(mel2)
        t_min = min(int(mel2.shape[2]), int(fea_ref.shape[2]))
        mel2 = mel2[:, :, :t_min]
        fea_ref = fea_ref[:, :, :t_min]
        t_ref = int(vocoder_config["T_ref"])
        t_chunk = int(vocoder_config["T_chunk"])
        if t_min > t_ref:
            mel2 = mel2[:, :, -t_ref:]
            fea_ref = fea_ref[:, :, -t_ref:]
            t_min = t_ref
        chunk_len = t_chunk - t_min

        return {
            "refer_audio_spec": refer_audio_spec,
            "fea_ref": fea_ref,
            "ge": ge,
            "mel2": mel2.to(dtype=precision),
            "t_min": int(t_min),
            "chunk_len": int(chunk_len),
            "output_sr": int(vocoder_config["sr"]),
            "vocoder_config": vocoder_config,
        }

    def _decode_vocoder_with_prompt_context(
        self,
        pipeline: Any,
        prepared: GPTSoVITSDecodePreparedRequest,
        prompt_context: dict[str, Any],
    ) -> tuple[Any, int]:
        self._bind_pipeline_components(pipeline)
        device = torch.device(self._get_runtime_configs().device)
        refer_audio_spec = prompt_context["refer_audio_spec"]
        ge = prompt_context["ge"]
        fea_ref_base = prompt_context["fea_ref"]
        mel2_base = prompt_context["mel2"]
        t_min = int(prompt_context["t_min"])
        chunk_len = int(prompt_context["chunk_len"])
        semantic_tokens = prepared.semantic_tokens.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.long)
        phones = prepared.phones.unsqueeze(0).to(device=device, dtype=torch.long)
        fea_todo, ge = self._run_vits_vocoder_feature_decode(
            self._get_runtime_vits_model(),
            codes=semantic_tokens,
            text=phones,
            refer_audio_spec=refer_audio_spec,
            ge=ge,
            speed=float(prepared.speed_factor),
        )

        mel2 = mel2_base.clone()
        fea_ref = fea_ref_base.clone()
        cfm_results: list[torch.Tensor] = []
        idx = 0
        while True:
            fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
            if int(fea_todo_chunk.shape[-1]) == 0:
                break
            idx += chunk_len
            fea = torch.cat([fea_ref, fea_todo_chunk], dim=2).transpose(2, 1)
            cfm_res = self._get_runtime_vits_model().cfm.inference(
                fea,
                torch.LongTensor([int(fea.size(1))]).to(fea.device),
                mel2,
                int(prepared.sample_steps),
                inference_cfg_rate=0,
            )
            cfm_res = cfm_res[:, :, mel2.shape[2] :]
            mel2 = cfm_res[:, :, -t_min:]
            fea_ref = fea_todo_chunk[:, :, -t_min:]
            cfm_results.append(cfm_res)

        cfm_res = torch.cat(cfm_results, dim=2)
        cfm_res = self._denorm_vocoder_spec(cfm_res)
        wav_gen = self._run_vocoder_module(self._get_runtime_vocoder(), cfm_res)
        return wav_gen[0][0], int(prompt_context["output_sr"])

    @staticmethod
    def _sola_merge_audio_fragments(
        audio_fragments: Sequence[torch.Tensor],
        overlap_len: int,
        *,
        search_len: int = 320,
    ) -> torch.Tensor:
        if not audio_fragments:
            return torch.zeros((0,), dtype=torch.float32)
        dtype = audio_fragments[0].dtype
        if len(audio_fragments) == 1 or overlap_len <= 0:
            return torch.cat(list(audio_fragments), dim=0).to(dtype)

        merged_fragments = [fragment for fragment in audio_fragments]
        for index in range(len(merged_fragments) - 1):
            first = merged_fragments[index].float()
            second = merged_fragments[index + 1].float()
            overlap = min(int(overlap_len), int(first.shape[-1]), int(second.shape[-1]))
            if overlap <= 0:
                continue
            available_search = max(int(second.shape[-1]) - overlap, 0)
            current_search = min(int(search_len), available_search)
            window_first = first[-overlap:]
            window_second = second[: overlap + current_search]
            if int(window_second.shape[-1]) < overlap:
                continue

            corr_norm = F.conv1d(window_second.view(1, 1, -1), window_first.view(1, 1, -1)).view(-1)
            corr_den = (
                F.conv1d(window_second.view(1, 1, -1) ** 2, torch.ones_like(window_first).view(1, 1, -1)).view(-1)
                + 1e-8
            )
            offset = int((corr_norm / corr_den.sqrt()).argmax().item())

            merged_fragments[index] = first[:-overlap]
            aligned_second = second[offset:]
            if int(aligned_second.shape[-1]) < overlap:
                overlap = int(aligned_second.shape[-1])
                if overlap <= 0:
                    merged_fragments[index + 1] = aligned_second
                    continue
                window_first = first[-overlap:]
            window = torch.hann_window(overlap * 2, device=first.device, dtype=first.dtype)
            aligned_second[:overlap] = (
                window[:overlap] * aligned_second[:overlap]
                + window[overlap:] * window_first
            )
            merged_fragments[index + 1] = aligned_second
        return torch.cat(merged_fragments, dim=0).to(dtype)

    def _decode_prepared_requests_batched_vocoder(
        self,
        pipeline: Any,
        prepared_requests: list[GPTSoVITSDecodePreparedRequest],
        prompt_context: dict[str, Any],
    ) -> list[GPTSoVITSDecodedAudio]:
        self._bind_pipeline_components(pipeline)
        if not prepared_requests:
            return []

        first_speed = float(prepared_requests[0].speed_factor)
        first_sample_steps = int(prepared_requests[0].sample_steps)
        if any(abs(float(item.speed_factor) - first_speed) > 1e-6 for item in prepared_requests):
            raise ValueError("GPT-SoVITS batched vocoder decode requires identical speed_factor")
        if any(int(item.sample_steps) != first_sample_steps for item in prepared_requests):
            raise ValueError("GPT-SoVITS batched vocoder decode requires identical sample_steps")

        chunk_len = int(prompt_context["chunk_len"])
        vocoder_config = dict(prompt_context.get("vocoder_config", {}) or {})
        overlapped_len = int(vocoder_config.get("overlapped_len", 0))
        upsample_rate = int(vocoder_config.get("upsample_rate", 1))
        if chunk_len <= 0 or overlapped_len < 0 or upsample_rate <= 0:
            return [
                GPTSoVITSDecodedAudio(
                    request_id=prepared.request_id,
                    audio_fragment=audio_fragment,
                    output_sr=int(output_sr),
                    speed_factor=float(prepared.speed_factor),
                    super_sampling=bool(prepared.super_sampling),
                )
                for prepared, (audio_fragment, output_sr) in (
                    (
                        prepared,
                        self._decode_vocoder_with_prompt_context(
                            pipeline,
                            prepared,
                            prompt_context,
                        ),
                    )
                    for prepared in prepared_requests
                )
            ]

        device = torch.device(self._get_runtime_configs().device)
        refer_audio_spec = prompt_context["refer_audio_spec"]
        ge = prompt_context["ge"]
        fea_ref = prompt_context["fea_ref"]
        mel2 = prompt_context["mel2"]

        feat_chunks: list[torch.Tensor] = []
        feat_lens: list[int] = []
        feats: list[torch.Tensor] = []
        for prepared in prepared_requests:
            semantic_tokens = prepared.semantic_tokens.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.long)
            phones = prepared.phones.unsqueeze(0).to(device=device, dtype=torch.long)
            feat, _ = self._run_vits_vocoder_feature_decode(
                self._get_runtime_vits_model(),
                codes=semantic_tokens,
                text=phones,
                refer_audio_spec=refer_audio_spec,
                ge=ge,
                speed=first_speed,
            )
            feats.append(feat)
            feat_lens.append(int(feat.shape[2]))

        merged_feats = torch.cat(feats, dim=2)
        padded_feats = F.pad(merged_feats, (overlapped_len, 0), "constant", 0)
        position = 0
        padding_len = 0
        while True:
            if position != 0:
                position -= overlapped_len
            chunk = padded_feats[:, :, position : position + chunk_len]
            position += chunk_len
            if int(chunk.shape[-1]) == 0:
                break
            padding_len = chunk_len - int(chunk.shape[2])
            if padding_len > 0:
                chunk = F.pad(chunk, (0, padding_len), "constant", 0)
            feat_chunks.append(chunk)

        if not feat_chunks:
            output_sr = int(prompt_context["output_sr"])
            return [
                GPTSoVITSDecodedAudio(
                    request_id=prepared.request_id,
                    audio_fragment=torch.zeros((0,), dtype=torch.float32, device=device),
                    output_sr=output_sr,
                    speed_factor=float(prepared.speed_factor),
                    super_sampling=bool(prepared.super_sampling),
                )
                for prepared in prepared_requests
            ]

        batched_feat_chunks = torch.cat(feat_chunks, dim=0)
        batch_size = int(batched_feat_chunks.shape[0])
        expanded_ref = fea_ref.repeat(batch_size, 1, 1)
        expanded_mel2 = mel2.repeat(batch_size, 1, 1)
        cfm_input = torch.cat([expanded_ref, batched_feat_chunks], dim=2).transpose(2, 1)
        cfm_lengths = torch.full((batch_size,), int(cfm_input.size(1)), dtype=torch.long, device=cfm_input.device)
        pred_spec = self._get_runtime_vits_model().cfm.inference(
            cfm_input,
            cfm_lengths,
            expanded_mel2,
            first_sample_steps,
            inference_cfg_rate=0,
        )
        pred_spec = pred_spec[:, :, -chunk_len:]
        channel_dim = int(pred_spec.shape[1])
        pred_spec = pred_spec.permute(1, 0, 2).contiguous().view(channel_dim, -1).unsqueeze(0)
        pred_spec = self._denorm_vocoder_spec(pred_spec)

        wav_gen = self._run_vocoder_module(self._get_runtime_vocoder(), pred_spec)
        audio = wav_gen[0][0]
        audio_chunks: list[torch.Tensor] = []
        audio_position = 0
        audio_chunk_len = chunk_len * upsample_rate
        while audio_position < int(audio.shape[-1]):
            audio_chunks.append(audio[audio_position : audio_position + audio_chunk_len])
            audio_position += audio_chunk_len
        merged_audio = self._sola_merge_audio_fragments(audio_chunks, overlapped_len * upsample_rate)
        trim_prefix = overlapped_len * upsample_rate
        if trim_prefix > 0:
            merged_audio = merged_audio[trim_prefix:]
        trim_suffix = padding_len * upsample_rate
        if trim_suffix > 0:
            merged_audio = merged_audio[:-trim_suffix] if trim_suffix < int(merged_audio.shape[-1]) else merged_audio[:0]

        output_sr = int(prompt_context["output_sr"])
        decoded_items: list[GPTSoVITSDecodedAudio] = []
        cursor = 0
        for prepared, feat_len in zip(prepared_requests, feat_lens):
            audio_len = feat_len * upsample_rate
            audio_fragment = merged_audio[cursor : cursor + audio_len]
            cursor += audio_len
            decoded_items.append(
                GPTSoVITSDecodedAudio(
                    request_id=prepared.request_id,
                    audio_fragment=audio_fragment,
                    output_sr=output_sr,
                    speed_factor=float(prepared.speed_factor),
                    super_sampling=bool(prepared.super_sampling),
                )
            )
        return decoded_items

    def _decode_prepared_requests_grouped_vocoder(
        self,
        pipeline: Any,
        prepared_requests: list[GPTSoVITSDecodePreparedRequest],
    ) -> list[GPTSoVITSDecodedAudio]:
        if not prepared_requests:
            return []
        refer_audio_spec = self._build_refer_spec_from_prepared(prepared_requests[0]).spec_audio
        with self._run_lock:
            prompt_context = self._build_vocoder_prompt_context(
                pipeline,
                prepared_requests[0],
                refer_audio_spec,
            )
            decoded: list[GPTSoVITSDecodedAudio | None] = [None] * len(prepared_requests)
            grouped_by_target: dict[tuple[float, int], list[tuple[int, GPTSoVITSDecodePreparedRequest]]] = {}
            for index, prepared in enumerate(prepared_requests):
                grouped_by_target.setdefault(
                    (float(prepared.speed_factor), int(prepared.sample_steps)),
                    [],
                ).append((index, prepared))
            for grouped_items in grouped_by_target.values():
                grouped_indices = [item[0] for item in grouped_items]
                grouped_prepared = [item[1] for item in grouped_items]
                if len(grouped_prepared) == 1:
                    prepared = grouped_prepared[0]
                    audio_fragment, output_sr = self._decode_vocoder_with_prompt_context(
                        pipeline,
                        prepared,
                        prompt_context,
                    )
                    grouped_decoded = [
                        GPTSoVITSDecodedAudio(
                            request_id=prepared.request_id,
                            audio_fragment=audio_fragment,
                            output_sr=int(output_sr),
                            speed_factor=float(prepared.speed_factor),
                            super_sampling=bool(prepared.super_sampling),
                        )
                    ]
                else:
                    grouped_decoded = self._decode_prepared_requests_batched_vocoder(
                        pipeline,
                        grouped_prepared,
                        prompt_context,
                    )
                for index, decoded_item in zip(grouped_indices, grouped_decoded):
                    decoded[index] = decoded_item
        return [item for item in decoded if item is not None]

    def _compute_vits_reference_ge(
        self,
        vits_model: Any,
        refer_audio_spec: torch.Tensor,
        *,
        refer_lengths: torch.Tensor | None = None,
        sv_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        commons = self._get_text_frontend_symbol("GPT_SoVITS.module", "commons")
        if refer_audio_spec.ndim != 3:
            raise ValueError(f"GPT-SoVITS refer_audio_spec 维度非法: expected 3D, got {tuple(refer_audio_spec.shape)}")
        batch_size = int(refer_audio_spec.size(0))
        if refer_lengths is None:
            refer_lengths = torch.full(
                (batch_size,),
                int(refer_audio_spec.size(2)),
                dtype=torch.long,
                device=refer_audio_spec.device,
            )
        else:
            refer_lengths = refer_lengths.to(device=refer_audio_spec.device, dtype=torch.long)
        refer_mask = torch.unsqueeze(
            commons.sequence_mask(refer_lengths, int(refer_audio_spec.size(2))),
            1,
        ).to(refer_audio_spec.dtype)
        refer_source = refer_audio_spec if getattr(vits_model, "version", "v2") == "v1" else refer_audio_spec[:, :704]
        ge = vits_model.ref_enc(refer_source * refer_mask, refer_mask)
        if bool(getattr(vits_model, "is_v2pro", False)):
            if sv_emb is None:
                raise ValueError("GPT-SoVITS v2Pro request-local synthesis 缺少 sv_emb")
            ge = ge + vits_model.sv_emb(sv_emb).unsqueeze(-1)
            ge = vits_model.prelu(ge)
        return ge

    def _run_vits_non_vocoder_decode(
        self,
        vits_model: Any,
        *,
        codes: torch.Tensor,
        code_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        ge: torch.Tensor,
        speed: float,
        noise_scale: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quantized = vits_model.quantizer.decode(codes)
        if getattr(vits_model, "semantic_frame_rate", "") == "25hz":
            quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")
        code_lengths = code_lengths.to(device=codes.device, dtype=torch.long)
        text_lengths = text_lengths.to(device=text.device, dtype=torch.long)
        y_lengths = code_lengths * 2
        encoder_ge = (
            vits_model.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
            if bool(getattr(vits_model, "is_v2pro", False))
            else ge
        )
        x, m_p, logs_p, y_mask, _, _ = vits_model.enc_p(
            quantized,
            y_lengths,
            text,
            text_lengths,
            encoder_ge,
            speed,
        )
        del x
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * float(noise_scale)
        z = vits_model.flow(z_p, y_mask, g=ge, reverse=True)
        audio = vits_model._decode_audio_runtime((z * y_mask)[:, :, :], g=ge)
        return audio, y_mask

    @staticmethod
    def _get_vits_streaming_chunk_tokens() -> int:
        raw_value = str(os.environ.get("GPTSOVITS_VITS_STREAMING_CHUNK_TOKENS", "128")).strip()
        try:
            return max(0, int(raw_value))
        except Exception:
            return 128

    @staticmethod
    def _get_vits_streaming_overlap_tokens() -> int:
        raw_value = str(os.environ.get("GPTSOVITS_VITS_STREAMING_OVERLAP_TOKENS", "2")).strip()
        try:
            return max(0, int(raw_value))
        except Exception:
            return 2

    @staticmethod
    def _get_vits_streaming_alignment_mode() -> str:
        return str(os.environ.get("GPTSOVITS_VITS_STREAMING_ALIGNMENT_MODE", "local_window")).strip().lower()

    def _run_vits_non_vocoder_streaming_chunk(
        self,
        vits_model: Any,
        *,
        codes: torch.Tensor,
        text: torch.Tensor,
        ge: torch.Tensor,
        speed: float,
        result_length: int | None,
        overlap_frames: torch.Tensor | None,
        padding_length: int | None = 0,
        noise_scale: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized = vits_model.quantizer.decode(codes)
        if getattr(vits_model, "semantic_frame_rate", "") == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
            result_length = (2 * int(result_length)) if result_length is not None else None
            padding_length = (2 * int(padding_length)) if padding_length is not None else None
        y_lengths = torch.LongTensor([int(codes.size(2) * 2)]).to(codes.device)
        text_lengths = torch.LongTensor([int(text.size(-1))]).to(text.device)
        x, m_p, logs_p, y_mask, latent, latent_mask = vits_model.enc_p(
            quantized,
            y_lengths,
            text,
            text_lengths,
            vits_model.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
            if bool(getattr(vits_model, "is_v2pro", False))
            else ge,
            speed,
            result_length=result_length,
            overlap_frames=overlap_frames,
            padding_length=padding_length,
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * float(noise_scale)
        z = vits_model.flow(z_p, y_mask, g=ge, reverse=True)
        decoder_input = (z * y_mask)[:, :, :]
        use_compiled_decoder = str(os.environ.get("GPTSOVITS_VITS_STREAMING_USE_COMPILED_DEC", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        audio = (
            vits_model._decode_audio_runtime(decoder_input, g=ge)
            if use_compiled_decoder
            else vits_model.dec(decoder_input, g=ge)
        )
        return audio, latent, latent_mask

    def _run_vits_non_vocoder_streaming_decode(
        self,
        pipeline: Any,
        vits_model: Any,
        *,
        semantic_tokens: torch.Tensor,
        phones: torch.Tensor,
        ge: torch.Tensor,
        speed: float,
        chunk_tokens: int,
        overlap_tokens: int,
    ) -> torch.Tensor:
        min_chunk_tokens_raw = str(os.environ.get("GPTSOVITS_VITS_STREAMING_MIN_CHUNK_TOKENS", "32")).strip()
        try:
            min_chunk_tokens = max(8, int(min_chunk_tokens_raw))
        except Exception:
            min_chunk_tokens = 32
        current_chunk_tokens = max(chunk_tokens, min_chunk_tokens)
        while True:
            try:
                return self._run_vits_non_vocoder_streaming_decode_impl(
                    pipeline,
                    vits_model,
                    semantic_tokens=semantic_tokens,
                    phones=phones,
                    ge=ge,
                    speed=speed,
                    chunk_tokens=current_chunk_tokens,
                    overlap_tokens=overlap_tokens,
                )
            except torch.OutOfMemoryError:
                if current_chunk_tokens <= min_chunk_tokens:
                    raise
                next_chunk_tokens = max(min_chunk_tokens, current_chunk_tokens // 2)
                if next_chunk_tokens >= current_chunk_tokens:
                    raise
                logger.warning(
                    "GPT-SoVITS chunked VITS decode OOM at chunk=%d; retrying with chunk=%d",
                    int(current_chunk_tokens),
                    int(next_chunk_tokens),
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current_chunk_tokens = next_chunk_tokens

    def _run_vits_non_vocoder_streaming_decode_local_window_impl(
        self,
        pipeline: Any,
        vits_model: Any,
        *,
        semantic_tokens: torch.Tensor,
        phones: torch.Tensor,
        ge: torch.Tensor,
        speed: float,
        chunk_tokens: int,
        overlap_tokens: int,
    ) -> torch.Tensor:
        total_tokens = int(semantic_tokens.shape[-1])
        if total_tokens == 0:
            return torch.zeros((0,), dtype=ge.dtype, device=ge.device)
        if chunk_tokens <= 0 or total_tokens <= chunk_tokens:
            raise ValueError("streaming decode helper expects semantic length above chunk threshold")

        upsample_factor = 1
        for up_layer in getattr(getattr(vits_model, "dec", None), "ups", []):
            stride = up_layer.stride[0] if isinstance(up_layer.stride, tuple) else int(up_layer.stride)
            upsample_factor *= int(stride)
        semantic_rate_scale = 2 if getattr(vits_model, "semantic_frame_rate", "") == "25hz" else 1
        speed_value = max(float(speed), 1e-6)
        upsample_rate = float(upsample_factor) * (float(semantic_rate_scale) / speed_value)
        overlap_size = int(math.ceil(float(overlap_tokens) * upsample_rate))

        audio_fragments: list[torch.Tensor] = []
        last_audio_chunk: torch.Tensor | None = None
        last_latent: torch.Tensor | None = None
        chunk_start = 0
        total_phones = int(phones.shape[-1])
        phone_overlap = 0
        if total_tokens > 0 and overlap_tokens > 0 and total_phones > 0:
            phone_overlap = max(1, int(math.ceil(float(overlap_tokens) * float(total_phones) / float(total_tokens))))

        while chunk_start < total_tokens:
            chunk_end = min(total_tokens, chunk_start + chunk_tokens)
            new_chunk_tokens = int(chunk_end - chunk_start)
            semantic_chunk = semantic_tokens[chunk_start:chunk_end]
            is_first_chunk = chunk_start == 0
            is_final_chunk = chunk_end >= total_tokens
            overlap_len = min(int(overlap_tokens), max(0, new_chunk_tokens - 1))

            phone_start = 0 if total_phones <= 0 else int((chunk_start * total_phones) // total_tokens)
            phone_end = total_phones if total_phones <= 0 else int(math.ceil((chunk_end * total_phones) / total_tokens))
            if not is_first_chunk:
                phone_start = max(0, phone_start - phone_overlap)
            phone_end = max(phone_start + 1, min(total_phones, phone_end))
            phone_chunk = phones[phone_start:phone_end]

            overlap_frames = None
            if last_latent is not None and overlap_len > 0:
                overlap_frames = last_latent[
                    :,
                    :,
                    -overlap_len * semantic_rate_scale :,
                ]

            audio_chunk, latent, _latent_mask = self._run_vits_non_vocoder_streaming_chunk(
                vits_model,
                codes=semantic_chunk.unsqueeze(0).unsqueeze(0).to(device=semantic_tokens.device, dtype=torch.long),
                text=phone_chunk.unsqueeze(0).to(device=phones.device, dtype=torch.long),
                ge=ge,
                speed=speed_value,
                result_length=None,
                overlap_frames=overlap_frames,
                padding_length=0,
            )
            audio_chunk = audio_chunk.detach()[0, 0, :]

            if is_first_chunk:
                if is_final_chunk:
                    audio_fragment = audio_chunk
                elif overlap_size > 0:
                    audio_fragment = audio_chunk[:-overlap_size]
                else:
                    audio_fragment = audio_chunk
            elif last_audio_chunk is None:
                audio_fragment = audio_chunk
            elif overlap_size > 0:
                merged_audio = pipeline.sola_algorithm([last_audio_chunk, audio_chunk], overlap_size)
                start_index = int(last_audio_chunk.shape[0] - overlap_size)
                audio_fragment = merged_audio[start_index:] if is_final_chunk else merged_audio[start_index:-overlap_size]
            else:
                audio_fragment = audio_chunk

            audio_fragments.append(audio_fragment)
            last_audio_chunk = audio_chunk
            last_latent = latent
            chunk_start = chunk_end

        return torch.cat(audio_fragments, dim=0)

    def _run_vits_non_vocoder_streaming_decode_impl(
        self,
        pipeline: Any,
        vits_model: Any,
        *,
        semantic_tokens: torch.Tensor,
        phones: torch.Tensor,
        ge: torch.Tensor,
        speed: float,
        chunk_tokens: int,
        overlap_tokens: int,
    ) -> torch.Tensor:
        alignment_mode = self._get_vits_streaming_alignment_mode()
        if alignment_mode not in {"prefix_full_text", "prefix_full_phones", "quality"}:
            return self._run_vits_non_vocoder_streaming_decode_local_window_impl(
                pipeline,
                vits_model,
                semantic_tokens=semantic_tokens,
                phones=phones,
                ge=ge,
                speed=speed,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
            )

        total_tokens = int(semantic_tokens.shape[-1])
        if total_tokens == 0:
            return torch.zeros((0,), dtype=ge.dtype, device=ge.device)
        if chunk_tokens <= 0 or total_tokens <= chunk_tokens:
            raise ValueError("streaming decode helper expects semantic length above chunk threshold")

        upsample_factor = 1
        for up_layer in getattr(getattr(vits_model, "dec", None), "ups", []):
            stride = up_layer.stride[0] if isinstance(up_layer.stride, tuple) else int(up_layer.stride)
            upsample_factor *= int(stride)
        semantic_rate_scale = 2 if getattr(vits_model, "semantic_frame_rate", "") == "25hz" else 1
        speed_value = max(float(speed), 1e-6)
        overlap_size = int(math.ceil(float(overlap_tokens) * float(upsample_factor) * (float(semantic_rate_scale) / speed_value)))

        audio_fragments: list[torch.Tensor] = []
        last_audio_chunk: torch.Tensor | None = None
        last_latent: torch.Tensor | None = None
        chunk_start = 0

        try:
            while chunk_start < total_tokens:
                chunk_end = min(total_tokens, chunk_start + chunk_tokens)
                new_chunk_tokens = int(chunk_end - chunk_start)
                prefix_semantic = semantic_tokens[:chunk_end]
                is_first_chunk = chunk_start == 0
                is_final_chunk = chunk_end >= total_tokens
                overlap_len = min(int(overlap_tokens), max(0, new_chunk_tokens - 1))
                if not is_first_chunk and new_chunk_tokens < 10:
                    overlap_len = min(int(chunk_end), int(overlap_tokens) + int(10 - new_chunk_tokens))

                overlap_frames = None
                if last_latent is not None and overlap_len > 0:
                    overlap_frames = last_latent[:, :, -overlap_len * semantic_rate_scale :]

                result_length = None if is_first_chunk else int(new_chunk_tokens + overlap_len)
                audio_chunk, latent, _latent_mask = self._run_vits_non_vocoder_streaming_chunk(
                    vits_model,
                    codes=prefix_semantic.unsqueeze(0).unsqueeze(0).to(device=semantic_tokens.device, dtype=torch.long),
                    text=phones.unsqueeze(0).to(device=phones.device, dtype=torch.long),
                    ge=ge,
                    speed=speed_value,
                    result_length=result_length,
                    overlap_frames=overlap_frames,
                    padding_length=0,
                )
                audio_chunk = audio_chunk.detach()[0, 0, :]

                if overlap_len > overlap_tokens:
                    audio_chunk = audio_chunk[-int((int(overlap_tokens) + int(new_chunk_tokens)) * float(upsample_factor) * (float(semantic_rate_scale) / speed_value)) :]

                if is_first_chunk:
                    if is_final_chunk:
                        audio_fragment = audio_chunk
                    elif overlap_size > 0:
                        audio_fragment = audio_chunk[:-overlap_size]
                    else:
                        audio_fragment = audio_chunk
                elif last_audio_chunk is None:
                    audio_fragment = audio_chunk
                elif overlap_size > 0:
                    merged_audio = pipeline.sola_algorithm([last_audio_chunk, audio_chunk], overlap_size)
                    start_index = int(last_audio_chunk.shape[0] - overlap_size)
                    audio_fragment = merged_audio[start_index:] if is_final_chunk else merged_audio[start_index:-overlap_size]
                else:
                    audio_fragment = audio_chunk

                audio_fragments.append(audio_fragment)
                last_audio_chunk = audio_chunk
                last_latent = latent
                chunk_start = chunk_end
        except torch.OutOfMemoryError:
            logger.warning(
                "GPT-SoVITS prefix/full-phone chunked VITS decode OOM at chunk=%d; falling back to local-window decode",
                int(chunk_tokens),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._run_vits_non_vocoder_streaming_decode_local_window_impl(
                pipeline,
                vits_model,
                semantic_tokens=semantic_tokens,
                phones=phones,
                ge=ge,
                speed=speed,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
            )

        return torch.cat(audio_fragments, dim=0)

    @staticmethod
    def _measure_vits_audio_lengths(vits_model: Any, y_mask: torch.Tensor) -> torch.Tensor:
        upsample_factor = 1
        for up_layer in getattr(getattr(vits_model, "dec", None), "ups", []):
            stride = up_layer.stride[0] if isinstance(up_layer.stride, tuple) else int(up_layer.stride)
            upsample_factor *= int(stride)
        return y_mask.squeeze(1).sum(dim=1).to(dtype=torch.long) * int(upsample_factor)

    def _run_vits_vocoder_feature_decode(
        self,
        vits_model: Any,
        *,
        codes: torch.Tensor,
        text: torch.Tensor,
        refer_audio_spec: torch.Tensor,
        ge: torch.Tensor | None = None,
        speed: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        commons = self._get_text_frontend_symbol("GPT_SoVITS.module", "commons")
        if ge is None:
            refer_lengths = torch.LongTensor([int(refer_audio_spec.size(2))]).to(refer_audio_spec.device)
            refer_mask = torch.unsqueeze(
                commons.sequence_mask(refer_lengths, int(refer_audio_spec.size(2))),
                1,
            ).to(refer_audio_spec.dtype)
            ge = vits_model.ref_enc(refer_audio_spec[:, :704] * refer_mask, refer_mask)
        y_lengths = torch.LongTensor([int(codes.size(2) * 2)]).to(codes.device)
        version = str(getattr(vits_model, "version", "v2"))
        base_scale = 3.875 if version == "v3" else 4.0
        if float(speed) == 1.0:
            sizee = int(codes.size(2) * base_scale)
        else:
            sizee = int(codes.size(2) * base_scale / float(speed)) + 1
        y_lengths1 = torch.LongTensor([sizee]).to(codes.device)
        text_lengths = torch.LongTensor([int(text.size(-1))]).to(text.device)
        quantized = vits_model.quantizer.decode(codes)
        if getattr(vits_model, "semantic_frame_rate", "") == "25hz":
            quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")
        try:
            x, _m_p, _logs_p, _y_mask, *_ = vits_model.enc_p(quantized, y_lengths, text, text_lengths, ge, speed)
        except TypeError:
            x, _m_p, _logs_p, _y_mask, *_ = vits_model.enc_p(quantized, y_lengths, text, text_lengths, ge)
        fea = vits_model.bridge(x)
        fea = F.interpolate(fea, scale_factor=(1.875 if version == "v3" else 2), mode="nearest")
        fea, _ = vits_model.wns1(fea, y_lengths1, ge)
        return fea, ge

    def _decode_prepared_request_fragment(
        self,
        pipeline: Any,
        prepared: GPTSoVITSDecodePreparedRequest,
    ) -> tuple[Any, int]:
        self._bind_pipeline_components(pipeline)
        refer_spec = self._build_refer_spec_from_prepared(prepared)
        if bool(getattr(pipeline.configs, "use_vocoder", False)):
            return self._decode_prepared_request_vocoder_fragment(
                pipeline,
                prepared,
                refer_spec.spec_audio,
            )

        device = torch.device(self._get_runtime_configs().device)
        vits_model = self._get_runtime_vits_model()
        refer_audio_spec = refer_spec.spec_audio.to(dtype=self._get_runtime_precision(), device=device)
        sv_emb = None
        if self._is_runtime_v2pro():
            audio_tensor = refer_spec.audio_16k
            if audio_tensor is None:
                raise ValueError("GPT-SoVITS v2Pro request-local synthesis 缺少 16k 参考音频")
            sv_emb = self._get_runtime_sv_model().compute_embedding3(audio_tensor).to(device)
        ge = self._compute_vits_reference_ge(
            vits_model,
            refer_audio_spec,
            sv_emb=sv_emb,
        )
        chunk_tokens = self._get_vits_streaming_chunk_tokens()
        if (
            chunk_tokens > 0
            and int(prepared.semantic_tokens.shape[-1]) > chunk_tokens
            and hasattr(pipeline, "sola_algorithm")
        ):
            logger.info(
                "GPT-SoVITS using chunked non-vocoder VITS decode: tokens=%d chunk=%d overlap=%d",
                int(prepared.semantic_tokens.shape[-1]),
                int(chunk_tokens),
                int(self._get_vits_streaming_overlap_tokens()),
            )
            audio_fragment = self._run_vits_non_vocoder_streaming_decode(
                pipeline,
                vits_model,
                semantic_tokens=prepared.semantic_tokens,
                phones=prepared.phones,
                ge=ge,
                speed=float(prepared.speed_factor),
                chunk_tokens=chunk_tokens,
                overlap_tokens=self._get_vits_streaming_overlap_tokens(),
            )
            return audio_fragment, int(self._get_runtime_configs().sampling_rate)
        audio_batch, y_mask = self._run_vits_non_vocoder_decode(
            vits_model,
            codes=prepared.semantic_tokens.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.long),
            code_lengths=torch.LongTensor([int(prepared.semantic_tokens.shape[-1])]).to(device),
            text=prepared.phones.unsqueeze(0).to(device=device, dtype=torch.long),
            text_lengths=torch.LongTensor([int(prepared.phones.shape[-1])]).to(device),
            ge=ge,
            speed=float(prepared.speed_factor),
        )
        audio_lengths = self._measure_vits_audio_lengths(vits_model, y_mask)
        audio_fragment = audio_batch[0, 0, : int(audio_lengths[0].item())].detach()
        return audio_fragment, int(self._get_runtime_configs().sampling_rate)

    def _decode_prepared_requests_batched_non_vocoder(
        self,
        pipeline: Any,
        prepared_requests: list[GPTSoVITSDecodePreparedRequest],
    ) -> list[GPTSoVITSDecodedAudio]:
        self._bind_pipeline_components(pipeline)
        if not prepared_requests:
            return []
        if bool(getattr(pipeline.configs, "use_vocoder", False)):
            raise ValueError("non-vocoder batched decode helper received vocoder pipeline")

        device = torch.device(self._get_runtime_configs().device)
        vits_model = self._get_runtime_vits_model()
        batch_size = len(prepared_requests)
        first_speed = float(prepared_requests[0].speed_factor)
        first_sample_steps = int(prepared_requests[0].sample_steps)
        if any(abs(float(item.speed_factor) - first_speed) > 1e-6 for item in prepared_requests):
            raise ValueError("GPT-SoVITS batched non-vocoder decode requires identical speed_factor")
        if any(int(item.sample_steps) != first_sample_steps for item in prepared_requests):
            raise ValueError("GPT-SoVITS batched non-vocoder decode requires identical sample_steps")

        refer_specs = [self._build_refer_spec_from_prepared(prepared) for prepared in prepared_requests]
        shared_single_refer = False
        shared_refer_audio_spec = None
        if refer_specs:
            first_spec = refer_specs[0].spec_audio
            first_audio = refer_specs[0].audio_16k
            first_spec_key = (
                str(first_spec.device),
                str(first_spec.dtype),
                tuple(int(item) for item in first_spec.shape),
                int(first_spec.data_ptr()),
            )
            first_audio_key = None
            if first_audio is not None:
                first_audio_key = (
                    str(first_audio.device),
                    str(first_audio.dtype),
                    tuple(int(item) for item in first_audio.shape),
                    int(first_audio.data_ptr()),
                )
            shared_single_refer = True
            for refer_spec in refer_specs[1:]:
                refer_audio_spec = refer_spec.spec_audio
                audio_tensor = refer_spec.audio_16k
                refer_spec_key = (
                    str(refer_audio_spec.device),
                    str(refer_audio_spec.dtype),
                    tuple(int(item) for item in refer_audio_spec.shape),
                    int(refer_audio_spec.data_ptr()),
                )
                if refer_spec_key != first_spec_key:
                    shared_single_refer = False
                    break
                if first_audio_key is None:
                    if audio_tensor is not None:
                        shared_single_refer = False
                        break
                else:
                    if audio_tensor is None:
                        shared_single_refer = False
                        break
                    audio_key = (
                        str(audio_tensor.device),
                        str(audio_tensor.dtype),
                        tuple(int(item) for item in audio_tensor.shape),
                        int(audio_tensor.data_ptr()),
                    )
                    if audio_key != first_audio_key:
                        shared_single_refer = False
                        break
            if shared_single_refer:
                shared_refer_audio_spec = first_spec.to(dtype=self._get_runtime_precision(), device=device)

        max_semantic_len = max(int(item.semantic_tokens.shape[-1]) for item in prepared_requests)
        max_phone_len = max(int(item.phones.shape[-1]) for item in prepared_requests)
        semantic_batch = torch.zeros((1, batch_size, max_semantic_len), dtype=torch.long, device=device)
        phone_batch = torch.zeros((batch_size, max_phone_len), dtype=torch.long, device=device)
        semantic_lengths: list[int] = []
        phone_lengths: list[int] = []
        refer_audio_specs: list[torch.Tensor] = []
        sv_emb_batch = None
        sv_emb_list: list[torch.Tensor] = []

        for batch_index, prepared in enumerate(prepared_requests):
            semantic_len = int(prepared.semantic_tokens.shape[-1])
            phone_len = int(prepared.phones.shape[-1])
            semantic_batch[0, batch_index, :semantic_len] = prepared.semantic_tokens.to(device=device, dtype=torch.long)
            phone_batch[batch_index, :phone_len] = prepared.phones.to(device=device, dtype=torch.long)
            semantic_lengths.append(semantic_len)
            phone_lengths.append(phone_len)
            if shared_single_refer:
                continue
            refer_spec = refer_specs[batch_index]
            refer_audio_specs.append(refer_spec.spec_audio.to(dtype=self._get_runtime_precision(), device=device))
            if self._is_runtime_v2pro():
                if refer_spec.audio_16k is None:
                    raise ValueError("GPT-SoVITS v2Pro batched non-vocoder decode 缺少 16k 参考音频")
                sv_emb_list.append(self._get_runtime_sv_model().compute_embedding3(refer_spec.audio_16k).to(device))

        if self._is_runtime_v2pro():
            if shared_single_refer:
                shared_audio_tensor = refer_specs[0].audio_16k
                if shared_audio_tensor is None:
                    raise ValueError("GPT-SoVITS v2Pro batched non-vocoder decode 缺少 16k 参考音频")
                sv_emb_batch = self._get_runtime_sv_model().compute_embedding3(shared_audio_tensor).to(device)
            else:
                sv_emb_batch = torch.cat(sv_emb_list, dim=0)

        ge_batch = None
        if shared_single_refer:
            if shared_refer_audio_spec is None:
                raise ValueError("shared_single_refer detected but refer_audio_spec missing")
            refer_audio_specs = [shared_refer_audio_spec]
            ge_batch = self._compute_vits_reference_ge(
                vits_model,
                shared_refer_audio_spec,
                sv_emb=(sv_emb_batch[:1] if sv_emb_batch is not None else None),
            )
            ge_batch = ge_batch.expand(batch_size, -1, -1)
        else:
            refer_lengths = torch.LongTensor([int(item.size(2)) for item in refer_audio_specs]).to(device)
            max_refer_len = int(refer_lengths.max().item())
            refer_batch = torch.zeros(
                (batch_size, int(refer_audio_specs[0].size(1)), max_refer_len),
                dtype=refer_audio_specs[0].dtype,
                device=device,
            )
            for batch_index, refer in enumerate(refer_audio_specs):
                refer_batch[batch_index, :, : int(refer.size(2))] = refer.squeeze(0)
            ge_batch = self._compute_vits_reference_ge(
                vits_model,
                refer_batch,
                refer_lengths=refer_lengths,
                sv_emb=sv_emb_batch,
            )

        with self._run_lock:
            audio_batch, y_mask = self._run_vits_non_vocoder_decode(
                vits_model,
                codes=semantic_batch,
                code_lengths=torch.LongTensor(semantic_lengths).to(device),
                text=phone_batch,
                text_lengths=torch.LongTensor(phone_lengths).to(device),
                ge=ge_batch,
                speed=first_speed,
            )
        audio_lengths = self._measure_vits_audio_lengths(vits_model, y_mask)
        audio_fragments = [
            audio_batch[batch_index, 0, : int(audio_lengths[batch_index].item())].detach()
            for batch_index in range(batch_size)
        ]
        output_sr = int(self._get_runtime_configs().sampling_rate)
        return [
            GPTSoVITSDecodedAudio(
                request_id=prepared.request_id,
                audio_fragment=audio_fragment,
                output_sr=output_sr,
                speed_factor=float(prepared.speed_factor),
                super_sampling=bool(prepared.super_sampling),
            )
            for prepared, audio_fragment in zip(prepared_requests, audio_fragments)
        ]

    def decode_prepared_request(
        self,
        prepared: GPTSoVITSDecodePreparedRequest,
    ) -> GPTSoVITSDecodedAudio:
        pipeline = self._ensure_pipeline()
        if prepared.semantic_tokens.numel() == 0:
            return GPTSoVITSDecodedAudio(
                request_id=prepared.request_id,
                audio_fragment=np.zeros((0,), dtype=np.float32),
                output_sr=int(getattr(pipeline.configs, "sampling_rate", 32000)),
                speed_factor=float(prepared.speed_factor),
                super_sampling=bool(prepared.super_sampling),
            )
        with self._run_lock:
            with self._project_root_cwd():
                audio_fragment, output_sr = self._decode_prepared_request_fragment(pipeline, prepared)
        return GPTSoVITSDecodedAudio(
            request_id=prepared.request_id,
            audio_fragment=audio_fragment,
            output_sr=int(output_sr),
            speed_factor=float(prepared.speed_factor),
            super_sampling=bool(prepared.super_sampling),
        )

    def decode_prepared_requests(
        self,
        prepared_requests: list[GPTSoVITSDecodePreparedRequest],
    ) -> list[GPTSoVITSDecodedAudio]:
        if not prepared_requests:
            return []
        pipeline = self._ensure_pipeline()
        decoded: list[GPTSoVITSDecodedAudio | None] = [None] * len(prepared_requests)
        output_sr_default = int(getattr(pipeline.configs, "sampling_rate", 32000))
        sequential_indices: list[int] = []
        grouped_non_vocoder: dict[tuple[float, int], list[tuple[int, GPTSoVITSDecodePreparedRequest]]] = {}
        grouped_vocoder: dict[tuple[Any, ...], list[tuple[int, GPTSoVITSDecodePreparedRequest]]] = {}

        for index, prepared in enumerate(prepared_requests):
            if prepared.semantic_tokens.numel() == 0:
                decoded[index] = GPTSoVITSDecodedAudio(
                    request_id=prepared.request_id,
                    audio_fragment=np.zeros((0,), dtype=np.float32),
                    output_sr=output_sr_default,
                    speed_factor=float(prepared.speed_factor),
                    super_sampling=bool(prepared.super_sampling),
                )
                continue
            if bool(getattr(pipeline.configs, "use_vocoder", False)):
                refer_audio_spec = self._build_refer_spec_from_prepared(prepared).spec_audio
                grouped_vocoder.setdefault(
                    self._build_vocoder_prompt_cache_key(prepared, refer_audio_spec),
                    [],
                ).append((index, prepared))
                continue
            grouped_non_vocoder.setdefault(
                (float(prepared.speed_factor), int(prepared.sample_steps)),
                [],
            ).append((index, prepared))

        for grouped_items in grouped_non_vocoder.values():
            if len(grouped_items) == 1:
                sequential_indices.append(grouped_items[0][0])
                continue
            grouped_indices = [item[0] for item in grouped_items]
            grouped_prepared = [item[1] for item in grouped_items]
            grouped_decoded = self._decode_prepared_requests_batched_non_vocoder(pipeline, grouped_prepared)
            for index, decoded_item in zip(grouped_indices, grouped_decoded):
                decoded[index] = decoded_item

        for grouped_items in grouped_vocoder.values():
            grouped_indices = [item[0] for item in grouped_items]
            grouped_prepared = [item[1] for item in grouped_items]
            if len(grouped_prepared) == 1:
                sequential_indices.append(grouped_indices[0])
                continue
            grouped_decoded = self._decode_prepared_requests_grouped_vocoder(
                pipeline,
                grouped_prepared,
            )
            for index, decoded_item in zip(grouped_indices, grouped_decoded):
                decoded[index] = decoded_item

        for index in sequential_indices:
            if decoded[index] is None:
                decoded[index] = self.decode_prepared_request(prepared_requests[index])

        return [item for item in decoded if item is not None]

    @staticmethod
    def _audio_fragment_to_tensor(audio_fragment: Any, *, device: torch.device) -> torch.Tensor:
        if isinstance(audio_fragment, torch.Tensor):
            return audio_fragment.detach().to(device=device, dtype=torch.float32).reshape(-1)
        return torch.as_tensor(audio_fragment, dtype=torch.float32, device=device).reshape(-1)

    def _audio_postprocess_native(
        self,
        pipeline: Any,
        *,
        audio_fragments: list[Any],
        sr: int,
        speed_factor: float,
        super_sampling: bool,
        fragment_interval: float = 0.0,
    ) -> tuple[int, np.ndarray]:
        del speed_factor
        device = torch.device(pipeline.configs.device)
        fragment_tensors: list[torch.Tensor] = []
        zero_wav = None
        if fragment_interval > 0:
            zero_wav = torch.zeros(
                int(pipeline.configs.sampling_rate * fragment_interval),
                dtype=torch.float32,
                device=device,
            )

        for audio_fragment in audio_fragments:
            fragment = self._audio_fragment_to_tensor(audio_fragment, device=device)
            max_audio = torch.abs(fragment).max() if fragment.numel() > 0 else torch.tensor(0.0, device=device)
            if float(max_audio.item()) > 1.0:
                fragment = fragment / max_audio
            if zero_wav is not None:
                fragment = torch.cat([fragment, zero_wav], dim=0)
            fragment_tensors.append(fragment)

        audio = torch.cat(fragment_tensors, dim=0) if fragment_tensors else torch.zeros((0,), dtype=torch.float32, device=device)

        if super_sampling:
            sr_model, sr_model_not_exist = self._ensure_runtime_sr_model()
            if not sr_model_not_exist and sr_model is not None:
                audio, sr = sr_model(audio.unsqueeze(0), sr)
                max_audio = float(torch.abs(audio).max().item()) if isinstance(audio, torch.Tensor) and audio.numel() > 0 else 0.0
                if max_audio > 1.0:
                    audio = audio / max_audio

        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().float().cpu().numpy()
        else:
            audio_np = np.asarray(audio)
        audio_np = (audio_np.reshape(-1) * 32768).astype(np.int16)
        return int(sr), audio_np

    def finalize_decoded_audio(
        self,
        decoded: GPTSoVITSDecodedAudio,
    ) -> GPTSoVITSResult:
        pipeline = self._ensure_pipeline()
        with self._run_lock:
            sample_rate, audio = self._audio_postprocess_native(
                pipeline,
                audio_fragments=[decoded.audio_fragment],
                sr=int(decoded.output_sr),
                speed_factor=float(decoded.speed_factor),
                fragment_interval=0.0,
                super_sampling=bool(decoded.super_sampling),
            )
        return GPTSoVITSResult(sample_rate=int(sample_rate), audio=self._normalize_audio(np.asarray(audio)))

    def finalize_decoded_audios(
        self,
        decoded_items: list[GPTSoVITSDecodedAudio],
    ) -> list[GPTSoVITSResult]:
        return [self.finalize_decoded_audio(decoded) for decoded in decoded_items]

    def decode_semantic_tokens_from_transport(
        self,
        semantic_tokens: torch.Tensor,
        transport_info: GPTSoVITSStageTransport | dict[str, Any],
    ) -> GPTSoVITSResult:
        prepared = self.prepare_decode_request(semantic_tokens, transport_info)
        if prepared.semantic_tokens.numel() == 0:
            pipeline = self._ensure_pipeline()
            return GPTSoVITSResult(
                sample_rate=int(getattr(pipeline.configs, "sampling_rate", 32000)),
                audio=np.zeros((0,), dtype=np.float32),
            )
        decoded = self.decode_prepared_request(prepared)
        return self.finalize_decoded_audio(decoded)

    @staticmethod
    def _normalize_audio(audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio)
        if audio.ndim > 1:
            audio = np.squeeze(audio)
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        if audio.dtype == np.int32:
            return audio.astype(np.float32) / 2147483648.0
        return audio.astype(np.float32, copy=False)

    @staticmethod
    def _normalize_lang(lang: Any, *, fallback: str) -> str:
        if lang is None:
            return fallback
        value = str(lang).strip().lower()
        return value or fallback

    @staticmethod
    def _default_cut_method(text_lang: str) -> str:
        del text_lang
        return "cut1"

    def build_tts_inputs(self, request: dict[str, Any]) -> dict[str, Any]:
        text = str(request.get("text") or "").strip()
        if not text:
            raise ValueError("GPT-SoVITS request requires non-empty 'text'")

        ref_audio_path = request.get("ref_audio_path")
        if not ref_audio_path:
            raise ValueError("GPT-SoVITS request requires 'ref_audio_path'")
        ref_audio_path = self._resolve_path(str(ref_audio_path)) if not os.path.isabs(str(ref_audio_path)) else str(ref_audio_path)
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

        prompt_text = str(request.get("prompt_text") or "").strip()
        if not prompt_text:
            raise ValueError("GPT-SoVITS request requires non-empty 'prompt_text'")

        text_lang = self._normalize_lang(request.get("text_lang"), fallback="auto")
        prompt_lang = self._normalize_lang(request.get("prompt_lang"), fallback=text_lang)
        text_split_method = str(request.get("text_split_method") or self._default_cut_method(text_lang)).strip()

        return {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": list(request.get("aux_ref_audio_paths") or []),
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "top_k": int(request.get("top_k", 15)),
            "top_p": float(request.get("top_p", 1.0)),
            "temperature": float(request.get("temperature", 1.0)),
            "text_split_method": text_split_method,
            "batch_size": int(request.get("batch_size", 4)),
            "batch_threshold": float(request.get("batch_threshold", 0.75)),
            "split_bucket": bool(request.get("split_bucket", True)),
            "speed_factor": float(request.get("speed_factor", request.get("speed", 1.0) or 1.0)),
            "fragment_interval": float(request.get("fragment_interval", 0.3)),
            "seed": int(request.get("seed", -1)),
            "parallel_infer": bool(request.get("parallel_infer", True)),
            "repetition_penalty": float(request.get("repetition_penalty", 1.35)),
            "sample_steps": int(request.get("sample_steps", 32)),
            "super_sampling": bool(request.get("super_sampling", False)),
            "streaming_mode": False,
            "return_fragment": False,
        }

    def synthesize(self, request: dict[str, Any]) -> GPTSoVITSResult:
        spec = self.build_request_spec(request)
        prepared = self.prepare_request_spec(spec)
        semantic_tokens = self.generate_semantic_tokens([prepared]).get(str(prepared.request_id))
        if semantic_tokens is None:
            raise RuntimeError(f"GPT-SoVITS native synthesize missing semantic tokens for request {prepared.request_id}")
        return self.decode_semantic_tokens_from_transport(semantic_tokens, prepared.transport_info)


_RUNTIME_CACHE: dict[tuple[str, str], GPTSoVITSRuntime] = {}
_RUNTIME_CACHE_LOCK = threading.Lock()


def get_gpt_sovits_runtime(
    *,
    project_root: str | None = None,
    config_path: str | None = None,
) -> GPTSoVITSRuntime:
    resolved_root = os.path.abspath(project_root or os.environ.get("GPT_SOVITS_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT))
    resolved_config = os.path.abspath(
        config_path
        or os.environ.get("GPT_SOVITS_CONFIG_PATH")
        or os.path.join(resolved_root, _DEFAULT_CONFIG_PATH)
    )
    cache_key = (resolved_root, resolved_config)
    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE.get(cache_key)
        if runtime is None:
            runtime = GPTSoVITSRuntime(project_root=resolved_root, config_path=resolved_config)
            _RUNTIME_CACHE[cache_key] = runtime
        return runtime
