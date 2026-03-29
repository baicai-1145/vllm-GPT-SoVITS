from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys
import threading
import ctypes
import types
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Sequence

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


@dataclass(slots=True)
class GPTSoVITSPreparedRequest:
    request_id: str
    state: Any
    transport_info: dict[str, Any]


@dataclass(slots=True)
class GPTSoVITSRequestSpec:
    request_id: str
    ref_audio_path: str
    prompt_text: str
    prompt_lang: str
    text: str
    text_lang: str
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    early_stop_num: int
    aux_ref_audio_paths: list[str]
    ready_step: int = 0


@dataclass(slots=True)
class GPTSoVITSPreparedCpuStage:
    request_id: str
    request: dict[str, Any]
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

    def submit_prepare_ref_audio_asset(self, ref_audio_path: str, *, submit_at: float | None = None) -> Any:
        submit_fn = self.submit_prepare_ref_audio_asset_fn
        if not callable(submit_fn):
            raise RuntimeError("GPT-SoVITS prepare coordinator does not provide ref-audio preload submission")
        return self._wrap_prepare_future(submit_fn(ref_audio_path, submit_at=submit_at))

    async def acquire_prepare_admission(self) -> dict[str, float]:
        return await self.inflight_gate.acquire()

    def mark_prepare_enter(self) -> tuple[int, int]:
        mark_enter = self.mark_enter_fn
        if callable(mark_enter):
            current_inflight, peak_inflight = mark_enter()
            return int(current_inflight), int(peak_inflight)
        return 0, 0

    def release_split_stage_slot(self) -> None:
        release_slot = self.release_split_stage_slot_fn
        if callable(release_slot):
            release_slot()


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
    transport_info: dict[str, Any]
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
                pipeline = TTS(config)
            self._install_ref_audio_loader_fallback(pipeline)
            self._install_sv_half_safe_patch(pipeline)
            self._config = config
            self._pipeline = pipeline
            logger.info(
                "Initialized GPT-SoVITS runtime from %s using config %s",
                self.project_root,
                self.config_path,
            )
        return self._pipeline

    def get_t2s_model(self) -> Any:
        pipeline = self._ensure_pipeline()
        return pipeline.t2s_model.model

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
            with self._project_root_cwd():
                from GPT_SoVITS.TTS_infer_pack.prepare_coordinator import PrepareCoordinator

                self._prepare_coordinator = self._coerce_prepare_coordinator(
                    PrepareCoordinator(pipeline)
                )
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

    def _get_sampling_ops(self) -> tuple[Any, Any]:
        self._ensure_import_path()
        self._ensure_native_runtime_deps()
        with self._project_root_cwd():
            from AR.models.utils import logits_to_probs, multinomial_sample_one_no_sync

        return logits_to_probs, multinomial_sample_one_no_sync

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits_to_probs, multinomial_sample_one_no_sync = self._get_sampling_ops()
        top_k, top_p, temperature, repetition_penalty, trim_eos = sampling_key
        sample_logits = logits[:, :-1] if trim_eos else logits
        padded_histories = self._stack_token_sequences_if_same_length(histories)
        history_mask = None
        if padded_histories is None:
            padded_histories, history_mask = _pad_token_sequences(histories)
        probs = logits_to_probs(
            logits=sample_logits,
            previous_tokens=padded_histories,
            previous_token_mask=history_mask,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )
        sampled = multinomial_sample_one_no_sync(probs)
        argmax_tokens = torch.argmax(sample_logits, dim=-1)
        return sampled, argmax_tokens

    def _batched_sample_by_group(
        self,
        logits: torch.Tensor,
        histories: Sequence[torch.LongTensor],
        sampling_keys: Sequence[tuple[int, float, float, float, bool]],
    ) -> tuple[list[torch.Tensor], list[int]]:
        logits_to_probs, multinomial_sample_one_no_sync = self._get_sampling_ops()
        sampled_list: list[torch.Tensor | None] = [None] * len(histories)
        argmax_list: list[int | None] = [None] * len(histories)
        for group_key, group_indices in self._iter_contiguous_sampling_groups(sampling_keys):
            top_k, top_p, temperature, repetition_penalty, trim_eos = group_key
            index_tensor = torch.tensor(group_indices, dtype=torch.long, device=logits.device)
            group_logits = torch.index_select(logits, dim=0, index=index_tensor)
            if trim_eos:
                group_logits = group_logits[:, :-1]
            group_histories = [histories[index] for index in group_indices]
            padded_histories = self._stack_token_sequences_if_same_length(group_histories)
            history_mask = None
            if padded_histories is None:
                padded_histories, history_mask = _pad_token_sequences(group_histories)
            probs = logits_to_probs(
                logits=group_logits,
                previous_tokens=padded_histories,
                previous_token_mask=history_mask,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
            )
            argmax_tokens = torch.argmax(group_logits, dim=-1)
            for local_index, global_index in enumerate(group_indices):
                sampled_list[global_index] = multinomial_sample_one_no_sync(probs[local_index : local_index + 1])
                argmax_list[global_index] = int(argmax_tokens[local_index].item())

        return [item for item in sampled_list if item is not None], [int(item) for item in argmax_list if item is not None]

    def _sample_active_batch_requests(
        self,
        model: Any,
        active_batch: Any,
        logits: torch.Tensor,
        *,
        max_steps: int,
    ) -> tuple[list[GPTSoVITSARFinishedItem], list[int], list[torch.LongTensor]]:
        finished_items: list[GPTSoVITSARFinishedItem] = []
        keep_indices: list[int] = []
        updated_sequences: list[torch.LongTensor] = []

        uniform_sampling_key = self._uniform_sampling_group_key(active_batch)
        sampled_items: list[torch.Tensor]
        argmax_tokens: list[int]
        sampled_token_tensor: torch.Tensor | None = None
        argmax_token_tensor: torch.Tensor | None = None
        if uniform_sampling_key is not None:
            sampled_tensor, argmax_tensor = self._batched_sample_uniform(
                logits=logits,
                histories=active_batch.y_sequences,
                sampling_key=uniform_sampling_key,
            )
            sampled_token_tensor = sampled_tensor.view(-1)
            argmax_token_tensor = argmax_tensor.view(-1)
            stacked_histories = self._stack_token_sequences_if_same_length(active_batch.y_sequences)
            if (
                all(state.early_stop_num == -1 for state in active_batch.states)
                and int(active_batch.step_indices[0].item()) + 1 < max_steps
                and not bool(sampled_token_tensor.eq(model.EOS).any().item())
                and not bool(argmax_token_tensor.eq(model.EOS).any().item())
            ):
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
            sampled_items = [sampled_tensor[index : index + 1] for index in range(sampled_tensor.shape[0])]
            argmax_tokens = [int(item) for item in argmax_tensor.tolist()]
        else:
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
            )

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
        return finished_items, keep_indices, updated_sequences

    def _decode_active_batch_one_step(
        self,
        model: Any,
        active_batch: Any,
        *,
        max_steps: int,
    ) -> tuple[Any | None, list[GPTSoVITSARFinishedItem]]:
        was_prefill = not active_batch.prefill_done
        if was_prefill:
            if active_batch.prefill_attn_mask is None or active_batch.key_padding_mask is None:
                raise ValueError("GPT-SoVITS AR prefill stage is missing masks")
            xy_dec, active_batch.k_cache, active_batch.v_cache = model.t2s_transformer.process_prompt(
                active_batch.xy_pos,
                active_batch.prefill_attn_mask,
                None,
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
                batched_decode_attn_mask = pool.build_decode_mask(next_kv_lens)
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
                active_batch.decode_attn_mask = self._advance_decode_mask(active_batch.decode_attn_mask, active_batch.kv_lens)

        logits = model.ar_predict_layer(xy_dec[:, -1])
        finished_items, keep_indices, updated_sequences = self._sample_active_batch_requests(
            model,
            active_batch,
            logits,
            max_steps=max_steps,
        )

        if len(keep_indices) == 0:
            if active_batch.kv_cache_pooled:
                self._set_kv_pool_active_rows(model, 0)
            return None, finished_items

        if len(keep_indices) == len(active_batch.request_ids):
            active_batch.y_sequences = updated_sequences
            active_batch.step_indices = active_batch.step_indices + 1
            if not was_prefill and active_batch.kv_lens is not None:
                active_batch.kv_lens = active_batch.kv_lens + 1
            active_batch.xy_pos = self._build_next_xy_pos(model, active_batch.y_sequences)
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

        active_batch.xy_pos = self._build_next_xy_pos(model, active_batch.y_sequences)
        return active_batch, finished_items

    def _run_prefill_active_batch(
        self,
        model: Any,
        states: Sequence[Any],
        *,
        max_steps: int,
    ) -> tuple[Any | None, list[GPTSoVITSARFinishedItem]]:
        if not states:
            return None, []
        active_batch = self._build_prefill_active_batch(model, states)
        return self._decode_active_batch_one_step(model, active_batch, max_steps=max_steps)

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
        pending = sorted(states, key=lambda item: (item.ready_step, item.request_id))
        active_batch: Any | None = None
        finished: list[GPTSoVITSARFinishedItem] = []
        current_tick = 0

        while pending or active_batch is not None:
            admitted: list[Any] = []
            while pending and pending[0].ready_step <= current_tick:
                admitted.append(pending.pop(0))

            admitted_active_batch, admitted_finished = self._run_prefill_active_batch(
                model,
                admitted,
                max_steps=max_steps,
            )
            finished.extend(admitted_finished)
            active_batch = self._merge_active_batches(model, active_batch, admitted_active_batch)

            if active_batch is not None:
                active_batch, step_finished = self._decode_active_batch_one_step(
                    model,
                    active_batch,
                    max_steps=max_steps,
                )
                finished.extend(step_finished)

            if active_batch is None and pending:
                current_tick = max(current_tick + 1, int(pending[0].ready_step))
                continue

            current_tick += 1

        finished.sort(key=lambda item: item.request_id)
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
                model_device = next(sv_self.embedding_model.parameters()).device
                feat = feat.to(device=model_device)
                if getattr(sv_self, "is_half", False):
                    feat = feat.half()
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
            top_k=int(inputs["top_k"]),
            top_p=float(inputs["top_p"]),
            temperature=float(inputs["temperature"]),
            repetition_penalty=float(inputs["repetition_penalty"]),
            early_stop_num=int(getattr(pipeline.configs, "hz", 50) * getattr(pipeline.configs, "max_sec", 30)),
            aux_ref_audio_paths=[str(item) for item in list(inputs.get("aux_ref_audio_paths") or [])],
            ready_step=int(request.get("ready_step", 0)),
        )

    @staticmethod
    def _clone_tensor_to_cpu(value: torch.Tensor | None, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        if value is None:
            return torch.empty((0,), dtype=dtype or torch.float32)
        cloned = value.detach().to("cpu").contiguous()
        if dtype is not None:
            cloned = cloned.to(dtype=dtype)
        return cloned

    def _state_to_transport_info(self, state: Any, request: dict[str, Any]) -> dict[str, Any]:
        refer_spec = getattr(state, "refer_spec", None)
        refer_audio_spec = refer_spec.spec_audio if refer_spec is not None else None
        refer_audio_16k = refer_spec.audio_16k if refer_spec is not None else None
        return {
            "gpt_sovits_request_id": str(state.request_id),
            "gpt_sovits_phones": self._clone_tensor_to_cpu(getattr(state, "phones", None), dtype=torch.long),
            "gpt_sovits_prompt_phones": self._clone_tensor_to_cpu(getattr(state, "prompt_phones", None), dtype=torch.long),
            "gpt_sovits_prompt_semantic": self._clone_tensor_to_cpu(getattr(state, "prompt_semantic", None), dtype=torch.long),
            "gpt_sovits_refer_audio_spec": self._clone_tensor_to_cpu(refer_audio_spec),
            "gpt_sovits_refer_audio_16k": self._clone_tensor_to_cpu(refer_audio_16k),
            "gpt_sovits_raw_audio": self._clone_tensor_to_cpu(getattr(state, "raw_audio", None)),
            "gpt_sovits_raw_sr": int(getattr(state, "raw_sr", 0)),
            "gpt_sovits_speed_factor": float(request.get("speed_factor", request.get("speed", 1.0) or 1.0)),
            "gpt_sovits_sample_steps": int(request.get("sample_steps", 32)),
            "gpt_sovits_super_sampling": bool(request.get("super_sampling", False)),
        }

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

    def _prepare_text_cpu(self, text: str, language: str) -> Any:
        pipeline = self._ensure_pipeline()
        return pipeline.prepare_text_segments(text, language)

    def _resolve_g2pw_segments(self, prepared_segments: Any) -> tuple[Any, dict[str, float]]:
        pipeline = self._ensure_pipeline()
        profile: dict[str, float] = {}
        resolved_segments = pipeline.resolve_g2pw_segments(prepared_segments, profile=profile)
        return resolved_segments, profile

    def _resolve_g2pw_segment_batches(
        self,
        prepared_segment_batches: list[Any],
    ) -> tuple[Any, list[dict[str, float]]]:
        pipeline = self._ensure_pipeline()
        profiles: list[dict[str, float]] = [{} for _ in prepared_segment_batches]
        resolved_batches = pipeline.resolve_g2pw_segments_batch(prepared_segment_batches, profiles=profiles)
        return resolved_batches, profiles

    def _load_ref_audio_raw(self, ref_audio_path: str) -> Any:
        pipeline = self._ensure_pipeline()
        return pipeline._load_ref_audio_raw(ref_audio_path)

    def _extract_ref_spec_from_raw(self, raw_audio: Any, raw_sr: int) -> tuple[GPTSoVITSReferSpec, dict[str, float]]:
        pipeline = self._ensure_pipeline()
        spec, audio, _, _, profile = pipeline._extract_ref_spec_profile_from_raw(raw_audio, raw_sr)
        return GPTSoVITSReferSpec(spec_audio=spec, audio_16k=audio), profile

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

    def _build_text_features(
        self,
        prepared_segments: Any,
        language: str | None,
        cpu_run_ms: float,
        base_profile: dict[str, float] | None = None,
    ) -> GPTSoVITSTextFeatures:
        pipeline = self._ensure_pipeline()
        profile: dict[str, float] = dict(base_profile or {})
        profile["cpu_preprocess_ms"] = float(cpu_run_ms)
        branch_start = time.perf_counter()
        phones, bert_features, norm_text = pipeline.build_text_features_from_segments(
            prepared_segments,
            profile=profile,
        )
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

    def _build_ref_prompt_semantic_from_raw(self, raw_audio: Any, raw_sr: int) -> GPTSoVITSRefAudioBundle:
        pipeline = self._ensure_pipeline()
        load_profile = {"audio_load_ms": 0.0}
        if getattr(pipeline, "prepare_ref_semantic_batch_worker", None) is not None:
            wav16k, local_cpu_prepare_profile = pipeline._prepare_ref_prompt_wav16k_for_worker(raw_audio, raw_sr)
            prompt_semantic, worker_profile = pipeline.prepare_ref_semantic_batch_worker.submit(
                raw_audio,
                raw_sr,
                wav16k=wav16k,
            )
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

        wav16k, cpu_prepare_ms, limiter_stats = pipeline._prepare_prompt_semantic_wav16k_profile(raw_audio, raw_sr)
        with pipeline.prepare_ref_semantic_stage_limiter.enter() as stage_stats:
            prompt_semantic, runtime_profile = pipeline._extract_prompt_semantic_profile_from_prepared_wav16k(wav16k)
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

    async def _run_text_cpu_stage(
        self,
        coordinator: Any,
        text: str,
        language: str,
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

        pipeline = self._ensure_pipeline()
        text_cpu_worker = getattr(pipeline, "prepare_text_cpu_worker", None)
        executor = getattr(pipeline, "prepare_text_cpu_executor", None)
        try:
            if text_cpu_worker is not None:
                submit_at = time.perf_counter()
                result, worker_profile = await text_cpu_worker.submit_async(text, language)
                return self._build_text_cpu_profiled_result(submit_at, result, dict(worker_profile))
            if executor is None:
                submit_at = time.perf_counter()
                return self._prepare_run_profiled(self._prepare_text_cpu, submit_at, text, language)
            return await self._prepare_run_on_executor(executor, self._prepare_text_cpu, text, language)
        finally:
            coordinator.text_cpu_gate.release()

    async def _run_text_cpu_stage_pair(
        self,
        coordinator: Any,
        prompt_text: str,
        prompt_lang: str,
        text: str,
        text_lang: str,
    ) -> tuple[GPTSoVITSPrepareProfiledResult, GPTSoVITSPrepareProfiledResult]:
        coordinator = self._coerce_prepare_coordinator(coordinator)
        pipeline = self._ensure_pipeline()
        text_cpu_worker = getattr(pipeline, "prepare_text_cpu_worker", None)
        if (
            text_cpu_worker is None
            or not hasattr(text_cpu_worker, "submit_many_async")
            or int(getattr(coordinator.text_cpu_gate, "max_inflight", 0)) > 0
        ):
            prompt_cpu_task = asyncio.create_task(self._run_text_cpu_stage(coordinator, prompt_text, prompt_lang))
            target_cpu_task = asyncio.create_task(self._run_text_cpu_stage(coordinator, text, text_lang))
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
        coordinator = self._ensure_prepare_coordinator()
        spec = self._build_scheduler_request_spec(request, request_id=request_id)
        ref_audio_prepare_future = None
        submit_at = time.perf_counter()
        if preload_ref_audio:
            ref_audio_prepare_future = self.preload_ref_audio_asset(str(spec.ref_audio_path), submit_at=submit_at)
        cpu_stage = self._run_awaitable_sync(
            self._prepare_cpu_stage_async(coordinator, spec, prepare_submit_at=submit_at)
        )
        return GPTSoVITSPreparedCpuStage(
            request_id=str(spec.request_id),
            request=dict(request),
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
        coordinator = self._ensure_prepare_coordinator()
        specs: list[Any] = []
        submit_ats: list[float] = []
        ref_audio_prepare_futures: list[Any | None] = []
        for index, request in enumerate(requests):
            request_id = str(request.get("engine_request_id") or f"gpt_sovits_{index}")
            spec = self._build_scheduler_request_spec(request, request_id=request_id)
            specs.append(spec)
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
        if len(cpu_stage_results) != len(requests):
            raise ValueError("GPT-SoVITS batch prepare CPU stage count mismatch")
        prepared_cpu_stages: list[GPTSoVITSPreparedCpuStage] = []
        for request, spec, cpu_stage, ref_audio_prepare_future in zip(
            requests,
            specs,
            cpu_stage_results,
            ref_audio_prepare_futures,
        ):
            if isinstance(cpu_stage, Exception):
                raise cpu_stage
            prepared_cpu_stages.append(
                GPTSoVITSPreparedCpuStage(
                    request_id=str(spec.request_id),
                    request=dict(request),
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

        pipeline = self._ensure_pipeline()
        await coordinator.text_feature_gate.acquire()
        profile: dict[str, float] = dict(base_profile or {})
        profile["cpu_preprocess_ms"] = float(cpu_run_ms)
        submit_at = time.perf_counter()
        started_at = float(submit_at)
        try:
            result_raw = await pipeline.build_text_features_from_segments_async(
                prepared_segments,
                profile=profile,
            )
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
        pipeline = self._ensure_pipeline()
        g2pw_batch_worker = getattr(pipeline, "prepare_g2pw_batch_worker", None)
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

        await coordinator.text_feature_gate.acquire()
        target_profile: dict[str, float] = dict(target_base_profile or {})
        target_profile["cpu_preprocess_ms"] = float(target_cpu_run_ms)
        submit_at = time.perf_counter()
        started_at = float(submit_at)
        try:
            if prompt_is_empty:
                target_result_raw = await pipeline.build_text_features_from_segments_async(
                    target_segments,
                    profile=target_profile,
                )
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
            prompt_result_raw, target_result_raw = await pipeline.build_text_feature_pair_from_segments_async(
                prompt_segments,
                target_segments,
                prompt_profile=prompt_profile,
                target_profile=target_profile,
            )
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
        pipeline = self._ensure_pipeline()
        if getattr(pipeline, "prepare_ref_semantic_batch_worker", None) is not None:
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

            prompt_semantic_task = asyncio.create_task(
                pipeline.prepare_ref_semantic_batch_worker.submit_async(raw_audio, raw_sr, wav16k=wav16k)
            )
            prompt_semantic, prompt_semantic_profile = await prompt_semantic_task
            limiter_snapshot = (
                pipeline.prepare_ref_semantic_stage_limiter.snapshot()
                if getattr(pipeline, "prepare_ref_semantic_stage_limiter", None) is not None
                else {}
            )
            prompt_semantic_ms = (
                float(prompt_semantic_profile.get("prompt_semantic_cpu_prepare_ms", 0.0))
                + float(prompt_semantic_profile.get("prompt_semantic_forward_ms", 0.0))
                + float(prompt_semantic_profile.get("prompt_semantic_scatter_ms", 0.0))
            )
            finished_at = time.perf_counter()
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
            )
            return GPTSoVITSPrepareProfiledResult(
                result=result,
                submit_at=float(submit_at),
                started_at=started_at,
                finished_at=float(finished_at),
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
                with pipeline.prepare_ref_semantic_stage_limiter.enter() as stage_stats:
                    prompt_semantic, runtime_profile = await asyncio.to_thread(
                        pipeline._extract_prompt_semantic_profile_from_prepared_wav16k,
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
        ref_audio_tasks = [
            asyncio.create_task(
                self._run_ref_prompt_semantic_stage(
                    coordinator,
                    str(prepared_cpu_stage.spec.ref_audio_path),
                    prepared_asset_future=prepared_cpu_stage.ref_audio_prepare_future,
                )
            )
            for prepared_cpu_stage in prepared_cpu_stages
        ]
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
            ref_audio_results = await asyncio.gather(*ref_audio_tasks, return_exceptions=True)
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
            for task in ref_audio_tasks:
                if not task.done():
                    task.cancel()

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
            aux_spec_audio, aux_audio_16k, _, _ = pipeline.extract_ref_spec(str(aux_ref_audio_path))
            aux_refer_specs.append((aux_spec_audio, aux_audio_16k))
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
            transport_info=self._state_to_transport_info(state, prepared_cpu_stage.request),
        )

    def prepare_request(self, request: dict[str, Any], *, request_id: str | None = None) -> GPTSoVITSPreparedRequest:
        prepared_cpu_stage = self.prepare_request_cpu_stage(request, request_id=request_id)
        prepared_audio_phase = self.prepare_request_gpu_audio_phase(prepared_cpu_stage)
        prepared_ref_spec_phase = self.prepare_request_ref_spec_phase(prepared_audio_phase)
        prepared_text_phase = self.prepare_request_gpu_text_phase(prepared_audio_phase)
        return self.build_prepared_request_from_phases(prepared_text_phase, prepared_ref_spec_phase=prepared_ref_spec_phase)

    def prepare_requests(self, requests: list[dict[str, Any]]) -> list[GPTSoVITSPreparedRequest]:
        if not requests:
            return []
        prepared_cpu_stages = self.prepare_request_cpu_stages(requests)
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
        enable_pooled_kv = str(os.environ.get("GPT_SOVITS_ENABLE_AR_KV_POOL", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        xy_dec, active_batch.k_cache, active_batch.v_cache = model.t2s_transformer.process_prompt(
            active_batch.xy_pos,
            active_batch.prefill_attn_mask,
            None,
        )
        active_batch.kv_lens = active_batch.x_lens + active_batch.prefix_lens
        if active_batch.k_cache is None or active_batch.v_cache is None or active_batch.kv_lens is None:
            raise ValueError("GPT-SoVITS AR prefill did not produce KV cache")

        packed_into_pool = False
        if enable_pooled_kv:
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
        resolved_request_id = request_id or request.get("engine_request_id") or f"gpt_sovits_ar_{time.time_ns()}"
        with self._run_lock:
            with torch.inference_mode(False), torch.no_grad():
                prepared = self.prepare_request(request, request_id=str(resolved_request_id))
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

    @staticmethod
    def _tensor_from_transport(
        info: dict[str, Any],
        key: str,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = info.get(key)
        if isinstance(value, torch.Tensor):
            return value.detach().contiguous().to(dtype=dtype)
        if value is None:
            return torch.empty((0,), dtype=dtype)
        return torch.as_tensor(value, dtype=dtype)

    def prepare_decode_request(
        self,
        semantic_tokens: torch.Tensor,
        transport_info: dict[str, Any],
    ) -> GPTSoVITSDecodePreparedRequest:
        pipeline = self._ensure_pipeline()
        device = torch.device(getattr(pipeline.configs, "device", "cpu"))
        request_id = str(
            transport_info.get("gpt_sovits_request_id")
            or transport_info.get("engine_request_id")
            or transport_info.get("request_id")
            or f"gpt_sovits_decode_{time.time_ns()}"
        )
        return GPTSoVITSDecodePreparedRequest(
            request_id=request_id,
            semantic_tokens=semantic_tokens.detach().reshape(-1).to(device=device, dtype=torch.long),
            phones=self._tensor_from_transport(transport_info, "gpt_sovits_phones", dtype=torch.long).to(device=device),
            prompt_phones=self._tensor_from_transport(
                transport_info,
                "gpt_sovits_prompt_phones",
                dtype=torch.long,
            ).to(device=device),
            prompt_semantic=self._tensor_from_transport(
                transport_info,
                "gpt_sovits_prompt_semantic",
                dtype=torch.long,
            ).to(device=device),
            refer_audio_spec=self._tensor_from_transport(
                transport_info,
                "gpt_sovits_refer_audio_spec",
                dtype=torch.float32,
            ),
            refer_audio_16k=self._tensor_from_transport(
                transport_info,
                "gpt_sovits_refer_audio_16k",
                dtype=torch.float32,
            ),
            raw_audio=self._tensor_from_transport(transport_info, "gpt_sovits_raw_audio", dtype=torch.float32),
            raw_sr=int(transport_info.get("gpt_sovits_raw_sr", 0)),
            speed_factor=float(transport_info.get("gpt_sovits_speed_factor", 1.0)),
            sample_steps=int(transport_info.get("gpt_sovits_sample_steps", 32)),
            super_sampling=bool(transport_info.get("gpt_sovits_super_sampling", False)),
        )

    def prepare_decode_requests(
        self,
        semantic_tokens_list: list[torch.Tensor],
        transport_infos: list[dict[str, Any]],
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

    def _build_vocoder_prompt_context(
        self,
        pipeline: Any,
        prepared: GPTSoVITSDecodePreparedRequest,
        refer_audio_spec: torch.Tensor,
    ) -> dict[str, Any]:
        device = torch.device(pipeline.configs.device)
        prompt_semantic_tokens = prepared.prompt_semantic.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.long)
        prompt_phones = prepared.prompt_phones.unsqueeze(0).to(device=device, dtype=torch.long)
        refer_audio_spec = refer_audio_spec.to(dtype=pipeline.precision, device=device)

        fea_ref, ge = pipeline.vits_model.decode_encp(prompt_semantic_tokens, prompt_phones, refer_audio_spec)
        ref_audio = prepared.raw_audio.to(device=device, dtype=torch.float32)
        if ref_audio.ndim == 1:
            ref_audio = ref_audio.unsqueeze(0)
        if ref_audio.shape[0] == 2:
            ref_audio = ref_audio.mean(0).unsqueeze(0)

        target_sr = 24000 if pipeline.configs.version == "v3" else 32000
        if int(prepared.raw_sr) != target_sr:
            ref_audio = self._resample_audio(ref_audio, int(prepared.raw_sr), target_sr, device)

        mel2 = self._compute_vocoder_mel(ref_audio, version=str(pipeline.configs.version))
        mel2 = self._norm_vocoder_spec(mel2)
        t_min = min(int(mel2.shape[2]), int(fea_ref.shape[2]))
        mel2 = mel2[:, :, :t_min]
        fea_ref = fea_ref[:, :, :t_min]
        t_ref = int(pipeline.vocoder_configs["T_ref"])
        t_chunk = int(pipeline.vocoder_configs["T_chunk"])
        if t_min > t_ref:
            mel2 = mel2[:, :, -t_ref:]
            fea_ref = fea_ref[:, :, -t_ref:]
            t_min = t_ref
        chunk_len = t_chunk - t_min

        return {
            "refer_audio_spec": refer_audio_spec,
            "fea_ref": fea_ref,
            "ge": ge,
            "mel2": mel2.to(dtype=pipeline.precision),
            "t_min": int(t_min),
            "chunk_len": int(chunk_len),
            "output_sr": int(pipeline.vocoder_configs["sr"]),
        }

    def _decode_vocoder_with_prompt_context(
        self,
        pipeline: Any,
        prepared: GPTSoVITSDecodePreparedRequest,
        prompt_context: dict[str, Any],
    ) -> tuple[Any, int]:
        device = torch.device(pipeline.configs.device)
        refer_audio_spec = prompt_context["refer_audio_spec"]
        ge = prompt_context["ge"]
        fea_ref_base = prompt_context["fea_ref"]
        mel2_base = prompt_context["mel2"]
        t_min = int(prompt_context["t_min"])
        chunk_len = int(prompt_context["chunk_len"])
        semantic_tokens = prepared.semantic_tokens.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.long)
        phones = prepared.phones.unsqueeze(0).to(device=device, dtype=torch.long)
        fea_todo, ge = pipeline.vits_model.decode_encp(
            semantic_tokens,
            phones,
            refer_audio_spec,
            ge,
            float(prepared.speed_factor),
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
            cfm_res = pipeline.vits_model.cfm.inference(
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
        wav_gen = pipeline.vocoder(cfm_res)
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
        if not prepared_requests:
            return []

        first_speed = float(prepared_requests[0].speed_factor)
        first_sample_steps = int(prepared_requests[0].sample_steps)
        if any(abs(float(item.speed_factor) - first_speed) > 1e-6 for item in prepared_requests):
            raise ValueError("GPT-SoVITS batched vocoder decode requires identical speed_factor")
        if any(int(item.sample_steps) != first_sample_steps for item in prepared_requests):
            raise ValueError("GPT-SoVITS batched vocoder decode requires identical sample_steps")

        chunk_len = int(prompt_context["chunk_len"])
        overlapped_len = int(pipeline.vocoder_configs.get("overlapped_len", 0))
        upsample_rate = int(pipeline.vocoder_configs.get("upsample_rate", 1))
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

        device = torch.device(pipeline.configs.device)
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
            feat, _ = pipeline.vits_model.decode_encp(
                semantic_tokens,
                phones,
                refer_audio_spec,
                ge,
                first_speed,
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
        pred_spec = pipeline.vits_model.cfm.inference(
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

        wav_gen = pipeline.vocoder(pred_spec)
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

    def _decode_prepared_request_fragment(
        self,
        pipeline: Any,
        prepared: GPTSoVITSDecodePreparedRequest,
    ) -> tuple[Any, int]:
        refer_spec = self._build_refer_spec_from_prepared(prepared)
        if bool(getattr(pipeline.configs, "use_vocoder", False)):
            return self._decode_prepared_request_vocoder_fragment(
                pipeline,
                prepared,
                refer_spec.spec_audio,
            )

        refer_audio_spec_list = [refer_spec.spec_audio.to(dtype=pipeline.precision, device=pipeline.configs.device)]
        sv_emb = None
        if bool(getattr(pipeline, "is_v2pro", False)):
            audio_tensor = refer_spec.audio_16k
            if audio_tensor is None:
                raise ValueError("GPT-SoVITS v2Pro request-local synthesis 缺少 16k 参考音频")
            sv_emb = [pipeline.sv_model.compute_embedding3(audio_tensor).to(pipeline.configs.device)]
        audio_fragment = pipeline.vits_model.decode(
            prepared.semantic_tokens.unsqueeze(0).unsqueeze(0),
            prepared.phones.unsqueeze(0),
            refer_audio_spec_list,
            speed=float(prepared.speed_factor),
            sv_emb=sv_emb,
        ).detach()[0, 0, :]
        return audio_fragment, int(pipeline.configs.sampling_rate)

    def _decode_prepared_requests_batched_non_vocoder(
        self,
        pipeline: Any,
        prepared_requests: list[GPTSoVITSDecodePreparedRequest],
    ) -> list[GPTSoVITSDecodedAudio]:
        if not prepared_requests:
            return []
        if bool(getattr(pipeline.configs, "use_vocoder", False)):
            raise ValueError("non-vocoder batched decode helper received vocoder pipeline")

        device = torch.device(pipeline.configs.device)
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
                shared_refer_audio_spec = first_spec.to(dtype=pipeline.precision, device=device)

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
            refer_audio_specs.append(refer_spec.spec_audio.to(dtype=pipeline.precision, device=device))
            if bool(getattr(pipeline, "is_v2pro", False)):
                if refer_spec.audio_16k is None:
                    raise ValueError("GPT-SoVITS v2Pro batched non-vocoder decode 缺少 16k 参考音频")
                sv_emb_list.append(pipeline.sv_model.compute_embedding3(refer_spec.audio_16k).to(device))

        if bool(getattr(pipeline, "is_v2pro", False)):
            if shared_single_refer:
                shared_audio_tensor = refer_specs[0].audio_16k
                if shared_audio_tensor is None:
                    raise ValueError("GPT-SoVITS v2Pro batched non-vocoder decode 缺少 16k 参考音频")
                sv_emb_batch = pipeline.sv_model.compute_embedding3(shared_audio_tensor).to(device)
            else:
                sv_emb_batch = torch.cat(sv_emb_list, dim=0)

        precomputed_ge = None
        if shared_single_refer:
            if shared_refer_audio_spec is None:
                raise ValueError("shared_single_refer detected but refer_audio_spec missing")
            refer_audio_specs = [shared_refer_audio_spec]
            with self._project_root_cwd():
                from GPT_SoVITS.module import commons

            refer_lengths = torch.LongTensor([int(shared_refer_audio_spec.size(2))]).to(device)
            refer_mask = torch.unsqueeze(
                commons.sequence_mask(refer_lengths, int(shared_refer_audio_spec.size(2))),
                1,
            ).to(shared_refer_audio_spec.dtype)
            if pipeline.vits_model.version == "v1":
                precomputed_ge = pipeline.vits_model.ref_enc(shared_refer_audio_spec * refer_mask, refer_mask)
            else:
                precomputed_ge = pipeline.vits_model.ref_enc(shared_refer_audio_spec[:, :704] * refer_mask, refer_mask)
            if bool(getattr(pipeline, "is_v2pro", False)):
                if sv_emb_batch is None:
                    raise ValueError("GPT-SoVITS v2Pro batched non-vocoder decode 缺少 sv_emb")
                precomputed_ge = precomputed_ge + pipeline.vits_model.sv_emb(sv_emb_batch[:1]).unsqueeze(-1)
                precomputed_ge = pipeline.vits_model.prelu(precomputed_ge)

        with self._run_lock:
            audio_batch, audio_lengths = pipeline.vits_model.decode_batched_request_local(
                codes=semantic_batch,
                code_lengths=torch.LongTensor(semantic_lengths).to(device),
                text=phone_batch,
                text_lengths=torch.LongTensor(phone_lengths).to(device),
                refer_list=refer_audio_specs,
                speed=first_speed,
                sv_emb=sv_emb_batch,
                shared_refer=shared_single_refer,
                precomputed_ge=precomputed_ge,
            )
        audio_fragments = [
            audio_batch[batch_index, 0, : int(audio_lengths[batch_index].item())].detach()
            for batch_index in range(batch_size)
        ]
        output_sr = int(pipeline.configs.sampling_rate)
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
            pipeline.init_sr_model()
            if not getattr(pipeline, "sr_model_not_exist", False):
                audio, sr = pipeline.sr_model(audio.unsqueeze(0), sr)
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
        transport_info: dict[str, Any],
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
        return "cut4" if text_lang == "en" else "cut5"

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
            "speed_factor": float(request.get("speed_factor", 1.0)),
            "fragment_interval": float(request.get("fragment_interval", 0.3)),
            "seed": int(request.get("seed", -1)),
            "parallel_infer": bool(request.get("parallel_infer", True)),
            "repetition_penalty": float(request.get("repetition_penalty", 1.35)),
            "sample_steps": int(request.get("sample_steps", 32)),
            "streaming_mode": False,
            "return_fragment": False,
        }

    def synthesize(self, request: dict[str, Any]) -> GPTSoVITSResult:
        pipeline = self._ensure_pipeline()
        inputs = self.build_tts_inputs(request)

        final_chunk: tuple[int, np.ndarray] | None = None
        with self._run_lock:
            with self._project_root_cwd():
                for sample_rate, audio in pipeline.run(inputs):
                    final_chunk = (int(sample_rate), np.asarray(audio))

        if final_chunk is None:
            raise RuntimeError("GPT-SoVITS returned no audio chunk")

        sample_rate, audio = final_chunk
        audio = self._normalize_audio(audio)
        return GPTSoVITSResult(sample_rate=sample_rate, audio=audio)


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
