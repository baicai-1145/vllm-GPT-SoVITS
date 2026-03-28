from __future__ import annotations

import asyncio
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
class GPTSoVITSPreparedCpuStage:
    request_id: str
    request: dict[str, Any]
    spec: Any
    cpu_stage: Any
    ref_audio_prepare_future: Any | None = None


@dataclass(slots=True)
class GPTSoVITSPreparedAudioPhase:
    request_id: str
    prepared_cpu_stage: GPTSoVITSPreparedCpuStage
    phase_one: dict[str, Any]


@dataclass(slots=True)
class GPTSoVITSPreparedRefSpecPhase:
    request_id: str
    prepared_audio_phase: GPTSoVITSPreparedAudioPhase
    ref_spec_result: tuple[tuple[Any, Any], dict[str, float]]


@dataclass(slots=True)
class GPTSoVITSPreparedTextPhase:
    request_id: str
    prepared_audio_phase: GPTSoVITSPreparedAudioPhase
    phase_two: dict[str, Any]


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
        self._t2s_active_batch_cls: Any | None = None
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

    def _ensure_prepare_coordinator(self) -> Any:
        if self._prepare_coordinator is not None:
            return self._prepare_coordinator
        with self._init_lock:
            if self._prepare_coordinator is not None:
                return self._prepare_coordinator
            pipeline = self._ensure_pipeline()
            with self._project_root_cwd():
                from GPT_SoVITS.TTS_infer_pack.prepare_coordinator import PrepareCoordinator

                self._prepare_coordinator = PrepareCoordinator(pipeline)
        return self._prepare_coordinator

    def _get_t2s_active_batch_cls(self) -> Any:
        if self._t2s_active_batch_cls is not None:
            return self._t2s_active_batch_cls
        with self._init_lock:
            if self._t2s_active_batch_cls is not None:
                return self._t2s_active_batch_cls
            self._ensure_import_path()
            self._ensure_native_runtime_deps()
            with self._project_root_cwd():
                from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SActiveBatch

            self._t2s_active_batch_cls = T2SActiveBatch
        return self._t2s_active_batch_cls

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

        active_batch_cls = self._get_t2s_active_batch_cls()
        return active_batch_cls(
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

    def _build_scheduler_request_spec(self, request: dict[str, Any], request_id: str | None = None) -> Any:
        pipeline = self._ensure_pipeline()
        inputs = self.build_tts_inputs(request)
        with self._project_root_cwd():
            from pathlib import Path

            from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import SchedulerRequestSpec

            return SchedulerRequestSpec(
                request_id=str(request_id or request.get("request_id") or request.get("engine_request_id") or "gpt_sovits"),
                ref_audio_path=Path(inputs["ref_audio_path"]),
                prompt_text=inputs["prompt_text"],
                prompt_lang=inputs["prompt_lang"],
                text=inputs["text"],
                text_lang=inputs["text_lang"],
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
        refer_audio_spec = refer_spec[0] if refer_spec is not None else None
        refer_audio_16k = refer_spec[1] if refer_spec is not None else None
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
        with self._project_root_cwd():
            return coordinator.submit_prepare_ref_audio_asset(ref_audio_path, submit_at=submit_at)

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
        if preload_ref_audio:
            ref_audio_prepare_future = self.preload_ref_audio_asset(str(spec.ref_audio_path), submit_at=time.perf_counter())
        cpu_stage = self._run_awaitable_sync(
            coordinator.prepare_cpu_stage_profiled_async(spec, time.perf_counter())
        )
        return GPTSoVITSPreparedCpuStage(
            request_id=str(spec.request_id),
            request=dict(request),
            spec=spec,
            cpu_stage=cpu_stage,
            ref_audio_prepare_future=ref_audio_prepare_future,
        )

    def prepare_request_gpu_audio_phase(
        self,
        prepared_cpu_stage: GPTSoVITSPreparedCpuStage,
    ) -> GPTSoVITSPreparedAudioPhase:
        coordinator = self._ensure_prepare_coordinator()
        phase_one_results = self._run_awaitable_sync(
            coordinator.prepare_gpu_audio_phases_async(
                [prepared_cpu_stage.cpu_stage],
                prepared_ref_audio_futures=[prepared_cpu_stage.ref_audio_prepare_future],
            )
        )
        if not phase_one_results:
            raise ValueError("GPT-SoVITS prepare_gpu_audio_phases_async returned no results")
        phase_one = phase_one_results[0]
        if isinstance(phase_one, Exception):
            raise phase_one
        return GPTSoVITSPreparedAudioPhase(
            request_id=prepared_cpu_stage.request_id,
            prepared_cpu_stage=prepared_cpu_stage,
            phase_one=phase_one,
        )

    def prepare_request_gpu_text_phase(
        self,
        prepared_audio_phase: GPTSoVITSPreparedAudioPhase,
    ) -> GPTSoVITSPreparedTextPhase:
        coordinator = self._ensure_prepare_coordinator()
        phase_two_results = self._run_awaitable_sync(
            coordinator.prepare_gpu_text_phases_async(
                [(prepared_audio_phase.prepared_cpu_stage.cpu_stage, prepared_audio_phase.phase_one)]
            )
        )
        if not phase_two_results:
            raise ValueError("GPT-SoVITS prepare_gpu_text_phases_async returned no results")
        phase_two = phase_two_results[0]
        if isinstance(phase_two, Exception):
            raise phase_two
        return GPTSoVITSPreparedTextPhase(
            request_id=prepared_audio_phase.request_id,
            prepared_audio_phase=prepared_audio_phase,
            phase_two=phase_two,
        )

    def prepare_request_ref_spec_phase(
        self,
        prepared_audio_phase: GPTSoVITSPreparedAudioPhase,
    ) -> GPTSoVITSPreparedRefSpecPhase:
        coordinator = self._ensure_prepare_coordinator()
        ref_spec_results = self._run_awaitable_sync(
            coordinator.prepare_ref_spec_stages_async([prepared_audio_phase.phase_one])
        )
        if not ref_spec_results:
            raise ValueError("GPT-SoVITS prepare_ref_spec_stages_async returned no results")
        ref_spec_result = ref_spec_results[0]
        if isinstance(ref_spec_result, Exception):
            raise ref_spec_result
        return GPTSoVITSPreparedRefSpecPhase(
            request_id=prepared_audio_phase.request_id,
            prepared_audio_phase=prepared_audio_phase,
            ref_spec_result=ref_spec_result,
        )

    def build_prepared_request_from_phases(
        self,
        prepared_text_phase: GPTSoVITSPreparedTextPhase,
        prepared_ref_spec_phase: GPTSoVITSPreparedRefSpecPhase | None = None,
    ) -> GPTSoVITSPreparedRequest:
        coordinator = self._ensure_prepare_coordinator()
        prepared_cpu_stage = prepared_text_phase.prepared_audio_phase.prepared_cpu_stage
        state, _, _ = coordinator.build_gpu_prepare_result_from_phases(
            prepared_cpu_stage.cpu_stage,
            prepared_text_phase.prepared_audio_phase.phase_one,
            prepared_text_phase.phase_two,
            extra_profile=None,
        )
        if prepared_ref_spec_phase is not None:
            coordinator.apply_ref_spec_result_to_state(state, prepared_ref_spec_phase.ref_spec_result)
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
        prepared: list[GPTSoVITSPreparedRequest] = []
        for index, request in enumerate(requests):
            prepared.append(self.prepare_request(request, request_id=str(request.get("engine_request_id") or f"gpt_sovits_{index}")))
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
            with self._project_root_cwd():
                from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import run_scheduler_continuous

                finished_items = run_scheduler_continuous(pipeline.t2s_model.model, states, max_steps=max_steps)
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

        refer_spec = (
            prepared.refer_audio_spec,
            None if prepared.refer_audio_16k.numel() == 0 else prepared.refer_audio_16k,
        )
        with self._run_lock:
            with self._project_root_cwd():
                audio_fragment = pipeline.synthesize_audio_request_local(
                    semantic_tokens=prepared.semantic_tokens.unsqueeze(0).unsqueeze(0),
                    phones=prepared.phones.unsqueeze(0),
                    prompt_semantic=prepared.prompt_semantic,
                    prompt_phones=prepared.prompt_phones,
                    refer_spec=[refer_spec],
                    raw_audio=prepared.raw_audio,
                    raw_sr=int(prepared.raw_sr),
                    speed=float(prepared.speed_factor),
                    sample_steps=int(prepared.sample_steps),
                )
                output_sr = (
                    int(pipeline.configs.sampling_rate)
                    if not pipeline.configs.use_vocoder
                    else int(pipeline.vocoder_configs["sr"])
                )
        return GPTSoVITSDecodedAudio(
            request_id=prepared.request_id,
            audio_fragment=audio_fragment,
            output_sr=int(output_sr),
            speed_factor=float(prepared.speed_factor),
            super_sampling=bool(prepared.super_sampling),
        )

    def finalize_decoded_audio(
        self,
        decoded: GPTSoVITSDecodedAudio,
    ) -> GPTSoVITSResult:
        pipeline = self._ensure_pipeline()
        with self._run_lock:
            with self._project_root_cwd():
                sample_rate, audio = pipeline.audio_postprocess(
                    audio=[[decoded.audio_fragment]],
                    sr=int(decoded.output_sr),
                    batch_index_list=None,
                    speed_factor=float(decoded.speed_factor),
                    split_bucket=False,
                    fragment_interval=0.0,
                    super_sampling=bool(decoded.super_sampling),
                )
        return GPTSoVITSResult(sample_rate=int(sample_rate), audio=self._normalize_audio(np.asarray(audio)))

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
