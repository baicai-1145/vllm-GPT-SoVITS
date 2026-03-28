from __future__ import annotations

import os
import sys
import threading
import ctypes
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

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


@dataclass(slots=True)
class GPTSoVITSResult:
    sample_rate: int
    audio: np.ndarray


@dataclass(slots=True)
class GPTSoVITSPreparedRequest:
    request_id: str
    state: Any
    transport_info: dict[str, Any]


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

    def prepare_request(self, request: dict[str, Any], *, request_id: str | None = None) -> GPTSoVITSPreparedRequest:
        pipeline = self._ensure_pipeline()
        spec = self._build_scheduler_request_spec(request, request_id=request_id)
        with self._project_root_cwd():
            from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import _prepare_request_state_legacy

            state = _prepare_request_state_legacy(pipeline, spec)
        return GPTSoVITSPreparedRequest(
            request_id=str(spec.request_id),
            state=state,
            transport_info=self._state_to_transport_info(state, request),
        )

    def prepare_requests(self, requests: list[dict[str, Any]]) -> list[GPTSoVITSPreparedRequest]:
        prepared: list[GPTSoVITSPreparedRequest] = []
        for index, request in enumerate(requests):
            prepared.append(self.prepare_request(request, request_id=str(request.get("engine_request_id") or f"gpt_sovits_{index}")))
        return prepared

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

    def decode_semantic_tokens_from_transport(
        self,
        semantic_tokens: torch.Tensor,
        transport_info: dict[str, Any],
    ) -> GPTSoVITSResult:
        pipeline = self._ensure_pipeline()
        device = torch.device(getattr(pipeline.configs, "device", "cpu"))
        semantic_tokens = semantic_tokens.detach().reshape(-1).to(device=device, dtype=torch.long)
        if semantic_tokens.numel() == 0:
            return GPTSoVITSResult(
                sample_rate=int(getattr(pipeline.configs, "sampling_rate", 32000)),
                audio=np.zeros((0,), dtype=np.float32),
            )

        phones = self._tensor_from_transport(transport_info, "gpt_sovits_phones", dtype=torch.long).to(device=device)
        prompt_phones = self._tensor_from_transport(transport_info, "gpt_sovits_prompt_phones", dtype=torch.long).to(device=device)
        prompt_semantic = self._tensor_from_transport(transport_info, "gpt_sovits_prompt_semantic", dtype=torch.long).to(
            device=device
        )
        refer_audio_spec = self._tensor_from_transport(transport_info, "gpt_sovits_refer_audio_spec", dtype=torch.float32)
        refer_audio_16k = self._tensor_from_transport(transport_info, "gpt_sovits_refer_audio_16k", dtype=torch.float32)
        raw_audio = self._tensor_from_transport(transport_info, "gpt_sovits_raw_audio", dtype=torch.float32)
        raw_sr = int(transport_info.get("gpt_sovits_raw_sr", 0))
        speed_factor = float(transport_info.get("gpt_sovits_speed_factor", 1.0))
        sample_steps = int(transport_info.get("gpt_sovits_sample_steps", 32))
        super_sampling = bool(transport_info.get("gpt_sovits_super_sampling", False))

        refer_spec = (refer_audio_spec, None if refer_audio_16k.numel() == 0 else refer_audio_16k)
        with self._run_lock:
            with self._project_root_cwd():
                audio_fragment = pipeline.synthesize_audio_request_local(
                    semantic_tokens=semantic_tokens.unsqueeze(0).unsqueeze(0),
                    phones=phones.unsqueeze(0),
                    prompt_semantic=prompt_semantic,
                    prompt_phones=prompt_phones,
                    refer_spec=[refer_spec],
                    raw_audio=raw_audio,
                    raw_sr=raw_sr,
                    speed=speed_factor,
                    sample_steps=sample_steps,
                )
                output_sr = (
                    int(pipeline.configs.sampling_rate)
                    if not pipeline.configs.use_vocoder
                    else int(pipeline.vocoder_configs["sr"])
                )
                sample_rate, audio = pipeline.audio_postprocess(
                    audio=[[audio_fragment]],
                    sr=output_sr,
                    batch_index_list=None,
                    speed_factor=speed_factor,
                    split_bucket=False,
                    fragment_interval=0.0,
                    super_sampling=super_sampling,
                )
        return GPTSoVITSResult(sample_rate=int(sample_rate), audio=self._normalize_audio(np.asarray(audio)))

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
