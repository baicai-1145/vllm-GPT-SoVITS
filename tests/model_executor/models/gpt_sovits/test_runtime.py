from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.gpt_sovits.runtime import (
    GPTSoVITSActiveBatch,
    GPTSoVITSARFinishedItem,
    GPTSoVITSARSession,
    GPTSoVITSDecodePreparedRequest,
    GPTSoVITSDecodePreparedRequestGroup,
    GPTSoVITSDecodedAudio,
    GPTSoVITSDecodedAudioGroup,
    GPTSoVITSMultiSegmentRequestState,
    GPTSoVITSNativePreparedCpuStage,
    GPTSoVITSPrepareAudioPhaseData,
    GPTSoVITSPreparedAudioPhase,
    GPTSoVITSPreparedCpuStage,
    GPTSoVITSPrepareProfiledResult,
    GPTSoVITSReferSpec,
    GPTSoVITSRefAudioBundle,
    GPTSoVITSPrepareRefSpecResult,
    GPTSoVITSPrepareRuntimeCoordinator,
    GPTSoVITSRequestSpec,
    GPTSoVITSResult,
    GPTSoVITSPrepareTextPhaseData,
    GPTSoVITSPreparedRefAudioAsset,
    GPTSoVITSPreparedRefSpecPhase,
    GPTSoVITSPreparedTextPhase,
    GPTSoVITSStageTransport,
    GPTSoVITST2SRequestState,
    GPTSoVITSTextSegmentFeatures,
    GPTSoVITSRuntime,
)


_REPO_ROOT = Path(__file__).resolve().parents[4]
_GPT_SOVITS_PROJECT_ROOT = Path("/root/GPT-SoVITS")
_G2PW_CU_RUNTIME_LIB = _GPT_SOVITS_PROJECT_ROOT / "third_party" / "g2pw-cu" / "build" / "libg2pw_runtime.so"
_G2PW_CU_MANIFEST = _GPT_SOVITS_PROJECT_ROOT / "third_party" / "g2pw-cu" / "artifacts" / "model" / "manifest.txt"
_G2PW_CU_WEIGHTS = _GPT_SOVITS_PROJECT_ROOT / "third_party" / "g2pw-cu" / "artifacts" / "model" / "weights.bin"
_GPT_SOVITS_S1_V3 = _GPT_SOVITS_PROJECT_ROOT / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"
_GPT_SOVITS_V2PROPLUS = (
    _GPT_SOVITS_PROJECT_ROOT / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Gv2ProPlus.pth"
)
_G2PW_CU_RUNTIME_CONFIG = (
    _REPO_ROOT
    / "tests"
    / "model_executor"
    / "models"
    / "gpt_sovits"
    / "fixtures"
    / "tts_infer_v2proplus_cpu_longprompt.yaml"
)
_VENDORED_GPT_SOVITS_RUNTIME_ROOT = (
    _REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "gpt_sovits" / "runtime_lib"
)


def _ensure_vendored_gpt_sovits_import_path() -> None:
    runtime_root = str(_VENDORED_GPT_SOVITS_RUNTIME_ROOT)
    package_root = str(_VENDORED_GPT_SOVITS_RUNTIME_ROOT / "GPT_SoVITS")
    for candidate in (runtime_root, package_root):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def _has_g2pw_cuda_runtime_test_assets() -> bool:
    return torch.cuda.is_available() and all(
        path.exists()
        for path in (
            _G2PW_CU_RUNTIME_LIB,
            _G2PW_CU_MANIFEST,
            _G2PW_CU_WEIGHTS,
            _GPT_SOVITS_S1_V3,
            _GPT_SOVITS_V2PROPLUS,
            _G2PW_CU_RUNTIME_CONFIG,
        )
    )


def test_ref_audio_loader_falls_back_to_soundfile(tmp_path):
    wav_path = tmp_path / "ref.wav"
    samples = np.linspace(-0.2, 0.2, num=320, dtype=np.float32)
    sf.write(wav_path, samples, 16000)

    class DummyPipeline:
        def __init__(self) -> None:
            self.torchaudio_calls = 0

        def _load_ref_audio_raw(self, ref_audio_path: str):
            self.torchaudio_calls += 1
            raise RuntimeError(f"torchaudio failed for {ref_audio_path}")

    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = DummyPipeline()

    runtime._install_ref_audio_loader_fallback(pipeline)
    raw_audio, raw_sr = pipeline._load_ref_audio_raw(str(wav_path))

    assert pipeline.torchaudio_calls == 1
    assert raw_sr == 16000
    assert raw_audio.dtype == torch.float32
    assert raw_audio.shape == (1, samples.shape[0])
    np.testing.assert_allclose(raw_audio.numpy()[0], samples, atol=5e-4)


def test_sv_embedding_patch_keeps_fbank_in_float32(tmp_path):
    fbank_dtypes: list[torch.dtype] = []

    class DummyKaldi:
        @staticmethod
        def fbank(wav, num_mel_bins: int, sample_frequency: int, dither: int):
            del num_mel_bins, sample_frequency, dither
            fbank_dtypes.append(wav.dtype)
            return torch.ones((4, 80), dtype=torch.float32, device=wav.device)

    class DummyEmbedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.probe = nn.Parameter(torch.zeros(1, dtype=torch.float16))
            self.last_dtype: torch.dtype | None = None

        def forward3(self, feat: torch.Tensor) -> torch.Tensor:
            self.last_dtype = feat.dtype
            return feat.mean(dim=(1, 2))

    def original_compute_embedding3(self, wav):
        return wav

    original_compute_embedding3.__globals__["Kaldi"] = DummyKaldi

    class DummySV:
        def __init__(self) -> None:
            self.embedding_model = DummyEmbedding()
            self.is_half = True

        compute_embedding3 = original_compute_embedding3

    class DummyPipeline:
        def __init__(self) -> None:
            self.sv_model = DummySV()

    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = DummyPipeline()

    runtime._install_sv_half_safe_patch(pipeline)
    wav = torch.randn(2, 160, dtype=torch.float32)
    output = pipeline.sv_model.compute_embedding3(wav)

    assert output.shape == (2,)
    assert fbank_dtypes == [torch.float32, torch.float32]
    assert pipeline.sv_model.embedding_model.last_dtype == torch.float16


def test_build_tts_inputs_defaults_to_upstream_cut1(tmp_path):
    wav_path = tmp_path / "ref.wav"
    sf.write(wav_path, np.zeros(1600, dtype=np.float32), 16000)

    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))

    inputs = runtime.build_tts_inputs(
        {
            "text": "今天风很轻，路边的树叶在阳光里慢慢摇晃。",
            "text_lang": "zh",
            "ref_audio_path": str(wav_path),
            "prompt_text": "朝阳下的朝圣者重申着重获新生的愿望。",
            "prompt_lang": "zh",
        }
    )

    assert inputs["text_split_method"] == "cut1"


def test_bind_pipeline_components_exposes_runtime_owned_refs(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    bert_worker = object()
    ref_worker = object()
    g2pw_worker = object()
    text_cpu_worker = object()
    text_cpu_executor = object()
    bert_stage_limiter = object()
    ref_stage_limiter = object()
    pipeline = SimpleNamespace(
        configs=SimpleNamespace(device="cpu", version="v3"),
        precision=torch.float16,
        is_v2pro=True,
        t2s_model=SimpleNamespace(model="t2s-model"),
        vits_model="vits-model",
        bert_tokenizer="bert-tokenizer",
        bert_model="bert-model",
        cnhuhbert_model="cnhubert-model",
        vocoder="vocoder-model",
        sv_model="sv-model",
        sr_model="sr-model",
        sr_model_not_exist=False,
        prepare_bert_batch_worker=bert_worker,
        prepare_ref_semantic_batch_worker=ref_worker,
        prepare_g2pw_batch_worker=g2pw_worker,
        prepare_text_cpu_worker=text_cpu_worker,
        prepare_text_cpu_executor=text_cpu_executor,
        prepare_bert_stage_limiter=bert_stage_limiter,
        prepare_ref_semantic_stage_limiter=ref_stage_limiter,
    )

    runtime._bind_pipeline_components(pipeline)

    assert runtime.get_t2s_model() == "t2s-model"
    assert runtime._get_runtime_vits_model() == "vits-model"
    assert runtime._get_runtime_bert_tokenizer() == "bert-tokenizer"
    assert runtime._get_runtime_bert_model() == "bert-model"
    assert runtime._get_runtime_cnhuhbert_model() == "cnhubert-model"
    assert runtime._get_runtime_vocoder() == "vocoder-model"
    assert runtime._get_runtime_sv_model() == "sv-model"
    assert runtime._get_runtime_sr_model() == ("sr-model", False)
    assert runtime._get_runtime_prepare_bert_batch_worker() is bert_worker
    assert runtime._get_runtime_prepare_ref_semantic_batch_worker() is ref_worker
    assert runtime._get_runtime_prepare_g2pw_batch_worker() is g2pw_worker
    assert runtime._get_runtime_prepare_text_cpu_worker() is text_cpu_worker
    assert runtime._get_runtime_prepare_text_cpu_executor() is text_cpu_executor
    assert runtime._get_runtime_prepare_bert_stage_limiter() is bert_stage_limiter
    assert runtime._get_runtime_prepare_ref_semantic_stage_limiter() is ref_stage_limiter
    assert runtime._get_runtime_precision() == torch.float16
    assert runtime._is_runtime_v2pro() is True


def test_run_vits_non_vocoder_streaming_decode_uses_prefix_chunks(tmp_path, monkeypatch):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    calls: list[dict[str, int | None]] = []
    monkeypatch.setenv("GPTSOVITS_VITS_STREAMING_MIN_CHUNK_TOKENS", "8")
    monkeypatch.setenv("GPTSOVITS_VITS_STREAMING_ALIGNMENT_MODE", "prefix_full_phones")

    def fake_streaming_chunk(
        _vits_model,
        *,
        codes,
        text,
        ge,
        speed,
        result_length,
        overlap_frames,
        padding_length=0,
        noise_scale=0.5,
    ):
        del ge, speed, overlap_frames, padding_length, noise_scale
        prefix_len = int(codes.shape[-1])
        calls.append(
            {
                "prefix_len": prefix_len,
                "text_len": int(text.shape[-1]),
                "result_length": result_length,
            }
        )
        audio_len = 40 if prefix_len >= 16 else 32
        audio = torch.arange(audio_len, dtype=torch.float32).view(1, 1, -1) + prefix_len * 100
        latent = torch.zeros((1, 1, max(prefix_len, 16)), dtype=torch.float32)
        latent_mask = torch.ones_like(latent)
        return audio, latent, latent_mask

    runtime._run_vits_non_vocoder_streaming_chunk = fake_streaming_chunk  # type: ignore[method-assign]
    pipeline = SimpleNamespace(sola_algorithm=lambda fragments, overlap_len: torch.cat(fragments, dim=0))
    vits_model = SimpleNamespace(
        dec=SimpleNamespace(ups=[SimpleNamespace(stride=(2,)), SimpleNamespace(stride=(2,))]),
        semantic_frame_rate="",
    )

    audio = runtime._run_vits_non_vocoder_streaming_decode(
        pipeline,
        vits_model,
        semantic_tokens=torch.arange(18, dtype=torch.long),
        phones=torch.arange(12, dtype=torch.long),
        ge=torch.zeros((1, 1, 1), dtype=torch.float32),
        speed=1.0,
        chunk_tokens=8,
        overlap_tokens=2,
    )

    assert calls == [
        {"prefix_len": 8, "text_len": 12, "result_length": None},
        {"prefix_len": 16, "text_len": 12, "result_length": 12},
        {"prefix_len": 18, "text_len": 12, "result_length": 12},
    ]
    assert audio.ndim == 1
    assert int(audio.numel()) > 0


def test_decode_prepared_request_fragment_uses_chunked_vits_path_for_long_tokens(tmp_path, monkeypatch):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = SimpleNamespace(
        configs=SimpleNamespace(use_vocoder=False),
        sola_algorithm=lambda fragments, overlap_len: torch.cat(fragments, dim=0),
    )
    prepared = GPTSoVITSDecodePreparedRequest(
        request_id="req",
        semantic_tokens=torch.arange(16, dtype=torch.long),
        phones=torch.arange(20, dtype=torch.long),
        prompt_phones=torch.arange(4, dtype=torch.long),
        prompt_semantic=torch.arange(4, dtype=torch.long),
        refer_audio_spec=torch.zeros((1, 100, 32), dtype=torch.float32),
        refer_audio_16k=torch.zeros((1, 16000), dtype=torch.float32),
        raw_audio=torch.zeros((1, 16000), dtype=torch.float32),
        raw_sr=16000,
        speed_factor=1.0,
        fragment_interval=0.3,
        sample_steps=32,
        super_sampling=False,
    )
    refer_spec = GPTSoVITSReferSpec(
        spec_audio=torch.zeros((1, 100, 32), dtype=torch.float32),
        audio_16k=torch.zeros((1, 16000), dtype=torch.float32),
    )
    vits_model = SimpleNamespace(dec=SimpleNamespace(ups=[]), semantic_frame_rate="")
    chunked_audio = torch.linspace(-0.1, 0.1, steps=32)

    monkeypatch.setenv("GPTSOVITS_VITS_STREAMING_CHUNK_TOKENS", "8")
    runtime._bind_pipeline_components = Mock()  # type: ignore[method-assign]
    runtime._build_refer_spec_from_prepared = Mock(return_value=refer_spec)  # type: ignore[method-assign]
    runtime._get_runtime_configs = Mock(return_value=SimpleNamespace(device="cpu", sampling_rate=32000))  # type: ignore[method-assign]
    runtime._get_runtime_vits_model = Mock(return_value=vits_model)  # type: ignore[method-assign]
    runtime._get_runtime_precision = Mock(return_value=torch.float32)  # type: ignore[method-assign]
    runtime._is_runtime_v2pro = Mock(return_value=False)  # type: ignore[method-assign]
    runtime._compute_vits_reference_ge = Mock(return_value=torch.zeros((1, 1, 1), dtype=torch.float32))  # type: ignore[method-assign]
    runtime._run_vits_non_vocoder_decode = Mock(side_effect=AssertionError("full decode should not run"))  # type: ignore[method-assign]
    runtime._run_vits_non_vocoder_streaming_decode = Mock(return_value=chunked_audio)  # type: ignore[method-assign]

    audio_fragment, output_sr = runtime._decode_prepared_request_fragment(pipeline, prepared)

    assert output_sr == 32000
    assert torch.equal(audio_fragment, chunked_audio)
    runtime._run_vits_non_vocoder_streaming_decode.assert_called_once()


def test_install_runtime_prepare_components_rebinds_pipeline_ownership(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = SimpleNamespace(
        _prewarm_g2pw_runtime=Mock(),
        _prewarm_prepare_ref_runtime=Mock(),
    )
    runtime._build_runtime_prepare_bert_batch_worker = Mock(return_value="bert-worker")  # type: ignore[method-assign]
    runtime._build_runtime_prepare_ref_semantic_batch_worker = Mock(return_value="ref-worker")  # type: ignore[method-assign]
    runtime._build_runtime_text_preprocessor = Mock(return_value="text-preprocessor")  # type: ignore[method-assign]
    runtime._build_runtime_prepare_g2pw_batch_worker = Mock(return_value="g2pw-worker")  # type: ignore[method-assign]
    runtime._build_runtime_prepare_text_cpu_worker = Mock(return_value="text-cpu-worker")  # type: ignore[method-assign]

    runtime._install_runtime_prepare_components(pipeline)

    runtime._build_runtime_prepare_bert_batch_worker.assert_called_once_with(pipeline)
    runtime._build_runtime_prepare_ref_semantic_batch_worker.assert_called_once_with(pipeline)
    runtime._build_runtime_text_preprocessor.assert_called_once_with(pipeline, "bert-worker")
    runtime._build_runtime_prepare_g2pw_batch_worker.assert_called_once_with(pipeline)
    runtime._build_runtime_prepare_text_cpu_worker.assert_called_once_with(pipeline)
    assert pipeline.prepare_bert_batch_worker == "bert-worker"
    assert pipeline.prepare_ref_semantic_batch_worker == "ref-worker"
    assert pipeline.text_preprocessor == "text-preprocessor"
    assert pipeline.prepare_g2pw_batch_worker == "g2pw-worker"
    assert pipeline.prepare_text_cpu_worker == "text-cpu-worker"
    assert callable(pipeline._vllm_runtime_prepare_state_provider)
    assert callable(pipeline._vllm_runtime_prepare_coordinator_factory)
    assert callable(pipeline._vllm_runtime_refresh_prepare_components)
    assert callable(pipeline.refresh_runtime_components)
    assert callable(pipeline.snapshot_prepare_runtime_components)
    assert pipeline.snapshot_prepare_runtime_components() == pipeline._vllm_runtime_prepare_state_provider()
    assert pipeline._vllm_runtime_prepare_generation == 1
    assert pipeline._vllm_runtime_owned_prepare_components is True
    pipeline._prewarm_g2pw_runtime.assert_called_once_with()
    pipeline._prewarm_prepare_ref_runtime.assert_called_once_with()


def test_ensure_native_runtime_deps_skips_global_preload_by_default(tmp_path, monkeypatch):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    preload = Mock()
    monkeypatch.setattr("vllm_omni.model_executor.models.gpt_sovits.runtime.ctypes.CDLL", preload)
    monkeypatch.delenv("GPTSOVITS_PRELOAD_NATIVE_RUNTIME_DEPS", raising=False)

    runtime._ensure_native_runtime_deps()

    preload.assert_not_called()
    assert runtime._native_runtime_ready is True


def test_temporary_env_override_restores_previous_value(tmp_path, monkeypatch):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    monkeypatch.setenv("GPTSOVITS_RUNTIME_SKIP_PREPARE_COMPONENTS", "old")

    with runtime._temporary_env_override("GPTSOVITS_RUNTIME_SKIP_PREPARE_COMPONENTS", "1"):
        assert os.environ["GPTSOVITS_RUNTIME_SKIP_PREPARE_COMPONENTS"] == "1"

    assert os.environ["GPTSOVITS_RUNTIME_SKIP_PREPARE_COMPONENTS"] == "old"


def test_ensure_pipeline_skips_legacy_tts_refresh_during_runtime_owned_bootstrap(tmp_path, monkeypatch):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    refresh_calls: list[str] = []

    gpt_pkg = ModuleType("GPT_SoVITS")
    infer_pkg = ModuleType("GPT_SoVITS.TTS_infer_pack")
    tts_module = ModuleType("GPT_SoVITS.TTS_infer_pack.TTS")

    class DummyConfig:
        def __init__(self, config_path):
            self.config_path = config_path

    class DummyTTS:
        def __init__(self, configs):
            self.received_configs = configs
            self.configs = SimpleNamespace(device="cpu", version="v2", is_half=False)
            self.precision = torch.float16
            self.is_v2pro = True
            self.t2s_model = SimpleNamespace(model=None)
            self.vits_model = None
            self.bert_tokenizer = None
            self.bert_model = None
            self.cnhuhbert_model = None
            self.vocoder = None
            self.sv_model = None
            self.sr_model = None
            self.sr_model_not_exist = False
            self.prepare_text_cpu_workers = 1
            self.prepare_bert_stage_limiter = None
            self.prepare_ref_semantic_stage_limiter = None
            self.refresh_runtime_components()

        def refresh_runtime_components(self):
            refresh_calls.append("refresh")

    tts_module.TTS = DummyTTS
    tts_module.TTS_Config = DummyConfig
    monkeypatch.setitem(sys.modules, "GPT_SoVITS", gpt_pkg)
    monkeypatch.setitem(sys.modules, "GPT_SoVITS.TTS_infer_pack", infer_pkg)
    monkeypatch.setitem(sys.modules, "GPT_SoVITS.TTS_infer_pack.TTS", tts_module)
    monkeypatch.setattr(runtime, "_ensure_import_path", lambda: None)
    monkeypatch.setattr(runtime, "_ensure_native_runtime_deps", lambda: None)
    install_ref_audio_loader = Mock()
    install_sv_half_safe_patch = Mock()
    install_runtime_prepare_components = Mock()
    bind_pipeline_components = Mock()
    monkeypatch.setattr(runtime, "_install_ref_audio_loader_fallback", install_ref_audio_loader)
    monkeypatch.setattr(runtime, "_install_sv_half_safe_patch", install_sv_half_safe_patch)
    monkeypatch.setattr(runtime, "_install_runtime_prepare_components", install_runtime_prepare_components)
    monkeypatch.setattr(runtime, "_bind_pipeline_components", bind_pipeline_components)

    pipeline = runtime._ensure_pipeline()

    assert isinstance(pipeline, DummyTTS)
    assert refresh_calls == []
    assert getattr(pipeline, "runtime_prepare_components_deferred", False) is True
    install_ref_audio_loader.assert_called_once_with(pipeline)
    install_sv_half_safe_patch.assert_called_once_with(pipeline)
    install_runtime_prepare_components.assert_called_once_with(pipeline)
    bind_pipeline_components.assert_called_once_with(pipeline)


def test_refresh_runtime_prepare_components_reinstalls_runtime_owned_components_and_resets_coordinator(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = SimpleNamespace()
    runtime._prepare_coordinator = object()
    refresh_kv = Mock()
    install_prepare = Mock()
    bind_components = Mock()
    runtime._refresh_runtime_t2s_kv_cache_pool_state = refresh_kv  # type: ignore[method-assign]
    runtime._install_runtime_prepare_components = install_prepare  # type: ignore[method-assign]
    runtime._bind_pipeline_components = bind_components  # type: ignore[method-assign]

    runtime._refresh_runtime_prepare_components(pipeline)

    refresh_kv.assert_called_once_with(pipeline)
    install_prepare.assert_called_once_with(pipeline)
    bind_components.assert_called_once_with(pipeline)
    assert runtime._prepare_coordinator is None


def test_build_prepare_coordinator_prefers_runtime_native_factory(tmp_path):
    _ensure_vendored_gpt_sovits_import_path()
    from vllm_omni.model_executor.models.gpt_sovits.runtime_lib.GPT_SoVITS.TTS_infer_pack.prepare_coordinator import (
        RuntimePrepareCoordinatorAdapter,
        build_prepare_coordinator,
    )

    native = SimpleNamespace(
        inflight=1,
        peak_inflight=2,
        inflight_gate=SimpleNamespace(max_inflight=3),
        text_feature_executor=None,
        g2pw_executor=None,
        ref_audio_executor=None,
        ref_audio_asset_cache={},
        ref_audio_asset_inflight={},
    )
    runtime_owner = SimpleNamespace(
        _ensure_prepare_coordinator=Mock(return_value=native),
    )
    tts = SimpleNamespace(
        _vllm_runtime_owner=runtime_owner,
        _vllm_runtime_prepare_coordinator_factory=Mock(return_value=object()),
        snapshot_prepare_runtime_components=Mock(return_value={"g2pw": {"worker_count": 2}}),
    )

    coordinator = build_prepare_coordinator(tts)

    assert isinstance(coordinator, RuntimePrepareCoordinatorAdapter)
    assert coordinator.runtime_owner is runtime_owner
    assert coordinator.snapshot()["max_inflight"] == 3
    runtime_owner._ensure_prepare_coordinator.assert_called_once_with()
    tts._vllm_runtime_prepare_coordinator_factory.assert_not_called()


def test_build_prepare_coordinator_falls_back_to_legacy_prepare_coordinator(tmp_path):
    _ensure_vendored_gpt_sovits_import_path()
    from vllm_omni.model_executor.models.gpt_sovits.runtime_lib.GPT_SoVITS.TTS_infer_pack.prepare_coordinator import (
        PrepareCoordinator,
        build_prepare_coordinator,
    )

    tts = SimpleNamespace(
        prepare_bert_batch_worker=None,
        prepare_text_cpu_workers=1,
    )

    coordinator = build_prepare_coordinator(tts)

    try:
        assert isinstance(coordinator, PrepareCoordinator)
    finally:
        if coordinator.text_feature_executor is not None:
            coordinator.text_feature_executor.shutdown(wait=True, cancel_futures=True)
        if coordinator.g2pw_executor is not None:
            coordinator.g2pw_executor.shutdown(wait=True, cancel_futures=True)
        if coordinator.ref_audio_executor is not None:
            coordinator.ref_audio_executor.shutdown(wait=True, cancel_futures=True)


def test_install_runtime_prepare_components_rebinds_pipeline_refresh_method_to_runtime_callback(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    refresh_prepare = Mock()
    runtime._refresh_runtime_prepare_components = refresh_prepare  # type: ignore[method-assign]
    pipeline = SimpleNamespace(
        _prewarm_g2pw_runtime=Mock(),
        _prewarm_prepare_ref_runtime=Mock(),
    )
    runtime._build_runtime_prepare_bert_batch_worker = Mock(return_value=None)  # type: ignore[method-assign]
    runtime._build_runtime_prepare_ref_semantic_batch_worker = Mock(return_value=None)  # type: ignore[method-assign]
    runtime._build_runtime_text_preprocessor = Mock(return_value=None)  # type: ignore[method-assign]
    runtime._build_runtime_prepare_g2pw_batch_worker = Mock(return_value=None)  # type: ignore[method-assign]
    runtime._build_runtime_prepare_text_cpu_worker = Mock(return_value=None)  # type: ignore[method-assign]

    runtime._install_runtime_prepare_components(pipeline)
    pipeline.refresh_runtime_components()

    refresh_prepare.assert_called_once_with(pipeline)


def test_scheduler_prepare_coordinator_prefers_runtime_native_factory_and_rebuilds_on_generation_change(tmp_path):
    _ensure_vendored_gpt_sovits_import_path()
    from GPT_SoVITS.TTS_infer_pack.prepare_coordinator import RuntimePrepareCoordinatorAdapter
    from vllm_omni.model_executor.models.gpt_sovits.runtime_lib.GPT_SoVITS.TTS_infer_pack.t2s_scheduler import (
        _get_scheduler_prepare_coordinator,
    )

    runtime_owner = SimpleNamespace(_ensure_prepare_coordinator=Mock())
    tts = SimpleNamespace(
        _vllm_runtime_owner=runtime_owner,
        _vllm_runtime_prepare_generation=1,
        _vllm_runtime_prepare_coordinator_factory=Mock(return_value=object()),
    )

    coordinator_v1 = _get_scheduler_prepare_coordinator(tts)
    assert isinstance(coordinator_v1, RuntimePrepareCoordinatorAdapter)
    assert _get_scheduler_prepare_coordinator(tts) is coordinator_v1
    tts._vllm_runtime_prepare_generation = 2
    coordinator_v2 = _get_scheduler_prepare_coordinator(tts)
    assert isinstance(coordinator_v2, RuntimePrepareCoordinatorAdapter)
    assert coordinator_v2 is not coordinator_v1
    runtime_owner._ensure_prepare_coordinator.assert_not_called()
    tts._vllm_runtime_prepare_coordinator_factory.assert_not_called()


def test_scheduler_prepare_request_state_uses_coordinator_automatically_for_runtime_owned_tts(tmp_path, monkeypatch):
    _ensure_vendored_gpt_sovits_import_path()
    from vllm_omni.model_executor.models.gpt_sovits.runtime_lib.GPT_SoVITS.TTS_infer_pack import (
        t2s_scheduler as scheduler,
    )

    expected_state = object()
    coordinator = SimpleNamespace(
        prepare_state_profiled_async=Mock(return_value="coro"),
    )
    monkeypatch.setattr(scheduler, "_prepare_request_state_legacy", Mock(side_effect=AssertionError("legacy path")))
    monkeypatch.setattr(scheduler, "_get_scheduler_prepare_coordinator", Mock(return_value=coordinator))
    monkeypatch.setattr(scheduler, "_run_coro_sync", Mock(return_value=(expected_state, 0.0, 0.0)))
    monkeypatch.delenv("GPTSOVITS_PREPARE_SCHEDULER_USE_COORDINATOR", raising=False)
    tts = SimpleNamespace(
        _vllm_runtime_prepare_coordinator_factory=Mock(return_value=object()),
    )
    spec = SimpleNamespace()

    state = scheduler.prepare_request_state(tts, spec)

    assert state is expected_state
    scheduler._get_scheduler_prepare_coordinator.assert_called_once_with(tts)
    coordinator.prepare_state_profiled_async.assert_called_once()


def test_runtime_prepare_coordinator_adapter_prepare_state_profiled_async_uses_runtime_native_path(tmp_path):
    _ensure_vendored_gpt_sovits_import_path()
    from GPT_SoVITS.TTS_infer_pack.prepare_coordinator import RuntimePrepareCoordinatorAdapter

    @contextmanager
    def _noop_project_root_cwd():
        yield

    native_coordinator = SimpleNamespace(enable_g2pw_audio_batch_merge=False)
    runtime_owner = SimpleNamespace(
        _ensure_prepare_coordinator=Mock(return_value=native_coordinator),
        _project_root_cwd=_noop_project_root_cwd,
        _prepare_cpu_stage_async=AsyncMock(),
        _prepare_gpu_audio_phase_async=AsyncMock(),
        _prepare_gpu_text_phase_async=AsyncMock(),
        _build_request_state_from_prepare_phases=Mock(return_value=SimpleNamespace(state="ok")),
        _release_prepare_split_stage_slot=Mock(),
    )
    tts = SimpleNamespace(
        _vllm_runtime_owner=runtime_owner,
        _vllm_runtime_prepare_coordinator_factory=Mock(return_value=object()),
    )
    coordinator = RuntimePrepareCoordinatorAdapter(tts, runtime_owner)
    spec = SimpleNamespace(request_id="req-1", prompt_text="prompt", prompt_lang="zh", text="text", text_lang="zh")
    prepared_cpu = GPTSoVITSNativePreparedCpuStage(
        spec=spec,
        prepare_submit_at=1.0,
        prepare_start=2.0,
        prompt_text="prompt",
        text="text",
        prepare_admission_wait_ms=0.0,
        current_inflight=1,
        peak_inflight=1,
        prompt_cpu_profiled=SimpleNamespace(),
        target_cpu_profiled=SimpleNamespace(),
    )
    audio_phase = GPTSoVITSPrepareAudioPhaseData(
        prompt_g2pw_profiled="prompt-g2pw",
        target_g2pw_profiled="target-g2pw",
        ref_audio_profiled=SimpleNamespace(result=SimpleNamespace(raw_audio="raw", raw_sr=16000)),
        g2pw_pair_ms=1.0,
        phase_wall_ms=2.0,
    )
    text_phase = GPTSoVITSPrepareTextPhaseData(
        prompt_feature_profiled="prompt-feature",
        target_feature_profiled="target-feature",
        phase_wall_ms=3.0,
    )
    runtime_owner._prepare_cpu_stage_async.return_value = prepared_cpu
    runtime_owner._prepare_gpu_audio_phase_async.return_value = audio_phase
    runtime_owner._prepare_gpu_text_phase_async.return_value = text_phase

    state, prepare_start, prepare_finished_at = asyncio.run(
        coordinator.prepare_state_profiled_async(spec, prepare_submit_at=1.0)
    )

    runtime_owner._prepare_cpu_stage_async.assert_awaited_once_with(
        native_coordinator,
        spec,
        prepare_submit_at=1.0,
    )
    runtime_owner._prepare_gpu_audio_phase_async.assert_awaited_once_with(native_coordinator, prepared_cpu)
    runtime_owner._prepare_gpu_text_phase_async.assert_awaited_once()
    runtime_owner._build_request_state_from_prepare_phases.assert_called_once_with(
        prepared_cpu,
        audio_phase,
        text_phase,
        ref_spec_result=None,
        extra_profile={
            "engine_prepare_audio_phase_mode": 0.0,
            "engine_prepare_audio_phase_wall_ms": 2.0,
            "engine_prepare_audio_phase_batch_size": 1.0,
            "engine_prepare_text_phase_wall_ms": 3.0,
            "engine_prepare_text_phase_batch_size": 1.0,
        },
    )
    runtime_owner._release_prepare_split_stage_slot.assert_called_once_with(native_coordinator)
    assert state.state == "ok"
    assert prepare_start == 2.0
    assert prepare_finished_at >= prepare_start


def test_runtime_prepare_coordinator_adapter_prepare_direct_shared_segments_uses_runtime_native_path(tmp_path):
    _ensure_vendored_gpt_sovits_import_path()
    from GPT_SoVITS.TTS_infer_pack.prepare_coordinator import RuntimePrepareCoordinatorAdapter

    native_coordinator = SimpleNamespace()
    runtime_owner = SimpleNamespace(
        _ensure_prepare_coordinator=Mock(return_value=native_coordinator),
        _prepare_direct_shared_segments_async=AsyncMock(return_value=["shared-ok"]),
    )
    coordinator = RuntimePrepareCoordinatorAdapter(
        SimpleNamespace(
            _vllm_runtime_owner=runtime_owner,
            _vllm_runtime_prepare_coordinator_factory=Mock(return_value=object()),
        ),
        runtime_owner,
    )
    specs = [SimpleNamespace(request_id="req-1"), SimpleNamespace(request_id="req-2")]

    results = asyncio.run(coordinator.prepare_direct_shared_segments_profiled_async(specs))

    assert results == ["shared-ok"]
    runtime_owner._prepare_direct_shared_segments_async.assert_awaited_once_with(native_coordinator, specs)


def test_ensure_native_runtime_deps_supports_opt_in_preload(tmp_path, monkeypatch):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    preload = Mock()
    monkeypatch.setattr("vllm_omni.model_executor.models.gpt_sovits.runtime.ctypes.CDLL", preload)
    monkeypatch.setattr("vllm_omni.model_executor.models.gpt_sovits.runtime.os.path.isdir", lambda _path: True)
    monkeypatch.setattr("vllm_omni.model_executor.models.gpt_sovits.runtime.os.path.exists", lambda _path: True)
    monkeypatch.setenv("GPTSOVITS_PRELOAD_NATIVE_RUNTIME_DEPS", "1")

    runtime._ensure_native_runtime_deps()

    assert runtime._native_runtime_ready is True
    assert preload.call_count >= 2


def test_get_ar_session_semantic_tokens_drops_prefix_and_trailing_eos(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime.get_semantic_eos_id = lambda: 1024  # type: ignore[method-assign]
    session = GPTSoVITSARSession(
        request_id="req",
        active_batch=SimpleNamespace(
            prefix_lens=torch.tensor([2], dtype=torch.long),
            y_sequences=[torch.tensor([7, 8, 101, 102, 1024], dtype=torch.long)],
        ),
        transport_info={},
        current_logits=torch.zeros((1, 1025), dtype=torch.float32),
    )

    tokens = runtime.get_ar_session_semantic_tokens(session)

    assert tokens.tolist() == [101, 102]


def test_get_ar_session_logits_suppresses_eos_before_min_step(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime.get_semantic_eos_id = lambda: 4  # type: ignore[method-assign]
    session = GPTSoVITSARSession(
        request_id="req",
        active_batch=SimpleNamespace(step_indices=torch.tensor([5], dtype=torch.long)),
        transport_info={},
        current_logits=torch.zeros((1, 6), dtype=torch.float32),
    )

    logits = runtime.get_ar_session_logits(session, suppress_eos_until_step=11)

    assert torch.isneginf(logits[0, 4])


def test_uniform_sampling_group_key_requires_same_step_and_sampling_params(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    active_batch = SimpleNamespace(
        states=[
            SimpleNamespace(top_k=5, top_p=0.8, temperature=1.0, repetition_penalty=1.1),
            SimpleNamespace(top_k=5, top_p=0.8, temperature=1.0, repetition_penalty=1.1),
        ],
        step_indices=torch.tensor([3, 3], dtype=torch.long),
    )

    uniform_key = runtime._uniform_sampling_group_key(active_batch)
    non_uniform_key = runtime._uniform_sampling_group_key(
        SimpleNamespace(
            states=active_batch.states,
            step_indices=torch.tensor([3, 4], dtype=torch.long),
        )
    )

    assert uniform_key == (5, 0.8, 1.0, 1.1, True)
    assert non_uniform_key is None


def test_batched_sample_by_group_uses_runtime_native_sampling_helpers(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))

    def fake_logits_to_probs(**kwargs):
        logits = kwargs["logits"]
        return torch.softmax(logits.float(), dim=-1)

    def fake_multinomial_sample_one_no_sync(probs):
        return torch.argmax(probs, dim=-1, keepdim=True)

    runtime._get_sampling_ops = Mock(return_value=(fake_logits_to_probs, fake_multinomial_sample_one_no_sync))  # type: ignore[method-assign]

    sampled, argmax_tokens = runtime._batched_sample_by_group(
        logits=torch.tensor(
            [
                [0.1, 0.2, 0.9, 0.0],
                [0.3, 0.8, 0.1, 0.0],
                [0.7, 0.2, 0.1, 0.0],
            ],
            dtype=torch.float32,
        ),
        histories=[
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([2, 3, 4], dtype=torch.long),
        ],
        sampling_keys=[
            (5, 0.8, 1.0, 1.1, True),
            (5, 0.8, 1.0, 1.1, True),
            (2, 0.9, 0.7, 1.0, False),
        ],
    )

    assert [item.shape for item in sampled] == [(1, 1), (1, 1), (1, 1)]
    assert [int(item.item()) for item in sampled] == [2, 1, 0]
    assert argmax_tokens == [2, 1, 0]


def test_batched_sample_uniform_records_fine_grained_stats(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))

    def fake_logits_to_probs(**kwargs):
        logits = kwargs["logits"]
        return torch.softmax(logits.float(), dim=-1)

    def fake_multinomial_sample_one_no_sync(probs):
        return torch.argmax(probs, dim=-1, keepdim=True)

    runtime._get_sampling_ops = Mock(return_value=(fake_logits_to_probs, fake_multinomial_sample_one_no_sync))  # type: ignore[method-assign]

    stats: dict[str, float] = {}
    sampled, argmax_tokens = runtime._batched_sample_uniform(
        logits=torch.tensor([[0.1, 0.2, 0.9, 0.0]], dtype=torch.float32),
        histories=[torch.tensor([1, 2], dtype=torch.long)],
        sampling_key=(5, 0.8, 1.0, 1.1, True),
        stats=stats,
    )

    assert sampled.shape == (1, 1)
    assert argmax_tokens.shape == (1,)
    for key in (
        "sampling_history_stack_pad_ms",
        "sampling_logits_to_probs_ms",
        "sampling_multinomial_ms",
        "sampling_argmax_ms",
    ):
        assert key in stats
        assert stats[key] >= 0.0


def test_sample_active_batch_requests_uses_runtime_native_uniform_fast_path(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._batched_sample_uniform = Mock(  # type: ignore[method-assign]
        return_value=(
            torch.tensor([[7], [8]], dtype=torch.long),
            torch.tensor([7, 8], dtype=torch.long),
        )
    )

    active_batch = SimpleNamespace(
        states=[
            SimpleNamespace(request_id="a", top_k=5, top_p=0.8, temperature=1.0, repetition_penalty=1.0, early_stop_num=-1),
            SimpleNamespace(request_id="b", top_k=5, top_p=0.8, temperature=1.0, repetition_penalty=1.0, early_stop_num=-1),
        ],
        step_indices=torch.tensor([3, 3], dtype=torch.long),
        y_sequences=[
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([3, 4], dtype=torch.long),
        ],
        prefix_lens=torch.tensor([1, 1], dtype=torch.long),
    )

    finished, keep_indices, updated_sequences = runtime._sample_active_batch_requests(
        SimpleNamespace(EOS=1024),
        active_batch,
        torch.zeros((2, 1025), dtype=torch.float32),
        max_steps=16,
    )

    runtime._batched_sample_uniform.assert_called_once()
    assert finished == []
    assert keep_indices == [0, 1]
    assert [seq.tolist() for seq in updated_sequences] == [[1, 2, 7], [3, 4, 8]]


def test_sample_active_batch_requests_records_single_request_sampling_stats(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._sample_single_request = Mock(  # type: ignore[method-assign]
        return_value=(
            torch.tensor([[7]], dtype=torch.long),
            torch.tensor([7], dtype=torch.long),
        )
    )

    active_batch = SimpleNamespace(
        states=[
            SimpleNamespace(request_id="solo", top_k=5, top_p=0.8, temperature=1.0, repetition_penalty=1.0, early_stop_num=-1),
        ],
        step_indices=torch.tensor([3], dtype=torch.long),
        y_sequences=[torch.tensor([1, 2], dtype=torch.long)],
        prefix_lens=torch.tensor([1], dtype=torch.long),
    )
    stats: dict[str, float | int] = {}

    finished, keep_indices, updated_sequences = runtime._sample_active_batch_requests(
        SimpleNamespace(EOS=1024),
        active_batch,
        torch.zeros((1, 1025), dtype=torch.float32),
        max_steps=16,
        stats=stats,
    )

    assert finished == []
    assert keep_indices == [0]
    assert [seq.tolist() for seq in updated_sequences] == [[1, 2, 7]]
    assert int(stats["sampling_single_request_calls"]) == 1
    assert float(stats["sampling_finish_scan_ms"]) >= 0.0
    assert float(stats["sampling_total_ms"]) >= float(stats["sampling_finish_scan_ms"])


def test_sample_single_request_records_sampling_stats(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))

    def fake_logits_to_probs(**kwargs):
        logits = kwargs["logits"]
        return torch.softmax(logits.float(), dim=-1)

    def fake_multinomial_sample_one_no_sync(probs):
        return torch.argmax(probs, dim=-1, keepdim=True)

    runtime._get_sampling_ops = Mock(return_value=(fake_logits_to_probs, fake_multinomial_sample_one_no_sync))  # type: ignore[method-assign]

    stats: dict[str, float] = {}
    sampled, argmax_tokens = runtime._sample_single_request(
        logits=torch.tensor([[0.1, 0.2, 0.9, 0.0]], dtype=torch.float32),
        history=torch.tensor([1, 2], dtype=torch.long),
        sampling_key=(5, 0.8, 1.0, 1.1, True),
        stats=stats,
    )

    assert sampled.shape == (1, 1)
    assert argmax_tokens.shape == (1,)
    assert float(stats["sampling_history_stack_pad_ms"]) == 0.0
    for key in (
        "sampling_logits_to_probs_ms",
        "sampling_multinomial_ms",
        "sampling_argmax_ms",
    ):
        assert key in stats
        assert stats[key] >= 0.0


def test_prepare_decode_request_moves_transport_to_pipeline_device(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._pipeline = SimpleNamespace(configs=SimpleNamespace(device="cpu", sampling_rate=32000))

    prepared = runtime.prepare_decode_request(
        torch.tensor([10, 11], dtype=torch.int32),
        GPTSoVITSStageTransport(
            request_id="decode_req",
            semantic_tokens=torch.tensor([], dtype=torch.long),
            semantic_token_segments=(),
            phones=torch.tensor([1, 2], dtype=torch.int32),
            segment_phones=(),
            prompt_phones=torch.tensor([3], dtype=torch.int32),
            prompt_semantic=torch.tensor([4], dtype=torch.int32),
            refer_audio_spec=torch.tensor([0.1], dtype=torch.float16),
            refer_audio_16k=torch.tensor([0.2], dtype=torch.float16),
            raw_audio=torch.tensor([0.3], dtype=torch.float64),
            raw_sr=16000,
            speed_factor=1.25,
            fragment_interval=0.3,
            sample_steps=48,
            super_sampling=True,
        ),
    )

    assert prepared.request_id == "decode_req"
    assert prepared.semantic_tokens.dtype == torch.long
    assert prepared.semantic_tokens.device.type == "cpu"
    assert prepared.phones.dtype == torch.long
    assert prepared.refer_audio_spec.dtype == torch.float32
    assert prepared.raw_audio.dtype == torch.float32
    assert prepared.raw_sr == 16000
    assert prepared.speed_factor == 1.25
    assert prepared.sample_steps == 48
    assert prepared.super_sampling is True


def test_prepare_decode_request_builds_group_for_segmented_transport(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._pipeline = SimpleNamespace(configs=SimpleNamespace(device="cpu", sampling_rate=32000))

    prepared = runtime.prepare_decode_request(
        [
            torch.tensor([10, 11], dtype=torch.long),
            torch.tensor([20], dtype=torch.long),
        ],
        GPTSoVITSStageTransport(
            request_id="decode_req",
            semantic_tokens=torch.tensor([], dtype=torch.long),
            semantic_token_segments=(),
            phones=torch.tensor([], dtype=torch.long),
            segment_phones=(
                torch.tensor([1, 2], dtype=torch.long),
                torch.tensor([3], dtype=torch.long),
            ),
            prompt_phones=torch.tensor([4], dtype=torch.long),
            prompt_semantic=torch.tensor([5], dtype=torch.long),
            refer_audio_spec=torch.tensor([0.1], dtype=torch.float32),
            refer_audio_16k=torch.tensor([0.2], dtype=torch.float32),
            raw_audio=torch.tensor([0.3], dtype=torch.float32),
            raw_sr=16000,
            speed_factor=1.0,
            fragment_interval=0.3,
            sample_steps=32,
            super_sampling=False,
        ),
    )

    assert isinstance(prepared, GPTSoVITSDecodePreparedRequestGroup)
    assert prepared.request_id == "decode_req"
    assert len(prepared.segment_requests) == 2
    assert torch.equal(prepared.segment_requests[0].semantic_tokens, torch.tensor([10, 11], dtype=torch.long))
    assert torch.equal(prepared.segment_requests[1].phones, torch.tensor([3], dtype=torch.long))


def test_decode_semantic_tokens_from_transport_uses_prepare_decode_finalize_pipeline(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    prepared = SimpleNamespace(semantic_tokens=torch.tensor([1], dtype=torch.long))
    decoded = SimpleNamespace(audio_fragment=np.array([0.1], dtype=np.float32))
    expected = SimpleNamespace(sample_rate=24000, audio=np.array([0.2], dtype=np.float32))
    runtime.prepare_decode_request = Mock(return_value=prepared)
    runtime.decode_prepared_request = Mock(return_value=decoded)
    runtime.finalize_decoded_audio = Mock(return_value=expected)

    result = runtime.decode_semantic_tokens_from_transport(torch.tensor([1], dtype=torch.long), {"foo": "bar"})

    runtime.prepare_decode_request.assert_called_once()
    runtime.decode_prepared_request.assert_called_once_with(prepared)
    runtime.finalize_decoded_audio.assert_called_once_with(decoded)
    assert result is expected


def test_finalize_decoded_audio_group_uses_segment_fragments(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._pipeline = SimpleNamespace(configs=SimpleNamespace(device="cpu", sampling_rate=32000))
    runtime._audio_postprocess_native = Mock(return_value=(32000, np.array([1, 2, 3], dtype=np.int16)))  # type: ignore[method-assign]

    result = runtime.finalize_decoded_audio(
        GPTSoVITSDecodedAudioGroup(
            request_id="req",
            segment_items=[
                GPTSoVITSDecodedAudio("req::seg0000", np.array([0.1], dtype=np.float32), 32000, 1.0, 0.3, False),
                GPTSoVITSDecodedAudio("req::seg0001", np.array([0.2], dtype=np.float32), 32000, 1.0, 0.3, False),
            ],
            speed_factor=1.0,
            fragment_interval=0.3,
            super_sampling=False,
        )
    )

    runtime._audio_postprocess_native.assert_called_once()
    assert result.sample_rate == 32000
    assert result.audio.shape == (3,)


def test_decode_prepared_request_group_decodes_segments_in_chunks(tmp_path, monkeypatch):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    monkeypatch.setenv("GPTSOVITS_SEGMENT_DECODE_MAX_BATCH", "2")
    runtime._pipeline = SimpleNamespace(configs=SimpleNamespace(use_vocoder=False, sampling_rate=32000))
    chunk_calls: list[list[str]] = []
    runtime._build_non_vocoder_prompt_context = Mock(return_value={"ge": torch.ones((1, 1, 1)), "device": torch.device("cpu"), "vits_model": object()})  # type: ignore[method-assign]

    def _fake_batched_decode(_pipeline, chunk, prompt_context=None):
        assert prompt_context is not None
        chunk_calls.append([item.request_id for item in chunk])
        return [
            GPTSoVITSDecodedAudio(item.request_id, f"audio-{item.request_id}", 32000, 1.0, 0.3, False)
            for item in chunk
        ]

    runtime._decode_prepared_requests_batched_non_vocoder = Mock(side_effect=_fake_batched_decode)  # type: ignore[method-assign]

    decoded = runtime.decode_prepared_request(
        GPTSoVITSDecodePreparedRequestGroup(
            request_id="req",
            segment_requests=[
                GPTSoVITSDecodePreparedRequest(
                    request_id=f"req::seg{index:04d}",
                    semantic_tokens=torch.tensor([index + 1], dtype=torch.long),
                    phones=torch.tensor([1], dtype=torch.long),
                    prompt_phones=torch.tensor([2], dtype=torch.long),
                    prompt_semantic=torch.tensor([3], dtype=torch.long),
                    refer_audio_spec=torch.tensor([0.1], dtype=torch.float32),
                    refer_audio_16k=torch.tensor([0.2], dtype=torch.float32),
                    raw_audio=torch.tensor([0.3], dtype=torch.float32),
                    raw_sr=16000,
                    speed_factor=1.0,
                    fragment_interval=0.3,
                    sample_steps=32,
                    super_sampling=False,
                )
                for index in range(5)
            ],
            speed_factor=1.0,
            fragment_interval=0.3,
            sample_steps=32,
            super_sampling=False,
        )
    )

    assert isinstance(decoded, GPTSoVITSDecodedAudioGroup)
    assert chunk_calls == [
        ["req::seg0000", "req::seg0001"],
        ["req::seg0002", "req::seg0003"],
        ["req::seg0004"],
    ]
    runtime._build_non_vocoder_prompt_context.assert_called_once()
    assert [item.request_id for item in decoded.segment_items] == [
        "req::seg0000",
        "req::seg0001",
        "req::seg0002",
        "req::seg0003",
        "req::seg0004",
    ]


def test_decode_prepared_request_group_uses_length_aware_batches_and_restores_original_order(tmp_path, monkeypatch):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    monkeypatch.setenv("GPTSOVITS_SEGMENT_DECODE_MAX_BATCH", "8")
    monkeypatch.setenv("GPTSOVITS_SEGMENT_DECODE_MAX_SEMANTIC_TOKENS", "6")
    monkeypatch.setenv("GPTSOVITS_SEGMENT_DECODE_MAX_PHONE_TOKENS", "0")
    runtime._pipeline = SimpleNamespace(configs=SimpleNamespace(use_vocoder=False, sampling_rate=32000))
    chunk_calls: list[list[str]] = []
    runtime._build_non_vocoder_prompt_context = Mock(return_value={"ge": torch.ones((1, 1, 1)), "device": torch.device("cpu"), "vits_model": object()})  # type: ignore[method-assign]

    def _fake_batched_decode(_pipeline, chunk, prompt_context=None):
        assert prompt_context is not None
        chunk_calls.append([item.request_id for item in chunk])
        return [
            GPTSoVITSDecodedAudio(item.request_id, f"audio-{item.request_id}", 32000, 1.0, 0.3, False)
            for item in chunk
        ]

    runtime._decode_prepared_requests_batched_non_vocoder = Mock(side_effect=_fake_batched_decode)  # type: ignore[method-assign]

    decoded = runtime.decode_prepared_request(
        GPTSoVITSDecodePreparedRequestGroup(
            request_id="req",
            segment_requests=[
                GPTSoVITSDecodePreparedRequest(
                    request_id="req::seg0000",
                    semantic_tokens=torch.tensor([1, 2], dtype=torch.long),
                    phones=torch.tensor([1], dtype=torch.long),
                    prompt_phones=torch.tensor([2], dtype=torch.long),
                    prompt_semantic=torch.tensor([3], dtype=torch.long),
                    refer_audio_spec=torch.tensor([0.1], dtype=torch.float32),
                    refer_audio_16k=torch.tensor([0.2], dtype=torch.float32),
                    raw_audio=torch.tensor([0.3], dtype=torch.float32),
                    raw_sr=16000,
                    speed_factor=1.0,
                    fragment_interval=0.3,
                    sample_steps=32,
                    super_sampling=False,
                ),
                GPTSoVITSDecodePreparedRequest(
                    request_id="req::seg0001",
                    semantic_tokens=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                    phones=torch.tensor([1], dtype=torch.long),
                    prompt_phones=torch.tensor([2], dtype=torch.long),
                    prompt_semantic=torch.tensor([3], dtype=torch.long),
                    refer_audio_spec=torch.tensor([0.1], dtype=torch.float32),
                    refer_audio_16k=torch.tensor([0.2], dtype=torch.float32),
                    raw_audio=torch.tensor([0.3], dtype=torch.float32),
                    raw_sr=16000,
                    speed_factor=1.0,
                    fragment_interval=0.3,
                    sample_steps=32,
                    super_sampling=False,
                ),
                GPTSoVITSDecodePreparedRequest(
                    request_id="req::seg0002",
                    semantic_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
                    phones=torch.tensor([1], dtype=torch.long),
                    prompt_phones=torch.tensor([2], dtype=torch.long),
                    prompt_semantic=torch.tensor([3], dtype=torch.long),
                    refer_audio_spec=torch.tensor([0.1], dtype=torch.float32),
                    refer_audio_16k=torch.tensor([0.2], dtype=torch.float32),
                    raw_audio=torch.tensor([0.3], dtype=torch.float32),
                    raw_sr=16000,
                    speed_factor=1.0,
                    fragment_interval=0.3,
                    sample_steps=32,
                    super_sampling=False,
                ),
                GPTSoVITSDecodePreparedRequest(
                    request_id="req::seg0003",
                    semantic_tokens=torch.tensor([1, 2, 3, 4], dtype=torch.long),
                    phones=torch.tensor([1], dtype=torch.long),
                    prompt_phones=torch.tensor([2], dtype=torch.long),
                    prompt_semantic=torch.tensor([3], dtype=torch.long),
                    refer_audio_spec=torch.tensor([0.1], dtype=torch.float32),
                    refer_audio_16k=torch.tensor([0.2], dtype=torch.float32),
                    raw_audio=torch.tensor([0.3], dtype=torch.float32),
                    raw_sr=16000,
                    speed_factor=1.0,
                    fragment_interval=0.3,
                    sample_steps=32,
                    super_sampling=False,
                ),
            ],
            speed_factor=1.0,
            fragment_interval=0.3,
            sample_steps=32,
            super_sampling=False,
        )
    )

    assert isinstance(decoded, GPTSoVITSDecodedAudioGroup)
    assert chunk_calls == [
        ["req::seg0001"],
        ["req::seg0003"],
        ["req::seg0002", "req::seg0000"],
    ]
    assert [item.request_id for item in decoded.segment_items] == [
        "req::seg0000",
        "req::seg0001",
        "req::seg0002",
        "req::seg0003",
    ]


def test_synthesize_routes_through_runtime_native_prepare_generate_decode_pipeline(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    spec = SimpleNamespace(request_id="req-native")
    prepared = SimpleNamespace(request_id="req-native", transport_info={"transport": "ok"})
    semantic_tokens = torch.tensor([1, 2, 3], dtype=torch.long)
    expected = GPTSoVITSResult(sample_rate=32000, audio=np.array([0.1, -0.1], dtype=np.float32))
    runtime.build_request_spec = Mock(return_value=spec)  # type: ignore[method-assign]
    runtime.prepare_request_spec = Mock(return_value=prepared)  # type: ignore[method-assign]
    runtime.generate_semantic_tokens = Mock(return_value={"req-native": semantic_tokens})  # type: ignore[method-assign]
    runtime.decode_semantic_tokens_from_transport = Mock(return_value=expected)  # type: ignore[method-assign]

    result = runtime.synthesize({"text": "hello"})

    runtime.build_request_spec.assert_called_once_with({"text": "hello"})
    runtime.prepare_request_spec.assert_called_once_with(spec)
    runtime.generate_semantic_tokens.assert_called_once_with([prepared])
    runtime.decode_semantic_tokens_from_transport.assert_called_once_with(semantic_tokens, {"transport": "ok"})
    assert result is expected


def test_extract_ref_spec_from_raw_uses_runtime_native_components(tmp_path):
    runtime = GPTSoVITSRuntime(
        project_root="/root/vllm-omni/vllm_omni/model_executor/models/gpt_sovits/runtime_lib",
        config_path=str(tmp_path / "dummy.yaml"),
    )
    pipeline = SimpleNamespace(
        configs=SimpleNamespace(
            device="cpu",
            sampling_rate=32000,
            filter_length=1280,
            hop_length=320,
            win_length=1280,
            is_half=False,
        ),
        is_v2pro=True,
        _extract_ref_spec_profile_from_raw=Mock(side_effect=AssertionError("should not call pipeline ref_spec helper")),
    )
    runtime._ensure_pipeline = Mock(return_value=pipeline)  # type: ignore[method-assign]
    runtime._resample_audio = Mock(return_value=torch.ones((1, 8), dtype=torch.float32))  # type: ignore[method-assign]

    refer_spec, profile = runtime._extract_ref_spec_from_raw(torch.ones((1, 1600), dtype=torch.float32), 32000)

    pipeline._extract_ref_spec_profile_from_raw.assert_not_called()
    runtime._resample_audio.assert_called_once()
    assert isinstance(refer_spec, GPTSoVITSReferSpec)
    assert refer_spec.spec_audio.ndim == 3
    assert torch.equal(refer_spec.audio_16k, torch.ones((1, 8), dtype=torch.float32))
    assert profile["ref_spec_to_device_ms"] >= 0.0
    assert profile["ref_spec_spectrogram_ms"] >= 0.0
    assert profile["ref_spec_post_resample_ms"] >= 0.0


def test_run_ref_prompt_semantic_stage_uses_runtime_native_helper_without_worker(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))

    @contextmanager
    def _enter_stage():
        yield {"wait_ms": 1.0, "slots": 2.0, "peak_inflight": 3.0}

    pipeline = SimpleNamespace(
        prepare_ref_semantic_batch_worker=None,
        prepare_ref_semantic_stage_limiter=SimpleNamespace(enter=_enter_stage),
        _extract_prompt_semantic_profile_from_prepared_wav16k=Mock(
            side_effect=AssertionError("should not call pipeline prompt semantic helper")
        ),
    )
    runtime._pipeline = pipeline
    runtime._extract_prompt_semantic_profile_from_prepared_wav16k = Mock(  # type: ignore[method-assign]
        return_value=(
            torch.tensor([7, 8], dtype=torch.long),
            {
                "prompt_semantic_h2d_ms": 1.0,
                "prompt_semantic_ssl_forward_ms": 2.0,
                "prompt_semantic_hidden_length_ms": 0.0,
                "prompt_semantic_extract_latent_ms": 3.0,
                "prompt_semantic_forward_ms": 6.0,
            },
        )
    )
    prepared_asset = GPTSoVITSPreparedRefAudioAsset(
        raw_audio=torch.ones((1, 3200), dtype=torch.float32),
        raw_sr=16000,
        wav16k=torch.ones((3200,), dtype=torch.float32),
        profile={
            "audio_load_ms": 4.0,
            "prompt_semantic_cpu_prepare_wait_ms": 5.0,
            "prompt_semantic_cpu_prepare_slots": 6.0,
            "prompt_semantic_cpu_prepare_inflight_peak": 7.0,
            "prompt_semantic_cpu_prepare_ms": 8.0,
        },
    )

    profiled = runtime._run_awaitable_sync(
        runtime._run_ref_prompt_semantic_stage(SimpleNamespace(), "/tmp/ref.wav", prepared_asset=prepared_asset)
    )

    runtime._extract_prompt_semantic_profile_from_prepared_wav16k.assert_called_once_with(prepared_asset.wav16k)
    pipeline._extract_prompt_semantic_profile_from_prepared_wav16k.assert_not_called()
    assert torch.equal(profiled.result.prompt_semantic, torch.tensor([7, 8], dtype=torch.long))
    assert profiled.result.profile["audio_load_ms"] == 4.0
    assert profiled.result.profile["prompt_semantic_forward_ms"] == 6.0
    assert profiled.result.profile["prompt_semantic_cpu_prepare_wait_ms"] == 5.0
    assert profiled.result.profile["audio_stage_slots"] == 2.0


def test_build_ref_prompt_semantic_from_raw_uses_runtime_native_wav16k_helper_for_worker(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    worker = SimpleNamespace(
        bucket_index_for_inputs=Mock(return_value=4),
        pick_runtime_first_hit_shard_index=Mock(return_value=3),
        estimate_runtime_exact_prewarm_target_samples=Mock(return_value=3200),
        run_runtime_exact_prewarm=Mock(
            return_value={
                "prompt_semantic_runtime_exact_prewarm_applied": 1.0,
                "prompt_semantic_runtime_exact_prewarm_ms": 1.5,
            }
        ),
        submit=Mock(
            return_value=(
                torch.tensor([9, 10], dtype=torch.long),
                {
                    "prompt_semantic_wait_ms": 1.0,
                    "prompt_semantic_stage_slots": 2.0,
                    "prompt_semantic_stage_inflight_peak": 3.0,
                    "prompt_semantic_cpu_prepare_ms": 4.0,
                    "prompt_semantic_forward_ms": 5.0,
                    "prompt_semantic_scatter_ms": 6.0,
                },
            )
        )
    )
    pipeline = SimpleNamespace(
        prepare_ref_semantic_batch_worker=worker,
        _prepare_ref_prompt_wav16k_for_worker=Mock(
            side_effect=AssertionError("should not call pipeline wav16k helper")
        ),
    )
    runtime._pipeline = pipeline
    runtime._prepare_coordinator = runtime._coerce_prepare_coordinator(
        SimpleNamespace(
            ref_prompt_semantic_runtime_exact_prewarm_enabled=True,
            ref_prompt_semantic_runtime_exact_prewarm_max_unique=4,
            ref_prompt_semantic_runtime_exact_prewarm_batch_sizes=(1, 4),
            ref_prompt_semantic_runtime_exact_prewarm_lock=threading.Lock(),
            ref_prompt_semantic_runtime_exact_prewarmed_samples=set(),
            ref_prompt_semantic_runtime_exact_prewarm_inflight_samples=set(),
            ref_prompt_semantic_runtime_exact_prewarm_total=0,
            ref_prompt_semantic_runtime_exact_prewarm_total_ms=0.0,
            ref_prompt_semantic_runtime_exact_prewarm_peak_ms=0.0,
            ref_prompt_semantic_bucket_first_hit_serialization_enabled=True,
            ref_prompt_semantic_bucket_first_hit_required_hits=1,
            ref_prompt_semantic_bucket_first_hit_bucket_indices=(4,),
            ref_prompt_semantic_bucket_first_hit_lock=threading.Lock(),
            ref_prompt_semantic_bucket_first_hit_states={},
        )
    )
    runtime._prepare_ref_prompt_wav16k_for_worker = Mock(  # type: ignore[method-assign]
        return_value=(
            torch.ones((3200,), dtype=torch.float32),
            {
                "prompt_semantic_cpu_prepare_wait_ms": 7.0,
                "prompt_semantic_cpu_prepare_slots": 8.0,
                "prompt_semantic_cpu_prepare_inflight_peak": 9.0,
                "prompt_semantic_cpu_prepare_ms": 10.0,
            },
        )
    )

    result = runtime._build_ref_prompt_semantic_from_raw(torch.ones((1, 3200), dtype=torch.float32), 16000)

    runtime._prepare_ref_prompt_wav16k_for_worker.assert_called_once()
    pipeline._prepare_ref_prompt_wav16k_for_worker.assert_not_called()
    worker.submit.assert_called_once()
    worker.estimate_runtime_exact_prewarm_target_samples.assert_called_once()
    worker.run_runtime_exact_prewarm.assert_called_once_with(
        ANY,
        16000,
        wav16k=ANY,
        batch_sizes=[1, 4],
    )
    assert worker.submit.call_args.kwargs["runtime_exact_prewarm_profile"] == {
        "prompt_semantic_runtime_exact_prewarm_applied": 1.0,
        "prompt_semantic_runtime_exact_prewarm_ms": 1.5,
        "prompt_semantic_runtime_exact_prewarm_target_samples": 3200.0,
        "prompt_semantic_runtime_exact_prewarm_batch_sizes": 2.0,
        "prompt_semantic_runtime_exact_prewarm_skipped_capacity": 0.0,
    }
    assert worker.submit.call_args.kwargs["bucket_index"] == 4
    assert worker.submit.call_args.kwargs["preferred_shard_index"] == 3
    assert worker.submit.call_args.kwargs["bucket_first_hit_serialized"] is True
    assert torch.equal(result.prompt_semantic, torch.tensor([9, 10], dtype=torch.long))
    assert result.profile["prompt_semantic_cpu_prepare_ms"] == 14.0
    assert result.profile["prompt_semantic_cpu_prepare_wait_ms"] == 7.0
    assert result.profile["prompt_semantic_cpu_prepare_slots"] == 8.0


def test_build_ref_prompt_semantic_from_raw_prefers_runtime_owned_worker_ref(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    worker = SimpleNamespace(
        bucket_index_for_inputs=Mock(return_value=4),
        pick_runtime_first_hit_shard_index=Mock(return_value=3),
        estimate_runtime_exact_prewarm_target_samples=Mock(return_value=3200),
        run_runtime_exact_prewarm=Mock(
            return_value={
                "prompt_semantic_runtime_exact_prewarm_applied": 1.0,
                "prompt_semantic_runtime_exact_prewarm_ms": 1.5,
            }
        ),
        submit=Mock(
            return_value=(
                torch.tensor([9, 10], dtype=torch.long),
                {
                    "prompt_semantic_wait_ms": 1.0,
                    "prompt_semantic_stage_slots": 2.0,
                    "prompt_semantic_stage_inflight_peak": 3.0,
                    "prompt_semantic_cpu_prepare_ms": 4.0,
                    "prompt_semantic_forward_ms": 5.0,
                    "prompt_semantic_scatter_ms": 6.0,
                },
            )
        ),
    )
    broken_worker = SimpleNamespace(
        submit=Mock(side_effect=AssertionError("should not use pipeline ref worker directly"))
    )
    runtime._runtime_prepare_ref_semantic_batch_worker = worker
    runtime._pipeline = SimpleNamespace(
        prepare_ref_semantic_batch_worker=broken_worker,
        prepare_ref_semantic_stage_limiter=SimpleNamespace(
            enter=Mock(side_effect=AssertionError("should not enter fallback limiter path"))
        ),
    )
    runtime._prepare_coordinator = runtime._coerce_prepare_coordinator(
        SimpleNamespace(
            ref_prompt_semantic_runtime_exact_prewarm_enabled=True,
            ref_prompt_semantic_runtime_exact_prewarm_max_unique=4,
            ref_prompt_semantic_runtime_exact_prewarm_batch_sizes=(1, 4),
            ref_prompt_semantic_runtime_exact_prewarm_lock=threading.Lock(),
            ref_prompt_semantic_runtime_exact_prewarmed_samples=set(),
            ref_prompt_semantic_runtime_exact_prewarm_inflight_samples=set(),
            ref_prompt_semantic_runtime_exact_prewarm_total=0,
            ref_prompt_semantic_runtime_exact_prewarm_total_ms=0.0,
            ref_prompt_semantic_runtime_exact_prewarm_peak_ms=0.0,
            ref_prompt_semantic_bucket_first_hit_serialization_enabled=True,
            ref_prompt_semantic_bucket_first_hit_required_hits=1,
            ref_prompt_semantic_bucket_first_hit_bucket_indices=(4,),
            ref_prompt_semantic_bucket_first_hit_lock=threading.Lock(),
            ref_prompt_semantic_bucket_first_hit_states={},
        )
    )
    runtime._prepare_ref_prompt_wav16k_for_worker = Mock(  # type: ignore[method-assign]
        return_value=(
            torch.ones((3200,), dtype=torch.float32),
            {
                "prompt_semantic_cpu_prepare_wait_ms": 7.0,
                "prompt_semantic_cpu_prepare_slots": 8.0,
                "prompt_semantic_cpu_prepare_inflight_peak": 9.0,
                "prompt_semantic_cpu_prepare_ms": 10.0,
            },
        )
    )

    result = runtime._build_ref_prompt_semantic_from_raw(torch.ones((1, 3200), dtype=torch.float32), 16000)

    worker.submit.assert_called_once()
    broken_worker.submit.assert_not_called()
    assert torch.equal(result.prompt_semantic, torch.tensor([9, 10], dtype=torch.long))
    assert result.profile["prompt_semantic_cpu_prepare_ms"] == 14.0


def test_run_ref_prompt_semantic_stage_prepares_wav16k_before_worker_submit_without_preload(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    worker = SimpleNamespace(
        bucket_index_for_inputs=Mock(return_value=4),
        pick_runtime_first_hit_shard_index=Mock(return_value=3),
        estimate_runtime_exact_prewarm_target_samples=Mock(return_value=3200),
        run_runtime_exact_prewarm=Mock(
            return_value={
                "prompt_semantic_runtime_exact_prewarm_applied": 1.0,
                "prompt_semantic_runtime_exact_prewarm_ms": 2.5,
            }
        ),
        submit_async=AsyncMock(
            return_value=(
                torch.tensor([5, 6], dtype=torch.long),
                {
                    "prompt_semantic_wait_ms": 1.0,
                    "prompt_semantic_stage_slots": 2.0,
                    "prompt_semantic_stage_inflight_peak": 3.0,
                    "prompt_semantic_cpu_prepare_ms": 4.0,
                    "prompt_semantic_forward_ms": 5.0,
                    "prompt_semantic_scatter_ms": 6.0,
                },
            )
        )
    )
    pipeline = SimpleNamespace(
        prepare_ref_semantic_batch_worker=worker,
        prepare_ref_semantic_stage_limiter=None,
    )
    runtime._pipeline = pipeline
    runtime._prepare_coordinator = runtime._coerce_prepare_coordinator(
        SimpleNamespace(
            ref_prompt_semantic_runtime_exact_prewarm_enabled=True,
            ref_prompt_semantic_runtime_exact_prewarm_max_unique=4,
            ref_prompt_semantic_runtime_exact_prewarm_batch_sizes=(1, 2),
            ref_prompt_semantic_runtime_exact_prewarm_lock=threading.Lock(),
            ref_prompt_semantic_runtime_exact_prewarmed_samples=set(),
            ref_prompt_semantic_runtime_exact_prewarm_inflight_samples=set(),
            ref_prompt_semantic_runtime_exact_prewarm_total=0,
            ref_prompt_semantic_runtime_exact_prewarm_total_ms=0.0,
            ref_prompt_semantic_runtime_exact_prewarm_peak_ms=0.0,
            ref_prompt_semantic_bucket_first_hit_serialization_enabled=True,
            ref_prompt_semantic_bucket_first_hit_required_hits=1,
            ref_prompt_semantic_bucket_first_hit_bucket_indices=(4,),
            ref_prompt_semantic_bucket_first_hit_lock=threading.Lock(),
            ref_prompt_semantic_bucket_first_hit_states={},
        )
    )
    runtime._prepare_run_on_executor = AsyncMock(  # type: ignore[method-assign]
        return_value=GPTSoVITSPrepareProfiledResult(
            result=(torch.ones((1, 3200), dtype=torch.float32), 16000),
            submit_at=1.0,
            started_at=1.5,
            finished_at=2.0,
        )
    )
    runtime._prepare_ref_prompt_wav16k_for_worker = Mock(  # type: ignore[method-assign]
        return_value=(
            torch.ones((3200,), dtype=torch.float32),
            {
                "prompt_semantic_cpu_prepare_wait_ms": 7.0,
                "prompt_semantic_cpu_prepare_slots": 8.0,
                "prompt_semantic_cpu_prepare_inflight_peak": 9.0,
                "prompt_semantic_cpu_prepare_ms": 10.0,
            },
        )
    )

    profiled = runtime._run_awaitable_sync(
        runtime._run_ref_prompt_semantic_stage(runtime._prepare_coordinator, "/tmp/ref.wav", prepared_asset_future=None)
    )

    runtime._prepare_ref_prompt_wav16k_for_worker.assert_called_once()
    worker.submit_async.assert_awaited_once()
    submit_args = worker.submit_async.await_args.args
    submit_kwargs = worker.submit_async.await_args.kwargs
    worker.estimate_runtime_exact_prewarm_target_samples.assert_called_once()
    worker.run_runtime_exact_prewarm.assert_called_once_with(
        ANY,
        16000,
        wav16k=ANY,
        batch_sizes=[1, 2],
    )
    assert submit_args[1] == 16000
    assert torch.equal(submit_kwargs["wav16k"], torch.ones((3200,), dtype=torch.float32))
    assert submit_kwargs["runtime_exact_prewarm_profile"] == {
        "prompt_semantic_runtime_exact_prewarm_applied": 1.0,
        "prompt_semantic_runtime_exact_prewarm_ms": 2.5,
        "prompt_semantic_runtime_exact_prewarm_target_samples": 3200.0,
        "prompt_semantic_runtime_exact_prewarm_batch_sizes": 2.0,
        "prompt_semantic_runtime_exact_prewarm_skipped_capacity": 0.0,
    }
    assert submit_kwargs["bucket_index"] == 4
    assert submit_kwargs["preferred_shard_index"] == 3
    assert submit_kwargs["bucket_first_hit_serialized"] is True
    assert profiled.result.profile["prompt_semantic_preload_cpu_prepare_ms"] == 10.0
    assert profiled.result.profile["prompt_semantic_cpu_prepare_wait_ms"] == 7.0
    assert profiled.result.profile["prompt_semantic_cpu_prepare_slots"] == 8.0


def test_run_ref_prompt_semantic_stage_batch_uses_worker_with_preloaded_assets(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    worker = SimpleNamespace(
        bucket_index_for_inputs=Mock(side_effect=[4, 4]),
        pick_runtime_first_hit_shard_index=Mock(return_value=3),
        estimate_runtime_exact_prewarm_target_samples=Mock(side_effect=[3200, 6400]),
        run_runtime_exact_prewarm=Mock(
            side_effect=[
                {
                    "prompt_semantic_runtime_exact_prewarm_applied": 1.0,
                    "prompt_semantic_runtime_exact_prewarm_ms": 3.5,
                    "prompt_semantic_runtime_exact_prewarm_target_samples": 3200.0,
                    "prompt_semantic_runtime_exact_prewarm_batch_sizes": 2.0,
                    "prompt_semantic_runtime_exact_prewarm_skipped_capacity": 0.0,
                },
                {
                    "prompt_semantic_runtime_exact_prewarm_applied": 0.0,
                    "prompt_semantic_runtime_exact_prewarm_ms": 0.0,
                    "prompt_semantic_runtime_exact_prewarm_target_samples": 6400.0,
                    "prompt_semantic_runtime_exact_prewarm_batch_sizes": 2.0,
                    "prompt_semantic_runtime_exact_prewarm_skipped_capacity": 0.0,
                },
            ]
        ),
        submit_async=AsyncMock(
            side_effect=[
                (
                    torch.tensor([5, 6], dtype=torch.long),
                    {
                        "prompt_semantic_wait_ms": 1.0,
                        "prompt_semantic_stage_slots": 2.0,
                        "prompt_semantic_stage_inflight_peak": 3.0,
                        "prompt_semantic_cpu_prepare_ms": 4.0,
                        "prompt_semantic_forward_ms": 5.0,
                        "prompt_semantic_scatter_ms": 6.0,
                    },
                ),
                (
                    torch.tensor([7, 8], dtype=torch.long),
                    {
                        "prompt_semantic_wait_ms": 1.5,
                        "prompt_semantic_stage_slots": 2.5,
                        "prompt_semantic_stage_inflight_peak": 3.5,
                        "prompt_semantic_cpu_prepare_ms": 4.5,
                        "prompt_semantic_forward_ms": 5.5,
                        "prompt_semantic_scatter_ms": 6.5,
                    },
                ),
            ]
        )
    )
    pipeline = SimpleNamespace(
        prepare_ref_semantic_batch_worker=worker,
        prepare_ref_semantic_stage_limiter=SimpleNamespace(snapshot=Mock(return_value={"slots": 9.0, "peak_inflight": 10.0})),
    )
    runtime._pipeline = pipeline
    runtime._prepare_coordinator = runtime._coerce_prepare_coordinator(
        SimpleNamespace(
            ref_prompt_semantic_runtime_exact_prewarm_enabled=True,
            ref_prompt_semantic_runtime_exact_prewarm_max_unique=4,
            ref_prompt_semantic_runtime_exact_prewarm_batch_sizes=(1, 2),
            ref_prompt_semantic_runtime_exact_prewarm_lock=threading.Lock(),
            ref_prompt_semantic_runtime_exact_prewarmed_samples=set(),
            ref_prompt_semantic_runtime_exact_prewarm_inflight_samples=set(),
            ref_prompt_semantic_runtime_exact_prewarm_total=0,
            ref_prompt_semantic_runtime_exact_prewarm_total_ms=0.0,
            ref_prompt_semantic_runtime_exact_prewarm_peak_ms=0.0,
            ref_prompt_semantic_bucket_first_hit_serialization_enabled=True,
            ref_prompt_semantic_bucket_first_hit_required_hits=2,
            ref_prompt_semantic_bucket_first_hit_bucket_indices=(4,),
            ref_prompt_semantic_bucket_first_hit_lock=threading.Lock(),
            ref_prompt_semantic_bucket_first_hit_states={},
        )
    )

    future_a = concurrent.futures.Future()
    future_a.set_result(
        GPTSoVITSPrepareProfiledResult(
            result=GPTSoVITSPreparedRefAudioAsset(
                raw_audio=torch.ones((1, 3200), dtype=torch.float32),
                raw_sr=16000,
                wav16k=torch.ones((3200,), dtype=torch.float32),
                profile={
                    "audio_load_ms": 4.0,
                    "prompt_semantic_cpu_prepare_wait_ms": 5.0,
                    "prompt_semantic_cpu_prepare_slots": 6.0,
                    "prompt_semantic_cpu_prepare_inflight_peak": 7.0,
                    "prompt_semantic_cpu_prepare_ms": 8.0,
                },
            ),
            submit_at=1.0,
            started_at=1.0,
            finished_at=1.5,
        )
    )
    future_b = concurrent.futures.Future()
    future_b.set_result(
        GPTSoVITSPrepareProfiledResult(
            result=GPTSoVITSPreparedRefAudioAsset(
                raw_audio=torch.full((1, 3200), 2.0, dtype=torch.float32),
                raw_sr=16000,
                wav16k=torch.full((3200,), 2.0, dtype=torch.float32),
                profile={
                    "audio_load_ms": 9.0,
                    "prompt_semantic_cpu_prepare_wait_ms": 10.0,
                    "prompt_semantic_cpu_prepare_slots": 11.0,
                    "prompt_semantic_cpu_prepare_inflight_peak": 12.0,
                    "prompt_semantic_cpu_prepare_ms": 13.0,
                },
            ),
            submit_at=2.0,
            started_at=2.0,
            finished_at=2.5,
        )
    )

    results = runtime._run_awaitable_sync(
        runtime._run_ref_prompt_semantic_stage_batch(
            runtime._prepare_coordinator,
            [
                ("/tmp/a.wav", future_a, None),
                ("/tmp/b.wav", future_b, None),
            ],
        )
    )

    assert worker.submit_async.await_count == 2
    assert worker.estimate_runtime_exact_prewarm_target_samples.call_count == 2
    assert worker.run_runtime_exact_prewarm.call_count == 2
    first_kwargs = worker.submit_async.await_args_list[0].kwargs
    second_kwargs = worker.submit_async.await_args_list[1].kwargs
    assert torch.equal(first_kwargs["wav16k"], torch.ones((3200,), dtype=torch.float32))
    assert torch.equal(second_kwargs["wav16k"], torch.full((3200,), 2.0, dtype=torch.float32))
    assert first_kwargs["runtime_exact_prewarm_profile"] == {
        "prompt_semantic_runtime_exact_prewarm_applied": 1.0,
        "prompt_semantic_runtime_exact_prewarm_ms": 3.5,
        "prompt_semantic_runtime_exact_prewarm_target_samples": 3200.0,
        "prompt_semantic_runtime_exact_prewarm_batch_sizes": 2.0,
        "prompt_semantic_runtime_exact_prewarm_skipped_capacity": 0.0,
    }
    assert first_kwargs["bucket_index"] == 4
    assert first_kwargs["preferred_shard_index"] == 3
    assert first_kwargs["bucket_first_hit_serialized"] is True
    assert second_kwargs["runtime_exact_prewarm_profile"] == {
        "prompt_semantic_runtime_exact_prewarm_applied": 0.0,
        "prompt_semantic_runtime_exact_prewarm_ms": 0.0,
        "prompt_semantic_runtime_exact_prewarm_target_samples": 6400.0,
        "prompt_semantic_runtime_exact_prewarm_batch_sizes": 2.0,
        "prompt_semantic_runtime_exact_prewarm_skipped_capacity": 0.0,
    }
    assert second_kwargs["bucket_index"] == 4
    assert second_kwargs["preferred_shard_index"] == 3
    assert second_kwargs["bucket_first_hit_serialized"] is True
    assert all(not isinstance(item, Exception) for item in results)
    first = results[0]
    second = results[1]
    assert isinstance(first, GPTSoVITSPrepareProfiledResult)
    assert isinstance(second, GPTSoVITSPrepareProfiledResult)
    assert first.result.profile["prompt_semantic_preload_cpu_prepare_ms"] == 8.0
    assert first.result.profile["prompt_semantic_cpu_prepare_wait_ms"] == 5.0
    assert first.result.profile["audio_stage_slots"] == 9.0
    assert second.result.profile["prompt_semantic_preload_cpu_prepare_ms"] == 13.0
    assert second.result.profile["prompt_semantic_cpu_prepare_slots"] == 11.0
    assert second.result.profile["audio_stage_inflight_peak"] == 10.0


def test_build_ref_prompt_semantic_runtime_exact_prewarm_profile_uses_runtime_coordinator_dedupe_and_capacity(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    coordinator = runtime._coerce_prepare_coordinator(
        SimpleNamespace(
            ref_prompt_semantic_runtime_exact_prewarm_enabled=True,
            ref_prompt_semantic_runtime_exact_prewarm_max_unique=1,
            ref_prompt_semantic_runtime_exact_prewarm_batch_sizes=(1, 3),
            ref_prompt_semantic_runtime_exact_prewarm_lock=threading.Lock(),
            ref_prompt_semantic_runtime_exact_prewarmed_samples=set(),
            ref_prompt_semantic_runtime_exact_prewarm_inflight_samples=set(),
            ref_prompt_semantic_runtime_exact_prewarm_total=0,
            ref_prompt_semantic_runtime_exact_prewarm_total_ms=0.0,
            ref_prompt_semantic_runtime_exact_prewarm_peak_ms=0.0,
        )
    )
    worker = SimpleNamespace(
        estimate_runtime_exact_prewarm_target_samples=Mock(side_effect=[3200, 3200, 6400]),
        run_runtime_exact_prewarm=Mock(
            return_value={
                "prompt_semantic_runtime_exact_prewarm_applied": 1.0,
                "prompt_semantic_runtime_exact_prewarm_ms": 4.0,
                "prompt_semantic_runtime_exact_prewarm_target_samples": 3200.0,
                "prompt_semantic_runtime_exact_prewarm_batch_sizes": 2.0,
                "prompt_semantic_runtime_exact_prewarm_skipped_capacity": 0.0,
            }
        ),
    )

    first = runtime._build_ref_prompt_semantic_runtime_exact_prewarm_profile(
        coordinator,
        worker,
        torch.ones((1, 3200), dtype=torch.float32),
        16000,
        wav16k=torch.ones((3200,), dtype=torch.float32),
    )
    second = runtime._build_ref_prompt_semantic_runtime_exact_prewarm_profile(
        coordinator,
        worker,
        torch.ones((1, 3200), dtype=torch.float32),
        16000,
        wav16k=torch.ones((3200,), dtype=torch.float32),
    )
    third = runtime._build_ref_prompt_semantic_runtime_exact_prewarm_profile(
        coordinator,
        worker,
        torch.ones((1, 6400), dtype=torch.float32),
        16000,
        wav16k=torch.ones((6400,), dtype=torch.float32),
    )

    worker.run_runtime_exact_prewarm.assert_called_once_with(
        ANY,
        16000,
        wav16k=ANY,
        batch_sizes=[1, 3],
    )
    assert first["prompt_semantic_runtime_exact_prewarm_applied"] == 1.0
    assert second["prompt_semantic_runtime_exact_prewarm_applied"] == 0.0
    assert third["prompt_semantic_runtime_exact_prewarm_skipped_capacity"] == 1.0
    assert third["prompt_semantic_runtime_exact_prewarm_target_samples"] == 6400.0
    assert coordinator.ref_prompt_semantic_runtime_exact_prewarm_total == 1
    assert coordinator.ref_prompt_semantic_runtime_exact_prewarm_total_ms == 4.0
    assert coordinator.ref_prompt_semantic_runtime_exact_prewarm_peak_ms == 4.0


def test_build_ref_prompt_semantic_worker_routing_uses_runtime_coordinator_first_hit_state(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    coordinator = runtime._coerce_prepare_coordinator(
        SimpleNamespace(
            ref_prompt_semantic_bucket_first_hit_serialization_enabled=True,
            ref_prompt_semantic_bucket_first_hit_required_hits=2,
            ref_prompt_semantic_bucket_first_hit_bucket_indices=(4,),
            ref_prompt_semantic_bucket_first_hit_lock=threading.Lock(),
            ref_prompt_semantic_bucket_first_hit_states={},
        )
    )
    worker = SimpleNamespace(
        bucket_index_for_inputs=Mock(side_effect=[4, 4, 4]),
        pick_runtime_first_hit_shard_index=Mock(return_value=7),
    )

    route_a = runtime._build_ref_prompt_semantic_worker_routing(
        coordinator,
        worker,
        torch.ones((1, 3200), dtype=torch.float32),
        16000,
        wav16k=torch.ones((3200,), dtype=torch.float32),
    )
    route_b = runtime._build_ref_prompt_semantic_worker_routing(
        coordinator,
        worker,
        torch.ones((1, 3200), dtype=torch.float32),
        16000,
        wav16k=torch.ones((3200,), dtype=torch.float32),
    )
    route_c = runtime._build_ref_prompt_semantic_worker_routing(
        coordinator,
        worker,
        torch.ones((1, 3200), dtype=torch.float32),
        16000,
        wav16k=torch.ones((3200,), dtype=torch.float32),
    )

    runtime._mark_ref_prompt_semantic_worker_routing_completed(coordinator, route_a)
    runtime._mark_ref_prompt_semantic_worker_routing_completed(coordinator, route_b)

    worker.pick_runtime_first_hit_shard_index.assert_called_once()
    assert route_a == {"bucket_index": 4, "preferred_shard_index": 7, "bucket_first_hit_serialized": True}
    assert route_b == {"bucket_index": 4, "preferred_shard_index": 7, "bucket_first_hit_serialized": True}
    assert route_c == {"bucket_index": 4, "preferred_shard_index": None, "bucket_first_hit_serialized": False}
    assert coordinator.ref_prompt_semantic_bucket_first_hit_states[4]["dispatched_hits"] == 2
    assert coordinator.ref_prompt_semantic_bucket_first_hit_states[4]["completed_hits"] == 2


def test_select_ref_prompt_semantic_worker_shard_index_uses_runtime_bucket_aware_policy(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    coordinator = runtime._coerce_prepare_coordinator(
        SimpleNamespace(
            ref_prompt_semantic_bucket_aware_sharding=True,
            ref_prompt_semantic_bucket_aware_max_outstanding_gap=2,
        )
    )
    worker = SimpleNamespace(
        runtime_routing_snapshots_for_bucket=Mock(
            return_value=[
                {
                    "shard_index": 0,
                    "mergeable_pending": 0,
                    "exact_pending": 0,
                    "outstanding": 1,
                    "outstanding_samples": 100,
                    "active_mergeable": 0,
                    "active_batch_size": 0,
                },
                {
                    "shard_index": 1,
                    "mergeable_pending": 3,
                    "exact_pending": 1,
                    "outstanding": 2,
                    "outstanding_samples": 140,
                    "active_mergeable": 1,
                    "active_batch_size": 1,
                },
                {
                    "shard_index": 2,
                    "mergeable_pending": 2,
                    "exact_pending": 2,
                    "outstanding": 3,
                    "outstanding_samples": 120,
                    "active_mergeable": 1,
                    "active_batch_size": 2,
                },
            ]
        )
    )

    shard_index = runtime._select_ref_prompt_semantic_worker_shard_index(
        coordinator,
        worker,
        bucket_index=4,
    )

    worker.runtime_routing_snapshots_for_bucket.assert_called_once_with(4)
    assert shard_index == 1


def test_prepare_decode_requests_batches_prepare_contract(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime.prepare_decode_request = Mock(side_effect=["prepared-a", "prepared-b"])  # type: ignore[method-assign]

    prepared = runtime.prepare_decode_requests(
        [torch.tensor([1], dtype=torch.long), torch.tensor([2, 3], dtype=torch.long)],
        [{"request_id": "a"}, {"request_id": "b"}],
    )

    assert prepared == ["prepared-a", "prepared-b"]
    assert runtime.prepare_decode_request.call_count == 2


def test_decode_prepared_request_uses_native_decode_fragment_helper(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = SimpleNamespace(configs=SimpleNamespace(sampling_rate=32000))
    runtime._ensure_pipeline = Mock(return_value=pipeline)  # type: ignore[method-assign]
    runtime._decode_prepared_request_fragment = Mock(return_value=("fragment", 32000))  # type: ignore[method-assign]

    decoded = runtime.decode_prepared_request(
        SimpleNamespace(
            request_id="req",
            semantic_tokens=torch.tensor([1, 2], dtype=torch.long),
            speed_factor=1.0,
            super_sampling=False,
        )
    )

    runtime._decode_prepared_request_fragment.assert_called_once_with(pipeline, ANY)
    assert decoded.audio_fragment == "fragment"
    assert decoded.output_sr == 32000


def test_decode_prepared_request_fragment_routes_vocoder_to_native_helper(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = SimpleNamespace(
        configs=SimpleNamespace(use_vocoder=True, device="cpu"),
        vocoder_configs={"sr": 24000},
    )
    runtime._decode_prepared_request_vocoder_fragment = Mock(return_value=("vocoder-audio", 24000))  # type: ignore[method-assign]

    audio_fragment, output_sr = runtime._decode_prepared_request_fragment(
        pipeline,
        SimpleNamespace(
            refer_audio_spec=torch.ones((1, 10, 4), dtype=torch.float32),
            refer_audio_16k=torch.zeros((0,), dtype=torch.float32),
        ),
    )

    runtime._decode_prepared_request_vocoder_fragment.assert_called_once_with(pipeline, ANY, ANY)
    assert audio_fragment == "vocoder-audio"
    assert output_sr == 24000


def test_decode_prepared_requests_batches_non_vocoder_groups(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = SimpleNamespace(configs=SimpleNamespace(sampling_rate=32000, use_vocoder=False))
    runtime._ensure_pipeline = Mock(return_value=pipeline)  # type: ignore[method-assign]
    runtime._decode_prepared_requests_batched_non_vocoder = Mock(  # type: ignore[method-assign]
        return_value=[
            GPTSoVITSDecodedAudio("a", "audio-a", 32000, 1.0, 0.3, False),
            GPTSoVITSDecodedAudio("b", "audio-b", 32000, 1.0, 0.3, False),
        ]
    )
    runtime.decode_prepared_request = Mock(  # type: ignore[method-assign]
        return_value=GPTSoVITSDecodedAudio("c", "audio-c", 32000, 1.5, 0.3, False)
    )

    decoded = runtime.decode_prepared_requests(
        [
            SimpleNamespace(request_id="a", semantic_tokens=torch.tensor([1]), speed_factor=1.0, fragment_interval=0.3, sample_steps=32, super_sampling=False),
            SimpleNamespace(request_id="b", semantic_tokens=torch.tensor([2]), speed_factor=1.0, fragment_interval=0.3, sample_steps=32, super_sampling=False),
            SimpleNamespace(request_id="c", semantic_tokens=torch.tensor([3]), speed_factor=1.5, fragment_interval=0.3, sample_steps=32, super_sampling=False),
        ]
    )

    runtime._decode_prepared_requests_batched_non_vocoder.assert_called_once()
    runtime.decode_prepared_request.assert_called_once()
    assert [item.request_id for item in decoded] == ["a", "b", "c"]
    assert [item.audio_fragment for item in decoded] == ["audio-a", "audio-b", "audio-c"]


def test_decode_prepared_requests_batches_vocoder_groups_by_prompt_context(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = SimpleNamespace(configs=SimpleNamespace(sampling_rate=32000, use_vocoder=True))
    shared_prompt_key = ("shared",)
    other_prompt_key = ("other",)
    runtime._ensure_pipeline = Mock(return_value=pipeline)  # type: ignore[method-assign]
    runtime._build_refer_spec_from_prepared = Mock(  # type: ignore[method-assign]
        side_effect=lambda prepared: GPTSoVITSReferSpec(prepared.refer_audio_spec, None)
    )
    runtime._build_vocoder_prompt_cache_key = Mock(  # type: ignore[method-assign]
        side_effect=lambda prepared, refer_audio_spec: shared_prompt_key if prepared.request_id in {"a", "b"} else other_prompt_key
    )
    runtime._decode_prepared_requests_grouped_vocoder = Mock(  # type: ignore[method-assign]
        return_value=[
            GPTSoVITSDecodedAudio("a", "audio-a", 24000, 1.0, 0.3, False),
            GPTSoVITSDecodedAudio("b", "audio-b", 24000, 1.5, 0.3, True),
        ]
    )
    runtime.decode_prepared_request = Mock(  # type: ignore[method-assign]
        return_value=GPTSoVITSDecodedAudio("c", "audio-c", 24000, 0.8, 0.3, False)
    )

    decoded = runtime.decode_prepared_requests(
        [
            SimpleNamespace(
                request_id="a",
                semantic_tokens=torch.tensor([1], dtype=torch.long),
                refer_audio_spec=torch.ones((1, 10, 4), dtype=torch.float32),
                speed_factor=1.0,
                sample_steps=16,
                super_sampling=False,
            ),
            SimpleNamespace(
                request_id="b",
                semantic_tokens=torch.tensor([2], dtype=torch.long),
                refer_audio_spec=torch.ones((1, 10, 4), dtype=torch.float32),
                speed_factor=1.5,
                sample_steps=32,
                super_sampling=True,
            ),
            SimpleNamespace(
                request_id="c",
                semantic_tokens=torch.tensor([3], dtype=torch.long),
                refer_audio_spec=torch.ones((1, 10, 4), dtype=torch.float32),
                speed_factor=0.8,
                sample_steps=8,
                super_sampling=False,
            ),
        ]
    )

    runtime._decode_prepared_requests_grouped_vocoder.assert_called_once()
    grouped_prepared = runtime._decode_prepared_requests_grouped_vocoder.call_args.args[1]
    assert [item.request_id for item in grouped_prepared] == ["a", "b"]
    runtime.decode_prepared_request.assert_called_once()
    assert [item.request_id for item in decoded] == ["a", "b", "c"]
    assert [item.audio_fragment for item in decoded] == ["audio-a", "audio-b", "audio-c"]


def test_finalize_decoded_audios_batches_finalize_contract(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime.finalize_decoded_audio = Mock(side_effect=["result-a", "result-b"])  # type: ignore[method-assign]

    results = runtime.finalize_decoded_audios(["decoded-a", "decoded-b"])  # type: ignore[arg-type]

    assert results == ["result-a", "result-b"]
    assert runtime.finalize_decoded_audio.call_count == 2


def test_finalize_decoded_audio_normalizes_output(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))

    pipeline = SimpleNamespace()
    runtime._audio_postprocess_native = Mock(return_value=(32000, np.array([32767, -32768], dtype=np.int16)))  # type: ignore[method-assign]
    runtime._pipeline = pipeline

    result = runtime.finalize_decoded_audio(
        SimpleNamespace(
            audio_fragment=np.array([0.1], dtype=np.float32),
            output_sr=32000,
            speed_factor=1.0,
            super_sampling=False,
        )
    )

    runtime._audio_postprocess_native.assert_called_once()
    assert result.sample_rate == 32000
    np.testing.assert_allclose(result.audio, np.array([32767 / 32768.0, -1.0], dtype=np.float32))


def test_audio_postprocess_native_super_sampling_uses_runtime_sr_loader(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    sr_model = Mock(return_value=(torch.tensor([[0.25, -0.5]], dtype=torch.float32), 48000))
    runtime._ensure_runtime_sr_model = Mock(return_value=(sr_model, False))  # type: ignore[method-assign]
    pipeline = SimpleNamespace(
        configs=SimpleNamespace(device="cpu", sampling_rate=32000),
        init_sr_model=Mock(side_effect=AssertionError("legacy sr init should not be used")),
    )

    sample_rate, audio = runtime._audio_postprocess_native(
        pipeline,
        audio_fragments=[torch.tensor([0.1, 0.2], dtype=torch.float32)],
        sr=32000,
        speed_factor=1.0,
        super_sampling=True,
    )

    runtime._ensure_runtime_sr_model.assert_called_once_with()
    sr_model.assert_called_once()
    assert sample_rate == 48000
    assert audio.tolist() == [8192, -16384]


def test_decode_prepared_requests_batched_non_vocoder_uses_native_vits_batch(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._get_text_frontend_symbol = Mock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            sequence_mask=lambda lengths, max_len: (
                torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
            )
        )
    )
    vits_model = SimpleNamespace(
        version="v2",
        is_v2pro=False,
        semantic_frame_rate="50hz",
        ref_enc=Mock(side_effect=lambda refer, mask: torch.ones((refer.shape[0], 2, 3), dtype=torch.float32)),
        quantizer=SimpleNamespace(decode=Mock(return_value=torch.ones((1, 2, 3), dtype=torch.float32))),
        enc_p=Mock(
            return_value=(
                torch.zeros((2, 1, 4), dtype=torch.float32),
                torch.zeros((2, 1, 4), dtype=torch.float32),
                torch.zeros((2, 1, 4), dtype=torch.float32),
                torch.tensor(
                    [
                        [[1.0, 1.0, 1.0, 0.0]],
                        [[1.0, 1.0, 0.0, 0.0]],
                    ],
                    dtype=torch.float32,
                ),
                None,
                None,
            )
        ),
        flow=Mock(side_effect=lambda z_p, y_mask, g=None, reverse=True: z_p),
        _decode_audio_runtime=Mock(
            return_value=torch.tensor(
                [
                    [[0.1, 0.2, 0.3, 0.0]],
                    [[0.4, 0.5, 0.0, 0.0]],
                ],
                dtype=torch.float32,
            )
        ),
        dec=SimpleNamespace(ups=[SimpleNamespace(stride=(1,))]),
        decode_batched_request_local=Mock(side_effect=AssertionError("should not call vits wrapper")),
    )
    pipeline = SimpleNamespace(
        configs=SimpleNamespace(sampling_rate=32000, use_vocoder=False, device="cpu"),
        precision=torch.float16,
        is_v2pro=False,
        vits_model=vits_model,
    )

    decoded = runtime._decode_prepared_requests_batched_non_vocoder(
        pipeline,
        [
            SimpleNamespace(
                request_id="a",
                semantic_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
                phones=torch.tensor([10, 11], dtype=torch.long),
                refer_audio_spec=torch.ones((1, 704, 4), dtype=torch.float32),
                refer_audio_16k=torch.zeros((0,), dtype=torch.float32),
                speed_factor=1.0,
                sample_steps=32,
                super_sampling=False,
            ),
            SimpleNamespace(
                request_id="b",
                semantic_tokens=torch.tensor([4, 5], dtype=torch.long),
                phones=torch.tensor([12], dtype=torch.long),
                refer_audio_spec=torch.ones((1, 704, 4), dtype=torch.float32),
                refer_audio_16k=torch.zeros((0,), dtype=torch.float32),
                speed_factor=1.0,
                sample_steps=32,
                super_sampling=False,
            ),
        ],
    )

    vits_model.quantizer.decode.assert_called_once()
    vits_model.enc_p.assert_called_once()
    vits_model._decode_audio_runtime.assert_called_once()
    vits_model.decode_batched_request_local.assert_not_called()
    assert [item.request_id for item in decoded] == ["a", "b"]
    assert decoded[0].audio_fragment.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert decoded[1].audio_fragment.tolist() == pytest.approx([0.4, 0.5])


def test_decode_prepared_request_fragment_uses_runtime_native_vits_path(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._get_text_frontend_symbol = Mock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            sequence_mask=lambda lengths, max_len: (
                torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
            )
        )
    )
    vits_model = SimpleNamespace(
        version="v2",
        is_v2pro=False,
        semantic_frame_rate="50hz",
        ref_enc=Mock(return_value=torch.ones((1, 2, 3), dtype=torch.float32)),
        quantizer=SimpleNamespace(decode=Mock(return_value=torch.ones((1, 1, 3), dtype=torch.float32))),
        enc_p=Mock(
            return_value=(
                torch.zeros((1, 1, 3), dtype=torch.float32),
                torch.zeros((1, 1, 3), dtype=torch.float32),
                torch.zeros((1, 1, 3), dtype=torch.float32),
                torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32),
                None,
                None,
            )
        ),
        flow=Mock(side_effect=lambda z_p, y_mask, g=None, reverse=True: z_p),
        _decode_audio_runtime=Mock(return_value=torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32)),
        dec=SimpleNamespace(ups=[SimpleNamespace(stride=(1,))]),
        decode=Mock(side_effect=AssertionError("should not call vits decode wrapper")),
    )
    pipeline = SimpleNamespace(
        configs=SimpleNamespace(sampling_rate=32000, use_vocoder=False, device="cpu"),
        precision=torch.float16,
        is_v2pro=False,
        vits_model=vits_model,
    )

    audio_fragment, output_sr = runtime._decode_prepared_request_fragment(
        pipeline,
        SimpleNamespace(
            semantic_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            phones=torch.tensor([10, 11], dtype=torch.long),
            refer_audio_spec=torch.ones((1, 704, 4), dtype=torch.float32),
            refer_audio_16k=torch.zeros((0,), dtype=torch.float32),
            speed_factor=1.0,
            sample_steps=32,
        ),
    )

    vits_model.quantizer.decode.assert_called_once()
    vits_model.enc_p.assert_called_once()
    vits_model._decode_audio_runtime.assert_called_once()
    vits_model.decode.assert_not_called()
    assert audio_fragment.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert output_sr == 32000


def test_decode_prepared_request_vocoder_fragment_uses_native_components(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._compute_vocoder_mel = Mock(return_value=torch.ones((1, 100, 6), dtype=torch.float32))  # type: ignore[method-assign]
    runtime._resample_audio = Mock(side_effect=lambda audio, sr0, sr1, device: audio)  # type: ignore[method-assign]
    runtime._get_text_frontend_symbol = Mock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            sequence_mask=lambda lengths, max_len: (
                torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
            )
        )
    )

    vits_model = SimpleNamespace(
        version="v3",
        semantic_frame_rate="50hz",
        ref_enc=Mock(return_value=torch.ones((1, 2, 1), dtype=torch.float32)),
        quantizer=SimpleNamespace(
            decode=Mock(
                side_effect=[
                    torch.ones((1, 1, 3), dtype=torch.float32),
                    torch.ones((1, 1, 4), dtype=torch.float32),
                ]
            )
        ),
        enc_p=Mock(
            side_effect=[
                (
                    torch.ones((1, 50, 3), dtype=torch.float32),
                    None,
                    None,
                    torch.ones((1, 1, 3), dtype=torch.float32),
                    None,
                    None,
                ),
                (
                    torch.ones((1, 50, 4), dtype=torch.float32),
                    None,
                    None,
                    torch.ones((1, 1, 4), dtype=torch.float32),
                    None,
                    None,
                ),
            ]
        ),
        bridge=Mock(
            side_effect=[
                torch.ones((1, 100, 4), dtype=torch.float32),
                torch.ones((1, 100, 5), dtype=torch.float32),
            ]
        ),
        wns1=Mock(
            side_effect=[
                (torch.ones((1, 100, 6), dtype=torch.float32), torch.ones((1, 1, 6), dtype=torch.float32)),
                (torch.ones((1, 100, 8), dtype=torch.float32), torch.ones((1, 1, 8), dtype=torch.float32)),
            ]
        ),
        decode_encp=Mock(side_effect=AssertionError("should not call decode_encp wrapper")),
        cfm=SimpleNamespace(inference=Mock(return_value=torch.ones((1, 100, 10), dtype=torch.float32))),
    )
    pipeline = SimpleNamespace(
        configs=SimpleNamespace(device="cpu", version="v3"),
        precision=torch.float16,
        vits_model=vits_model,
        vocoder_configs={"T_ref": 4, "T_chunk": 8, "sr": 24000},
        vocoder=Mock(return_value=torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32)),
    )

    audio_fragment, output_sr = runtime._decode_prepared_request_vocoder_fragment(
        pipeline,
        SimpleNamespace(
            prompt_semantic=torch.tensor([1, 2], dtype=torch.long),
            prompt_phones=torch.tensor([3, 4], dtype=torch.long),
            raw_audio=torch.ones((1, 12), dtype=torch.float32),
            raw_sr=24000,
            semantic_tokens=torch.tensor([5, 6, 7], dtype=torch.long),
            phones=torch.tensor([8, 9], dtype=torch.long),
            speed_factor=1.0,
            sample_steps=16,
        ),
        torch.ones((1, 704, 4), dtype=torch.float32),
    )

    assert vits_model.quantizer.decode.call_count == 2
    assert vits_model.enc_p.call_count == 2
    assert vits_model.bridge.call_count == 2
    assert vits_model.wns1.call_count == 2
    vits_model.decode_encp.assert_not_called()
    vits_model.cfm.inference.assert_called()
    pipeline.vocoder.assert_called_once()
    assert audio_fragment.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert output_sr == 24000


def test_resolve_vocoder_runtime_config_fills_defaults_from_version(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))

    config = runtime._resolve_vocoder_runtime_config(
        SimpleNamespace(
            configs=SimpleNamespace(version="v3"),
            vocoder_configs={"sr": None, "T_ref": 256},
        )
    )

    assert config == {
        "sr": 24000,
        "T_ref": 256,
        "T_chunk": 934,
        "upsample_rate": 256,
        "overlapped_len": 12,
    }


def test_decode_vocoder_with_prompt_context_uses_runtime_vocoder_helper(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._run_vits_vocoder_feature_decode = Mock(  # type: ignore[method-assign]
        return_value=(torch.ones((1, 100, 4), dtype=torch.float32), torch.ones((1, 2, 1), dtype=torch.float32))
    )
    runtime._run_vocoder_module = Mock(return_value=torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32))  # type: ignore[method-assign]
    vits_model = SimpleNamespace(
        cfm=SimpleNamespace(inference=Mock(return_value=torch.ones((1, 100, 6), dtype=torch.float32))),
    )
    pipeline = SimpleNamespace(
        configs=SimpleNamespace(device="cpu"),
        vits_model=vits_model,
        vocoder="vocoder-module",
    )
    prompt_context = {
        "refer_audio_spec": torch.ones((1, 704, 4), dtype=torch.float32),
        "ge": torch.ones((1, 2, 1), dtype=torch.float32),
        "fea_ref": torch.ones((1, 100, 2), dtype=torch.float32),
        "mel2": torch.ones((1, 100, 2), dtype=torch.float32),
        "t_min": 2,
        "chunk_len": 4,
        "output_sr": 24000,
    }

    audio_fragment, output_sr = runtime._decode_vocoder_with_prompt_context(
        pipeline,
        SimpleNamespace(
            semantic_tokens=torch.tensor([1, 2], dtype=torch.long),
            phones=torch.tensor([3, 4], dtype=torch.long),
            speed_factor=1.0,
            sample_steps=16,
        ),
        prompt_context,
    )

    runtime._run_vits_vocoder_feature_decode.assert_called_once()
    runtime._run_vocoder_module.assert_called_once()
    assert audio_fragment.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert output_sr == 24000


def test_decode_prepared_requests_grouped_vocoder_reuses_prompt_context_once(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = SimpleNamespace(configs=SimpleNamespace(device="cpu"))
    prompt_context = {"output_sr": 24000}
    runtime._build_refer_spec_from_prepared = Mock(  # type: ignore[method-assign]
        side_effect=lambda prepared: GPTSoVITSReferSpec(prepared.refer_audio_spec, None)
    )
    runtime._build_vocoder_prompt_context = Mock(return_value=prompt_context)  # type: ignore[method-assign]
    runtime._decode_vocoder_with_prompt_context = Mock(  # type: ignore[method-assign]
        side_effect=[
            (torch.tensor([0.1, 0.2], dtype=torch.float32), 24000),
            (torch.tensor([0.3], dtype=torch.float32), 24000),
        ]
    )

    decoded = runtime._decode_prepared_requests_grouped_vocoder(
        pipeline,
        [
            SimpleNamespace(
                request_id="a",
                refer_audio_spec=torch.ones((1, 10, 4), dtype=torch.float32),
                speed_factor=1.0,
                sample_steps=16,
                super_sampling=False,
            ),
            SimpleNamespace(
                request_id="b",
                refer_audio_spec=torch.ones((1, 10, 4), dtype=torch.float32),
                speed_factor=1.3,
                sample_steps=16,
                super_sampling=True,
            ),
        ],
    )

    runtime._build_vocoder_prompt_context.assert_called_once_with(
        pipeline,
        ANY,
        ANY,
    )
    assert runtime._decode_vocoder_with_prompt_context.call_count == 2
    assert all(call.args[2] is prompt_context for call in runtime._decode_vocoder_with_prompt_context.call_args_list)
    assert [item.request_id for item in decoded] == ["a", "b"]
    assert decoded[0].audio_fragment.tolist() == pytest.approx([0.1, 0.2])
    assert decoded[1].audio_fragment.tolist() == pytest.approx([0.3])
    assert decoded[1].super_sampling is True


def test_decode_prepared_requests_grouped_vocoder_batches_same_target_config(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = SimpleNamespace(configs=SimpleNamespace(device="cpu"))
    prompt_context = {"output_sr": 24000}
    runtime._build_refer_spec_from_prepared = Mock(  # type: ignore[method-assign]
        side_effect=lambda prepared: GPTSoVITSReferSpec(prepared.refer_audio_spec, None)
    )
    runtime._build_vocoder_prompt_context = Mock(return_value=prompt_context)  # type: ignore[method-assign]
    runtime._decode_prepared_requests_batched_vocoder = Mock(  # type: ignore[method-assign]
        return_value=[
            GPTSoVITSDecodedAudio("a", "audio-a", 24000, 1.0, 0.3, False),
            GPTSoVITSDecodedAudio("b", "audio-b", 24000, 1.0, 0.3, True),
        ]
    )
    runtime._decode_vocoder_with_prompt_context = Mock(  # type: ignore[method-assign]
        return_value=("audio-c", 24000)
    )

    decoded = runtime._decode_prepared_requests_grouped_vocoder(
        pipeline,
        [
            SimpleNamespace(
                request_id="a",
                refer_audio_spec=torch.ones((1, 10, 4), dtype=torch.float32),
                speed_factor=1.0,
                sample_steps=16,
                super_sampling=False,
            ),
            SimpleNamespace(
                request_id="b",
                refer_audio_spec=torch.ones((1, 10, 4), dtype=torch.float32),
                speed_factor=1.0,
                sample_steps=16,
                super_sampling=True,
            ),
            SimpleNamespace(
                request_id="c",
                refer_audio_spec=torch.ones((1, 10, 4), dtype=torch.float32),
                speed_factor=1.2,
                sample_steps=16,
                super_sampling=False,
            ),
        ],
    )

    runtime._build_vocoder_prompt_context.assert_called_once_with(pipeline, ANY, ANY)
    runtime._decode_prepared_requests_batched_vocoder.assert_called_once()
    grouped_prepared = runtime._decode_prepared_requests_batched_vocoder.call_args.args[1]
    assert [item.request_id for item in grouped_prepared] == ["a", "b"]
    runtime._decode_vocoder_with_prompt_context.assert_called_once_with(pipeline, ANY, prompt_context)
    assert [item.request_id for item in decoded] == ["a", "b", "c"]
    assert [item.audio_fragment for item in decoded] == ["audio-a", "audio-b", "audio-c"]


def test_decode_prepared_requests_batched_vocoder_uses_native_components(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._sola_merge_audio_fragments = Mock(  # type: ignore[method-assign]
        return_value=torch.tensor([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=torch.float32)
    )
    runtime._get_text_frontend_symbol = Mock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            sequence_mask=lambda lengths, max_len: (
                torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
            )
        )
    )
    ge = torch.ones((1, 2, 1), dtype=torch.float32)
    vits_model = SimpleNamespace(
        version="v2",
        semantic_frame_rate="50hz",
        ref_enc=Mock(return_value=ge),
        quantizer=SimpleNamespace(
            decode=Mock(
                side_effect=[
                    torch.ones((1, 1, 2), dtype=torch.float32),
                    torch.ones((1, 1, 1), dtype=torch.float32) * 2,
                ]
            )
        ),
        enc_p=Mock(
            side_effect=[
                (
                    torch.ones((1, 50, 2), dtype=torch.float32),
                    None,
                    None,
                    torch.ones((1, 1, 2), dtype=torch.float32),
                    None,
                    None,
                ),
                (
                    torch.ones((1, 50, 1), dtype=torch.float32),
                    None,
                    None,
                    torch.ones((1, 1, 1), dtype=torch.float32),
                    None,
                    None,
                ),
            ]
        ),
        bridge=Mock(
            side_effect=[
                torch.ones((1, 100, 2), dtype=torch.float32),
                torch.ones((1, 100, 1), dtype=torch.float32) * 2,
            ]
        ),
        wns1=Mock(
            side_effect=[
                (torch.ones((1, 100, 3), dtype=torch.float32), torch.ones((1, 1, 3), dtype=torch.float32)),
                (torch.ones((1, 100, 2), dtype=torch.float32) * 2, torch.ones((1, 1, 2), dtype=torch.float32)),
            ]
        ),
        decode_encp=Mock(side_effect=AssertionError("should not call decode_encp wrapper")),
        cfm=SimpleNamespace(
            inference=Mock(return_value=torch.ones((2, 100, 6), dtype=torch.float32)),
        ),
    )
    pipeline = SimpleNamespace(
        configs=SimpleNamespace(device="cpu"),
        vocoder_configs={"overlapped_len": 1, "upsample_rate": 1},
        vits_model=vits_model,
        vocoder=Mock(return_value=torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]], dtype=torch.float32)),
    )
    prompt_context = {
        "refer_audio_spec": torch.ones((1, 704, 4), dtype=torch.float32),
        "ge": ge,
        "fea_ref": torch.ones((1, 100, 2), dtype=torch.float32),
        "mel2": torch.ones((1, 100, 2), dtype=torch.float32),
        "chunk_len": 4,
        "output_sr": 24000,
        "vocoder_config": {"overlapped_len": 1, "upsample_rate": 1},
    }

    decoded = runtime._decode_prepared_requests_batched_vocoder(
        pipeline,
        [
            SimpleNamespace(
                request_id="a",
                semantic_tokens=torch.tensor([1, 2], dtype=torch.long),
                phones=torch.tensor([3, 4], dtype=torch.long),
                speed_factor=1.0,
                sample_steps=16,
                super_sampling=False,
            ),
            SimpleNamespace(
                request_id="b",
                semantic_tokens=torch.tensor([5], dtype=torch.long),
                phones=torch.tensor([6], dtype=torch.long),
                speed_factor=1.0,
                sample_steps=16,
                super_sampling=True,
            ),
        ],
        prompt_context,
    )

    assert vits_model.quantizer.decode.call_count == 2
    assert vits_model.enc_p.call_count == 2
    assert vits_model.bridge.call_count == 2
    assert vits_model.wns1.call_count == 2
    vits_model.decode_encp.assert_not_called()
    vits_model.cfm.inference.assert_called_once()
    pipeline.vocoder.assert_called_once()
    runtime._sola_merge_audio_fragments.assert_called_once()
    assert [item.request_id for item in decoded] == ["a", "b"]
    assert decoded[0].audio_fragment.tolist() == pytest.approx([10.0, 11.0, 12.0])
    assert decoded[1].audio_fragment.tolist() == pytest.approx([13.0, 14.0])
    assert decoded[1].super_sampling is True


def test_prepare_request_spec_cpu_stage_uses_prepare_coordinator_and_ref_audio_preload(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    spec = SimpleNamespace(request_id="req-1", ref_audio_path="/tmp/ref.wav")
    prompt_cpu_profiled = SimpleNamespace()
    target_cpu_profiled = SimpleNamespace()
    native_cpu_stage = GPTSoVITSNativePreparedCpuStage(
        spec=spec,
        prepare_submit_at=1.0,
        prepare_start=2.0,
        prompt_text="prompt",
        text="text",
        prepare_admission_wait_ms=3.0,
        current_inflight=4,
        peak_inflight=5,
        prompt_cpu_profiled=prompt_cpu_profiled,
        target_cpu_profiled=target_cpu_profiled,
    )
    coordinator = SimpleNamespace()
    runtime._ensure_prepare_coordinator = Mock(return_value=coordinator)  # type: ignore[method-assign]
    runtime.preload_ref_audio_asset = Mock(return_value="future")  # type: ignore[method-assign]
    runtime._prepare_cpu_stage_async = Mock(return_value="awaitable")  # type: ignore[method-assign]
    runtime._run_awaitable_sync = Mock(return_value=native_cpu_stage)  # type: ignore[method-assign]

    prepared = runtime.prepare_request_spec_cpu_stage(spec)

    runtime.preload_ref_audio_asset.assert_called_once_with("/tmp/ref.wav", submit_at=ANY)
    runtime._prepare_cpu_stage_async.assert_called_once()
    assert prepared.request_id == "req-1"
    assert prepared.spec is spec
    assert prepared.prompt_text == "prompt"
    assert prepared.text == "text"
    assert prepared.current_inflight == 4
    assert prepared.prompt_cpu_profiled is prompt_cpu_profiled
    assert prepared.ref_audio_prepare_future == "future"


def test_prepare_request_cpu_stage_builds_spec_and_delegates_to_spec_path(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    spec = SimpleNamespace(request_id="req-1")
    prepared_cpu_stage = GPTSoVITSPreparedCpuStage(
        request_id="req-1",
        spec=spec,
        prepare_submit_at=1.0,
        prepare_start=2.0,
        prompt_text="prompt",
        text="text",
        prepare_admission_wait_ms=3.0,
        current_inflight=4,
        peak_inflight=5,
        prompt_cpu_profiled=SimpleNamespace(),
        target_cpu_profiled=SimpleNamespace(),
        ref_audio_prepare_future="future",
    )
    runtime._build_scheduler_request_spec = Mock(return_value=spec)  # type: ignore[method-assign]
    runtime.prepare_request_spec_cpu_stage = Mock(return_value=prepared_cpu_stage)  # type: ignore[method-assign]

    prepared = runtime.prepare_request_cpu_stage({"text": "hello"}, request_id="req-1")

    runtime._build_scheduler_request_spec.assert_called_once_with({"text": "hello"}, request_id="req-1")
    runtime.prepare_request_spec_cpu_stage.assert_called_once_with(spec, preload_ref_audio=True)
    assert prepared.spec is spec


def test_build_scheduler_request_spec_returns_runtime_native_spec(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._ensure_pipeline = Mock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(configs=SimpleNamespace(hz=50, max_sec=30))
    )
    runtime.build_tts_inputs = Mock(  # type: ignore[method-assign]
        return_value={
            "ref_audio_path": "/tmp/ref.wav",
            "prompt_text": "prompt",
            "prompt_lang": "zh",
            "text": "text",
            "text_lang": "zh",
            "text_split_method": "cut5",
            "top_k": 5,
            "top_p": 0.9,
            "temperature": 1.1,
            "repetition_penalty": 1.2,
            "aux_ref_audio_paths": ["/tmp/a.wav", "/tmp/b.wav"],
            "speed_factor": 1.4,
            "sample_steps": 48,
            "super_sampling": True,
        }
    )

    spec = runtime._build_scheduler_request_spec({"ready_step": 7}, request_id="req-native-spec")

    assert isinstance(spec, GPTSoVITSRequestSpec)
    assert spec.request_id == "req-native-spec"
    assert spec.ref_audio_path == "/tmp/ref.wav"
    assert spec.prompt_text == "prompt"
    assert spec.text == "text"
    assert spec.text_split_method == "cut5"
    assert spec.top_k == 5
    assert spec.top_p == pytest.approx(0.9)
    assert spec.temperature == pytest.approx(1.1)
    assert spec.repetition_penalty == pytest.approx(1.2)
    assert spec.early_stop_num == 1500
    assert spec.aux_ref_audio_paths == ["/tmp/a.wav", "/tmp/b.wav"]
    assert spec.speed_factor == pytest.approx(1.4)
    assert spec.sample_steps == 48
    assert spec.super_sampling is True
    assert spec.ready_step == 7


def test_build_tts_inputs_accepts_speed_alias_and_super_sampling(tmp_path):
    wav_path = tmp_path / "ref.wav"
    sf.write(wav_path, np.zeros(1600, dtype=np.float32), 16000)
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))

    inputs = runtime.build_tts_inputs(
        {
            "text": "今天风很轻。",
            "text_lang": "zh",
            "ref_audio_path": str(wav_path),
            "prompt_text": "朝阳下的朝圣者。",
            "prompt_lang": "zh",
            "speed": 1.25,
            "sample_steps": 40,
            "super_sampling": True,
        }
    )

    assert inputs["speed_factor"] == pytest.approx(1.25)
    assert inputs["sample_steps"] == 40
    assert inputs["super_sampling"] is True


def test_ensure_prepare_coordinator_builds_runtime_native_coordinator(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = SimpleNamespace(
        prepare_bert_batch_worker=None,
        prepare_text_cpu_workers=3,
        _vllm_runtime_prepare_state_provider=Mock(return_value={"g2pw": {"worker_count": 2}}),
        snapshot_prepare_runtime_components=Mock(side_effect=AssertionError("legacy snapshot should not be used")),
    )
    runtime._ensure_pipeline = Mock(return_value=pipeline)  # type: ignore[method-assign]

    coordinator = runtime._ensure_prepare_coordinator()

    assert isinstance(coordinator, GPTSoVITSPrepareRuntimeCoordinator)
    assert coordinator.tts is pipeline
    assert coordinator.g2pw_executor is not None
    assert coordinator.ref_audio_executor is not None
    assert coordinator.text_cpu_gate is not None
    assert coordinator.inflight_gate is not None
    assert coordinator.g2pw_gate.snapshot()["max_inflight"] == pytest.approx(2.0)
    pipeline._vllm_runtime_prepare_state_provider.assert_called_once_with()
    pipeline.snapshot_prepare_runtime_components.assert_not_called()
    if coordinator.text_feature_executor is not None:
        coordinator.text_feature_executor.shutdown(wait=True, cancel_futures=True)
    coordinator.g2pw_executor.shutdown(wait=True, cancel_futures=True)
    coordinator.ref_audio_executor.shutdown(wait=True, cancel_futures=True)


def test_preload_ref_audio_asset_wraps_runtime_future_into_runtime_native_types(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    source_future: concurrent.futures.Future = concurrent.futures.Future()
    runtime._prepare_coordinator = runtime._coerce_prepare_coordinator(
        SimpleNamespace(
            submit_prepare_ref_audio_asset=Mock(return_value=source_future),
        )
    )

    wrapped_future = runtime.preload_ref_audio_asset("/tmp/ref.wav", submit_at=1.25)
    source_future.set_result(
        SimpleNamespace(
            result=SimpleNamespace(
                raw_audio=torch.tensor([0.1], dtype=torch.float32),
                raw_sr=16000,
                wav16k=torch.tensor([0.2], dtype=torch.float32),
                profile={"prepared_ref_audio_cache_hit": 1.0},
            ),
            submit_at=1.25,
            started_at=1.5,
            finished_at=2.0,
            profile={"future_profile": 3.0},
        )
    )
    profiled = wrapped_future.result(timeout=1.0)

    assert isinstance(profiled, GPTSoVITSPrepareProfiledResult)
    assert isinstance(profiled.result, GPTSoVITSPreparedRefAudioAsset)
    assert profiled.result.raw_sr == 16000
    assert profiled.result.profile["prepared_ref_audio_cache_hit"] == 1.0
    assert profiled.profile == {"future_profile": 3.0}


def test_native_prepare_coordinator_ref_audio_preload_uses_cache(tmp_path):
    wav_path = tmp_path / "ref.wav"
    sf.write(wav_path, np.zeros(48000, dtype=np.float32), 16000)

    load_calls = []

    def _load_ref_audio_raw(path: str):
        load_calls.append(path)
        return torch.ones((1, 48000), dtype=torch.float32), 16000

    pipeline = SimpleNamespace(
        prepare_bert_batch_worker=None,
        prepare_text_cpu_workers=2,
        configs=SimpleNamespace(sampling_rate=32000),
        prepare_ref_audio_cpu_limiter=None,
        _vllm_runtime_prepare_state_provider=Mock(return_value={"g2pw": {"worker_count": 1}}),
        snapshot_prepare_runtime_components=Mock(side_effect=AssertionError("legacy snapshot should not be used")),
        _load_ref_audio_raw=_load_ref_audio_raw,
        _prepare_prompt_semantic_wav16k_profile=Mock(
            side_effect=AssertionError("should not call tts preload wav16k helper")
        ),
    )
    coordinator = GPTSoVITSPrepareRuntimeCoordinator.build_native(pipeline)

    try:
        first = coordinator.submit_prepare_ref_audio_asset(str(wav_path), submit_at=1.0).result(timeout=2.0)
        second = coordinator.submit_prepare_ref_audio_asset(str(wav_path), submit_at=2.0).result(timeout=2.0)
    finally:
        if coordinator.text_feature_executor is not None:
            coordinator.text_feature_executor.shutdown(wait=True, cancel_futures=True)
        if coordinator.g2pw_executor is not None:
            coordinator.g2pw_executor.shutdown(wait=True, cancel_futures=True)
        if coordinator.ref_audio_executor is not None:
            coordinator.ref_audio_executor.shutdown(wait=True, cancel_futures=True)

    assert len(load_calls) == 1
    pipeline._vllm_runtime_prepare_state_provider.assert_called_once_with()
    pipeline.snapshot_prepare_runtime_components.assert_not_called()
    pipeline._prepare_prompt_semantic_wav16k_profile.assert_not_called()
    assert isinstance(first, GPTSoVITSPrepareProfiledResult)
    assert isinstance(first.result, GPTSoVITSPreparedRefAudioAsset)
    assert first.result.profile["prepared_ref_audio_cache_hit"] == 0.0
    assert second.result.profile["prepared_ref_audio_cache_hit"] == 1.0
    assert second.result.profile["prepared_ref_audio_cache_age_ms"] >= 0.0


def test_prepare_cpu_stage_async_uses_runtime_native_text_cpu_pair(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    prompt_cpu_profiled = GPTSoVITSPrepareProfiledResult(result=["prompt"], submit_at=1.0, started_at=1.0, finished_at=2.0)
    target_cpu_profiled = GPTSoVITSPrepareProfiledResult(result=["target"], submit_at=1.5, started_at=1.5, finished_at=2.5)
    coordinator = SimpleNamespace(
        _inflight_gate=SimpleNamespace(acquire=AsyncMock(return_value={"wait_ms": 0.5})),
        _mark_enter=Mock(return_value=(2, 4)),
        _release_split_stage_slot=Mock(),
    )
    runtime._normalize_prepare_sentence = Mock(return_value="normalized prompt")  # type: ignore[method-assign]
    runtime._run_text_cpu_stage_pair = AsyncMock(return_value=(prompt_cpu_profiled, target_cpu_profiled))  # type: ignore[method-assign]

    stage = runtime._run_awaitable_sync(
        runtime._prepare_cpu_stage_async(
            coordinator,
            SimpleNamespace(prompt_text="prompt", prompt_lang="zh", text="target\n", text_lang="zh"),
            prepare_submit_at=11.0,
        )
    )

    runtime._run_text_cpu_stage_pair.assert_awaited_once_with(
        ANY,
        "normalized prompt",
        "zh",
        "target",
        "zh",
    )
    forwarded_coordinator = runtime._run_text_cpu_stage_pair.await_args.args[0]
    assert isinstance(forwarded_coordinator, GPTSoVITSPrepareRuntimeCoordinator)
    assert forwarded_coordinator.mark_prepare_enter() == (2, 4)
    assert isinstance(stage, GPTSoVITSNativePreparedCpuStage)
    assert stage.prompt_text == "normalized prompt"
    assert stage.text == "target"
    assert stage.current_inflight == 2
    assert stage.peak_inflight == 4
    coordinator._release_split_stage_slot.assert_not_called()


def test_prepare_request_split_phase_helpers_build_state_from_explicit_phases(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    coordinator = SimpleNamespace(_release_split_stage_slot=Mock())
    runtime._ensure_prepare_coordinator = Mock(return_value=coordinator)  # type: ignore[method-assign]
    audio_phase = GPTSoVITSPrepareAudioPhaseData(
        prompt_g2pw_profiled="prompt-g2pw",
        target_g2pw_profiled="target-g2pw",
        ref_audio_profiled="ref-audio",
        g2pw_pair_ms=1.0,
        phase_wall_ms=2.0,
    )
    ref_spec_result = GPTSoVITSPrepareRefSpecResult(
        refer_spec=GPTSoVITSReferSpec("spec", "wav16k"),
        profile={"ref_spec_ms": 1.0},
    )
    text_phase = GPTSoVITSPrepareTextPhaseData(
        prompt_feature_profiled="prompt-feature",
        target_feature_profiled="target-feature",
        phase_wall_ms=3.0,
    )
    runtime._prepare_gpu_audio_phase_async = AsyncMock(return_value=audio_phase)  # type: ignore[method-assign]
    runtime._prepare_ref_spec_phase_async = AsyncMock(return_value=ref_spec_result)  # type: ignore[method-assign]
    runtime._prepare_gpu_text_phase_async = AsyncMock(return_value=text_phase)  # type: ignore[method-assign]
    runtime._build_request_state_from_prepare_phases = Mock(return_value=SimpleNamespace(state="ok"))  # type: ignore[method-assign]
    runtime._state_to_transport_info = Mock(return_value={"transport": "ok"})  # type: ignore[method-assign]

    prepared_cpu = GPTSoVITSPreparedCpuStage(
        request_id="req-2",
        spec=SimpleNamespace(request_id="req-2", ref_audio_path="/tmp/ref.wav"),
        prepare_submit_at=0.0,
        prepare_start=0.0,
        prompt_text="prompt",
        text="text",
        prepare_admission_wait_ms=0.0,
        current_inflight=0,
        peak_inflight=0,
        prompt_cpu_profiled=SimpleNamespace(),
        target_cpu_profiled=SimpleNamespace(),
        ref_audio_prepare_future="future",
    )

    prepared_audio = runtime.prepare_request_gpu_audio_phase(prepared_cpu)
    prepared_ref_spec = runtime.prepare_request_ref_spec_phase(prepared_audio)
    prepared_text = runtime.prepare_request_gpu_text_phase(prepared_audio)
    prepared_request = runtime.build_prepared_request_from_phases(
        prepared_text,
        prepared_ref_spec_phase=prepared_ref_spec,
    )

    assert isinstance(prepared_audio, GPTSoVITSPreparedAudioPhase)
    assert prepared_audio.phase_one is audio_phase
    assert isinstance(prepared_ref_spec, GPTSoVITSPreparedRefSpecPhase)
    assert prepared_ref_spec.ref_spec_result is ref_spec_result
    assert isinstance(prepared_text, GPTSoVITSPreparedTextPhase)
    assert prepared_text.phase_two is text_phase
    runtime._build_request_state_from_prepare_phases.assert_called_once_with(
        prepared_cpu,
        audio_phase,
        text_phase,
        ref_spec_result=ref_spec_result,
        extra_profile=None,
    )
    coordinator._release_split_stage_slot.assert_called_once_with()
    runtime._state_to_transport_info.assert_called_once_with(prepared_request.state, prepared_cpu.spec)
    assert prepared_request.request_id == "req-2"
    assert prepared_request.transport_info == {"transport": "ok"}


def test_prepare_direct_shared_segments_async_uses_runtime_native_shared_prompt_and_ref(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    coordinator = SimpleNamespace()
    spec_a = SimpleNamespace(
        request_id="req-a",
        prompt_text="shared prompt",
        prompt_lang="zh",
        text="text-a",
        text_lang="zh",
        ref_audio_path="/tmp/ref.wav",
    )
    spec_b = SimpleNamespace(
        request_id="req-b",
        prompt_text="shared prompt",
        prompt_lang="zh",
        text="text-b",
        text_lang="zh",
        ref_audio_path="/tmp/ref.wav",
    )
    shared_prompt_cpu = GPTSoVITSPrepareProfiledResult(
        result=["prompt-cpu"],
        submit_at=1.0,
        started_at=1.0,
        finished_at=2.0,
        profile={},
    )
    target_cpu_a = GPTSoVITSPrepareProfiledResult(
        result=["target-a"],
        submit_at=2.0,
        started_at=2.0,
        finished_at=3.0,
        profile={},
    )
    target_cpu_b = GPTSoVITSPrepareProfiledResult(
        result=["target-b"],
        submit_at=3.0,
        started_at=3.0,
        finished_at=4.0,
        profile={},
    )
    shared_prompt_g2pw = GPTSoVITSPrepareProfiledResult(
        result=["prompt-g2pw"],
        submit_at=4.0,
        started_at=4.0,
        finished_at=5.0,
        profile={"prompt_g2pw_ms": 1.0},
    )
    target_g2pw_a = GPTSoVITSPrepareProfiledResult(
        result=["target-g2pw-a"],
        submit_at=5.0,
        started_at=5.0,
        finished_at=6.0,
        profile={"target_g2pw_ms": 1.0},
    )
    target_g2pw_b = GPTSoVITSPrepareProfiledResult(
        result=["target-g2pw-b"],
        submit_at=6.0,
        started_at=6.0,
        finished_at=7.0,
        profile={"target_g2pw_ms": 1.0},
    )
    shared_prompt_feature = GPTSoVITSPrepareProfiledResult(
        result="prompt-feature",
        submit_at=7.0,
        started_at=7.0,
        finished_at=8.0,
        profile={"bert_total_ms": 1.0},
    )
    target_feature_a = GPTSoVITSPrepareProfiledResult(
        result="target-feature-a",
        submit_at=8.0,
        started_at=8.0,
        finished_at=9.0,
        profile={"bert_total_ms": 1.0},
    )
    target_feature_b = GPTSoVITSPrepareProfiledResult(
        result="target-feature-b",
        submit_at=9.0,
        started_at=9.0,
        finished_at=10.0,
        profile={"bert_total_ms": 1.0},
    )
    shared_ref_audio = GPTSoVITSPrepareProfiledResult(
        result=GPTSoVITSRefAudioBundle(
            prompt_semantic="prompt-semantic",
            raw_audio="raw-audio",
            raw_sr=16000,
            profile={"bundle_total_ms": 1.0},
        ),
        submit_at=10.0,
        started_at=10.0,
        finished_at=11.0,
        profile={},
    )
    shared_ref_spec = GPTSoVITSPrepareProfiledResult(
        result=(GPTSoVITSReferSpec("spec-audio", "audio-16k"), {"ref_spec_to_device_ms": 1.0}),
        submit_at=11.0,
        started_at=11.0,
        finished_at=12.0,
        profile={},
    )
    runtime._normalize_prepare_sentence = Mock(return_value="shared prompt")  # type: ignore[method-assign]
    runtime._run_text_cpu_stage = AsyncMock(  # type: ignore[method-assign]
        side_effect=[shared_prompt_cpu, target_cpu_a, target_cpu_b]
    )
    runtime._run_g2pw_stage = AsyncMock(  # type: ignore[method-assign]
        side_effect=[shared_prompt_g2pw, target_g2pw_a, target_g2pw_b]
    )
    runtime._run_text_feature_stage = AsyncMock(  # type: ignore[method-assign]
        side_effect=[shared_prompt_feature, target_feature_a, target_feature_b]
    )
    runtime.preload_ref_audio_asset = Mock(return_value="shared-future")  # type: ignore[method-assign]
    runtime._run_ref_prompt_semantic_stage = AsyncMock(return_value=shared_ref_audio)  # type: ignore[method-assign]
    runtime._run_ref_spec_stage = AsyncMock(return_value=shared_ref_spec)  # type: ignore[method-assign]
    runtime._build_request_state_from_prepare_phases = Mock(  # type: ignore[method-assign]
        side_effect=[SimpleNamespace(state="state-a"), SimpleNamespace(state="state-b")]
    )
    runtime._release_prepare_split_stage_slot = Mock()  # type: ignore[method-assign]

    outputs = runtime._run_awaitable_sync(
        runtime._prepare_direct_shared_segments_async(coordinator, [spec_a, spec_b])
    )

    runtime._normalize_prepare_sentence.assert_called_once_with("shared prompt", "zh")
    assert runtime._run_text_cpu_stage.await_count == 3
    runtime.preload_ref_audio_asset.assert_called_once_with("/tmp/ref.wav", submit_at=ANY)
    runtime._run_ref_prompt_semantic_stage.assert_awaited_once_with(
        ANY,
        "/tmp/ref.wav",
        prepared_asset_future="shared-future",
    )
    runtime._run_ref_spec_stage.assert_awaited_once_with(ANY, "raw-audio", 16000)
    assert runtime._build_request_state_from_prepare_phases.call_count == 2
    assert runtime._release_prepare_split_stage_slot.call_count == 2
    assert [result[0].state for result in outputs] == ["state-a", "state-b"]


def test_build_request_state_from_prepare_phases_uses_runtime_native_state_assembly(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = SimpleNamespace(
        configs=SimpleNamespace(device="cpu"),
        precision=torch.float16,
        prepare_text_cpu_workers=3,
    )
    runtime._ensure_pipeline = Mock(return_value=pipeline)  # type: ignore[method-assign]
    runtime._build_prepare_profile_overrides = Mock(return_value={"custom_metric": 7.0})  # type: ignore[method-assign]
    runtime._build_ref_audio_bundle_from_phase = Mock(  # type: ignore[method-assign]
        return_value=GPTSoVITSRefAudioBundle(
            prompt_semantic=torch.tensor([5, 6], dtype=torch.int32),
            refer_spec=GPTSoVITSReferSpec(
                torch.ones((1, 5, 4), dtype=torch.float32),
                torch.ones((1, 160), dtype=torch.float32),
            ),
            raw_audio=torch.ones((1, 12), dtype=torch.float32),
            raw_sr=16000,
            profile={
                "bundle_total_ms": 4.0,
                "prompt_semantic_ms": 1.5,
                "ref_spec_ms": 2.0,
                "audio_load_ms": 0.5,
            },
        )
    )
    runtime._extract_ref_spec_native = Mock(  # type: ignore[method-assign]
        return_value=(GPTSoVITSReferSpec("aux-spec", "aux-16k"), torch.ones((1, 8), dtype=torch.float32), 16000)
    )

    prompt_result = SimpleNamespace(
        phones=[1, 2],
        bert_features=torch.ones((4, 2), dtype=torch.float32),
        norm_text="prompt-norm",
        total_ms=11.0,
        cpu_preprocess_ms=2.0,
        profile={"bert_wait_ms": 1.0},
    )
    target_result = SimpleNamespace(
        phones=[3, 4, 5],
        bert_features=torch.ones((4, 3), dtype=torch.float32) * 2,
        norm_text="target-norm",
        total_ms=13.0,
        cpu_preprocess_ms=3.0,
        profile={"bert_wait_ms": 2.0},
    )

    prepared_cpu_stage = GPTSoVITSPreparedCpuStage(
        request_id="req-native",
        spec=SimpleNamespace(
            request_id="req-native",
            ref_audio_path="/tmp/ref.wav",
            prompt_lang="zh",
            text_lang="zh",
            top_k=5,
            top_p=0.8,
            temperature=1.0,
            repetition_penalty=1.1,
            early_stop_num=123,
            ready_step=4,
            aux_ref_audio_paths=[str(tmp_path / "aux.wav"), str(tmp_path / "missing.wav")],
        ),
        prepare_submit_at=0.0,
        prepare_start=time.time() - 0.01,
        prompt_text="prompt",
        text="target",
        prepare_admission_wait_ms=0.0,
        current_inflight=0,
        peak_inflight=0,
        prompt_cpu_profiled=SimpleNamespace(),
        target_cpu_profiled=SimpleNamespace(),
    )
    (tmp_path / "aux.wav").write_bytes(b"wav")

    state = runtime._build_request_state_from_prepare_phases(
        prepared_cpu_stage,
        GPTSoVITSPrepareAudioPhaseData(
            prompt_g2pw_profiled=SimpleNamespace(),
            target_g2pw_profiled=SimpleNamespace(),
            ref_audio_profiled=SimpleNamespace(),
        ),
        GPTSoVITSPrepareTextPhaseData(
            prompt_feature_profiled=SimpleNamespace(result=prompt_result),
            target_feature_profiled=SimpleNamespace(result=target_result),
        ),
    )

    assert state.request_id == "req-native"
    assert state.norm_prompt_text == "prompt-norm"
    assert state.norm_text == "target-norm"
    assert state.phones.tolist() == [3, 4, 5]
    assert state.prompt_phones.tolist() == [1, 2]
    assert state.all_phones.tolist() == [1, 2, 3, 4, 5]
    assert state.prompt_semantic.dtype == torch.long
    assert state.refer_spec is not None
    assert state.refer_spec.spec_audio.shape == (1, 5, 4)
    assert state.raw_sr == 16000
    assert state.top_k == 5
    assert state.ready_step == 4
    assert state.prepare_profile["custom_metric"] == 7.0
    assert "executor_run_wall_ms" in state.prepare_profile
    assert state.all_bert_features.dtype == torch.float16
    assert len(state.aux_refer_specs) == 1
    runtime._extract_ref_spec_native.assert_called_once_with(str(tmp_path / "aux.wav"))


def test_prepare_gpu_audio_phase_async_uses_runtime_native_prepare_kernels(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    prompt_g2pw = GPTSoVITSPrepareProfiledResult(result=["prompt-g2pw"], submit_at=1.0, started_at=1.0, finished_at=2.0)
    target_g2pw = GPTSoVITSPrepareProfiledResult(result=["target-g2pw"], submit_at=1.0, started_at=1.0, finished_at=2.5)
    ref_audio = GPTSoVITSPrepareProfiledResult(
        result=GPTSoVITSRefAudioBundle(
            prompt_semantic=torch.tensor([1, 2], dtype=torch.long),
            raw_audio=torch.tensor([0.1]),
            raw_sr=16000,
            profile={},
        ),
        submit_at=1.0,
        started_at=1.0,
        finished_at=3.0,
    )
    runtime._run_g2pw_pair_stage = AsyncMock(return_value=(prompt_g2pw, target_g2pw))  # type: ignore[method-assign]
    runtime._run_ref_prompt_semantic_stage = AsyncMock(return_value=ref_audio)  # type: ignore[method-assign]

    prepared_cpu = GPTSoVITSPreparedCpuStage(
        request_id="req-audio",
        spec=SimpleNamespace(ref_audio_path="/tmp/ref.wav"),
        prepare_submit_at=0.0,
        prepare_start=0.0,
        prompt_text="prompt",
        text="target",
        prepare_admission_wait_ms=0.0,
        current_inflight=0,
        peak_inflight=0,
        prompt_cpu_profiled=SimpleNamespace(result=["prompt"]),
        target_cpu_profiled=SimpleNamespace(result=["target"]),
        ref_audio_prepare_future="future",
    )

    phase_one = runtime._run_awaitable_sync(runtime._prepare_gpu_audio_phase_async(SimpleNamespace(), prepared_cpu))

    runtime._run_g2pw_pair_stage.assert_awaited_once()
    runtime._run_ref_prompt_semantic_stage.assert_awaited_once_with(
        ANY,
        "/tmp/ref.wav",
        prepared_asset_future="future",
    )
    assert phase_one.prompt_g2pw_profiled is prompt_g2pw
    assert phase_one.target_g2pw_profiled is target_g2pw
    assert phase_one.ref_audio_profiled is ref_audio


def test_prepare_gpu_audio_phase_batch_async_uses_runtime_native_ref_audio_batch_helper(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    coordinator = SimpleNamespace(g2pw_audio_batch_merge_group_size=8)
    prompt_a = GPTSoVITSPrepareProfiledResult(result=["prompt-a"], submit_at=1.0, started_at=1.0, finished_at=2.0)
    target_a = GPTSoVITSPrepareProfiledResult(result=["target-a"], submit_at=1.0, started_at=1.0, finished_at=2.0)
    prompt_b = GPTSoVITSPrepareProfiledResult(result=["prompt-b"], submit_at=1.0, started_at=1.0, finished_at=2.0)
    target_b = GPTSoVITSPrepareProfiledResult(result=["target-b"], submit_at=1.0, started_at=1.0, finished_at=2.0)
    ref_a = GPTSoVITSPrepareProfiledResult(
        result=GPTSoVITSRefAudioBundle(
            prompt_semantic=torch.tensor([1], dtype=torch.long),
            raw_audio=torch.tensor([0.1]),
            raw_sr=16000,
            profile={},
        ),
        submit_at=1.0,
        started_at=1.0,
        finished_at=3.0,
    )
    ref_b = GPTSoVITSPrepareProfiledResult(
        result=GPTSoVITSRefAudioBundle(
            prompt_semantic=torch.tensor([2], dtype=torch.long),
            raw_audio=torch.tensor([0.2]),
            raw_sr=16000,
            profile={},
        ),
        submit_at=1.0,
        started_at=1.0,
        finished_at=3.5,
    )
    runtime._run_g2pw_pair_stage_batch = AsyncMock(return_value=[(prompt_a, target_a), (prompt_b, target_b)])  # type: ignore[method-assign]
    runtime._run_ref_prompt_semantic_stage_batch = AsyncMock(return_value=[ref_a, ref_b])  # type: ignore[method-assign]
    runtime._run_ref_prompt_semantic_stage = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError("should not call per-request ref audio helper")
    )

    prepared_cpu_stages = [
        GPTSoVITSPreparedCpuStage(
            request_id="req-a",
            spec=SimpleNamespace(ref_audio_path="/tmp/a.wav"),
            prepare_submit_at=0.0,
            prepare_start=0.0,
            prompt_text="prompt-a",
            text="target-a",
            prepare_admission_wait_ms=0.0,
            current_inflight=0,
            peak_inflight=0,
            prompt_cpu_profiled=SimpleNamespace(result=["prompt-a"]),
            target_cpu_profiled=SimpleNamespace(result=["target-a"]),
            ref_audio_prepare_future="future-a",
        ),
        GPTSoVITSPreparedCpuStage(
            request_id="req-b",
            spec=SimpleNamespace(ref_audio_path="/tmp/b.wav"),
            prepare_submit_at=0.0,
            prepare_start=0.0,
            prompt_text="prompt-b",
            text="target-b",
            prepare_admission_wait_ms=0.0,
            current_inflight=0,
            peak_inflight=0,
            prompt_cpu_profiled=SimpleNamespace(result=["prompt-b"]),
            target_cpu_profiled=SimpleNamespace(result=["target-b"]),
            ref_audio_prepare_future="future-b",
        ),
    ]

    outputs = runtime._run_awaitable_sync(runtime._prepare_gpu_audio_phase_batch_async(coordinator, prepared_cpu_stages))

    runtime._run_ref_prompt_semantic_stage_batch.assert_awaited_once_with(
        ANY,
        [
            ("/tmp/a.wav", "future-a", None),
            ("/tmp/b.wav", "future-b", None),
        ],
    )
    runtime._run_ref_prompt_semantic_stage.assert_not_awaited()
    assert len(outputs) == 2
    assert outputs[0].ref_audio_profiled is ref_a
    assert outputs[1].ref_audio_profiled is ref_b


def test_prepare_request_gpu_audio_phases_batch_merge_uses_runtime_native_batch_helper(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    coordinator = SimpleNamespace(enable_g2pw_audio_batch_merge=True)
    runtime._ensure_prepare_coordinator = Mock(return_value=coordinator)  # type: ignore[method-assign]
    phase_a = GPTSoVITSPrepareAudioPhaseData(
        prompt_g2pw_profiled="prompt-a",
        target_g2pw_profiled="target-a",
        ref_audio_profiled="ref-a",
    )
    phase_b = GPTSoVITSPrepareAudioPhaseData(
        prompt_g2pw_profiled="prompt-b",
        target_g2pw_profiled="target-b",
        ref_audio_profiled="ref-b",
    )
    runtime._prepare_gpu_audio_phase_batch_async = AsyncMock(return_value=[phase_a, phase_b])  # type: ignore[method-assign]

    prepared_audio = runtime.prepare_request_gpu_audio_phases(
        [
            GPTSoVITSPreparedCpuStage("req-a", SimpleNamespace(), 0.0, 0.0, "", "", 0.0, 0, 0, SimpleNamespace(), SimpleNamespace(), None),
            GPTSoVITSPreparedCpuStage("req-b", SimpleNamespace(), 0.0, 0.0, "", "", 0.0, 0, 0, SimpleNamespace(), SimpleNamespace(), None),
        ]
    )

    runtime._prepare_gpu_audio_phase_batch_async.assert_awaited_once_with(
        coordinator,
        ANY,
    )
    assert [item.phase_one for item in prepared_audio] == [phase_a, phase_b]


def test_prepare_request_spec_uses_explicit_split_prepare_pipeline(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    spec = SimpleNamespace(request_id="req-3")
    prepared_cpu = SimpleNamespace(name="cpu")
    prepared_audio = SimpleNamespace(name="audio")
    prepared_ref_spec = SimpleNamespace(name="ref_spec")
    prepared_text = SimpleNamespace(name="text")
    prepared_request = SimpleNamespace(name="prepared")
    runtime.prepare_request_spec_cpu_stage = Mock(return_value=prepared_cpu)  # type: ignore[method-assign]
    runtime.prepare_request_gpu_audio_phase = Mock(return_value=prepared_audio)  # type: ignore[method-assign]
    runtime.prepare_request_ref_spec_phase = Mock(return_value=prepared_ref_spec)  # type: ignore[method-assign]
    runtime.prepare_request_gpu_text_phase = Mock(return_value=prepared_text)  # type: ignore[method-assign]
    runtime.build_prepared_request_from_phases = Mock(return_value=prepared_request)  # type: ignore[method-assign]

    result = runtime.prepare_request_spec(spec)

    runtime.prepare_request_spec_cpu_stage.assert_called_once_with(spec)
    runtime.prepare_request_gpu_audio_phase.assert_called_once_with(prepared_cpu)
    runtime.prepare_request_ref_spec_phase.assert_called_once_with(prepared_audio)
    runtime.prepare_request_gpu_text_phase.assert_called_once_with(prepared_audio)
    runtime.build_prepared_request_from_phases.assert_called_once_with(
        prepared_text,
        prepared_ref_spec_phase=prepared_ref_spec,
    )
    assert result is prepared_request


def test_prepare_request_builds_spec_and_delegates_to_spec_pipeline(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    spec = SimpleNamespace(request_id="req-3")
    prepared_request = SimpleNamespace(name="prepared")
    runtime._build_scheduler_request_spec = Mock(return_value=spec)  # type: ignore[method-assign]
    runtime.prepare_request_spec = Mock(return_value=prepared_request)  # type: ignore[method-assign]

    result = runtime.prepare_request({"text": "hello"}, request_id="req-3")

    runtime._build_scheduler_request_spec.assert_called_once_with({"text": "hello"}, request_id="req-3")
    runtime.prepare_request_spec.assert_called_once_with(spec)
    assert result is prepared_request


def test_prepare_gpu_text_and_ref_spec_phases_use_runtime_native_helpers(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    coordinator = SimpleNamespace()
    runtime._run_text_feature_pair_stage = AsyncMock(  # type: ignore[method-assign]
        return_value=("prompt-feature", "target-feature")
    )
    runtime._run_ref_spec_stage = AsyncMock(  # type: ignore[method-assign]
        return_value=GPTSoVITSPrepareProfiledResult(
            result=(("spec", "wav16k"), {"ref_spec_to_device_ms": 1.0}),
            submit_at=1.0,
            started_at=1.5,
            finished_at=3.0,
        )
    )
    prepared_audio = GPTSoVITSPreparedAudioPhase(
        request_id="req-phase",
        prepared_cpu_stage=GPTSoVITSPreparedCpuStage(
            request_id="req-phase",
            spec=SimpleNamespace(),
            prepare_submit_at=0.0,
            prepare_start=0.0,
            prompt_text="prompt",
            text="text",
            prepare_admission_wait_ms=0.0,
            current_inflight=0,
            peak_inflight=0,
            prompt_cpu_profiled=SimpleNamespace(run_ms=5.0),
            target_cpu_profiled=SimpleNamespace(run_ms=6.0),
            ref_audio_prepare_future=None,
        ),
        phase_one=GPTSoVITSPrepareAudioPhaseData(
            prompt_g2pw_profiled=SimpleNamespace(result=["prompt"], profile={"a": 1.0}),
            target_g2pw_profiled=SimpleNamespace(result=["target"], profile={"b": 2.0}),
            ref_audio_profiled=GPTSoVITSPrepareProfiledResult(
                result=GPTSoVITSRefAudioBundle(
                    prompt_semantic=torch.tensor([3, 4], dtype=torch.long),
                    raw_audio=torch.tensor([0.2]),
                    raw_sr=16000,
                    profile={},
                ),
                submit_at=1.0,
                started_at=1.0,
                finished_at=1.0,
            ),
        ),
    )

    phase_two = runtime._run_awaitable_sync(runtime._prepare_gpu_text_phase_async(coordinator, prepared_audio))
    ref_spec = runtime._run_awaitable_sync(runtime._prepare_ref_spec_phase_async(coordinator, prepared_audio))

    runtime._run_text_feature_pair_stage.assert_awaited_once_with(
        ANY,
        ["prompt"],
        ["target"],
        5.0,
        6.0,
        prompt_base_profile={"a": 1.0},
        target_base_profile={"b": 2.0},
    )
    forwarded_coordinator = runtime._run_text_feature_pair_stage.await_args.args[0]
    assert isinstance(forwarded_coordinator, GPTSoVITSPrepareRuntimeCoordinator)
    runtime._run_ref_spec_stage.assert_awaited_once_with(ANY, ANY, 16000)
    assert isinstance(runtime._run_ref_spec_stage.await_args.args[0], GPTSoVITSPrepareRuntimeCoordinator)
    assert phase_two.prompt_feature_profiled == "prompt-feature"
    assert phase_two.target_feature_profiled == "target-feature"
    assert ref_spec.refer_spec.spec_audio == "spec"
    assert ref_spec.refer_spec.audio_16k == "wav16k"
    assert ref_spec.profile["ref_spec_wait_ms"] == 500.0
    assert ref_spec.profile["ref_spec_ms"] == 1500.0


def test_prepare_request_spec_cpu_stages_batches_cpu_prepare_calls(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    coordinator = SimpleNamespace()
    runtime._ensure_prepare_coordinator = Mock(return_value=coordinator)  # type: ignore[method-assign]
    spec_a = SimpleNamespace(request_id="req-a", ref_audio_path="/tmp/a.wav")
    spec_b = SimpleNamespace(request_id="req-b", ref_audio_path="/tmp/b.wav")
    runtime.preload_ref_audio_asset = Mock(side_effect=["future-a", "future-b"])  # type: ignore[method-assign]
    runtime._prepare_cpu_stage_async = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            GPTSoVITSNativePreparedCpuStage(spec_a, 1.0, 2.0, "pa", "ta", 0.1, 1, 2, SimpleNamespace(), SimpleNamespace()),
            GPTSoVITSNativePreparedCpuStage(spec_b, 1.5, 2.5, "pb", "tb", 0.2, 3, 4, SimpleNamespace(), SimpleNamespace()),
        ]
    )

    prepared = runtime.prepare_request_spec_cpu_stages([spec_a, spec_b])

    assert len(prepared) == 2
    assert prepared[0].request_id == "req-a"
    assert prepared[0].prompt_text == "pa"
    assert prepared[0].ref_audio_prepare_future == "future-a"
    assert prepared[1].request_id == "req-b"
    assert prepared[1].prompt_text == "pb"
    assert prepared[1].ref_audio_prepare_future == "future-b"
    assert runtime._prepare_cpu_stage_async.await_count == 2


def test_prepare_request_cpu_stages_builds_specs_and_delegates_to_spec_path(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    spec_a = SimpleNamespace(request_id="req-a")
    spec_b = SimpleNamespace(request_id="req-b")
    prepared_cpu_stages = [
        GPTSoVITSPreparedCpuStage("req-a", spec_a, 0.0, 0.0, "pa", "ta", 0.0, 0, 0, SimpleNamespace(), SimpleNamespace(), "future-a"),
        GPTSoVITSPreparedCpuStage("req-b", spec_b, 0.0, 0.0, "pb", "tb", 0.0, 0, 0, SimpleNamespace(), SimpleNamespace(), "future-b"),
    ]
    runtime._build_scheduler_request_spec = Mock(side_effect=[spec_a, spec_b])  # type: ignore[method-assign]
    runtime.prepare_request_spec_cpu_stages = Mock(return_value=prepared_cpu_stages)  # type: ignore[method-assign]

    requests = [{"text": "a"}, {"text": "b", "engine_request_id": "req-b"}]
    prepared = runtime.prepare_request_cpu_stages(requests)

    assert prepared == prepared_cpu_stages
    runtime.prepare_request_spec_cpu_stages.assert_called_once_with([spec_a, spec_b])


def test_prepare_request_specs_uses_batched_split_prepare_pipeline(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    specs = [SimpleNamespace(request_id="req-a"), SimpleNamespace(request_id="req-b")]
    prepared_cpu_stages = [SimpleNamespace(name="cpu-a"), SimpleNamespace(name="cpu-b")]
    prepared_audio_phases = [SimpleNamespace(name="audio-a"), SimpleNamespace(name="audio-b")]
    prepared_ref_spec_phases = [SimpleNamespace(name="ref-a"), SimpleNamespace(name="ref-b")]
    prepared_text_phases = [SimpleNamespace(name="text-a"), SimpleNamespace(name="text-b")]
    prepared_requests = [SimpleNamespace(name="prepared-a"), SimpleNamespace(name="prepared-b")]

    runtime.prepare_request_spec_cpu_stages = Mock(return_value=prepared_cpu_stages)  # type: ignore[method-assign]
    runtime.prepare_request_gpu_audio_phases = Mock(return_value=prepared_audio_phases)  # type: ignore[method-assign]
    runtime.prepare_request_ref_spec_phases = Mock(return_value=prepared_ref_spec_phases)  # type: ignore[method-assign]
    runtime.prepare_request_gpu_text_phases = Mock(return_value=prepared_text_phases)  # type: ignore[method-assign]
    runtime.build_prepared_request_from_phases = Mock(side_effect=prepared_requests)  # type: ignore[method-assign]

    result = runtime.prepare_request_specs(specs)

    runtime.prepare_request_spec_cpu_stages.assert_called_once_with(specs)
    runtime.prepare_request_gpu_audio_phases.assert_called_once_with(prepared_cpu_stages)
    runtime.prepare_request_ref_spec_phases.assert_called_once_with(prepared_audio_phases)
    runtime.prepare_request_gpu_text_phases.assert_called_once_with(prepared_audio_phases)
    assert runtime.build_prepared_request_from_phases.call_count == 2
    for call_args in runtime.build_prepared_request_from_phases.call_args_list:
        assert "extra_profile" in call_args.kwargs
        extra_profile = call_args.kwargs["extra_profile"]
        assert extra_profile["engine_prepare_audio_phase_mode"] == 1.0
        assert extra_profile["engine_prepare_audio_phase_batch_size"] == 2.0
        assert extra_profile["engine_prepare_text_phase_batch_size"] == 2.0
    assert result == prepared_requests


def test_prepare_requests_builds_specs_and_delegates_to_spec_pipeline(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    spec_a = SimpleNamespace(request_id="req-a")
    spec_b = SimpleNamespace(request_id="req-b")
    prepared_requests = [SimpleNamespace(name="prepared-a"), SimpleNamespace(name="prepared-b")]
    runtime._build_scheduler_request_spec = Mock(side_effect=[spec_a, spec_b])  # type: ignore[method-assign]
    runtime.prepare_request_specs = Mock(return_value=prepared_requests)  # type: ignore[method-assign]

    requests = [{"text": "a"}, {"text": "b"}]
    result = runtime.prepare_requests(requests)

    runtime.prepare_request_specs.assert_called_once_with([spec_a, spec_b])
    assert result == prepared_requests


def test_start_ar_session_builds_spec_and_delegates_to_spec_pipeline(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    spec = SimpleNamespace(request_id="req-ar")
    session = SimpleNamespace(name="session")
    runtime._build_scheduler_request_spec = Mock(return_value=spec)  # type: ignore[method-assign]
    runtime.start_ar_session_from_spec = Mock(return_value=session)  # type: ignore[method-assign]

    result = runtime.start_ar_session({"text": "hello"}, request_id="req-ar")

    runtime._build_scheduler_request_spec.assert_called_once_with({"text": "hello"}, request_id="req-ar")
    runtime.start_ar_session_from_spec.assert_called_once_with(spec)
    assert result is session


def test_run_text_cpu_stage_pair_uses_worker_batch_submission(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    worker = SimpleNamespace(
        submit_many_async=AsyncMock(
            return_value=[
                (["prompt-seg"], {"text_cpu_admission_wait_ms": 1.0, "text_cpu_queue_wait_ms": 2.0, "text_cpu_run_ms": 3.0}),
                (["target-seg"], {"text_cpu_admission_wait_ms": 4.0, "text_cpu_queue_wait_ms": 5.0, "text_cpu_run_ms": 6.0}),
            ]
        )
    )
    runtime._pipeline = SimpleNamespace(prepare_text_cpu_worker=worker)
    coordinator = SimpleNamespace(text_cpu_gate=SimpleNamespace(max_inflight=0))

    prompt_profiled, target_profiled = runtime._run_awaitable_sync(
        runtime._run_text_cpu_stage_pair(coordinator, "prompt", "zh", "target", "zh")
    )

    worker.submit_many_async.assert_awaited_once_with([("prompt", "zh", "cut1"), ("target", "zh", "cut1")])
    assert prompt_profiled.result == ["prompt-seg"]
    assert prompt_profiled.queue_ms == pytest.approx(3.0, abs=5.0)
    assert target_profiled.result == ["target-seg"]
    assert target_profiled.run_ms == pytest.approx(6.0, abs=1e-6)


def test_run_text_cpu_stage_pair_prefers_runtime_owned_worker_over_pipeline_attr(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    worker = SimpleNamespace(
        submit_many_async=AsyncMock(
            return_value=[
                (["prompt-seg"], {"text_cpu_admission_wait_ms": 1.0, "text_cpu_queue_wait_ms": 2.0, "text_cpu_run_ms": 3.0}),
                (["target-seg"], {"text_cpu_admission_wait_ms": 4.0, "text_cpu_queue_wait_ms": 5.0, "text_cpu_run_ms": 6.0}),
            ]
        )
    )
    broken_worker = SimpleNamespace(
        submit_many_async=AsyncMock(side_effect=AssertionError("should not use pipeline worker directly"))
    )
    runtime._runtime_prepare_text_cpu_worker = worker
    runtime._pipeline = SimpleNamespace(prepare_text_cpu_worker=broken_worker)
    coordinator = SimpleNamespace(text_cpu_gate=SimpleNamespace(max_inflight=0))

    prompt_profiled, target_profiled = runtime._run_awaitable_sync(
        runtime._run_text_cpu_stage_pair(coordinator, "prompt", "zh", "target", "zh")
    )

    worker.submit_many_async.assert_awaited_once_with([("prompt", "zh", "cut1"), ("target", "zh", "cut1")])
    broken_worker.submit_many_async.assert_not_awaited()
    assert prompt_profiled.result == ["prompt-seg"]
    assert target_profiled.result == ["target-seg"]


def test_prepare_text_cpu_uses_runtime_native_payload_conversion(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._current_text_frontend_version = Mock(return_value="v2")  # type: ignore[method-assign]
    runtime._preprocess_text_segments_payload = Mock(  # type: ignore[method-assign]
        return_value=[
            {
                "language": "zh",
                "phones": [1, 2],
                "word2ph": [1, 1],
                "norm_text": "你好",
                "needs_g2pw": False,
            }
        ]
    )
    runtime._payloads_to_prepared_text_segments = Mock(  # type: ignore[method-assign]
        return_value=["prepared-segments"]
    )
    runtime._pipeline = SimpleNamespace(prepare_text_segments=Mock(side_effect=AssertionError("should not call pipeline helper")))

    result = runtime._prepare_text_cpu("你好", "zh")

    runtime._current_text_frontend_version.assert_called_once_with()
    runtime._preprocess_text_segments_payload.assert_called_once_with("你好", "zh", "v2")
    runtime._payloads_to_prepared_text_segments.assert_called_once()
    assert result == ["prepared-segments"]


def test_resolve_g2pw_segments_uses_runtime_native_frontend_path(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._current_text_frontend_version = Mock(return_value="v2")  # type: ignore[method-assign]
    runtime._get_text_frontend_symbol = Mock(  # type: ignore[method-assign]
        side_effect=[
            Mock(return_value=([(["ni3"], [1], "你")], {"g2pw_predict_ms": 3.0})),
            Mock(return_value=[9]),
        ]
    )
    runtime._get_text_preprocessor_symbol = Mock(  # type: ignore[method-assign]
        return_value=lambda **kwargs: SimpleNamespace(**kwargs)
    )
    runtime._pipeline = SimpleNamespace(resolve_g2pw_segments=Mock(side_effect=AssertionError("should not call pipeline helper")))
    prepared_segments = [SimpleNamespace(language="zh", phones=[], word2ph=None, norm_text="你", needs_g2pw=True)]

    resolved_segments, profile = runtime._resolve_g2pw_segments(prepared_segments)

    assert resolved_segments[0].language == "zh"
    assert resolved_segments[0].phones == [9]
    assert resolved_segments[0].word2ph == [1]
    assert resolved_segments[0].norm_text == "你"
    assert resolved_segments[0].needs_g2pw is False
    assert profile["g2pw_predict_ms"] == pytest.approx(3.0)


def test_build_text_features_uses_runtime_native_sync_helpers(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._resolve_g2pw_segments = Mock(  # type: ignore[method-assign]
        return_value=(
            [
                SimpleNamespace(language="zh", phones=[1, 2], word2ph=[1, 1], norm_text="你好", needs_g2pw=False),
                SimpleNamespace(language="en", phones=[3], word2ph=None, norm_text="a", needs_g2pw=False),
            ],
            {"g2pw_predict_ms": 1.5},
        )
    )
    runtime._build_segment_bert_feature = Mock(  # type: ignore[method-assign]
        side_effect=[
            torch.ones((1024, 2), dtype=torch.float32),
            torch.full((1024, 1), 2.0, dtype=torch.float32),
        ]
    )
    runtime._pipeline = SimpleNamespace(build_text_features_from_segments=Mock(side_effect=AssertionError("should not call pipeline helper")))

    result = runtime._build_text_features(["prepared"], "zh", 5.0, base_profile={"seed": 1.0})

    runtime._resolve_g2pw_segments.assert_called_once_with(["prepared"])
    assert runtime._build_segment_bert_feature.call_count == 2
    assert result.phones == [1, 2, 3]
    assert result.norm_text == "你好a"
    assert result.bert_features.shape == (1024, 3)
    assert result.profile["seed"] == pytest.approx(1.0)
    assert result.profile["cpu_preprocess_ms"] == pytest.approx(5.0)
    assert result.profile["g2pw_predict_ms"] == pytest.approx(1.5)


def test_run_text_feature_stage_uses_native_bert_worker_async_path(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    worker = SimpleNamespace(
        submit_async=AsyncMock(
            return_value=(
                torch.ones((1024, 2), dtype=torch.float32),
                {
                    "bert_wait_ms": 1.0,
                    "bert_tokenize_ms": 2.0,
                    "bert_forward_ms": 3.0,
                    "bert_scatter_ms": 4.0,
                    "bert_calls": 1.0,
                },
            )
        )
    )
    build_text_features_async = AsyncMock(side_effect=AssertionError("should not call pipeline async helper"))
    runtime._pipeline = SimpleNamespace(
        prepare_bert_batch_worker=worker,
        configs=SimpleNamespace(device="cpu"),
        build_text_features_from_segments_async=build_text_features_async,
    )
    prepared_segments = [
        SimpleNamespace(
            language="zh",
            phones=[1, 2],
            word2ph=[1, 1],
            norm_text="你好",
        )
    ]

    profiled = runtime._run_awaitable_sync(
        runtime._run_text_feature_stage(
            SimpleNamespace(text_feature_executor=None),
            prepared_segments,
            "zh",
            5.0,
            base_profile={"_branch_start_ts": time.perf_counter()},
        )
    )

    worker.submit_async.assert_awaited_once_with("你好", [1, 1])
    build_text_features_async.assert_not_awaited()
    assert profiled.result.phones == [1, 2]
    assert profiled.result.norm_text == "你好"
    assert profiled.result.bert_features.shape == (1024, 2)
    assert profiled.result.profile["cpu_preprocess_ms"] == 5.0
    assert profiled.result.profile["bert_calls"] == pytest.approx(1.0)


def test_run_text_feature_stage_prefers_native_bert_submit_many_async(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    worker = SimpleNamespace(
        submit_many_async=AsyncMock(
            return_value=[
                (
                    torch.ones((1024, 1), dtype=torch.float32),
                    {"bert_wait_ms": 1.0, "bert_forward_ms": 2.0, "bert_calls": 1.0},
                ),
                (
                    torch.full((1024, 2), 2.0, dtype=torch.float32),
                    {"bert_wait_ms": 3.0, "bert_forward_ms": 4.0, "bert_calls": 1.0},
                ),
            ]
        ),
        submit_async=AsyncMock(side_effect=AssertionError("should not call submit_async")),
    )
    runtime._pipeline = SimpleNamespace(
        prepare_bert_batch_worker=worker,
        configs=SimpleNamespace(device="cpu"),
    )
    prepared_segments = [
        SimpleNamespace(language="zh", phones=[10], word2ph=[1], norm_text="你"),
        SimpleNamespace(language="zh", phones=[20, 21], word2ph=[1, 1], norm_text="好呀"),
    ]

    profiled = runtime._run_awaitable_sync(
        runtime._run_text_feature_stage(
            SimpleNamespace(text_feature_executor=None),
            prepared_segments,
            "zh",
            5.0,
            base_profile={"_branch_start_ts": time.perf_counter()},
        )
    )

    worker.submit_many_async.assert_awaited_once_with([("你", [1]), ("好呀", [1, 1])])
    worker.submit_async.assert_not_awaited()
    assert profiled.result.phones == [10, 20, 21]
    assert profiled.result.norm_text == "你好呀"
    assert profiled.result.bert_features.shape == (1024, 3)
    assert profiled.result.profile["bert_calls"] == pytest.approx(2.0)


def test_run_text_feature_pair_stage_uses_native_bert_worker_async_path(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    worker = SimpleNamespace(
        submit_async=AsyncMock(
            side_effect=[
                (
                    torch.ones((1024, 1), dtype=torch.float32),
                    {"bert_wait_ms": 1.0, "bert_forward_ms": 2.0, "bert_calls": 1.0},
                ),
                (
                    torch.full((1024, 2), 2.0, dtype=torch.float32),
                    {"bert_wait_ms": 3.0, "bert_forward_ms": 4.0, "bert_calls": 1.0},
                ),
            ]
        )
    )
    build_text_features_async = AsyncMock(side_effect=AssertionError("should not call single async helper"))
    build_text_feature_pair_async = AsyncMock(side_effect=AssertionError("should not call pair async helper"))
    runtime._pipeline = SimpleNamespace(
        prepare_bert_batch_worker=worker,
        configs=SimpleNamespace(device="cpu"),
        build_text_features_from_segments_async=build_text_features_async,
        build_text_feature_pair_from_segments_async=build_text_feature_pair_async,
    )
    prompt_segments = [
        SimpleNamespace(
            language="zh",
            phones=[10],
            word2ph=[1],
            norm_text="你",
        )
    ]
    target_segments = [
        SimpleNamespace(
            language="zh",
            phones=[20, 21],
            word2ph=[1, 1],
            norm_text="好呀",
        )
    ]

    prompt_profiled, target_profiled = runtime._run_awaitable_sync(
        runtime._run_text_feature_pair_stage(
            SimpleNamespace(text_feature_executor=None),
            prompt_segments,
            target_segments,
            6.0,
            7.0,
            prompt_base_profile={"_branch_start_ts": time.perf_counter()},
            target_base_profile={"_branch_start_ts": time.perf_counter()},
        )
    )

    assert worker.submit_async.await_count == 2
    worker.submit_async.assert_any_await("你", [1])
    worker.submit_async.assert_any_await("好呀", [1, 1])
    build_text_features_async.assert_not_awaited()
    build_text_feature_pair_async.assert_not_awaited()
    assert prompt_profiled.result.phones == [10]
    assert prompt_profiled.result.norm_text == "你"
    assert prompt_profiled.result.bert_features.shape == (1024, 1)
    assert prompt_profiled.result.profile["cpu_preprocess_ms"] == 6.0
    assert target_profiled.result.phones == [20, 21]
    assert target_profiled.result.norm_text == "好呀"
    assert target_profiled.result.bert_features.shape == (1024, 2)
    assert target_profiled.result.profile["cpu_preprocess_ms"] == 7.0


def test_run_text_feature_pair_stage_prefers_native_bert_submit_many_async(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    worker = SimpleNamespace(
        submit_many_async=AsyncMock(
            return_value=[
                (
                    torch.ones((1024, 1), dtype=torch.float32),
                    {"bert_wait_ms": 1.0, "bert_forward_ms": 2.0, "bert_calls": 1.0},
                ),
                (
                    torch.full((1024, 2), 2.0, dtype=torch.float32),
                    {"bert_wait_ms": 3.0, "bert_forward_ms": 4.0, "bert_calls": 1.0},
                ),
            ]
        ),
        submit_async=AsyncMock(side_effect=AssertionError("should not call submit_async")),
    )
    runtime._pipeline = SimpleNamespace(
        prepare_bert_batch_worker=worker,
        configs=SimpleNamespace(device="cpu"),
    )
    prompt_segments = [SimpleNamespace(language="zh", phones=[10], word2ph=[1], norm_text="你")]
    target_segments = [SimpleNamespace(language="zh", phones=[20, 21], word2ph=[1, 1], norm_text="好呀")]

    prompt_profiled, target_profiled = runtime._run_awaitable_sync(
        runtime._run_text_feature_pair_stage(
            SimpleNamespace(text_feature_executor=None),
            prompt_segments,
            target_segments,
            6.0,
            7.0,
            prompt_base_profile={"_branch_start_ts": time.perf_counter()},
            target_base_profile={"_branch_start_ts": time.perf_counter()},
        )
    )

    worker.submit_many_async.assert_awaited_once_with([("你", [1]), ("好呀", [1, 1])])
    worker.submit_async.assert_not_awaited()
    assert prompt_profiled.result.phones == [10]
    assert target_profiled.result.phones == [20, 21]
    assert prompt_profiled.result.profile["bert_calls"] == pytest.approx(1.0)
    assert target_profiled.result.profile["bert_calls"] == pytest.approx(1.0)


def test_compute_sync_bert_feature_returns_empty_feature_for_zero_repeat_counts(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))

    class DummyTokenizer:
        def __call__(self, text, return_tensors="pt"):
            del text, return_tensors
            return {
                "input_ids": torch.ones((1, 4), dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
            }

    class DummyBert:
        def __call__(self, **kwargs):
            del kwargs
            hidden = torch.arange(4 * 1024, dtype=torch.float32).view(1, 4, 1024)
            return {"hidden_states": [torch.zeros_like(hidden), hidden, torch.zeros_like(hidden), torch.zeros_like(hidden)]}

    runtime._pipeline = SimpleNamespace(
        prepare_bert_batch_worker=None,
        bert_tokenizer=DummyTokenizer(),
        bert_model=DummyBert(),
        configs=SimpleNamespace(device="cpu"),
        prepare_bert_stage_limiter=None,
    )

    feature = runtime._compute_sync_bert_feature("你好", [0, 0], profile={})

    assert feature.shape == (1024, 0)


def test_build_ar_session_routes_through_runtime_native_ar_helpers(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))

    active_batch = SimpleNamespace(
        request_ids=["req-native"],
        states=[SimpleNamespace(request_id="req-native")],
        x=torch.ones((1, 1, 4), dtype=torch.float32),
        x_lens=torch.tensor([1], dtype=torch.long),
        y_sequences=[torch.tensor([8], dtype=torch.long)],
        prefix_lens=torch.tensor([1], dtype=torch.long),
        xy_pos=torch.ones((1, 2, 4), dtype=torch.float32),
        key_padding_mask=torch.tensor([[False, False]], dtype=torch.bool),
        prefill_attn_mask=torch.zeros((1, 1, 2, 2), dtype=torch.bool),
        decode_attn_mask=None,
        k_cache=None,
        v_cache=None,
        kv_lens=None,
        step_indices=torch.zeros((1,), dtype=torch.long),
        prefill_done=False,
        kv_cache_pooled=False,
        kv_cache_capacity=0,
        kv_cache_batch_capacity=0,
    )
    fake_model = SimpleNamespace(
        t2s_transformer=SimpleNamespace(
            process_prompt=Mock(
                return_value=(
                    torch.ones((1, 2, 4), dtype=torch.float32),
                    [torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)],
                    [torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)],
                )
            )
        ),
        ar_predict_layer=Mock(return_value=torch.tensor([[0.1, 0.2]], dtype=torch.float32)),
    )
    runtime._pipeline = SimpleNamespace(t2s_model=SimpleNamespace(model=fake_model))
    runtime._build_prefill_active_batch = Mock(return_value=active_batch)  # type: ignore[method-assign]
    runtime._pack_active_batch_into_pool = Mock(return_value=False)  # type: ignore[method-assign]
    runtime._compact_cache_to_kv_lens = Mock(side_effect=lambda layer, kv_lens: layer)  # type: ignore[method-assign]
    runtime._compact_decode_mask_to_kv_lens = Mock(side_effect=lambda mask, kv_lens: mask)  # type: ignore[method-assign]
    runtime._build_next_xy_pos = Mock(return_value=torch.ones((1, 1, 4), dtype=torch.float32))  # type: ignore[method-assign]

    prepared = SimpleNamespace(
        request_id="req-native",
        state=SimpleNamespace(request_id="req-native"),
        transport_info={"transport": "ok"},
    )

    session = runtime._build_ar_session_from_prepared(prepared)

    runtime._build_prefill_active_batch.assert_called_once_with(fake_model, [prepared.state])
    runtime._pack_active_batch_into_pool.assert_called_once_with(fake_model, active_batch)
    assert runtime._compact_cache_to_kv_lens.call_count == 2
    runtime._compact_decode_mask_to_kv_lens.assert_called_once()
    runtime._build_next_xy_pos.assert_called_once_with(fake_model, active_batch.y_sequences)
    assert active_batch.prefill_done is True
    assert active_batch.x is None
    assert active_batch.x_lens is None
    assert active_batch.key_padding_mask is None
    assert active_batch.prefill_attn_mask is None
    assert session.request_id == "req-native"
    assert session.transport_info == {"transport": "ok"}
    assert torch.equal(session.current_logits, torch.tensor([[0.1, 0.2]], dtype=torch.float32))


def test_pack_active_batch_into_pool_rejects_decode_headroom_overflow(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pool = SimpleNamespace(
        max_seq_len=4,
        max_batch_size=8,
        state=SimpleNamespace(enabled=True),
        record_fallback=Mock(),
        set_active_rows=Mock(),
        pack_dynamic_cache_layers=Mock(),
    )
    active_batch = SimpleNamespace(
        request_ids=["req-overflow"],
        k_cache=[torch.ones((1, 4, 3), dtype=torch.float32)],
        v_cache=[torch.ones((1, 4, 3), dtype=torch.float32)],
        kv_lens=torch.tensor([4], dtype=torch.long),
        kv_cache_pooled=True,
        kv_cache_capacity=4,
        kv_cache_batch_capacity=8,
        decode_attn_mask=None,
    )

    packed = runtime._pack_active_batch_into_pool(SimpleNamespace(kv_cache_pool=pool), active_batch)

    assert packed is False
    pool.pack_dynamic_cache_layers.assert_not_called()
    pool.record_fallback.assert_called_once()
    assert active_batch.kv_cache_pooled is False
    assert active_batch.kv_cache_capacity == 0
    assert active_batch.kv_cache_batch_capacity == 0


def test_advance_ar_session_routes_non_pooled_decode_through_runtime_helpers(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))

    initial_decode_mask = torch.zeros((1, 1, 1, 2), dtype=torch.bool)
    initial_kv_lens = torch.tensor([1], dtype=torch.long)
    active_batch = SimpleNamespace(
        y_sequences=[torch.tensor([5], dtype=torch.long)],
        step_indices=torch.tensor([0], dtype=torch.long),
        xy_pos=torch.zeros((1, 1, 4), dtype=torch.float32),
        k_cache=[torch.ones((1, 1, 3), dtype=torch.float32)],
        v_cache=[torch.ones((1, 1, 3), dtype=torch.float32)],
        kv_lens=initial_kv_lens.clone(),
        kv_cache_pooled=False,
        decode_attn_mask=initial_decode_mask,
    )
    fake_model = SimpleNamespace(
        t2s_transformer=SimpleNamespace(
            decode_next_token=Mock(
                return_value=(
                    torch.ones((1, 1, 4), dtype=torch.float32),
                    [torch.ones((1, 2, 3), dtype=torch.float32)],
                    [torch.ones((1, 2, 3), dtype=torch.float32)],
                )
            )
        ),
        ar_predict_layer=Mock(return_value=torch.tensor([[0.3, 0.4]], dtype=torch.float32)),
    )
    runtime._pipeline = SimpleNamespace(t2s_model=SimpleNamespace(model=fake_model))
    runtime._build_next_xy_pos = Mock(return_value=torch.ones((1, 1, 4), dtype=torch.float32))  # type: ignore[method-assign]
    runtime._materialize_decode_mask_for_active_batch = Mock(  # type: ignore[method-assign]
        return_value=torch.zeros((1, 1, 1, 2), dtype=torch.bool)
    )
    runtime._advance_decode_mask = Mock(  # type: ignore[method-assign]
        return_value=torch.ones((1, 1, 1, 3), dtype=torch.bool)
    )

    session = GPTSoVITSARSession(
        request_id="req-advance",
        active_batch=active_batch,
        transport_info={},
        current_logits=torch.zeros((1, 2), dtype=torch.float32),
    )

    logits = runtime.advance_ar_session(session, 9)

    assert active_batch.y_sequences[0].tolist() == [5, 9]
    assert int(active_batch.step_indices.item()) == 1
    runtime._build_next_xy_pos.assert_called_once_with(fake_model, active_batch.y_sequences)
    runtime._materialize_decode_mask_for_active_batch.assert_called_once_with(active_batch)
    fake_model.t2s_transformer.decode_next_token.assert_called_once()
    assert fake_model.t2s_transformer.decode_next_token.call_args.args[3] is None
    runtime._advance_decode_mask.assert_called_once_with(initial_decode_mask, initial_kv_lens)
    assert torch.equal(active_batch.kv_lens, torch.tensor([2], dtype=torch.long))
    assert torch.equal(active_batch.decode_attn_mask, torch.ones((1, 1, 1, 3), dtype=torch.bool))
    assert torch.equal(session.current_logits, torch.tensor([[0.3, 0.4]], dtype=torch.float32))
    assert torch.equal(logits, session.current_logits)


def test_decode_active_batch_falls_back_from_pooled_cache_when_decode_headroom_exhausted(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pool = SimpleNamespace(
        max_seq_len=4,
        max_batch_size=8,
        state=SimpleNamespace(enabled=True),
        record_fallback=Mock(),
        set_active_rows=Mock(),
    )
    active_batch = SimpleNamespace(
        request_ids=["req-a", "req-b"],
        states=[SimpleNamespace(request_id="req-a"), SimpleNamespace(request_id="req-b")],
        y_sequences=[torch.tensor([5], dtype=torch.long), torch.tensor([6], dtype=torch.long)],
        prefix_lens=torch.tensor([1, 1], dtype=torch.long),
        step_indices=torch.tensor([0, 0], dtype=torch.long),
        xy_pos=torch.zeros((2, 1, 4), dtype=torch.float32),
        k_cache=[
            torch.tensor(
                [
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [0.0, 0.0, 0.0]],
                    [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]],
                ],
                dtype=torch.float32,
            )
        ],
        v_cache=[
            torch.tensor(
                [
                    [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0], [0.0, 0.0, 0.0]],
                    [[40.0, 40.0, 40.0], [50.0, 50.0, 50.0], [60.0, 60.0, 60.0], [70.0, 70.0, 70.0]],
                ],
                dtype=torch.float32,
            )
        ],
        kv_lens=torch.tensor([3, 4], dtype=torch.long),
        kv_cache_pooled=True,
        kv_cache_capacity=4,
        kv_cache_batch_capacity=8,
        decode_attn_mask=None,
        prefill_done=True,
    )
    fake_model = SimpleNamespace(
        kv_cache_pool=pool,
        t2s_transformer=SimpleNamespace(
            decode_next_token=Mock(
                return_value=(
                    torch.ones((2, 1, 4), dtype=torch.float32),
                    [torch.ones((2, 5, 3), dtype=torch.float32)],
                    [torch.ones((2, 5, 3), dtype=torch.float32)],
                )
            )
        ),
        decode_next_token_prealloc_runtime=Mock(),
        ar_predict_layer=Mock(return_value=torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)),
    )
    runtime._sample_active_batch_requests = Mock(  # type: ignore[method-assign]
        return_value=(
            [],
            [0, 1],
            [torch.tensor([5, 9], dtype=torch.long), torch.tensor([6, 8], dtype=torch.long)],
        )
    )
    runtime._build_next_xy_pos = Mock(return_value=torch.ones((2, 1, 4), dtype=torch.float32))  # type: ignore[method-assign]

    updated_batch, finished_items = runtime._decode_active_batch_one_step(fake_model, active_batch, max_steps=8)

    assert finished_items == []
    assert updated_batch is active_batch
    fake_model.decode_next_token_prealloc_runtime.assert_not_called()
    fake_model.t2s_transformer.decode_next_token.assert_called_once()
    decode_mask = fake_model.t2s_transformer.decode_next_token.call_args.args[3]
    assert decode_mask.shape == (2, 1, 1, 5)
    assert torch.equal(decode_mask[0, 0, 0], torch.tensor([True, False, False, False, False], dtype=torch.bool))
    assert torch.equal(decode_mask[1, 0, 0], torch.tensor([False, False, False, False, False], dtype=torch.bool))
    assert active_batch.kv_cache_pooled is False
    assert active_batch.kv_cache_capacity == 0
    assert active_batch.kv_cache_batch_capacity == 0
    assert torch.equal(active_batch.kv_lens, torch.tensor([4, 5], dtype=torch.long))
    pool.record_fallback.assert_called_once()


def test_decode_active_batch_merges_profiled_prealloc_decode_stats(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pool = SimpleNamespace(
        max_seq_len=8,
        state=SimpleNamespace(enabled=True),
        build_decode_mask=Mock(return_value=torch.zeros((1, 1, 1, 3), dtype=torch.bool)),
        set_active_rows=Mock(),
    )
    active_batch = SimpleNamespace(
        request_ids=["req-a"],
        states=[SimpleNamespace(request_id="req-a", early_stop_num=-1)],
        y_sequences=[torch.tensor([5, 6], dtype=torch.long)],
        prefix_lens=torch.tensor([1], dtype=torch.long),
        step_indices=torch.tensor([1], dtype=torch.long),
        xy_pos=torch.zeros((1, 1, 4), dtype=torch.float32),
        k_cache=[torch.ones((1, 8, 3), dtype=torch.float32)],
        v_cache=[torch.ones((1, 8, 3), dtype=torch.float32)],
        kv_lens=torch.tensor([2], dtype=torch.long),
        kv_cache_pooled=True,
        kv_cache_capacity=8,
        kv_cache_batch_capacity=1,
        decode_attn_mask=None,
        prefill_done=True,
    )
    fake_model = SimpleNamespace(
        kv_cache_pool=pool,
        decode_next_token_prealloc_runtime=Mock(
            return_value=(
                torch.ones((1, 1, 4), dtype=torch.float32),
                [torch.ones((1, 8, 3), dtype=torch.float32)],
                [torch.ones((1, 8, 3), dtype=torch.float32)],
            )
        ),
        get_last_prealloc_decode_profile=Mock(
            return_value={
                "pooled_prealloc_profiled_decode_calls": 1,
                "pooled_prealloc_profiled_layer_calls": 24,
                "pooled_prealloc_qkv_linear_ms": 12.5,
                "pooled_prealloc_sdpa_ms": 34.0,
                "pooled_prealloc_layer_total_ms": 56.0,
            }
        ),
        ar_predict_layer=Mock(return_value=torch.tensor([[0.1, 0.9]], dtype=torch.float32)),
    )
    runtime._sample_active_batch_requests = Mock(  # type: ignore[method-assign]
        return_value=([], [0], [torch.tensor([5, 6, 7], dtype=torch.long)])
    )
    runtime._build_next_xy_pos = Mock(return_value=torch.ones((1, 1, 4), dtype=torch.float32))  # type: ignore[method-assign]
    stats: dict[str, float | int] = {}

    updated_batch, finished_items = runtime._decode_active_batch_one_step(
        fake_model,
        active_batch,
        max_steps=8,
        stats=stats,
    )

    assert finished_items == []
    assert updated_batch is active_batch
    fake_model.decode_next_token_prealloc_runtime.assert_called_once()
    fake_model.get_last_prealloc_decode_profile.assert_called_once()
    assert int(stats["pooled_prealloc_profiled_decode_calls"]) == 1
    assert int(stats["pooled_prealloc_profiled_layer_calls"]) == 24
    assert float(stats["pooled_prealloc_qkv_linear_ms"]) == 12.5
    assert float(stats["pooled_prealloc_sdpa_ms"]) == 34.0
    assert float(stats["pooled_prealloc_layer_total_ms"]) == 56.0


def test_runtime_build_next_xy_pos_single_request_fast_path_matches_expected(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    embedding = nn.Embedding(16, 4)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(64, dtype=torch.float32).reshape(16, 4))
    audio_position = SimpleNamespace(
        pe=torch.arange(32, dtype=torch.float32).reshape(1, 8, 4),
        x_scale=2.0,
        alpha=0.5,
        embedding_dim=4,
        extend_pe=Mock(side_effect=AssertionError("unexpected extend_pe call")),
    )
    fake_model = SimpleNamespace(
        ar_audio_embedding=embedding,
        ar_audio_position=audio_position,
    )
    y_sequences = [torch.tensor([1, 4, 7], dtype=torch.long)]

    xy_pos = runtime._build_next_xy_pos(fake_model, y_sequences)

    expected = embedding(torch.tensor([[7]], dtype=torch.long)) * 2.0 + 0.5 * audio_position.pe[:, 2:3, :]
    assert torch.allclose(xy_pos, expected)


def test_scheduler_build_next_xy_pos_single_request_fast_path_matches_expected():
    _ensure_vendored_gpt_sovits_import_path()
    from vllm_omni.model_executor.models.gpt_sovits.runtime_lib.GPT_SoVITS.TTS_infer_pack.t2s_scheduler import (
        build_next_xy_pos,
    )

    embedding = nn.Embedding(16, 4)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(64, dtype=torch.float32).reshape(16, 4))
    audio_position = SimpleNamespace(
        pe=torch.arange(32, dtype=torch.float32).reshape(1, 8, 4),
        x_scale=2.0,
        alpha=0.5,
        embedding_dim=4,
        extend_pe=Mock(side_effect=AssertionError("unexpected extend_pe call")),
    )
    fake_model = SimpleNamespace(
        ar_audio_embedding=embedding,
        ar_audio_position=audio_position,
    )
    y_sequences = [torch.tensor([2, 5, 9], dtype=torch.long)]

    xy_pos = build_next_xy_pos(fake_model, y_sequences)

    expected = embedding(torch.tensor([[9]], dtype=torch.long)) * 2.0 + 0.5 * audio_position.pe[:, 2:3, :]
    assert torch.allclose(xy_pos, expected)


def test_t2s_kv_cache_pool_build_decode_mask_skips_single_request_padding():
    _ensure_vendored_gpt_sovits_import_path()
    from vllm_omni.model_executor.models.gpt_sovits.runtime_lib.GPT_SoVITS.TTS_infer_pack.t2s_kv_cache_pool import (
        T2SKVCachePool,
    )

    pool = T2SKVCachePool(
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_layers=1,
        hidden_dim=2,
        max_batch_size=2,
        max_seq_len=8,
    )

    assert pool.build_decode_mask(torch.tensor([3], dtype=torch.long)) is None

    mask = pool.build_decode_mask(torch.tensor([2, 3], dtype=torch.long))

    assert mask is not None
    assert torch.equal(mask[0, 0, 0], torch.tensor([False, False, True], dtype=torch.bool))
    assert torch.equal(mask[1, 0, 0], torch.tensor([False, False, False], dtype=torch.bool))


def test_t2s_prepare_prealloc_decode_inputs_stabilizes_x_and_hoists_metadata():
    _ensure_vendored_gpt_sovits_import_path()
    from vllm_omni.model_executor.models.gpt_sovits.runtime_lib.GPT_SoVITS.AR.models.t2s_model import (
        Text2SemanticDecoder,
    )

    base = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    x = base[:, 1:, :]
    kv_lens = torch.tensor([5], dtype=torch.long)
    attn_mask = torch.tensor([[[[False, False, False, True, True, True]]]], dtype=torch.bool)

    stable_x, batch_index, max_kv_index, next_max_kv_len, sdpa_attn_mask = Text2SemanticDecoder._prepare_prealloc_decode_inputs(
        x,
        kv_lens,
        attn_mask,
    )

    assert torch.equal(stable_x, x)
    assert stable_x.is_contiguous()
    assert torch.equal(batch_index, torch.tensor([0], dtype=torch.long))
    assert max_kv_index == 5
    assert next_max_kv_len == 6
    assert torch.equal(sdpa_attn_mask, torch.tensor([[[[True, True, True, False, False, False]]]], dtype=torch.bool))


def test_generate_semantic_tokens_uses_runtime_native_batch_scheduler(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._pipeline = SimpleNamespace(t2s_model=SimpleNamespace(model="fake-model"))
    runtime._estimate_scheduler_max_steps = Mock(return_value=37)  # type: ignore[method-assign]
    runtime._run_continuous_batch_scheduler = Mock(  # type: ignore[method-assign]
        return_value=[
            GPTSoVITSARFinishedItem(
                request_id="req-a",
                semantic_tokens=torch.tensor([1, 2], dtype=torch.long),
                finish_idx=1,
                finish_reason="eos_sample",
            ),
            GPTSoVITSARFinishedItem(
                request_id="req-b",
                semantic_tokens=torch.tensor([3], dtype=torch.long),
                finish_idx=0,
                finish_reason="max_step",
            ),
        ]
    )
    prepared_requests = [
        SimpleNamespace(state=SimpleNamespace(request_id="req-a")),
        SimpleNamespace(state=SimpleNamespace(request_id="req-b")),
    ]

    result = runtime.generate_semantic_tokens(prepared_requests)

    runtime._estimate_scheduler_max_steps.assert_called_once_with([item.state for item in prepared_requests])
    runtime._run_continuous_batch_scheduler.assert_called_once_with("fake-model", [item.state for item in prepared_requests], max_steps=37)
    assert torch.equal(result["req-a"], torch.tensor([1, 2], dtype=torch.long))
    assert torch.equal(result["req-b"], torch.tensor([3], dtype=torch.long))


def test_g2pw_pinyin_defaults_to_pypinyin_when_cuda_backend_fails(monkeypatch):
    package_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "vllm_omni",
            "model_executor",
            "models",
            "gpt_sovits",
            "runtime_lib",
            "GPT_SoVITS",
        )
    )
    module_name = "text.g2pw.g2pw"
    cuda_module_name = "text.g2pw.cuda_api"
    onnx_module_name = "text.g2pw.onnx_api"

    cuda_stub = ModuleType(cuda_module_name)

    class BrokenCudaConverter:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("cuda backend unavailable")

    cuda_stub.G2PWCudaConverter = BrokenCudaConverter

    onnx_stub = ModuleType(onnx_module_name)

    class ForbiddenOnnxConverter:
        def __init__(self, *args, **kwargs):
            raise AssertionError("onnx fallback should stay disabled by default")

    onnx_stub.G2PWOnnxConverter = ForbiddenOnnxConverter

    monkeypatch.setenv("GPTSOVITS_G2PW_BACKEND", "cuda")
    monkeypatch.delenv("GPTSOVITS_G2PW_ONNX_FALLBACK", raising=False)
    monkeypatch.syspath_prepend(package_root)
    monkeypatch.setitem(sys.modules, cuda_module_name, cuda_stub)
    monkeypatch.setitem(sys.modules, onnx_module_name, onnx_stub)

    g2pw_module = importlib.import_module(module_name)
    instance = g2pw_module.G2PWPinyin(model_dir="dummy", model_source="dummy")

    assert instance._g2pw is None


def test_attach_t2s_kv_cache_pool_honors_legacy_enable_env(monkeypatch):
    from vllm_omni.model_executor.models.gpt_sovits.runtime_lib.GPT_SoVITS.TTS_infer_pack.t2s_kv_cache_pool import (
        attach_t2s_kv_cache_pool,
    )

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.num_layers = 1
            self.model_dim = 1

    model = DummyModel()
    monkeypatch.delenv("GPTSOVITS_ENGINE_KV_POOL_ENABLE", raising=False)
    monkeypatch.setenv("GPT_SOVITS_ENABLE_AR_KV_POOL", "0")

    state = attach_t2s_kv_cache_pool(model, "cpu")

    assert state.enabled is False
    assert getattr(model, "kv_cache_pool", None) is None


def test_attach_t2s_kv_cache_pool_defaults_batch_to_t2s_active_batch(monkeypatch):
    from vllm_omni.model_executor.models.gpt_sovits.runtime_lib.GPT_SoVITS.TTS_infer_pack.t2s_kv_cache_pool import (
        attach_t2s_kv_cache_pool,
    )

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.num_layers = 1
            self.model_dim = 1

    model = DummyModel()
    monkeypatch.delenv("GPTSOVITS_ENGINE_KV_POOL_MAX_BATCH", raising=False)
    monkeypatch.setenv("GPTSOVITS_T2S_MAX_ACTIVE_BATCH", "3")

    state = attach_t2s_kv_cache_pool(model, "cpu")

    assert state.enabled is True
    assert state.max_batch_size == 3


def test_chinese2_init_falls_back_when_g2pw_init_fails(monkeypatch):
    package_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "vllm_omni",
            "model_executor",
            "models",
            "gpt_sovits",
            "runtime_lib",
            "GPT_SoVITS",
        )
    )
    fake_g2pw = ModuleType("text.g2pw")

    class BrokenG2PWPinyin:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("g2pw init failed")

    fake_g2pw.G2PWPinyin = BrokenG2PWPinyin
    fake_g2pw.correct_pronunciation = lambda word, word_pinyins: word_pinyins

    monkeypatch.syspath_prepend(package_root)
    monkeypatch.setitem(sys.modules, "text.g2pw", fake_g2pw)
    monkeypatch.delitem(sys.modules, "text.chinese2", raising=False)

    chinese2 = importlib.import_module("text.chinese2")
    g2pw_results, profile = chinese2._predict_g2pw_batch(["重庆银行的行长。"])

    assert chinese2.is_g2pw is False
    assert chinese2.g2pw is None
    assert g2pw_results == []
    assert profile["g2pw_predict_ms"] == 0.0


@pytest.mark.slow
@pytest.mark.skipif(
    not _has_g2pw_cuda_runtime_test_assets(),
    reason="g2pw-cu runtime integration regression test requires CUDA plus local GPT-SoVITS/g2pw-cu assets",
)
def test_runtime_pipeline_keeps_g2pw_cuda_backend_in_fresh_subprocess():
    script = f"""
import json
from vllm_omni.model_executor.models.gpt_sovits.runtime import GPTSoVITSRuntime

runtime = GPTSoVITSRuntime(
    project_root={str(_GPT_SOVITS_PROJECT_ROOT)!r},
    config_path={str(_G2PW_CU_RUNTIME_CONFIG)!r},
)
runtime._ensure_pipeline()

import text.chinese2 as chinese2

results, profile = chinese2._predict_g2pw_batch(["重庆银行的行长。"])
backend = getattr(getattr(chinese2, "g2pw", None), "_g2pw", None)
print(
    json.dumps(
        {{
            "backend_type": type(backend).__name__ if backend is not None else None,
            "is_g2pw": bool(getattr(chinese2, "is_g2pw", False)),
            "result_count": len(results),
            "profile": profile,
        }},
        ensure_ascii=False,
    )
)
"""
    env = os.environ.copy()
    env.pop("GPTSOVITS_PRELOAD_NATIVE_RUNTIME_DEPS", None)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    assert completed.returncode == 0, combined_output
    assert "fallback to pypinyin" not in combined_output
    assert "invalid tensor header in manifest" not in combined_output

    payload_line = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")), None)
    assert payload_line is not None, combined_output
    payload = json.loads(payload_line)

    assert payload["backend_type"] == "G2PWCudaConverter"
    assert payload["is_g2pw"] is True
    assert payload["result_count"] == 1


def test_merge_active_batches_uses_runtime_native_merge_helpers(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._build_next_xy_pos = Mock(return_value=torch.ones((2, 1, 4), dtype=torch.float32))  # type: ignore[method-assign]
    runtime._pack_active_batch_into_pool = Mock(return_value=False)  # type: ignore[method-assign]

    left_batch = SimpleNamespace(
        request_ids=["left"],
        states=[SimpleNamespace(request_id="left")],
        y_sequences=[torch.tensor([11], dtype=torch.long)],
        prefix_lens=torch.tensor([1], dtype=torch.long),
        step_indices=torch.tensor([0], dtype=torch.long),
        k_cache=[torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)],
        v_cache=[torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)],
        kv_lens=torch.tensor([2], dtype=torch.long),
        prefill_done=True,
        kv_cache_pooled=False,
    )
    right_batch = SimpleNamespace(
        request_ids=["right"],
        states=[SimpleNamespace(request_id="right")],
        y_sequences=[torch.tensor([22], dtype=torch.long)],
        prefix_lens=torch.tensor([1], dtype=torch.long),
        step_indices=torch.tensor([1], dtype=torch.long),
        k_cache=[torch.arange(6, 12, dtype=torch.float32).reshape(1, 2, 3)],
        v_cache=[torch.arange(6, 12, dtype=torch.float32).reshape(1, 2, 3)],
        kv_lens=torch.tensor([2], dtype=torch.long),
        prefill_done=True,
        kv_cache_pooled=False,
    )

    merged = runtime._merge_active_batches(SimpleNamespace(), left_batch, right_batch)

    runtime._build_next_xy_pos.assert_called_once()
    runtime._pack_active_batch_into_pool.assert_called_once()
    assert isinstance(merged, GPTSoVITSActiveBatch)
    assert merged.request_ids == ["left", "right"]
    assert torch.equal(merged.kv_lens, torch.tensor([2, 2], dtype=torch.long))
    assert torch.equal(merged.step_indices, torch.tensor([0, 1], dtype=torch.long))
    assert merged.prefill_done is True
    assert merged.k_cache[0].shape == (2, 2, 3)


def test_run_continuous_batch_scheduler_uses_runtime_native_loop(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    state_late = SimpleNamespace(request_id="b", ready_step=2)
    state_now = SimpleNamespace(request_id="a", ready_step=0)
    admitted_batch = SimpleNamespace(name="admitted", kv_lens=None, request_ids=[])
    merged_batch = SimpleNamespace(name="merged", kv_lens=None, request_ids=[])
    finished_prefill = [GPTSoVITSARFinishedItem("a", torch.tensor([1]), 0, "prefill")]
    finished_decode = [GPTSoVITSARFinishedItem("b", torch.tensor([2]), 1, "decode")]
    runtime._run_prefill_active_batch = Mock(side_effect=[(admitted_batch, []), (None, finished_prefill)])  # type: ignore[method-assign]
    runtime._merge_active_batches = Mock(side_effect=[merged_batch, merged_batch])  # type: ignore[method-assign]
    runtime._decode_active_batch_one_step = Mock(side_effect=[(None, finished_decode), (None, [])])  # type: ignore[method-assign]

    finished = runtime._run_continuous_batch_scheduler(SimpleNamespace(), [state_late, state_now], max_steps=9)

    assert [item.request_id for item in finished] == ["a", "b"]
    assert runtime._run_prefill_active_batch.call_args_list[0].args[1] == [state_now]
    assert runtime._run_prefill_active_batch.call_args_list[1].args[1] == [state_late]


def test_run_continuous_batch_scheduler_limits_admitted_states_by_active_batch(tmp_path, monkeypatch):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    monkeypatch.setenv("GPTSOVITS_T2S_MAX_ACTIVE_BATCH", "1")
    states = [
        SimpleNamespace(request_id="a", ready_step=0),
        SimpleNamespace(request_id="b", ready_step=0),
        SimpleNamespace(request_id="c", ready_step=0),
    ]
    admitted_batches: list[list[str]] = []

    def _fake_prefill(_model, admitted, *, max_steps, stats=None):
        del max_steps, stats
        admitted_batches.append([item.request_id for item in admitted])
        return SimpleNamespace(request_ids=[item.request_id for item in admitted], kv_lens=None), []

    runtime._run_prefill_active_batch = Mock(side_effect=_fake_prefill)  # type: ignore[method-assign]
    runtime._merge_active_batches = Mock(side_effect=lambda _model, left, right: right if right is not None else left)  # type: ignore[method-assign]
    runtime._decode_active_batch_one_step = Mock(side_effect=[(None, []), (None, []), (None, [])])  # type: ignore[method-assign]

    finished = runtime._run_continuous_batch_scheduler(SimpleNamespace(), states, max_steps=9)

    assert finished == []
    assert admitted_batches == [["a"], ["b"], ["c"]]
    assert runtime.get_last_t2s_scheduler_stats()["max_active_batch_limit"] == 1


def test_generate_semantic_tokens_runs_scheduler_under_no_grad(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._ensure_pipeline = Mock(return_value=SimpleNamespace(t2s_model=SimpleNamespace(model="fake-model")))  # type: ignore[method-assign]
    runtime._estimate_scheduler_max_steps = Mock(return_value=8)  # type: ignore[method-assign]

    def _fake_scheduler(model, states, max_steps):
        assert model == "fake-model"
        assert max_steps == 8
        assert torch.is_grad_enabled() is False
        assert torch.is_inference_mode_enabled() is False
        return [GPTSoVITSARFinishedItem("req-a", torch.tensor([1, 2], dtype=torch.long), 0, "done")]

    runtime._run_continuous_batch_scheduler = Mock(side_effect=_fake_scheduler)  # type: ignore[method-assign]
    prepared = [SimpleNamespace(state=SimpleNamespace(request_id="req-a"), request_id="req-a")]

    result = runtime.generate_semantic_tokens(prepared)

    assert torch.equal(result["req-a"], torch.tensor([1, 2], dtype=torch.long))


def test_generate_semantic_tokens_groups_multi_segment_request_by_parent_id(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._ensure_pipeline = Mock(return_value=SimpleNamespace(t2s_model=SimpleNamespace(model="fake-model")))  # type: ignore[method-assign]
    runtime._estimate_scheduler_max_steps = Mock(return_value=8)  # type: ignore[method-assign]
    runtime._run_continuous_batch_scheduler = Mock(  # type: ignore[method-assign]
        return_value=[
            GPTSoVITSARFinishedItem("req-a::seg0001", torch.tensor([3], dtype=torch.long), 0, "done"),
            GPTSoVITSARFinishedItem("req-a::seg0000", torch.tensor([1, 2], dtype=torch.long), 0, "done"),
        ]
    )
    segment0 = GPTSoVITST2SRequestState(
        request_id="req-a::seg0000",
        parent_request_id="req-a",
        segment_index=0,
        ref_audio_path="ref.wav",
        prompt_text="prompt",
        prompt_lang="zh",
        text="text",
        text_lang="zh",
        norm_prompt_text="prompt",
        norm_text="seg0",
        phones=torch.tensor([1], dtype=torch.long),
        prompt_phones=torch.tensor([2], dtype=torch.long),
        all_phones=torch.tensor([2, 1], dtype=torch.long),
        all_bert_features=torch.zeros((1024, 2), dtype=torch.float32),
        prompt_semantic=torch.tensor([3], dtype=torch.long),
        refer_spec=None,
        aux_refer_specs=[],
        raw_audio=torch.zeros((1,), dtype=torch.float32),
        raw_sr=16000,
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.35,
        early_stop_num=1500,
        ready_step=0,
        prepare_profile={},
    )
    segment1 = GPTSoVITST2SRequestState(
        request_id="req-a::seg0001",
        parent_request_id="req-a",
        segment_index=1,
        ref_audio_path="ref.wav",
        prompt_text="prompt",
        prompt_lang="zh",
        text="text",
        text_lang="zh",
        norm_prompt_text="prompt",
        norm_text="seg1",
        phones=torch.tensor([4], dtype=torch.long),
        prompt_phones=torch.tensor([2], dtype=torch.long),
        all_phones=torch.tensor([2, 4], dtype=torch.long),
        all_bert_features=torch.zeros((1024, 2), dtype=torch.float32),
        prompt_semantic=torch.tensor([3], dtype=torch.long),
        refer_spec=None,
        aux_refer_specs=[],
        raw_audio=torch.zeros((1,), dtype=torch.float32),
        raw_sr=16000,
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.35,
        early_stop_num=1500,
        ready_step=0,
        prepare_profile={},
    )
    prepared = [
        SimpleNamespace(
            state=GPTSoVITSMultiSegmentRequestState(
                request_id="req-a",
                prompt_phones=torch.tensor([2], dtype=torch.long),
                prompt_semantic=torch.tensor([3], dtype=torch.long),
                refer_spec=None,
                raw_audio=torch.zeros((1,), dtype=torch.float32),
                raw_sr=16000,
                segment_states=[segment0, segment1],
                prepare_profile={},
            ),
            request_id="req-a",
        )
    ]

    result = runtime.generate_semantic_tokens(prepared)

    scheduler_states = runtime._run_continuous_batch_scheduler.call_args.args[1]
    assert [state.request_id for state in scheduler_states] == ["req-a::seg0000", "req-a::seg0001"]
    assert isinstance(result["req-a"], list)
    assert [item.tolist() for item in result["req-a"]] == [[1, 2], [3]]
