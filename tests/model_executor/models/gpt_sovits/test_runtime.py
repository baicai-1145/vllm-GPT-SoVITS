from __future__ import annotations

import time
from types import SimpleNamespace
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
    GPTSoVITSDecodedAudio,
    GPTSoVITSNativePreparedCpuStage,
    GPTSoVITSPreparedAudioPhase,
    GPTSoVITSPreparedCpuStage,
    GPTSoVITSPrepareProfiledResult,
    GPTSoVITSPreparedRefSpecPhase,
    GPTSoVITSPreparedTextPhase,
    GPTSoVITSRuntime,
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


def test_prepare_decode_request_moves_transport_to_pipeline_device(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._pipeline = SimpleNamespace(configs=SimpleNamespace(device="cpu", sampling_rate=32000))

    prepared = runtime.prepare_decode_request(
        torch.tensor([10, 11], dtype=torch.int32),
        {
            "gpt_sovits_request_id": "decode_req",
            "gpt_sovits_phones": torch.tensor([1, 2], dtype=torch.int32),
            "gpt_sovits_prompt_phones": torch.tensor([3], dtype=torch.int32),
            "gpt_sovits_prompt_semantic": torch.tensor([4], dtype=torch.int32),
            "gpt_sovits_refer_audio_spec": torch.tensor([0.1], dtype=torch.float16),
            "gpt_sovits_refer_audio_16k": torch.tensor([0.2], dtype=torch.float16),
            "gpt_sovits_raw_audio": torch.tensor([0.3], dtype=torch.float64),
            "gpt_sovits_raw_sr": 16000,
            "gpt_sovits_speed_factor": 1.25,
            "gpt_sovits_sample_steps": 48,
            "gpt_sovits_super_sampling": True,
        },
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
            GPTSoVITSDecodedAudio("a", "audio-a", 32000, 1.0, False),
            GPTSoVITSDecodedAudio("b", "audio-b", 32000, 1.0, False),
        ]
    )
    runtime.decode_prepared_request = Mock(  # type: ignore[method-assign]
        return_value=GPTSoVITSDecodedAudio("c", "audio-c", 32000, 1.5, False)
    )

    decoded = runtime.decode_prepared_requests(
        [
            SimpleNamespace(request_id="a", semantic_tokens=torch.tensor([1]), speed_factor=1.0, sample_steps=32, super_sampling=False),
            SimpleNamespace(request_id="b", semantic_tokens=torch.tensor([2]), speed_factor=1.0, sample_steps=32, super_sampling=False),
            SimpleNamespace(request_id="c", semantic_tokens=torch.tensor([3]), speed_factor=1.5, sample_steps=32, super_sampling=False),
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
        side_effect=lambda prepared: (prepared.refer_audio_spec, None)
    )
    runtime._build_vocoder_prompt_cache_key = Mock(  # type: ignore[method-assign]
        side_effect=lambda prepared, refer_audio_spec: shared_prompt_key if prepared.request_id in {"a", "b"} else other_prompt_key
    )
    runtime._decode_prepared_requests_grouped_vocoder = Mock(  # type: ignore[method-assign]
        return_value=[
            GPTSoVITSDecodedAudio("a", "audio-a", 24000, 1.0, False),
            GPTSoVITSDecodedAudio("b", "audio-b", 24000, 1.5, True),
        ]
    )
    runtime.decode_prepared_request = Mock(  # type: ignore[method-assign]
        return_value=GPTSoVITSDecodedAudio("c", "audio-c", 24000, 0.8, False)
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


def test_decode_prepared_requests_batched_non_vocoder_uses_native_vits_batch(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    vits_model = SimpleNamespace(
        version="v2",
        ref_enc=Mock(return_value=torch.ones((1, 2, 3), dtype=torch.float32)),
        decode_batched_request_local=Mock(
            return_value=(
                torch.tensor(
                    [
                        [[0.1, 0.2, 0.3]],
                        [[0.4, 0.5, 0.0]],
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor([3, 2], dtype=torch.long),
            )
        ),
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

    vits_model.decode_batched_request_local.assert_called_once()
    assert [item.request_id for item in decoded] == ["a", "b"]
    assert decoded[0].audio_fragment.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert decoded[1].audio_fragment.tolist() == pytest.approx([0.4, 0.5])


def test_decode_prepared_request_vocoder_fragment_uses_native_components(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    runtime._compute_vocoder_mel = Mock(return_value=torch.ones((1, 100, 6), dtype=torch.float32))  # type: ignore[method-assign]
    runtime._resample_audio = Mock(side_effect=lambda audio, sr0, sr1, device: audio)  # type: ignore[method-assign]

    decode_encp_calls = [
        (torch.ones((1, 100, 6), dtype=torch.float32), torch.ones((1, 2, 1), dtype=torch.float32)),
        (torch.ones((1, 100, 8), dtype=torch.float32), torch.ones((1, 2, 1), dtype=torch.float32)),
    ]
    vits_model = SimpleNamespace(
        decode_encp=Mock(side_effect=decode_encp_calls),
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

    assert vits_model.decode_encp.call_count == 2
    vits_model.cfm.inference.assert_called()
    pipeline.vocoder.assert_called_once()
    assert audio_fragment.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert output_sr == 24000


def test_decode_prepared_requests_grouped_vocoder_reuses_prompt_context_once(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = SimpleNamespace(configs=SimpleNamespace(device="cpu"))
    prompt_context = {"output_sr": 24000}
    runtime._build_refer_spec_from_prepared = Mock(  # type: ignore[method-assign]
        side_effect=lambda prepared: (prepared.refer_audio_spec, None)
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
        side_effect=lambda prepared: (prepared.refer_audio_spec, None)
    )
    runtime._build_vocoder_prompt_context = Mock(return_value=prompt_context)  # type: ignore[method-assign]
    runtime._decode_prepared_requests_batched_vocoder = Mock(  # type: ignore[method-assign]
        return_value=[
            GPTSoVITSDecodedAudio("a", "audio-a", 24000, 1.0, False),
            GPTSoVITSDecodedAudio("b", "audio-b", 24000, 1.0, True),
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
    ge = torch.ones((1, 2, 1), dtype=torch.float32)
    decode_encp_calls = [
        (torch.ones((1, 100, 3), dtype=torch.float32), ge),
        (torch.ones((1, 100, 2), dtype=torch.float32) * 2, ge),
    ]
    vits_model = SimpleNamespace(
        decode_encp=Mock(side_effect=decode_encp_calls),
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

    assert vits_model.decode_encp.call_count == 2
    vits_model.cfm.inference.assert_called_once()
    pipeline.vocoder.assert_called_once()
    runtime._sola_merge_audio_fragments.assert_called_once()
    assert [item.request_id for item in decoded] == ["a", "b"]
    assert decoded[0].audio_fragment.tolist() == pytest.approx([10.0, 11.0, 12.0])
    assert decoded[1].audio_fragment.tolist() == pytest.approx([13.0, 14.0])
    assert decoded[1].super_sampling is True


def test_prepare_request_cpu_stage_uses_prepare_coordinator_and_ref_audio_preload(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    spec = SimpleNamespace(request_id="req-1", ref_audio_path="/tmp/ref.wav")
    cpu_stage = SimpleNamespace(stage="cpu")
    coordinator = SimpleNamespace()
    runtime._ensure_prepare_coordinator = Mock(return_value=coordinator)  # type: ignore[method-assign]
    runtime._build_scheduler_request_spec = Mock(return_value=spec)  # type: ignore[method-assign]
    runtime.preload_ref_audio_asset = Mock(return_value="future")  # type: ignore[method-assign]
    runtime._prepare_cpu_stage_async = Mock(return_value="awaitable")  # type: ignore[method-assign]
    runtime._run_awaitable_sync = Mock(return_value=cpu_stage)  # type: ignore[method-assign]

    prepared = runtime.prepare_request_cpu_stage({"text": "hello"}, request_id="req-1")

    runtime.preload_ref_audio_asset.assert_called_once_with("/tmp/ref.wav", submit_at=ANY)
    runtime._prepare_cpu_stage_async.assert_called_once()
    assert prepared.request_id == "req-1"
    assert prepared.spec is spec
    assert prepared.cpu_stage is cpu_stage
    assert prepared.ref_audio_prepare_future == "future"


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
        coordinator,
        "normalized prompt",
        "zh",
        "target",
        "zh",
    )
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
    runtime._prepare_gpu_audio_phase_async = AsyncMock(return_value={"phase": "audio"})  # type: ignore[method-assign]
    runtime._prepare_ref_spec_phase_async = AsyncMock(return_value=(("spec", "wav16k"), {"ref_spec_ms": 1.0}))  # type: ignore[method-assign]
    runtime._prepare_gpu_text_phase_async = AsyncMock(return_value={"phase": "text"})  # type: ignore[method-assign]
    runtime._build_request_state_from_prepare_phases = Mock(return_value=SimpleNamespace(state="ok"))  # type: ignore[method-assign]
    runtime._state_to_transport_info = Mock(return_value={"transport": "ok"})  # type: ignore[method-assign]

    prepared_cpu = GPTSoVITSPreparedCpuStage(
        request_id="req-2",
        request={"text": "hello"},
        spec=SimpleNamespace(request_id="req-2", ref_audio_path="/tmp/ref.wav"),
        cpu_stage=SimpleNamespace(stage="cpu"),
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
    assert prepared_audio.phase_one == {"phase": "audio"}
    assert isinstance(prepared_ref_spec, GPTSoVITSPreparedRefSpecPhase)
    assert prepared_ref_spec.ref_spec_result == (("spec", "wav16k"), {"ref_spec_ms": 1.0})
    assert isinstance(prepared_text, GPTSoVITSPreparedTextPhase)
    assert prepared_text.phase_two == {"phase": "text"}
    runtime._build_request_state_from_prepare_phases.assert_called_once_with(
        prepared_cpu,
        {"phase": "audio"},
        {"phase": "text"},
        ref_spec_result=(("spec", "wav16k"), {"ref_spec_ms": 1.0}),
        extra_profile=None,
    )
    coordinator._release_split_stage_slot.assert_called_once_with()
    runtime._state_to_transport_info.assert_called_once_with(prepared_request.state, {"text": "hello"})
    assert prepared_request.request_id == "req-2"
    assert prepared_request.transport_info == {"transport": "ok"}


def test_build_request_state_from_prepare_phases_uses_runtime_native_state_assembly(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    pipeline = SimpleNamespace(
        configs=SimpleNamespace(device="cpu"),
        precision=torch.float16,
        prepare_text_cpu_workers=3,
        extract_ref_spec=Mock(return_value=("aux-spec", "aux-16k", None, None)),
    )
    runtime._ensure_pipeline = Mock(return_value=pipeline)  # type: ignore[method-assign]
    runtime._build_prepare_profile_overrides = Mock(return_value={"custom_metric": 7.0})  # type: ignore[method-assign]
    runtime._build_ref_audio_bundle_from_phase = Mock(  # type: ignore[method-assign]
        return_value={
            "prompt_semantic": torch.tensor([5, 6], dtype=torch.int32),
            "refer_spec": (torch.ones((1, 5, 4), dtype=torch.float32), torch.ones((1, 160), dtype=torch.float32)),
            "raw_audio": torch.ones((1, 12), dtype=torch.float32),
            "raw_sr": 16000,
            "profile": {
                "bundle_total_ms": 4.0,
                "prompt_semantic_ms": 1.5,
                "ref_spec_ms": 2.0,
                "audio_load_ms": 0.5,
            },
        }
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
        request={"text": "hello"},
        spec=SimpleNamespace(request_id="req-native"),
        cpu_stage=SimpleNamespace(
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
            prompt_text="prompt",
            text="target",
            prepare_start=time.time() - 0.01,
        ),
    )
    (tmp_path / "aux.wav").write_bytes(b"wav")

    state = runtime._build_request_state_from_prepare_phases(
        prepared_cpu_stage,
        {"phase": "audio"},
        {
            "prompt_feature_profiled": SimpleNamespace(result=prompt_result),
            "target_feature_profiled": SimpleNamespace(result=target_result),
        },
    )

    assert state.request_id == "req-native"
    assert state.norm_prompt_text == "prompt-norm"
    assert state.norm_text == "target-norm"
    assert state.phones.tolist() == [3, 4, 5]
    assert state.prompt_phones.tolist() == [1, 2]
    assert state.all_phones.tolist() == [1, 2, 3, 4, 5]
    assert state.prompt_semantic.dtype == torch.long
    assert state.refer_spec is not None
    assert state.raw_sr == 16000
    assert state.top_k == 5
    assert state.ready_step == 4
    assert state.prepare_profile["custom_metric"] == 7.0
    assert "executor_run_wall_ms" in state.prepare_profile
    assert state.all_bert_features.dtype == torch.float16
    assert len(state.aux_refer_specs) == 1
    pipeline.extract_ref_spec.assert_called_once()


def test_prepare_gpu_audio_phase_async_uses_runtime_native_prepare_kernels(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    prompt_g2pw = GPTSoVITSPrepareProfiledResult(result=["prompt-g2pw"], submit_at=1.0, started_at=1.0, finished_at=2.0)
    target_g2pw = GPTSoVITSPrepareProfiledResult(result=["target-g2pw"], submit_at=1.0, started_at=1.0, finished_at=2.5)
    ref_audio = GPTSoVITSPrepareProfiledResult(
        result={"raw_audio": torch.tensor([0.1]), "raw_sr": 16000, "profile": {}},
        submit_at=1.0,
        started_at=1.0,
        finished_at=3.0,
    )
    runtime._run_g2pw_pair_stage = AsyncMock(return_value=(prompt_g2pw, target_g2pw))  # type: ignore[method-assign]
    runtime._run_ref_prompt_semantic_stage = AsyncMock(return_value=ref_audio)  # type: ignore[method-assign]

    prepared_cpu = GPTSoVITSPreparedCpuStage(
        request_id="req-audio",
        request={},
        spec=SimpleNamespace(ref_audio_path="/tmp/ref.wav"),
        cpu_stage=SimpleNamespace(
            spec=SimpleNamespace(ref_audio_path="/tmp/ref.wav"),
            prompt_cpu_profiled=SimpleNamespace(result=["prompt"]),
            target_cpu_profiled=SimpleNamespace(result=["target"]),
        ),
        ref_audio_prepare_future="future",
    )

    phase_one = runtime._run_awaitable_sync(runtime._prepare_gpu_audio_phase_async(SimpleNamespace(), prepared_cpu))

    runtime._run_g2pw_pair_stage.assert_awaited_once()
    runtime._run_ref_prompt_semantic_stage.assert_awaited_once_with(
        ANY,
        "/tmp/ref.wav",
        prepared_asset_future="future",
    )
    assert phase_one["prompt_g2pw_profiled"] is prompt_g2pw
    assert phase_one["target_g2pw_profiled"] is target_g2pw
    assert phase_one["ref_audio_profiled"] is ref_audio


def test_prepare_request_gpu_audio_phases_batch_merge_uses_runtime_native_batch_helper(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    coordinator = SimpleNamespace(enable_g2pw_audio_batch_merge=True)
    runtime._ensure_prepare_coordinator = Mock(return_value=coordinator)  # type: ignore[method-assign]
    runtime._prepare_gpu_audio_phase_batch_async = AsyncMock(return_value=[{"phase": "a"}, {"phase": "b"}])  # type: ignore[method-assign]

    prepared_audio = runtime.prepare_request_gpu_audio_phases(
        [
            GPTSoVITSPreparedCpuStage("req-a", {}, SimpleNamespace(), SimpleNamespace(), None),
            GPTSoVITSPreparedCpuStage("req-b", {}, SimpleNamespace(), SimpleNamespace(), None),
        ]
    )

    runtime._prepare_gpu_audio_phase_batch_async.assert_awaited_once_with(
        coordinator,
        ANY,
    )
    assert [item.phase_one for item in prepared_audio] == [{"phase": "a"}, {"phase": "b"}]


def test_prepare_request_uses_explicit_split_prepare_pipeline(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    prepared_cpu = SimpleNamespace(name="cpu")
    prepared_audio = SimpleNamespace(name="audio")
    prepared_ref_spec = SimpleNamespace(name="ref_spec")
    prepared_text = SimpleNamespace(name="text")
    prepared_request = SimpleNamespace(name="prepared")
    runtime.prepare_request_cpu_stage = Mock(return_value=prepared_cpu)  # type: ignore[method-assign]
    runtime.prepare_request_gpu_audio_phase = Mock(return_value=prepared_audio)  # type: ignore[method-assign]
    runtime.prepare_request_ref_spec_phase = Mock(return_value=prepared_ref_spec)  # type: ignore[method-assign]
    runtime.prepare_request_gpu_text_phase = Mock(return_value=prepared_text)  # type: ignore[method-assign]
    runtime.build_prepared_request_from_phases = Mock(return_value=prepared_request)  # type: ignore[method-assign]

    result = runtime.prepare_request({"text": "hello"}, request_id="req-3")

    runtime.prepare_request_cpu_stage.assert_called_once_with({"text": "hello"}, request_id="req-3")
    runtime.prepare_request_gpu_audio_phase.assert_called_once_with(prepared_cpu)
    runtime.prepare_request_ref_spec_phase.assert_called_once_with(prepared_audio)
    runtime.prepare_request_gpu_text_phase.assert_called_once_with(prepared_audio)
    runtime.build_prepared_request_from_phases.assert_called_once_with(
        prepared_text,
        prepared_ref_spec_phase=prepared_ref_spec,
    )
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
            request={},
            spec=SimpleNamespace(),
            cpu_stage=SimpleNamespace(
                prompt_cpu_profiled=SimpleNamespace(run_ms=5.0),
                target_cpu_profiled=SimpleNamespace(run_ms=6.0),
            ),
            ref_audio_prepare_future=None,
        ),
        phase_one={
            "prompt_g2pw_profiled": SimpleNamespace(result=["prompt"], profile={"a": 1.0}),
            "target_g2pw_profiled": SimpleNamespace(result=["target"], profile={"b": 2.0}),
            "ref_audio_profiled": GPTSoVITSPrepareProfiledResult(
                result={"raw_audio": torch.tensor([0.2]), "raw_sr": 16000, "profile": {}},
                submit_at=1.0,
                started_at=1.0,
                finished_at=1.0,
            ),
        },
    )

    phase_two = runtime._run_awaitable_sync(runtime._prepare_gpu_text_phase_async(coordinator, prepared_audio))
    ref_spec = runtime._run_awaitable_sync(runtime._prepare_ref_spec_phase_async(coordinator, prepared_audio))

    runtime._run_text_feature_pair_stage.assert_awaited_once_with(
        coordinator,
        ["prompt"],
        ["target"],
        5.0,
        6.0,
        prompt_base_profile={"a": 1.0},
        target_base_profile={"b": 2.0},
    )
    runtime._run_ref_spec_stage.assert_awaited_once_with(coordinator, ANY, 16000)
    assert phase_two["prompt_feature_profiled"] == "prompt-feature"
    assert phase_two["target_feature_profiled"] == "target-feature"
    assert ref_spec[0] == ("spec", "wav16k")
    assert ref_spec[1]["ref_spec_wait_ms"] == 500.0
    assert ref_spec[1]["ref_spec_ms"] == 1500.0


def test_prepare_request_cpu_stages_batches_cpu_prepare_calls(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    coordinator = SimpleNamespace()
    runtime._ensure_prepare_coordinator = Mock(return_value=coordinator)  # type: ignore[method-assign]
    runtime._build_scheduler_request_spec = Mock(side_effect=[  # type: ignore[method-assign]
        SimpleNamespace(request_id="req-a", ref_audio_path="/tmp/a.wav"),
        SimpleNamespace(request_id="req-b", ref_audio_path="/tmp/b.wav"),
    ])
    runtime.preload_ref_audio_asset = Mock(side_effect=["future-a", "future-b"])  # type: ignore[method-assign]
    runtime._prepare_cpu_stage_async = AsyncMock(side_effect=["cpu-a", "cpu-b"])  # type: ignore[method-assign]

    prepared = runtime.prepare_request_cpu_stages([{"text": "a"}, {"text": "b", "engine_request_id": "req-b"}])

    assert len(prepared) == 2
    assert prepared[0].request_id == "req-a"
    assert prepared[0].cpu_stage == "cpu-a"
    assert prepared[0].ref_audio_prepare_future == "future-a"
    assert prepared[1].request_id == "req-b"
    assert prepared[1].cpu_stage == "cpu-b"
    assert prepared[1].ref_audio_prepare_future == "future-b"
    assert runtime._prepare_cpu_stage_async.await_count == 2


def test_prepare_requests_uses_batched_split_prepare_pipeline(tmp_path):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    prepared_cpu_stages = [SimpleNamespace(name="cpu-a"), SimpleNamespace(name="cpu-b")]
    prepared_audio_phases = [SimpleNamespace(name="audio-a"), SimpleNamespace(name="audio-b")]
    prepared_ref_spec_phases = [SimpleNamespace(name="ref-a"), SimpleNamespace(name="ref-b")]
    prepared_text_phases = [SimpleNamespace(name="text-a"), SimpleNamespace(name="text-b")]
    prepared_requests = [SimpleNamespace(name="prepared-a"), SimpleNamespace(name="prepared-b")]

    runtime.prepare_request_cpu_stages = Mock(return_value=prepared_cpu_stages)  # type: ignore[method-assign]
    runtime.prepare_request_gpu_audio_phases = Mock(return_value=prepared_audio_phases)  # type: ignore[method-assign]
    runtime.prepare_request_ref_spec_phases = Mock(return_value=prepared_ref_spec_phases)  # type: ignore[method-assign]
    runtime.prepare_request_gpu_text_phases = Mock(return_value=prepared_text_phases)  # type: ignore[method-assign]
    runtime.build_prepared_request_from_phases = Mock(side_effect=prepared_requests)  # type: ignore[method-assign]

    requests = [{"text": "a"}, {"text": "b"}]
    result = runtime.prepare_requests(requests)

    runtime.prepare_request_cpu_stages.assert_called_once_with(requests)
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

    worker.submit_many_async.assert_awaited_once_with([("prompt", "zh"), ("target", "zh")])
    assert prompt_profiled.result == ["prompt-seg"]
    assert prompt_profiled.queue_ms == pytest.approx(3.0, abs=5.0)
    assert target_profiled.result == ["target-seg"]
    assert target_profiled.run_ms == pytest.approx(6.0, abs=1e-6)


def test_build_ar_session_routes_through_runtime_native_ar_helpers(tmp_path, monkeypatch):
    runtime = GPTSoVITSRuntime(project_root=str(tmp_path), config_path=str(tmp_path / "dummy.yaml"))
    monkeypatch.setenv("GPT_SOVITS_ENABLE_AR_KV_POOL", "1")

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
    admitted_batch = SimpleNamespace(name="admitted")
    merged_batch = SimpleNamespace(name="merged")
    finished_prefill = [GPTSoVITSARFinishedItem("a", torch.tensor([1]), 0, "prefill")]
    finished_decode = [GPTSoVITSARFinishedItem("b", torch.tensor([2]), 1, "decode")]
    runtime._run_prefill_active_batch = Mock(side_effect=[(admitted_batch, []), (None, finished_prefill)])  # type: ignore[method-assign]
    runtime._merge_active_batches = Mock(side_effect=[merged_batch, merged_batch])  # type: ignore[method-assign]
    runtime._decode_active_batch_one_step = Mock(side_effect=[(None, finished_decode), (None, [])])  # type: ignore[method-assign]

    finished = runtime._run_continuous_batch_scheduler(SimpleNamespace(), [state_late, state_now], max_steps=9)

    assert [item.request_id for item in finished] == ["a", "b"]
    assert runtime._run_prefill_active_batch.call_args_list[0].args[1] == [state_now]
    assert runtime._run_prefill_active_batch.call_args_list[1].args[1] == [state_late]


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
