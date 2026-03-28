from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.gpt_sovits.runtime import GPTSoVITSRuntime


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
