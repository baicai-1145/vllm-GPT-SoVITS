from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

RUNTIME_LIB_ROOT = (
    Path(__file__).resolve().parents[4] / "vllm_omni" / "model_executor" / "models" / "gpt_sovits" / "runtime_lib"
)
GPT_SOVITS_ROOT = RUNTIME_LIB_ROOT / "GPT_SoVITS"
for path in (RUNTIME_LIB_ROOT, GPT_SOVITS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from module import models as gpt_models  # noqa: E402
from module import modules as gpt_modules  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_resblock1_forward_channels_last_matches_forward():
    torch.manual_seed(1234)
    block = gpt_modules.ResBlock1(32, kernel_size=11, dilation=(1, 3, 5)).cuda().half().eval()
    block.remove_weight_norm()
    block.prepare_channels_last_runtime()

    x = torch.randn((2, 32, 128), device="cuda", dtype=torch.float16)
    expected = block(x)
    actual = block.forward_channels_last(x.unsqueeze(2).contiguous(memory_format=torch.channels_last)).squeeze(2)

    assert torch.allclose(expected, actual, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generator_channels_last_resblocks_matches_default(monkeypatch):
    monkeypatch.setenv("GPTSOVITS_VITS_DECODER_CHANNELS_LAST_RESBLOCK", "1")
    torch.manual_seed(1234)
    generator = (
        gpt_models.Generator(
            initial_channel=8,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
            upsample_rates=[2, 2],
            upsample_initial_channel=32,
            upsample_kernel_sizes=[4, 4],
            gin_channels=4,
            is_bias=True,
        )
        .cuda()
        .half()
        .eval()
    )
    generator.remove_weight_norm()

    x = torch.randn((2, 8, 24), device="cuda", dtype=torch.float16)
    g = torch.randn((2, 4, 1), device="cuda", dtype=torch.float16)

    monkeypatch.setenv("GPTSOVITS_VITS_DECODER_CHANNELS_LAST_RESBLOCK", "0")
    expected = generator(x, g=g)
    monkeypatch.setenv("GPTSOVITS_VITS_DECODER_CHANNELS_LAST_RESBLOCK", "1")
    actual = generator(x, g=g)

    assert torch.allclose(expected, actual, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generator_forward_profiled_records_channels_last_stage_metrics(monkeypatch):
    monkeypatch.setenv("GPTSOVITS_VITS_DECODER_CHANNELS_LAST_RESBLOCK", "1")
    torch.manual_seed(1234)
    generator = (
        gpt_models.Generator(
            initial_channel=8,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
            upsample_rates=[2, 2],
            upsample_initial_channel=32,
            upsample_kernel_sizes=[4, 4],
            gin_channels=4,
            is_bias=True,
        )
        .cuda()
        .half()
        .eval()
    )
    generator.remove_weight_norm()

    x = torch.randn((2, 8, 24), device="cuda", dtype=torch.float16)
    g = torch.randn((2, 4, 1), device="cuda", dtype=torch.float16)

    _, stats = generator.forward_profiled(x, g=g)

    assert int(stats["decoder_profile_channels_last_resblock_path"]) == 1
    assert float(stats["decoder_stage_0_channels_last_prepare_ms"]) >= 0.0
    assert float(stats["decoder_stage_0_resblock_0_ms"]) >= 0.0
    assert float(stats["decoder_stage_0_resblock_0_layer_0_conv1_ms"]) >= 0.0
    assert float(stats["decoder_stage_0_resblock_0_layer_0_conv2_ms"]) >= 0.0
    assert float(stats["decoder_stage_0_resblock_0_layer_0_total_ms"]) >= 0.0
    assert float(stats["decoder_stage_0_resblock_avg_ms"]) >= 0.0
    assert float(stats["decoder_stage_0_total_ms"]) >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generator_stage_static_bucket_runtime(monkeypatch):
    monkeypatch.setenv("GPTSOVITS_VITS_DECODER_CHANNELS_LAST_RESBLOCK", "1")
    monkeypatch.setenv("GPTSOVITS_COMPILE_VITS_DEC_STAGE_STATIC_BUCKETS", "1")
    monkeypatch.setenv("GPTSOVITS_COMPILE_VITS_DEC_STAGE_STATIC_BUCKET_MIN_HITS", "1")
    monkeypatch.setenv("GPTSOVITS_COMPILE_VITS_DEC_STAGE_STATIC_BUCKET_MAX_ENTRIES", "4")
    monkeypatch.setenv("GPTSOVITS_COMPILE_VITS_DEC_STAGE_STATIC_BUCKET_STAGES", "0")
    torch.manual_seed(1234)
    generator = (
        gpt_models.Generator(
            initial_channel=8,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
            upsample_rates=[2, 2],
            upsample_initial_channel=32,
            upsample_kernel_sizes=[4, 4],
            gin_channels=4,
            is_bias=True,
        )
        .cuda()
        .half()
        .eval()
    )
    generator.remove_weight_norm()

    compile_calls = []

    def fake_compile(fn, mode, dynamic):
        compile_calls.append({"mode": mode, "dynamic": dynamic})
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    x = torch.randn((2, 8, 24), device="cuda", dtype=torch.float16)
    g = torch.randn((2, 4, 1), device="cuda", dtype=torch.float16)
    output = generator(x, g=g)
    stats = generator.get_last_stage_runtime_stats()

    assert output.shape[0] == 2
    assert int(stats["decoder_stage_static_runtime_calls"]) == 2
    assert int(stats["decoder_stage_static_runtime_compiled_calls"]) == 1
    assert int(stats["decoder_stage_static_runtime_eager_calls"]) == 1
    assert int(stats["decoder_stage_static_stage_0_compiled_calls"]) == 1
    assert compile_calls == [{"mode": "max-autotune-no-cudagraphs", "dynamic": False}]


def test_synthesizer_enc_p_static_bucket_runtime(monkeypatch):
    monkeypatch.setenv("GPTSOVITS_COMPILE_VITS_ENC_P_STATIC_BUCKETS", "1")
    monkeypatch.setenv("GPTSOVITS_COMPILE_VITS_ENC_P_STATIC_BUCKET_MIN_HITS", "2")
    monkeypatch.setenv("GPTSOVITS_COMPILE_VITS_ENC_P_STATIC_BUCKET_MAX_ENTRIES", "1")

    model = gpt_models.SynthesizerTrn(
        spec_channels=4,
        segment_size=4,
        inter_channels=4,
        hidden_channels=4,
        filter_channels=4,
        n_heads=1,
        n_layers=1,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[(1, 3, 5)],
        upsample_rates=[2],
        upsample_initial_channel=8,
        upsample_kernel_sizes=[4],
        gin_channels=4,
        semantic_frame_rate="25hz",
        freeze_quantizer=True,
        version="v2",
    ).eval()

    class FakeEncP(torch.nn.Module):
        def forward(self, quantized, y_lengths, text, text_lengths, ge, speed):
            del y_lengths, text, text_lengths, speed
            base = quantized + ge[..., : quantized.shape[-1]]
            return base, base + 1, base + 2, torch.ones_like(base), base + 3, base + 4

    compile_calls = []

    def fake_compile(fn, mode, dynamic):
        compile_calls.append({"mode": mode, "dynamic": dynamic})

        def compiled(*args, **kwargs):
            outputs = fn(*args, **kwargs)
            return tuple(item + 10 if torch.is_tensor(item) else item for item in outputs)

        return compiled

    monkeypatch.setattr(model, "enc_p", FakeEncP())
    monkeypatch.setattr(torch, "compile", fake_compile)

    quantized = torch.randn((1, 4, 6), dtype=torch.float32)
    y_lengths = torch.tensor([6], dtype=torch.long)
    text = torch.randint(0, 8, (1, 5), dtype=torch.long)
    text_lengths = torch.tensor([5], dtype=torch.long)
    ge = torch.randn((1, 4, 6), dtype=torch.float32)

    eager_outputs = model.run_enc_p_runtime(quantized, y_lengths, text, text_lengths, ge, 1.0)
    eager_stats = model.get_last_enc_p_runtime_stats()

    compiled_outputs = model.run_enc_p_runtime(quantized, y_lengths, text, text_lengths, ge, 1.0)
    compiled_stats = model.get_last_enc_p_runtime_stats()

    assert eager_stats["enc_p_runtime_path"] == "eager"
    assert int(eager_stats["enc_p_runtime_static_bucket_misses"]) == 1
    assert int(eager_stats["enc_p_runtime_compiled_calls"]) == 0
    assert compiled_stats["enc_p_runtime_path"] == "compiled_static_bucket"
    assert int(compiled_stats["enc_p_runtime_static_bucket_hits"]) == 1
    assert int(compiled_stats["enc_p_runtime_compiled_calls"]) == 1
    assert float(compiled_stats["enc_p_runtime_static_bucket_compile_ms"]) >= 0.0
    assert compile_calls == [{"mode": "max-autotune-no-cudagraphs", "dynamic": False}]
    assert torch.allclose(compiled_outputs[0], eager_outputs[0] + 10)
