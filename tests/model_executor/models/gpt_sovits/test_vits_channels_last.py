from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


RUNTIME_LIB_ROOT = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni"
    / "model_executor"
    / "models"
    / "gpt_sovits"
    / "runtime_lib"
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
    actual = block.forward_channels_last(
        x.unsqueeze(2).contiguous(memory_format=torch.channels_last)
    ).squeeze(2)

    assert torch.allclose(expected, actual, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generator_channels_last_resblocks_matches_default(monkeypatch):
    monkeypatch.setenv("GPTSOVITS_VITS_DECODER_CHANNELS_LAST_RESBLOCK", "1")
    torch.manual_seed(1234)
    generator = gpt_models.Generator(
        initial_channel=8,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        upsample_rates=[2, 2],
        upsample_initial_channel=32,
        upsample_kernel_sizes=[4, 4],
        gin_channels=4,
        is_bias=True,
    ).cuda().half().eval()
    generator.remove_weight_norm()

    x = torch.randn((2, 8, 24), device="cuda", dtype=torch.float16)
    g = torch.randn((2, 4, 1), device="cuda", dtype=torch.float16)

    monkeypatch.setenv("GPTSOVITS_VITS_DECODER_CHANNELS_LAST_RESBLOCK", "0")
    expected = generator(x, g=g)
    monkeypatch.setenv("GPTSOVITS_VITS_DECODER_CHANNELS_LAST_RESBLOCK", "1")
    actual = generator(x, g=g)

    assert torch.allclose(expected, actual, atol=5e-3, rtol=5e-3)
