import os
import pathlib
from typing import Optional

import torch
import torch.nn as nn
from torch.utils import cpp_extension


_EXTENSION = None
_EXTENSION_ERROR: Optional[Exception] = None
_EXTENSION_LOAD_ATTEMPTED = False


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def is_enabled() -> bool:
    return _env_flag("GPTSOVITS_CNHUBERT_FFN_CUSTOM_OP", False)


def _build_directory() -> pathlib.Path:
    build_dir = pathlib.Path(__file__).resolve().parent / "csrc" / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir


def _load_extension():
    global _EXTENSION, _EXTENSION_ERROR, _EXTENSION_LOAD_ATTEMPTED
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_LOAD_ATTEMPTED:
        return None
    _EXTENSION_LOAD_ATTEMPTED = True
    try:
        src_dir = pathlib.Path(__file__).resolve().parent / "csrc"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "")
        _EXTENSION = cpp_extension.load(
            name="gptsovits_cnhubert_ffn_cuda",
            sources=[
                str(src_dir / "cnhubert_ffn_binding.cpp"),
                str(src_dir / "cnhubert_ffn_cuda.cu"),
            ],
            build_directory=str(_build_directory()),
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ],
            extra_ldflags=["-lcublas", "-lcublasLt"],
            verbose=_env_flag("GPTSOVITS_CNHUBERT_FFN_CUSTOM_OP_VERBOSE", False),
        )
    except Exception as exc:
        _EXTENSION_ERROR = exc
        print(f"[cnhubert_ffn_cuda] extension unavailable, fallback to torch path: {exc}")
        return None
    return _EXTENSION


class HubertFeedForwardCuda(nn.Module):
    def __init__(self, original: nn.Module):
        super().__init__()
        self.intermediate_dropout = original.intermediate_dropout
        self.intermediate_dense = original.intermediate_dense
        self.intermediate_act_fn = original.intermediate_act_fn
        self.output_dense = original.output_dense
        self.output_dropout = original.output_dropout

    def _fallback(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._fallback(hidden_states)
        if hidden_states.device.type != "cuda":
            return self._fallback(hidden_states)
        if hidden_states.dtype != torch.float16:
            return self._fallback(hidden_states)
        ext = _load_extension()
        if ext is None:
            return self._fallback(hidden_states)
        try:
            return ext.forward(
                hidden_states.contiguous(),
                self.intermediate_dense.weight.contiguous(),
                self.intermediate_dense.bias.contiguous(),
                self.output_dense.weight.contiguous(),
                self.output_dense.bias.contiguous(),
            )
        except Exception as exc:
            global _EXTENSION_ERROR
            if _EXTENSION_ERROR is None:
                _EXTENSION_ERROR = exc
                print(f"[cnhubert_ffn_cuda] runtime failed, fallback to torch path: {exc}")
            return self._fallback(hidden_states)


def maybe_patch_hubert_feed_forward(model: nn.Module) -> bool:
    if not is_enabled():
        return False
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        return False
    patched = 0
    for layer in layers:
        feed_forward = getattr(layer, "feed_forward", None)
        if feed_forward is None or isinstance(feed_forward, HubertFeedForwardCuda):
            continue
        layer.feed_forward = HubertFeedForwardCuda(feed_forward)
        patched += 1
    if patched > 0:
        print(f"[cnhubert_ffn_cuda] patched {patched} HubertFeedForward blocks")
    return patched > 0
