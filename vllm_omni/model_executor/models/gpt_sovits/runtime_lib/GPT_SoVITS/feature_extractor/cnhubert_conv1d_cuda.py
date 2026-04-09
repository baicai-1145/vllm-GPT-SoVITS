import os
import pathlib

import torch
import torch.nn as nn
from torch.utils import cpp_extension

_EXTENSION = None
_EXTENSION_ERROR: Exception | None = None
_EXTENSION_LOAD_ATTEMPTED = False


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def is_enabled() -> bool:
    return _env_flag("GPTSOVITS_CNHUBERT_CONV1D_CUSTOM_OP", False)


def _target_layer_indices() -> set[int]:
    raw = str(os.environ.get("GPTSOVITS_CNHUBERT_CONV1D_CUSTOM_LAYER_INDICES", "1")).strip()
    values: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.add(int(item))
        except ValueError:
            continue
    if not values:
        values.add(1)
    return values


def _max_input_elements() -> int:
    raw = str(os.environ.get("GPTSOVITS_CNHUBERT_CONV1D_MAX_INPUT_ELEMENTS", "64000000")).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 64000000


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
            name="gptsovits_cnhubert_conv1d_cuda",
            sources=[
                str(src_dir / "cnhubert_conv1d_binding.cpp"),
                str(src_dir / "cnhubert_conv1d_cuda.cu"),
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
            extra_ldflags=["-lcudnn"],
            verbose=_env_flag("GPTSOVITS_CNHUBERT_CONV1D_CUSTOM_OP_VERBOSE", False),
        )
    except Exception as exc:
        _EXTENSION_ERROR = exc
        print(f"[cnhubert_conv1d_cuda] extension unavailable, fallback to torch path: {exc}")
        return None
    return _EXTENSION


class Conv1dCuda(nn.Module):
    def __init__(self, original: nn.Conv1d):
        super().__init__()
        self.original = original

    def _fallback(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.original(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._fallback(hidden_states)
        if hidden_states.device.type != "cuda":
            return self._fallback(hidden_states)
        if hidden_states.dtype != torch.float16:
            return self._fallback(hidden_states)
        if int(hidden_states.numel()) > _max_input_elements():
            return self._fallback(hidden_states)
        if self.original.bias is not None:
            return self._fallback(hidden_states)
        ext = _load_extension()
        if ext is None:
            return self._fallback(hidden_states)
        try:
            return ext.forward(
                hidden_states.contiguous(),
                self.original.weight.contiguous(),
                int(self.original.stride[0]),
                int(self.original.padding[0]),
                int(self.original.dilation[0]),
                int(self.original.groups),
            )
        except Exception as exc:
            global _EXTENSION_ERROR
            if _EXTENSION_ERROR is None:
                _EXTENSION_ERROR = exc
                print(f"[cnhubert_conv1d_cuda] runtime failed, fallback to torch path: {exc}")
            return self._fallback(hidden_states)


def maybe_patch_hubert_conv_layers(model: nn.Module) -> bool:
    if not is_enabled():
        return False
    feature_extractor = getattr(model, "feature_extractor", None)
    conv_layers = getattr(feature_extractor, "conv_layers", None)
    if conv_layers is None:
        return False
    target_indices = _target_layer_indices()
    patched_indices: list[int] = []
    for layer_index, layer in enumerate(conv_layers):
        if layer_index not in target_indices:
            continue
        conv = getattr(layer, "conv", None)
        if conv is None or isinstance(conv, Conv1dCuda) or not isinstance(conv, nn.Conv1d):
            continue
        layer.conv = Conv1dCuda(conv)
        patched_indices.append(layer_index)
    if patched_indices:
        print(f"[cnhubert_conv1d_cuda] patched conv_layers={patched_indices}")
    return bool(patched_indices)
