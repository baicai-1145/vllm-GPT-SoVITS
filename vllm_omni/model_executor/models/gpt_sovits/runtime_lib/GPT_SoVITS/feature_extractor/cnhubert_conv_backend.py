import os

import torch
import torch.nn as nn


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _backend_preset() -> str:
    return str(os.environ.get("GPTSOVITS_CNHUBERT_BACKEND_PRESET", "")).strip().lower()


def apply_backend_preset_env_defaults() -> None:
    preset = _backend_preset()
    if preset in {"", "exact", "baseline"}:
        return
    if preset in {"scoped_l1", "fast"}:
        os.environ.setdefault("GPTSOVITS_CNHUBERT_DISABLE_CUDNN_LAYER_INDICES", "1")
        return
    if preset == "aggressive":
        os.environ.setdefault("GPTSOVITS_CNHUBERT_DISABLE_CUDNN_LAYER_INDICES", "1")
        os.environ.setdefault("GPTSOVITS_CNHUBERT_MODEL_CUDNN_BENCHMARK", "1")
        return
    if preset in {"aggressive_tf32", "unsafe_tf32"}:
        os.environ.setdefault("GPTSOVITS_CNHUBERT_DISABLE_CUDNN_LAYER_INDICES", "1")
        os.environ.setdefault("GPTSOVITS_CNHUBERT_MODEL_CUDNN_BENCHMARK", "1")
        os.environ.setdefault("GPTSOVITS_CNHUBERT_MODEL_CUDNN_ALLOW_TF32", "1")
        return
    if preset == "global_off":
        os.environ.setdefault("GPTSOVITS_CNHUBERT_MODEL_CUDNN_ENABLED", "0")
        return


def _target_layer_indices() -> set[int]:
    raw = str(os.environ.get("GPTSOVITS_CNHUBERT_DISABLE_CUDNN_LAYER_INDICES", "")).strip()
    values: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.add(int(item))
        except ValueError:
            continue
    return values


class Conv1dScopedCudnnDisabled(nn.Module):
    def __init__(self, original: nn.Conv1d):
        super().__init__()
        self.original = original

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.device.type != "cuda":
            return self.original(hidden_states)
        with torch.backends.cudnn.flags(enabled=False):
            return self.original(hidden_states)


class ModuleScopedCudnnFlags(nn.Module):
    def __init__(
        self,
        original: nn.Module,
        *,
        enabled: bool | None = None,
        benchmark: bool | None = None,
        deterministic: bool | None = None,
        allow_tf32: bool | None = None,
    ):
        super().__init__()
        self.original = original
        self.enabled = enabled
        self.benchmark = benchmark
        self.deterministic = deterministic
        self.allow_tf32 = allow_tf32

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original, name)

    def forward(self, *args, **kwargs):
        first_tensor = None
        for value in args:
            if isinstance(value, torch.Tensor):
                first_tensor = value
                break
        if first_tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    first_tensor = value
                    break
        if first_tensor is None or first_tensor.device.type != "cuda":
            return self.original(*args, **kwargs)
        with torch.backends.cudnn.flags(
            enabled=self.enabled,
            benchmark=self.benchmark,
            deterministic=self.deterministic,
            allow_tf32=self.allow_tf32,
        ):
            return self.original(*args, **kwargs)


def maybe_wrap_hubert_model_backend(model: nn.Module) -> nn.Module:
    enabled = os.environ.get("GPTSOVITS_CNHUBERT_MODEL_CUDNN_ENABLED")
    benchmark = os.environ.get("GPTSOVITS_CNHUBERT_MODEL_CUDNN_BENCHMARK")
    deterministic = os.environ.get("GPTSOVITS_CNHUBERT_MODEL_CUDNN_DETERMINISTIC")
    allow_tf32 = os.environ.get("GPTSOVITS_CNHUBERT_MODEL_CUDNN_ALLOW_TF32")
    if enabled is None and benchmark is None and deterministic is None and allow_tf32 is None:
        return model
    wrapped = ModuleScopedCudnnFlags(
        model,
        enabled=_env_flag("GPTSOVITS_CNHUBERT_MODEL_CUDNN_ENABLED", True) if enabled is not None else None,
        benchmark=_env_flag("GPTSOVITS_CNHUBERT_MODEL_CUDNN_BENCHMARK", False) if benchmark is not None else None,
        deterministic=(
            _env_flag("GPTSOVITS_CNHUBERT_MODEL_CUDNN_DETERMINISTIC", False) if deterministic is not None else None
        ),
        allow_tf32=_env_flag("GPTSOVITS_CNHUBERT_MODEL_CUDNN_ALLOW_TF32", False) if allow_tf32 is not None else None,
    )
    print(
        "[cnhubert_conv_backend] model cudnn flags "
        f"enabled={wrapped.enabled} benchmark={wrapped.benchmark} "
        f"deterministic={wrapped.deterministic} allow_tf32={wrapped.allow_tf32}"
    )
    return wrapped


def maybe_patch_hubert_conv_backend(model: nn.Module) -> bool:
    target_indices = _target_layer_indices()
    if not target_indices:
        return False
    feature_extractor = getattr(model, "feature_extractor", None)
    conv_layers = getattr(feature_extractor, "conv_layers", None)
    if conv_layers is None:
        return False
    patched_indices: list[int] = []
    for layer_index, layer in enumerate(conv_layers):
        if layer_index not in target_indices:
            continue
        conv = getattr(layer, "conv", None)
        if conv is None or isinstance(conv, Conv1dScopedCudnnDisabled) or not isinstance(conv, nn.Conv1d):
            continue
        layer.conv = Conv1dScopedCudnnDisabled(conv)
        patched_indices.append(layer_index)
    if patched_indices:
        print(f"[cnhubert_conv_backend] disabled cuDNN for conv_layers={patched_indices}")
    return bool(patched_indices)
