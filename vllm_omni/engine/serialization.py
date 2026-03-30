"""Shared serialization helpers for omni engine request payloads."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.engine import (
    AdditionalInformationEntry,
    AdditionalInformationPayload,
)

logger = init_logger(__name__)

_TENSOR_MARKER = "__additional_information_tensor__"
_NDARRAY_MARKER = "__additional_information_ndarray__"
_TUPLE_MARKER = "__additional_information_tuple__"


def dtype_to_name(dtype: torch.dtype) -> str:
    """Convert torch dtype to a stable string name for serialization."""
    mapping = {
        torch.float32: "float32",
        torch.float: "float32",
        torch.float16: "float16",
        torch.half: "float16",
        torch.bfloat16: "bfloat16",
        torch.float64: "float64",
        torch.double: "float64",
        torch.int64: "int64",
        torch.long: "int64",
        torch.int32: "int32",
        torch.int: "int32",
        torch.int16: "int16",
        torch.short: "int16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    return mapping.get(dtype, str(dtype).replace("torch.", ""))


def _encode_nested_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value_cpu = value.detach().to("cpu").contiguous()
        return {
            _TENSOR_MARKER: True,
            "dtype": dtype_to_name(value_cpu.dtype),
            "shape": list(value_cpu.shape),
            "data": value_cpu.numpy().tobytes(),
        }

    if isinstance(value, np.ndarray):
        value_cpu = np.ascontiguousarray(value)
        return {
            _NDARRAY_MARKER: True,
            "dtype": value_cpu.dtype.str,
            "shape": list(value_cpu.shape),
            "data": value_cpu.tobytes(),
        }

    if isinstance(value, dict):
        return {str(key): _encode_nested_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_encode_nested_value(item) for item in value]

    if isinstance(value, tuple):
        return {
            _TUPLE_MARKER: [_encode_nested_value(item) for item in value],
        }

    return value


def _decode_nested_value(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get(_TENSOR_MARKER):
            dt = np.dtype(value.get("dtype", "float32"))
            arr = np.frombuffer(value["data"], dtype=dt).reshape(value.get("shape", ()))
            return torch.from_numpy(arr.copy())

        if value.get(_NDARRAY_MARKER):
            dt = np.dtype(value.get("dtype", "<f4"))
            return np.frombuffer(value["data"], dtype=dt).reshape(value.get("shape", ())).copy()

        if _TUPLE_MARKER in value:
            items = value[_TUPLE_MARKER]
            if isinstance(items, list):
                return tuple(_decode_nested_value(item) for item in items)

        return {key: _decode_nested_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_decode_nested_value(item) for item in value]

    return value


def serialize_additional_information(
    raw_info: dict[str, Any] | AdditionalInformationPayload | None,
    *,
    log_prefix: str | None = None,
) -> AdditionalInformationPayload | None:
    """Serialize omni request metadata for EngineCore transport."""
    if raw_info is None:
        return None
    if isinstance(raw_info, AdditionalInformationPayload):
        return raw_info

    entries: dict[str, AdditionalInformationEntry] = {}
    for key, value in raw_info.items():
        if isinstance(value, torch.Tensor):
            value_cpu = value.detach().to("cpu").contiguous()
            entries[key] = AdditionalInformationEntry(
                tensor_data=value_cpu.numpy().tobytes(),
                tensor_shape=list(value_cpu.shape),
                tensor_dtype=dtype_to_name(value_cpu.dtype),
            )
            continue

        if isinstance(value, list):
            entries[key] = AdditionalInformationEntry(
                list_data=_encode_nested_value(value),
            )
            continue

        entries[key] = AdditionalInformationEntry(
            scalar_data=_encode_nested_value(value),
        )

    return AdditionalInformationPayload(entries=entries) if entries else None


def deserialize_additional_information(
    payload: dict | AdditionalInformationPayload | object | None,
) -> dict:
    """Deserialize an *additional_information* payload into a plain dict.

    Accepts:
    - ``dict`` – returned as-is.
    - ``AdditionalInformationPayload`` (or duck-typed with
      ``.entries``) – decoded entry-by-entry.
    - ``None`` – returns ``{}``.
    """

    if payload is None:
        return {}

    if isinstance(payload, dict):
        return payload

    try:
        entries = getattr(payload, "entries", None)
        if not isinstance(entries, dict):
            logger.exception("Failed to decode additional_information payload, entries field not a dict")
            return {}
        info: dict[str, object] = {}
        for k, entry in entries.items():
            if getattr(entry, "tensor_data", None) is not None:
                dt = np.dtype(getattr(entry, "tensor_dtype", "float32"))
                arr = np.frombuffer(entry.tensor_data, dtype=dt)
                arr = arr.reshape(getattr(entry, "tensor_shape", ()))
                info[k] = torch.from_numpy(arr.copy())
            elif getattr(entry, "list_data", None) is not None:
                info[k] = _decode_nested_value(entry.list_data)
            elif getattr(entry, "scalar_data", None) is not None:
                info[k] = _decode_nested_value(entry.scalar_data)
            else:
                info[k] = None
        return info
    except Exception:
        logger.exception("Failed to decode additional_information payload")

    return {}
