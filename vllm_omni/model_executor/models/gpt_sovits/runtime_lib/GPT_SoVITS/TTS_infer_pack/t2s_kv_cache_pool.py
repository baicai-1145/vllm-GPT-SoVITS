from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch


@dataclass
class T2SKVCachePoolState:
    enabled: bool
    device: str
    dtype: str
    num_layers: int
    hidden_dim: int
    max_batch_size: int
    max_seq_len: int
    allocated_bytes: int
    allocated_at: float
    active_rows: int = 0
    pack_hits: int = 0
    fallback_count: int = 0
    last_fallback_reason: str = ""


class T2SKVCachePool:
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_layers: int,
        hidden_dim: int,
        max_batch_size: int,
        max_seq_len: int,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.k_buffers: List[torch.Tensor] = []
        self.v_buffers: List[torch.Tensor] = []
        self.decode_mask_buffer: Optional[torch.Tensor] = None
        self.positions: Optional[torch.Tensor] = None
        self.state = T2SKVCachePoolState(
            enabled=False,
            device=str(self.device),
            dtype=str(self.dtype),
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            allocated_bytes=0,
            allocated_at=time.time(),
        )
        self._allocate()

    def _allocate(self) -> None:
        if self.max_batch_size <= 0 or self.max_seq_len <= 0 or self.num_layers <= 0 or self.hidden_dim <= 0:
            self._disable("invalid_pool_shape")
            return
        try:
            self.k_buffers = [
                torch.empty(
                    (self.max_batch_size, self.max_seq_len, self.hidden_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                for _ in range(self.num_layers)
            ]
            self.v_buffers = [
                torch.empty(
                    (self.max_batch_size, self.max_seq_len, self.hidden_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                for _ in range(self.num_layers)
            ]
            self.decode_mask_buffer = torch.empty(
                (self.max_batch_size, 1, 1, self.max_seq_len + 1),
                dtype=torch.bool,
                device=self.device,
            )
            self.positions = torch.arange(self.max_seq_len + 1, device=self.device, dtype=torch.long)
            allocated_bytes = 0
            for tensor in self.k_buffers + self.v_buffers:
                allocated_bytes += int(tensor.numel() * tensor.element_size())
            if self.decode_mask_buffer is not None:
                allocated_bytes += int(self.decode_mask_buffer.numel() * self.decode_mask_buffer.element_size())
            self.state.enabled = True
            self.state.allocated_bytes = allocated_bytes
        except Exception as exc:
            self._disable(str(exc))

    def _disable(self, reason: str) -> None:
        self.k_buffers = []
        self.v_buffers = []
        self.decode_mask_buffer = None
        self.positions = None
        self.state.enabled = False
        self.state.last_fallback_reason = str(reason)

    @staticmethod
    def _copy_rows_left_aligned(dst: torch.Tensor, src: torch.Tensor, kv_lens: Sequence[int]) -> None:
        dst.zero_()
        if not kv_lens:
            return
        target_len = int(dst.shape[1])
        if all(kv_len == target_len for kv_len in kv_lens):
            dst.copy_(src[:, -target_len:, :])
            return
        for batch_index, kv_len in enumerate(kv_lens):
            if kv_len <= 0:
                continue
            dst[batch_index, :kv_len, :].copy_(src[batch_index, -kv_len:, :])

    def snapshot(self) -> dict:
        state = self.state
        return {
            "enabled": bool(state.enabled),
            "device": state.device,
            "dtype": state.dtype,
            "num_layers": int(state.num_layers),
            "hidden_dim": int(state.hidden_dim),
            "max_batch_size": int(state.max_batch_size),
            "max_seq_len": int(state.max_seq_len),
            "allocated_bytes": int(state.allocated_bytes),
            "allocated_mb": round(float(state.allocated_bytes) / (1024.0 * 1024.0), 3),
            "allocated_at": float(state.allocated_at),
            "active_rows": int(state.active_rows),
            "available_rows": max(0, int(state.max_batch_size) - int(state.active_rows)),
            "pack_hits": int(state.pack_hits),
            "fallback_count": int(state.fallback_count),
            "last_fallback_reason": str(state.last_fallback_reason),
        }

    def can_handle(self, batch_size: int, max_kv_len: int) -> bool:
        return bool(
            self.state.enabled
            and batch_size <= self.max_batch_size
            and max_kv_len <= self.max_seq_len
        )

    def set_active_rows(self, active_rows: int) -> None:
        self.state.active_rows = max(0, min(self.max_batch_size, int(active_rows)))

    def record_fallback(self, reason: str) -> None:
        self.state.fallback_count += 1
        self.state.last_fallback_reason = str(reason)

    def _build_views(self, batch_size: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return (
            [buffer[:batch_size, :, :] for buffer in self.k_buffers],
            [buffer[:batch_size, :, :] for buffer in self.v_buffers],
        )

    def pack_dynamic_cache_layers(
        self,
        *,
        k_layers: Sequence[torch.Tensor],
        v_layers: Sequence[torch.Tensor],
        kv_lens: torch.LongTensor,
    ) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if kv_lens.numel() <= 0:
            return self._build_views(0)
        batch_size = int(kv_lens.shape[0])
        max_kv_len = int(kv_lens.max().item())
        if not self.can_handle(batch_size, max_kv_len):
            self.record_fallback(f"pack_overflow(batch={batch_size},seq={max_kv_len})")
            return None
        kv_lens_list = [int(item) for item in kv_lens.tolist()]
        for layer_index in range(self.num_layers):
            dst_k = self.k_buffers[layer_index][:batch_size, :max_kv_len, :]
            dst_v = self.v_buffers[layer_index][:batch_size, :max_kv_len, :]
            self._copy_rows_left_aligned(dst_k, k_layers[layer_index], kv_lens_list)
            self._copy_rows_left_aligned(dst_v, v_layers[layer_index], kv_lens_list)
        self.set_active_rows(batch_size)
        self.state.pack_hits += 1
        return self._build_views(batch_size)

    def compact_rows(
        self,
        *,
        keep_indices: Sequence[int],
        kv_lens: torch.LongTensor,
    ) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        batch_size = int(len(keep_indices))
        if batch_size <= 0:
            return self._build_views(0)
        max_kv_len = int(kv_lens.max().item())
        if not self.can_handle(batch_size, max_kv_len):
            self.record_fallback(f"compact_overflow(batch={batch_size},seq={max_kv_len})")
            return None
        index = torch.tensor(list(keep_indices), dtype=torch.long, device=self.device)
        for layer_index in range(self.num_layers):
            selected_k = torch.index_select(self.k_buffers[layer_index][:, :max_kv_len, :], 0, index)
            selected_v = torch.index_select(self.v_buffers[layer_index][:, :max_kv_len, :], 0, index)
            self.k_buffers[layer_index][:batch_size, :max_kv_len, :].copy_(selected_k)
            self.v_buffers[layer_index][:batch_size, :max_kv_len, :].copy_(selected_v)
        self.set_active_rows(batch_size)
        return self._build_views(batch_size)

    def build_decode_mask(self, next_kv_lens: torch.LongTensor) -> Optional[torch.Tensor]:
        if next_kv_lens.numel() <= 0:
            return None
        batch_size = int(next_kv_lens.shape[0])
        max_kv_len = int(next_kv_lens.max().item())
        if not self.can_handle(batch_size, max_kv_len):
            self.record_fallback(f"mask_overflow(batch={batch_size},seq={max_kv_len})")
            return None
        if self.decode_mask_buffer is None or self.positions is None:
            self.record_fallback("mask_buffer_missing")
            return None
        mask = self.decode_mask_buffer[:batch_size, :, :, :max_kv_len]
        valid = self.positions[:max_kv_len].unsqueeze(0) >= next_kv_lens.view(-1, 1)
        mask[:, 0, 0, :].copy_(valid)
        if not valid.any().item():
            return None
        return mask


def attach_t2s_kv_cache_pool(model: Any, device: Any) -> T2SKVCachePoolState:
    enabled = str(os.environ.get("GPTSOVITS_ENGINE_KV_POOL_ENABLE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    if not enabled:
        state = T2SKVCachePoolState(
            enabled=False,
            device=str(device),
            dtype="unknown",
            num_layers=0,
            hidden_dim=0,
            max_batch_size=0,
            max_seq_len=0,
            allocated_bytes=0,
            allocated_at=time.time(),
            last_fallback_reason="disabled_by_env",
        )
        setattr(model, "kv_cache_pool", None)
        setattr(model, "kv_cache_pool_state", state)
        return state

    params = list(model.parameters())
    model_dtype = params[0].dtype if params else torch.float32
    max_batch_size = max(1, int(os.environ.get("GPTSOVITS_ENGINE_KV_POOL_MAX_BATCH", "128")))
    max_seq_len = max(1, int(os.environ.get("GPTSOVITS_ENGINE_KV_POOL_MAX_SEQ_LEN", "1024")))
    pool = T2SKVCachePool(
        device=torch.device(device),
        dtype=model_dtype,
        num_layers=int(getattr(model, "num_layers", 0)),
        hidden_dim=int(getattr(model, "model_dim", 0)),
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    setattr(model, "kv_cache_pool", pool if pool.state.enabled else None)
    setattr(model, "kv_cache_pool_state", pool.state)
    return pool.state
