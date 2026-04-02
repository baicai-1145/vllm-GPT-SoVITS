# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
import math
import os
import time
from typing import Any, Callable, List, Optional

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F
from tqdm import tqdm

from AR.models.utils import (
    dpo_loss,
    get_batch_logps,
    make_pad_mask,
    make_pad_mask_left,
    make_reject_y,
    sample,
    topk_sampling,
)
from AR.modules.embedding import SinePositionalEmbedding, TokenEmbedding
from AR.modules.transformer import LayerNorm, TransformerEncoder, TransformerEncoderLayer

default_config = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_head": 8,
    "num_layers": 12,
    "num_codebook": 8,
    "p_dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024,
}


class _MulticlassAccuracy(nn.Module):
    """Minimal top-k accuracy metric for inference/runtime environments."""

    def __init__(
        self,
        num_classes: int,
        *,
        top_k: int = 1,
        average: str = "micro",
        multidim_average: str = "global",
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        del num_classes, average, multidim_average
        self.top_k = max(1, int(top_k))
        self.ignore_index = ignore_index

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.numel() == 0 or target.numel() == 0:
            return preds.new_tensor(0.0)

        topk = min(self.top_k, int(preds.shape[1]))
        predicted = torch.topk(preds, k=topk, dim=1).indices
        correct = (predicted == target.unsqueeze(1)).any(dim=1).reshape(-1)
        target = target.reshape(-1)

        if self.ignore_index is not None:
            valid = target != self.ignore_index
            if not bool(valid.any()):
                return preds.new_tensor(0.0)
            correct = correct[valid]

        return correct.float().mean() if correct.numel() > 0 else preds.new_tensor(0.0)


# @torch.jit.script ## 使用的话首次推理会非常慢，而且推理速度不稳定
# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, H, L, S = query.size(0), query.size(1), query.size(-2), key.size(-2)
    if scale is None:
        scale_factor = torch.tensor(1 / math.sqrt(query.size(-1)))
    else:
        scale_factor = scale
    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight.masked_fill_(attn_mask, 0)
        else:
            attn_mask[attn_mask != float("-inf")] = 0
            attn_mask[attn_mask == float("-inf")] = 1
            attn_weight.masked_fill_(attn_mask, 0)

    return attn_weight @ value


@torch.jit.script
class T2SMLP:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self, x):
        x = F.relu(F.linear(x, self.w1, self.b1))
        x = F.linear(x, self.w2, self.b2)
        return x


@torch.jit.script
class T2SBlock:
    def __init__(
        self,
        num_heads,
        hidden_dim: int,
        mlp: T2SMLP,
        qkv_w,
        qkv_b,
        out_w,
        out_b,
        norm_w1,
        norm_b1,
        norm_eps1,
        norm_w2,
        norm_b2,
        norm_eps2,
    ):
        self.num_heads = num_heads
        self.mlp = mlp
        self.hidden_dim: int = hidden_dim
        self.qkv_w = qkv_w
        self.qkv_b = qkv_b
        self.out_w = out_w
        self.out_b = out_b
        self.norm_w1 = norm_w1
        self.norm_b1 = norm_b1
        self.norm_eps1 = norm_eps1
        self.norm_w2 = norm_w2
        self.norm_b2 = norm_b2
        self.norm_eps2 = norm_eps2

        self.false = torch.tensor(False, dtype=torch.bool)

    @torch.jit.ignore
    def to_mask(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ):
        if padding_mask is None:
            return x

        if padding_mask.dtype == torch.bool:
            return x.masked_fill(padding_mask, 0)
        else:
            return x * padding_mask

    def process_prompt(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        torch_sdpa: bool = True,
    ):
        q, k, v = F.linear(self.to_mask(x, padding_mask), self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k.shape[1]

        q = self.to_mask(q, padding_mask)
        k_cache = self.to_mask(k, padding_mask)
        v_cache = self.to_mask(v, padding_mask)

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)
        else:
            attn = scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = F.linear(self.to_mask(attn, padding_mask), self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1)
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache

    def decode_next_token(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_mask: torch.Tensor = None,
        torch_sdpa: bool = True,
    ):
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        k_cache = torch.cat([k_cache, k], dim=1)
        v_cache = torch.cat([v_cache, v], dim=1)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k_cache.shape[1]

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k, v, (~attn_mask) if attn_mask is not None else None)
        else:
            attn = scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = F.linear(attn, self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w1,
            self.norm_b1,
            self.norm_eps1,
        )
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache


@torch.jit.script
class T2STransformer:
    def __init__(self, num_blocks: int, blocks: List[T2SBlock]):
        self.num_blocks: int = num_blocks
        self.blocks = blocks

    def process_prompt(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        torch_sdpa: bool = True,
    ):
        k_cache: List[torch.Tensor] = []
        v_cache: List[torch.Tensor] = []
        for i in range(self.num_blocks):
            x, k_cache_, v_cache_ = self.blocks[i].process_prompt(x, attn_mask, padding_mask, torch_sdpa)
            k_cache.append(k_cache_)
            v_cache.append(v_cache_)
        return x, k_cache, v_cache

    def decode_next_token(
        self,
        x: torch.Tensor,
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        attn_mask: torch.Tensor = None,
        torch_sdpa: bool = True,
    ):
        for i in range(self.num_blocks):
            x, k_cache[i], v_cache[i] = self.blocks[i].decode_next_token(
                x, k_cache[i], v_cache[i], attn_mask, torch_sdpa
            )
        return x, k_cache, v_cache


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = config["model"]["dropout"]
        self.EOS = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        # should be same as num of kmeans bin
        # assert self.EOS == 1024
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim,
            self.phoneme_vocab_size,
            self.p_dropout,
        )
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim,
            self.vocab_size,
            self.p_dropout,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )

        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if norm_first else None,
        )

        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")

        self.ar_accuracy_metric = _MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        )

        blocks = []

        for i in range(self.num_layers):
            layer = self.h.layers[i]
            t2smlp = T2SMLP(
                layer.linear1.weight,
                layer.linear1.bias,
                layer.linear2.weight,
                layer.linear2.bias,
            )

            block = T2SBlock(
                self.num_head,
                self.model_dim,
                t2smlp,
                layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias,
                layer.self_attn.out_proj.weight,
                layer.self_attn.out_proj.bias,
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps,
            )

            blocks.append(block)

        self.t2s_transformer = T2STransformer(self.num_layers, blocks)
        self.last_infer_stats = {}
        self._compiled_decode_next_token_prealloc_with_metadata = None
        self._last_prealloc_decode_profile = {}
        self._last_dynamic_decode_profile = {}

    def _set_last_infer_stats(self, stats):
        self.last_infer_stats = stats

    def get_last_infer_stats(self):
        return dict(self.last_infer_stats)

    def get_last_prealloc_decode_profile(self):
        return dict(self._last_prealloc_decode_profile)

    def _set_last_prealloc_decode_profile(self, stats):
        self._last_prealloc_decode_profile = dict(stats)

    def get_last_dynamic_decode_profile(self):
        return dict(self._last_dynamic_decode_profile)

    def _set_last_dynamic_decode_profile(self, stats):
        self._last_dynamic_decode_profile = dict(stats)

    @staticmethod
    def _profile_prealloc_decode_enabled() -> bool:
        return os.environ.get("GPTSOVITS_PROFILE_T2S_PREALLOC_DECODE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _profile_dynamic_decode_enabled() -> bool:
        return os.environ.get("GPTSOVITS_PROFILE_T2S_DYNAMIC_DECODE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _get_decode_nomask_sdpa_backend() -> SDPBackend | None:
        raw_value = os.environ.get("GPTSOVITS_T2S_DECODE_NOMASK_SDPA_BACKEND", "efficient").strip().lower()
        backend_mapping = {
            "auto": None,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION,
            "math": SDPBackend.MATH,
        }
        return backend_mapping.get(raw_value, SDPBackend.EFFICIENT_ATTENTION)

    @classmethod
    def _run_decode_sdpa(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        backend = cls._get_decode_nomask_sdpa_backend()
        if attn_mask is None and query.device.type == "cuda" and backend is not None:
            with sdpa_kernel(backends=[backend]):
                return F.scaled_dot_product_attention(query, key, value, None)
        return F.scaled_dot_product_attention(query, key, value, attn_mask)

    @staticmethod
    def _profile_t2s_call(
        fn: Callable[[], Any],
        *,
        records: list[tuple[str, torch.cuda.Event | None, torch.cuda.Event | float]] | None,
        stat_key: str | list[str] | tuple[str, ...],
        device: torch.device | None,
    ) -> Any:
        stat_keys = [stat_key] if isinstance(stat_key, str) else list(stat_key)
        if records is not None and device is not None and device.type == "cuda" and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = fn()
            end_event.record()
            for key in stat_keys:
                records.append((key, start_event, end_event))
            return result

        started = time.perf_counter()
        result = fn()
        if records is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            for key in stat_keys:
                records.append((key, None, elapsed_ms))
        return result

    @staticmethod
    def _prepare_prealloc_decode_inputs(
        x: torch.Tensor,
        kv_lens: torch.LongTensor,
        attn_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.LongTensor, int, int, Optional[torch.Tensor]]:
        stable_x = x.clone(memory_format=torch.contiguous_format)
        batch_index = torch.arange(stable_x.shape[0], device=stable_x.device, dtype=torch.long)
        max_kv_index = int(kv_lens.max().item())
        next_max_kv_len = max_kv_index + 1
        sdpa_attn_mask = None if attn_mask is None else ~attn_mask
        return stable_x, batch_index, max_kv_index, next_max_kv_len, sdpa_attn_mask

    @staticmethod
    def _prealloc_kv_bucket_label(next_max_kv_len: int) -> str:
        boundaries = (1024, 2048, 3072, 4096, 5120)
        lower = 1
        for upper in boundaries:
            if next_max_kv_len <= upper:
                return f"kv_{lower}_{upper}"
            lower = upper + 1
        return f"kv_{boundaries[-1] + 1}_plus"

    @staticmethod
    def _flush_t2s_profile_records(
        stats: dict[str, float | int],
        records: list[tuple[str, torch.cuda.Event | None, torch.cuda.Event | float]],
        *,
        device: torch.device | None,
    ) -> None:
        if not records:
            return
        if device is not None and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.current_stream(device).synchronize()
        for stat_key, start_event, end_event_or_elapsed in records:
            if start_event is None:
                elapsed_ms = float(end_event_or_elapsed)
            else:
                elapsed_ms = float(start_event.elapsed_time(end_event_or_elapsed))
            if stat_key.endswith("_calls"):
                stats[stat_key] = int(stats.get(stat_key, 0)) + int(elapsed_ms)
            else:
                stats[stat_key] = float(stats.get(stat_key, 0.0)) + elapsed_ms
        records.clear()

    def _decode_block_next_token_prealloc(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        layer_k_cache: torch.Tensor,
        layer_v_cache: torch.Tensor,
        kv_lens: torch.LongTensor,
        sdpa_attn_mask: Optional[torch.Tensor],
        batch_index: torch.LongTensor,
        max_kv_index: int,
        next_max_kv_len: int,
    ) -> torch.Tensor:
        q, k, v = F.linear(x, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias).chunk(3, dim=-1)
        batch_size = int(batch_index.shape[0])
        if layer_k_cache.shape[0] < batch_size or layer_v_cache.shape[0] < batch_size:
            raise ValueError(
                f"prealloc KV cache batch capacity不足: batch={batch_size}, "
                f"k_cache_batch={layer_k_cache.shape[0]}, v_cache_batch={layer_v_cache.shape[0]}"
            )
        if max_kv_index >= layer_k_cache.shape[1] or max_kv_index >= layer_v_cache.shape[1]:
            raise ValueError(
                f"prealloc KV cache seq capacity不足: write_index={max_kv_index}, "
                f"k_cache_seq={layer_k_cache.shape[1]}, v_cache_seq={layer_v_cache.shape[1]}"
            )
        layer_k_cache[batch_index, kv_lens, :] = k[:, 0, :]
        layer_v_cache[batch_index, kv_lens, :] = v[:, 0, :]

        q = q.view(batch_size, 1, self.num_head, -1).transpose(1, 2)
        k_context = layer_k_cache[:, :next_max_kv_len, :]
        v_context = layer_v_cache[:, :next_max_kv_len, :]
        k_context = k_context.view(batch_size, next_max_kv_len, self.num_head, -1).transpose(1, 2)
        v_context = v_context.view(batch_size, next_max_kv_len, self.num_head, -1).transpose(1, 2)

        attn = self._run_decode_sdpa(q, k_context, v_context, sdpa_attn_mask)
        attn = attn.transpose(1, 2).reshape(batch_size, 1, -1)
        attn = F.linear(attn, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias)

        x = x + attn
        x = F.layer_norm(
            x,
            [self.model_dim],
            layer.norm1.weight,
            layer.norm1.bias,
            layer.norm1.eps,
        )
        x = x + F.linear(F.relu(F.linear(x, layer.linear1.weight, layer.linear1.bias)), layer.linear2.weight, layer.linear2.bias)
        x = F.layer_norm(
            x,
            [self.model_dim],
            layer.norm2.weight,
            layer.norm2.bias,
            layer.norm2.eps,
        )
        return x

    @torch.inference_mode()
    def _decode_next_token_prealloc_with_metadata(
        self,
        x: torch.Tensor,
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        kv_lens: torch.LongTensor,
        batch_index: torch.LongTensor,
        max_kv_index: int,
        next_max_kv_len: int,
        sdpa_attn_mask: Optional[torch.Tensor],
    ):
        for layer_index in range(self.num_layers):
            x = self._decode_block_next_token_prealloc(
                self.h.layers[layer_index],
                x,
                k_cache[layer_index],
                v_cache[layer_index],
                kv_lens,
                sdpa_attn_mask,
                batch_index,
                max_kv_index,
                next_max_kv_len,
            )
        return x, k_cache, v_cache

    def _decode_block_next_token_prealloc_profiled(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        layer_k_cache: torch.Tensor,
        layer_v_cache: torch.Tensor,
        kv_lens: torch.LongTensor,
        sdpa_attn_mask: Optional[torch.Tensor],
        batch_index: torch.LongTensor,
        max_kv_index: int,
        next_max_kv_len: int,
        *,
        stats: dict[str, float | int],
        records: list[tuple[str, torch.cuda.Event | None, torch.cuda.Event | float]],
    ) -> torch.Tensor:
        device = x.device
        kv_bucket = self._prealloc_kv_bucket_label(next_max_kv_len)
        q, k, v = self._profile_t2s_call(
            lambda: F.linear(x, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias).chunk(3, dim=-1),
            records=records,
            stat_key="pooled_prealloc_qkv_linear_ms",
            device=device,
        )
        batch_size = int(batch_index.shape[0])
        if layer_k_cache.shape[0] < batch_size or layer_v_cache.shape[0] < batch_size:
            raise ValueError(
                f"prealloc KV cache batch capacity不足: batch={batch_size}, "
                f"k_cache_batch={layer_k_cache.shape[0]}, v_cache_batch={layer_v_cache.shape[0]}"
            )
        if max_kv_index >= layer_k_cache.shape[1] or max_kv_index >= layer_v_cache.shape[1]:
            raise ValueError(
                f"prealloc KV cache seq capacity不足: write_index={max_kv_index}, "
                f"k_cache_seq={layer_k_cache.shape[1]}, v_cache_seq={layer_v_cache.shape[1]}"
            )
        self._profile_t2s_call(
            lambda: (
                layer_k_cache.__setitem__((batch_index, kv_lens, slice(None)), k[:, 0, :]),
                layer_v_cache.__setitem__((batch_index, kv_lens, slice(None)), v[:, 0, :]),
            ),
            records=records,
            stat_key="pooled_prealloc_kv_write_ms",
            device=device,
        )

        q, k_context, v_context = self._profile_t2s_call(
            lambda: (
                q.view(batch_size, 1, self.num_head, -1).transpose(1, 2),
                layer_k_cache[:, :next_max_kv_len, :].view(batch_size, next_max_kv_len, self.num_head, -1).transpose(1, 2),
                layer_v_cache[:, :next_max_kv_len, :].view(batch_size, next_max_kv_len, self.num_head, -1).transpose(1, 2),
            ),
            records=records,
            stat_key="pooled_prealloc_kv_context_ms",
            device=device,
        )

        attn = self._profile_t2s_call(
            lambda: self._run_decode_sdpa(q, k_context, v_context, sdpa_attn_mask),
            records=records,
            stat_key=(
                "pooled_prealloc_sdpa_ms",
                f"pooled_prealloc_sdpa_{kv_bucket}_ms",
            ),
            device=device,
        )
        stats[f"pooled_prealloc_sdpa_{kv_bucket}_calls"] = int(
            stats.get(f"pooled_prealloc_sdpa_{kv_bucket}_calls", 0)
        ) + 1
        attn = self._profile_t2s_call(
            lambda: attn.transpose(1, 2).reshape(batch_size, 1, -1),
            records=records,
            stat_key=("pooled_prealloc_out_proj_ms", "pooled_prealloc_out_proj_prep_ms"),
            device=device,
        )
        attn = self._profile_t2s_call(
            lambda: F.linear(
                attn,
                layer.self_attn.out_proj.weight,
                layer.self_attn.out_proj.bias,
            ),
            records=records,
            stat_key=("pooled_prealloc_out_proj_ms", "pooled_prealloc_out_proj_linear_ms"),
            device=device,
        )

        x = self._profile_t2s_call(
            lambda: x + attn,
            records=records,
            stat_key="pooled_prealloc_attn_residual_ms",
            device=device,
        )
        x = self._profile_t2s_call(
            lambda: F.layer_norm(
                x,
                [self.model_dim],
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
            ),
            records=records,
            stat_key="pooled_prealloc_norm1_ms",
            device=device,
        )
        ffn_hidden = self._profile_t2s_call(
            lambda: F.relu(F.linear(x, layer.linear1.weight, layer.linear1.bias)),
            records=records,
            stat_key=("pooled_prealloc_ffn_ms", "pooled_prealloc_ffn_up_ms"),
            device=device,
        )
        ffn_out = self._profile_t2s_call(
            lambda: F.linear(
                ffn_hidden,
                layer.linear2.weight,
                layer.linear2.bias,
            ),
            records=records,
            stat_key=("pooled_prealloc_ffn_ms", "pooled_prealloc_ffn_down_ms"),
            device=device,
        )
        x = self._profile_t2s_call(
            lambda: x + ffn_out,
            records=records,
            stat_key=("pooled_prealloc_ffn_ms", "pooled_prealloc_ffn_residual_ms"),
            device=device,
        )
        x = self._profile_t2s_call(
            lambda: F.layer_norm(
                x,
                [self.model_dim],
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps,
            ),
            records=records,
            stat_key="pooled_prealloc_norm2_ms",
            device=device,
        )
        stats["pooled_prealloc_profiled_layer_calls"] = int(stats.get("pooled_prealloc_profiled_layer_calls", 0)) + 1
        return x

    @torch.inference_mode()
    def decode_next_token_prealloc(
        self,
        x: torch.Tensor,
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        kv_lens: torch.LongTensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        prepared_x, batch_index, max_kv_index, next_max_kv_len, sdpa_attn_mask = self._prepare_prealloc_decode_inputs(
            x,
            kv_lens,
            attn_mask,
        )
        return self._decode_next_token_prealloc_with_metadata(
            prepared_x,
            k_cache,
            v_cache,
            kv_lens,
            batch_index,
            max_kv_index,
            next_max_kv_len,
            sdpa_attn_mask,
        )

    @torch.inference_mode()
    def decode_next_token_prealloc_profiled(
        self,
        x: torch.Tensor,
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        kv_lens: torch.LongTensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        x, batch_index, max_kv_index, next_max_kv_len, sdpa_attn_mask = self._prepare_prealloc_decode_inputs(
            x,
            kv_lens,
            attn_mask,
        )
        batch_size = int(batch_index.shape[0])
        stats: dict[str, float | int] = {
            "pooled_prealloc_profiled_decode_calls": 1,
            "pooled_prealloc_profiled_layer_calls": 0,
            "pooled_prealloc_qkv_linear_ms": 0.0,
            "pooled_prealloc_kv_write_ms": 0.0,
            "pooled_prealloc_kv_context_ms": 0.0,
            "pooled_prealloc_sdpa_ms": 0.0,
            "pooled_prealloc_out_proj_ms": 0.0,
            "pooled_prealloc_out_proj_prep_ms": 0.0,
            "pooled_prealloc_out_proj_linear_ms": 0.0,
            "pooled_prealloc_attn_residual_ms": 0.0,
            "pooled_prealloc_norm1_ms": 0.0,
            "pooled_prealloc_ffn_ms": 0.0,
            "pooled_prealloc_ffn_up_ms": 0.0,
            "pooled_prealloc_ffn_down_ms": 0.0,
            "pooled_prealloc_ffn_residual_ms": 0.0,
            "pooled_prealloc_norm2_ms": 0.0,
        }
        records: list[tuple[str, torch.cuda.Event | None, torch.cuda.Event | float]] = []
        layer_loop_started = time.perf_counter()
        kv_bucket = self._prealloc_kv_bucket_label(next_max_kv_len)
        for layer_index in range(self.num_layers):
            stats[f"pooled_prealloc_layer_total_{kv_bucket}_calls"] = int(
                stats.get(f"pooled_prealloc_layer_total_{kv_bucket}_calls", 0)
            ) + 1
            x = self._profile_t2s_call(
                lambda: self._decode_block_next_token_prealloc_profiled(
                    self.h.layers[layer_index],
                    x,
                    k_cache[layer_index],
                    v_cache[layer_index],
                    kv_lens,
                    sdpa_attn_mask,
                    batch_index,
                    max_kv_index,
                    next_max_kv_len,
                    stats=stats,
                    records=records,
                ),
                records=records,
                stat_key=f"pooled_prealloc_layer_total_{kv_bucket}_ms",
                device=x.device,
            )
        self._flush_t2s_profile_records(stats, records, device=x.device)
        stats["pooled_prealloc_layer_total_ms"] = float((time.perf_counter() - layer_loop_started) * 1000.0)
        self._set_last_prealloc_decode_profile(stats)
        return (
            x,
            [layer[:batch_size, :, :] for layer in k_cache],
            [layer[:batch_size, :, :] for layer in v_cache],
        )

    @torch.inference_mode()
    def decode_next_token_prealloc_runtime(
        self,
        x: torch.Tensor,
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        kv_lens: torch.LongTensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if self._profile_prealloc_decode_enabled():
            return self.decode_next_token_prealloc_profiled(x, k_cache, v_cache, kv_lens, attn_mask)
        self._set_last_prealloc_decode_profile({})
        prepared_x, batch_index, max_kv_index, next_max_kv_len, sdpa_attn_mask = self._prepare_prealloc_decode_inputs(
            x,
            kv_lens,
            attn_mask,
        )
        runtime = getattr(self, "_compiled_decode_next_token_prealloc_with_metadata", None)
        if runtime is None:
            return self._decode_next_token_prealloc_with_metadata(
                prepared_x,
                k_cache,
                v_cache,
                kv_lens,
                batch_index,
                max_kv_index,
                next_max_kv_len,
                sdpa_attn_mask,
            )
        try:
            return runtime(
                prepared_x,
                k_cache,
                v_cache,
                kv_lens,
                batch_index,
                max_kv_index,
                next_max_kv_len,
                sdpa_attn_mask,
            )
        except Exception as exc:
            self._compiled_decode_next_token_prealloc_with_metadata = None
            print(f"Compiled T2S prealloc decode disabled after runtime failure: {exc}")
            return self._decode_next_token_prealloc_with_metadata(
                prepared_x,
                k_cache,
                v_cache,
                kv_lens,
                batch_index,
                max_kv_index,
                next_max_kv_len,
                sdpa_attn_mask,
            )

    def _decode_block_next_token_dynamic_profiled(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        layer_k_cache: torch.Tensor,
        layer_v_cache: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        *,
        records: list[tuple[str, torch.cuda.Event | None, torch.cuda.Event | float]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        q, k, v = self._profile_t2s_call(
            lambda: F.linear(x, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias).chunk(3, dim=-1),
            records=records,
            stat_key="dynamic_qkv_linear_ms",
            device=device,
        )
        next_k_cache, next_v_cache = self._profile_t2s_call(
            lambda: (
                torch.cat([layer_k_cache, k], dim=1),
                torch.cat([layer_v_cache, v], dim=1),
            ),
            records=records,
            stat_key="dynamic_kv_append_ms",
            device=device,
        )

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = next_k_cache.shape[1]
        q, k_context, v_context = self._profile_t2s_call(
            lambda: (
                q.view(batch_size, q_len, self.num_head, -1).transpose(1, 2),
                next_k_cache.view(batch_size, kv_len, self.num_head, -1).transpose(1, 2),
                next_v_cache.view(batch_size, kv_len, self.num_head, -1).transpose(1, 2),
            ),
            records=records,
            stat_key="dynamic_kv_context_ms",
            device=device,
        )

        attn = self._profile_t2s_call(
            lambda: self._run_decode_sdpa(q, k_context, v_context, (~attn_mask) if attn_mask is not None else None),
            records=records,
            stat_key="dynamic_sdpa_ms",
            device=device,
        )
        attn = self._profile_t2s_call(
            lambda: attn.transpose(1, 2).reshape(batch_size, q_len, -1),
            records=records,
            stat_key="dynamic_out_proj_ms",
            device=device,
        )
        attn = self._profile_t2s_call(
            lambda: F.linear(attn, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias),
            records=records,
            stat_key="dynamic_out_proj_ms",
            device=device,
        )
        x = self._profile_t2s_call(
            lambda: x + attn,
            records=records,
            stat_key="dynamic_attn_residual_ms",
            device=device,
        )
        x = self._profile_t2s_call(
            lambda: F.layer_norm(x, [self.model_dim], layer.norm1.weight, layer.norm1.bias, layer.norm1.eps),
            records=records,
            stat_key="dynamic_norm1_ms",
            device=device,
        )
        ffn_hidden = self._profile_t2s_call(
            lambda: F.relu(F.linear(x, layer.linear1.weight, layer.linear1.bias)),
            records=records,
            stat_key="dynamic_ffn_ms",
            device=device,
        )
        ffn_out = self._profile_t2s_call(
            lambda: F.linear(ffn_hidden, layer.linear2.weight, layer.linear2.bias),
            records=records,
            stat_key="dynamic_ffn_ms",
            device=device,
        )
        x = self._profile_t2s_call(
            lambda: x + ffn_out,
            records=records,
            stat_key="dynamic_ffn_residual_ms",
            device=device,
        )
        x = self._profile_t2s_call(
            lambda: F.layer_norm(x, [self.model_dim], layer.norm2.weight, layer.norm2.bias, layer.norm2.eps),
            records=records,
            stat_key="dynamic_norm2_ms",
            device=device,
        )
        return x, next_k_cache, next_v_cache

    @torch.inference_mode()
    def decode_next_token_dynamic_profiled(
        self,
        x: torch.Tensor,
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ):
        stats: dict[str, float | int] = {
            "dynamic_profiled_decode_calls": 1,
            "dynamic_profiled_layer_calls": 0,
            "dynamic_qkv_linear_ms": 0.0,
            "dynamic_kv_append_ms": 0.0,
            "dynamic_kv_context_ms": 0.0,
            "dynamic_sdpa_ms": 0.0,
            "dynamic_out_proj_ms": 0.0,
            "dynamic_attn_residual_ms": 0.0,
            "dynamic_norm1_ms": 0.0,
            "dynamic_ffn_ms": 0.0,
            "dynamic_ffn_residual_ms": 0.0,
            "dynamic_norm2_ms": 0.0,
            "dynamic_layer_total_ms": 0.0,
        }
        records: list[tuple[str, torch.cuda.Event | None, torch.cuda.Event | float]] = []
        layer_loop_started = time.perf_counter()
        for layer_index in range(self.num_layers):
            x, k_cache[layer_index], v_cache[layer_index] = self._profile_t2s_call(
                lambda current_index=layer_index: self._decode_block_next_token_dynamic_profiled(
                    self.h.layers[current_index],
                    x,
                    k_cache[current_index],
                    v_cache[current_index],
                    attn_mask,
                    records=records,
                ),
                records=records,
                stat_key="dynamic_layer_total_ms",
                device=x.device,
            )
            stats["dynamic_profiled_layer_calls"] = int(stats.get("dynamic_profiled_layer_calls", 0)) + 1
        self._flush_t2s_profile_records(stats, records, device=x.device)
        stats["dynamic_layer_total_ms"] = float((time.perf_counter() - layer_loop_started) * 1000.0)
        self._set_last_dynamic_decode_profile(stats)
        return x, k_cache, v_cache

    @torch.inference_mode()
    def decode_next_token_dynamic_runtime(
        self,
        x: torch.Tensor,
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if self._profile_dynamic_decode_enabled():
            return self.decode_next_token_dynamic_profiled(x, k_cache, v_cache, attn_mask)
        self._set_last_dynamic_decode_profile({})
        return self.t2s_transformer.decode_next_token(x, k_cache, v_cache, attn_mask)

    def make_input_data(self, x, x_lens, y, y_lens, bert_feature):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask_left(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)

        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )
        # x_attn_mask[:, x_len]=False
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )

        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_head, -1, -1)
            .reshape(bsz * self.num_head, 1, src_len)
        )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask
        # x 和完整的 y 一次性输入模型
        xy_pos = torch.concat([x, y_pos], dim=1)

        return xy_pos, xy_attn_mask, targets

    def forward(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids
        y: semantic_ids
        """

        reject_y, reject_y_lens = make_reject_y(y, y_lens)

        xy_pos, xy_attn_mask, targets = self.make_input_data(x, x_lens, y, y_lens, bert_feature)

        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        x_len = x_lens.max()
        logits = self.ar_predict_layer(xy_dec[:, x_len-1:])

        ###### DPO #############
        reject_xy_pos, reject_xy_attn_mask, reject_targets = self.make_input_data(
            x, x_lens, reject_y, reject_y_lens, bert_feature
        )

        reject_xy_dec, _ = self.h(
            (reject_xy_pos, None),
            mask=reject_xy_attn_mask,
        )
        x_len = x_lens.max()
        reject_logits = self.ar_predict_layer(reject_xy_dec[:, x_len-1:])

        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum

        loss_1 = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction="sum")
        acc = self.ar_accuracy_metric(logits.permute(0, 2, 1).detach(), targets).item()

        A_logits, R_logits = get_batch_logps(logits, reject_logits, targets, reject_targets)
        loss_2, _, _ = dpo_loss(A_logits, R_logits, 0, 0, 0.2, reference_free=True)

        loss = loss_1 + loss_2

        return loss, acc

    def forward_old(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids
        y: semantic_ids
        """
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask_left(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.num_head, -1, -1)
            .reshape(bsz * self.num_head, 1, src_len)
        )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask
        # x 和完整的 y 一次性输入模型
        xy_pos = torch.concat([x, y_pos], dim=1)
        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        logits = self.ar_predict_layer(xy_dec[:, x_len-1:]).permute(0, 2, 1)
        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum
        loss = F.cross_entropy(logits, targets, reduction="sum")
        acc = self.ar_accuracy_metric(logits.detach(), targets).item()
        return loss, acc

    # 需要看下这个函数和 forward 的区别以及没有 semantic 的时候 prompts 输入什么
    def infer(
        self,
        x,
        x_lens,
        prompts,
        bert_feature,
        top_k: int = -100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts
        prefix_len = y.shape[1]
        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False
        for _ in tqdm(range(1500)):
            y_emb = self.ar_audio_embedding(y)
            y_pos = self.ar_audio_position(y_emb)
            # x 和逐渐增长的 y 一起输入给模型
            xy_pos = torch.concat([x, y_pos], dim=1)
            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0).to(y.device)

            xy_dec, _ = self.h(
                (xy_pos, None),
                mask=xy_attn_mask,
            )
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(logits, top_k=top_k, top_p=1.0, temperature=temperature)

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                # print(torch.argmax(logits, dim=-1)[0] == self.EOS, samples[0, 0] == self.EOS)
                stop = True
            if stop:
                if prompts.shape[1] == y.shape[1]:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break
            # 本次生成的 semantic_ids 和之前的 y 构成新的 y
            # print(samples.shape)#[1,1]#第一个1是bs
            # import os
            # os._exit(2333)
            y = torch.concat([y, samples], dim=1)
        return y

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(y_mask_int, (0, 1), value=1)
        # 错位
        return targets[:, :-1], targets

    def infer_panel_batch_infer(
        self,
        x: List[torch.LongTensor],  #####全部文本token
        x_lens: torch.LongTensor,
        prompts: torch.LongTensor,  ####参考音频token
        bert_feature: List[torch.LongTensor],
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        requested_enable_mask_free_fastpath = bool(kwargs.get("enable_mask_free_fastpath", True))
        if prompts is None:
            self._set_last_infer_stats(
                {
                    "infer_mode": "batch_infer_prompt_free_fallback",
                    "requested_enable_mask_free_fastpath": requested_enable_mask_free_fastpath,
                    "batch_size": int(len(x)),
                    "prefill_after_mask_all_visible": None,
                    "fastpath_hit": False,
                    "generated_token_count": 0,
                    "generated_token_count_list": [],
                }
            )
            print("Warning: Prompt free is not supported batch_infer! switch to naive_infer")
            return self.infer_panel_naive_batched(
                x,
                x_lens,
                prompts,
                bert_feature,
                top_k=top_k,
                top_p=top_p,
                early_stop_num=early_stop_num,
                temperature=temperature,
                **kwargs,
            )

        max_len = kwargs.get("max_len", x_lens.max())
        enable_mask_free_fastpath = requested_enable_mask_free_fastpath
        x_list = []
        for x_item, bert_item in zip(x, bert_feature):
            # max_len = max(max_len, x_item.shape[0], bert_item.shape[1])
            x_item = self.ar_text_embedding(x_item.unsqueeze(0))
            x_item = x_item + self.bert_proj(bert_item.transpose(0, 1).unsqueeze(0))
            x_item = self.ar_text_position(x_item).squeeze(0)
            # x_item = F.pad(x_item,(0,0,0,max_len-x_item.shape[0]),value=0) if x_item.shape[0]<max_len else x_item  ### padding right
            x_item = (
                F.pad(x_item, (0, 0, max_len - x_item.shape[0], 0), value=0) if x_item.shape[0] < max_len else x_item
            )  ### padding left
            x_list.append(x_item)
        x: torch.Tensor = torch.stack(x_list, dim=0)

        # AR Decoder
        y = prompts

        x_len = x.shape[1]
        stop = False

        k_cache = None
        v_cache = None
        ###################  first step ##########################
        assert y is not None, "Error: Prompt free is not supported batch_infer!"
        ref_free = False

        y_emb = self.ar_audio_embedding(y)
        y_len = y_emb.shape[1]
        prefix_len = y.shape[1]
        y_lens = torch.LongTensor([y_emb.shape[1]] * y_emb.shape[0]).to(x.device)
        y_pos = self.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1)

        ##### create mask #####
        bsz = x.shape[0]
        src_len = x_len + y_len
        y_paddind_mask = make_pad_mask_left(y_lens, y_len)
        x_paddind_mask = make_pad_mask_left(x_lens, max_len)

        # (bsz, x_len + y_len)
        padding_mask = torch.concat([x_paddind_mask, y_paddind_mask], dim=1)

        x_mask = F.pad(
            torch.zeros(x_len, x_len, dtype=torch.bool, device=x.device),
            (0, y_len),
            value=True,
        )

        y_mask = F.pad(  ###yy的右上1扩展到左边xy的0,(y,x+y)
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=x.device), diagonal=1),
            (x_len, 0),
            value=False,
        )

        causal_mask = torch.concat([x_mask, y_mask], dim=0).view(1, src_len, src_len).repeat(bsz, 1, 1).to(x.device)
        # padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2) ### [b, x+y, x+y]
        ### 上面是错误的，会导致padding的token被"看见"

        # 正确的padding_mask应该是：
        # |   pad_len   |  x_len  |  y_len  |
        # [[PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],  前3行按理说也应该被mask掉，但是为了防止计算attention时不出现nan，还是保留了，不影响结果
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6],
        # [PAD, PAD, PAD, 1, 2, 3, 4, 5, 6]]

        padding_mask = padding_mask.view(bsz, 1, src_len).repeat(1, src_len, 1)

        attn_mask: torch.Tensor = causal_mask.logical_or(padding_mask)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_head, -1, -1).bool()

        # 正确的attn_mask应该是这样的：
        # |   pad_len   |  x_len  |  y_len  |
        # [[PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],  前3行按理说也应该被mask掉，但是为了防止计算attention时不出现nan，还是保留了，不影响结果
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3, EOS, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3,   4, EOS, EOS],
        # [PAD, PAD, PAD, 1, 2, 3,   4,   5, EOS],
        # [PAD, PAD, PAD, 1, 2, 3,   4,   5,   6]]

        ###### decode #####
        y_list = [None] * y.shape[0]
        batch_idx_map = list(range(y.shape[0]))
        idx_list = [None] * y.shape[0]
        decode_attn_mask = attn_mask
        prefill_after_mask_all_visible = None
        fastpath_hit = False
        for idx in tqdm(range(1500)):
            if idx == 0:
                xy_dec, k_cache, v_cache = self.t2s_transformer.process_prompt(xy_pos, attn_mask, None)
            else:
                xy_dec, k_cache, v_cache = self.t2s_transformer.decode_next_token(
                    xy_pos, k_cache, v_cache, decode_attn_mask
                )
            logits = self.ar_predict_layer(xy_dec[:, -1])

            if idx == 0:
                attn_mask = F.pad(attn_mask[:, :, -1].unsqueeze(-2), (0, 1), value=False)
                prefill_after_mask_all_visible = not attn_mask.any().item()
                if enable_mask_free_fastpath and y.shape[0] == 1 and prefill_after_mask_all_visible:
                    decode_attn_mask = None
                    fastpath_hit = True
                else:
                    decode_attn_mask = attn_mask
            else:
                if decode_attn_mask is not None:
                    attn_mask = F.pad(attn_mask, (0, 1), value=False)
                    decode_attn_mask = attn_mask

            if idx < 11:  ###至少预测出10个token不然不给停止（0.4s）
                logits = logits[:, :-1] 

            samples = sample(
                logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature
            )[0]

            y = torch.concat([y, samples], dim=1)

            ####### 移除batch中已经生成完毕的序列,进一步优化计算量
            tokens = torch.argmax(logits, dim=-1)
            reserved_idx_of_batch_for_y = None
            if (self.EOS in samples[:, 0]) or (self.EOS in tokens):  ###如果生成到EOS，则停止
                l1 = samples[:, 0] == self.EOS
                l2 = tokens == self.EOS
                l = l1.logical_or(l2)
                removed_idx_of_batch_for_y = torch.where(l == True)[0].tolist()
                reserved_idx_of_batch_for_y = torch.where(l == False)[0]
                # batch_indexs = torch.tensor(batch_idx_map, device=y.device)[removed_idx_of_batch_for_y]
                for i in removed_idx_of_batch_for_y:
                    batch_index = batch_idx_map[i]
                    idx_list[batch_index] = idx
                    y_list[batch_index] = y[i, :-1]

                batch_idx_map = [batch_idx_map[i] for i in reserved_idx_of_batch_for_y.tolist()]

            # 只保留batch中未生成完毕的序列
            if reserved_idx_of_batch_for_y is not None:
                # index = torch.LongTensor(batch_idx_map).to(y.device)
                y = torch.index_select(y, dim=0, index=reserved_idx_of_batch_for_y)
                if decode_attn_mask is not None:
                    attn_mask = torch.index_select(attn_mask, dim=0, index=reserved_idx_of_batch_for_y)
                    decode_attn_mask = attn_mask
                if k_cache is not None:
                    for i in range(len(k_cache)):
                        k_cache[i] = torch.index_select(k_cache[i], dim=0, index=reserved_idx_of_batch_for_y)
                        v_cache[i] = torch.index_select(v_cache[i], dim=0, index=reserved_idx_of_batch_for_y)

            if (early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num) or idx == 1499:
                print("use early stop num:", early_stop_num)
                stop = True
                for i, batch_index in enumerate(batch_idx_map):
                    batch_index = batch_idx_map[i]
                    idx_list[batch_index] = idx
                    y_list[batch_index] = y[i, :-1]

            if None not in idx_list:
                stop = True

            if stop:
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break

            ####################### update next step ###################################
            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[
                :, y_len + idx
            ].to(dtype=y_emb.dtype, device=y_emb.device)

        if None in idx_list:
            for i in range(x.shape[0]):
                if idx_list[i] is None:
                    idx_list[i] = 1500 - 1  ###如果没有生成到EOS，就用最大长度代替

        self._set_last_infer_stats(
            {
                "infer_mode": "batch_infer",
                "requested_enable_mask_free_fastpath": enable_mask_free_fastpath,
                "batch_size": int(len(x)),
                "prefill_after_mask_all_visible": prefill_after_mask_all_visible,
                "fastpath_hit": fastpath_hit,
                "generated_token_count": int(sum(idx_list)),
                "generated_token_count_list": [int(item) for item in idx_list],
                "max_len": int(max_len),
            }
        )
        if ref_free:
            return y_list, [0] * x.shape[0]
        # print(idx_list)
        return y_list, idx_list

    def infer_panel_naive_batched(
        self,
        x: List[torch.LongTensor],  #####全部文本token
        x_lens: torch.LongTensor,
        prompts: torch.LongTensor,  ####参考音频token
        bert_feature: List[torch.LongTensor],
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        y_list = []
        idx_list = []
        for i in range(len(x)):
            y, idx = next(self.infer_panel_naive(
                x[i].unsqueeze(0),
                x_lens[i],
                prompts[i].unsqueeze(0) if prompts is not None else None,
                bert_feature[i].unsqueeze(0),
                top_k,
                top_p,
                early_stop_num,
                temperature,
                repetition_penalty,
                **kwargs,
            ))
            y_list.append(y[0])
            idx_list.append(idx)

        self._set_last_infer_stats(
            {
                "infer_mode": "naive_batched",
                "requested_enable_mask_free_fastpath": bool(kwargs.get("enable_mask_free_fastpath", True)),
                "batch_size": int(len(x)),
                "prefill_after_mask_all_visible": None,
                "fastpath_hit": False,
                "generated_token_count": int(sum(idx_list)),
                "generated_token_count_list": [int(item) for item in idx_list],
            }
        )
        return y_list, idx_list

    def infer_panel_naive(
        self,
        x: torch.LongTensor,  #####全部文本token
        x_lens: torch.LongTensor,
        prompts: torch.LongTensor,  ####参考音频token
        bert_feature: torch.LongTensor,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        streaming_mode: bool = False,
        chunk_length: int = 24,
        **kwargs,
    ):
        mute_emb_sim_matrix = kwargs.get("mute_emb_sim_matrix", None)
        chunk_split_thershold = kwargs.get("chunk_split_thershold", 0.3)
        check_token_num = 2


        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts

        x_len = x.shape[1]
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        stop = False
        # print(1111111,self.num_layers)

        k_cache = None
        v_cache = None
        ###################  first step ##########################
        if y is not None:
            y_emb = self.ar_audio_embedding(y)
            y_len = y_emb.shape[1]
            prefix_len = y.shape[1]
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)
            ref_free = False
        else:
            y_emb = None
            y_len = 0
            prefix_len = 0
            y_pos = None
            xy_pos = x
            y = torch.zeros(x.shape[0], 0, dtype=torch.int, device=x.device)
            ref_free = True

        bsz = x.shape[0]
        src_len = x_len + y_len
        x_attn_mask_pad = F.pad(
            x_attn_mask,
            (0, y_len),  ###xx的纯0扩展到xx纯0+xy纯1，(x,x+y)
            value=True,
        )
        y_attn_mask = F.pad(  ###yy的右上1扩展到左边xy的0,(y,x+y)
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = (
            torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
            .unsqueeze(0)
            .expand(bsz * self.num_head, -1, -1)
            .view(bsz, self.num_head, src_len, src_len)
            .to(device=x.device, dtype=torch.bool)
        )

        token_counter = 0
        curr_ptr = prefix_len
        for idx in tqdm(range(1500)):
            token_counter+=1
            if xy_attn_mask is not None:
                xy_dec, k_cache, v_cache = self.t2s_transformer.process_prompt(xy_pos, xy_attn_mask, None)
            else:
                xy_dec, k_cache, v_cache = self.t2s_transformer.decode_next_token(xy_pos, k_cache, v_cache)

            logits = self.ar_predict_layer(xy_dec[:, -1])

            if idx == 0:
                xy_attn_mask = None
            if idx < 11:  ###至少预测出10个token不然不给停止（0.4s）
                logits = logits[:, :-1]

            samples = sample(
                logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature
            )[0]

            y = torch.concat([y, samples], dim=1)

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
                y=y[:, :-1]
                token_counter -= 1

            if idx == 1499:
                stop = True

            if stop:
                if y.shape[1] == 0:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                    print("bad zero prediction")
                # print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                if streaming_mode:
                    yield y[:, curr_ptr:] if curr_ptr<y.shape[1] else None, True
                break


            if streaming_mode and (mute_emb_sim_matrix is not None) and (token_counter >= chunk_length+check_token_num):
                score = mute_emb_sim_matrix[y[0, curr_ptr:]] - chunk_split_thershold
                score[score<0]=-1
                score[:-1]=score[:-1]+score[1:] ##考虑连续两个token
                argmax_idx = score.argmax()

                if score[argmax_idx]>=0 and argmax_idx+1>=chunk_length: 
                    print(f"\n\ncurr_ptr:{curr_ptr}")
                    yield y[:, curr_ptr:], False
                    token_counter -= argmax_idx+1
                    curr_ptr += argmax_idx+1


            elif streaming_mode and (mute_emb_sim_matrix is None) and (token_counter >= chunk_length):
                yield y[:, -token_counter:], False
                curr_ptr+=token_counter
                token_counter = 0
                


            ####################### update next step ###################################
            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[
                :, y_len + idx
            ].to(dtype=y_emb.dtype, device=y_emb.device)



        if not streaming_mode:
            generated_token_count = max(int(y.shape[1] - prefix_len), 0)
            self._set_last_infer_stats(
                {
                    "infer_mode": "naive",
                    "requested_enable_mask_free_fastpath": bool(kwargs.get("enable_mask_free_fastpath", True)),
                    "batch_size": int(x.shape[0]),
                    "prefill_after_mask_all_visible": True if prompts is not None else None,
                    "fastpath_hit": True if prompts is not None else False,
                    "generated_token_count": generated_token_count,
                    "generated_token_count_list": [generated_token_count],
                }
            )
            if ref_free:
                yield y, 0
            yield y, idx



    def infer_panel(
        self,
        x: torch.LongTensor,  #####全部文本token
        x_lens: torch.LongTensor,
        prompts: torch.LongTensor,  ####参考音频token
        bert_feature: torch.LongTensor,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        return next(self.infer_panel_naive(
            x, x_lens, prompts, bert_feature, top_k, top_p, early_stop_num, temperature, repetition_penalty, **kwargs
        ))
