import warnings

warnings.filterwarnings("ignore")
import contextlib
import math
import os
import threading
import time
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from module import commons
from module import modules
from module import attentions
from f5_tts.model import DiT
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from module.commons import init_weights, get_padding
from module.mrte_model import MRTE
from module.quantize import ResidualVectorQuantizer

# from text import symbols
from text import symbols as symbols_v1
from text import symbols2 as symbols_v2
from torch.cuda.amp import autocast
import contextlib
import random


def _sync_profile_device(device):
    try:
        device_str = str(device)
        if device_str.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    except Exception:
        pass


def _vits_static_bucket_compile_enabled() -> bool:
    return os.environ.get("GPTSOVITS_COMPILE_VITS_DEC_STATIC_BUCKETS", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


def _vits_static_bucket_min_hits() -> int:
    raw_value = os.environ.get("GPTSOVITS_COMPILE_VITS_DEC_STATIC_BUCKET_MIN_HITS", "2").strip()
    try:
        return max(1, int(raw_value))
    except Exception:
        return 2


def _vits_static_bucket_compile_mode() -> str:
    return os.environ.get("GPTSOVITS_COMPILE_VITS_DEC_STATIC_BUCKET_MODE", "default").strip() or "default"


def _vits_decoder_cudagraph_enabled() -> bool:
    return os.environ.get("GPTSOVITS_VITS_DECODER_CUDAGRAPH", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


def _vits_decoder_cudagraph_min_hits() -> int:
    raw_value = os.environ.get("GPTSOVITS_VITS_DECODER_CUDAGRAPH_MIN_HITS", "2").strip()
    try:
        return max(1, int(raw_value))
    except Exception:
        return 2


def _vits_decoder_cudagraph_max_entries() -> int:
    raw_value = os.environ.get("GPTSOVITS_VITS_DECODER_CUDAGRAPH_MAX_ENTRIES", "2").strip()
    try:
        return max(1, int(raw_value))
    except Exception:
        return 2


def _vits_decoder_cudagraph_shape_whitelist() -> set[str]:
    raw_value = os.environ.get("GPTSOVITS_VITS_DECODER_CUDAGRAPH_SHAPES", "").strip()
    if not raw_value:
        return set()
    return {item.strip() for item in raw_value.split(",") if item.strip()}


def _vits_decoder_tf32_enabled() -> bool:
    return os.environ.get("GPTSOVITS_VITS_DECODER_TF32", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


def _vits_decoder_channels_last_resblock_enabled() -> bool:
    return os.environ.get("GPTSOVITS_VITS_DECODER_CHANNELS_LAST_RESBLOCK", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


@contextlib.contextmanager
def _vits_decoder_tf32_context(x: torch.Tensor):
    if (
        not _vits_decoder_tf32_enabled()
        or x.device.type != "cuda"
        or x.dtype != torch.float32
    ):
        yield
        return
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


WINDOW = {}

class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
        version="v2",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels
        self.version = version

        self.ssl_proj = nn.Conv1d(768, hidden_channels, 1)

        self.encoder_ssl = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.encoder_text = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

        if self.version == "v1":
            symbols = symbols_v1.symbols
        else:
            symbols = symbols_v2.symbols
        self.text_embedding = nn.Embedding(len(symbols), hidden_channels)

        self.mrte = MRTE()

        self.encoder2 = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, y, y_lengths, text, text_lengths, ge, speed=1, test=None, result_length:int=None, overlap_frames:torch.Tensor=None, padding_length:int=None):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)

        y = self.ssl_proj(y * y_mask) * y_mask

        y = self.encoder_ssl(y * y_mask, y_mask)

        text_mask = torch.unsqueeze(commons.sequence_mask(text_lengths, text.size(1)), 1).to(y.dtype)
        if test == 1:
            text[:, :] = 0
        text = self.text_embedding(text).transpose(1, 2)
        text = self.encoder_text(text * text_mask, text_mask)
        y = self.mrte(y, y_mask, text, text_mask, ge)

        if padding_length is not None and padding_length!=0:
            y = y[:, :, :-padding_length]
            y_mask = y_mask[:, :, :-padding_length]


        y = self.encoder2(y * y_mask, y_mask)

        if result_length is not None:
            y = y[:, :, -result_length:]
            y_mask = y_mask[:, :, -result_length:]

        if overlap_frames is not None:
            overlap_len = overlap_frames.shape[-1]
            window = WINDOW.get(overlap_len, None)
            if window is None:
                # WINDOW[overlap_len] = torch.hann_window(overlap_len*2, device=y.device, dtype=y.dtype)
                WINDOW[overlap_len] = torch.sin(torch.arange(overlap_len*2, device=y.device) * torch.pi / (overlap_len*2))
                window = WINDOW[overlap_len]


            window = window.to(y.device)
            y[:,:,:overlap_len] = (
                window[:overlap_len].view(1, 1, -1) * y[:,:,:overlap_len]
                + window[overlap_len:].view(1, 1, -1) * overlap_frames
            )
            
        y_ = y
        y_mask_ = y_mask



        if speed != 1:
            y = F.interpolate(y, size=int(y.shape[-1] / speed) + 1, mode="linear")
            y_mask = F.interpolate(y_mask, size=y.shape[-1], mode="nearest")
        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask, y_, y_mask_

    def extract_latent(self, x):
        x = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x)
        return codes.transpose(0, 1)

    def decode_latent(self, codes, y_mask, refer, refer_mask, ge):
        quantized = self.quantizer.decode(codes)

        y = self.vq_proj(quantized) * y_mask
        y = self.encoder_ssl(y * y_mask, y_mask)

        y = self.mrte(y, y_mask, refer, refer_mask, ge)

        y = self.encoder2(y * y_mask, y_mask)

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask, quantized


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        if g != None:
            g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Encoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_lengths, g=None):
        if g != None:
            g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        return stats, x_mask


class WNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.norm = modules.LayerNorm(out_channels)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        out = self.proj(x) * x_mask
        out = self.norm(out)
        return out


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
        is_bias=False,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=is_bias)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    @staticmethod
    def _profile_decoder_enabled() -> bool:
        return str(os.environ.get("GPTSOVITS_PROFILE_VITS_DECODE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _accumulate_profile_metric(stats, key, elapsed_ms):
        stats[key] = float(stats.get(key, 0.0)) + float(elapsed_ms)

    @staticmethod
    def _profile_call(fn, stats, key, device):
        _sync_profile_device(device)
        started = time.perf_counter()
        result = fn()
        _sync_profile_device(device)
        Generator._accumulate_profile_metric(stats, key, (time.perf_counter() - started) * 1000.0)
        return result

    def _run_resblock_profiled(self, block, x, stats, prefix, device):
        block_started = time.perf_counter()
        if hasattr(block, "convs1") and hasattr(block, "convs2"):
            for layer_index, (c1, c2) in enumerate(zip(block.convs1, block.convs2)):
                residual = x
                xt = self._profile_call(
                    lambda current=x: F.leaky_relu(current, modules.LRELU_SLOPE),
                    stats,
                    f"{prefix}_layer_{layer_index}_relu1_ms",
                    device,
                )
                xt = self._profile_call(
                    lambda current=xt, conv=c1: conv(current),
                    stats,
                    f"{prefix}_layer_{layer_index}_conv1_ms",
                    device,
                )
                xt = self._profile_call(
                    lambda current=xt: F.leaky_relu(current, modules.LRELU_SLOPE),
                    stats,
                    f"{prefix}_layer_{layer_index}_relu2_ms",
                    device,
                )
                xt = self._profile_call(
                    lambda current=xt, conv=c2: conv(current),
                    stats,
                    f"{prefix}_layer_{layer_index}_conv2_ms",
                    device,
                )
                x = self._profile_call(
                    lambda current=xt, residual=residual: current + residual,
                    stats,
                    f"{prefix}_layer_{layer_index}_residual_add_ms",
                    device,
                )
        elif hasattr(block, "convs"):
            for layer_index, conv in enumerate(block.convs):
                residual = x
                xt = self._profile_call(
                    lambda current=x: F.leaky_relu(current, modules.LRELU_SLOPE),
                    stats,
                    f"{prefix}_layer_{layer_index}_relu_ms",
                    device,
                )
                xt = self._profile_call(
                    lambda current=xt, conv=conv: conv(current),
                    stats,
                    f"{prefix}_layer_{layer_index}_conv_ms",
                    device,
                )
                x = self._profile_call(
                    lambda current=xt, residual=residual: current + residual,
                    stats,
                    f"{prefix}_layer_{layer_index}_residual_add_ms",
                    device,
                )
        else:
            x = self._profile_call(
                lambda block=block, current=x: block(current),
                stats,
                f"{prefix}_ms",
                device,
            )
        stats[f"{prefix}_total_ms"] = float((time.perf_counter() - block_started) * 1000.0)
        return x

    @staticmethod
    def _use_channels_last_resblock_runtime(x: torch.Tensor) -> bool:
        return (
            _vits_decoder_channels_last_resblock_enabled()
            and x.device.type == "cuda"
            and x.dtype in {torch.float16, torch.bfloat16}
        )

    def _run_stage_resblocks_channels_last(self, x, stage_index: int):
        stage_x = x.unsqueeze(2).contiguous(memory_format=torch.channels_last)
        xs = None
        stage_base = stage_index * self.num_kernels
        for j in range(self.num_kernels):
            block = self.resblocks[stage_base + j]
            block_out = block.forward_channels_last(stage_x)
            xs = block_out if xs is None else xs + block_out
        return (xs / self.num_kernels).squeeze(2)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            if self._use_channels_last_resblock_runtime(x):
                x = self._run_stage_resblocks_channels_last(x, i)
            else:
                xs = None
                for j in range(self.num_kernels):
                    if xs is None:
                        xs = self.resblocks[i * self.num_kernels + j](x)
                    else:
                        xs += self.resblocks[i * self.num_kernels + j](x)
                x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def forward_profiled(self, x, g=None):
        stats = {
            "decoder_profiled_calls": 1,
            "decoder_num_upsamples": int(self.num_upsamples),
            "decoder_num_kernels": int(self.num_kernels),
        }
        total_started = time.perf_counter()
        device = x.device
        x = self._profile_call(lambda: self.conv_pre(x), stats, "decoder_conv_pre_ms", device)
        if g is not None:
            x = self._profile_call(lambda: x + self.cond(g), stats, "decoder_cond_add_ms", device)

        for i in range(self.num_upsamples):
            stage_started = time.perf_counter()
            x = self._profile_call(
                lambda: F.leaky_relu(x, modules.LRELU_SLOPE),
                stats,
                f"decoder_stage_{i}_pre_relu_ms",
                device,
            )
            x = self._profile_call(
                lambda: self.ups[i](x),
                stats,
                f"decoder_stage_{i}_upsample_ms",
                device,
            )
            xs = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j]
                block_out = self._run_resblock_profiled(
                    block,
                    x,
                    stats,
                    f"decoder_stage_{i}_resblock_{j}",
                    device,
                )
                self._accumulate_profile_metric(
                    stats,
                    f"decoder_stage_{i}_resblock_{j}_ms",
                    stats.get(f"decoder_stage_{i}_resblock_{j}_total_ms", 0.0),
                )
                if xs is None:
                    xs = block_out
                else:
                    xs = self._profile_call(
                        lambda current=xs, block_out=block_out: current + block_out,
                        stats,
                        f"decoder_stage_{i}_resblock_accum_ms",
                        device,
                    )
            x = self._profile_call(
                lambda: xs / self.num_kernels,
                stats,
                f"decoder_stage_{i}_resblock_avg_ms",
                device,
            )
            stats[f"decoder_stage_{i}_total_ms"] = float((time.perf_counter() - stage_started) * 1000.0)
        x = self._profile_call(lambda: F.leaky_relu(x), stats, "decoder_post_relu_ms", device)
        x = self._profile_call(lambda: self.conv_post(x), stats, "decoder_conv_post_ms", device)
        x = self._profile_call(lambda: torch.tanh(x), stats, "decoder_tanh_ms", device)
        stats["decoder_total_ms"] = float((time.perf_counter() - total_started) * 1000.0)
        return x, stats

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
            if hasattr(l, "prepare_channels_last_runtime"):
                l.prepare_channels_last_runtime()
                if self.conv_pre.weight.device.type == "cuda":
                    l.prepare_channels_last_runtime(dtype=torch.float16)
                    if torch.cuda.is_bf16_supported():
                        l.prepare_channels_last_runtime(dtype=torch.bfloat16)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


v2pro_set = {"v2Pro", "v2ProPlus"}


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, version=None):
        super(MultiPeriodDiscriminator, self).__init__()
        if version in v2pro_set:
            periods = [2, 3, 5, 7, 11, 17, 23]
        else:
            periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0)).unsqueeze(-1)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class Quantizer_module(torch.nn.Module):
    def __init__(self, n_e, e_dim):
        super(Quantizer_module, self).__init__()
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, x):
        d = (
            torch.sum(x**2, 1, keepdim=True)
            + torch.sum(self.embedding.weight**2, 1)
            - 2 * torch.matmul(x, self.embedding.weight.T)
        )
        min_indicies = torch.argmin(d, 1)
        z_q = self.embedding(min_indicies)
        return z_q, min_indicies


class Quantizer(torch.nn.Module):
    def __init__(self, embed_dim=512, n_code_groups=4, n_codes=160):
        super(Quantizer, self).__init__()
        assert embed_dim % n_code_groups == 0
        self.quantizer_modules = nn.ModuleList(
            [Quantizer_module(n_codes, embed_dim // n_code_groups) for _ in range(n_code_groups)]
        )
        self.n_code_groups = n_code_groups
        self.embed_dim = embed_dim

    def forward(self, xin):
        # B, C, T
        B, C, T = xin.shape
        xin = xin.transpose(1, 2)
        x = xin.reshape(-1, self.embed_dim)
        x = torch.split(x, self.embed_dim // self.n_code_groups, dim=-1)
        min_indicies = []
        z_q = []
        for _x, m in zip(x, self.quantizer_modules):
            _z_q, _min_indicies = m(_x)
            z_q.append(_z_q)
            min_indicies.append(_min_indicies)  # B * T,
        z_q = torch.cat(z_q, -1).reshape(xin.shape)
        loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
        z_q = xin + (z_q - xin).detach()
        z_q = z_q.transpose(1, 2)
        codes = torch.stack(min_indicies, -1).reshape(B, T, self.n_code_groups)
        return z_q, loss, codes.transpose(1, 2)

    def embed(self, x):
        # idx: N, 4, T
        x = x.transpose(1, 2)
        x = torch.split(x, 1, 2)
        ret = []
        for q, embed in zip(x, self.quantizer_modules):
            q = embed.embedding(q.squeeze(-1))
            ret.append(q)
        ret = torch.cat(ret, -1)
        return ret.transpose(1, 2)  # N, C, T


class CodePredictor(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        n_q=8,
        dims=1024,
        ssl_dim=768,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.vq_proj = nn.Conv1d(ssl_dim, hidden_channels, 1)
        self.ref_enc = modules.MelStyleEncoder(ssl_dim, style_vector_dim=hidden_channels)

        self.encoder = attentions.Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)

        self.out_proj = nn.Conv1d(hidden_channels, (n_q - 1) * dims, 1)
        self.n_q = n_q
        self.dims = dims

    def forward(self, x, x_mask, refer, codes, infer=False):
        x = x.detach()
        x = self.vq_proj(x * x_mask) * x_mask
        g = self.ref_enc(refer, x_mask)
        x = x + g
        x = self.encoder(x * x_mask, x_mask)
        x = self.out_proj(x * x_mask) * x_mask
        logits = x.reshape(x.shape[0], self.n_q - 1, self.dims, x.shape[-1]).transpose(2, 3)
        target = codes[1:].transpose(0, 1)
        if not infer:
            logits = logits.reshape(-1, self.dims)
            target = target.reshape(-1)
            loss = torch.nn.functional.cross_entropy(logits, target)
            return loss
        else:
            _, top10_preds = torch.topk(logits, 10, dim=-1)
            correct_top10 = torch.any(top10_preds == target.unsqueeze(-1), dim=-1)
            top3_acc = 100 * torch.mean(correct_top10.float()).detach().cpu().item()

            print("Top-10 Accuracy:", top3_acc, "%")

            pred_codes = torch.argmax(logits, dim=-1)
            acc = 100 * torch.mean((pred_codes == target).float()).detach().cpu().item()
            print("Top-1 Accuracy:", acc, "%")

            return pred_codes.transpose(0, 1)


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        semantic_frame_rate=None,
        freeze_quantizer=None,
        version="v2",
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.version = version

        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            version=version,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        # self.version=os.environ.get("version","v1")
        if self.version == "v1":
            self.ref_enc = modules.MelStyleEncoder(spec_channels, style_vector_dim=gin_channels)
        else:
            self.ref_enc = modules.MelStyleEncoder(704, style_vector_dim=gin_channels)

        ssl_dim = 768
        assert semantic_frame_rate in ["25hz", "50hz"]
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer

        self.is_v2pro = self.version in v2pro_set
        if self.is_v2pro:
            self.sv_emb = nn.Linear(20480, gin_channels)
            self.ge_to512 = nn.Linear(gin_channels, 512)
            self.prelu = nn.PReLU(num_parameters=gin_channels)
        self._last_decoder_profile = {}
        self._last_decoder_runtime_stats = {}
        self._compiled_dec_static_buckets = {}
        self._compiled_dec_static_bucket_seen = {}
        self._compiled_dec_static_bucket_failures = set()
        self._compiled_dec_static_bucket_lock = threading.Lock()
        self._decoder_cuda_graphs = {}
        self._decoder_cuda_graph_seen = {}
        self._decoder_cuda_graph_failures = set()
        self._decoder_cuda_graph_failure_reasons = {}
        self._decoder_cuda_graph_lock = threading.Lock()

    def get_last_decoder_profile(self):
        return dict(self._last_decoder_profile)

    def _set_last_decoder_profile(self, stats):
        self._last_decoder_profile = dict(stats)

    def get_last_decoder_runtime_stats(self):
        return dict(self._last_decoder_runtime_stats)

    def _set_last_decoder_runtime_stats(self, stats):
        self._last_decoder_runtime_stats = dict(stats)

    @staticmethod
    def _decoder_static_bucket_key(x: torch.Tensor, g: torch.Tensor | None) -> tuple[object, ...]:
        return (
            tuple(int(dim) for dim in x.shape),
            str(x.dtype),
            str(x.device),
            None if g is None else tuple(int(dim) for dim in g.shape),
            None if g is None else str(g.dtype),
            None if g is None else str(g.device),
        )

    @staticmethod
    def _decoder_runtime_shape_id(x: torch.Tensor, g: torch.Tensor | None) -> str:
        x_part = "x" + "x".join(str(int(dim)) for dim in x.shape)
        if g is None:
            return f"{x_part}_gNone"
        g_part = "g" + "x".join(str(int(dim)) for dim in g.shape)
        return f"{x_part}_{g_part}"

    def _get_decoder_cuda_graph_entry(self, key: tuple[object, ...]):
        return self._decoder_cuda_graphs.get(key)

    def _maybe_capture_decoder_cuda_graph(
        self,
        key: tuple[object, ...],
        shape_id: str,
        runtime_dec,
        x: torch.Tensor,
        g: torch.Tensor | None,
        runtime_stats: dict[str, object],
    ) -> None:
        if not _vits_decoder_cudagraph_enabled():
            return
        if runtime_dec is None or x.device.type != "cuda":
            return
        shape_whitelist = _vits_decoder_cudagraph_shape_whitelist()
        if shape_whitelist and shape_id not in shape_whitelist:
            return
        if key in self._decoder_cuda_graphs or key in self._decoder_cuda_graph_failures:
            return
        seen_hits = int(self._decoder_cuda_graph_seen.get(key, 0)) + 1
        self._decoder_cuda_graph_seen[key] = seen_hits
        if seen_hits < _vits_decoder_cudagraph_min_hits():
            return
        if len(self._decoder_cuda_graphs) >= _vits_decoder_cudagraph_max_entries():
            return
        with self._decoder_cuda_graph_lock:
            if key in self._decoder_cuda_graphs or key in self._decoder_cuda_graph_failures:
                return
            try:
                started_at = time.perf_counter()
                from vllm.platforms import current_platform

                pool = current_platform.get_global_graph_pool()
                static_x = torch.empty_like(x)
                static_x.copy_(x)
                static_g = None if g is None else torch.empty_like(g)
                if static_g is not None:
                    static_g.copy_(g)
                with torch.no_grad():
                    _ = runtime_dec(static_x, g=static_g)
                _sync_profile_device(x.device)
                graph = torch.cuda.CUDAGraph()
                with torch.no_grad():
                    with torch.cuda.graph(graph, pool=pool):
                        static_out = runtime_dec(static_x, g=static_g)
                self._decoder_cuda_graphs[key] = {
                    "graph": graph,
                    "x": static_x,
                    "g": static_g,
                    "out": static_out,
                }
                runtime_stats["decoder_runtime_cudagraph_capture_ms"] = float((time.perf_counter() - started_at) * 1000.0)
                runtime_stats["decoder_runtime_cudagraph_captures"] = 1
            except Exception as exc:
                self._decoder_cuda_graph_failures.add(key)
                self._decoder_cuda_graph_failure_reasons[key] = f"{type(exc).__name__}: {exc}"
                runtime_stats["decoder_runtime_cudagraph_failure_reason"] = self._decoder_cuda_graph_failure_reasons[key]

    def _decode_audio_runtime(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
        *,
        borrow_output: bool = False,
    ):
        runtime_dec = getattr(self, "_compiled_dec", None)
        static_bucket_key = self._decoder_static_bucket_key(x, g)
        shape_id = self._decoder_runtime_shape_id(x, g)
        static_runtime = None
        cuda_graph_entry = self._get_decoder_cuda_graph_entry(static_bucket_key)
        runtime_stats = {
            "decoder_runtime_calls": 1,
            "decoder_runtime_compiled_available": bool(runtime_dec is not None),
            "decoder_runtime_x_shape": [int(dim) for dim in x.shape],
            "decoder_runtime_x_dtype": str(x.dtype),
            "decoder_runtime_x_device": str(x.device),
            "decoder_runtime_g_shape": None if g is None else [int(dim) for dim in g.shape],
            "decoder_runtime_static_bucket_key": repr(static_bucket_key),
            "decoder_runtime_shape_id": shape_id,
            "decoder_runtime_cudagraph_hits": 0,
            "decoder_runtime_cudagraph_misses": 0,
            "decoder_runtime_cudagraph_captures": 0,
            "decoder_runtime_path_hist": {},
            "decoder_runtime_shape_hist": {shape_id: 1},
            "decoder_runtime_cudagraph_shape_hist": {},
            "decoder_runtime_cudagraph_miss_shape_hist": {},
        }
        if getattr(self.dec, "_profile_decoder_enabled", None) is not None and self.dec._profile_decoder_enabled():
            audio, stats = self.dec.forward_profiled(x, g=g)
            runtime_stats["decoder_runtime_profiled_calls"] = 1
            runtime_stats["decoder_runtime_compiled_calls"] = 0
            runtime_stats["decoder_runtime_eager_calls"] = 0
            runtime_stats["decoder_runtime_static_bucket_hits"] = 0
            runtime_stats["decoder_runtime_static_bucket_misses"] = 0
            runtime_stats["decoder_runtime_path"] = "profiled_eager"
            runtime_stats["decoder_runtime_path_hist"] = {"profiled_eager": 1}
            self._set_last_decoder_profile(stats)
            self._set_last_decoder_runtime_stats(runtime_stats)
            return audio
        if cuda_graph_entry is not None:
            started_at = time.perf_counter()
            cuda_graph_entry["x"].copy_(x)
            runtime_stats["decoder_runtime_cudagraph_input_copy_ms"] = float((time.perf_counter() - started_at) * 1000.0)
            static_g = cuda_graph_entry["g"]
            if static_g is not None and g is not None:
                started_at = time.perf_counter()
                static_g.copy_(g)
                runtime_stats["decoder_runtime_cudagraph_cond_copy_ms"] = float((time.perf_counter() - started_at) * 1000.0)
            started_at = time.perf_counter()
            with _vits_decoder_tf32_context(x):
                cuda_graph_entry["graph"].replay()
            runtime_stats["decoder_runtime_cudagraph_replay_ms"] = float((time.perf_counter() - started_at) * 1000.0)
            runtime_stats["decoder_runtime_profiled_calls"] = 0
            runtime_stats["decoder_runtime_compiled_calls"] = 1
            runtime_stats["decoder_runtime_eager_calls"] = 0
            runtime_stats["decoder_runtime_static_bucket_hits"] = 0
            runtime_stats["decoder_runtime_static_bucket_misses"] = 0
            runtime_stats["decoder_runtime_cudagraph_hits"] = 1
            runtime_stats["decoder_runtime_path_hist"] = {"compiled_cudagraph": 1}
            runtime_stats["decoder_runtime_cudagraph_shape_hist"] = {shape_id: 1}
            runtime_stats["decoder_runtime_path"] = "compiled_cudagraph"
            self._set_last_decoder_profile({})
            if borrow_output:
                runtime_stats["decoder_runtime_output_borrowed"] = 1
                self._set_last_decoder_runtime_stats(runtime_stats)
                return cuda_graph_entry["out"]
            started_at = time.perf_counter()
            output = cuda_graph_entry["out"].clone()
            runtime_stats["decoder_runtime_cudagraph_output_clone_ms"] = float((time.perf_counter() - started_at) * 1000.0)
            self._set_last_decoder_runtime_stats(runtime_stats)
            return output
        if runtime_dec is not None and _vits_static_bucket_compile_enabled():
            static_runtime = self._compiled_dec_static_buckets.get(static_bucket_key)
            if static_runtime is not None:
                runtime_stats["decoder_runtime_profiled_calls"] = 0
                runtime_stats["decoder_runtime_compiled_calls"] = 1
                runtime_stats["decoder_runtime_eager_calls"] = 0
                runtime_stats["decoder_runtime_static_bucket_hits"] = 1
                runtime_stats["decoder_runtime_static_bucket_misses"] = 0
                runtime_stats["decoder_runtime_path_hist"] = {"compiled_static_bucket": 1}
                runtime_stats["decoder_runtime_path"] = "compiled_static_bucket"
                self._set_last_decoder_profile({})
                self._set_last_decoder_runtime_stats(runtime_stats)
                with _vits_decoder_tf32_context(x):
                    return static_runtime(x, g=g)
            runtime_stats["decoder_runtime_static_bucket_hits"] = 0
            runtime_stats["decoder_runtime_static_bucket_misses"] = 1
            if static_bucket_key not in self._compiled_dec_static_bucket_failures:
                seen_hits = int(self._compiled_dec_static_bucket_seen.get(static_bucket_key, 0)) + 1
                self._compiled_dec_static_bucket_seen[static_bucket_key] = seen_hits
                if seen_hits >= _vits_static_bucket_min_hits():
                    with self._compiled_dec_static_bucket_lock:
                        if (
                            static_bucket_key not in self._compiled_dec_static_buckets
                            and static_bucket_key not in self._compiled_dec_static_bucket_failures
                        ):
                            try:
                                started_at = time.perf_counter()
                                static_runtime = torch.compile(
                                    self.dec,
                                    mode=_vits_static_bucket_compile_mode(),
                                    dynamic=False,
                                )
                                static_runtime(x, g=g)
                                runtime_stats["decoder_runtime_static_bucket_compile_ms"] = float(
                                    (time.perf_counter() - started_at) * 1000.0
                                )
                                self._compiled_dec_static_buckets[static_bucket_key] = static_runtime
                            except Exception:
                                self._compiled_dec_static_bucket_failures.add(static_bucket_key)
        if runtime_dec is not None:
            runtime_stats["decoder_runtime_profiled_calls"] = 0
            runtime_stats["decoder_runtime_compiled_calls"] = 1
            runtime_stats["decoder_runtime_eager_calls"] = 0
            runtime_stats.setdefault("decoder_runtime_static_bucket_hits", 0)
            runtime_stats.setdefault("decoder_runtime_static_bucket_misses", 0)
            runtime_stats["decoder_runtime_cudagraph_misses"] = 1 if _vits_decoder_cudagraph_enabled() else 0
            if runtime_stats["decoder_runtime_cudagraph_misses"]:
                runtime_stats["decoder_runtime_cudagraph_miss_shape_hist"] = {shape_id: 1}
            self._maybe_capture_decoder_cuda_graph(static_bucket_key, shape_id, runtime_dec, x, g, runtime_stats)
            if static_bucket_key in self._decoder_cuda_graph_failure_reasons:
                runtime_stats["decoder_runtime_cudagraph_failure_reason"] = self._decoder_cuda_graph_failure_reasons[
                    static_bucket_key
                ]
            runtime_stats["decoder_runtime_path_hist"] = {"compiled": 1}
            runtime_stats["decoder_runtime_path"] = "compiled"
            self._set_last_decoder_profile({})
            self._set_last_decoder_runtime_stats(runtime_stats)
            with _vits_decoder_tf32_context(x):
                return runtime_dec(x, g=g)
        runtime_stats["decoder_runtime_profiled_calls"] = 0
        runtime_stats["decoder_runtime_compiled_calls"] = 0
        runtime_stats["decoder_runtime_eager_calls"] = 1
        runtime_stats["decoder_runtime_static_bucket_hits"] = 0
        runtime_stats["decoder_runtime_static_bucket_misses"] = 0
        runtime_stats["decoder_runtime_path_hist"] = {"eager": 1}
        runtime_stats["decoder_runtime_path"] = "eager"
        self._set_last_decoder_profile({})
        self._set_last_decoder_runtime_stats(runtime_stats)
        with _vits_decoder_tf32_context(x):
            return self.dec(x, g=g)

    def forward(self, ssl, y, y_lengths, text, text_lengths, sv_emb=None):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        if self.version == "v1":
            ge = self.ref_enc(y * y_mask, y_mask)
        else:
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)
        if self.is_v2pro:
            sv_emb = self.sv_emb(sv_emb)  # B*20480->B*512
            ge += sv_emb.unsqueeze(-1)
            ge = self.prelu(ge)
            ge512 = self.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
        with autocast(enabled=False):
            maybe_no_grad = torch.no_grad() if self.freeze_quantizer else contextlib.nullcontext()
            with maybe_no_grad:
                if self.freeze_quantizer:
                    self.ssl_proj.eval()
                    self.quantizer.eval()
            ssl = self.ssl_proj(ssl)
            quantized, codes, commit_loss, quantized_list = self.quantizer(ssl, layers=[0])

        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")

        x, m_p, logs_p, y_mask, _, _ = self.enc_p(quantized, y_lengths, text, text_lengths, ge512 if self.is_v2pro else ge)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=ge)
        z_p = self.flow(z, y_mask, g=ge)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=ge)
        return (
            o,
            commit_loss,
            ids_slice,
            y_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )

    def infer(self, ssl, y, y_lengths, text, text_lengths, test=None, noise_scale=0.5):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        if self.version == "v1":
            ge = self.ref_enc(y * y_mask, y_mask)
        else:
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)

        ssl = self.ssl_proj(ssl)
        quantized, codes, commit_loss, _ = self.quantizer(ssl, layers=[0])
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")

        x, m_p, logs_p, y_mask, _, _ = self.enc_p(quantized, y_lengths, text, text_lengths, ge, test=test)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self._decode_audio_runtime((z * y_mask)[:, :, :], g=ge)
        return o, y_mask, (z, z_p, m_p, logs_p)


    @torch.no_grad()
    def decode(self, codes, text, refer, noise_scale=0.5, speed=1, sv_emb=None):
        def get_ge(refer, sv_emb):
            ge = None
            if refer is not None:
                refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
                refer_mask = torch.unsqueeze(commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
                if self.version == "v1":
                    ge = self.ref_enc(refer * refer_mask, refer_mask)
                else:
                    ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
                if self.is_v2pro:
                    sv_emb = self.sv_emb(sv_emb)  # B*20480->B*512
                    ge += sv_emb.unsqueeze(-1)
                    ge = self.prelu(ge)
            return ge

        if type(refer) == list:
            ges = []
            for idx, _refer in enumerate(refer):
                ge = get_ge(_refer, sv_emb[idx] if self.is_v2pro else None)
                ges.append(ge)
            ge = torch.stack(ges, 0).mean(0)
        else:
            ge = get_ge(refer, sv_emb)

        y_lengths = torch.LongTensor([codes.size(2) * 2]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
        x, m_p, logs_p, y_mask, _, _ = self.enc_p(
            quantized,
            y_lengths,
            text,
            text_lengths,
            self.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if self.is_v2pro else ge,
            speed,
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self._decode_audio_runtime((z * y_mask)[:, :, :], g=ge)
        return o

    @torch.no_grad()
    def decode_batched_request_local(
        self,
        codes: torch.Tensor,
        code_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        refer_list: List[torch.Tensor],
        noise_scale: float = 0.5,
        speed: float = 1,
        sv_emb: torch.Tensor | None = None,
        shared_refer: bool = False,
        precomputed_ge: torch.Tensor | None = None,
    ):
        batch_size = int(codes.size(1))
        if batch_size <= 0:
            raise ValueError("decode_batched_request_local 收到空 batch")
        if shared_refer:
            if len(refer_list) <= 0:
                raise ValueError("shared_refer=True 时 refer_list 不能为空")
        elif len(refer_list) != batch_size:
            raise ValueError("refer_list 数量与 batch size 不一致")

        if shared_refer:
            if precomputed_ge is not None:
                ge = precomputed_ge
            else:
                shared_refer_tensor = refer_list[0]
                refer_lengths = torch.LongTensor([int(shared_refer_tensor.size(2))]).to(codes.device)
                refer_mask = torch.unsqueeze(
                    commons.sequence_mask(refer_lengths, int(shared_refer_tensor.size(2))),
                    1,
                ).to(shared_refer_tensor.dtype)
                if self.version == "v1":
                    ge = self.ref_enc(shared_refer_tensor * refer_mask, refer_mask)
                else:
                    ge = self.ref_enc(shared_refer_tensor[:, :704] * refer_mask, refer_mask)
                if self.is_v2pro:
                    if sv_emb is None:
                        raise ValueError("v2Pro batched request-local synthesis 缺少 sv_emb")
                    shared_sv_emb = sv_emb[:1]
                    ge = ge + self.sv_emb(shared_sv_emb).unsqueeze(-1)
                    ge = self.prelu(ge)
            ge = ge.expand(batch_size, -1, -1)
        else:
            refer_lengths = torch.LongTensor([int(item.size(2)) for item in refer_list]).to(codes.device)
            max_refer_len = int(refer_lengths.max().item())
            refer_batch = torch.zeros(
                (batch_size, int(refer_list[0].size(1)), max_refer_len),
                dtype=refer_list[0].dtype,
                device=codes.device,
            )
            for batch_index, refer in enumerate(refer_list):
                refer_batch[batch_index, :, : int(refer.size(2))] = refer.squeeze(0)
            refer_mask = torch.unsqueeze(commons.sequence_mask(refer_lengths, max_refer_len), 1).to(refer_batch.dtype)
            if self.version == "v1":
                ge = self.ref_enc(refer_batch * refer_mask, refer_mask)
            else:
                ge = self.ref_enc(refer_batch[:, :704] * refer_mask, refer_mask)
            if self.is_v2pro:
                if sv_emb is None:
                    raise ValueError("v2Pro batched request-local synthesis 缺少 sv_emb")
                ge = ge + self.sv_emb(sv_emb).unsqueeze(-1)
                ge = self.prelu(ge)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")
        y_lengths = code_lengths.to(device=codes.device, dtype=torch.long) * 2
        text_lengths = text_lengths.to(device=text.device, dtype=torch.long)
        x, m_p, logs_p, y_mask, _, _ = self.enc_p(
            quantized,
            y_lengths,
            text,
            text_lengths,
            self.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if self.is_v2pro else ge,
            speed,
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=ge, reverse=True)
        audio = self._decode_audio_runtime((z * y_mask)[:, :, :], g=ge)
        upsample_factor = 1
        for up_layer in self.dec.ups:
            stride = up_layer.stride[0] if isinstance(up_layer.stride, tuple) else int(up_layer.stride)
            upsample_factor *= int(stride)
        audio_lengths = y_mask.squeeze(1).sum(dim=1).to(dtype=torch.long) * int(upsample_factor)
        return audio, audio_lengths


    @torch.no_grad()
    def decode_streaming(self, codes, text, refer, noise_scale=0.5, speed=1, sv_emb=None, result_length:int=None, overlap_frames:torch.Tensor=None, padding_length:int=None):
        def get_ge(refer, sv_emb):
            ge = None
            if refer is not None:
                refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
                refer_mask = torch.unsqueeze(commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
                if self.version == "v1":
                    ge = self.ref_enc(refer * refer_mask, refer_mask)
                else:
                    ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
                if self.is_v2pro:
                    sv_emb = self.sv_emb(sv_emb)  # B*20480->B*512
                    ge += sv_emb.unsqueeze(-1)
                    ge = self.prelu(ge)
            return ge

        if type(refer) == list:
            ges = []
            for idx, _refer in enumerate(refer):
                ge = get_ge(_refer, sv_emb[idx] if self.is_v2pro else None)
                ges.append(ge)
            ge = torch.stack(ges, 0).mean(0)
        else:
            ge = get_ge(refer, sv_emb)

        y_lengths = torch.LongTensor([codes.size(2) * 2]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
            result_length = (2*result_length) if result_length is not None else None
            padding_length = (2*padding_length) if padding_length is not None else None
        x, m_p, logs_p, y_mask, y_, y_mask_ = self.enc_p(
            quantized,
            y_lengths,
            text,
            text_lengths,
            self.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if self.is_v2pro else ge,
            speed,
            result_length=result_length, 
            overlap_frames=overlap_frames, 
            padding_length=padding_length
            )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self._decode_audio_runtime((z * y_mask)[:, :, :], g=ge)
        return o, y_, y_mask_

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0, 1)


class CFM(torch.nn.Module):
    def __init__(self, in_channels, dit):
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator = dit

        self.in_channels = in_channels

        self.criterion = torch.nn.MSELoss()

        self.use_conditioner_cache = True

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, n_timesteps, temperature=1.0, inference_cfg_rate=0):
        """Forward diffusion"""
        B, T = mu.size(0), mu.size(1)
        x = torch.randn([B, self.in_channels, T], device=mu.device, dtype=mu.dtype) * temperature
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x, dtype=mu.dtype)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        mu = mu.transpose(2, 1)
        t = 0
        d = 1 / n_timesteps
        text_cache = None
        text_cfg_cache = None
        dt_cache = None
        d_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * d
        for j in range(n_timesteps):
            t_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * t
            # v_pred = model(x, t_tensor, d_tensor, **extra_args)
            v_pred, text_emb, dt = self.estimator(
                x,
                prompt_x,
                x_lens,
                t_tensor,
                d_tensor,
                mu,
                use_grad_ckpt=False,
                drop_audio_cond=False,
                drop_text=False,
                infer=True,
                text_cache=text_cache,
                dt_cache=dt_cache,
            )
            v_pred = v_pred.transpose(2, 1)
            if self.use_conditioner_cache:
                text_cache = text_emb
                dt_cache = dt
            if inference_cfg_rate > 1e-5:
                neg, text_cfg_emb, _ = self.estimator(
                    x,
                    prompt_x,
                    x_lens,
                    t_tensor,
                    d_tensor,
                    mu,
                    use_grad_ckpt=False,
                    drop_audio_cond=True,
                    drop_text=True,
                    infer=True,
                    text_cache=text_cfg_cache,
                    dt_cache=dt_cache,
                )
                neg = neg.transpose(2, 1)
                if self.use_conditioner_cache:
                    text_cfg_cache = text_cfg_emb
                v_pred = v_pred + (v_pred - neg) * inference_cfg_rate
            x = x + d * v_pred
            t = t + d
            x[:, :, :prompt_len] = 0
        return x

    def forward(self, x1, x_lens, prompt_lens, mu, use_grad_ckpt):
        b, _, t = x1.shape
        t = torch.rand([b], device=mu.device, dtype=x1.dtype)
        x0 = torch.randn_like(x1, device=mu.device)
        vt = x1 - x0
        xt = x0 + t[:, None, None] * vt
        dt = torch.zeros_like(t, device=mu.device)
        prompt = torch.zeros_like(x1)
        for i in range(b):
            prompt[i, :, : prompt_lens[i]] = x1[i, :, : prompt_lens[i]]
            xt[i, :, : prompt_lens[i]] = 0
        gailv = 0.3  # if ttime()>1736250488 else 0.1
        if random.random() < gailv:
            base = torch.randint(2, 8, (t.shape[0],), device=mu.device)
            d = 1 / torch.pow(2, base)
            d_input = d.clone()
            d_input[d_input < 1e-2] = 0
            # with torch.no_grad():
            v_pred_1 = self.estimator(xt, prompt, x_lens, t, d_input, mu, use_grad_ckpt).transpose(2, 1).detach()
            # v_pred_1 = self.diffusion(xt, t, d_input, cond=conditioning).detach()
            x_mid = xt + d[:, None, None] * v_pred_1
            # v_pred_2 = self.diffusion(x_mid, t+d, d_input, cond=conditioning).detach()
            v_pred_2 = self.estimator(x_mid, prompt, x_lens, t + d, d_input, mu, use_grad_ckpt).transpose(2, 1).detach()
            vt = (v_pred_1 + v_pred_2) / 2
            vt = vt.detach()
            dt = 2 * d

        vt_pred = self.estimator(xt, prompt, x_lens, t, dt, mu, use_grad_ckpt).transpose(2, 1)
        loss = 0
        for i in range(b):
            loss += self.criterion(vt_pred[i, :, prompt_lens[i] : x_lens[i]], vt[i, :, prompt_lens[i] : x_lens[i]])
        loss /= b

        return loss


def set_no_grad(net_g):
    for name, param in net_g.named_parameters():
        param.requires_grad = False


class SynthesizerTrnV3(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        semantic_frame_rate=None,
        freeze_quantizer=None,
        version="v3",
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.version = version

        self.model_dim = 512
        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(
            inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        # self.ref_enc = modules.MelStyleEncoder(spec_channels, style_vector_dim=gin_channels)###Rollback
        self.ref_enc = modules.MelStyleEncoder(704, style_vector_dim=gin_channels)  ###Rollback
        # self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
        #                      upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        # self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
        #                               gin_channels=gin_channels)
        # self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        ssl_dim = 768
        assert semantic_frame_rate in ["25hz", "50hz"]
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer
        inter_channels2 = 512
        self.bridge = nn.Sequential(nn.Conv1d(inter_channels, inter_channels2, 1, stride=1), nn.LeakyReLU())
        self.wns1 = Encoder(inter_channels2, inter_channels2, inter_channels2, 5, 1, 8, gin_channels=gin_channels)
        self.linear_mel = nn.Conv1d(inter_channels2, 100, 1, stride=1)
        self.cfm = CFM(
            100,
            DiT(**dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=inter_channels2, conv_layers=4)),
        )  # text_dim is condition feature dim
        if self.freeze_quantizer == True:
            set_no_grad(self.ssl_proj)
            set_no_grad(self.quantizer)
            set_no_grad(self.enc_p)

    def forward(
        self, ssl, y, mel, ssl_lengths, y_lengths, text, text_lengths, mel_lengths, use_grad_ckpt
    ):  # ssl_lengths no need now
        with autocast(enabled=False):
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)
            maybe_no_grad = torch.no_grad() if self.freeze_quantizer else contextlib.nullcontext()
            with maybe_no_grad:
                if self.freeze_quantizer:
                    self.ssl_proj.eval()  #
                    self.quantizer.eval()
                    self.enc_p.eval()
                ssl = self.ssl_proj(ssl)
                quantized, codes, commit_loss, quantized_list = self.quantizer(ssl, layers=[0])
                quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")  ##BCT
                x, m_p, logs_p, y_mask, y_, y_mask_ = self.enc_p(quantized, y_lengths, text, text_lengths, ge)
        fea = self.bridge(x)
        fea = F.interpolate(fea, scale_factor=(1.875 if self.version == "v3" else 2), mode="nearest")  ##BCT
        fea, y_mask_ = self.wns1(
            fea, mel_lengths, ge
        )  ##If the 1-minute fine-tuning works fine, no need to manually adjust the learning rate.
        B = ssl.shape[0]
        prompt_len_max = mel_lengths * 2 / 3
        prompt_len = (torch.rand([B], device=fea.device) * prompt_len_max).floor().to(dtype=torch.long)
        minn = min(mel.shape[-1], fea.shape[-1])
        mel = mel[:, :, :minn]
        fea = fea[:, :, :minn]
        cfm_loss = self.cfm(mel, mel_lengths, prompt_len, fea, use_grad_ckpt)
        return cfm_loss

    @torch.no_grad()
    def decode_encp(self, codes, text, refer, ge=None, speed=1):
        # print(2333333,refer.shape)
        # ge=None
        if ge == None:
            refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
            refer_mask = torch.unsqueeze(commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
            ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        y_lengths = torch.LongTensor([int(codes.size(2) * 2)]).to(codes.device)
        if speed == 1:
            sizee = int(codes.size(2) * (3.875 if self.version == "v3" else 4))
        else:
            sizee = int(codes.size(2) * (3.875 if self.version == "v3" else 4) / speed) + 1
        y_lengths1 = torch.LongTensor([sizee]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")  ##BCT
        x, m_p, logs_p, y_mask, _, _ = self.enc_p(quantized, y_lengths, text, text_lengths, ge, speed)
        fea = self.bridge(x)
        fea = F.interpolate(fea, scale_factor=(1.875 if self.version == "v3" else 2), mode="nearest")  ##BCT
        ####more wn paramter to learn mel
        fea, y_mask_ = self.wns1(fea, y_lengths1, ge)
        return fea, ge

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0, 1)


class SynthesizerTrnV3b(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        semantic_frame_rate=None,
        freeze_quantizer=None,
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.model_dim = 512
        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(
            inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        # self.ref_enc = modules.MelStyleEncoder(spec_channels, style_vector_dim=gin_channels)###Rollback
        self.ref_enc = modules.MelStyleEncoder(704, style_vector_dim=gin_channels)  ###Rollback
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels
        )
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        ssl_dim = 768
        assert semantic_frame_rate in ["25hz", "50hz"]
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer

        inter_channels2 = 512
        self.bridge = nn.Sequential(nn.Conv1d(inter_channels, inter_channels2, 1, stride=1), nn.LeakyReLU())
        self.wns1 = Encoder(inter_channels2, inter_channels2, inter_channels2, 5, 1, 8, gin_channels=gin_channels)
        self.linear_mel = nn.Conv1d(inter_channels2, 100, 1, stride=1)
        self.cfm = CFM(
            100,
            DiT(**dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=inter_channels2, conv_layers=4)),
        )  # text_dim is condition feature dim

    def forward(self, ssl, y, mel, ssl_lengths, y_lengths, text, text_lengths, mel_lengths):  # ssl_lengths no need now
        with autocast(enabled=False):
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)
            # ge = self.ref_enc(y * y_mask, y_mask)#change back, new spec setting is whole 24k
            # ge=None
            maybe_no_grad = torch.no_grad() if self.freeze_quantizer else contextlib.nullcontext()
            with maybe_no_grad:
                if self.freeze_quantizer:
                    self.ssl_proj.eval()
                    self.quantizer.eval()
                ssl = self.ssl_proj(ssl)
                quantized, codes, commit_loss, quantized_list = self.quantizer(ssl, layers=[0])
                quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")  ##BCT
                x, m_p, logs_p, y_mask, y_, y_mask_ = self.enc_p(quantized, y_lengths, text, text_lengths, ge)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=ge)
        z_p = self.flow(z, y_mask, g=ge)
        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=ge)
        fea = self.bridge(x)
        fea = F.interpolate(fea, scale_factor=1.875, mode="nearest")  ##BCT
        fea, y_mask_ = self.wns1(fea, mel_lengths, ge)
        learned_mel = self.linear_mel(fea)
        B = ssl.shape[0]
        prompt_len_max = mel_lengths * 2 / 3
        prompt_len = (torch.rand([B], device=fea.device) * prompt_len_max).floor().to(dtype=torch.long)  #
        minn = min(mel.shape[-1], fea.shape[-1])
        mel = mel[:, :, :minn]
        fea = fea[:, :, :minn]
        cfm_loss = self.cfm(mel, mel_lengths, prompt_len, fea)  # fea==cond,y_lengths==target_mel_lengths#ge not need
        return (
            commit_loss,
            cfm_loss,
            F.mse_loss(learned_mel, mel),
            o,
            ids_slice,
            y_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )

    @torch.no_grad()
    def decode_encp(self, codes, text, refer, ge=None):
        # print(2333333,refer.shape)
        # ge=None
        if ge == None:
            refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
            refer_mask = torch.unsqueeze(commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
            ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        y_lengths = torch.LongTensor([int(codes.size(2) * 2)]).to(codes.device)
        y_lengths1 = torch.LongTensor([int(codes.size(2) * 2.5 * 1.5)]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")  ##BCT
        x, m_p, logs_p, y_mask, y_, y_mask_ = self.enc_p(quantized, y_lengths, text, text_lengths, ge)
        fea = self.bridge(x)
        fea = F.interpolate(fea, scale_factor=1.875, mode="nearest")  ##BCT
        ####more wn paramter to learn mel
        fea, y_mask_ = self.wns1(fea, y_lengths1, ge)
        return fea, ge

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0, 1)
