from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.gpt_sovits.runtime import get_gpt_sovits_runtime
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _empty_float_tensor() -> torch.Tensor:
    return torch.zeros((0,), dtype=torch.float32)


def _empty_long_tensor() -> torch.Tensor:
    return torch.zeros((0,), dtype=torch.long)


class GPTSoVITSV2T2S(nn.Module):
    """Stage-0 GPT-SoVITS v2 semantic generator.

    This is an incremental integration step: it uses the vendored GPT-SoVITS
    prepare + continuous T2S scheduler as an explicit stage-0 model so stage-1
    waveform synthesis can be separated cleanly.
    """

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self.runtime = get_gpt_sovits_runtime()
        self._device = torch.device(vllm_config.device_config.device)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        return set()

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids is None:
            return torch.empty((0, 1), device=self._device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        del hidden_states, sampling_metadata
        return None

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, Any]]:
        return [{"skip_synthesis": True} for _ in range(max(1, num_reqs))]

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del input_ids, positions, intermediate_tensors, inputs_embeds, kwargs

        request_infos = runtime_additional_information or [{"skip_synthesis": True}]
        num_reqs = len(request_infos)

        semantic_tokens: list[torch.Tensor] = [_empty_long_tensor() for _ in range(num_reqs)]
        phones: list[torch.Tensor] = [_empty_long_tensor() for _ in range(num_reqs)]
        prompt_phones: list[torch.Tensor] = [_empty_long_tensor() for _ in range(num_reqs)]
        prompt_semantic: list[torch.Tensor] = [_empty_long_tensor() for _ in range(num_reqs)]
        refer_audio_spec: list[torch.Tensor] = [_empty_float_tensor() for _ in range(num_reqs)]
        refer_audio_16k: list[torch.Tensor] = [_empty_float_tensor() for _ in range(num_reqs)]
        raw_audio: list[torch.Tensor] = [_empty_float_tensor() for _ in range(num_reqs)]
        raw_sr: list[torch.Tensor] = [torch.tensor(0, dtype=torch.int32) for _ in range(num_reqs)]

        active_indices: list[int] = []
        prepared_requests = []
        for idx, info in enumerate(request_infos):
            if info.get("skip_synthesis"):
                continue
            prepared = self.runtime.prepare_request(
                info,
                request_id=str(info.get("engine_request_id") or f"gpt_sovits_v2_{idx}"),
            )
            active_indices.append(idx)
            prepared_requests.append(prepared)

        if prepared_requests:
            semantic_by_request = self.runtime.generate_semantic_tokens(prepared_requests)
            for idx, prepared in zip(active_indices, prepared_requests, strict=True):
                token_tensor = semantic_by_request.get(prepared.request_id)
                if token_tensor is None:
                    logger.warning("GPT-SoVITS v2 T2S produced no semantic tokens for %s", prepared.request_id)
                    continue
                semantic_tokens[idx] = token_tensor.to(dtype=torch.long)
                transport = prepared.transport_info
                phones[idx] = transport["gpt_sovits_phones"].to(dtype=torch.long)
                prompt_phones[idx] = transport["gpt_sovits_prompt_phones"].to(dtype=torch.long)
                prompt_semantic[idx] = transport["gpt_sovits_prompt_semantic"].to(dtype=torch.long)
                refer_audio_spec[idx] = transport["gpt_sovits_refer_audio_spec"].to(dtype=torch.float32)
                refer_audio_16k[idx] = transport["gpt_sovits_refer_audio_16k"].to(dtype=torch.float32)
                raw_audio[idx] = transport["gpt_sovits_raw_audio"].to(dtype=torch.float32)
                raw_sr[idx] = torch.tensor(int(transport["gpt_sovits_raw_sr"]), dtype=torch.int32)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "semantic_tokens": semantic_tokens,
                "gpt_sovits_phones": phones,
                "gpt_sovits_prompt_phones": prompt_phones,
                "gpt_sovits_prompt_semantic": prompt_semantic,
                "gpt_sovits_refer_audio_spec": refer_audio_spec,
                "gpt_sovits_refer_audio_16k": refer_audio_16k,
                "gpt_sovits_raw_audio": raw_audio,
                "gpt_sovits_raw_sr": raw_sr,
            },
        )
