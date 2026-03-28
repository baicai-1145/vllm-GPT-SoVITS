from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available

from vllm_omni.model_executor.models.gpt_sovits.runtime import get_gpt_sovits_runtime
from vllm_omni.model_executor.models.output_templates import OmniOutput


class GPTSoVITSV2Decode(nn.Module):
    """Stage-1 GPT-SoVITS v2 semantic-to-waveform decoder."""

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
        empty_long = torch.zeros((0,), dtype=torch.long)
        empty_float = torch.zeros((0,), dtype=torch.float32)
        return [
            {
                "gpt_sovits_phones": empty_long,
                "gpt_sovits_prompt_phones": empty_long,
                "gpt_sovits_prompt_semantic": empty_long,
                "gpt_sovits_refer_audio_spec": empty_float,
                "gpt_sovits_refer_audio_16k": empty_float,
                "gpt_sovits_raw_audio": empty_float,
                "gpt_sovits_raw_sr": 0,
                "gpt_sovits_semantic_token_count": 0,
            }
            for _ in range(max(1, num_reqs))
        ]

    @staticmethod
    def _has_decode_conditioning(info: dict[str, Any]) -> bool:
        required_keys = (
            "gpt_sovits_phones",
            "gpt_sovits_prompt_phones",
            "gpt_sovits_prompt_semantic",
            "gpt_sovits_refer_audio_spec",
            "gpt_sovits_refer_audio_16k",
            "gpt_sovits_raw_audio",
        )
        for key in required_keys:
            value = info.get(key)
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return False
                continue
            if value in (None, [], ()):
                return False
        return True

    def _split_request_ids(
        self,
        ids: torch.Tensor,
        seq_token_counts: list[int] | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
    ) -> list[torch.Tensor]:
        total = ids.numel()
        if total == 0:
            return [ids]
        if runtime_additional_information and all(
            isinstance(info.get("gpt_sovits_semantic_token_count"), int) and int(info["gpt_sovits_semantic_token_count"]) > 0
            for info in runtime_additional_information
        ):
            sizes = [int(info["gpt_sovits_semantic_token_count"]) for info in runtime_additional_information]
            if sum(sizes) == total:
                parts: list[torch.Tensor] = []
                offset = 0
                for size in sizes:
                    parts.append(ids[offset : offset + size])
                    offset += size
                return parts
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            return [ids[boundaries[i] : min(boundaries[i + 1], total)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(item, "token_slice") for item in slices):
                boundaries = [0]
                for item in slices:
                    boundaries.append(boundaries[-1] + item)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds
        empty = torch.zeros((0,), dtype=torch.float32)
        default_sr = torch.tensor(32000, dtype=torch.int32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": [empty], "sr": [default_sr]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_infos = runtime_additional_information or []
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"), request_infos)
        if runtime_additional_information is None:
            request_infos = [{} for _ in range(len(request_ids_list))]
        elif len(request_infos) < len(request_ids_list):
            request_infos = request_infos + ([{}] * (len(request_ids_list) - len(request_infos)))

        audio_outputs: list[torch.Tensor] = []
        sample_rates: list[torch.Tensor] = []
        for index, semantic_ids in enumerate(request_ids_list):
            info = request_infos[index] if index < len(request_infos) else {}
            if not self._has_decode_conditioning(info):
                audio_outputs.append(empty)
                sample_rates.append(default_sr)
                continue
            result = self.runtime.decode_semantic_tokens_from_transport(semantic_ids, info)
            audio_outputs.append(torch.from_numpy(result.audio).to(dtype=torch.float32))
            sample_rates.append(torch.tensor(int(result.sample_rate), dtype=torch.int32))

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "audio": audio_outputs,
                "sr": sample_rates,
            },
        )
