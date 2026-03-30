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


class GPTSoVITSGeneration(nn.Module):
    """Generation-only wrapper that calls the existing GPT-SoVITS pipeline."""

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

    @staticmethod
    def _request_infos_from_buffers(
        model_intermediate_buffer: list[dict[str, Any]] | None,
        runtime_additional_information: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if model_intermediate_buffer is not None:
            return model_intermediate_buffer
        if runtime_additional_information is not None:
            logger.warning_once("GPT-SoVITS generation received legacy runtime_additional_information")
            return runtime_additional_information
        return [{"skip_synthesis": True}]

    def get_dummy_model_intermediate_buffer(self, num_reqs: int) -> list[dict[str, Any]]:
        return [{"skip_synthesis": True} for _ in range(max(1, num_reqs))]

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, Any]]:
        return self.get_dummy_model_intermediate_buffer(num_reqs)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del input_ids, positions, intermediate_tensors, inputs_embeds, kwargs

        request_infos = self._request_infos_from_buffers(
            model_intermediate_buffer,
            runtime_additional_information,
        )
        audio_outputs: list[torch.Tensor] = []
        sample_rates: list[torch.Tensor] = []

        for info in request_infos:
            if info.get("skip_synthesis"):
                audio_outputs.append(torch.zeros((0,), dtype=torch.float32))
                sample_rates.append(torch.tensor(32000, dtype=torch.int32))
                continue

            result = self.runtime.synthesize(info)
            audio_outputs.append(torch.from_numpy(result.audio).to(dtype=torch.float32))
            sample_rates.append(torch.tensor(result.sample_rate, dtype=torch.int32))

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "audio": audio_outputs,
                "sr": sample_rates,
            },
        )
