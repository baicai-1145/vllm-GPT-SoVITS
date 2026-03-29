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
    _PREPARED_KEY = "gpt_sovits_decode_prepared"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.have_multimodal_outputs = True
        self.has_preprocess = True
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

    def _merge_request_info(self, info_dict: dict[str, Any]) -> dict[str, Any]:
        additional_information = info_dict.get("additional_information")
        if not isinstance(additional_information, dict):
            return dict(info_dict)
        merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
        for key, value in additional_information.items():
            merged.setdefault(key, value)
        return merged

    @staticmethod
    def _semantic_tokens_from_info(info: dict[str, Any]) -> torch.Tensor:
        value = info.get("gpt_sovits_semantic_tokens")
        if isinstance(value, torch.Tensor):
            return value.detach().reshape(-1).to(dtype=torch.long).cpu().contiguous()
        if value in (None, [], ()):
            return torch.zeros((0,), dtype=torch.long)
        return torch.as_tensor(value, dtype=torch.long).reshape(-1).cpu().contiguous()

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, Any]]:
        empty_long = torch.zeros((0,), dtype=torch.long)
        empty_float = torch.zeros((0,), dtype=torch.float32)
        return [
            {
                "gpt_sovits_request_id": "",
                "gpt_sovits_semantic_tokens": empty_long,
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
        if info.get(GPTSoVITSV2Decode._PREPARED_KEY) is not None:
            return True
        required_keys = (
            "gpt_sovits_semantic_tokens",
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

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        merged_info = self._merge_request_info(info_dict)
        embeds = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
        if not self._has_decode_conditioning(merged_info):
            return input_ids, embeds, {}

        prepared = merged_info.get(self._PREPARED_KEY)
        if prepared is None:
            semantic_tokens = self._semantic_tokens_from_info(merged_info)
            prepared = self.runtime.prepare_decode_request(semantic_tokens, merged_info)
        return input_ids, embeds, {self._PREPARED_KEY: prepared}

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
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds
        empty = torch.zeros((0,), dtype=torch.float32)
        default_sr = torch.tensor(32000, dtype=torch.int32)

        request_infos = model_intermediate_buffer or runtime_additional_information or []
        if not request_infos and (input_ids is None or input_ids.numel() == 0):
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": [empty], "sr": [default_sr]},
            )

        if request_infos:
            request_infos = [self._merge_request_info(info) for info in request_infos]

        audio_outputs: list[torch.Tensor] = []
        sample_rates: list[torch.Tensor] = []
        if not request_infos:
            ids = input_ids.reshape(-1).to(dtype=torch.long) if input_ids is not None else torch.zeros((0,), dtype=torch.long)
            request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"), runtime_additional_information)
            request_infos = [{} for _ in range(len(request_ids_list))]
        else:
            request_ids_list = [self._semantic_tokens_from_info(info) for info in request_infos]

        prepared_by_index: list[Any | None] = [None] * len(request_ids_list)
        batch_prepare_indices: list[int] = []
        batch_prepare_semantic_ids: list[torch.Tensor] = []
        batch_prepare_infos: list[dict[str, Any]] = []

        for index, semantic_ids in enumerate(request_ids_list):
            info = request_infos[index] if index < len(request_infos) else {}
            if not self._has_decode_conditioning(info):
                continue
            prepared = info.get(self._PREPARED_KEY)
            if prepared is not None:
                prepared_by_index[index] = prepared
                continue
            batch_prepare_indices.append(index)
            batch_prepare_semantic_ids.append(semantic_ids)
            batch_prepare_infos.append(info)

        if batch_prepare_indices:
            prepared_items = self.runtime.prepare_decode_requests(batch_prepare_semantic_ids, batch_prepare_infos)
            for index, prepared in zip(batch_prepare_indices, prepared_items):
                prepared_by_index[index] = prepared

        runnable_indices: list[int] = []
        runnable_prepared: list[Any] = []
        for index, prepared in enumerate(prepared_by_index):
            if prepared is None:
                audio_outputs.append(empty)
                sample_rates.append(default_sr)
                continue
            runnable_indices.append(index)
            runnable_prepared.append(prepared)
            audio_outputs.append(empty)
            sample_rates.append(default_sr)

        if runnable_prepared:
            decoded_items = self.runtime.decode_prepared_requests(runnable_prepared)
            finalized_items = self.runtime.finalize_decoded_audios(decoded_items)
            for index, result in zip(runnable_indices, finalized_items):
                audio_outputs[index] = torch.from_numpy(result.audio).to(dtype=torch.float32)
                sample_rates[index] = torch.tensor(int(result.sample_rate), dtype=torch.int32)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "audio": audio_outputs,
                "sr": sample_rates,
            },
        )
