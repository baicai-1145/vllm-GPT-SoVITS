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
    """Stage-0 GPT-SoVITS v2 semantic generator backed by the AR scheduler."""

    input_modalities = "audio"
    _SESSION_KEY = "gpt_sovits_ar_session"
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
        self._pending_logits: torch.Tensor | None = None
        self._semantic_eos_id: int | None = None
        self._semantic_vocab_size: int | None = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        return set()

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids is None:
            return torch.empty((0, 1), device=self._device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def _get_semantic_eos_id(self) -> int:
        if self._semantic_eos_id is None:
            self._semantic_eos_id = int(self.runtime.get_semantic_eos_id())
        return self._semantic_eos_id

    def _get_semantic_vocab_size(self) -> int:
        if self._semantic_vocab_size is None:
            self._semantic_vocab_size = int(self.runtime.get_semantic_vocab_size())
        return self._semantic_vocab_size

    def _merge_request_info(self, info_dict: dict[str, Any]) -> dict[str, Any]:
        additional_information = info_dict.get("additional_information")
        if not isinstance(additional_information, dict):
            return dict(info_dict)
        merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
        for key, value in additional_information.items():
            merged.setdefault(key, value)
        return merged

    def _request_id_from_info(self, info_dict: dict[str, Any], index: int = 0) -> str:
        request_id = (
            info_dict.get("engine_request_id")
            or info_dict.get("request_id")
            or info_dict.get("req_id")
            or info_dict.get("gpt_sovits_request_id")
        )
        if request_id is None:
            return f"gpt_sovits_v2_{index}"
        return str(request_id)

    def _skip_logits(self, device: torch.device) -> torch.Tensor:
        vocab_size = self._get_semantic_vocab_size()
        logits = torch.full((1, vocab_size), float("-inf"), device=device, dtype=torch.float32)
        logits[0, self._get_semantic_eos_id()] = 0.0
        return logits

    def _transport_payload_from_info(self, info_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        empty_long = _empty_long_tensor()
        empty_float = _empty_float_tensor()
        return {
            "semantic_tokens": empty_long,
            "gpt_sovits_phones": info_dict.get("gpt_sovits_phones", empty_long),
            "gpt_sovits_prompt_phones": info_dict.get("gpt_sovits_prompt_phones", empty_long),
            "gpt_sovits_prompt_semantic": info_dict.get("gpt_sovits_prompt_semantic", empty_long),
            "gpt_sovits_refer_audio_spec": info_dict.get("gpt_sovits_refer_audio_spec", empty_float),
            "gpt_sovits_refer_audio_16k": info_dict.get("gpt_sovits_refer_audio_16k", empty_float),
            "gpt_sovits_raw_audio": info_dict.get("gpt_sovits_raw_audio", empty_float),
            "gpt_sovits_raw_sr": torch.tensor(int(info_dict.get("gpt_sovits_raw_sr", 0)), dtype=torch.int32),
        }

    def _transport_payload_from_session(self, info_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        session = info_dict.get(self._SESSION_KEY)
        if session is None:
            return self._transport_payload_from_info(info_dict)
        transport = session.transport_info
        return {
            "semantic_tokens": self.runtime.get_ar_session_semantic_tokens(session),
            "gpt_sovits_phones": transport["gpt_sovits_phones"].to(dtype=torch.long),
            "gpt_sovits_prompt_phones": transport["gpt_sovits_prompt_phones"].to(dtype=torch.long),
            "gpt_sovits_prompt_semantic": transport["gpt_sovits_prompt_semantic"].to(dtype=torch.long),
            "gpt_sovits_refer_audio_spec": transport["gpt_sovits_refer_audio_spec"].to(dtype=torch.float32),
            "gpt_sovits_refer_audio_16k": transport["gpt_sovits_refer_audio_16k"].to(dtype=torch.float32),
            "gpt_sovits_raw_audio": transport["gpt_sovits_raw_audio"].to(dtype=torch.float32),
            "gpt_sovits_raw_sr": torch.tensor(int(transport["gpt_sovits_raw_sr"]), dtype=torch.int32),
        }

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        del sampling_metadata
        if self._pending_logits is None:
            device = hidden_states.device if isinstance(hidden_states, torch.Tensor) else self._device
            return self._skip_logits(device)
        return self._pending_logits

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, Any]]:
        empty_long = _empty_long_tensor()
        empty_float = _empty_float_tensor()
        return [
            {
                "skip_synthesis": True,
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

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        merged_info = self._merge_request_info(info_dict)
        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            embeds = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, embeds, {}
        if span_len != 1:
            raise ValueError(f"GPT-SoVITS v2 AR stage expects single-token steps, got span_len={span_len}")

        input_ids = input_ids.reshape(-1).to(dtype=torch.long)
        embeds = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
        if merged_info.get("skip_synthesis"):
            return input_ids, embeds, {}

        session = merged_info.get(self._SESSION_KEY)
        request_id = self._request_id_from_info(merged_info)
        if session is None:
            session = self.runtime.start_ar_session(merged_info, request_id=request_id)
            return input_ids, embeds, {self._SESSION_KEY: session}

        generated_len = int(merged_info.get("generated_len", 0) or 0)
        if generated_len > 0:
            sampled_token_id = int(input_ids[-1].item())
            self.runtime.advance_ar_session(session, sampled_token_id)
        return input_ids, embeds, {self._SESSION_KEY: session}

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
        del positions, intermediate_tensors, inputs_embeds, kwargs

        request_infos = model_intermediate_buffer or runtime_additional_information or [{"skip_synthesis": True}]
        num_reqs = len(request_infos)

        logits_rows: list[torch.Tensor] = []
        semantic_tokens: list[torch.Tensor] = []
        phones: list[torch.Tensor] = []
        prompt_phones: list[torch.Tensor] = []
        prompt_semantic: list[torch.Tensor] = []
        refer_audio_spec: list[torch.Tensor] = []
        refer_audio_16k: list[torch.Tensor] = []
        raw_audio: list[torch.Tensor] = []
        raw_sr: list[torch.Tensor] = []

        for index, raw_info in enumerate(request_infos):
            info = self._merge_request_info(raw_info)
            session = info.get(self._SESSION_KEY)
            if info.get("skip_synthesis") or session is None:
                logits_rows.append(self._skip_logits(self._device))
                payload = self._transport_payload_from_info(info)
            else:
                logits_rows.append(self.runtime.get_ar_session_logits(session))
                payload = self._transport_payload_from_session(info)

            semantic_tokens.append(payload["semantic_tokens"].to(dtype=torch.long))
            phones.append(payload["gpt_sovits_phones"].to(dtype=torch.long))
            prompt_phones.append(payload["gpt_sovits_prompt_phones"].to(dtype=torch.long))
            prompt_semantic.append(payload["gpt_sovits_prompt_semantic"].to(dtype=torch.long))
            refer_audio_spec.append(payload["gpt_sovits_refer_audio_spec"].to(dtype=torch.float32))
            refer_audio_16k.append(payload["gpt_sovits_refer_audio_16k"].to(dtype=torch.float32))
            raw_audio.append(payload["gpt_sovits_raw_audio"].to(dtype=torch.float32))
            raw_sr.append(payload["gpt_sovits_raw_sr"])

            if session is None and not info.get("skip_synthesis"):
                logger.warning("GPT-SoVITS v2 AR request %s has no active session", self._request_id_from_info(info, index))

        self._pending_logits = torch.cat(logits_rows, dim=0) if logits_rows else self._skip_logits(self._device)
        num_tokens = int(input_ids.numel()) if input_ids is not None and input_ids.numel() > 0 else max(1, num_reqs)
        dummy_hidden = torch.zeros((num_tokens, 1), device=self._device, dtype=torch.float32)

        return OmniOutput(
            text_hidden_states=dummy_hidden,
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
