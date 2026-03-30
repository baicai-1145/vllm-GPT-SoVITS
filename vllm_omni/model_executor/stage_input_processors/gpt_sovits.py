"""Stage input processor for GPT-SoVITS v2: semantic T2S -> waveform decode."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.gpt_sovits.runtime import GPTSoVITSStageTransport
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import _validate_stage_inputs

logger = init_logger(__name__)


def _prompt_additional_info(prompt: Any, index: int) -> dict[str, Any]:
    if prompt is None:
        return {}
    selected = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    if not isinstance(selected, dict):
        return {}
    info = selected.get("additional_information")
    return info if isinstance(info, dict) else {}


def _as_scalar(value: Any, default: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def t2s2decode(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    del requires_multimodal_data
    from vllm_omni.inputs.data import OmniTokensPrompt

    t2s_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    decode_inputs: list[OmniTokensPrompt] = []
    for index, t2s_output in enumerate(t2s_outputs):
        if not t2s_output.finished:
            continue
        output = t2s_output.outputs[0]
        semantic_tokens = output.multimodal_output["semantic_tokens"].to(torch.long).reshape(-1).cpu().contiguous()
        request_id = str(getattr(t2s_output, "request_id", getattr(output, "request_id", "")) or "")
        logger.info("GPT-SoVITS stage0 finished request %s with %d semantic tokens", request_id, int(semantic_tokens.numel()))
        if semantic_tokens.numel() == 0:
            continue
        prompt_info = _prompt_additional_info(prompt, index)
        raw_sr = output.multimodal_output["gpt_sovits_raw_sr"]
        if isinstance(raw_sr, torch.Tensor):
            raw_sr = int(raw_sr.reshape(-1)[0].item()) if raw_sr.numel() > 0 else 0
        transport = GPTSoVITSStageTransport(
            request_id=request_id,
            semantic_tokens=semantic_tokens,
            phones=output.multimodal_output["gpt_sovits_phones"].to(torch.long).cpu().contiguous(),
            prompt_phones=output.multimodal_output["gpt_sovits_prompt_phones"].to(torch.long).cpu().contiguous(),
            prompt_semantic=output.multimodal_output["gpt_sovits_prompt_semantic"].to(torch.long).cpu().contiguous(),
            refer_audio_spec=output.multimodal_output["gpt_sovits_refer_audio_spec"].to(torch.float32).cpu().contiguous(),
            refer_audio_16k=output.multimodal_output["gpt_sovits_refer_audio_16k"].to(torch.float32).cpu().contiguous(),
            raw_audio=output.multimodal_output["gpt_sovits_raw_audio"].to(torch.float32).cpu().contiguous(),
            raw_sr=int(raw_sr),
            speed_factor=float(_as_scalar(prompt_info.get("speed_factor", prompt_info.get("speed")), 1.0)),
            sample_steps=int(_as_scalar(prompt_info.get("sample_steps"), 32)),
            super_sampling=bool(_as_scalar(prompt_info.get("super_sampling"), False)),
        )
        decode_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                multi_modal_data=None,
                mm_processor_kwargs=None,
                model_intermediate_buffer=transport.to_model_intermediate_buffer(),
            )
        )
    return decode_inputs
