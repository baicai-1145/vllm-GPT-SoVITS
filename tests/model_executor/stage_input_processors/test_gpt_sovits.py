# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import msgspec
import pytest
import torch

from vllm_omni.engine import AdditionalInformationPayload
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)
from vllm_omni.model_executor.stage_input_processors.gpt_sovits import t2s2decode
from vllm_omni.model_executor.models.gpt_sovits.runtime import GPTSoVITSStageTransport

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_stage_output(
    *,
    semantic_tokens: torch.Tensor,
    finished: bool = True,
) -> SimpleNamespace:
    multimodal_output = {
        "semantic_tokens": semantic_tokens,
        "gpt_sovits_phones": torch.tensor([11, 12, 13], dtype=torch.int32),
        "gpt_sovits_prompt_phones": torch.tensor([21, 22], dtype=torch.int16),
        "gpt_sovits_prompt_semantic": torch.tensor([31, 32], dtype=torch.int64),
        "gpt_sovits_refer_audio_spec": torch.randn(4, dtype=torch.float16),
        "gpt_sovits_refer_audio_16k": torch.randn(8, dtype=torch.float16),
        "gpt_sovits_raw_audio": torch.randn(12, dtype=torch.float64),
        "gpt_sovits_raw_sr": torch.tensor([32000], dtype=torch.int32),
    }
    return SimpleNamespace(
        request_id="req-1",
        finished=finished,
        outputs=[SimpleNamespace(multimodal_output=multimodal_output)],
    )


def _make_stage(engine_outputs) -> SimpleNamespace:
    return SimpleNamespace(engine_outputs=engine_outputs)


def test_t2s2decode_transfers_conditioning_and_prompt_metadata():
    stage = _make_stage(
        [
            _make_stage_output(semantic_tokens=torch.tensor([[101, 102, 103]], dtype=torch.int32)),
            _make_stage_output(semantic_tokens=torch.tensor([], dtype=torch.int32), finished=True),
            _make_stage_output(semantic_tokens=torch.tensor([999], dtype=torch.int32), finished=False),
        ]
    )
    prompt = [
        {
            "additional_information": {
                "speed": [1.25],
                "sample_steps": [40],
                "super_sampling": [True],
            }
        }
    ]

    result = t2s2decode([stage], engine_input_source=[0], prompt=prompt)

    assert len(result) == 1
    engine_input = result[0]
    assert engine_input["prompt_token_ids"] == [0]
    assert engine_input["multi_modal_data"] is None
    assert "additional_information" not in engine_input

    info = engine_input["model_intermediate_buffer"]
    transport = GPTSoVITSStageTransport.from_info(info)
    assert info["gpt_sovits_request_id"] == "req-1"
    assert info["gpt_sovits_semantic_token_count"] == 3
    assert torch.equal(transport.semantic_tokens, torch.tensor([101, 102, 103], dtype=torch.long))
    assert transport.raw_sr == 32000
    assert transport.speed_factor == pytest.approx(1.25)
    assert transport.sample_steps == 40
    assert transport.super_sampling is True

    assert transport.phones.dtype == torch.long
    assert transport.prompt_phones.dtype == torch.long
    assert transport.prompt_semantic.dtype == torch.long
    assert transport.refer_audio_spec.dtype == torch.float32
    assert transport.refer_audio_16k.dtype == torch.float32
    assert transport.raw_audio.dtype == torch.float32

    assert transport.phones.device.type == "cpu"
    assert transport.refer_audio_spec.device.type == "cpu"


def test_t2s2decode_uses_default_runtime_metadata_when_prompt_missing():
    stage = _make_stage([_make_stage_output(semantic_tokens=torch.tensor([7, 8], dtype=torch.int32))])

    result = t2s2decode([stage], engine_input_source=[0], prompt=None)

    assert len(result) == 1
    info = result[0]["model_intermediate_buffer"]
    transport = GPTSoVITSStageTransport.from_info(info)
    assert transport.speed_factor == pytest.approx(1.0)
    assert transport.sample_steps == 32
    assert transport.super_sampling is False


def test_t2s2decode_transport_survives_additional_information_msgspec_roundtrip():
    stage = _make_stage([_make_stage_output(semantic_tokens=torch.tensor([7, 8, 9], dtype=torch.int32))])

    result = t2s2decode([stage], engine_input_source=[0], prompt=None)

    raw_info = result[0]["model_intermediate_buffer"]
    payload = serialize_additional_information(raw_info)
    assert payload is not None
    encoded = msgspec.msgpack.encode(payload)
    decoded = msgspec.msgpack.decode(encoded, type=AdditionalInformationPayload)
    transport = GPTSoVITSStageTransport.from_info(
        deserialize_additional_information(decoded)
    )
    assert torch.equal(transport.semantic_tokens, torch.tensor([7, 8, 9], dtype=torch.long))
    assert transport.has_decode_conditioning()
