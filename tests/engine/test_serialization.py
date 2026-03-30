# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
import numpy as np
import pytest
import torch

from vllm_omni.engine import AdditionalInformationPayload
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _round_trip(raw_info: dict[str, object]) -> dict[str, object]:
    payload = serialize_additional_information(raw_info)
    assert payload is not None
    encoded = msgspec.msgpack.encode(payload)
    decoded = msgspec.msgpack.decode(encoded, type=AdditionalInformationPayload)
    return deserialize_additional_information(decoded)


def test_additional_information_round_trips_nested_tensors_and_arrays():
    info = _round_trip(
        {
            "gpt_sovits_transport": {
                "semantic_tokens": torch.tensor([101, 102, 103], dtype=torch.long),
                "phones": torch.tensor([11, 12], dtype=torch.int32),
                "refer_audio_spec": np.arange(6, dtype=np.float32).reshape(2, 3),
                "raw_sr": 32000,
                "prompt_meta": {
                    "speed_factor": 1.25,
                },
            }
        }
    )

    transport = info["gpt_sovits_transport"]
    assert isinstance(transport, dict)
    assert torch.equal(transport["semantic_tokens"], torch.tensor([101, 102, 103], dtype=torch.long))
    assert torch.equal(transport["phones"], torch.tensor([11, 12], dtype=torch.int32))
    assert np.array_equal(
        transport["refer_audio_spec"],
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )
    assert transport["raw_sr"] == 32000
    assert transport["prompt_meta"] == {"speed_factor": 1.25}


def test_additional_information_round_trips_nested_list_payloads():
    info = _round_trip(
        {
            "voice_clone_prompt": [
                {
                    "ref_spk_embedding": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                    "aux": (1, 2, 3),
                }
            ]
        }
    )

    prompt_list = info["voice_clone_prompt"]
    assert isinstance(prompt_list, list)
    assert len(prompt_list) == 1
    item = prompt_list[0]
    assert torch.allclose(
        item["ref_spk_embedding"],
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
    )
    assert item["aux"] == (1, 2, 3)
