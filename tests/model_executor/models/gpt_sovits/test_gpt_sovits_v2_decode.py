# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.gpt_sovits.gpt_sovits_v2_decode import GPTSoVITSV2Decode

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _minimal_model() -> GPTSoVITSV2Decode:
    model = object.__new__(GPTSoVITSV2Decode)
    object.__setattr__(model, "runtime", Mock())
    object.__setattr__(model, "_device", torch.device("cpu"))
    return model


def _conditioning_info(*, semantic_count: int) -> dict[str, object]:
    return {
        "gpt_sovits_phones": torch.tensor([1, 2], dtype=torch.long),
        "gpt_sovits_prompt_phones": torch.tensor([3], dtype=torch.long),
        "gpt_sovits_prompt_semantic": torch.tensor([4], dtype=torch.long),
        "gpt_sovits_refer_audio_spec": torch.randn(2, dtype=torch.float32),
        "gpt_sovits_refer_audio_16k": torch.randn(2, dtype=torch.float32),
        "gpt_sovits_raw_audio": torch.randn(4, dtype=torch.float32),
        "gpt_sovits_raw_sr": 32000,
        "gpt_sovits_semantic_token_count": semantic_count,
    }


def test_split_request_ids_prefers_runtime_semantic_token_counts():
    model = _minimal_model()
    ids = torch.tensor([10, 11, 20, 21, 22], dtype=torch.long)

    parts = model._split_request_ids(
        ids,
        seq_token_counts=[4, 1],
        runtime_additional_information=[
            {"gpt_sovits_semantic_token_count": 2},
            {"gpt_sovits_semantic_token_count": 3},
        ],
    )

    assert [part.tolist() for part in parts] == [[10, 11], [20, 21, 22]]


def test_forward_decodes_each_request_using_runtime_split_contract():
    model = _minimal_model()
    model.runtime.decode_semantic_tokens_from_transport.side_effect = [
        SimpleNamespace(audio=np.array([0.1, 0.2], dtype=np.float32), sample_rate=24000),
        SimpleNamespace(audio=np.array([0.3], dtype=np.float32), sample_rate=16000),
    ]

    result = model.forward(
        input_ids=torch.tensor([10, 11, 20, 21, 22], dtype=torch.long),
        runtime_additional_information=[
            _conditioning_info(semantic_count=2),
            _conditioning_info(semantic_count=3),
        ],
        seq_token_counts=[4, 1],
    )

    calls = model.runtime.decode_semantic_tokens_from_transport.call_args_list
    assert len(calls) == 2
    assert calls[0].args[0].tolist() == [10, 11]
    assert calls[1].args[0].tolist() == [20, 21, 22]

    audios = result.multimodal_outputs["audio"]
    sample_rates = result.multimodal_outputs["sr"]
    assert len(audios) == 2
    assert torch.equal(audios[0], torch.tensor([0.1, 0.2], dtype=torch.float32))
    assert torch.equal(audios[1], torch.tensor([0.3], dtype=torch.float32))
    assert [int(sr.item()) for sr in sample_rates] == [24000, 16000]


def test_dummy_runtime_information_and_missing_conditioning_return_silence():
    model = _minimal_model()
    dummy_infos = model.get_dummy_runtime_additional_information(2)

    assert len(dummy_infos) == 2
    assert all(info["gpt_sovits_semantic_token_count"] == 0 for info in dummy_infos)

    result = model.forward(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        runtime_additional_information=dummy_infos,
        seq_token_counts=[1, 2],
    )

    model.runtime.decode_semantic_tokens_from_transport.assert_not_called()
    audios = result.multimodal_outputs["audio"]
    sample_rates = result.multimodal_outputs["sr"]
    assert len(audios) == 2
    assert all(audio.numel() == 0 for audio in audios)
    assert [int(sr.item()) for sr in sample_rates] == [32000, 32000]
