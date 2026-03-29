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
        "gpt_sovits_semantic_tokens": torch.arange(semantic_count, dtype=torch.long) + 100,
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
    model.runtime.prepare_decode_requests.return_value = [
        SimpleNamespace(request_id="a"),
        SimpleNamespace(request_id="b"),
    ]
    model.runtime.decode_prepared_requests.return_value = [
        SimpleNamespace(request_id="a"),
        SimpleNamespace(request_id="b"),
    ]
    model.runtime.finalize_decoded_audios.return_value = [
        SimpleNamespace(audio=np.array([0.1, 0.2], dtype=np.float32), sample_rate=24000),
        SimpleNamespace(audio=np.array([0.3], dtype=np.float32), sample_rate=16000),
    ]

    result = model.forward(
        input_ids=torch.tensor([0, 0], dtype=torch.long),
        runtime_additional_information=[
            {**_conditioning_info(semantic_count=2), "gpt_sovits_semantic_tokens": torch.tensor([10, 11], dtype=torch.long)},
            {
                **_conditioning_info(semantic_count=3),
                "gpt_sovits_semantic_tokens": torch.tensor([20, 21, 22], dtype=torch.long),
            },
        ],
        seq_token_counts=[1, 1],
    )

    prepare_call = model.runtime.prepare_decode_requests.call_args
    assert [item.tolist() for item in prepare_call.args[0]] == [[10, 11], [20, 21, 22]]
    assert model.runtime.decode_prepared_requests.call_count == 1
    assert model.runtime.finalize_decoded_audios.call_count == 1

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

    model.runtime.prepare_decode_request.assert_not_called()
    model.runtime.prepare_decode_requests.assert_not_called()
    model.runtime.decode_prepared_requests.assert_not_called()
    model.runtime.finalize_decoded_audios.assert_not_called()
    audios = result.multimodal_outputs["audio"]
    sample_rates = result.multimodal_outputs["sr"]
    assert len(audios) == 2
    assert all(audio.numel() == 0 for audio in audios)
    assert [int(sr.item()) for sr in sample_rates] == [32000, 32000]


def test_preprocess_builds_and_reuses_prepared_decode_request():
    model = _minimal_model()
    prepared = SimpleNamespace(request_id="prepared")
    model.runtime.prepare_decode_request.return_value = prepared
    info = _conditioning_info(semantic_count=2)

    input_ids = torch.tensor([0], dtype=torch.long)
    embeds = torch.zeros((1, 1), dtype=torch.float32)

    _, _, update_dict = model.preprocess(input_ids=input_ids, input_embeds=embeds, **info)
    assert update_dict[model._PREPARED_KEY] is prepared
    call = model.runtime.prepare_decode_request.call_args
    assert call.args[0].tolist() == [100, 101]

    _, _, second_update = model.preprocess(
        input_ids=input_ids,
        input_embeds=embeds,
        **{**info, model._PREPARED_KEY: prepared},
    )
    assert second_update[model._PREPARED_KEY] is prepared
    assert model.runtime.prepare_decode_request.call_count == 1
