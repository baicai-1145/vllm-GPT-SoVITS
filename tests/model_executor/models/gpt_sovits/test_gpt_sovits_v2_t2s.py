# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.gpt_sovits.gpt_sovits_v2_t2s import GPTSoVITSV2T2S
from vllm_omni.model_executor.models.gpt_sovits.runtime import GPTSoVITSStageTransport

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _minimal_model() -> GPTSoVITSV2T2S:
    model = object.__new__(GPTSoVITSV2T2S)
    object.__setattr__(model, "runtime", Mock())
    object.__setattr__(model, "_device", torch.device("cpu"))
    object.__setattr__(model, "_pending_logits", None)
    object.__setattr__(model, "_semantic_eos_id", 3)
    object.__setattr__(model, "_semantic_vocab_size", 5)
    return model


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        transport_info=GPTSoVITSStageTransport(
            request_id="req-1",
            semantic_tokens=torch.zeros((0,), dtype=torch.long),
            phones=torch.tensor([1, 2], dtype=torch.int16),
            prompt_phones=torch.tensor([3], dtype=torch.int32),
            prompt_semantic=torch.tensor([4, 5], dtype=torch.int64),
            refer_audio_spec=torch.randn(3, dtype=torch.float16),
            refer_audio_16k=torch.randn(4, dtype=torch.float16),
            raw_audio=torch.randn(8, dtype=torch.float64),
            raw_sr=32000,
            speed_factor=1.0,
            sample_steps=32,
            super_sampling=False,
        ),
    )


def test_preprocess_starts_ar_session_on_first_step():
    model = _minimal_model()
    session = _session()
    model.runtime.build_request_spec.return_value = "spec"
    model.runtime.start_ar_session_from_spec.return_value = session

    input_ids, embeds, update = model.preprocess(
        input_ids=torch.tensor([1], dtype=torch.long),
        input_embeds=None,
        additional_information={"text": "hello", "ref_audio_path": "/tmp/ref.wav", "prompt_text": "ref"},
        engine_request_id="req-1",
    )

    model.runtime.build_request_spec.assert_called_once_with(
        {"text": "hello", "ref_audio_path": "/tmp/ref.wav", "prompt_text": "ref", "engine_request_id": "req-1"},
        request_id="req-1",
    )
    model.runtime.start_ar_session_from_spec.assert_called_once_with("spec")
    model.runtime.start_ar_session.assert_not_called()
    model.runtime.advance_ar_session.assert_not_called()
    assert input_ids.tolist() == [1]
    assert embeds.shape == (1, 1)
    assert update["gpt_sovits_ar_session"] is session


def test_preprocess_advances_existing_session_with_sampled_token():
    model = _minimal_model()
    session = _session()

    input_ids, embeds, update = model.preprocess(
        input_ids=torch.tensor([77], dtype=torch.long),
        input_embeds=None,
        gpt_sovits_ar_session=session,
        generated_len=1,
    )

    model.runtime.start_ar_session.assert_not_called()
    model.runtime.advance_ar_session.assert_called_once_with(session, 77)
    assert input_ids.tolist() == [77]
    assert embeds.shape == (1, 1)
    assert update["gpt_sovits_ar_session"] is session


def test_forward_and_compute_logits_follow_session_batch_order():
    model = _minimal_model()
    session = _session()
    model.runtime.get_ar_session_logits.return_value = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
    model.runtime.get_ar_session_semantic_tokens.return_value = torch.tensor([10, 11], dtype=torch.long)

    result = model.forward(
        input_ids=torch.tensor([1, 2], dtype=torch.long),
        model_intermediate_buffer=[
            {"gpt_sovits_ar_session": session},
            {"skip_synthesis": True},
        ],
    )

    logits = model.compute_logits(torch.zeros((2, 1), dtype=torch.float32))
    assert logits.shape == (2, 5)
    assert torch.allclose(logits[0], torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32))
    assert logits[1, 3].item() == pytest.approx(0.0)
    assert torch.isneginf(logits[1, 0])

    semantic_tokens = result.multimodal_outputs["semantic_tokens"]
    assert len(semantic_tokens) == 2
    assert semantic_tokens[0].tolist() == [10, 11]
    assert semantic_tokens[1].numel() == 0

    assert result.multimodal_outputs["gpt_sovits_phones"][0].dtype == torch.long
    assert result.multimodal_outputs["gpt_sovits_refer_audio_spec"][0].dtype == torch.float32
    assert int(result.multimodal_outputs["gpt_sovits_raw_sr"][0].item()) == 32000
