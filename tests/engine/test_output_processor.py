# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.engine.output_processor import OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request_state() -> OmniRequestState:
    state = object.__new__(OmniRequestState)
    state.mm_type = None
    state.mm_accumulated = None
    state.mm_keep_last_keys = set()
    return state


def test_multimodal_keep_last_contract_is_model_owned():
    state = _request_state()

    state.add_multimodal_tensor(
        {
            "__omni_keep_last_mm_keys__": ["semantic_tokens"],
            "semantic_tokens": torch.tensor([1], dtype=torch.long),
            "foo": torch.tensor([10], dtype=torch.long),
        },
        mm_type="latent",
    )
    state.add_multimodal_tensor(
        {
            "__omni_keep_last_mm_keys__": ["semantic_tokens"],
            "semantic_tokens": torch.tensor([1, 2], dtype=torch.long),
            "foo": torch.tensor([20], dtype=torch.long),
        },
        mm_type="latent",
    )

    state._consolidate_multimodal_tensors()

    assert state.mm_keep_last_keys == {"semantic_tokens"}
    assert torch.equal(state.mm_accumulated["semantic_tokens"], torch.tensor([1, 2], dtype=torch.long))
    assert torch.equal(state.mm_accumulated["foo"], torch.tensor([10, 20], dtype=torch.long))
