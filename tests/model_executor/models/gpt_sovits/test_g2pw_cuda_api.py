from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_VENDORED_RUNTIME_ROOT = _REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "gpt_sovits" / "runtime_lib"
for candidate in (str(_VENDORED_RUNTIME_ROOT), str(_VENDORED_RUNTIME_ROOT / "GPT_SoVITS")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
G2PWRuntimeWrapper = importlib.import_module(
    "vllm_omni.model_executor.models.gpt_sovits.runtime_lib.GPT_SoVITS.text.g2pw.cuda_api"
).G2PWRuntimeWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _build_model_input() -> dict[str, np.ndarray]:
    return {
        "input_ids": np.asarray(
            [
                [11, 12, 13, 0],
                [21, 22, 23, 0],
                [31, 32, 33, 0],
            ],
            dtype=np.int64,
        ),
        "token_type_ids": np.zeros((3, 4), dtype=np.int64),
        "attention_masks": np.ones((3, 4), dtype=np.int64),
        "phoneme_masks": np.ones((4, 2), dtype=np.float32),
        "char_ids": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "position_ids": np.asarray([5, 6, 7, 8], dtype=np.int64),
        "sequence_ids": np.asarray([0, 0, 1, 2], dtype=np.int64),
    }


def _build_wrapper() -> G2PWRuntimeWrapper:
    wrapper = object.__new__(G2PWRuntimeWrapper)
    wrapper.direct_max_rows = 2
    wrapper.direct_max_sequences = 2
    wrapper.direct_max_tokens = 8
    wrapper.batch_enabled = True
    wrapper.shard_index = 0
    return wrapper


def test_split_model_input_for_direct_run_remaps_sequence_ids():
    wrapper = _build_wrapper()

    chunks = wrapper._split_model_input_for_direct_run(_build_model_input())

    assert len(chunks) == 2
    assert chunks[0]["char_ids"].tolist() == [0, 1]
    assert chunks[0]["sequence_ids"].tolist() == [0, 0]
    assert chunks[0]["input_ids"].shape == (1, 4)

    assert chunks[1]["char_ids"].tolist() == [2, 3]
    assert chunks[1]["sequence_ids"].tolist() == [0, 1]
    assert chunks[1]["input_ids"].shape == (2, 4)
    assert np.array_equal(chunks[1]["input_ids"][0], np.asarray([21, 22, 23, 0], dtype=np.int64))
    assert np.array_equal(chunks[1]["input_ids"][1], np.asarray([31, 32, 33, 0], dtype=np.int64))


def test_run_with_profile_bypasses_batch_worker_for_oversized_input():
    wrapper = _build_wrapper()
    calls: list[tuple[list[int], list[int], int]] = []

    def fake_run_direct(model_input: dict[str, np.ndarray]) -> np.ndarray:
        calls.append(
            (
                model_input["char_ids"].tolist(),
                model_input["sequence_ids"].tolist(),
                int(model_input["input_ids"].shape[0]),
            )
        )
        fill_value = float(len(calls))
        return np.full((model_input["char_ids"].shape[0], 3), fill_value=fill_value, dtype=np.float32)

    wrapper._run_direct = fake_run_direct  # type: ignore[method-assign]
    wrapper._submit_batched = lambda _model_input: (_ for _ in ()).throw(AssertionError("unexpected batching"))  # type: ignore[method-assign]

    output, profile = wrapper.run_with_profile(_build_model_input())

    assert calls == [([0, 1], [0, 0], 1), ([2, 3], [0, 1], 2)]
    assert output.shape == (4, 3)
    assert np.allclose(output[:2], 1.0)
    assert np.allclose(output[2:], 2.0)
    assert profile["g2pw_runtime_chunk_count"] == pytest.approx(2.0)
    assert profile["g2pw_runtime_batch_requests"] == pytest.approx(2.0)
    assert profile["g2pw_runtime_batch_rows"] == pytest.approx(4.0)
