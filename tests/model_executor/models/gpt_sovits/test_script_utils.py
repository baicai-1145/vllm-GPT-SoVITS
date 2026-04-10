from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf

from vllm_omni.model_executor.models.gpt_sovits.script_utils import (
    build_default_request,
    configure_runtime_cache_root,
    read_input_text,
    render_output_stem,
    resolve_synthesis_inputs,
    summarize_audio_file,
)


def test_read_input_text_rejects_both_sources(tmp_path):
    text_file = tmp_path / "input.txt"
    text_file.write_text("hello", encoding="utf-8")

    try:
        read_input_text(text="inline", text_file=text_file, required_arg_hint="missing input")
    except ValueError as exc:
        assert "either --text or --text-file" in str(exc)
    else:
        raise AssertionError("expected ValueError when both text and text_file are provided")


def test_build_default_request_allows_overrides(tmp_path):
    ref_audio = tmp_path / "ref.wav"
    request = build_default_request(
        "hello",
        ref_audio,
        ref_text="prompt",
        text_lang="en",
        prompt_lang="zh",
        text_split_method="cut1",
        speed_factor=1.25,
        sample_steps=48,
        super_sampling=True,
        parallel_infer=False,
        fragment_interval=0.0,
        seed=7,
        batch_size=2,
        batch_threshold=0.5,
        split_bucket=False,
        repetition_penalty=1.1,
    )

    assert request["ref_audio_path"] == str(ref_audio)
    assert request["prompt_text"] == "prompt"
    assert request["text_lang"] == "en"
    assert request["prompt_lang"] == "zh"
    assert request["text_split_method"] == "cut1"
    assert request["speed_factor"] == 1.25
    assert request["sample_steps"] == 48
    assert request["super_sampling"] is True
    assert request["parallel_infer"] is False
    assert request["fragment_interval"] == 0.0
    assert request["seed"] == 7
    assert request["batch_size"] == 2
    assert request["batch_threshold"] == 0.5
    assert request["split_bucket"] is False
    assert request["repetition_penalty"] == 1.1


def test_configure_runtime_cache_root_sets_env_and_dirs(tmp_path):
    resolved = configure_runtime_cache_root(tmp_path / "cache_root")

    assert resolved == str((tmp_path / "cache_root").resolve())
    assert Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).is_dir()
    assert Path(os.environ["TRITON_CACHE_DIR"]).is_dir()
    assert os.environ["GPTSOVITS_RUNTIME_CACHE_ROOT"] == resolved


def test_summarize_audio_file_reads_wav_metadata(tmp_path):
    wav_path = tmp_path / "audio.wav"
    sf.write(wav_path, np.zeros(3200, dtype=np.float32), 16000)

    summary = summarize_audio_file(wav_path, sample_rate=16000, mode="test")

    assert summary["path"] == str(wav_path)
    assert summary["sample_rate"] == 16000
    assert summary["file_sample_rate"] == 16000
    assert summary["frames"] == 3200
    assert summary["duration_sec"] == 0.2
    assert summary["format"] == "WAV"
    assert summary["mode"] == "test"


def test_resolve_synthesis_inputs_reads_sorted_batch_files(tmp_path):
    alpha = tmp_path / "b_file.txt"
    beta = tmp_path / "a_file.txt"
    alpha.write_text("bravo", encoding="utf-8")
    beta.write_text("alpha", encoding="utf-8")

    items = resolve_synthesis_inputs(
        text=None,
        text_file=None,
        input_dir=tmp_path,
        glob_pattern="*.txt",
        required_arg_hint="missing input",
    )

    assert [item.item_id for item in items] == ["a_file", "b_file"]
    assert [item.text for item in items] == ["alpha", "bravo"]
    assert all(item.source_kind == "file" for item in items)


def test_render_output_stem_uses_template_and_sanitizes():
    rendered = render_output_stem("{index:03d}_{stem}_voice", index=7, item_id="hello world")

    assert rendered == "007_hello_world_voice"
