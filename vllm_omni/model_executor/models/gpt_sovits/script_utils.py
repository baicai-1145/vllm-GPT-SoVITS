from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
TEMP_ROOT = REPO_ROOT / "TEMP"
DEFAULT_PROJECT_ROOT = REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "gpt_sovits" / "runtime_lib"
DEFAULT_BENCH_CONFIG_PATH = (
    REPO_ROOT
    / "tests"
    / "model_executor"
    / "models"
    / "gpt_sovits"
    / "fixtures"
    / "tts_infer_v2proplus_cuda_longprompt.yaml"
)
DEFAULT_EXAMPLE_CONFIG_PATH = DEFAULT_PROJECT_ROOT / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
DEFAULT_REF_AUDIO = REPO_ROOT / "test.wav"
DEFAULT_REF_TEXT = "又或者说，你已经察觉到了…却还想拿「它」干什么好事？"
DEFAULT_TEXT_FILES: dict[str, list[Path]] = {
    "cn": [REPO_ROOT / "test_cn.txt", Path("/root/GPT-SoVITS/test_cn.txt")],
    "en": [REPO_ROOT / "test_en.txt", Path("/root/GPT-SoVITS/test_en.txt")],
    "4lang": [REPO_ROOT / "test_4lang.txt", Path("/root/GPT-SoVITS/test_4lang.txt")],
}


@dataclass(slots=True)
class SynthesisTextInput:
    item_id: str
    text: str
    source_path: str | None
    source_kind: str


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def stage_timer() -> float:
    sync_cuda()
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    sync_cuda()
    return (time.perf_counter() - start) * 1000.0


def elapsed_s(start: float) -> float:
    sync_cuda()
    return time.perf_counter() - start


def resolve_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of the candidate paths exist: {joined}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def read_input_text(*, text: str | None, text_file: Path | None, required_arg_hint: str) -> str:
    inline_text = (text or "").strip()
    if inline_text and text_file is not None:
        raise ValueError("Use either --text or --text-file, not both")
    if text_file is not None:
        loaded = read_text(text_file)
        if not loaded:
            raise ValueError(f"Input text file is empty: {text_file}")
        return loaded
    if not inline_text:
        raise ValueError(required_arg_hint)
    return inline_text


def _sanitize_output_token(value: str, *, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def resolve_synthesis_inputs(
    *,
    text: str | None,
    text_file: Path | None,
    input_dir: Path | None,
    glob_pattern: str,
    required_arg_hint: str,
) -> list[SynthesisTextInput]:
    has_single = bool((text or "").strip()) or text_file is not None
    has_batch = input_dir is not None
    if has_single and has_batch:
        raise ValueError("Use either --text/--text-file or --input-dir, not both")
    if has_batch:
        resolved_dir = input_dir.resolve()
        if not resolved_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {resolved_dir}")
        if not resolved_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {resolved_dir}")
        matched_files = sorted(path for path in resolved_dir.glob(glob_pattern) if path.is_file())
        if not matched_files:
            raise FileNotFoundError(f"No files matched pattern '{glob_pattern}' under {resolved_dir}")
        items: list[SynthesisTextInput] = []
        for path in matched_files:
            loaded = read_text(path)
            if not loaded:
                raise ValueError(f"Input text file is empty: {path}")
            items.append(
                SynthesisTextInput(
                    item_id=_sanitize_output_token(path.stem, fallback=f"item_{len(items):03d}"),
                    text=loaded,
                    source_path=str(path.resolve()),
                    source_kind="file",
                )
            )
        return items

    single_text = read_input_text(text=text, text_file=text_file, required_arg_hint=required_arg_hint)
    if text_file is not None:
        return [
            SynthesisTextInput(
                item_id=_sanitize_output_token(text_file.stem, fallback="text"),
                text=single_text,
                source_path=str(text_file.resolve()),
                source_kind="file",
            )
        ]
    return [
        SynthesisTextInput(
            item_id="inline",
            text=single_text,
            source_path=None,
            source_kind="inline",
        )
    ]


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return value


def build_default_request(
    text: str,
    ref_audio_path: Path,
    *,
    ref_text: str = DEFAULT_REF_TEXT,
    text_lang: str = "auto",
    prompt_lang: str = "auto",
    text_split_method: str = "cut5",
    speed_factor: float = 1.0,
    sample_steps: int = 32,
    super_sampling: bool = False,
    parallel_infer: bool = True,
    fragment_interval: float = 0.3,
    seed: int = 1234,
    batch_size: int = 4,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    repetition_penalty: float = 1.35,
) -> dict[str, Any]:
    return {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": str(ref_audio_path),
        "prompt_text": ref_text,
        "prompt_lang": prompt_lang,
        "text_split_method": text_split_method,
        "speed_factor": float(speed_factor),
        "sample_steps": int(sample_steps),
        "super_sampling": bool(super_sampling),
        "parallel_infer": bool(parallel_infer),
        "fragment_interval": float(fragment_interval),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "split_bucket": bool(split_bucket),
        "repetition_penalty": float(repetition_penalty),
    }


def configure_runtime_cache_root(runtime_cache_root: Path | None) -> str | None:
    if runtime_cache_root is None:
        return None
    resolved_root = runtime_cache_root.resolve()
    torchinductor_cache_dir = resolved_root / "torchinductor"
    triton_cache_dir = resolved_root / "triton"
    torchinductor_cache_dir.mkdir(parents=True, exist_ok=True)
    triton_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GPTSOVITS_RUNTIME_CACHE_ROOT"] = str(resolved_root)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(torchinductor_cache_dir)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache_dir)
    return str(resolved_root)


def build_timestamped_output_path(output_dir: Path, *, prefix: str) -> Path:
    return output_dir / f"{prefix}_{int(time.time())}.wav"


def render_output_stem(template: str, *, index: int, item_id: str) -> str:
    safe_id = _sanitize_output_token(item_id, fallback=f"item_{index:03d}")
    rendered = template.format(index=index, id=safe_id, stem=safe_id)
    return _sanitize_output_token(rendered, fallback=f"item_{index:03d}")


def summarize_audio_file(
    output_path: Path,
    *,
    sample_rate: int,
    mode: str,
) -> dict[str, Any]:
    info = sf.info(output_path)
    file_sample_rate = int(info.samplerate)
    frames = int(info.frames)
    duration_sec = float(frames / file_sample_rate) if file_sample_rate > 0 else 0.0
    return {
        "path": str(output_path),
        "sample_rate": int(sample_rate),
        "file_sample_rate": file_sample_rate,
        "frames": frames,
        "duration_sec": duration_sec,
        "format": str(info.format),
        "subtype": str(info.subtype),
        "mode": mode,
    }
