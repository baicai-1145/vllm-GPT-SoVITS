#!/usr/bin/env python3
"""GPT-SoVITS VITS-only hot benchmark.

Builds a fixed request once, runs prepare/t2s/decode-prepare once to obtain a
stable prepared decode request, then repeatedly measures only:
  1. `decode_prepared_request`
  2. optional `finalize_decoded_audio`

This is intended for validating VITS shape bucketing / compile / cudagraph
changes without paying the full T2S cost every iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm_omni.model_executor.models.gpt_sovits.runtime import GPTSoVITSRuntime

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMP_ROOT = REPO_ROOT / "TEMP"
DEFAULT_PROJECT_ROOT = REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "gpt_sovits" / "runtime_lib"
DEFAULT_CONFIG_PATH = (
    REPO_ROOT
    / "tests"
    / "model_executor"
    / "models"
    / "gpt_sovits"
    / "fixtures"
    / "tts_infer_v2proplus_cuda_longprompt.yaml"
)
DEFAULT_REF_AUDIO = REPO_ROOT / "test.wav"
DEFAULT_REF_TEXT = "又或者说，你已经察觉到了…却还想拿「它」干什么好事？"
DEFAULT_TEXT_FILES = {
    "cn": [REPO_ROOT / "test_cn.txt", Path("/root/GPT-SoVITS/test_cn.txt")],
    "en": [REPO_ROOT / "test_en.txt", Path("/root/GPT-SoVITS/test_en.txt")],
    "4lang": [REPO_ROOT / "test_4lang.txt", Path("/root/GPT-SoVITS/test_4lang.txt")],
}


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stage_timer() -> float:
    _sync_cuda()
    return time.perf_counter()


def _elapsed_ms(start: float) -> float:
    _sync_cuda()
    return (time.perf_counter() - start) * 1000.0


def _resolve_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of the candidate paths exist: {joined}")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return value


def _set_env_if_not_none(key: str, value: str | None) -> None:
    if value is not None:
        os.environ[key] = value


def _default_request(text: str, ref_audio_path: Path, *, text_split_method: str) -> dict[str, Any]:
    return {
        "text": text,
        "text_lang": "auto",
        "ref_audio_path": str(ref_audio_path),
        "prompt_text": DEFAULT_REF_TEXT,
        "prompt_lang": "auto",
        "text_split_method": text_split_method,
        "speed_factor": 1.0,
        "sample_steps": 32,
        "super_sampling": False,
        "parallel_infer": True,
        "fragment_interval": 0.3,
        "seed": 1234,
        "batch_size": 4,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "repetition_penalty": 1.35,
    }


@dataclass
class IterationMetrics:
    iteration: int
    vits_ms: float
    postprocess_ms: float
    output_samples: int
    sample_rate: int
    vits_profile: dict[str, Any]

    def summarized(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "vits_ms": self.vits_ms,
            "postprocess_ms": self.postprocess_ms,
            "output_samples": self.output_samples,
            "sample_rate": self.sample_rate,
            "vits_profile": _to_serializable(self.vits_profile),
        }


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "stdev_ms": 0.0}
    return {
        "mean_ms": float(statistics.mean(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "stdev_ms": float(statistics.pstdev(values)),
    }


def run_iteration(
    runtime: GPTSoVITSRuntime,
    *,
    iteration: int,
    prepared_request: Any,
) -> IterationMetrics:
    start = _stage_timer()
    decoded = runtime.decode_prepared_request(prepared_request)
    vits_ms = _elapsed_ms(start)
    vits_profile = runtime.get_last_vits_decode_profile()

    start = _stage_timer()
    result = runtime.finalize_decoded_audio(decoded)
    postprocess_ms = _elapsed_ms(start)

    return IterationMetrics(
        iteration=iteration,
        vits_ms=vits_ms,
        postprocess_ms=postprocess_ms,
        output_samples=int(result.audio.shape[0]),
        sample_rate=int(result.sample_rate),
        vits_profile=vits_profile,
    )


def export_prepared_request_audio_streaming(
    runtime: GPTSoVITSRuntime,
    *,
    prepared_request: Any,
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decoded = runtime.decode_prepared_request(prepared_request)
    sample_rate = runtime.finalize_decoded_audio_to_file(decoded, output_path)
    import soundfile as sf

    info = sf.info(output_path)
    return {
        "path": str(output_path),
        "sample_rate": int(sample_rate),
        "file_sample_rate": int(info.samplerate),
        "frames": int(info.frames),
        "duration_sec": float(info.frames / info.samplerate) if int(info.samplerate) > 0 else 0.0,
        "format": str(info.format),
        "subtype": str(info.subtype),
        "mode": "bounded_memory_extra_pass",
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark GPT-SoVITS VITS-only hot path.")
    parser.add_argument("--suite", choices=["cn", "en", "4lang"], default="cn")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ref-audio", type=Path, default=DEFAULT_REF_AUDIO)
    parser.add_argument("--text-split-method", default="cut5")
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Export one extra bounded-memory audio file after measured iterations.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--prepared-cache",
        type=Path,
        default=None,
        help=(
            "Optional torch cache path for the prepared decode request. "
            "If the file exists it will be loaded; otherwise it will be built and saved."
        ),
    )
    parser.add_argument("--segment-decode-bucketing", choices=["0", "1"], default=None)
    parser.add_argument("--segment-decode-max-batch", default=None)
    parser.add_argument("--segment-decode-max-semantic-tokens", default=None)
    parser.add_argument("--segment-decode-max-phone-tokens", default=None)
    parser.add_argument("--segment-decode-semantic-buckets", default=None)
    parser.add_argument("--segment-decode-phone-buckets", default=None)
    parser.add_argument(
        "--runtime-cache-root",
        type=Path,
        default=None,
        help=(
            "Optional runtime cache root for TorchInductor/Triton. Useful when the default cache path has quota issues."
        ),
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    _set_env_if_not_none("GPTSOVITS_SEGMENT_DECODE_BUCKETING", args.segment_decode_bucketing)
    _set_env_if_not_none("GPTSOVITS_SEGMENT_DECODE_MAX_BATCH", args.segment_decode_max_batch)
    _set_env_if_not_none("GPTSOVITS_SEGMENT_DECODE_MAX_SEMANTIC_TOKENS", args.segment_decode_max_semantic_tokens)
    _set_env_if_not_none("GPTSOVITS_SEGMENT_DECODE_MAX_PHONE_TOKENS", args.segment_decode_max_phone_tokens)
    _set_env_if_not_none("GPTSOVITS_SEGMENT_DECODE_SEMANTIC_BUCKETS", args.segment_decode_semantic_buckets)
    _set_env_if_not_none("GPTSOVITS_SEGMENT_DECODE_PHONE_BUCKETS", args.segment_decode_phone_buckets)
    if args.runtime_cache_root is not None:
        runtime_cache_root = args.runtime_cache_root.resolve()
        torchinductor_cache_dir = runtime_cache_root / "torchinductor"
        triton_cache_dir = runtime_cache_root / "triton"
        torchinductor_cache_dir.mkdir(parents=True, exist_ok=True)
        triton_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["GPTSOVITS_RUNTIME_CACHE_ROOT"] = str(runtime_cache_root)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(torchinductor_cache_dir)
        os.environ["TRITON_CACHE_DIR"] = str(triton_cache_dir)

    ref_audio = args.ref_audio.resolve()
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
    text_path = _resolve_existing(DEFAULT_TEXT_FILES[args.suite])
    text = _read_text(text_path)

    runtime = GPTSoVITSRuntime(
        project_root=str(args.project_root.resolve()),
        config_path=str(args.config_path.resolve()),
    )
    request = _default_request(text, ref_audio, text_split_method=args.text_split_method)

    setup_ms = 0.0
    cache_loaded = False
    decode_prepared = None
    if args.prepared_cache is not None and args.prepared_cache.exists():
        build_start = _stage_timer()
        decode_prepared = torch.load(args.prepared_cache, map_location="cpu", weights_only=False)
        setup_ms = _elapsed_ms(build_start)
        cache_loaded = True
    if decode_prepared is None:
        build_start = _stage_timer()
        spec = runtime.build_request_spec(request, request_id=f"vits-only-{args.suite}")
        prepared = runtime.prepare_request_spec(spec)
        semantic_tokens = runtime.generate_semantic_tokens([prepared])[prepared.request_id]
        decode_prepared = runtime.prepare_decode_request(semantic_tokens, prepared.transport_info)
        setup_ms = _elapsed_ms(build_start)
        if args.prepared_cache is not None:
            args.prepared_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(decode_prepared, args.prepared_cache)

    measured: list[IterationMetrics] = []
    for warmup_index in range(args.warmup):
        run_iteration(
            runtime,
            iteration=-(warmup_index + 1),
            prepared_request=decode_prepared,
        )
    for iteration in range(args.iters):
        measured.append(
            run_iteration(
                runtime,
                iteration=iteration,
                prepared_request=decode_prepared,
            )
        )

    summarized = [item.summarized() for item in measured]
    payload = {
        "meta": {
            "suite": args.suite,
            "project_root": str(args.project_root.resolve()),
            "config_path": str(args.config_path.resolve()),
            "ref_audio": str(ref_audio),
            "ref_text": DEFAULT_REF_TEXT,
            "text_path": str(text_path),
            "text_chars": len(text),
            "text_split_method": args.text_split_method,
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "setup_ms": float(setup_ms),
            "prepared_cache": None if args.prepared_cache is None else str(args.prepared_cache),
            "prepared_cache_loaded": bool(cache_loaded),
            "runtime_cache_root": None if args.runtime_cache_root is None else str(args.runtime_cache_root.resolve()),
            "segment_decode_bucketing": os.environ.get("GPTSOVITS_SEGMENT_DECODE_BUCKETING"),
            "segment_decode_max_batch": os.environ.get("GPTSOVITS_SEGMENT_DECODE_MAX_BATCH"),
            "segment_decode_max_semantic_tokens": os.environ.get("GPTSOVITS_SEGMENT_DECODE_MAX_SEMANTIC_TOKENS"),
            "segment_decode_max_phone_tokens": os.environ.get("GPTSOVITS_SEGMENT_DECODE_MAX_PHONE_TOKENS"),
            "segment_decode_semantic_buckets": os.environ.get("GPTSOVITS_SEGMENT_DECODE_SEMANTIC_BUCKETS"),
            "segment_decode_phone_buckets": os.environ.get("GPTSOVITS_SEGMENT_DECODE_PHONE_BUCKETS"),
            "saved_audio": (
                export_prepared_request_audio_streaming(
                    runtime,
                    prepared_request=decode_prepared,
                    output_path=TEMP_ROOT / f"gpt_sovits_vits_only_{args.suite}.wav",
                )
                if args.save_audio
                else None
            ),
        },
        "iterations": summarized,
        "aggregate": {
            "vits_ms": _stats([item["vits_ms"] for item in summarized]),
            "postprocess_ms": _stats([item["postprocess_ms"] for item in summarized]),
            "output_samples": {
                "mean": float(statistics.mean([item["output_samples"] for item in summarized])),
                "min": float(min(item["output_samples"] for item in summarized)),
                "max": float(max(item["output_samples"] for item in summarized)),
            },
            "sample_rate": int(summarized[0]["sample_rate"]) if summarized else 32000,
        },
    }

    output_json = args.output_json or (TEMP_ROOT / f"gpt_sovits_vits_only_{args.suite}_{int(time.time())}.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_to_serializable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_to_serializable(payload), ensure_ascii=False, indent=2))
    print(f"\nWrote VITS-only benchmark report to: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
