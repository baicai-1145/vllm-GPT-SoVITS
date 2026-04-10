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
from vllm_omni.model_executor.models.gpt_sovits.script_utils import (
    DEFAULT_BENCH_CONFIG_PATH,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_REF_AUDIO,
    DEFAULT_REF_TEXT,
    DEFAULT_TEXT_FILES,
    TEMP_ROOT,
    build_default_request,
    configure_runtime_cache_root,
    elapsed_ms,
    read_text,
    resolve_existing,
    stage_timer,
    summarize_audio_file,
    to_serializable,
)


def _set_env_if_not_none(key: str, value: str | None) -> None:
    if value is not None:
        os.environ[key] = value


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
            "vits_profile": to_serializable(self.vits_profile),
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
    start = stage_timer()
    decoded = runtime.decode_prepared_request(prepared_request)
    vits_ms = elapsed_ms(start)
    vits_profile = runtime.get_last_vits_decode_profile()

    start = stage_timer()
    result = runtime.finalize_decoded_audio(decoded)
    postprocess_ms = elapsed_ms(start)

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
    return summarize_audio_file(output_path, sample_rate=sample_rate, mode="bounded_memory_extra_pass")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark GPT-SoVITS VITS-only hot path.")
    parser.add_argument("--suite", choices=["cn", "en", "4lang"], default="cn")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_BENCH_CONFIG_PATH)
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
        configure_runtime_cache_root(args.runtime_cache_root)

    ref_audio = args.ref_audio.resolve()
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
    text_path = resolve_existing(DEFAULT_TEXT_FILES[args.suite])
    text = read_text(text_path)

    runtime = GPTSoVITSRuntime(
        project_root=str(args.project_root.resolve()),
        config_path=str(args.config_path.resolve()),
    )
    request = build_default_request(text, ref_audio, text_split_method=args.text_split_method)

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
    output_json.write_text(json.dumps(to_serializable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(to_serializable(payload), ensure_ascii=False, indent=2))
    print(f"\nWrote VITS-only benchmark report to: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
