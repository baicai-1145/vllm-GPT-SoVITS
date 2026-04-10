#!/usr/bin/env python3
"""GPT-SoVITS v2ProPlus single-request hot benchmark.

This benchmark targets the native runtime path inside vLLM-Omni and keeps the
inputs fixed so later optimization rounds remain comparable.
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


def _semantic_token_count(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return sum(_semantic_token_count(item) for item in value)
    return 0


@dataclass
class IterationMetrics:
    suite: str
    iteration: int
    build_spec_ms: float
    prepare_ms: float
    t2s_ms: float
    decode_prepare_ms: float
    vits_ms: float
    postprocess_ms: float
    end_to_end_ms: float
    semantic_token_count: int
    output_samples: int
    sample_rate: int
    prepare_profile: dict[str, float]
    t2s_profile: dict[str, Any]
    vits_profile: dict[str, Any]

    def summarized(self) -> dict[str, Any]:
        profile = self.prepare_profile
        return {
            "suite": self.suite,
            "iteration": self.iteration,
            "build_spec_ms": self.build_spec_ms,
            "prepare_ms": self.prepare_ms,
            "t2s_ms": self.t2s_ms,
            "decode_prepare_ms": self.decode_prepare_ms,
            "vits_ms": self.vits_ms,
            "postprocess_ms": self.postprocess_ms,
            "end_to_end_ms": self.end_to_end_ms,
            "semantic_token_count": self.semantic_token_count,
            "output_samples": self.output_samples,
            "sample_rate": self.sample_rate,
            "text_features_ms": float(profile.get("text_features_ms", 0.0)),
            "prompt_text_features_ms": float(profile.get("prompt_text_features_ms", 0.0)),
            "text_cpu_preprocess_ms": float(profile.get("text_cpu_preprocess_ms", 0.0)),
            "prompt_text_cpu_preprocess_ms": float(profile.get("prompt_text_cpu_preprocess_ms", 0.0)),
            "text_g2pw_total_ms": float(profile.get("text_g2pw_total_ms", 0.0)),
            "prompt_text_g2pw_total_ms": float(profile.get("prompt_text_g2pw_total_ms", 0.0)),
            "text_bert_forward_ms": float(profile.get("text_bert_forward_ms", 0.0)),
            "prompt_text_bert_forward_ms": float(profile.get("prompt_text_bert_forward_ms", 0.0)),
            "prompt_semantic_ms": float(profile.get("prompt_semantic_ms", 0.0)),
            "ref_spec_ms": float(profile.get("ref_spec_ms", 0.0)),
            "tensorize_ms": float(profile.get("tensorize_ms", 0.0)),
            "prepare_total_ms": float(profile.get("total_ms", 0.0)),
            "prepare_wall_total_ms": float(profile.get("wall_total_ms", 0.0)),
            "g2pw_total_ms": float(profile.get("text_g2pw_total_ms", 0.0))
            + float(profile.get("prompt_text_g2pw_total_ms", 0.0)),
            "t2s_profile": to_serializable(self.t2s_profile),
            "vits_profile": to_serializable(self.vits_profile),
        }


def run_iteration(
    runtime: GPTSoVITSRuntime,
    *,
    suite: str,
    iteration: int,
    request: dict[str, Any],
) -> IterationMetrics:
    total_start = stage_timer()

    start = stage_timer()
    spec = runtime.build_request_spec(request, request_id=f"{suite}-{iteration}")
    build_spec_ms = elapsed_ms(start)

    start = stage_timer()
    prepared = runtime.prepare_request_spec(spec)
    prepare_ms = elapsed_ms(start)

    start = stage_timer()
    semantic_tokens = runtime.generate_semantic_tokens([prepared])[prepared.request_id]
    t2s_ms = elapsed_ms(start)
    t2s_profile = runtime.get_last_t2s_scheduler_stats()

    start = stage_timer()
    decode_prepared = runtime.prepare_decode_request(semantic_tokens, prepared.transport_info)
    decode_prepare_ms = elapsed_ms(start)

    start = stage_timer()
    decoded = runtime.decode_prepared_request(decode_prepared)
    vits_ms = elapsed_ms(start)
    vits_profile = runtime.get_last_vits_decode_profile()

    start = stage_timer()
    result = runtime.finalize_decoded_audio(decoded)
    postprocess_ms = elapsed_ms(start)

    end_to_end_ms = elapsed_ms(total_start)

    return IterationMetrics(
        suite=suite,
        iteration=iteration,
        build_spec_ms=build_spec_ms,
        prepare_ms=prepare_ms,
        t2s_ms=t2s_ms,
        decode_prepare_ms=decode_prepare_ms,
        vits_ms=vits_ms,
        postprocess_ms=postprocess_ms,
        end_to_end_ms=end_to_end_ms,
        semantic_token_count=_semantic_token_count(semantic_tokens),
        output_samples=int(result.audio.shape[0]),
        sample_rate=int(result.sample_rate),
        prepare_profile={str(key): float(value) for key, value in prepared.state.prepare_profile.items()},
        t2s_profile=t2s_profile,
        vits_profile=vits_profile,
    )


def export_audio_streaming(
    runtime: GPTSoVITSRuntime,
    *,
    request: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = runtime.synthesize_to_file(request, output_path)
    return summarize_audio_file(output_path, sample_rate=sample_rate, mode="bounded_memory_extra_pass")


def summarize_iterations(items: list[IterationMetrics]) -> dict[str, Any]:
    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "stdev_ms": 0.0}
        return {
            "mean_ms": float(statistics.mean(values)),
            "min_ms": float(min(values)),
            "max_ms": float(max(values)),
            "stdev_ms": float(statistics.pstdev(values)),
        }

    summarized = [item.summarized() for item in items]
    first = summarized[0]
    return {
        "iterations": summarized,
        "aggregate": {
            "build_spec_ms": stats([item["build_spec_ms"] for item in summarized]),
            "prepare_ms": stats([item["prepare_ms"] for item in summarized]),
            "t2s_ms": stats([item["t2s_ms"] for item in summarized]),
            "decode_prepare_ms": stats([item["decode_prepare_ms"] for item in summarized]),
            "vits_ms": stats([item["vits_ms"] for item in summarized]),
            "postprocess_ms": stats([item["postprocess_ms"] for item in summarized]),
            "end_to_end_ms": stats([item["end_to_end_ms"] for item in summarized]),
            "g2pw_total_ms": stats([item["g2pw_total_ms"] for item in summarized]),
            "text_cpu_preprocess_ms": stats([item["text_cpu_preprocess_ms"] for item in summarized]),
            "text_bert_forward_ms": stats([item["text_bert_forward_ms"] for item in summarized]),
            "prompt_semantic_ms": stats([item["prompt_semantic_ms"] for item in summarized]),
            "ref_spec_ms": stats([item["ref_spec_ms"] for item in summarized]),
            "semantic_token_count": {
                "mean": float(statistics.mean([item["semantic_token_count"] for item in summarized])),
                "min": float(min(item["semantic_token_count"] for item in summarized)),
                "max": float(max(item["semantic_token_count"] for item in summarized)),
            },
            "output_samples": {
                "mean": float(statistics.mean([item["output_samples"] for item in summarized])),
                "min": float(min(item["output_samples"] for item in summarized)),
                "max": float(max(item["output_samples"] for item in summarized)),
            },
            "sample_rate": int(first["sample_rate"]),
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark GPT-SoVITS native single-request hot path.")
    parser.add_argument("--suite", choices=["cn", "en", "4lang", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations per suite.")
    parser.add_argument("--iters", type=int, default=3, help="Number of measured iterations per suite.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_BENCH_CONFIG_PATH)
    parser.add_argument("--ref-audio", type=Path, default=DEFAULT_REF_AUDIO)
    parser.add_argument("--text-split-method", default="cut5", help="Segmentation method passed into runtime.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Export one extra bounded-memory audio file per suite after measured iterations.",
    )
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

    if args.runtime_cache_root is not None:
        configure_runtime_cache_root(args.runtime_cache_root)

    suites = ["cn", "en", "4lang"] if args.suite == "all" else [args.suite]
    ref_audio = args.ref_audio.resolve()
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    runtime = GPTSoVITSRuntime(
        project_root=str(args.project_root.resolve()),
        config_path=str(args.config_path.resolve()),
    )

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    result_payload: dict[str, Any] = {
        "meta": {
            "project_root": str(args.project_root.resolve()),
            "config_path": str(args.config_path.resolve()),
            "ref_audio": str(ref_audio),
            "ref_text": DEFAULT_REF_TEXT,
            "text_split_method": args.text_split_method,
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "device": device_name,
            "cuda_available": bool(torch.cuda.is_available()),
            "runtime_cache_root": None if args.runtime_cache_root is None else str(args.runtime_cache_root.resolve()),
            "torch_version": torch.__version__,
            "python": os.sys.version,
        },
        "suites": {},
    }

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    for suite in suites:
        text_path = resolve_existing(DEFAULT_TEXT_FILES[suite])
        text = read_text(text_path)
        request = build_default_request(text, ref_audio, text_split_method=args.text_split_method)

        for warmup_index in range(args.warmup):
            run_iteration(
                runtime,
                suite=suite,
                iteration=-(warmup_index + 1),
                request=request,
            )

        measured: list[IterationMetrics] = []
        for iteration in range(args.iters):
            measured.append(
                run_iteration(
                    runtime,
                    suite=suite,
                    iteration=iteration,
                    request=request,
                )
            )

        suite_summary = summarize_iterations(measured)
        suite_summary["text_path"] = str(text_path)
        suite_summary["text_chars"] = len(text)
        if args.save_audio:
            suite_summary["saved_audio"] = export_audio_streaming(
                runtime,
                request=request,
                output_path=TEMP_ROOT / f"gpt_sovits_{suite}_bench.wav",
            )
        result_payload["suites"][suite] = suite_summary

    output_json = args.output_json
    if output_json is None:
        output_json = TEMP_ROOT / f"gpt_sovits_single_request_bench_{int(time.time())}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(to_serializable(result_payload), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(to_serializable(result_payload), ensure_ascii=False, indent=2))
    print(f"\nWrote benchmark report to: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
