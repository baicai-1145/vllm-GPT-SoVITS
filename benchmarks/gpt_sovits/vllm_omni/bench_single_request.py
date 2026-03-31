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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from vllm_omni.model_executor.models.gpt_sovits.runtime import GPTSoVITSRuntime


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMP_ROOT = REPO_ROOT / "TEMP"
DEFAULT_PROJECT_ROOT = REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "gpt_sovits" / "runtime_lib"
DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "tests" / "model_executor" / "models" / "gpt_sovits" / "fixtures" / "tts_infer_v2proplus_cuda_longprompt.yaml"
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
            "t2s_profile": _to_serializable(self.t2s_profile),
        }


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


def run_iteration(
    runtime: GPTSoVITSRuntime,
    *,
    suite: str,
    iteration: int,
    request: dict[str, Any],
    save_audio_path: Path | None,
) -> IterationMetrics:
    total_start = _stage_timer()

    start = _stage_timer()
    spec = runtime.build_request_spec(request, request_id=f"{suite}-{iteration}")
    build_spec_ms = _elapsed_ms(start)

    start = _stage_timer()
    prepared = runtime.prepare_request_spec(spec)
    prepare_ms = _elapsed_ms(start)

    start = _stage_timer()
    semantic_tokens = runtime.generate_semantic_tokens([prepared])[prepared.request_id]
    t2s_ms = _elapsed_ms(start)
    t2s_profile = runtime.get_last_t2s_scheduler_stats()

    start = _stage_timer()
    decode_prepared = runtime.prepare_decode_request(semantic_tokens, prepared.transport_info)
    decode_prepare_ms = _elapsed_ms(start)

    start = _stage_timer()
    decoded = runtime.decode_prepared_request(decode_prepared)
    vits_ms = _elapsed_ms(start)

    start = _stage_timer()
    result = runtime.finalize_decoded_audio(decoded)
    postprocess_ms = _elapsed_ms(start)

    end_to_end_ms = _elapsed_ms(total_start)

    if save_audio_path is not None:
        save_audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(save_audio_path, result.audio, result.sample_rate)

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
        semantic_token_count=int(semantic_tokens.numel()),
        output_samples=int(result.audio.shape[0]),
        sample_rate=int(result.sample_rate),
        prepare_profile={str(key): float(value) for key, value in prepared.state.prepare_profile.items()},
        t2s_profile=t2s_profile,
    )


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
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ref-audio", type=Path, default=DEFAULT_REF_AUDIO)
    parser.add_argument("--text-split-method", default="cut5", help="Segmentation method passed into runtime.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--save-audio", action="store_true", help="Save the last measured audio for each suite to TEMP.")
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

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
            "torch_version": torch.__version__,
            "python": os.sys.version,
        },
        "suites": {},
    }

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    for suite in suites:
        text_path = _resolve_existing(DEFAULT_TEXT_FILES[suite])
        text = _read_text(text_path)
        request = _default_request(text, ref_audio, text_split_method=args.text_split_method)

        for warmup_index in range(args.warmup):
            run_iteration(
                runtime,
                suite=suite,
                iteration=-(warmup_index + 1),
                request=request,
                save_audio_path=None,
            )

        measured: list[IterationMetrics] = []
        for iteration in range(args.iters):
            audio_path = None
            if args.save_audio and iteration == args.iters - 1:
                audio_path = TEMP_ROOT / f"gpt_sovits_{suite}_bench.wav"
            measured.append(
                run_iteration(
                    runtime,
                    suite=suite,
                    iteration=iteration,
                    request=request,
                    save_audio_path=audio_path,
                )
            )

        suite_summary = summarize_iterations(measured)
        suite_summary["text_path"] = str(text_path)
        suite_summary["text_chars"] = len(text)
        result_payload["suites"][suite] = suite_summary

    output_json = args.output_json
    if output_json is None:
        output_json = TEMP_ROOT / f"gpt_sovits_single_request_bench_{int(time.time())}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_to_serializable(result_payload), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(_to_serializable(result_payload), ensure_ascii=False, indent=2))
    print(f"\nWrote benchmark report to: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
