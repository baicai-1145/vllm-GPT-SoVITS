#!/usr/bin/env python3
"""Offline GPT-SoVITS synthesis using the bounded-memory file path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vllm_omni.model_executor.models.gpt_sovits.runtime import GPTSoVITSRuntime
from vllm_omni.model_executor.models.gpt_sovits.script_utils import (
    DEFAULT_EXAMPLE_CONFIG_PATH,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_REF_AUDIO,
    DEFAULT_REF_TEXT,
    build_default_request,
    build_timestamped_output_path,
    configure_runtime_cache_root,
    elapsed_s,
    render_output_stem,
    resolve_synthesis_inputs,
    stage_timer,
    summarize_audio_file,
    to_serializable,
)

DEFAULT_OUTPUT_DIR = Path("output_audio")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GPT-SoVITS offline synthesis directly to file without accumulating full audio in memory."
    )
    parser.add_argument("--text", default=None, help="Inline text to synthesize.")
    parser.add_argument("--text-file", type=Path, default=None, help="UTF-8 text file to synthesize.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory of UTF-8 text files to synthesize.")
    parser.add_argument("--glob", default="*.txt", help="Glob pattern used with --input-dir. Default: *.txt")
    parser.add_argument("--text-lang", default="auto", help="Target text language passed to GPT-SoVITS.")
    parser.add_argument("--ref-audio", type=Path, default=DEFAULT_REF_AUDIO, help="Reference audio path.")
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT, help="Transcript for the reference audio.")
    parser.add_argument("--prompt-lang", default="auto", help="Reference transcript language.")
    parser.add_argument("--text-split-method", default="cut5", help="Segmentation method passed into runtime.")
    parser.add_argument("--speed-factor", type=float, default=1.0)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--fragment-interval", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--super-sampling", action="store_true", help="Enable super sampling for single-segment cases.")
    parser.add_argument("--parallel-infer", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-threshold", type=float, default=0.75)
    parser.add_argument("--split-bucket", action="store_true", default=True)
    parser.add_argument("--repetition-penalty", type=float, default=1.35)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_EXAMPLE_CONFIG_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output WAV path. Defaults to output_audio/gpt_sovits_<timestamp>.wav",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional summary JSON path. In batch mode this becomes the aggregate summary path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for batch mode and default single-file exports.",
    )
    parser.add_argument(
        "--output-name-template",
        default="{index:03d}_{stem}",
        help="Batch output filename template without extension. Fields: {index}, {id}, {stem}.",
    )
    parser.add_argument(
        "--runtime-cache-root",
        type=Path,
        default=None,
        help="Optional cache root for TorchInductor/Triton when the default cache path has quota issues.",
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    items = resolve_synthesis_inputs(
        text=args.text,
        text_file=args.text_file,
        input_dir=args.input_dir,
        glob_pattern=args.glob,
        required_arg_hint="GPT-SoVITS offline example requires --text or --text-file",
    )
    ref_audio = args.ref_audio.resolve()
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
    is_batch = args.input_dir is not None or len(items) > 1
    if is_batch and args.output is not None:
        raise ValueError("--output is only supported for single-input mode")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_batch:
        aggregate_json = (args.output_json or (output_dir / f"gpt_sovits_batch_{int(time.time())}.json")).resolve()
    else:
        default_output = build_timestamped_output_path(output_dir, prefix="gpt_sovits")
        single_output_path = (args.output or default_output).resolve()
        single_output_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_json = (args.output_json or single_output_path.with_suffix(".json")).resolve()
    aggregate_json.parent.mkdir(parents=True, exist_ok=True)

    runtime_cache_root = configure_runtime_cache_root(args.runtime_cache_root)
    runtime = GPTSoVITSRuntime(
        project_root=str(args.project_root.resolve()),
        config_path=str(args.config_path.resolve()),
    )
    aggregate_items: list[dict[str, object]] = []
    single_output_print_path: Path | None = None
    single_summary_print_path: Path | None = None
    for index, item in enumerate(items):
        if is_batch:
            output_stem = render_output_stem(args.output_name_template, index=index, item_id=item.item_id)
            output_path = (output_dir / f"{output_stem}.wav").resolve()
            item_summary_path = output_path.with_suffix(".json")
        else:
            output_path = single_output_path
            item_summary_path = aggregate_json
            single_output_print_path = output_path
            single_summary_print_path = item_summary_path

        request = build_default_request(
            item.text,
            ref_audio,
            ref_text=args.ref_text,
            text_lang=args.text_lang,
            prompt_lang=args.prompt_lang,
            text_split_method=args.text_split_method,
            speed_factor=args.speed_factor,
            sample_steps=args.sample_steps,
            super_sampling=args.super_sampling,
            parallel_infer=args.parallel_infer,
            fragment_interval=args.fragment_interval,
            seed=args.seed,
            batch_size=args.batch_size,
            batch_threshold=args.batch_threshold,
            split_bucket=args.split_bucket,
            repetition_penalty=args.repetition_penalty,
        )
        start = stage_timer()
        sample_rate = runtime.synthesize_to_file(request, output_path)
        wall_time_s = elapsed_s(start)
        audio_summary = summarize_audio_file(output_path, sample_rate=sample_rate, mode="bounded_memory")
        duration_sec = float(audio_summary["duration_sec"])
        item_summary = {
            "output_path": str(output_path),
            "output_json": str(item_summary_path),
            "sample_rate_returned": int(sample_rate),
            "sample_rate_file": int(audio_summary["file_sample_rate"]),
            "frames": int(audio_summary["frames"]),
            "duration_sec": duration_sec,
            "wall_time_sec": float(wall_time_s),
            "rtf": float(wall_time_s / duration_sec) if duration_sec > 0 else None,
            "format": str(audio_summary["format"]),
            "subtype": str(audio_summary["subtype"]),
            "text_chars": len(item.text),
            "input_id": item.item_id,
            "input_source_kind": item.source_kind,
            "input_source_path": item.source_path,
            "ref_audio": str(ref_audio),
            "project_root": str(args.project_root.resolve()),
            "config_path": str(args.config_path.resolve()),
            "runtime_cache_root": runtime_cache_root,
            "bounded_memory": True,
        }
        item_summary_path.write_text(json.dumps(item_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        aggregate_items.append(item_summary)

    if is_batch:
        aggregate_summary = {
            "mode": "batch",
            "count": len(aggregate_items),
            "input_dir": str(args.input_dir.resolve()) if args.input_dir is not None else None,
            "glob": str(args.glob),
            "output_dir": str(output_dir),
            "output_name_template": str(args.output_name_template),
            "ref_audio": str(ref_audio),
            "project_root": str(args.project_root.resolve()),
            "config_path": str(args.config_path.resolve()),
            "runtime_cache_root": runtime_cache_root,
            "items": aggregate_items,
        }
        aggregate_json.write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(to_serializable(aggregate_summary), ensure_ascii=False, indent=2))
        print(f"\nWrote GPT-SoVITS batch summary to: {aggregate_json}")
    else:
        single_summary = aggregate_items[0]
        print(json.dumps(to_serializable(single_summary), ensure_ascii=False, indent=2))
        print(f"\nWrote GPT-SoVITS audio to: {single_output_print_path}")
        print(f"Wrote GPT-SoVITS summary to: {single_summary_print_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
