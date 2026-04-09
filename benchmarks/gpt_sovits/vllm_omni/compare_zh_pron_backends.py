#!/usr/bin/env python3
"""Compare Chinese pronunciation backend outputs on long text.

This script is designed for offline backend A/B analysis before touching the
full GPT-SoVITS runtime path. It compares two backends by:

1. Normalizing and segmenting the same input text with `text.chinese2`
2. Dumping per-segment raw pronunciation outputs in isolated subprocesses
3. Reporting timing/profile deltas and character-level pronunciation diffs

Note: current `g2pm` integration may be a compatibility stub. The report
includes backend metadata so results can be interpreted correctly.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMP_ROOT = REPO_ROOT / "TEMP"
DEFAULT_TEXT_PATH = REPO_ROOT / "test_cn.txt"
DEFAULT_OUTPUT_JSON = TEMP_ROOT / "zh_pron_backend_compare.json"
DEFAULT_BACKENDS = ("g2pw", "g2pm")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _expand_text(text: str, repeat_until_chars: int | None) -> str:
    if repeat_until_chars is None or repeat_until_chars <= 0 or len(text) >= repeat_until_chars:
        return text
    chunks: list[str] = []
    total = 0
    while total < repeat_until_chars:
        chunks.append(text)
        total += len(text)
    return "\n".join(chunks)


def _run_backend_dump(
    *,
    backend: str,
    text_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dump-backend-json",
        "--backend",
        backend,
        "--text-path",
        str(text_path),
        "--project-root",
        str(project_root),
    ]
    env = os.environ.copy()
    env["GPTSOVITS_ZH_PRON_BACKEND"] = backend
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


def _build_diff_summary(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    max_examples: int,
) -> dict[str, Any]:
    left_segments = left["segments"]
    right_segments = right["segments"]
    segment_count = min(len(left_segments), len(right_segments))
    differing_segments = 0
    differing_chars = 0
    examples: list[dict[str, Any]] = []
    left_only_segments = max(0, len(left_segments) - segment_count)
    right_only_segments = max(0, len(right_segments) - segment_count)

    for seg_index in range(segment_count):
        lseg = left_segments[seg_index]
        rseg = right_segments[seg_index]
        segment_text = lseg["segment"]
        if segment_text != rseg["segment"]:
            differing_segments += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "segment_index": seg_index,
                        "kind": "segment_text_mismatch",
                        "left_segment": segment_text,
                        "right_segment": rseg["segment"],
                    }
                )
            continue

        lp = lseg["pinyins"]
        rp = rseg["pinyins"]
        segment_diff = False
        char_count = min(len(segment_text), len(lp), len(rp))
        for char_index in range(char_count):
            if lp[char_index] == rp[char_index]:
                continue
            differing_chars += 1
            segment_diff = True
            if len(examples) < max_examples:
                examples.append(
                    {
                        "segment_index": seg_index,
                        "char_index": char_index,
                        "char": segment_text[char_index],
                        "left_pinyin": lp[char_index],
                        "right_pinyin": rp[char_index],
                        "segment_preview": segment_text[max(0, char_index - 8) : char_index + 9],
                    }
                )
        if len(lp) != len(rp):
            segment_diff = True
            if len(examples) < max_examples:
                examples.append(
                    {
                        "segment_index": seg_index,
                        "kind": "segment_length_mismatch",
                        "segment": segment_text,
                        "left_pinyin_len": len(lp),
                        "right_pinyin_len": len(rp),
                    }
                )
        if segment_diff:
            differing_segments += 1

    return {
        "segment_count_compared": segment_count,
        "left_only_segments": left_only_segments,
        "right_only_segments": right_only_segments,
        "differing_segments": differing_segments,
        "differing_chars": differing_chars,
        "examples": examples,
    }


def _top_diff_chars(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], int] = {}
    for lseg, rseg in zip(left["segments"], right["segments"]):
        if lseg["segment"] != rseg["segment"]:
            continue
        for char, lp, rp in zip(lseg["segment"], lseg["pinyins"], rseg["pinyins"]):
            if lp == rp:
                continue
            key = (char, lp, rp)
            counts[key] = int(counts.get(key, 0)) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "char": key[0],
            "left_pinyin": key[1],
            "right_pinyin": key[2],
            "count": count,
        }
        for key, count in ranked[:top_k]
    ]


def _inner_dump_backend(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    gpt_root = project_root / "GPT_SoVITS"
    if str(gpt_root) not in sys.path:
        sys.path.insert(0, str(gpt_root))

    import re
    import time

    from text import chinese2

    text = _read_text(Path(args.text_path).resolve())
    normalized = chinese2.text_normalize(text)
    pattern = r"(?<=[{}])\s*".format("".join(chinese2.punctuation))
    segments = [item for item in re.split(pattern, normalized) if item.strip()]
    prepared_segments, batch_inputs = chinese2._prepare_g2p_segments(list(segments))

    started = time.perf_counter()
    backend_results, backend_profile = chinese2._predict_g2pw_batch(batch_inputs)
    elapsed_ms = float((time.perf_counter() - started) * 1000.0)

    converter = getattr(getattr(chinese2, "g2pw", None), "_g2pw", None)
    backend_meta = {
        "selected_backend": str(args.backend),
        "is_g2pw_enabled": bool(getattr(chinese2, "is_g2pw", False)),
        "converter_type": type(converter).__name__ if converter is not None else None,
        "converter_backend": getattr(converter, "backend", None) if converter is not None else None,
        "converter_providers": list(getattr(converter, "providers", [])) if converter is not None else [],
    }

    payload_segments: list[dict[str, Any]] = []
    cursor = 0
    for item in prepared_segments:
        segment = str(item["segment"])
        pinyins: list[str] = []
        if segment and cursor < len(backend_results):
            pinyins = [str(value) for value in backend_results[cursor]]
            cursor += 1
        payload_segments.append(
            {
                "segment": segment,
                "pinyins": pinyins,
                "char_count": len(segment),
            }
        )

    payload = {
        "meta": {
            "backend": str(args.backend),
            "text_path": str(Path(args.text_path).resolve()),
            "normalized_chars": len(normalized),
            "segment_count": len(payload_segments),
            "predict_wall_ms": elapsed_ms,
            "project_root": str(project_root),
        },
        "backend_meta": backend_meta,
        "profile": dict(backend_profile or {}),
        "segments": payload_segments,
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare GPT-SoVITS Chinese pronunciation backends.")
    parser.add_argument("--backend-a", default=DEFAULT_BACKENDS[0], choices=["g2pw", "g2pm", "pypinyin"])
    parser.add_argument("--backend-b", default=DEFAULT_BACKENDS[1], choices=["g2pw", "g2pm", "pypinyin"])
    parser.add_argument("--text-path", type=Path, default=DEFAULT_TEXT_PATH)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "gpt_sovits" / "runtime_lib",
    )
    parser.add_argument("--repeat-until-chars", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dump-backend-json", action="store_true")
    parser.add_argument("--backend", choices=["g2pw", "g2pm", "pypinyin"])
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    if args.dump_backend_json:
        return _inner_dump_backend(args)

    source_text = _read_text(args.text_path.resolve())
    expanded_text = _expand_text(source_text, args.repeat_until_chars if args.repeat_until_chars > 0 else None)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(expanded_text)
        temp_text_path = Path(handle.name)

    try:
        left = _run_backend_dump(backend=args.backend_a, text_path=temp_text_path, project_root=args.project_root)
        right = _run_backend_dump(backend=args.backend_b, text_path=temp_text_path, project_root=args.project_root)
    finally:
        temp_text_path.unlink(missing_ok=True)

    diff_summary = _build_diff_summary(left, right, max_examples=max(1, int(args.max_examples)))
    payload = {
        "meta": {
            "text_path": str(args.text_path.resolve()),
            "source_chars": len(source_text),
            "compared_chars": len(expanded_text),
            "repeat_until_chars": int(args.repeat_until_chars),
            "backend_a": args.backend_a,
            "backend_b": args.backend_b,
        },
        "backend_a": left,
        "backend_b": right,
        "diff_summary": diff_summary,
        "top_diff_pairs": _top_diff_chars(left, right, top_k=max(1, int(args.top_k))),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    predict_wall_values = [
        float(left["meta"]["predict_wall_ms"]),
        float(right["meta"]["predict_wall_ms"]),
    ]
    summary = {
        "output_json": str(args.output_json.resolve()),
        "compared_chars": int(payload["meta"]["compared_chars"]),
        "differing_segments": int(diff_summary["differing_segments"]),
        "differing_chars": int(diff_summary["differing_chars"]),
        "backend_predict_wall_ms": {
            args.backend_a: float(left["meta"]["predict_wall_ms"]),
            args.backend_b: float(right["meta"]["predict_wall_ms"]),
            "mean_ms": float(statistics.mean(predict_wall_values)),
        },
        "top_diff_pairs": payload["top_diff_pairs"][:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote zh pronunciation compare report to: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
