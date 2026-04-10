#!/usr/bin/env python3
"""Extract top GPT-SoVITS enc_p runtime shapes from benchmark JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMP_ROOT = REPO_ROOT / "TEMP"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_suite_iterations(payload: dict[str, Any], suite: str) -> list[dict[str, Any]]:
    suites = payload.get("suites", {})
    suite_payload = suites.get(suite)
    if not isinstance(suite_payload, dict):
        raise KeyError(f"Suite not found in benchmark payload: {suite}")
    iterations = suite_payload.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        raise ValueError(f"Suite {suite} has no iterations")
    return iterations


def _aggregate_shape_hist(iterations: list[dict[str, Any]]) -> Counter[str]:
    shape_hist: Counter[str] = Counter()
    for iteration in iterations:
        vits_profile = iteration.get("vits_profile", {})
        iteration_hist = vits_profile.get("enc_p_runtime_shape_hist", {})
        if not isinstance(iteration_hist, dict):
            continue
        for shape_id, count in iteration_hist.items():
            if not isinstance(shape_id, str):
                continue
            try:
                shape_hist[shape_id] += int(count)
            except Exception:
                continue
    return shape_hist


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract enc_p top shapes from benchmark JSON.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--suite", default="cn")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    payload = _load_json(args.input_json.resolve())
    iterations = _iter_suite_iterations(payload, str(args.suite))
    shape_hist = _aggregate_shape_hist(iterations)
    min_hits = max(1, int(args.min_hits))
    top_k = max(1, int(args.top_k))

    filtered_shapes = [(shape_id, count) for shape_id, count in shape_hist.most_common() if int(count) >= min_hits]
    selected_shapes = filtered_shapes[:top_k]
    whitelist = ",".join(shape_id for shape_id, _count in selected_shapes)

    result = {
        "meta": {
            "input_json": str(args.input_json.resolve()),
            "suite": str(args.suite),
            "top_k": top_k,
            "min_hits": min_hits,
            "num_iterations": len(iterations),
            "num_unique_shapes": len(shape_hist),
        },
        "top_shapes": [{"shape_id": shape_id, "count": int(count)} for shape_id, count in selected_shapes],
        "all_shapes": [{"shape_id": shape_id, "count": int(count)} for shape_id, count in shape_hist.most_common()],
        "env": {
            "GPTSOVITS_COMPILE_VITS_ENC_P_STATIC_BUCKET_SHAPES": whitelist,
        },
    }

    output_json = args.output_json
    if output_json is None:
        output_json = TEMP_ROOT / f"encp_shape_whitelist_{args.suite}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nWrote whitelist report to: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
