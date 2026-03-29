from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Sequence

from GPT_SoVITS.TTS_infer_pack.TTS import TTS
from GPT_SoVITS.TTS_infer_pack.unified_engine_builder import EngineCompositionBuilder
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import RuntimeControlCallbacks
from GPT_SoVITS.TTS_infer_pack.unified_engine_delegates import EngineApiDelegates, EngineBridgeDelegates, EngineRuntimeDelegates
from GPT_SoVITS.TTS_infer_pack.unified_engine_public import EngineCompatInterface, EnginePublicInterface


class UnifiedTTSEngine(EnginePublicInterface, EngineCompatInterface, EngineBridgeDelegates, EngineApiDelegates, EngineRuntimeDelegates):
    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return bool(default)
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value in [None, ""]:
            return int(default)
        return int(value)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.environ.get(name)
        if value in [None, ""]:
            return float(default)
        return float(value)

    def __init__(
        self,
        tts: TTS,
        cut_method_names: Sequence[str],
        control_callbacks: RuntimeControlCallbacks | None = None,
        max_steps: int = 1500,
        micro_batch_wait_ms: int = 5,
    ) -> None:
        self.tts = tts
        self.cut_method_names = set(cut_method_names)
        self.control_callbacks = control_callbacks or RuntimeControlCallbacks()
        self.runtime_cache_env = {
            "cache_root": os.environ.get("GPTSOVITS_RUNTIME_CACHE_ROOT"),
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        }
        self.startup_prewarm_summary = {
            "enabled": False,
            "ran": False,
            "reason": "not_started",
            "rounds": 0,
            "plan_count": 0,
            "total_ms": 0.0,
            "plans": [],
        }
        EngineCompositionBuilder(self).build(max_steps=max_steps, micro_batch_wait_ms=micro_batch_wait_ms)

    @staticmethod
    def _split_warmup_text_units(text: str, text_lang: str) -> list[str]:
        normalized_lang = str(text_lang or "").strip().lower()
        raw_text = str(text or "").strip()
        if not raw_text:
            return []
        if normalized_lang in {"en", "all_en"}:
            parts = re.split(r"(?<=[.!?])\s+", raw_text)
            return [part.strip() for part in parts if part.strip()]
        line_units = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if len(line_units) > 1:
            return line_units
        parts = re.split(r"(?<=[。！？!?])", raw_text)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _join_warmup_text_units(units: list[str], text_lang: str) -> str:
        normalized_lang = str(text_lang or "").strip().lower()
        if normalized_lang in {"en", "all_en"}:
            return " ".join(units)
        return "\n".join(units)

    @staticmethod
    def _resolve_warmup_prompt_text(
        *,
        repo_root: Path,
        ref_audio_path: Path,
    ) -> str | None:
        prompt_text = os.environ.get("GPTSOVITS_STARTUP_PREWARM_PROMPT_TEXT")
        if prompt_text not in [None, ""]:
            return str(prompt_text)
        prompt_text_path = os.environ.get("GPTSOVITS_STARTUP_PREWARM_PROMPT_TEXT_PATH")
        if prompt_text_path:
            candidate = Path(prompt_text_path).expanduser()
            if candidate.exists():
                return candidate.read_text(encoding="utf-8").strip()
        ref_lab_path = ref_audio_path.with_suffix(".lab")
        if ref_lab_path.exists():
            return ref_lab_path.read_text(encoding="utf-8").strip()
        default_prompt_path = repo_root / "test.txt"
        if default_prompt_path.exists():
            return default_prompt_path.read_text(encoding="utf-8").strip()
        return None

    def _resolve_startup_prewarm_reference(self) -> tuple[str, str, str] | None:
        repo_root = Path(__file__).resolve().parents[2]
        ref_audio_env = os.environ.get("GPTSOVITS_STARTUP_PREWARM_REF_AUDIO_PATH")
        ref_audio_path = Path(ref_audio_env).expanduser() if ref_audio_env else None
        if ref_audio_path is None or not ref_audio_path.exists():
            testwav_dir = repo_root / "testwav"
            wav_candidates = sorted(testwav_dir.glob("*.wav")) if testwav_dir.exists() else []
            ref_audio_path = wav_candidates[0] if wav_candidates else (repo_root / "test.wav")
        if ref_audio_path is None or not ref_audio_path.exists():
            return None
        prompt_text = self._resolve_warmup_prompt_text(repo_root=repo_root, ref_audio_path=ref_audio_path)
        if not prompt_text:
            return None
        prompt_lang = str(os.environ.get("GPTSOVITS_STARTUP_PREWARM_PROMPT_LANG", "zh")).strip() or "zh"
        return str(ref_audio_path), prompt_text, prompt_lang

    def _load_startup_prewarm_text(
        self,
        *,
        text_path: Path,
        text_lang: str,
        unit_limit: int,
    ) -> str:
        raw_text = text_path.read_text(encoding="utf-8").strip()
        if unit_limit <= 0:
            return raw_text
        units = self._split_warmup_text_units(raw_text, text_lang)
        if not units:
            return raw_text
        return self._join_warmup_text_units(units[:unit_limit], text_lang)

    @staticmethod
    def _parse_startup_prewarm_bucket_spec(raw_bucket_spec: str) -> tuple[list[int], bool]:
        bucket_values: list[int] = []
        include_exact = False
        for raw_item in str(raw_bucket_spec or "").split(","):
            item = raw_item.strip().lower()
            if not item:
                continue
            if item in {"exact", "full", "max"}:
                include_exact = True
                continue
            try:
                value = int(item)
            except ValueError:
                continue
            if value > 0:
                bucket_values.append(int(value))
        return sorted(set(bucket_values)), bool(include_exact)

    def _resolve_startup_prewarm_bucket_limits(
        self,
        *,
        available_units: int,
        unit_limit: int,
        bucket_spec: str,
    ) -> list[int]:
        capped_limit = int(available_units) if unit_limit <= 0 else min(int(available_units), int(unit_limit))
        if capped_limit <= 0:
            return []
        bucket_values, include_exact = self._parse_startup_prewarm_bucket_spec(bucket_spec)
        resolved = [value for value in bucket_values if 0 < value <= capped_limit]
        if include_exact:
            resolved.append(int(capped_limit))
        return sorted(set(resolved))

    def _startup_prewarm_manifest_path(self) -> Path | None:
        cache_root = str((self.runtime_cache_env or {}).get("cache_root") or "").strip()
        if not cache_root:
            return None
        cache_root_path = Path(cache_root).expanduser()
        cache_root_path.mkdir(parents=True, exist_ok=True)
        return cache_root_path / "startup_prewarm_manifest.json"

    def _build_startup_prewarm_signature(self, requests: list[dict], rounds: int) -> dict:
        repo_root = Path(__file__).resolve().parents[2]
        compile_env_keys = [
            "GPTSOVITS_COMPILE_T2S_PREALLOC_MODE",
            "GPTSOVITS_COMPILE_T2S_PREALLOC_DYNAMIC",
            "GPTSOVITS_COMPILE_T2S_PREALLOC_CAPTURE_SCALAR_OUTPUTS",
            "GPTSOVITS_COMPILE_T2S_PREALLOC_SKIP_DYNAMIC_CUDAGRAPH",
            "GPTSOVITS_COMPILE_VITS_DEC_MODE",
            "GPTSOVITS_COMPILE_VITS_DEC_DYNAMIC",
            "GPTSOVITS_COMPILE_VITS_DEC_SKIP_DYNAMIC_CUDAGRAPH",
            "GPTSOVITS_STARTUP_PREWARM_MODE",
            "GPTSOVITS_STARTUP_PREWARM_SHAPE_EXTRA_DECODE_STEPS",
            "GPTSOVITS_STARTUP_PREWARM_SHAPE_SEMANTIC_MULTIPLIER",
            "TORCHINDUCTOR_FX_GRAPH_CACHE",
            "TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE",
        ]
        tracked_files = [
            repo_root / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py",
            repo_root / "GPT_SoVITS" / "AR" / "models" / "t2s_model.py",
            repo_root / "GPT_SoVITS" / "TTS_infer_pack" / "unified_engine.py",
            repo_root / "GPT_SoVITS" / "TTS_infer_pack" / "unified_engine_api.py",
            repo_root / "GPT_SoVITS" / "TTS_infer_pack" / "unified_engine_api_direct.py",
        ]
        tracked_sources = {}
        for path in tracked_files:
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            tracked_sources[str(path)] = {
                "mtime_ns": int(stat.st_mtime_ns),
                "size": int(stat.st_size),
            }
        request_signatures = []
        for request in requests:
            meta = dict(request.get("_warmup_meta") or {})
            request_signatures.append(
                {
                    "text_path": str(meta.get("text_path") or ""),
                    "text_lang": str(meta.get("text_lang") or ""),
                    "text_split_method": str(meta.get("text_split_method") or ""),
                    "unit_limit": int(meta.get("unit_limit", 0) or 0),
                    "available_unit_count": int(meta.get("available_unit_count", 0) or 0),
                    "raw_segment_count_estimate": int(meta.get("raw_segment_count_estimate", 0) or 0),
                    "warmup_mode": str(meta.get("warmup_mode") or "full"),
                }
            )
        return {
            "python_version": sys.version,
            "t2s_weights_path": str(getattr(self.tts.configs, "t2s_weights_path", "")),
            "vits_weights_path": str(getattr(self.tts.configs, "vits_weights_path", "")),
            "rounds": int(rounds),
            "requests": request_signatures,
            "compile_env": {key: str(os.environ.get(key, "")) for key in compile_env_keys},
            "tracked_sources": tracked_sources,
        }

    def _load_startup_prewarm_manifest(self, signature: dict) -> dict | None:
        manifest_path = self._startup_prewarm_manifest_path()
        if manifest_path is None or not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if dict(manifest.get("signature") or {}) != dict(signature):
            return None
        summary = dict(manifest.get("summary") or {})
        if not summary or str(summary.get("reason")) != "completed":
            return None
        plans = list(summary.get("plans") or [])
        if not plans or not all(bool(item.get("ok", False)) for item in plans):
            return None
        return manifest

    def _write_startup_prewarm_manifest(self, *, signature: dict, summary: dict) -> None:
        manifest_path = self._startup_prewarm_manifest_path()
        if manifest_path is None:
            return
        payload = {
            "signature": dict(signature),
            "summary": dict(summary),
            "written_at": float(time.time()),
        }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_startup_prewarm_requests(self) -> list[dict]:
        reference = self._resolve_startup_prewarm_reference()
        if reference is None:
            return []
        ref_audio_path, prompt_text, prompt_lang = reference
        repo_root = Path(__file__).resolve().parents[2]
        plan_spec_env = os.environ.get(
            "GPTSOVITS_STARTUP_PREWARM_PLAN",
            "test_en.txt|en|cut4|32|1,8,16,32;test_cn.txt|zh|cut5|32|1,8,16,32",
        )
        default_bucket_spec = str(os.environ.get("GPTSOVITS_STARTUP_PREWARM_BUCKETS", "") or "").strip()
        timeout_sec = max(30.0, float(os.environ.get("GPTSOVITS_STARTUP_PREWARM_TIMEOUT_SEC", "900")))
        sample_steps = max(1, int(os.environ.get("GPTSOVITS_STARTUP_PREWARM_SAMPLE_STEPS", "32")))
        auto_bucket_enabled = self._env_flag("GPTSOVITS_STARTUP_PREWARM_AUTO_BUCKETS", True)
        warmup_mode = str(os.environ.get("GPTSOVITS_STARTUP_PREWARM_MODE", "full")).strip().lower() or "full"
        requests: list[dict] = []
        for plan_index, raw_plan in enumerate(str(plan_spec_env).split(";")):
            raw_plan = raw_plan.strip()
            if not raw_plan:
                continue
            fields = [item.strip() for item in raw_plan.split("|")]
            if len(fields) < 3:
                continue
            text_path = Path(fields[0]).expanduser()
            if not text_path.is_absolute():
                text_path = repo_root / text_path
            if not text_path.exists():
                continue
            text_lang = fields[1] or "zh"
            text_split_method = fields[2] or "cut1"
            unit_limit = int(fields[3]) if len(fields) >= 4 and fields[3] not in {"", None} else 0
            raw_text = text_path.read_text(encoding="utf-8").strip()
            if not raw_text:
                continue
            units = self._split_warmup_text_units(raw_text, text_lang)
            available_units = len(units) if units else 0
            if available_units > 0:
                effective_unit_limit = int(available_units) if unit_limit <= 0 else min(int(unit_limit), int(available_units))
            else:
                effective_unit_limit = max(1, int(unit_limit) if unit_limit > 0 else 1)
            bucket_spec = fields[4] if len(fields) >= 5 else default_bucket_spec
            unit_limits = [int(effective_unit_limit)]
            if auto_bucket_enabled:
                bucket_limits = self._resolve_startup_prewarm_bucket_limits(
                    available_units=available_units if available_units > 0 else max(1, effective_unit_limit),
                    unit_limit=effective_unit_limit,
                    bucket_spec=bucket_spec,
                )
                if bucket_limits:
                    unit_limits = bucket_limits
            seen_unit_limits: set[int] = set()
            for bucket_index, bucket_limit in enumerate(unit_limits):
                resolved_limit = int(bucket_limit)
                if resolved_limit <= 0 or resolved_limit in seen_unit_limits:
                    continue
                seen_unit_limits.add(resolved_limit)
                text = self._load_startup_prewarm_text(
                    text_path=text_path,
                    text_lang=text_lang,
                    unit_limit=resolved_limit,
                )
                if not text:
                    continue
                raw_segment_count_estimate = 0
                try:
                    raw_segment_count_estimate = len(
                        self.tts.text_preprocessor.pre_seg_text(text, text_lang, text_split_method)
                    )
                except Exception:
                    raw_segment_count_estimate = max(0, resolved_limit)
                requests.append(
                    {
                        "request_id": f"startup_prewarm_{plan_index:02d}_{bucket_index:02d}",
                        "text": text,
                        "text_lang": text_lang,
                        "ref_audio_path": ref_audio_path,
                        "prompt_lang": prompt_lang,
                        "prompt_text": prompt_text,
                        "text_split_method": text_split_method,
                        "media_type": "wav",
                        "streaming_mode": False,
                        "parallel_infer": True,
                        "sample_steps": sample_steps,
                        "timeout_sec": timeout_sec,
                        "_warmup_meta": {
                            "text_path": str(text_path),
                            "unit_limit": int(resolved_limit),
                            "text_chars": int(len(text)),
                            "text_lang": text_lang,
                            "text_split_method": text_split_method,
                            "available_unit_count": int(available_units),
                            "raw_segment_count_estimate": int(raw_segment_count_estimate),
                            "bucket_spec": str(bucket_spec),
                            "warmup_mode": str(warmup_mode),
                        },
                    }
                )
        return requests

    def run_startup_prewarm(self) -> dict:
        enabled = self._env_flag("GPTSOVITS_STARTUP_PREWARM", True)
        summary = {
            "enabled": bool(enabled),
            "ran": False,
            "reason": "",
            "rounds": 0,
            "plan_count": 0,
            "total_ms": 0.0,
            "plans": [],
            "manifest_reused": False,
        }
        if not enabled:
            summary["reason"] = "disabled"
            self.startup_prewarm_summary = summary
            return dict(summary)

        requests = self._build_startup_prewarm_requests()
        if not requests:
            summary["reason"] = "no_valid_plan"
            self.startup_prewarm_summary = summary
            print("Startup prewarm skipped: no valid warmup request plan")
            return dict(summary)

        rounds = max(1, self._env_int("GPTSOVITS_STARTUP_PREWARM_ROUNDS", 1))
        strict = self._env_flag("GPTSOVITS_STARTUP_PREWARM_STRICT", False)
        signature = self._build_startup_prewarm_signature(requests, rounds)
        if self._env_flag("GPTSOVITS_STARTUP_PREWARM_SKIP_IF_MANIFEST_MATCH", True):
            cached_manifest = self._load_startup_prewarm_manifest(signature)
            if cached_manifest is not None:
                cached_summary = dict(cached_manifest.get("summary") or {})
                cached_summary["enabled"] = True
                cached_summary["ran"] = False
                cached_summary["reason"] = "reused_manifest"
                cached_summary["manifest_reused"] = True
                cached_summary["reused_total_ms"] = float((cached_manifest.get("summary") or {}).get("total_ms", 0.0))
                self.startup_prewarm_summary = cached_summary
                print(
                    "Startup prewarm skipped via manifest reuse: "
                    f"plans={cached_summary.get('plan_count', 0)} reused_total_ms="
                    f"{float(cached_summary.get('reused_total_ms', 0.0)):.2f}"
                )
                return dict(cached_summary)
        started_at = time.perf_counter()
        for warm_round in range(rounds):
            for plan_index, plan_template in enumerate(requests):
                req = {key: value for key, value in plan_template.items() if key != "_warmup_meta"}
                req["request_id"] = f"{req['request_id']}_r{warm_round:02d}"
                warmup_meta = dict(plan_template.get("_warmup_meta") or {})
                warmup_mode = str(warmup_meta.get("warmup_mode") or "full").strip().lower()
                plan_started_at = time.perf_counter()
                try:
                    if warmup_mode == "shape_only":
                        result = self.run_direct_tts_shape_prewarm(req)
                        audio_bytes = int(result.get("audio_bytes", 0) or 0)
                        extra_summary = {
                            key: value
                            for key, value in result.items()
                            if key not in {"audio_bytes"}
                        }
                    else:
                        result = self.run_direct_tts_startup_prewarm(req)
                        audio_bytes = 0 if result.audio_bytes is None else int(len(result.audio_bytes))
                        extra_summary = {}
                    plan_ms = max(0.0, (time.perf_counter() - plan_started_at) * 1000.0)
                    summary["plans"].append(
                        {
                            "round": int(warm_round),
                            "plan_index": int(plan_index),
                            "request_id": req["request_id"],
                            "ok": True,
                            "elapsed_ms": float(plan_ms),
                            "audio_bytes": int(audio_bytes),
                            **warmup_meta,
                            **extra_summary,
                        }
                    )
                except Exception as exc:
                    plan_ms = max(0.0, (time.perf_counter() - plan_started_at) * 1000.0)
                    plan_summary = {
                        "round": int(warm_round),
                        "plan_index": int(plan_index),
                        "request_id": req["request_id"],
                        "ok": False,
                        "elapsed_ms": float(plan_ms),
                        "error": str(exc),
                        **warmup_meta,
                    }
                    summary["plans"].append(plan_summary)
                    print("Startup prewarm failed:", plan_summary)
                    traceback.print_exc()
                    if strict:
                        summary["reason"] = "failed"
                        summary["rounds"] = int(rounds)
                        summary["plan_count"] = int(len(requests))
                        summary["total_ms"] = max(0.0, (time.perf_counter() - started_at) * 1000.0)
                        self.startup_prewarm_summary = summary
                        raise
        summary["ran"] = True
        summary["reason"] = "completed"
        summary["rounds"] = int(rounds)
        summary["plan_count"] = int(len(requests))
        summary["total_ms"] = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        summary["manifest_reused"] = False
        self.startup_prewarm_summary = summary
        try:
            self._write_startup_prewarm_manifest(signature=signature, summary=summary)
        except Exception as exc:
            print(f"Startup prewarm manifest write skipped: {exc}")
        print(
            "Startup prewarm completed: "
            f"rounds={summary['rounds']} plans={summary['plan_count']} total_ms={summary['total_ms']:.2f}"
        )
        return dict(summary)
