from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[4]
_VENDORED_GPT_SOVITS_RUNTIME_ROOT = (
    _REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "gpt_sovits" / "runtime_lib"
)


def _load_text_cpu_preprocess_module():
    runtime_root = str(_VENDORED_GPT_SOVITS_RUNTIME_ROOT)
    package_root = str(_VENDORED_GPT_SOVITS_RUNTIME_ROOT / "GPT_SoVITS")
    for candidate in (runtime_root, package_root):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    return importlib.import_module("TTS_infer_pack.text_cpu_preprocess")


def test_preprocess_text_segments_payload_auto_langsegment_compat_has_separate_cache(monkeypatch):
    module = _load_text_cpu_preprocess_module()
    module._PAYLOAD_CACHE.clear()
    monkeypatch.delenv("GPTSOVITS_PREPARE_TEXT_CPU_AUTO_LANGSEG_COMPAT", raising=False)
    monkeypatch.setenv("GPTSOVITS_PREPARE_TEXT_CPU_AUTO_FASTPATH", "1")
    monkeypatch.setattr(module.chinese2, "text_normalize", lambda text: f"ZH<{text}>")
    monkeypatch.setattr(
        module,
        "clean_text_segment",
        lambda text, language, version: ([101, 102], [1, 1], f"{language.upper()}<{text}>"),
    )
    monkeypatch.setattr(
        module.LangSegmenter,
        "getTexts",
        staticmethod(
            lambda text, *args: [
                {"lang": "zh", "text": "汉皇重色"},
                {"lang": "ja", "text": "思倾国，"},
            ]
        ),
    )

    fast_payloads = module.preprocess_text_segments_payload("汉皇重色思倾国，", "auto", "v2")
    assert [payload["language"] for payload in fast_payloads] == ["zh"]
    assert fast_payloads[0]["needs_g2pw"] is True

    monkeypatch.setenv("GPTSOVITS_PREPARE_TEXT_CPU_AUTO_LANGSEG_COMPAT", "1")
    compat_payloads = module.preprocess_text_segments_payload("汉皇重色思倾国，", "auto", "v2")
    assert [payload["language"] for payload in compat_payloads] == ["zh", "ja"]
    assert compat_payloads[0]["needs_g2pw"] is True
    assert compat_payloads[1]["needs_g2pw"] is False


def test_split_texts_by_language_batch_auto_langsegment_compat_bypasses_fast_paths(monkeypatch):
    module = _load_text_cpu_preprocess_module()
    module._PAYLOAD_CACHE.clear()
    monkeypatch.setenv("GPTSOVITS_PREPARE_TEXT_CPU_AUTO_LANGSEG_COMPAT", "1")
    monkeypatch.setattr(
        module,
        "_split_texts_by_language_batch_selective_direct_runs",
        lambda texts, language: (_ for _ in ()).throw(AssertionError("selective fast path should be bypassed")),
    )
    monkeypatch.setattr(
        module,
        "_split_texts_by_language_batch_auto_whitespace_mixed",
        lambda texts, language: (_ for _ in ()).throw(AssertionError("whitespace fast path should be bypassed")),
    )
    monkeypatch.setattr(
        module.LangSegmenter,
        "getTextsBatch",
        staticmethod(
            lambda texts, *args: [
                [
                    {"lang": "zh", "text": "汉皇重色"},
                    {"lang": "ja", "text": "思倾国，"},
                ],
                [
                    {"lang": "zh", "text": "六宫粉黛无"},
                    {"lang": "zh", "text": "颜色。"},
                ],
            ]
        ),
    )

    results = module.split_texts_by_language_batch(
        ["汉皇重色思倾国，", "六宫粉黛无颜色。"],
        "auto",
    )

    assert results == [
        (["汉皇重色", "思倾国，"], ["zh", "ja"]),
        (["六宫粉黛无颜色。"], ["zh"]),
    ]
