from __future__ import annotations

import time
from typing import Dict, List, Tuple

from pypinyin import Style, pinyin


class G2PMConverter:
    """Compatibility stub for a future g2pm backend.

    This adapter preserves the existing g2pw-compatible contract used by
    `chinese2.py`: batch input is a list of sentences and output is a
    per-sentence, per-character pinyin list in tone3 form.

    The current implementation intentionally uses `pypinyin` as a placeholder
    so the rest of the runtime can be wired up and benchmarked before a real
    g2pm model/runtime is integrated.
    """

    def __init__(
        self,
        model_dir: str = "G2PWModel/",
        style: str = "pinyin",
        model_source: str | None = None,
        enable_non_tradional_chinese: bool = False,
    ) -> None:
        del model_dir, model_source, enable_non_tradional_chinese
        if style != "pinyin":
            raise ValueError(f"G2PMConverter currently only supports style='pinyin', got {style!r}")
        self.backend = "g2pm"
        self.providers = ["g2pm-stub-pypinyin"]
        self._prewarmed = False

    def _predict_impl(self, sentences: List[str]) -> List[List[str]]:
        results: List[List[str]] = []
        for sent in sentences:
            tone3 = pinyin(sent, neutral_tone_with_five=True, style=Style.TONE3)
            results.append([item[0] if item else "" for item in tone3])
        return results

    def __call__(self, sentences: List[str]) -> List[List[str]]:
        if isinstance(sentences, str):
            sentences = [sentences]
        return self._predict_impl(list(sentences))

    def predict_sentences_with_profile(self, sentences: List[str]) -> Tuple[List[List[str]], Dict[str, float]]:
        if isinstance(sentences, str):
            sentences = [sentences]
        started = time.perf_counter()
        results = self._predict_impl(list(sentences))
        elapsed_ms = float((time.perf_counter() - started) * 1000.0)
        return results, {
            "g2pw_predict_ms": elapsed_ms,
            "g2pw_runtime_total_ms": elapsed_ms,
        }

    def prewarm(self, sentences=None, rounds: int = 1) -> bool:
        warm_sentences = list(sentences or ["重庆银行的行长在长安见到了重要的人。"])
        warm_sentences = [str(item).strip() for item in warm_sentences if str(item).strip()]
        if not warm_sentences:
            warm_sentences = ["重庆银行的行长在长安见到了重要的人。"]
        for _ in range(max(1, int(rounds))):
            self._predict_impl(warm_sentences)
        self._prewarmed = True
        return True

    def snapshot(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "providers": list(self.providers),
            "prewarmed": bool(self._prewarmed),
        }
