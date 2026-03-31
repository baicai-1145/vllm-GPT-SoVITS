import os
import re
import sys
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

now_dir = os.getcwd()
sys.path.append(now_dir)

from text.LangSegmenter import LangSegmenter
from text.split_fastpath_native import scan_selective_direct_runs as scan_selective_direct_runs_native
from text import cleaned_text_to_sequence
from text import chinese2
from text.cleaner import clean_text, clean_text_batch


PreparedTextSegmentPayload = Dict[str, object]
PreparedTextSegmentBatchItem = Tuple[str, str, str, bool]
_PayloadCacheKey = Tuple[str, str, str, bool, str]
_SegmentJob = Tuple[int, str, str, str]
_MULTISPACE_PATTERN = re.compile(r" {2,}")
_AUTO_ZH_FASTPATH_ALLOWED_PATTERN = re.compile(r"^[\u4e00-\u9fff0-9\s、，。！？,.!?…：；\-—~～/·]+$")
_AUTO_EN_FASTPATH_PATTERN = re.compile(
    r"^(?=.*[A-Za-z])[A-Za-z0-9\s\u0020-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]+$"
)
_AUTO_ZH_FASTPATH_LATIN_PATTERN = re.compile(r"[A-Za-z\uff21-\uff3a\uff41-\uff5a]")
_AUTO_ZH_FASTPATH_JAKO_PATTERN = re.compile(r"[\u3040-\u30ff\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
_AUTO_JA_FASTPATH_ALLOWED_PATTERN = re.compile(
    r"^[\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff66-\uff9d0-9\s、，。！？,.!?…：；\-—~～/·]+$"
)
_AUTO_JA_DISTINCTIVE_PATTERN = re.compile(r"[\u3040-\u30ff\uff66-\uff9d]")
_AUTO_KO_DISTINCTIVE_PATTERN = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
_YUE_FASTPATH_ALLOWED_PATTERN = re.compile(r"^[\u4e00-\u9fff0-9\s、，。！？,.!?…：；\-—~～/·]+$")
_JA_FASTPATH_ALLOWED_PATTERN = re.compile(
    r"^[\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d0-9\s、，。！？,.!?…：；\-—~～/·]+$"
)
_KO_FASTPATH_ALLOWED_PATTERN = re.compile(r"^[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af0-9\s、，。！？,.!?…：；\-—~～/·]+$")
_DIRECT_FASTPATH_LATIN_PATTERN = re.compile(r"[A-Za-z\uff21-\uff3a\uff41-\uff5a]")
_WHITESPACE_TOKEN_PATTERN = re.compile(r"\S+\s*")
_TOKEN_STRIP_PUNCT_PATTERN = re.compile(r"^[、，。！？,.!?…：；\-—~～/·]+|[、，。！？,.!?…：；\-—~～/·]+$")
_TOKEN_HAS_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_TRIVIAL_BRIDGE_PATTERN = re.compile(r"^[0-9\s、，。！？,.!?…：；\-—~～/·]*$")
_ASCII_TRIVIAL_BRIDGE_CHARS = frozenset(",.!?-/~")
_UNICODE_TRIVIAL_BRIDGE_CHARS = frozenset("、，。！？…：；—～/·")
_PAYLOAD_CACHE_LOCK = threading.Lock()
_PAYLOAD_CACHE: "OrderedDict[_PayloadCacheKey, List[PreparedTextSegmentPayload]]" = OrderedDict()


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


_PAYLOAD_CACHE_MAX_ITEMS = max(0, int(os.environ.get("GPTSOVITS_PREPARE_TEXT_CPU_CACHE_MAX_ITEMS", "4096")))
_PAYLOAD_CACHE_ENABLED = _PAYLOAD_CACHE_MAX_ITEMS > 0 and _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_CACHE", True)
_NATIVE_SPAN_AMBIGUOUS = 0
_NATIVE_SPAN_EN = 2
_NATIVE_SPAN_KO = 3
_NATIVE_SPAN_JA = 4


def _normalize_spaces(text: str) -> str:
    return _MULTISPACE_PATTERN.sub(" ", str(text))


def _is_direct_zh_fast_path(language: str) -> bool:
    return _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_ZH_FASTPATH", True) and str(language) in {"zh", "all_zh"}


def _use_auto_langsegment_compat(language: str) -> bool:
    return str(language) in {"auto", "auto_yue"} and _env_flag(
        "GPTSOVITS_PREPARE_TEXT_CPU_AUTO_LANGSEG_COMPAT",
        False,
    )


def _payload_cache_mode_tag(language: str) -> str:
    return "|".join(
        (
            f"compat={int(_use_auto_langsegment_compat(language))}",
            f"zh={int(_env_flag('GPTSOVITS_PREPARE_TEXT_CPU_ZH_FASTPATH', True))}",
            f"yue={int(_env_flag('GPTSOVITS_PREPARE_TEXT_CPU_YUE_FASTPATH', True))}",
            f"ja={int(_env_flag('GPTSOVITS_PREPARE_TEXT_CPU_JA_FASTPATH', True))}",
            f"ko={int(_env_flag('GPTSOVITS_PREPARE_TEXT_CPU_KO_FASTPATH', True))}",
            f"auto_zh={int(_env_flag('GPTSOVITS_PREPARE_TEXT_CPU_AUTO_ZH_FASTPATH', False))}",
            f"auto={int(_env_flag('GPTSOVITS_PREPARE_TEXT_CPU_AUTO_FASTPATH', True))}",
        )
    )


def _payload_cache_key(text: str, language: str, version: str, final: bool) -> _PayloadCacheKey:
    return (
        _normalize_spaces(str(text)),
        str(language),
        str(version),
        bool(final),
        _payload_cache_mode_tag(str(language)),
    )


def _get_direct_language_fast_path(language: str) -> str | None:
    normalized = str(language)
    if normalized in {"all_yue"} and _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_YUE_FASTPATH", True):
        return "yue"
    if normalized in {"all_ja"} and _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_JA_FASTPATH", True):
        return "ja"
    if normalized in {"all_ko"} and _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_KO_FASTPATH", True):
        return "ko"
    return None


def _can_use_direct_language_fast_path(text: str, language: str) -> str | None:
    target_language = _get_direct_language_fast_path(language)
    if target_language is None or not text:
        return None
    if target_language == "yue":
        if _YUE_FASTPATH_ALLOWED_PATTERN.fullmatch(text) and not _DIRECT_FASTPATH_LATIN_PATTERN.search(text):
            return target_language
        return None
    if target_language == "ja":
        if _JA_FASTPATH_ALLOWED_PATTERN.fullmatch(text) and not _DIRECT_FASTPATH_LATIN_PATTERN.search(text):
            return target_language
        return None
    if target_language == "ko":
        if _KO_FASTPATH_ALLOWED_PATTERN.fullmatch(text) and not _DIRECT_FASTPATH_LATIN_PATTERN.search(text):
            return target_language
        return None
    return None


def _is_auto_zh_fast_path(text: str, language: str) -> bool:
    if str(language) not in {"auto", "auto_yue"}:
        return False
    if _use_auto_langsegment_compat(language):
        return False
    if not _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_AUTO_ZH_FASTPATH", False):
        return False
    if not text or not _AUTO_ZH_FASTPATH_ALLOWED_PATTERN.fullmatch(text):
        return False
    if _AUTO_ZH_FASTPATH_LATIN_PATTERN.search(text) or _AUTO_ZH_FASTPATH_JAKO_PATTERN.search(text):
        return False
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk_count > 0


def _get_auto_single_language_fast_path(text: str, language: str) -> str | None:
    normalized = str(language)
    if normalized not in {"auto", "auto_yue"}:
        return None
    if _use_auto_langsegment_compat(language):
        return None
    if not _env_flag("GPTSOVITS_PREPARE_TEXT_CPU_AUTO_FASTPATH", True):
        return None
    if not text:
        return None
    if _AUTO_EN_FASTPATH_PATTERN.fullmatch(text):
        return "en"
    if _KO_FASTPATH_ALLOWED_PATTERN.fullmatch(text) and _AUTO_KO_DISTINCTIVE_PATTERN.search(text):
        return "ko"
    if (
        _AUTO_JA_FASTPATH_ALLOWED_PATTERN.fullmatch(text)
        and _AUTO_JA_DISTINCTIVE_PATTERN.search(text)
        and not _TOKEN_HAS_CJK_PATTERN.search(text)
    ):
        return "ja"
    if _AUTO_ZH_FASTPATH_ALLOWED_PATTERN.fullmatch(text):
        if _AUTO_ZH_FASTPATH_LATIN_PATTERN.search(text) or _AUTO_ZH_FASTPATH_JAKO_PATTERN.search(text):
            return None
        cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        if cjk_count > 0:
            return "yue" if normalized == "auto_yue" else "zh"
    return None


def _should_use_zh_fast_path(text: str, language: str) -> bool:
    return _is_direct_zh_fast_path(language) or _is_auto_zh_fast_path(text, language)


def _classify_whitespace_token(text: str, language: str) -> str | None:
    stripped = _TOKEN_STRIP_PUNCT_PATTERN.sub("", str(text).strip())
    if not stripped:
        return None
    if _KO_FASTPATH_ALLOWED_PATTERN.fullmatch(stripped) and _AUTO_KO_DISTINCTIVE_PATTERN.search(stripped):
        return "ko"
    if (
        _AUTO_JA_FASTPATH_ALLOWED_PATTERN.fullmatch(stripped)
        and _AUTO_JA_DISTINCTIVE_PATTERN.search(stripped)
        and not _TOKEN_HAS_CJK_PATTERN.search(stripped)
    ):
        return "ja"
    if _AUTO_EN_FASTPATH_PATTERN.fullmatch(stripped):
        return "en"
    return None


def _try_whitespace_mixed_fast_path(text: str, language: str) -> Tuple[List[str], List[str]] | None:
    if str(language) not in {"auto", "auto_yue"}:
        return None
    token_matches = list(_WHITESPACE_TOKEN_PATTERN.finditer(str(text)))
    if len(token_matches) <= 1:
        return None
    textlist: List[str] = []
    langlist: List[str] = []
    for match in token_matches:
        chunk = match.group(0)
        detected_lang = _classify_whitespace_token(chunk, language)
        if detected_lang is None:
            return None
        if langlist and langlist[-1] == detected_lang:
            textlist[-1] += chunk
            continue
        textlist.append(chunk)
        langlist.append(detected_lang)
    return textlist, langlist


def _is_ascii_or_fullwidth_latin(ch: str) -> bool:
    codepoint = ord(ch)
    return (
        0x41 <= codepoint <= 0x5A
        or 0x61 <= codepoint <= 0x7A
        or 0xFF21 <= codepoint <= 0xFF3A
        or 0xFF41 <= codepoint <= 0xFF5A
    )


def _is_korean_char(ch: str) -> bool:
    codepoint = ord(ch)
    return (
        0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def _is_kana_char(ch: str) -> bool:
    codepoint = ord(ch)
    return 0x3040 <= codepoint <= 0x30FF or 0xFF66 <= codepoint <= 0xFF9D


def _is_han_char(ch: str) -> bool:
    codepoint = ord(ch)
    return 0x4E00 <= codepoint <= 0x9FFF or codepoint == 0x3005


def _is_trivial_bridge_char(ch: str) -> bool:
    if not ch:
        return False
    codepoint = ord(ch)
    return (
        0x30 <= codepoint <= 0x39
        or ch.isspace()
        or ch in _ASCII_TRIVIAL_BRIDGE_CHARS
        or ch in _UNICODE_TRIVIAL_BRIDGE_CHARS
    )


def _consume_direct_script_run(text: str, start: int, script: str) -> int:
    text_length = len(text)
    cursor = start

    if script == "en":
        def _is_core_char(value: str) -> bool:
            return _is_ascii_or_fullwidth_latin(value) or value.isdigit() or value in {"'", "_", "-"}
    elif script == "ko":
        def _is_core_char(value: str) -> bool:
            return _is_korean_char(value)
    else:
        def _is_core_char(value: str) -> bool:
            return _is_kana_char(value)

    while cursor < text_length and _is_core_char(text[cursor]):
        cursor += 1
    while cursor < text_length:
        bridge_start = cursor
        while cursor < text_length and _is_trivial_bridge_char(text[cursor]):
            cursor += 1
        if cursor >= text_length or not _is_core_char(text[cursor]):
            return bridge_start if bridge_start > start else cursor
        while cursor < text_length and _is_core_char(text[cursor]):
            cursor += 1
    return cursor


def _consume_cjk_block(text: str, start: int) -> Tuple[int, bool, bool]:
    text_length = len(text)
    cursor = start
    saw_kana = False
    saw_han = False
    last_core_end = start
    while cursor < text_length:
        current = text[cursor]
        if _is_han_char(current):
            saw_han = True
            cursor += 1
            last_core_end = cursor
            continue
        if _is_kana_char(current):
            saw_kana = True
            cursor += 1
            last_core_end = cursor
            continue
        if _is_trivial_bridge_char(current):
            cursor += 1
            continue
        break
    return max(last_core_end, start + 1), saw_kana, saw_han


def _consume_misc_ambiguous_block(text: str, start: int) -> int:
    text_length = len(text)
    cursor = start
    last_core_end = start
    while cursor < text_length:
        current = text[cursor]
        if (
            _is_ascii_or_fullwidth_latin(current)
            or _is_korean_char(current)
            or _is_kana_char(current)
            or _is_han_char(current)
        ):
            break
        cursor += 1
        if not _is_trivial_bridge_char(current):
            last_core_end = cursor
    return max(last_core_end, start + 1)


def _merge_direct_and_resolved_specs(
    direct_specs: Sequence[Tuple[str, str] | None],
    resolved_specs: Sequence[List[Tuple[str, str]]],
) -> Tuple[List[str], List[str]]:
    merged_specs: List[Tuple[str, str]] = []
    resolved_index = 0
    for spec in direct_specs:
        if spec is None:
            merged_specs.extend(resolved_specs[resolved_index])
            resolved_index += 1
            continue
        merged_specs.append(spec)
    return _merge_segment_specs(merged_specs)


def _resolve_selective_direct_native_rows(
    texts: Sequence[str],
    native_rows: Sequence[Sequence[Tuple[str, int]]],
    language: str,
) -> List[Tuple[List[str], List[str]]]:
    per_text_specs: List[List[Tuple[str, str] | None]] = []
    ambiguous_chunks: List[str] = []
    lang_map = {
        _NATIVE_SPAN_EN: "en",
        _NATIVE_SPAN_KO: "ko",
        _NATIVE_SPAN_JA: "ja",
    }

    for _text, row in zip(texts, native_rows):
        specs: List[Tuple[str, str] | None] = []
        for chunk, span_type in row:
            if int(span_type) == _NATIVE_SPAN_AMBIGUOUS:
                chunk_text = str(chunk)
                specs.append(None)
                ambiguous_chunks.append(chunk_text)
            else:
                specs.append((str(chunk), lang_map[int(span_type)]))
        per_text_specs.append(specs)

    resolved_specs: List[List[Tuple[str, str]]] = []
    if ambiguous_chunks:
        ambiguous_rows = LangSegmenter.getTextsBatch(ambiguous_chunks)
        for items in ambiguous_rows:
            textlist, langlist = _langsegmenter_items_to_segment_lists(items, language)
            resolved_specs.append(list(zip(textlist, langlist)))

    results: List[Tuple[List[str], List[str]]] = []
    resolved_offset = 0
    for specs in per_text_specs:
        ambiguous_count = sum(1 for spec in specs if spec is None)
        resolved_slice = resolved_specs[resolved_offset : resolved_offset + ambiguous_count]
        resolved_offset += ambiguous_count
        results.append(_merge_direct_and_resolved_specs(specs, resolved_slice))
    return results


def _split_texts_by_language_batch_selective_direct_runs(
    texts: Sequence[str],
    language: str,
) -> List[Tuple[List[str], List[str]]] | None:
    if str(language) not in {"auto", "auto_yue"}:
        return None
    native_rows = scan_selective_direct_runs_native(texts)
    if native_rows is not None:
        return _resolve_selective_direct_native_rows(texts, native_rows, language)
    per_text_specs: List[List[Tuple[str, str] | None]] = []
    ambiguous_chunks: List[str] = []
    has_direct_run = False
    ascii_trivial_bridge_chars = _ASCII_TRIVIAL_BRIDGE_CHARS
    unicode_trivial_bridge_chars = _UNICODE_TRIVIAL_BRIDGE_CHARS
    bridge_type = 1
    en_type = 2
    ko_type = 3
    kana_type = 4
    han_type = 5

    for raw_text in texts:
        text = str(raw_text)
        specs: List[Tuple[str, str] | None] = []
        cursor = 0
        text_length = len(text)
        char_types = bytearray(text_length)
        for index, char in enumerate(text):
            codepoint = ord(char)
            if (
                0x30 <= codepoint <= 0x39
                or char.isspace()
                or char in ascii_trivial_bridge_chars
                or char in unicode_trivial_bridge_chars
            ):
                char_types[index] = bridge_type
                continue
            if (
                0x41 <= codepoint <= 0x5A
                or 0x61 <= codepoint <= 0x7A
                or 0xFF21 <= codepoint <= 0xFF3A
                or 0xFF41 <= codepoint <= 0xFF5A
            ):
                char_types[index] = en_type
                continue
            if (
                0x1100 <= codepoint <= 0x11FF
                or 0x3130 <= codepoint <= 0x318F
                or 0xAC00 <= codepoint <= 0xD7AF
            ):
                char_types[index] = ko_type
                continue
            if 0x3040 <= codepoint <= 0x30FF or 0xFF66 <= codepoint <= 0xFF9D:
                char_types[index] = kana_type
                continue
            if 0x4E00 <= codepoint <= 0x9FFF or codepoint == 0x3005:
                char_types[index] = han_type

        pending_bridge_start = -1
        while cursor < text_length:
            current = text[cursor]
            current_type = char_types[cursor]
            if current_type == bridge_type:
                if pending_bridge_start < 0:
                    pending_bridge_start = cursor
                cursor += 1
                continue

            segment_start = pending_bridge_start if pending_bridge_start >= 0 else cursor
            pending_bridge_start = -1

            if current_type == en_type:
                cursor += 1
                while cursor < text_length:
                    next_char = text[cursor]
                    next_type = char_types[cursor]
                    if next_type == en_type or next_char.isdigit() or next_char in {"'", "_", "-"}:
                        cursor += 1
                        continue
                    bridge_start = cursor
                    while cursor < text_length and char_types[cursor] == bridge_type:
                        cursor += 1
                    if cursor >= text_length:
                        cursor = bridge_start
                        break
                    next_char = text[cursor]
                    next_type = char_types[cursor]
                    if not (next_type == en_type or next_char.isdigit() or next_char in {"'", "_", "-"}):
                        cursor = bridge_start
                        break
                specs.append((text[segment_start:cursor], "en"))
                has_direct_run = True
                continue

            if current_type == ko_type:
                cursor += 1
                while cursor < text_length:
                    if char_types[cursor] == ko_type:
                        cursor += 1
                        continue
                    bridge_start = cursor
                    while cursor < text_length and char_types[cursor] == bridge_type:
                        cursor += 1
                    if cursor >= text_length:
                        cursor = bridge_start
                        break
                    if char_types[cursor] != ko_type:
                        cursor = bridge_start
                        break
                specs.append((text[segment_start:cursor], "ko"))
                has_direct_run = True
                continue

            if current_type == kana_type or current_type == han_type:
                saw_kana = current_type == kana_type
                saw_han = current_type == han_type
                cursor += 1
                last_core_end = cursor
                while cursor < text_length:
                    next_type = char_types[cursor]
                    if next_type == han_type:
                        saw_han = True
                        cursor += 1
                        last_core_end = cursor
                        continue
                    if next_type == kana_type:
                        saw_kana = True
                        cursor += 1
                        last_core_end = cursor
                        continue
                    if next_type == bridge_type:
                        cursor += 1
                        continue
                    break
                chunk_end = last_core_end
                chunk = text[segment_start:chunk_end]
                if saw_kana and not saw_han:
                    specs.append((chunk, "ja"))
                    has_direct_run = True
                else:
                    specs.append(None)
                    ambiguous_chunks.append(chunk)
                cursor = chunk_end
                continue

            cursor += 1
            last_core_end = cursor
            while cursor < text_length:
                next_type = char_types[cursor]
                if next_type != 0 and next_type != bridge_type:
                    break
                cursor += 1
                if next_type != bridge_type:
                    last_core_end = cursor
            chunk_end = last_core_end
            chunk = text[segment_start:chunk_end]
            specs.append(None)
            ambiguous_chunks.append(chunk)
            cursor = chunk_end

        if pending_bridge_start >= 0:
            pending_bridge = text[pending_bridge_start:text_length]
            if specs:
                last_spec = specs[-1]
                if last_spec is None:
                    ambiguous_chunks[-1] += pending_bridge
                else:
                    last_text, last_lang = last_spec
                    specs[-1] = (last_text + pending_bridge, last_lang)
            else:
                specs.append(None)
                ambiguous_chunks.append(pending_bridge)
        per_text_specs.append(specs)

    if not has_direct_run:
        return None

    resolved_specs: List[List[Tuple[str, str]]] = []
    if ambiguous_chunks:
        ambiguous_rows = LangSegmenter.getTextsBatch(ambiguous_chunks)
        for items in ambiguous_rows:
            textlist, langlist = _langsegmenter_items_to_segment_lists(items, language)
            resolved_specs.append(list(zip(textlist, langlist)))

    results: List[Tuple[List[str], List[str]]] = []
    resolved_offset = 0
    for specs in per_text_specs:
        ambiguous_count = sum(1 for spec in specs if spec is None)
        resolved_slice = resolved_specs[resolved_offset : resolved_offset + ambiguous_count]
        resolved_offset += ambiguous_count
        results.append(_merge_direct_and_resolved_specs(specs, resolved_slice))
    return results


def _merge_segment_specs(
    specs: Sequence[Tuple[str, str]],
) -> Tuple[List[str], List[str]]:
    textlist: List[str] = []
    langlist: List[str] = []
    for text, lang in specs:
        if not text:
            continue
        if langlist and langlist[-1] == lang:
            textlist[-1] += text
            continue
        textlist.append(text)
        langlist.append(lang)
    return textlist, langlist


def _langsegmenter_items_to_segment_lists(
    items: Sequence[Dict[str, str]],
    language: str,
) -> Tuple[List[str], List[str]]:
    textlist: List[str] = []
    langlist: List[str] = []
    normalized_language = str(language)
    pending_prefix = ""
    for item in items:
        item_lang = str(item["lang"])
        item_text = str(item["text"])
        if item_lang in {"punctuation", "newline"}:
            if textlist:
                textlist[-1] += item_text
            else:
                pending_prefix += item_text
            continue
        if normalized_language == "auto_yue" and item_lang == "zh":
            item_lang = "yue"
        if pending_prefix:
            item_text = pending_prefix + item_text
            pending_prefix = ""
        if langlist and langlist[-1] == item_lang:
            textlist[-1] += item_text
            continue
        textlist.append(item_text)
        langlist.append(item_lang)
    if pending_prefix:
        if textlist:
            textlist[-1] += pending_prefix
        else:
            fallback_lang = "yue" if normalized_language == "auto_yue" else "zh"
            textlist.append(pending_prefix)
            langlist.append(fallback_lang)
    return textlist, langlist


def _split_texts_by_language_batch_auto_whitespace_mixed(
    texts: Sequence[str],
    language: str,
) -> List[Tuple[List[str], List[str]]] | None:
    if str(language) not in {"auto", "auto_yue"}:
        return None
    per_text_specs: List[List[Tuple[str, str] | None]] = []
    ambiguous_chunks: List[str] = []
    ambiguous_targets: List[Tuple[int, int]] = []
    has_any_direct_token = False

    for text in texts:
        token_matches = list(_WHITESPACE_TOKEN_PATTERN.finditer(str(text)))
        if len(token_matches) <= 1:
            return None
        specs: List[Tuple[str, str] | None] = []
        pending_ambiguous = ""
        direct_token_count = 0
        for match in token_matches:
            chunk = match.group(0)
            detected_lang = _classify_whitespace_token(chunk, language)
            if detected_lang is None:
                pending_ambiguous += chunk
                continue
            direct_token_count += 1
            has_any_direct_token = True
            if pending_ambiguous:
                specs.append(None)
                ambiguous_chunks.append(pending_ambiguous)
                ambiguous_targets.append((len(per_text_specs), len(specs) - 1))
                pending_ambiguous = ""
            specs.append((chunk, detected_lang))
        if pending_ambiguous:
            specs.append(None)
            ambiguous_chunks.append(pending_ambiguous)
            ambiguous_targets.append((len(per_text_specs), len(specs) - 1))
        if direct_token_count == 0:
            return None
        per_text_specs.append(specs)

    if not has_any_direct_token:
        return None

    ambiguous_rows = LangSegmenter.getTextsBatch(ambiguous_chunks)
    for (text_index, spec_index), items in zip(ambiguous_targets, ambiguous_rows):
        textlist, langlist = _langsegmenter_items_to_segment_lists(items, language)
        per_text_specs[text_index][spec_index] = ("\0".join(textlist), "\0".join(langlist))

    results: List[Tuple[List[str], List[str]]] = []
    for specs in per_text_specs:
        expanded_specs: List[Tuple[str, str]] = []
        for spec in specs:
            assert spec is not None
            text_value, lang_value = spec
            if "\0" in text_value or "\0" in lang_value:
                text_parts = text_value.split("\0")
                lang_parts = lang_value.split("\0")
                expanded_specs.extend(zip(text_parts, lang_parts))
            else:
                expanded_specs.append((text_value, lang_value))
        results.append(_merge_segment_specs(expanded_specs))
    return results


def _build_zh_fast_path_payload(norm_text: str) -> List[PreparedTextSegmentPayload]:
    return [
        {
            "language": "zh",
            "phones": [],
            "word2ph": None,
            "norm_text": str(norm_text),
            "needs_g2pw": True,
        }
    ]


def _build_direct_language_payload(
    text: str,
    language: str,
    version: str,
) -> List[PreparedTextSegmentPayload]:
    phones, word2ph, norm_text = clean_text_segment(text, language, version)
    return [
        {
            "language": str(language).replace("all_", ""),
            "phones": phones,
            "word2ph": word2ph,
            "norm_text": norm_text,
            "needs_g2pw": False,
        }
    ]


def _estimate_payload_phones_len(payloads: Sequence[PreparedTextSegmentPayload]) -> int:
    total_phones_len = 0
    for payload in payloads:
        if bool(payload.get("needs_g2pw", False)):
            total_phones_len += max(0, len(str(payload.get("norm_text", ""))) * 2)
            continue
        total_phones_len += len(payload.get("phones", []))
    return int(total_phones_len)


def _build_segment_payload(
    *,
    language: str,
    phones: Sequence[int] | None,
    word2ph: Sequence[int] | None,
    norm_text: str,
    needs_g2pw: bool,
) -> PreparedTextSegmentPayload:
    return {
        "language": str(language),
        "phones": [] if phones is None else list(phones),
        "word2ph": None if word2ph is None else list(word2ph),
        "norm_text": str(norm_text),
        "needs_g2pw": bool(needs_g2pw),
    }


def _clone_payloads(payloads: Sequence[PreparedTextSegmentPayload]) -> List[PreparedTextSegmentPayload]:
    return [
        {
            "language": str(payload["language"]),
            "phones": list(payload["phones"]),
            "word2ph": None if payload["word2ph"] is None else list(payload["word2ph"]),
            "norm_text": str(payload["norm_text"]),
            "needs_g2pw": bool(payload.get("needs_g2pw", False)),
        }
        for payload in payloads
    ]


def _cache_get_payloads(item: PreparedTextSegmentBatchItem) -> List[PreparedTextSegmentPayload] | None:
    if not _PAYLOAD_CACHE_ENABLED:
        return None
    cache_key = _payload_cache_key(*item)
    with _PAYLOAD_CACHE_LOCK:
        cached = _PAYLOAD_CACHE.get(cache_key)
        if cached is None:
            return None
        _PAYLOAD_CACHE.move_to_end(cache_key)
        return _clone_payloads(cached)


def _cache_store_payloads(
    item: PreparedTextSegmentBatchItem,
    payloads: Sequence[PreparedTextSegmentPayload],
) -> None:
    if not _PAYLOAD_CACHE_ENABLED:
        return
    cached_payloads = _clone_payloads(payloads)
    cache_key = _payload_cache_key(*item)
    with _PAYLOAD_CACHE_LOCK:
        _PAYLOAD_CACHE[cache_key] = cached_payloads
        _PAYLOAD_CACHE.move_to_end(cache_key)
        while len(_PAYLOAD_CACHE) > _PAYLOAD_CACHE_MAX_ITEMS:
            _PAYLOAD_CACHE.popitem(last=False)


def _build_nonzh_segment_payloads_batch(
    jobs: Sequence[_SegmentJob],
) -> Dict[int, PreparedTextSegmentPayload]:
    payloads_by_index: Dict[int, PreparedTextSegmentPayload] = {}
    if not jobs:
        return payloads_by_index
    texts = [segment_text for _segment_index, segment_text, _segment_lang, _version in jobs]
    segment_lang = str(jobs[0][2])
    version = str(jobs[0][3])
    rows = clean_text_batch(texts, segment_lang, version)
    for (segment_index, _segment_text, segment_lang, version), (phones, word2ph, norm_text) in zip(jobs, rows):
        payloads_by_index[segment_index] = _build_segment_payload(
            language=segment_lang.replace("all_", ""),
            phones=cleaned_text_to_sequence(phones, version),
            word2ph=word2ph,
            norm_text=norm_text,
            needs_g2pw=False,
        )
    return payloads_by_index


def _build_zh_segment_payloads_batch(
    jobs: Sequence[_SegmentJob],
) -> Dict[int, PreparedTextSegmentPayload]:
    payloads_by_index: Dict[int, PreparedTextSegmentPayload] = {}
    if not jobs:
        return payloads_by_index
    norm_texts = chinese2.text_normalize_batch([segment_text for _, segment_text, _, _ in jobs])
    for (segment_index, _segment_text, _segment_lang, _version), norm_text in zip(jobs, norm_texts):
        payloads_by_index[segment_index] = _build_segment_payload(
            language="zh",
            phones=[],
            word2ph=None,
            norm_text=str(norm_text),
            needs_g2pw=True,
        )
    return payloads_by_index


def _build_segment_payloads_batch(
    jobs_by_language: Dict[Tuple[str, str], List[_SegmentJob]],
) -> Dict[int, PreparedTextSegmentPayload]:
    payloads_by_index: Dict[int, PreparedTextSegmentPayload] = {}
    for (normalized_language, _version), jobs in jobs_by_language.items():
        if normalized_language == "zh":
            payloads_by_index.update(_build_zh_segment_payloads_batch(jobs))
            continue
        payloads_by_index.update(_build_nonzh_segment_payloads_batch(jobs))
    return payloads_by_index


def split_text_by_language(text: str, language: str) -> Tuple[List[str], List[str]]:
    if _use_auto_langsegment_compat(language):
        return _langsegmenter_items_to_segment_lists(LangSegmenter.getTexts(text), str(language))
    if _should_use_zh_fast_path(text, language):
        return [text], ["zh"]
    auto_language = _get_auto_single_language_fast_path(text, language)
    if auto_language is not None:
        return [text], [auto_language]
    direct_language = _can_use_direct_language_fast_path(text, language)
    if direct_language is not None:
        return [text], [direct_language]
    if str(language) in {"auto", "auto_yue"}:
        selective_direct_result = _split_texts_by_language_batch_selective_direct_runs([text], str(language))
        if selective_direct_result is not None:
            return selective_direct_result[0]
    whitespace_fast_path = _try_whitespace_mixed_fast_path(text, language)
    if whitespace_fast_path is not None:
        return whitespace_fast_path
    textlist: List[str] = []
    langlist: List[str] = []
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text, "zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_yue":
        for tmp in LangSegmenter.getTexts(text, "zh"):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ja":
        for tmp in LangSegmenter.getTexts(text, "ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text, "ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "auto_yue":
        for tmp in LangSegmenter.getTexts(text):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                same_group = (tmp["lang"] == "en" and langlist[-1] == "en") or (
                    tmp["lang"] != "en" and langlist[-1] != "en"
                )
                if same_group:
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                langlist.append(language)
            textlist.append(tmp["text"])
    return textlist, langlist


def split_texts_by_language_batch(
    texts: Sequence[str],
    language: str,
) -> List[Tuple[List[str], List[str]]]:
    normalized_language = str(language)
    if not texts:
        return []
    if _use_auto_langsegment_compat(normalized_language):
        return [
            _langsegmenter_items_to_segment_lists(items, normalized_language)
            for items in LangSegmenter.getTextsBatch(texts)
        ]
    if normalized_language in {"auto", "auto_yue"}:
        selective_direct_results = _split_texts_by_language_batch_selective_direct_runs(texts, normalized_language)
        if selective_direct_results is not None:
            return selective_direct_results
        mixed_fast_path_results = _split_texts_by_language_batch_auto_whitespace_mixed(texts, normalized_language)
        if mixed_fast_path_results is not None:
            return mixed_fast_path_results
        results: List[Tuple[List[str], List[str]] | None] = [None] * len(texts)
        fallback_pairs: List[Tuple[int, str]] = []
        for index, text in enumerate(texts):
            fast_path = _try_whitespace_mixed_fast_path(text, normalized_language)
            if fast_path is not None:
                results[index] = fast_path
            else:
                fallback_pairs.append((index, text))
        if not fallback_pairs:
            return [result for result in results if result is not None]
        fallback_rows = (
            LangSegmenter.getTextsBatch([text for _, text in fallback_pairs])
            if normalized_language == "auto"
            else LangSegmenter.getTextsBatch([text for _, text in fallback_pairs])
        )
        for (index, _text), items in zip(fallback_pairs, fallback_rows):
            if normalized_language == "auto_yue":
                textlist = []
                langlist = []
                for item in items:
                    item_lang = "yue" if item["lang"] == "zh" else item["lang"]
                    textlist.append(item["text"])
                    langlist.append(item_lang)
                results[index] = (textlist, langlist)
            else:
                results[index] = ([item["text"] for item in items], [item["lang"] for item in items])
        return [result for result in results if result is not None]
    if normalized_language == "all_zh":
        return [
            ([item["text"] for item in items], [item["lang"] for item in items])
            for items in LangSegmenter.getTextsBatch(texts, "zh")
        ]
    if normalized_language == "all_yue":
        results: List[Tuple[List[str], List[str]]] = []
        for items in LangSegmenter.getTextsBatch(texts, "zh"):
            textlist: List[str] = []
            langlist: List[str] = []
            for item in items:
                item_lang = "yue" if item["lang"] == "zh" else item["lang"]
                textlist.append(item["text"])
                langlist.append(item_lang)
            results.append((textlist, langlist))
        return results
    if normalized_language == "all_ja":
        return [
            ([item["text"] for item in items], [item["lang"] for item in items])
            for items in LangSegmenter.getTextsBatch(texts, "ja")
        ]
    if normalized_language == "all_ko":
        return [
            ([item["text"] for item in items], [item["lang"] for item in items])
            for items in LangSegmenter.getTextsBatch(texts, "ko")
        ]
    if normalized_language == "en":
        return [([text], ["en"]) for text in texts]
    return [split_text_by_language(text, normalized_language) for text in texts]


def clean_text_segment(text: str, language: str, version: str) -> Tuple[List[int], Optional[List[int]], str]:
    normalized_language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, normalized_language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return list(phones), None if word2ph is None else list(word2ph), str(norm_text)


def _preprocess_text_segments_payload_impl(
    text: str,
    language: str,
    version: str,
    final: bool = False,
) -> List[PreparedTextSegmentPayload]:
    text = _normalize_spaces(text)
    if _should_use_zh_fast_path(text, language):
        norm_text = chinese2.text_normalize(text)
        if not final and max(0, len(norm_text) * 2) < 6:
            return _preprocess_text_segments_payload_impl("." + text, language, version, final=True)
        return _build_zh_fast_path_payload(norm_text)
    auto_language = _get_auto_single_language_fast_path(text, language)
    if auto_language is not None:
        if auto_language == "zh":
            norm_text = chinese2.text_normalize(text)
            if not final and max(0, len(norm_text) * 2) < 6:
                return _preprocess_text_segments_payload_impl("." + text, language, version, final=True)
            return _build_zh_fast_path_payload(norm_text)
        payloads = _build_direct_language_payload(text, auto_language, version)
        estimated_phones_len = len(payloads[0]["phones"])
        if not final and estimated_phones_len < 6:
            return _preprocess_text_segments_payload_impl("." + text, language, version, final=True)
        return payloads
    direct_language = _can_use_direct_language_fast_path(text, language)
    if direct_language is not None:
        payloads = _build_direct_language_payload(text, direct_language, version)
        estimated_phones_len = len(payloads[0]["phones"])
        if not final and estimated_phones_len < 6:
            return _preprocess_text_segments_payload_impl("." + text, language, version, final=True)
        return payloads
    textlist, langlist = split_text_by_language(text, language)
    payloads: List[PreparedTextSegmentPayload] = []
    total_phones_len = 0
    for segment_text, segment_lang in zip(textlist, langlist):
        normalized_language = segment_lang.replace("all_", "")
        if normalized_language == "zh":
            norm_text = chinese2.text_normalize(segment_text)
            phones = []
            word2ph = None
            needs_g2pw = True
            estimated_phones_len = max(0, len(norm_text) * 2)
        else:
            phones, word2ph, norm_text = clean_text_segment(segment_text, segment_lang, version)
            needs_g2pw = False
            estimated_phones_len = len(phones)
        payloads.append(
            {
                "language": normalized_language,
                "phones": phones,
                "word2ph": word2ph,
                "norm_text": norm_text,
                "needs_g2pw": needs_g2pw,
            }
        )
        total_phones_len += int(estimated_phones_len)

    if not final and total_phones_len < 6:
        return _preprocess_text_segments_payload_impl("." + text, language, version, final=True)

    return payloads


def preprocess_text_segments_payload(
    text: str,
    language: str,
    version: str,
    final: bool = False,
) -> List[PreparedTextSegmentPayload]:
    item = (_normalize_spaces(str(text)), str(language), str(version), bool(final))
    cached = _cache_get_payloads(item)
    if cached is not None:
        return cached
    payloads = _preprocess_text_segments_payload_impl(*item)
    _cache_store_payloads(item, payloads)
    return _clone_payloads(payloads)


def preprocess_text_segments_payload_batch(
    items: Sequence[PreparedTextSegmentBatchItem],
) -> List[List[PreparedTextSegmentPayload]]:
    normalized_items = [
        (_normalize_spaces(str(text)), str(language), str(version), bool(final))
        for text, language, version, final in items
    ]
    results: List[List[PreparedTextSegmentPayload] | None] = [None] * len(normalized_items)
    duplicate_indices_by_root: Dict[int, List[int]] = {}
    unique_items: List[PreparedTextSegmentBatchItem] = []
    unique_result_indices: List[int] = []
    first_index_by_item: Dict[PreparedTextSegmentBatchItem, int] = {}

    for index, item in enumerate(normalized_items):
        cached = _cache_get_payloads(item)
        if cached is not None:
            results[index] = cached
            continue
        root_index = first_index_by_item.get(item)
        if root_index is not None:
            duplicate_indices_by_root.setdefault(root_index, []).append(index)
            continue
        first_index_by_item[item] = index
        unique_items.append(item)
        unique_result_indices.append(index)

    normalized_items = unique_items
    item_segment_indices: List[List[int]] = [[] for _ in normalized_items]
    jobs_by_language: Dict[Tuple[str, str], List[_SegmentJob]] = {}
    retry_items: List[PreparedTextSegmentBatchItem] = []
    retry_result_indices: List[int] = []
    next_segment_index = 0
    segment_specs_by_unique: List[List[Tuple[str, str]] | None] = [None] * len(normalized_items)
    split_batches_by_language: Dict[str, List[Tuple[int, str]]] = {}

    for unique_index, (text, language, version, final) in enumerate(normalized_items):
        segment_specs: List[Tuple[str, str]] = []
        if _should_use_zh_fast_path(text, language):
            segment_specs = [(text, "zh")]
        else:
            auto_language = _get_auto_single_language_fast_path(text, language)
            if auto_language is not None:
                segment_specs = [(text, auto_language)]
            else:
                direct_language = _can_use_direct_language_fast_path(text, language)
                if direct_language is not None:
                    segment_specs = [(text, direct_language)]
                else:
                    split_batches_by_language.setdefault(str(language), []).append((unique_index, text))

        if segment_specs:
            segment_specs_by_unique[unique_index] = segment_specs

    for language, pending_items in split_batches_by_language.items():
        split_results = split_texts_by_language_batch(
            [text for _unique_index, text in pending_items],
            language,
        )
        for (unique_index, _text), (textlist, langlist) in zip(pending_items, split_results):
            segment_specs_by_unique[unique_index] = list(zip(textlist, langlist))

    for unique_index, segment_specs in enumerate(segment_specs_by_unique):
        assert segment_specs is not None
        for segment_text, segment_lang in segment_specs:
            segment_index = next_segment_index
            next_segment_index += 1
            item_segment_indices[unique_index].append(segment_index)
            normalized_language = str(segment_lang).replace("all_", "")
            jobs_by_language.setdefault((normalized_language, version), []).append(
                (segment_index, str(segment_text), str(segment_lang), version)
            )

    payloads_by_segment = _build_segment_payloads_batch(jobs_by_language)

    for unique_index, segment_indices in enumerate(item_segment_indices):
        result_index = unique_result_indices[unique_index]
        payloads = [payloads_by_segment[segment_index] for segment_index in segment_indices]
        text, language, version, final = normalized_items[unique_index]
        if not final and _estimate_payload_phones_len(payloads) < 6:
            retry_items.append(("." + text, language, version, True))
            retry_result_indices.append(result_index)
            continue
        _cache_store_payloads((text, language, version, final), payloads)
        results[result_index] = payloads

    if retry_items:
        retry_results = preprocess_text_segments_payload_batch(retry_items)
        for result_index, payloads in zip(retry_result_indices, retry_results):
            results[result_index] = payloads

    for root_index, duplicate_indices in duplicate_indices_by_root.items():
        root_payloads = results[root_index]
        assert root_payloads is not None
        for duplicate_index in duplicate_indices:
            results[duplicate_index] = _clone_payloads(root_payloads)

    return [result if result is not None else [] for result in results]
