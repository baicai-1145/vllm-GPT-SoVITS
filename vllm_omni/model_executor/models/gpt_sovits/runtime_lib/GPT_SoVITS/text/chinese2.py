import os
import re
import threading
import time
from typing import Dict, List, Sequence, Tuple

import cn2an
from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

from text.symbols import punctuation
from text.tone_sandhi import ToneSandhi
from text.zh_normalization.text_normlization import TextNormalizer

normalizer = lambda x: cn2an.transform(x, "an2cn")

current_file_path = os.path.dirname(__file__)
_TEXT_ROOT = os.path.abspath(current_file_path)
_PROJECT_PACKAGE_ROOT = os.path.abspath(os.path.join(_TEXT_ROOT, ".."))
_DEFAULT_G2PW_MODEL_DIR = os.path.join(_PROJECT_PACKAGE_ROOT, "text", "G2PWModel")
_DEFAULT_G2PW_BERT_PATH = os.path.join(_PROJECT_PACKAGE_ROOT, "pretrained_models", "chinese-roberta-wwm-ext-large")
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba_fast
import logging

jieba_fast.setLogLevel(logging.CRITICAL)
import jieba_fast.posseg as psg

def _env_enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


def correct_pronunciation(word, word_pinyins):
    return word_pinyins


g2pw = None
is_g2pw = _env_enabled("is_g2pw", "1")
if is_g2pw:
    try:
        from text.g2pw import G2PWPinyin, correct_pronunciation

        parent_directory = os.path.dirname(current_file_path)
        g2pw = G2PWPinyin(
            model_dir=os.environ.get("g2pw_model_dir", _DEFAULT_G2PW_MODEL_DIR),
            model_source=os.environ.get("bert_path", _DEFAULT_G2PW_BERT_PATH),
            v_to_u=False,
            neutral_tone_with_five=True,
        )
        if getattr(g2pw, "_g2pw", None) is None:
            is_g2pw = False
    except Exception as exc:
        print(f"[g2pw] disabled during chinese2 init: {exc}")
        g2pw = None
        is_g2pw = False

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "/": ",",
    "—": "-",
    "~": "…",
    "～": "…",
}
_REP_PATTERN = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
_ZH_ONLY_PATTERN = re.compile(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+")
_ZH_EN_PATTERN = re.compile(r"[^\u4e00-\u9fa5A-Za-z" + "".join(punctuation) + r"]+")
_PUNCTUATION_CHARS = "".join(re.escape(p) for p in punctuation)
_PUNCTUATIONS_PATTERN = re.compile(f"([{_PUNCTUATION_CHARS}])([{_PUNCTUATION_CHARS}])+")

tone_modifier = ToneSandhi()
_THREAD_LOCAL = threading.local()
_G2PW_PIPELINE_PREWARM_LOCK = threading.Lock()
_G2PW_PIPELINE_PREWARMED = False


def _get_text_normalizer() -> TextNormalizer:
    normalizer = getattr(_THREAD_LOCAL, "text_normalizer", None)
    if normalizer is None:
        normalizer = TextNormalizer()
        _THREAD_LOCAL.text_normalizer = normalizer
    return normalizer


def _normalize_text_with_cached_normalizer(text: str) -> str:
    normalizer = _get_text_normalizer()
    sentences = normalizer.normalize(text)
    dest_text = "".join(replace_punctuation(sentence) for sentence in sentences)
    return replace_consecutive_punctuation(dest_text)


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    replaced_text = _REP_PATTERN.sub(lambda x: rep_map[x.group()], text)
    replaced_text = _ZH_ONLY_PATTERN.sub("", replaced_text)

    return replaced_text


def g2p(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, word2ph = _g2p(sentences)
    return phones, word2ph


def _prepare_g2p_segments(segments):
    prepared_segments = []
    batch_inputs = []
    for segment in segments:
        processed_segment = re.sub("[a-zA-Z]+", "", segment)
        seg_cut = psg.lcut(processed_segment)
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        prepared_segments.append(
            {
                "segment": processed_segment,
                "seg_cut": seg_cut,
            }
        )
        if processed_segment:
            batch_inputs.append(processed_segment)
    return prepared_segments, batch_inputs


def _new_g2pw_profile() -> Dict[str, float]:
    return {
        "g2pw_prepare_ms": 0.0,
        "g2pw_predict_ms": 0.0,
        "g2pw_post_ms": 0.0,
        "g2pw_total_ms": 0.0,
        "g2pw_runtime_total_ms": 0.0,
        "g2pw_runtime_queue_wait_ms": 0.0,
        "g2pw_runtime_collect_wait_ms": 0.0,
        "g2pw_runtime_run_ms": 0.0,
        "g2pw_runtime_batch_rows": 0.0,
        "g2pw_runtime_batch_requests": 0.0,
        "g2pw_runtime_pool_workers": 0.0,
        "g2pw_runtime_shard_index": 0.0,
    }


def _predict_g2pw_batch(batch_inputs: Sequence[str]) -> Tuple[List[List[str]], Dict[str, float]]:
    profile = _new_g2pw_profile()
    if not (is_g2pw and batch_inputs and g2pw is not None):
        return [], profile
    converter = getattr(g2pw, "_g2pw", None)
    if converter is None:
        return [], profile
    if hasattr(converter, "predict_sentences_with_profile"):
        g2pw_batch_results, predict_profile = converter.predict_sentences_with_profile(list(batch_inputs))
        for key, value in dict(predict_profile or {}).items():
            profile[key] = float(value)
        return g2pw_batch_results, profile
    predict_start = time.perf_counter()
    g2pw_batch_results = converter(list(batch_inputs))
    profile["g2pw_predict_ms"] = float((time.perf_counter() - predict_start) * 1000.0)
    return g2pw_batch_results, profile


def _g2pw_batch_weight(prepared_segments, batch_inputs: Sequence[str]) -> float:
    char_count = sum(len(item["segment"]) for item in prepared_segments if item["segment"])
    if char_count > 0:
        return float(char_count)
    if batch_inputs:
        return float(len(batch_inputs))
    return 0.0


def _merge_shared_g2pw_profile(
    profile: Dict[str, float],
    shared_profile: Dict[str, float],
    weight: float,
    total_weight: float,
) -> None:
    if total_weight <= 0.0 or weight <= 0.0:
        return
    scale = float(weight) / float(total_weight)
    for key in (
        "g2pw_predict_ms",
        "g2pw_runtime_total_ms",
        "g2pw_runtime_queue_wait_ms",
        "g2pw_runtime_collect_wait_ms",
        "g2pw_runtime_run_ms",
    ):
        profile[key] = float(profile.get(key, 0.0)) + float(shared_profile.get(key, 0.0)) * scale
    for key in (
        "g2pw_runtime_batch_rows",
        "g2pw_runtime_batch_requests",
        "g2pw_runtime_pool_workers",
        "g2pw_runtime_shard_index",
    ):
        profile[key] = float(shared_profile.get(key, 0.0))


def g2p_segments_batch(
    segment_batches: Sequence[Sequence[str]],
    return_profiles: bool = False,
):
    prepared_batches = []
    all_batch_inputs: List[str] = []
    batch_weights: List[float] = []
    for segments in segment_batches:
        prepare_start = time.perf_counter()
        prepared_segments, batch_inputs = _prepare_g2p_segments(list(segments))
        profile = _new_g2pw_profile()
        profile["g2pw_prepare_ms"] = float((time.perf_counter() - prepare_start) * 1000.0)
        prepared_batches.append(
            {
                "prepared_segments": prepared_segments,
                "batch_inputs": batch_inputs,
                "profile": profile,
            }
        )
        all_batch_inputs.extend(batch_inputs)
        batch_weights.append(_g2pw_batch_weight(prepared_segments, batch_inputs))

    g2pw_batch_results, shared_profile = _predict_g2pw_batch(all_batch_inputs)
    total_weight = float(sum(batch_weights))
    results_batches = []
    batch_cursor = 0
    for batch_info, batch_weight in zip(prepared_batches, batch_weights):
        profile = batch_info["profile"]
        batch_inputs = batch_info["batch_inputs"]
        if batch_inputs:
            _merge_shared_g2pw_profile(profile, shared_profile, batch_weight, total_weight)
        post_start = time.perf_counter()
        results = []
        for item in batch_info["prepared_segments"]:
            segment = item["segment"]
            if not segment:
                results.append(([], [], segment))
                continue
            if not is_g2pw or g2pw is None or batch_cursor >= len(g2pw_batch_results):
                phones, word2ph = _build_segment_without_g2pw(segment, item["seg_cut"])
                results.append((phones, word2ph, segment))
                continue
            pinyins = g2pw_batch_results[batch_cursor]
            batch_cursor += 1
            if not pinyins:
                phones, word2ph = _build_segment_without_g2pw(segment, item["seg_cut"])
                results.append((phones, word2ph, segment))
                continue
            phones, word2ph = _build_segment_from_g2pw(segment, item["seg_cut"], pinyins)
            results.append((phones, word2ph, segment))
        profile["g2pw_post_ms"] = float((time.perf_counter() - post_start) * 1000.0)
        profile["g2pw_total_ms"] = float(
            profile["g2pw_prepare_ms"] + profile["g2pw_predict_ms"] + profile["g2pw_post_ms"]
        )
        results_batches.append(results)
    if return_profiles:
        return results_batches, [dict(batch_info["profile"]) for batch_info in prepared_batches]
    return results_batches


def prewarm_g2pw_pipeline(sentences: Sequence[str] | None = None, rounds: int = 1) -> bool:
    global _G2PW_PIPELINE_PREWARMED
    if not is_g2pw or g2pw is None or getattr(g2pw, "_g2pw", None) is None:
        return False
    with _G2PW_PIPELINE_PREWARM_LOCK:
        if _G2PW_PIPELINE_PREWARMED:
            return False
        warm_sentences = list(
            sentences
            or [
                "重庆银行的行长在长安见到了重要的人。",
                "音乐老师重新调整了长句里的重音和节奏。",
                "我们准备把参考文本和目标文本一起处理。",
                "这个系统需要在高并发下稳定完成预处理。",
            ]
        )
        warm_sentences = [str(item).strip() for item in warm_sentences if str(item).strip()]
        if not warm_sentences:
            warm_sentences = ["重庆银行的行长在长安见到了重要的人。"]
        multiplier_values = []
        for raw in str(os.environ.get("GPTSOVITS_G2PW_CUDA_PREWARM_BATCH_MULTIPLIERS", "1,2,4")).split(","):
            raw = raw.strip()
            if not raw:
                continue
            multiplier_values.append(max(1, int(raw)))
        if not multiplier_values:
            multiplier_values = [1]
        warm_rounds = max(1, int(rounds))
        warm_batches = []
        for multiplier in sorted(set(multiplier_values)):
            batch = list(warm_sentences) * int(multiplier)
            if not batch:
                continue
            warm_batches.append(batch)
            if len(batch) > 1:
                warm_batches.append(list(reversed(batch)))
        for _ in range(warm_rounds):
            for batch in warm_batches:
                g2p_segments_batch([batch], return_profiles=False)
        _G2PW_PIPELINE_PREWARMED = True
        return True


def _build_segment_from_g2pw(segment: str, seg_cut, pinyins):
    phones_list = []
    word2ph = []
    initials = []
    finals = []
    pre_word_length = 0
    for word, pos in seg_cut:
        sub_initials = []
        sub_finals = []
        now_word_length = pre_word_length + len(word)

        if pos == "eng":
            pre_word_length = now_word_length
            continue

        word_pinyins = pinyins[pre_word_length:now_word_length]
        word_pinyins = correct_pronunciation(word, word_pinyins)

        for pinyin in word_pinyins:
            if pinyin[0].isalpha():
                sub_initials.append(to_initials(pinyin))
                sub_finals.append(to_finals_tone3(pinyin, neutral_tone_with_five=True))
            else:
                sub_initials.append(pinyin)
                sub_finals.append(pinyin)

        pre_word_length = now_word_length
        sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
        sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)
        initials.append(sub_initials)
        finals.append(sub_finals)

    initials = sum(initials, [])
    finals = sum(finals, [])
    for c, v in zip(initials, finals):
        raw_pinyin = c + v
        if c == v:
            assert c in punctuation
            phone = [c]
            word2ph.append(1)
        else:
            v_without_tone = v[:-1]
            tone = v[-1]

            pinyin = c + v_without_tone
            assert tone in "12345"

            if c:
                v_rep_map = {
                    "uei": "ui",
                    "iou": "iu",
                    "uen": "un",
                }
                if v_without_tone in v_rep_map.keys():
                    pinyin = c + v_rep_map[v_without_tone]
            else:
                pinyin_rep_map = {
                    "ing": "ying",
                    "i": "yi",
                    "in": "yin",
                    "u": "wu",
                }
                if pinyin in pinyin_rep_map.keys():
                    pinyin = pinyin_rep_map[pinyin]
                else:
                    single_rep_map = {
                        "v": "yu",
                        "e": "e",
                        "i": "y",
                        "u": "w",
                    }
                    if pinyin[0] in single_rep_map.keys():
                        pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

            assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, segment, raw_pinyin)
            new_c, new_v = pinyin_to_symbol_map[pinyin].split(" ")
            new_v = new_v + tone
            phone = [new_c, new_v]
            word2ph.append(len(phone))

        phones_list += phone
    return phones_list, word2ph


def _build_segment_without_g2pw(segment: str, seg_cut):
    initials = []
    finals = []
    for word, pos in seg_cut:
        if pos == "eng":
            continue
        sub_initials, sub_finals = _get_initials_finals(word)
        sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
        sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)
        initials.append(sub_initials)
        finals.append(sub_finals)
    phones_list = []
    word2ph = []
    for c, v in zip(sum(initials, []), sum(finals, [])):
        raw_pinyin = c + v
        if c == v:
            assert c in punctuation
            phone = [c]
            word2ph.append(1)
        else:
            v_without_tone = v[:-1]
            tone = v[-1]
            pinyin = c + v_without_tone
            assert tone in "12345"
            if c:
                v_rep_map = {"uei": "ui", "iou": "iu", "uen": "un"}
                if v_without_tone in v_rep_map:
                    pinyin = c + v_rep_map[v_without_tone]
            else:
                pinyin_rep_map = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
                if pinyin in pinyin_rep_map:
                    pinyin = pinyin_rep_map[pinyin]
                else:
                    single_rep_map = {"v": "yu", "e": "e", "i": "y", "u": "w"}
                    if pinyin[0] in single_rep_map:
                        pinyin = single_rep_map[pinyin[0]] + pinyin[1:]
            assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, segment, raw_pinyin)
            new_c, new_v = pinyin_to_symbol_map[pinyin].split(" ")
            new_v = new_v + tone
            phone = [new_c, new_v]
            word2ph.append(len(phone))
        phones_list += phone
    return phones_list, word2ph


def g2p_segments(segments, return_profile: bool = False):
    results_batches, profile_batches = g2p_segments_batch([segments], return_profiles=True)
    results = results_batches[0]
    profile = profile_batches[0]
    if return_profile:
        return results, profile
    return results


def _get_initials_finals(word):
    initials = []
    finals = []

    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)

    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


must_erhua = {"小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"}
not_erhua = {
    "虐儿",
    "为儿",
    "护儿",
    "瞒儿",
    "救儿",
    "替儿",
    "有儿",
    "一儿",
    "我儿",
    "俺儿",
    "妻儿",
    "拐儿",
    "聋儿",
    "乞儿",
    "患儿",
    "幼儿",
    "孤儿",
    "婴儿",
    "婴幼儿",
    "连体儿",
    "脑瘫儿",
    "流浪儿",
    "体弱儿",
    "混血儿",
    "蜜雪儿",
    "舫儿",
    "祖儿",
    "美儿",
    "应采儿",
    "可儿",
    "侄儿",
    "孙儿",
    "侄孙儿",
    "女儿",
    "男儿",
    "红孩儿",
    "花儿",
    "虫儿",
    "马儿",
    "鸟儿",
    "猪儿",
    "猫儿",
    "狗儿",
    "少儿",
}


def _merge_erhua(initials: list[str], finals: list[str], word: str, pos: str) -> list[list[str]]:
    """
    Do erhub.
    """
    # fix er1
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "儿" and phn == "er1":
            finals[i] = "er2"

    # 发音
    if word not in must_erhua and (word in not_erhua or pos in {"a", "j", "nr"}):
        return initials, finals

    # "……" 等情况直接返回
    if len(finals) != len(word):
        return initials, finals

    assert len(finals) == len(word)

    # 与前一个字发同音
    new_initials = []
    new_finals = []
    for i, phn in enumerate(finals):
        if (
            i == len(finals) - 1
            and word[i] == "儿"
            and phn in {"er2", "er5"}
            and word[-2:] not in not_erhua
            and new_finals
        ):
            phn = "er" + new_finals[-1][-1]

        new_initials.append(initials[i])
        new_finals.append(phn)

    return new_initials, new_finals


def _g2p(segments):
    phones_list = []
    word2ph = []
    for phones, item_word2ph, _segment in g2p_segments(segments):
        phones_list += phones
        word2ph += item_word2ph
    return phones_list, word2ph


def replace_punctuation_with_en(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    replaced_text = _REP_PATTERN.sub(lambda x: rep_map[x.group()], text)
    replaced_text = _ZH_EN_PATTERN.sub("", replaced_text)

    return replaced_text


def replace_consecutive_punctuation(text):
    return _PUNCTUATIONS_PATTERN.sub(r"\1", text)


def text_normalize(text):
    # https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    return _normalize_text_with_cached_normalizer(text)


def text_normalize_batch(texts):
    normalizer = _get_text_normalizer()
    results = []
    for text in texts:
        sentences = normalizer.normalize(text)
        dest_text = "".join(replace_punctuation(sentence) for sentence in sentences)
        results.append(replace_consecutive_punctuation(dest_text))
    return results


if __name__ == "__main__":
    text = "啊——但是《原神》是由,米哈\游自主，研发的一款全.新开放世界.冒险游戏"
    text = "呣呣呣～就是…大人的鼹鼠党吧？"
    text = "你好"
    text = text_normalize(text)
    print(g2p(text))


# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试..."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
