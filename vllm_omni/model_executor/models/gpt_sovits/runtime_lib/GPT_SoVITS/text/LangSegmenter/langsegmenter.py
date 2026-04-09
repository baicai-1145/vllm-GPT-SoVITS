import logging
import re
import threading

# jieba静音
import jieba

jieba.setLogLevel(logging.CRITICAL)

# 更改fast_langdetect大模型位置
from pathlib import Path

import fast_langdetect

fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(
    fast_langdetect.infer.LangDetectConfig(
        cache_dir=Path(__file__).parent.parent.parent / "pretrained_models" / "fast_langdetect"
    )
)


from split_lang import LangSplitter

_FULL_EN_PATTERN = re.compile(r"^(?=.*[A-Za-z])[A-Za-z0-9\s\u0020-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]+$")
_CJK_CHAR_PATTERN = re.compile(r"[\u3400-\u4db5\u4e00-\u9fff\U00020000-\U0002EE5D]")
_CJK_ALLOWED_NONCHAR_PATTERN = re.compile(r"[0-9、\-〜。！？.!?… /]")
_JA_SPLIT_PATTERN = re.compile(
    r"([\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]+(?:[0-9、\-〜。！？.!?… ]+[\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]*)*)"
)
_KO_SPLIT_PATTERN = re.compile(
    r"([\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]+(?:[0-9、\-〜。！？.!?… ]+[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]*)*)"
)


def full_en(text):
    return bool(_FULL_EN_PATTERN.match(str(text)))


def full_cjk(text):
    chars: list[str] = []
    for char in str(text):
        if _CJK_CHAR_PATTERN.match(char) or _CJK_ALLOWED_NONCHAR_PATTERN.match(char):
            chars.append(char)
    return "".join(chars)


def split_jako(tag_lang, item):
    pattern = _JA_SPLIT_PATTERN if tag_lang == "ja" else _KO_SPLIT_PATTERN

    lang_list: list[dict] = []
    tag = 0
    for match in pattern.finditer(item["text"]):
        if match.start() > tag:
            lang_list.append({"lang": item["lang"], "text": item["text"][tag : match.start()]})

        tag = match.end()
        lang_list.append({"lang": tag_lang, "text": item["text"][match.start() : match.end()]})

    if tag < len(item["text"]):
        lang_list.append({"lang": item["lang"], "text": item["text"][tag : len(item["text"])]})

    return lang_list


def merge_lang(lang_list, item):
    if lang_list and item["lang"] == lang_list[-1]["lang"]:
        lang_list[-1]["text"] += item["text"]
    else:
        lang_list.append(item)
    return lang_list


class LangSegmenter:
    # 默认过滤器, 基于gsv目前四种语言
    DEFAULT_LANG_MAP = {
        "zh": "zh",
        "yue": "zh",  # 粤语
        "wuu": "zh",  # 吴语
        "zh-cn": "zh",
        "zh-tw": "x",  # 繁体设置为x
        "ko": "ko",
        "ja": "ja",
        "en": "en",
    }
    _splitter_lock = threading.Lock()
    _splitter_cache: dict[str, LangSplitter] = {}

    @classmethod
    def _get_splitter(cls, default_lang: str = "") -> LangSplitter:
        cache_key = str(default_lang)
        splitter = cls._splitter_cache.get(cache_key)
        if splitter is not None:
            return splitter
        with cls._splitter_lock:
            splitter = cls._splitter_cache.get(cache_key)
            if splitter is None:
                splitter = LangSplitter(
                    lang_map=cls.DEFAULT_LANG_MAP,
                    default_lang=cache_key or "x",
                    merge_across_digit=False,
                    debug=False,
                )
                splitter.merge_across_digit = False
                cls._splitter_cache[cache_key] = splitter
            return splitter

    @classmethod
    def _getTexts_with_splitter(cls, text: str, splitter: LangSplitter, default_lang: str = ""):
        substr = splitter.split_by_lang(text=str(text))

        lang_list: list[dict] = []
        have_num = False

        for _, item in enumerate(substr):
            dict_item = {"lang": item.lang, "text": item.text}

            if dict_item["lang"] == "digit":
                if default_lang != "":
                    dict_item["lang"] = default_lang
                else:
                    have_num = True
                lang_list = merge_lang(lang_list, dict_item)
                continue

            if full_en(dict_item["text"]):
                dict_item["lang"] = "en"
                lang_list = merge_lang(lang_list, dict_item)
                continue

            if default_lang != "":
                dict_item["lang"] = default_lang
                lang_list = merge_lang(lang_list, dict_item)
                continue

            ja_list: list[dict] = []
            if dict_item["lang"] != "ja":
                ja_list = split_jako("ja", dict_item)

            if not ja_list:
                ja_list.append(dict_item)

            ko_list: list[dict] = []
            temp_list: list[dict] = []
            for _, ko_item in enumerate(ja_list):
                if ko_item["lang"] != "ko":
                    ko_list = split_jako("ko", ko_item)

                if ko_list:
                    temp_list.extend(ko_list)
                else:
                    temp_list.append(ko_item)

            if len(temp_list) == 1:
                if dict_item["lang"] == "x":
                    cjk_text = full_cjk(dict_item["text"])
                    if cjk_text:
                        dict_item = {"lang": "zh", "text": cjk_text}
                        lang_list = merge_lang(lang_list, dict_item)
                    else:
                        lang_list = merge_lang(lang_list, dict_item)
                    continue
                lang_list = merge_lang(lang_list, dict_item)
                continue

            for _, temp_item in enumerate(temp_list):
                if temp_item["lang"] == "x":
                    cjk_text = full_cjk(temp_item["text"])
                    if cjk_text:
                        lang_list = merge_lang(lang_list, {"lang": "zh", "text": cjk_text})
                    else:
                        lang_list = merge_lang(lang_list, temp_item)
                else:
                    lang_list = merge_lang(lang_list, temp_item)

        if have_num:
            temp_list = lang_list
            lang_list = []
            for i, temp_item in enumerate(temp_list):
                if temp_item["lang"] == "digit":
                    if default_lang:
                        temp_item["lang"] = default_lang
                    elif lang_list and i == len(temp_list) - 1:
                        temp_item["lang"] = lang_list[-1]["lang"]
                    elif not lang_list and i < len(temp_list) - 1:
                        temp_item["lang"] = temp_list[1]["lang"]
                    elif lang_list and i < len(temp_list) - 1:
                        if lang_list[-1]["lang"] == temp_list[i + 1]["lang"]:
                            temp_item["lang"] = lang_list[-1]["lang"]
                        elif lang_list[-1]["text"][-1] in [",", ".", "!", "?", "，", "。", "！", "？"]:
                            temp_item["lang"] = temp_list[i + 1]["lang"]
                        elif temp_list[i + 1]["text"][0] in [",", ".", "!", "?", "，", "。", "！", "？"]:
                            temp_item["lang"] = lang_list[-1]["lang"]
                        elif temp_item["text"][-1] in ["。", "."]:
                            temp_item["lang"] = lang_list[-1]["lang"]
                        elif len(lang_list[-1]["text"]) >= len(temp_list[i + 1]["text"]):
                            temp_item["lang"] = lang_list[-1]["lang"]
                        else:
                            temp_item["lang"] = temp_list[i + 1]["lang"]
                    else:
                        temp_item["lang"] = "zh"

                lang_list = merge_lang(lang_list, temp_item)

        temp_list = lang_list
        lang_list = []
        for _, temp_item in enumerate(temp_list):
            if temp_item["lang"] == "x":
                if lang_list:
                    temp_item["lang"] = lang_list[-1]["lang"]
                elif len(temp_list) > 1:
                    temp_item["lang"] = temp_list[1]["lang"]
                else:
                    temp_item["lang"] = "zh"

            lang_list = merge_lang(lang_list, temp_item)

        return lang_list

    @classmethod
    def getTextsBatch(cls, texts, default_lang=""):
        splitter = cls._get_splitter(default_lang)
        rows = splitter.split_by_lang_batch([str(text) for text in texts])
        return [cls._build_lang_list_from_splitter_rows(substr, default_lang) for substr in rows]

    def getTexts(self, default_lang=""):
        splitter = LangSegmenter._get_splitter(default_lang)
        return LangSegmenter._getTexts_with_splitter(self, splitter, default_lang)

    @classmethod
    def _build_lang_list_from_splitter_rows(cls, substr, default_lang: str = ""):
        lang_list: list[dict] = []
        have_num = False

        for _, item in enumerate(substr):
            dict_item = {"lang": item.lang, "text": item.text}

            if dict_item["lang"] == "digit":
                if default_lang != "":
                    dict_item["lang"] = default_lang
                else:
                    have_num = True
                lang_list = merge_lang(lang_list, dict_item)
                continue

            if full_en(dict_item["text"]):
                dict_item["lang"] = "en"
                lang_list = merge_lang(lang_list, dict_item)
                continue

            if default_lang != "":
                dict_item["lang"] = default_lang
                lang_list = merge_lang(lang_list, dict_item)
                continue

            ja_list: list[dict] = []
            if dict_item["lang"] != "ja":
                ja_list = split_jako("ja", dict_item)

            if not ja_list:
                ja_list.append(dict_item)

            ko_list: list[dict] = []
            temp_list: list[dict] = []
            for _, ko_item in enumerate(ja_list):
                if ko_item["lang"] != "ko":
                    ko_list = split_jako("ko", ko_item)

                if ko_list:
                    temp_list.extend(ko_list)
                else:
                    temp_list.append(ko_item)

            if len(temp_list) == 1:
                if dict_item["lang"] == "x":
                    cjk_text = full_cjk(dict_item["text"])
                    if cjk_text:
                        dict_item = {"lang": "zh", "text": cjk_text}
                        lang_list = merge_lang(lang_list, dict_item)
                    else:
                        lang_list = merge_lang(lang_list, dict_item)
                    continue
                lang_list = merge_lang(lang_list, dict_item)
                continue

            for _, temp_item in enumerate(temp_list):
                if temp_item["lang"] == "x":
                    cjk_text = full_cjk(temp_item["text"])
                    if cjk_text:
                        lang_list = merge_lang(lang_list, {"lang": "zh", "text": cjk_text})
                    else:
                        lang_list = merge_lang(lang_list, temp_item)
                else:
                    lang_list = merge_lang(lang_list, temp_item)

        if have_num:
            temp_list = lang_list
            lang_list = []
            for i, temp_item in enumerate(temp_list):
                if temp_item["lang"] == "digit":
                    if default_lang:
                        temp_item["lang"] = default_lang
                    elif lang_list and i == len(temp_list) - 1:
                        temp_item["lang"] = lang_list[-1]["lang"]
                    elif not lang_list and i < len(temp_list) - 1:
                        temp_item["lang"] = temp_list[1]["lang"]
                    elif lang_list and i < len(temp_list) - 1:
                        if lang_list[-1]["lang"] == temp_list[i + 1]["lang"]:
                            temp_item["lang"] = lang_list[-1]["lang"]
                        elif lang_list[-1]["text"][-1] in [",", ".", "!", "?", "，", "。", "！", "？"]:
                            temp_item["lang"] = temp_list[i + 1]["lang"]
                        elif temp_list[i + 1]["text"][0] in [",", ".", "!", "?", "，", "。", "！", "？"]:
                            temp_item["lang"] = lang_list[-1]["lang"]
                        elif temp_item["text"][-1] in ["。", "."]:
                            temp_item["lang"] = lang_list[-1]["lang"]
                        elif len(lang_list[-1]["text"]) >= len(temp_list[i + 1]["text"]):
                            temp_item["lang"] = lang_list[-1]["lang"]
                        else:
                            temp_item["lang"] = temp_list[i + 1]["lang"]
                    else:
                        temp_item["lang"] = "zh"

                lang_list = merge_lang(lang_list, temp_item)

        temp_list = lang_list
        lang_list = []
        for _, temp_item in enumerate(temp_list):
            if temp_item["lang"] == "x":
                if lang_list:
                    temp_item["lang"] = lang_list[-1]["lang"]
                elif len(temp_list) > 1:
                    temp_item["lang"] = temp_list[1]["lang"]
                else:
                    temp_item["lang"] = "zh"

            lang_list = merge_lang(lang_list, temp_item)

        return lang_list


if __name__ == "__main__":
    text = "MyGO?,你也喜欢まいご吗？"
    print(LangSegmenter.getTexts(text))

    text = "ねえ、知ってる？最近、僕は天文学を勉強してるんだ。君の瞳が星空みたいにキラキラしてるからさ。"
    print(LangSegmenter.getTexts(text))

    text = "当时ThinkPad T60刚刚发布，一同推出的还有一款名为Advanced Dock的扩展坞配件。这款扩展坞通过连接T60底部的插槽，扩展出包括PCIe在内的一大堆接口，并且自带电源，让T60可以安装桌面显卡来提升性能。"
    print(LangSegmenter.getTexts(text, "zh"))
    print(LangSegmenter.getTexts(text))
