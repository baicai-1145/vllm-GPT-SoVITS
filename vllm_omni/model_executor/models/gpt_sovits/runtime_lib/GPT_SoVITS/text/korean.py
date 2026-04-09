# reference: https://github.com/ORI-Muchim/MB-iSTFT-VITS-Korean/blob/main/text/korean.py

import importlib
import os
import re
from functools import lru_cache

import g2pk2.g2pk2 as g2pk2_module
import ko_pron
from g2pk2 import G2p
from jamo import h2j, j2hcj

# 防止win下无法读取模型
if os.name == "nt":

    class win_G2p(G2p):
        def check_mecab(self):
            super().check_mecab()
            spam_spec = importlib.util.find_spec("eunjeon")
            non_found = spam_spec is None
            if non_found:
                print("you have to install eunjeon. install it...")
            else:
                installpath = spam_spec.submodule_search_locations[0]
                if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", installpath)):
                    import sys

                    from eunjeon import Mecab as _Mecab

                    class Mecab(_Mecab):
                        def get_dicpath(installpath):
                            if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", installpath)):
                                import shutil

                                python_dir = os.getcwd()
                                if installpath[: len(python_dir)].upper() == python_dir.upper():
                                    dicpath = os.path.join(os.path.relpath(installpath, python_dir), "data", "mecabrc")
                                else:
                                    if not os.path.exists("TEMP"):
                                        os.mkdir("TEMP")
                                    if not os.path.exists(os.path.join("TEMP", "ko")):
                                        os.mkdir(os.path.join("TEMP", "ko"))
                                    if os.path.exists(os.path.join("TEMP", "ko", "ko_dict")):
                                        shutil.rmtree(os.path.join("TEMP", "ko", "ko_dict"))

                                    shutil.copytree(
                                        os.path.join(installpath, "data"), os.path.join("TEMP", "ko", "ko_dict")
                                    )
                                    dicpath = os.path.join("TEMP", "ko", "ko_dict", "mecabrc")
                            else:
                                dicpath = os.path.abspath(os.path.join(installpath, "data/mecabrc"))
                            return dicpath

                        def __init__(self, dicpath=get_dicpath(installpath)):
                            super().__init__(dicpath=dicpath)

                    sys.modules["eunjeon"].Mecab = Mecab

    G2p = win_G2p


from text.symbols2 import symbols

_REGEX_CACHE_TARGET = 8192
try:
    if int(getattr(re, "_MAXCACHE", 0)) < _REGEX_CACHE_TARGET:
        re._MAXCACHE = _REGEX_CACHE_TARGET
except Exception:
    pass

_ASCII_ALPHA_PATTERN = re.compile(r"[A-Za-z]")
_DIGIT_PATTERN = re.compile(r"\d")
_ANNOTATION_MARK_PATTERN = re.compile(r"/[PJEB]")
_TAIL_JAMO_PATTERN = re.compile(r"([\u3131-\u3163])$")
_IDIOM_RULE_SEPARATOR = "==="
_TRAILING_PASSTHROUGH_CHARS = frozenset(" \n：；，。！？·、,.!?")
_SPECIAL_G2PK2_FUNCS = (
    g2pk2_module.jyeo,
    g2pk2_module.ye,
    g2pk2_module.consonant_ui,
    g2pk2_module.josa_ui,
    g2pk2_module.vowel_ui,
    g2pk2_module.jamo,
    g2pk2_module.rieulgiyeok,
    g2pk2_module.rieulbieub,
    g2pk2_module.verb_nieun,
    g2pk2_module.balb,
    g2pk2_module.palatalize,
    g2pk2_module.modifying_rieul,
)
_LINK_G2PK2_FUNCS = (
    g2pk2_module.link1,
    g2pk2_module.link2,
    g2pk2_module.link3,
    g2pk2_module.link4,
)


def _load_compiled_idiom_rules(path: str) -> list[tuple[re.Pattern[str], str]]:
    rules: list[tuple[re.Pattern[str], str]] = []
    with open(path, encoding="utf8") as idiom_file:
        for raw_line in idiom_file:
            line = raw_line.split("#")[0].strip()
            if _IDIOM_RULE_SEPARATOR not in line:
                continue
            pattern_text, replacement = line.split(_IDIOM_RULE_SEPARATOR, 1)
            rules.append((re.compile(pattern_text), replacement))
    return rules


class OptimizedG2p(G2p):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._idiom_rules = _load_compiled_idiom_rules(self.idioms_path)
        self._compiled_table = [
            (re.compile(pattern_text), replacement, tuple(rule_ids))
            for pattern_text, replacement, rule_ids in self.table
        ]

    def idioms(self, string, descriptive=False, verbose=False):
        rule = "from idioms.txt"
        out = string
        for pattern, replacement in self._idiom_rules:
            out = pattern.sub(replacement, out)
        g2pk2_module.gloss(verbose, out, string, rule)
        return out

    def __call__(self, string, descriptive=False, verbose=False, group_vowels=False, to_syl=True):
        string = self.idioms(string, descriptive, verbose)

        if _ASCII_ALPHA_PATTERN.search(string):
            string = g2pk2_module.convert_eng(string, self.cmu)

        string = g2pk2_module.annotate(string, self.mecab)

        if _DIGIT_PATTERN.search(string):
            string = g2pk2_module.convert_num(string)

        inp = h2j(string)

        for func in _SPECIAL_G2PK2_FUNCS:
            inp = func(inp, descriptive, verbose)
        inp = _ANNOTATION_MARK_PATTERN.sub("", inp)

        for pattern, replacement, rule_ids in self._compiled_table:
            previous = inp
            inp = pattern.sub(replacement, inp)
            if verbose:
                if rule_ids:
                    rule = "\n".join(self.rule2text.get(rule_id, "") for rule_id in rule_ids)
                else:
                    rule = ""
                g2pk2_module.gloss(verbose, inp, previous, rule)

        for func in _LINK_G2PK2_FUNCS:
            inp = func(inp, descriptive, verbose)

        if group_vowels:
            inp = g2pk2_module.group(inp)

        if to_syl:
            inp = g2pk2_module.compose(inp)
        return inp


# This is a list of Korean classifiers preceded by pure Korean numerals.
_korean_classifiers = (
    "군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통"
)

# List of (hangul, hangul divided) pairs:
_hangul_divided = [
    (re.compile(f"{x[0]}"), x[1])
    for x in [
        # ('ㄳ', 'ㄱㅅ'),   # g2pk2, A Syllable-ending Rule
        # ('ㄵ', 'ㄴㅈ'),
        # ('ㄶ', 'ㄴㅎ'),
        # ('ㄺ', 'ㄹㄱ'),
        # ('ㄻ', 'ㄹㅁ'),
        # ('ㄼ', 'ㄹㅂ'),
        # ('ㄽ', 'ㄹㅅ'),
        # ('ㄾ', 'ㄹㅌ'),
        # ('ㄿ', 'ㄹㅍ'),
        # ('ㅀ', 'ㄹㅎ'),
        # ('ㅄ', 'ㅂㅅ'),
        ("ㅘ", "ㅗㅏ"),
        ("ㅙ", "ㅗㅐ"),
        ("ㅚ", "ㅗㅣ"),
        ("ㅝ", "ㅜㅓ"),
        ("ㅞ", "ㅜㅔ"),
        ("ㅟ", "ㅜㅣ"),
        ("ㅢ", "ㅡㅣ"),
        ("ㅑ", "ㅣㅏ"),
        ("ㅒ", "ㅣㅐ"),
        ("ㅕ", "ㅣㅓ"),
        ("ㅖ", "ㅣㅔ"),
        ("ㅛ", "ㅣㅗ"),
        ("ㅠ", "ㅣㅜ"),
    ]
]

# List of (Latin alphabet, hangul) pairs:
_latin_to_hangul = [
    (re.compile(f"{x[0]}", re.IGNORECASE), x[1])
    for x in [
        ("a", "에이"),
        ("b", "비"),
        ("c", "시"),
        ("d", "디"),
        ("e", "이"),
        ("f", "에프"),
        ("g", "지"),
        ("h", "에이치"),
        ("i", "아이"),
        ("j", "제이"),
        ("k", "케이"),
        ("l", "엘"),
        ("m", "엠"),
        ("n", "엔"),
        ("o", "오"),
        ("p", "피"),
        ("q", "큐"),
        ("r", "아르"),
        ("s", "에스"),
        ("t", "티"),
        ("u", "유"),
        ("v", "브이"),
        ("w", "더블유"),
        ("x", "엑스"),
        ("y", "와이"),
        ("z", "제트"),
    ]
]
_latin_char_to_hangul = {}
for _regex, _replacement in _latin_to_hangul:
    _pattern_text = str(_regex.pattern)
    if len(_pattern_text) == 1 and _pattern_text.isalpha():
        _latin_char_to_hangul[_pattern_text.lower()] = _replacement
_hangul_divided_map = {
    "ㅘ": "ㅗㅏ",
    "ㅙ": "ㅗㅐ",
    "ㅚ": "ㅗㅣ",
    "ㅝ": "ㅜㅓ",
    "ㅞ": "ㅜㅔ",
    "ㅟ": "ㅜㅣ",
    "ㅢ": "ㅡㅣ",
    "ㅑ": "ㅣㅏ",
    "ㅒ": "ㅣㅐ",
    "ㅕ": "ㅣㅓ",
    "ㅖ": "ㅣㅔ",
    "ㅛ": "ㅣㅗ",
    "ㅠ": "ㅣㅜ",
}

# List of (ipa, lazy ipa) pairs:
_ipa_to_lazy_ipa = [
    (re.compile(f"{x[0]}", re.IGNORECASE), x[1])
    for x in [
        ("t͡ɕ", "ʧ"),
        ("d͡ʑ", "ʥ"),
        ("ɲ", "n^"),
        ("ɕ", "ʃ"),
        ("ʷ", "w"),
        ("ɭ", "l`"),
        ("ʎ", "ɾ"),
        ("ɣ", "ŋ"),
        ("ɰ", "ɯ"),
        ("ʝ", "j"),
        ("ʌ", "ə"),
        ("ɡ", "g"),
        ("\u031a", "#"),
        ("\u0348", "="),
        ("\u031e", ""),
        ("\u0320", ""),
        ("\u0339", ""),
    ]
]


def fix_g2pk2_error(text):
    if "ㅇㅡㄹ " not in text and "ㄹㅡㄹ " not in text:
        return text
    new_text = ""
    i = 0
    while i < len(text) - 4:
        if (text[i : i + 3] == "ㅇㅡㄹ" or text[i : i + 3] == "ㄹㅡㄹ") and text[i + 3] == " " and text[i + 4] == "ㄹ":
            new_text += text[i : i + 3] + " " + "ㄴ"
            i += 5
        else:
            new_text += text[i]
            i += 1

    new_text += text[i:]
    return new_text


def latin_to_hangul(text):
    if not _ASCII_ALPHA_PATTERN.search(text):
        return text
    converted = []
    append = converted.append
    for char in str(text):
        replacement = _latin_char_to_hangul.get(char.lower()) if char.isascii() and char.isalpha() else None
        append(replacement if replacement is not None else char)
    return "".join(converted)


def divide_hangul(text):
    text = j2hcj(h2j(text))
    if not any(char in _hangul_divided_map for char in text):
        return text
    return "".join(_hangul_divided_map.get(char, char) for char in text)


def hangul_number(num, sino=True):
    """Reference https://github.com/Kyubyong/g2pK"""
    num = re.sub(",", "", num)

    if num == "0":
        return "영"
    if not sino and num == "20":
        return "스무"

    digits = "123456789"
    names = "일이삼사오육칠팔구"
    digit2name = {d: n for d, n in zip(digits, names)}

    modifiers = "한 두 세 네 다섯 여섯 일곱 여덟 아홉"
    decimals = "열 스물 서른 마흔 쉰 예순 일흔 여든 아흔"
    digit2mod = {d: mod for d, mod in zip(digits, modifiers.split())}
    digit2dec = {d: dec for d, dec in zip(digits, decimals.split())}

    spelledout = []
    for i, digit in enumerate(num):
        i = len(num) - i - 1
        if sino:
            if i == 0:
                name = digit2name.get(digit, "")
            elif i == 1:
                name = digit2name.get(digit, "") + "십"
                name = name.replace("일십", "십")
        else:
            if i == 0:
                name = digit2mod.get(digit, "")
            elif i == 1:
                name = digit2dec.get(digit, "")
        if digit == "0":
            if i % 4 == 0:
                last_three = spelledout[-min(3, len(spelledout)) :]
                if "".join(last_three) == "":
                    spelledout.append("")
                    continue
            else:
                spelledout.append("")
                continue
        if i == 2:
            name = digit2name.get(digit, "") + "백"
            name = name.replace("일백", "백")
        elif i == 3:
            name = digit2name.get(digit, "") + "천"
            name = name.replace("일천", "천")
        elif i == 4:
            name = digit2name.get(digit, "") + "만"
            name = name.replace("일만", "만")
        elif i == 5:
            name = digit2name.get(digit, "") + "십"
            name = name.replace("일십", "십")
        elif i == 6:
            name = digit2name.get(digit, "") + "백"
            name = name.replace("일백", "백")
        elif i == 7:
            name = digit2name.get(digit, "") + "천"
            name = name.replace("일천", "천")
        elif i == 8:
            name = digit2name.get(digit, "") + "억"
        elif i == 9:
            name = digit2name.get(digit, "") + "십"
        elif i == 10:
            name = digit2name.get(digit, "") + "백"
        elif i == 11:
            name = digit2name.get(digit, "") + "천"
        elif i == 12:
            name = digit2name.get(digit, "") + "조"
        elif i == 13:
            name = digit2name.get(digit, "") + "십"
        elif i == 14:
            name = digit2name.get(digit, "") + "백"
        elif i == 15:
            name = digit2name.get(digit, "") + "천"
        spelledout.append(name)
    return "".join(elem for elem in spelledout)


def number_to_hangul(text):
    """Reference https://github.com/Kyubyong/g2pK"""
    tokens = set(re.findall(r"(\d[\d,]*)([\uac00-\ud71f]+)", text))
    for token in tokens:
        num, classifier = token
        if classifier[:2] in _korean_classifiers or classifier[0] in _korean_classifiers:
            spelledout = hangul_number(num, sino=False)
        else:
            spelledout = hangul_number(num, sino=True)
        text = text.replace(f"{num}{classifier}", f"{spelledout}{classifier}")
    # digit by digit for remaining digits
    digits = "0123456789"
    names = "영일이삼사오육칠팔구"
    for d, n in zip(digits, names):
        text = text.replace(d, n)
    return text


def korean_to_lazy_ipa(text):
    text = latin_to_hangul(text)
    text = number_to_hangul(text)
    text = re.sub("[\uac00-\ud7af]+", lambda x: ko_pron.romanise(x.group(0), "ipa").split("] ~ [")[0], text)
    for regex, replacement in _ipa_to_lazy_ipa:
        text = re.sub(regex, replacement, text)
    return text


_g2p = OptimizedG2p()


def korean_to_ipa(text):
    text = latin_to_hangul(text)
    text = number_to_hangul(text)
    text = _g2p(text)
    text = fix_g2pk2_error(text)
    text = korean_to_lazy_ipa(text)
    return text.replace("ʧ", "tʃ").replace("ʥ", "dʑ")


def post_replace_ph(ph):
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
        " ": "空",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "停"
    return ph


def g2p(text):
    return _finalize_g2p_row(str(text))


@lru_cache(maxsize=8192)
def _g2p_cached(text: str):
    return tuple(_finalize_g2p_row(str(text)))


def _split_trailing_passthrough_suffix(text: str) -> tuple[str, str]:
    suffix_start = len(text)
    while suffix_start > 0 and text[suffix_start - 1] in _TRAILING_PASSTHROUGH_CHARS:
        suffix_start -= 1
    return text[:suffix_start], text[suffix_start:]


@lru_cache(maxsize=8192)
def _transform_core_to_jamo_text(core_text: str) -> str:
    value = latin_to_hangul(str(core_text))
    value = _g2p(value)
    value = divide_hangul(value)
    value = fix_g2pk2_error(value)
    return value


def _finalize_core_and_suffix(core_text: str, suffix_text: str) -> list[str]:
    if not core_text:
        return [post_replace_ph(char) for char in suffix_text]
    value = _transform_core_to_jamo_text(core_text)
    if not suffix_text:
        value = _TAIL_JAMO_PATTERN.sub(r"\1.", value)
    row = [post_replace_ph(char) for char in value]
    if suffix_text:
        row.extend(post_replace_ph(char) for char in suffix_text)
    return row


def _finalize_g2p_row(text: str) -> list[str]:
    core_text, suffix_text = _split_trailing_passthrough_suffix(str(text))
    return _finalize_core_and_suffix(core_text, suffix_text)


def g2p_batch(texts):
    normalized_texts = [str(text) for text in texts]
    if not normalized_texts:
        return []
    if len(normalized_texts) == 1:
        return [list(_g2p_cached(normalized_texts[0]))]
    rows: list[list[str] | None] = [None] * len(normalized_texts)
    core_indices: list[int | None] = [None] * len(normalized_texts)
    unique_core_order: list[str] = []
    core_index_by_text: dict[str, int] = {}
    text_parts: list[tuple[int, str]] = []

    for index, text in enumerate(normalized_texts):
        core_text, suffix_text = _split_trailing_passthrough_suffix(text)
        if not core_text:
            rows[index] = [post_replace_ph(char) for char in suffix_text]
            continue
        core_index = core_index_by_text.get(core_text)
        if core_index is None:
            core_index = len(unique_core_order)
            core_index_by_text[core_text] = core_index
            unique_core_order.append(core_text)
        core_indices[index] = core_index
        text_parts.append((index, suffix_text))

    core_rows = [_transform_core_to_jamo_text(core_text) for core_text in unique_core_order]
    for index, suffix_text in text_parts:
        core_index = int(core_indices[index])
        value = core_rows[core_index]
        if not suffix_text:
            value = _TAIL_JAMO_PATTERN.sub(r"\1.", value)
        row = [post_replace_ph(char) for char in value]
        if suffix_text:
            row.extend(post_replace_ph(char) for char in suffix_text)
        rows[index] = row
    return [row if row is not None else [] for row in rows]


if __name__ == "__main__":
    text = "안녕하세요"
    print(g2p(text))
