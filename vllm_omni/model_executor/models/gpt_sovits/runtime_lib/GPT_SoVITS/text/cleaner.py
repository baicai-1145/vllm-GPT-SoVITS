from text import cleaned_text_to_sequence
import os
# if os.environ.get("version","v1")=="v1":
#     from text import chinese
#     from text.symbols import symbols
# else:
#     from text import chinese2 as chinese
#     from text.symbols2 import symbols

from text import symbols as symbols_v1
from text import symbols2 as symbols_v2

_SYMBOLS_SET_V1 = frozenset(symbols_v1.symbols)
_SYMBOLS_SET_V2 = frozenset(symbols_v2.symbols)

special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]


def clean_text(text, language, version=None):
    if version is None:
        version = os.environ.get("version", "v2")
    if version == "v1":
        symbols = symbols_v1.symbols
        symbols_set = _SYMBOLS_SET_V1
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english"}
    else:
        symbols = symbols_v2.symbols
        symbols_set = _SYMBOLS_SET_V2
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean", "yue": "cantonese"}

    if language not in language_module_map:
        language = "en"
        text = " "
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol, version)
    language_module = __import__("text." + language_module_map[language], fromlist=[language_module_map[language]])
    if hasattr(language_module, "text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text = text
    if language == "zh" or language == "yue":  ##########
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    elif language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [","] + phones
        word2ph = None
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None
    phones = ["UNK" if ph not in symbols_set else ph for ph in phones]
    return phones, word2ph, norm_text


def clean_text_batch(texts, language, version=None):
    if version is None:
        version = os.environ.get("version", "v2")
    if version == "v1":
        symbols_set = _SYMBOLS_SET_V1
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english"}
    else:
        symbols_set = _SYMBOLS_SET_V2
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean", "yue": "cantonese"}

    normalized_language = str(language)
    if normalized_language not in language_module_map:
        normalized_language = "en"
        texts = [" " for _ in texts]

    # Rare special-symbol branches keep the original single-item path for exact behavior.
    for special_s, special_l, _target_symbol in special:
        if normalized_language == special_l and any(special_s in str(text) for text in texts):
            return [clean_text(text, normalized_language, version) for text in texts]

    language_module = __import__(
        "text." + language_module_map[normalized_language],
        fromlist=[language_module_map[normalized_language]],
    )

    if hasattr(language_module, "text_normalize_batch"):
        norm_texts = list(language_module.text_normalize_batch([str(text) for text in texts]))
    elif hasattr(language_module, "text_normalize"):
        norm_texts = [language_module.text_normalize(str(text)) for text in texts]
    else:
        norm_texts = [str(text) for text in texts]

    rows = []
    if normalized_language == "zh" or normalized_language == "yue":
        return [clean_text(text, normalized_language, version) for text in texts]

    unique_norm_texts: List[str] = []
    unique_index_by_text = {}
    norm_text_indices: List[int] = []
    for norm_text in norm_texts:
        key = str(norm_text)
        index = unique_index_by_text.get(key)
        if index is None:
            index = len(unique_norm_texts)
            unique_index_by_text[key] = index
            unique_norm_texts.append(key)
        norm_text_indices.append(index)

    if hasattr(language_module, "g2p_batch"):
        unique_phone_rows = list(language_module.g2p_batch(unique_norm_texts))
    else:
        unique_phone_rows = [language_module.g2p(norm_text) for norm_text in unique_norm_texts]
    phone_rows = [unique_phone_rows[index] for index in norm_text_indices]

    for phones, norm_text in zip(phone_rows, norm_texts):
        mapped_phones = ["UNK" if ph not in symbols_set else ph for ph in phones]
        word2ph = None
        if normalized_language == "en" and len(mapped_phones) < 4:
            mapped_phones = [","] + mapped_phones
        rows.append((mapped_phones, word2ph, norm_text))
    return rows


def clean_special(text, language, special_s, target_symbol, version=None):
    if version is None:
        version = os.environ.get("version", "v2")
    if version == "v1":
        symbols_set = _SYMBOLS_SET_V1
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english"}
    else:
        symbols_set = _SYMBOLS_SET_V2
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean", "yue": "cantonese"}

    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = __import__("text." + language_module_map[language], fromlist=[language_module_map[language]])
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols_set
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def text_to_sequence(text, language, version=None):
    version = os.environ.get("version", version)
    if version is None:
        version = "v2"
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones, version)


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
