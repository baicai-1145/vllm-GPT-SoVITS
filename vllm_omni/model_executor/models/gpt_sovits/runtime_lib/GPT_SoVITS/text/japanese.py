# modified from https://github.com/CjangCjengh/vits/blob/main/text/japanese.py
import re
import os
import hashlib
from functools import lru_cache

try:
    import pyopenjtalk

    current_file_path = os.path.dirname(__file__)

    # 防止win下无法读取模型
    if os.name == "nt":
        python_dir = os.getcwd()
        OPEN_JTALK_DICT_DIR = pyopenjtalk.OPEN_JTALK_DICT_DIR.decode("utf-8")
        if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", OPEN_JTALK_DICT_DIR)):
            if OPEN_JTALK_DICT_DIR[: len(python_dir)].upper() == python_dir.upper():
                OPEN_JTALK_DICT_DIR = os.path.join(os.path.relpath(OPEN_JTALK_DICT_DIR, python_dir))
            else:
                import shutil

                if not os.path.exists("TEMP"):
                    os.mkdir("TEMP")
                if not os.path.exists(os.path.join("TEMP", "ja")):
                    os.mkdir(os.path.join("TEMP", "ja"))
                if os.path.exists(os.path.join("TEMP", "ja", "open_jtalk_dic")):
                    shutil.rmtree(os.path.join("TEMP", "ja", "open_jtalk_dic"))
                shutil.copytree(
                    pyopenjtalk.OPEN_JTALK_DICT_DIR.decode("utf-8"),
                    os.path.join("TEMP", "ja", "open_jtalk_dic"),
                )
                OPEN_JTALK_DICT_DIR = os.path.join("TEMP", "ja", "open_jtalk_dic")
            pyopenjtalk.OPEN_JTALK_DICT_DIR = OPEN_JTALK_DICT_DIR.encode("utf-8")

        if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", current_file_path)):
            if current_file_path[: len(python_dir)].upper() == python_dir.upper():
                current_file_path = os.path.join(os.path.relpath(current_file_path, python_dir))
            else:
                if not os.path.exists("TEMP"):
                    os.mkdir("TEMP")
                if not os.path.exists(os.path.join("TEMP", "ja")):
                    os.mkdir(os.path.join("TEMP", "ja"))
                if not os.path.exists(os.path.join("TEMP", "ja", "ja_userdic")):
                    os.mkdir(os.path.join("TEMP", "ja", "ja_userdic"))
                    shutil.copyfile(
                        os.path.join(current_file_path, "ja_userdic", "userdict.csv"),
                        os.path.join("TEMP", "ja", "ja_userdic", "userdict.csv"),
                    )
                current_file_path = os.path.join("TEMP", "ja")

    def get_hash(fp: str) -> str:
        hash_md5 = hashlib.md5()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    USERDIC_CSV_PATH = os.path.join(current_file_path, "ja_userdic", "userdict.csv")
    USERDIC_BIN_PATH = os.path.join(current_file_path, "ja_userdic", "user.dict")
    USERDIC_HASH_PATH = os.path.join(current_file_path, "ja_userdic", "userdict.md5")
    # 如果没有用户词典，就生成一个；如果有，就检查md5，如果不一样，就重新生成
    if os.path.exists(USERDIC_CSV_PATH):
        if (
            not os.path.exists(USERDIC_BIN_PATH)
            or get_hash(USERDIC_CSV_PATH) != open(USERDIC_HASH_PATH, "r", encoding="utf-8").read()
        ):
            pyopenjtalk.mecab_dict_index(USERDIC_CSV_PATH, USERDIC_BIN_PATH)
            with open(USERDIC_HASH_PATH, "w", encoding="utf-8") as f:
                f.write(get_hash(USERDIC_CSV_PATH))

    if os.path.exists(USERDIC_BIN_PATH):
        pyopenjtalk.update_global_jtalk_with_user_dict(USERDIC_BIN_PATH)
except Exception:
    # print(e)
    import pyopenjtalk

    # failed to load user dictionary, ignore.
    pass


from text.symbols import punctuation

# Regular expression matching Japanese without punctuation marks:
_japanese_characters = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# Regular expression matching non-Japanese characters or punctuation marks:
_japanese_marks = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# List of (symbol, Japanese) pairs for marks:
_symbols_to_japanese = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]
_TRAILING_PASSTHROUGH_CHARS = frozenset(" \n：；，。！？·、,.!?")


# List of (consonant, sokuon) pairs:
_real_sokuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"Q([↑↓]*[kg])", r"k#\1"),
        (r"Q([↑↓]*[tdjʧ])", r"t#\1"),
        (r"Q([↑↓]*[sʃ])", r"s\1"),
        (r"Q([↑↓]*[pb])", r"p#\1"),
    ]
]

# List of (consonant, hatsuon) pairs:
_real_hatsuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"N([↑↓]*[pbm])", r"m\1"),
        (r"N([↑↓]*[ʧʥj])", r"n^\1"),
        (r"N([↑↓]*[tdn])", r"n\1"),
        (r"N([↑↓]*[kg])", r"ŋ\1"),
    ]
]


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
    }

    if ph in rep_map.keys():
        ph = rep_map[ph]
    return ph


def replace_consecutive_punctuation(text):
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def symbols_to_japanese(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text


def preprocess_jap(text, with_prosody=False):
    """Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html"""
    text = symbols_to_japanese(text)
    # English words to lower case, should have no influence on japanese words.
    text = text.lower()
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)
    text = []
    for i, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            if with_prosody:
                text += pyopenjtalk_g2p_prosody(sentence)[1:-1]
            else:
                p = pyopenjtalk.g2p(sentence)
                text += p.split(" ")

        if i < len(marks):
            if marks[i] == " ":  # 防止意外的UNK
                continue
            text += [marks[i].replace(" ", "")]
    return text


def preprocess_jap_batch(texts, with_prosody=False):
    to_japanese = symbols_to_japanese
    split_marks = _japanese_marks
    match_chars = _japanese_characters.match
    post_marks = post_replace_ph
    use_prosody = bool(with_prosody)
    if use_prosody:
        phonemize_sentence = lambda sentence: pyopenjtalk_g2p_prosody(sentence)[1:-1]
    else:
        phonemize_sentence = lambda sentence: pyopenjtalk.g2p(sentence).split(" ")

    rows = []
    pending_sentence_refs = []
    unique_sentences = []
    sentence_to_index = {}
    for raw_text in texts:
        text = to_japanese(str(raw_text)).lower()
        sentences = re.split(split_marks, text)
        marks = re.findall(split_marks, text)
        row = []
        for index, sentence in enumerate(sentences):
            if match_chars(sentence):
                sentence_index = sentence_to_index.get(sentence)
                if sentence_index is None:
                    sentence_index = len(unique_sentences)
                    sentence_to_index[sentence] = sentence_index
                    unique_sentences.append(sentence)
                pending_sentence_refs.append((row, sentence_index))
            if index < len(marks):
                if marks[index] == " ":
                    continue
                row.append(post_marks(marks[index].replace(" ", "")))
        rows.append(row)
    sentence_rows = []
    if unique_sentences:
        sentence_rows = [phonemize_sentence(sentence) for sentence in unique_sentences]
    for row, sentence_index in pending_sentence_refs:
        row.extend(sentence_rows[sentence_index])
    return rows


def _suffix_to_phones(suffix_text):
    if not suffix_text:
        return ()
    phones = []
    for char in str(suffix_text):
        if char == " ":
            continue
        phones.append(post_replace_ph(char.replace(" ", "")))
    return tuple(phones)


def _split_trailing_passthrough_suffix(text: str):
    suffix_start = len(text)
    while suffix_start > 0 and text[suffix_start - 1] in _TRAILING_PASSTHROUGH_CHARS:
        suffix_start -= 1
    return text[:suffix_start], text[suffix_start:]


def text_normalize(text):
    # todo: jap text normalize

    # 避免重复标点引起的参考泄露
    text = replace_consecutive_punctuation(text)
    return text


def text_normalize_batch(texts):
    return [text_normalize(text) for text in texts]


# Copied from espnet https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
def pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True):
    """Extract phoneme + prosoody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104

    """
    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)
        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones


# Copied from espnet https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


@lru_cache(maxsize=8192)
def _g2p_cached(norm_text: str, with_prosody: bool = True):
    phones = preprocess_jap(norm_text, with_prosody)
    return tuple(post_replace_ph(i) for i in phones)


def g2p(norm_text, with_prosody=True):
    return list(_g2p_cached(str(norm_text), bool(with_prosody)))


def g2p_batch(norm_texts, with_prosody=True):
    normalized_texts = [str(norm_text) for norm_text in norm_texts]
    if not normalized_texts:
        return []
    if len(normalized_texts) == 1:
        return [list(_g2p_cached(normalized_texts[0], bool(with_prosody)))]
    rows = [None] * len(normalized_texts)
    unique_core_texts = []
    core_to_pos = {}
    use_prosody = bool(with_prosody)
    pending_indices = []

    for index, text in enumerate(normalized_texts):
        core_text, suffix_text = _split_trailing_passthrough_suffix(text)
        if not core_text:
            rows[index] = list(_suffix_to_phones(suffix_text))
            continue
        core_pos = core_to_pos.get(core_text)
        if core_pos is None:
            core_pos = len(unique_core_texts)
            core_to_pos[core_text] = core_pos
            unique_core_texts.append(core_text)
        pending_indices.append(index)

    core_rows = []
    if unique_core_texts:
        core_rows = [
            tuple(post_replace_ph(phone) for phone in row)
            for row in preprocess_jap_batch(unique_core_texts, with_prosody=use_prosody)
        ]
    for index in pending_indices:
        core_text, suffix_text = _split_trailing_passthrough_suffix(normalized_texts[index])
        rows[index] = list(core_rows[core_to_pos[core_text]] + _suffix_to_phones(suffix_text))
    return [row if row is not None else [] for row in rows]


if __name__ == "__main__":
    phones = g2p("Hello.こんにちは！今日もNiCe天気ですね！tokyotowerに行きましょう！")
    print(phones)
