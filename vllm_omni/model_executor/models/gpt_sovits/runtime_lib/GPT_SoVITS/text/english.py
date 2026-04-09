import os
import pickle
import re
from builtins import str as unicode
from functools import lru_cache

import nltk
import wordsegment
from g2p_en import G2p
from nltk.tokenize import TweetTokenizer

from text.en_normalization.expend import normalize
from text.symbols import punctuation
from text.symbols2 import symbols

word_tokenize = TweetTokenizer().tokenize
from nltk import pos_tag, pos_tag_sents

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CMU_DICT_FAST_PATH = os.path.join(current_file_path, "cmudict-fast.rep")
CMU_DICT_HOT_PATH = os.path.join(current_file_path, "engdict-hot.rep")
CACHE_PATH = os.path.join(current_file_path, "engdict_cache.pickle")
NAMECACHE_PATH = os.path.join(current_file_path, "namedict_cache.pickle")


def _ensure_nltk_data_paths() -> None:
    candidate_roots: list[str] = []
    env_value = str(os.environ.get("GPTSOVITS_NLTK_DATA_DIRS", os.environ.get("NLTK_DATA", ""))).strip()
    if env_value:
        candidate_roots.extend([item.strip() for item in env_value.split(os.pathsep) if item and item.strip()])
    candidate_roots.extend(
        [
            "/root/miniconda3/envs/GPTSoVits/nltk_data",
            os.path.join(os.path.dirname(current_file_path), "nltk_data"),
            os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "nltk_data"),
        ]
    )
    seen: set[str] = set()
    for candidate in candidate_roots:
        resolved = os.path.abspath(os.path.expanduser(candidate))
        if resolved in seen or not os.path.isdir(resolved):
            continue
        seen.add(resolved)
        if resolved not in nltk.data.path:
            nltk.data.path.insert(0, resolved)


_ensure_nltk_data_paths()


# 适配中文及 g2p_en 标点
rep_map = {
    "[;:：，；]": ",",
    '["’]': "'",
    "。": ".",
    "！": "!",
    "？": "?",
}

_TEXT_NORMALIZE_PATTERN = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
_TRAILING_PASSTHROUGH_CHARS = frozenset(" \n：；，。！？·、,.!?")
_SIMPLE_WORD_PATTERN = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)?$")


arpa = {
    "AH0",
    "S",
    "AH1",
    "EY2",
    "AE2",
    "EH0",
    "OW2",
    "UH0",
    "NG",
    "B",
    "G",
    "AY0",
    "M",
    "AA0",
    "F",
    "AO0",
    "ER2",
    "UH1",
    "IT1",
    "AH2",
    "DH",
    "IT0",
    "EY1",
    "IH0",
    "K",
    "N",
    "W",
    "IT2",
    "T",
    "AA1",
    "ER1",
    "EH2",
    "OY0",
    "UH2",
    "UW1",
    "Z",
    "AW2",
    "AW1",
    "V",
    "UW2",
    "AA2",
    "ER",
    "AW0",
    "UW0",
    "R",
    "OW1",
    "EH1",
    "ZH",
    "AE0",
    "IH2",
    "IH",
    "Y",
    "JH",
    "P",
    "AY1",
    "EY0",
    "OY2",
    "TH",
    "HH",
    "D",
    "ER0",
    "CH",
    "AO1",
    "AE1",
    "AO2",
    "OY1",
    "AY2",
    "IH1",
    "OW0",
    "L",
    "SH",
}


def replace_phs(phs):
    rep_map = {"'": "-"}
    phs_new = []
    for ph in phs:
        if ph in symbols:
            phs_new.append(ph)
        elif ph in rep_map.keys():
            phs_new.append(rep_map[ph])
        else:
            print("ph not in symbols: ", ph)
    return phs_new


def replace_consecutive_punctuation(text):
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = rf"([{punctuations}\s])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0].lower()

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def read_dict_new():
    g2p_dict = {}
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 57:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0].lower()
                g2p_dict[word] = [word_split[1].split(" ")]

            line_index = line_index + 1
            line = f.readline()

    with open(CMU_DICT_FAST_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0].lower()
                if word not in g2p_dict:
                    g2p_dict[word] = [word_split[1:]]

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def hot_reload_hot(g2p_dict):
    with open(CMU_DICT_HOT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0].lower()
                # 自定义发音词直接覆盖字典
                g2p_dict[word] = [word_split[1:]]

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict_new()
        cache_dict(g2p_dict, CACHE_PATH)

    g2p_dict = hot_reload_hot(g2p_dict)

    return g2p_dict


def get_namedict():
    if os.path.exists(NAMECACHE_PATH):
        with open(NAMECACHE_PATH, "rb") as pickle_file:
            name_dict = pickle.load(pickle_file)
    else:
        name_dict = {}

    return name_dict


@lru_cache(maxsize=8192)
def _text_normalize_cached(text: str):
    text = _TEXT_NORMALIZE_PATTERN.sub(lambda x: rep_map[x.group()], str(text))
    text = unicode(text)
    text = normalize(text)
    return replace_consecutive_punctuation(text)


def text_normalize(text):
    return _text_normalize_cached(str(text))


def text_normalize_batch(texts):
    return [text_normalize(text) for text in texts]


class en_G2p(G2p):
    def __init__(self):
        super().__init__()
        # 分词初始化
        wordsegment.load()

        # 扩展过时字典, 添加姓名字典
        self.cmu = get_dict()
        self.namedict = get_namedict()

        # 剔除读音错误的几个缩写
        for word in ["AE", "AI", "AR", "IOS", "HUD", "OS"]:
            del self.cmu[word.lower()]

        # 修正多音字
        self.homograph2features["read"] = (["R", "IT1", "D"], ["R", "EH1", "D"], "VBP")
        self.homograph2features["complex"] = (
            ["K", "AH0", "M", "P", "L", "EH1", "K", "S"],
            ["K", "AA1", "M", "P", "L", "EH0", "K", "S"],
            "JJ",
        )

    def __call__(self, text):
        words = word_tokenize(text)
        return self._phonemize_tagged_tokens(pos_tag(words))

    def _phonemize_tagged_tokens(self, tokens):
        prons = []
        for o_word, pos in tokens:
            # 还原 g2p_en 小写操作逻辑
            word = o_word.lower()

            if re.search("[a-z]", word) is None:
                pron = [word]
            # 先把单字母推出去
            elif len(word) == 1:
                # 单读 A 发音修正, 这里需要原格式 o_word 判断大写
                if o_word == "A":
                    pron = ["EY1"]
                else:
                    pron = self.cmu[word][0]
            # g2p_en 原版多音字处理
            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                # pos1比pos长仅出现在read
                elif len(pos) < len(pos1) and pos == pos1[: len(pos)]:
                    pron = pron1
                else:
                    pron = pron2
            else:
                # 递归查找预测
                pron = self.qryword(o_word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]

    def batch(self, texts):
        tokenized_sentences = [word_tokenize(text) for text in texts]
        tagged_sentences = pos_tag_sents(tokenized_sentences)
        return [self._phonemize_tagged_tokens(tokens) for tokens in tagged_sentences]

    def qryword(self, o_word):
        word = o_word.lower()

        # 查字典, 单字母除外
        if len(word) > 1 and word in self.cmu:  # lookup CMU dict
            return self.cmu[word][0]

        # 单词仅首字母大写时查找姓名字典
        if o_word.istitle() and word in self.namedict:
            return self.namedict[word][0]

        # oov 长度小于等于 3 直接读字母
        if len(word) <= 3:
            phones = []
            for w in word:
                # 单读 A 发音修正, 此处不存在大写的情况
                if w == "a":
                    phones.extend(["EY1"])
                elif not w.isalpha():
                    phones.extend([w])
                else:
                    phones.extend(self.cmu[w][0])
            return phones

        # 尝试分离所有格
        if re.match(r"^([a-z]+)('s)$", word):
            phones = self.qryword(word[:-2])[:]
            # P T K F TH HH 无声辅音结尾 's 发 ['S']
            if phones[-1] in ["P", "T", "K", "F", "TH", "HH"]:
                phones.extend(["S"])
            # S Z SH ZH CH JH 擦声结尾 's 发 ['IH1', 'Z'] 或 ['AH0', 'Z']
            elif phones[-1] in ["S", "Z", "SH", "ZH", "CH", "JH"]:
                phones.extend(["AH0", "Z"])
            # B D G DH V M N NG L R W Y 有声辅音结尾 's 发 ['Z']
            # AH0 AH1 AH2 EY0 EY1 EY2 AE0 AE1 AE2 EH0 EH1 EH2 OW0 OW1 OW2 UH0 UH1 UH2 IT0 IT1 IT2 AA0 AA1 AA2 AO0 AO1 AO2
            # ER ER0 ER1 ER2 UW0 UW1 UW2 AY0 AY1 AY2 AW0 AW1 AW2 OY0 OY1 OY2 IH IH0 IH1 IH2 元音结尾 's 发 ['Z']
            else:
                phones.extend(["Z"])
            return phones

        # 尝试进行分词，应对复合词
        comps = wordsegment.segment(word.lower())

        # 无法分词的送回去预测
        if len(comps) == 1:
            return self.predict(word)

        # 可以分词的递归处理
        return [phone for comp in comps for phone in self.qryword(comp)]


_g2p = en_G2p()


def _postprocess_phone_list(phone_list):
    phones = [ph if ph != "<unk>" else "UNK" for ph in phone_list if ph not in [" ", "<pad>", "UW", "</s>", "<s>"]]
    return tuple(replace_phs(phones))


def _suffix_to_phones(suffix_text: str):
    if not suffix_text:
        return ()
    phones = []
    for char in str(suffix_text):
        if char == " ":
            continue
        phones.extend(replace_phs([char]))
    return tuple(phones)


@lru_cache(maxsize=8192)
def _g2p_cached(text: str):
    return _postprocess_phone_list(_g2p(str(text)))


def _split_trailing_passthrough_suffix(text: str):
    suffix_start = len(text)
    while suffix_start > 0 and text[suffix_start - 1] in _TRAILING_PASSTHROUGH_CHARS:
        suffix_start -= 1
    return text[:suffix_start], text[suffix_start:]


def _is_simple_word_core(text: str):
    return bool(_SIMPLE_WORD_PATTERN.fullmatch(str(text)))


@lru_cache(maxsize=8192)
def _g2p_simple_word_cached(text: str):
    return _postprocess_phone_list(_g2p.qryword(str(text)))


def g2p(text):
    return list(_g2p_cached(str(text)))


def g2p_batch(texts):
    normalized_texts = [str(text) for text in texts]
    if not normalized_texts:
        return []
    if len(normalized_texts) == 1:
        return [list(_g2p_cached(normalized_texts[0]))]
    rows = [None] * len(normalized_texts)
    hard_indices = []
    hard_core_texts = []
    hard_cached_rows = {}

    for index, text in enumerate(normalized_texts):
        core_text, suffix_text = _split_trailing_passthrough_suffix(text)
        suffix_phones = _suffix_to_phones(suffix_text)
        if not core_text:
            rows[index] = list(suffix_phones)
            continue
        if _is_simple_word_core(core_text) and core_text.lower() not in _g2p.homograph2features:
            rows[index] = list(_g2p_simple_word_cached(core_text) + suffix_phones)
            continue
        cached_core = hard_cached_rows.get(core_text)
        if cached_core is None:
            hard_cached_rows[core_text] = ()
            hard_indices.append(index)
            hard_core_texts.append(core_text)
            continue
        if cached_core:
            rows[index] = list(cached_core + suffix_phones)
            continue
        hard_indices.append(index)
        hard_core_texts.append(core_text)

    if hard_core_texts:
        unique_hard_cores = []
        unique_core_to_pos = {}
        hard_index_to_unique_pos = []
        for core_text in hard_core_texts:
            unique_pos = unique_core_to_pos.get(core_text)
            if unique_pos is None:
                unique_pos = len(unique_hard_cores)
                unique_core_to_pos[core_text] = unique_pos
                unique_hard_cores.append(core_text)
            hard_index_to_unique_pos.append(unique_pos)
        hard_phone_rows = [_postprocess_phone_list(phone_list) for phone_list in _g2p.batch(unique_hard_cores)]
        for core_text, unique_pos in unique_core_to_pos.items():
            hard_cached_rows[core_text] = hard_phone_rows[unique_pos]
        for index, unique_pos in zip(hard_indices, hard_index_to_unique_pos):
            core_text, suffix_text = _split_trailing_passthrough_suffix(normalized_texts[index])
            rows[index] = list(hard_phone_rows[unique_pos] + _suffix_to_phones(suffix_text))

    return [row if row is not None else [] for row in rows]


if __name__ == "__main__":
    print(g2p("hello"))
    print(g2p(text_normalize("e.g. I used openai's AI tool to draw a picture.")))
    print(g2p(text_normalize("In this; paper, we propose 1 DSPGAN, a GAN-based universal vocoder.")))
