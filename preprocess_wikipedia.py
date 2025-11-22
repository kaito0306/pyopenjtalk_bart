# Wikipediaのダンプデータからテキストを抽出し、前処理を行うスクリプト

# https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja から引用・一部改変
from __future__ import unicode_literals
import re
import unicodedata

def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s, enable_remove_extra_spaces=True):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = re.sub('[″〝〟˝＂]', '”', s) # normalize double quotes


    # 半角のアルファベットを全角に変換
    s = s.translate(
        maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                  'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'))

    # 半角数字を全角数字に変換
    s = s.translate(
        maketrans('0123456789', '０１２３４５６７８９'))

    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    if enable_remove_extra_spaces:
        s = remove_extra_spaces(s)
    # s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    # s = re.sub('[’]', '\'', s)
    # s = re.sub('[”]', '"', s)
    return s


def split_and_noramlize_text(text, len_max=100):
    texts = text.split('\n')
    normalized_texts = []
    for t in texts:
        if t == '':
            continue
        if '。' not in t:
            normalized_texts.append(normalize_neologd(t))
            continue
        sents = t.split('。')
        current_output = ''
        for s in sents:
            if s == '':
                continue
            if current_output == '':
                current_output = s + '。'
            elif len(current_output) + len(s) < len_max:
                current_output += s + '。'
            else:
                normalized_texts.append(normalize_neologd(current_output))
                current_output = s + '。'
        if current_output != '':
            normalized_texts.append(normalize_neologd(current_output))
    return normalized_texts

##
import os
os.environ['HF_HOME'] = '/autofs/diamond3/share/cache/huggingface'

# load wikipedia dataset
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset
dataset = load_dataset('wikipedia', date='20240801', language='ja')

# preprocess
from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
from transformers import MBartForConditionalGeneration, AutoTokenizer

model_name = "ku-nlp/bart-base-japanese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

from tqdm import tqdm
import sys

def data_generator():
    # for i, example in tqdm(enumerate(dataset['train']), desc='Processing', file=sys.stdout, total=len(dataset['train']), ncols=0):
    for i, example in enumerate(dataset['train'].shuffle(seed=42).select(range(100_000))):
        text = example['text']
        example_id = example['id']
        normalized_texts = split_and_noramlize_text(text)
        for j, t in enumerate(normalized_texts):
            if len(t) <= 0 or len(t) > 300:
                continue
            tokens = tokenizer(t, max_length=512, padding="max_length", truncation=True)
            # print(f"{len(t):04d}, {len(tokens):04d}: {' '.join(tokens)}")
            phones = pyopenjtalk_g2p_prosody(t)
            phones_text = ' '.join(phones)
            normalized_phones_text = normalize_neologd(phones_text, enable_remove_extra_spaces=False)
            normalized_phones_tokens = tokenizer(normalized_phones_text, max_length=512, padding="max_length", truncation=True)
            # print(f"{len(normalized_phones_text):04d}, {len(normalized_phones_tokens):04d}: {' '.join(normalized_phones_tokens)}")
            # print(f"{example_id},{j},{t},{phones_text}")
            data = {'id': example_id, 
                    'sentence_id': j,
                    'text': t, 
                    'phones': phones_text,
                    'normalized_phones': normalized_phones_text,}
            data.update(tokens)
            data.update({'labels': normalized_phones_tokens['input_ids']})
            yield data

        # if i > 10:
        #    break

# build dataset
print("Building dataset...")
new_dataset = Dataset.from_generator(data_generator, cache_dir='./cache')
# new_dataset = IterableDataset.from_generator(data_generator)
print("Saving dataset...")
new_dataset.save_to_disk('./wikipedia_ja_20240801')
# new_dataset_dict = DatasetDict({'train': new_dataset})
# print("Saving dataset to disk...")
# new_dataset_dict.save_to_disk('./wikipedia_ja_20240801')
