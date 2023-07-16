'''
作业描述
-英汉（繁体）翻译
-输入：一个英语句子（例如tom is a student）
-输出：中文翻译（例如湯姆 是 個 學生。)

-TODO
-训练一个简单的RNN seq2seq以实现高效翻译
-切换到transformer model 以提高性能
-应用Back-translation 以进一步提高性能
'''
import sys
import pdb
import pprint
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from argparse import Namespace
from fairseq import utils

import matplotlib.pyplot as plt

#固定随机种子
seed = 73
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#数据集
'''
双语平行语料库2020年:
原始：398066（句子）
已处理：393980（句）
测试数据:
大小：4000（句子）
中文翻译未公开。提供的（.zh）文件是psuedo翻译，每行都是一个“。”
'''
#数据集下载
#数据集有两个压缩包：ted2020.tgz(包含raw.en, raw.zh两个文件)，test.tgz(包含test.en, test.zh两个文件)。
data_dir = './DATA/rawdata'   #解压后的文件全部放到一层目录中，目录位置是："DATA/rawdata/ted2020"
dataset_name = 'ted2020'
urls = (
    "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/ted2020.tgz",
    "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/test.tgz",
)
file_names = (
    'ted2020.tgz', # train & dev
    'test.tgz', # test
)
prefix = Path(data_dir).absolute() / dataset_name

prefix.mkdir(parents=True, exist_ok=True)


#语言
src_lang = 'en'
tgt_lang = 'zh'

data_prefix = f'{prefix}/train_dev.raw'
test_prefix = f'{prefix}/test.raw'
# 预处理文件
import re


def strQ2B(ustring):
    """Full width -> half width"""
    # reference:https://ithelp.ithome.com.tw/articles/10233122
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全宽空间：直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全宽字符（空格除外）转换
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)  # 将字符串ss里面的内容以‘’分隔


# 去掉或者替换一些特殊字符
def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s)  # remove ([text])
        s = s.replace('-', '')  # remove '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s)  # 保留标点符号
    elif lang == 'zh':
        s = strQ2B(s)  # Q2B
        s = re.sub(r"\([^()]*\)", "", s)  # remove ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s)  # keep punctuation
    s = ' '.join(s.strip().split())
    return s


def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split())


# clean后的文件名称是，train_dev.raw.clean.en, train_dev.raw.clean.zh, test.raw.clean.en, test.raw.clean.zh。
def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return
    with open(f'{prefix}.{l1}', 'r',encoding='utf-8') as l1_in_f:
        with open(f'{prefix}.{l2}', 'r',encoding='utf-8') as l2_in_f:
            with open(f'{prefix}.clean.{l1}', 'w',encoding='utf-8') as l1_out_f:
                with open(f'{prefix}.clean.{l2}', 'w',encoding='utf-8') as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_s(s1, l1)
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0:  # remove short sentence
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0:  # remove long sentence
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0:  # remove by ratio of length
                            if s1_len / s2_len > ratio or s2_len / s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)


clean_corpus(data_prefix, src_lang, tgt_lang)
clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)
print(torch.__version__)
#分成train/valid
#划分训练集和验证集，train_dev.raw.clean.en和train_dev.clean.zh被分成train.clean.en, valid.clean.en和train.clean.zh, train.clean.zh。
valid_ratio = 0.01 # 3000~4000就足够了
train_ratio = 1 - valid_ratio

if (prefix/f'train.clean.{src_lang}').exists() \
and (prefix/f'train.clean.{tgt_lang}').exists() \
and (prefix/f'valid.clean.{src_lang}').exists() \
and (prefix/f'valid.clean.{tgt_lang}').exists():
    print(f'train/valid splits exists. skipping split.')
else:
    line_num = sum(1 for line in open(f'{data_prefix}.clean.{src_lang}',encoding='utf-8'))
    labels = list(range(line_num))
    random.shuffle(labels)
    for lang in [src_lang, tgt_lang]:
        train_f = open(os.path.join(data_dir, dataset_name, f'train.clean.{lang}'), 'w',encoding='utf-8')
        valid_f = open(os.path.join(data_dir, dataset_name, f'valid.clean.{lang}'), 'w',encoding='utf-8')
        count = 0
        for line in open(f'{data_prefix}.clean.{lang}', 'r',encoding='utf-8'):
            if labels[count]/line_num < train_ratio:
                train_f.write(line)
            else:
                valid_f.write(line)
            count += 1
        train_f.close()
        valid_f.close()

'''
Subword Units
词汇表外（OOV）一直是机器翻译中的一个主要问题。这可以通过使用子字单元来缓解。
我们将使用句子包,选择“unigram”或“字节对编码（BPE）”算法

分词
使用sentencepiece中的spm对训练集和验证集进行分词建模，模型名称是spm8000.model，同时产生词汇库spm8000.vocab。
使用模型对训练集、验证集、以及测试集进行分词处理，得到文件train.en, train.zh, valid.en, valid.zh, test.en, test.zh。
'''
import sentencepiece as spm
vocab_size = 8000
if (prefix/f'spm{vocab_size}.model').exists():
    print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
else:
    spm.SentencePieceTrainer.train(
        input=','.join([f'{prefix}/train.clean.{src_lang}',
                        f'{prefix}/valid.clean.{src_lang}',
                        f'{prefix}/train.clean.{tgt_lang}',
                        f'{prefix}/valid.clean.{tgt_lang}']),
        model_prefix=prefix/f'spm{vocab_size}',
        vocab_size=vocab_size,
        character_coverage=1,
        model_type='unigram', # 'bpe' works as well
        input_sentence_size=1e6,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
    )

spm_model = spm.SentencePieceProcessor(model_file=str(prefix/f'spm{vocab_size}.model'))
in_tag = {
    'train': 'train.clean',
    'valid': 'valid.clean',
    'test': 'test.raw.clean',
}
for split in ['train', 'valid', 'test']:
    for lang in [src_lang, tgt_lang]:
        out_path = prefix/f'{split}.{lang}'
        if out_path.exists():
            print(f"{out_path} exists. skipping spm_encode.")
        else:
            with open(prefix/f'{split}.{lang}', 'w',encoding='utf-8') as out_f:
                with open(prefix/f'{in_tag[split]}.{lang}', 'r',encoding='utf-8') as in_f:
                    for line in in_f:
                        line = line.strip()#用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                        tok = spm_model.encode(line, out_type=str)
                        print(' '.join(tok), file=out_f)#token是" "，加入句子




