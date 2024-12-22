# 1. 读取数据
# 2. 分词器 tokenizer
# 3. 构建词汇表 vocabulary
# 4. 预处理 preprocess
# 5. 构建数据集 dataset

import torch
import re

with open(r'C:\Users\steve\Desktop\xz\transformer\corpus.en_ru.1m.en', 'r', encoding='utf-8') as f:
    en = f.readlines()

with open(r'C:\Users\steve\Desktop\xz\transformer\corpus.en_ru.1m.ru', 'r', encoding='utf-8') as f:
    ru = f.readlines()

train_dataset = list(zip(ru, en))

# 2. 分词器 tokenizer

def tokenizer(text):
    # 标点加空格
    text = re.sub(r'([.,!?()"\'-])', r' \1 ', text)
    # 省略号
    text = re.sub(r'\.{3}', ' ... ', text)
    # 多个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip().split()

en_token = []
ru_token = []

for ru_sentence, en_sentence in train_dataset:
    ru_token.append(tokenizer(ru_sentence))
    en_token.append(tokenizer(en_sentence))


# 3. 构建词汇表 vocabulary

UNK_SYM = '<unk>'
PAD_SYM = '<pad>'
BOS_SYM = '<bos>'
EOS_SYM = '<eos>'
UNK_NUM = 0
PAD_NUM = 1
BOS_NUM = 2
EOS_NUM = 3

class Vocabulary:
    def __init__(self, specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM]):
        self.itos = specials[:]  # index to string
        self.stoi = {token: idx for idx, token in enumerate(specials)}  # string to index
    
    def build_vocab(self, token_lists):
        # 统计所有词频
        word_freq = {}
        for tokens in token_lists:
            for token in tokens:
                if token not in self.stoi:  # 不统计特殊词
                    word_freq[token] = word_freq.get(token, 0) + 1
        # 排序
        word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        # 构建词汇表
        for token, freq in word_freq:
            self.itos.append(token)
            self.stoi[token] = len(self.itos) - 1

    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, idx):
        return self.itos[idx]

    def __call__(self, tokens_list):
        # 处理嵌套列表的情况
        if isinstance(tokens_list[0], list):
            return [[self.stoi.get(token, self.stoi[UNK_SYM]) for token in tokens] 
                    for tokens in tokens_list]
        # 处理单个列表的情况
        return [self.stoi.get(token, self.stoi[UNK_SYM]) for token in tokens_list]

    def lookup_tokens(self, ids):
        return [self.itos[id] for id in ids]

en_vocab = Vocabulary()
ru_vocab = Vocabulary()
en_vocab.build_vocab(en_token)
ru_vocab.build_vocab(ru_token)

# 4. 预处理 preprocess

def ru_preprocess(ru_sentence, ru_vocab):
    tokens = tokenizer(ru_sentence)
    tokens = [BOS_SYM] + tokens + [EOS_SYM]
    ids = ru_vocab(tokens)
    return tokens, ids

def en_preprocess(en_sentence, en_vocab):
    tokens = tokenizer(en_sentence)
    tokens = [BOS_SYM] + tokens + [EOS_SYM]
    ids = en_vocab(tokens)
    return tokens, ids

if __name__ == '__main__':


    print(train_dataset[0][0])
    print(train_dataset[0][1])
        
    print(ru_preprocess(train_dataset[0][0], ru_vocab))
    print(en_preprocess(train_dataset[0][1], en_vocab))


