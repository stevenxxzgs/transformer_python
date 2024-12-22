# unsqueeze(0) 在第0维上增加一个维度
# unsqueeze(1) 在第1维上增加一个维度
# unsqueeze(-1) 在最后一维上增加一个维度

# squeeze(0) 在第0维上减少一个维度

# arange(start, end, step) 生成从start到end，步长为step的等差数列

# a[:, 0::2] 表示从第0维开始，每隔2个元素取一个元素

# a[:, 1::2] 表示从第1维开始，每隔2个元素取一个元素

from torch import nn
import torch
from dataset import ru_vocab, ru_preprocess, train_dataset
import math

class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, emb_size, dropout=0.3, seq_max_len=5000):
        super().__init__()
        self.seq_emb = nn.Embedding(vocab_size, emb_size)
        # print(self.seq_emb)
        position_idx = torch.arange(0, seq_max_len, dtype=torch.float).unsqueeze(-1)
        # print(position_idx.shape)
        
        position_emb_fill = position_idx * torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000.0) / emb_size)
        # print(position_emb_fill.shape)
        pos_encoding = torch.zeros(seq_max_len, emb_size)
        pos_encoding[:, 0::2] = torch.sin(position_emb_fill)
        pos_encoding[:, 1::2] = torch.cos(position_emb_fill)
        # print(pos_encoding.shape)
        # print(pos_encoding)
        self.register_buffer('pos_encoding', pos_encoding)
        # self.pos_encoding = pos_encoding

        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # x : [batch_size, seq_len]
        x = self.seq_emb(x) # x : [batch_size, seq_len, emb_size]
        x = x + self.pos_encoding.unsqueeze(0)[:, :x.size(1), :] # x : [batch_size, seq_len, emb_size]
        return self.dropout(x)

if __name__ == '__main__':
    emb = EmbeddingWithPosition(vocab_size=len(ru_vocab), emb_size=128)
    # 把句子转序列ID
    ru_tokens, ru_ids = ru_preprocess(train_dataset[0][0], ru_vocab)
    # 把序列ID转tensor
    ru_ids_tensor = torch.tensor(ru_ids, dtype=torch.long)
    # 把tensor送入embedding层
    emb_result = emb(ru_ids_tensor.unsqueeze(0))

