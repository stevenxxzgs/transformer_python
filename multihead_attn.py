from torch import nn
import torch
from emb import EmbeddingWithPosition
from dataset import train_dataset, ru_vocab, ru_preprocess
import math

class MultiheadAttention(nn.Module):
    def __init__(self, emb_size, q_k_size, v_size, head, dropout=0.3):
        super().__init__() # 调用父类的构造函数，nn.Module
        self.emb_size = emb_size
        self.q_k_size = q_k_size
        self.v_size = v_size
        self.head = head

        self.w_q = nn.Linear(emb_size, head*q_k_size)
        self.w_k = nn.Linear(emb_size, head*q_k_size)
        self.w_v = nn.Linear(emb_size, head*v_size)

    def forward(self, key_value, query, attn_mask):
        # [batch_size, seq_len, emb_size]
        ####### !!!!!!!!!!!!!!犯病直接把权重乘input，而不是用层去乘 之前写的是  k = torch.matmul(x, self.w_k.weight.T) ; v = torch.matmul(x, self.w_v.weight.T) ; q = torch.matmul(x, self.w_q.weight.T)
        k = self.w_k(key_value)
        v = self.w_v(key_value)
        q = self.w_q(query)

        # 多头兼容
        k = k.view(k.size()[0], k.size()[1], self.head, self.q_k_size).transpose(1, 2).transpose(2, 3)  # [batch_size, seq_len, head*q_k_size] -> [batch_size, head, seq_len, q_k_size]
        v = v.view(v.size()[0], v.size()[1], self.head, self.v_size).transpose(1, 2)  # [batch_size, seq_len, head*v_size] -> [batch_size, head, seq_len, v_size]
        q = q.view(q.size()[0], q.size()[1], self.head, self.q_k_size).transpose(1, 2)  # [batch_size, seq_len, head*q_k_size] -> [batch_size, head, seq_len, q_k_size]

        # 注意力分数
        attn = torch.matmul(q, k)/ math.sqrt(self.q_k_size)  # 这个操作被称为缩放点积注意力（Scaled Dot-Product Attention），其主要目的是计算点积时避免因键向量长度较大而导致的梯度消失或爆炸问题，并提高训练的稳定性。
        
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.head, -1, -1) # :(batch_size, head, seq_len, seq_len)
        attn = attn.masked_fill(attn_mask, -1e9)
        attn = torch.softmax(attn, dim=-1)

        # V 和 评分 相乘
        z = torch.matmul(attn, v)
        z = z.transpose(1, 2)
        z = z.contiguous().view(z.size(0), z.size(1), -1) # [batch_size, head, seq_len, v_size] -> [batch_size, seq_len, head*v_size]
        return z

if __name__ == '__main__':
    emb = EmbeddingWithPosition(vocab_size=len(ru_vocab), emb_size=128)
    ru_tokens, ru_ids = ru_preprocess(train_dataset[0][0], ru_vocab)
    ru_ids_tensor = torch.tensor(ru_ids, dtype=torch.long)
    emb_result = emb(ru_ids_tensor.unsqueeze(0)) # [batch_size, seq_len, emb_size]
    print("emb_result size:", emb_result.shape)
    multihead_attn = MultiheadAttention(emb_size=128, q_k_size=256, v_size=512, head=8)
    print(ru_ids_tensor.size()[0])
    attn_mask = torch.zeros((1, ru_ids_tensor.size()[0], ru_ids_tensor.size()[0]), dtype=torch.bool)
    multihead_attn_result = multihead_attn(key_value=emb_result, query=emb_result ,attn_mask=attn_mask)
    print('multihead_attn_result size:', multihead_attn_result.size())