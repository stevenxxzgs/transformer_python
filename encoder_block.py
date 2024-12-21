# 在 Transformer 的编码器和解码器层中，feedforward 是必不可少的组成部分。它的作用主要有以下几点：

# 引入非线性性： 多头注意力机制本质上是线性变换的组合。如果没有前馈网络中的 ReLU 激活函数，整个 Transformer 模型就只能进行线性变换，表达能力会大大受限。非线性性使得模型能够学习更复杂的模式。
# 增加模型容量： 前馈网络的隐藏层维度 (f_size) 通常比输入维度 (emb_size) 大很多（例如 4 倍）。这增加了模型的参数量，使其能够学习更复杂的函数。
# 逐位置（Position-wise）处理： 前馈网络是独立地应用于序列中每个位置的。这意味着每个位置的词向量都会通过相同的网络进行变换，这有助于模型更好地理解每个词的上下文信息。
# 如果去掉前馈网络，Transformer 模型将失去上述能力，性能会显著下降。因此，在 Transformer 中，通常不建议去掉前馈网络。

import torch 
from torch import nn
import math
from dataset import ru_vocab, ru_preprocess, train_dataset
from emb import EmbeddingWithPosition
from multihead_attn import MultiheadAttention

class EncoderBlock(nn.Module):
    def __init__(self, emb_size, q_k_size, v_size, f_size, head, dropout=0.3):
        super().__init__()
        self.multihead_attn = MultiheadAttention(emb_size=emb_size, q_k_size=q_k_size, v_size=v_size, head=head)
        self.z_linear = nn.Linear(head*v_size, emb_size)
        self.addnorm1 = nn.LayerNorm(emb_size)
        self.feedforward = nn.Sequential(
            nn.Linear(emb_size, f_size),
            nn.ReLU(),
            nn.Linear(f_size, emb_size),
        )
        self.addnorm2 = nn.LayerNorm(emb_size)

    def forward(self, x, attn_mask):
        attn_result = self.multihead_attn(x, attn_mask)
        z = self.z_linear(attn_result)
        z = x + z
        z = self.addnorm1(z)
        f_result = self.feedforward(z)
        z = z + f_result
        z = self.addnorm2(z)
        return z

if __name__ == '__main__':
    emb = EmbeddingWithPosition(vocab_size=len(ru_vocab), emb_size=128)
    ru_tokens, ru_ids = ru_preprocess(train_dataset[0][0], ru_vocab)
    ru_ids_tensor = torch.tensor(ru_ids, dtype=torch.long)
    emb_result = emb(ru_ids_tensor.unsqueeze(0)) # [batch_size, seq_len, emb_size]
    print(emb_result.shape)

    attn_mask = torch.zeros((1, ru_ids_tensor.size()[0], ru_ids_tensor.size()[0]), dtype=torch.bool)

    encoder_block = []
    for i in range(6):
        encoder_block.append(EncoderBlock(emb_size=128, q_k_size=256, v_size=512, f_size=1024, head=8))

    encoder_result = emb_result
    for i in range(6):
        encoder_result = encoder_block[i](encoder_result, attn_mask)
    print(encoder_result.size())
