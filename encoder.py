import torch
from torch import nn
from encoder_block import EncoderBlock
from dataset import train_dataset, ru_vocab, ru_preprocess, en_vocab, en_preprocess, PAD_NUM
from config import DEVICE
from emb import EmbeddingWithPosition


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, q_k_size, v_size, f_size, head, nblocks, dropout=0.3):
        super().__init__()
        self.emb = EmbeddingWithPosition(vocab_size=vocab_size, emb_size=emb_size)
        self.blocks = nn.ModuleList()
        for _ in range(nblocks):
            self.blocks.append(EncoderBlock(emb_size=emb_size, q_k_size=q_k_size, v_size=v_size, f_size=f_size, head=head, dropout=dropout))
    
    def forward(self, x): # 这个输入是sequence的ids
        pad_mask = (x == PAD_NUM).unsqueeze(1)
        pad_mask = pad_mask.expand(x.size(0), x.size(1), x.size(1))
        pad_mask = pad_mask.to(DEVICE)
        x = self.emb(x) # 所以这里要emb
        for block in self.blocks:
            x = block(key_value=x, query=x, attn_mask=pad_mask)
        return x

if __name__ == '__main__':

    ru_tokens, ru_ids = ru_preprocess(train_dataset[0][0], ru_vocab)
    ru_tokens_1, ru_ids_1 = ru_preprocess(train_dataset[1][0], ru_vocab)

    print("len(ru_ids), len(ru_ids_1):", len(ru_ids), len(ru_ids_1))

    if len(ru_tokens) > len(ru_tokens_1):
        ru_ids_1 = ru_ids_1 + [PAD_NUM] * (len(ru_ids) - len(ru_ids_1))
    else:
        ru_ids = ru_ids + [PAD_NUM] * (len(ru_ids_1) - len(ru_ids))

    print("len(ru_ids), len(ru_ids_1):", len(ru_ids), len(ru_ids_1))

    batch = torch.tensor([ru_ids, ru_ids_1], dtype=torch.long).to(DEVICE)
    print('batch size:', batch.size())

    encoder = Encoder(vocab_size=len(ru_vocab), emb_size=128, q_k_size=128, v_size=128, f_size=128, head=8, nblocks=3, dropout=0.3).to(DEVICE)
    encoder_result = encoder(batch)
    print('encoder_result size:', encoder_result.size())
    print('encoder_result:', encoder_result)