import torch
from decoder_block import DecoderBlock
from config import DEVICE
from emb import EmbeddingWithPosition
from encoder import Encoder
from dataset import ru_vocab, ru_preprocess, en_vocab, en_preprocess, train_dataset, PAD_NUM
from torch import nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, q_k_size, v_size, f_size, head, nblocks, dropout=0.3):
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(nblocks):
            self.blocks.append(DecoderBlock(emb_size, q_k_size, v_size, f_size, head))
        self.linear = nn.Linear(emb_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb = EmbeddingWithPosition(vocab_size=vocab_size, emb_size=emb_size, seq_max_len=5000).to(DEVICE)
        self.nblocks = nblocks
        
    def forward(self, x, encoder_input, encoder_result):
        emb_result = self.emb(x)
        decoder_result = emb_result
        first_attn_mask = (x == PAD_NUM).unsqueeze(1).expand(x.size(0), x.size(1), x.size(1)).to(DEVICE)
        matrix_mask = torch.tril(torch.ones(x.size(1), x.size(1),dtype=torch.bool).unsqueeze(0), diagonal=1).to(DEVICE)
        first_attn_mask = first_attn_mask | matrix_mask
        second_attn_mask = (encoder_input == PAD_NUM).unsqueeze(1).expand(encoder_input.size(0), x.size(1), encoder_input.size(1)).to(DEVICE)
        for block in self.blocks:
            decoder_result = block(encoder_result, decoder_result, first_attn_mask, second_attn_mask)
        z = self.linear(decoder_result)
        # z = self.softmax(z) ### 这里不能softmax ？？？？？？？？？ 明明结构是要的呀，直接导致训练loss不动
        return z


if __name__ == '__main__':
    ru_tokens, ru_ids = ru_preprocess(train_dataset[0][0], ru_vocab)
    ru_tokens_1, ru_ids_1 = ru_preprocess(train_dataset[1][0], ru_vocab)
    print("len(ru_ids), len(ru_ids_1):", len(ru_ids), len(ru_ids_1))

    if len(ru_tokens) > len(ru_tokens_1):
        ru_ids_1 = ru_ids_1 + [PAD_NUM] * (len(ru_ids) - len(ru_ids_1))
    else:
        ru_ids = ru_ids + [PAD_NUM] * (len(ru_ids_1) - len(ru_ids))
    print("len(ru_ids), len(ru_ids_1):", len(ru_ids), len(ru_ids_1))

    ru_ids_tensor = torch.tensor(ru_ids, dtype=torch.long)  
    ru_ids_1_tensor = torch.tensor(ru_ids_1, dtype=torch.long)
    batch_ru = torch.tensor([ru_ids, ru_ids_1], dtype=torch.long).to(DEVICE)
    encoder = Encoder(vocab_size=len(ru_vocab), emb_size=128, q_k_size=256, v_size=512, f_size=512, head=8, nblocks=3, dropout=0.3).to(DEVICE)
    encoder_result = encoder(batch_ru).to(DEVICE)

    
    en_tokens, en_ids = en_preprocess(train_dataset[0][1], en_vocab)
    en_tokens_1, en_ids_1 = en_preprocess(train_dataset[1][1], en_vocab)
    if len(en_tokens) > len(en_tokens_1):
        en_ids_1 = en_ids_1 + [PAD_NUM] * (len(en_ids) - len(en_ids_1))
    else:
        en_ids = en_ids + [PAD_NUM] * (len(en_ids_1) - len(en_ids))
    en_ids_tensor = torch.tensor(en_ids, dtype=torch.long)
    en_ids_1_tensor = torch.tensor(en_ids_1, dtype=torch.long)
    batch_en = torch.tensor([en_ids, en_ids_1], dtype=torch.long).to(DEVICE)
    decoder = Decoder(vocab_size=len(en_vocab), emb_size=128, q_k_size=256, v_size=512, f_size=512, head=8, nblocks=3, dropout=0.3).to(DEVICE)
    decoder_result = decoder(batch_en, batch_ru, encoder_result).to(DEVICE)
    print('decoder_result size:', decoder_result.size())
    print('decoder_result:', decoder_result)
