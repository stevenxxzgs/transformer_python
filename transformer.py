import torch
from encoder import Encoder
from decoder import Decoder
from torch import nn
from dataset import ru_vocab, en_vocab, ru_preprocess, en_preprocess, train_dataset, PAD_NUM
from config import DEVICE

class Transformer(nn.Module):
    def __init__(self, vocab_size_de, vocab_size_en, emb_size, q_k_size, v_size, f_size, head, nblocks, dropout=0.3):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size=vocab_size_de, emb_size=emb_size, q_k_size=q_k_size, v_size=v_size, f_size=f_size, head=head, nblocks=nblocks, dropout=dropout)
        self.decoder = Decoder(vocab_size=vocab_size_en, emb_size=emb_size, q_k_size=q_k_size, v_size=v_size, f_size=f_size, head=head, nblocks=nblocks, dropout=dropout)

    def forward(self, encoder_x, decoder_x):
        encoder_result = self.encoder(encoder_x)
        decoder_result = self.decoder(decoder_x, encoder_x, encoder_result)
        return decoder_result
    
    def encode(self, encoder_x):
        return self.encoder(encoder_x)
    
    def decode(self, decoder_x, encoder_x, encoder_result):
        return self.decoder(decoder_x, encoder_x, encoder_result)

if __name__ == '__main__':
    transformer = Transformer(vocab_size_de=len(ru_vocab), vocab_size_en=len(en_vocab), emb_size=128, q_k_size=256, v_size=512, f_size=512, head=8, nblocks=3, dropout=0.3).to(DEVICE)
    print(transformer)
    ru_tokens, ru_ids = ru_preprocess(train_dataset[0][0], ru_vocab)
    ru_tokens_1, ru_ids_1 = ru_preprocess(train_dataset[1][0], ru_vocab)
    en_tokens, en_ids = en_preprocess(train_dataset[0][1], en_vocab)
    en_tokens_1, en_ids_1 = en_preprocess(train_dataset[1][1], en_vocab)
    if len(ru_ids) > len(ru_ids_1):
        ru_ids_1 = ru_ids_1 + [PAD_NUM] * (len(ru_ids) - len(ru_ids_1))
    else:
        ru_ids = ru_ids + [PAD_NUM] * (len(ru_ids_1) - len(ru_ids))

    if len(en_ids) > len(en_ids_1):
        en_ids_1 = en_ids_1 + [PAD_NUM] * (len(en_ids) - len(en_ids_1))
    else:
        en_ids = en_ids + [PAD_NUM] * (len(en_ids_1) - len(en_ids))
    batch_ru = torch.tensor([ru_ids, ru_ids_1]).to(DEVICE)
    batch_en = torch.tensor([en_ids, en_ids_1]).to(DEVICE)
    print(batch_ru.size())
    print(batch_en.size())
    result = transformer(batch_ru, batch_en)
    print(result.size())
