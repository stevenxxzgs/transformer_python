import torch
from torch import nn
from dataset import ru_vocab, ru_preprocess, en_vocab, en_preprocess, train_dataset, PAD_NUM
from emb import EmbeddingWithPosition
from encoder import Encoder
from config import DEVICE
from multihead_attn import MultiheadAttention

class DecoderBlock(nn.Module):
    def __init__(self, emb_size, q_k_size, v_size, f_size, head, dropout=0.3):
        super(DecoderBlock, self).__init__()
        self.multihead_attn_masked = MultiheadAttention(emb_size, q_k_size, v_size, head)
        self.z_linear1 = nn.Linear(v_size*head, emb_size)
        self.multihead_attn = MultiheadAttention(emb_size, q_k_size, v_size, head)
        self.z_linear2 = nn.Linear(v_size*head, emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, f_size),
            nn.ReLU(),
            nn.Linear(f_size, emb_size)
        )
        self.addnorm1 = nn.LayerNorm(emb_size)
        self.addnorm2 = nn.LayerNorm(emb_size)
        self.addnorm3 = nn.LayerNorm(emb_size)

    def forward(self, encoder_result, emb_result, first_attn_mask, second_attn_mask):
        z = self.multihead_attn_masked(key_value=emb_result, query=emb_result, attn_mask=first_attn_mask) # [batch_size, seq_len, head*emb_size]
        z = self.z_linear1(z) # [batch_size, seq_len, emb_size]
        # RuntimeError: The size of tensor a (128) must match the size of tensor b (262144) at non-singleton dimension 2
        # emb_result: [batch_size, seq_len, emb_size], z: [batch_size, seq_len, emb_size]
        z = self.addnorm1(emb_result + z) # [batch_size, seq_len, emb_size]
        attn_out = self.multihead_attn(key_value=encoder_result, query=emb_result, attn_mask=second_attn_mask)
        attn_out = self.z_linear2(attn_out) # [batch_size, seq_len, emb_size*head] -> [batch_size, seq_len, emb_size]
        z = self.addnorm2(z + attn_out) # [batch_size, seq_len, emb_size]
        ffn_out = self.feed_forward(z) # [batch_size, seq_len, emb_size]
        z = self.addnorm3(z + ffn_out) # [batch_size, seq_len, emb_size]
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
    emb_en = EmbeddingWithPosition(vocab_size=len(en_vocab), emb_size=128, seq_max_len=5000).to(DEVICE)
    emb_en_result = emb_en(batch_en).to(DEVICE)
    print('emb_en_result size:', emb_en_result.size())
    print('emb_en_result:', emb_en_result)

# the most import thing is the attn_mask !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! this is wrong
    # first_attn_mask = torch.zeros((1, en_ids_tensor.size()[0], en_ids_tensor.size()[0]), dtype=torch.bool).to(DEVICE)
    # if len(en_ids) < len(ru_ids):
    #     second_attn_mask = torch.zeros((1, ru_ids_tensor.size()[0], ru_ids_tensor.size()[0]), dtype=torch.bool).to(DEVICE)
    # else:
    #     second_attn_mask = torch.zeros((1, en_ids_tensor.size()[0], en_ids_tensor.size()[0]), dtype=torch.bool).to(DEVICE)

# the right attn_mask is:
# 这里不需要用embbding来做，因为mask是一个方阵
# 第一个mask是用来decoder_emb用的，这个有一个重要细节，就是不能看到未来的词，所以要做一个对应的mask，比如第一个词只能看到第一个，第二个词只能看到前两个
    first_attn_mask = (batch_en == PAD_NUM)
    first_attn_mask = (batch_en == PAD_NUM).unsqueeze(1)
    first_attn_mask = (batch_en == PAD_NUM).unsqueeze(1).expand(batch_en.size(0), batch_en.size(1), batch_en.size(1)).to(DEVICE)
    matrix = torch.ones((batch_en.size(1), batch_en.size(1)), dtype=torch.bool).to(DEVICE)
    upper_triangle = torch.triu(matrix, diagonal=1).to(DEVICE) # diagonal=1表示对角线上的元素不取
    first_attn_mask = first_attn_mask | upper_triangle.unsqueeze(0)
    print('first_attn_mask size:', first_attn_mask.size())
    print('first_attn_mask:', first_attn_mask)

# 接着是第二个mask，这个mask是用来decoder_emb和encoder_result用的，这个mask是用来把对应的pad的词给mask掉
    second_attn_mask = (batch_ru == PAD_NUM).unsqueeze(1).expand(batch_ru.size(0), batch_en.size(1), batch_ru.size(1)).to(DEVICE)
    print('second_attn_mask size:', second_attn_mask.size())
    print('second_attn_mask:', second_attn_mask)
    decoder_result = emb_en_result.to(DEVICE)
    decoder_block = DecoderBlock(emb_size=128, q_k_size=256, v_size=512, f_size=512, head=8).to(DEVICE)
    decoder_result = decoder_block(encoder_result=encoder_result, emb_result=decoder_result, first_attn_mask=first_attn_mask, second_attn_mask=second_attn_mask)
    print('decoder_result size:', decoder_result.size())
    print('decoder_result:', decoder_result)
