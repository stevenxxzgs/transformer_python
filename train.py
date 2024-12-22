import torch
from torch import nn
from transformer import Transformer
from config import DEVICE, SEQ_MAX_LEN
from dataset import train_dataset, ru_vocab, en_vocab, ru_preprocess, en_preprocess, PAD_NUM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class De2EnDataset(Dataset):
    def __init__(self, train_dataset):
        super().__init__()
        self.enc_x = []
        self.dec_x = []
        for ru, en in train_dataset:
            # 分词
            ru_tokens, ru_ids = ru_preprocess(ru, ru_vocab)
            en_tokens, en_ids = en_preprocess(en, en_vocab)
            if len(ru_ids) > SEQ_MAX_LEN or len(en_ids) > SEQ_MAX_LEN:
                continue
            self.enc_x.append(ru_ids)
            self.dec_x.append(en_ids)

    def __len__(self):
        return len(self.enc_x)

    def __getitem__(self, idx):
        return self.enc_x[idx], self.dec_x[idx]

def collate_fn(batch):
    enc_x_batch = []
    dec_x_batch = []
    for enc_x, dec_x in batch:
        enc_x_batch.append(torch.tensor(enc_x, dtype=torch.long))
        dec_x_batch.append(torch.tensor(dec_x, dtype=torch.long))
    
    pad_enc = pad_sequence(enc_x_batch, batch_first=True, padding_value=PAD_NUM)
    pad_dec = pad_sequence(dec_x_batch, batch_first=True, padding_value=PAD_NUM)
    return pad_enc, pad_dec

# input batch seqence 
if __name__ == '__main__':
    print(len(train_dataset))
    dataset = De2EnDataset(train_dataset)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, persistent_workers=True, collate_fn=collate_fn)

    try:
        transformer = torch.load('checkpoints/transformer.pth').to(DEVICE)
    except:
        transformer = Transformer(vocab_size_de=len(ru_vocab), vocab_size_en=len(en_vocab), emb_size=32, q_k_size=32, v_size=32, f_size=256, head=4, nblocks=6, dropout=0.3).to(DEVICE)

    optimizer = torch.optim.SGD(transformer.parameters(), lr=1e-3, momentum=0.99)
    loss_fn=nn.CrossEntropyLoss(ignore_index=PAD_NUM)# pad_num 不计算损失

    transformer.train()
    epoch = 100

    for epoch in range(epoch):
        batch_i = 0
        loss_sum = 0
        for pad_enc_x, pad_dec_x in dataloader:
            ground_true_dec_z = pad_dec_x[:, 1:].to(DEVICE) # 真实值需要去掉第一个
            pad_enc_x = pad_enc_x.to(DEVICE)
            input_dec_z = pad_dec_x[:, :-1].to(DEVICE) # decode输入需要去掉最后一个

            predcit_output = transformer(pad_enc_x, input_dec_z)
            batch_i += 1
            loss = loss_fn(predcit_output.view(-1, predcit_output.size(-1)), ground_true_dec_z.view(-1)) # 展平 !!!
            print(predcit_output.view(-1, predcit_output.size(-1)).shape, ground_true_dec_z.view(-1).shape)
            # loss = loss_fn(predcit_output.reshape(-1, predcit_output.size(-1)), ground_true_dec_z.reshape(-1))
            # print(predcit_output.reshape(-1, predcit_output.size(-1)).shape, ground_true_dec_z.reshape(-1).shape)
            loss_sum += loss.item()
            print('epoch:{} batch:{} loss:{}'.format(epoch, batch_i, loss.item()))
            optimizer.zero_grad() #清除优化器中所有已累积的梯度。
            loss.backward()
            optimizer.step() #根据计算出的梯度更新模型的参数。
        print('epoch:{} loss:{}'.format(epoch, loss_sum))
        torch.save(transformer, 'checkpoints/transformer.pth'.format(epoch))
