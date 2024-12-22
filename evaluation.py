import torch
from transformer import Transformer
from dataset import train_dataset, ru_vocab, en_vocab, ru_preprocess, en_preprocess, PAD_NUM, BOS_NUM, EOS_NUM, UNK_NUM
from config import DEVICE, SEQ_MAX_LEN

def translate(transformer, sentence):
    ru_tokens, ru_ids = ru_preprocess(sentence, ru_vocab)
    if len(ru_tokens) > SEQ_MAX_LEN:
        raise Exception('不支持超过{}的句子'.format(SEQ_MAX_LEN))
    with torch.no_grad():
        ru_ids_tensor = torch.tensor([ru_ids], dtype=torch.long).to(DEVICE)
        encoder_z = transformer.encode(ru_ids_tensor)
        en_token_ids = [BOS_NUM]
        while len(en_token_ids) < 10:
            en_ids_tensor = torch.tensor([en_token_ids], dtype=torch.long).to(DEVICE)
            decoder_z = transformer.decode(en_ids_tensor, ru_ids_tensor, encoder_z)
            next_token_probs = decoder_z[0, en_ids_tensor.size(-1)-1, :]
            next_token_id = torch.argmax(next_token_probs)
            en_token_ids.append(next_token_id)

            if next_token_id == EOS_NUM:
                break

    en_token_ids = [id for id in en_token_ids if id not in [BOS_NUM, EOS_NUM, PAD_NUM, UNK_NUM]]
    en_tokens = en_vocab.lookup_tokens(en_token_ids)
    return ' '.join(en_tokens)

if __name__ == '__main__':
    transformer = torch.load(r'C:\Users\steve\Desktop\xz\transformer\checkpoints\transformer.pth').to(DEVICE)
    transformer.eval()
    print(transformer)

    en = translate(transformer, 'В постановлениях о роспуске ассоциации "Илинден".')
    print(en)

