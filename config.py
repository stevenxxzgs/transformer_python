import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_MAX_LEN = 5000