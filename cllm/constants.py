import torch

# system params
SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input params
vocab_size = 50000
batch_size = 3
embed_dim = 128
seq_len = 4
batch_size = 2
num_heads = 8
qdim = embed_dim
kdim = embed_dim
vdim = embed_dim