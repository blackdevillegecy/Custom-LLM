import numpy as np
import torch
from torch import nn

SEED=42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 k_dim: int = None, 
                 v_dim: int = None,
                 device: str = device) -> None:
        
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        if k_dim == None:
            k_dim = embed_dim
        if v_dim == None:
            v_dim = embed_dim

        self.w_query = nn.Parameter(torch.randn(embed_dim, embed_dim)).cuda()
        self.w_key = nn.Parameter(torch.randn(k_dim, embed_dim)).cuda()
        self.w_value = nn.Parameter(torch.randn(v_dim, embed_dim)).cuda()

        self.query_size = int(embed_dim/num_heads)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = x @ self.w_query
        key = x @ self.w_key
        value = x @ self.w_value   
        query = query.reshape((x.shape[0], self.num_heads, self.query_size))   
        query = query.reshape((query.shape[1], query.shape[0], query.shape[2]))
        key = key.reshape((x.shape[0], self.num_heads, self.query_size))
        key = key.reshape((key.shape[1], key.shape[0], key.shape[2]))
        value = value.reshape((x.shape[0], self.num_heads, self.query_size))
        value = value.reshape((value.shape[1], value.shape[0], value.shape[2]))

        
        
        

if __name__=='__main__':
    embed_dim = 6
    sequence_length=20
    num_heads=1
    x = torch.randn((sequence_length, embed_dim))
    obj = MultiHeadAttention(embed_dim, num_heads)
    out = obj.forward(x)
