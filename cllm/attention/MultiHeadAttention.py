import numpy as np
import torch
from torch import nn
from cllm.activation.Softmax import Softmax

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
                 batch: bool = False) -> None:
        
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        if k_dim == None:
            k_dim = embed_dim
        if v_dim == None:
            v_dim = embed_dim

        self.w_query = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.w_key = nn.Parameter(torch.randn(k_dim, embed_dim))
        self.w_value = nn.Parameter(torch.randn(v_dim, embed_dim))

        self.query_size = int(embed_dim/num_heads)
        self.batch = batch

        self.soft = Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = x @ self.w_query
        key = x @ self.w_key
        value = x @ self.w_value

        if (self.batch):
            seq_len = x.shape[1]
        else:
            seq_len = x.shape[0]  

        query = query.reshape((batch_size, seq_len, self.num_heads, self.query_size))   
        query = query.reshape((batch_size, query.shape[2], query.shape[1], query.shape[3]))
        key = key.reshape((batch_size, seq_len, self.num_heads, self.query_size))   
        key = key.reshape((batch_size, key.shape[2], key.shape[1], key.shape[3]))
        value = value.reshape((batch_size, seq_len, self.num_heads, self.query_size))   
        value = value.reshape((batch_size, value.shape[2], value.shape[1], value.shape[3]))

        attn = []
        for i in range(query.shape[0]):
            temp_i = []
            for j in range(query.shape[1]):
                q_ij = query[i][j].detach().numpy()
                k_ij = key[i][j].detach().numpy()
                score_ij = (q_ij @ k_ij.T) / np.sqrt(self.query_size)
                soft_score = self.soft(torch.from_numpy(score_ij))
                # print(soft_score.shape)
                weight_ij = soft_score.numpy() @ value[i][j].detach().numpy()
                # print(weight_ij)
                temp_i.append(weight_ij)
            attn.append(temp_i)
        attn = np.array(attn)

        attn = attn.reshape((attn.shape[0], attn.shape[2], attn.shape[1], attn.shape[3]))
        attn = attn.reshape((attn.shape[0], attn.shape[1], attn.shape[2]*attn.shape[3]))
        # print(attn.shape)
        # print(query.shape)
        # print(key.shape)
        # print(value.shape)
        # print(seq_len)
        return torch.from_numpy(attn)
        
        
        

if __name__=='__main__':
    embed_dim = 128
    sequence_length=20
    num_heads=32 # num_heads should be completely divisible by embed_dim
    batch_size=10
    x = torch.randn((batch_size, sequence_length, embed_dim))
    print(x.shape)
    obj = MultiHeadAttention(embed_dim, num_heads, batch=True)
    out = obj.forward(x)
    print(out.shape)