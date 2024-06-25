import torch
from torch import nn
from cllm.activation.Softmax import Softmax
from cllm.utils import BMM, _matmul_2d_3d
from cllm.embedding.TokenEmbedding import TokenEmbedding
from cllm.embedding.StandardPositionEmbedding import StandardPositionEmbedding
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfAttention(nn.Module):
    def __init__(self, 
                 seq_len: int,
                 q_dim: int, 
                 k_dim: int,
                 v_dim: int,
                 device: str = device,
                 batch: bool = False) -> None:
        super(SelfAttention, self).__init__()
        self.seq_len = seq_len

        self.softmax = Softmax(dim=-1)

        self.batch = batch
        
    def forward(self, 
                x: torch.Tensor,
                ) -> torch.Tensor:

        if self.batch:
            query = _matmul_2d_3d(self.w_query, x)
            key = _matmul_2d_3d(self.w_key, x).transpose(1, 2)
            value = _matmul_2d_3d(self.w_value, x).transpose(1, 2)
        else:
            query = self.w_query.T.matmul(x.T)
            key = self.w_key.T.matmul(x.T).T
            value = self.w_value.T.matmul(x.T).T
        print(query.shape, key.shape)
        if (query.shape[1] != key.shape[0]):
                raise ValueError(f"input shape not matching the mat2 shape!!: inp_shape: {query.shape} and mat2_shape: {key.shape}")

        scoreobj = BMM(query, key, batch=self.batch)
        scores = scoreobj.forward()

        scores = scores/(self.seq_len**0.5)
        attention = self.softmax(scores)
        
        weightobj = BMM(attention, value, batch=self.batch)
        weights = weightobj.forward()

        return weights


## EXPERIMENT ##

if __name__=='__main__':
    sentence = "For me, you are the priority no matter what, is that okay"
    dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}
    tokens = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
    # print (tokens)

    vocab_size = 50000
    embed_dim = 3
    embed = TokenEmbedding(vocab_size, embed_dim)
    embed_sentence = embed(tokens)

    print(embed_sentence)
    spe = StandardPositionEmbedding(embed_sentence.shape[0], embed_dim, 10000.0)
    x = spe.forward(embed_sentence)
    
    print(x)
    
    seq_len = x.shape[0]
    print(x.shape) # seq_len, embed_dim
    q_dim, k_dim, v_dim = 4, 4, 6
    tar_len = seq_len
    w_query = nn.Parameter(torch.randn(seq_len, q_dim)).to(device)
    w_key = nn.Parameter(torch.randn(tar_len, k_dim)).to(device)
    w_value = nn.Parameter(torch.randn(tar_len, v_dim)).to(device)
    
      
    obj = SelfAttention(embed_dim, q_dim, k_dim, v_dim, device)
    ans = obj.forward(x)
    print(ans)

