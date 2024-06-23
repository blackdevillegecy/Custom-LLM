import torch
from torch import nn
from cllm.activation.Softmax import Softmax
from cllm.utils import BMM, _matmul_2d_3d
from cllm.embedding.TokenEmbedding import TokenEmbedding

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfAttention(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 q_dim: int, 
                 k_dim: int,
                 v_dim: int,
                 device: str = device,
                 batch: bool = False) -> None:
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim

        self.w_query = nn.Parameter(torch.randn(input_dim, q_dim)).to(device)
        self.w_key = nn.Parameter(torch.randn(input_dim, k_dim)).to(device)
        self.w_value = nn.Parameter(torch.randn(input_dim, v_dim)).to(device)
        self.softmax = Softmax(dim=-1)

        self.batch = batch
        
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:

        if self.batch:
            query = _matmul_2d_3d(self.w_query, x).transpose(1, 2)
            key = _matmul_2d_3d(self.w_key, x)
            value = _matmul_2d_3d(self.w_value, x).transpose(1, 2)
        else:
            query = self.w_query.T.matmul(x.T).T
            key = self.w_key.T.matmul(x.T)
            value = self.w_value.T.matmul(x.T).T

        scoreobj = BMM(query, key, batch=self.batch)
        scores = scoreobj.forward()

        scores = scores/(self.input_dim**2)
        attention = self.softmax(scores)
        
        weightobj = BMM(attention, value, batch=self.batch)
        weighted = weightobj.forward()

        return weighted


## EXPERIMENT ##

if __name__=='__main__':
    sentence = "For me, you are the priority no matter what, is that okay"
    dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}
    tokens = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
    # print (tokens)

    vocab_size = 50000
    embed_dim = 3
    embed = TokenEmbedding(vocab_size, embed_dim)
    embed_sentence = embed(tokens).detach()

    # print(embed_sentence)
    # print(embed_sentence[1])

    x = embed_sentence
    input_dim = embed_sentence.shape[1]
    q_dim, k_dim, v_dim = 4, 4, 6
    # x = torch.randn(3, 12, 3)
    # print(x.shape)
    # print(x, input_dim, q_dim, k_dim, v_dim)
    
    obj = SelfAttention(input_dim, q_dim, k_dim, v_dim, device)
    ans = obj.forward(x)
    # print(ans)

