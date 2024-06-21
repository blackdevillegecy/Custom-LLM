import torch 
from torch import nn
from cllm.activation.Softmax import Softmax
from cllm.utils import BMM, Embedding, _matmul_2d_3d

SEED = 42
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CrossAttention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 qk_dim: int,
                 v_dim: int,
                 device: str = device,
                 batch: bool = False) -> None:
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim

        self.w_query = nn.Parameter(torch.randn(input_dim, qk_dim)).to(device)
        self.w_key = nn.Parameter(torch.randn(input_dim, qk_dim)).to(device)
        self.w_value = nn.Parameter(torch.randn(input_dim, v_dim)).to(device)
        self.softmax = Softmax(dim=-1)

        self.batch = batch

    def forward(self, 
                x1: torch.Tensor, 
                x2: torch.Tensor) -> torch.Tensor:
        if self.batch:
            query = _matmul_2d_3d(self.w_query, x1).transpose(1, 2)
            key = _matmul_2d_3d(self.w_key, x2)
            value = _matmul_2d_3d(self.w_value, x2).transpose(1, 2)
        else:
            query = self.w_query.T.matmul(x1.T).T
            key = self.w_key.T.matmul(x2.T)
            value = self.w_value.T.matmul(x2.T).T

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
    sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])

    vocab_size = 50000
    embed = Embedding(vocab_size, 3)
    embed_sentence = embed(sentence_int).detach()

    x1 = embed_sentence
    input_dim = embed_sentence.shape[1]
    x2 = torch.randn(8, input_dim)
    qk_dim, v_dim = 2, 4

    crossattn = CrossAttention(input_dim, qk_dim, v_dim)
    cvector = crossattn.forward(x1, x2)
    print(cvector.shape)