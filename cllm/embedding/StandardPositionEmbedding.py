import torch
from torch import nn
import math
# from cllm.embedding.TokenEmbedding import TokenEmbedding

class StandardPositionEmbedding(nn.Module):
    def __init__(self, 
                 seq_len: int,
                 embed_dim: int,
                 batch_size: int = 0,
                 dropout: float = 0.1,
                 n: float = 1000.0) -> None:
        super(StandardPositionEmbedding, self).__init__()
        if dropout < 0.0: 
            raise ValueError("Embedding: negative value of dropout is not allowed!!")

        self.sl = seq_len
        self.embed_dim = embed_dim
        self.bsize = batch_size
        self.n = n
        self.drop = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        posEmbedding = torch.zeros(self.sl, self.embed_dim)
        pos = torch.arange(0, self.sl, dtype=torch.float)
        pos = pos.unsqueeze(dim=1) ## Change the dimension of the pos from 1 to 2

        c = torch.exp(torch.arange(0, self.embed_dim, 2) * -(math.log(self.n) / self.embed_dim))
        pos = pos * c
        for i in range(posEmbedding.shape[0]):
            for j in range(posEmbedding.shape[1]):
                if (j%2==0):
                    posEmbedding[i][j] = torch.sin(pos[i][j//2])
                else:
                    posEmbedding[i][j] = torch.cos(pos[i][j//2])
        posEmbedding.unsqueeze(0)
        x = x + posEmbedding.requires_grad_(False)
        self.drop(x)
        return x


## EXPERIMENT ##
if __name__=='__main__':
    # sentence = "Hello, my name is Ayush, how are you"
    # dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}
    # tokens = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
    # vocab_size = 50000
    # embed_dim = 128
    # print(tokens.shape)
    # te = TokenEmbedding(vocab_size, embed_dim)
    # token_embeddings = te.forward(tokens)
    # print(token_embeddings)
    # seq_len = token_embeddings.shape[0]
    # spe = StandardPositionEmbedding(seq_len, embed_dim, 10000.0)
    # final_embeddings = spe.forward(token_embeddings)
    # print(final_embeddings)
    pass
        