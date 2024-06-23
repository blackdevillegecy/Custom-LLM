import torch
from torch import nn

SEED=42
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TokenEmbedding(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 embed_dim: int,) -> None:
        
        super(TokenEmbedding, self).__init__()

        self.w = nn.Parameter(torch.zeros((vocab_size, embed_dim))).to(device)
        self.vocab_size = vocab_size
        nn.init.uniform_(self.w, -0.2, 0.2)
        nn.init.normal_(self.w)
    
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        
        if (torch.max(x) > self.vocab_size):
            raise IndexError("max value of x is greater than or equal to vocabulary size!! ")

        return self.w[x]
    
## EXPERIMENT ##
if __name__=='__main__':
    vocab_size = 1000
    embed_dim = 128
    x = torch.randint(0, 10, size=(3, 4))
    obj = TokenEmbedding(vocab_size, embed_dim)
    em = obj.forward(x)
    print(em)