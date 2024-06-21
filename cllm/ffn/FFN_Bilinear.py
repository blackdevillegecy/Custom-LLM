import torch
from torch import nn

class FeedForwardBilinear(nn.Module):
    def __init__(self, dim: int, multiplier: int) -> None: 
        super(FeedForwardBilinear, self).__init__()
        hidden = multiplier * ((dim + multiplier - 1) // multiplier)

        self.lin1 = nn.Linear(dim, hidden, bias=False)
        self.lin2 = nn.Linear(dim, hidden, bias=False)
        self.lin3 = nn.Linear(hidden, dim, bias=False)
    
    def forward(self, x: torch.Tensor, beta: int = 1) -> torch.Tensor:
        xW_b = self.lin1(x)
        xV_c = self.lin2(x)
        x = xW_b * xV_c
        x = self.lin3(x)
        return x



## EXPERIMENTS ##
if __name__ == '__main__':
    dim=4
    multiplier=256
    obj = FeedForwardBilinear(dim, multiplier)
    x = torch.randn(2, dim)
    bi = obj.forward(x)
    print(bi)