import torch
from torch import nn
import numpy as np

class FeedForwardGLU(nn.Module):
    def __init__(self, dim: int, multiplier: int) -> None: 
        super(FeedForwardGLU, self).__init__()
        hidden = multiplier * ((dim + multiplier - 1) // multiplier)

        self.lin1 = nn.Linear(dim, hidden, bias=False)
        self.lin2 = nn.Linear(dim, hidden, bias=False)
        self.lin3 = nn.Linear(hidden, dim, bias=False)

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(1/(1+np.exp(x)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xW_b = self.lin1(x)
        sigmoid_xW_b = self._sigmoid(xW_b.detach().numpy())
        xV_c = self.lin2(x)
        x = sigmoid_xW_b * xV_c
        x = self.lin3(x)
        return x



## EXPERIMENTS ##
if __name__ == '__main__':
    dim=4
    multiplier=256
    obj = FeedForwardGLU(dim, multiplier)
    x = torch.randn(2, dim)
    glu = obj.forward(x)
    print(glu)