import torch
from torch import nn
import numpy as np

class FeedForwardGEGLU(nn.Module):
    def __init__(self, dim: int, multiplier: int) -> None:
        super(FeedForwardGEGLU, self).__init__()

        hidden = multiplier * ((dim + multiplier - 1) // multiplier)

        self.lin1 = nn.Linear(dim, hidden, bias=False)
        self.lin2 = nn.Linear(dim, hidden, bias=False)
        self.lin3 = nn.Linear(hidden, dim, bias=False)

    def _gelu(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(0.5*x*(1+np.tanh((np.sqrt(2/np.pi))*(x+0.44715*np.power(x, 3)))))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xW_b = self.lin1(x)
        gelu_xW_b = self._gelu(xW_b.detach().numpy())
        xV_c = self.lin2(x)
        x = gelu_xW_b * xV_c
        x = self.lin3(x)
        return x


## EXPERIMENTS ##
if __name__ == '__main__':
    dim=4
    multiplier=256
    obj = FeedForwardGEGLU(dim, multiplier)
    x = torch.randn(3, 2, dim)
    ge = obj.forward(x)
    print(ge)