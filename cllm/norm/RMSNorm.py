import torch
from torch import nn
import numpy as np

class RMSNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-5) -> None:
        super(RMSNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.epsilon = epsilon

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g * self._normalize(x.float()).type_as(x)
    

## EXPERIMENT ##
if __name__ == '__main__':
    dim=3
    obj = RMSNorm(dim)
    x = torch.randn(2, dim)
    print("x:", x)
    rms = obj._normalize(x)
    print("rms:", rms)