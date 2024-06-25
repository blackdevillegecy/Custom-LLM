import torch
from torch import nn
import numpy as np

SEED=42
torch.cuda.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Softmax(nn.Module):
    def __init__(self, dim: int) -> None:
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num = torch.exp(x)
        denom = torch.sum(torch.exp(x), dim=self.dim, keepdim=True).to(device)
        return num/denom

## EXPERIMENT ##
if __name__=='__main__':
    dim=0
    x = torch.randn(2, 3, 4)
    obj = Softmax(dim=dim)
    ans = obj.forward(x)
    print(ans)