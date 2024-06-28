import torch
from torch import nn
from typing import Union

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[list, int, torch.Size], epsilon: float = 1e-5) -> None:
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.ones(normalized_shape))
        self.epsilon = epsilon

    def _mean(self, x: torch.Tensor) -> torch.Tensor:
        mean = None
        if (type(self.normalized_shape)==list):
            length = len(self.normalized_shape)
            dim = ()
            for i in range(-1*length, 0):
                if (x.shape[i] != self.normalized_shape[i]):
                    raise RuntimeError("dimension of x and normalized shape does not match!!")
                dim = dim + (i,)
            mean = x.mean(dim, keepdim=True)
        elif (type(self.normalized_shape)==int):
            mean = x.mean((-1), keepdim=True)
        else:
            raise TypeError("Enter a valid type of normalized_shape")

        return mean
    
    def _std(self, x: torch.Tensor) -> torch.Tensor:
        var = None
        if (type(self.normalized_shape)==list):
            length = len(self.normalized_shape)
            dim = ()
            for i in range(-1*length, 0):
                if (x.shape[i] != self.normalized_shape[i]):
                    raise RuntimeError("dimension of x and normalized shape does not match!!")
                dim = dim + (i,)
            var = x.var(dim, keepdim=True)
        elif (type(self.normalized_shape==int)):
            var = x.var((-1), keepdim=True)
        else:
            raise TypeError("Enter a valid type of normalized_shape")
        std = torch.sqrt(var + self.epsilon)
        return std
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x - self._mean(x)
        y = y / self._std(x)
        y = y*self.gamma + self.beta
        return y

## EXPERMIMENT ##
if __name__ == '__main__':
    normalized_shape =[2, 4]
    obj = LayerNorm(normalized_shape)
    x = torch.randn(5, 2, 4)
    m = obj.forward(x)
    print(m)
