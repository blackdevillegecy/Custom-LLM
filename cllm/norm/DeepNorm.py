import torch
from torch import nn
from typing import Union
from cllm.norm.LayerNorm import LayerNorm

class DeepNorm(nn.Module):
    def __init__(self, alpha: float, normalized_shape: Union[list, int, torch.Size], epsilon: float = 1e-5) -> None:
        super(DeepNorm, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.normalized_shape = normalized_shape
        self.layernorm = LayerNorm(normalized_shape, epsilon)

    def forward(self, x: torch.Tensor, gx: torch.Tensor):
        return self.layernorm(x + self.alpha * gx)
    

## EXPERIMENT ##
if __name__ == '__main__':    
    alpha, normalized_shape = 1.2, [3, 4]
    x = torch.randn((5, 3, 4))
    gx = torch.randn((5, 3, 4))
    obj = DeepNorm(alpha, normalized_shape)
    dn = obj.forward(x, gx)
    print(dn)