import torch
from torch import nn
import numpy as np
import constants

torch.manual_seed(constants.SEED)

class BMM(nn.Module):
    def __init__(self, 
                 inp: torch.Tensor, 
                 mat2: torch.Tensor,
                 batch: bool = False) -> None:
        
        super(BMM, self).__init__()
        self.batch = batch
        if self.batch:
            if (inp.shape[0] != mat2.shape[0] or inp.shape[2] != mat2.shape[1]):
                raise ValueError(f"input shape not matching the mat2 shape!!: inp_shape: {inp.shape} and mat2_shape: {mat2.shape}")
        else: 
            if (inp.shape[1] != mat2.shape[0]):
                raise ValueError(f"input shape not matching the mat2 shape!!: inp_shape: {inp.shape} and mat2_shape: {mat2.shape}")
        self.inp = inp ## if batch: b x n x m, else: n x m
        self.mat2 = mat2 ## if batch: b x m x p, else: m x p
        ## output dimension will be - if batch: b x n x p, else: n x p
   

    def forward(self, ) -> torch.Tensor:
        out = []
        if self.batch:
            for bi in range(self.inp.shape[0]):
                m1 = self.inp[bi].detach().numpy()
                m2 = self.mat2[bi].detach().numpy()
                out.append(m1 @ m2)
            out = np.array(out)
        else:
            out = self.inp.detach().numpy() @ self.mat2.detach().numpy()

        if not torch.is_tensor(out):
            out = torch.tensor(out).to(constants.device)
        return out

def _tril(x: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
    out = x.detach().clone()
    m, n = x.shape[0], x.shape[1]
    
    for i in range(m):
        for j in range(i+1+diagonal, n):
            out[i][j] = 0.00
    
    return out

def _matmul_2d_3d(two_d_x: torch.Tensor, three_d_x: torch.Tensor) -> torch.Tensor:
    results = []
    two_d_x = two_d_x.detach().numpy()
    three_d_x = three_d_x.detach().numpy()
    for i in range(three_d_x.shape[0]):
        result = np.matmul(two_d_x.T, three_d_x[i].T)
        results.append(result)
    return torch.tensor(np.array(results))

## EXPERIMENT ##
if __name__ == '__main__':
    x = torch.randn(4, 6)
    out = _tril(x, diagonal=-1)
    print(out)

    b, n, m, p = 10, 2, 3, 4
    inp, mat2 = torch.randn((b, n, m)), torch.randn((b, m, p))
    obj = BMM(inp, mat2)
    out = obj.forward()
    print(out)