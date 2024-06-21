import torch
import numpy as np

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