import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RotatoryPositionEmbedding(nn.Module):
    def __init__(self, ) -> None:

        pass