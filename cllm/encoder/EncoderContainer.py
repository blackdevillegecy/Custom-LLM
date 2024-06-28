import torch
from torch import nn
from cllm.attention import CrossAttention, MultiHeadAttention, SelfAttention
from cllm.ffn import FeedForwardBilinear, FeedForwardGEGLU, FeedForwardGLU, FeedForwardReLU, FeedForwardSwiGLU
from cllm.activation import Softmax
from cllm.norm import DeepNorm, LayerNorm, RMSNorm
from cllm import constants

'''
TERMINOLOGIES:

Attention:
CA - Cross Attention
MHA - Multi Head Attention
SA - Self Attention

Normalization:
DN - Deep Norm
LN - Layer Norm
RMSN - RMS Norm

Embedding:
TE - Token Embedding
SPE - Standared Positon Embedding
RPE - Rotatory Position Embedding

Feed Forward Network:
FFN_SwiGLU - Feed Forward Network with SwiGLU activation 
FFN_GEGLU - Feed Forward Network with GEGLU activation 
FFN_GLU - Feed Forward Network with GLU activation 
FFN_ReLU - Feed Forward Network with ReLU activation 
FFN_Bilinear - Feed Forward Network with Bilinear activation 
'''

class EncoderContainer(nn.Module):
    def __init__(self,
                 basic_architecture: list = [],
                 Nx: int = 1) -> None:
        '''
        Sample architecture - ["MHA", "LN", "FFN_SwiGLU", "LN"]*6
        '''
        super(EncoderContainer, self).__init__()
        self.container = nn.ModuleList()
        for i in range(Nx):
            for component in basic_architecture:
                if (component =='SA'):
                    self.container.append(SelfAttention(constants.embed_dim, batch=True))
                elif component == "MHA":
                    self.container.append(MultiHeadAttention(constants.embed_dim, constants.num_heads))
                elif component == "CA":
                    self.container.append(CrossAttention(constants.embed_dim, constants.qdim))
                elif component == "LN":
                    self.container.append(LayerNorm([constants.seq_len, constants.embed_dim]))
                elif component == "DN":
                    self.container.append(DeepNorm)
                elif component == "RMSN":
                    self.container.append(RMSNorm)
                elif component == "FFN_SwiGLU":
                    self.container.append(FeedForwardSwiGLU(constants.embed_dim))
                elif component == "FFN_GEGLU":
                    self.container.append(FeedForwardGEGLU)
                elif component == "FFN_GLU":
                    self.container.append(FeedForwardGLU)
                elif component == "FFN_ReLU":
                    self.container.append(FeedForwardReLU)
                elif component == "FFN_Bilinear":
                    self.container.append(FeedForwardBilinear)
                else:
                    raise ValueError("Enter a valid component code!!")
        print(self.container)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

if __name__=='__main__':
    # ls = ["MHA", "LN", "FFN_SwiGLU", "LN"]*6
    # print(ls)

    encoder = EncoderContainer(["MHA", "LN", "FFN_SwiGLU", "LN"], 6)