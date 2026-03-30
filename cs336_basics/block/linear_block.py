import torch
import torch.nn as nn

from einops import einsum

from .init_utils import linear_init

class Linear(nn.Module):
    def __init__(
            self,
            in_features:int,
            out_features:int,
            device:torch.device = None,
            dtype:torch.dtype = None,
    ):
        super().__init__()

        self.device = device if device else 'cpu'
        self.dtype = dtype if dtype else torch.float32

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=self.device,
                dtype=self.dtype
            )
        )
        linear_init(self.weight,in_features,out_features)
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        out = einsum(
            x,self.weight,
            "... d_in,d_out d_in -> ... d_out"
        )
        return out
