import torch
import torch.nn as nn

from einops import einsum

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model:int,
        eps:float = 1e-5,
        device:torch.device = None,
        dtype:torch.dtype = None,
    ):
        super().__init__()

        self.device = device if device else 'cpu'
        self.dtype = dtype if dtype else torch.float32
        self.d_model = d_model

        self.weight = nn.Parameter(torch.ones(self.d_model,device = self.device,dtype = self.dtype))
        self.eps = eps

    def _rms(self,x:torch.Tensor) -> torch.Tensor:
        dot = einsum(
            x,
            x,
            "... d_model , ... d_model -> ..."
        )
        rms = torch.sqrt(dot / self.d_model + self.eps)
        return rms
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(device = self.device,dtype= torch.float32)
        rms_x = self._rms(x)
        rms_x = rms_x.unsqueeze(dim = -1)

        res = self.weight * x / rms_x
        return res.to(in_dtype)