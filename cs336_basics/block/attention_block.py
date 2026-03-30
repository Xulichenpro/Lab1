import math
import torch
import torch.nn as nn

from einops import einsum,rearrange

from .rope_block import RoPE
from .linear_block import Linear

def softmax(x:torch.Tensor,dim:int) -> torch.Tensor:
    max_mat,_ = torch.max(x,dim=dim,keepdim=True)
    return torch.exp(x - max_mat) / torch.sum(torch.exp(x - max_mat),dim=dim,keepdim=True)

def scaled_dot_product_attention(
    q:torch.Tensor,
    k:torch.Tensor,
    v:torch.tensor,
    masked:torch.Tensor = None,
) -> torch.Tensor:
    d_k = q.shape[-1]
    pre_soft = einsum(
        q,
        k,
        "... seq_1 d_k,... seq_2 d_k -> ... seq_1 seq_2",
    )
    pre_soft = pre_soft / math.sqrt(d_k)
    
    if masked is not None:
    # 假设 masked 是布尔类型（True表示保留，False表示遮蔽）
        pre_soft = pre_soft.masked_fill(masked == 0, float('-inf'))
    
    soft = softmax(pre_soft,-1)        
    return einsum(
        soft,
        v,
        "... seq_len1 seq_len2,... seq_len2 d_v ->... seq_len1 d_v"
    )

class MutiheadAttention(nn.Module):
    def __init__(
        self,
        d_model:int,
        num_heads:int,
    ):
        super().__init__()

        self.d_model = d_model
        self.h = num_heads
        self.d_k = self.d_model // self.h
        self.d_v = self.d_k
        self.WO = Linear(in_features=self.d_model,out_features=(self.h * self.d_k))
        self.Q = Linear(in_features=self.d_model,out_features=(self.h * self.d_k))
        self.K = Linear(in_features=self.d_model,out_features=(self.h * self.d_k))
        self.V = Linear(in_features=self.d_model,out_features=(self.h * self.d_v))
        

