import torch
import math

def linear_init(weight:torch.Tensor, d_in:int, d_out:int):
    std_2 = 2 / (d_in + d_out)
    std = math.sqrt(std_2)

    return torch.nn.init.trunc_normal_(
        weight,
        mean = 0,
        std = std,
        a = -3 * std,
        b = 3 * std,
    )

def embedding_init(weight:torch.Tensor):
    std = 1
    return torch.nn.init.trunc_normal_(
        weight,
        mean = 0,
        std = std,
        a = -3 * std,
        b = 3 * std,
    )