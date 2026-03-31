import torch
import torch.nn as nn

from .init_utils import embedding_init

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings:int,
        embedding_dim:int,
        device:torch.device = None,
        dtype:torch.dtype = None,
    ):
        super().__init__()

        self.device = device if device else 'cpu'
        self.dtype = dtype if dtype else torch.float32
        self.d_model = embedding_dim
        
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,embedding_dim,
                device = self.device,
                dtype = self.dtype
            )
        )
        embedding_init(self.weight)

    # **Note**
    def forward(self,token_ids:torch.Tensor):      
        return self.weight[token_ids]