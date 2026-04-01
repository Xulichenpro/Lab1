import torch
import numpy as np

def data_loader(
    dataset:np.ndarray,
    batch_size:int,
    context_length:int,
    device:str
) -> torch.Tensor:
    size = dataset.size
    token_input =[]   
    token_target = []

    for i in range(size - context_length):
        token_input.append(torch.from_numpy(dataset[i:i + context_length]).to(device))
        token_target.append(torch.from_numpy(dataset[i + 1:i + 1 + context_length]).to(device))

    input =  torch.stack(token_input,dim = 0)
    target = torch.stack(token_target,dim = 0)
    indices = torch.randperm(size - context_length)[:batch_size]

    return input[indices],target[indices]