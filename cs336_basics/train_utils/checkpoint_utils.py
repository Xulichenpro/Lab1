import torch
import torch.nn as nn

def save_checkpoint(
    model:nn.Module,
    optimizer:torch.optim.Optimizer,
    iter:int,
    out:str,
):
    save_obj = {
        "model_weights":model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "iter":iter,
    }
    torch.save(obj = save_obj, f = out)

def load_checkpoint(
    src:str,
    model:nn.Module,
    optimizer:torch.optim.Optimizer,
):
    save_dict = torch.load(src)
    model_weights = save_dict['model_weights']
    optimizer_state = save_dict['optimizer_state']
    model.load_state_dict(model_weights)
    optimizer.load_state_dict(optimizer_state)
    return save_dict['iter']