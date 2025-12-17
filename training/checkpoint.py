import torch
from pathlib import Path

def save_checkpoint(path, model, optimizer, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state,
    }, path)

def load_checkpoint(path, model, optimizer=None, load_optimizer=True):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model"])

    if load_optimizer and optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as e:
            print("[RESUME] Optimizer state incompatible, rebuilding optimizer")
    
    return ckpt["state"]

