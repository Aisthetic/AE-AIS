"""Device selection: prefer Apple MPS, then CUDA, then CPU."""
from __future__ import annotations

import torch


def get_device(requested: str = "auto") -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
