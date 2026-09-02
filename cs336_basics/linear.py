import torch
import torch.nn as nn
import einops
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_features: int , out_features: int , device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        std = np.sqrt(2.0 / (in_features+out_features))
        W = torch.empty((out_features, in_features))
        nn.init.trunc_normal_(W, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(self.W, x, "out_features in_features, ... in_features -> ... out_features")