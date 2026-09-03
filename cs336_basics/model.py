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
        self.W = nn.Parameter(W).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(self.W, x, "out_features in_features, ... in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self, vocab_size: int , d_model: int , device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        std = np.sqrt(2.0 / (vocab_size + d_model))
        W = torch.empty((vocab_size, d_model))
        nn.init.trunc_normal_(W, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        self.W = nn.Parameter(W).to(device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int , eps: float = 1e-5 , device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        std = np.sqrt(2.0 / d_model)
        W = torch.ones(d_model)
        self.W = nn.Parameter(W).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        sum_quad = einops.reduce(x**2, "... d_model -> ... 1", reduction="sum")
        result = x / torch.sqrt(sum_quad / self.d_model + self.eps) * self.W
        return result.to(in_type)