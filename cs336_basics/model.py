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
        W = torch.ones(d_model)
        self.W = nn.Parameter(W).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        sum_quad = einops.reduce(x**2, "... d_model -> ... 1", reduction="sum")
        result = x / torch.sqrt(sum_quad / self.d_model + self.eps) * self.W
        return result.to(in_type)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        std = np.sqrt(2.0 / (self.d_ff + self.d_model))
        # init three matrices, matmul(W2, (SiLU(W1x) * W3x))
        W1 = torch.empty((self.d_ff, self.d_model))
        nn.init.trunc_normal_(W1, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        self.W1 = nn.Parameter(W1).to(device=device, dtype=dtype)

        W2 = torch.empty((self.d_model, self.d_ff))
        nn.init.trunc_normal_(W2, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        self.W2 = nn.Parameter(W2).to(device=device, dtype=dtype)

        W3 = torch.empty((self.d_ff, self.d_model))
        nn.init.trunc_normal_(W3, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        self.W3 = nn.Parameter(W3).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einops.einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = y * torch.sigmoid(y)
        z = silu * einops.einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einops.einsum(self.W2, z, "d_model d_ff, ... d_ff -> ... d_model")

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        position = torch.arange(max_seq_len).unsqueeze(-1)
        theta_tensor = torch.tensor([theta**((2*k-2) / d_k) for k in range(1, d_k//2+1)])
        theta_tensor = einops.rearrange(theta_tensor, "d_k -> 1 d_k")
        theta_ik = position / theta_tensor
        assert theta_ik.shape == (max_seq_len, d_k//2)
        self.register_buffer("cos_cached", torch.cos(theta_ik), persistent=False)
        self.register_buffer("sin_cached", torch.sin(theta_ik), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # token_positions.shape = (batch, seq_length)
        cos = self.cos_cached[token_positions] # shape = (batch, seq_length, d_k//2)
        sin = self.sin_cached[token_positions]
        x_even = x[..., 0::2] # shape = (batch, seq_length, d_k/2)
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        # 要把rotated_even和rotated_odd交错堆叠起来
        res = torch.stack([rotated_even, rotated_odd], dim=-1)
        return einops.rearrange(res, "... half two -> ... (half two)")
    