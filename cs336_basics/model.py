import torch
import torch.nn as nn
import einops
import numpy as np
from cs336_basics.utils import scaled_dot_product_attn
from jaxtyping import Int

class Linear(nn.Module):
    def __init__(self, in_features: int , out_features: int , device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        std = np.sqrt(2.0 / (in_features+out_features))
        weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self, vocab_size: int , d_model: int , device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        std = np.sqrt(2.0 / (vocab_size + d_model))
        weight = torch.empty((vocab_size, d_model), device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int , eps: float = 1e-5 , device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        weight = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        sum_quad = einops.reduce(x**2, "... d_model -> ... 1", reduction="sum")
        result = x / torch.sqrt(sum_quad / self.d_model + self.eps) * self.weight
        return result.to(in_type)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w1.forward(x)
        silu = y * torch.sigmoid(y)
        z = silu * self.w3.forward(x)
        return self.w2.forward(z)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        position = torch.arange(max_seq_len, device=device).unsqueeze(-1)
        theta_tensor = torch.tensor([theta**((2*k-2) / d_k) for k in range(1, d_k//2+1)], device=device)
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

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float = None, max_seq_len: int = None,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # dimension for each attn head

        # init q_proj, k_proj, v_proj of shape (d_k * num_heads, d_model)
        output_dim = self.d_k * self.num_heads
        self.q_proj = Linear(d_model, output_dim, device, dtype)
        self.k_proj = Linear(d_model, output_dim, device, dtype)
        self.v_proj = Linear(d_model, output_dim, device, dtype)
        self.output_proj = Linear(output_dim, d_model, device, dtype)

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.q_proj.forward(x)
        Q = einops.rearrange(Q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)

        K = self.k_proj.forward(x)
        K = einops.rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)

        V = self.v_proj.forward(x)
        V = einops.rearrange(V, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)

        seq_len = Q.shape[-2]
        if self.rope and token_positions is not None:
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)
        # mask[i,j] = True means qi does attend to kj
        mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        # align mask's dim with Q/K/V for indexing purpose
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(Q.shape[:-2] + (seq_len, seq_len)).to(Q.device)
        heads = scaled_dot_product_attn(Q, K, V, mask)
        heads = einops.rearrange(heads, "... h seq_len d_k -> ... seq_len (h d_k)")
        return self.output_proj.forward(heads)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, eps: float = 1e-5,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps, device, dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.ln2 = RMSNorm(d_model, eps, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        y = x + self.attn.forward(self.ln1.forward(x), torch.arange(seq_len, device=self.device)) 
        return y + self.ffn.forward(self.ln2(y))

class TransformerLM(nn.Module):
    def __init__(self, num_layers: int, vocab_size:int, d_model: int, num_heads: int, d_ff: int, theta: float,
                 max_seq_len: int, eps: float = 1e-5, device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len, eps, device, dtype)
                                     for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, eps, device, dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, token_ids: Int[torch.Tensor, "..."]) -> torch.Tensor:
        res = self.token_embeddings.forward(token_ids)
        for block in self.layers:
            res = block.forward(res)
        return self.lm_head.forward(self.ln_final.forward(res))

class TransformerBlockWithoutLayerNorm(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, eps: float = 1e-5,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        y = x + self.attn.forward(x, torch.arange(seq_len, device=self.device))
        return y + self.ffn.forward(y)

class TransformerLMWithoutLayerNorm(nn.Module):
    def __init__(self, num_layers: int, vocab_size:int, d_model: int, num_heads: int, d_ff: int, theta: float,
                 max_seq_len: int, eps: float = 1e-5, device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([TransformerBlockWithoutLayerNorm(d_model, num_heads, d_ff, theta, max_seq_len, eps, device, dtype)
                                     for _ in range(num_layers)])
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, token_ids: Int[torch.Tensor, "..."]) -> torch.Tensor:
        res = self.token_embeddings.forward(token_ids)
        for block in self.layers:
            res = block.forward(res)
        return self.lm_head.forward(res)

class TransformerBlockPostNorm(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, eps: float = 1e-5,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps, device, dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.ln2 = RMSNorm(d_model, eps, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        y = self.ln1.forward(x + self.attn.forward(x, torch.arange(seq_len, device=self.device)))
        return self.ln2.forward(y + self.ffn.forward(y))

class TransformerLMPostNorm(nn.Module):
    def __init__(self, num_layers: int, vocab_size:int, d_model: int, num_heads: int, d_ff: int, theta: float,
                 max_seq_len: int, eps: float = 1e-5, device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([TransformerBlockPostNorm(d_model, num_heads, d_ff, theta, max_seq_len, eps, device, dtype)
                                     for _ in range(num_layers)])
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, token_ids: Int[torch.Tensor, "..."]) -> torch.Tensor:
        res = self.token_embeddings.forward(token_ids)
        for block in self.layers:
            res = block.forward(res)
        return self.lm_head.forward(res)