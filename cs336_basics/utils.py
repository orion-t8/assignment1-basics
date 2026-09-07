import torch
import einops
import numpy as np
from jaxtyping import Float, Int

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    m = torch.max(x, dim=i, keepdim=True).values
    rescaled = torch.exp(x - m)
    return rescaled / torch.sum(rescaled, dim=i, keepdim=True)

def scaled_dot_product_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    z = einops.einsum(Q, K, "batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k") / np.sqrt(Q.shape[-1])
    if mask is not None:
        z[~mask] = -torch.inf
    factor = softmax(z, i=-1)
    return einops.einsum(factor, V, "batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v")

def cross_entropy(logits: Float[torch.Tensor, "batch vocab_size"], targets: Int[torch.Tensor, "batch"]) -> Float[torch.Tensor, ""]:
    subtracted = logits - torch.max(logits, dim=-1, keepdim=True).values # shape = (batch, )
    logsumexp = torch.log(torch.sum(torch.exp(subtracted), dim=-1))
    selected = torch.gather(subtracted, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(logsumexp - selected)

def model_size(vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int) -> int:
    embedding = vocab_size * d_model
    d_k = d_model // num_heads
    transformer_block = num_layers * (2 * d_model + 4 * num_heads * d_k * d_model + 3 * d_model * d_ff)
    ln_final = d_model
    lm_head = d_model * vocab_size
    return embedding + transformer_block + ln_final + lm_head