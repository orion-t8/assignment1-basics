import torch
import einops
import numpy as np

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    m = torch.max(x, dim=i, keepdim=True).values
    rescaled = torch.exp(x - m)
    return rescaled / torch.sum(rescaled, dim=i, keepdim=True)

def scaled_dot_product_attn(Q: torch.tensor, K: torch.tensor, V: torch.tensor, mask: bool[torch.Tensor, "seq_len seq_len"]):
    z = einops.einsum(Q, K, "batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k") / np.sqrt(Q.shape[-1])