import torch
import einops
import numpy as np
from jaxtyping import Float, Int
from collections.abc import Iterable
import os
import typing

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

def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return t / T_w * alpha_max
    if t <= T_c:
        return alpha_min + 0.5 * (1 + np.cos((t - T_w)  / (T_c - T_w) * np.pi)) * (alpha_max - alpha_min)
    return alpha_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params_with_grad = list([p for p in parameters if p.grad is not None])
    sqrd_sum = 0
    for p in params_with_grad:
        sqrd_sum += torch.sum(p.grad.data ** 2)
    G = torch.sqrt(sqrd_sum)
    if G > max_l2_norm:
        factor = max_l2_norm / (G + 1e-6)
        for p in params_with_grad:
            p.grad.data *= factor

def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_data = torch.from_numpy(x)
    max_idx = len(x) - (context_length + 1) # need to sample length = context_length + 1
    start_indices = torch.randint(0, max_idx+1, (batch_size, 1)) # shape = (B, 1)
    offsets = torch.arange(context_length + 1) # shape = (m,)
    idx_grid = start_indices + offsets # shape = (B, m) by broadcasting
    sampled_data = tensor_data[idx_grid].to(device)
    return sampled_data[:, :-1], sampled_data[:, 1:]

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj = dict()
    obj['model'] = model.state_dict()
    obj['optimizer'] = optimizer.state_dict()
    obj['iteration'] = iteration
    torch.save(obj, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']