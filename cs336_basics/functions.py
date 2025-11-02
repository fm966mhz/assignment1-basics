"""Functions."""

import torch

from jaxtyping import Float


def silu(x: Float[torch.Tensor, "..."]):
    """Runs SiLU."""
    return x * torch.sigmoid(x)


def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    """Runs softmax."""
    x_diffed = x - x.max(dim=dim, keepdim=True)[0]
    x_exped = torch.exp(x_diffed)
    x_exped_sumed = torch.sum(x_exped, dim=dim, keepdim=True)
    return x_exped / x_exped_sumed
