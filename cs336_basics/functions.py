import torch

from jaxtyping import Float


def silu(x: Float[torch.Tensor, "..."]):
    """Runs SiLU."""
    return x * torch.sigmoid(x)
