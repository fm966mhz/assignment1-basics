import einops
import numpy as np
import torch

from jaxtyping import Float
from torch import nn


class Linear(nn.Module):
    """Linear module."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features, dtype=dtype, device=device),
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_out"]:
        """Forward pass."""
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
