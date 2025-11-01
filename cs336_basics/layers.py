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


class Embedding(nn.Module):
    """Embedding module."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device),
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(
        self, token_ids: Float[torch.LongTensor, "..."]
    ) -> Float[torch.Tensor, "... embedding_dim"]:
        """Foward pass."""
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """RMSNorm module."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.gain = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[torch.Tensor, "... d_model"]):
        """Forward pass."""
        assert x.shape[-1] == self.gain.shape[0], (
            f"Input shape {x.shape}'s last dimension is different from what this module is expected"
            " {self.gain.shape}"
        )
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(self.eps + einops.reduce(x**2, "... d_model -> ...", "mean"))
        result = einops.einsum(
            x, 1.0 / rms, self.gain, "... d_model, ..., d_model -> ... d_model"
        )
        return result.to(in_dtype)
