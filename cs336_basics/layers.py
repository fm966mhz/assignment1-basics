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
        self, token_ids: Float[torch.LongTensor, "batch_size sequence_length"]
    ) -> Float[torch.Tensor, "batch_size sequence_length embedding_dim"]:
        """Foward pass."""
        return self.weight[token_ids]
