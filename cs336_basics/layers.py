"""Layers."""

import einops
import numpy as np
import torch

from jaxtyping import Float
from torch import nn

from cs336_basics.functions import silu


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


class SwiGLU(nn.Module):
    """SwiGLU module."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.in_projection_layer_1 = Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )
        self.in_projection_layer_3 = Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )
        self.out_projection_layer_2 = Linear(
            in_features=d_ff, out_features=d_model, device=device, dtype=dtype
        )

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        """Forward."""
        out_1 = self.in_projection_layer_1(x)
        out_3 = self.in_projection_layer_3(x)
        return self.out_projection_layer_2(silu(out_1) * out_3)


class Rope(nn.Module):
    """RoPE module."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        theta_tensor = einops.einsum(
            # Positions are zero-indexed.
            torch.arange(max_seq_len, device=device),
            1.0 / theta ** (2 * torch.arange(d_k // 2, device=device) / d_k),
            "seq_len, half_d_k -> seq_len half_d_k",
        )
        cosine_matrix = einops.einsum(
            torch.cos(theta_tensor),
            torch.tensor([[1.0, 0], [0, 1.0]], device=device),
            "seq_len half_d_k, r_out r_in -> seq_len half_d_k r_out r_in",
        )
        sine_matrix = einops.einsum(
            torch.sin(theta_tensor),
            torch.tensor([[0, -1.0], [1.0, 0]], device=device),
            "seq_len half_d_k, r_out r_in -> seq_len half_d_k r_out r_in",
        )
        self.register_buffer(
            "rope_matrix", cosine_matrix + sine_matrix, persistent=False
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Float[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """Forward pass."""
        position_embeddings: Float[torch.Tensor, "... seq_len half_d_k r_out r_in"] = (
            self.rope_matrix[token_positions]
        )
        x_rearanged = einops.rearrange(
            x, "... seq_len (half_d_k r_in) -> ... seq_len half_d_k r_in", r_in=2
        )
        output = einops.einsum(
            x_rearanged,
            position_embeddings,
            (
                "... seq_len half_d_k r_in, ... seq_len half_d_k r_out r_in -> "
                "... seq_len half_d_k r_out"
            ),
        )
        return einops.rearrange(
            output,
            "... seq_len half_d_out r_out -> ... seq_len (half_d_out r_out)",
        )
