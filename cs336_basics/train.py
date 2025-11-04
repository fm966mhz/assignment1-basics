"""The training utils."""

from dataclasses import dataclass

import torch

from torch import nn
from torch import optim


@dataclass(frozen=True)
class TrainingConfig:
    num_steps: int
    batch_size: int
    context_length: int
    checkpoint_freq: int


def train_loop(
    model: nn.Module, optimizer: optim.Optimizer, config: TrainingConfig, ckpt_path: str
):
    pass
