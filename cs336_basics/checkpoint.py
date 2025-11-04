"""Checkpoint."""

import os
import typing

import torch

from torch import nn
from torch import optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the
            model, optimizer, and iteration to.
    """
    torch.save(
        {
            "iteration": iteration,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint_data = torch.load(src)
    print(f"Loaded checkpoint data: {checkpoint_data}")
    assert "iteration" in checkpoint_data, "`iteration` not found in checkpoint data."
    assert "model" in checkpoint_data, "`model` not found in checkpoint data."
    assert "optimizer" in checkpoint_data, "`optimizer` not found in checkpoint data."
    model.load_state_dict(checkpoint_data["model"])
    optimizer.load_state_dict(checkpoint_data["optimizer"])
    return checkpoint_data["iteration"]
