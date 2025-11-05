"""The training utils."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import wandb

from absl import logging
from jaxtyping import Float
from torch import nn
from torch import optim
from tqdm import tqdm

from cs336_basics.checkpoint import CheckpointManager
from cs336_basics.data_loader import get_batch
from cs336_basics.functions import cross_entropy


@dataclass(frozen=True)
class TrainingConfig:
    """Training config."""

    num_steps: int
    training_batch_size: int
    context_length: int
    checkpoint_freq: int

    validation_batch_size: int
    validation_freq: int

    device: torch.device | None = None


def train_loop(
    model: nn.Module,
    optimizer: optim.Optimizer,
    training_dataset: npt.NDArray,
    validation_dataset: npt.NDArray,
    config: TrainingConfig,
    checkpoint_manager: CheckpointManager,
    wandb_run: wandb.Run,
    log_to_console: bool = True,
) -> dict[str, list[float]]:
    """The main training loop."""
    metric_history = {
        "training_loss": [],
        "validation_loss": [],
        "validation_perplexity": [],
    }
    latest_checkpointed_iteration = (
        checkpoint_manager.checkpoint_metadata.latest_checkpointed_iteration
    )
    for t in tqdm(
        range(
            latest_checkpointed_iteration,
            latest_checkpointed_iteration + config.num_steps,
        )
    ):
        optimizer.zero_grad()
        # TODO(djwenren): pinning CPU memories?
        input_seq, label_seq = get_batch(
            dataset=training_dataset,
            batch_size=config.training_batch_size,
            context_length=config.context_length,
            device=config.device,
        )
        logits: Float[torch.Tensor, "batch_size seq_len vocab_size"] = model(input_seq)
        loss = cross_entropy(logits=logits, targets=label_seq)
        loss_val = loss.detach().cpu().item()
        metric_history["training_loss"].append(loss_val)
        if wandb_run is not None:
            wandb_run.log({"training_loss": loss_val})

        loss.backward()
        optimizer.step()

        if (t + 1 - latest_checkpointed_iteration) % config.validation_freq == 0:
            validation_loss, validation_perplexity = run_validation(
                model=model,
                validation_dataset=validation_dataset,
                config=config,
                wandb_run=wandb_run,
            )
            metric_history["validation_los"].append(validation_loss)
            metric_history["validation_perplexity"].append(validation_perplexity)
            if log_to_console:
                logging.info(
                    f"Iteration {t+1}. Training loss: {loss_val}. Validation loss: "
                    f"{validation_loss}. Validation perplexity: {validation_perplexity}."
                )

        if (t + 1 - latest_checkpointed_iteration) % config.checkpoint_freq == 0:
            checkpoint_manager.save_checkpoint(
                model=model, optimizer=optimizer, iteration=t + 1
            )
    return metric_history


def run_validation(
    model: nn.Module,
    validation_dataset: npt.NDArray,
    config: TrainingConfig,
    wandb_run: wandb.Run,
) -> tuple[float, float]:
    """Runs valiation.

    Returns the validation loss and perplexity.
    """
    input_seq, label_seq = get_batch(
        dataset=validation_dataset,
        batch_size=config.validation_batch_size,
        context_length=config.context_length,
        device=config.device,
    )
    with torch.no_grad():
        logits: Float[torch.Tensor, "batch_size seq_len vocab_size"] = model(input_seq)
    loss = cross_entropy(logits=logits, targets=label_seq).detach().cpu().item()
    # Perplexity is just the exponential of the cross entropy loss.
    perplexity = np.exp(loss)
    if wandb_run is not None:
        wandb_run.log(
            {
                "validation_loss": loss,
                "validation_perplexity": perplexity,
            }
        )
    return loss, perplexity
