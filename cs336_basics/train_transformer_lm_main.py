"""Train a Transfomer LM."""

import pickle

import numpy as np
import numpy.typing as npt
import wandb

from absl import app
from absl import logging
from absl import flags
from torch import nn
from torch import optim

from cs336_basics import checkpoint
from cs336_basics import optimizers
from cs336_basics import train_model
from cs336_basics import transformer


FLAGS = flags.FLAGS

# Input data.
flags.DEFINE_string("training_dataset_path", "", "The training data path.")
flags.DEFINE_string(
    "validation_dataset_path", "", "The path to the validation dataset."
)
# Output paths.
flags.DEFINE_string("metric_history_dump_path", "", "Path to the metric history dump.")
# Configs of the transformer.
flags.DEFINE_integer("vocab_size", None, "The vocab size.")
flags.DEFINE_integer("max_context_length", None, "The max context length.")
flags.DEFINE_integer("num_layers", None, "Number of layers.")
flags.DEFINE_integer("num_heads", None, "Number of heads.")
flags.DEFINE_float("rope_theta", None, "RoPE theta.")
flags.DEFINE_integer("d_model", None, "d_model.")
flags.DEFINE_float("d_ff_to_d_model", 8.0 / 3.0, "d_ff_to_d_model.")
# Configs of the AdamD optimizer.
flags.DEFINE_float("lr", 1e-3, "The learning rate.")
flags.DEFINE_float("weight_decay", 0.01, "Weight decay.")
flags.DEFINE_float("adamw_beta_1", 0.9, "AdamW beta_1.")
flags.DEFINE_float("adamw_beta_2", 0.999, "AdamW beta_2.")
flags.DEFINE_float("adamw_eps", 1e-8, "AdamW's eps.")
#  Configs of the checkpointing.
flags.DEFINE_string("checkpoint_dir_path", "", "Path to the checkpoint directory.")
flags.DEFINE_integer("max_num_checkpoints", None, "Max number of checkpoints to store.")
flags.DEFINE_integer("checkpoint_freq", None, "Checkpointing frequency.")
# Configs of WANDB.
flags.DEFINE_string("wandb_entity", "cs336-assignment-1", "wandb entity.")
flags.DEFINE_string("wandb_project", "test_train", "wandb project.")
# Configs of training.
flags.DEFINE_integer("num_steps", None, "Number of training steps.")
flags.DEFINE_integer("batch_size", None, "Training batch size.")
flags.DEFINE_integer("validation_batch_size", None, "Validation batch size.")
flags.DEFINE_integer("validation_freq", None, "Validation frequency.")
flags.DEFINE_string("device", None, "Device of the training.")


def _get_train_and_validaton_datasets() -> tuple[npt.NDArray, npt.NDArray]:
    return (
        np.load(FLAGS.training_dataset_path, mmap_mode="r"),
        np.load(FLAGS.validation_dataset_path, mmap_mode="r"),
    )


def _load_or_create_checkpoint_manager() -> checkpoint.CheckpointManager:
    return checkpoint.CheckpointManager(
        checkpoint_dir=FLAGS.checkpoint_dir_path,
        max_num_checkpoints=FLAGS.max_num_checkpoints,
    )


def _load_or_init_state(
    checkpoint_manager: checkpoint.CheckpointManager,
) -> tuple[nn.Module, optim.Optimizer, int]:
    model = transformer.TransformerLm(
        transformer.TransformerConfig(
            vocab_size=FLAGS.vocab_size,
            context_length=FLAGS.max_context_length,
            num_layers=FLAGS.num_layers,
            num_heads=FLAGS.num_heads,
            rope_theta=FLAGS.rope_theta,
            d_model=FLAGS.d_model,
            d_ff_to_d_model=FLAGS.d_ff_to_d_model,
        ),
        device=FLAGS.device,
    )
    optimizer = optimizers.AdamW(
        model.parameters(),
        lr=FLAGS.lr,
        weight_decay=FLAGS.weight_decay,
        betas=(FLAGS.adamw_beta_1, FLAGS.adamw_beta_2),
        eps=FLAGS.adamw_eps,
    )
    latest_checkpointed_iteration = checkpoint_manager.load_checkpoint(
        model=model, optimizer=optimizer
    )
    return (model, optimizer, latest_checkpointed_iteration)


def main(argv):
    """Runs the training."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line args,")

    logging.info(f"Loading checkpoint from {FLAGS.checkpoint_dir_path}...")
    checkpoint_manager = _load_or_create_checkpoint_manager()
    model, optimizer, latest_checkpointed_iteration = _load_or_init_state(
        checkpoint_manager=checkpoint_manager
    )
    if latest_checkpointed_iteration == 0:
        logging.info(
            f"No existing checkpoints in {FLAGS.checkpoint_dir_path}. Model and optimizer "
            "initialized."
        )
    else:
        logging.info(
            "Model and optimizer loaded from the latest checkpoint at iteration "
            f"{latest_checkpointed_iteration}."
        )

    logging.info(
        f"Mapping training and validation dataset from {FLAGS.training_dataset_path} and "
        f"{FLAGS.validation_dataset_path}."
    )
    training_dataset, validation_dataset = _get_train_and_validaton_datasets()
    logging.info("Training and validation datasets created.")

    logging.info("Creating wandb run...")
    wandb_run = wandb.init(
        entity=FLAGS.wandb_entity,
        project=FLAGS.wandb_project,
        config={"lr": FLAGS.lr, "dataset": "TinyStories"},
    )
    logging.info("wandb run created.")

    logging.info(f"Running main training loop for {FLAGS.num_steps} steps...")
    metric_history = train_model.train_loop(
        model=model,
        optimizer=optimizer,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        config=train_model.TrainingConfig(
            num_steps=FLAGS.num_steps,
            training_batch_size=FLAGS.batch_size,
            context_length=FLAGS.max_context_length,
            checkpoint_freq=FLAGS.checkpoint_freq,
            validation_batch_size=FLAGS.validation_batch_size,
            validation_freq=FLAGS.validation_freq,
            device=FLAGS.device,
        ),
        checkpoint_manager=checkpoint_manager,
        wandb_run=wandb_run,
        log_to_console=True,
    )
    logging.info("Main training loop completed.")
    with open(FLAGS.metric_history_dump_path, "wb") as f:
        pickle.dump(metric_history, f)


if __name__ == "__main__":
    app.run(main)
