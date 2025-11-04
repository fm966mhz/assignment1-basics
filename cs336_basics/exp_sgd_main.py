"""Experiment with SGD."""

import torch

from absl import app
from absl import flags
from tqdm import tqdm

from cs336_basics.optimizers import SGD


FLAGS = flags.FLAGS

flags.DEFINE_float("lr", 1.0, "The learning rate.")


def main(argv):
    """Main function."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=FLAGS.lr)
