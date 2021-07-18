"""mobility_cli.py

A Command-Line Interface for interacting with the mobility models and datasets.
"""
import logging
import random
import sys

import click
import numpy as np
import pandas as pd
from tensorflow.random import set_seed

sys.path.append(".")
from src import datasets, models

LOG = logging.getLogger("mobility")
LOG.setLevel(logging.DEBUG)
fh = logging.FileHandler("mobility.log")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
LOG.addHandler(fh)

DATASET_CHOICES = {klass.__name__: klass for klass in datasets.DATASETS}
MODEL_CHOICES = {klass.__name__: klass for klass in models.MODELS}


def set_seeds(seed):
    """Set random seed for Python, NumPy and TensorFlow"""
    np.random.seed(seed)
    set_seed(seed)
    random.seed(seed)


@click.command()
@click.argument("saved_model", type=click.Path(exists=True, file_okay=False))
@click.argument("dataset", type=click.Choice(DATASET_CHOICES))
@click.argument("output_path", type=click.Path(dir_okay=False))
@click.option(
    "--n",
    type=click.INT,
    default=-1,
    help="The number of synthetic examples to generate. Default is the size of original dataset.",
)
def generate(saved_model, dataset, output_path, n):
    """Use SAVED_MODEL to generate NUM trajectories based on DATASET and write to OUTPUT_PATH as CSV."""
    # TODO


@click.command()
@click.argument("saved_model", type=click.Path(exists=True, file_okay=False))
@click.argument("dataset", type=click.Choice(DATASET_CHOICES))
def predict():
    """Use SAVED_MODEL to predict the labels of DATASET."""
    # TODO


@click.command()
@click.argument("model", type=click.Choice(MODEL_CHOICES.keys()))
@click.argument("dataset", type=click.Choice(DATASET_CHOICES.keys()))
@click.argument("dataset_path", type=click.Path(exists=True))
@click.argument("epochs", type=click.INT)
def train(model, dataset, dataset_path, epochs):
    """Train MODEL on DATASET stored in DATASET_PATH for EPOCHS."""
    the_dataset = DATASET_CHOICES.get(dataset)(dataset_path)
    the_model = MODEL_CHOICES.get(model)(the_dataset)
    the_model.train(epochs=epochs)


@click.group()
def cli():
    """Command line interface for the mobility learning framework."""


def main():
    cli.add_command(train)
    cli.add_command(generate)
    cli.add_command(predict)
    cli()


if __name__ == "__main__":
    SEED = 11
    set_seeds(SEED)
    main()
