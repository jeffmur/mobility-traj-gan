import logging
import random
import sys

import click
import numpy as np
import pandas as pd
from tensorflow.random import set_seed

sys.path.append(".")
from src.models import marc

MODELS = ["trajgan", "marc"]
DATASETS = ["mdc_lausanne", "foursquare_nyc", "geolife_beijing"]

LOG = logging.getLogger("mobility")
LOG.setLevel(logging.DEBUG)
fh = logging.FileHandler("mobility.log")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
LOG.addHandler(fh)

SEED = 11


def set_seeds(seed):
    """Set random seed for Python, NumPy and TensorFlow"""
    np.random.seed(seed)
    set_seed(seed)
    random.seed(seed)


@click.command()
@click.argument("model", type=click.Choice(MODELS))
@click.argument("dataset", type=click.Choice(DATASETS))
@click.argument("epochs", type=click.INT)
def train(model):
    """Train MODEL on DATASET for EPOCHS"""
    # TODO


@click.command()
@click.argument("saved_model", type=click.Path(exists=True, file_okay=False))
@click.argument("dataset", type=click.Choice(DATASETS))
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
@click.argument("dataset", type=click.Choice(DATASETS))
def predict():
    """Use SAVED_MODEL to predict the labels of DATASET."""
    # TODO


@click.command()
@click.argument("train_file", type=click.Path(exists=True))
@click.argument("test_file", type=click.Path(exists=True))
@click.argument("result_file", type=click.Path())
@click.argument("epochs", type=click.INT)
def train_marc(train_file, test_file, result_file, epochs):
    """Train the MARC trajectory-user linking classifier."""

    # TODO


@click.group()
def cli():
    """Command line interface for the mobility learning framework."""


def main():
    cli.add_command(train)
    cli.add_command(generate)
    cli.add_command(predict)
    cli()


if __name__ == "__main__":
    main()
