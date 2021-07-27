"""mobility_cli.py

A Command-Line Interface for interacting with the mobility models and datasets.
"""
import logging
import os
import random
import sys

import click
import numpy as np
import pandas as pd
from tensorflow.random import set_seed

from src import datasets, models, config

LOG = logging.getLogger("src")
LOG.setLevel(logging.DEBUG)
fh = logging.FileHandler("mobility.log")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
LOG.addHandler(fh)
LOG.addHandler(ch)

DATASET_CHOICES = {klass.__name__: klass for klass in datasets.DATASETS}
MODEL_CHOICES = {klass.__name__: klass for klass in models.MODELS}


def set_seeds(seed):
    """Set random seed for Python, NumPy and TensorFlow"""
    np.random.seed(seed)
    set_seed(seed)
    random.seed(seed)


@click.command()
@click.argument("model", type=click.Choice(MODEL_CHOICES.keys()))
@click.argument("dataset", type=click.Choice(DATASET_CHOICES.keys()))
@click.argument("epochs", type=click.INT)
def train(model, dataset, epochs):
    """Train MODEL on DATASET stored in DATASET_PATH for EPOCHS."""
    the_dataset = DATASET_CHOICES.get(dataset)()
    the_model = MODEL_CHOICES.get(model)(the_dataset)
    LOG.info("Training model %s on %s for %d epochs.", model, dataset, epochs)
    the_model.train(epochs=epochs)


@click.command()
@click.argument("model", type=click.Choice(MODEL_CHOICES.keys()))
@click.argument("saved_path", type=click.Path(exists=True, file_okay=False))
@click.argument("dataset", type=click.Choice(DATASET_CHOICES))
@click.argument("output_path", type=click.Path(dir_okay=False))
def predict(model, saved_path, dataset, output_path):
    """Use trained MODEL saved in SAVED_PATH to make predictions based on DATASET
    and write to OUTPUT_PATH as CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    the_dataset = DATASET_CHOICES.get(dataset)()
    the_model = MODEL_CHOICES.get(model).restore(saved_path)
    _, df_test = the_dataset.train_test_split()
    the_model.predict(df_test).to_csv(output_path, index=False)
    LOG.info(
        "Model %s in %s predictions on %s saved to %s", model, saved_path, dataset, output_path
    )


@click.command()
@click.argument("model", type=click.Choice(MODEL_CHOICES.keys()))
@click.argument("saved_path", type=click.Path(exists=True, file_okay=False))
@click.argument("dataset", type=click.Choice(DATASET_CHOICES))
def evaluate(model, saved_path, dataset):
    """Use SAVED_MODEL to predict the labels of DATASET."""
    the_dataset = DATASET_CHOICES.get(dataset)()
    the_model = MODEL_CHOICES.get(model).restore(saved_path)
    _, df_test = the_dataset.train_test_split()
    metrics = the_model.evaluate(df_test)
    LOG.info("Model %s in %s evaluated on %s with metrics %s.", model, saved_path, dataset, metrics)


@click.group()
def cli():
    """Command line interface for the mobility learning framework."""


def main():
    cli.add_command(train)
    cli.add_command(predict)
    cli.add_command(evaluate)
    cli()


if __name__ == "__main__":
    set_seeds(config.SEED)
    main()
