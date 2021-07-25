"""src/models/base.py

Base classes for models.
"""
import abc
import logging
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import pandas as pd
from src.datasets import Dataset


class TrajectoryModel(abc.ABC):
    """A base class for trajectory machine learning models."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.vocab_sizes = dataset.get_vocab_sizes()
        self.start_time = None
        self.end_time = None
        self.duration = None

    @abc.abstractmethod
    def train_test_split(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataset into a train and test set."""
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, optimizer, epochs: int, batch_size: int, **kwargs):
        """Train the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame):
        """Use the model to predict (or generate) given new input data."""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, save_path: os.PathLike):
        """Serialize the model to a checkpoint on disk."""

    @abc.abstractclassmethod
    def restore(cls, save_path: os.PathLike):
        """Restore the model from a checkpoint on disk."""

    def __repr__(self):
        return type(self).__name__


def log_start(log: logging.Logger, exp_name: str, **hparams: Dict[str, Any]):
    """Write a log entry for training experiment start."""
    start_time = datetime.now()
    log.info(f"Running experiment {exp_name} with hparams: {hparams}")
    return start_time


def log_end(log: logging.Logger, exp_name: str, start_time: datetime):
    """Write a log entry for training experiment end."""
    end_time = datetime.now()
    duration = end_time - start_time
    log.info(f"Experiment {exp_name} finished in {duration}")
    return duration
