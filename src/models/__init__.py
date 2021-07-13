"""models

Base class for trajectory machine learning models.
"""
import abc
from datetime import datetime
import logging
import os

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
    def train(self, optimizer, epochs: int, batch_size: int, **kwargs):
        """Train the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, save_path: os.PathLike):
        """Serialize the model to a checkpoint on disk."""

    @abc.abstractmethod
    def restore(self, save_path: os.PathLike):
        """Restore the model from a checkpoint on disk."""


class Generative(abc.ABC):
    """"""

    @abc.abstractmethod
    def generate(self, dataset: Dataset, n: int):
        """Generate synthetic examples from this model."""
