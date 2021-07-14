"""models

Base class for trajectory machine learning models.
"""

from .lstm_trajgan import LSTMTrajGAN
from .marc import MARC
from src.datasets import Dataset

MODELS = [LSTMTrajGAN, MARC]
