"""models

Base class for trajectory machine learning models.
"""
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from .lstm_trajgan import LSTMTrajGAN
from .marc import MARC
from src.datasets import Dataset

MODELS = [LSTMTrajGAN, MARC]
