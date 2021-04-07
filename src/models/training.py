from src.lib import config
from src.models.lstm import LSTMAutoEncoder
import pandas as pd
import tensorflow as tf

# TODO: decide between ragged tensor input vs. padding & masking
# considering distribution of input sequence lengths
# TODO: decide whether to aggregate / downsample time resolution


def train_lstm_ae():
    """"""
    x = pd.read_csv(config.FM_MATRIX, index_col=0)
    n_subjects = x["UID"].nunique()
    model = LSTMAutoEncoder(
        optimizer="adam",
        loss="mse",
        n_subjects=n_subjects,
        n_features=2,
        lstm_units=64,
    )

    history = model.fit(x, x, epochs=20)
    return history


if __name__ == "__main__":
    train_lstm_ae()