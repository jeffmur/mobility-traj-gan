import pandas as pd
import tensorflow as tf
from src.lib import config


def train_lstm_ae(epochs: int, batch_size: int = 64, learning_rate: float = 0.0001):
    """Current approach, simple sequential LSTM-AE model"""
    x = get_trajectory_samples(
        cell_size=config.CELL_SIZE_METERS * config.MILES_PER_METER,
        start_date=pd.Timestamp("2010-01-04"),
        end_date=pd.Timestamp("2011-01-03"),
        sample_resolution="4H",
        trajectory_period="W",
        min_points_per_trajectory=3,
    )

    n_trajectories = x[["UID", "Period"]].drop_duplicates()["UID"].count()
    n_timesteps = x.groupby(["UID", "Period"])["DateTime"].count().max()
    feature_values = x.Position.drop_duplicates().sort_values().tolist()
    n_feature_values = len(feature_values)
    x["Position"] = x["Position"].apply(lambda p: feature_values.index(p))
    x = x[["Position"]].values.reshape((n_trajectories, n_timesteps, 1))
    x = tf.convert_to_tensor(x)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(512, input_shape=(n_timesteps, 1), return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))
    model.add(
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(n_feature_values, activation="softmax")
        )
    )
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    history = model.fit(x, x, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)
    return history


if __name__ == "__main__":
    import os

    os.getcwd()
    os.chdir("../..")
    history = train_lstm_ae(epochs=500, batch_size=64, learning_rate=0.0001)
