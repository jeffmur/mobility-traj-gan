from src.lib import config
from src.models.lstm import LSTMAutoEncoder
import pandas as pd
import tensorflow as tf


def preprocess_time(df, start_date, end_date, sample_resolution, pad_value=-1):
    """Preprocess the time dimension of the GPS time series.
    Filter window to start and end date, resample the time series
    and pad missing periods with sentinel value
    """
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df[(df["DateTime"] >= start_date) & (df["DateTime"] < end_date)]
    expanded_range = pd.date_range(
        start=start_date,
        end=end_date,
        closed="left",
        freq=sample_resolution,
        name="DateTime",
    )
    df = (
        df[["DateTime", "UID", "Row", "Column"]]
        .groupby("UID")
        .apply(
            lambda g: g.set_index("DateTime")
            .resample(sample_resolution)
            .first()[["Row", "Column"]]
            .reindex(expanded_range, fill_value=-1)
        )
    ).reset_index()
    return df


def train_lstm_ae():
    """"""
    x = pd.read_csv(config.FM_MATRIX, index_col=0)
    x = preprocess_time(
        x,
        pd.to_datetime("2010-04-01"),
        pd.to_datetime("2010-04-15"),
        sample_resolution="60min",
    )
    n_subjects = x["UID"].nunique()
    n_timesteps = x["DateTime"].nunique()
    n_features = 2
    x = x[["Row", "Column"]].values.reshape(
        (n_subjects, n_timesteps, n_features)
    )
    x = tf.convert_to_tensor(x)

    model = LSTMAutoEncoder(
        optimizer="adam",
        loss="mse",
        n_timesteps=n_timesteps,
        n_features=2,
        lstm_units=32,
    )

    # TODO:
    history = model.fit(x, x, batch_size=64, epochs=20)
    return history


if __name__ == "__main__":
    train_lstm_ae()
