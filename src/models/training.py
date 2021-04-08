from src.lib import config
from src.models.lstm import LSTMAutoEncoder
import pandas as pd
import tensorflow as tf

# TODO: parameterize max sequence length
# TODO: add padding and masking
# TODO: parameterize time series resampling resolution


def preprocess_time(df, start_date, end_date, sample_resolution, pad_value=-1):
    """Preprocess the time dimension of the GPS time series.
    Filter window to start and end date, resample the time series
    and pad missing periods with sentinel value
    """
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df[(df["DateTime"] >= start_date) & (df["DateTime"] <= end_date)]
    expanded_range = pd.date_range(start=start_date, end=end_date, freq=sample_resolution)
    df = (
        df[["DateTime", "UID", "Row", "Column"]]
        .groupby("UID")
        .apply(
            lambda g: g.set_index("DateTime")
            .resample(sample_resolution)
            .first()[["Row", "Column"]]
            .reindex(expanded_range, fill_value=-1)
        )
    )
    return df


def train_lstm_ae():
    """"""
    x = pd.read_csv(config.FM_MATRIX, index_col=0)
    n_subjects = x["UID"].nunique()
    model = LSTMAutoEncoder(
        optimizer="adam",
        loss="mse",
        n_subjects=n_subjects,
        n_features=2,
        lstm_units=32,
    )

    history = model.fit(x, x, epochs=20)
    return history


if __name__ == "__main__":
    train_lstm_ae()