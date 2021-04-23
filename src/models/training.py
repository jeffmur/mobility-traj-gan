from src.lib import config
from src.models.lstm import LSTMAutoEncoder
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit


def preprocess_time(df, start_date, end_date, sample_resolution, pad_value=0):
    """Preprocess the time dimension of the GPS time series.
    Filter window to start and end date, resample the time series
    and pad missing periods with sentinel value (default 0)
    """
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df[(df["DateTime"] >= start_date) & (df["DateTime"] < end_date)]
    df["Position"] = (df["Row"] * df["Column"].max()) + df["Column"] + 1
    expanded_range = pd.date_range(
        start=start_date,
        end=end_date,
        closed="left",
        freq=sample_resolution,
        name="DateTime",
    )
    df = (
        (
            df[["DateTime", "UID", "Position"]]
            .groupby("UID")
            .apply(
                lambda g: g.set_index("DateTime")
                .resample(sample_resolution)
                .first()[["Position"]]
                .reindex(expanded_range, fill_value=pad_value)
            )
        )
        .fillna(pad_value)
        .reset_index()
    )
    df["Position"] = df["Position"].astype(int)
    return df


def train_lstm_ae():
    """"""
    x = pd.read_csv(config.FM_MATRIX, index_col=0)
    x = preprocess_time(
        x,
        pd.to_datetime("2010-05-01"),
        pd.to_datetime("2010-05-08"),
        sample_resolution="60min",
    )
    x_idx, x_test_idx = next(
        GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=7).split(
            x, groups=x["UID"]
        )
    )
    x = x.iloc[x_idx]
    n_subjects = x["UID"].nunique()
    n_timesteps = x["DateTime"].nunique()
    n_feature_values = x["Position"].max() + 1
    x = x[["Position"]].values.reshape((n_subjects, n_timesteps))
    x = tf.convert_to_tensor(x)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model = LSTMAutoEncoder(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        n_timesteps=n_timesteps,
        n_feature_values=n_feature_values,
        embedding_length=32,
        lstm_units=100,
        activation="tanh",
        metrics=["sparse_categorical_accuracy"],
    )

    history = model.fit(x, x, batch_size=32, epochs=10, validation_split=0.2)
    return history


def train_synthetic():
    """"""


if __name__ == "__main__":
    history = train_lstm_ae()
