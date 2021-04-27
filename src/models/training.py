import numpy as np
from src.lib import config
from src.models.lstm import LSTMAutoEncoder
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder


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


def label_user_gaps(df, max_gap="60min"):
    df["Gap"] = (df["DateTime"].diff() > max_gap).astype(int).cumsum()
    return df


def split_gaps(df, max_gap="60min"):
    users = df.sort_values("DateTime").groupby("UID")
    df = users.apply(label_user_gaps).reset_index()
    return df


def merge_datetime(df):
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.drop(["Date", "Time"], axis=1)
    return df


def rowcol_to_1d(df):
    df["Position"] = ((df["Row"] * df["Column"].max()) + df["Column"] + 1).astype(int)
    df = df.drop(["Row", "Column"], axis=1)
    return df


def prep_data_sessions():
    """Prep mobility sequence session data"""
    x = pd.read_csv(config.FM_MATRIX, index_col=0)
    x = merge_datetime(x)
    x = rowcol_to_1d(x)
    x = split_gaps(x)
    # get indexes
    idxs = [list(v.values) for v in x.groupby(["UID", "Gap"]).groups.values() if len(v.values) > 1]
    x_sessions = [x.iloc[i] for i in idxs]
    x_nested = [s.Position.tolist() for s in x_sessions]
    # get 90th percentile sequence length
    x_lengths = sorted([len(l) for l in x_nested])
    pct_idx = round(len(x_lengths) * 0.9)
    maxlen = x_lengths[pct_idx]
    x_pad = pad_sequences(x_nested, maxlen=maxlen, padding="pre", truncating="pre", value=0)
    x_enc = OrdinalEncoder().fit_transform(x_pad)
    return x_enc


def train_lstm_ae_seq():
    """"""
    x = prep_data_sessions()
    n_feature_values = len(np.unique(x))
    n_timesteps = x.shape[1]
    x = tf.convert_to_tensor(x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model = LSTMAutoEncoder(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        n_timesteps=n_timesteps,
        n_feature_values=n_feature_values,
        embedding_length=3,
        lstm_units=100,
        activation="tanh",
        metrics=["sparse_categorical_accuracy"],
    )

    history = model.fit(x, x, batch_size=32, epochs=10)
    return history


def train_lstm_ae():
    """"""
    x = pd.read_csv(config.FM_MATRIX, index_col=0)

    x = preprocess_time(
        x,
        pd.to_datetime("2010-05-01"),
        pd.to_datetime("2010-05-08"),
        sample_resolution="1min",
    )
    x_idx, x_test_idx = next(
        GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=7).split(x, groups=x["UID"])
    )
    x = x.iloc[x_idx]
    n_subjects = x["UID"].nunique()
    n_timesteps = x["DateTime"].nunique()
    feature_values = x.Position.drop_duplicates().sort_values().tolist()
    n_feature_values = len(feature_values)
    x["Position"] = x["Position"].apply(lambda p: feature_values.index(p))
    x = x[["Position"]].values.reshape((n_subjects, n_timesteps))
    x = tf.convert_to_tensor(x)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model = LSTMAutoEncoder(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        n_timesteps=n_timesteps,
        n_feature_values=n_feature_values,
        embedding_length=3,
        lstm_units=100,
        activation="tanh",
        metrics=["sparse_categorical_accuracy"],
    )

    history = model.fit(x, x, batch_size=32, epochs=10)
    return history


if __name__ == "__main__":
    history = train_lstm_ae()
