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


def preprocess_time_gps(df, start_date, end_date, sample_resolution, pad_value=0):
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
        (
            df[["DateTime", "UID", "Latitude", "Longitude"]]
            .groupby("UID")
            .apply(
                lambda g: g.set_index("DateTime")
                .resample(sample_resolution)
                .mean()[["Latitude", "Longitude"]]
                .reindex(expanded_range, fill_value=pad_value)
            )
        )
        .fillna(pad_value)
        .reset_index()
    )
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


def prep_data_sessions(maxlen_ntile=0.5):
    """Prep mobility sequence session data"""
    x = pd.read_csv(config.FM_MATRIX, index_col=0)
    x = merge_datetime(x)
    x = rowcol_to_1d(x)
    x = split_gaps(x)
    # get indexes
    idxs = [
        list(v.values)
        for v in x.groupby(["UID", "Gap"]).groups.values()
        if len(v.values) > 1
    ]
    x_sessions = [x.iloc[i] for i in idxs]
    x_nested = [s.Position.tolist() for s in x_sessions]
    # get 90th percentile sequence length
    x_lengths = sorted([len(l) for l in x_nested])
    pct_idx = round(len(x_lengths) * 0.5)
    maxlen = x_lengths[pct_idx]
    x_pad = pad_sequences(
        x_nested, maxlen=maxlen, padding="pre", truncating="pre", value=0
    )
    x_enc = OrdinalEncoder().fit_transform(x_pad)
    return x_enc


def train_lstm_ae_seq():
    """"""
    x = prep_data_sessions()
    n_feature_values = len(np.unique(x))
    n_timesteps = x.shape[1]
    x = tf.convert_to_tensor(x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # opt = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.1, beta_2=0.001, amsgrad=True)
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


def prep_gps():
    """prepare raw gps data"""
    df = pd.read_csv(config.GPS_BB)
    start_date = pd.to_datetime("2010-01-04")
    end_date = pd.to_datetime("2011-01-02")
    traj_len = "7d"
    date_range = pd.date_range(
        start=start_date,
        end=end_date,
        closed="left",
        freq=traj_len,
        name="DateRange",
    )
    x = pd.concat(
        [
            preprocess_time_gps(
                df,
                date_range[i],
                date_range[i + 1],
                sample_resolution="60min",
                pad_value=0.0,
            ).assign(i=i)
            for i in range(len(date_range) - 1)
        ]
    )
    return x


def prep_fm():
    """prepare raw gps data"""
    sample_resolution = "4H"
    pad_value = 0.0
    df = pd.read_csv(config.FM_MATRIX, index_col=0)
    start_date = pd.to_datetime("2010-01-04")
    end_date = pd.to_datetime("2011-01-03")
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
    df["Period"] = pd.DatetimeIndex(df["DateTime"]).to_period("W").start_time
    # exclude periods where the user didn't have at least 3 points
    df = df.groupby(["UID", "Period"]).filter(
        lambda x: np.count_nonzero(x["Position"]) >= 3
    )
    return df


def train_lstm_ae():
    """"""
    x = prep_fm()

    # x_idx, x_test_idx = next(
    #     GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=7).split(
    #         x, groups=x["UID"]
    #     )
    # )
    # x = x.iloc[x_idx]
    n_trajectories = x[["UID", "Period"]].drop_duplicates()["UID"].count()
    n_timesteps = x.groupby(["UID", "Period"])["DateTime"].count().max()
    feature_values = x.Position.drop_duplicates().sort_values().tolist()
    n_feature_values = len(feature_values)
    x["Position"] = x["Position"].apply(lambda p: feature_values.index(p))
    x = x[["Position"]].values.reshape((n_trajectories, n_timesteps, 1))
    x = tf.convert_to_tensor(x)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # model = LSTMAutoEncoder(
    #     optimizer=opt,
    #     loss="sparse_categorical_crossentropy",
    #     n_timesteps=n_timesteps,
    #     n_feature_values=n_feature_values,
    #     embedding_length=2,
    #     lstm_units=1024,
    #     activation="tanh",
    #     metrics=["sparse_categorical_accuracy"],
    # )
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.LSTM(512, input_shape=(n_timesteps, 1), return_sequences=True)
    )
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
    history = model.fit(x, x, batch_size=64, epochs=100, verbose=1)
    return history


def train_lstmae_gps():
    x = prep_gps()
    x_idx, x_test_idx = next(
        GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=7).split(
            x, groups=x["UID"]
        )
    )
    x = x.iloc[x_idx]
    n_trajectories = x[["UID", "i"]].drop_duplicates()["UID"].count()
    n_timesteps = x.groupby(["UID", "i"])["DateTime"].count().max()
    x = x[["Latitude", "Longitude"]].values.reshape((n_trajectories, n_timesteps, 2))
    x = tf.convert_to_tensor(x)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.LSTM(512, input_shape=(n_timesteps, 2), return_sequences=True)
    )
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2)))
    model.compile(optimizer=opt, loss="mse", metrics=["mean_squared_error"])
    model.summary()
    history = model.fit(x, x, batch_size=64, epochs=500)
    return history


if __name__ == "__main__":
    import os

    os.chdir("../..")

    # x = prep_data_sessions()
