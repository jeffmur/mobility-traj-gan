from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from src.lib import config, freq_matrix, preprocess
from src.models.lstm import LSTMAutoEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def _get_grid(cell_size: float) -> pd.DataFrame:
    """Get input data for the model
    Parameters
    ----------
    cell_size: float
        The width of a square cell in miles
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the timestamped grid row and column positions of each user.
    """
    bounding_box = preprocess.fetchGeoLocation("Lausanne, District de Lausanne, Vaud, Switzerland")
    # ['46.5043006', '46.6025773', '6.5838681', '6.7208137']

    bounds, step, pix = freq_matrix.set_map(bounding_box, cell_size)
    df = pd.read_csv(Path(config.DATA_INPUT_DIR) / config.DATA_INPUT_FILE, index_col=0)
    _, _, df = freq_matrix.create_2d_freq(df, bounds, step, pix)
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    return df[["UID", "DateTime", "Column", "Row"]]


def get_trajectory_samples(
    cell_size: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    sample_resolution: str,
    trajectory_period: str,
    min_points_per_trajectory: int,
    pad_value: float = 0.0,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    cell_size: float
        The width of a square cell in miles
    start_date: pandas.Timestamp
        The start date of the study period
    end_date: pandas.Timestamp
        The end date of the study period
    sample_resolution: str
        A pandas DateOffset string indicating the resampling frequency
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    trajectory_period: str
        A pandas DateOffset string indicating the length of one trajectory example in time.
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    min_points_per_trajectory: int
        User trajectories with fewer points than this will get filtered out.
    pad_value: float
        The value to fill / pad missing periods in the time series.
    """
    df = _get_grid(cell_size)
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
    df["Period"] = pd.DatetimeIndex(df["DateTime"]).to_period(trajectory_period).start_time
    # exclude periods where the user didn't have enough points
    df = df.groupby(["UID", "Period"]).filter(
        lambda x: np.count_nonzero(x["Position"]) >= min_points_per_trajectory
    )
    return df


def preprocess_time_gps(df, start_date, end_date, sample_resolution, pad_value=0):
    """In case we need to train on raw GPS later."""
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
    """Prep mobility sequence session data split by session gaps"""
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
    pct_idx = round(len(x_lengths) * 0.5)
    maxlen = x_lengths[pct_idx]
    x_pad = pad_sequences(x_nested, maxlen=maxlen, padding="pre", truncating="pre", value=0)
    x_enc = OrdinalEncoder().fit_transform(x_pad)
    return x_enc


def get_gps_traj(min_points=2):
    """Split the GPS data into user-week trajectories at 10 minute resolution."""
    df = pd.read_csv(config.GPS_BB)
    df.columns = ["uid", "date", "time", "lat", "lon"]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.sort_values(["uid", "datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["weekday"] = df["datetime"].dt.weekday
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["week"] = df["datetime"].dt.isocalendar().week
    df["tenminute"] = (df["datetime"].dt.minute // 10 * 10).astype(int)
    df = (
        df.groupby(["uid", "year", "month", "week", "day", "weekday", "hour", "tenminute"])
        .agg({"lat": "mean", "lon": "mean"})
        .reset_index()
    )
    # filter out trajectories with fewer than min points
    df = df.groupby(["uid", "year", "week"]).filter(lambda x: len(x) >= min_points).reset_index()
    df["tid"] = df.groupby(["uid", "year", "week"]).ngroup()
    return df[["uid", "tid", "lat", "lon", "weekday", "hour"]]


def scale_gps(df):
    """Prepare GPS data with standardized distance from a centroid."""
    df1 = df.copy()
    kmeans = KMeans(n_clusters=1).fit(df1[["lat", "lon"]])
    centr_lat, centr_lon = kmeans.cluster_centers_[0]
    df1.loc[:, "lat"] = df1["lat"] - centr_lat
    df1.loc[:, "lon"] = df1["lon"] - centr_lon
    scaler = MinMaxScaler().fit(df1[["lat", "lon"]])
    df1.loc[:, ["lat", "lon"]] = scaler.transform(df1[["lat", "lon"]])
    return df1, kmeans, scaler


def gps_to_tensor(df, max_len_qtile=0.95):
    """transform trajectories DataFrame to a NumPy tensor"""
    tid_groups = df.groupby("tid").groups
    tid_dfs = [df.iloc[g] for g in tid_groups.values()]
    # UID is a label, split it into a separate vector
    uids = np.array([tdf.uid.values[0] for tdf in tid_dfs])
    # get percentile of sequence length
    x_lengths = sorted([len(l) for l in tid_dfs])
    pct_idx = round(len(x_lengths) * max_len_qtile)
    maxlen = x_lengths[pct_idx]
    x_nested = [tdf[["lat", "lon", "weekday", "hour"]].to_numpy() for tdf in tid_dfs]
    x_pad = pad_sequences(
        x_nested, maxlen=maxlen, padding="pre", truncating="pre", value=0.0, dtype=float
    )
    weekday_cat = to_categorical(x_pad[:, :, 2:3])
    hour_cat = to_categorical(x_pad[:, :, 3:4])
    x_pad = np.concatenate([x_pad[:, :, 0:2], weekday_cat, hour_cat], axis=2)
    return x_pad, uids


def tensor_to_gps(x, uids, kmeans, scaler):
    """Convert the tensor representation back to a data frame of GPS records."""
    tids = np.repeat(np.expand_dims(np.arange(0, x.shape[0]), axis=1), x.shape[1], axis=1).reshape(
        x.shape[0] * x.shape[1]
    )
    # tids = np.repeat(np.expand_dims(tids, axis=1), x.shape[1], axis=1).reshape(
    #     x.shape[0] * x.shape[1]
    # )
    uids = np.repeat(np.expand_dims(uids, axis=1), x.shape[1], axis=1).reshape(
        x.shape[0] * x.shape[1]
    )
    x_res = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
    df = pd.DataFrame(x_res)
    df["tid"] = tids
    df["uid"] = uids
    df = df[df[0] + df[1] != 0.0].reset_index()
    df.loc[:, ["lat", "lon"]] = scaler.inverse_transform(df[[0, 1]])
    df["lat"] = df["lat"] + kmeans.cluster_centers_[0][0]
    df["lon"] = df["lon"] + kmeans.cluster_centers_[0][1]
    df["weekday"] = df[list(range(2, 9))].idxmax(axis=1) - 2
    df["hour"] = df[list(range(9, 33))].idxmax(axis=1) - 9
    return df[["uid", "tid", "lat", "lon", "weekday", "hour"]]
