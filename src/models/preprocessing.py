from pathlib import Path
import numpy as np
import pandas as pd

from src.lib import config, freq_matrix, preprocess


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
    bounding_box = preprocess.fetchGeoLocation(
        "Lausanne, District de Lausanne, Vaud, Switzerland"
    )
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
    pad_value: float,
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
    df["Period"] = (
        pd.DatetimeIndex(df["DateTime"]).to_period(trajectory_period).start_time
    )
    # exclude periods where the user didn't have at least 3 points
    df = df.groupby(["UID", "Period"]).filter(
        lambda x: np.count_nonzero(x["Position"]) >= 3
    )
    return df
