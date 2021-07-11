"""datasets/mdc_lausanne.py

The MDC dataset filtered to the city limits of Lausanne, CH.
"""
from logging import Logger
import abc
import os
from pathlib import Path

import pandas as pd

from src.datasets import Dataset
from src.lib import config, preprocess, freq_matrix

LOG = Logger(__name__)


class MDCLausanne(Dataset):
    """The Mobility Data Challenge (MDC) dataset filtered to Lausanne, CH."""

    def __init__(self, raw_data_path):
        self.city_name = config.CITY
        self.bounding_box = preprocess.fetch_geo_location(self.city_name)
        self.raw_data_path = Path(raw_data_path)
        self.raw_columns = [
            "rid",
            "unix",
            "lon",
            "lat",
            "alt",
            "speed",
            "h_accuracy",
            "h_dop",
            "v_accuracy",
            "v_dop",
            "speed_accuracy",
            "time_since_boot",
        ]
        self.label_column = "label"
        self.trajectory_column = "tid"
        self.lat_column = "lat"
        self.lon_column = "lon"
        self.columns = [self.label_column, self.trajectory_column, self.lat_column, self.lon_column]

    def preprocess(self, output_file: os.PathLike = "data/mdc_lausanne.csv"):
        """Preprocess the raw data into a single CSV file of trajectory data.

        Long-running (takes a few minutes).

        Parameters
        ----------
        output_file : os.PathLike
            The file path to save the processed CSV data.
        """

        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            df.datetime = pd.to_datetime(df.datetime)
            return df
        output_file = Path(output_file)
        raw_gps = self.raw_data_path / "gps.csv"
        raw_records = self.raw_data_path / "records.csv"
        # Read in raw gps
        LOG.info("Reading file: %s", raw_gps)
        df = pd.read_table(raw_gps, names=self.raw_columns)
        # Drop extra columns
        df = df[["rid", "unix", "lat", "lon"]]
        # filter to bounds
        cell_size = config.CELL_SIZE_METERS * config.MILES_PER_METER  # meters * conversion to miles
        bounds, _, _ = freq_matrix.set_map(self.bounding_box, cell_size)
        df = freq_matrix.filter_bounds(df, bounds, "lat", "lon")
        # Convert Unix to datetime
        df.loc[:, "datetime"] = pd.to_datetime(df.unix, unit="s")
        df = df.drop(["unix"], axis=1)
        # Read GPS records in chunks to join and get user id label
        rec_headers = ["rid", "label", "tz", "time", "type"]
        LOG.info("Reading file: %s", raw_records)
        iter_rec = pd.read_table(raw_records, names=rec_headers, iterator=True, chunksize=100000)
        df_rec = pd.concat([chunk[chunk["type"] == "gps"] for chunk in iter_rec])
        # Join gps to records on RID
        df = df.set_index("rid")
        df_rec = df_rec.set_index("rid")
        df = df.join(df_rec, how="inner").reset_index()
        df = df.drop_duplicates(subset=["rid"])
        df = df[self.columns]
        df.to_csv(output_file, index=False)
        LOG.info("Preprocessed data written to: %s", output_file)
        return df

    def to_trajectories(self, min_points: int = 2):
        """Return the dataset as a Pandas DataFrame split into user-week trajectories.

        Multiple points within a ten minute interval will be removed.

        Parameters
        ----------
        min_points : int
            The minimum number of location points (rows) in a
            trajectory to include it in the dataset.
        """

        df = self.preprocess()
        df = df.sort_values(["label", "datetime"])
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["day"] = df["datetime"].dt.day
        df["weekday"] = df["datetime"].dt.weekday
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["week"] = df["datetime"].dt.isocalendar().week
        df["tenminute"] = (df["datetime"].dt.minute // 10 * 10).astype(int)
        df = (
            df.groupby(
                [self.label_column, "year", "month", "week", "day", "weekday", "hour", "tenminute"]
            )
            .agg({"lat": "mean", "lon": "mean"})
            .reset_index()
        )
        # filter out trajectories with fewer than min points
        df = (
            df.groupby(["label", "year", "week"])
            .filter(lambda x: len(x) >= min_points)
            .reset_index()
        )
        df[self.trajectory_column] = df.groupby([self.label_column, "year", "week"]).ngroup()
        df = df[[self.label_column, self.trajectory_column, "lat", "lon", "weekday", "hour"]]

        return df

    def get_vocab_sizes(self):
        """Get a dictionary of categorical features and their cardinalities."""
        df = self.to_trajectories()
        return (
            df.drop([self.label_column, self.trajectory_column], axis=1, errors="ignore")
            .select_dtypes("int")
            .nunique()
            .to_dict()
        )
