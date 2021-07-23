import os
from logging import Logger
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.datasets import Dataset
from src import config

LOG = Logger(__name__)


class FourSquareNYC(Dataset):
    """The FourSquare NYC dataset."""

    def __init__(
        self,
        raw_data_path: os.PathLike = config.FSN_INPUT_DIR,
        processed_file: os.PathLike = config.FSN_INPUT_FILE,
    ):
        self.raw_data_path = Path(raw_data_path)
        self.processed_file = Path(processed_file)
        self.raw_columns = config.DATA_HEADERS.get("FSN")
        self.label_column = "label"
        self.trajectory_column = "tid"
        self.datetime_column = "datetime"
        self.lat_column = "lat"
        self.lon_column = "lon"
        self.category_column = "category"
        self.columns = [
            self.label_column,
            self.trajectory_column,
            self.lat_column,
            self.lon_column,
            self.category_column,
        ]
        self.category_encoder = OrdinalEncoder()

    def preprocess(self):
        """Preprocess the raw data into a single CSV file of trajectory data.

        Long-running (may take around a minute)
        """
        if os.path.exists(self.processed_file):
            return pd.read_csv(self.processed_file, parse_dates=["datetime"])
        nyc_file = self.raw_data_path / "dataset_TSMC2014_NYC.txt"
        LOG.info("Reading file: %s", nyc_file)
        df = pd.read_table(nyc_file, names=self.raw_columns)
        df["datetime"] = pd.to_datetime(df["utc_time"])
        df["category"] = self.category_encoder.fit_transform(df[["category_id"]]).astype(int)
        df = df.rename(columns={"user_id": "label"})
        df = df[["label", "datetime", "lat", "lon", "category"]]
        df.to_csv(self.processed_file, index=False)
        LOG.info("Preprocessed data written to: %s", self.processed_file)
        return df

    def to_trajectories(self, min_points=2, *args):
        """Return the dataset as a Pandas DataFrame split into user-week trajectories.

        Multiple points within a ten minute interval will be removed.

        Parameters
        ----------
        min_points : int
           The minimum number of location points (rows) in a
           trajectory to include it in the dataset.
        args : str
            The names of any extra categorical columns to pass through.
        """
        return super().to_trajectories(min_points, "category", *args)
