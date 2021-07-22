import os
from pathlib import Path

from src.datasets.base import Dataset
from src import config


class FourSquareNYC(Dataset):
    """The FourSquare NYC dataset."""

    def __init__(
        self,
        raw_data_path: os.PathLike = config.FSN_INPUT_DIR,
        processed_file: os.PathLike = config.FSN_INPUT_FILE,
    ):
        self.raw_data_path = Path(raw_data_path)
        self.processed_file = Path(processed_file)
        self.raw_columns = []
        self.label_column = "label"
        self.trajectory_column = "tid"
        self.datetime_column = "datetime"
        self.lat_column = "lat"
        self.lon_column = "lon"
        self.columns = [self.label_column, self.trajectory_column, self.lat_column, self.lon_column]

    def preprocess(self):
        """Preprocess the raw data into a single CSV file of trajectory data.

        Long-running (takes a few minutes).
        """
        # TODO
