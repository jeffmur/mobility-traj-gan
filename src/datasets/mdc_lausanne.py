"""datasets/mdc_lausanne.py

The MDC dataset filtered to the city limits of Lausanne, CH.
"""
import os
from logging import Logger
from pathlib import Path

import pandas as pd
from src import config, freq_matrix, preprocess
from src.datasets.base import Dataset

LOG = Logger(__name__)


class MDCLausanne(Dataset):
    """The Mobility Data Challenge (MDC) dataset filtered to Lausanne, CH."""

    def __init__(
        self, raw_data_path: os.PathLike = config.MDC_INPUT_DIR, processed_file: os.PathLike = config.MDC_INPUT_FILE
    ):
        self.city_name = "Lausanne, District de Lausanne, Vaud, Switzerland"
        self.bounding_box = preprocess.fetch_geo_location(self.city_name)
        self.raw_data_path = Path(raw_data_path)
        self.processed_file = Path(processed_file)
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
        self.datetime_column = "datetime"
        self.lat_column = "lat"
        self.lon_column = "lon"
        self.columns = [self.label_column, self.trajectory_column, self.lat_column, self.lon_column]

    def preprocess(self):
        """Preprocess the raw data into a single CSV file of trajectory data.

        Long-running (takes a few minutes).

        Parameters
        ----------
        output_file : os.PathLike
            The file path to save the processed CSV data.
        """

        if os.path.exists(self.processed_file):
            df = pd.read_csv(self.processed_file)
            df.datetime = pd.to_datetime(df.datetime)
            return df
            
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
        df.to_csv(self.processed_file, index=False)
        LOG.info("Preprocessed data written to: %s", self.processed_file)
        return df
