"""datasets/privamov_lyon.py

Lyon, France
City Limits = ['45.7073666', '45.8082628', '4.7718134', '4.8983774']
Number of Users: 116

"""
import os
from logging import Logger
from pathlib import Path

import pandas as pd
from src import config, freq_matrix, preprocess
from src.datasets import Dataset

LOG = Logger(__name__)


class PrivamovLyon(Dataset):
    """
    PRIVA'MOV has been filtered to the greater area of Lyon, France. 

    Sonia Ben Mokhtar, Antoine Boutet, Louafi Bouzouina, Patrick Bonnel, Olivier Brette, et al..
        PRIVA’MOV: Analysing Human Mobility Through Multi-Sensor Datasets. NetMob 2017, Apr 2017,
        Milan, Italy. hal-01578557
    """

    def __init__(
        self,
        raw_data_path: os.PathLike = config.PRV_INPUT_DIR,
        processed_file: os.PathLike = config.PRV_INPUT_FILE,
    ):
        self.city_name = "Lyon, France"
        self.bounding_box = preprocess.fetch_geo_location(self.city_name)

        self.raw_data_path = Path(raw_data_path)
        self.processed_file = Path(processed_file)
        self.raw_columns = [
            "label",
            "datetime",
            "lon",
            "lat",
        ]
        self.label_column = "label"
        self.trajectory_column = "tid"
        self.datetime_column = "datetime"
        self.lat_column = "lat"
        self.lon_column = "lon"
        self.columns = [self.label_column, self.datetime_column, self.lat_column, self.lon_column]

    def preprocess(self):
        """Preprocess the raw data into a single CSV file of trajectory data.

        Long-running (takes a few minutes).
        """
        ## If it exits, return df
        if os.path.exists(self.processed_file):
            df = pd.read_csv(self.processed_file, parse_dates=[["datetime"]])
            return df

        # Must be using raw-gps (not a csv)
        raw_gps = self.raw_data_path / "raw-gps"

        LOG.info("Reading file: %s", raw_gps)
        df = pd.read_table(raw_gps, names=self.raw_columns)
        df.datetime = pd.to_datetime(df.datetime)

        ## Spatial bound 
        cell_size = config.CELL_SIZE_METERS * config.MILES_PER_METER  # meters * conversion to miles
        bounds, _, _ = freq_matrix.set_map(self.bounding_box, cell_size)
        df = freq_matrix.filter_bounds(df, bounds, "lat", "lon")

        ## Final format & Save
        df = df[self.columns]
        df.to_csv(self.processed_file, index=False)
        return df