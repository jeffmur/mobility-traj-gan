"""datasets/geolife_beijing.py

The GeoLife dataset is filtered by the city limits of Beijing
City Limits = ['39.4416113', '41.0595584', '115.4172086', '117.5079852']

"""
import os
from logging import Logger, raiseExceptions
from pathlib import Path
import glob

import pandas as pd
from src import config, freq_matrix, preprocess
from src.datasets import Dataset

LOG = Logger(__name__)


class GeoLifeBeijing(Dataset):
    """ 
    GeoLife by Microsoft has been filtered with OpenStreetMaps Nominum API within 'Bejing, China' 
    Data is organized as User/Trajectory/csv
    
    """

    def __init__(
        self, raw_data_path: os.PathLike = config.GEO_INPUT_DIR, processed_file: os.PathLike = config.GEO_INPUT_FILE
    ):
        self.city_name = "Beijing, China"
        self.bounding_box = preprocess.fetch_geo_location(self.city_name)

        self.raw_data_path = Path(raw_data_path)
        self.processed_file = Path(processed_file)
        self.raw_columns = [ 
            "lat",
            "lon",
            "zero",
            "alt",
            "n_days",
            "date",
            "time",
        ]
        self.label_column = "label"
        self.trajectory_column = "tid"
        self.datetime_column = "datetime"
        self.lat_column = "lat"
        self.lon_column = "lon"
        self.preprocess_col = [self.label_column, 'date', 'time', self.lat_column, self.lon_column]
        self.columns = [self.label_column, self.trajectory_column, self.lat_column, self.lon_column]            

    def fetchGeoDF(self):
        """
        Purpose:  Check for existing csv

        Returns df['datetime', 'label', 'lat', 'lon']
        """
        df = pd.read_csv(self.processed_file, parse_dates=[['date', 'time']])
        df = df.rename(columns={'date_time': 'datetime'}) 
        return df

    def preprocess(self):
        """
        ----------
        First stage of pre-processing GeoLife

        Takes raw data {GeoLife/Data/UID/Trajectory/*.plt} -> one csv 
        
        Preprocess the raw data into a single CSV file of trajectory data.

        Filters out by cell_size grid & boundry, see freq_matrix.py

        Long-running (takes a few minutes).

        """
        if os.path.exists(self.processed_file): return self.fetchGeoDF()

        # Single call for header & boundries
        oneTimeHeader = True
        cell_size = config.CELL_SIZE_METERS * config.MILES_PER_METER  # meters * conversion to miles
        bounds, _, _ = freq_matrix.set_map(self.bounding_box, cell_size)

        # Get all User Directories within GeoLife/Data directory
        for user_dir in self.raw_data_path.glob('*'):
            frames = []
            uid = str(user_dir)[-3:]
            # Read in raw gps
            LOG.info("On User: %s", uid)
            # Iterate past /Trajectory/ directory 
            dir = user_dir / Path("Trajectory") 
            # Set current path to this dir
            os.chdir(dir)
            # Create a list of relative paths
            all_plt_files = list(glob.glob("*.plt"))
            # Parse through each file
            for plt in all_plt_files: # working one plt file at a time, then append to csv
                # Read in raw gps
                LOG.info("\t Reading file: %s", plt)
                # Open each .plt file 
                with open(plt, 'r') as file: 
                    
                    plt_f = pd.read_csv(plt, skiprows=6, names=self.raw_columns)

                    # Attach UID of cur user
                    plt_f['label'] = uid

                    # # Correcting Format 
                    plt_f = plt_f[['label', 'date', 'time' ,'lat', 'lon']]

                    # Filter data before writing to file
                    plt_f = freq_matrix.filter_bounds(plt_f, bounds, "lat", "lon")

                    # Only save header once, because appending data to INPUT_FILE
                    if(oneTimeHeader) : 
                        plt_f.to_csv(self.processed_file, mode='a', index=False)
                        oneTimeHeader = False
                    
                    else : 
                        plt_f.to_csv(self.processed_file, mode='a', index=False, header=False)

        LOG.info("Preprocessed data written to: %s", self.processed_file)
        
        # Fetch & Verify Dataframe
        return self.fetchGeoDF()