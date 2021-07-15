"""datasets/geolife_beijing.py

The GeoLife dataset is filtered by the city limits of Beijing
City Limits = ['39.4416113', '41.0595584', '115.4172086', '117.5079852']

!Important Note: 
 - Before using this dataset, 
### TODO

        # Assume running split_by_month.py has been run??
        # Add implementation error check ^^
"""
import os
from logging import Logger
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

    def preprocess(self):
        self._sanitize()
        return self._filter()
        

    def _sanitize(self):
        """
        Step 1 
        ----------
        First stage of pre-processing GeoLife

        Takes raw data {GeoLife/Data/UID/Trajectory/*.plt} -> one csv 
        
        Preprocess the raw data into a single CSV file of trajectory data.

        Long-running (takes a few minutes).

        """
        oneTimeHeader = True
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
                    # Skip first 6 lines of meta-data
                    [next(file) for _ in range(6)]
                    # concat records to list
                    lines = [line.rstrip() for line in file]
                    # Create DF and parse
                    df = pd.DataFrame(lines)
                    # Expand single column to all
                    df = df[0].str.split(',',expand=True)
                    # Set column names
                    df.columns = self.raw_columns

                    # Attach UID of cur user
                    df['label'] = uid

                    # # Final Format 
                    df = df[['label', 'date', 'time' ,'lat', 'lon']]
                    # # save as INPUT_FILE (one csv)
                    if(oneTimeHeader) : 
                        df.to_csv(self.processed_file, mode='a', index=False)
                        oneTimeHeader = False
                    else : df.to_csv(self.processed_file, mode='a', index=False, header=False)
                    # Note: to avoid many duplicates, it has been removed from the saved csv file

        LOG.info("Preprocessed data written to: %s", self.processed_file)

    def _filter(self):
        """
        Step 2
        ---------
        There must be a geoLife_beijing.csv file set as the self.processesed_file, as this will be used as an input. 

        Filter will condense the data by space and time dimensions & prepare it for analysis. 

        Parameters
        ----------
        :output: DataFrame 
            [ label, datetime, lat, lon ]
        """

        if not os.path.exists(self.processed_file):
            self.preprocess()
            return -1 

        # Import file 
        df = pd.read_csv(self.processed_file, parse_dates=[['date', 'time']]) # Assuming static placement of date time
        df = df.rename(columns={'date_time': 'datetime'}) 
        df = df[['label', 'datetime', 'lat', 'lon']]

        # # filter to bounds
        cell_size = config.CELL_SIZE_METERS * config.MILES_PER_METER  # meters * conversion to miles
        bounds, _, _ = freq_matrix.set_map(self.bounding_box, cell_size)
        df = freq_matrix.filter_bounds(df, bounds, "lat", "lon")

        # # time bounding??! TODO

        # Overwrite previous cached dataset
        df.to_csv(self.processed_file, mode='a', index=False)
        return df