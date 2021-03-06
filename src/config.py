## Temporary Path Storage for simple plug-in-play development
import os
import dotenv
import numpy as np

dotenv.load_dotenv()

"""
Dataset Specific Path Locations
"""

GEO_INPUT_DIR = os.getenv("GEO_INPUT_DIR")
"""
Note: Used with os.walk which assumes it is a directory (hence no trailing backslash)
Important: This is the RAW dataset (not santized or preprocessed)
ex: data/GeoLife/Data/
"""

GEO_INPUT_FILE = os.getenv("GEO_INPUT_FILE")
"""
Input data filename, if a single CSV file (not a dir)
ex: data/geoLife_beijing.csv
"""

MDC_INPUT_DIR = os.getenv("MDC_INPUT_DIR")
MDC_INPUT_FILE = os.getenv("MDC_INPUT_FILE")
"""
MDC Lausanne
"""

PRV_INPUT_DIR = os.getenv("PRV_INPUT_DIR")
PRV_INPUT_FILE = os.getenv("PRV_INPUT_FILE")
"""
Privamov Lyon
"""

FSN_INPUT_DIR = os.getenv("FSN_INPUT_DIR")
FSN_INPUT_FILE = os.getenv("FSN_INPUT_FILE")
"""
Foursquare NYC
"""

DATA_HEADERS = {
    "MDC": ["Index", "UID", "Date", "Time", "Latitude", "Longitude"],
    "GEO": ["Latitude", "Longitude", "Zero", "Altitude", "Num of Days", "Date", "Time"],
    "PRV": ["ID", "Date", "Time", "Longitude", "Latitude"],
    "FSN": ["user_id", "venue_id", "category_id", "category", "lat", "lon", "tz", "utc_time"],
}
"""
(mdc || geoLife || privamov || etc.)
RAW HEADERs
TODO: Setup .env interface for each dataset with applicable paths & modularity
"""

CELL_SIZE_METERS = 300
"""
Spatial resolution config: Dimension of a grid cell in meters
"""

MILES_PER_METER = 0.00062137119
"""
Distance unit conversion constant
"""

GPS_BB = os.getenv("GPS_BB")
"""
Path to the file with raw GPS coordinates but filtered to bounding box region
"""

SEED = 11
"""
Random number generator seed
"""
