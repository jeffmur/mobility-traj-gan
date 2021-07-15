## Temporary Path Storage for simple plug-in-play development
import os
import dotenv

dotenv.load_dotenv()

DATA_INPUT_DIR = str(os.getenv("DATA_INPUT_DIR"))
"""
Note: Used with os.walk which assumes it is a directory (hence no trailing backslash)
Important: This is the RAW dataset (not santized or preprocessed)
"""

DATA_INPUT_FILE = str(os.getenv("DATA_INPUT_FILE"))
"""
Input data filename, if a single CSV file (not a dir)
"""

DATA_OUTPUT_DIR = str(os.getenv("DATA_OUTPUT_DIR"))
"""
Used for exporting data (as images or csvs)
"""

GIT_PATH = str(os.getenv("GIT_PATH"))
"""
Project Directory
"""

DATASET = str(os.getenv("DATASET"))
"""
Name of the dataset we are processing. Could be "GeoLife", "MDC", "Privamov", NYC, etc.
"""

DATA_HEADERS = {
    "MDC": ["Index", "UID", "Date", "Time", "Latitude", "Longitude"],
    "GeoLife": ["Latitude", "Longitude", "Zero", "Altitude", "Num of Days", "Date", "Time"],
    "Privamov": ['ID', 'Date', 'Time', 'Longitude', 'Latitude']
    }.get(DATASET)
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

FM_MATRIX = os.getenv("FM_MATRIX")
"""
Post processing pah in .env for output of FM_all_users.py
( .csv file with all users and locations inside of Frequency Matrix )
"""

GPS_BB = os.getenv("GPS_BB")
"""
Path to the file with raw GPS coordinates but filtered to bounding box region
"""

CITY = os.getenv("CITY")
"""
Name of the city to retrieve the bounding box for.
"""



## MY BUCKET LIST 

# 1. Get a job , buy a house, other adult things
# 2. Travel - Italy, Switzerland, Germany 
# 3. Great something new - invention 
# 4. Learn how to fly a plane 
# 5. Try something new