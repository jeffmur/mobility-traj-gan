## Temporary Path Storage for simple plug-in-play development
import os
import dotenv

dotenv.load_dotenv()

DATA_INPUT_DIR = os.getenv("DATA_INPUT_DIR")
"""
Note: Used with os.walk which assumes it is a directory (hence no trailing backslash)
Important: This is the RAW dataset (not santized or preprocessed)
"""

DATA_INPUT_FILE = os.getenv("DATA_INPUT_FILE")
"""
Input data filename, if a single CSV file (not a dir)
"""

DATA_OUTPUT_DIR = os.getenv("DATA_OUTPUT_DIR")
"""
Used for exporting data (as images or csvs)
"""

GIT_PATH = os.getenv("GIT_PATH")
"""
Project Directory
"""

DATASET = os.getenv("DATASET")
"""
Name of the dataset we are processing. Could be "GeoLife", "MDC", etc.
"""

DATA_HEADERS = {"MDC": ["Index", "UID", "Date", "Time", "Latitude", "Longitude"]}.get(DATASET)
"""
Path to parsed dataset / (mdc || geoLife || privamov || etc.) /user_by_month/ included
"""

CELL_SIZE_METERS = 300
"""
Spatial resolution config: Dimension of a grid cell in meters
"""

MILES_PER_METER = 0.00062137119
"""
Distance unit conversion constant
"""