## Temporary Path Storage for simple plug-in-play development
import os
import dotenv

dotenv.load_dotenv()
DATA_INPUT_DIR = os.getenv("DATA_INPUT_DIR")
"""
Note: Used with os.walk which assumes it is a directory (hence no trailing backslash)
Important: This is the RAW dataset (not santized or preprocessed)
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
DATA_HEADERS = {
    "MDC": ["Index", "UID", "Date", "Time", "Latitude", "Longitude"]
}.get(DATASET)
"""
Path to parsed dataset / (mdc || geoLife || privamov || etc.) /user_by_month/ included
"""
