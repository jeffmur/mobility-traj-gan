## Temporary Path Storage for simple plug-in-play development
import os
import dotenv

dotenv.load_dotenv()
DATA_INPUT_DIR = os.getenv("DATA_INPUT_DIR")
"""
Note: Used with os.walk which assumes it is a directory (hence no trailing backslash)
"""

DATA_OUTPUT_DIR = os.getenv("DATA_OUTPUT_DIR")
"""
Used for exporting data (as images or csvs)
"""

GIT_PATH = os.getenv("GIT_PATH")
"""
Project Directory
"""

CONDA_ENV = os.getenv("CONDA_ENV")

DATASET = os.getenv("DATASET")
DATA_HEADERS = {
    "MDC": ["Index", "UID", "Date", "Time", "Latitude", "Longitude"]
}.get(DATASET)
"""
Specific to dataset in question
"""
