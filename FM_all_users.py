from pathlib import Path
import pandas as pd
from src.lib import config, freqMatrix as FM, preprocess as pre

"""
Purpose: Generate Frequency Matrix for all users within a dataset
            - Requires UID, Time, Date, Longitude, Latitude headers
            - Drops all outlying data from bounding region

Input: (.env) DATASET (gps-santized.csv)
Output: cell_size (meters or miles)_all_users.csv
            - Does not care about Temporal Resolution
            - Maintains as much data as possible w/in region & tiles ( cell_size sq. miles )
"""

# HTTP request or set to static bounds
boundingBox = pre.fetchGeoLocation("Lausanne, District de Lausanne, Vaud, Switzerland")
# ['46.5043006', '46.6025773', '6.5838681', '6.7208137']

# Data setup, modular cell_size for spatial resolution
cell_size = config.CELL_SIZE_METERS * config.MILES_PER_METER  # meters * conversion to miles
bounds, step, pix = FM.setMap(boundingBox, cell_size)
df = pd.read_csv(Path(config.DATA_INPUT_DIR) / config.DATA_INPUT_FILE, index_col=0)

# Useful analytics + exportList
maxVal, freq_heatmap, exportList = FM.create2DFreq(df, bounds, step, pix)

# Export List to CSV in OUTPUT DIR
exportList[["UID", "Date", "Time", "Column", "Row"]].to_csv(
    Path(config.DATA_OUTPUT_DIR) / f"{config.CELL_SIZE_METERS}m_all_users.csv"
)
