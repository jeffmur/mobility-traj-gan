import pandas as pd 
import numpy as np 
from src.lib import freqMatrix as FM 
from src.lib import preprocess as pre 
from src.lib import config as config

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
boundingBox = pre.fetchGeoLocation('Lausanne, District de Lausanne, Vaud, Switzerland')
# ['46.5043006', '46.6025773', '6.5838681', '6.7208137']

# Data setup, modular cell_size for spatial resolution
cell_size = 300 * 0.00062137119 # meters * conversion to miles
bounds, step, pix = FM.setMap(boundingBox, cell_size)
df = pd.read_csv(config.DATASET)

# Useful analytics + exportList
maxVal, freq_heatmap, exportList = FM.create2DFreq(df, bounds, step, pix)
# print(freq_heatmap.head())
# print(maxVal)

# Export List to CSV in OUTPUT DIR
toExport = pd.DataFrame(exportList, columns=['UID', 'Date', 'Time', 'Column', 'Row'])
toExport.to_csv(f'{config.DATA_OUTPUT_DIR}+{cell_size}m_all_users')