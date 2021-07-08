import sys

sys.path.append(".")
from pathlib import Path
import pandas as pd
from src.lib import config, freq_matrix, preprocess

"""
Purpose: Generate Frequency Matrix for all users within a dataset
            - Requires UID, Time, Date, Longitude, Latitude headers
            - Drops all outlying data from bounding region

Input: (.env) DATASET (gps-santized.csv)
Output: cell_size (meters or miles)_all_users.csv
            - Does not care about Temporal Resolution
            - Maintains as much data as possible w/in region & tiles ( cell_size sq. miles )
"""
if __name__ == "__main__":
    # HTTP request or set to static bounds
    bounding_box = preprocess.fetch_geo_location(config.CITY)
    # "Lausanne, District de Lausanne, Vaud, Switzerland" =
    # ['46.5043006', '46.6025773', '6.5838681', '6.7208137']

    # Data setup, modular cell_size for spatial resolution
    cell_size = config.CELL_SIZE_METERS * config.MILES_PER_METER  # meters * conversion to miles
    bounds, step, pix = freq_matrix.set_map(bounding_box, cell_size)
    df = pd.read_csv(Path(config.DATA_OUTPUT_DIR) / config.DATA_INPUT_FILE)

    # output file with raw GPS coordinates
    df_bb = freq_matrix.filter_bounds(df, bounds, "Latitude", "Longitude")
    df_bb[["UID", "Date", "Time", "Latitude", "Longitude"]].to_csv(
        Path(config.DATA_OUTPUT_DIR) / "gps_bb_all_users.csv", index=False
    )

    # Useful analytics + exportList
    _, _, exportList = freq_matrix.create_2d_freq(df, bounds, step, pix)

    # Export List to CSV in OUTPUT DIR
    exportList[["UID", "Date", "Time", "Column", "Row"]].to_csv(
        Path(config.DATA_OUTPUT_DIR) / f"{config.CELL_SIZE_METERS}m_all_users.csv"
    )
