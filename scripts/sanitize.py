"""
Sanitizing the MDC dataset
Jeffrey Murray Jr
Updated by Alex Kyllo 2021-07-08

* Input of Raw Set
* Removes extra spaces and Nulls
* Converts Unix to DateTime respectively
* Saves to csv
"""
import sys
from pathlib import Path

sys.path.append(".")
import pandas as pd
from src import config

## Paths : TO CHANGE FOR LOCAL SETUP ## TODO
# raw gps.csv file in mdc
input_path = Path(config.DATA_INPUT_DIR)
print(f"Reading MDC data from {input_path}")
output_path = Path(config.DATA_OUTPUT_DIR)
raw_gps = input_path / "gps.csv"
# raw records.csv file in mdc
raw_records = input_path / "records.csv"
output_file = output_path / config.DATA_INPUT_FILE

# column names
headers = [
    "RID",
    "Unix",
    "Longitude",
    "Latitude",
    "Altitude",
    "Speed",
    "H. Accuracy",
    "H. Dop",
    "V. Accuracy",
    "V. Dop",
    "Speed Accuracy",
    "Time_Since_Boot",
]

# Read in raw gps
df = pd.read_table(raw_gps, names=headers)

# Drop extra headers
df = df[["RID", "Unix", "Latitude", "Longitude"]]

# Convert Unix to datetime
df.loc[:, "DateTime"] = pd.to_datetime(df.Unix, unit="s")
df = df.drop(["Unix"], axis=1)

rec_headers = ["RID", "UID", "tz", "time", "type"]
iter_rec = pd.read_table(raw_records, names=rec_headers, iterator=True, chunksize=10000)
df_rec = pd.concat([chunk[chunk["type"] == "gps"] for chunk in iter_rec])

# Join gps to records on RID
df = df.set_index("RID")
df_rec = df_rec.set_index("RID")

df = df.join(df_rec, how="inner")
df = df[["UID", "DateTime", "Latitude", "Longitude"]]

df.to_csv(output_file, index=False)
print(f"Cleaned MDC data written to {output_file}")
