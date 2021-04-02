import pandas as pd
import glob
from pathlib import Path
import os
from src.lib import config

# TODO: add header & combine split files

input_dir = config.DATA_INPUT_DIR
output_dir = config.DATA_OUTPUT_DIR
gps_header = config.DATA_HEADERS

# Make dir if does NOT exist
if not (os.path.isdir(output_dir)):
    output_dir.mkdir()

all_files = glob.glob(input_dir + "*")

for path in all_files:
    print(path)
    l = len(path)
    fileName = path[l - 11 : l - 4]
    monthDF = pd.read_csv(path, names=gps_header)
    # monthDF.head()

    # Number of Users in Month File
    numOfUsers = len(pd.unique(monthDF["UID"]))

    # Group by ID
    grouped = monthDF.groupby(monthDF["UID"])

    # Write each user group to monthFile in sub dir
    for userName, group in grouped:
        user_dir = output_dir / Path(str(userName))

        group = group.reset_index().drop("index", axis=1)

        # Make dir if does NOT exist
        if not (os.path.isdir(user_dir)):
            user_dir.mkdir()
        # print(name)
        group.to_csv(f"{user_dir}/{fileName}.csv")
