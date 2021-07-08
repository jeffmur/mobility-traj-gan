# Sanitizing the MDC dataset
# Please read through before executing
# Jeffrey Murray Jr
"""
* Input of Raw Set
* Removes extra spaces and Nulls
* Converts Unix to Date, Time respectively
* Saves to csv
"""

import src.lib.preprocess as pre

## Paths : TO CHANGE FOR LOCAL SETUP ## TODO
# raw gps.csv file in mdc
rawGPS = "PATH/FROM/gps-raw.csv"
cleanGPS = "PATH/TO/cleanGPS.csv"

# raw records.csv file in mdc
rawRecords = "PATH/FROM/records-raw.csv"
cleanRecords = "PATH/TO/cleanRecords.csv"

# records that only contain gps label (use grep) -- important
gpsRecords = "PATH/FROM/CLEANED/gps-ONLY-records.csv"

## ------------------------------

# Drop spaces and save as cleanGPS.csv
pre.removeSpaces(rawGPS, cleanGPS)

# Create Pandas Dataset
headers = [
    "UID",
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
gpsLong = pre.to_pandas(cleanGPS, headers, " ")

# Drop extra headers
newGPS = gpsLong[["UID", "Unix", "Latitude", "Longitude"]]

# Convert Unix to Date, Time respectively
gps = pre.unix_to_timestamp(newGPS)

# # Then save
gps.to_csv("cleanGPS.csv")

"""
With the data cleaned, we can now decode Database keys for User IDs
To do this, we must use records.csv (which also must be cleaned)
"""
pre.removeSpaces(rawRecords, cleanRecords)

# --- IMPORTANT ---
# --- $ cat records.csv | grep -w 'gps' > gpsrecords.csv
# --- Execute Before Continuing ---


# headers = ['DB', 'UID', 'tz', 'time', 'type']
# records = pre.toPandas(gpsRecords, headers, ' ')

# print(f" There are {records['UID'].nunique()} unique users")

# ## Create a dictionary for O(1) reading, O(N) for intialization
# dictOfRecords = {}
# for row in records.itertuples():
#     KEY = row.DB
#     VAL = row.UID

#     dictOfRecords[KEY] = VAL

# replaceUID = gps['UID'].to_numpy()

# # O(N) to replace each key (Database) w/ value (UID)
# for i in range(0, len(replaceUID)):
#     replaceUID[i] = dictOfRecords[replaceUID[i]]

# final_gps = gps[['UID', 'Date', 'Time' ,'Latitude', 'Longitude']]
# final_gps['UID'] = replaceUID

# final_gps.to_csv('gps-sanitized.csv')
