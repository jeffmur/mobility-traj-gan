#!/home/jeffmur/anaconda3/envs/geoLife/bin/python

"""
Note: Change Enviornment ^^^, Must be executable (chmod 777 for dispatch.py)
Otherwise, see config.py to set correct paths

Purpose: Generate User heatmap images for each user
"""

import sys
import os
from pathlib import Path
import sys

## TODO: THERE HAS TO BE A BETTER WAY
sys.path.insert(1, '../')

import lib.config as c
import lib.preprocess as pre
import lib.frequency as fre
import lib.heatmap as hp

gpsHeader = c.datasetHeaders

meters_size = 300 # sq meters
CELL_SIZE = meters_size * 0.00062137 #sq miles

args = sys.argv

if(len(args) <= 1):
    print(f"Usage: py3 gen_user_freq.py {c.DataInputDirectory}/USER_ID ")
    print("Usage: See dispatch.py")
    exit()

outDir = c.DataOutputDirectory+'gps_{CELL_SIZE}_all/'
p = Path(outDir)

if(not (os.path.isdir(p))):
    p.mkdir()

boundingBox = pre.fetchGeoLocation('Lausanne, District de Lausanne, Vaud, Switzerland')

for user in args[1:]:
    val = hp.parse4User(user)
    print(f"On User: {val} ..... ", end="")

    # temp = dataDir+val+'/'
    # user_months = glob.glob(temp+'*')
    # print(f"parsing {len(user_months)} months")

    fre.imagePerUser(
        boundingBox=boundingBox,
        userDir=user,
        outDir=outDir,
        cell_size=CELL_SIZE,
    )