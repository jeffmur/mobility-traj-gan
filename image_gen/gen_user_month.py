#!/home/jeffmur/anaconda3/envs/geoLife/bin/python

"""
Note: Change Enviornment ^^^, Must be executable (chmod 777 for dispatch.py)
Otherwise, see config.py to set correct paths

Purpose: Generate User heatmap images for each month
"""
import sys
import os
from pathlib import Path
sys.path.insert(1, '../')

import lib.preprocess as pre 
import lib.frequency as fre
import lib.config as c

gpsHeader = c.datasetHeaders

meters_size = 300 # sq meters
CELL_SIZE = meters_size * 0.00062137 #sq miles

args = sys.argv

if(len(args) <= 1):
    print(f"Usage: py3 gen_user_freq.py {c.DataInputDirectory}/USER_ID ")
    print("Usage: See dispatch.py")
    exit()

outDir = c.DataOutputDirectory+'gps_{CELL_SIZE}_month/'
p = Path(outDir)

print(f"Going to generate {len(args)-1} of them...")

if(not (os.path.isdir(p))):
    p.mkdir()


# HTTP 
boundingBox = pre.fetchGeoLocation('Lausanne, District de Lausanne, Vaud, Switzerland')

for user in args[1:]:
    uid = user.split('/')
    val = uid[len(uid)-1]
    print(f"On User: {val} ..... ", end="")
    fre.imagePerMonth(
        boundingBox= boundingBox,
        userDir = user,
        outDir = Path(os.path.join(outDir, val)),
        cell_size = CELL_SIZE,
    )
