import os
import subprocess
import sys
import glob
import time
import numpy as np
from src.lib import config as c

args = sys.argv

if len(args) != 2:
    print(f"Usage: py3 :file:")
    print("Usage: Allows for Multi-Threading Parsing & Image Saves")
    exit()

pyfile = args[1]

totalUsers = 0
totalFiles = 0
# Get static ref of all users sorted by userID
temp = glob.glob(c.DATA_INPUT_DIR + "/*")
all_users = sorted(
    temp, key=lambda i: int(os.path.splitext(os.path.basename(i))[0])
)

monthPerUser = []

for base, dirs, files in os.walk(c.DATA_INPUT_DIR):
    # print('Searching in : ', base)
    c = 0
    for Files in files:
        c += 1

    if base != c.DATA_INPUT_DIR:
        baseName = int(base.split("/")[6])
        monthPerUser.append([baseName, c])

# Sorted by UserID
monthPerUser.sort(key=lambda x: x[0])

for i in range(len(monthPerUser)):
    monthPerUser[i] = [i, monthPerUser[i][0], monthPerUser[i][1]]

values = range(1, np.amax(monthPerUser))

# Group batches
grouped = [[y[0] for y in monthPerUser if y[2] == x] for x in values]

# In Ascending Size Order
ascendingSize = list(filter(lambda x: x, grouped))
length = len(ascendingSize)
print(f"Total Number of Priority Lists: {length}")
print(f"Total CPU Cores: {os.cpu_count()}")

# TODO: Watch for overflow cases (not enough cores)
#       Add Log func for every child processes
#       -- Time it, file size,

pathAscendSize = []

for group in ascendingSize:
    batch = []
    for each in group:
        batch.append(all_users[each])

    pathAscendSize.append(batch)

for i in range(len(pathAscendSize)):
    ## Drop all commas and extra spaces
    batchFiles = str(pathAscendSize[i])[1:-1].replace(",", "").replace("'", "")
    print(len(pathAscendSize[i]))
    print(batchFiles)
    # Create Child Process on new session
    cmdChild = [
        f"gnome-terminal",
        "--",
        "bash",
        "-c",
        f"./{pyfile} {batchFiles}",
    ]
    subprocess.run(cmdChild)
    time.sleep(2)
