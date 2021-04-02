import pandas as pd
import numpy as np
from PIL import Image
import glob
import os
import sys

sys.path.insert(1, "../")
from src.lib import heatmap as hp


##################################################
# FM Prime
##################################################
def genFMprime(log_df):
    """"""
    dim = log_df.shape

    img = Image.new("RGB", (dim[0], dim[1]), color="red")
    pixels = img.load()

    for row in log_df.itertuples():
        # Need row index for assignment
        for c in range(1, len(row)):
            # Capture data point @ [row, column]
            data = row[c]

            freq = int(255 * data)

            pixels[row[0], c - 1] = (freq, freq, freq)

    return img


##############################################
# Frequency Matrix
##############################################
def getFreqInMonth(bb, inFile, cell_size):
    """
    Generates a Frequency Matrix(pixelX, pixelY)
    Number of visits per cell_size within bounding box

    :bb:        Bounding Box of city limits
                (min lat, max lat, min lon, max lon)

    :inFile:    File to parse
    :cell_size: in miles
    :pixelX:    Pixel Width Demension
    :pixelY:    Pixel Length Demension

    -> returns DataFrame
    """
    df = pd.read_csv(inFile)

    bounds, step, pix = hp.setMap(bb, cell_size)
    maxVal, freqDF = hp.create2DFreq(df, bounds, step, pix)
    print(f"Max Value: {maxVal}")
    log_df = hp.takeLog(maxVal, freqDF)

    return pix, log_df


def prodImage(bb, inFile, cell_size):
    """
    Generates an image representation of the Frequency Matrix

    :df_header: Headers ("names") for column labels
    :inFile:    File to parse
    :cell_size: in miles
    :pixelX:    Pixel Width Demension
    :pixelY:    Pixel Length Demension

    -> returns Image (pixelX, pixelY)
    Representation (in black/white) of log dataframe
    """
    _, df = getFreqInMonth(bb, inFile, cell_size)

    return genFMprime(df)


def imagePerMonth(boundingBox, userDir, outDir, cell_size):
    """
    Generates an image representation of the Frequency Matrix

    :cityCountry:   Ex. Lyon, France.
                    Location name for OpenStreetMap API

    :userDir:       UserID to parse each month
    :outDir:        Location to save files
    :cell_size:     in square miles
    :pixelX:        Pixel Width Demension
    :pixelY:        Pixel Length Demension

    -> returns Image (pixelX, pixelY)
    Representation (in black/white) of log dataframe
    """
    print(userDir)
    all_months = glob.glob(userDir + "/*")
    # Create User Out Directory
    if not (os.path.isdir(outDir)):
        outDir.mkdir()

    # dir = list all months in userDir
    #       /NNN/monthN.csv
    for month in all_months:
        img = prodImage(boundingBox, month, cell_size)
        date = hp.parse4Date(month)

        if np.mean(img) != 0:
            img.save(f"{outDir}/{date}.png")
            print(f"Saving {date}.png")
        else:
            print(f"Warning! Image {date}.png has no data")


###### Image Per User #######


def monthFM(monthFile, boundingBox, cell_size):
    #  Parse
    df = pd.read_csv(monthFile)
    # Set Map Grid
    bounds, step, pix = hp.setMap(boundingBox, cell_size)
    # Return
    return hp.create2DFreq(df, bounds, step, pix)


def imagePerUser(boundingBox, userDir, outDir, cell_size):
    """
    Generates an image representation of the Frequency Matrix

    :cityCountry:   Ex. Lyon, France.
                    Location name for OpenStreetMap API

    :userDir:       UserID to parse each month
    :outDir:        Location to save files
    :cell_size:     in square miles
    :pixelX:        Pixel Width Demension
    :pixelY:        Pixel Length Demension

    -> returns Image (pixelX, pixelY)
    Representation (in black/white) of log dataframe
    """
    all_months = glob.glob(userDir + "/*")

    print(boundingBox)

    # Create User Out Directory
    if not (os.path.isdir(outDir)):
        outDir.mkdir()

    all_dfs = pd.DataFrame()
    onePass = True
    maxVal = 0
    for month in all_months:
        val, tmp = monthFM(month, boundingBox, cell_size)
        if onePass:
            onePass = False
            all_dfs = tmp
            maxVal = val
        else:
            # Append Frequency Point to dataframe
            for i in range(0, len(tmp.columns)):
                all_dfs[i] += tmp[i]

            if val > maxVal:
                maxVal = val

    # print(all_dfs)

    log_df = hp.takeLog(maxVal, all_dfs)
    userName = hp.parse4User(userDir)

    # Save to OUTPUT / USER
    prime = genFMprime(log_df)

    if np.mean(prime) != 0:
        prime.save(f"{outDir}/{userName}.png")
        print(f"Saving {userName}.png")
    else:
        print(f"Warning! Image {userName}.png has no data")