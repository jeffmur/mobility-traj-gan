import pandas as pd
import matplotlib.pyplot as plt
import math
import src.lib.preprocess as pre
import numpy as np


def setMap(boundingBox, cell_size):
    """
    Purpose: Create HeatMap Outline

    @boundingBox: City or region limits to calculate grid dimensions in miles
    @cell_size: in miles

    :returns:
    bounds: Lon, Lat coordinates {"SE", "SW", "NE", "NW"}
    step: conversion between lat/lon with respect to cell_size (mi)
    pix: returns image dimensions (frequency matrix columns and rows)
    """

    ## Calculate bounds
    sLat = float(boundingBox[0])
    nLat = float(boundingBox[1])
    wLon = float(boundingBox[2])
    eLon = float(boundingBox[3])

    # all four corners
    SE = [sLat, eLon]
    SW = [sLat, wLon]
    NE = [nLat, eLon]
    NW = [nLat, wLon]

    bounds = {"SE": SE, "SW": SW, "NE": NE, "NW": NW}

    ## Calculate Haversine Distance of bounds
    # SW -> NW
    # Width
    dWest = pre.haversine_distance(SW[0], SW[1], NW[0], NW[1])

    # NW -> NE
    # Length
    dNorth = pre.haversine_distance(NW[0], NW[1], NE[0], NE[1])

    # Round val to the nearest 1
    length = math.ceil(dNorth)
    width = math.ceil(dWest)

    # Image Dimensions
    l_pix = int(math.ceil(length / cell_size))
    w_pix = int(math.ceil(width / cell_size))

    # Step Size for Lat/Lon comparison
    # Max distance / num of pixels
    step_length = (nLat - sLat) / l_pix  #  Step Lenth
    step_width = (eLon - wLon) / w_pix  #  Step Width

    # Steps in degrees
    step = {"width": step_width, "length": step_length}

    # Calculated Width and Length of image
    pix = {"length": l_pix, "width": w_pix}

    return bounds, step, pix


def create2DFreq(df, bounds, step, pix):
    """
    Frequency Matrix
    For every data point (lon, lat) within cell_size increment by 1

    @df: Dataframe['Longitude', 'Latitdue']
    @bounds, step, pix: result of SetMap

    :returns:
    maxVal: Highest tally value within matrix
    freq_heat: Tallied results where columns are pix['width'] and rows are pix['length']
    """

    n_lat = bounds["NE"][0]
    e_lon = bounds["NE"][1]

    columns = pix["width"]
    rows = pix["length"]

    step_w = step["width"]
    step_l = step["length"]

    # Difference between max Point (NE)
    # And Location (lonLat)
    # Within Frequency Matrix bounds
    df["r"] = np.round((n_lat - df.iloc[:, 3]) / step_l)
    df["c"] = np.round((e_lon - df.iloc[:, 4]) / step_w)
    df = df[(df.c <= columns) & (df.c >= 0) & (df.r <= rows) & (df.r >= 0)]

    freq_heat = pd.pivot_table(
        df[["UID", "r", "c"]],
        values="UID",
        index="r",
        columns="c",
        aggfunc="count",
        fill_value=0,
    )

    max_val = freq_heat.max().max()

    exportList = df.iloc[:, [0, 1, 2, 5, 6]]
    exportList.columns = ["UID", "Date", "Time", "Row", "Column"]

    return max_val, freq_heat, exportList


def takeLog(maxVal, freq_heat):
    """
    For each row, normalize each data point
    By their maximum values between 0 and 1

    Uses log_base(maxVal, FM[i,j])

    :returns: normalization approach
    """
    shape = freq_heat.shape
    log_freq = pd.DataFrame(0, index=range(shape[0]), columns=range(shape[1]))
    if maxVal <= 1:
        return log_freq

    for row in freq_heat.itertuples():
        # Need row index for assignment
        for c in range(1, len(row)):
            # Capture data point @ [row, column]
            data = row[c]
            # print(data, end="")
            # Expecting 0.5 -> inf (nan)
            d = pre.log_base(maxVal, data)

            # # -inf or < 0
            # if(d < 0): l = d * -1
            # # inf or > 0
            # else: l = d
            # print(f"Data: {data} ... logit: {d}")

            # row[0] = Index ; c = columns
            # Offest Columns by the included index
            log_freq.loc[row[0], c - 1] = d

    # print("Calculated Logit")

    return log_freq


# Ex. split_by_month_output/000/2008_10.csv
def parse4Date(path):
    l = len(path)
    return path[l - 11 : l - 4]


def parse4User(path):
    uid = path.split("/")
    return uid[len(uid) - 1]
