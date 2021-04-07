import pandas as pd
import matplotlib.pyplot as plt
import math
import src.lib.preprocess as pre
import numpy as np


def set_map(bounding_box, cell_size):
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
    s_lat = float(bounding_box[0])
    n_lat = float(bounding_box[1])
    w_lon = float(bounding_box[2])
    e_lon = float(bounding_box[3])

    # all four corners
    SE = [s_lat, e_lon]
    SW = [s_lat, w_lon]
    NE = [n_lat, e_lon]
    NW = [n_lat, w_lon]

    bounds = {"SE": SE, "SW": SW, "NE": NE, "NW": NW}

    ## Calculate Haversine Distance of bounds
    # SW -> NW
    # Width
    d_west = pre.haversine_distance(SW[0], SW[1], NW[0], NW[1])

    # NW -> NE
    # Length
    d_north = pre.haversine_distance(NW[0], NW[1], NE[0], NE[1])

    # Round val to the nearest 1
    length = math.ceil(d_north)
    width = math.ceil(d_west)

    # Image Dimensions
    l_pix = int(math.ceil(length / cell_size))
    w_pix = int(math.ceil(width / cell_size))

    # Step Size for Lat/Lon comparison
    # Max distance / num of pixels
    step_length = (n_lat - s_lat) / l_pix  #  Step Lenth
    step_width = (e_lon - w_lon) / w_pix  #  Step Width

    # Steps in degrees
    step = {"width": step_width, "length": step_length}

    # Calculated Width and Length of image
    pix = {"length": l_pix, "width": w_pix}

    return bounds, step, pix


def create_2d_freq(df, bounds, step, pix):
    """
    Frequency Matrix
    For every data point (lon, lat) within cell_size increment by 1

    @df: Dataframe['Longitude', 'Latitdue']
    @bounds, step, pix: result of SetMap

    :returns:
    max_val: Highest tally value within matrix
    freq_heat: Tallied results where columns are pix['width'] and rows are pix['length']
    export_list: The input data filtered and aggregated to cells
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

    export_list = df.iloc[:, [0, 1, 2, 5, 6]]
    export_list.columns = ["UID", "Date", "Time", "Row", "Column"]

    return max_val, freq_heat, export_list


def take_log(maxVal, freq_heat):
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
