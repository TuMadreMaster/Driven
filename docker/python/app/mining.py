#/usr/bin/python3

from scipy.stats import median_absolute_deviation
from scipy.ndimage import binary_dilation, label

from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# LOAD DEFINITIONS
# ================================================================= #

CONFIG_FILE = Path(__file__).parent.resolve() / "model_variables.yml"

with open(CONFIG_FILE, "r") as file:
    defs = yaml.load(file, Loader=yaml.FullLoader)

time_col = defs["time_col"]
delta_col = defs["delta_col"]
prod_col = defs["prod_col"]
rate_col = defs["rate_col"]
chunk_col= defs["chunk_col"]
counter_col = defs["counter_col"]

categorical_cols = defs["categorical_cols"]
numeric_cols = defs["numeric_cols"]
antropic_cols = defs["antropic_cols"]

# ================================================================= #

def generate_anomaly_mask(
                        df: pd.DataFrame,
                        lda: float,
                        mode="median",
                        ) -> list[pd.Series, float]:

    """
    Provides a mask to filter the dataset, using either a 
    mean-based or a median-based threshold.
    """
    
    assert(mode in ["mean", "median", "arbitrary"])

    if mode == "median":
        threshold = np.median(df[delta_col]) + lda*median_absolute_deviation(df[delta_col])
    elif mode == "mean":
        threshold = np.mean(df[delta_col]) + lda*np.std(df[delta_col])
    elif mode == "arbitrary":
        threshold = lda
    
    return df[delta_col] > threshold, threshold

def generate_product_change_mask(
                    df: pd.DataFrame, 
                    new_product_start: bool = True,
                    new_product_end: bool = False,
                    ) -> pd.Series:

    """ Generates a mask """

    # basic product change mask 
    mask = df[prod_col].ne(df[prod_col].shift(1))

    # count day change (24h) as a new batch
    if delta_col in df:
        mask[df[delta_col] >= 86400] = True

    if new_product_start:
        mask.iloc[0] = True
    else:
        mask.iloc[0] = False

    # useful when chunking by product
    if new_product_end:
        mask.iloc[0] = True

    return mask


def chunk_by_product(
                df: pd.DataFrame, 
                group_by_product: bool = True,
                join_same_product: bool = False,
                ) -> list[pd.DataFrame]:

    """ Splits the dataframe in chunks for each product. """

    # make a copy of the input dataframe

    df = df.copy()

    cog_mask = generate_product_change_mask(df)
    df[chunk_col] = cog_mask.cumsum() 
    df[chunk_col] = df[prod_col].str.cat(df[chunk_col].astype(str),sep="_")

    if group_by_product:

        if join_same_product:

            chunk_df = {prod: pd.DataFrame(columns=df.columns) for prod in df[prod_col].unique()}

            for chunk_id, data in df.groupby(chunk_col):
                
                data: pd.DataFrame

                product = data[prod_col].unique()
                assert(len(product) == 1)
                product = product[0]

                chunk: pd.DataFrame = data.copy()
                chunk_df[product] = pd.concat([chunk_df[product], chunk], ignore_index=True)
                chunk_df[product].reset_index(inplace=True, drop=True)

        else:

            chunk_df = {prod: {} for prod in df[prod_col].unique()}
            
            for chunk_id, data in df.groupby(chunk_col):

                data: pd.DataFrame

                product = data[prod_col].unique()
                assert(len(product) == 1)
                product = product[0]

                chunk: pd.DataFrame = data.copy()
                chunk.reset_index(inplace=True, drop=True)

                chunk_df[product][chunk_id] = chunk
    else:

        chunk_df = {}

        for chunk_id, data in df.groupby(chunk_col):

            data: pd.DataFrame

            chunk: pd.DataFrame = data.copy()
            chunk.reset_index(inplace=True, drop=True)

            chunk_df[chunk_id] = chunk

    return chunk_df

def generate_target_sets_by_events(
                    df: pd.DataFrame, 
                    anomaly_mask: pd.Series,
                    nsamples: int,
                    sample_shift: int = 0,
                    new_product_start: bool = True,
                    new_product_end: bool = True,
                    prod_change_margin: int = 5,
                    ) -> list[pd.DataFrame]:

    # copy dataframe and reset index
    df = df.copy()
    df.reset_index(inplace=True, drop=True)

    # copy anomaly mask and reset index
    anomaly_mask = np.array(anomaly_mask)

    # mask with the product changes
    prod_changes = generate_product_change_mask(df, 
        new_product_start=new_product_start, new_product_end=new_product_end)

    # dilate to cover times
    prod_changes = binary_dilation(prod_changes, iterations=prod_change_margin)

    events_mask = np.logical_and(~prod_changes, anomaly_mask)
    colission_mask = np.logical_or(prod_changes, anomaly_mask)

    # possible events
    possible_events: pd.DataFrame = df[events_mask]

    events = []
    trainings_sets = []
    ts_mask = np.zeros_like(events_mask, dtype=bool)
    # iterate over possible events
    for idx, event in possible_events.iterrows():

        # get event window
        window = np.logical_and(df.index <= idx-1-sample_shift, df.index >= idx-nsamples-sample_shift)

        # check if window overlaps with reference border or other event
        if np.any(np.logical_and(window, colission_mask)):
            continue

        # TODO investigate why this is needed
        if np.sum(window) < nsamples:
            continue

        # update training set mask
        ts_mask = np.logical_or(ts_mask, window)

        # save event 
        events.append(event)

        # slice the window
        event_ts = df[window].copy()
        event_ts.reset_index(inplace=True, drop=True)

        # add training set
        trainings_sets.append(event_ts)

    return events, trainings_sets, ts_mask


# NORMAL APPROACH
# (fill interval)
def generate_blank_sets_by_events(
                    df: pd.DataFrame, 
                    anomaly_mask: pd.Series,
                    nsamples: int,
                    new_product_start: bool = True,
                    new_product_end: bool = True,
                    prod_change_margin: int = 5
                    ) -> list[pd.DataFrame]:

    # copy dataframe and reset index
    df = df.copy()
    df.reset_index(inplace=True, drop=True)

    # copy anomaly mask and reset index
    anomaly_mask = np.array(anomaly_mask)

    # mask with the product changes
    prod_changes = generate_product_change_mask(df, 
            new_product_start=new_product_start, new_product_end=new_product_end)

    # dilate to cover times
    prod_changes = binary_dilation(prod_changes, iterations=prod_change_margin)

    # generate colission mask
    colission_mask = np.logical_or(prod_changes, anomaly_mask)

    # find gaps in colission_mask
    labeled, ncomponents = label(~colission_mask)

    # iterate over the gaps
    blank_sets = []
    ts_mask = np.zeros_like(anomaly_mask, dtype=bool)
    for i in range(1, ncomponents+1):

        # create a window
        window_mask = (labeled == i)
        
        # safety measure #1
        # window_mask = binary_erosion(window_mask, iterations=nsamples) 

        # safety measure #2
        if np.sum(window_mask) < 3*nsamples:
            continue

        # find possible interval range
        minp = np.argmax(window_mask)
        maxp = len(window_mask) - np.argmax(window_mask[::-1])

        # split the dataframe 
        for p in range(int((maxp - minp)/nsamples)):

            # sample range
            lowb = minp + p*nsamples
            highb = minp + p*nsamples + nsamples

            # update ts_mask
            ts_mask[lowb:highb] = True

            # generate ts
            blank_ts: pd.DataFrame = df.iloc[lowb:highb].copy()
            blank_ts.reset_index(inplace=True, drop=True)

            # add ts
            blank_sets.append(blank_ts)

    return blank_sets, ts_mask