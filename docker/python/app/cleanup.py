#/usr/bin/python3

"""


"""

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

def print_summary(df: pd.DataFrame) -> None:

    """ Prints a summary of the data columns. """

    print("====================================")
    for col, dtype in zip(df.columns, df.dtypes):
        print(col, dtype, df[col].unique())
    print("Total columns:", len(df.columns))
    print("Total rows:", len(df))
    print("====================================")

def convert_booleans_to_integers(df: pd.DataFrame):

    """" Converts boolean columns to integers. """
    
    df = df.copy()  # make a copy of the input dataframe
    bool_cols = df.select_dtypes(include='bool').columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    return df

def add_time_deltas(df: pd.DataFrame) -> pd.DataFrame:

    """ Adds a colum with a time difference between events in seconds. """
    
    df = df.copy()  # make a copy of the input dataframe
    df.insert(1, delta_col, df[time_col].diff(1).dt.total_seconds())
    return df

def add_kit_rate(df: pd.DataFrame) -> pd.DataFrame:

    """ Adds a column with the kit production rate. """
    
    df = df.copy()  # make a copy of the input dataframe
    rate = df[counter_col].diff(1)
    rate[rate < 0] = 0
    df.insert(2, rate_col, rate)
    return df

def remove_first_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:

    """ Removes first n rows. """
    
    df = df.copy()  # make a copy of the input dataframe
    df.drop(df.head(n).index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df

def assing_correct_dtypes(df: pd.DataFrame,) -> pd.DataFrame:

    """ Optimiza los dtypes, booleanizes categories. """
    
    df = df.copy() # make a copy of the input dataframe
    df = df.infer_objects()  # automagically infer dtypes

    # correct floats to integers
    float_cols = df.select_dtypes(include="float64").columns
    float_cols = [c for c in float_cols if c != delta_col]
    
    for col in float_cols:
        df[col] = df[col].astype("int")

    # denote categorical columns
    for col in categorical_cols + [prod_col]:
        df[col] = df[col].astype("category")

    # make a copy because "fragmentation"
    df = df.copy()

    # move product name to the beggining
    df.insert(2, prod_col, df.pop(prod_col))

    return df

def booleanize_categories(
                df: pd.DataFrame, 
                exclude_product_name: bool = True) -> pd.DataFrame:
    
    """ Adds a boolean column for each unique value in the categorical columns and removes them. """

    df = df.copy()  # make a copy of the input dataframe
    cols = categorical_cols.copy()

    if not exclude_product_name:
        cols.append(prod_col)

    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=bool)
        df.drop(col, axis=1, inplace=True)
        df = pd.concat([df, dummies], axis=1)

    return df

def drop_categories(
                df: pd.DataFrame, 
                exclude_product_name: bool = True) -> pd.DataFrame:
   
    """ Drops categorical columns. """

    df = df.copy()  # make a copy of the input dataframe

    cols = categorical_cols.copy()

    if not exclude_product_name:
        cols.append(prod_col)

    df.drop(cols, axis=1, inplace=True)

    return df

def drop_antropic_cols(df: pd.DataFrame) -> pd.DataFrame:

    """ Drops antropic columns. """

    df = df.copy()
    cols = [col for col in df.columns if col not in antropic_cols]
    df = df[cols]
    df = df.copy()
    return df

def minmax_scaler(
        series: np.ndarray,
        frange: list[int] = [0,1], 
        orange: list[int] = None
        ) -> np.ndarray:

    """ MinMax Scaler. """

    if orange is None:
        maxs = np.max(series)
        mins = np.min(series)
        if maxs != mins:
            std_series = (series - np.min(series))/(np.max(series) - np.min(series))
        else:
            std_series = series.copy()
            std_series = 0
    else:
        std_series = (series - orange[0])/(orange[1] - orange[0])
    scaled_series = std_series*(frange[1] - frange[0]) + frange[0]
    scaled_series = np.clip(scaled_series, frange[0], frange[1])

    return scaled_series

def scale_numeric_nonbinary(df: pd.DataFrame) -> pd.DataFrame:

    """ Scale numeric non-binary columns between 0 and 1. """

    df = df.copy()  # make a copy of the input dataframe
    for col in numeric_cols:
        df[col] = minmax_scaler(np.array(df[col]))

    return df

def drop_numeric_nonbinary(df: pd.DataFrame) -> pd.DataFrame:

    """ Drop numeric non-binary columns. """

    df = df.copy()  # make a copy of the input dataframe
    df.drop(numeric_cols, axis=1, inplace=True)

    return df

def split_into_samples(df: pd.DataFrame, sample_size: int):

    """ Split dataframe into samples of size 'sample_size' """

    if sample_size > len(df):
        raise MemoryError

    times = []
    samples = []
    for i in range(len(df) + 1 - sample_size):  
        df_slice: pd.DataFrame = df.iloc[i:i+sample_size]
        times.append(pd.to_datetime(df_slice[time_col]).max())
        samples.append(df_slice.copy())

    return samples, times

def remove_first_product_chunks(
                df: pd.DataFrame, 
                chunks_to_remove: int, 
                product_col: str = "nombre_receta") -> list[pd.DataFrame]:

    """ Splits the dataframe in chunks for each product. """

    # make a copy of the input dataframe
    df = df.copy()

    group = df[product_col].ne(df[product_col].shift(1))
    group.iloc[0] = False

    df["group"] = group.cumsum() 

    remaining_chunks = []
    for i, (group, data) in enumerate(df.groupby("group")):

        if i < chunks_to_remove:
            continue
        else:
            product = data[product_col].unique()
            assert(len(product) == 1)
            remaining_chunks.append(data)

    df = pd.concat(remaining_chunks)
    df.drop("group", axis=1, inplace=True)

    return df

def convert_dataset_to_array(df: pd.DataFrame, return_cols: bool = False) -> np.ndarray:

    """ Converts the dataframe to an array suitable for the TNN. """

    df = drop_categories(df, exclude_product_name=False)
    # df = scale_numeric_nonbinary(df)
    df = drop_numeric_nonbinary(df)
    
    cols = [c for c in df.columns if "index" not in c]
    cols = [c for c in cols if delta_col not in c] 
    cols = [c for c in cols if time_col not in c] 

    if return_cols:
        return cols

    return np.array(df[cols].values, dtype=np.float16)