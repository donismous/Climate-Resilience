""" Functions for preparing the dataset for modeling."""


import pandas as pd
from config.indicators import indicator_map

def prepare_for_model(df):

    df = pivot_indicators(df)
    df = sort_time_series(df)

    return df


def pivot_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pivot_table(
            index=["Country", "Year"],
            columns="Indicator",
            values="Value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

def sort_time_series(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["Country", "Year"])
          .reset_index(drop=True)
    )
