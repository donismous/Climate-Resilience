"""This module provides a function to clean and preprocess the data by
standardizing column names, converting data types, removing duplicates,
and sorting the data."""

import pandas as pd

def clean_data(df):
    df = standardize_columns(df)
    df = convert_types(df)
    df = remove_duplicates(df)
    df = sort_data(df)
    return df


COLUMN_MAP = {
    "REF_AREA": "Country",
    "TIME_PERIOD": "Year",
    "INDICATOR": "Indicator",
    "OBS_VALUE": "Value",
}


def standardize_columns(df):
    return df.rename(columns=COLUMN_MAP)

def convert_types(df):

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    return df

def remove_duplicates(df):

    return df.drop_duplicates(
        subset=["Country", "Year", "Indicator"]
    )

def sort_data(df):

    return df.sort_values(
        ["Country", "Indicator", "Year"]
    ).reset_index(drop=True)
