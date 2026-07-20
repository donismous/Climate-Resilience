""" Functions for preparing the dataset for modeling."""


import pandas as pd

DROP_COUNTRIES = pd.read_csv("config/drop_countries.csv")["Name"]

def prepare_for_model(df: pd.DataFrame) -> pd.DataFrame:

    df = pivot_indicators(df)
    df = drop_countries(df)
    df = impute_missing(df)
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


def drop_countries(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~df["Country"].isin(DROP_COUNTRIES)].copy()


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:

    indicator_cols = df.columns.difference(["Country", "Year"])

    df[indicator_cols] = (
        df.groupby("Country")[indicator_cols]
          .transform(
              lambda x: (
                  x.interpolate(method="linear")
                   .ffill()
                   .bfill()
              )
          )
    )

    return df


def sort_time_series(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["Country", "Year"])
          .reset_index(drop=True)
    )
