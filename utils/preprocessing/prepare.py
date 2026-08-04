""" Functions for preparing the dataset for modeling."""


import pandas as pd

DROP_COUNTRIES = pd.read_csv("config/drop_countries.csv")["ISO3"]

def prepare_for_model(df: pd.DataFrame, imputation: str = "global") -> pd.DataFrame:

    df = pivot_indicators(df)
    df = drop_countries(df)
    df = impute_missing(df)
    df = reverse_readiness_indicators(df)
    df = sort_time_series(df)

    return df


def reverse_readiness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reverse the scale of readiness indicators (Economic, Governance, Social) using (1 - value).

    In the raw ND-GAIN dataset, higher readiness values represent higher adaptive capacity (lower risk).
    Reversing the scale with (1 - value) converts them into lack-of-readiness metrics, so that across all
    indicators, higher values consistently represent higher risk contribution.
    """
    readiness_cols = [col for col in ["Economic", "Governance", "Social"] if col in df.columns]
    if readiness_cols:
        df[readiness_cols] = 1 - df[readiness_cols]
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
        df.groupby("Year")[indicator_cols]
          .transform(lambda x: x.fillna(x.mean()))
    )

    return df


def sort_time_series(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["Country", "Year"])
          .reset_index(drop=True)
    )

def add_country_names(df: pd.DataFrame) -> pd.DataFrame:
    country_names = pd.read_csv("config/iso3_to_country_name.csv", sep=";")
    country_map = country_names.set_index("ISO3")["Name"]
    df.insert(
    loc=df.columns.get_loc("Country") + 1,
    column="Name",
    value=df["Country"].map(country_map)
    )
    return df


def add_region_names(df: pd.DataFrame, country_col: str = "Country") -> pd.DataFrame:
    region_names = pd.read_csv("config/iso3_to_region_name.csv")
    region_map = region_names.set_index("ISO3")["sub_region"]
    df.insert(
        loc=df.columns.get_loc(country_col) + 1,
        column="Region",
        value=df[country_col].map(region_map),
    )
    return df
