""" Validation module for ND-GAIN data. """

from config.indicators import indicator_map

VALID_INDICATORS = set(indicator_map.keys())

REQUIRED_COLUMNS = [
    "Country",
    "Year",
    "Indicator",
    "Value",
]

def validate(df):

    check_required_columns(df)
    check_missing_values(df)
    check_duplicate_rows(df)
    check_indicators(df)


def check_required_columns(df):

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

def check_missing_values(df):

    critical = ["Country", "Year", "Indicator"]
    missing = df[critical].isna().sum()

    if missing.any():

        raise ValueError(
            f"Missing values detected:\n{missing}"
        )


def check_duplicate_rows(df):

    duplicates = df.duplicated(
        subset=["Country", "Year", "Indicator"]
    )

    if duplicates.any():

        raise ValueError(
            f"{duplicates.sum()} duplicate observations found."
        )


def check_indicators(df):

    invalid = set(df["Indicator"]) - VALID_INDICATORS

    if invalid:

        raise ValueError(
            f"Unknown indicators: {sorted(invalid)}"
        )
