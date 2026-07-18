""" Filter ND-GAIN indicators and replace API codes with user-friendly names. """

import pandas as pd

# Import the indicator mapping and renaming dictionaries from the config module
from config.indicators import (
    indicator_map,
    rename_map,
)

# Filter the DataFrame to keep only the ND-GAIN indicators used by the project and replace API indicator codes with user-friendly names.
def filter_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the ND-GAIN indicators used by the project and
    replace API indicator codes with user-friendly names.
    """

    df = df[df["INDICATOR"].isin(indicator_map.values())].copy()

    df["INDICATOR"] = (
        df["INDICATOR"]
        .replace(rename_map)
    )

    return df
