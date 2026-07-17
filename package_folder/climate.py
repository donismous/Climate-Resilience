import os
from functools import lru_cache

import pandas as pd

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
FORECAST_PATH = os.path.join(ROOT_PATH, "data", "outputs", "risk_score_with_forecast.csv")


@lru_cache(maxsize=1)
def _load_forecast() -> pd.DataFrame:
    """Load the precomputed actual + forecast risk scores from disk.

    Cached after the first call so the CSV is only read once per running
    container, not once per request.

    Returns:
        DataFrame with columns ``country``, ``year``, ``risk_score``,
        ``source``, ``lower``, ``upper`` (see
        ``model.basic_arima_model.extend_with_forecast``).

    Raises:
        FileNotFoundError: If the forecast CSV hasn't been generated /
            copied into the image.
    """
    if not os.path.exists(FORECAST_PATH):
        raise FileNotFoundError(
            f"Forecast data not found at {FORECAST_PATH!r}. Run "
            "`python model/basic_arima_model.py` and make sure the output "
            "CSV is copied into the image as data/outputs/risk_score_with_forecast.csv."
        )
    return pd.read_csv(FORECAST_PATH)


def prediction_function(country: str, year: int) -> dict:
    """Look up the risk score for a country/year from the precomputed data.

    Args:
        country: ISO3 country code, e.g. "FRA".
        year: Calendar year to look up (actual or forecast, whichever is
            available up to the horizon baked into the CSV).

    Returns:
        A dict with ``risk_score``, ``source`` ("actual" or "forecast"),
        and ``lower``/``upper`` confidence bounds (``None`` for actual rows).

    Raises:
        ValueError: If there's no row for that country/year combination.
    """
    df = _load_forecast()
    match = df[(df["country"] == country.upper()) & (df["year"] == year)]

    if match.empty:
        available = sorted(df.loc[df["country"] == country.upper(), "year"].unique())
        raise ValueError(
            f"No data for country={country!r}, year={year!r}. "
            f"Available years for {country.upper()}: {available or 'none'}."
        )

    row = match.iloc[0]
    return {
        "risk_score": float(row["risk_score"]),
        "source": row["source"],
        "lower": None if pd.isna(row.get("lower")) else float(row["lower"]),
        "upper": None if pd.isna(row.get("upper")) else float(row["upper"]),
    }
