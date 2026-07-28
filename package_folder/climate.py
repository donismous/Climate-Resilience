import os
import numpy as np
import pandas as pd
from functools import lru_cache
from scipy.stats import linregress


ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
FORECAST_PATH = os.path.join(ROOT_PATH, "data", "outputs", "risk_score_with_forecast.csv")
PROCESSED_DATA_PATH = os.path.join(ROOT_PATH, "data", "data_preprocessed", "processed_data.csv")

READINESS_INDICATORS = ["Economic", "Governance", "Social"]
VULNERABILITY_INDICATORS = [
    "Exposure", "Sensitivity", "Capacity", "Food", "Water",
    "Health", "Ecosystems", "Habitat", "Infrastructure",
]

# Load country names
COUNTRY_NAMES_PATH = os.path.join(ROOT_PATH, "config", "iso3_to_country_name.csv")


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


@lru_cache(maxsize=1)
def _load_country_names() -> dict:
    """Load the ISO3 -> country name mapping, cached after first read."""
    names_df = pd.read_csv(COUNTRY_NAMES_PATH, sep=";", usecols=["ISO3", "Name"])
    return dict(zip(names_df["ISO3"], names_df["Name"]))

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
        "country_name": _load_country_names().get(country.upper()),
        "risk_score": float(row["risk_score"]),
        "source": row["source"],
        "lower": None if pd.isna(row.get("lower")) else float(row["lower"]),
        "upper": None if pd.isna(row.get("upper")) else float(row["upper"]),
    }

def all_predictions(year: int | None = None) -> list[dict]:
    """Return every country/year row from the precomputed data.

    Args:
        year: If given, restrict to this calendar year (still all countries).
            If omitted, returns every row in the dataset.

    Returns:
        A list of dicts, each shaped like the /predict response: country,
        year, risk_score, source, lower, upper.
    """
    df = _load_forecast()
    country_names = _load_country_names()
    if year is not None:
        df = df[df["year"] == year]

    records = []
    for _, row in df.iterrows():
        records.append({
            "country": row["country"],
            "country_name": country_names.get(row["country"]),
            "year": int(row["year"]),
            "risk_score": float(row["risk_score"]),
            "source": row["source"],
            "lower": None if pd.isna(row.get("lower")) else float(row["lower"]),
            "upper": None if pd.isna(row.get("upper")) else float(row["upper"]),
        })
    return records

@lru_cache(maxsize=1)
def _load_processed_data() -> pd.DataFrame:
    """Load the real historical (non-forecast) preprocessed dataset."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH!r}.")
    return pd.read_csv(PROCESSED_DATA_PATH)

def _country_trend(df: pd.DataFrame, country: str) -> dict:
    """Real historical trend for one country (not a forecast)."""
    sub = df[df["Country"] == country].sort_values("Year")
    slope, intercept, r_value, p_value, std_err = linregress(
        sub["Year"].values.astype(float), sub["risk_score"].values.astype(float)
    )
    return {
        "country": country,
        "country_name": _load_country_names().get(country),
        "slope_per_year": slope,
        "significant": p_value < 0.05,
        "direction": "improving" if slope < 0 else "worsening",
    }

def get_global_movers(top_n: int = 5) -> dict:
    """Countries that improved/worsened the most long-term, plus global trend.

    Returns:
        dict with ``improved`` (list of top_n dicts, most negative slope),
        ``worsened`` (top_n dicts, most positive slope), and
        ``global_mean_slope`` (float).
    """
    from model.composite_risk_score import compute_composite_risk

    raw = _load_processed_data()
    risk_df = compute_composite_risk(raw)

    trends = [_country_trend(risk_df, c) for c in risk_df["Country"].unique()]
    trends = [t for t in trends if t["significant"]]  # only real trends, not noise

    trends_sorted = sorted(trends, key=lambda t: t["slope_per_year"])
    improved = trends_sorted[:top_n]           # most negative slope = most improved
    worsened = list(reversed(trends_sorted[-top_n:]))  # most positive slope = most worsened

    global_mean_slope = float(np.mean([t["slope_per_year"] for t in trends]))

    return {
        "improved": improved,
        "worsened": worsened,
        "global_mean_slope": global_mean_slope,
        "global_direction": "improving" if global_mean_slope < 0 else "worsening",
    }

def get_country_detail(country: str) -> dict:
    """Country-specific trend + indicator breakdown.

    Returns:
        dict with ``trend`` (from _country_trend) and ``indicators``: a list
        of {name, latest_value, category, is_favorable} sorted so the
        strongest (most favorable) indicators come first.
    """
    from model.composite_risk_score import compute_composite_risk

    raw = _load_processed_data()
    risk_df = compute_composite_risk(raw)

    country = country.upper()
    sub = risk_df[risk_df["Country"] == country]
    if sub.empty:
        raise ValueError(f"No data for country={country!r}.")

    trend = _country_trend(risk_df, country)
    latest = sub.sort_values("Year").iloc[-1]

    indicators = []
    for name in READINESS_INDICATORS:
        indicators.append({
            "name": name,
            "category": "readiness",
            "latest_value": float(latest[name]),
            "favorable_score": float(latest[name]),  # higher = better, use directly
        })
    for name in VULNERABILITY_INDICATORS:
        indicators.append({
            "name": name,
            "category": "vulnerability",
            "latest_value": float(latest[name]),
            "favorable_score": 1 - float(latest[name]),  # invert: higher = better
        })

    indicators.sort(key=lambda i: i["favorable_score"], reverse=True)

    return {
        "country": country,
        "country_name": _load_country_names().get(country),
        "trend": trend,
        "strongest_indicators": indicators[:3],
        "weakest_indicators": indicators[-3:],
    }
