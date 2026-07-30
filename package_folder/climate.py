import os
import json
import pandas as pd
from functools import lru_cache


ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
FORECAST_PATH = os.path.join(ROOT_PATH, "data", "outputs", "risk_score_with_forecast.csv")
LLM_CACHE_PATH = os.path.join(ROOT_PATH, "data", "outputs", "llm_summaries_cache.csv")
WORLD_MOVERS_PATH = os.path.join(ROOT_PATH, "data", "outputs", "world_movers.json")
COUNTRY_DETAILS_PATH = os.path.join(ROOT_PATH, "data", "outputs", "country_details.json")

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


@lru_cache(maxsize=1)
def _load_llm_cache() -> pd.DataFrame:
    """Load precomputed LLM summaries, or an empty frame if none exist yet."""
    if not os.path.exists(LLM_CACHE_PATH):
        return pd.DataFrame(columns=["scope", "summary"])
    return pd.read_csv(LLM_CACHE_PATH)


def get_cached_summary(scope: str) -> str | None:
    """Look up a precomputed summary by scope ('world' or an ISO3 code)."""
    df = _load_llm_cache()
    match = df[df["scope"] == scope]
    return None if match.empty else match.iloc[0]["summary"]


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
def _load_world_movers() -> dict:
    if not os.path.exists(WORLD_MOVERS_PATH):
        raise FileNotFoundError(
            f"{WORLD_MOVERS_PATH!r} not found. Run `python model/generate_dashboard_data.py` first."
        )
    with open(WORLD_MOVERS_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_country_details() -> dict:
    if not os.path.exists(COUNTRY_DETAILS_PATH):
        raise FileNotFoundError(
            f"{COUNTRY_DETAILS_PATH!r} not found. Run `python model/generate_dashboard_data.py` first."
        )
    with open(COUNTRY_DETAILS_PATH) as f:
        return json.load(f)


def get_global_movers(top_n: int = 5) -> dict:
    """Return precomputed global movers, sliced to top_n (precomputed with TOP_N=10)."""
    data = _load_world_movers()
    return {
        "improved": data["improved"][:top_n],
        "worsened": data["worsened"][:top_n],
        "best_current": data["best_current"][:top_n],
        "worst_current": data["worst_current"][:top_n],
        "global_mean_slope": data["global_mean_slope"],
        "global_direction": data["global_direction"],
    }


def get_country_detail(country: str) -> dict:
    """Return precomputed detail for one country."""
    data = _load_country_details()
    country = country.upper()
    if country not in data:
        raise ValueError(f"No detail for country={country!r}.")
    return data[country]
