import os
import json
import pandas as pd
import pycountry
from functools import lru_cache


ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
FORECAST_PATH = os.path.join(ROOT_PATH, "data", "outputs", "all_indicator_values.csv")
LLM_CACHE_PATH = os.path.join(ROOT_PATH, "data", "outputs", "llm_summaries_cache.csv")
WORLD_MOVERS_PATH = os.path.join(ROOT_PATH, "data", "outputs", "world_movers.json")
COUNTRY_DETAILS_PATH = os.path.join(ROOT_PATH, "data", "outputs", "country_details.json")
TIERS_PATH = os.path.join(ROOT_PATH, "data", "outputs", "country_tiers.json")
FEATURE_IMPORTANCE_PATH = os.path.join(ROOT_PATH, "data", "outputs", "global_feature_importance.csv")

# SHAP values behind global_feature_importance.csv were fit against an
# earlier, PCA-weighted version of CompositeRisk, not the current
# 0.6*Vulnerability + 0.4*(1-Readiness) risk_score. Surfaced to API
# consumers so nobody presents these numbers as exact for the live score.
FEATURE_IMPORTANCE_CAVEAT = (
    "Estimated from a SHAP analysis of historical data using an earlier "
    "version of the scoring formula. Directionally accurate, but will be "
    "refreshed once the model is retrained on the current formula."
)

# Load new country names
COUNTRY_NAMES_PATH = os.path.join(ROOT_PATH, "config", "iso3_to_region_name.csv")


@lru_cache(maxsize=1)
def _load_forecast() -> pd.DataFrame:
    """Load the precomputed actual + forecast indicator values from disk.

    Cached after the first call so the CSV is only read once per running
    container, not once per request.

    Returns:
        Long-format DataFrame with columns ``country``, ``indicator``,
        ``year``, ``value``, ``source``, ``lower``, ``upper`` -- one row
        per country/indicator/year.

    Raises:
        FileNotFoundError: If the forecast CSV hasn't been generated /
            copied into the image.
    """
    if not os.path.exists(FORECAST_PATH):
        raise FileNotFoundError(
            f"Forecast data not found at {FORECAST_PATH!r}. Generate it and "
            "make sure the output CSV is copied into the image as "
            "data/outputs/all_indicators_ets_forecast.csv."
        )
    return pd.read_csv(FORECAST_PATH)


@lru_cache(maxsize=1)
def _load_country_names() -> dict:
    """Load the ISO3 -> country name mapping, cached after first read."""
    names_df = pd.read_csv(COUNTRY_NAMES_PATH, sep=",", usecols=["ISO3", "Name", "region","sub_region"])
    # return dict(zip(names_df["ISO3"], names_df["Name"],names_df["region"],names_df["sub_region"] ))
    return {
        iso: {"Name": name, "region": region, "sub_region": sub_region}
        for iso, name, region, sub_region in zip(
            names_df["ISO3"],
            names_df["Name"],
            names_df["region"],
            names_df["sub_region"]
        )
    }

@lru_cache(maxsize=1)
def _load_llm_cache() -> pd.DataFrame:
    """Load precomputed LLM summaries/recommendations, or an empty frame."""
    if not os.path.exists(LLM_CACHE_PATH):
        return pd.DataFrame(columns=["scope", "kind", "summary"])
    return pd.read_csv(LLM_CACHE_PATH)

@lru_cache(maxsize=1)
def _load_tiers() -> list[dict]:
    if not os.path.exists(TIERS_PATH):
        raise FileNotFoundError(
            f"{TIERS_PATH!r} not found. Run `python model/generate_tiers.py` first."
        )
    with open(TIERS_PATH) as f:
        return json.load(f)


def get_cached_summary(scope: str) -> str | None:
    """Look up a precomputed dashboard/country summary by scope."""
    df = _load_llm_cache()
    match = df[(df["scope"] == scope) & (df["kind"] == "summary")]
    return None if match.empty else match.iloc[0]["summary"]


def get_cached_recommendation(scope: str, persona: str) -> str | None:
    """Look up a precomputed recommendation by scope + persona.

    Only 'individual' and 'government/institution' are ever cached, since
    'business' varies by free-text industry and can't be precomputed.
    """
    df = _load_llm_cache()
    match = df[(df["scope"] == scope) & (df["kind"] == f"recommendation_{persona}")]
    return None if match.empty else match.iloc[0]["summary"]


@lru_cache(maxsize=None)
def flag_url(iso3: str) -> str | None:
    """Flag image URL (flagcdn.com) for an ISO3 code, or None if unmapped.

    flagcdn addresses flags by lowercase ISO alpha-2 code, so the ISO3
    code is converted first.
    """
    country = pycountry.countries.get(alpha_3=iso3)
    if country is None:
        return None
    return f"https://flagcdn.com/24x18/{country.alpha_2.lower()}.png"


def _format_record(row: pd.Series, country_names: dict) -> dict:
    """Shape one row of the long-format forecast data into an API record.

    Shared by prediction_function() and all_predictions() so /predict and
    /predict_all always return identically-shaped records.
    """
    country_info = country_names.get(row["country"], {})
    return {
        "country": row["country"],
        "indicator": row["indicator"],
        "country_name": country_info.get("Name"),
        "region": country_info.get("region"),
        "sub_region": country_info.get("sub_region"),
        "year": int(row["year"]),
        "value": float(row["value"]),
        "source": row["source"],
        "flag": flag_url(row["country"]),
    }


def prediction_function(country: str, year: int) -> list[dict]:
    """Look up every indicator's value for a country/year from the precomputed data.

    Args:
        country: ISO3 country code, e.g. "FRA".
        year: Calendar year to look up (actual or forecast, whichever is
            available up to the horizon baked into the CSV).

    Returns:
        A list of dicts, one per ND-GAIN indicator, shaped identically to
        all_predictions()'s records (country, indicator, country_name,
        region, sub_region, year, value, source, lower, upper).

    Raises:
        ValueError: If there's no row for that country/year combination.
    """
    df = _load_forecast()
    country_names = _load_country_names()
    match = df[(df["country"] == country.upper()) & (df["year"] == year)]

    if match.empty:
        available = sorted(df.loc[df["country"] == country.upper(), "year"].unique())
        raise ValueError(
            f"No data for country={country!r}, year={year!r}. "
            f"Available years for {country.upper()}: {available or 'none'}."
        )

    return [_format_record(row, country_names) for _, row in match.iterrows()]


def all_predictions(year: int | None = None) -> list[dict]:
    """Return every country/year/indicator row from the precomputed data.

    Args:
        year: If given, restrict to this calendar year (still all countries).
            If omitted, returns every row in the dataset.

    Returns:
        A list of dicts, each shaped like the /predict response: country,
        indicator, country_name, region, sub_region, year, value, source,
        lower, upper.
    """
    df = _load_forecast()
    country_names = _load_country_names()

    if year is not None:
        df = df[df["year"] == year]

    return [_format_record(row, country_names) for _, row in df.iterrows()]


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
        "best_indicators_global": data.get("best_indicators_global", []),
        "worst_indicators_global": data.get("worst_indicators_global", []),
        "forecast_target_year": data.get("forecast_target_year"),
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





def get_country_tiers(year: int | None = None) -> list[dict]:
    """Return precomputed Vulnerability x Readiness tier assignments.

    Args:
        year: Optional calendar year to narrow results to (still all countries).
    """
    records = _load_tiers()
    if year is not None:
        records = [r for r in records if r["year"] == year]
    return [{**r, "flag": flag_url(r["country"])} for r in records]


@lru_cache(maxsize=1)
def _load_feature_importance() -> pd.DataFrame:
    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        raise FileNotFoundError(
            f"{FEATURE_IMPORTANCE_PATH!r} not found. Run the SHAP analysis "
            "step (model/convert_shap_pca.py) first."
        )
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


def get_feature_importance() -> dict:
    """Return each ND-GAIN sub-indicator's global contribution to the risk
    score, ranked by SHAP importance (highest first)."""
    df = _load_feature_importance().sort_values("SHAP Importance (%)", ascending=False)
    records = [
        {
            "indicator": row["Feature"],
            "importance_pct": round(float(row["SHAP Importance (%)"]), 2),
            "mean_abs_shap": float(row["Mean |SHAP|"]),
            "rf_importance": float(row["RF Importance"]),
        }
        for _, row in df.iterrows()
    ]
    return {"data": records, "caveat": FEATURE_IMPORTANCE_CAVEAT}
