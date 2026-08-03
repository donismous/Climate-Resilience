"""Precompute all structured data the API needs for /world-summary and
/country-detail, so those endpoints do zero pandas/regression computation
at request time -- just JSON lookups.

Reads the single long-format forecast file (country, year, indicator,
value, source) covering all raw indicators + CompositeRisk, actual and
forecast. Also reads shap_analysis.json if it exists yet; if not, SHAP
fields are simply left as None and can be backfilled later by re-running
this script once that file is available -> no code changes needed then.

Usage:
    python model/generate_dashboard_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# CONFIRM/UPDATE this filename once the final long-format file exists.
LONG_FORMAT_PATH = ROOT / "data" / "outputs" / "all_indicator_values.csv"
SHAP_PATH = ROOT / "data" / "outputs" / "shap_analysis.json"
COUNTRY_NAMES_PATH = ROOT / "config" / "iso3_to_country_name.csv"

READINESS_INDICATORS = ["Economic", "Governance", "Social"]
VULNERABILITY_INDICATORS = [
    "Exposure", "Sensitivity", "Capacity", "Food", "Water",
    "Health", "Ecosystems", "Habitat", "Infrastructure",
]
COMPOSITE_LABEL = "CompositeRisk"  # confirm this matches the actual indicator label in the file
TOP_N = 10


def get_indicator_forecast(country_df: pd.DataFrame, country: str, indicator: str, year: int) -> float | None:
    """Look up a single indicator's value for one country/year, or None."""
    match = country_df[(country_df["indicator"] == indicator) & (country_df["year"] == year)]
    return float(match.iloc[0]["value"]) if not match.empty else None


def load_country_names() -> dict:
    names_df = pd.read_csv(COUNTRY_NAMES_PATH, sep=";", usecols=["ISO3", "Name"])
    return dict(zip(names_df["ISO3"], names_df["Name"]))


def load_shap_data() -> dict | None:
    if not SHAP_PATH.exists():
        print(f"{SHAP_PATH} not found yet -- proceeding without SHAP contributions.")
        return None
    with open(SHAP_PATH) as f:
        return json.load(f)


def generate_world_movers(long_df: pd.DataFrame, country_names: dict, shap_data: dict | None) -> dict:
    risk_actual = long_df[(long_df["indicator"] == COMPOSITE_LABEL) & (long_df["source"] == "actual")]
    risk_forecast = long_df[(long_df["indicator"] == COMPOSITE_LABEL) & (long_df["source"] == "forecast")]

    FORECAST_TARGET_YEAR = 2040

    trends = []
    for country in risk_actual["country"].unique():
        sub_actual = risk_actual[risk_actual["country"] == country].sort_values("year")
        if sub_actual.empty:
            continue
        latest_year = sub_actual["year"].max()
        latest_value = float(sub_actual[sub_actual["year"] == latest_year]["value"].iloc[0])

        forecast_row = risk_forecast[
            (risk_forecast["country"] == country) & (risk_forecast["year"] == FORECAST_TARGET_YEAR)
        ]
        if forecast_row.empty or FORECAST_TARGET_YEAR <= latest_year:
            continue
        forecast_value = float(forecast_row["value"].iloc[0])

        forecast_slope = (forecast_value - latest_value) / (FORECAST_TARGET_YEAR - latest_year)
        trends.append({
            "country": country,
            "country_name": country_names.get(country),
            "forecast_slope_per_year": forecast_slope,
            "direction": "improving" if forecast_slope < 0 else "worsening",
        })

    trends_sorted = sorted(trends, key=lambda t: t["forecast_slope_per_year"])
    improved = trends_sorted[:TOP_N]
    worsened = list(reversed(trends_sorted[-TOP_N:]))
    global_mean_slope = float(np.mean([t["forecast_slope_per_year"] for t in trends])) if trends else None

    latest_year = risk_actual["year"].max()
    latest = risk_actual[risk_actual["year"] == latest_year][["country", "value"]].sort_values("value")

    best_current = [
        {"country": row["country"], "country_name": country_names.get(row["country"]), "risk_score": float(row["value"])}
        for _, row in latest.head(TOP_N).iterrows()
    ]
    worst_current = [
        {"country": row["country"], "country_name": country_names.get(row["country"]), "risk_score": float(row["value"])}
        for _, row in latest.tail(TOP_N).iloc[::-1].iterrows()
    ]

    # Global strongest/weakest indicators: signed average SHAP contribution
    # across all countries. Negative = pulls risk down (a global strength);
    # positive = pushes risk up (a global weakness). Falls back to raw
    # favorable-value averaging if SHAP data isn't available yet.
    all_indicators = READINESS_INDICATORS + VULNERABILITY_INDICATORS
    local_shap = shap_data["local_shap"] if shap_data else {}

    if local_shap:
        indicator_shap_avgs = []
        for name in all_indicators:
            values = [country_shap[name] for country_shap in local_shap.values() if name in country_shap]
            if not values:
                continue
            indicator_shap_avgs.append({
                "name": name,
                "category": "readiness" if name in READINESS_INDICATORS else "vulnerability",
                "mean_shap_contribution": float(np.mean(values)),
            })
        indicator_shap_avgs.sort(key=lambda i: i["mean_shap_contribution"])
        best_indicators_global = indicator_shap_avgs[:3]
        worst_indicators_global = list(reversed(indicator_shap_avgs[-3:]))
    else:
        indicator_averages = []
        for name in all_indicators:
            is_readiness = name in READINESS_INDICATORS
            rows = long_df[
                (long_df["indicator"] == name) & (long_df["year"] == latest_year) & (long_df["source"] == "actual")
            ]
            if rows.empty:
                continue
            mean_value = float(rows["value"].mean())
            favorable_score = 1 - mean_value
            indicator_averages.append({
                "name": name,
                "category": "readiness" if is_readiness else "vulnerability",
                "global_mean_value": mean_value,
                "favorable_score": favorable_score,
            })
        indicator_averages.sort(key=lambda i: i["favorable_score"], reverse=True)
        best_indicators_global = indicator_averages[:3]
        worst_indicators_global = indicator_averages[-3:]

    return {
        "improved": improved,
        "worsened": worsened,
        "best_current": best_current,
        "worst_current": worst_current,
        "best_indicators_global": best_indicators_global,
        "worst_indicators_global": worst_indicators_global,
        "forecast_target_year": FORECAST_TARGET_YEAR,
        "global_mean_slope": global_mean_slope,
        "global_direction": "improving" if (global_mean_slope or 0) < 0 else "worsening",
    }


def generate_country_details(long_df: pd.DataFrame, country_names: dict, shap_data: dict | None) -> dict:
    global_importance = shap_data["global_importance"] if shap_data else {}
    local_shap = shap_data["local_shap"] if shap_data else {}
    importance_order = sorted(global_importance, key=global_importance.get, reverse=True) if global_importance else []

    FORECAST_TARGET_YEAR = 2040

    details = {}
    for country in long_df["country"].unique():
        country_df = long_df[long_df["country"] == country]
        risk_rows = country_df[(country_df["indicator"] == COMPOSITE_LABEL) & (country_df["source"] == "actual")]
        if risk_rows.empty:
            continue
        risk_rows = risk_rows.sort_values("year")

        latest_year = risk_rows["year"].max()
        latest_actual_value = float(risk_rows[risk_rows["year"] == latest_year]["value"].iloc[0])

        forecast_target_value = get_indicator_forecast(country_df, country, COMPOSITE_LABEL, FORECAST_TARGET_YEAR)
        if forecast_target_value is not None and FORECAST_TARGET_YEAR > latest_year:
            forecast_slope = (forecast_target_value - latest_actual_value) / (FORECAST_TARGET_YEAR - latest_year)
        else:
            forecast_slope = None

        trend = {
            "forecast_slope_per_year": forecast_slope,
            "direction": "improving" if (forecast_slope or 0) < 0 else "worsening",
            "latest_actual_value": latest_actual_value,
            "latest_actual_year": int(latest_year),
            f"forecast_{FORECAST_TARGET_YEAR}_value": forecast_target_value,
        }

        country_shap = local_shap.get(country, {})

        indicators = []
        for name in READINESS_INDICATORS + VULNERABILITY_INDICATORS:
            is_readiness = name in READINESS_INDICATORS
            latest_value = get_indicator_forecast(country_df, country, name, latest_year)
            if latest_value is None:
                continue
            favorable_score = 1 - latest_value
            shap_value = country_shap.get(name)

            indicators.append({
                "name": name,
                "category": "readiness" if is_readiness else "vulnerability",
                "latest_value": latest_value,
                "favorable_score": favorable_score,
                "forecast_2030": get_indicator_forecast(country_df, country, name, 2030),
                "forecast_2040": get_indicator_forecast(country_df, country, name, 2040),
                "shap_contribution": shap_value,
                "global_importance_rank": importance_order.index(name) + 1 if name in importance_order else None,
            })

        if not indicators:
            continue

        # Only rank by SHAP among indicators that actually have a SHAP
        # value; if SHAP data isn't available yet, fall back to
        # favorable_score so the pipeline still produces useful output.
        has_shap = any(i["shap_contribution"] is not None for i in indicators)
        if has_shap:
            with_shap = [i for i in indicators if i["shap_contribution"] is not None]
            # Strongest = most negative SHAP (biggest downward pull on risk)
            # Weakest = most positive SHAP (biggest upward pull on risk)
            with_shap.sort(key=lambda i: i["shap_contribution"])
            strongest = with_shap[:3]
            weakest = list(reversed(with_shap[-3:]))
        else:
            indicators.sort(key=lambda i: i["favorable_score"], reverse=True)
            strongest = indicators[:3]
            weakest = indicators[-3:]

        details[country] = {
            "country": country,
            "country_name": country_names.get(country),
            "trend": trend,
            "strongest_indicators": strongest,
            "weakest_indicators": weakest,
        }

    return details

def compute_feature_dependence_correlation(attribution_df: pd.DataFrame, feature_col: str = "Feature Value") -> dict:
    """For each feature, correlation between its raw value and its SHAP
    contribution across all country-year observations. Strong negative =
    higher values consistently reduce risk; strong positive = higher
    values consistently increase risk; near zero = inconsistent effect.
    """
    correlations = {}
    for feature, group in attribution_df.groupby("Feature"):
        if group[feature_col].nunique() < 2:
            continue
        corr = group[feature_col].corr(group["SHAP Value"])
        correlations[feature] = {
            "correlation": float(corr) if not pd.isna(corr) else None,
            "consistency": (
                "consistent" if abs(corr) >= 0.5 else "inconsistent"
            ) if not pd.isna(corr) else "unknown",
        }
    return correlations


if __name__ == "__main__":
    if not LONG_FORMAT_PATH.exists():
        sys.exit(f"{LONG_FORMAT_PATH} not found. Generate it first.")

    long_df = pd.read_csv(LONG_FORMAT_PATH)
    country_names = load_country_names()
    shap_data = load_shap_data()

    world_movers = generate_world_movers(long_df, country_names, shap_data)

    dependence_df = pd.read_csv(ROOT / "data" / "outputs" / "feature_dependence.csv")
    feature_dependence = compute_feature_dependence_correlation(dependence_df)
    world_movers["feature_dependence"] = feature_dependence

    world_movers_path = ROOT / "data" / "outputs" / "world_movers.json"
    with open(world_movers_path, "w") as f:
        json.dump(world_movers, f, indent=2)
    print(f"Saved {world_movers_path} ({len(world_movers['improved'])} improved, {len(world_movers['worsened'])} worsened)")

    country_details = generate_country_details(long_df, country_names, shap_data)
    country_details_path = ROOT / "data" / "outputs" / "country_details.json"
    with open(country_details_path, "w") as f:
        json.dump(country_details, f, indent=2)
    print(f"Saved {country_details_path} ({len(country_details)} countries, SHAP included: {shap_data is not None})")
