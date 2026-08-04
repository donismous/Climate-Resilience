"""Precompute Vulnerability x Readiness tier assignments for every
country/year, so the API can serve them as a plain JSON lookup instead
of clustering at request time (or in the frontend).

Each year's cohort is split into 4 tiers by that year's median
Vulnerability and median Readiness (both ND-GAIN indicators, 0-1 scale,
already present in data/outputs/all_indicators_with_forecast.csv):

    High Risk  vulnerability above avg, lack of readiness above avg   -- most exposed, least prepared
    Prepared  vulnerability above avg, lack of readiness below avg   -- most exposed, most prepared
    Developing  vulnerability below avg, lack of readiness above avg   -- less exposed, least prepared
    Resilient  vulnerability below avg, lack of readiness below avg   -- less exposed, most prepared

Usage:
    python model/generate_tiers.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INDICATORS_PATH = ROOT / "data" / "outputs" / "all_indicator_values.csv"
COUNTRY_NAMES_PATH = ROOT / "config" / "iso3_to_region_name.csv"
OUTPUT_PATH = ROOT / "data" / "outputs" / "country_tiers.json"

TIERS = {
    4: {
        "tier": 4,
        "name": "High Risk",
        "label": "Most exposed, Least prepared",
        "emoji": "🔴",
        "color": "#B22222",
        "short": "High vulnerability, high lack of readiness",
        "note": "Most exposed to climate risk and least prepared to respond.",
    },
    3: {
        "tier": 3,
        "name": "Prepared",
        "label": "Most exposed, Most prepared",
        "emoji": "🟡",
        "color": "#D9A62E",
        "short": "High vulnerability, low lack of readiness",
        "note": "Faces serious climate exposure, but has institutions and resources in place.",
    },
    2: {
        "tier": 2,
        "name": "Developing",
        "label": "Least exposed, still unprepared",
        "emoji": "🔵",
        "color": "#3A6EA8",
        "short": "Low vulnerability, high lack of readiness",
        "note": "Not yet under severe climate pressure, but under-resourced if that changes.",
    },
    1: {
        "tier": 1,
        "name": "Resilient",
        "label": "Least exposed, Most prepared",
        "emoji": "🟢",
        "color": "#3F5730",
        "short": "Low vulnerability, low lack of readiness",
        "note": "Relatively low exposure and strong capacity to adapt.",
    },
}


def assign_tier(vulnerability: float, readiness: float, vuln_median: float, ready_median: float) -> int:
    high_vulnerability = vulnerability >= vuln_median
    high_readiness = readiness >= ready_median
    if high_vulnerability and high_readiness:
        return 4
    if high_vulnerability and not high_readiness:
        return 3
    if not high_vulnerability and high_readiness:
        return 2
    return 1


def load_country_names() -> dict:
    names_df = pd.read_csv(COUNTRY_NAMES_PATH, sep=",", usecols=["ISO3", "Name"])
    return dict(zip(names_df["ISO3"], names_df["Name"]))


def generate_tiers(long_df: pd.DataFrame, country_names: dict) -> list[dict]:
    subset = long_df[long_df["indicator"].isin(["Vulnerability", "Readiness"])]
    pivoted = subset.pivot_table(
        index=["country", "year"], columns="indicator", values="value"
    ).reset_index()
    pivoted = pivoted.dropna(subset=["Vulnerability", "Readiness"])

    records = []
    for year, year_df in pivoted.groupby("year"):
        vuln_median = year_df["Vulnerability"].median()
        ready_median = year_df["Readiness"].median()

        for _, row in year_df.iterrows():
            tier = assign_tier(row["Vulnerability"], row["Readiness"], vuln_median, ready_median)
            tier_info = TIERS[tier]
            records.append({
                "country": row["country"],
                "country_name": country_names.get(row["country"]),
                "year": int(year),
                "vulnerability": float(row["Vulnerability"]),
                "readiness": float(row["Readiness"]),
                "tier": tier,
                "tier_name": tier_info["name"],
                "tier_emoji": tier_info["emoji"],
                "tier_color": tier_info["color"],
            })
    return records


if __name__ == "__main__":
    if not INDICATORS_PATH.exists():
        sys.exit(f"{INDICATORS_PATH} not found. Generate it first.")

    long_df = pd.read_csv(INDICATORS_PATH)
    country_names = load_country_names()

    tiers = generate_tiers(long_df, country_names)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(tiers, f, indent=2)
    print(f"Saved {OUTPUT_PATH} ({len(tiers)} country-year tier assignments)")
