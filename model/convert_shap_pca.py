"""Convert SHAP/PCA output files into the two JSON files
this pipeline expects: shap_analysis.json (global + local SHAP, latest
year per country) and pca_summary.json (explained variance per component).

!!! Requires updated output files once model is finalized. !!!

Usage:
    python model/convert_shap_pca.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GLOBAL_IMPORTANCE_PATH = ROOT / "data" / "outputs" / "global_feature_importance.csv"
LOCAL_ATTRIBUTION_PATH = ROOT / "data" / "outputs" / "country_feature_attribution.csv"
PCA_PATH = ROOT / "data" / "outputs" / "PCA_explained_variation.csv"


def build_global_importance() -> dict:
    """Global feature importance, using SHAP Importance (%) -- already
    normalized and directly comparable to the local per-row percentages.
    """
    df = pd.read_csv(GLOBAL_IMPORTANCE_PATH)
    return dict(zip(df["Feature"], df["SHAP Importance (%)"]))


def build_local_shap() -> dict:
    """Per-country SHAP values, using each country's most recent year
    (matching the 'latest_value' year used elsewhere in country_details.json).
    """
    df = pd.read_csv(LOCAL_ATTRIBUTION_PATH)

    local_shap = {}
    for country, group in df.groupby("Country"):
        latest_year = group["Year"].max()
        latest = group[group["Year"] == latest_year]
        local_shap[country] = dict(zip(latest["Feature"], latest["SHAP Value"]))

    return local_shap


def build_pca_summary() -> dict:
    """PCA explained variance per component, for explain_drivers."""
    df = pd.read_csv(PCA_PATH)
    return {
        "explained_variance": dict(zip(df["PC"], df["Explained Variance"])),
        "cumulative_variance": dict(zip(df["PC"], df["Cumulative"])),
    }


if __name__ == "__main__":
    shap_analysis = {
        "global_importance": build_global_importance(),
        "local_shap": build_local_shap(),
        "_caveat": (
            "SHAP computed against a PCA-weighted CompositeRisk, not the "
            "ND-GAIN-based risk_score currently forecast/served elsewhere "
            "in this project. See module docstring."
        ),
    }
    shap_output_path = ROOT / "data" / "outputs" / "shap_analysis.json"
    with open(shap_output_path, "w") as f:
        json.dump(shap_analysis, f, indent=2)
    print(f"Saved {shap_output_path} ({len(shap_analysis['local_shap'])} countries)")

    pca_summary = build_pca_summary()
    pca_output_path = ROOT / "data" / "outputs" / "pca_summary.json"
    with open(pca_output_path, "w") as f:
        json.dump(pca_summary, f, indent=2)
    print(f"Saved {pca_output_path}")
