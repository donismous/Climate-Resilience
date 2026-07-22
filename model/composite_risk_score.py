"""Composite climate risk score from ND-GAIN vulnerability and readiness.

Combines the two headline ND-GAIN indicators into a single risk score per
country and year:

    Risk Score = (Vulnerability x 0.6) + (1 - Readiness) x 0.4

Both indicators are on a 0-1 scale, where higher vulnerability means more
exposed and higher readiness means better prepared, so the score is also on
a 0-1 scale with higher values meaning higher climate risk.

The input is the preprocessed dataset produced by
``data.preprocessing.pipeline.preprocess`` (saved to
``data/data_preprocessed/processed_data.csv``): one row per (country, year)
with one column per indicator, already cleaned, filtered and imputed. All
indicator columns are kept in the output alongside the new ``risk_score``
column, so the saved dataset is the single starting point for every
downstream model (composite score and per-indicator ARIMA alike). The raw
API download is fetched once by ``fetch_ndgain`` and is not used here.

Usage:
    Run as a script to load the preprocessed data, compute the scores and
    save them to ``data/outputs/composite_risk_score.csv``:

        python model/composite_risk_score.py

    Or import from another module / notebook:

        from model.composite_risk_score import get_composite_risk

        risk = get_composite_risk()
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT / "data" / "data_preprocessed" / "processed_data.csv"
OUTPUT_PATH = ROOT / "data" / "outputs" / "composite_risk_score.csv"

# Headline indicator columns in the preprocessed dataset.
VULNERABILITY_COLUMN = "Vulnerability"
READINESS_COLUMN = "Readiness"

# Weights of the composite score.
VULNERABILITY_WEIGHT = 0.6
READINESS_WEIGHT = 0.4


def compute_composite_risk(
    df: pd.DataFrame,
    vulnerability_column: str = VULNERABILITY_COLUMN,
    readiness_column: str = READINESS_COLUMN,
) -> pd.DataFrame:
    """Compute the composite risk score by country and year.

    Args:
        df: Preprocessed DataFrame from the preprocessing pipeline (or read
            from ``data/data_preprocessed/processed_data.csv``), with one
            row per (country, year) and one column per indicator. Must
            contain the ``Country``, ``Year``, vulnerability and readiness
            columns.
        vulnerability_column: Column holding the vulnerability score.
        readiness_column: Column holding the readiness score.

    Returns:
        The input DataFrame (all indicator columns preserved) with an added
        ``risk_score`` column, restricted to rows where both headline
        indicators are available, sorted by country then year.

    Raises:
        ValueError: If either indicator column is missing from ``df``.
    """
    for column in (vulnerability_column, readiness_column):
        if column not in df.columns:
            raise ValueError(f"Column {column!r} not found in the data.")

    # The score needs both components, so drop years missing either.
    risk = df.dropna(subset=[vulnerability_column, readiness_column]).copy()
    risk["risk_score"] = (
        risk[vulnerability_column] * VULNERABILITY_WEIGHT
        + (1 - risk[readiness_column]) * READINESS_WEIGHT
    )
    return risk.sort_values(["Country", "Year"]).reset_index(drop=True)


def get_composite_risk() -> pd.DataFrame:
    """Load the composite risk dataset, computing and saving it if missing.

    Resolution order:
        1. ``data/outputs/composite_risk_score.csv`` if it exists.
        2. Otherwise compute it from
           ``data/data_preprocessed/processed_data.csv`` and save it.
        3. If the preprocessed file is missing too, run the preprocessing
           pipeline on the one-off raw download first (requires the current
           working directory to be the repo root, as the pipeline reads its
           config files with relative paths).

    Returns:
        A DataFrame with one row per (country, year), one column per
        indicator and a ``risk_score`` column.
    """
    if OUTPUT_PATH.exists():
        return pd.read_csv(OUTPUT_PATH)

    if PROCESSED_PATH.exists():
        df = pd.read_csv(PROCESSED_PATH)
    else:
        from data.preprocessing.pipeline import preprocess

        raw = pd.read_csv(ROOT / "data" / "inputs" / "fetch" / "ndgain_raw.csv")
        df = preprocess(raw)

    risk = compute_composite_risk(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    risk.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(risk)} rows to {OUTPUT_PATH}")
    return risk

    out = out.rename(
        columns={
            "Country": "country",
            "Year": "year",
            vulnerability_col: "vulnerability",
            readiness_col: "readiness",
        }
    )

if __name__ == "__main__":
    import sys

    # Allow running from the repo root or the model/ folder.
    sys.path.insert(0, str(ROOT))

    # Force a fresh computation: running the script means regenerating the
    # output, so bypass the existing-file shortcut of get_composite_risk.
    if PROCESSED_PATH.exists():
        df = pd.read_csv(PROCESSED_PATH)
    else:
        from data.preprocessing.pipeline import preprocess

        df = preprocess(
            pd.read_csv(ROOT / "data" / "inputs" / "fetch" / "ndgain_raw.csv")
        )

    risk = compute_composite_risk(df)
    print(risk.head())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    risk.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(risk)} rows to {OUTPUT_PATH}")
