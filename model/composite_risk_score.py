"""Composite climate risk score from ND-GAIN vulnerability and readiness.

Combines the two headline ND-GAIN indicators into a single risk score per
country and year:

    Risk Score = (Vulnerability x 0.6) + (1 - Readiness) x 0.4

Both indicators are on a 0-1 scale, where higher vulnerability means more
exposed and higher readiness means better prepared, so the score is also on
a 0-1 scale with higher values meaning higher climate risk.

This version reads directly from the already-preprocessed, wide-format ND-GAIN
export (one row per Country/Year, one column per ND-GAIN sub-indicator,
including ``Vulnerability`` and ``Readiness`` as columns in their own right).
No filtering, pivoting, or renaming is needed here -- this module only does
the weighted-sum arithmetic and normalizes column names to lowercase
(``country``, ``year``, ``risk_score``) for the downstream ARIMA/ETS models.

Usage:
    Run as a script to load ``processed_data.csv``, compute the scores and
    save them to ``composite_risk_score.csv``:

        python model/composite_risk_score.py

    Or import from another module / notebook:

        import pandas as pd
        from model.composite_risk_score import compute_composite_risk

        df = pd.read_csv("processed_data.csv")
        risk = compute_composite_risk(df)
"""

import pandas as pd

# Column names as they appear in the processed ND-GAIN export.
VULNERABILITY_COLUMN = "Vulnerability"
READINESS_COLUMN = "Readiness"

# Weights of the composite score.
VULNERABILITY_WEIGHT = 0.6
READINESS_WEIGHT = 0.4


def compute_composite_risk(
    df: pd.DataFrame,
    vulnerability_col: str = VULNERABILITY_COLUMN,
    readiness_col: str = READINESS_COLUMN,
) -> pd.DataFrame:
    """Compute the composite risk score by country and year.

    Args:
        df: Wide-format DataFrame with one row per (Country, Year) and a
            column per ND-GAIN sub-indicator. Must contain the vulnerability
            and readiness columns, plus ``Country`` and ``Year``.
        vulnerability_col: Column name holding the vulnerability score.
        readiness_col: Column name holding the readiness score.

    Returns:
        A DataFrame with one row per (country, year) where both indicators
        are available, and columns ``country``, ``year``, ``vulnerability``,
        ``readiness`` and ``risk_score``, sorted by country then year.

    Raises:
        ValueError: If a required column is missing from ``df``.
    """
    required = ["Country", "Year", vulnerability_col, readiness_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    out = df.dropna(subset=[vulnerability_col, readiness_col]).copy()

    out["risk_score"] = (
        out[vulnerability_col] * VULNERABILITY_WEIGHT
        + (1 - out[readiness_col]) * READINESS_WEIGHT
    )

    out = out.rename(
        columns={
            "Country": "country",
            "Year": "year",
            vulnerability_col: "vulnerability",
            readiness_col: "readiness",
        }
    )

    columns = ["country", "year", "vulnerability", "readiness", "risk_score"]
    return out[columns].sort_values(["country", "year"]).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("processed_data.csv")
    risk = compute_composite_risk(df)
    print(risk.head())
    risk.to_csv("composite_risk_score.csv", index=False)
    print(f"Saved {len(risk)} rows to composite_risk_score.csv")
