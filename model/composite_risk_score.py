"""Composite climate risk score from ND-GAIN vulnerability and readiness.

Combines the two headline ND-GAIN indicators into a single risk score per
country and year:

    Risk Score = (Vulnerability x 0.6) + (1 - Readiness) x 0.4

Both indicators are on a 0-1 scale, where higher vulnerability means more
exposed and higher readiness means better prepared, so the score is also on
a 0-1 scale with higher values meaning higher climate risk.

Usage:
    Run as a script to download the data, compute the scores and save them
    to ``composite_risk_score.csv``:

        python model/composite_risk_score.py

    Or import from another module / notebook:

        from fetch_ndgain import fetch_ndgain
        from model.composite_risk_score import compute_composite_risk

        df = fetch_ndgain()
        risk = compute_composite_risk(df)
"""

import pandas as pd

# Headline ND-GAIN indicator codes in the Data360 dataset.
VULNERABILITY_INDICATOR = "UND_NDGAIN_VULNERABILITY"
READINESS_INDICATOR = "UND_NDGAIN_READINESS"

# Weights of the composite score.
VULNERABILITY_WEIGHT = 0.6
READINESS_WEIGHT = 0.4


def compute_composite_risk(
    df: pd.DataFrame,
    vulnerability_indicator: str = VULNERABILITY_INDICATOR,
    readiness_indicator: str = READINESS_INDICATOR,
) -> pd.DataFrame:
    """Compute the composite risk score by country and year.

    Args:
        df: DataFrame from ``fetch_ndgain`` with one row per
            (country, indicator, year) observation. Must contain both the
            vulnerability and readiness indicators.
        vulnerability_indicator: Indicator code for the vulnerability score.
        readiness_indicator: Indicator code for the readiness score.

    Returns:
        A DataFrame with one row per (country, year) where both indicators
        are available, and columns ``country``, ``year``, ``vulnerability``,
        ``readiness`` and ``risk_score``, sorted by country then year.

    Raises:
        ValueError: If either indicator is missing from ``df``.
    """
    for indicator in (vulnerability_indicator, readiness_indicator):
        if indicator not in df["INDICATOR"].values:
            raise ValueError(f"Indicator {indicator!r} not found in the data.")

    # One column per indicator, indexed by (country, year).
    wide = (
        df[df["INDICATOR"].isin([vulnerability_indicator, readiness_indicator])]
        .pivot_table(
            index=["REF_AREA", "TIME_PERIOD"],
            columns="INDICATOR",
            values="OBS_VALUE",
        )
        .rename(
            columns={
                vulnerability_indicator: "vulnerability",
                readiness_indicator: "readiness",
            }
        )
        # The score needs both components, so drop years missing either.
        .dropna(subset=["vulnerability", "readiness"])
        .reset_index()
        .rename(columns={"REF_AREA": "country", "TIME_PERIOD": "year"})
    )
    wide.columns.name = None

    wide["risk_score"] = (
        wide["vulnerability"] * VULNERABILITY_WEIGHT
        + (1 - wide["readiness"]) * READINESS_WEIGHT
    )
    return wide.sort_values(["country", "year"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow running from the repo root or the model/ folder.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fetch_ndgain import fetch_ndgain

    df = fetch_ndgain()
    risk = compute_composite_risk(df)
    print(risk.head())
    risk.to_csv("composite_risk_score.csv", index=False)
    print(f"Saved {len(risk)} rows to composite_risk_score.csv")
