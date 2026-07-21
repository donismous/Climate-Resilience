"""Composite climate risk score from ND-GAIN vulnerability and readiness.

Combines the two headline ND-GAIN indicators into a single risk score per
country and year:

    Risk Score = (Vulnerability x 0.6) + (1 - Readiness) x 0.4

Both indicators are on a 0-1 scale, where higher vulnerability means more
exposed and higher readiness means better prepared, so the score is also on
a 0-1 scale with higher values meaning higher climate risk.

Usage:"""Composite climate risk score from ND-GAIN vulnerability and readiness.

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
VULNERABILITY_INDICATOR = "NDGAIN_VULNERABILITY"
READINESS_INDICATOR = "NDGAIN_READINESS"

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
    from data.inputs.fetch.fetch_ndgain import fetch_ndgain

    df = fetch_ndgain()
    risk = compute_composite_risk(df)
    print(risk.head())
    risk.to_csv("composite_risk_score.csv", index=False)
    print(f"Saved {len(risk)} rows to composite_risk_score.csv")
