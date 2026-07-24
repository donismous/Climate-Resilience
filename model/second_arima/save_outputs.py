"""
ARIMA Time Series Forecasting Model
Strategy: 1 order per indicator

This module handles:

#add exposure back to the dataframe (constant value for each country)
#calculate vulnerability based on results of vulnerability sectors and calculate readiness based on readiness sectors
#calculate composite risk score with (0.4*(1-readiness) + 0.6*vulnerability)
#save all_indicators_values.csv (including vulnerability, readiness, composite risk score) to data/outputs
#save composite_risk_score.csv to data/outputs

"""

import pandas as pd
import numpy as np


def calculate_composite_metrics(forecast_df, df_original):

    vulnerability_indicators = [
    "Food",
    "Water",
    "Health",
    "Habitat",
    "Infrastructure",
    "Ecosystems",
    "Sensitivity",
    "Capacity"
    ]

    readiness_indicators = [
    "Economic",
    "Governance",
    "Social"
    ]

    exposure = (
        df_original
        .reset_index()
        .groupby(["Country", "Year"])["Exposure"]
        .first()
        .rename("Exposure")
    )

    vulnerability = (
    forecast_df[
        forecast_df["Indicator"].isin(vulnerability_indicators)
    ]
    .groupby(["Country","Year"])["Value"]
    .mean()
    .rename("Vulnerability")
    )

    readiness = (
    forecast_df[
        forecast_df["Indicator"].isin(readiness_indicators)
    ]
    .groupby(["Country","Year"])["Value"]
    .mean()
    .rename("Readiness")
    )

    summary = pd.concat(
    [vulnerability, readiness, exposure],
    axis=1
    ).reset_index()

    summary["CompositeRisk"] = (
    0.4 * (1 - summary["Readiness"])
    + 0.6 * summary["Vulnerability"]
    )

    extra = []

    indicators = [
        "Exposure",
        "Vulnerability",
        "Readiness",
        "CompositeRisk"
        ]

    extra = []

    for _, row in summary.iterrows():
        for indicator in indicators:
            extra.append({
                "Country": row.Country,
                "Year": row.Year,
                "Indicator": indicator,
                "Value": row[indicator]
            })

    forecast_df = pd.concat(
        [forecast_df, pd.DataFrame(extra)],
        ignore_index=True
    )

    forecast_df = forecast_df.sort_values(
        ["Country", "Year"]
        ).reset_index(drop=True)

    return forecast_df, summary

def save_outputs(forecast_df, summary):
    forecast_df.to_csv(
        "data/outputs/all_indicator_values.csv",
        index=False
    )

    summary.to_csv(
        "data/outputs/composite_risk_score.csv",
        index=False
        )

def main():

    forecast_df = pd.read_parquet("data/intermediate/forecast.parquet")
    df_original = pd.read_parquet("data/intermediate/df_original.parquet")

    # Post-process
    forecast_df, summary = calculate_composite_metrics(forecast_df, df_original)

    # Save
    save_outputs(
        forecast_df,
        summary,
    )

if __name__ == "__main__":
    main()
