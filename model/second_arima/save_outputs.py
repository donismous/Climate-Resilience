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


def calculate_composite_metrics(forecast_df, df_original, weights):

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
        .groupby(["Country", "Year"])["Value"]
        .mean()
        .rename("Vulnerability")
    )

    readiness = (
        forecast_df[
            forecast_df["Indicator"].isin(readiness_indicators)
        ]
        .groupby(["Country", "Year"])["Value"]
        .mean()
        .rename("Readiness")
    )

    summary = pd.concat(
        [vulnerability, readiness, exposure],
        axis=1
    ).reset_index()

    # Create Country-Year x Indicator matrix
    indicator_matrix = (
        forecast_df
        .pivot(
            index=["Country", "Year"],
            columns="Indicator",
            values="Value"
        )
    )

    # Add Exposure to the matrix (it's in df_original, not forecast_df)
    indicator_matrix["Exposure"] = exposure

    # Reorder columns to match the weights
    indicator_matrix = indicator_matrix[weights.index]

    # Calculate PC1-weighted Composite Risk
    summary["CompositeRisk"] = (
        indicator_matrix @ weights
    ).values

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

    forecast_df = (
        forecast_df
        .sort_values(["Country", "Indicator", "Year"])
        .reset_index(drop=True)
    )

    return forecast_df, summary

def add_source(forecast_df):

    forecast_df["source"] = forecast_df["Year"].apply(
    lambda year: "actual" if year <= 2023 else "forecast"
    )

def save_outputs(forecast_df, summary):
    forecast_df.to_csv(
        "data/outputs/all_indicator_values.csv",
        index=False
    )

def change_column_names(forecast_df):
    forecast_df.rename(columns={
    "Country" : "country",
    "Year" : "year",
    "Indicator" : "indicator",
    "Value" : "value"
    }, inplace=True
    )

def main():

    forecast_df = pd.read_parquet("data/intermediate/forecast.parquet")
    df_original = pd.read_parquet("data/intermediate/df_original.parquet")

    weights = pd.read_csv("config/risk_score_weights.csv")
    weights = weights.rename(columns={"Unnamed: 0": "Indicator"})
    weights = weights.set_index("Indicator")
    weights = weights["PC1"]

    # Post-process
    forecast_df, summary = calculate_composite_metrics(forecast_df, df_original, weights)
    add_source(forecast_df)
    change_column_names(forecast_df)

    # Save
    save_outputs(
        forecast_df,
        summary,
    )

if __name__ == "__main__":
    main()
