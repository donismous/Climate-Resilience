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
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


def calculate_composite_metrics(forecast_df, df_original, weights):
    print(forecast_df["Year"].min(), forecast_df["Year"].max())
    print(sorted(forecast_df["Year"].unique())[-10:])

    vulnerability_indicators = [
        "Food", "Water", "Health", "Habitat",
        "Infrastructure", "Ecosystems",
        "Sensitivity", "Capacity"
    ]

    readiness_indicators = [
        "Economic", "Governance", "Social"
    ]

    calculated = [
        "Exposure",
        "Vulnerability",
        "Readiness",
        "CompositeRisk"
    ]

    # Remove previously calculated indicators
    forecast_df = forecast_df[
        ~forecast_df["Indicator"].isin(calculated)
    ].copy()

    # Exposure (constant across years)
    last_year = forecast_df["Year"].max()

    exposure = (
        df_original
        .groupby(["Country", "Year"])["Exposure"]
        .first()
    )

    full_index = pd.MultiIndex.from_product(
        [
            exposure.index.get_level_values("Country").unique(),
            range(
                exposure.index.get_level_values("Year").min(),
                last_year + 1
            ),
        ],
        names=["Country", "Year"],
    )

    exposure = (
        exposure
        .reindex(full_index)
        .groupby(level=0)
        .ffill()
    )

    # One row per Country-Year
    indicator_matrix = (
        forecast_df
        .pivot_table(
            index=["Country", "Year"],
            columns="Indicator",
            values="Value",
            aggfunc="first"
        )
    )

    # Keep only Country-Year pairs that exist in forecast_df
    indicator_matrix["Exposure"] = exposure.loc[indicator_matrix.index]

    indicator_matrix["Vulnerability"] = (
        indicator_matrix[vulnerability_indicators]
        .mean(axis=1)
    )

    indicator_matrix["Readiness"] = (
        indicator_matrix[readiness_indicators]
        .mean(axis=1)
    )

    indicator_matrix["CompositeRisk"] = (
        indicator_matrix[weights.index] @ weights
    ) / weights.sum()

    summary = (
        indicator_matrix[
            [
                "Exposure",
                "Vulnerability",
                "Readiness",
                "CompositeRisk"
            ]
        ]
        .reset_index()
    )

    extra = summary.melt(
        id_vars=["Country", "Year"],
        var_name="Indicator",
        value_name="Value"
    )

    forecast_df = (
        pd.concat([forecast_df, extra], ignore_index=True)
        .sort_values(["Country", "Indicator", "Year"])
        .reset_index(drop=True)
    )

    print(indicator_matrix.index.get_level_values("Year").max())
    print(summary["Year"].max())
    print(forecast_df["Year"].max())
    print(df_original.index.names)
    print(df_original.columns.tolist())
    print(
        forecast_df.groupby("Year")["Indicator"]
        .nunique()
        .tail(20)
    )

    print(
        forecast_df[forecast_df["Year"] > 2023]["Indicator"].unique()
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
