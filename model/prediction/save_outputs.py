"""
ARIMA Time Series Forecasting Model
Strategy: 1 order per indicator

This module handles:

#add exposure back to the dataframe (constant value for each country)
#calculate vulnerability based on results of vulnerability sectors and calculate readiness based on readiness sectors
#calculate composite risk score with PC1 loadings
#save all_indicators_values.csv (including vulnerability, readiness, composite risk score) to data/outputs
#save composite_risk_score.csv to data/outputs

"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


def calculate_composite_metrics(forecast_df, df_original, weights):
    """Calculate composite metrics (Vulnerability, Readiness, CompositeRisk) from forecast data.

    Expects lowercase column names: year, country, indicator, value
    """
    print(forecast_df["year"].min(), forecast_df["year"].max())
    print(sorted(forecast_df["year"].unique())[-10:])

    vulnerability_indicators = [
        "Food", "Water", "Health", "Habitat",
        "Infrastructure", "Ecosystems",
        "Sensitivity", "Capacity",
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
        ~forecast_df["indicator"].isin(calculated)
    ].copy()

    # Exposure (constant across years)
    last_year = forecast_df["year"].max()

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
            index=["country", "year"],
            columns="indicator",
            values="value",
            aggfunc="first"
        )
    )

    # Keep only Country-Year pairs that exist in forecast_df
    indicator_matrix["Exposure"] = exposure.loc[indicator_matrix.index]

    indicator_matrix["Vulnerability"] = (
        indicator_matrix[vulnerability_indicators + ["Exposure"]]
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
        id_vars=["country", "year"],
        var_name="indicator",
        value_name="value"
    )

    forecast_df = (
        pd.concat([forecast_df, extra], ignore_index=True)
        .sort_values(["country", "indicator", "year"])
        .reset_index(drop=True)
    )

    print(indicator_matrix.index.get_level_values("year").max())
    print(summary["year"].max())
    print(forecast_df["year"].max())
    print(df_original.index.names)
    print(df_original.columns.tolist())
    print(
        forecast_df.groupby("year")["indicator"]
        .nunique()
        .tail(20)
    )

    print(
        forecast_df[forecast_df["year"] > 2023]["indicator"].unique()
    )

    return forecast_df, summary


def add_source(forecast_df):
    """Add 'source' column indicating 'actual' or 'forecast' based on year."""
    forecast_df["source"] = forecast_df["year"].apply(
        lambda year: "actual" if year <= 2023 else "forecast"
    )
    return forecast_df


def change_column_names(forecast_df):
    """Convert lowercase column names to match output format."""
    forecast_df.rename(columns={
        "country": "country",  # Already lowercase, but keeping for clarity
        "year": "year",
        "indicator": "indicator",
        "value": "value"
    }, inplace=True)
    return forecast_df


def save_outputs(forecast_df, summary):
    """Save forecast results to CSV files."""
    forecast_df.to_csv(
        "data/outputs/all_indicator_values.csv",
        index=False
    )
    print(f"Saved {len(forecast_df)} rows to data/outputs/all_indicator_values.csv")


def main():
    forecast_df = pd.read_parquet("data/intermediate/forecast.parquet")
    df_original = pd.read_parquet("data/intermediate/df_original.parquet")

    weights = pd.read_csv("config/risk_score_weights.csv")
    weights = weights.rename(columns={"Unnamed: 0": "Indicator"})
    weights = weights.set_index("Indicator")
    weights = weights["PC1"]

    # Post-process (order matters: calculate first, then add source)
    forecast_df, summary = calculate_composite_metrics(forecast_df, df_original, weights)
    forecast_df = add_source(forecast_df)
    forecast_df = change_column_names(forecast_df)

    # Save
    save_outputs(forecast_df, summary)


if __name__ == "__main__":
    main()
