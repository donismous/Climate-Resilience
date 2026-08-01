"""
Feature Importance Comparison.

This module combines feature importance metrics from multiple methods into a
single comparison table.

The implemented comparison includes:
    - Random Forest feature importance,
    - Mean absolute SHAP values,
    - PCA-derived feature weights.

These complementary approaches allow interpretation of feature influence from
both statistical (PCA) and machine learning (Random Forest and SHAP)
perspectives.
"""

from pathlib import Path
import pandas as pd


def create_importance_table(
    rf_model,
    shap_df,
    pca_weights,
):
    importance = pd.DataFrame({
        "Feature": rf_model.feature_names_in_,
        "RF_Importance": rf_model.feature_importances_,
        "MeanAbsSHAP": shap_df.abs().mean(),
    })

    importance = importance.merge(
        pca_weights,
        on="Feature",
        how="left",
    )

    return importance.sort_values(
        "RF_Importance",
        ascending=False,
    )


def create_country_feature_attribution(shap_df):
    """
    Compute country-level SHAP feature attributions with relative SHAP importance %.
    Matches notebook Section 5.
    """
    shap_long = (
        shap_df
        .reset_index()
        .rename(columns={"country": "Country", "year": "Year"})
        .melt(
            id_vars=["Country", "Year"],
            var_name="Feature",
            value_name="SHAP Value"
        )
    )

    shap_long["SHAP Importance (%)"] = (
        shap_long.groupby(["Country", "Year"])["SHAP Value"]
        .transform(lambda x: x.abs() / x.abs().sum() * 100)
    )

    return shap_long.sort_values(
        ["Country", "Year", "SHAP Importance (%)"],
        ascending=[True, True, False]
    ).reset_index(drop=True)


def create_global_feature_importance(rf_model, shap_df):
    """
    Compute global feature importance combining RF importance % and SHAP importance %.
    Matches notebook Section 5.
    """
    mean_abs_shap = shap_df.abs().mean(axis=0)

    global_importance = pd.DataFrame({
        "Feature": rf_model.feature_names_in_,
        "RF Importance": rf_model.feature_importances_ * 100,
        "Mean |SHAP|": mean_abs_shap.values,
    })

    global_importance["SHAP Importance (%)"] = (
        global_importance["Mean |SHAP|"]
        / global_importance["Mean |SHAP|"].sum()
        * 100
    )

    return global_importance.sort_values(
        "SHAP Importance (%)",
        ascending=False
    ).reset_index(drop=True)


def create_feature_dependence(X, country_attribution, raw_df_path="data/outputs/all_indicator_values.csv"):
    """
    Combine feature values, SHAP attributions, and composite metrics for dependence analysis.
    Matches notebook Section 6.
    """
    X_long = X.reset_index().rename(columns={"country": "Country", "year": "Year"})
    feature_values = X_long.melt(
        id_vars=["Country", "Year"],
        var_name="Feature",
        value_name="Feature Value"
    )

    feature_dependence = feature_values.merge(
        country_attribution,
        on=["Country", "Year", "Feature"],
        how="left"
    )

    raw_path = Path(raw_df_path)
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        metrics = raw_df.rename(columns={
            "country": "Country",
            "year": "Year",
            "indicator": "Indicator",
            "value": "Value"
        })
        metrics = (
            metrics[metrics["Indicator"].isin(["Readiness", "Vulnerability", "CompositeRisk"])]
            .pivot(
                index=["Country", "Year"],
                columns="Indicator",
                values="Value"
            )
            .reset_index()
        )
        feature_dependence = feature_dependence.merge(
            metrics[["Country", "Year", "Readiness", "Vulnerability", "CompositeRisk"]],
            on=["Country", "Year"],
            how="left"
        )

    return feature_dependence.sort_values(
        ["Country", "Year", "SHAP Importance (%)"],
        ascending=[True, True, False]
    ).reset_index(drop=True)
