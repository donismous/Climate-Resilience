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
