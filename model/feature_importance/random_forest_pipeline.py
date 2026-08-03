"""
Random Forest Feature Attribution Pipeline.

This module orchestrates the complete feature attribution workflow for the
Climate Resilience Dashboard.

The pipeline performs the following steps:
    1. Load and preprocess the modelling dataset.
    2. Train a Random Forest regression model to predict Composite Risk.
    3. Compute SHAP values to explain individual feature contributions.
    4. Compare Random Forest feature importance with PCA-derived feature weights.
    5. Save feature importance tables and SHAP attribution outputs.

The pipeline serves as the main entry point for the Random Forest analysis and
can be executed directly to reproduce all feature attribution results.
"""

from pathlib import Path
import pandas as pd

from utils.data_loader import load_data
from model.feature_importance.random_forest import train_random_forest
from model.feature_importance.shap_analysis import calculate_shap_values
from model.feature_importance.feature_importance import (
    create_importance_table,
    create_country_feature_attribution,
    create_global_feature_importance,
    create_feature_dependence,
)


OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROOT_OUTPUT_DIR = Path("data/outputs")
ROOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading data...")
    X, y, df = load_data()

    print("Training Random Forest...")
    rf = train_random_forest(X, y)

    print("Calculating SHAP values...")
    shap_df = calculate_shap_values(rf, X)

    print("Creating SHAP attribution table...")
    shap_long = (
        shap_df
        .reset_index()
        .rename(columns={"country": "Country", "year": "Year"})
        .melt(
            id_vars=["Country", "Year"],
            var_name="Feature",
            value_name="SHAP_Value"
        )
    )

    print("Creating country feature attribution table...")
    country_attribution = create_country_feature_attribution(shap_df)

    print("Creating global feature importance table...")
    global_importance = create_global_feature_importance(rf, shap_df)

    print("Creating feature dependence table...")
    feature_dependence = create_feature_dependence(X, country_attribution)

    print("Loading PCA weights...")
    pca_weights_path = Path("config/risk_score_weights.csv")
    if not pca_weights_path.exists():
        pca_weights_path = OUTPUT_DIR / "risk_score_weights.csv"

    pca_weights = (
        pd.read_csv(pca_weights_path)
        .rename(columns={"Unnamed: 0": "Feature", "PC1": "PCA_Weight"})
    )

    print("Creating feature importance table...")
    importance = create_importance_table(
        rf,
        shap_df,
        pca_weights,
    )

    print("Saving outputs...")
    # Save to OUTPUT_DIR (data/outputs)
    country_attribution.to_csv(OUTPUT_DIR / "country_feature_attribution.csv", index=False)
    global_importance.to_csv(OUTPUT_DIR / "global_feature_importance.csv", index=False)
    feature_dependence.to_csv(OUTPUT_DIR / "feature_dependence.csv", index=False)

    print("Done!")



if __name__ == "__main__":
    main()
