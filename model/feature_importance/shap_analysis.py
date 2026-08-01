"""
SHAP Feature Attribution.

This module calculates SHAP (SHapley Additive exPlanations) values for the
trained Random Forest model.

SHAP values quantify the contribution of each predictor to the predicted
Composite Risk score for every country-year observation, enabling local and
global model interpretability.

The resulting SHAP values are used for feature attribution, comparison with
PCA-derived weights, and visualization.
"""

import os
# Prevent OpenMP thread deadlocks on macOS in C++ extensions (shap._cext)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import shap
import pandas as pd


def calculate_shap_values(model, X, check_additivity=False, approximate=True):
    """
    Calculate SHAP values for a trained tree-based model.

    Args:
        model: Trained tree-based model (e.g. RandomForestRegressor).
        X (pd.DataFrame): Predictor variables matrix.
        check_additivity (bool): Whether to check additivity of SHAP values.
            Defaults to False to prevent numerical precision stalls with Random Forest ensembles.
        approximate (bool): Whether to use fast tree-path SHAP approximation.
            Defaults to True for sub-second calculation speed and robust C-extension performance.

    Returns:
        pd.DataFrame: Matrix of SHAP values matching X's shape and indices.
    """
    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(
        X,
        check_additivity=check_additivity,
        approximate=approximate,
    )

    print("Creating DataFrame...")
    shap_df = pd.DataFrame(
        shap_values,
        columns=X.columns,
        index=X.index,
    )

    return shap_df
