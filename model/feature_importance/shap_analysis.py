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

import shap
print(shap.__version__)
import pandas as pd


def calculate_shap_values(model, X):

    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")
    explanation = explainer(X)

    print("Creating DataFrame...")
    shap_df = pd.DataFrame(
        explanation.values,
        columns=X.columns,
        index=X.index,
    )

    return shap_df
