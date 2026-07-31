"""
Random Forest Model.

This module contains functions for training and evaluating the Random Forest
regression model used to predict the Composite Risk score.

The trained model provides:
    - predictions of Composite Risk,
    - Random Forest feature importance scores,
    - the fitted estimator required for SHAP analysis.

The functions are intentionally independent from data loading and output saving
to encourage modularity and code reuse.
"""

from sklearn.ensemble import RandomForestRegressor


def train_random_forest(
    X,
    y,
    n_estimators=500,
    max_depth=None,
    random_state=42,
):
    """
    Train a Random Forest regression model.

    Args:
        X (pd.DataFrame):
            Predictor variables.

        y (pd.Series):
            Target variable (Composite Risk).

    Returns:
        RandomForestRegressor:
            Trained Random Forest model.
    """

    print(f"X shape: {X.shape}")


    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

    rf.fit(X, y)

    print(f"Number of trees: {len(rf.estimators_)}")

    return rf
