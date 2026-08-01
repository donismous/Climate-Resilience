"""
Data Loading Utilities.

This module provides reusable functions for loading and preparing the modelling
dataset used in the Random Forest feature attribution workflow.

Note: Readiness indicators (Economic, Governance, Social) are centrally reversed
(1 - value) in the preprocessing pipeline (utils/preprocessing/prepare.py) so that
higher values consistently represent higher risk contribution.
"""

import pandas as pd


def load_data(filepath="data/outputs/all_indicator_values.csv"):
    """
    Load dataset and split into features and target.

    Returns
    -------
    X : pd.DataFrame
        Predictor variables.
    y : pd.Series
        Composite Risk score.
    """

    df = pd.read_csv(filepath)

    target = (
        df.query("indicator == 'CompositeRisk'")
          .rename(columns={"value": "CompositeRisk"})
          [["country", "year", "CompositeRisk"]]
    )

    features = (
        df.query("indicator != 'CompositeRisk'")
          .pivot(
              index=["country", "year"],
              columns="indicator",
              values="value",
          )
          .reset_index()
    )

    dataset = (
        features
        .merge(target, on=["country", "year"])
        .set_index(["country", "year"])
        .drop(columns=["Vulnerability", "Readiness"])
    )

    X = dataset.drop(columns="CompositeRisk")
    y = dataset["CompositeRisk"]

    return X, y, dataset
