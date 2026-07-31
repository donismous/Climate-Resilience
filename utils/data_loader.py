"""
Data Loading and Preprocessing Utilities.

This module provides reusable functions for loading and preparing the modelling
dataset used in the Random Forest feature attribution workflow.

The preprocessing steps include:
    - extracting the Composite Risk target,
    - reshaping indicator values into feature columns,
    - reversing readiness component scales where required,
    - removing composite indicators that would introduce target leakage,
    - returning predictor and target datasets ready for modelling.

Separating data preparation from model training improves readability,
maintainability, and reuse across multiple analysis pipelines.
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

    print("NEW load_data() is running")


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

    features[["Economic", "Governance", "Social"]] = (
        1 - features[["Economic", "Governance", "Social"]]
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
