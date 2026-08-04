"""Shared backtest machinery for the model evaluation notebooks.

Evaluates the three forecasting models on a common train/test split of the
wide processed dataset (one row per (Country, Year), one column per
indicator):

- ``arima``: per-series ARIMA, order chosen per (country, indicator) from
  its own PACF/ACF (``model.old_models.arima``). Used by
  ``model/old_models/arima_all_indicators.py``.
- ``ets``: Holt's linear trend, damped vs non-damped chosen by AIC
  (``model.old_models.exp_smoothing``). Used by
  ``model/prediction/ETS_all_indicators.py``.
- ``arima_shared_order``: one ARIMA order per indicator (lowest average AIC
  across countries), reused for every country
  (``model.old_models.second_arima.second_arima``). Does not model
  Exposure, so that indicator has no rows for this model.

The last ``TEST_YEARS`` years of each country's series are held out and
each model forecasts them from the preceding history. Both the error
metrics and the raw year-by-year predictions are returned, so notebooks can
build MAE tables and forecast plots from a single backtest run.
"""

import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.old_models.arima import prepare_series, select_order, fit_arima
from model.old_models.arima import forecast as arima_forecast
from model.old_models.exp_smoothing import fit_ets
from model.old_models.second_arima.second_arima import ARIMACountryModel

TEST_YEARS = 5
MIN_TRAIN_POINTS = 10

INDICATORS = [
    "Capacity", "Economic", "Ecosystems", "Exposure", "Food", "Governance",
    "Habitat", "Health", "Infrastructure", "Sensitivity", "Social", "Water",
]

# second_arima's methodology excludes Exposure (alongside the derived
# Readiness/Vulnerability columns, which are not in INDICATORS anyway).
SHARED_ORDER_EXCLUDED = ["Exposure"]

MODEL_LABELS = {
    "arima": "ARIMA (per-series order)",
    "ets": "ETS (Holt)",
    "arima_shared_order": "ARIMA (shared order)",
}


def metrics(actual, predicted):
    """MAE and RMSE of predictions against actuals."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    return {"mae": np.mean(np.abs(errors)), "rmse": np.sqrt(np.mean(errors ** 2))}


def backtest_per_series(df, indicators=None, test_years=TEST_YEARS, progress=True):
    """Backtest the per-series ARIMA and ETS models.

    For every (country, indicator) pair with enough history, holds out the
    last ``test_years`` observations, fits each model on the rest and
    forecasts the held-out window.

    Returns:
        (results, preds): ``results`` has one row per (country, indicator,
        model) with ``mae``/``rmse`` (or an ``error`` message on failure);
        ``preds`` has one row per forecast year with the predicted and
        actual value.
    """
    if indicators is None:
        indicators = INDICATORS
    result_rows, pred_rows = [], []
    countries = sorted(df["Country"].unique())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, country in enumerate(countries):
            if progress and i % 10 == 0:
                print(f"Per-series backtest {i + 1}/{len(countries)}: {country}",
                      file=sys.__stdout__)
            for indicator in indicators:
                try:
                    series = prepare_series(df, country, indicator)
                except ValueError:
                    continue
                if len(series) < MIN_TRAIN_POINTS + test_years:
                    continue

                train, test = series.iloc[:-test_years], series.iloc[-test_years:]
                test_year_values = test.index.year

                fits = {
                    "arima": lambda: arima_forecast(
                        fit_arima(train, order=select_order(train)), steps=test_years
                    )["forecast"].values,
                    # Point forecast only: ets_forecast additionally simulates
                    # 1000 paths for confidence bounds the backtest never uses,
                    # which dominates the runtime.
                    "ets": lambda: fit_ets(train).forecast(test_years).values,
                }
                for model_name, fit in fits.items():
                    try:
                        predicted = fit()
                        result_rows.append({
                            "country": country, "indicator": indicator,
                            "model": model_name, **metrics(test.values, predicted),
                            "error": None,
                        })
                        for year, pred, actual in zip(test_year_values, predicted, test.values):
                            pred_rows.append({
                                "country": country, "indicator": indicator,
                                "model": model_name, "year": int(year),
                                "predicted": float(pred), "actual": float(actual),
                            })
                    except Exception as exc:
                        result_rows.append({
                            "country": country, "indicator": indicator,
                            "model": model_name, "mae": None, "rmse": None,
                            "error": str(exc),
                        })

    return pd.DataFrame(result_rows), pd.DataFrame(pred_rows)


def backtest_shared_order(df, indicators=None, test_years=TEST_YEARS, progress=True):
    """Backtest the shared-order-per-indicator ARIMA (second_arima).

    Splits off the last ``test_years`` years of each country, selects one
    order per indicator on the training window (lowest average AIC across
    countries), fits every (country, indicator) with it and forecasts the
    held-out window. The model's own console output is captured.

    Returns:
        (results, preds, best_orders): same shapes as
        ``backtest_per_series`` plus the selected order per indicator.
    """
    if indicators is None:
        indicators = [c for c in INDICATORS if c not in SHARED_ORDER_EXCLUDED]

    wide = df.set_index(["Country", "Year"]).sort_index()
    last_years = wide.reset_index().groupby("Country")["Year"].transform("max").values
    years = wide.index.get_level_values("Year")
    train_mask = years <= (last_years - test_years)

    df_train = wide[train_mask]
    df_test = wide[~train_mask]
    countries = df_train.index.get_level_values("Country").unique()

    if progress:
        print(f"Shared-order backtest: selecting orders and fitting "
              f"{len(countries)} countries x {len(indicators)} indicators...",
              file=sys.__stdout__)

    model = ARIMACountryModel()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        model.fit(df_train, list(countries), list(indicators))
        forecasts = model.forecast(years=test_years)

    result_rows, pred_rows = [], []
    for country in forecasts:
        for indicator, predicted in forecasts[country].items():
            try:
                actual = df_test.loc[country, indicator].sort_index()
                n = min(len(actual), len(predicted))
                if n == 0:
                    continue
                predicted_values = np.asarray(predicted)[:n]
                result_rows.append({
                    "country": country, "indicator": indicator,
                    "model": "arima_shared_order",
                    **metrics(actual.values[:n], predicted_values),
                    "error": None,
                })
                for year, pred, act in zip(actual.index[:n], predicted_values,
                                           actual.values[:n]):
                    pred_rows.append({
                        "country": country, "indicator": indicator,
                        "model": "arima_shared_order", "year": int(year),
                        "predicted": float(pred), "actual": float(act),
                    })
            except Exception as exc:
                result_rows.append({
                    "country": country, "indicator": indicator,
                    "model": "arima_shared_order", "mae": None, "rmse": None,
                    "error": str(exc),
                })

    return pd.DataFrame(result_rows), pd.DataFrame(pred_rows), dict(model.best_orders)


def run_backtests(df, test_years=TEST_YEARS, progress=True):
    """Run all three models on one dataset and combine the outputs.

    Returns:
        (results, preds, info): concatenated metrics and predictions for
        the three models, plus a dict with the shared-order selection and
        the ARIMA convergence-fallback count.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        res_ps, preds_ps = backtest_per_series(df, test_years=test_years,
                                               progress=progress)
        res_so, preds_so, best_orders = backtest_shared_order(
            df, test_years=test_years, progress=progress
        )

    results = pd.concat([res_ps, res_so], ignore_index=True)
    preds = pd.concat([preds_ps, preds_so], ignore_index=True)
    info = {
        "shared_orders": best_orders,
        "arima_convergence_fallbacks": buffer.getvalue().count("did not converge"),
    }
    return results, preds, info
