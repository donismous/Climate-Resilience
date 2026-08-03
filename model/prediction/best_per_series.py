"""Per-series champion model on every ND-GAIN indicator.

Instead of using one model family everywhere (like ``ETS_all_indicators``) or
blending two (like ``Ensemble_all_indicators``), this module picks, for every
(country, indicator) pair, whichever of the three candidate models won the
5-year-holdout backtest for that pair, and fits that winner on the pair's
full history:

- ``ets``: Holt's linear trend (``model.old_models.exp_smoothing.fit_ets``)
- ``arima``: per-series ARIMA, order from the series' own PACF/ACF
  (``model.old_models.arima``)
- ``arima_shared_order``: ARIMA with one order per indicator, chosen by
  average AIC across countries during the backtest
  (``model.old_models.second_arima``'s methodology)

The winner table comes from ``notebooks/eval_utils.run_backtests`` and is
cached in ``data/intermediate/best_model_selection.csv`` (delete that file to
force a re-run of the ~20 minute backtest). Pairs without a backtest result
fall back to ETS, the best model overall.

Note on methodology: the champion is picked on the same 2019-2023 holdout the
models were scored on, so the selection's own backtest MAE is an optimistic
estimate of future accuracy. Judging the champion strategy fairly would need
an earlier, second holdout window.

Usage:
    python model/prediction/best_per_series.py

    Or from a notebook:

        from model.prediction.best_per_series import build_selection, extend_with_forecast

        selection = build_selection(df)          # or load the cached CSV
        forecast_df = extend_with_forecast(df, selection, end_year=2040)
"""

import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Allow running from the repo root or nested folders.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.indicators import indicator_map
from model.old_models.arima import prepare_series, select_order, fit_arima
from model.old_models.arima import forecast as arima_forecast
from model.old_models.exp_smoothing import fit_ets

SELECTION_PATH = ROOT / "data" / "intermediate" / "best_model_selection.csv"
FALLBACK_MODEL = "ets"  # best overall in the backtest; used when a pair has no result


def build_selection(df: pd.DataFrame, test_years: int = 5) -> pd.DataFrame:
    """Backtest the three candidate models and pick a winner per pair.

    Runs ``notebooks.eval_utils.run_backtests`` (three models, last
    ``test_years`` years held out per country) and keeps, for every
    (country, indicator), the model with the lowest holdout MAE.

    Args:
        df: Wide processed dataset (one row per (Country, Year), one column
            per indicator).
        test_years: Holdout length used to score the candidates.

    Returns:
        DataFrame with columns ``country``, ``indicator``, ``model``,
        ``mae`` (the winner's holdout MAE) and ``shared_order`` (the
        indicator's (p, d, q) as a string, filled only on rows won by
        ``arima_shared_order``).
    """
    sys.path.insert(0, str(ROOT / "notebooks"))
    import eval_utils

    results, _, info = eval_utils.run_backtests(df, test_years=test_years)
    valid = results[results["error"].isna()]

    winners = (
        valid.loc[valid.groupby(["country", "indicator"])["mae"].idxmin()]
        [["country", "indicator", "model", "mae"]]
        .reset_index(drop=True)
    )
    winners["shared_order"] = winners.apply(
        lambda row: str(info["shared_orders"][row["indicator"]])
        if row["model"] == "arima_shared_order"
        else "",
        axis=1,
    )
    return winners


def load_selection(path: Path = SELECTION_PATH) -> pd.DataFrame:
    """Load a cached selection table written by ``main``."""
    return pd.read_csv(path, keep_default_na=False)


def _forecast_winner(series: pd.Series, model: str, shared_order: str, steps: int):
    """Fit one series' winning model on its full history and forecast.

    Args:
        series: Yearly series from ``prepare_series`` (PeriodIndex).
        model: One of ``ets``, ``arima``, ``arima_shared_order``.
        shared_order: The indicator's (p, d, q) string, used only by
            ``arima_shared_order``.
        steps: Number of years to forecast.

    Returns:
        Array-like of ``steps`` point forecasts.
    """
    if model == "ets":
        return fit_ets(series).forecast(steps).values
    if model == "arima":
        fitted = fit_arima(series, order=select_order(series))
        return arima_forecast(fitted, steps=steps)["forecast"].values
    if model == "arima_shared_order":
        # second_arima's methodology: plain ARIMA on the raw series with the
        # indicator-wide order (no standardization).
        order = ast.literal_eval(shared_order)
        raw = pd.Series(series.values, index=series.index.year)
        fitted = ARIMA(raw, order=order).fit()
        return fitted.get_forecast(steps=steps).predicted_mean.values
    raise ValueError(f"Unknown model {model!r}.")


def extend_with_forecast(
    df: pd.DataFrame,
    selection: pd.DataFrame,
    indicators: list = None,
    countries: list = None,
    end_year: int = 2040,
) -> pd.DataFrame:
    """Concatenate actual values with each pair's champion-model forecast.

    For every (country, indicator) pair, looks up the backtest winner in
    ``selection``, fits it on the pair's full history and appends forecasts
    through ``end_year``. Pairs missing from ``selection`` use
    ``FALLBACK_MODEL``; a pair whose winner fails to fit falls back to ETS,
    then to persistence of the last observation.

    Args:
        df: Wide processed dataset.
        selection: Winner table from ``build_selection``/``load_selection``.
        indicators: Optional indicator subset. Defaults to
            ``config.indicators.indicator_map`` columns present in ``df``.
        countries: Optional ISO3 subset. Defaults to every country in ``df``.
        end_year: Last year to forecast (inclusive).

    Returns:
        A DataFrame with columns ``country``, ``indicator``, ``year``,
        ``value``, ``source`` ("actual" or "forecast") and ``model`` (which
        model produced the row; "actual" rows carry the winner too), sorted
        by country, indicator, then year.
    """
    if indicators is None:
        indicators = [name for name in indicator_map if name in df.columns]
    if countries is None:
        countries = sorted(df["Country"].unique())

    chosen = {
        (row.country, row.indicator): (row.model, row.shared_order)
        for row in selection.itertuples()
    }

    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for country in countries:
            for name in indicators:
                try:
                    series = prepare_series(df, country, name)
                except ValueError:
                    continue

                model, shared_order = chosen.get((country, name), (FALLBACK_MODEL, ""))

                frames.append(
                    pd.DataFrame(
                        {
                            "country": country,
                            "indicator": name,
                            "year": series.index.year,
                            "value": series.values,
                            "source": "actual",
                            "model": model,
                        }
                    )
                )

                steps = end_year - series.index[-1].year
                if steps <= 0:
                    continue
                if len(series) < 4:
                    print(f"Skipping forecast for {country}/{name}: fewer than 4 observations.")
                    continue

                used = model
                try:
                    predicted = _forecast_winner(series, model, shared_order, steps)
                except Exception:
                    try:
                        used = "ets (fallback)"
                        predicted = fit_ets(series).forecast(steps).values
                    except Exception as error:
                        print(f"{country}/{name}: {model} and ETS fallback failed "
                              f"({error}); using persistence.")
                        used = "persistence (fallback)"
                        predicted = [series.iloc[-1]] * steps

                forecast_years = range(series.index[-1].year + 1, end_year + 1)
                frames.append(
                    pd.DataFrame(
                        {
                            "country": country,
                            "indicator": name,
                            "year": list(forecast_years),
                            # Indicators live on a [0, 1] scale; a linear
                            # trend extrapolated to 2040 can drift outside it.
                            "value": np.clip(pd.Series(predicted, dtype=float), 0.0, 1.0).round(6),
                            "source": "forecast",
                            "model": used,
                        }
                    )
                )

    forecast_df = pd.concat(frames, ignore_index=True)
    return forecast_df.sort_values(["country", "indicator", "year"]).reset_index(drop=True)


def main():
    df_original = pd.read_csv(ROOT / "data" / "data_preprocessed" / "processed_data.csv")
    df = df_original.drop(
        columns=["Exposure", "Readiness", "Vulnerability"], errors="ignore"
    )
    # Exposure is constant over time in ND-GAIN, so it is not forecast by the
    # shared-order model; keep it for the per-series candidates' selection.
    df["Exposure"] = df_original["Exposure"]

    if SELECTION_PATH.exists():
        print(f"Loading cached model selection from {SELECTION_PATH}")
        selection = load_selection()
    else:
        print("No cached selection found - running the three-model backtest "
              "(~20 minutes)...")
        selection = build_selection(df)
        SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        selection.to_csv(SELECTION_PATH, index=False)
        print(f"Saved selection to {SELECTION_PATH}")

    print("Winners per model:")
    print(selection["model"].value_counts().to_string())

    forecast_df = extend_with_forecast(df, selection, end_year=2040)
    print("Actuals + champion-model forecasts to 2040:")
    print(forecast_df.tail(12))

    output_path = ROOT / "data" / "intermediate" / "forecast_best_model.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_parquet(output_path, index=False)
    print(f"Saved {len(forecast_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
