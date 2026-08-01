"""Enhanced ETS + ARIMA Ensemble Time Series Forecasting Model on ND-GAIN indicators.

Enhancements:
1. Logit transformation logit(y) = log(y / (1 - y)) applied prior to model fitting;
   sigmoid back-transformation applied to forecasts to strictly bound predictions in [0, 1].
2. Candidate model selection using small-sample corrected AIC (AICc) across:
   - Simple Exponential Smoothing (SES, no trend)
   - Damped Holt's Linear Trend
   - Undamped Holt's Linear Trend
3. Ensemble forecasting combining ETS and ARIMA predictions via 50/50 averaging.

Usage:
    python model/prediction/ES_all_indicators.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# Allow running from the repo root or nested folders.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.indicators import indicator_map
from model.old_models.arima import prepare_series


def calculate_aicc(aic: float, k: int, n: int) -> float:
    """Calculate small-sample corrected AIC (AICc)."""
    if n - k - 1 <= 0:
        return aic
    return aic + (2 * k ** 2 + 2 * k) / (n - k - 1)


def logit_transform(series: pd.Series, eps: float = 1e-5) -> pd.Series:
    """Transform series to log-odds scale logit(y)."""
    clipped = np.clip(series.astype(float).values, eps, 1.0 - eps)
    return pd.Series(logit(clipped), index=series.index, name=series.name)


def sigmoid_transform(values: np.ndarray) -> np.ndarray:
    """Transform logit values back to original [0, 1] probability scale."""
    return expit(values)


def fit_ets_candidate(series_logit: pd.Series):
    """Fit ETS candidates (SES, Damped Trend, Undamped Trend) and return best by AICc."""
    n = len(series_logit)
    if n < 4:
        raise ValueError(f"Need at least 4 observations, got {n}.")

    candidates = []
    # Candidate 1: Simple Exponential Smoothing (no trend)
    try:
        m1 = ExponentialSmoothing(
            series_logit, trend=None, initialization_method="estimated"
        ).fit()
        k1 = 2
        aicc1 = calculate_aicc(m1.aic, k1, n)
        candidates.append((aicc1, m1))
    except Exception:
        pass

    # Candidate 2: Damped Holt's Linear Trend
    try:
        m2 = ExponentialSmoothing(
            series_logit, trend="add", damped_trend=True, initialization_method="estimated"
        ).fit()
        k2 = 5
        aicc2 = calculate_aicc(m2.aic, k2, n)
        candidates.append((aicc2, m2))
    except Exception:
        pass

    # Candidate 3: Undamped Holt's Linear Trend
    try:
        m3 = ExponentialSmoothing(
            series_logit, trend="add", damped_trend=False, initialization_method="estimated"
        ).fit()
        k3 = 4
        aicc3 = calculate_aicc(m3.aic, k3, n)
        candidates.append((aicc3, m3))
    except Exception:
        pass

    if not candidates:
        raise RuntimeError("All ETS candidate models failed to fit.")

    return min(candidates, key=lambda item: item[0])[1]


def fit_arima_candidate(series_logit: pd.Series):
    """Fit ARIMA candidate models on standardized logit series and return best by AICc."""
    n = len(series_logit)
    mean, scale = series_logit.mean(), series_logit.std()
    if scale == 0 or np.isnan(scale):
        scale = 1.0
    std_series = (series_logit - mean) / scale

    orders = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0)]
    fitted_models = []
    for order in orders:
        try:
            m = ARIMA(std_series, order=order, trend="c").fit()
            k = sum(order) + 2
            aicc = calculate_aicc(m.aic, k, n)
            fitted_models.append((aicc, m, mean, scale))
        except Exception:
            pass

    if not fitted_models:
        raise RuntimeError("ARIMA model fit failed.")

    best_aicc, best_m, m_mean, m_scale = min(fitted_models, key=lambda item: item[0])
    return best_m, m_mean, m_scale


def forecast_ensemble(series: pd.Series, steps: int = 15) -> np.ndarray:
    """Forecast future values using an ensemble of ETS and ARIMA on logit-transformed series."""
    logit_series = logit_transform(series)

    # 1. ETS Forecast
    ets_pred = None
    try:
        ets_model = fit_ets_candidate(logit_series)
        ets_pred_logit = ets_model.forecast(steps=steps)
        ets_pred = sigmoid_transform(np.asarray(ets_pred_logit))
    except Exception:
        pass

    # 2. ARIMA Forecast
    arima_pred = None
    try:
        arima_model, mean, scale = fit_arima_candidate(logit_series)
        arima_fc_std = arima_model.get_forecast(steps=steps).predicted_mean
        arima_pred_logit = np.asarray(arima_fc_std) * scale + mean
        arima_pred = sigmoid_transform(arima_pred_logit)
    except Exception:
        pass

    # 3. Combine Predictions
    if ets_pred is not None and arima_pred is not None:
        ensemble_pred = 0.5 * ets_pred + 0.5 * arima_pred
    elif ets_pred is not None:
        ensemble_pred = ets_pred
    elif arima_pred is not None:
        ensemble_pred = arima_pred
    else:
        # Fallback: Flat persistence of last observation
        ensemble_pred = np.repeat(series.iloc[-1], steps)

    return np.clip(ensemble_pred, 0.0, 1.0)


def extend_with_forecast(
    df: pd.DataFrame,
    indicators: list = None,
    countries: list = None,
    end_year: int = 2040,
) -> pd.DataFrame:
    """Concatenate actual indicator values with ETS+ARIMA ensemble forecasts up to end_year."""
    if indicators is None:
        indicators = [name for name in indicator_map if name in df.columns]
    if countries is None:
        countries = sorted(df["Country"].unique())

    frames = []
    for country in countries:
        for name in indicators:
            try:
                series = prepare_series(df, country, name)
            except ValueError:
                continue

            frames.append(
                pd.DataFrame(
                    {
                        "country": country,
                        "indicator": name,
                        "year": series.index.year,
                        "value": series.values,
                        "source": "actual",
                    }
                )
            )

            steps = end_year - series.index[-1].year
            if steps <= 0:
                continue
            if len(series) < 4:
                print(f"Skipping forecast for {country}/{name}: fewer than 4 observations.")
                continue

            try:
                predicted_values = forecast_ensemble(series, steps=steps)
            except Exception as error:
                print(f"Skipping forecast for {country}/{name}: {error}")
                continue

            forecast_years = range(series.index[-1].year + 1, end_year + 1)
            frames.append(
                pd.DataFrame(
                    {
                        "country": country,
                        "indicator": name,
                        "year": list(forecast_years),
                        "value": np.round(predicted_values, 6),
                        "source": "forecast",
                    }
                )
            )

    forecast_df = pd.concat(frames, ignore_index=True)
    return forecast_df.sort_values(["country", "indicator", "year"]).reset_index(drop=True)


if __name__ == "__main__":
    processed_path = ROOT / "data" / "data_preprocessed" / "processed_data.csv"
    if not processed_path.exists():
        from utils.preprocessing.pipeline import preprocess
        raw_path = ROOT / "data" / "inputs" / "fetch" / "ndgain_raw.csv"
        df_original = preprocess(pd.read_csv(raw_path))
    else:
        df_original = pd.read_csv(processed_path)

    df = df_original.drop(
        columns=["Exposure", "Readiness", "Vulnerability"], errors="ignore"
    )

    intermediate_dir = ROOT / "data" / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    df_original.reset_index().to_parquet(
        intermediate_dir / "df_original.parquet",
        index=False,
    )

    forecast_df = extend_with_forecast(df, end_year=2040)
    print("Actuals + ETS/ARIMA ensemble forecasts to 2040 (all countries, all indicators):")
    print(forecast_df.tail(12))

    forecast_df.to_parquet(
        intermediate_dir / "forecast.parquet",
        index=False,
    )


