"""Shared ARIMA machinery for the ND-GAIN forecasting models.

Holds the series preparation, order selection, fitting and forecasting
functions used by both ``basic_arima_model`` (composite risk score) and
``arima_all_indicators`` (every indicator). All of them operate on the wide
composite risk dataset from ``model.composite_risk_score.get_composite_risk``:
one row per (country, year), one column per indicator plus ``risk_score``.

The series are assumed stationary, so the differencing order is fixed at
d = 0, and the AR / MA orders (p, q) are read from the sample PACF and ACF
respectively: p is the number of leading consecutive PACF lags outside the
95% confidence band, and q the same count on the ACF.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf


def prepare_series(df: pd.DataFrame, country: str, indicator: str) -> pd.Series:
    """Extract one country's yearly series for one indicator column.

    Args:
        df: Composite risk DataFrame from ``get_composite_risk`` with one
            row per (country, year) and one column per indicator.
        country: ISO3 country code to select, e.g. "FRA".
        indicator: Column to extract, e.g. "Vulnerability" or "risk_score".

    Returns:
        A Series of indicator values indexed by year (yearly PeriodIndex,
        sorted), so ARIMA forecasts carry proper date labels.

    Raises:
        ValueError: If the (country, indicator) pair has no data.
    """
    sub = df[df["Country"] == country].dropna(subset=[indicator])
    if sub.empty:
        raise ValueError(f"No data for country {country!r}, indicator {indicator!r}.")

    series = (
        sub.assign(year=pd.PeriodIndex(sub["Year"].astype(str), freq="Y"))
        .set_index("year")[indicator]
        .sort_index()
    )
    series.name = f"{country}_{indicator}"
    return series


def select_order(series: pd.Series, max_lag: int = 5) -> tuple:
    """Choose the ARIMA (p, d, q) order from the sample PACF and ACF.

    Standard Box-Jenkins identification: an AR(p) process shows a PACF that
    cuts off after lag p while its ACF tails off slowly, and an MA(q)
    process an ACF that cuts off after lag q while its PACF tails off. So p
    (resp. q) is the number of leading consecutive PACF (resp. ACF) lags
    outside the 95% confidence band ±1.96/sqrt(n) — but if a function never
    cuts off within the inspected window, it is read as tailing off and its
    order is set to 0 rather than to the window length (an ACF that decays
    slowly signals AR behaviour, not a high-order MA). d is always 0 since
    the series is assumed stationary.

    Args:
        series: Stationary yearly time series.
        max_lag: Largest lag to inspect; also caps the resulting p and q.
            Kept small by default because the ND-GAIN series are short
            (~28 yearly points) and very persistent, so an uncapped ACF
            count overestimates q and the fit fails to converge.
            Effectively bounded by the PACF's sample-size limit (n // 2 - 1).

    Returns:
        The (p, 0, q) order tuple.
    """
    n = len(series)
    nlags = min(max_lag, n // 2 - 1)
    threshold = 1.96 / np.sqrt(n)

    def leading_significant(values: np.ndarray) -> int:
        # values[0] is lag 0 (always 1); count from lag 1 until the first
        # lag inside the confidence band.
        count = 0
        for value in values[1:]:
            if abs(value) <= threshold:
                break
            count += 1
        return count

    p = leading_significant(pacf(series, nlags=nlags))
    q = leading_significant(acf(series, nlags=nlags))

    # A count equal to the window means the function never cut off — it
    # tails off, so that side contributes no order of its own.
    if p == nlags and q == nlags:
        return (1, 0, 1)  # both tail off: mixed ARMA signature
    if q == nlags:
        q = 0  # ACF tails off: AR(p) signature
    elif p == nlags:
        p = 0  # PACF tails off: MA(q) signature
    return (p, 0, q)


def fit_arima(series: pd.Series, order: tuple = None):
    """Fit a non-seasonal ARIMA on a stationary series.

    Args:
        series: Yearly time series from ``prepare_series``.
        order: Optional (p, d, q) override. If omitted, the order is chosen
            with ``select_order``.

    Returns:
        A fitted ``ARIMAResults`` object. The fit runs on the standardized
        series (the indicator values have such a small variance that the raw
        likelihood surface is too flat for the optimizer), so the summary's
        ``const`` and ``sigma2`` are in standardized units; the AR/MA
        coefficients are unaffected and ``forecast`` converts back to the
        original scale.
    """
    if order is None:
        order = select_order(series)

    mean, scale = series.mean(), series.std()
    if not scale > 0:
        scale = 1.0
    standardized = (series - mean) / scale
    result = ARIMA(standardized, order=order, trend="c").fit()

    # On short series the ACF/PACF heuristic can over-parameterize the MA
    # side and the likelihood optimization fails; fall back to the pure AR
    # part, which is parsimonious and reliably converges.
    if not result.mle_retvals.get("converged", True) and order[2] > 0:
        fallback = (max(order[0], 1), order[1], 0)
        print(f"ARIMA{order} did not converge, falling back to ARIMA{fallback}")
        result = ARIMA(standardized, order=fallback, trend="c").fit()

    result.series_mean = mean
    result.series_scale = scale
    return result


def forecast(result, steps: int = 15) -> pd.DataFrame:
    """Forecast future values with 95% confidence intervals.

    Args:
        result: Fitted ``ARIMAResults`` from ``fit_arima``.
        steps: Number of years to forecast ahead.

    Returns:
        A DataFrame indexed by forecast year with columns ``forecast``,
        ``lower`` and ``upper`` (95% confidence bounds), in the original
        indicator units.
    """
    prediction = result.get_forecast(steps=steps)
    conf_int = prediction.conf_int(alpha=0.05)
    # Undo the standardization applied in fit_arima.
    mean = getattr(result, "series_mean", 0.0)
    scale = getattr(result, "series_scale", 1.0)
    return pd.DataFrame(
        {
            "forecast": prediction.predicted_mean * scale + mean,
            "lower": conf_int.iloc[:, 0] * scale + mean,
            "upper": conf_int.iloc[:, 1] * scale + mean,
        }
    )
