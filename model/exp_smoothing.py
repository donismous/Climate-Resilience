"""Exponential Smoothing (Holt's linear trend) model on the composite climate risk score.

Alternative to ``basic_arima_model.py``. Where the ARIMA baseline assumes the
risk_score series is stationary (d fixed at 0) and picks (p, q) from the
ACF/PACF, this module fits Holt's linear trend method, which explicitly
models a (possibly damped) trend. This matters if a country's risk score is
drifting over time rather than fluctuating around a fixed mean -- a case the
ARIMA baseline cannot capture by construction.

For each country, both a damped and non-damped trend are fit and the one
with the lower AIC is kept, since damping matters a lot on short (~28 point)
series: an undamped trend can extrapolate unrealistically far by the
forecast horizon.

Confidence intervals are not available in closed form for
``ExponentialSmoothing`` the way they are for ARIMA, so they are estimated
by simulating future paths from the fitted model's innovations
distribution and taking the 2.5/97.5 percentiles.

Usage:
    Run as a script to fit a model per country and save the actual scores
    concatenated with forecasts up to 2030:

        python model/ets_model.py

    Or import from another module / notebook:

        from fetch_ndgain import fetch_ndgain
        from model.composite_risk_score import compute_composite_risk
        from model.ets_model import extend_with_forecast

        risk = compute_composite_risk(fetch_ndgain())
        combined = extend_with_forecast(risk, end_year=2030)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

N_SIMULATIONS = 1000
RANDOM_SEED = 0


def prepare_series(risk_df: pd.DataFrame, country: str) -> pd.Series:
    """Extract one country's yearly risk score series.

    Identical to ``basic_arima_model.prepare_series`` -- duplicated here so
    this module has no import-time dependency on the ARIMA file.

    Args:
        risk_df: DataFrame from ``compute_composite_risk`` with ``country``,
            ``year`` and ``risk_score`` columns.
        country: ISO3 country code to select, e.g. "FRA".

    Returns:
        A Series of risk_score indexed by year (yearly PeriodIndex, sorted).

    Raises:
        ValueError: If the country has no data.
    """
    sub = risk_df[risk_df["country"] == country]
    if sub.empty:
        raise ValueError(f"No risk score data for country {country!r}.")

    series = (
        sub.assign(year=pd.PeriodIndex(sub["year"].astype(str), freq="Y"))
        .set_index("year")["risk_score"]
        .sort_index()
    )
    series.name = f"{country}_risk_score"
    return series


def fit_ets(series: pd.Series):
    """Fit Holt's linear trend method, choosing damped vs. non-damped by AIC.

    Args:
        series: Yearly time series from ``prepare_series``. Needs at least
            4 points for a trend fit to be meaningful.

    Returns:
        The fitted ``HoltWintersResults`` with the lower AIC of the damped
        and non-damped trend variants.

    Raises:
        ValueError: If the series has fewer than 4 points.
    """
    if len(series) < 4:
        raise ValueError(
            f"Need at least 4 points to fit a trend model, got {len(series)}."
        )

    candidates = []
    for damped in (False, True):
        model = ExponentialSmoothing(
            series.astype(float),
            trend="add",
            damped_trend=damped,
            initialization_method="estimated",
        )
        candidates.append(model.fit())

    return min(candidates, key=lambda result: result.aic)


def forecast(result, steps: int = 5) -> pd.DataFrame:
    """Forecast future values with simulated 95% confidence intervals.

    Args:
        result: Fitted ``HoltWintersResults`` from ``fit_ets``.
        steps: Number of years to forecast ahead.

    Returns:
        A DataFrame indexed by forecast step (0-based) with columns
        ``forecast``, ``lower`` and ``upper`` (95% bounds from simulation),
        in the original risk-score units. The caller is responsible for
        attaching real year labels (see ``extend_with_forecast``).
    """
    point_forecast = result.forecast(steps=steps)

    simulations = result.simulate(
        nsimulations=steps, repetitions=N_SIMULATIONS, random_state=RANDOM_SEED
    )
    lower = np.percentile(simulations, 2.5, axis=1)
    upper = np.percentile(simulations, 97.5, axis=1)

    return pd.DataFrame(
        {
            "forecast": point_forecast.values,
            "lower": lower,
            "upper": upper,
        }
    )


def extend_with_forecast(
    risk_df: pd.DataFrame, countries: list = None, end_year: int = 2030
) -> pd.DataFrame:
    """Concatenate actual risk scores with ETS forecasts up to a year.

    For each country, fits Holt's linear trend method on its risk score
    history and appends forecasts from the year after its last observation
    through ``end_year``. Countries with fewer than 4 observations are
    skipped (with a warning printed) since a trend model needs a minimum
    amount of history to be meaningful.

    Args:
        risk_df: DataFrame from ``compute_composite_risk``.
        countries: Optional list of ISO3 codes to process. Defaults to every
            country in ``risk_df``.
        end_year: Last year to forecast (inclusive).

    Returns:
        A DataFrame with columns ``country``, ``year``, ``risk_score``,
        ``source`` ("actual" or "forecast"), ``lower`` and ``upper`` (95%
        bounds, only filled on forecast rows), sorted by country then year.
    """
    if countries is None:
        countries = sorted(risk_df["country"].unique())

    frames = []
    for country in countries:
        series = prepare_series(risk_df, country)
        frames.append(
            pd.DataFrame(
                {
                    "country": country,
                    "year": series.index.year,
                    "risk_score": series.values,
                    "source": "actual",
                }
            )
        )

        steps = end_year - series.index[-1].year
        if steps <= 0:
            continue
        if len(series) < 4:
            print(f"Skipping forecast for {country!r}: fewer than 4 observations.")
            continue

        predicted = forecast(fit_ets(series), steps=steps)
        forecast_years = range(series.index[-1].year + 1, end_year + 1)
        frames.append(
            pd.DataFrame(
                {
                    "country": country,
                    "year": list(forecast_years),
                    "risk_score": predicted["forecast"].values,
                    "source": "forecast",
                    "lower": predicted["lower"].values,
                    "upper": predicted["upper"].values,
                }
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["country", "year"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import pandas as pd

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    from model.composite_risk_score import compute_composite_risk

    df = pd.read_csv(ROOT / "data" / "data_preprocessed" / "processed_data.csv")
    risk = compute_composite_risk(df)

    combined = extend_with_forecast(risk, end_year=2030)
    print("Actuals + forecasts to 2030 (all countries):")
    print(combined.tail(12))

    output_dir = ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "risk_score_with_ets_forecast.csv"
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} rows to {output_path}")
