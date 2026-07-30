"""ETS (Holt's linear trend) models on every ND-GAIN indicator.

Same approach as ``arima_all_indicators`` but using exponential smoothing
(``model.exp_smoothing.fit_ets``) instead of ARIMA: for every (country,
indicator) pair, fits Holt's linear trend on the yearly series and appends
forecasts through ``end_year``.

Usage:
    Run as a script to fit a model per country and indicator and save the
    actual values concatenated with forecasts up to 2040:

        python model/ets_all_indicators.py

    Or import from another module / notebook:

        from model.composite_risk_score import get_composite_risk
        from model.ets_all_indicators import extend_with_forecast

        combined = extend_with_forecast(get_composite_risk(), end_year=2040)
"""

import sys
from pathlib import Path

import pandas as pd

# Allow running from the repo root or the model/ folder.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.indicators import indicator_map
from model.old_models.arima import prepare_series
from model.old_models.exp_smoothing import fit_ets, forecast  # if exp_smoothing also moved under old_models
from model.old_models.composite_risk_score import get_composite_risk


def extend_with_forecast(
    df: pd.DataFrame,
    indicators: list = None,
    countries: list = None,
    end_year: int = 2040,
) -> pd.DataFrame:
    """Concatenate actual indicator values with ETS forecasts up to a year.

    For each (country, indicator) pair, fits Holt's linear trend (damped vs.
    non-damped chosen by AIC, per ``model.exp_smoothing.fit_ets``) on the
    indicator's history and appends forecasts from the year after its last
    observation through ``end_year``. Pairs with fewer than 4 observations,
    or whose fit fails, are kept as actuals only, with a message printed.

    Args:
        df: Composite risk DataFrame from ``get_composite_risk``.
        indicators: Optional list of indicator column names to process.
            Defaults to the names in ``config.indicators.indicator_map``.
        countries: Optional list of ISO3 codes to process. Defaults to every
            country in ``df``.
        end_year: Last year to forecast (inclusive).

    Returns:
        A DataFrame with columns ``country``, ``indicator``, ``year``,
        ``value``, ``source`` ("actual" or "forecast"), ``lower`` and
        ``upper`` (95% bounds, only filled on forecast rows), sorted by
        country, indicator, then year.
    """
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
                predicted = forecast(fit_ets(series), steps=steps)
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
                        "value": predicted["forecast"].values,
                        "source": "forecast",
                        "lower": predicted["lower"].values,
                        "upper": predicted["upper"].values,
                    }
                )
            )

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["country", "indicator", "year"]).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "data_preprocessed" / "processed_data.csv")

    combined = extend_with_forecast(df, end_year=2040)
    print("Actuals + forecasts to 2040 (all countries, all indicators):")
    print(combined.tail(12))

    output_path = ROOT / "data" / "outputs" / "all_indicators_ets_forecast.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} rows to {output_path}")
