"""ARIMA models on every ND-GAIN indicator.

Same approach as ``basic_arima_model`` (which handles only the composite
risk score), but applied to each indicator in
``config.indicators.indicator_map``: for every (country, indicator) pair a
non-seasonal ARIMA is fitted on the yearly series from the composite risk
dataset. The modelling machinery (series preparation, Box-Jenkins order
selection, fitting, forecasting) lives in ``model.arima`` and is shared
with ``basic_arima_model``.

Usage:
    Run as a script to fit a model per country and indicator and save the
    actual values concatenated with forecasts up to 2030:

        python model/arima_all_indicators.py

    Or import from another module / notebook:

        from model.composite_risk_score import get_composite_risk
        from model.arima_all_indicators import extend_with_forecast

        combined = extend_with_forecast(get_composite_risk(), end_year=2030)
"""

import sys
from pathlib import Path

import pandas as pd

# Allow running from the repo root or the model/ folder.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.indicators import indicator_map
from model.arima import prepare_series, select_order, fit_arima, forecast


def extend_with_forecast(
    df: pd.DataFrame,
    indicators: list = None,
    countries: list = None,
    end_year: int = 2030,
) -> pd.DataFrame:
    """Concatenate actual indicator values with ARIMA forecasts up to a year.

    For each (country, indicator) pair, fits an ARIMA (order chosen per
    series with ``model.arima.select_order``) on the indicator's history and
    appends forecasts from the year after its last observation through
    ``end_year``. Pairs with no data or whose fit fails are kept as actuals
    only, with a warning printed.

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
        ``upper`` (95% confidence bounds, only filled on forecast rows),
        sorted by country, indicator, then year.
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
            try:
                # Hyperparameters are per (country, indicator) pair: each
                # series gets its own (p, 0, q) order from its own PACF / ACF.
                order = select_order(series)
                predicted = forecast(fit_arima(series, order=order), steps=steps)
            except Exception as error:
                print(f"Skipping forecast for {country}/{name}: {error}")
                continue

            frames.append(
                pd.DataFrame(
                    {
                        "country": country,
                        "indicator": name,
                        "year": predicted.index.year,
                        "value": predicted["forecast"].values,
                        "source": "forecast",
                        "lower": predicted["lower"].values,
                        "upper": predicted["upper"].values,
                    }
                )
            )

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["country", "indicator", "year"]).reset_index(
        drop=True
    )


if __name__ == "__main__":
    from model.composite_risk_score import get_composite_risk

    combined = extend_with_forecast(get_composite_risk(), end_year=2030)
    print("Actuals + forecasts to 2030 (all countries, all indicators):")
    print(combined.tail(12))

    output_path = ROOT / "data" / "outputs" / "all_indicators_with_forecast.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} rows to {output_path}")
