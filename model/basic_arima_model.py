"""Basic ARIMA model on the composite climate risk score.

Fits a non-seasonal ARIMA on one country's yearly ``risk_score`` series from
the composite risk dataset. The modelling machinery (series preparation,
Box-Jenkins order selection, fitting, forecasting) lives in ``model.arima``
and is shared with ``arima_all_indicators``.

Usage:
    Run as a script to fit a model per country and save the actual scores
    concatenated with forecasts up to 2030:

        python model/basic_arima_model.py

    Or import from another module / notebook:

        from model.composite_risk_score import get_composite_risk
        from model.basic_arima_model import extend_with_forecast

        combined = extend_with_forecast(get_composite_risk(), end_year=2030)
"""

import sys
from pathlib import Path

import pandas as pd

# Allow running from the repo root or the model/ folder.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.arima import prepare_series, select_order, fit_arima, forecast


def extend_with_forecast(
    risk_df: pd.DataFrame, countries: list = None, end_year: int = 2030
) -> pd.DataFrame:
    """Concatenate actual risk scores with ARIMA forecasts up to a year.

    For each country, fits an ARIMA (order chosen per country with
    ``model.arima.select_order``) on its risk score history and appends
    forecasts from the year after its last observation through ``end_year``.

    Args:
        risk_df: DataFrame from ``compute_composite_risk`` /
            ``get_composite_risk``.
        countries: Optional list of ISO3 codes to process. Defaults to every
            country in ``risk_df``.
        end_year: Last year to forecast (inclusive).

    Returns:
        A DataFrame with columns ``country``, ``year``, ``risk_score``,
        ``source`` ("actual" or "forecast"), ``lower`` and ``upper`` (95%
        confidence bounds, only filled on forecast rows), sorted by country
        then year.
    """
    if countries is None:
        countries = sorted(risk_df["Country"].unique())

    frames = []
    for country in countries:
        series = prepare_series(risk_df, country, "risk_score")
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
        if steps > 0:
            # Hyperparameters are per country: each risk-score series gets
            # its own (p, 0, q) order from its own PACF / ACF.
            order = select_order(series)
            predicted = forecast(fit_arima(series, order=order), steps=steps)
            frames.append(
                pd.DataFrame(
                    {
                        "country": country,
                        "year": predicted.index.year,
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
    from model.composite_risk_score import get_composite_risk

    combined = extend_with_forecast(get_composite_risk(), end_year=2030)
    print("Actuals + forecasts to 2030 (all countries):")
    print(combined.tail(12))

    output_path = ROOT / "data" / "outputs" / "risk_score_with_forecast.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} rows to {output_path}")
