"""
ARIMA Time Series Forecasting Model
Strategy: 1 order per indicator

This module handles:

#load_data
#drop_columns (exposure, readiness, vulnerability)
#select best order per country (out of the 2 Default: [(0, 0, 1), (2, 0, 1)])
#fit
#create new dataframe with all country-year-indications (actual data + forecast until 2030)

"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Optional
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def load_data():
    df = pd.read_csv("data/data_preprocessed/processed_data.csv")
    df = df.set_index(["Country", "Year"])
    return df

class ARIMACountryModel:
    """
    Wrapper for ARIMA models trained using one order per indicator

    Strategy: 1 order per indicator
    - Each indicator gets ONE order that is used for every country.
    """

    def __init__(self, grid: list = None):
        """
        Initialize model

        Args:
            grid: List of (p, d, q) tuples to search.
                  Default: [(0, 0, 1), (2, 0, 0), (2, 0, 1)]
        """
        self.grid = grid or [(0, 0, 1), (2, 0 ,1), (2, 0, 1)]
        self.models = {}  # {indicator: {country: fitted_model}}
        self.best_orders = {}  # {indicator: (p, d, q)}
        self.training_results = []

    def _fit_arima(self, series: pd.Series, p: int, d: int, q: int) -> Optional[Dict]:
        """
        Fit single ARIMA model

        Args:
            series: Time series data
            p, d, q: ARIMA parameters

        Returns:
            Dict with model and metrics, or None if fitting failed
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(series, order=(p, d, q)).fit()
                train_mse = np.mean(model.resid ** 2)

                return {
                    # 'order': (p, d, q),
                    'aic': model.aic,
                    # 'bic': model.bic,
                     'train_mse': train_mse,
                    # 'model': model
                }
        except Exception as e:
            return None

    def _select_best_order_per_indicator(
        self,
        df: pd.DataFrame,
        countries: list,
        indicators: list,
    ) -> Dict:
            """ Select one ARIMA order for each indicator.

        The selected order is the one with the lowest average AIC
        across all countries.
        """
            best_orders = {}

            for indicator in indicators:

                    order_scores = {}

                    for order in self.grid:

                        p, d, q = order
                        aics = []

                        for country in countries:

                            try:
                                series = df.loc[country, indicator]
                            except KeyError:
                                continue

                            result = self._fit_arima(series, p, d, q)

                            if result:
                                aics.append(result["aic"])

                        if aics:
                            order_scores[order] = np.mean(aics)

                    if order_scores:
                        best_order = min(order_scores, key=order_scores.get)
                        best_orders[indicator] = best_order

            return best_orders

    def fit(self, df: pd.DataFrame, countries: list, indicators: list) -> None:
        """
        Train ARIMA models using the best order for each indicator.
        """
        print("=" * 80)
        print("TRAINING ARIMA MODELS - STRATEGY: 1 ORDER PER INDICATOR")
        print("=" * 80)

        print("Step 1: Selecting best order for each indicator...")
        self.best_orders = self._select_best_order_per_indicator(df, countries, indicators)
        print("✓ Selected orders:")
        for indicator, order in self.best_orders.items():
            print(f"  {indicator}: {order}")

        print("Step 2: Fitting final models on full data...")
        total = len(countries) * len(indicators)
        count = 0
        self.models = {}

        for country in countries:
            self.models[country] = {}

            for indicator in indicators:
                count += 1

                if indicator not in self.best_orders:
                    continue

                order = self.best_orders[indicator]

                try:
                    series = df.loc[country, indicator]
                    print(type(series))
                    print(series.name)
                    print(series.index)
                    if country == "AFG" and indicator == "Capacity":

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = ARIMA(series, order=order).fit()

                    self.models[country][indicator] = model

                    print(
                        f"  [{count}/{total}] {country:20s} - {indicator:30s} | "
                        f"AIC: {model.aic:10.2f} | MSE: {np.mean(model.resid**2):10.6f}"
                    )

                    self.training_results.append({
                        "country": country,
                        "indicator": indicator,
                        "order": order,
                        "aic": model.aic,
                        "bic": model.bic,
                        "train_mse": np.mean(model.resid ** 2),
                    })

                except Exception as e:
                    print(f"  ✗ {country} - {indicator}: {e}")

        print("✓ Training complete!")
        print(f"  Total models trained: {sum(len(v) for v in self.models.values())}")

    def forecast(self, years=5):

        print(f"Forecast horizon: {years}")

        forecasts = {}

        for country in self.models:
            forecasts[country] = {}

            for indicator, model in self.models[country].items():

                try:
                    print(f"Forecasting {country} - {indicator}")

                    print(f"years: {years} ({type(years)})")
                    print(f"nobs: {model.nobs}")

                    pred = model.get_forecast(steps=int(years)).predicted_mean

                    forecasts[country][indicator] = pred

                except Exception as e:
                    print(f"FAILED: {country} - {indicator}")
                    print(f"Years: {years}")
                    print(f"Training observations: {len(model.model.endog)}")
                    raise

        print("Finished forecasting all countries")

        return forecasts

def forecast_to_dataframe(df, model, last_year=2030):

    rows = []

    final_year = df.index.get_level_values("Year").max()
    forecast_years = last_year - final_year

    print(f"Final year: {final_year}")
    print(f"Forecast years: {forecast_years}")

    forecasts = model.forecast(forecast_years)

    for country in model.models:

        for indicator in model.models[country]:

            # historical values
            history = df.loc[country, indicator]

            for year, value in history.items():
                rows.append({
                "Country": country,
                "Year": year,
                "Indicator": indicator,
                "Value": value
                })

            # forecasts
            future = forecasts[country][indicator]

            for i, value in enumerate(future):

                rows.append({
                "Country": country,
                "Year": final_year + i + 1,
                "Indicator": indicator,
                "Value": value
                })

    return pd.DataFrame(rows)

def main():

    # Load
    df_original = load_data()
    df = df_original.drop(
        columns=["Exposure", "Readiness", "Vulnerability"]
    )

    df_original.reset_index().to_parquet(
    "data/intermediate/df_original.parquet",
    index=False,
    )

    countries = df.index.get_level_values("Country").unique()
    indicators = df.columns

    # Train
    model = ARIMACountryModel()
    model.fit(df, countries, indicators)

    # Forecast
    forecast_df = forecast_to_dataframe(
        df,
        model,
        last_year=2030
    )

    forecast_df.to_parquet(
    "data/intermediate/forecast.parquet",
    index=False,
    )
if __name__ == "__main__":
    main()
