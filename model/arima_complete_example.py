"""
Complete ARIMA Strategy Validation Workflow
============================================
Copy-paste this script and update the data loading section at the bottom
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Your grid from yesterday
GRID = [(0, 0, 1), (0, 0, 0), (0, 1, 0), (1, 0, 0), (2, 0, 0), (1, 0, 0), (2, 0, 1)]

def fit_arima_on_train(series, p, d, q):
    """Fit ARIMA on training data"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(series, order=(p, d, q)).fit()
            train_mse = np.mean(model.resid ** 2)
            return {
                'order': (p, d, q),
                'aic': model.aic,
                'train_mse': train_mse,
                'model': model
            }
    except:
        return None

def evaluate_test(model, test_series):
    """Evaluate on test set"""
    try:
        forecast = model.get_forecast(steps=len(test_series))
        test_mse = mean_squared_error(test_series, forecast.predicted_mean)
        test_mae = mean_absolute_error(test_series, forecast.predicted_mean)
        return test_mse, test_mae
    except:
        return None, None

class ARIMAStrategyValidator:
    """Validate 4 ARIMA order selection strategies"""

    def __init__(self, df, countries, indicators, grid, train_split=0.8):
        self.df = df
        self.countries = countries
        self.indicators = indicators
        self.grid = grid
        self.train_split = train_split
        self.results = []

        print("Step 1: Fitting all grid combinations on training data...")
        self._fit_all_on_train()

        print("Step 2: Selecting best orders for each strategy...")
        self._select_strategy_orders()

        print("Step 3: Evaluating strategies on test data...")
        self._evaluate_on_test()

    def _fit_all_on_train(self):
        """Fit all grid combinations on training data for each country-indicator"""
        self.train_fits = {}

        total = len(self.countries) * len(self.indicators)
        count = 0

        for country in self.countries:
            for indicator in self.indicators:
                count += 1
                print(f"  [{count}/{total}] Fitting {country} - {indicator}")

                series = self.df.loc[country, indicator]
                split_idx = int(len(series) * self.train_split)
                train_series = series[:split_idx]

                self.train_fits[(country, indicator)] = {}

                for p, d, q in self.grid:
                    result = fit_arima_on_train(train_series, p, d, q)
                    if result:
                        self.train_fits[(country, indicator)][(p, d, q)] = result

    def _select_strategy_orders(self):
        """Select best order for each of 4 strategies"""

        # Strategy 1: Best order for each country-indicator pair
        print("  → Strategy 1: 1 order per country-indicator")
        self.strategy_orders = {'1 order per country-indicator': {}}
        for key, fits in self.train_fits.items():
            if fits:
                best = min(fits.values(), key=lambda x: x['train_mse'])
                self.strategy_orders['1 order per country-indicator'][key] = best['order']

        # Strategy 2: Best order for each indicator (across all countries)
        print("  → Strategy 2: 1 order per indicator")
        self.strategy_orders['1 order per indicator'] = {}
        for indicator in self.indicators:
            all_fits = []
            for country in self.countries:
                key = (country, indicator)
                if key in self.train_fits:
                    all_fits.extend(self.train_fits[key].values())

            if all_fits:
                best_order = min(all_fits, key=lambda x: x['train_mse'])['order']
                for country in self.countries:
                    self.strategy_orders['1 order per indicator'][(country, indicator)] = best_order

        # Strategy 3: Best order for each country (across all indicators)
        print("  → Strategy 3: 1 order per country")
        self.strategy_orders['1 order per country'] = {}
        for country in self.countries:
            all_fits = []
            for indicator in self.indicators:
                key = (country, indicator)
                if key in self.train_fits:
                    all_fits.extend(self.train_fits[key].values())

            if all_fits:
                best_order = min(all_fits, key=lambda x: x['train_mse'])['order']
                for indicator in self.indicators:
                    self.strategy_orders['1 order per country'][(country, indicator)] = best_order

        # Strategy 4: Global best order (across everything)
        print("  → Strategy 4: 1 global order")
        self.strategy_orders['1 global order'] = {}
        all_fits = []
        for fits in self.train_fits.values():
            all_fits.extend(fits.values())

        if all_fits:
            best_order = min(all_fits, key=lambda x: x['train_mse'])['order']
            for country in self.countries:
                for indicator in self.indicators:
                    self.strategy_orders['1 global order'][(country, indicator)] = best_order

    def _evaluate_on_test(self):
        """Evaluate all strategies on held-out test data"""

        total = len(self.countries) * len(self.indicators)
        count = 0

        for country in self.countries:
            for indicator in self.indicators:
                count += 1

                series = self.df.loc[country, indicator]
                split_idx = int(len(series) * self.train_split)
                test_series = series[split_idx:]

                key = (country, indicator)
                if key not in self.train_fits:
                    continue

                for strategy, orders_dict in self.strategy_orders.items():
                    if key not in orders_dict:
                        continue

                    order = orders_dict[key]

                    # Get the fitted model from training
                    if order in self.train_fits[key]:
                        fit_result = self.train_fits[key][order]
                        model = fit_result['model']
                        train_mse = fit_result['train_mse']

                        # Evaluate on test set
                        test_mse, test_mae = evaluate_test(model, test_series)

                        if test_mse is not None:
                            self.results.append({
                                'country': country,
                                'indicator': indicator,
                                'strategy': strategy,
                                'order': str(order),
                                'train_mse': train_mse,
                                'test_mse': test_mse,
                                'test_mae': test_mae,
                                'overfitting_gap': test_mse - train_mse,
                            })

    def get_results_df(self):
        """Get results as DataFrame"""
        return pd.DataFrame(self.results)

    def summarize(self):
        """Print comprehensive summary"""
        results_df = self.get_results_df()

        print("\n" + "="*100)
        print("HOLDOUT VALIDATION RESULTS - STRATEGY COMPARISON")
        print("="*100 + "\n")

        summary = results_df.groupby('strategy').agg({
            'test_mse': ['mean', 'std', 'min', 'max'],
            'test_mae': ['mean', 'std'],
            'train_mse': ['mean'],
            'overfitting_gap': ['mean', 'std']
        }).round(8)

        print(summary)

        print("\n" + "="*100)
        print("SIMPLE SUMMARY TABLE")
        print("="*100 + "\n")

        simple = results_df.groupby('strategy').agg({
            'test_mse': 'mean',
            'test_mae': 'mean',
            'train_mse': 'mean',
            'overfitting_gap': 'mean',
        }).round(8)

        simple['num_combinations'] = results_df.groupby('strategy').size()
        simple = simple.sort_values('test_mse')

        print(simple.to_string())

        # Best strategy
        best_strategy = simple.index[0]
        best_mse = simple.loc[best_strategy, 'test_mse']
        best_mae = simple.loc[best_strategy, 'test_mae']
        best_gap = simple.loc[best_strategy, 'overfitting_gap']

        print("\n" + "="*100)
        print(f"✓ BEST STRATEGY: {best_strategy}")
        print("="*100)
        print(f"  Test MSE:           {best_mse:.8f}")
        print(f"  Test MAE:           {best_mae:.8f}")
        print(f"  Overfitting gap:    {best_gap:.8f}")

        # Show orders selected
        print("\n" + "="*100)
        print("ORDERS SELECTED BY STRATEGY")
        print("="*100 + "\n")

        for strategy in sorted(results_df['strategy'].unique()):
            unique_orders = results_df[results_df['strategy'] == strategy]['order'].unique()
            print(f"{strategy}:")
            print(f"  {len(unique_orders)} unique order(s): {', '.join(sorted(unique_orders))}\n")

        return simple

    def plot_results(self, save_path='arima_validation_results.png'):
        """Visualize validation results"""
        results_df = self.get_results_df()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('ARIMA Strategy Validation Results', fontsize=16, fontweight='bold')

        # Plot 1: Test MSE by strategy
        ax = axes[0, 0]
        results_df.boxplot(column='test_mse', by='strategy', ax=ax)
        ax.set_title('Test MSE Distribution')
        ax.set_ylabel('Test MSE')
        ax.set_xlabel('Strategy')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Test MAE by strategy
        ax = axes[0, 1]
        results_df.boxplot(column='test_mae', by='strategy', ax=ax)
        ax.set_title('Test MAE Distribution')
        ax.set_ylabel('Test MAE')
        ax.set_xlabel('Strategy')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 3: Overfitting gap
        ax = axes[1, 0]
        results_df.boxplot(column='overfitting_gap', by='strategy', ax=ax)
        ax.set_title('Overfitting Gap (Test MSE - Train MSE)')
        ax.set_ylabel('Gap')
        ax.set_xlabel('Strategy')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 4: Mean comparison
        ax = axes[1, 1]
        mean_mse = results_df.groupby('strategy')['test_mse'].mean().sort_values()
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(mean_mse))]
        mean_mse.plot(kind='barh', ax=ax, color=colors)
        ax.set_title('Mean Test MSE (Lower is Better)')
        ax.set_xlabel('Mean Test MSE')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to '{save_path}'")

        return fig


# ============================================================================
# MAIN EXECUTION - UPDATE THIS SECTION WITH YOUR DATA
# ============================================================================

if __name__ == "__main__":

    # ========== STEP 1: Load your data ==========
    print("Loading data...")

    def load_data():
        df = pd.read_csv("data/data_preprocessed/processed_data.csv")
        df = df.set_index(["Country", "Year"])
        return df

    df = load_data()
    df = df.drop(columns='Exposure')

    countries = df.index.get_level_values(0).unique().tolist()
    indicators = df.columns.tolist()

    print(f"Data loaded: {len(countries)} countries, {len(indicators)} indicators")
    print(f"Countries: {countries}")
    print(f"Indicators: {indicators}")

    # ========== STEP 2: Run validation ==========
    print("\n" + "="*100)
    print("RUNNING ARIMA STRATEGY VALIDATION")
    print("="*100 + "\n")

    validator = ARIMAStrategyValidator(
        df=df,
        countries=countries,
        indicators=indicators,
        grid=GRID,
        train_split=0.8  # 80% train, 20% test
    )

    # ========== STEP 3: Get and save results ==========
    results_df = validator.get_results_df()

    # Save raw results
    results_df.to_csv('arima_validation_results.csv', index=False)
    print("\n✓ Results saved to 'arima_validation_results.csv'")

    # ========== STEP 4: Print summary ==========
    summary = validator.summarize()

    # Save summary
    summary.to_csv('arima_validation_summary.csv')
    print("\n✓ Summary saved to 'arima_validation_summary.csv'")

    # ========== STEP 5: Visualize ==========
    validator.plot_results()

    # ========== STEP 6: Use best strategy ==========
    best_strategy = summary.index[0]
    print(f"\n{'='*100}")
    print(f"RECOMMENDED NEXT STEPS:")
    print(f"{'='*100}")
    print(f"\n1. Use strategy: {best_strategy}")
    print(f"\n2. Extract selected orders:")
    print(f"   best_results = results_df[results_df['strategy'] == '{best_strategy}']")
    print(f"   best_results[['country', 'indicator', 'order']].to_csv('best_orders.csv')")
    print(f"\n3. Retrain final model on full data with these orders")
    print(f"\n4. Document in your report why this strategy outperformed others")
