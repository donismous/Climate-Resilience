"""
Visualization Utilities.

This module contains reusable plotting functions for the Random Forest feature
attribution workflow.

Visualizations may include:
    - Random Forest feature importance,
    - SHAP summary plots,
    - SHAP dependence plots,
    - comparisons between PCA weights and model-derived feature importance,
    - country-level feature attribution analyses.

Keeping plotting functions separate from modelling code ensures a clean
separation between analysis and visualization.
"""


plot_rf_importance()

plot_shap_summary()

plot_importance_comparison()

plot_country_attribution()
