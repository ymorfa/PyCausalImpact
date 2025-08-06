
# PyCausalImpact – Agents Guide

This document defines the core development context for **PyCausalImpact**. It provides precise and minimal instructions for code generation, file structure, and agent-oriented development.

## Purpose

PyCausalImpact estimates the **causal effect of an intervention** on a time series by:
1. Fitting a forecasting model on pre-intervention data.
2. Predicting a counterfactual trajectory for the post-intervention period.
3. Computing differences between observed and predicted values (causal effect) with confidence intervals.

## Technical Requirements

- Implemented in **Python 3.9+**.
- Built around **pandas** for data handling.
- Supports **pluggable forecasting models**: models must expose `fit()` and `predict()` methods.
- Should be compatible with:
  - **Statsmodels** (ARIMA, ETS, structural models)
  - **Prophet** (via optional adapter)
  - **sktime** forecasters
  - Any `scikit-learn`-style regressor (for feature-based forecasting)

## File Structure

```bash
pycausalimpact/
  ├── __init__.py
  ├── core.py            # Main engine: data splitting, model fitting, impact computation
  ├── reporting.py        # Summary tables, narrative text, and plotting
  ├── models/
  │   ├── base.py         # Model interface adapter
  │   ├── statsmodels.py  # Wrapper for statsmodels forecasters
  │   ├── prophet.py      # Wrapper for Prophet (optional dependency)
  │   └── sktime.py       # Wrapper for sktime models
  └── utils.py            # Helpers (confidence intervals, bootstrapping, etc.)
  tests/
  ├── test\_core.py
  ├── test\_models.py
  ├── test\_reporting.py
  docs/
  ├── README.md
  └── examples/
```

## Development Priorities

1. **Core Engine**: Input validation, pre/post splitting, model interface, effect computation.
2. **Reporting**: Implement `summary()` with:
   - Numerical outputs (observed, predicted, effects, CIs).
   - Visual outputs (3-panel matplotlib plot).
3. **Adapters**: Create wrappers for external forecasting libraries.
4. **Confidence Intervals**: Use predictive intervals if available; otherwise, fallback to bootstrap.
5. **Statistical Outputs**: Provide significance testing and causal probability estimation.

## API Outline

```python
impact = CausalImpactPy(
    data: pd.DataFrame,
    index: str | None,
    y: list[str],
    pre_period: tuple,
    post_period: tuple,
    model: Any,
    alpha: float = 0.05
)

impact.summary(plot: bool = True)
impact.results  # DataFrame with observed, predicted, effects, intervals
```

## Constraints

* Minimal dependencies (core: pandas, numpy, matplotlib; extras: statsmodels, prophet, sktime).
* Clear separation between **engine**, **model adapters**, and **reporting**.
* Unit tests for reproducibility (pytest).
* Modular design to allow Bayesian extensions later.
