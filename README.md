# PyCausalImpact

**PyCausalImpact** is a Python library for estimating the **causal impact** of an intervention or event on a time series. It follows the principles of Google's [CausalImpact](https://github.com/google/CausalImpact) but introduces a **model-agnostic design**, allowing users to plug in any forecasting model (ARIMA, Prophet, machine learning regressors, etc.) to generate counterfactual predictions and quantify the difference between **observed** and **expected** values.

## Key Features

- **Flexible forecasting backend:** Use models from libraries such as **Statsmodels**, **Prophet**, **sktime**, **Darts**, or any compatible `fit/predict` estimator.
- **Causal effect estimation:** Compute the difference between observed and predicted values after an intervention.
- **Rich statistical outputs:**
  - Pointwise causal effect (observed – predicted)
  - Average and cumulative effects
  - Confidence intervals for effects (derived from predictive intervals or bootstrapped residuals)
  - Statistical significance and a causal confidence score
- **Intuitive interface with Pandas:** Accepts data as a `pandas.DataFrame`, with easy specification of target series, control features, and pre/post-intervention periods.
- **Visual and textual reporting:** Built-in `summary()` method provides a textual summary, a metrics table, and a 3-panel visualization (observed vs. predicted, pointwise effect, cumulative effect).
- **Compatible with univariate or multivariate time series:** Supports models with or without exogenous regressors (control variables).

## Usage Overview

```python
from pycausalimpact import CausalImpactPy
from sktime.forecasting.arima import AutoARIMA

impact = CausalImpactPy(
    data=df,
    index="date",
    y=["sales", "price", "temperature"],  # first column = target, others = controls
    pre_period=("2021-01-01", "2021-06-30"),
    post_period=("2021-07-01", "2021-09-30"),
    model=AutoARIMA()
)

impact.summary(plot=True)
```

This will:

1. Fit the chosen model on the **pre-intervention period**.
2. Predict the **counterfactual** for the post-period.
3. Calculate pointwise, average, and cumulative causal effects with confidence intervals.
4. Output a **visual and textual report** of the estimated causal impact.

## Planned Features

* Model wrappers for **Statsmodels**, **Prophet**, and **sktime** out of the box.
* Bootstrapping-based confidence intervals for models lacking predictive intervals.
* Narrative-style report generation for executive summaries.
* Extended plotting options (matplotlib, seaborn, or Plotly).

## Roadmap

* [ ] Core engine for causal effect estimation (v0.1)
* [ ] Forecasting model adapters (Statsmodels, Prophet, sktime)
* [ ] Reporting API (`summary`, `plot`)
* [ ] Advanced statistical options (bootstrap, Bayesian models)
* [ ] Release initial stable version (v1.0)

## License

MIT License (planned).