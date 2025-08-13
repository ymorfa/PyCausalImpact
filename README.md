# PyCausalImpact

**PyCausalImpact** is a Python library for estimating the **causal impact** of an intervention or event on a time series. It follows the principles of Google's [CausalImpact](https://github.com/google/CausalImpact) but introduces a **model-agnostic design**, allowing users to plug in any forecasting model (ARIMA, Prophet, machine learning regressors, etc.) to generate counterfactual predictions and quantify the difference between **observed** and **expected** values.

## Key Features

- **Flexible forecasting backend:** Use models from libraries such as **Statsmodels**, **Prophet**, **sktime**, **TensorFlow Probability**, **Darts**, or any compatible `fit/predict` estimator.
- **Causal effect estimation:** Compute the difference between observed and predicted values after an intervention.
- **Rich statistical outputs:**
  - Pointwise causal effect (observed – predicted)
  - Average and cumulative effects
  - Confidence intervals for effects (derived from predictive intervals or bootstrapped residuals)
  - Statistical significance and a causal confidence score
- **Intuitive interface with Pandas:** Accepts data as a `pandas.DataFrame`, with easy specification of target series, control features, and pre/post-intervention periods.
- **Visual and textual reporting:** Built-in `summary()` method provides a textual summary, a metrics table, and a 3-panel visualization (observed vs. predicted, pointwise effect, cumulative effect).
- **Compatible with univariate or multivariate time series:** Supports models with or without exogenous regressors (control variables).

## Installation

```bash
pip install pycausalimpact

# or from source
git clone https://github.com/pycausalimpact/PyCausalImpact.git
cd PyCausalImpact
pip install -e .
```

To enable TensorFlow Probability-based models, install the optional `tfp` extra:

```bash
pip install pycausalimpact[tfp]
```

## Multiple Inference Methods

The TensorFlow Probability adapter supports multiple inference algorithms.
Variational inference is used by default, while Hamiltonian Monte Carlo (HMC)
is available for more exact posterior sampling:

```python
from pycausalimpact.models import TFPStructuralTimeSeries

# Variational inference (default)
model_vi = TFPStructuralTimeSeries()

# Hamiltonian Monte Carlo
model_hmc = TFPStructuralTimeSeries(inference_method="hmc")
```

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

## Parameters

`CausalImpactPy` accepts the following arguments:

- `data` (`pd.DataFrame`): Time series with the target column and optional controls.
- `index` (`str | None`): Name of the column to use as the time index if the DataFrame isn't already indexed.
- `y` (`list[str]`): Columns to use; the first is the target series, remaining columns are controls.
- `pre_period` (`tuple`): `(start, end)` defining the pre-intervention period.
- `post_period` (`tuple`): `(start, end)` defining the post-intervention period.
- `model` (`Any`): Forecasting model instance exposing `fit` and `predict` methods.
- `alpha` (`float`, optional): Significance level for confidence intervals; defaults to `0.05`.

## End-to-End Example

```python
import numpy as np
import pandas as pd
from functools import partial
from statsmodels.tsa.arima.model import ARIMA

from pycausalimpact import CausalImpactPy
from pycausalimpact.models.statsmodels import StatsmodelsAdapter

# Synthetic data with an intervention after index 40
np.random.seed(0)
data = pd.Series(np.random.normal(size=60)).cumsum()
data.iloc[40:] += 5
df = pd.DataFrame({"y": data})

pre_period = (0, 39)
post_period = (40, 59)
model = StatsmodelsAdapter(partial(ARIMA, order=(1, 0, 0)))

impact = CausalImpactPy(
    data=df,
    index=None,
    y=["y"],
    pre_period=pre_period,
    post_period=post_period,
    model=model,
)

impact.run()
impact.summary(plot=True)
```

See more runnable scripts in [`docs/examples/`](docs/examples/).

## Planned Features

- [x] Model wrappers for **Statsmodels**, **Prophet**, **sktime**, and **TensorFlow Probability** out of the box.
- [x] Bootstrapping-based confidence intervals for models lacking predictive intervals.
- [x] Narrative-style report generation for executive summaries.
- [ ] Extended plotting options (seaborn, Plotly, etc.).
- [ ] Additional adapters for libraries such as **scikit-learn** or **Darts**.

## Roadmap

- [x] Core engine for causal effect estimation (v0.1)
- [x] Forecasting model adapters (Statsmodels, Prophet, sktime)
- [x] Reporting API (`summary`, `plot`)
- [ ] Advanced statistical options (Bayesian models, richer bootstrap methods)
- [ ] Packaging and PyPI release
- [ ] Additional model adapters and integrations
- [ ] Release initial stable version (v1.0)

## License

MIT License (planned).
