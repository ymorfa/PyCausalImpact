# PyCausalImpact Documentation

Examples and additional guides for using PyCausalImpact live here.

- [`examples/basic_usage.ipynb`](examples/basic_usage.ipynb) –
  minimal script showing how to estimate causal impact with a
  Statsmodels ARIMA model.

## Summary Output

Calling `CausalImpactPy(...).run()` produces a results table that can be passed to `ReportGenerator.generate()` (or the high‑level `summary()` helper) to obtain a two‑row summary of the intervention effect. The rows are:

- **average** – mean values across the post‑intervention period.
- **cumulative** – sums over the post‑intervention period.

Each column describes a different quantity derived from the observed series and its counterfactual prediction:

- **observed** – actual outcomes in the post period. Calculated as the mean or sum of the observed data.
- **predicted** – modelled counterfactual had the intervention not occurred. Obtained from the forecasting model; averaged or summed as above.
- **predicted_lower / predicted_upper** – bounds of the prediction interval for the counterfactual. These come from model‑specific predictive intervals or bootstrap simulations. If a model does not supply intervals, these remain missing and should be estimated via bootstrapping or Bayesian simulation.
- **abs_effect** – absolute causal effect, computed as `observed - predicted`. For the cumulative row the last value of the cumulative effect series is used.
- **abs_effect_lower / abs_effect_upper** – interval bounds for the absolute effect, derived from the corresponding predicted bounds (`observed` minus `predicted_upper` or `predicted_lower`). They remain missing when prediction intervals are unavailable.
- **rel_effect** – relative effect expressed as a fraction of the prediction: `abs_effect / predicted`. A value of `0.10` indicates a 10 % increase over the counterfactual baseline.
- **rel_effect_lower / rel_effect_upper** – interval bounds for the relative effect, obtained by dividing the absolute effect bounds by the opposite prediction bounds (e.g. `abs_effect_lower / predicted_upper`).
- **p_value** – two‑sided p‑value for the hypothesis that the effect is zero. Calculated using a normal approximation based on the width of the effect interval. When interval estimates are missing this value is `NaN`; future versions may employ model‑specific or Bayesian tests.
- **causal_probability** – probability that the intervention produced an effect with the observed sign, computed as `1 - p_value/2`. When p‑values cannot be computed, a Bayesian or bootstrap‑based estimate should be implemented.

Together these metrics summarise both the magnitude and the statistical significance of the intervention's impact.
