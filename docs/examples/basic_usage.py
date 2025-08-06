"""Minimal end-to-end example for PyCausalImpact."""

import pathlib
import sys
from functools import partial

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from pycausalimpact import CausalImpactPy
from pycausalimpact.models.statsmodels import StatsmodelsAdapter

# Generate synthetic data with an intervention after index 40
np.random.seed(0)
series = pd.Series(np.random.normal(size=60)).cumsum()
series.iloc[40:] += 5

df = pd.DataFrame({"y": series})
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
print(impact.summary(plot=False))
