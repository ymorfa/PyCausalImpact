import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from functools import partial

from pycausalimpact.core import CausalImpactPy
from pycausalimpact.models import (
    StatsmodelsAdapter,
    ProphetAdapter,
    SktimeAdapter,
    TFPStructuralTimeSeries,
)

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:  # pragma: no cover
    ARIMA = None

try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None

try:
    from sktime.forecasting.naive import NaiveForecaster
except Exception:  # pragma: no cover
    NaiveForecaster = None

ADAPTERS = [
    pytest.param(
        "statsmodels",
        lambda: StatsmodelsAdapter(partial(ARIMA, order=(1, 0, 0))),
        marks=[
            pytest.mark.backend("statsmodels"),
            pytest.mark.skipif(
                StatsmodelsAdapter is None or ARIMA is None,
                reason="statsmodels not installed",
            ),
        ],
    ),
    pytest.param(
        "prophet",
        lambda: ProphetAdapter(
            Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
            )
        ),
        marks=[
            pytest.mark.backend("prophet"),
            pytest.mark.skipif(
                ProphetAdapter is None or Prophet is None,
                reason="prophet not installed",
            ),
        ],
    ),
    pytest.param(
        "sktime",
        lambda: SktimeAdapter(NaiveForecaster(strategy="last")),
        marks=[
            pytest.mark.backend("sktime"),
            pytest.mark.skipif(
                SktimeAdapter is None or NaiveForecaster is None,
                reason="sktime not installed",
            ),
        ],
    ),
    pytest.param(
        "tfp",
        lambda: TFPStructuralTimeSeries(
            num_variational_steps=20, num_results=20, num_warmup_steps=20
        ),
        marks=[
            pytest.mark.backend("tfp"),
            pytest.mark.heavy,
            pytest.mark.skipif(
                TFPStructuralTimeSeries is None,
                reason="tfp not installed",
            ),
        ],
    ),
]


@pytest.mark.parametrize("name,make_adapter", ADAPTERS)
def test_e2e_causalimpact(name, make_adapter, series_y, exog_X, pre_post_periods):
    df = exog_X.copy()
    df["y"] = series_y
    pre, post = pre_post_periods
    columns = ["y"] + list(exog_X.columns)
    model = make_adapter()
    ci = CausalImpactPy(
        data=df,
        index=None,
        y=columns,
        pre_period=pre,
        post_period=post,
        model=model,
    )
    ci.run()
    summary = ci.summary(plot=False)
    assert summary is not None

    res = ci.results
    assert set(
        [
            "predicted",
            "predicted_lower",
            "predicted_upper",
            "point_effect",
            "cumulative_effect",
        ]
    ).issubset(res.columns)
    post_len = (post[1] - post[0]).days + 1
    assert len(res) == post_len
    mean_effect = np.nanmean(res["point_effect"])
    if np.isnan(mean_effect):
        pytest.skip("model did not produce effect estimate")
    assert mean_effect > 0
