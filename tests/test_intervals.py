import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from functools import partial

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
@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_interval_shapes_and_order(name, make_adapter, series_y, exog_X, alpha):
    adapter = make_adapter()
    adapter.fit(series_y[:-5], exog_X[:-5])
    mean = adapter.predict(5, exog_X[-5:])
    interval = adapter.predict_interval(5, exog_X[-5:], alpha=alpha)
    assert interval.shape == (5, 2)
    lower = interval["lower"].to_numpy()
    upper = interval["upper"].to_numpy()
    mean_arr = np.asarray(mean)
    assert np.all(lower <= mean_arr)
    assert np.all(mean_arr <= upper)
