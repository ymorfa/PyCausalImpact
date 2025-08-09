import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from functools import partial

from pycausalimpact.models import (
    StatsmodelsAdapter,
    ProphetAdapter,
    SktimeAdapter,
    TFPStructuralTimeSeries,
)

try:  # Optional imports for constructors
    from statsmodels.tsa.arima.model import ARIMA
except Exception:  # pragma: no cover - handled via skip
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
def test_fit_predict_no_exog(name, make_adapter, series_y):
    adapter = make_adapter()
    adapter.fit(series_y[:-10])
    mean = adapter.predict(10)
    assert len(mean) == 10
    assert np.isfinite(np.asarray(mean)).all()


@pytest.mark.parametrize("name,make_adapter", ADAPTERS)
def test_fit_predict_with_exog(name, make_adapter, series_y, exog_X):
    adapter = make_adapter()
    adapter.fit(series_y[:-10], exog_X[:-10])
    mean = adapter.predict(10, exog_X[-10:])
    assert len(mean) == 10
    assert np.isfinite(np.asarray(mean)).all()


@pytest.mark.parametrize("name,make_adapter", ADAPTERS)
def test_predict_interval(name, make_adapter, series_y, exog_X):
    adapter = make_adapter()
    adapter.fit(series_y[:-10], exog_X[:-10])
    mean = adapter.predict(10, exog_X[-10:])
    interval = adapter.predict_interval(10, exog_X[-10:], alpha=0.1)
    assert len(interval) == 10
    assert list(interval.columns) == ["lower", "upper"]
    lower = interval["lower"].to_numpy()
    upper = interval["upper"].to_numpy()
    mean_arr = np.asarray(mean)
    assert np.all(lower <= mean_arr)
    assert np.all(mean_arr <= upper)
    assert not np.isnan(lower).any() and not np.isnan(upper).any()


@pytest.mark.parametrize("name,make_adapter", ADAPTERS)
def test_input_validation(name, make_adapter, series_y, exog_X):
    if name != "statsmodels":
        pytest.skip("validation handled by backend")
    adapter = make_adapter()
    with pytest.raises(Exception):
        adapter.fit(series_y, exog_X.iloc[:-1])
