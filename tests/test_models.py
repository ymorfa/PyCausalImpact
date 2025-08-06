import pandas as pd
import pytest
from functools import partial

from statsmodels.tsa.arima.model import ARIMA

from pycausalimpact.models.statsmodels import StatsmodelsAdapter
from pycausalimpact.models.prophet import ProphetAdapter
from pycausalimpact.models.sktime import SktimeAdapter


def test_statsmodels_adapter_forecast_with_exog():
    y = pd.Series(range(10))
    X = pd.DataFrame({"x": range(10)})

    pre_y, post_y = y[:8], y[8:]
    pre_X, post_X = X[:8], X[8:]

    adapter = StatsmodelsAdapter(partial(ARIMA, order=(1, 0, 0)))
    adapter.fit(pre_y, X=pre_X)
    result = adapter.predict(steps=len(post_y), X=post_X)

    manual = ARIMA(pre_y, exog=pre_X, order=(1, 0, 0)).fit()
    expected = manual.forecast(steps=len(post_y), exog=post_X)

    pd.testing.assert_series_equal(result, expected)


def test_statsmodels_adapter_interval_dataframe():
    y = pd.Series(range(10))
    X = pd.DataFrame({"x": range(10)})

    adapter = StatsmodelsAdapter(partial(ARIMA, order=(1, 0, 0)))
    adapter.fit(y[:8], X=X[:8])
    interval = adapter.predict_interval(steps=2, X=X[8:])

    assert list(interval.columns) == ["lower", "upper"]
    assert len(interval) == 2


class MockProphet:
    def __init__(self):
        self.fit_df = None
        self.predict_df = None

    def add_regressor(self, name):
        pass

    def fit(self, df):
        self.fit_df = df

    def make_future_dataframe(self, periods, freq):
        return pd.DataFrame({"ds": range(len(self.fit_df) + periods)})

    def predict(self, df):
        self.predict_df = df
        n = len(df)
        return pd.DataFrame({
            "ds": df["ds"],
            "yhat": range(n),
            "yhat_lower": range(n),
            "yhat_upper": range(n),
        })


def test_prophet_adapter_handles_exog_and_interval():
    y = pd.Series(range(5))
    X = pd.DataFrame({"x": range(5)})
    future_X = pd.DataFrame({"x": [5, 6]})

    model = MockProphet()
    adapter = ProphetAdapter(model)

    adapter.fit(y, X=X)
    pred = adapter.predict(steps=2, X=future_X)
    interval = adapter.predict_interval(steps=2, X=future_X)

    # ensure exogenous data passed through
    assert "x" in model.fit_df.columns
    assert model.predict_df["x"].iloc[-2:].reset_index(drop=True).equals(future_X["x"])
    assert list(interval.columns) == ["lower", "upper"]
    assert list(pred) == [5, 6]


class MockSktimeModel:
    def __init__(self):
        self.fit_X = None
        self.predict_X = None

    def fit(self, y, X=None):
        self.fit_X = X

    def predict(self, fh, X=None):
        self.predict_X = X
        return pd.Series([0] * len(fh))


def test_sktime_adapter_exog_and_interval_error():
    y = pd.Series(range(5))
    X = pd.DataFrame({"x": range(5)})
    future_X = X.iloc[-2:]

    model = MockSktimeModel()
    adapter = SktimeAdapter(model)

    adapter.fit(y, X=X)
    adapter.predict(steps=2, X=future_X)

    assert model.fit_X.equals(X)
    assert model.predict_X.equals(future_X)

    with pytest.raises(NotImplementedError):
        adapter.predict_interval(steps=2, X=future_X)

