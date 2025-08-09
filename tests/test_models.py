import pandas as pd
import pytest
from functools import partial

from statsmodels.tsa.arima.model import ARIMA

from pycausalimpact.models.statsmodels import StatsmodelsAdapter
from pycausalimpact.models.prophet import ProphetAdapter
from pycausalimpact.models.sktime import SktimeAdapter


def test_statsmodels_adapter_fit_predict_interval():
    y = pd.Series(range(10))
    X = pd.DataFrame({"x": range(10)})

    adapter = StatsmodelsAdapter(partial(ARIMA, order=(1, 0, 0)))
    assert adapter.fit(y[:8], X=X[:8]) is adapter

    pred = adapter.predict(steps=2, X=X[8:])
    manual = ARIMA(y[:8], exog=X[:8], order=(1, 0, 0)).fit()
    expected = manual.forecast(steps=2, exog=X[8:])
    pd.testing.assert_series_equal(pred, expected)

    interval = adapter.predict_interval(steps=2, X=X[8:])
    assert list(interval.columns) == ["lower", "upper"]
    assert len(interval) == 2


def test_statsmodels_adapter_no_exog():
    y = pd.Series(range(10))
    empty_X = pd.DataFrame(index=y.index)

    adapter = StatsmodelsAdapter(partial(ARIMA, order=(1, 0, 0)))
    adapter.fit(y, X=empty_X)

    # After fitting with empty X, exogenous data shouldn't be required
    assert adapter._use_exog is False
    pred = adapter.predict(steps=2)
    interval = adapter.predict_interval(steps=2)

    assert len(pred) == 2
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
        return pd.DataFrame(
            {
                "ds": df["ds"],
                "yhat": range(n),
                "yhat_lower": range(n),
                "yhat_upper": range(n),
            }
        )


def test_prophet_adapter_fit_predict_interval():
    y = pd.Series(range(5))
    X = pd.DataFrame({"x": range(5)})
    future_X = pd.DataFrame({"x": [5, 6]})

    model = MockProphet()
    adapter = ProphetAdapter(model)

    assert adapter.fit(y, X=X) is adapter
    pred = adapter.predict(steps=2, X=future_X)
    interval = adapter.predict_interval(steps=2, X=future_X)

    assert "x" in model.fit_df.columns
    assert model.predict_df["x"].iloc[-2:].reset_index(drop=True).equals(future_X["x"])
    assert list(pred) == [5, 6]
    assert list(interval.columns) == ["lower", "upper"]


class MockSktimeIntervalModel:
    def __init__(self):
        self.fit_X = None
        self.predict_X = None

    def fit(self, y, X=None):
        self.fit_X = X

    def predict(self, fh, X=None):
        self.predict_X = X
        return pd.Series(range(len(fh)))

    def predict_interval(self, fh, X=None, coverage=0.95):
        index = range(len(fh))
        arrays = [["y"] * 2, [coverage] * 2, ["lower", "upper"]]
        columns = pd.MultiIndex.from_arrays(arrays)
        data = [[i, i + 1] for i in index]
        return pd.DataFrame(data, columns=columns, index=index)


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


def test_sktime_adapter_fit_predict_interval():
    y = pd.Series(range(5))
    X = pd.DataFrame({"x": range(5)})
    future_X = X.iloc[-2:]

    model = MockSktimeIntervalModel()
    adapter = SktimeAdapter(model)

    assert adapter.fit(y, X=X) is adapter
    pred = adapter.predict(steps=2, X=future_X)
    interval = adapter.predict_interval(steps=2, X=future_X)

    assert model.fit_X.equals(X)
    assert model.predict_X.equals(future_X)
    assert list(pred) == [0, 1]
    assert list(interval.columns) == ["lower", "upper"]
    assert len(interval) == 2


class MockSktimeQuantileModel:
    def __init__(self):
        self.fit_X = None
        self.predict_X = None

    def fit(self, y, X=None):
        self.fit_X = X

    def predict(self, fh, X=None):
        self.predict_X = X
        return pd.Series(range(len(fh)))

    def predict_quantiles(self, fh, X=None, alpha=None):
        self.predict_X = X
        index = range(len(fh))
        arrays = [["y"] * len(alpha), alpha]
        columns = pd.MultiIndex.from_arrays(arrays)
        data = [[-1, 1] for _ in index]
        return pd.DataFrame(data, columns=columns, index=index)


def test_sktime_adapter_predict_quantiles_interval():
    y = pd.Series(range(5))
    X = pd.DataFrame({"x": range(5)})
    future_X = X.iloc[-2:]

    model = MockSktimeQuantileModel()
    adapter = SktimeAdapter(model)

    adapter.fit(y, X=X)
    interval = adapter.predict_interval(steps=2, X=future_X)

    assert model.fit_X.equals(X)
    assert model.predict_X.equals(future_X)
    assert list(interval.columns) == ["lower", "upper"]
    assert interval["lower"].tolist() == [-1, -1]
    assert interval["upper"].tolist() == [1, 1]
