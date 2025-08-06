import pandas as pd
from functools import partial

from statsmodels.tsa.arima.model import ARIMA

from pycausalimpact.models.statsmodels import StatsmodelsAdapter


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

