import pathlib
import sys

import pandas as pd
import pytest
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pycausalimpact import CausalImpactPy
from pycausalimpact.utils import validate_periods, split_pre_post
from pycausalimpact.models.statsmodels import StatsmodelsAdapter
from pycausalimpact.models.sktime import SktimeAdapter
from statsmodels.tsa.arima.model import ARIMA


def _create_df():
    index = pd.RangeIndex(6)
    return pd.DataFrame({"y": range(6)}, index=index)


def test_split_pre_post_valid_periods():
    df = _create_df()
    pre = (0, 2)
    post = (3, 5)
    validate_periods(df, pre, post)
    pre_df, post_df = split_pre_post(df, pre, post)
    assert list(pre_df.index) == [0, 1, 2]
    assert list(post_df.index) == [3, 4, 5]


def test_validate_periods_overlap():
    df = _create_df()
    pre = (0, 3)
    post = (2, 5)
    with pytest.raises(ValueError):
        validate_periods(df, pre, post)


def test_validate_periods_out_of_range():
    df = _create_df()
    pre = (-1, 2)
    post = (3, 5)
    with pytest.raises(ValueError):
        validate_periods(df, pre, post)


class MeanModel:
    def fit(self, X, y):
        self.mean_ = float(y.mean())
        self.fitted = self
        self.fittedvalues = pd.Series([self.mean_] * len(y), index=y.index)

    def predict(self, X):
        return pd.Series(self.mean_, index=X.index)


def test_effects_and_ci_dimensions():
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6]})
    pre = (0, 2)
    post = (3, 5)
    impact = CausalImpactPy(
        df,
        index=None,
        y=["y"],
        pre_period=pre,
        post_period=post,
        model=MeanModel(),
    )
    impact.run(n_sim=200)
    res = impact.results

    assert list(res.index) == [3, 4, 5]
    assert res["point_effect"].tolist() == [2, 3, 4]
    assert res["cumulative_effect"].tolist() == [2, 5, 9]
    assert res["relative_effect"].round(2).tolist() == [1.0, 1.5, 2.0]

    for col in [
        "predicted_lower",
        "predicted_upper",
        "point_effect_lower",
        "point_effect_upper",
    ]:
        assert col in res.columns
        assert len(res[col]) == len(res)


def test_random_state_reproducibility():
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6]})
    pre = (0, 2)
    post = (3, 5)
    impact1 = CausalImpactPy(
        df,
        index=None,
        y=["y"],
        pre_period=pre,
        post_period=post,
        model=MeanModel(),
    )
    impact2 = CausalImpactPy(
        df,
        index=None,
        y=["y"],
        pre_period=pre,
        post_period=post,
        model=MeanModel(),
    )
    res1 = impact1.run(n_sim=200, random_state=42)
    res2 = impact2.run(n_sim=200, random_state=42)
    pd.testing.assert_frame_equal(res1, res2)


def test_statsmodels_adapter_integration():
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    pre = (0, 5)
    post = (6, 9)
    model = partial(ARIMA, order=(1, 0, 0))
    adapter = StatsmodelsAdapter(model)

    impact = CausalImpactPy(
        df,
        index=None,
        y=["y"],
        pre_period=pre,
        post_period=post,
        model=adapter,
    )

    res = impact.run()
    assert len(res) == (post[1] - post[0] + 1)
    assert "predicted_lower" in res.columns
    assert "predicted_upper" in res.columns


class MockSktimeQuantileModel:
    def __init__(self):
        self.fit_X = None
        self.predict_X = None

    def fit(self, y, X=None):
        self.fit_X = X

    def predict(self, fh, X=None):
        self.predict_X = X
        return pd.Series([0] * len(fh))

    def predict_quantiles(self, fh, X=None, alpha=None):
        self.predict_X = X
        index = range(len(fh))
        arrays = [["y"] * len(alpha), alpha]
        columns = pd.MultiIndex.from_arrays(arrays)
        data = [[-1, 1] for _ in index]
        return pd.DataFrame(data, columns=columns, index=index)


def test_sktime_adapter_integration():
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6]})
    pre = (0, 2)
    post = (3, 5)
    adapter = SktimeAdapter(MockSktimeQuantileModel())

    impact = CausalImpactPy(
        df,
        index=None,
        y=["y"],
        pre_period=pre,
        post_period=post,
        model=adapter,
    )

    res = impact.run()
    assert res["predicted_lower"].tolist() == [-1, -1, -1]
    assert res["predicted_upper"].tolist() == [1, 1, 1]


def test_causalimpact_no_exog_statsmodels():
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6]})
    pre = (0, 2)
    post = (3, 5)
    adapter = StatsmodelsAdapter(partial(ARIMA, order=(1, 0, 0)))

    impact = CausalImpactPy(
        df,
        index=None,
        y=["y"],
        pre_period=pre,
        post_period=post,
        model=adapter,
    )

    res = impact.run()
    assert len(res) == (post[1] - post[0] + 1)
    assert impact.model._use_exog is False
