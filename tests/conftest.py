import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Random number generator with a fixed seed."""
    return np.random.default_rng(42)


@pytest.fixture
def series_y(rng: np.random.Generator) -> pd.Series:
    """Synthetic time series with a positive shift in the post period."""
    n = 80
    shift_at = 50
    shift = 0.7
    base = rng.normal(size=n)
    y = base.copy()
    y[shift_at:] += shift
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(y, index=index)


@pytest.fixture
def exog_X(rng: np.random.Generator, series_y: pd.Series) -> pd.DataFrame:
    """Exogenous features correlated with ``series_y`` but without the shift."""
    n = len(series_y)
    baseline = series_y.to_numpy().copy()
    baseline[50:] -= 0.7  # remove intervention effect
    noise = rng.normal(scale=0.1, size=n)
    x1 = baseline + noise
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}, index=series_y.index)


@pytest.fixture
def pre_post_periods(series_y: pd.Series) -> tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]]:
    """Default pre/post intervention periods."""
    split = 50
    index = series_y.index
    return (index[0], index[split - 1]), (index[split], index[-1])
