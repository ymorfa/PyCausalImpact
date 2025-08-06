import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pycausalimpact.utils import validate_periods, split_pre_post


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
