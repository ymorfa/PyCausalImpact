import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from pycausalimpact.models import ProphetAdapter, SktimeAdapter, TFPStructuralTimeSeries


@pytest.mark.backend("prophet")
@pytest.mark.skipif(ProphetAdapter is None, reason="prophet not installed")
def test_prophet_available():
    assert ProphetAdapter is not None


@pytest.mark.backend("sktime")
@pytest.mark.skipif(SktimeAdapter is None, reason="sktime not installed")
def test_sktime_available():
    assert SktimeAdapter is not None


@pytest.mark.backend("tfp")
@pytest.mark.heavy
@pytest.mark.skipif(TFPStructuralTimeSeries is None, reason="tfp not installed")
def test_tfp_available():
    assert TFPStructuralTimeSeries is not None
