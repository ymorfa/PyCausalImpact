import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from pycausalimpact import CausalImpactPy
from pycausalimpact.models import TFPStructuralTimeSeries


@pytest.mark.tfp
@pytest.mark.parametrize("method", ["variational", "hmc"])
def test_tfp_adapter_inference_methods(method):
    np.random.seed(0)
    tf.random.set_seed(0)
    n = 30
    series = pd.Series(
        np.sin(np.linspace(0, 3 * np.pi, n)) + np.random.normal(scale=0.1, size=n)
    )
    data = pd.DataFrame({"y": series})
    pre_period = (0, 19)
    post_period = (20, 29)

    model = TFPStructuralTimeSeries(
        num_variational_steps=50,
        num_results=20,
        num_warmup_steps=20,
        inference_method=method,
    )
    impact = CausalImpactPy(
        data=data,
        index=None,
        y=["y"],
        pre_period=pre_period,
        post_period=post_period,
        model=model,
        inference_method=method,
    )
    results = impact.run()
    assert results.shape[0] == post_period[1] - post_period[0] + 1
    assert not results["predicted"].isna().any()
    assert (results["predicted_upper"] > results["predicted_lower"]).all()


@pytest.mark.tfp
def test_posterior_and_sample_predictions():
    np.random.seed(0)
    tf.random.set_seed(0)
    n = 20
    series = pd.Series(
        np.sin(np.linspace(0, 3 * np.pi, n)) + np.random.normal(scale=0.1, size=n)
    )
    model = TFPStructuralTimeSeries(num_variational_steps=20, num_results=10)
    model.fit(series)
    assert model.posterior_samples() is not None
    samples = model.predict_samples(5, n_samples=4)
    assert samples.shape == (4, 5)
