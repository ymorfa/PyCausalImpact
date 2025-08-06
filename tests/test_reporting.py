import pathlib
import sys

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pycausalimpact import CausalImpactPy
from pycausalimpact.reporting import ReportGenerator


class MeanModel:
    def fit(self, X, y):
        self.mean_ = float(y.mean())
        self.fitted = self
        self.fittedvalues = pd.Series([self.mean_] * len(y), index=y.index)

    def predict(self, X):
        return pd.Series(self.mean_, index=X.index)


def _generate_results():
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6]})
    impact = CausalImpactPy(
        df,
        index=None,
        y=["y"],
        pre_period=(0, 2),
        post_period=(3, 5),
        model=MeanModel(),
    )
    return impact.run(n_sim=200)


def test_summary_table_structure_and_values():
    results = _generate_results()
    report = ReportGenerator(results)
    summary = report.generate(plot=False)

    expected_cols = [
        "observed",
        "predicted",
        "predicted_lower",
        "predicted_upper",
        "abs_effect",
        "abs_effect_lower",
        "abs_effect_upper",
        "rel_effect",
        "rel_effect_lower",
        "rel_effect_upper",
        "p_value",
        "causal_probability",
    ]

    assert list(summary.index) == ["average", "cumulative"]
    assert list(summary.columns) == expected_cols
    assert summary.loc["average", "observed"] == pytest.approx(5.0)
    assert summary.loc["average", "abs_effect"] == pytest.approx(3.0)
    assert summary.loc["cumulative", "abs_effect"] == pytest.approx(9.0)
    assert summary.loc["cumulative", "rel_effect"] == pytest.approx(1.5)
    assert summary.loc["cumulative", "p_value"] < 0.05
    assert summary.loc["cumulative", "causal_probability"] > 0.95


def test_plotting_runs_without_error():
    results = _generate_results()
    report = ReportGenerator(results)
    report.generate(plot=True)  # Should not raise


def test_narrative_generation():
    results = _generate_results()
    report = ReportGenerator(results)
    summary, narrative = report.generate(plot=False, narrative=True)

    assert isinstance(narrative, str)
    # narrative should mention average effect and significance
    assert "3.00" in narrative
    assert "statistically significant" in narrative
