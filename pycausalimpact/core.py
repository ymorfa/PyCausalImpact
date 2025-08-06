"""Core functionality for the CausalImpactPy package."""

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .reporting import ReportGenerator
from .utils import split_pre_post, validate_periods


class CausalImpactPy:
    """
    Main class for causal impact estimation using any forecasting model.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        index: Optional[str],
        y: List[str],
        pre_period: Tuple,
        post_period: Tuple,
        model: Any,
        alpha: float = 0.05,
    ):
        """
        :param data: Pandas DataFrame with target and optional control vars.
        :param index: Column name to set as time index (optional if set).
        :param y: List of columns: first is target (y), rest are controls (X).
        :param pre_period: Tuple for pre-intervention period (dates or idx).
        :param post_period: Tuple for post-intervention period (dates or idx).
        :param model: Forecasting model instance with fit/predict interface.
        :param alpha: Significance level for confidence intervals.
        """
        self.data = data.copy()
        self.index = index
        self.y_cols = y
        self.pre_period = pre_period
        self.post_period = post_period
        self.model = model
        self.alpha = alpha

        self.results: Optional[pd.DataFrame] = None

        self._prepare_data()

    def _prepare_data(self):
        """Set index, validate inputs, and split into pre and post periods."""
        if self.index:
            self.data = self.data.set_index(self.index)
        validate_periods(self.data, self.pre_period, self.post_period)
        self.pre_data, self.post_data = split_pre_post(
            self.data, self.pre_period, self.post_period
        )

    def run(self, n_sim: int = 1000):
        """Fit model and generate counterfactual predictions for post period.

        Parameters
        ----------
        n_sim: int, optional
            Number of bootstrap simulations to use when model lacks prediction
            intervals. Defaults to ``1000``.
        """

        y_col = self.y_cols[0]
        x_cols = self.y_cols[1:]

        pre_y = self.pre_data[y_col]
        if x_cols:
            pre_X = self.pre_data[x_cols]
        else:
            pre_X = pd.DataFrame(index=pre_y.index)

        post_y = self.post_data[y_col]

        if x_cols:
            post_X = self.post_data[x_cols]
        else:
            post_X = pd.DataFrame(index=post_y.index)

        try:
            self.model.fit(pre_y, pre_X)
        except Exception:
            # Fall back to scikit-learn style (X, y)
            self.model.fit(pre_X, pre_y)

        try:
            y_pred_post = pd.Series(
                self.model.predict(len(post_y), post_X),
                index=post_y.index,
            )
        except TypeError:
            y_pred_post = pd.Series(
                self.model.predict(post_X),
                index=post_y.index,
            )

        # Residuals for bootstrap simulations
        pre_pred = None
        hasattr_fitted = hasattr(self.model, "fitted")
        if hasattr_fitted and hasattr(self.model.fitted, "fittedvalues"):
            pre_pred = pd.Series(
                self.model.fitted.fittedvalues,
                index=pre_y.index,
            )
        if pre_pred is None:
            try:
                pre_pred = pd.Series(
                    self.model.predict(len(pre_y), pre_X),
                    index=pre_y.index,
                )
            except Exception:  # pragma: no cover
                pre_pred = pd.Series(
                    np.repeat(pre_y.mean(), len(pre_y)),
                    index=pre_y.index,
                )
        residuals = pre_y - pre_pred

        y_pred_lower = y_pred_upper = None

        if hasattr(self.model, "predict_interval"):
            try:
                intervals = self.model.predict_interval(
                    len(post_y), post_X, alpha=self.alpha
                )
                if isinstance(intervals, pd.DataFrame):
                    lower = intervals.iloc[:, 0]
                    upper = intervals.iloc[:, -1]
                elif isinstance(intervals, (list, tuple)) and len(intervals) == 2:
                    lower, upper = intervals
                else:  # pragma: no cover
                    lower = intervals[0]
                    upper = intervals[1]
                y_pred_lower = pd.Series(lower, index=post_y.index)
                y_pred_upper = pd.Series(upper, index=post_y.index)
            except Exception:  # pragma: no cover
                y_pred_lower = y_pred_upper = None

        if y_pred_lower is None or y_pred_upper is None:
            y_pred_lower, y_pred_upper = self._bootstrap_intervals(
                residuals, y_pred_post, n_sim
            )

        self.results = self._compute_effects(
            post_y, y_pred_post, y_pred_lower, y_pred_upper
        )
        return self.results

    def _bootstrap_intervals(
        self,
        residuals: pd.Series,
        y_pred_post: pd.Series,
        n_sim: int,
    ) -> Tuple[pd.Series, pd.Series]:
        """Generate prediction intervals via residual bootstrapping."""

        rng = np.random.default_rng(0)
        sims = rng.choice(
            residuals.values,
            size=(n_sim, len(y_pred_post)),
            replace=True,
        )
        sims = sims + y_pred_post.values
        lower = np.quantile(sims, self.alpha / 2, axis=0)
        upper = np.quantile(sims, 1 - self.alpha / 2, axis=0)
        return pd.Series(lower, index=y_pred_post.index), pd.Series(
            upper, index=y_pred_post.index
        )

    def _compute_effects(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_pred_lower: Optional[pd.Series] = None,
        y_pred_upper: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Compute point, cumulative, average and relative effects with CIs."""

        df = pd.DataFrame({"observed": y_true, "predicted": y_pred})

        if y_pred_lower is not None and y_pred_upper is not None:
            df["predicted_lower"] = y_pred_lower
            df["predicted_upper"] = y_pred_upper

        df["point_effect"] = df["observed"] - df["predicted"]
        df["cumulative_effect"] = df["point_effect"].cumsum()
        df["relative_effect"] = df["point_effect"] / df["predicted"]
        df["average_effect"] = df["point_effect"].mean()

        if "predicted_lower" in df.columns and "predicted_upper" in df.columns:
            df["point_effect_lower"] = df["observed"] - df["predicted_upper"]
            df["point_effect_upper"] = df["observed"] - df["predicted_lower"]
            df["cumulative_effect_lower"] = df["point_effect_lower"].cumsum()
            df["cumulative_effect_upper"] = df["point_effect_upper"].cumsum()

        return df

    def summary(self, plot: bool = True):
        """Generate a textual and visual summary of the causal impact."""
        if self.results is None:
            raise ValueError("No results found. Run .run() first.")
        return ReportGenerator(self.results, alpha=self.alpha).generate(plot=plot)
