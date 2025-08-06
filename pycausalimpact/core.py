import pandas as pd
from typing import Optional, Tuple, List, Any
from .reporting import ReportGenerator
from .utils import validate_periods, split_pre_post


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
        :param data: Pandas DataFrame with target and optional control variables.
        :param index: Column name to set as time index (optional if already set).
        :param y: List of columns: first is target (y), rest are controls (X).
        :param pre_period: Tuple defining pre-intervention period (dates or indices).
        :param post_period: Tuple defining post-intervention period (dates or indices).
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
        self.pre_data, self.post_data = split_pre_post(self.data, self.pre_period, self.post_period)

    def run(self):
        """Fit the model on pre-period data and compute predictions for post-period."""
        # TODO: Implement training and prediction logic
        pass

    def _compute_effects(self, y_true: pd.Series, y_pred: pd.Series):
        """Compute pointwise and cumulative effects, with confidence intervals."""
        # TODO: Implement effect calculations and uncertainty estimation
        pass

    def summary(self, plot: bool = True):
        """Generate a textual and visual summary of the causal impact."""
        if self.results is None:
            raise ValueError("No results found. Run .run() first.")
        return ReportGenerator(self.results, alpha=self.alpha).generate(plot=plot)
