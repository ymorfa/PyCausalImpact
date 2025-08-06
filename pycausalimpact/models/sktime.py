import pandas as pd
from .base import BaseForecastModel


class SktimeAdapter(BaseForecastModel):
    """
    Adapter for sktime forecasters.
    """

    def __init__(self, model):
        self.model = model
        self._use_exog = False

    def fit(self, y: pd.Series, X: pd.DataFrame = None):
        self.model.fit(y, X=X)
        self._use_exog = X is not None
        return self

    def predict(self, steps: int, X: pd.DataFrame = None):
        if self._use_exog and X is None:
            raise ValueError("Exogenous data must be provided for prediction.")
        return self.model.predict(fh=range(1, steps + 1), X=X)

    def predict_interval(self, steps: int, X: pd.DataFrame = None, alpha: float = 0.05):
        if self._use_exog and X is None:
            raise ValueError("Exogenous data must be provided for prediction.")

        fh = range(1, steps + 1)

        if hasattr(self.model, "predict_interval"):
            intervals = self.model.predict_interval(fh=fh, X=X, coverage=1 - alpha)
            first_var = intervals.columns.get_level_values(0)[0]
            coverage = intervals.columns.get_level_values(1)[0]
            df = intervals[first_var, coverage]
            df.columns = ["lower", "upper"]
            return df

        if hasattr(self.model, "predict_quantiles"):
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2
            quantiles = self.model.predict_quantiles(
                fh=fh, X=X, alpha=[lower_q, upper_q]
            )
            first_var = quantiles.columns.get_level_values(0)[0]
            df = quantiles[first_var][[lower_q, upper_q]]
            df.columns = ["lower", "upper"]
            return df

        raise NotImplementedError(
            "Prediction intervals not supported by this sktime model."
        )
