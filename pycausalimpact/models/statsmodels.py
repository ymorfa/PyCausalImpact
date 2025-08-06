import pandas as pd
from .base import BaseForecastModel

class StatsmodelsAdapter(BaseForecastModel):
    """
    Adapter for Statsmodels forecasting models.
    """

    def __init__(self, model):
        self.model = model
        self.fitted = None
        self._use_exog = False

    def fit(self, y: pd.Series, X: pd.DataFrame = None):
        model = self.model(y, exog=X)
        self.fitted = model.fit()
        self._use_exog = X is not None
        return self

    def predict(self, steps: int, X: pd.DataFrame = None):
        if self._use_exog and X is None:
            raise ValueError("Exogenous data must be provided for prediction.")
        return self.fitted.forecast(steps=steps, exog=X)

    def predict_interval(self, steps: int, X: pd.DataFrame = None, alpha: float = 0.05):
        if not hasattr(self.fitted, "get_forecast"):
            raise NotImplementedError(
                "Prediction intervals not supported by this statsmodels model."
            )
        if self._use_exog and X is None:
            raise ValueError("Exogenous data must be provided for prediction.")
        forecast = self.fitted.get_forecast(steps=steps, exog=X)
        ci = forecast.conf_int(alpha=alpha)
        ci.columns = ["lower", "upper"]
        return ci
