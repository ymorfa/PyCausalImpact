import pandas as pd
from .base import BaseForecastModel

class StatsmodelsAdapter(BaseForecastModel):
    """
    Adapter for Statsmodels forecasting models.
    """

    def __init__(self, model):
        self.model = model
        self.fitted = None

    def fit(self, y: pd.Series, X: pd.DataFrame = None):
        self.fitted = self.model.fit()
        return self

    def predict(self, steps: int, X: pd.DataFrame = None):
        return self.fitted.forecast(steps=steps)

    def predict_interval(self, steps: int, X: pd.DataFrame = None, alpha: float = 0.05):
        forecast = self.fitted.get_forecast(steps=steps)
        return forecast.conf_int(alpha=alpha)
