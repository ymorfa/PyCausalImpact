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
        # Treat empty DataFrames as missing exogenous data
        if isinstance(X, pd.DataFrame) and X.shape[1] == 0:
            X = None

        if X is not None:
            model = self.model(y, exog=X)
        else:
            model = self.model(y)

        self.fitted = model.fit()
        self._use_exog = X is not None
        return self

    def predict(self, steps: int, X: pd.DataFrame = None):
        if isinstance(X, pd.DataFrame) and X.shape[1] == 0:
            X = None

        if self._use_exog:
            if X is None:
                raise ValueError("Exogenous data must be provided for prediction.")
            return self.fitted.forecast(steps=steps, exog=X)
        return self.fitted.forecast(steps=steps)

    def predict_interval(self, steps: int, X: pd.DataFrame = None, alpha: float = 0.05):
        if not hasattr(self.fitted, "get_forecast"):
            raise NotImplementedError(
                "Prediction intervals not supported by this statsmodels model."
            )
        if isinstance(X, pd.DataFrame) and X.shape[1] == 0:
            X = None

        if self._use_exog:
            if X is None:
                raise ValueError("Exogenous data must be provided for prediction.")
            forecast = self.fitted.get_forecast(steps=steps, exog=X)
        else:
            forecast = self.fitted.get_forecast(steps=steps)
        ci = forecast.conf_int(alpha=alpha)
        ci.columns = ["lower", "upper"]
        return ci
