import pandas as pd
from .base import BaseForecastModel


class ProphetAdapter(BaseForecastModel):
    """
    Adapter for Facebook Prophet model.
    """

    def __init__(self, model):
        self.model = model
        self._use_exog = False
        self._X = None

    def fit(self, y: pd.Series, X: pd.DataFrame = None):
        df = pd.DataFrame({"ds": y.index, "y": y.values}).reset_index(drop=True)
        if X is not None:
            X = X.reset_index(drop=True)
            if hasattr(self.model, "add_regressor"):
                for col in X.columns:
                    self.model.add_regressor(col)
            df = pd.concat([df, X], axis=1)
            self._use_exog = True
            self._X = X
        self.model.fit(df)
        return self

    def _prepare_future(self, steps: int, X: pd.DataFrame = None) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=steps, freq="D").reset_index(
            drop=True
        )
        if self._use_exog:
            if X is None:
                raise ValueError("Exogenous data must be provided for prediction.")
            future_exog = pd.concat([self._X, X.reset_index(drop=True)], axis=0)
            future = pd.concat([future, future_exog.reset_index(drop=True)], axis=1)
        return future

    def predict(self, steps: int, X: pd.DataFrame = None):
        future = self._prepare_future(steps, X=X)
        forecast = self.model.predict(future)
        return forecast["yhat"].iloc[-steps:].reset_index(drop=True)

    def predict_interval(self, steps: int, X: pd.DataFrame = None, alpha: float = 0.05):
        future = self._prepare_future(steps, X=X)
        forecast = self.model.predict(future)
        interval = forecast[["yhat_lower", "yhat_upper"]].iloc[-steps:]
        interval = interval.reset_index(drop=True)
        interval.columns = ["lower", "upper"]
        return interval
