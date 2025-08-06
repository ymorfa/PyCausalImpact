import pandas as pd
from .base import BaseForecastModel

class ProphetAdapter(BaseForecastModel):
    """
    Adapter for Facebook Prophet model.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, y: pd.Series, X: pd.DataFrame = None):
        df = pd.DataFrame({"ds": y.index, "y": y})
        if X is not None:
            df = pd.concat([df, X.reset_index(drop=True)], axis=1)
        self.model.fit(df)
        return self

    def predict(self, steps: int, X: pd.DataFrame = None):
        future = self.model.make_future_dataframe(periods=steps, freq='D')
        forecast = self.model.predict(future)
        return forecast["yhat"].iloc[-steps:]

    def predict_interval(self, steps: int, X: pd.DataFrame = None, alpha: float = 0.05):
        future = self.model.make_future_dataframe(periods=steps, freq='D')
        forecast = self.model.predict(future)
        return forecast[["yhat_lower", "yhat_upper"]].iloc[-steps:]
