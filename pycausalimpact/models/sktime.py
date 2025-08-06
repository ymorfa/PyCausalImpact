import pandas as pd
from .base import BaseForecastModel

class SktimeAdapter(BaseForecastModel):
    """
    Adapter for sktime forecasters.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, y: pd.Series, X: pd.DataFrame = None):
        self.model.fit(y, X=X)
        return self

    def predict(self, steps: int, X: pd.DataFrame = None):
        return self.model.predict(fh=range(1, steps+1), X=X)
