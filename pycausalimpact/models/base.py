from abc import ABC, abstractmethod
import pandas as pd


class BaseForecastModel(ABC):
    """
    Abstract base class for forecast model adapters.
    """

    @abstractmethod
    def fit(self, y: pd.Series, X: pd.DataFrame = None):
        """Fit the model on pre-period data."""
        pass

    @abstractmethod
    def predict(self, steps: int, X: pd.DataFrame = None):
        """Generate predictions for post-period."""
        pass

    def predict_interval(self, steps: int, X: pd.DataFrame = None, alpha: float = 0.05):
        """
        Optional: return prediction intervals if the model supports it.
        """
        raise NotImplementedError("Prediction intervals not supported for this model.")
