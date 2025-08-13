import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from .base import BaseForecastModel


tfd = tfp.distributions
sts = tfp.sts


class TFPStructuralTimeSeries(BaseForecastModel):
    """Adapter for TensorFlow Probability structural time series models."""

    def __init__(
        self,
        num_variational_steps: int = 200,
        num_results: int = 100,
        num_warmup_steps: int = 100,
        inference_method: str = "variational",
    ):
        """
        Parameters
        ----------
        num_variational_steps: int, optional
            Number of steps for variational inference optimisation.
        num_results: int, optional
            Number of posterior samples to draw.
        num_warmup_steps: int, optional
            Number of burn-in steps for HMC.
        inference_method: str, optional
            Inference algorithm to use. Must be ``"variational"`` or ``"hmc"``.
        """

        valid_methods = {"variational", "hmc"}
        if inference_method not in valid_methods:
            raise ValueError(f"inference_method must be one of {sorted(valid_methods)}")

        self.num_variational_steps = num_variational_steps
        self.num_results = num_results
        self.num_warmup_steps = num_warmup_steps
        self.inference_method = inference_method

        self._model = None
        self._surrogate_posterior = None
        self._posterior_samples = None
        self._y = None
        self._design_matrix = None

    def _build_model(self, y: np.ndarray, design_matrix: np.ndarray | None):
        components = [sts.LocalLinearTrend(observed_time_series=y)]
        if design_matrix is not None:
            components.append(sts.LinearRegression(design_matrix=design_matrix))
        return sts.Sum(components, observed_time_series=y)

    def fit(
        self,
        y: pd.Series,
        X: pd.DataFrame | None = None,
        inference_method: str | None = None,
    ):
        if isinstance(X, pd.DataFrame) and X.shape[1] == 0:
            X = None
        if inference_method is not None:
            valid_methods = {"variational", "hmc"}
            if inference_method not in valid_methods:
                raise ValueError(
                    f"inference_method must be one of {sorted(valid_methods)}"
                )
            self.inference_method = inference_method

        self._y = y.to_numpy(dtype=np.float32)
        self._design_matrix = X.to_numpy(dtype=np.float32) if X is not None else None
        self._model = self._build_model(self._y, self._design_matrix)

        if self.inference_method == "hmc":
            self._posterior_samples, _ = sts.fit_with_hmc(
                self._model,
                observed_time_series=self._y,
                num_results=self.num_results,
                num_warmup_steps=self.num_warmup_steps,
            )
        else:
            target_log_prob_fn = self._model.joint_log_prob(
                observed_time_series=self._y
            )
            surrogate_posterior = sts.build_factored_surrogate_posterior(self._model)
            optimizer = tf.optimizers.Adam(0.1)
            tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn=target_log_prob_fn,
                surrogate_posterior=surrogate_posterior,
                optimizer=optimizer,
                num_steps=self.num_variational_steps,
            )
            self._posterior_samples = surrogate_posterior.sample(self.num_results)
            self._surrogate_posterior = surrogate_posterior

        return self

    def _forecast_dist(self, steps: int, X: pd.DataFrame | None):
        if self._y is None:
            raise ValueError("Model must be fit before prediction.")

        if self._design_matrix is not None:
            if X is None:
                X_future = np.zeros(
                    (steps, self._design_matrix.shape[1]), dtype=np.float32
                )
            else:
                X_future = X.to_numpy(dtype=np.float32)
            design_matrix = np.vstack([self._design_matrix, X_future])
            y_full = np.concatenate([self._y, np.zeros(steps, dtype=np.float32)])
            model = self._build_model(y_full, design_matrix)
            return sts.forecast(
                model,
                self._y,
                self._posterior_samples,
                num_steps_forecast=steps,
            )
        return sts.forecast(
            self._model,
            self._y,
            self._posterior_samples,
            num_steps_forecast=steps,
        )

    def predict(self, steps: int, X: pd.DataFrame = None):
        forecast_dist = self._forecast_dist(steps, X)
        mean = forecast_dist.mean().numpy().squeeze(-1)
        return mean

    def predict_interval(self, steps: int, X: pd.DataFrame = None, alpha: float = 0.05):
        forecast_dist = self._forecast_dist(steps, X)
        samples = forecast_dist.sample(self.num_results).numpy().squeeze(-1)
        lower = np.quantile(samples, alpha / 2, axis=0)
        upper = np.quantile(samples, 1 - alpha / 2, axis=0)
        return pd.DataFrame({"lower": lower, "upper": upper})
