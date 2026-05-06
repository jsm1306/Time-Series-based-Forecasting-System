from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

from app.models.base_forecaster import BaseForecaster


class SarimaForecaster(BaseForecaster):
    """SARIMA forecasting model for seasonal time series data."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
    ) -> None:
        super().__init__(model_name="SARIMA")
        self.order: Tuple[int, int, int] = order
        self.seasonal_order: Tuple[int, int, int, int] = seasonal_order

    def train(self, train_data: pd.DataFrame, target_column: str = "Total", **kwargs) -> None:
        """Train SARIMA model on the provided training data."""
        self.logger.info("Starting SARIMA training with order=%s, seasonal_order=%s", self.order, self.seasonal_order)

        try:
            y_train = train_data[target_column].values

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.model = SARIMAX(
                    y_train,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self.model = self.model.fit(disp=False, maxiter=200)

            self.logger.info("SARIMA training completed successfully")
        except Exception as error:
            self.logger.exception("SARIMA training failed")
            raise error

    def predict(self, steps: int) -> np.ndarray:
        """Generate predictions for the specified number of future steps."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() before predict().")

        self.logger.info("Generating SARIMA predictions for %d steps", steps)

        try:
            forecast = self.model.get_forecast(steps=steps)
            predictions = np.asarray(forecast.predicted_mean)
            self.logger.info("SARIMA predictions generated successfully")
            return predictions
        except Exception as error:
            self.logger.exception("SARIMA prediction failed")
            raise error

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics comparing true and predicted values."""
        self.logger.info("Evaluating SARIMA model")
        metrics = self.calculate_metrics(y_true, y_pred)
        self.logger.info("SARIMA metrics: RMSE=%.4f, MAE=%.4f, MAPE=%.4f", metrics["rmse"], metrics["mae"], metrics["mape"])
        return metrics

    def save_model(self, model_path: Path) -> None:
        """Persist the trained model to disk."""
        if self.model is None:
            raise ValueError("No trained model to save.")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("Saving SARIMA model to %s", model_path)

        try:
            self.model.save(str(model_path))
            self.logger.info("SARIMA model saved successfully")
        except Exception as error:
            self.logger.exception("Failed to save SARIMA model")
            raise error

    def load_model(self, model_path: Path) -> None:
        """Load a previously trained model from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.logger.info("Loading SARIMA model from %s", model_path)

        try:
            self.model = SARIMAX.load(str(model_path))
            self.logger.info("SARIMA model loaded successfully")
        except Exception as error:
            self.logger.exception("Failed to load SARIMA model")
            raise error
