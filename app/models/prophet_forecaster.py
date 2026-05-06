from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from prophet import Prophet
import json

from app.models.base_forecaster import BaseForecaster


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet forecasting model for time series with seasonality."""

    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True) -> None:
        super().__init__(model_name="Prophet")
        self.yearly_seasonality: bool = yearly_seasonality
        self.weekly_seasonality: bool = weekly_seasonality

    def train(self, train_data: pd.DataFrame, target_column: str = "Total", date_column: str = "Date", **kwargs) -> None:
        """Train Prophet model on the provided training data."""
        self.logger.info("Starting Prophet training")

        try:
            prophet_df = train_data[[date_column, target_column]].copy()
            prophet_df.columns = ["ds", "y"]
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

            self.model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                interval_width=0.95,
            )
            self.model.fit(prophet_df)

            self.logger.info("Prophet training completed successfully")
        except Exception as error:
            self.logger.exception("Prophet training failed")
            raise error

    def predict(self, steps: int) -> np.ndarray:
        """Generate predictions for the specified number of future steps."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() before predict().")

        self.logger.info("Generating Prophet predictions for %d steps", steps)

        try:
            future = self.model.make_future_dataframe(periods=steps, freq="D")
            forecast = self.model.predict(future)
            predictions = forecast.tail(steps)["yhat"].values
            self.logger.info("Prophet predictions generated successfully")
            return predictions
        except Exception as error:
            self.logger.exception("Prophet prediction failed")
            raise error

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics comparing true and predicted values."""
        self.logger.info("Evaluating Prophet model")
        metrics = self.calculate_metrics(y_true, y_pred)
        self.logger.info("Prophet metrics: RMSE=%.4f, MAE=%.4f, MAPE=%.4f", metrics["rmse"], metrics["mae"], metrics["mape"])
        return metrics

    def save_model(self, model_path: Path) -> None:
        """Persist the trained model to disk."""
        if self.model is None:
            raise ValueError("No trained model to save.")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("Saving Prophet model to %s", model_path)

        try:
            with open(model_path, "w", encoding="utf-8") as f:
                json.dump(self.model.params, f, indent=4, default=str)
            self.logger.info("Prophet model saved successfully")
        except Exception as error:
            self.logger.exception("Failed to save Prophet model")
            raise error

    def load_model(self, model_path: Path) -> None:
        """Load a previously trained model from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.logger.info("Loading Prophet model from %s", model_path)

        try:
            with open(model_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            self.model = Prophet(**{k: v for k, v in params.items() if k in ["yearly_seasonality", "weekly_seasonality"]})
            self.logger.info("Prophet model loaded successfully")
        except Exception as error:
            self.logger.exception("Failed to load Prophet model")
            raise error
