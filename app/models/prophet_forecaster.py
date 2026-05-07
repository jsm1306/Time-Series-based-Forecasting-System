from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import pickle
from prophet import Prophet

from app.models.base_forecaster import BaseForecaster


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet forecasting model for time series with seasonality."""

    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True) -> None:
        super().__init__(model_name="Prophet")
        self.yearly_seasonality: bool = yearly_seasonality
        self.weekly_seasonality: bool = weekly_seasonality

    def train(
        self,
        train_data: pd.DataFrame,
        target_column: str = "Total",
        date_column: str = "Date",
        regressor_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Train Prophet model on the provided training data."""
        self.logger.info("Starting Prophet training")

        try:
            columns = [date_column, target_column]
            if regressor_columns:
                columns += regressor_columns

            prophet_df = train_data[columns].copy()
            prophet_df.columns = ["ds", "y"] + (regressor_columns or [])
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

            self.model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                interval_width=0.95,
            )

            if regressor_columns:
                for regressor in regressor_columns:
                    self.model.add_regressor(regressor)

            self.model.fit(prophet_df)
            self.logger.info("Prophet training completed successfully")
        except Exception as error:
            self.logger.exception("Prophet training failed")
            raise error

    def predict(self, steps: int, future_regressors: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate predictions for the specified number of future steps."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() before predict().")

        self.logger.info("Generating Prophet predictions for %d steps", steps)

        try:
            if future_regressors is not None:
                future = future_regressors.copy()
                if "Date" in future.columns:
                    future = future.rename(columns={"Date": "ds"})
                future["ds"] = pd.to_datetime(future["ds"])
            else:
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
            joblib.dump(self.model, model_path)
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
            self.model = joblib.load(model_path)
            self.logger.info("Prophet model loaded successfully")
        except Exception as load_error:
            self.logger.exception("Failed to load Prophet model")

            file_contents = ""
            try:
                with open(model_path, "r", encoding="utf-8", errors="ignore") as file:
                    file_contents = file.read(256)
            except Exception:
                file_contents = ""

            if isinstance(load_error, (pickle.UnpicklingError, EOFError, AttributeError, KeyError)) or file_contents.lstrip().startswith("{"):
                raise ValueError(
                    f"Unable to load Prophet model from {model_path}. "
                    "The file appears to be an unsupported or non-serialized format. "
                    "Please retrain the Prophet model and save it using the current code's `joblib.dump(model, model_path)` format."
                ) from load_error

            raise ValueError(
                f"Unable to load Prophet model from {model_path}. "
                "If the file is not compatible with the current Prophet loader, retrain or regenerate it."
            ) from load_error
