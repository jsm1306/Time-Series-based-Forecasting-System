from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from app.models.base_forecaster import BaseForecaster


class XGBoostForecaster(BaseForecaster):
    """XGBoost supervised learning model for time series forecasting."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        random_state: int = 42,
    ) -> None:
        super().__init__(model_name="XGBoost")
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.max_depth: int = max_depth
        self.random_state: int = random_state
        self.feature_columns: List[str] = [
            "lag_1",
            "lag_7",
            "lag_30",
            "rolling_mean_7",
            "rolling_std_7",
            "month",
            "week_of_year",
            "quarter",
            "is_holiday",
        ]

    def train(self, train_data: pd.DataFrame, target_column: str = "Total", **kwargs) -> None:
        """Train XGBoost model using engineered lag and rolling features."""
        self.logger.info("Starting XGBoost training with n_estimators=%d, max_depth=%d", self.n_estimators, self.max_depth)

        try:
            X_train = train_data[self.feature_columns].copy()
            y_train = train_data[target_column].values

            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbosity=0,
            )
            self.model.fit(X_train, y_train)

            self.logger.info("XGBoost training completed successfully")
        except Exception as error:
            self.logger.exception("XGBoost training failed")
            raise error

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions using feature-engineered test data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() before predict().")

        self.logger.info("Generating XGBoost predictions")

        try:
            X_test = test_data[self.feature_columns].copy()
            predictions = self.model.predict(X_test)
            self.logger.info("XGBoost predictions generated successfully")
            return predictions
        except Exception as error:
            self.logger.exception("XGBoost prediction failed")
            raise error

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics comparing true and predicted values."""
        self.logger.info("Evaluating XGBoost model")
        metrics = self.calculate_metrics(y_true, y_pred)
        self.logger.info("XGBoost metrics: RMSE=%.4f, MAE=%.4f, MAPE=%.4f", metrics["rmse"], metrics["mae"], metrics["mape"])
        return metrics

    def save_model(self, model_path: Path) -> None:
        """Persist the trained model to disk."""
        if self.model is None:
            raise ValueError("No trained model to save.")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("Saving XGBoost model to %s", model_path)

        try:
            joblib.dump(self.model, model_path)
            self.logger.info("XGBoost model saved successfully")
        except Exception as error:
            self.logger.exception("Failed to save XGBoost model")
            raise error

    def load_model(self, model_path: Path) -> None:
        """Load a previously trained model from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.logger.info("Loading XGBoost model from %s", model_path)

        try:
            self.model = joblib.load(model_path)
            self.logger.info("XGBoost model loaded successfully")
        except Exception as error:
            self.logger.exception("Failed to load XGBoost model")
            raise error
