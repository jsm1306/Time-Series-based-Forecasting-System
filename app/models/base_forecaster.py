from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from app.config.settings import get_logger


class BaseForecaster(ABC):
    """Abstract base class defining the standard forecasting interface."""

    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name
        self.logger = get_logger(self.__class__.__name__)
        self.model = None

    @abstractmethod
    def train(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Train the forecasting model on the provided training data."""
        pass

    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """Generate predictions for the specified number of future steps."""
        pass

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics comparing true and predicted values."""
        pass

    @abstractmethod
    def save_model(self, model_path: Path) -> None:
        """Persist the trained model to disk."""
        pass

    @abstractmethod
    def load_model(self, model_path: Path) -> None:
        """Load a previously trained model from disk."""
        pass

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard forecasting metrics."""
        rmse: float = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae: float = float(mean_absolute_error(y_true, y_pred))
        mape: float = float(mean_absolute_percentage_error(y_true, y_pred))

        return {"rmse": rmse, "mae": mae, "mape": mape}

    def train_test_split(
        self, data: pd.DataFrame, target_column: str = "Total", train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Split time series data into train and validation sets without shuffling."""
        split_point: int = int(len(data) * train_ratio)

        train_data = data.iloc[:split_point].copy()
        val_data = data.iloc[split_point:].copy()

        y_train = train_data[target_column].values
        y_val = val_data[target_column].values

        return train_data, val_data, y_train, y_val
