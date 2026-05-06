from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

from app.models.base_forecaster import BaseForecaster


class LSTMForecaster(BaseForecaster):
    """LSTM recurrent neural network for sequence-based time series forecasting."""

    def __init__(
        self,
        sequence_length: int = 7,
        lstm_units: int = 50,
        dropout_rate: float = 0.2,
        epochs: int = 50,
        batch_size: int = 16,
    ) -> None:
        super().__init__(model_name="LSTM")
        self.sequence_length: int = sequence_length
        self.lstm_units: int = lstm_units
        self.dropout_rate: float = dropout_rate
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.scaler = MinMaxScaler()

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sequences from univariate time series data."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def train(self, train_data: pd.DataFrame, target_column: str = "Total", **kwargs) -> None:
        """Train LSTM model on the provided training data."""
        self.logger.info("Starting LSTM training with sequence_length=%d, lstm_units=%d", self.sequence_length, self.lstm_units)

        try:
            y_train = train_data[target_column].values.reshape(-1, 1)
            y_train_scaled = self.scaler.fit_transform(y_train)

            X_train, y_train_seq = self._create_sequences(y_train_scaled)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

            self.model = Sequential([
                LSTM(self.lstm_units, activation="relu", input_shape=(self.sequence_length, 1)),
                Dropout(self.dropout_rate),
                Dense(1),
            ])
            self.model.compile(optimizer="adam", loss="mse")

            early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
            self.model.fit(
                X_train,
                y_train_seq,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early_stop],
                verbose=0,
            )

            self.logger.info("LSTM training completed successfully")
        except Exception as error:
            self.logger.exception("LSTM training failed")
            raise error

    def predict(self, steps: int, last_sequence: np.ndarray) -> np.ndarray:
        """Generate predictions for the specified number of future steps."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() before predict().")

        self.logger.info("Generating LSTM predictions for %d steps", steps)

        try:
            current_sequence = last_sequence.copy()
            predictions = []

            for _ in range(steps):
                input_seq = current_sequence.reshape(1, self.sequence_length, 1)
                next_pred = self.model.predict(input_seq, verbose=0)[0, 0]
                predictions.append(next_pred)

                current_sequence = np.append(current_sequence[1:], next_pred)

            predictions_array = np.array(predictions).reshape(-1, 1)
            predictions_inverse = self.scaler.inverse_transform(predictions_array)

            self.logger.info("LSTM predictions generated successfully")
            return predictions_inverse.flatten()
        except Exception as error:
            self.logger.exception("LSTM prediction failed")
            raise error

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics comparing true and predicted values."""
        self.logger.info("Evaluating LSTM model")
        metrics = self.calculate_metrics(y_true, y_pred)
        self.logger.info("LSTM metrics: RMSE=%.4f, MAE=%.4f, MAPE=%.4f", metrics["rmse"], metrics["mae"], metrics["mape"])
        return metrics

    def save_model(self, model_path: Path) -> None:
        """Persist the trained model and scaler to disk."""
        if self.model is None:
            raise ValueError("No trained model to save.")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("Saving LSTM model to %s", model_path)

        try:
            model_weights_path = model_path.with_suffix(".h5")
            scaler_path = model_path.with_stem(model_path.stem + "_scaler").with_suffix(".pkl")

            self.model.save(str(model_weights_path))
            joblib.dump(self.scaler, scaler_path)
            self.logger.info("LSTM model and scaler saved successfully")
        except Exception as error:
            self.logger.exception("Failed to save LSTM model")
            raise error

    def load_model(self, model_path: Path) -> None:
        """Load a previously trained model and scaler from disk."""
        from tensorflow.keras.models import load_model

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.logger.info("Loading LSTM model from %s", model_path)

        try:
            model_weights_path = model_path.with_suffix(".h5")
            scaler_path = model_path.with_stem(model_path.stem + "_scaler").with_suffix(".pkl")

            self.model = load_model(str(model_weights_path))
            self.scaler = joblib.load(scaler_path)
            self.logger.info("LSTM model and scaler loaded successfully")
        except Exception as error:
            self.logger.exception("Failed to load LSTM model")
            raise error
