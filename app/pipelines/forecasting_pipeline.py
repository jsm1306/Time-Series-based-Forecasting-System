import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_logger
from app.models.lstm_forecaster import LSTMForecaster
from app.models.prophet_forecaster import ProphetForecaster
from app.models.sarima_forecaster import SarimaForecaster
from app.models.xgboost_forecaster import XGBoostForecaster
from app.services.model_registry import ModelRegistry


class ForecastingPipeline:
    """Orchestrate end-to-end forecasting model training, evaluation, and selection."""

    def __init__(
        self,
        processed_data_path: Path,
        model_output_dir: Path,
        registry_path: Path,
    ) -> None:
        self.processed_data_path: Path = processed_data_path
        self.model_output_dir: Path = model_output_dir
        self.registry = ModelRegistry(registry_path=registry_path)
        self.logger = get_logger(self.__class__.__name__)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load the processed time series dataset from disk."""
        if not self.processed_data_path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {self.processed_data_path}")

        df = pd.read_csv(self.processed_data_path, parse_dates=["Date"])
        self.logger.info("Loaded processed dataset with %d rows", len(df))
        return df

    def get_states(self, df: pd.DataFrame) -> List[str]:
        """Return the sorted list of unique states in the dataset."""
        states = sorted(df["State"].dropna().unique())
        self.logger.info("Found %d unique states", len(states))
        return states

    def split_time_series(self, df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into chronological train and validation sets."""
        split_index = int(len(df) * train_ratio)
        if split_index < 1 or split_index == len(df):
            raise ValueError("Dataset too small for chronological train/validation split")

        train_df = df.iloc[:split_index].copy()
        val_df = df.iloc[split_index:].copy()
        self.logger.debug("State split: %d train rows, %d validation rows", len(train_df), len(val_df))
        return train_df, val_df

    def _sanitize_state(self, state: str) -> str:
        """Create a filesystem-safe state identifier."""
        return state.strip().replace(" ", "_").lower()

    def _state_data(self, df: pd.DataFrame, state: str) -> pd.DataFrame:
        """Return chronological subset for the requested state."""
        state_df = df[df["State"] == state].copy()
        if state_df.empty:
            raise ValueError(f"No data available for state: {state}")

        state_df = state_df.sort_values("Date").reset_index(drop=True)
        if "is_holiday" in state_df.columns:
            state_df["is_holiday"] = state_df["is_holiday"].astype(int)
        return state_df

    def _save_model(self, model: Any, model_name: str, state: str) -> Path:
        """Save a trained model to disk using a consistent naming convention."""
        state_key = self._sanitize_state(state)
        extension = ".keras" if model_name == "LSTM" else ".pkl"
        model_path = self.model_output_dir / f"{model_name.lower()}_{state_key}{extension}"
        model.save_model(model_path)
        return model_path

    def _evaluate_model(self, model_name: str, model: Any, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Train, predict, and evaluate a single model instance."""
        try:
            if model_name == "SARIMA":
                model.train(train_df)
                predictions = model.predict(len(val_df))
            elif model_name == "Prophet":
                model.train(train_df, regressor_columns=["is_holiday"])
                future_regressors = val_df[["Date", "is_holiday"]].copy()
                predictions = model.predict(len(val_df), future_regressors=future_regressors)
            elif model_name == "XGBoost":
                model.train(train_df)
                predictions = model.predict(val_df)
            elif model_name == "LSTM":
                model.train(train_df, feature_columns=["is_holiday"])
                scaled_target = model.scaler.transform(train_df["Total"].values.reshape(-1, 1))
                last_sequence = np.concatenate(
                    [
                        scaled_target[-model.sequence_length:],
                        train_df["is_holiday"].astype(float).values[-model.sequence_length:].reshape(-1, 1),
                    ],
                    axis=1,
                )
                future_features = val_df["is_holiday"].astype(float).values.reshape(-1, 1)
                predictions = model.predict(len(val_df), last_sequence, future_features=future_features)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            metrics = model.evaluate(val_df["Total"].values, predictions)
            model_path = self._save_model(model, model_name, train_df["State"].iloc[0])
            metrics.update({"model_path": str(model_path.as_posix())})
            self.logger.info("Completed evaluation for %s on state %s", model_name, train_df["State"].iloc[0])
            return metrics
        except Exception as error:
            self.logger.exception("Model %s failed for state %s", model_name, train_df["State"].iloc[0])
            return None

    def train_models_for_state(self, state: str, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Train and evaluate all candidate models for a given state."""
        state_df = self._state_data(df, state)
        train_df, val_df = self.split_time_series(state_df)

        model_factories = {
            "SARIMA": SarimaForecaster,
            "Prophet": ProphetForecaster,
            "XGBoost": XGBoostForecaster,
            "LSTM": LSTMForecaster,
        }

        results: Dict[str, Dict[str, Any]] = {}
        for model_name, factory in model_factories.items():
            self.logger.info("Starting model %s for state %s", model_name, state)
            model = factory()
            metrics = self._evaluate_model(model_name, model, train_df, val_df)
            if metrics is not None:
                results[model_name] = metrics
        return results

    def select_best_model(self, results: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best model for a given state based on RMSE."""
        if not results:
            return None

        best_model = min(results.items(), key=lambda item: item[1]["rmse"])
        name, metrics = best_model
        return {"model_name": name, **metrics}

    def run(self) -> Dict[str, Any]:
        """Execute the full state-level forecasting pipeline."""
        df = self.load_data()
        states = self.get_states(df)

        summary: Dict[str, Any] = {
            "total_states": len(states),
            "successful_states": 0,
            "failed_states": 0,
            "selection_counts": {},
            "average_rmse": None,
            "average_mae": None,
            "average_mape": None,
        }
        all_metrics: List[Dict[str, float]] = []

        for state in states:
            self.logger.info("Processing state=%s", state)
            results = self.train_models_for_state(state, df)
            if not results:
                self.logger.warning("No successful models for state=%s", state)
                summary["failed_states"] += 1
                continue

            best = self.select_best_model(results)
            if best is None:
                summary["failed_states"] += 1
                continue

            self.registry.add_entry(
                state=state,
                best_model=best["model_name"],
                model_path=Path(best["model_path"]),
                rmse=best["rmse"],
                mae=best["mae"],
                mape=best["mape"],
            )
            self.registry.save_registry()
            all_metrics.append({"rmse": best["rmse"], "mae": best["mae"], "mape": best["mape"]})
            summary["successful_states"] += 1
            summary["selection_counts"][best["model_name"]] = summary["selection_counts"].get(best["model_name"], 0) + 1

        if all_metrics:
            summary["average_rmse"] = float(np.mean([m["rmse"] for m in all_metrics]))
            summary["average_mae"] = float(np.mean([m["mae"] for m in all_metrics]))
            summary["average_mape"] = float(np.mean([m["mape"] for m in all_metrics]))

        self.logger.info("Forecasting pipeline completed")
        return summary
