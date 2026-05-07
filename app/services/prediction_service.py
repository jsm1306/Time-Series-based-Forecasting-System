from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import holidays
import numpy as np
import pandas as pd
from app.config.settings import BASE_DIR, PROCESSED_DATA_DIR, get_logger
from app.models.lstm_forecaster import LSTMForecaster
from app.models.prophet_forecaster import ProphetForecaster
from app.models.sarima_forecaster import SarimaForecaster
from app.models.xgboost_forecaster import XGBoostForecaster
from app.services.model_registry import ModelRegistry


class PredictionService:
    """Service responsible for loading models and generating forecasts."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.registry = ModelRegistry(registry_path=BASE_DIR / "trained_models" / "model_registry.json")
        self.data_path = PROCESSED_DATA_DIR / "processed_timeseries.csv"
        self.holiday_calendar = holidays.US()

    def _load_state_data(self, state: str) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {self.data_path}")

        df = pd.read_csv(self.data_path, parse_dates=["Date"])
        state_df = df[df["State"] == state].copy()
        if state_df.empty:
            raise ValueError(f"No data available for state: {state}")

        state_df = state_df.sort_values("Date").reset_index(drop=True)
        if "is_holiday" in state_df.columns:
            state_df["is_holiday"] = state_df["is_holiday"].astype(int)
        return state_df

    def _build_future_dates(self, last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
        return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")

    def _holiday_flags(self, dates: pd.DatetimeIndex) -> List[int]:
        return [1 if date in self.holiday_calendar else 0 for date in dates]

    def _load_model(self, model_name: str, model_path: Path) -> Any:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_name == "SARIMA":
            model = SarimaForecaster()
        elif model_name == "Prophet":
            model = ProphetForecaster()
        elif model_name == "XGBoost":
            model = XGBoostForecaster()
        elif model_name == "LSTM":
            model = LSTMForecaster()
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        model.load_model(model_path)
        self.logger.info("Loaded %s model from %s", model_name, model_path)
        return model

    def _build_feature_row(self, totals: List[float], date: pd.Timestamp) -> Dict[str, Any]:
        lag_1 = totals[-1] if len(totals) >= 1 else 0.0
        lag_7 = totals[-7] if len(totals) >= 7 else totals[0] if totals else 0.0
        lag_30 = totals[-30] if len(totals) >= 30 else totals[0] if totals else 0.0
        recent = totals[-7:] if len(totals) >= 1 else [0.0]
        rolling_mean_7 = float(np.mean(recent))
        rolling_std_7 = float(np.std(recent, ddof=0))

        return {
            "lag_1": float(lag_1),
            "lag_7": float(lag_7),
            "lag_30": float(lag_30),
            "rolling_mean_7": rolling_mean_7,
            "rolling_std_7": rolling_std_7,
            "month": int(date.month),
            "week_of_year": int(date.isocalendar()[1]),
            "quarter": int(date.quarter),
            "is_holiday": int(date in self.holiday_calendar),
        }

    def _predict_xgboost(self, model: XGBoostForecaster, state_df: pd.DataFrame, periods: int) -> List[float]:
        totals = state_df["Total"].astype(float).tolist()
        future_dates = self._build_future_dates(state_df["Date"].iloc[-1], periods)
        predictions: List[float] = []

        for date in future_dates:
            feature_row = self._build_feature_row(totals, date)
            forecast = model.predict(pd.DataFrame([feature_row]))
            prediction = float(forecast[0])
            predictions.append(prediction)
            totals.append(prediction)

        return predictions

    def _predict_lstm(self, model: LSTMForecaster, state_df: pd.DataFrame, periods: int) -> List[float]:
        totals = state_df["Total"].astype(float).values.reshape(-1, 1)
        scaled_totals = model.scaler.transform(totals)
        last_targets = scaled_totals[-model.sequence_length:]
        last_holidays = state_df["is_holiday"].astype(float).values[-model.sequence_length:].reshape(-1, 1)
        last_sequence = np.concatenate([last_targets, last_holidays], axis=1)

        future_dates = self._build_future_dates(state_df["Date"].iloc[-1], periods)
        future_holidays = np.array(self._holiday_flags(future_dates)).reshape(-1, 1)

        predictions = model.predict(periods, last_sequence, future_features=future_holidays)
        return [float(value) for value in predictions]

    def _predict_prophet(self, model: ProphetForecaster, state_df: pd.DataFrame, periods: int) -> List[float]:
        future_dates = self._build_future_dates(state_df["Date"].iloc[-1], periods)
        future_df = pd.DataFrame({"Date": future_dates, "is_holiday": self._holiday_flags(future_dates)})
        predictions = model.predict(periods, future_regressors=future_df)
        return [float(value) for value in predictions]

    def _predict_sarima(self, model: SarimaForecaster, periods: int) -> List[float]:
        predictions = model.predict(periods)
        return [float(value) for value in predictions]

    def _sanitize_state(self, state: str) -> str:
        return state.strip().replace(" ", "_").lower()

    def _derive_model_path(self, model_name: str, state: str) -> Path:
        state_key = self._sanitize_state(state)
        extension = ".h5" if model_name == "LSTM" else ".pkl"
        return BASE_DIR / "trained_models" / f"{model_name.lower()}_{state_key}{extension}"

    def predict(self, state: str, forecast_periods: int = 8, model_name: Optional[str] = None) -> Dict[str, Any]:
        self.logger.info(
            "Generating forecast for state=%s periods=%d model_override=%s",
            state,
            forecast_periods,
            model_name,
        )
        if model_name:
            model_path = self._derive_model_path(model_name, state)
            if not model_path.exists():
                raise ValueError(f"Requested model file not found for {model_name} and state {state}")
        else:
            entry = self.registry.get_best_model_for_state(state)
            if entry is None:
                raise ValueError(f"No model registry entry found for state: {state}")
            model_name = entry["best_model"]
            model_path = Path(entry["model_path"])

        model = self._load_model(model_name, model_path)
        state_df = self._load_state_data(state)

        if model_name == "SARIMA":
            predictions = self._predict_sarima(model, forecast_periods)
        elif model_name == "Prophet":
            predictions = self._predict_prophet(model, state_df, forecast_periods)
        elif model_name == "XGBoost":
            predictions = self._predict_xgboost(model, state_df, forecast_periods)
        elif model_name == "LSTM":
            predictions = self._predict_lstm(model, state_df, forecast_periods)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        response = {
            "state": state,
            "model_used": model_name,
            "forecast_periods": forecast_periods,
            "predictions": predictions,
            "generated_at": datetime.utcnow().isoformat(),
        }
        self.logger.info("Forecast generated for state=%s using model=%s", state, model_name)
        return response
