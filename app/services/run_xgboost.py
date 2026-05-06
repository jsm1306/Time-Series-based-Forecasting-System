import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_logger
from app.models.xgboost_forecaster import XGBoostForecaster


logger = get_logger("run_xgboost")


def load_processed_data(data_path: Path) -> pd.DataFrame:
    """Load the preprocessed dataset from the configured path."""
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["Date"])
    logger.info("Loaded processed dataset with %d rows", len(df))
    return df


def filter_dataset(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Filter the dataset for a single state."""
    filtered = df[df["State"] == state].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for State={state}")

    filtered = filtered.sort_values("Date").reset_index(drop=True)
    logger.info("Filtered dataset to %d rows for %s", len(filtered), state)
    return filtered


def validate_feature_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Ensure the required feature columns are available before training."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")


def split_time_series(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into chronological train and validation segments."""
    split_index = int(len(df) * train_ratio)
    if split_index < 1 or split_index == len(df):
        raise ValueError("Dataset size is too small for a proper train/validation split")

    train_df = df.iloc[:split_index].copy()
    val_df = df.iloc[split_index:].copy()
    logger.info("Split data into train (%d rows) and validation (%d rows)", len(train_df), len(val_df))
    return train_df, val_df


def run() -> None:
    """Run the XGBoost forecasting validation workflow end-to-end."""
    data_path = PROJECT_ROOT / "data" / "processed" / "processed_timeseries.csv"
    model_path = PROJECT_ROOT / "trained_models" / "xgboost_texas_beverages.pkl"

    try:
        df = load_processed_data(data_path)
        df = filter_dataset(df, state="Texas")

        df["is_holiday"] = df["is_holiday"].astype(int)
        forecaster = XGBoostForecaster()
        validate_feature_columns(df, forecaster.feature_columns + ["Total"])

        train_df, val_df = split_time_series(df)

        forecaster.train(train_df)
        predictions = forecaster.predict(val_df)
        metrics = forecaster.evaluate(val_df["Total"].values, predictions)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        forecaster.save_model(model_path)

        print(f"Train shape: {train_df.shape}")
        print(f"Validation shape: {val_df.shape}")
        print(f"Prediction sample: {predictions[:5].tolist()}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MAPE: {metrics['mape']:.4f}")
        print(f"Saved model to: {model_path}")
    except Exception as error:
        logger.exception("XGBoost pipeline failed")
        raise error


if __name__ == "__main__":
    run()
