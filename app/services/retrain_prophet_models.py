import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import BASE_DIR, PROCESSED_DATA_DIR, get_logger
from app.models.prophet_forecaster import ProphetForecaster

logger = get_logger("retrain_prophet_models")


def load_processed_data() -> pd.DataFrame:
    data_path = PROCESSED_DATA_DIR / "processed_timeseries.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["Date"])
    logger.info("Loaded processed dataset with %d rows", len(df))
    return df


def get_states(df: pd.DataFrame) -> List[str]:
    states = sorted(df["State"].dropna().unique())
    logger.info("Found %d unique states", len(states))
    return states


def sanitize_state(state: str) -> str:
    return state.strip().replace(" ", "_").lower()


def prepare_state_df(df: pd.DataFrame, state: str) -> pd.DataFrame:
    state_df = df[df["State"] == state].copy()
    if state_df.empty:
        raise ValueError(f"No data available for state: {state}")

    state_df = state_df.sort_values("Date").reset_index(drop=True)
    if "is_holiday" in state_df.columns:
        state_df["is_holiday"] = state_df["is_holiday"].astype(int)
    else:
        state_df["is_holiday"] = 0

    return state_df


def train_and_save_prophet(state: str, state_df: pd.DataFrame, model_dir: Path) -> Path:
    model = ProphetForecaster()
    model.train(state_df, regressor_columns=["is_holiday"])

    state_key = sanitize_state(state)
    model_path = model_dir / f"prophet_{state_key}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)

    logger.info("Saved Prophet model for %s to %s", state, model_path)
    return model_path


def main() -> None:
    df = load_processed_data()
    states = get_states(df)
    model_dir = BASE_DIR / "trained_models"

    retrained_states = []
    failed_states = []

    for state in states:
        try:
            logger.info("Retraining Prophet model for state=%s", state)
            state_df = prepare_state_df(df, state)
            train_and_save_prophet(state, state_df, model_dir)
            retrained_states.append(state)
        except Exception as exc:
            logger.exception("Failed to retrain Prophet model for state=%s", state)
            failed_states.append((state, str(exc)))

    logger.info("Retrained Prophet models for %d states", len(retrained_states))
    if failed_states:
        logger.warning("Prophet model training failed for %d states", len(failed_states))
        for state, error in failed_states:
            logger.warning("State=%s error=%s", state, error)

    print("Retrained Prophet models:", len(retrained_states))
    if failed_states:
        print("Failed states:")
        for state, error in failed_states:
            print(f"- {state}: {error}")


if __name__ == "__main__":
    main()
