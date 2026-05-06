from pathlib import Path
from typing import List, Optional

import pandas as pd

from app.config.settings import RAW_DATA_PATH, get_logger


class DataLoader:
    """Load and validate raw time series data from the configured source."""

    def __init__(self, raw_data_path: Path = RAW_DATA_PATH, required_columns: Optional[List[str]] = None) -> None:
        self.raw_data_path: Path = raw_data_path
        self.required_columns: List[str] = required_columns or ["State", "Date", "Total", "Category"]
        self.logger = get_logger(self.__class__.__name__)

    def load(self) -> pd.DataFrame:
        """Load the raw CSV dataset and prepare the date column."""
        self.logger.info("Loading raw dataset from %s", self.raw_data_path)

        if not self.raw_data_path.exists():
            message = f"Raw data file not found at path: {self.raw_data_path}"
            self.logger.error(message)
            raise FileNotFoundError(message)

        try:
            df = pd.read_csv(self.raw_data_path)
            self._validate_columns(df)
            df = self._prepare_dates(df)
            df = df.sort_values(by="Date").reset_index(drop=True)
            self.logger.info("Loaded dataset with %d rows", len(df))
            return df
        except Exception as error:
            self.logger.exception("Failed to load raw dataset")
            raise error

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns are present in the dataset."""
        missing_columns = [column for column in self.required_columns if column not in df.columns]
        if missing_columns:
            message = f"Missing required columns: {missing_columns}"
            self.logger.error(message)
            raise ValueError(message)

    def _prepare_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the date column to datetime while handling mixed formats."""
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"],format="mixed", errors="coerce", dayfirst=True)

        invalid_dates = df["Date"].isna().sum()
        if invalid_dates:
            self.logger.warning("Found %d rows with invalid date values", int(invalid_dates))

        return df
