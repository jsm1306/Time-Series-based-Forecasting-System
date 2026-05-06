import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from app.config.settings import get_logger


class DataPreprocessor:
    """Preprocess raw time series data for weekly forecasting."""

    def __init__(self, required_columns: Optional[List[str]] = None, logger: Optional[logging.Logger] = None) -> None:
        self.required_columns: List[str] = required_columns or ["State", "Date", "Total", "Category"]
        self.logger = logger or get_logger(self.__class__.__name__)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full preprocessing sequence for the raw dataset."""
        df = self._validate_input(df)
        df = self._clean_data(df)
        df = self._aggregate_weekly(df)
        return df

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input dataset structure before preprocessing."""
        missing_columns = [column for column in self.required_columns if column not in df.columns]
        if missing_columns:
            message = f"Missing required preprocessing columns: {missing_columns}"
            self.logger.error(message)
            raise ValueError(message)
        return df.copy()

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and remove duplicate rows."""
        df = df.copy()
        df = df.dropna(subset=["State", "Date"]).reset_index(drop=True)
        df["Total"] = (
            df["Total"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
        df["Total"] = df["Total"].fillna(0.0)

        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        after = len(df)
        self.logger.info("Removed %d duplicate rows", before - after)
        return df

    def _aggregate_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate records by state, date, and category without generating artificial rows."""
        df = df.copy()

        grouped = (
            df.groupby(["State", "Date", "Category"], as_index=False)
            .agg(Total=("Total", "sum"))
        )

        grouped = grouped.sort_values(["State", "Category", "Date"]).reset_index(drop=True)
        grouped["week_start"] = grouped["Date"]
        return grouped
