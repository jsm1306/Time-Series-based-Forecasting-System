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
        self.missing_week_fill_count: int = 0

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
        """Aggregate records by state, date, and category and fill missing weekly dates."""
        df = df.copy()

        grouped = (
            df.groupby(["State", "Date", "Category"], as_index=False)
            .agg(Total=("Total", "sum"))
        )
        grouped = grouped.sort_values(["State", "Category", "Date"]).reset_index(drop=True)
        grouped["Date"] = pd.to_datetime(grouped["Date"])
        grouped["week_start"] = grouped["Date"]

        expanded_frames = []
        self.missing_week_fill_count = 0
        for (state, category), subset in grouped.groupby(["State", "Category"], sort=False):
            subset = subset.set_index("week_start").sort_index()
            full_range = pd.date_range(start=subset.index.min(), end=subset.index.max(), freq="7D")
            missing_rows = len(full_range) - len(subset.index)
            self.missing_week_fill_count += missing_rows
            subset = subset.reindex(full_range)
            subset["State"] = state
            subset["Category"] = category
            subset["Date"] = subset.index
            subset["Total"] = subset["Total"].interpolate(method="linear", limit_direction="both")
            subset["Total"] = subset["Total"].fillna(0.0)
            expanded_frames.append(subset.reset_index(drop=True))

        result = pd.concat(expanded_frames, ignore_index=True)
        result = result[["State", "Date", "Category", "Total"]]
        result["week_start"] = result["Date"]
        return result
