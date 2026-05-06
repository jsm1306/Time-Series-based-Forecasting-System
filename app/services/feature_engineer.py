import logging
from typing import Optional

import pandas as pd

from app.config.settings import get_logger


class FeatureEngineer:
    """Build lag, rolling, and date-based features for weekly forecasting."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or get_logger(self.__class__.__name__)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features and remove rows that cannot support lagged inputs."""
        df = df.copy()
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_date_features(df)
        df = self._add_holiday_flag(df)
        df = self._drop_incomplete_rows(df)
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged total values for multiple horizons."""
        grouped = df.groupby("State")["Total"]

        for lag in (1, 7, 30):
            df[f"lag_{lag}"] = grouped.shift(lag)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling mean and standard deviation features over past weeks."""
        grouped = df.groupby("State")["Total"]
        shifted = grouped.shift(1)

        df["rolling_mean_7"] = shifted.rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        df["rolling_std_7"] = (
            shifted.rolling(7, min_periods=1).std(ddof=0).fillna(0).reset_index(level=0, drop=True)
        )

        return df

    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar features derived from the weekly time index."""
        df["month"] = df["week_start"].dt.month
        df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
        df["quarter"] = df["week_start"].dt.quarter
        return df

    def _add_holiday_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a placeholder holiday indicator for future enrichment."""
        df["is_holiday"] = False
        return df

    def _drop_incomplete_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values created by lagged feature generation."""
        before = len(df)
        required = ["lag_1", "lag_7", "lag_30"]
        df = df.dropna(subset=required).reset_index(drop=True)
        self.logger.info("Dropped %d rows with incomplete lag features", before - len(df))
        return df
