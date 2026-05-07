import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from app.config.settings import PROCESSED_DATA_DIR, get_logger
from app.services.data_loader import DataLoader
from app.services.feature_engineer import FeatureEngineer
from app.services.preprocessor import DataPreprocessor


class DataPipeline:
    """Orchestrate data loading, preprocessing, and feature engineering."""

    def __init__(
        self,
        loader: Optional[DataLoader] = None,
        preprocessor: Optional[DataPreprocessor] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
    ) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.loader = loader or DataLoader()
        self.preprocessor = preprocessor or DataPreprocessor()
        self.feature_engineer = feature_engineer or FeatureEngineer()

    def run(self) -> pd.DataFrame:
        """Execute the end-to-end data pipeline and return the processed DataFrame."""
        self.logger.info("Starting data pipeline")

        raw_df = self.loader.load()
        preprocessed_df = self.preprocessor.process(raw_df)
        featured_df = self.feature_engineer.process(preprocessed_df)
        processed_df = self._save(featured_df)

        self.logger.info(
            "Data pipeline complete. Filled %d missing weekly data points.",
            self.preprocessor.missing_week_fill_count,
        )
        return processed_df

    def _save(self, df: pd.DataFrame) -> pd.DataFrame:
        """Persist the processed dataset to the configured output directory."""
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path: Path = PROCESSED_DATA_DIR / "processed_timeseries.csv"
        df.to_csv(output_path, index=False)
        self.logger.info("Saved processed dataset to %s", output_path)
        return df


if __name__ == "__main__":
    pipeline = DataPipeline()
    processed_df = pipeline.run()
    print(f"Processed dataset shape: {processed_df.shape}")    
    print(f"Filled missing weekly data points: {pipeline.preprocessor.missing_week_fill_count}")    
    print(processed_df.groupby(["State", "Category"]).size().head())
