from typing import Dict, List, Optional
import pandas as pd

from app.config.settings import get_logger


class ModelEvaluator:
    """Compare and rank forecasting models based on evaluation metrics."""

    def __init__(self) -> None:
        self.results: Dict[str, Dict[str, float]] = {}
        self.logger = get_logger(self.__class__.__name__)

    def add_result(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Store evaluation metrics for a trained model."""
        self.logger.info("Adding evaluation result for model: %s", model_name)

        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be a dictionary")

        required_keys = {"rmse", "mae", "mape"}
        if not required_keys.issubset(metrics.keys()):
            raise ValueError(f"Metrics must contain keys: {required_keys}")

        self.results[model_name] = metrics
        self.logger.info("Model %s stored with RMSE=%.4f", model_name, metrics["rmse"])

    def get_best_model(self, primary_metric: str = "rmse") -> Optional[str]:
        """Identify and return the best-performing model by primary metric."""
        if not self.results:
            self.logger.warning("No model results available")
            return None

        self.logger.info("Selecting best model based on primary metric: %s", primary_metric)

        if primary_metric not in {"rmse", "mae", "mape"}:
            raise ValueError(f"Primary metric must be one of: rmse, mae, mape")

        best_model = min(self.results, key=lambda m: self.results[m][primary_metric])
        self.logger.info("Best model selected: %s with %s=%.4f", best_model, primary_metric, self.results[best_model][primary_metric])
        return best_model

    def generate_report(self) -> pd.DataFrame:
        """Create a ranked summary of all evaluated models."""
        if not self.results:
            self.logger.warning("No model results available for report")
            return pd.DataFrame()

        self.logger.info("Generating evaluation report for %d models", len(self.results))

        report_df = pd.DataFrame(self.results).T.reset_index()
        report_df.columns = ["Model", "RMSE", "MAE", "MAPE"]

        report_df = report_df.sort_values("RMSE").reset_index(drop=True)
        report_df["Rank"] = range(1, len(report_df) + 1)

        self.logger.info("Evaluation report generated with rankings")
        return report_df[["Rank", "Model", "RMSE", "MAE", "MAPE"]]

    def get_summary(self) -> str:
        """Return a formatted text summary of model rankings and metrics."""
        report = self.generate_report()

        if report.empty:
            return "No evaluation results available"

        summary = "\n" + "=" * 70 + "\n"
        summary += "MODEL EVALUATION REPORT\n"
        summary += "=" * 70 + "\n"
        summary += report.to_string(index=False)
        summary += "\n" + "=" * 70 + "\n"

        return summary
