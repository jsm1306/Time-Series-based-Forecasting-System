import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_logger
from app.pipelines.forecasting_pipeline import ForecastingPipeline


logger = get_logger("run_full_pipeline")


def run() -> None:
    """Execute the full forecasting pipeline for all states."""
    processed_data_path = PROJECT_ROOT / "data" / "processed" / "processed_timeseries.csv"
    model_output_dir = PROJECT_ROOT / "trained_models"
    registry_path = model_output_dir / "model_registry.json"

    try:
        pipeline = ForecastingPipeline(
            processed_data_path=processed_data_path,
            model_output_dir=model_output_dir,
            registry_path=registry_path,
        )

        summary = pipeline.run()

        print(f"Total states processed: {summary['total_states']}")
        print(f"Successful states: {summary['successful_states']}")
        print(f"Failed states: {summary['failed_states']}")

        if summary.get("selection_counts"):
            print("Best model distribution:")
            for model_name, count in summary["selection_counts"].items():
                print(f"{model_name}: {count}")

        if summary.get("average_rmse") is not None:
            print(f"Average RMSE: {summary['average_rmse']:.4f}")
            print(f"Average MAE: {summary['average_mae']:.4f}")
            print(f"Average MAPE: {summary['average_mape']:.4f}")
    except Exception as error:
        logger.exception("Full forecasting pipeline execution failed")
        raise error


if __name__ == "__main__":
    run()
