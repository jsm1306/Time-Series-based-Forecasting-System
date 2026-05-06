import logging
from logging import Logger
from pathlib import Path
from typing import Optional

BASE_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_FILENAME: str = r"Forecasting Case.csv"
RAW_DATA_PATH: Path = DATA_DIR / RAW_DATA_FILENAME
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
LOG_DIR: Path = BASE_DIR / "logs"
DEFAULT_LOG_LEVEL: int = logging.INFO


def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists for the provided path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name: str, level: int = DEFAULT_LOG_LEVEL) -> Logger:
    """Create or retrieve a configured logger instance."""
    ensure_directory(LOG_DIR)

    logger: Logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
