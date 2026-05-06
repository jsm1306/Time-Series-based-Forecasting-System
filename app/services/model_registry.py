import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from app.config.settings import get_logger


class ModelRegistry:
    """Maintain metadata of trained forecasting models and persist registry state."""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path: Path = registry_path
        self.logger = get_logger(self.__class__.__name__)
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.load_registry()

    def add_entry(
        self,
        state: str,
        best_model: str,
        model_path: Path,
        rmse: float,
        mae: float,
        mape: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """Add a state-level registry entry for the selected model."""
        self.entries[state] = {
            "best_model": best_model,
            "model_path": str(model_path.as_posix()),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "timestamp": timestamp or datetime.utcnow().isoformat(),
        }
        self.logger.info("Registry entry added for state=%s model=%s", state, best_model)

    def save_registry(self) -> None:
        """Persist the registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("w", encoding="utf-8") as file:
            json.dump(self.entries, file, indent=2)
        self.logger.info("Saved model registry to %s", self.registry_path)

    def load_registry(self) -> None:
        """Load existing registry metadata from disk if available."""
        if not self.registry_path.exists():
            self.entries = {}
            self.logger.info("No existing registry found at %s", self.registry_path)
            return

        with self.registry_path.open("r", encoding="utf-8") as file:
            self.entries = json.load(file)
        self.logger.info("Loaded existing registry from %s", self.registry_path)

    def get_best_model_for_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Retrieve the registry entry for a specific state."""
        return self.entries.get(state)
