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
        models: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Add a state-level registry entry for the selected model."""
        best_metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
        }

        model_store = models if models is not None else {best_model: best_metrics}
        entry: Dict[str, Any] = {
            "best_model": best_model,
            "model_path": str(model_path.as_posix()),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "best_model_metrics": best_metrics,
            "models": model_store,
            "all_models": model_store,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
        }

        self.entries[state] = entry
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
            raw_entries = json.load(file)

        self.entries = {}
        for state, entry in raw_entries.items():
            self.entries[state] = self._normalize_entry(entry)

        self.logger.info("Loaded existing registry from %s", self.registry_path)

    def _normalize_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize legacy or new registry entry formats to the latest structure."""
        normalized = dict(entry)

        if "best_model_metrics" not in normalized:
            normalized["best_model_metrics"] = {
                "rmse": float(normalized.get("rmse", 0.0)) if normalized.get("rmse") is not None else None,
                "mae": float(normalized.get("mae", 0.0)) if normalized.get("mae") is not None else None,
                "mape": float(normalized.get("mape", 0.0)) if normalized.get("mape") is not None else None,
            }

        if "models" not in normalized:
            normalized["models"] = {}
            best_model = normalized.get("best_model")
            if best_model and normalized.get("best_model_metrics") is not None:
                normalized["models"][best_model] = normalized["best_model_metrics"]

        if "all_models" not in normalized:
            normalized["all_models"] = normalized["models"]

        return normalized

    def get_best_model_for_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Retrieve the registry entry for a specific state."""
        return self.entries.get(state)
