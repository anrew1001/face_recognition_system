"""Configuration management for the face recognition system."""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class AppConfig:
    """Application configuration loaded from YAML."""

    _instance: Optional["AppConfig"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> "AppConfig":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "AppConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            instance = cls()
            instance._config = data or {}
            logger.info(f"Loaded config from {config_path}")
            return instance
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        models_config = self.get("recognition.models", {})
        return models_config.get(model_name, {})

    def get_active_model(self) -> str:
        """Get the active model name."""
        return self.get("recognition.active_model", "scrfd_10g")

    def use_composite_architecture(self) -> bool:
        """Check if composite architecture (detection + recognition) is enabled."""
        return self.get("recognition.use_composite", False)

    def get_detection_model(self) -> str:
        """Get the detection model name for composite architecture."""
        return self.get("recognition.detection_model", "scrfd_2.5g")

    def get_recognition_model(self) -> str:
        """Get the recognition model name for composite architecture."""
        return self.get("recognition.recognition_model", "arcface")

    def get_model_param(self, model_name: str, param: str, default: Any = None) -> Any:
        """Get a specific parameter for a model.

        Args:
            model_name: Name of the model.
            param: Parameter name.
            default: Default value if parameter not found.

        Returns:
            Parameter value or default.
        """
        model_config = self.get_model_config(model_name)
        return model_config.get(param, default)

    def to_dict(self) -> Dict[str, Any]:
        """Get the entire configuration as a dictionary."""
        return dict(self._config)
