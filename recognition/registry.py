"""ModelRegistry singleton pattern"""
from typing import Dict, Optional, Type, Any
from threading import Lock
import logging

from .base import RecognitionModel
from .types import ModelInfo

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when model is not found in registry."""
    pass


class ModelLoadError(Exception):
    """Raised when model fails to load."""
    pass


class ModelRegistry:
    """Thread-safe registry for recognition models."""

    _instance: Optional["ModelRegistry"] = None
    _lock: Lock = Lock()
    _models_lock: Lock = Lock()

    def __new__(cls) -> "ModelRegistry":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize registry."""
        if self._initialized:
            return
        self._model_classes: Dict[str, Type[RecognitionModel]] = {}
        self._model_instances: Dict[str, RecognitionModel] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        self._priorities: Dict[str, int] = {}
        self._initialized = True

    def register(
        self,
        name: str,
        model_class: Type[RecognitionModel],
        priority: int = 0,
    ) -> None:
        """Register a model class."""
        with self._models_lock:
            if name in self._model_classes:
                logger.warning(f"Overwriting registered model: {name}")
            self._model_classes[name] = model_class
            self._priorities[name] = priority
            logger.info(f"Registered model: {name} (priority={priority})")

    def get(self, name: str) -> RecognitionModel:
        """Get or instantiate a model by name (lazy loading)."""
        with self._models_lock:
            if name not in self._model_classes:
                raise ModelNotFoundError(f"Model '{name}' not found in registry")

            # Return cached instance if available
            if name in self._model_instances:
                return self._model_instances[name]

            # Instantiate and load model
            try:
                model_class = self._model_classes[name]
                instance = model_class()
                instance.load()
                self._model_instances[name] = instance
                logger.info(f"Loaded model: {name}")
                return instance
            except Exception as e:
                raise ModelLoadError(f"Failed to load model '{name}': {str(e)}")

    def get_info(self, name: str) -> ModelInfo:
        """Get model info by instantiating the model temporarily."""
        with self._models_lock:
            if name not in self._model_classes:
                raise ModelNotFoundError(f"Model '{name}' not found in registry")

            # Return cached info if model is loaded
            if name in self._model_instances:
                return self._model_instances[name].info

            # If not cached in _model_info, instantiate temporarily
            if name not in self._model_info:
                try:
                    temp_instance = self._model_classes[name]()
                    self._model_info[name] = temp_instance.info
                except Exception as e:
                    logger.warning(f"Could not get info for {name}: {e}")
                    raise ModelNotFoundError(f"Could not get info for '{name}'")

            return self._model_info[name]

    def list_models(self) -> Dict[str, ModelInfo]:
        """List all registered models with their info."""
        with self._models_lock:
            result = {}
            for name in self._model_classes.keys():
                try:
                    result[name] = self.get_info(name)
                except Exception as e:
                    logger.warning(f"Could not get info for model {name}: {e}")
            return result

    def unload(self, name: str) -> None:
        """Unload a model instance."""
        with self._models_lock:
            if name in self._model_instances:
                try:
                    self._model_instances[name].unload()
                    del self._model_instances[name]
                    logger.info(f"Unloaded model: {name}")
                except Exception as e:
                    logger.error(f"Error unloading model '{name}': {str(e)}")

    def unload_all(self) -> None:
        """Unload all model instances."""
        with self._models_lock:
            for name in list(self._model_instances.keys()):
                try:
                    self._model_instances[name].unload()
                    del self._model_instances[name]
                except Exception as e:
                    logger.error(f"Error unloading model '{name}': {str(e)}")


def register_model(name: str, priority: int = 0) -> Any:
    """Decorator for registering a model class."""
    def decorator(cls: Type[RecognitionModel]) -> Type[RecognitionModel]:
        registry = ModelRegistry()
        registry.register(name, cls, priority=priority)
        return cls
    return decorator


# Singleton instance
registry = ModelRegistry()
