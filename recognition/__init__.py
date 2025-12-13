"""Recognition module - face detection and embedding models."""

from .base import RecognitionModel
from .registry import ModelRegistry, register_model, registry
from .types import EmbeddingResult, FaceDetection, ModelInfo

# Import adapters to trigger registration
from . import scrfd_adapter  # noqa: F401
from . import scrfd_25g_adapter  # noqa: F401
from . import insightface_adapter  # noqa: F401
from . import arcface_adapter  # noqa: F401

__all__ = [
    "RecognitionModel",
    "ModelRegistry",
    "register_model",
    "registry",
    "EmbeddingResult",
    "FaceDetection",
    "ModelInfo",
]
