"""ModelInfo, FaceDetection, EmbeddingResult data types for face recognition system."""
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ModelInfo:
    """Immutable metadata about a recognition model.

    Attributes:
        name: Model identifier name.
        version: Semantic version string (e.g., "1.0.0").
        embedding_dim: Dimensionality of face embeddings produced by this model.
        input_size: Expected input image dimensions as (height, width).
    """
    name: str
    version: str
    embedding_dim: int
    input_size: Tuple[int, int]

    def fingerprint(self) -> str:
        """Generate a deterministic fingerprint for model identity.

        Returns:
            First 16 characters of SHA256 hash computed from name:version:embedding_dim.
            Used to verify embedding compatibility and detect model changes.

        Example:
            >>> info = ModelInfo("scrfd_10g", "1.0.0", 512, (640, 640))
            >>> info.fingerprint()  # "a1b2c3d4e5f6g7h8"
        """
        content = f"{self.name}:{self.version}:{self.embedding_dim}"
        hash_obj = hashlib.sha256(content.encode("utf-8"))
        return hash_obj.hexdigest()[:16]


@dataclass
class FaceDetection:
    """Detected face region with confidence score and optional landmarks.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        confidence: Detection confidence score in range [0.0, 1.0].
        landmarks: Optional face landmarks (e.g., eyes, nose) as (N, 2) array.

    Guarantees:
        - bbox values are non-negative integers
        - confidence is in valid range [0.0, 1.0]
        - landmarks shape is (N, 2) or None
    """
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmarks: Optional[np.ndarray] = None

    @property
    def width(self) -> int:
        """Bounding box width in pixels."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Bounding box height in pixels."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        """Bounding box area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of bounding box as (x, y)."""
        x = (self.bbox[0] + self.bbox[2]) / 2.0
        y = (self.bbox[1] + self.bbox[3]) / 2.0
        return (x, y)


@dataclass
class EmbeddingResult:
    """Face embedding with associated detection and model information.

    Attributes:
        embedding: Face feature vector as L2-normalized numpy array of shape (D,).
        face_crop: Cropped face image from source.
        detection: Original face detection that produced this embedding.
        model_fingerprint: Fingerprint of the model that computed this embedding.

    Guarantees:
        - embedding is L2-normalized (np.linalg.norm(embedding) ≈ 1.0)
        - Can be safely used for similarity computation via dot product

    Raises:
        ValueError: If embedding L2 norm deviates significantly from 1.0 after
                    post-initialization normalization.
    """
    embedding: np.ndarray
    face_crop: Image.Image
    detection: FaceDetection
    model_fingerprint: str

    def __post_init__(self) -> None:
        """Validate and normalize embedding vector.

        Ensures the embedding is L2-normalized by dividing by its norm.
        Raises ValueError if normalization fails (e.g., zero vector).
        """
        norm = np.linalg.norm(self.embedding)
        if norm < 1e-10:
            raise ValueError("Cannot normalize embedding with near-zero norm")

        # Normalize in-place
        self.embedding = self.embedding / norm

        # Verify normalization
        actual_norm = np.linalg.norm(self.embedding)
        if not np.isclose(actual_norm, 1.0, atol=1e-6):
            raise ValueError(
                f"Embedding normalization failed: norm={actual_norm}, "
                f"expected ~1.0"
            )
