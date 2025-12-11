"""Abstract RecognitionModel interface for face detection and embedding extraction."""
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from .types import EmbeddingResult, FaceDetection, ModelInfo


class RecognitionModel(ABC):
    """Abstract base class for face recognition models.

    Defines the interface for models that detect faces and extract embeddings.
    Implementations must handle model loading/unloading and provide methods
    for face detection and embedding extraction.

    Subclasses must implement all abstract methods and properties.
    """

    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Get immutable model metadata.

        Returns:
            ModelInfo containing name, version, embedding_dim, and input_size.
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model weights are currently loaded in memory.

        Returns:
            True if model is ready for inference, False otherwise.

        Note:
            All inference methods require is_loaded == True.
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """Load model weights and initialize for inference.

        Must be called before detect_faces or extract_embedding.
        Multiple calls to load() should be safe (idempotent).

        Raises:
            RuntimeError: If model weights cannot be loaded or dependencies
                         are missing (ONNX runtime, device unavailable, etc.).
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free GPU/memory resources.

        Must be called when model is no longer needed.
        Multiple calls to unload() should be safe (idempotent).

        After unload(), is_loaded must return False.
        Calling inference methods after unload() raises RuntimeError.
        """
        pass

    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect all faces in an image.

        Args:
            image: Input image as HxWx3 uint8 BGR numpy array.

        Returns:
            List of FaceDetection objects, empty list if no faces found.

        Raises:
            RuntimeError: If model is not loaded (is_loaded == False).
            ValueError: If image format is invalid (not uint8, wrong shape, etc.).

        Guarantees:
            - All returned FaceDetection.bbox values are valid pixel coordinates
            - All returned FaceDetection.confidence is in [0.0, 1.0]
            - Results are sorted by confidence descending

        Note:
            Implementations should crop and normalize input to model.input_size.
        """
        pass

    @abstractmethod
    def extract_embedding(
        self,
        image: np.ndarray,
        detection: Optional[FaceDetection] = None,
    ) -> Optional[EmbeddingResult]:
        """Extract face embedding from image region.

        If detection is provided, extracts embedding from the detected face region.
        If detection is None, assumes the entire image is a face.

        Args:
            image: Input image as HxWx3 uint8 BGR numpy array.
            detection: Optional face detection to extract from. If None, uses
                      entire image as face region.

        Returns:
            EmbeddingResult with L2-normalized embedding, or None if extraction fails.

        Raises:
            RuntimeError: If model is not loaded (is_loaded == False).
            ValueError: If image format is invalid or detection bbox is out of bounds.

        Guarantees:
            - Returned EmbeddingResult.embedding has shape (embedding_dim,) and L2 norm ≈ 1.0
            - Returned EmbeddingResult.model_fingerprint matches self.info.fingerprint()
            - Result can be safely used for similarity computation via dot product

        Note:
            - Implementations should crop image to detection.bbox if provided
            - Embedding dimensionality must match self.info.embedding_dim
        """
        pass

    def compute_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two L2-normalized embeddings.

        Args:
            emb1: First L2-normalized embedding vector.
            emb2: Second L2-normalized embedding vector.

        Returns:
            Similarity score in range [-1.0, 1.0], where:
            - 1.0 = identical embeddings (same person)
            - 0.0 = orthogonal (dissimilar)
            - -1.0 = opposite (anti-correlated)

        Guarantees:
            - Both inputs should be L2-normalized (norm ≈ 1.0)
            - Uses dot product: cos_sim = emb1 · emb2

        Note:
            For L2-normalized vectors, dot product equals cosine similarity.
            Result is stable and efficient for batch comparisons.
        """
        similarity = float(np.dot(emb1, emb2))
        return float(np.clip(similarity, -1.0, 1.0))
