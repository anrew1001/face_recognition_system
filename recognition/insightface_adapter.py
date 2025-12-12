"""InsightFace face detection and recognition adapter.

InsightFace is a state-of-the-art face recognition framework that provides
end-to-end face detection and embedding extraction. This adapter integrates
InsightFace with our RecognitionModel interface.

Note:
    InsightFace uses buffalo_l model variant which provides both face detection
    and 512-dimensional embedding extraction in a single forward pass.
"""
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .base import RecognitionModel
from .registry import register_model
from .types import EmbeddingResult, FaceDetection, ModelInfo

logger = logging.getLogger(__name__)


@register_model("insightface", priority=20)
class InsightFaceAdapter(RecognitionModel):
    """InsightFace face detection and recognition model adapter.

    This adapter provides face detection and embedding extraction using InsightFace
    with the buffalo_l model. It supports CPU execution optimized for inference speed.

    Attributes:
        _model: InsightFace FaceAnalysis model instance.
        _det_size: Input size for detection (height, width).
        _ctx_id: Context ID for execution (-1 for CPU, >=0 for GPU).
        _model_fingerprint: Unique fingerprint for embedding compatibility.

    Example:
        >>> adapter = InsightFaceAdapter(det_size=(640, 640))
        >>> adapter.load()
        >>> detections = adapter.detect_faces(image)
        >>> for det in detections:
        ...     print(f"Face at {det.bbox} with confidence {det.confidence:.2f}")
        >>> result = adapter.extract_embedding(image, detections[0])
        >>> print(f"Embedding shape: {result.embedding.shape}")
    """

    def __init__(
        self,
        det_size: Tuple[int, int] = (640, 640),
        ctx_id: int = -1
    ) -> None:
        """Initialize InsightFace adapter.

        Args:
            det_size: Detection input size as (height, width).
            ctx_id: Context ID for execution (-1 for CPU, >=0 for GPU).

        Raises:
            ValueError: If ctx_id is invalid.
        """
        if ctx_id < -1:
            raise ValueError(f"Invalid ctx_id: {ctx_id}. Must be >= -1")

        self._det_size = det_size
        self._ctx_id = ctx_id
        self._model = None
        # Fingerprint computed from ModelInfo.fingerprint(): sha256("insightface:0.7.0:512")[:16]
        self._model_fingerprint = "212a5ec8dbd9c95e"

        # Frame caching to avoid redundant inference
        self._cached_faces = None
        self._cache_frame_id = -1

        logger.info(
            f"Initialized InsightFace adapter: det_size={det_size}, "
            f"ctx_id={ctx_id}"
        )

    @property
    def info(self) -> ModelInfo:
        """Get model metadata.

        Returns:
            ModelInfo with name, version, and configuration.

        Note:
            InsightFace buffalo_l produces 512-dimensional embeddings.
        """
        return ModelInfo(
            name="insightface",
            version="0.7.0",
            embedding_dim=512,
            input_size=self._det_size
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference.

        Returns:
            True if FaceAnalysis model is initialized, False otherwise.
        """
        return self._model is not None

    def load(self) -> None:
        """Load InsightFace model and initialize for inference.

        Configures FaceAnalysis with:
        - allowed_modules=['detection', 'recognition'] for both tasks
        - CPU execution (ctx_id=-1) or GPU if specified

        Raises:
            RuntimeError: If insightface package is not installed or
                         model initialization fails.
        """
        if self.is_loaded:
            logger.info("Model already loaded, skipping")
            return

        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise RuntimeError(
                "insightface not installed. Install with: "
                "pip install insightface>=0.7.0"
            ) from e

        try:
            self._model = FaceAnalysis(
                allowed_modules=['detection', 'recognition']
            )
            self._model.prepare(
                ctx_id=self._ctx_id,
                det_size=self._det_size
            )

            logger.info(
                f"Loaded InsightFace model with det_size={self._det_size}, "
                f"ctx_id={self._ctx_id}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize InsightFace: {e}") from e

    def unload(self) -> None:
        """Release model resources.

        Idempotent - safe to call multiple times.
        """
        if self._model is not None:
            self._model = None
            logger.info("Unloaded InsightFace model")

    def detect_faces(self, image: np.ndarray, frame_id: Optional[int] = None) -> List[FaceDetection]:
        """Detect all faces in image.

        Args:
            image: Input image as HxWx3 BGR uint8 numpy array.
            frame_id: Optional frame identifier for caching. If provided and matches
                     cached frame_id, uses cached results for better performance.

        Returns:
            List of FaceDetection objects sorted by confidence descending.
            Empty list if no faces detected.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If image format is invalid.

        Example:
            >>> image = cv2.imread("photo.jpg")
            >>> detections = adapter.detect_faces(image, frame_id=0)
            >>> print(f"Found {len(detections)} faces")
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Validate input
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

        if image.dtype != np.uint8:
            raise ValueError(f"Image must be uint8, got {image.dtype}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Image must be HxWx3, got shape {image.shape}"
            )

        try:
            # Run detection and cache results
            faces = self._model.get(image)

            # Cache faces for this frame
            if frame_id is not None:
                self._cached_faces = faces
                self._cache_frame_id = frame_id
                logger.debug(f"Cached {len(faces)} faces for frame_id={frame_id}")

        except Exception as e:
            raise RuntimeError(f"Face detection failed: {e}") from e

        detections = []
        for face in faces:
            # InsightFace returns bbox as [x1, y1, x2, y2]
            bbox = tuple(int(x) for x in face.bbox)
            confidence = float(face.det_score)

            # Extract landmarks if available (5 keypoints)
            landmarks = None
            if hasattr(face, 'kps') and face.kps is not None:
                landmarks = np.array(face.kps, dtype=np.float32)

            detection = FaceDetection(
                bbox=bbox,
                confidence=confidence,
                landmarks=landmarks
            )
            detections.append(detection)

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.debug(f"Detected {len(detections)} faces")

        return detections

    def extract_embedding(
        self,
        image: np.ndarray,
        detection: Optional[FaceDetection] = None,
        frame_id: Optional[int] = None,
    ) -> Optional[EmbeddingResult]:
        """Extract face embedding from image region.

        If detection is provided, extracts embedding from the detected face region.
        If detection is None, performs detection first and uses the highest
        confidence face.

        Args:
            image: Input image as HxWx3 BGR uint8 numpy array.
            detection: Optional face detection to extract from. If None, detects
                      faces first.
            frame_id: Optional frame identifier for cache lookup. If provided and
                     matches cached frame_id, reuses cached face objects.

        Returns:
            EmbeddingResult with L2-normalized 512-dim embedding, or None if
            no face can be detected/extracted.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If image format is invalid or detection bbox is out of bounds.

        Example:
            >>> image = cv2.imread("photo.jpg")
            >>> detections = adapter.detect_faces(image, frame_id=0)
            >>> result = adapter.extract_embedding(image, detections[0], frame_id=0)
            >>> print(f"Embedding shape: {result.embedding.shape}")
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Validate input
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

        if image.dtype != np.uint8:
            raise ValueError(f"Image must be uint8, got {image.dtype}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Image must be HxWx3, got shape {image.shape}"
            )

        h, w = image.shape[:2]

        try:
            # If no detection provided, detect faces first
            if detection is None:
                faces = self._model.get(image)
                if not faces:
                    logger.warning("No faces detected for embedding extraction")
                    return None
                # Use highest confidence detection
                face = faces[0]
            else:
                # Clip bbox to image bounds instead of raising error
                x1, y1, x2, y2 = detection.bbox
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                # Verify bbox is still valid after clipping
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid bbox after clipping: ({x1}, {y1}, {x2}, {y2})")
                    # Fall back to full image detection
                    faces = self._model.get(image)
                    if not faces:
                        logger.warning("No faces detected for embedding extraction")
                        return None
                    face = faces[0]
                else:
                    # Update detection with clipped bbox
                    detection = FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=detection.confidence,
                        landmarks=detection.landmarks
                    )

                    # Try to use cached faces if available (PERFORMANCE OPTIMIZATION)
                    use_cache = (
                        frame_id is not None
                        and frame_id == self._cache_frame_id
                        and self._cached_faces is not None
                    )

                    if use_cache:
                        # Reuse cached face objects - avoid redundant inference!
                        faces = self._cached_faces
                        logger.debug(f"Using cached faces for frame_id={frame_id} (cache hit)")
                    else:
                        # Cache miss - run inference
                        faces = self._model.get(image)
                        logger.debug(f"Cache miss for frame_id={frame_id}, running inference")

                    if not faces:
                        logger.warning("No faces detected for embedding extraction")
                        return None

                    # Find the face that matches our detection (by bbox proximity)
                    face = faces[0]  # Use first detected face as fallback
                    for f in faces:
                        f_bbox = tuple(int(x) for x in f.bbox)
                        if f_bbox == detection.bbox:
                            face = f
                            break

        except Exception as e:
            raise RuntimeError(f"Embedding extraction failed: {e}") from e

        # Extract and normalize embedding
        embedding = face.embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            logger.warning("Face embedding has near-zero norm")
            return None

        normalized_embedding = embedding / norm

        # Crop face region for storage using detection bbox
        x1, y1, x2, y2 = detection.bbox
        crop_bgr = image[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        # Create FaceDetection for the result
        result_detection = FaceDetection(
            bbox=(x1, y1, x2, y2),
            confidence=float(face.det_score),
            landmarks=np.array(face.kps, dtype=np.float32) if hasattr(face, 'kps') else None
        )

        result = EmbeddingResult(
            embedding=normalized_embedding,
            face_crop=crop_pil,
            detection=result_detection,
            model_fingerprint=self._model_fingerprint
        )

        logger.debug(f"Extracted embedding with norm={np.linalg.norm(normalized_embedding):.6f}")

        return result