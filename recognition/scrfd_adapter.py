"""SCRFD_10G face detection adapter.

SCRFD (Sample and Computation Redistribution Face Detector) is an efficient
face detection model optimized for performance. This adapter integrates the
ONNX model with our RecognitionModel interface.

Note:
    SCRFD_10G is a detection-only model. Embedding extraction requires a
    separate model (e.g., ArcFace) which will be integrated in the future.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .base import RecognitionModel
from .registry import register_model
from .types import EmbeddingResult, FaceDetection, ModelInfo

logger = logging.getLogger(__name__)


@register_model("scrfd_10g", priority=10)
class SCRFDAdapter(RecognitionModel):
    """SCRFD_10G face detection model adapter.

    This adapter provides face detection capabilities using the SCRFD_10G
    ONNX model. It supports CPU and Apple Silicon (CoreML) execution.

    Attributes:
        _model_path: Path to ONNX model file.
        _det_size: Input size for detection (height, width).
        _confidence_threshold: Minimum confidence for valid detections.
        _device: Device to run inference on ("cpu" or "coreml").
        _session: ONNX Runtime inference session.

    Example:
        >>> adapter = SCRFDAdapter(device="cpu")
        >>> adapter.load()
        >>> detections = adapter.detect_faces(image)
        >>> for det in detections:
        ...     print(f"Face at {det.bbox} with confidence {det.confidence:.2f}")
    """

    def __init__(
        self,
        model_path: str = "models/scrfd_10g.onnx",
        det_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ) -> None:
        """Initialize SCRFD adapter.

        Args:
            model_path: Path to ONNX model file.
            det_size: Detection input size as (height, width).
            confidence_threshold: Minimum confidence threshold for detections.
            device: Device to run on ("cpu" or "coreml").

        Raises:
            ValueError: If device is not "cpu" or "coreml".
        """
        if device not in ("cpu", "coreml"):
            raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'coreml'")

        self._model_path = Path(model_path)
        self._det_size = det_size
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._session = None

        logger.info(
            f"Initialized SCRFD adapter: model_path={model_path}, "
            f"det_size={det_size}, device={device}"
        )

    @property
    def info(self) -> ModelInfo:
        """Get model metadata.

        Returns:
            ModelInfo with name, version, and configuration.

        Note:
            embedding_dim is set to 512 as a placeholder. This will be
            replaced when ArcFace integration is added.
        """
        return ModelInfo(
            name="scrfd_10g",
            version="1.0",
            embedding_dim=512,  # Placeholder, will use ArcFace later
            input_size=self._det_size
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference.

        Returns:
            True if ONNX session is initialized, False otherwise.
        """
        return self._session is not None

    def load(self) -> None:
        """Load ONNX model and initialize inference session.

        Configures execution providers based on device setting:
        - "coreml": Attempts CoreMLExecutionProvider (Apple Silicon optimization)
        - "cpu": Uses CPUExecutionProvider

        Raises:
            RuntimeError: If onnxruntime is not installed or model file not found.
        """
        if self.is_loaded:
            logger.info("Model already loaded, skipping")
            return

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime not installed. Install with: "
                "pip install onnxruntime>=1.16.0"
            ) from e

        if not self._model_path.exists():
            raise RuntimeError(
                f"Model file not found: {self._model_path}. "
                f"Please download SCRFD_10G ONNX model to this location."
            )

        # Configure execution providers
        providers = []
        if self._device == "coreml":
            # Check if CoreML provider is available
            available_providers = ort.get_available_providers()
            if "CoreMLExecutionProvider" in available_providers:
                providers.append("CoreMLExecutionProvider")
                logger.info("Using CoreML execution provider")
            else:
                logger.warning(
                    "CoreML provider not available, falling back to CPU. "
                    "Install onnxruntime-silicon for Apple Silicon optimization."
                )
                providers.append("CPUExecutionProvider")
        else:
            providers.append("CPUExecutionProvider")

        # Create inference session
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            self._session = ort.InferenceSession(
                str(self._model_path),
                sess_options=sess_options,
                providers=providers
            )

            logger.info(
                f"Loaded SCRFD model from {self._model_path} "
                f"with providers: {self._session.get_providers()}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

    def unload(self) -> None:
        """Release model resources.

        Idempotent - safe to call multiple times.
        """
        if self._session is not None:
            self._session = None
            logger.info("Unloaded SCRFD model")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for SCRFD inference.

        Args:
            image: Input image as HxWx3 BGR uint8 array.

        Returns:
            Preprocessed tensor as (1, 3, H, W) float32 array normalized to [0, 1].

        Note:
            - Resizes to self._det_size
            - Converts BGR to RGB
            - Normalizes to [0, 1]
            - Transposes to NCHW format
        """
        # Resize to detection size
        resized = cv2.resize(image, self._det_size)

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # HWC to CHW
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched

    def _decode_predictions(
        self,
        outputs: List[np.ndarray],
        orig_shape: Tuple[int, int]
    ) -> List[FaceDetection]:
        """Decode SCRFD anchor-based outputs."""
        detections = []

        # SCRFD outputs: multiple scales with [score, bbox, landmarks]
        # Typically 3 scales × 3 outputs = 9 tensors

        stride_fpn = [8, 16, 32]
        num_anchors = 2

        for idx, stride in enumerate(stride_fpn):
            # Find corresponding outputs for this scale
            score_idx = idx * 3
            bbox_idx = idx * 3 + 1

            if score_idx >= len(outputs) or bbox_idx >= len(outputs):
                continue

            scores = outputs[score_idx]
            bboxes = outputs[bbox_idx]

            # scores shape: (H*W*num_anchors, 1)
            # bboxes shape: (H*W*num_anchors, 4)

            # Validate shapes match
            if scores.shape[0] != bboxes.shape[0]:
                logger.warning(
                    f"Shape mismatch: scores {scores.shape} vs bboxes {bboxes.shape}"
                )
                continue

            height = self._det_size[0] // stride
            width = self._det_size[1] // stride

            for i in range(scores.shape[0]):
                score = scores[i, 0]

                if score < self._confidence_threshold:
                    continue

                # Get anchor center
                anchor_idx = i // num_anchors
                y = (anchor_idx // width) * stride
                x = (anchor_idx % width) * stride

                # Decode bbox (distance format)
                bbox = bboxes[i]
                x1 = (x - bbox[0]) * orig_shape[1] / self._det_size[1]
                y1 = (y - bbox[1]) * orig_shape[0] / self._det_size[0]
                x2 = (x + bbox[2]) * orig_shape[1] / self._det_size[1]
                y2 = (y + bbox[3]) * orig_shape[0] / self._det_size[0]

                # Clip to bounds
                x1 = int(max(0, min(x1, orig_shape[1])))
                y1 = int(max(0, min(y1, orig_shape[0])))
                x2 = int(max(0, min(x2, orig_shape[1])))
                y2 = int(max(0, min(y2, orig_shape[0])))

                if x2 > x1 and y2 > y1:
                    detections.append(FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(score)
                    ))

        # NMS (simple - keep top K by score)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:50]  # Keep max 50 faces

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect all faces in image.

        Args:
            image: Input image as HxWx3 BGR uint8 numpy array.

        Returns:
            List of FaceDetection objects sorted by confidence descending.
            Empty list if no faces detected.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If image format is invalid.

        Example:
            >>> image = cv2.imread("photo.jpg")
            >>> detections = adapter.detect_faces(image)
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

        orig_shape = image.shape[:2]  # (height, width)

        # Preprocess
        input_tensor = self._preprocess_image(image)

        # Run inference
        try:
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: input_tensor})
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}") from e

        logger.debug(f"SCRFD inference outputs: {len(outputs)} tensors")
        for i, out in enumerate(outputs):
            logger.debug(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")

        # Decode predictions
        detections = self._decode_predictions(outputs, orig_shape)

        logger.debug(f"Detected {len(detections)} faces")

        return detections

    def extract_embedding(
        self,
        image: np.ndarray,
        detection: Optional[FaceDetection] = None,
    ) -> Optional[EmbeddingResult]:
        """Extract face embedding from image region.

        Note:
            SCRFD_10G is a detection-only model. This method performs face
            detection and returns a placeholder embedding until ArcFace
            integration is implemented.

        TODO:
            Integrate ArcFace model for actual embedding extraction:
            1. Crop face region from detection.bbox
            2. Align face using landmarks
            3. Pass aligned face to ArcFace model
            4. Return normalized embedding from ArcFace

        Args:
            image: Input image as HxWx3 BGR uint8 numpy array.
            detection: Optional face detection. If None, detects face first.

        Returns:
            EmbeddingResult with placeholder zero embedding, or None if no face.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If image format is invalid or bbox out of bounds.

        Example:
            >>> result = adapter.extract_embedding(image, detection)
            >>> if result:
            ...     print(f"Embedding shape: {result.embedding.shape}")
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # If no detection provided, detect faces first
        if detection is None:
            detections = self.detect_faces(image)
            if not detections:
                logger.warning("No faces detected for embedding extraction")
                return None
            detection = detections[0]  # Use highest confidence detection

        # Validate bbox is within image bounds
        h, w = image.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            raise ValueError(
                f"Detection bbox {detection.bbox} is out of image bounds "
                f"(width={w}, height={h})"
            )

        # Crop face region
        face_crop_array = image[y1:y2, x1:x2]
        face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop_array, cv2.COLOR_BGR2RGB))

        # TODO: Replace with actual ArcFace embedding extraction
        # For now, return a placeholder zero embedding
        placeholder_embedding = np.zeros(self.info.embedding_dim, dtype=np.float32)
        # Add small random noise to avoid zero-norm error
        placeholder_embedding[0] = 1.0

        result = EmbeddingResult(
            embedding=placeholder_embedding,
            face_crop=face_crop_pil,
            detection=detection,
            model_fingerprint=self.info.fingerprint()
        )

        logger.warning(
            "Using placeholder embedding. ArcFace integration pending."
        )

        return result