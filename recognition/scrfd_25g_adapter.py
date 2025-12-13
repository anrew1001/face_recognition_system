"""SCRFD_2.5G_KPS face detection adapter.

SCRFD (Sample and Computation Redistribution Face Detector) is an efficient
face detection model optimized for performance. This adapter integrates the
SCRFD_2.5G_KPS ONNX model with our RecognitionModel interface.

The SCRFD_2.5G_KPS model is optimized for better accuracy/speed tradeoff
compared to SCRFD_10G, making it suitable for composite detection architectures.

Note:
    SCRFD_2.5G_KPS is a detection-only model. Embedding extraction requires a
    separate model (e.g., ArcFace) which will be integrated in the future.
    The model provides 5-point facial landmarks critical for face alignment.
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


@register_model("scrfd_2.5g", priority=15)
class SCRFD25GAdapter(RecognitionModel):
    """SCRFD_2.5G_KPS face detection model adapter.

    This adapter provides face detection capabilities using the SCRFD_2.5G_KPS
    ONNX model. It supports CPU and Apple Silicon (CoreML) execution.

    Attributes:
        _model_path: Path to ONNX model file.
        _det_size: Input size for detection (height, width).
        _confidence_threshold: Minimum confidence for valid detections.
        _device: Device to run inference on ("cpu" or "coreml").
        _session: ONNX Runtime inference session.

    Example:
        >>> adapter = SCRFD25GAdapter(device="cpu")
        >>> adapter.load()
        >>> detections = adapter.detect_faces(image)
        >>> for det in detections:
        ...     print(f"Face at {det.bbox} with confidence {det.confidence:.2f}")
    """

    def __init__(
        self,
        model_path: str = "models/det_2.5g.onnx",
        det_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        num_threads: int = 4,
        graph_optimization: bool = True,
        **kwargs  # Accept extra config parameters (e.g., 'enabled')
    ) -> None:
        """Initialize SCRFD_2.5G adapter.

        Args:
            model_path: Path to ONNX model file.
            det_size: Detection input size as (height, width).
            confidence_threshold: Minimum confidence threshold for detections.
            device: Device to run on ("cpu" or "coreml").
            num_threads: Number of threads for CPU inference (default: 4).
            graph_optimization: Enable ONNX graph optimizations (default: True).
            **kwargs: Additional config parameters (ignored for compatibility).

        Raises:
            ValueError: If device is not "cpu" or "coreml".
        """
        if device not in ("cpu", "coreml"):
            raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'coreml'")

        self._model_path = Path(model_path)
        self._det_size = det_size
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._num_threads = num_threads
        self._graph_optimization = graph_optimization
        self._session = None

        # Log ignored parameters for debugging
        if kwargs:
            logger.debug(f"Ignoring extra config parameters: {list(kwargs.keys())}")

        logger.info(
            f"Initialized SCRFD_2.5G adapter: model_path={model_path}, "
            f"det_size={det_size}, device={device}, num_threads={num_threads}"
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
            name="scrfd_2.5g",
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
                f"Please download SCRFD_2.5G_KPS ONNX model from:\n"
                f"https://huggingface.co/public-data/insightface/tree/main/models/scrfd_2.5g_bnkps/"
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

        # Create inference session with optimizations
        try:
            sess_options = ort.SessionOptions()

            # Enable graph optimizations for better performance
            if self._graph_optimization:
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                logger.info("Enabled ONNX graph optimizations (ORT_ENABLE_ALL)")

            # Configure threading for CPU inference
            if self._device == "cpu":
                sess_options.intra_op_num_threads = self._num_threads
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                logger.info(
                    f"Configured CPU inference: {self._num_threads} threads, "
                    "sequential execution"
                )

            self._session = ort.InferenceSession(
                str(self._model_path),
                sess_options=sess_options,
                providers=providers
            )

            logger.info(
                f"Loaded SCRFD_2.5G model from {self._model_path} "
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
            logger.info("Unloaded SCRFD_2.5G model")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for SCRFD inference.

        Args:
            image: Input image as HxWx3 BGR uint8 array.

        Returns:
            Preprocessed tensor as (1, 3, H, W) float32 array normalized to [-1, 1].

        Note:
            - Resizes to self._det_size
            - Converts BGR to RGB
            - Normalizes with SCRFD formula: (pixel - 127.5) / 128.0
            - Transposes to NCHW format
        """
        # Resize to detection size
        resized = cv2.resize(image, self._det_size)

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # SCRFD-specific normalization: (pixel - 127.5) / 128.0
        # This scales [0, 255] to approximately [-1, 1]
        normalized = (rgb.astype(np.float32) - 127.5) / 128.0

        # HWC to CHW
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched

    def _nms(
        self,
        detections: List[FaceDetection],
        iou_threshold: float = 0.4
    ) -> List[FaceDetection]:
        """Non-Maximum Suppression to remove overlapping detections.

        Args:
            detections: List of face detections.
            iou_threshold: IoU threshold for suppression (default: 0.4).

        Returns:
            Filtered list of detections after NMS.
        """
        if not detections:
            return []

        # Sort by confidence descending
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        while detections:
            # Take highest confidence detection
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping detections
            detections = [
                det for det in detections
                if self._compute_iou(best.bbox, det.bbox) < iou_threshold
            ]

        return keep

    def _compute_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union between two bboxes.

        Args:
            bbox1: First bbox as (x1, y1, x2, y2).
            bbox2: Second bbox as (x1, y1, x2, y2).

        Returns:
            IoU score in [0, 1].
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _decode_predictions(
        self,
        outputs: List[np.ndarray],
        orig_shape: Tuple[int, int]
    ) -> List[FaceDetection]:
        """Decode SCRFD anchor-based outputs with proper NMS.

        SCRFD_2.5G_KPS outputs 15 tensors for 5 scales:
        - Outputs 0-4: scores (one per scale)
        - Outputs 5-9: bboxes (4 coords per anchor)
        - Outputs 10-14: landmarks (10 values = 5 points × 2 coords)

        Args:
            outputs: List of 15 output tensors from ONNX model.
            orig_shape: Original image shape (height, width).

        Returns:
            List of FaceDetection objects after NMS.
        """
        detections = []

        # SCRFD_2.5G uses 5 FPN scales with strides [8, 16, 32, 64, 128]
        # However, for 640x640 input, effective scales are smaller
        # We determine stride from the output grid size
        num_scales = 5

        if len(outputs) != 15:
            logger.error(f"Expected 15 outputs, got {len(outputs)}")
            return []

        for scale_idx in range(num_scales):
            score_idx = scale_idx
            bbox_idx = num_scales + scale_idx
            kps_idx = 2 * num_scales + scale_idx

            scores = outputs[score_idx]
            bboxes = outputs[bbox_idx]
            kps = outputs[kps_idx]

            # Determine grid size and stride from output shape
            num_anchors = scores.shape[0]
            grid_size = int(np.sqrt(num_anchors))

            if grid_size * grid_size != num_anchors:
                logger.warning(
                    f"Scale {scale_idx}: non-square grid {num_anchors}, "
                    f"expected {grid_size}×{grid_size}"
                )
                # Try to proceed anyway
                grid_h = grid_w = grid_size
            else:
                grid_h = grid_w = grid_size

            stride = self._det_size[0] / grid_size

            for i in range(num_anchors):
                score = scores[i, 0]

                if score < self._confidence_threshold:
                    continue

                # Get anchor center (row-major order)
                anchor_y = (i // grid_w) * stride
                anchor_x = (i % grid_w) * stride

                # Decode bbox from distance format
                bbox = bboxes[i]
                x1 = (anchor_x - bbox[0] * stride) * orig_shape[1] / self._det_size[1]
                y1 = (anchor_y - bbox[1] * stride) * orig_shape[0] / self._det_size[0]
                x2 = (anchor_x + bbox[2] * stride) * orig_shape[1] / self._det_size[1]
                y2 = (anchor_y + bbox[3] * stride) * orig_shape[0] / self._det_size[0]

                # Clip to image bounds
                x1 = int(max(0, min(x1, orig_shape[1])))
                y1 = int(max(0, min(y1, orig_shape[0])))
                x2 = int(max(0, min(x2, orig_shape[1])))
                y2 = int(max(0, min(y2, orig_shape[0])))

                # Decode landmarks (5-point keypoints)
                landmarks_raw = kps[i]
                landmarks_list = []
                for j in range(0, 10, 2):
                    lm_x = (anchor_x + landmarks_raw[j] * stride) * orig_shape[1] / self._det_size[1]
                    lm_y = (anchor_y + landmarks_raw[j + 1] * stride) * orig_shape[0] / self._det_size[0]
                    landmarks_list.append([lm_x, lm_y])
                landmarks = np.array(landmarks_list, dtype=np.float32)

                # Only add valid detections
                if x2 > x1 and y2 > y1:
                    detections.append(FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(score),
                        landmarks=landmarks
                    ))

        # Apply NMS to remove overlapping detections
        detections = self._nms(detections, iou_threshold=0.4)

        logger.debug(f"Decoded {len(detections)} faces after NMS")

        return detections

    def detect_faces(
        self,
        image: np.ndarray,
        frame_id: Optional[int] = None  # For compatibility with main.py
    ) -> List[FaceDetection]:
        """Detect all faces in image.

        Args:
            image: Input image as HxWx3 BGR uint8 numpy array.
            frame_id: Optional frame ID (unused, for compatibility).

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

        logger.debug(f"SCRFD_2.5G inference outputs: {len(outputs)} tensors")
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
        frame_id: Optional[int] = None  # For compatibility with main.py
    ) -> Optional[EmbeddingResult]:
        """Extract face embedding from image region.

        Note:
            SCRFD_2.5G_KPS is a detection-only model. This method performs face
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
            frame_id: Optional frame ID (unused, for compatibility).

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
