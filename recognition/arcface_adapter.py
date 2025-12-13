"""ArcFace face recognition adapter.

ArcFace is a state-of-the-art face recognition model that extracts high-quality
embeddings from aligned face images. This adapter is designed for recognition-only
tasks and requires pre-aligned 112x112 face images.

Note:
    This is a recognition-only model. Face detection must be performed separately
    using SCRFD or another detector. ArcFace expects aligned face images as input.
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


@register_model("arcface", priority=5)
class ArcFaceAdapter(RecognitionModel):
    """ArcFace face recognition model adapter.

    This adapter provides embedding extraction using the ArcFace R50 ONNX model
    trained on WebFace600K dataset. It supports CPU and Apple Silicon (CoreML)
    execution.

    ArcFace is a recognition-only model and does not perform face detection.
    Input images must be pre-aligned to 112x112 RGB format.

    Attributes:
        _model_path: Path to ONNX model file.
        _input_size: Expected input size (112, 112) for aligned faces.
        _device: Device to run inference on ("cpu" or "coreml").
        _num_threads: Number of CPU threads for inference.
        _graph_optimization: Enable ONNX graph optimizations.
        _session: ONNX Runtime inference session.

    Example:
        >>> adapter = ArcFaceAdapter(device="cpu")
        >>> adapter.load()
        >>> # Note: image must be aligned 112x112 face
        >>> result = adapter.extract_embedding(aligned_face_image)
        >>> print(f"Embedding shape: {result.embedding.shape}")
    """

    def __init__(
        self,
        model_path: str = "models/w600k_r50.onnx",
        device: str = "cpu",
        num_threads: int = 4,
        graph_optimization: bool = True,
        **kwargs  # Accept extra config parameters (e.g., 'enabled')
    ) -> None:
        """Initialize ArcFace adapter.

        Args:
            model_path: Path to ONNX model file (default: models/w600k_r50.onnx).
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
        self._input_size = (112, 112)  # ArcFace standard input size
        self._device = device
        self._num_threads = num_threads
        self._graph_optimization = graph_optimization
        self._session = None

        # Log ignored parameters for debugging
        if kwargs:
            logger.debug(f"Ignoring extra config parameters: {list(kwargs.keys())}")

        logger.info(
            f"Initialized ArcFace adapter: model_path={model_path}, "
            f"device={device}, num_threads={num_threads}"
        )

    @property
    def info(self) -> ModelInfo:
        """Get model metadata.

        Returns:
            ModelInfo with name, version, embedding_dim, and input_size.

        Note:
            ArcFace R50 produces 512-dimensional embeddings.
            Fingerprint is "arcface_r50_v1" for model compatibility tracking.
        """
        return ModelInfo(
            name="arcface",
            version="r50_v1",
            embedding_dim=512,
            input_size=self._input_size
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
        - "cpu": Uses CPUExecutionProvider with optimized threading

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
                f"Please download ArcFace R50 ONNX model from:\n"
                f"https://github.com/deepinsight/insightface/tree/master/model_zoo"
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
                f"Loaded ArcFace model from {self._model_path} "
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
            logger.info("Unloaded ArcFace model")

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in image.

        Note:
            ArcFace is a recognition-only model and does not support face detection.
            Use SCRFD_2.5G or another detector for face detection.

        Args:
            image: Input image (not used).

        Raises:
            NotImplementedError: Always raised, as ArcFace does not support detection.

        Example:
            >>> adapter = ArcFaceAdapter()
            >>> adapter.detect_faces(image)  # Raises NotImplementedError
        """
        raise NotImplementedError(
            "ArcFace is a recognition-only model and does not support face detection. "
            "Use SCRFD_2.5G (scrfd_2.5g) or InsightFace (insightface) for detection, "
            "then pass aligned faces to extract_embedding()."
        )

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess aligned face image for ArcFace inference.

        Args:
            image: Input aligned face as 112x112x3 RGB uint8 array.

        Returns:
            Preprocessed tensor as (1, 3, 112, 112) float32 array normalized to [-1, 1].

        Note:
            - ArcFace expects RGB input (not BGR)
            - Normalization formula: (pixel - 127.5) / 127.5
            - Transposes to NCHW format
        """
        # Verify input is 112x112
        if image.shape[:2] != self._input_size:
            raise ValueError(
                f"Input image must be {self._input_size}, got {image.shape[:2]}. "
                f"ArcFace requires aligned face images."
            )

        # ArcFace-specific normalization: (pixel - 127.5) / 127.5
        # This scales [0, 255] to [-1, 1]
        normalized = (image.astype(np.float32) - 127.5) / 127.5

        # HWC to CHW
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched

    def extract_embedding(
        self,
        image: np.ndarray,
        detection: Optional[FaceDetection] = None,
    ) -> Optional[EmbeddingResult]:
        """Extract face embedding from aligned face image.

        This method expects a pre-aligned 112x112 RGB face image. If detection
        is provided, it will be included in the result metadata, but the image
        itself must already be aligned to 112x112.

        Args:
            image: Input aligned face as HxWx3 uint8 RGB numpy array.
                  Must be 112x112 pixels (ArcFace standard size).
            detection: Optional face detection metadata. If provided, will be
                      included in result but does not affect embedding extraction.

        Returns:
            EmbeddingResult with L2-normalized 512-dim embedding, or None if
            extraction fails.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If image format is invalid or not 112x112.

        Example:
            >>> # After detecting and aligning face to 112x112
            >>> aligned_face_rgb = align_face(image, detection)
            >>> result = adapter.extract_embedding(aligned_face_rgb, detection)
            >>> print(f"Embedding: {result.embedding.shape}")  # (512,)
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

        # Verify input is 112x112
        if image.shape[:2] != self._input_size:
            raise ValueError(
                f"Input image must be {self._input_size}, got {image.shape[:2]}. "
                f"ArcFace requires pre-aligned face images. "
                f"Use face alignment with landmarks before calling extract_embedding()."
            )

        # Preprocess (expects RGB input)
        input_tensor = self._preprocess_image(image)

        # Run inference
        try:
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: input_tensor})
            embedding = outputs[0][0]  # Shape: (512,)
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}") from e

        # Normalize embedding
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            logger.warning("Face embedding has near-zero norm")
            return None

        normalized_embedding = embedding / norm

        # Convert aligned face to PIL image for storage
        face_crop_pil = Image.fromarray(image)  # Already RGB

        # Create or use provided detection
        if detection is None:
            # Create dummy detection for aligned face
            detection = FaceDetection(
                bbox=(0, 0, self._input_size[1], self._input_size[0]),
                confidence=1.0,
                landmarks=None
            )

        result = EmbeddingResult(
            embedding=normalized_embedding,
            face_crop=face_crop_pil,
            detection=detection,
            model_fingerprint=self.info.fingerprint()
        )

        logger.debug(
            f"Extracted embedding with norm={np.linalg.norm(normalized_embedding):.6f}"
        )

        return result
