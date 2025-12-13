"""Test SCRFD performance with optimizations."""
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from core.config import AppConfig
from recognition import registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def benchmark_detection(model, image, num_iterations=100):
    """Benchmark face detection performance.

    Args:
        model: Loaded RecognitionModel instance.
        image: Test image (BGR numpy array).
        num_iterations: Number of iterations for benchmarking.

    Returns:
        Tuple of (avg_fps, avg_time_ms, detections).
    """
    logger.info(f"Running {num_iterations} iterations...")

    # Warmup
    for _ in range(10):
        model.detect_faces(image)

    # Benchmark
    start_time = time.time()
    detections = None

    for _ in range(num_iterations):
        detections = model.detect_faces(image)

    total_time = time.time() - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    avg_fps = num_iterations / total_time

    return avg_fps, avg_time_ms, detections


def main():
    """Test SCRFD performance with optimizations."""
    logger.info("=" * 70)
    logger.info("SCRFD Performance Test with Optimizations")
    logger.info("=" * 70)

    # Load config
    config_path = "config/recognition.yaml"
    config = AppConfig.from_yaml(config_path)

    # Get active model name
    model_name = config.get_active_model()
    logger.info(f"Active model: {model_name}")

    # Load model
    logger.info("Loading model...")
    model = registry.get(model_name)
    model.load()

    logger.info(f"Model info: {model.info.name} v{model.info.version}")
    logger.info(f"Input size: {model.info.input_size}")

    # Load or create test image
    test_image_path = Path("test_image.jpg")

    if test_image_path.exists():
        logger.info(f"Loading test image: {test_image_path}")
        image = cv2.imread(str(test_image_path))
    else:
        logger.warning("Test image not found, creating synthetic test image")
        # Create a synthetic test image (640x640)
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    logger.info(f"Image shape: {image.shape}")

    # Run benchmark
    logger.info("\n" + "=" * 70)
    logger.info("Running performance benchmark...")
    logger.info("=" * 70)

    avg_fps, avg_time_ms, detections = benchmark_detection(model, image, num_iterations=100)

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Average FPS: {avg_fps:.2f}")
    logger.info(f"Average inference time: {avg_time_ms:.2f} ms")
    logger.info(f"Detected faces: {len(detections)}")

    if detections:
        logger.info("\nDetection details:")
        for i, det in enumerate(detections[:5]):  # Show first 5
            logger.info(f"  Face {i+1}: bbox={det.bbox}, confidence={det.confidence:.3f}")

    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION STATUS")
    logger.info("=" * 70)

    # Show configuration
    model_config = config.get_model_config(model_name)
    logger.info(f"Configuration for {model_name}:")
    for key, value in model_config.items():
        logger.info(f"  {key}: {value}")

    # Performance assessment
    logger.info("\n" + "=" * 70)
    logger.info("PERFORMANCE ASSESSMENT")
    logger.info("=" * 70)

    if avg_fps >= 15:
        logger.info("✅ EXCELLENT: Target FPS (15-20) achieved!")
    elif avg_fps >= 10:
        logger.info("⚠️  GOOD: Close to target, consider further optimizations")
    else:
        logger.info("❌ NEEDS IMPROVEMENT: Below target FPS")
        logger.info("\nSuggested optimizations:")
        logger.info("  1. Reduce det_size to [320, 320] or [480, 480]")
        logger.info("  2. Increase confidence_threshold to reduce detections")
        logger.info("  3. Adjust num_threads (try 2, 4, or 8)")
        logger.info("  4. Consider using a smaller model (e.g., scrfd_2.5g)")

    logger.info("=" * 70)

    # Cleanup
    model.unload()


if __name__ == "__main__":
    main()
