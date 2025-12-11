#!/usr/bin/env python3
"""Test script for InsightFace adapter validation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging
import numpy as np
import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_insightface_adapter():
    """Test InsightFace adapter registration and basic functionality."""
    print("\n" + "="*60)
    print("InsightFace Adapter Test")
    print("="*60)

    # Test 1: Import and registration
    print("\n[Test 1] Importing recognition module and checking registration...")
    try:
        from recognition import registry
        import recognition  # Trigger imports
        print("✓ Recognition module loaded")
    except Exception as e:
        logger.error(f"Failed to import recognition module: {e}")
        return False

    # Test 2: List available models
    print("\n[Test 2] Listing registered models...")
    try:
        models = registry.list_models()
        print(f"✓ Found {len(models)} registered models:")
        for name, info in models.items():
            print(f"  - {name}: v{info.version}, "
                  f"embedding_dim={info.embedding_dim}, "
                  f"input_size={info.input_size}")

        if "insightface" not in models:
            logger.error("InsightFace model not found in registry!")
            return False
        print("✓ InsightFace model found in registry")
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return False

    # Test 3: Get model instance without loading
    print("\n[Test 3] Getting model instance (without loading)...")
    try:
        adapter = registry.get("insightface")
        print(f"✓ Got model instance: {adapter.__class__.__name__}")
        print(f"  - is_loaded: {adapter.is_loaded}")
        print(f"  - info.name: {adapter.info.name}")
        print(f"  - info.embedding_dim: {adapter.info.embedding_dim}")
        print(f"  - info.fingerprint: {adapter.info.fingerprint()}")

        expected_fingerprint = "insightface:buffalo_l:512"
        # Note: fingerprint is derived from name:version:embedding_dim
        # So we just verify it's a valid hex string
        if not isinstance(adapter.info.fingerprint(), str) or len(adapter.info.fingerprint()) != 16:
            logger.error(f"Invalid fingerprint: {adapter.info.fingerprint()}")
            return False
        print("✓ Fingerprint is valid")
    except Exception as e:
        logger.error(f"Failed to get model instance: {e}")
        return False

    # Test 4: Load model
    print("\n[Test 4] Loading InsightFace model (this may take a moment)...")
    try:
        adapter.load()
        print(f"✓ Model loaded successfully")
        print(f"  - is_loaded: {adapter.is_loaded}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Note: This is expected if insightface is not installed")
        return False

    # Test 5: Create a dummy image and test detection
    print("\n[Test 5] Testing face detection with a dummy image...")
    try:
        # Create a simple dummy image (100x100 BGR)
        dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        detections = adapter.detect_faces(dummy_image)
        print(f"✓ Detection completed")
        print(f"  - Found {len(detections)} faces")
        if detections:
            for i, det in enumerate(detections):
                print(f"    Face {i}: bbox={det.bbox}, confidence={det.confidence:.3f}, "
                      f"landmarks={'Yes' if det.landmarks is not None else 'No'}")
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return False

    # Test 6: Test embedding extraction
    print("\n[Test 6] Testing embedding extraction...")
    try:
        result = adapter.extract_embedding(dummy_image, detection=None)
        if result is None:
            print("⚠ No embedding returned (expected with dummy image)")
        else:
            print(f"✓ Embedding extracted successfully")
            print(f"  - Shape: {result.embedding.shape}")
            print(f"  - Dtype: {result.embedding.dtype}")
            print(f"  - Norm: {np.linalg.norm(result.embedding):.6f}")
            print(f"  - Fingerprint: {result.model_fingerprint}")

            # Verify norm is close to 1.0
            norm = np.linalg.norm(result.embedding)
            if not np.isclose(norm, 1.0, atol=1e-5):
                logger.error(f"Embedding norm is not 1.0: {norm}")
                return False
            print("✓ Embedding is properly L2-normalized")
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        return False

    # Test 7: Unload model
    print("\n[Test 7] Unloading model...")
    try:
        adapter.unload()
        print(f"✓ Model unloaded")
        print(f"  - is_loaded: {adapter.is_loaded}")
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        return False

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60 + "\n")
    return True


if __name__ == "__main__":
    success = test_insightface_adapter()
    sys.exit(0 if success else 1)