"""Verification script for SCRFD adapter implementation.

This script demonstrates basic usage of the SCRFD adapter without requiring
the actual model file. It verifies:
1. Adapter registration in the registry
2. Model instantiation and configuration
3. Interface compliance with RecognitionModel
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_registration():
    """Verify SCRFD adapter is registered in the model registry."""
    from recognition import registry

    # List all registered models
    models = registry.list_models()
    logger.info(f"Registered models: {list(models.keys())}")

    # Check if scrfd_10g is registered
    if "scrfd_10g" not in models:
        logger.error("SCRFD model not found in registry!")
        return False

    logger.info("✓ SCRFD model successfully registered")
    return True


def verify_interface():
    """Verify SCRFD adapter implements RecognitionModel interface."""
    from recognition.scrfd_adapter import SCRFDAdapter
    from recognition import RecognitionModel

    # Check if SCRFDAdapter is a subclass of RecognitionModel
    if not issubclass(SCRFDAdapter, RecognitionModel):
        logger.error("SCRFDAdapter does not implement RecognitionModel!")
        return False

    logger.info("✓ SCRFD adapter implements RecognitionModel interface")

    # Instantiate adapter
    adapter = SCRFDAdapter(
        model_path="models/scrfd_10g.onnx",
        det_size=(640, 640),
        confidence_threshold=0.5,
        device="cpu"
    )

    # Verify properties
    info = adapter.info
    logger.info(f"  Model name: {info.name}")
    logger.info(f"  Model version: {info.version}")
    logger.info(f"  Embedding dim: {info.embedding_dim}")
    logger.info(f"  Input size: {info.input_size}")
    logger.info(f"  Fingerprint: {info.fingerprint()}")

    # Verify is_loaded before loading
    if adapter.is_loaded:
        logger.error("Adapter should not be loaded before load() call")
        return False

    logger.info("✓ Adapter interface working correctly")
    return True


def verify_error_handling():
    """Verify proper error handling."""
    from recognition.scrfd_adapter import SCRFDAdapter
    import numpy as np

    adapter = SCRFDAdapter()

    # Test detection without loading (should raise RuntimeError)
    try:
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        adapter.detect_faces(dummy_image)
        logger.error("Should have raised RuntimeError for unloaded model")
        return False
    except RuntimeError as e:
        logger.info(f"✓ Proper error handling: {e}")

    # Test invalid device
    try:
        SCRFDAdapter(device="invalid_device")
        logger.error("Should have raised ValueError for invalid device")
        return False
    except ValueError as e:
        logger.info(f"✓ Device validation working: {e}")

    return True


def main():
    """Run all verification tests."""
    logger.info("=" * 60)
    logger.info("SCRFD Adapter Verification")
    logger.info("=" * 60)

    tests = [
        ("Registration", verify_registration),
        ("Interface Compliance", verify_interface),
        ("Error Handling", verify_error_handling),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n[TEST] {test_name}")
        logger.info("-" * 60)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ Test failed with exception: {e}", exc_info=True)
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    all_passed = all(result for _, result in results)
    if all_passed:
        logger.info("\n✓ All verifications passed!")
        logger.info("\nNext steps:")
        logger.info("1. Download SCRFD_10G ONNX model to models/scrfd_10g.onnx")
        logger.info("2. Test with actual images using adapter.load() and adapter.detect_faces()")
        logger.info("3. Integrate ArcFace for embedding extraction")
    else:
        logger.error("\n✗ Some verifications failed")

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
