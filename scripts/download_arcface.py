#!/usr/bin/env python3
"""Download ArcFace R50 model for face recognition.

This script downloads the ArcFace R50 model trained on WebFace600K dataset
from the InsightFace model zoo.
"""
import sys
from pathlib import Path
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ArcFace R50 model URL (from OneDrive mirror)
# Alternative sources:
# - HuggingFace: https://huggingface.co/public-data/insightface/resolve/main/models/antelopev2/glintr100.onnx
# - GitHub: https://github.com/deepinsight/insightface/tree/master/model_zoo
ARCFACE_MODEL_URL = "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
MODEL_PATH = Path("models/w600k_r50.onnx")


def download_model():
    """Download ArcFace model if not present."""
    if MODEL_PATH.exists():
        logger.info(f"Model already exists at {MODEL_PATH}")
        return True

    # Create models directory
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading ArcFace R50 model from {ARCFACE_MODEL_URL}")
    logger.info(f"Saving to {MODEL_PATH}")
    logger.info("This may take a few minutes...")

    try:
        urllib.request.urlretrieve(ARCFACE_MODEL_URL, MODEL_PATH)
        logger.info(f"✓ Download complete! Model saved to {MODEL_PATH}")
        logger.info(f"Model size: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")
        return True
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        logger.error("")
        logger.error("Please download manually from:")
        logger.error(f"  {ARCFACE_MODEL_URL}")
        logger.error(f"And save to:")
        logger.error(f"  {MODEL_PATH.absolute()}")
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
