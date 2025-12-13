#!/usr/bin/env python3
"""Download SCRFD_2.5G_KPS model from HuggingFace.

This script downloads the SCRFD_2.5G_KPS ONNX model from the official
InsightFace repository on HuggingFace.

Model source:
https://huggingface.co/public-data/insightface/tree/main/models/scrfd_2.5g_bnkps

Usage:
    python scripts/download_scrfd_25g.py
"""
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Model URL from HuggingFace
MODEL_URL = (
    "https://huggingface.co/public-data/insightface/resolve/main/"
    "models/scrfd_2.5g_bnkps/det_2.5g.onnx"
)

# Target path
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TARGET_PATH = MODELS_DIR / "det_2.5g.onnx"


def download_progress(block_num, block_size, total_size):
    """Display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        print(
            f"\rDownloading: {percent:.1f}% "
            f"({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)",
            end="",
            flush=True
        )


def main():
    """Download SCRFD_2.5G_KPS model."""
    print("SCRFD_2.5G_KPS Model Downloader")
    print("=" * 50)
    print(f"Source: {MODEL_URL}")
    print(f"Target: {TARGET_PATH}")
    print()

    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if model already exists
    if TARGET_PATH.exists():
        size_mb = TARGET_PATH.stat().st_size / 1024 / 1024
        print(f"Model already exists: {TARGET_PATH} ({size_mb:.1f}MB)")
        response = input("Download again? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return 0

    # Download model
    try:
        print("Starting download...")
        urlretrieve(MODEL_URL, TARGET_PATH, reporthook=download_progress)
        print()  # New line after progress

        # Verify download
        if TARGET_PATH.exists():
            size_mb = TARGET_PATH.stat().st_size / 1024 / 1024
            print(f"\nDownload successful!")
            print(f"Model saved to: {TARGET_PATH}")
            print(f"File size: {size_mb:.1f}MB")

            # Verify it's a valid ONNX file (basic check)
            with open(TARGET_PATH, "rb") as f:
                header = f.read(4)
                if header[:4] != b'\x08\x03\x12\x02' and header[:4] != b'\x08\x07\x12\x02':
                    # ONNX files typically start with protobuf headers
                    print("\nWARNING: Downloaded file may not be a valid ONNX model!")
                    print("Please verify the download manually.")
                else:
                    print("\nFile appears to be a valid ONNX model.")

            return 0
        else:
            print("\nERROR: Download completed but file not found!")
            return 1

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        if TARGET_PATH.exists():
            print("Removing incomplete download...")
            TARGET_PATH.unlink()
        return 1

    except Exception as e:
        print(f"\n\nERROR: Download failed: {e}")
        if TARGET_PATH.exists():
            print("Removing incomplete download...")
            TARGET_PATH.unlink()
        return 1


if __name__ == "__main__":
    sys.exit(main())
