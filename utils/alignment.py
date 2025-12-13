"""Face alignment utilities for ArcFace preprocessing.

This module provides face alignment functionality to prepare detected faces
for ArcFace embedding extraction. The alignment process uses 5-point facial
landmarks to normalize face pose and scale to 112x112 pixels.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

# ArcFace standard 5-point landmark reference positions for 112x112 output
# These are the canonical positions where we want landmarks to be mapped
# Coordinates are (x, y) for: left_eye, right_eye, nose, left_mouth, right_mouth
ARCFACE_SRC_POINTS = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041]   # Right mouth corner
], dtype=np.float32)


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: Tuple[int, int] = (112, 112),
) -> Optional[np.ndarray]:
    """Align face using 5-point landmarks to ArcFace standard format.

    Applies similarity transformation (translation, rotation, scaling) to align
    facial landmarks to canonical positions. This normalization improves
    embedding quality by ensuring consistent face pose and scale.

    Args:
        image: Input BGR image containing the face (HxWx3 uint8).
        landmarks: 5-point facial landmarks as Nx2 array (N>=5).
                  Expected order: left_eye, right_eye, nose, left_mouth, right_mouth.
        output_size: Output image size (height, width). Default: (112, 112) for ArcFace.

    Returns:
        Aligned face image as HxWx3 uint8 RGB array (note: BGR->RGB conversion).
        Returns None if alignment fails (e.g., invalid landmarks).

    Example:
        >>> # After face detection with SCRFD
        >>> detections = detector.detect_faces(image)
        >>> for det in detections:
        ...     aligned = align_face(image, det.landmarks)
        ...     embedding = arcface.extract_embedding(aligned)

    Note:
        - Input image should be BGR (OpenCV format)
        - Output image is RGB (ArcFace expects RGB)
        - Landmarks must be in the same coordinate system as the input image
    """
    if landmarks is None or len(landmarks) < 5:
        return None

    # Use first 5 landmarks (standard 5-point format)
    src_landmarks = landmarks[:5].astype(np.float32)

    # Compute similarity transformation matrix
    # This finds the best affine transform that maps src_landmarks to ARCFACE_SRC_POINTS
    tform = cv2.estimateAffinePartial2D(
        src_landmarks,
        ARCFACE_SRC_POINTS,
        method=cv2.LMEDS  # Robust estimation (handles outliers)
    )[0]

    if tform is None:
        return None

    # Apply transformation to get aligned face
    aligned = cv2.warpAffine(
        image,
        tform,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Convert BGR to RGB (ArcFace expects RGB input)
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

    return aligned_rgb


def align_face_similarity(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: Tuple[int, int] = (112, 112),
) -> Optional[np.ndarray]:
    """Alternative alignment using skimage-style similarity transform.

    This is a fallback method that uses only eye landmarks for alignment.
    Useful when full 5-point landmarks are unreliable.

    Args:
        image: Input BGR image (HxWx3 uint8).
        landmarks: Facial landmarks with at least 2 eye points.
        output_size: Output size (height, width).

    Returns:
        Aligned face as RGB uint8 array, or None if fails.
    """
    if landmarks is None or len(landmarks) < 2:
        return None

    # Use only eye landmarks (first 2 points)
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    # Compute eye center and angle
    eye_center = (left_eye + right_eye) / 2.0
    eye_angle = np.degrees(np.arctan2(
        right_eye[1] - left_eye[1],
        right_eye[0] - left_eye[0]
    ))

    # Target eye distance (based on ArcFace canonical positions)
    target_eye_dist = ARCFACE_SRC_POINTS[1][0] - ARCFACE_SRC_POINTS[0][0]
    current_eye_dist = np.linalg.norm(right_eye - left_eye)
    scale = target_eye_dist / current_eye_dist if current_eye_dist > 0 else 1.0

    # Target center (middle of output image)
    target_center = np.array(output_size) / 2.0

    # Build transformation matrix
    M = cv2.getRotationMatrix2D(
        tuple(eye_center),
        eye_angle,
        scale
    )

    # Adjust translation to center the face
    M[0, 2] += target_center[1] - eye_center[0]
    M[1, 2] += target_center[0] - eye_center[1]

    # Apply transformation
    aligned = cv2.warpAffine(
        image,
        M,
        output_size,
        flags=cv2.INTER_LINEAR
    )

    # Convert BGR to RGB
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

    return aligned_rgb
