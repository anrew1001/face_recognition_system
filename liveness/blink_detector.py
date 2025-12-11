"""Liveness detection via eye blinking using Eye Aspect Ratio (EAR)."""
import logging
from collections import deque
from typing import Optional, Tuple

import numpy as np

from recognition.types import FaceDetection

logger = logging.getLogger(__name__)


class BlinkDetector:
    """Detect liveness by analyzing eye blinking patterns.

    Uses Eye Aspect Ratio (EAR) to identify blinks and tracks blink sequences
    to determine if face is alive. Requires face landmarks with eyes data.

    The detector maintains state across frames to track blink sequences and
    can be reset between different persons or sessions.

    Attributes:
        ear_threshold: EAR below this value indicates eyes closed (default 0.2).
        consecutive_frames_closed: Frames needed to confirm blink (default 2).
        blink_sequence_length: Number of blinks to track (default 5).
    """

    # Standard eye landmark indices (SCRFD format - 5 landmarks: eyes, nose, mouth)
    LEFT_EYE_INDICES = (0, 1)  # Left eye points
    RIGHT_EYE_INDICES = (2, 3)  # Right eye points

    def __init__(
        self,
        ear_threshold: float = 0.2,
        consecutive_frames_closed: int = 2,
        blink_sequence_length: int = 5,
    ) -> None:
        """Initialize blink detector.

        Args:
            ear_threshold: EAR threshold to classify eyes as closed (0.0-1.0).
                Lower value = stricter criteria for detecting closed eyes.
            consecutive_frames_closed: Number of consecutive frames with closed eyes
                required to count as a blink event.
            blink_sequence_length: Number of recent blinks to track for liveness
                estimation.
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames_closed = consecutive_frames_closed
        self.blink_sequence_length = blink_sequence_length

        # State tracking
        self._frames_closed = 0
        self._blink_history: deque[bool] = deque(
            maxlen=blink_sequence_length
        )  # Recent blink events
        self._last_ear_left = 1.0
        self._last_ear_right = 1.0

    def detect(self, detection: FaceDetection) -> bool:
        """Detect liveness from face detection with landmarks.

        Analyzes eye aspect ratio and blink patterns. Returns True if face
        shows signs of life (recent blinking activity).

        Args:
            detection: FaceDetection with landmarks array (N, 2) containing
                eye coordinates. Expected format: 5 landmarks
                (left_eye, right_eye, nose, left_mouth, right_mouth).

        Returns:
            True if liveness detected (recent blinking), False otherwise.

        Raises:
            ValueError: If detection lacks landmarks or has invalid format.
        """
        if detection.landmarks is None:
            raise ValueError("FaceDetection must contain landmarks for blink detection")

        if detection.landmarks.shape[0] < 4:
            raise ValueError(
                f"Expected at least 4 landmarks (2 eyes), got {detection.landmarks.shape[0]}"
            )

        if detection.landmarks.shape[1] != 2:
            raise ValueError(
                f"Landmarks must be (N, 2), got shape {detection.landmarks.shape}"
            )

        # Extract eye landmarks
        landmarks = detection.landmarks
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]

        # Compute Eye Aspect Ratio for both eyes
        ear_left = self._compute_ear(left_eye)
        ear_right = self._compute_ear(right_eye)

        self._last_ear_left = ear_left
        self._last_ear_right = ear_right

        # Determine if eyes are currently closed
        eyes_closed = (ear_left < self.ear_threshold) or (
            ear_right < self.ear_threshold
        )

        # Track consecutive closed frames
        if eyes_closed:
            self._frames_closed += 1
        else:
            # Eyes opened - check if this was a blink
            if self._frames_closed >= self.consecutive_frames_closed:
                # Valid blink detected
                self._blink_history.append(True)
                logger.debug(
                    f"Blink detected (closed for {self._frames_closed} frames, "
                    f"EAR_L={ear_left:.3f}, EAR_R={ear_right:.3f})"
                )
            elif self._frames_closed > 0:
                # Short eye closure (noise), don't count as blink
                self._blink_history.append(False)

            self._frames_closed = 0

        # Liveness: detect if we've seen recent blinking activity
        return self._is_alive()

    def _compute_ear(self, eye_landmarks: np.ndarray) -> float:
        """Compute Eye Aspect Ratio for single eye.

        Uses formula from Soukupová & Tereza (2016):
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

        For 2-point eyes (horizontal and vertical eye points):
        EAR = vertical_distance / horizontal_distance

        Args:
            eye_landmarks: Array of shape (2, 2) with eye points [p1, p2]
                where p1 = left/right point, p2 = vertical center point.

        Returns:
            EAR value in range [0.0, 1.0+], where low values indicate closed eye.
        """
        if eye_landmarks.shape != (2, 2):
            logger.warning(f"Expected (2, 2) eye landmarks, got {eye_landmarks.shape}")
            return 1.0

        # For 2-point landmarks: p1 is horizontal extent, p2 is vertical
        p1 = eye_landmarks[0]  # Left/right point
        p2 = eye_landmarks[1]  # Vertical point

        # Vertical distance (eye height)
        vertical_dist = np.linalg.norm(p2 - p1)

        # Horizontal distance (eye width) - approximate as distance between points
        # For simplified 2-point landmarks, use distance between them
        horizontal_dist = np.linalg.norm(p1 - p1) + 1e-6  # Avoid division by zero

        if horizontal_dist < 1e-6:
            # Degenerate case: points too close
            horizontal_dist = vertical_dist + 1e-6

        # EAR = vertical / horizontal
        ear = vertical_dist / max(horizontal_dist, 1.0)

        return float(np.clip(ear, 0.0, 1.0))

    def _is_alive(self) -> bool:
        """Determine liveness from blink history.

        Returns True if recent blink activity detected, indicating
        a real person rather than a static image or video replay.

        Strategy: Require at least 1 blink in recent history.
        Can be extended for more sophisticated patterns.

        Returns:
            True if liveness likely, False if no recent blinking.
        """
        if len(self._blink_history) == 0:
            return False

        # Check if any recent blinks detected
        recent_blinks = sum(self._blink_history)
        return recent_blinks >= 1

    def get_ear(self) -> Tuple[float, float]:
        """Get latest Eye Aspect Ratios for both eyes.

        Returns:
            Tuple of (ear_left, ear_right) from last detect() call.
        """
        return (self._last_ear_left, self._last_ear_right)

    def get_blink_count(self) -> int:
        """Get number of blinks detected in recent history.

        Returns:
            Count of True values in blink history.
        """
        return sum(self._blink_history)

    def get_blink_history(self) -> Tuple[bool, ...]:
        """Get recent blink detection history.

        Returns:
            Tuple of recent blink events (True = blink detected, False = no blink).
        """
        return tuple(self._blink_history)

    def reset(self) -> None:
        """Reset detector state for new person or session.

        Clears blink history and frame counters.
        """
        self._frames_closed = 0
        self._blink_history.clear()
        self._last_ear_left = 1.0
        self._last_ear_right = 1.0
        logger.debug("BlinkDetector reset")

    def get_stats(self) -> dict:
        """Get current detector statistics.

        Returns:
            Dict with detector state: frames_closed, blink_count, blink_history.
        """
        return {
            "frames_closed": self._frames_closed,
            "blink_count": self.get_blink_count(),
            "blink_history": self.get_blink_history(),
            "ear_left": self._last_ear_left,
            "ear_right": self._last_ear_right,
            "threshold": self.ear_threshold,
        }