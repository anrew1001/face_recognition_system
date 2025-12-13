"""MediaPipe-based liveness detection using Eye Aspect Ratio (EAR) for blink detection.

This detector uses MediaPipe FaceMesh's 478 facial landmarks to compute accurate
EAR values for blink detection. Requires 6 landmarks per eye (12 total) for proper
vertical and horizontal distance calculations.
"""
import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MediaPipeLivenessDetector:
    """Liveness detection using MediaPipe FaceMesh and Eye Aspect Ratio (EAR).

    Uses 6-point EAR calculation for each eye to detect blinks. A person is
    considered "alive" if they blink within the configured timeout period.

    Attributes:
        ear_threshold: EAR value below which eye is considered closed.
        blink_consecutive_frames: Number of consecutive frames required to register a blink.
        live_timeout: Seconds after last blink before person is considered "not alive".

    Example:
        >>> detector = MediaPipeLivenessDetector()
        >>> is_alive, ear = detector.process_frame(frame)
        >>> print(f"Alive: {is_alive}, EAR: {ear:.3f}")
    """

    # MediaPipe FaceMesh landmark indices for 6-point EAR calculation
    # Left eye: 6 points around the eye contour
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    # Right eye: 6 points around the eye contour
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

    def __init__(
        self,
        ear_threshold: float = 0.25,
        blink_consecutive_frames: int = 2,
        live_timeout: float = 10.0,
    ) -> None:
        """Initialize MediaPipe liveness detector.

        Args:
            ear_threshold: EAR value below which eye is considered closed (default 0.25).
            blink_consecutive_frames: Frames with closed eyes to register blink (default 2).
            live_timeout: Seconds after last blink before "not alive" (default 10.0).
        """
        self.ear_threshold = ear_threshold
        self.blink_consecutive_frames = blink_consecutive_frames
        self.live_timeout = live_timeout

        # Blink tracking state
        self._closed_frames = 0
        self._last_blink_time: Optional[float] = None
        self._blink_count = 0

        # Initialize MediaPipe FaceMesh
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info(
                f"MediaPipe liveness detector initialized: "
                f"EAR={ear_threshold}, frames={blink_consecutive_frames}, "
                f"timeout={live_timeout}s"
            )
        except ImportError as e:
            raise RuntimeError(
                "mediapipe not installed. Install with: pip install mediapipe>=0.10.0"
            ) from e

    def _calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio for 6-point eye landmark configuration.

        EAR formula: (vertical_dist_1 + vertical_dist_2) / (2 * horizontal_dist)

        Args:
            eye_landmarks: Array of 6 (x, y) landmarks for one eye.

        Returns:
            EAR value (typically 0.2-0.4 when eye open, <0.25 when closed).
        """
        # Vertical distances
        # Points 1-5 and 2-4 form the vertical measurements
        vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

        # Horizontal distance
        # Points 0-3 form the horizontal measurement
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

        # Avoid division by zero
        if horizontal < 1e-6:
            return 0.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return float(ear)

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Process frame and detect liveness based on blink detection.

        Args:
            frame: Input BGR image from camera (HxWx3 uint8 numpy array).

        Returns:
            Tuple of (is_alive, average_ear):
            - is_alive: True if blink detected within timeout period
            - average_ear: Current average EAR for both eyes (0.0 if no face detected)

        Example:
            >>> is_alive, ear = detector.process_frame(frame)
            >>> if is_alive:
            ...     print(f"Person is alive! EAR: {ear:.3f}")
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        # No face detected
        if not results.multi_face_landmarks:
            return False, 0.0

        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        # Extract left eye landmarks
        left_eye = []
        for idx in self.LEFT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            left_eye.append([x, y])
        left_eye = np.array(left_eye, dtype=np.float32)

        # Extract right eye landmarks
        right_eye = []
        for idx in self.RIGHT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            right_eye.append([x, y])
        right_eye = np.array(right_eye, dtype=np.float32)

        # Calculate EAR for both eyes
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Blink detection logic
        current_time = time.time()

        if avg_ear < self.ear_threshold:
            # Eye is closed
            self._closed_frames += 1
        else:
            # Eye is open
            if self._closed_frames >= self.blink_consecutive_frames:
                # Blink detected!
                self._blink_count += 1
                self._last_blink_time = current_time
                logger.debug(f"Blink detected! Total blinks: {self._blink_count}")

            # Reset closed frame counter
            self._closed_frames = 0

        # Determine liveness
        is_alive = False
        if self._last_blink_time is not None:
            time_since_blink = current_time - self._last_blink_time
            is_alive = time_since_blink <= self.live_timeout

        return is_alive, avg_ear

    def reset(self) -> None:
        """Reset blink history and liveness state.

        Call this when a new person appears or when you want to restart tracking.
        """
        self._closed_frames = 0
        self._last_blink_time = None
        self._blink_count = 0
        logger.info("Liveness detector reset")

    def get_stats(self) -> dict:
        """Get current detection statistics.

        Returns:
            Dictionary with blink_count, last_blink_time, and time_since_blink.
        """
        current_time = time.time()
        time_since_blink = None
        if self._last_blink_time is not None:
            time_since_blink = current_time - self._last_blink_time

        return {
            "blink_count": self._blink_count,
            "last_blink_time": self._last_blink_time,
            "time_since_blink": time_since_blink,
            "closed_frames": self._closed_frames,
        }

    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()