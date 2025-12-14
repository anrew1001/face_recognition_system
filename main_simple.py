"""Simple Face Recognition System - Unified InsightFace pipeline.

This is a simplified version of the face recognition system that uses InsightFace
for both detection and recognition in a single unified pipeline. It's optimized
for M1 Macs with careful performance tuning.

Key differences from main.py:
- Uses InsightFace buffalo_l model for both detection and recognition
- Simpler architecture with fewer components
- Optimized for 12-18 FPS on M1 16GB
- Detection size: 320x320 (optimal for M1 performance)
- Frame downscaling to 480px (same as main.py)
- Frame skipping: every 2nd frame
- Liveness checking: every 5th processed frame

Run with:
    python main_simple.py

For encrypted database:
    bash run_encrypted.sh
    # or
    export FACE_DB_PASSPHRASE="your_password"
    python main_simple.py
"""
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from database import IdentityDatabase
from liveness import MediaPipeLivenessDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SimpleFaceRecognitionPipeline:
    """Simplified face recognition pipeline using InsightFace unified model.

    This pipeline uses InsightFace FaceAnalysis for both detection and recognition,
    providing a simpler architecture compared to the composite detector+recognizer
    approach in main.py.

    Performance optimizations:
    - Detection size: 320x320 (optimized for M1)
    - Frame downscaling: max 480px on longer side
    - Frame skipping: process every 2nd frame
    - Liveness checking: every 5th processed frame
    - CPU inference with ctx_id=-1

    Expected performance: 12-18 FPS on M1 16GB
    """

    # Display constants
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    THICKNESS = 2
    COLOR_MATCH = (0, 255, 0)  # Green
    COLOR_MISMATCH = (0, 0, 255)  # Red
    COLOR_UNKNOWN = (255, 165, 0)  # Orange
    COLOR_DEAD = (0, 165, 255)  # Orange-Red (not alive)

    # Performance optimization constants
    MAX_PROCESS_DIM = 480  # Downscale to 480p for processing
    FRAME_SKIP_INTERVAL = 2  # Process every 2nd frame
    LIVENESS_CHECK_INTERVAL = 5  # Check liveness every 5th processed frame

    # InsightFace configuration
    DETECTION_SIZE = (320, 320)  # Optimized for M1 performance
    CTX_ID = -1  # CPU inference

    # Model fingerprint for InsightFace buffalo_l
    # This matches the fingerprint from recognition/insightface_adapter.py
    MODEL_FINGERPRINT = "212a5ec8dbd9c95e"

    def __init__(
        self,
        db_path: str = "data/identities.npz",
        camera_id: int = 0,
        similarity_threshold: float = 0.5,
    ) -> None:
        """Initialize simplified face recognition pipeline.

        Args:
            db_path: Path to identity database NPZ file.
            camera_id: Camera device ID (default 0 = primary camera).
            similarity_threshold: Minimum similarity for identity match (0.0-1.0).
        """
        self.db_path = db_path
        self.camera_id = camera_id
        self.similarity_threshold = similarity_threshold

        # Initialize components
        self.model = None  # InsightFace FaceAnalysis instance
        self.db = IdentityDatabase(max_embeddings_per_identity=5)
        self.liveness_detector = MediaPipeLivenessDetector(
            ear_threshold=0.175,
            blink_consecutive_frames=1,
            live_timeout=3.0
        )

        logger.info(
            f"Simple Face Recognition Pipeline initialized "
            f"(detection_size={self.DETECTION_SIZE}, ctx_id={self.CTX_ID})"
        )

    def setup(self) -> bool:
        """Load InsightFace model and database. Returns True if successful."""
        try:
            # Load InsightFace model
            logger.info("Loading InsightFace buffalo_l model...")
            from insightface.app import FaceAnalysis

            self.model = FaceAnalysis(
                allowed_modules=['detection', 'recognition']
            )
            self.model.prepare(
                ctx_id=self.CTX_ID,
                det_size=self.DETECTION_SIZE
            )

            logger.info(
                f"✓ InsightFace loaded: det_size={self.DETECTION_SIZE}, "
                f"ctx_id={self.CTX_ID}"
            )
            logger.info(f"Model fingerprint: {self.MODEL_FINGERPRINT}")

            # Check for encryption passphrase
            passphrase = os.getenv("FACE_DB_PASSPHRASE")
            if passphrase:
                logger.info("Encryption passphrase found in environment (FACE_DB_PASSPHRASE)")

            # Load identity database if exists
            db_path = Path(self.db_path)
            encrypted_path = db_path.with_suffix(db_path.suffix + '.enc')

            if encrypted_path.exists():
                # Encrypted database exists
                if not passphrase:
                    logger.error("")
                    logger.error("=" * 70)
                    logger.error("ENCRYPTED DATABASE FOUND BUT NO PASSPHRASE PROVIDED")
                    logger.error("=" * 70)
                    logger.error("")
                    logger.error(f"Encrypted database: {encrypted_path}")
                    logger.error("")
                    logger.error("SOLUTION: Set the encryption password before running:")
                    logger.error("")
                    logger.error("  Option 1: Use the run script (EASIEST)")
                    logger.error("    bash run_encrypted.sh")
                    logger.error("")
                    logger.error("  Option 2: Set environment variable")
                    logger.error('    export FACE_DB_PASSPHRASE="123456789"')
                    logger.error("    python main_simple.py")
                    logger.error("")
                    logger.error("  Option 3: Use Python directly")
                    logger.error('    FACE_DB_PASSPHRASE="123456789" python main_simple.py')
                    logger.error("")
                    logger.error("=" * 70)
                    return False

                self.db.load(str(db_path), passphrase=passphrase)
                logger.info(
                    f"Loaded encrypted identity database: {len(self.db.list_identities())} identities"
                )
            elif db_path.exists():
                # Unencrypted database exists
                self.db.load(str(db_path), passphrase=passphrase)
                logger.info(f"Loaded identity database: {len(self.db.list_identities())} identities")
            else:
                logger.info("Identity database not found, starting with empty DB")

            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False

    def _downscale_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Downscale frame to MAX_PROCESS_DIM for processing.

        Args:
            frame: Original frame (HxWx3 BGR uint8).

        Returns:
            Tuple of (downscaled_frame, scale_factor).
            scale_factor maps coordinates from downscaled to original frame.
        """
        h, w = frame.shape[:2]

        if max(h, w) > self.MAX_PROCESS_DIM:
            scale = self.MAX_PROCESS_DIM / float(max(h, w))
            proc_frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )
            scale_factor = 1.0 / scale  # Scale from proc_frame back to original
        else:
            proc_frame = frame
            scale_factor = 1.0

        return proc_frame, scale_factor

    def run(self) -> None:
        """Run real-time face recognition from camera."""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return

        logger.info("Starting camera feed (press 'q' to quit, 's' to save identity)")
        frame_count = 0

        # FPS tracking
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0

        # Frame processing state
        frame_skip_counter = 0
        liveness_check_counter = 0

        # Cached state for skipped frames
        cached_faces = []
        cached_match_results = {}  # Maps face index to (match_text, match_color)
        cached_is_alive = False
        cached_ear_value = 0.0
        cached_aligned_crops = {}  # Maps face index to aligned face crop

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                frame_count += 1
                fps_frame_count += 1
                h, w = frame.shape[:2]

                # Downscale frame for processing
                proc_frame, scale_factor = self._downscale_frame(frame)

                # Calculate FPS every 30 frames
                if fps_frame_count >= 30:
                    fps_end_time = time.time()
                    elapsed = fps_end_time - fps_start_time
                    current_fps = fps_frame_count / elapsed if elapsed > 0 else 0.0
                    fps_start_time = fps_end_time
                    fps_frame_count = 0

                # Frame skipping: Process every Nth frame
                frame_skip_counter += 1
                should_process = (frame_skip_counter >= self.FRAME_SKIP_INTERVAL)

                if should_process:
                    frame_skip_counter = 0

                    # 1. Face Detection and Recognition (unified in InsightFace)
                    # InsightFace.get() performs detection and embedding extraction in one call
                    faces = self.model.get(proc_frame)
                    cached_faces = faces

                    # 2. Liveness Detection (only if faces detected and interval met)
                    if len(faces) > 0:
                        liveness_check_counter += 1
                        if liveness_check_counter >= self.LIVENESS_CHECK_INTERVAL:
                            # Use proc_frame to reduce MediaPipe overhead
                            is_alive, ear_value = self.liveness_detector.process_frame(proc_frame)
                            cached_is_alive = is_alive
                            cached_ear_value = ear_value
                            liveness_check_counter = 0
                        else:
                            is_alive = cached_is_alive
                            ear_value = cached_ear_value
                    else:
                        # No faces detected
                        is_alive = False
                        ear_value = 0.0
                        cached_is_alive = False
                        cached_ear_value = 0.0

                    # 3. Match with Database
                    cached_match_results = {}
                    cached_aligned_crops = {}

                    for idx, face in enumerate(faces):
                        # Extract embedding (already computed by InsightFace)
                        embedding = face.embedding.astype(np.float32)

                        # Normalize embedding
                        norm = np.linalg.norm(embedding)
                        if norm > 1e-10:
                            embedding = embedding / norm
                        else:
                            logger.warning("Face embedding has near-zero norm")
                            continue

                        # Match with database
                        match_text = "UNKNOWN"
                        match_color = self.COLOR_UNKNOWN

                        if self.db.list_identities():
                            match = self.db.find_match(
                                embedding,
                                threshold=self.similarity_threshold,
                                model_fingerprint=self.MODEL_FINGERPRINT,
                            )
                            if match:
                                name, score = match
                                match_text = f"{name} ({score:.3f})"
                                match_color = self.COLOR_MATCH

                        # Cache match result
                        cached_match_results[idx] = (match_text, match_color)

                        # Extract aligned face crop for visualization
                        # InsightFace provides aligned face in face.normed_embedding
                        # But we can also manually extract the bbox region
                        if hasattr(face, 'bbox'):
                            x1, y1, x2, y2 = [int(v) for v in face.bbox]
                            # Ensure bbox is within proc_frame bounds
                            ph, pw = proc_frame.shape[:2]
                            x1 = max(0, min(x1, pw))
                            y1 = max(0, min(y1, ph))
                            x2 = max(0, min(x2, pw))
                            y2 = max(0, min(y2, ph))

                            if x2 > x1 and y2 > y1:
                                aligned_crop = proc_frame[y1:y2, x1:x2].copy()
                                cached_aligned_crops[idx] = aligned_crop

                else:
                    # Use cached results on skip frames
                    faces = cached_faces
                    is_alive = cached_is_alive
                    ear_value = cached_ear_value

                # Draw detections and results
                for idx, face in enumerate(faces):
                    # Get bbox and scale to original frame coordinates
                    x1, y1, x2, y2 = [int(v * scale_factor) for v in face.bbox]

                    # Ensure bbox is within frame bounds
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))

                    # Get cached match results
                    match_text, match_color = cached_match_results.get(
                        idx, ("UNKNOWN", self.COLOR_UNKNOWN)
                    )

                    # Draw bounding box (color based on liveness and match)
                    bbox_color = match_color if is_alive else self.COLOR_DEAD
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        bbox_color,
                        self.THICKNESS,
                    )

                    # Display identity label
                    y_offset = y1 - 10
                    cv2.putText(
                        frame,
                        match_text,
                        (x1, y_offset),
                        self.FONT,
                        self.FONT_SCALE,
                        bbox_color,
                        self.THICKNESS,
                    )

                    # Display aligned face crop in corner (for verification)
                    if idx in cached_aligned_crops:
                        aligned_crop = cached_aligned_crops[idx]
                        # Resize to fixed size for display
                        crop_display_size = (100, 100)
                        aligned_resized = cv2.resize(aligned_crop, crop_display_size)

                        # Position in top-right corner (with offset for multiple faces)
                        crop_x = w - crop_display_size[0] - 10
                        crop_y = 10 + idx * (crop_display_size[1] + 10)

                        # Ensure crop fits in frame
                        if crop_y + crop_display_size[1] <= h:
                            frame[
                                crop_y:crop_y + crop_display_size[1],
                                crop_x:crop_x + crop_display_size[0]
                            ] = aligned_resized

                            # Draw border around crop
                            cv2.rectangle(
                                frame,
                                (crop_x, crop_y),
                                (crop_x + crop_display_size[0], crop_y + crop_display_size[1]),
                                bbox_color,
                                2
                            )

                # Display frame info with FPS and liveness
                liveness_status = "ALIVE" if is_alive else "NO BLINK"
                liveness_color = self.COLOR_MATCH if is_alive else self.COLOR_DEAD

                info_text = f"FPS: {current_fps:.1f} | InsightFace buffalo_l | Faces: {len(faces)}"
                cv2.putText(
                    frame,
                    info_text,
                    (10, 30),
                    self.FONT,
                    0.5,
                    liveness_color,
                    1,
                )

                # Liveness info on second line
                liveness_text = f"Liveness: {liveness_status} (EAR: {ear_value:.3f})"
                cv2.putText(
                    frame,
                    liveness_text,
                    (10, 50),
                    self.FONT,
                    0.5,
                    liveness_color,
                    1,
                )

                # Database info
                db_text = f"Identities in DB: {len(self.db.list_identities())}"
                cv2.putText(
                    frame,
                    db_text,
                    (10, 70),
                    self.FONT,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                help_text = "Press 'q' to quit, 'r' to reset, 's' to save DB"
                cv2.putText(
                    frame,
                    help_text,
                    (10, h - 10),
                    self.FONT,
                    0.4,
                    (200, 200, 200),
                    1,
                )

                # Display frame
                cv2.imshow("Face Recognition - Simple Pipeline", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Exiting...")
                    break
                elif key == ord("s"):
                    self._save_database()
                elif key == ord("r"):
                    self._reset_database()
                    self.liveness_detector.reset()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera feed closed")

    def _save_database(self) -> None:
        """Save identity database to disk with optional encryption."""
        try:
            db_path = Path(self.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Check for encryption passphrase
            passphrase = os.getenv("FACE_DB_PASSPHRASE")

            if passphrase:
                self.db.save(str(db_path), passphrase=passphrase)
                encrypted_path = db_path.with_suffix(db_path.suffix + '.enc')
                logger.info(f"Database saved with encryption to {encrypted_path}")
            else:
                self.db.save(str(db_path))
                logger.info(f"Database saved (unencrypted) to {db_path}")

        except Exception as e:
            logger.error(f"Failed to save database: {e}")

    def _reset_database(self) -> None:
        """Clear all identities from database."""
        self.db.clear()
        logger.info("Database cleared")


def main() -> None:
    """Main entry point for simple face recognition system."""
    logger.info("Simple Face Recognition System v0.1")
    logger.info("Using unified InsightFace pipeline (buffalo_l)")

    # Initialize pipeline
    pipeline = SimpleFaceRecognitionPipeline(
        db_path="data/identities.npz",
        camera_id=0,
        similarity_threshold=0.5,
    )

    # Setup and run
    if pipeline.setup():
        pipeline.run()
    else:
        logger.error("Pipeline setup failed")


if __name__ == "__main__":
    main()
