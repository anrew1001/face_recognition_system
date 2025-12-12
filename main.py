"""Face Recognition System - Real-time face detection, liveness, and identification."""
import logging
import os
from pathlib import Path

import cv2

from core.config import AppConfig
from database import IdentityDatabase
from liveness import MediaPipeLivenessDetector
from recognition import registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FaceRecognitionPipeline:
    """Real-time face recognition pipeline with liveness detection."""

    # Display constants
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    THICKNESS = 2
    COLOR_MATCH = (0, 255, 0)  # Green
    COLOR_MISMATCH = (0, 0, 255)  # Red
    COLOR_UNKNOWN = (255, 165, 0)  # Orange
    COLOR_DEAD = (0, 165, 255)  # Orange-Red (not alive)

    def __init__(
        self,
        config_path: str = "config/recognition.yaml",
        db_path: str = "data/identities.npz",
        camera_id: int = 0,
        confidence_threshold: float = 0.5,
        similarity_threshold: float = 0.5,
    ) -> None:
        """Initialize face recognition pipeline.

        Args:
            config_path: Path to config YAML file.
            db_path: Path to identity database NPZ file.
            camera_id: Camera device ID (default 0 = primary camera).
            confidence_threshold: Minimum face detection confidence.
            similarity_threshold: Minimum similarity for identity match.
        """
        self.config_path = config_path
        self.db_path = db_path
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold

        # Initialize components
        self.config = AppConfig.from_yaml(config_path)
        self.model = None
        self.db = IdentityDatabase(max_embeddings_per_identity=5)
        self.liveness_detector = MediaPipeLivenessDetector(
            ear_threshold=0.25,
            blink_consecutive_frames=2,
            live_timeout=10.0
        )

        logger.info("Face Recognition Pipeline initialized")

    def setup(self) -> bool:
        """Load model and database. Returns True if successful."""
        try:
            # Load model
            model_name = self.config.get_active_model()
            self.model = registry.get(model_name)
            self.model.load()
            logger.info(f"Loaded model: {self.model.info.name} v{self.model.info.version}")
            logger.info(f"Model fingerprint: {self.model.info.fingerprint()}")

            # Check for encryption passphrase in environment variable
            passphrase = os.getenv("FACE_DB_PASSPHRASE")
            if passphrase:
                logger.info("Encryption passphrase found in environment (FACE_DB_PASSPHRASE)")

            # Load identity database if exists
            db_path = Path(self.db_path)
            encrypted_path = db_path.with_suffix(db_path.suffix + '.enc')

            if encrypted_path.exists():
                # Encrypted database exists
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

    def run(self) -> None:
        """Run real-time face recognition from camera."""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return

        logger.info("Starting camera feed (press 'q' to quit, 's' to save identity)")
        frame_count = 0
        pending_identity_name = None

        # FPS tracking
        import time
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                frame_count += 1
                fps_frame_count += 1
                h, w = frame.shape[:2]

                # Calculate FPS every 30 frames
                if fps_frame_count >= 30:
                    fps_end_time = time.time()
                    elapsed = fps_end_time - fps_start_time
                    current_fps = fps_frame_count / elapsed if elapsed > 0 else 0.0
                    fps_start_time = fps_end_time
                    fps_frame_count = 0

                # 1. Liveness Detection (MediaPipe)
                is_alive, ear_value = self.liveness_detector.process_frame(frame)

                # 2. Face Detection (with frame_id for caching)
                detections = self.model.detect_faces(frame, frame_id=frame_count)

                for detection in detections:
                    x1, y1, x2, y2 = detection.bbox

                    # 3. Extract Embedding (uses cached results from detect_faces)
                    embedding_result = self.model.extract_embedding(
                        frame, detection, frame_id=frame_count
                    )

                    # 4. Match with Database
                    match_text = "UNKNOWN"
                    match_color = self.COLOR_UNKNOWN

                    if embedding_result and self.db.list_identities():
                        match = self.db.find_match(
                            embedding_result.embedding,
                            threshold=self.similarity_threshold,
                            model_fingerprint=embedding_result.model_fingerprint,
                        )
                        if match:
                            name, score = match
                            match_text = f"{name} ({score:.3f})"
                            match_color = self.COLOR_MATCH

                    # 5. Draw bounding box (color based on liveness and match)
                    bbox_color = match_color if is_alive else self.COLOR_DEAD
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        bbox_color,
                        self.THICKNESS,
                    )

                    # 6. Display identity label
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

                    # Identity match
                    # y_offset -= 25
                    # cv2.putText(
                    #     frame,
                    #     match_text,
                    #     (x1, y_offset),
                    #     self.FONT,
                    #     self.FONT_SCALE,
                    #     match_color,
                    #     self.THICKNESS,
                    # )

                    # Confidence score
                    # conf_text = f"Det: {detection.confidence:.2f}"
                    # y_offset = y2 + 20
                    # cv2.putText(
                    #     frame,
                    #     conf_text,
                    #     (x1, y_offset),
                    #     self.FONT,
                    #     self.FONT_SCALE,
                    #     self.COLOR_MATCH,
                    #     1,
                    # )

                    # Store embedding for pending identity
                    # if pending_identity_name and embedding_result:
                    #     identity_name = pending_identity_name
                    #     self.db.add_embedding_result(identity_name, embedding_result)
                    #     pending_identity_name = None
                    #     logger.info(f"Added embedding for '{identity_name}'")

                # Display frame info with FPS and liveness
                liveness_status = "ALIVE" if is_alive else "NO BLINK"
                liveness_color = self.COLOR_MATCH if is_alive else self.COLOR_DEAD
                info_text = f"FPS: {current_fps:.1f} | Liveness: {liveness_status} (EAR: {ear_value:.3f}) | Faces: {len(detections)}"
                cv2.putText(
                    frame,
                    info_text,
                    (10, 30),
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
                    (10, 55),
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
                cv2.imshow("Face Recognition", frame)

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
    """Main entry point for face recognition system."""
    logger.info("Face Recognition System v0.1")

    # Initialize pipeline
    pipeline = FaceRecognitionPipeline(
        config_path="config/recognition.yaml",
        db_path="data/identities.npz",
        camera_id=0,
        confidence_threshold=0.5,
        similarity_threshold=0.5,
    )

    # Setup and run
    if pipeline.setup():
        pipeline.run()
    else:
        logger.error("Pipeline setup failed")


if __name__ == "__main__":
    main()
