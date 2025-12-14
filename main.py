"""Face Recognition System - Real-time face detection, liveness, and identification."""
import logging
import os
from pathlib import Path

import cv2

from core.config import AppConfig
from database import IdentityDatabase
from liveness import MediaPipeLivenessDetector
from recognition import registry
from utils.alignment import align_face

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

        # Support both composite (detection + recognition) and single model architectures
        self.use_composite = self.config.use_composite_architecture()
        self.detector = None  # For composite: fast face detector (SCRFD)
        self.recognizer = None  # For composite: embedding extractor (ArcFace)
        self.model = None  # For single model: all-in-one (InsightFace)

        self.db = IdentityDatabase(max_embeddings_per_identity=5)
        self.liveness_detector = MediaPipeLivenessDetector(
            ear_threshold=0.25,
            blink_consecutive_frames=2,
            live_timeout=10.0
        )

        logger.info(
            f"Face Recognition Pipeline initialized "
            f"(composite={self.use_composite})"
        )

    def setup(self) -> bool:
        """Load model and database. Returns True if successful."""
        try:
            # Clear model cache to ensure fresh config is used
            # This prevents stale configuration from being cached
            registry._model_instances.clear()
            logger.debug("Cleared model cache to use fresh configuration")

            # Load models based on architecture choice
            if self.use_composite:
                # Composite architecture: fast detector + specialized recognizer
                detector_name = self.config.get_detection_model()
                recognizer_name = self.config.get_recognition_model()

                logger.info(f"Loading composite architecture:")
                logger.info(f"  Detector: {detector_name}")
                logger.info(f"  Recognizer: {recognizer_name}")

                # Load detector (e.g., SCRFD_2.5G)
                self.detector = registry.get(detector_name)
                self.detector.load()
                logger.info(
                    f"✓ Detector loaded: {self.detector.info.name} "
                    f"v{self.detector.info.version}"
                )
                logger.info(
                    f"  Detector config: model_path={self.detector._model_path}, "
                    f"det_size={self.detector._det_size}, "
                    f"confidence_threshold={self.detector._confidence_threshold}"
                )

                # Load recognizer (e.g., ArcFace)
                self.recognizer = registry.get(recognizer_name)
                self.recognizer.load()
                logger.info(
                    f"✓ Recognizer loaded: {self.recognizer.info.name} "
                    f"v{self.recognizer.info.version}"
                )
                logger.info(f"Model fingerprint: {self.recognizer.info.fingerprint()}")
            else:
                # Single model architecture (legacy)
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
                    logger.error("    python main.py")
                    logger.error("")
                    logger.error("  Option 3: Use Python directly")
                    logger.error('    FACE_DB_PASSPHRASE="123456789" python main.py')
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

        # Liveness state (cached when no faces detected)
        is_alive = False
        ear_value = 0.0

        # Frame processing parameters (performance optimization)
        MAX_PROCESS_DIM = 480  # Downscale to 480p for processing, keep original for display
        FRAME_SKIP_INTERVAL = 2  # Process every 2nd frame
        LIVENESS_CHECK_INTERVAL = 5  # Check liveness every 5th processed frame
        frame_skip_counter = 0
        liveness_check_counter = 0
        cached_detections = []
        cached_match_results = {}  # Maps detection bbox tuple to match_text and color
        cached_is_alive = False  # Cache liveness state
        cached_ear_value = 0.0  # Cache ear value for display

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                frame_count += 1
                fps_frame_count += 1
                h, w = frame.shape[:2]

                # Downscale frame for processing (reduce pixels by ~2.7x if 1280x720)
                if max(h, w) > MAX_PROCESS_DIM:
                    scale = MAX_PROCESS_DIM / float(max(h, w))
                    proc_frame = cv2.resize(
                        frame,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    proc_frame = frame

                # Calculate FPS every 30 frames
                if fps_frame_count >= 30:
                    fps_end_time = time.time()
                    elapsed = fps_end_time - fps_start_time
                    current_fps = fps_frame_count / elapsed if elapsed > 0 else 0.0
                    fps_start_time = fps_end_time
                    fps_frame_count = 0

                # Frame skipping: Process every Nth frame, use cached results on skip
                frame_skip_counter += 1
                should_process = (frame_skip_counter >= FRAME_SKIP_INTERVAL)

                if should_process:
                    frame_skip_counter = 0
                    # 1. Face Detection (MOVED BEFORE LIVENESS FOR PERFORMANCE)
                    if self.use_composite:
                        # Composite: Use fast detector (SCRFD_2.5G)
                        detections = self.detector.detect_faces(proc_frame, frame_id=frame_count)
                        # Debug logging for detection issues
                        if frame_count % 30 == 0:  # Log every 30 frames to avoid spam
                            logger.debug(
                                f"Frame {frame_count}: proc_frame shape={proc_frame.shape}, "
                                f"detector={self.detector.__class__.__name__}, "
                                f"threshold={self.detector._confidence_threshold}, "
                                f"detections={len(detections)}"
                            )
                    else:
                        # Legacy: Use single model
                        detections = self.model.detect_faces(proc_frame, frame_id=frame_count)

                    cached_detections = detections

                    # 2. Liveness Detection (ONLY IF FACES DETECTED AND INTERVAL MET)
                    if len(detections) > 0:
                        liveness_check_counter += 1
                        if liveness_check_counter >= LIVENESS_CHECK_INTERVAL:
                            # Use proc_frame to reduce MediaPipe overhead (~2.7x fewer pixels)
                            is_alive, ear_value = self.liveness_detector.process_frame(proc_frame)
                            cached_is_alive = is_alive
                            cached_ear_value = ear_value
                            liveness_check_counter = 0
                        else:
                            is_alive = cached_is_alive
                            ear_value = cached_ear_value
                else:
                    # Use cached detections and liveness state on skip frames
                    detections = cached_detections
                    is_alive = cached_is_alive
                    ear_value = cached_ear_value

                # Scale factor to map detection bbox from proc_frame back to original frame
                scale_factor = max(h, w) / float(MAX_PROCESS_DIM) if max(h, w) > MAX_PROCESS_DIM else 1.0

                # Process detections: extract embeddings and match with DB only on processed frames
                if should_process:
                    # Clear cache for new detections
                    cached_match_results = {}
                    for detection in detections:
                        x1, y1, x2, y2 = detection.bbox
                        # Scale bbox coordinates from proc_frame to original frame
                        if scale_factor != 1.0:
                            x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)

                        # 3. Extract Embedding
                        embedding_result = None
                        if self.use_composite:
                            # Composite: Align face and use specialized recognizer
                            if detection.landmarks is not None and len(detection.landmarks) >= 5:
                                # CRITICAL: detection.landmarks are in proc_frame coordinates
                                # align_face must use proc_frame with landmarks from proc_frame
                                aligned_face = align_face(proc_frame, detection.landmarks)
                                if aligned_face is not None:
                                    # Extract embedding from aligned face
                                    embedding_result = self.recognizer.extract_embedding(
                                        aligned_face, detection
                                    )
                                else:
                                    logger.debug(f"Face alignment failed for detection at {detection.bbox}")
                            else:
                                logger.debug(f"No landmarks for detection at {detection.bbox}")
                        else:
                            # Legacy: Use single model (with caching)
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

                        # Cache the match result
                        bbox_key = (x1, y1, x2, y2)
                        cached_match_results[bbox_key] = (match_text, match_color)

                # Draw detections using cached match results
                for idx, detection in enumerate(detections):
                    x1, y1, x2, y2 = detection.bbox
                    # Scale bbox coordinates from proc_frame to original frame
                    if scale_factor != 1.0:
                        x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)

                    # Get cached match results (default if not found)
                    bbox_key = (x1, y1, x2, y2)
                    match_text, match_color = cached_match_results.get(bbox_key, ("UNKNOWN", self.COLOR_UNKNOWN))

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

                # Show model info
                if self.use_composite:
                    model_info = f"Detector: {self.detector.info.name} | Recognizer: {self.recognizer.info.name}"
                else:
                    model_info = f"Model: {self.model.info.name}"

                info_text = f"FPS: {current_fps:.1f} | {model_info} | Faces: {len(detections)}"
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
