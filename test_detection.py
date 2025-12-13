#!/usr/bin/env python3
"""Test face detection with composite architecture."""
import cv2
import logging
from core.config import AppConfig
from recognition import registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_detection():
    """Test face detection from camera."""
    # Load config
    config = AppConfig.from_yaml('config/recognition.yaml')

    # Load detector
    detector_name = config.get_detection_model()
    logger.info(f"Loading detector: {detector_name}")

    detector = registry.get(detector_name)
    logger.info(f"Model path: {detector._model_path}")
    logger.info(f"Model exists: {detector._model_path.exists()}")
    logger.info(f"Confidence threshold: {detector._confidence_threshold}")

    detector.load()
    logger.info(f"✓ Detector loaded: {detector.info.name}")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return

    logger.info("Press 'q' to quit, 'd' to toggle debug info")
    show_debug = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect faces
        detections = detector.detect_faces(frame)

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw confidence
            conf_text = f"{det.confidence:.2f}"
            cv2.putText(frame, conf_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw landmarks if available
            if det.landmarks is not None and show_debug:
                for i, (lx, ly) in enumerate(det.landmarks):
                    cv2.circle(frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)

        # Show info
        info = f"Faces: {len(detections)} | Threshold: {detector._confidence_threshold:.2f} | Frame: {frame_count}"
        cv2.putText(frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if show_debug:
            debug_info = f"Input size: {detector._det_size} | Model: {detector._model_path.name}"
            cv2.putText(frame, debug_info, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Detection Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_debug = not show_debug
            logger.info(f"Debug mode: {show_debug}")

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Test complete")

if __name__ == "__main__":
    test_detection()
