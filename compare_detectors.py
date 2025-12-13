#!/usr/bin/env python3
"""Compare detection between InsightFace and SCRFD."""
import cv2
import logging
from recognition import registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_detectors():
    """Compare InsightFace vs SCRFD detection."""
    # Load both models
    logger.info("Loading InsightFace...")
    insightface = registry.get('insightface')
    insightface.load()

    logger.info("Loading SCRFD...")
    scrfd = registry.get('scrfd_2.5g')
    scrfd.load()

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return

    logger.info("Press 'q' to quit, '1' for InsightFace, '2' for SCRFD")
    use_insightface = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        if use_insightface:
            detections = insightface.detect_faces(frame)
            model_name = "InsightFace"
            color = (255, 0, 0)  # Blue
        else:
            detections = scrfd.detect_faces(frame)
            model_name = "SCRFD_2.5G"
            color = (0, 255, 0)  # Green

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            conf_text = f"{det.confidence:.2f}"
            cv2.putText(frame, conf_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show info
        info = f"{model_name} | Faces: {len(detections)}"
        cv2.putText(frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        help_text = "1: InsightFace | 2: SCRFD | q: quit"
        cv2.putText(frame, help_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Detector Comparison", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            use_insightface = True
            logger.info("Switched to InsightFace")
        elif key == ord('2'):
            use_insightface = False
            logger.info("Switched to SCRFD")

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Comparison complete")

if __name__ == "__main__":
    compare_detectors()
