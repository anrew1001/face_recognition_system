# test_scrfd_real.py
import cv2
from recognition import registry
import logging
logging.basicConfig(level=logging.DEBUG)


model = registry.get("scrfd_10g")
model.load()

# Укажи реальный путь к фото
img = cv2.imread("known/person1/person_1_1.png")

if img is None:
    print("ERROR: Cannot read image")
else:
    faces = model.detect_faces(img)
    print(f"Found {len(faces)} faces")
    for i, face in enumerate(faces):
        print(f"Face {i}: bbox={face.bbox}, conf={face.confidence:.2f}")