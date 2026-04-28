"""This script uses OpenCV's Haar cascades to detect face
and eyes in a single input image.
"""

import cv2 as cv
import numpy as np
import os

# ---------- PATHS ----------
FACE_CASCADE_PATH = os.path.join("haarcascades", "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join("haarcascades", "haarcascade_eye.xml")
IMAGE_PATH = os.path.join("images", "test.jpeg")

print("[INFO] Checking files...")

# Check cascades
if not os.path.exists(FACE_CASCADE_PATH):
    raise FileNotFoundError(f"Face cascade not found at: {FACE_CASCADE_PATH}")
if not os.path.exists(EYE_CASCADE_PATH):
    raise FileNotFoundError(f"Eye cascade not found at: {EYE_CASCADE_PATH}")

# Check image
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found at: {IMAGE_PATH}")

print("[INFO] All files found. Loading cascades...")

# Load cascades
face_cascade = cv.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv.CascadeClassifier(EYE_CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError("Failed to load face cascade.")
if eye_cascade.empty():
    raise RuntimeError("Failed to load eye cascade.")

# Read image
img = cv.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError("cv.imread failed to load the image.")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(f"[INFO] Faces detected: {len(faces)}")

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    print(f"[INFO] Eyes detected in this face: {len(eyes)}")

    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv.imshow("Face and Eye Detection - Single Image", img)
cv.waitKey(0)
cv.destroyAllWindows()
