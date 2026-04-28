import os
import pygame
import cv2 as cv

# ---------- PATHS ----------
FACE_CASCADE_PATH = os.path.join("haarcascades", "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join("haarcascades", "haarcascade_eye.xml")
IMAGE_PATH = os.path.join("images", "test.jpeg")
ALERT_SOUND_PATH = os.path.join("audio", "alert.wav")

print("=== FILE EXISTENCE CHECK ===")

# Haar cascades
print(f"Face cascade path: {FACE_CASCADE_PATH}")
print("  Exists:", os.path.exists(FACE_CASCADE_PATH))

print(f"Eye cascade path:  {EYE_CASCADE_PATH}")
print("  Exists:", os.path.exists(EYE_CASCADE_PATH))

# Test image
print(f"Test image path:   {IMAGE_PATH}")
print("  Exists:", os.path.exists(IMAGE_PATH))

# Sound file
print(f"Alert sound path:  {ALERT_SOUND_PATH}")
print("  Exists:", os.path.exists(ALERT_SOUND_PATH))

# ---------- CASCADE LOADING TEST ----------
print("\n=== CASCADE LOADING CHECK ===")
if os.path.exists(FACE_CASCADE_PATH):
    face_cascade = cv.CascadeClassifier(FACE_CASCADE_PATH)
    print("Face cascade loaded:", not face_cascade.empty())
else:
    print("Face cascade NOT found, cannot load.")

if os.path.exists(EYE_CASCADE_PATH):
    eye_cascade = cv.CascadeClassifier(EYE_CASCADE_PATH)
    print("Eye cascade loaded:", not eye_cascade.empty())
else:
    print("Eye cascade NOT found, cannot load.")

# ---------- SOUND PLAYBACK TEST ----------
print("\n=== SOUND TEST ===")
if not os.path.exists(ALERT_SOUND_PATH):
    print("Sound file missing, skipping sound test.")
else:
    try:
        # Use a safe Windows audio driver
        os.environ["SDL_AUDIODRIVER"] = "directsound"

        pygame.mixer.init()
        pygame.mixer.music.load(ALERT_SOUND_PATH)
        pygame.mixer.music.play()

        print("Playing alert sound now...")
        input("Press Enter to stop sound and exit...")
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    except Exception as e:
        print("Error while playing sound:", e)
