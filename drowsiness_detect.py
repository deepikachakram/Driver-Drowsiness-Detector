"""
Advanced Driver Drowsiness Detector (TESLA STYLE HUD)
----------------------------------------------------
Features:
- MediaPipe FaceMesh (no dlib required)
- EAR + MAR fatigue detection
- Blink counter
- Yawn detection + yawn counts
- Talking vs yawning distinction (short smaller openings = talking)
- Sleepiness score calculation
- Distraction detection (turning face / not looking forward)
- Intelligent Recommendation Based on Yawns:
    * 3rd yawn → Coffee shops nearby (Google Maps)
    * 4th yawn → Voice alert (parkaside.wav) + pull-over message
    * 5th+ yawns → Hotels/Lodges nearby (Google Maps)
- Continuous alarm when drowsy (alert.wav)
- CSV logging
- Clean HUD-style UI with Tesla-like theme:
    * Thin top bar with title + status + sleepiness bar
    * Small metrics overlay in top-left
    * Notification strip at bottom
- Manual: 't'=test alarm, 's'=stop alarm, 'q'=quit
"""

import cv2
import mediapipe as mp
import numpy as np
import winsound
import pygame
import os
import time
import csv
import webbrowser

# ================= THEME CONFIG (TESLA STYLE) =================

# Colors are in BGR format
PRIMARY_COLOR = (255, 255, 255)      # white text
ACCENT_COLOR = (255, 255, 0)         # cyan-ish (blue + green)
INFO_COLOR = (200, 200, 200)         # soft gray
ALERT_COLOR = (0, 0, 255)            # red
NORMAL_STATUS_COLOR = ACCENT_COLOR   # cyan for normal
FONT = cv2.FONT_HERSHEY_DUPLEX       # clean modern font

TOP_BAR_ALPHA = 0.35
BOTTOM_BAR_ALPHA = 0.45

# ================= CORE CONFIG =================

EAR_THRESH = 0.23
EAR_CONSEC_FRAMES = 20

MAR_THRESH = 0.60          # big mouth open → possible yawn
MAR_TALK = 0.30            # medium mouth open → likely talking
YAWN_TIME_THRESH = 3.0     # seconds above MAR_THRESH to consider as a yawn and trigger alarm
TALK_FRAMES_RESET = 4      # if talking for some frames, reset yawn state

DISTRACTION_X_THRESHOLD = 0.35

ALERT_SOUND_PATH = os.path.join("audio", "alert.wav")
PULL_OVER_AUDIO = os.path.join("audio", "parkaside.wav")

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "drowsiness_log.csv")
LOG_INTERVAL_SEC = 0.5


# ================ SOUND (ALARM) ====================

def start_alarm():
    """Start looping alert sound (if available)."""
    if not os.path.exists(ALERT_SOUND_PATH):
        print("[ERROR] Alert sound not found:", ALERT_SOUND_PATH)
        return
    print("[DEBUG] Alarm ON")
    winsound.PlaySound(
        ALERT_SOUND_PATH,
        winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
    )


def stop_alarm():
    """Stop any playing alert sound."""
    print("[DEBUG] Alarm OFF")
    winsound.PlaySound(None, winsound.SND_PURGE)


# ================ MATH HELPERS =====================

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, idxs):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idxs]
    return (euclidean_dist(p2, p6) + euclidean_dist(p3, p5)) / (2.0 * euclidean_dist(p1, p4))


def mouth_aspect_ratio(landmarks, idxs):
    left, right, top, bottom = [landmarks[i] for i in idxs]
    return euclidean_dist(top, bottom) / (euclidean_dist(left, right) + 1e-6)


# ================ SLEEPINESS SCORE =================

def compute_sleepiness_score(closed_frames, yawn_duration, distracted):
    """Combine eye closure, yawning, and distraction into a 0–100 score."""
    eye_score = min(1.0, closed_frames / EAR_CONSEC_FRAMES) * 50.0
    yawn_score = min(1.0, yawn_duration / YAWN_TIME_THRESH) * 30.0
    distraction_score = 20.0 if distracted else 0.0
    return min(100.0, max(0.0, eye_score + yawn_score + distraction_score))


# ================ LOGGING =========================

def init_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "EAR", "MAR",
                "blink_count", "yawn_count",
                "drowsy", "yawn_alert", "distracted",
                "sleepiness_score"
            ])


def log_state(timestamp, ear, mar, blinks, yawns, drowsy, yawn_alert, distracted, score):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            f"{ear:.4f}",
            f"{mar:.4f}",
            blinks,
            yawns,
            int(drowsy),
            int(yawn_alert),
            int(distracted),
            f"{score:.2f}",
        ])


# ================ MAIN ============================

def main():
    print("[INFO] Starting TESLA STYLE HUD Driver Drowsiness Detector")

    init_logger()

    # init pygame mixer for mp3 voice playback
    try:
        pygame.mixer.init()
        print("[INFO] Pygame mixer initialized.")
    except Exception as e:
        print("[WARNING] Could not init pygame mixer:", e)

    mp_face_mesh = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils
    draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not detected.")
        return

    # FaceMesh landmark indices
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH = [78, 308, 13, 14]

    closed_frames = 0
    yawn_start_time = None
    talk_frames = 0
    blink_count = 0
    yawn_count = 0

    last_ear = 0.0
    last_mar = 0.0
    blink_state_closed = False
    yawn_state_open = False

    alarm_on = False
    last_log_time = time.time()

    rest_message = ""
    last_suggested_yawn = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            distracted = False

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                landmarks = [(int(l.x * w), int(l.y * h)) for l in face.landmark]

                # EAR / MAR
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0
                last_ear = ear

                mar = mouth_aspect_ratio(landmarks, MOUTH)
                last_mar = mar

                # Distraction check (face center too far left/right)
                xs = [p[0] for p in landmarks]
                face_center_x_norm = (min(xs) + max(xs)) / 2 / w
                distracted = (
                    face_center_x_norm < (0.5 - DISTRACTION_X_THRESHOLD)
                    or face_center_x_norm > (0.5 + DISTRACTION_X_THRESHOLD)
                )

                # Eye closure / blink logic
                if ear < EAR_THRESH:
                    closed_frames += 1
                else:
                    if blink_state_closed:
                        blink_count += 1
                    blink_state_closed = False
                    closed_frames = 0
                blink_state_closed = ear < EAR_THRESH

                # ===== YAWN vs TALK detection =====
                if mar > MAR_THRESH:
                    # big opening → possible yawn
                    if yawn_start_time is None:
                        yawn_start_time = time.time()
                    yawn_state_open = True
                    talk_frames = 0
                elif mar > MAR_TALK:
                    # medium opening → likely talking
                    talk_frames += 1
                    # if talking persists, cancel any partial yawn
                    if talk_frames >= TALK_FRAMES_RESET:
                        yawn_start_time = None
                        yawn_state_open = False
                else:
                    # mouth closed - finalize any yawn
                    yawn_duration = time.time() - yawn_start_time if yawn_start_time else 0
                    if yawn_state_open and yawn_duration >= YAWN_TIME_THRESH:
                        yawn_count += 1
                        print(f"[INFO] Yawn #{yawn_count}")
                    yawn_start_time = None
                    yawn_state_open = False
                    talk_frames = 0

                # Draw facemesh
                mp_draw.draw_landmarks(
                    frame, face,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    None, draw_spec
                )

            else:
                # no face
                closed_frames = 0
                yawn_start_time = None
                talk_frames = 0
                distracted = True

            # drowsiness & score
            drowsy = closed_frames >= EAR_CONSEC_FRAMES
            current_yawn_duration = time.time() - yawn_start_time if yawn_start_time else 0
            yawn_alert = current_yawn_duration >= YAWN_TIME_THRESH
            sleepiness_score = compute_sleepiness_score(
                closed_frames, current_yawn_duration, distracted
            )

            # alarm logic
            if drowsy or yawn_alert or distracted:
                if not alarm_on:
                    alarm_on = True
                    start_alarm()
                status_text = "ALERT"
            else:
                status_text = "NORMAL"
                if alarm_on:
                    alarm_on = False
                    stop_alarm()

            # ===== SMART YAWN-BASED RECOMMENDATION ENGINE =====
            if yawn_count >= 3 and yawn_count > last_suggested_yawn:
                last_suggested_yawn = yawn_count

                if yawn_count == 3:
                    # 3rd yawn → coffee shops
                    rest_message = "You yawned 3 times. Suggesting nearby coffee shops..."
                    print("[INFO] Coffee shop suggestion triggered.")
                    try:
                        webbrowser.open(
                            "https://www.google.com/maps/search/coffee+shop+near+me"
                        )
                    except Exception:
                        pass

                elif yawn_count == 4:
                    # 4th yawn → voice alert + pull over
                    rest_message = "Severe fatigue! STOP the car safely and rest for a few minutes."
                    print("[WARNING] Pull over message displayed.")
                    try:
                        if os.path.exists(PULL_OVER_AUDIO):
                            pygame.mixer.music.load(PULL_OVER_AUDIO)
                            pygame.mixer.music.play()
                        else:
                            print("[ERROR] parkaside.wav not found in audio folder.")
                    except Exception as e:
                        print("[ERROR] Could not play pull-over audio:", e)

                else:
                    # 5th+ yawns → hotels
                    rest_message = "Extreme fatigue! Suggesting nearby hotels/lodges..."
                    print("[INFO] Hotel suggestion triggered.")
                    try:
                        webbrowser.open(
                            "https://www.google.com/maps/search/hotels+near+me"
                        )
                    except Exception:
                        pass

            # ===== LOGGING =====
            if time.time() - last_log_time > LOG_INTERVAL_SEC:
                ts = time.strftime("%H:%M:%S")
                log_state(
                    ts,
                    last_ear,
                    last_mar,
                    blink_count,
                    yawn_count,
                    drowsy,
                    yawn_alert,
                    distracted,
                    sleepiness_score,
                )
                last_log_time = time.time()

            # ================= TESLA STYLE CLEAN HUD UI =================
            # Top translucent bar
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, TOP_BAR_ALPHA, frame, 1 - TOP_BAR_ALPHA, 0)

            # Title (left)
            cv2.putText(
                frame,
                "Driver Assist Drowsiness Monitor",
                (10, 30),
                FONT,
                0.7,
                PRIMARY_COLOR,
                2,
                cv2.LINE_AA,
            )

            # Status + mini sleepiness bar (right)
            status_color = NORMAL_STATUS_COLOR if status_text == "NORMAL" else ALERT_COLOR
            status_label = f"Status: {status_text}"
            cv2.putText(
                frame,
                status_label,
                (w - 260, 23),
                FONT,
                0.6,
                status_color,
                1,
                cv2.LINE_AA,
            )

            # Sleepiness mini-bar just below status
            mini_bar_x = w - 260
            mini_bar_y = 32
            mini_bar_w = 240
            mini_bar_h = 8
            fill_width = int(mini_bar_w * sleepiness_score / 100)

            if sleepiness_score <= 33:
                mini_color = ACCENT_COLOR
            elif sleepiness_score <= 66:
                mini_color = (0, 215, 255)  # more yellowish cyan
            else:
                mini_color = ALERT_COLOR

            cv2.rectangle(frame, (mini_bar_x, mini_bar_y),
                          (mini_bar_x + mini_bar_w, mini_bar_y + mini_bar_h),
                          INFO_COLOR, 1)
            cv2.rectangle(frame, (mini_bar_x, mini_bar_y),
                          (mini_bar_x + fill_width, mini_bar_y + mini_bar_h),
                          mini_color, -1)

            # Small metrics cluster under top-left
            metrics_x = 10
            metrics_y = 70
            line_h = 22
            cv2.putText(frame, f"EAR: {last_ear:.2f}",
                        (metrics_x, metrics_y), FONT, 0.55, PRIMARY_COLOR, 1, cv2.LINE_AA)
            cv2.putText(frame, f"MAR: {last_mar:.2f}",
                        (metrics_x, metrics_y + line_h), FONT, 0.55, PRIMARY_COLOR, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Blinks: {blink_count}",
                        (metrics_x, metrics_y + 2 * line_h), FONT, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Yawns: {yawn_count}",
                        (metrics_x, metrics_y + 3 * line_h), FONT, 0.55, ACCENT_COLOR, 1, cv2.LINE_AA)

            distraction_txt = "Yes" if distracted else "No"
            cv2.putText(frame, f"Distracted: {distraction_txt}",
                        (metrics_x, metrics_y + 4 * line_h), FONT, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

            # Notification bar at bottom
            notif_overlay = frame.copy()
            bar_h = 35
            cv2.rectangle(notif_overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(notif_overlay, BOTTOM_BAR_ALPHA, frame, 1 - BOTTOM_BAR_ALPHA, 0)

            notif_text = rest_message if rest_message else "System monitoring driver state..."
            cv2.putText(frame, notif_text,
                        (10, h - 10), FONT, 0.55, ACCENT_COLOR, 1, cv2.LINE_AA)

            # Help text (small, bottom-right)
            help_msg = "Keys: 't' test | 's' stop | 'q' quit"
            text_size, _ = cv2.getTextSize(help_msg, FONT, 0.45, 1)
            cv2.putText(frame, help_msg,
                        (w - text_size[0] - 10, h - 10),
                        FONT, 0.45, INFO_COLOR, 1, cv2.LINE_AA)

            cv2.imshow("Driver Drowsiness System (Tesla HUD)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("t"):
                start_alarm()
            if key == ord("s"):
                stop_alarm()
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()
    print("[INFO] System Closed.")


if __name__ == "__main__":
    main()
