"""
Gesture Data Collector
======================
Records MediaPipe hand landmarks for training the gesture classifier.

Usage:
    python collect_data.py

Controls:
    SPACE  - Start/stop recording for current gesture
    N      - Next gesture
    P      - Previous gesture
    Q      - Quit and save
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from picamera2 import Picamera2

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ── Load gesture names from config ───────────────────────────────────────────
with open("config.json", "r") as f:
    config = json.load(f)

gesture_names = list(config["gestures"].keys())

# ── Data storage ─────────────────────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def extract_landmarks(hand_landmarks) -> list[float]:
    """Extract normalized landmark coordinates as a flat list of 63 values."""
    wrist = hand_landmarks.landmark[0]
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    return coords


def main():
    # ── Camera setup using picamera2 ─────────────────────────────────────
    picam2 = Picamera2()
    cam_config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(cam_config)
    picam2.start()
    time.sleep(1)  # Let camera warm up

    gesture_idx = 0
    recording = False
    samples = {name: [] for name in gesture_names}

    # Load existing data if present
    data_file = os.path.join(DATA_DIR, "gesture_data.npz")
    if os.path.exists(data_file):
        existing = np.load(data_file, allow_pickle=True)
        for name in gesture_names:
            if name in existing:
                samples[name] = existing[name].tolist()
        print(f"Loaded existing data: { {k: len(v) for k, v in samples.items() if v} }")

    print("\n╔══════════════════════════════════════════════╗")
    print("║         Gesture Data Collector               ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  SPACE = Start/Stop recording                ║")
    print("║  N     = Next gesture                        ║")
    print("║  P     = Previous gesture                    ║")
    print("║  Q     = Quit and save                       ║")
    print("╚══════════════════════════════════════════════╝\n")

    while True:
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Camera is upside down
        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = hands.process(frame)  # MediaPipe expects RGB

        current_gesture = gesture_names[gesture_idx]
        sample_count = len(samples[current_gesture])

        # ── Draw hand landmarks ──────────────────────────────────────────
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr, hand_lm, mp_hands.HAND_CONNECTIONS
                )

                if recording:
                    landmarks = extract_landmarks(hand_lm)
                    samples[current_gesture].append(landmarks)

        # ── UI overlay ───────────────────────────────────────────────────
        color = (0, 0, 255) if recording else (0, 200, 0)
        status = "REC" if recording else "Ready"

        cv2.rectangle(frame_bgr, (0, 0), (640, 80), (0, 0, 0), -1)
        cv2.putText(
            frame_bgr, f"Gesture: {current_gesture} ({gesture_idx+1}/{len(gesture_names)})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv2.putText(
            frame_bgr, f"{status}  |  Samples: {sample_count}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )

        if sample_count < 200:
            cv2.putText(
                frame_bgr, f"Need ~{200 - sample_count} more",
                (430, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
            )
        else:
            cv2.putText(
                frame_bgr, "Good!",
                (530, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        if not results.multi_hand_landmarks:
            cv2.putText(
                frame_bgr, "No hand detected",
                (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
            )

        cv2.imshow("Gesture Collector", frame_bgr)

        # ── Key handling ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            recording = not recording
            if recording:
                print(f"  ▶ Recording '{current_gesture}'...")
            else:
                print(f"  ⏸ Paused. {len(samples[current_gesture])} samples total.")
        elif key == ord("n"):
            recording = False
            gesture_idx = (gesture_idx + 1) % len(gesture_names)
            print(f"\n→ Switched to: {gesture_names[gesture_idx]}")
        elif key == ord("p"):
            recording = False
            gesture_idx = (gesture_idx - 1) % len(gesture_names)
            print(f"\n→ Switched to: {gesture_names[gesture_idx]}")

    # ── Save data ────────────────────────────────────────────────────────
    picam2.stop()
    cv2.destroyAllWindows()

    save_data = {}
    total = 0
    print("\n── Data Summary ────────────────────────────────")
    for name in gesture_names:
        count = len(samples[name])
        total += count
        bar = "█" * min(count // 10, 30)
        print(f"  {name:15s}: {count:4d} samples  {bar}")
        if samples[name]:
            save_data[name] = np.array(samples[name])

    np.savez(data_file, **save_data)
    print(f"\n✓ Saved {total} total samples to {data_file}")


if __name__ == "__main__":
    main()