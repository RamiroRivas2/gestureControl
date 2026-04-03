"""
Hand Gesture Data Collector

Records MediaPipe hand landmarks for each gesture defined in config.json.
Each frame captures 21 landmarks × 3 coordinates = 63 features.

Controls:
  SPACE  - Start/pause recording
  N      - Next gesture
  P      - Previous gesture
  Q/ESC  - Quit and save
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def extract_landmarks(hand_landmarks):
    """Extract 63 features (21 landmarks × 3 coords) from MediaPipe hand landmarks."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks


def main():
    config = load_config()
    gesture_names = list(config["gestures"].keys())
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(config.get("camera_index", 0))
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    current_gesture_idx = 0
    recording = False
    all_data = {}

    # Load existing data if available
    for gesture in gesture_names:
        filepath = os.path.join(data_dir, f"{gesture}.npy")
        if os.path.exists(filepath):
            all_data[gesture] = list(np.load(filepath))
            print(f"  Loaded {len(all_data[gesture])} existing samples for '{gesture}'")
        else:
            all_data[gesture] = []

    print("\n=== Hand Gesture Data Collector ===")
    print(f"Gestures to record: {gesture_names}")
    print("SPACE=record/pause | N=next | P=prev | Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_gesture = gesture_names[current_gesture_idx]
        sample_count = len(all_data[current_gesture])

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                if recording:
                    landmarks = extract_landmarks(hand_lms)
                    all_data[current_gesture].append(landmarks)

        # UI overlay
        status = "RECORDING" if recording else "PAUSED"
        color = (0, 0, 255) if recording else (0, 255, 0)

        cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples: {len(all_data[current_gesture])}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, status, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"[{current_gesture_idx + 1}/{len(gesture_names)}]", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Hand detection indicator
        hand_detected = results.multi_hand_landmarks is not None
        indicator_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.circle(frame, (frame.shape[1] - 30, 30), 15, indicator_color, -1)

        cv2.imshow("Gesture Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # Q or ESC
            break
        elif key == ord(" "):  # SPACE - toggle recording
            recording = not recording
            state = "Recording" if recording else "Paused"
            print(f"  [{current_gesture}] {state} - {len(all_data[current_gesture])} samples")
        elif key == ord("n"):  # N - next gesture
            recording = False
            current_gesture_idx = (current_gesture_idx + 1) % len(gesture_names)
            print(f"  Switched to: {gesture_names[current_gesture_idx]}")
        elif key == ord("p"):  # P - previous gesture
            recording = False
            current_gesture_idx = (current_gesture_idx - 1) % len(gesture_names)
            print(f"  Switched to: {gesture_names[current_gesture_idx]}")

    # Save all data
    print("\nSaving data...")
    for gesture, samples in all_data.items():
        if samples:
            filepath = os.path.join(data_dir, f"{gesture}.npy")
            np.save(filepath, np.array(samples))
            print(f"  Saved {len(samples)} samples for '{gesture}'")
        else:
            print(f"  No samples for '{gesture}' - skipped")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
