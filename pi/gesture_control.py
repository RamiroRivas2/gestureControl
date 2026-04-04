"""
Gesture Control — Main Loop
============================
Runs on the Raspberry Pi. Captures camera frames, classifies gestures
via MediaPipe landmarks + trained classifier, and sends commands to
the PC agent over WebSocket.

Usage:
    python gesture_control.py

Controls:
    Q - Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets
import json
import time
import joblib
import os
import threading
from collections import deque
from picamera2 import Picamera2

# ── Config ───────────────────────────────────────────────────────────────
with open("config.json", "r") as f:
    config = json.load(f)

PC_HOST = config["pc_host"]
PC_PORT = config["pc_port"]
CONFIDENCE_THRESHOLD = config.get("confidence_threshold", 0.75)
COOLDOWN = config.get("cooldown_seconds", 1.5)
SHOW_PREVIEW = config.get("show_preview", True)

# ── Load model ───────────────────────────────────────────────────────────
MODEL_PATH = "model/gesture_classifier.joblib"
ENCODER_PATH = "model/label_encoder.joblib"

if not os.path.exists(MODEL_PATH):
    print("✗ No trained model found. Run train_classifier.py first.")
    exit(1)

clf = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)
print(f"✓ Loaded model with gestures: {list(le.classes_)}")

# ── MediaPipe ────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ── State ────────────────────────────────────────────────────────────────
last_action_time = 0
gesture_buffer = deque(maxlen=5)  # Smooth predictions over 5 frames
ws_connection = None
ws_connected = False
pending_confirm = None  # Gesture awaiting confirmation


def extract_landmarks(hand_landmarks) -> list[float]:
    """Extract wrist-normalized landmarks as flat list."""
    wrist = hand_landmarks.landmark[0]
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    return coords


def classify_gesture(landmarks: list[float]) -> tuple[str, float]:
    """Classify a gesture from landmarks. Returns (name, confidence)."""
    features = np.array(landmarks).reshape(1, -1)
    proba = clf.predict_proba(features)[0]
    idx = np.argmax(proba)
    return le.classes_[idx], proba[idx]


def get_stable_gesture() -> tuple[str | None, float]:
    """Get the most common gesture in the buffer for smoothing."""
    if len(gesture_buffer) < 3:
        return None, 0.0

    counts = {}
    confidences = {}
    for name, conf in gesture_buffer:
        counts[name] = counts.get(name, 0) + 1
        confidences[name] = confidences.get(name, [])
        confidences[name].append(conf)

    best = max(counts, key=counts.get)
    if counts[best] >= 3:
        avg_conf = np.mean(confidences[best])
        return best, avg_conf

    return None, 0.0


# ── WebSocket connection (runs in background thread) ─────────────────────
command_queue = asyncio.Queue()


async def ws_sender():
    """Maintain WebSocket connection and send commands."""
    global ws_connection, ws_connected

    uri = f"ws://{PC_HOST}:{PC_PORT}"
    while True:
        try:
            async with websockets.connect(uri) as ws:
                ws_connected = True
                print(f"✓ Connected to PC agent at {uri}")

                while True:
                    command = await command_queue.get()
                    await ws.send(json.dumps(command))
                    response = await ws.recv()
                    resp_data = json.loads(response)
                    print(f"  PC response: {resp_data.get('status', 'unknown')}")

        except (ConnectionRefusedError, OSError):
            ws_connected = False
            print(f"  ⚠ Can't reach PC agent at {uri}. Retrying in 3s...")
            await asyncio.sleep(3)
        except websockets.exceptions.ConnectionClosed:
            ws_connected = False
            print("  ⚠ Connection lost. Reconnecting...")
            await asyncio.sleep(1)


def start_ws_thread():
    """Run the WebSocket sender in a background thread."""
    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ws_sender())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return loop


def send_command(gesture_name: str):
    """Queue a command to be sent to the PC."""
    gesture_config = config["gestures"].get(gesture_name, {})
    command = {
        "gesture": gesture_name,
        "action": gesture_config.get("action", "none"),
        "target": gesture_config.get("target", ""),
        "keys": gesture_config.get("keys", []),
        "timestamp": time.time(),
    }
    try:
        ws_loop.call_soon_threadsafe(command_queue.put_nowait, command)
    except Exception as e:
        print(f"  ⚠ Failed to queue command: {e}")


# ── Main loop ────────────────────────────────────────────────────────────
def main():
    global last_action_time, pending_confirm, ws_loop

    ws_loop = start_ws_thread()

    # ── Camera setup using picamera2 ─────────────────────────────────
    picam2 = Picamera2()
    cam_config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(cam_config)
    picam2.start()
    time.sleep(1)

    print("\n╔══════════════════════════════════════════════╗")
    print("║         Gesture Control — Running            ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  Show your gestures to the camera!           ║")
    print("║  Q = Quit                                    ║")
    print("╚══════════════════════════════════════════════╝\n")

    fps_counter = deque(maxlen=30)

    while True:
        t0 = time.time()
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Camera is upside down
        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = hands.process(frame)  # MediaPipe expects RGB

        gesture_name = None
        confidence = 0.0
        now = time.time()

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                if SHOW_PREVIEW:
                    mp_drawing.draw_landmarks(
                        frame_bgr, hand_lm, mp_hands.HAND_CONNECTIONS
                    )

                landmarks = extract_landmarks(hand_lm)
                raw_gesture, raw_conf = classify_gesture(landmarks)

                gesture_buffer.append((raw_gesture, raw_conf))
                gesture_name, confidence = get_stable_gesture()

        # ── Handle confirmed gestures ────────────────────────────────
        if gesture_name and confidence >= CONFIDENCE_THRESHOLD:
            gesture_cfg = config["gestures"].get(gesture_name, {})
            needs_confirm = gesture_cfg.get("confirm", False)

            if now - last_action_time > COOLDOWN:
                if needs_confirm and pending_confirm != gesture_name:
                    pending_confirm = gesture_name
                    print(f"  ⚠ {gesture_cfg['label']} — repeat gesture to confirm!")
                    last_action_time = now
                elif needs_confirm and pending_confirm == gesture_name:
                    print(f"  ✓ Confirmed: {gesture_cfg['label']}")
                    send_command(gesture_name)
                    pending_confirm = None
                    last_action_time = now
                elif not needs_confirm:
                    label = gesture_cfg.get("label", gesture_name)
                    print(f"  → {label} (confidence: {confidence:.0%})")
                    send_command(gesture_name)
                    last_action_time = now
        else:
            if gesture_name != pending_confirm:
                pending_confirm = None

        # ── Preview overlay ──────────────────────────────────────────
        if SHOW_PREVIEW:
            fps_counter.append(time.time() - t0)
            fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0

            cv2.rectangle(frame_bgr, (0, 0), (640, 60), (0, 0, 0), -1)

            conn_color = (0, 255, 0) if ws_connected else (0, 0, 255)
            conn_text = "● PC Connected" if ws_connected else "○ PC Disconnected"
            cv2.putText(frame_bgr, conn_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, conn_color, 2)

            if gesture_name:
                label = config["gestures"].get(gesture_name, {}).get("label", gesture_name)
                color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 200, 255)
                cv2.putText(frame_bgr, f"{label} ({confidence:.0%})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(frame_bgr, f"{fps:.0f} FPS", (560, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if pending_confirm:
                cv2.putText(frame_bgr, "Repeat to confirm!", (200, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Gesture Control", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()
    print("\n✓ Gesture control stopped.")


if __name__ == "__main__":
    main()