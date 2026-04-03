"""
Main gesture control loop.

Captures video from Pi Camera, classifies hand gestures using the trained model,
and sends commands to the PC agent over WebSocket. Falls back to local WoL
if the PC is unreachable and a wake gesture is detected.

Features:
  - Smoothing buffer (N-frame majority vote) to prevent false triggers
  - Cooldown timer between actions
  - Confirmation for dangerous actions (shutdown, restart)
  - Auto-reconnect to PC agent
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
import threading
import joblib
import websocket


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def extract_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks


class GestureController:
    def __init__(self):
        self.config = load_config()
        self.gesture_config = self.config["gestures"]
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.smoothing_frames = self.config.get("smoothing_frames", 5)
        self.cooldown_seconds = self.config.get("cooldown_seconds", 1.5)

        # Load trained model
        model_dir = os.path.join(os.path.dirname(__file__), "model")
        self.classifier = joblib.load(os.path.join(model_dir, "gesture_classifier.pkl"))
        with open(os.path.join(model_dir, "label_map.json"), "r") as f:
            raw_map = json.load(f)
            self.label_map = {int(k): v for k, v in raw_map.items()}

        # State
        self.gesture_buffer = []
        self.last_action_time = 0
        self.pending_confirm = None
        self.pending_confirm_time = 0
        self.ws = None
        self.ws_connected = False
        self.running = True

        # Start WebSocket connection in background
        self.ws_thread = threading.Thread(target=self._ws_connect_loop, daemon=True)
        self.ws_thread.start()

    def _ws_connect_loop(self):
        """Maintain WebSocket connection with auto-reconnect."""
        url = f"ws://{self.config['pc_ip']}:{self.config['pc_port']}"
        while self.running:
            try:
                print(f"Connecting to PC agent at {url}...")
                self.ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_ws_open,
                    on_close=self._on_ws_close,
                    on_error=self._on_ws_error,
                )
                self.ws.run_forever(ping_interval=10, ping_timeout=5)
            except Exception as e:
                print(f"WebSocket error: {e}")
            self.ws_connected = False
            if self.running:
                print("Reconnecting in 3 seconds...")
                time.sleep(3)

    def _on_ws_open(self, ws):
        self.ws_connected = True
        print("Connected to PC agent!")

    def _on_ws_close(self, ws, close_status, close_msg):
        self.ws_connected = False
        print("Disconnected from PC agent")

    def _on_ws_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def send_command(self, gesture_name):
        """Send a gesture command to the PC agent."""
        gesture_cfg = self.gesture_config[gesture_name]
        action = gesture_cfg["action"]

        # WoL is handled locally on the Pi
        if action == "wake_on_lan":
            self._handle_wol()
            return

        # Cancel pending confirmations
        if action == "cancel":
            if self.pending_confirm:
                print(f"Cancelled pending {self.pending_confirm}")
                self.pending_confirm = None
            return

        # Actions requiring confirmation
        if gesture_cfg.get("confirm", False):
            if self.pending_confirm == gesture_name:
                # Second time — confirmed
                elapsed = time.time() - self.pending_confirm_time
                if elapsed < 5.0:
                    print(f"CONFIRMED: {gesture_cfg['label']}")
                    self.pending_confirm = None
                    self._send_to_pc(gesture_cfg)
                else:
                    # Timed out, re-request
                    print(f"Confirmation timed out. Show '{gesture_name}' again to confirm {gesture_cfg['label']}")
                    self.pending_confirm = gesture_name
                    self.pending_confirm_time = time.time()
            else:
                print(f"Confirm {gesture_cfg['label']}? Show '{gesture_name}' again within 5s (or open palm to cancel)")
                self.pending_confirm = gesture_name
                self.pending_confirm_time = time.time()
            return

        # Regular actions
        self._send_to_pc(gesture_cfg)

    def _send_to_pc(self, gesture_cfg):
        """Send command payload to PC agent via WebSocket."""
        if not self.ws_connected:
            print(f"PC not connected — cannot execute '{gesture_cfg['label']}'")
            return

        payload = json.dumps(gesture_cfg)
        try:
            self.ws.send(payload)
            print(f"Sent: {gesture_cfg['label']}")
        except Exception as e:
            print(f"Failed to send: {e}")
            self.ws_connected = False

    def _handle_wol(self):
        """Send Wake-on-LAN packet directly from Pi."""
        if self.ws_connected:
            print("PC is already on!")
            return
        from wake_pc import wake_pc
        wake_pc(self.config["pc_mac"])

    def classify_gesture(self, landmarks):
        """Classify landmarks and return (gesture_name, confidence) or (None, 0)."""
        features = np.array(landmarks).reshape(1, -1)
        probabilities = self.classifier.predict_proba(features)[0]
        max_idx = np.argmax(probabilities)
        confidence = probabilities[max_idx]

        if confidence >= self.confidence_threshold:
            return self.label_map[max_idx], confidence
        return None, confidence

    def get_smoothed_gesture(self, gesture_name):
        """Apply majority-vote smoothing over the last N frames."""
        self.gesture_buffer.append(gesture_name)
        if len(self.gesture_buffer) > self.smoothing_frames:
            self.gesture_buffer.pop(0)

        if len(self.gesture_buffer) < self.smoothing_frames:
            return None

        # Majority vote
        from collections import Counter
        counts = Counter(self.gesture_buffer)
        most_common, count = counts.most_common(1)[0]

        if most_common is None:
            return None
        # Require majority
        if count >= (self.smoothing_frames // 2 + 1):
            return most_common
        return None

    def run(self):
        """Main loop: capture → detect → classify → act."""
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(self.config.get("camera_index", 0))
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return

        print("\n=== Gesture Control Active ===")
        print("Press Q to quit\n")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                gesture_name = None
                confidence = 0

                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                    landmarks = extract_landmarks(hand_lms)
                    gesture_name, confidence = self.classify_gesture(landmarks)

                smoothed = self.get_smoothed_gesture(gesture_name)

                # Execute action if cooldown has passed
                if smoothed and smoothed in self.gesture_config:
                    now = time.time()
                    if now - self.last_action_time >= self.cooldown_seconds:
                        self.send_command(smoothed)
                        self.last_action_time = now
                        self.gesture_buffer.clear()

                # UI overlay
                conn_status = "PC: Connected" if self.ws_connected else "PC: Disconnected"
                conn_color = (0, 255, 0) if self.ws_connected else (0, 0, 255)
                cv2.putText(frame, conn_status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, conn_color, 2)

                if gesture_name:
                    label = self.gesture_config.get(gesture_name, {}).get("label", gesture_name)
                    cv2.putText(frame, f"{label} ({confidence:.0%})", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if self.pending_confirm:
                    cv2.putText(frame, f"CONFIRM: {self.pending_confirm}?", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Gesture Control", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            if self.ws:
                self.ws.close()
            print("Gesture control stopped.")


if __name__ == "__main__":
    controller = GestureController()
    controller.run()
