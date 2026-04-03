"""
PC-side action handlers.

Executes system commands based on gesture payloads received from the Pi.
Supports: open_app, open_url, hotkey, shutdown, restart, lock, volume_up, volume_down.
"""

import subprocess
import platform
import pyautogui
import time

# Prevent pyautogui from throwing errors on edge of screen
pyautogui.FAILSAFE = False

SYSTEM = platform.system()


def execute_action(payload):
    """Route an action payload to the correct handler."""
    action = payload.get("action")
    label = payload.get("label", action)

    handlers = {
        "open_app": _open_app,
        "open_url": _open_url,
        "hotkey": _hotkey,
        "shutdown": _shutdown,
        "restart": _restart,
        "lock": _lock,
        "volume_up": _volume_up,
        "volume_down": _volume_down,
    }

    handler = handlers.get(action)
    if handler:
        print(f"Executing: {label}")
        handler(payload)
    else:
        print(f"Unknown action: {action}")


def _open_app(payload):
    target = payload.get("target", "")
    if SYSTEM == "Windows":
        # Common app shortcuts
        app_paths = {
            "discord": "Discord",
            "spotify": "Spotify",
            "chrome": "chrome",
            "firefox": "firefox",
            "steam": "steam",
        }
        app = app_paths.get(target.lower(), target)
        try:
            subprocess.Popen(["start", app], shell=True)
        except Exception as e:
            print(f"Failed to open {target}: {e}")
    elif SYSTEM == "Linux":
        subprocess.Popen([target])
    elif SYSTEM == "Darwin":
        subprocess.Popen(["open", "-a", target])


def _open_url(payload):
    import webbrowser
    url = payload.get("target", "")
    if url:
        webbrowser.open(url)


def _hotkey(payload):
    keys = payload.get("keys", [])
    if keys:
        pyautogui.hotkey(*keys)
        print(f"  Pressed: {' + '.join(keys)}")


def _shutdown(payload):
    print("  SHUTTING DOWN in 5 seconds...")
    if SYSTEM == "Windows":
        subprocess.run(["shutdown", "/s", "/t", "5"])
    elif SYSTEM == "Linux":
        subprocess.run(["shutdown", "-h", "+0"])
    elif SYSTEM == "Darwin":
        subprocess.run(["sudo", "shutdown", "-h", "now"])


def _restart(payload):
    print("  RESTARTING in 5 seconds...")
    if SYSTEM == "Windows":
        subprocess.run(["shutdown", "/r", "/t", "5"])
    elif SYSTEM == "Linux":
        subprocess.run(["shutdown", "-r", "+0"])
    elif SYSTEM == "Darwin":
        subprocess.run(["sudo", "shutdown", "-r", "now"])


def _lock(payload):
    if SYSTEM == "Windows":
        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
    elif SYSTEM == "Linux":
        subprocess.run(["loginctl", "lock-session"])
    elif SYSTEM == "Darwin":
        subprocess.run(["pmset", "displaysleepnow"])


def _volume_up(payload):
    if SYSTEM == "Windows":
        for _ in range(5):
            pyautogui.press("volumeup")
    elif SYSTEM == "Linux":
        subprocess.run(["amixer", "set", "Master", "10%+"])
    elif SYSTEM == "Darwin":
        subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 10)"])


def _volume_down(payload):
    if SYSTEM == "Windows":
        for _ in range(5):
            pyautogui.press("volumedown")
    elif SYSTEM == "Linux":
        subprocess.run(["amixer", "set", "Master", "10%-"])
    elif SYSTEM == "Darwin":
        subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 10)"])
