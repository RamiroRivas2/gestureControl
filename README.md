# Hand Gesture Control

Control your PC with hand gestures using a Raspberry Pi camera, MediaPipe hand tracking, and a custom gesture classifier.

## Architecture

```
Pi Camera → MediaPipe Landmarks → Gesture Classifier → WebSocket → PC Agent → Action
                                                    ↓ (if PC is off)
                                              Wake-on-LAN via Ethernet
```

## Hardware

- Raspberry Pi 5 (or 4) with Pi Camera Module
- Gaming PC with Ethernet port
- 3ft Ethernet cable (direct Pi↔PC link for Wake-on-LAN)

## Gesture Map

| Gesture | Action | Notes |
|---------|--------|-------|
| Fist | Shutdown | Requires confirmation (repeat gesture) |
| Open Palm | Cancel | Cancels pending shutdown/restart |
| Thumbs Up | Open Discord | |
| Peace Sign | Open YouTube | Opens in default browser |
| Index Up | Toggle Discord Mute | Sends Ctrl+Shift+M hotkey |
| Rock Sign | Restart | Requires confirmation |
| Wave | Wake PC | Sends WoL packet over Ethernet |
| Pinch | Volume Down | |

## Setup

### Pi Side

```bash
cd pi
pip install -r requirements.txt
```

1. **Collect data**: `python collect_data.py`
   - SPACE to start/stop recording, N for next gesture, Q to quit
   - Hold each gesture for ~7-10 seconds (~200 samples at 30fps)

2. **Train model**: `python train_classifier.py`

3. **Run controller**: `python gesture_control.py`

### PC Side

```bash
cd pc
pip install -r requirements.txt
python agent.py
```

### Network Setup

**Wi-Fi**: Both Pi and PC on the same LAN for normal gesture commands.

**Direct Ethernet (for WoL)**:
- Pi: Set `eth0` static IP to `10.0.0.1/24`
- PC: Set Ethernet adapter to `10.0.0.2`, subnet `255.255.255.0`
- Enable Wake-on-LAN in BIOS and Windows network adapter settings

### Configuration

Edit `pi/config.json` to:
- Set your PC's IP address and MAC address
- Customize gesture-to-action mappings
- Adjust confidence threshold, smoothing, and cooldown

### Discord Mute Setup

Set Discord mute keybind: Settings → Keybinds → Toggle Mute → `Ctrl+Shift+M`
