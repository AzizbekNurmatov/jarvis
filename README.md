# AURA — Spatial Tracking HUD

A webcam desktop app for **head pose**, **blink / wink detection**, **basic gesture recognition**, and optional **hand landmark tracking**. Built with MediaPipe, OpenCV, and a Tkinter HUD.

## Features

- Live mirrored camera preview (fills the main panel)
- Head rotation (yaw / pitch / roll) and relative position (X / Y / Z)
- Neutral calibration with wrap-aware angle smoothing
- Eye blink / wink detection with left / right openness meters
- One-shot gestures: winks, blink, double blink, nod, shake, turn left / right
- Optional MediaPipe Hands overlay (21 landmarks + skeleton, up to two hands)
- Compact dark dashboard with FPS and live gesture status in the header

## Requirements

- Windows 10 or 11 (also works on supported macOS / Linux)
- **Python 3.11, 64-bit** recommended
- A working webcam

## Easiest Windows setup

1. Download or clone this project into a folder.
2. Install Python 3.11 from [python.org](https://www.python.org/downloads/) if needed (ensure the `py` launcher is available).
3. Double-click **`start_windows.bat`**. It creates a local `.venv`, installs packages, and launches the app.
4. Allow webcam permissions. Sit facing the camera, then click **Calibrate Neutral**.

Later launches reuse the virtual environment. The batch script re-checks dependencies on every run.

## Terminal setup

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

**macOS / Linux:**

```bash
python3.11 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python main.py
```

## Using the HUD

| Control | Action |
| --- | --- |
| **Calibrate Neutral** | Zero rotation / position from your current pose; resets blink baselines and gesture state |
| **Face Mesh** | Toggle face landmark dots on the preview |
| **Pose Axes** | Toggle orientation guides from the nose |
| **Hand Tracking** | Toggle MediaPipe Hands (off by default; only runs when enabled) |

Close the window to quit. The app releases the camera and any hand model on exit.

### Gestures

Gestures fire as **one-shot** events (not every frame):

| Gesture | How to trigger |
| --- | --- |
| Left / Right wink | Close one eye while the other stays open |
| Both blink | Blink both eyes together |
| Double blink | Two bilateral blinks in quick succession |
| Head nod | Pitch down (or up) then return to neutral |
| Head shake | Yaw both directions, then return |
| Head turn left / right | Hold a left or right yaw briefly |

Latest gesture also appears in the header as `GESTURE: …`.

## Configuration

Camera and smoothing defaults live at the top of `main.py`:

```python
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
SMOOTHING = 0.25
```

- Raise `CAMERA_INDEX` if the wrong webcam opens.
- Lower resolution (for example `640` × `480`) if FPS drops.
- Higher `SMOOTHING` reacts faster but can look jittery; lower values are smoother with more lag.

Blink and gesture thresholds are tunable on `BlinkDetector` and `GestureEngine` in `aura/core/`.

## What you should see

- **Left:** large live camera preview with optional overlays
- **Right (compact rail):** head metrics, gesture panel, eye tracking, movement axes
- Header: tracking status, latest gesture, FPS

**Notes**

- This is **head pose**, not eye gaze. Rotation estimates are approximate.
- Position values are relative image-space measures, not centimeters.
- Hand LEFT / RIGHT labels are anatomical and corrected for the mirrored preview.
- Hand tracking is optional — leave it off when you want maximum face-tracking FPS.

## Project layout

```
main.py                      # Tkinter HUD + camera loop
aura/core/blink.py           # EAR blink / wink detection
aura/core/gesture_engine.py  # One-shot head + blink gestures
aura/core/hand_tracking.py   # Optional MediaPipe Hands overlay
aura/core/pose.py            # Angle wrap / pose helpers
aura/core/visuals.py         # Older OpenCV drawing helpers (unused by HUD)
aura/core/model.py           # Optional Face Landmarker downloader (unused by HUD)
tests/                       # Blink + gesture unit tests
start_windows.bat            # Windows one-click setup + launch
requirements.txt
```

Live path: `main.py` → Face Mesh (+ optional Hands) → blink / pose / gestures → Tkinter HUD.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Troubleshooting

- **Camera not found / CAMERA ERROR:** Close Zoom, Teams, or other webcam apps. Check Windows Settings → Privacy & security → Camera. Try a different `CAMERA_INDEX` in `main.py`.
- **Black or frozen preview:** Unplug/replug the camera, lower `CAMERA_WIDTH` / `CAMERA_HEIGHT`, or try another index.
- **Low FPS (~15 or below):** Lower camera resolution, disable **Hand Tracking** when you do not need it, close other webcam apps, and use good lighting.
- **Hands not appearing:** Enable the **Hand Tracking** checkbox and keep hands clearly in frame with decent light.
- **Winks registering as both eyes:** Recalibrate, face the camera squarely, and blink a bit more deliberately with the other eye open.
- **Installation fails:** Confirm Python 3.11 is **64-bit**, then run the terminal setup commands to see pip errors.
- **Missing Pillow / PIL error:** Re-run `pip install -r requirements.txt`.

## Dependencies

- `mediapipe` — Face Mesh and Hands
- `opencv-python` — capture, pose math, overlay drawing
- `numpy` — numeric helpers
- `Pillow` — camera frames in the Tkinter UI
