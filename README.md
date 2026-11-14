# AURA — Spatial Tracking HUD

A webcam desktop app for **head pose**, **blink / wink detection**, **hand gesture recognition**, an **Interaction Lab**, and optional **Windows desktop control**. Built with MediaPipe, OpenCV, and a Tkinter HUD.

## Features

- Live mirrored camera preview (fills the main panel)
- Head rotation (yaw / pitch / roll) and relative position (X / Y / Z)
- Neutral calibration with wrap-aware angle smoothing
- Eye blink / wink detection with left / right openness meters
- Toggleable eye-socket bounds on the camera feed
- One-shot head / eye gestures: winks, blink, double blink, nod, shake, turn left / right
- Optional MediaPipe Hands overlay (21 landmarks + skeleton, up to two hands)
- Rule-based hand gestures: open palm, fist, pointing, peace sign, pinch
- Interaction Lab: index-finger cursor, pinch-to-click, pinch-and-drag
- Desktop Control (off by default): move / click / drag the Windows cursor with your hand
- Compact dark dashboard with FPS and live gesture status in the header

## Requirements

- Windows 10 or 11 (also works on supported macOS / Linux; **Desktop Control is Windows-only**)
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
| **Calibrate Neutral** | Zero rotation / position from your current pose; resets blink baselines and head-gesture state |
| **Face Mesh** | Toggle face landmark dots on the preview |
| **Pose Axes** | Toggle orientation guides from the nose |
| **Eye Tracking** | Toggle eyelid / eye-socket bounds on the preview (blink detection still runs either way) |
| **Hand Tracking** | Toggle the hand skeleton overlay |
| **Gesture Recognition** | Toggle the hand-gesture HUD panel and camera labels |
| **Interaction Lab** | Show the pinch-driven lab canvas under the camera |
| **Desktop Control** | Move the Windows cursor with your index finger (off by default) |

Close the window to quit. The app releases the camera, any hand model, and a held mouse button on exit.

Hand detection runs once and is shared. It starts when **Hand Tracking**, **Gesture Recognition**, **Interaction Lab**, or **Desktop Control** is on, and stops when all of those are off.

### Head / eye gestures

These fire as **one-shot** events (not every frame):

| Gesture | How to trigger |
| --- | --- |
| Left / Right wink | Close one eye while the other stays open |
| Both blink | Blink both eyes together |
| Double blink | Two bilateral blinks in quick succession |
| Head nod | Pitch down (or up) then return to neutral |
| Head shake | Yaw both directions, then return |
| Head turn left / right | Hold a left or right yaw briefly |

Latest head / eye gesture also appears in the header as `GESTURE: …`.

### Hand gestures

Enable **Gesture Recognition**. Poses are classified from the existing 21 hand landmarks (no second detector, no extra ML model). Recognition is stabilized for a couple of frames so the HUD does not flicker.

| Gesture | Pose |
| --- | --- |
| Open palm | Four fingers extended |
| Fist | Fingers curled inward |
| Pointing | Index extended, other fingers curled |
| Peace sign | Index and middle extended, ring and pinky curled |
| Pinch | Thumb near the index tip or last index joint |

Unmatched poses show `UNKNOWN`. No hand shows `NO HAND`. Left and right hands are tracked separately. Pinch is intentionally a bit loose so a natural grip still counts; a closed fist is not treated as a pinch.

### Interaction Lab

Click **Interaction Lab** (the button turns green). An index-finger cursor appears on the lab canvas:

- **NODE** — pinch and drag, then release
- **CLICK** — one count per pinch (holding does not repeat)
- **LEVEL** — pinch the slider and drag sideways

You do not need the Hand Tracking or Gesture Recognition checkboxes for the lab. Turning **Hand Tracking** on is useful so you can see the skeleton while aiming.

### Desktop Control

**Off by default.** This moves your real Windows cursor.

1. Park the pointer somewhere harmless (empty desktop).
2. Click **Desktop Control** (the button turns orange). Keep a finger near **Esc**.
3. Index finger aims the cursor. Short pinch = click. Hold pinch (~0.24s) = drag.
4. Press **Esc** to stop immediately. The held mouse button is always released.

Also disengages if the hand leaves the frame for about a second, if you toggle the mode off, if an error occurs, or when AURA exits.

While it is on, AURA drives the cursor every frame and will fight a physical mouse. Use **Esc** rather than trying to pinch-click the orange button. A slim strip shows FPS, interaction latency, status (`AIMING` / `PINCH` / `DRAG` / `LOST`), and a sensitivity slider.

Desktop Control is Windows-only. macOS / Linux still run tracking, gestures, and the Interaction Lab.

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

Blink, head-gesture, hand-gesture, and desktop click/drag thresholds are tunable on the classes in `aura/core/`.

## What you should see

- **Left:** large live camera preview with optional overlays; Interaction Lab and Desktop Control strips appear under the camera when enabled
- **Right (compact rail):** head metrics, head-gesture panel, eye tracking, movement axes, hand-gesture panel
- Header: tracking status, latest head/eye gesture, FPS

**Notes**

- This is **head pose**, not eye gaze. Rotation estimates are approximate.
- Position values are relative image-space measures, not centimeters.
- Hand LEFT / RIGHT labels are anatomical and corrected for the mirrored preview.
- Hand tracking is optional — leave Hands-related modes off when you want maximum face-tracking FPS.
- Desktop Control does not send commands to TARS yet; it only consumes the same interaction events the HUD uses.

## Project layout

```
main.py                         # Tkinter HUD + camera loop
aura/core/blink.py              # EAR blink / wink detection + eye contours
aura/core/desktop.py            # Windows mouse destination (click / drag / safety)
aura/core/gesture_engine.py     # One-shot head + blink gestures
aura/core/hand_gestures.py      # Rule-based hand pose classification + pinch
aura/core/hand_tracking.py      # Optional MediaPipe Hands overlay
aura/core/interaction.py        # Cursor + pinch events for HUD / OS / TARS
aura/core/pose.py               # Angle wrap / pose helpers
aura/core/visuals.py            # Older OpenCV drawing helpers (unused by HUD)
aura/core/model.py              # Optional Face Landmarker downloader (unused by HUD)
aura/hud/lab.py                 # Interaction Lab canvas
aura/hud/desktop_panel.py       # Desktop Control status / sensitivity strip
tests/                          # Blink, head gesture, hand gesture, lab, desktop tests
start_windows.bat               # Windows one-click setup + launch
requirements.txt
```

Live path: `main.py` → Face Mesh (+ shared Hands when needed) → blink / pose / gestures / interaction events → Tkinter HUD (and optional Windows mouse).

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover blink classification, head gestures, hand poses, pinch hysteresis, Interaction Lab events, and Desktop Control click/drag/safety with a fake mouse (they do not move the real cursor).

## Troubleshooting

- **Camera not found / CAMERA ERROR:** Close Zoom, Teams, or other webcam apps. Check Windows Settings → Privacy & security → Camera. Try a different `CAMERA_INDEX` in `main.py`.
- **Black or frozen preview:** Unplug/replug the camera, lower `CAMERA_WIDTH` / `CAMERA_HEIGHT`, or try another index.
- **Low FPS (~15 or below):** Lower camera resolution, disable Hands-related modes when you do not need them, close other webcam apps, and use good lighting.
- **Hands not appearing:** Enable **Hand Tracking**, **Gesture Recognition**, **Interaction Lab**, or **Desktop Control**, and keep hands clearly in frame with decent light.
- **Pinch not registering:** Bring thumb and index close (a pad pinch counts). Face the palm somewhat toward the camera. A tight fist is ignored on purpose.
- **Desktop cursor fighting the mouse:** Press **Esc**, or wait for the hand to leave the frame. Desktop Control must be enabled explicitly each session.
- **Winks registering as both eyes:** Recalibrate, face the camera squarely, and blink a bit more deliberately with the other eye open.
- **Installation fails:** Confirm Python 3.11 is **64-bit**, then run the terminal setup commands to see pip errors.
- **Missing Pillow / PIL error:** Re-run `pip install -r requirements.txt`.

## Dependencies

- `mediapipe` — Face Mesh and Hands
- `opencv-python` — capture, pose math, overlay drawing
- `numpy` — numeric helpers
- `Pillow` — camera frames in the Tkinter UI

Desktop Control uses the Windows API through `ctypes` (no extra pip package).
