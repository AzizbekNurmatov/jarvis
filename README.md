# AURA — Spatial Tracking HUD

A webcam desktop app for **head pose**, **blink / wink detection**, **hand gesture recognition**, **head-anchored widgets**, an **Interaction Lab**, and optional **Windows desktop control**. Built with MediaPipe, OpenCV, and a Tkinter HUD.

## Features

- Live mirrored camera preview (fills the main panel)
- Head rotation (yaw / pitch / roll) and relative position (X / Y / Z)
- Neutral calibration with wrap-aware angle smoothing
- Eye blink / wink detection with left / right openness meters
- Toggleable eye-socket bounds on the camera feed
- One-shot head / eye gestures: winks, blink, double blink, nod, shake, turn left / right
- Optional MediaPipe Hands overlay (21 landmarks + skeleton, up to two hands)
- Rule-based hand gestures: open palm, fist, pointing, peace sign, pinch
- Interaction Lab: index-finger cursor, fist-to-click, fist-and-drag
- Desktop Control (off by default): move / click / drag the Windows cursor with your hand
- Pointer follow that stays steady when the hand is still and stays with the finger when you move, including a glide between camera frames on Windows
- Head-anchored widgets (off by default): Clock, Date, and Counter sit beside the face, follow the head, and are driven by nods plus a pointing pose
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
| **Widgets** | Show Clock, Date, and Counter beside the head (off by default) |
| **Interaction Lab** | Show the fist-driven lab canvas under the camera |
| **Desktop Control** | Move the Windows cursor with your index finger (off by default) |

Close the window to quit. The app releases the camera, any hand model, and a held mouse button on exit.

Hand detection runs once and is shared. It starts when **Hand Tracking**, **Gesture Recognition**, **Widgets**, **Interaction Lab**, or **Desktop Control** is on, and stops when all of those are off.

### Head / eye gestures

These fire as **one-shot** events (not every frame):

| Gesture | How to trigger |
| --- | --- |
| Left / Right wink | Close one eye while the other stays open |
| Both blink | Blink both eyes together |
| Double blink | Two bilateral blinks in quick succession |
| Head nod | Pitch down or up, then return to neutral. With **Widgets** on, a downward nod moves the highlight down and an upward nod moves it up |
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

Unmatched poses show `UNKNOWN`. No hand shows `NO HAND`. Left and right hands are tracked separately. Pinch is still classified in the HUD; **Interaction Lab** and **Desktop Control** grab with a **fist**. **Widgets** confirm with **pointing**, not a blink.

### Head widgets

Click **Widgets**. Three compact chips appear beside the face, on the side of the frame with more room, and glide with the head. They do not sit on the face. You do not need **Gesture Recognition** or **Hand Tracking** checked; Widgets starts the shared hand model on its own. **Hand Tracking** is still useful if you want to see the skeleton.

| Widget | While off | After you confirm it |
| --- | --- | --- |
| Clock | Title only | Current time (`HH:MM:SS`), chip turns green |
| Date | Title only | Today’s date (`Nov 17`), chip turns green |
| Counter | Title only | Increments by one and shows the number. Turning it off does not add another count |

**Nod** up or down only moves the highlight. Landing on a widget does not turn it on.

**Point** to confirm: index finger extended, the other fingers curled. The toggle happens once, when that pose is first confirmed. Holding the point does not repeat it. A second point within about 0.7 seconds is ignored. Point again after the cooldown to turn the same widget off. A blink, including a double blink, does not select a widget.

Calibrate neutral first so nods line up with your resting pose. Positive pitch is treated as a nod down.

### Interaction Lab

Click **Interaction Lab** (the button turns green). An index-finger cursor appears on the lab canvas. The same adaptive follow used by Desktop Control drives it: a resting fingertip stays steady, and a deliberate move stays with the finger instead of trailing it.

- **NODE** — make a fist and drag, then open the hand to release
- **CLICK** — one count per fist (holding does not repeat)
- **LEVEL** — fist the slider and drag sideways

You do not need the Hand Tracking or Gesture Recognition checkboxes for the lab. Turning **Hand Tracking** on is useful so you can see the skeleton while aiming.

### Desktop Control

**Off by default.** This moves your real Windows cursor.

1. Park the pointer somewhere harmless (empty desktop).
2. Click **Desktop Control** (the button turns orange). Keep a finger near **Esc**.
3. Index finger aims the cursor. Short fist = click. Hold fist (~0.24s) = drag.
4. Press **Esc** to stop immediately. The held mouse button is always released.

Also disengages if the hand leaves the frame for about a second, if you toggle the mode off, if an error occurs, or when AURA exits.

While it is on, AURA owns the cursor and will fight a physical mouse. Use **Esc** rather than trying to fist-click the orange button. A slim strip shows FPS, interaction latency, status (`AIMING` / `FIST` / `DRAG` / `LOST`), and a sensitivity slider.

**How the pointer follows**

- The fingertip is filtered adaptively. Holding still damps camera shake. A real move raises the response so the cursor stays on the finger instead of easing along behind it.
- The camera pump keeps the newest frame and drops older frames sitting in the webcam queue, so the aim point is where the hand is now.
- On Windows the cursor is updated between camera frames (a short glide plus a matching bit of the current velocity). Motion stays continuous, and a stop settles on the fingertip instead of coasting past it.

The lab canvas updates with each new hand sample. The Windows cursor is the one that glides between samples.

Desktop Control is Windows-only. macOS / Linux still run tracking, gestures, and the Interaction Lab.

## Configuration

Camera size and **head-pose** smoothing live at the top of `main.py`:

```python
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
SMOOTHING = 0.25
```

- Raise `CAMERA_INDEX` if the wrong webcam opens.
- Lower resolution (for example `640` × `480`) if FPS drops. The hand cursor can only follow as often as new hand samples arrive.
- Higher `SMOOTHING` makes yaw / pitch / roll and head position react faster but can look jittery. Lower values are smoother with more lag. This does **not** control the hand cursor.

Hand-cursor response is on `InteractionEngine` in `aura/core/interaction.py`:

```python
InteractionEngine(
    min_cutoff=1.05,  # lower = steadier when the hand is still
    beta=24.0,        # higher = a real move sticks to the finger sooner
)
```

`smooth=1` turns that filter off (the tests use this). Windows cursor glide timing is `_GLIDE_TAU_S` and `_GLIDE_LEAD_S` in `aura/core/desktop.py` (both default to 10 ms).

Widget follow and the confirm cooldown live on `HeadWidgets` in `aura/hud/head_widgets.py`:

```python
HeadWidgets(
    select_cooldown_s=0.70,  # ignore another point this soon after a toggle
    follow_s=0.09,           # how quickly the stack catches the head
)
```

Blink, head-gesture, hand-gesture, and desktop click/drag thresholds are tunable on the classes in `aura/core/`.

## What you should see

- **Left:** large live camera preview with optional overlays. With **Widgets** on, Clock, Date, and Counter sit beside the head. Interaction Lab and Desktop Control strips appear under the camera when enabled
- **Right (compact rail):** head metrics, head-gesture panel, eye tracking, movement axes, hand-gesture panel
- Header: tracking status, latest head/eye gesture, FPS

**Notes**

- This is **head pose**, not eye gaze. Rotation estimates are approximate.
- Position values are relative image-space measures, not centimeters.
- Hand LEFT / RIGHT labels are anatomical and corrected for the mirrored preview.
- Hand tracking is optional — leave Hands-related modes, including **Widgets**, off when you want maximum face-tracking FPS.
- Desktop Control does not send commands to TARS yet; it only consumes the same interaction events the HUD uses.

## Project layout

```
main.py                         # Tkinter HUD + camera loop
aura/core/blink.py              # EAR blink / wink detection + eye contours
aura/core/desktop.py            # Windows mouse destination, click / drag / safety, cursor glide
aura/core/gesture_engine.py     # One-shot head + blink gestures
aura/core/hand_gestures.py      # Rule-based hand pose classification + pinch
aura/core/hand_tracking.py      # Optional MediaPipe Hands overlay
aura/core/interaction.py        # Adaptive fingertip cursor + fist events for HUD / OS / TARS
aura/core/pose.py               # Angle wrap / pose helpers
aura/core/visuals.py            # Older OpenCV drawing helpers (unused by HUD)
aura/core/model.py              # Optional Face Landmarker downloader (unused by HUD)
aura/hud/head_widgets.py        # Head-anchored Clock / Date / Counter
aura/hud/lab.py                 # Interaction Lab canvas
aura/hud/desktop_panel.py       # Desktop Control status / sensitivity strip
tests/                          # Blink, head gesture, hand gesture, widgets, lab, desktop tests
start_windows.bat               # Windows one-click setup + launch
requirements.txt
```

Live path: `main.py` → Face Mesh (+ shared Hands when needed) → blink / pose / gestures / head widgets / interaction events → Tkinter HUD (and optional Windows mouse).

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover blink classification, head gestures (including nod up versus nod down), hand poses, pinch hysteresis, head widgets (nods only move focus, a confirmed point toggles, a held point does not), Interaction Lab events, fingertip follow (a fast move stays with the finger, resting jitter is rejected), and Desktop Control click/drag/safety plus cursor glide. Desktop tests use a fake mouse and do not move the real cursor.

## Troubleshooting

- **Camera not found / CAMERA ERROR:** Close Zoom, Teams, or other webcam apps. Check Windows Settings → Privacy & security → Camera. Try a different `CAMERA_INDEX` in `main.py`.
- **Black or frozen preview:** Unplug/replug the camera, lower `CAMERA_WIDTH` / `CAMERA_HEIGHT`, or try another index.
- **Low FPS (~15 or below):** Lower camera resolution, disable Hands-related modes when you do not need them, close other webcam apps, and use good lighting.
- **Hands not appearing:** Enable **Hand Tracking**, **Gesture Recognition**, **Widgets**, **Interaction Lab**, or **Desktop Control**, and keep hands clearly in frame with decent light.
- **Widget highlight will not move:** Calibrate neutral, then nod clearly down or up and return to center. A small head shift is not a nod.
- **Pointing does not toggle a widget:** Extend only the index finger and curl the others, hold it briefly so the pose can confirm, then drop the hand. Holding the point, or pointing again inside about 0.7 seconds, will not toggle a second time. Blinks do not select widgets.
- **Fist not grabbing in the lab / desktop:** Close all four fingers. Aim with a pointed index first, then fist; the cursor stays on the aim point instead of jumping to the knuckles.
- **Pinch not showing in the gesture panel:** Bring thumb and index close (a pad pinch still counts). A tight fist is classified as FIST, not PINCH.
- **Desktop cursor fighting the mouse:** Press **Esc**, or wait for the hand to leave the frame. Desktop Control must be enabled explicitly each session.
- **Cursor still feels late or stepped:** Restart AURA so the current pointer code is running. Keep the hand well lit and fully in frame. If FPS is very low, lower the camera resolution — glide fills the gaps, but it still depends on fresh hand samples.
- **Winks registering as both eyes:** Recalibrate, face the camera squarely, and blink a bit more deliberately with the other eye open.
- **Installation fails:** Confirm Python 3.11 is **64-bit**, then run the terminal setup commands to see pip errors.
- **Missing Pillow / PIL error:** Re-run `pip install -r requirements.txt`.

## Dependencies

- `mediapipe` — Face Mesh and Hands
- `opencv-python` — capture, pose math, overlay drawing
- `numpy` — numeric helpers
- `Pillow` — camera frames in the Tkinter UI

Desktop Control uses the Windows API through `ctypes` (no extra pip package).
