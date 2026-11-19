
import threading
import time
import tkinter as tk

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

from aura.core.blink import BlinkDetector
from aura.core.gesture_engine import GestureEngine
from aura.core.hand_gestures import (
    EVENT_STARTED,
    PINCH,
    POINTING,
    STATUS_STABLE,
    HandGestureRecognizer,
    draw_hand_gesture_overlay,
)
from aura.core.targets import TargetSelector, normalize_region
from aura.core.hand_tracking import HandTracker
from aura.core.desktop import DesktopController
from aura.core.interaction import InteractionEngine, draw_index_pointer
from aura.core.pose import wrap_angles
from aura.hud.desktop_panel import DesktopPanel
from aura.hud.head_widgets import HeadWidgets
from aura.hud.lab import InteractionLab
from aura.hud.target_overlay import draw_target_overlay
from aura.hud.target_panel import TargetPanel


# ----------------------------
# CONFIGURATION
# ----------------------------

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

SMOOTHING = 0.25
BG = "#0b1120"
PANEL = "#151e30"
PANEL_2 = "#1b2940"
TEXT = "#eef4ff"
MUTED = "#8da0ba"
ACCENT = "#38bdf8"
GREEN = "#34d399"
YELLOW = "#fbbf24"
PURPLE = "#a78bfa"
ORANGE = "#fb923c"

# MediaPipe Face Mesh landmarks for head pose.
POSE_POINTS = (1, 152, 33, 263, 61, 291)

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def smooth(previous, current, factor=SMOOTHING):
    if previous is None:
        return current
    return previous * (1.0 - factor) + current * factor


def smooth_angles(previous, current, factor=SMOOTHING):
    """EMA for Euler angles with wrap-aware deltas."""
    current = np.asarray(current, dtype=np.float64)
    if previous is None:
        return current.copy()
    previous = np.asarray(previous, dtype=np.float64)
    delta = wrap_angles(current - previous)
    return previous + factor * delta


class _LatestFrame:
    """Keep the newest camera image so the pointer is not a buffered frame behind.

    DirectShow queues several frames. Reading them in order makes the cursor
    chase where the hand was, not where it is. This pump drops that backlog.
    """

    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="aura-camera",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self._stop.wait(0.02):
                    break
                continue
            snapshot = frame.copy()
            with self._lock:
                self._frame = snapshot

    def read(self):
        with self._lock:
            frame = self._frame
        if frame is None:
            return False, None
        return True, frame

    def close(self):
        self._stop.set()

    def join(self, timeout=0.3):
        self._thread.join(timeout=timeout)


class AuraHUD:
    def __init__(self, root):
        self.root = root
        self.root.title("AURA | Head Tracking HUD")
        self.root.geometry("1480x940")
        self.root.minsize(1200, 820)
        self.root.configure(bg=BG)

        self.running = True

        self.cap = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW,
        )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self._frames = _LatestFrame(self.cap)
        self._seen_frame = None

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.blink_detector = BlinkDetector()
        self.gesture_engine = GestureEngine()
        self.hand_tracker = HandTracker(mirrored=True)
        self.hand_gesture_recognizer = HandGestureRecognizer()
        self.hand_gesture_snapshot = None
        self.interaction_engine = InteractionEngine()
        self.desktop = DesktopController(
            on_stop=self._on_desktop_stopped,
        )
        self._lab_on = False
        self._desktop_on = False
        self._targets_on = False
        self.targets = TargetSelector()
        self._target_drag = None

        self.show_mesh = tk.BooleanVar(value=False)
        self.show_axes = tk.BooleanVar(value=False)
        self.show_eyes = tk.BooleanVar(value=False)
        self.show_hands = tk.BooleanVar(value=False)
        self.recognize_gestures = tk.BooleanVar(value=False)
        self.show_widgets = tk.BooleanVar(value=False)
        self.head_widgets = HeadWidgets()

        self.neutral_rotation = None
        self.neutral_position = None

        self.smoothed_rotation = None
        self.smoothed_position = None

        self.rotation = np.zeros(3)
        self.position = np.zeros(3)

        self.face_visible = False
        self.last_time = time.perf_counter()
        self.fps = 0.0

        self.latest_blink = "None"
        self.blink_result = None
        self.latest_gesture = "None"
        self.gesture_flash_until = 0.0

        self.video_image = None

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind_all("<Escape>", self._desktop_emergency_stop)
        self.update_frame()

    # ----------------------------
    # UI HELPERS
    # ----------------------------

    def card(self, parent, title, subtitle=None):
        outer = tk.Frame(
            parent,
            bg=PANEL,
            padx=12,
            pady=8,
            highlightbackground="#26354d",
            highlightthickness=1,
        )

        tk.Label(
            outer,
            text=title,
            fg=TEXT,
            bg=PANEL,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(fill="x")

        if subtitle:
            tk.Label(
                outer,
                text=subtitle,
                fg=MUTED,
                bg=PANEL,
                font=("Segoe UI", 8),
                anchor="w",
            ).pack(fill="x", pady=(1, 4))

        return outer

    def metric(self, parent, name, color):
        item = tk.Frame(parent, bg=PANEL)
        item.pack(side="left", expand=True, fill="x")

        tk.Label(
            item,
            text=name,
            fg=MUTED,
            bg=PANEL,
            font=("Segoe UI", 8),
        ).pack()

        label = tk.Label(
            item,
            text="--",
            fg=color,
            bg=PANEL,
            font=("Consolas", 13, "bold"),
            width=6,
        )
        label.pack(pady=(2, 0))
        return label

    def build_ui(self):
        header = tk.Frame(self.root, bg=BG, padx=18, pady=10)
        header.pack(fill="x")

        tk.Label(
            header,
            text="AURA",
            fg=ACCENT,
            bg=BG,
            font=("Segoe UI", 20, "bold"),
        ).pack(side="left")

        tk.Label(
            header,
            text="  /  SPATIAL TRACKING",
            fg=MUTED,
            bg=BG,
            font=("Segoe UI", 10),
        ).pack(side="left", pady=(7, 0))

        self.fps_label = tk.Label(
            header,
            text="FPS: --",
            fg=MUTED,
            bg=BG,
            font=("Consolas", 10),
        )
        self.fps_label.pack(side="right", padx=(16, 0))

        self.status_label = tk.Label(
            header,
            text="● INITIALIZING",
            fg=YELLOW,
            bg=BG,
            font=("Segoe UI", 10, "bold"),
        )
        self.status_label.pack(side="right")

        self.header_gesture_label = tk.Label(
            header,
            text="GESTURE: —",
            fg=MUTED,
            bg=BG,
            font=("Segoe UI", 10, "bold"),
        )
        self.header_gesture_label.pack(side="right", padx=(0, 16))

        body = tk.Frame(self.root, bg=BG, padx=16, pady=6)
        body.pack(fill="both", expand=True)

        # Hard split: camera column grows; sidebar never steals width.
        # (pack_propagate alone was not reliable once metric cards got wide.)
        self._sidebar_width = 360
        body.grid_columnconfigure(0, weight=1, minsize=720)
        body.grid_columnconfigure(1, weight=0)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._left_column = left

        right = tk.Frame(body, bg=BG, width=self._sidebar_width)
        right.grid(row=0, column=1, sticky="ns")
        right.grid_propagate(False)
        right.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(1, weight=1)
        right.grid_rowconfigure(2, weight=1)
        self._sidebar = right
        body.bind("<Configure>", self._enforce_sidebar_width, add="+")

        # Camera preview.
        camera_card = self.card(
            left, "LIVE CAMERA",
            "Face / pose / eyes / optional hands / gestures / widgets / targets"
        )
        camera_card.pack(fill="both", expand=True)

        controls = tk.Frame(camera_card, bg=PANEL)
        controls.pack(side="bottom", fill="x", pady=(6, 0))

        toggles = tk.Frame(controls, bg=PANEL)
        toggles.pack(side="left", fill="x", expand=True)

        for text, variable, command in (
            ("Face Mesh", self.show_mesh, None),
            ("Pose Axes", self.show_axes, None),
            ("Eye Tracking", self.show_eyes, None),
            ("Hand Tracking", self.show_hands, self._sync_hand_tracker),
            (
                "Gesture Recognition",
                self.recognize_gestures,
                self._on_gesture_toggle,
            ),
            ("Widgets", self.show_widgets, self._on_widgets_toggle),
        ):
            tk.Checkbutton(
                toggles,
                text=text,
                variable=variable,
                command=command,
                bg=PANEL, fg=TEXT,
                selectcolor=PANEL_2,
                activebackground=PANEL,
                activeforeground=TEXT,
            ).pack(side="left", padx=4)

        self.desktop_button = tk.Button(
            controls,
            text="Desktop Control",
            command=self._toggle_desktop_control,
            bg=PANEL_2,
            fg=TEXT,
            relief="flat",
            padx=12,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        )
        self.desktop_button.pack(side="right", padx=4)

        self.targets_button = tk.Button(
            controls,
            text="Targets",
            command=self._toggle_targets,
            bg=PANEL_2,
            fg=TEXT,
            relief="flat",
            padx=12,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        )
        self.targets_button.pack(side="right", padx=4)

        self.lab_button = tk.Button(
            controls,
            text="Interaction Lab",
            command=self._toggle_interaction_lab,
            bg=PANEL_2,
            fg=TEXT,
            relief="flat",
            padx=12,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        )
        self.lab_button.pack(side="right", padx=4)

        tk.Button(
            controls,
            text="Calibrate Neutral",
            command=self.calibrate,
            bg=ACCENT,
            fg="#06101b",
            relief="flat",
            padx=12,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="right", padx=4)

        self.video_frame = tk.Frame(
            camera_card,
            bg="#050a14",
            highlightthickness=0,
        )
        self.video_frame.pack(fill="both", expand=True, pady=(4, 0))
        self.video_frame.pack_propagate(False)

        self.camera_label = tk.Label(
            self.video_frame,
            bg="#050a14",
            text="Initializing camera...",
            fg=MUTED,
        )
        # Centered contain-fit — never stretch or letterbox-crop the face away.
        self.camera_label.place(relx=0.5, rely=0.5, anchor="center")

        self.lab = InteractionLab(left)
        self.target_panel = TargetPanel(left, on_clear=self._clear_targets)
        self.desktop_panel = DesktopPanel(
            left, on_sensitivity=self._set_desktop_sensitivity
        )

        # --- Compact right dashboard (no scroll at normal size) ---
        metrics_card = self.card(
            right, "HEAD METRICS",
            "Rotation (°) and relative position"
        )
        metrics_card.grid(
            row=0, column=0, columnspan=2,
            sticky="ew", pady=(0, 6)
        )

        rotation_row = tk.Frame(metrics_card, bg=PANEL)
        rotation_row.pack(fill="x")
        self.yaw_label = self.metric(rotation_row, "YAW", ACCENT)
        self.pitch_label = self.metric(rotation_row, "PITCH", GREEN)
        self.roll_label = self.metric(rotation_row, "ROLL", PURPLE)

        position_row = tk.Frame(metrics_card, bg=PANEL)
        position_row.pack(fill="x", pady=(6, 0))
        self.x_label = self.metric(position_row, "X", ACCENT)
        self.y_label = self.metric(position_row, "Y", GREEN)
        self.z_label = self.metric(position_row, "Z", PURPLE)

        gesture_card = self.card(right, "GESTURE RECOGNITION")
        gesture_card.grid(
            row=1, column=0, columnspan=2,
            sticky="ew", pady=(0, 6)
        )

        gesture_panel = tk.Frame(gesture_card, bg=PANEL_2, padx=10, pady=6)
        gesture_panel.pack(fill="x")

        top_row = tk.Frame(gesture_panel, bg=PANEL_2)
        top_row.pack(fill="x")

        tk.Label(
            top_row,
            text="LATEST",
            bg=PANEL_2,
            fg=MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="left")

        self.gesture_status_label = tk.Label(
            top_row,
            text="Ready",
            bg=PANEL_2,
            fg=GREEN,
            font=("Segoe UI", 8, "bold"),
        )
        self.gesture_status_label.pack(side="right")

        self.gesture_label = tk.Label(
            gesture_panel,
            text="Waiting for gesture...",
            bg=PANEL_2,
            fg=TEXT,
            font=("Segoe UI", 13, "bold"),
        )
        self.gesture_label.pack(anchor="w", pady=(2, 0))

        self.gesture_flash = tk.Canvas(
            gesture_card,
            height=5,
            bg=PANEL,
            highlightthickness=0,
        )
        self.gesture_flash.pack(fill="x", pady=(6, 0))
        self._gesture_flash_bar = self.gesture_flash.create_rectangle(
            0, 0, 0, 5, fill=ACCENT, outline=""
        )

        blink_card = self.card(right, "EYE TRACKING")
        blink_card.grid(row=2, column=0, sticky="nsew", padx=(0, 4), pady=(0, 6))

        self.blink_label = tk.Label(
            blink_card,
            text="Waiting for blink...",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 12, "bold"),
            anchor="w",
        )
        self.blink_label.pack(fill="x")

        indicators = tk.Frame(blink_card, bg=PANEL)
        indicators.pack(fill="x", pady=(6, 0))

        self.eye_indicators = {}
        for key, title in [
            ("Left Eye", "L"),
            ("Right Eye", "R"),
            ("Both Eyes", "BOTH"),
        ]:
            indicator = tk.Label(
                indicators,
                text=title,
                bg=PANEL_2,
                fg=MUTED,
                padx=6,
                pady=4,
                font=("Segoe UI", 8, "bold"),
            )
            indicator.pack(side="left", fill="x", expand=True, padx=2)
            self.eye_indicators[key] = indicator

        openness = tk.Frame(blink_card, bg=PANEL)
        openness.pack(fill="x", pady=(8, 0))
        self.left_open_bar, self.left_open_label = (
            self._openness_meter(openness, "L")
        )
        self.right_open_bar, self.right_open_label = (
            self._openness_meter(openness, "R")
        )

        graph_card = self.card(right, "MOVEMENT AXES")
        graph_card.grid(row=2, column=1, sticky="nsew", padx=(4, 0), pady=(0, 6))

        self.axis_canvas = tk.Canvas(
            graph_card,
            bg="#0c1626",
            height=110,
            highlightthickness=0,
        )
        self.axis_canvas.pack(fill="both", expand=True, pady=(2, 0))
        self._axis_size = (0, 0)
        self.axis_canvas.bind("<Configure>", self._on_axis_configure)

        hand_g_card = self.card(right, "HAND GESTURE RECOGNITION")
        hand_g_card.grid(
            row=3, column=0, columnspan=2,
            sticky="ew",
        )

        hand_g_panel = tk.Frame(hand_g_card, bg=PANEL_2, padx=10, pady=6)
        hand_g_panel.pack(fill="x")

        tk.Label(
            hand_g_panel,
            text="CURRENT GESTURE",
            bg=PANEL_2,
            fg=MUTED,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill="x")

        self.hand_g_name = tk.Label(
            hand_g_panel,
            text="—",
            bg=PANEL_2,
            fg=MUTED,
            font=("Segoe UI", 13, "bold"),
            anchor="w",
        )
        self.hand_g_name.pack(fill="x", pady=(1, 4))

        meta = tk.Frame(hand_g_panel, bg=PANEL_2)
        meta.pack(fill="x")

        tk.Label(
            meta,
            text="HAND",
            bg=PANEL_2,
            fg=MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="left")

        self.hand_g_hand = tk.Label(
            meta,
            text="—",
            bg=PANEL_2,
            fg=TEXT,
            font=("Consolas", 10, "bold"),
        )
        self.hand_g_hand.pack(side="left", padx=(6, 16))

        tk.Label(
            meta,
            text="STATUS",
            bg=PANEL_2,
            fg=MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="left")

        self.hand_g_status = tk.Label(
            meta,
            text="NO HAND",
            bg=PANEL_2,
            fg=MUTED,
            font=("Segoe UI", 8, "bold"),
        )
        self.hand_g_status.pack(side="left", padx=(6, 0))

        stab_row = tk.Frame(hand_g_panel, bg=PANEL_2)
        stab_row.pack(fill="x", pady=(6, 0))

        tk.Label(
            stab_row,
            text="STABILITY",
            bg=PANEL_2,
            fg=MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="left")

        self.hand_g_stab_bar = tk.Canvas(
            stab_row,
            height=8,
            bg="#0c1626",
            highlightthickness=0,
        )
        self.hand_g_stab_bar.pack(
            side="left", fill="x", expand=True, padx=(8, 8)
        )
        self._hand_g_stab_rect = self.hand_g_stab_bar.create_rectangle(
            0, 0, 0, 8, fill=ACCENT, outline=""
        )

        self.hand_g_stab_label = tk.Label(
            stab_row,
            text="—",
            bg=PANEL_2,
            fg=MUTED,
            font=("Consolas", 8),
            width=5,
        )
        self.hand_g_stab_label.pack(side="right")

        tk.Label(
            hand_g_panel,
            text="RECENT",
            bg=PANEL_2,
            fg=MUTED,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(6, 0))

        self.hand_g_recent = tk.Label(
            hand_g_panel,
            text="—",
            bg=PANEL_2,
            fg=MUTED,
            font=("Consolas", 8),
            anchor="nw",
            justify="left",
        )
        self.hand_g_recent.pack(fill="x")

    def _openness_meter(self, parent, title):
        wrap = tk.Frame(parent, bg=PANEL)
        wrap.pack(side="left", expand=True, fill="x", padx=3)

        header = tk.Frame(wrap, bg=PANEL)
        header.pack(fill="x")

        tk.Label(
            header,
            text=title,
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="left")

        value = tk.Label(
            header,
            text="100%",
            bg=PANEL,
            fg=TEXT,
            font=("Consolas", 8),
        )
        value.pack(side="right")

        canvas = tk.Canvas(
            wrap,
            height=8,
            bg="#0c1626",
            highlightthickness=0,
        )
        canvas.pack(fill="x", pady=(4, 0))
        bar = canvas.create_rectangle(
            0, 0, 0, 8, fill=ACCENT, outline=""
        )
        canvas._bar_id = bar
        return canvas, value

    def _hands_needed(self):
        return bool(
            self.show_hands.get()
            or self.recognize_gestures.get()
            or self.show_widgets.get()
            or self._targets_on
            or self._lab_on
            or self._desktop_on
        )

    def _interaction_needed(self):
        return bool(self._lab_on or self._desktop_on)

    def _maybe_reset_engine(self):
        if not self._interaction_needed():
            self.interaction_engine.reset()

    def _toggle_interaction_lab(self):
        self._lab_on = not self._lab_on
        if self._lab_on:
            self.lab.frame.pack(side="bottom", fill="x", pady=(8, 0))
            self.lab.reset()
            self.lab_button.config(bg=GREEN, fg="#06101b")
        else:
            self.lab.frame.pack_forget()
            self.lab.reset()
            self.lab_button.config(bg=PANEL_2, fg=TEXT)
        self._maybe_reset_engine()
        self._sync_hand_tracker()

    def _set_desktop_sensitivity(self, value):
        self.desktop.sensitivity = float(value)

    def _show_desktop_ui(self):
        self._desktop_on = True
        self.desktop_panel.frame.pack(side="bottom", fill="x", pady=(8, 0))
        self.desktop_button.config(bg=ORANGE, fg="#06101b")
        self.desktop_panel.render(self.desktop.diagnostics())

    def _hide_desktop_ui(self):
        self._desktop_on = False
        self.desktop_panel.frame.pack_forget()
        self.desktop_button.config(bg=PANEL_2, fg=TEXT)

    def _toggle_desktop_control(self):
        if self._desktop_on:
            self.desktop.disable(reason="DISABLED")
            return
        self.desktop.enable()
        self._show_desktop_ui()
        self._sync_hand_tracker()

    def _on_desktop_stopped(self, reason):
        self._hide_desktop_ui()
        self._maybe_reset_engine()
        self._sync_hand_tracker()
        if reason and reason != "DISABLED":
            self.status_label.config(
                text=f"● {reason}",
                fg=ORANGE,
            )

    def _desktop_emergency_stop(self, _event=None):
        if self.desktop.enabled or self._desktop_on:
            self.desktop.emergency_stop()
        return "break"

    def _sync_hand_tracker(self):
        needed = self._hands_needed()
        if needed != self.hand_tracker.enabled:
            self.hand_tracker.set_enabled(needed)

    def _on_gesture_toggle(self):
        if not self.recognize_gestures.get():
            self.hand_gesture_recognizer.reset()
            self.hand_gesture_snapshot = None
            self.render_hand_gesture()
        self._sync_hand_tracker()

    def _on_widgets_toggle(self):
        if not self.show_widgets.get():
            self.head_widgets.reset_anchor()
        self._sync_hand_tracker()

    def _toggle_targets(self):
        self._targets_on = not self._targets_on
        if self._targets_on:
            self.target_panel.frame.pack(side="bottom", fill="x", pady=(8, 0))
            self.targets_button.config(bg=PURPLE, fg="#06101b")
            self.camera_label.bind("<ButtonPress-1>", self._target_press)
            self.camera_label.bind("<B1-Motion>", self._target_drag_move)
            self.camera_label.bind("<ButtonRelease-1>", self._target_release)
        else:
            self._target_drag = None
            self.target_panel.frame.pack_forget()
            self.targets_button.config(bg=PANEL_2, fg=TEXT)
            self.camera_label.unbind("<ButtonPress-1>")
            self.camera_label.unbind("<B1-Motion>")
            self.camera_label.unbind("<ButtonRelease-1>")
        self._sync_hand_tracker()

    def _clear_targets(self):
        self.targets.clear()
        self._target_drag = None
        self.target_panel.note("")

    def _preview_norm(self, event):
        image = self.video_image
        if image is None:
            return None
        width = max(image.width(), 1)
        height = max(image.height(), 1)
        x = 0.0 if event.x < 0 else 1.0 if event.x > width else event.x / width
        y = 0.0 if event.y < 0 else 1.0 if event.y > height else event.y / height
        return x, y

    def _target_press(self, event):
        if not self._targets_on:
            return
        origin = self._preview_norm(event)
        if origin is None:
            return
        self._target_drag = (origin[0], origin[1], origin[0], origin[1])

    def _target_drag_move(self, event):
        if self._target_drag is None:
            return
        point = self._preview_norm(event)
        if point is None:
            return
        x0, y0 = self._target_drag[0], self._target_drag[1]
        self._target_drag = (x0, y0, point[0], point[1])

    def _target_release(self, event):
        drag = self._target_drag
        self._target_drag = None
        if drag is None or not self._targets_on:
            return
        point = self._preview_norm(event)
        if point is None:
            return
        box = normalize_region(drag[0], drag[1], point[0], point[1])
        if box is None:
            self.target_panel.note("That box is too small. Drag a larger region.")
            return
        region = self.targets.add_region(self.target_panel.region_name(), *box)
        if region is None:
            self.target_panel.note("Three regions already. Clear one set before adding more.")
            return
        self.target_panel.clear_name()
        self.target_panel.note("")

    def _target_draft(self):
        drag = self._target_drag
        if drag is None:
            return None
        left = min(drag[0], drag[2])
        top = min(drag[1], drag[3])
        return (left, top, abs(drag[2] - drag[0]), abs(drag[3] - drag[1]))

    def _target_inputs(self, hands, snapshot):
        """Landmarks plus whether the stabilized pose is a point or a new pinch."""
        if not hands or snapshot is None:
            return None, False, False, False
        chosen = None
        for hand, result in zip(hands, snapshot.hands):
            pinch = result.gesture == PINCH and result.event == EVENT_STARTED
            stable = result.gesture == POINTING and result.status == STATUS_STABLE
            raw = result.raw_gesture == POINTING or stable
            rank = 3 if pinch else 2 if stable else 1 if raw else 0
            if chosen is None or rank > chosen[0]:
                chosen = (rank, getattr(hand, "landmarks", None), stable, raw, pinch)
        if chosen is None:
            return None, False, False, False
        return chosen[1], chosen[2], chosen[3], chosen[4]

    def _enforce_sidebar_width(self, _event=None):
        """Keep the metrics rail locked so the camera cannot collapse."""
        if not hasattr(self, "_sidebar"):
            return
        try:
            if int(self._sidebar.cget("width")) != self._sidebar_width:
                self._sidebar.configure(width=self._sidebar_width)
            self._sidebar.grid_propagate(False)
        except tk.TclError:
            pass

    def _on_axis_configure(self, event):
        size = (event.width, event.height)
        if size == self._axis_size:
            return
        self._axis_size = size
        self.draw_axes()

    # ----------------------------
    # CALIBRATION
    # ----------------------------

    def calibrate(self):
        if not self.face_visible:
            return

        if self.smoothed_rotation is not None:
            self.neutral_rotation = (
                self.smoothed_rotation.copy()
            )

        if self.smoothed_position is not None:
            self.neutral_position = (
                self.smoothed_position.copy()
            )

        self.blink_detector.reset()
        self.gesture_engine.reset()
        self.latest_blink = "None"
        self.latest_gesture = "None"
        self.gesture_flash_until = 0.0
        self.render_blink()
        self.render_gesture()

    # ----------------------------
    # HEAD ROTATION / POSITION
    # ----------------------------

    def calculate_pose(self, landmarks, width, height):
        image_points = np.array(
            [
                (
                    landmarks[i].x * width,
                    landmarks[i].y * height,
                )
                for i in POSE_POINTS
            ],
            dtype=np.float64,
        )

        focal_length = width

        camera_matrix = np.array(
            [
                [focal_length, 0, width / 2],
                [0, focal_length, height / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        distortion = np.zeros((4, 1))

        success, rotation_vector, translation = (
            cv2.solvePnP(
                MODEL_POINTS,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        )

        if not success:
            return None, None

        matrix, _ = cv2.Rodrigues(rotation_vector)
        angles = cv2.RQDecomp3x3(matrix)[0]

        # Wrap immediately so near-360° absolute pitch/yaw
        # never produces +356° relative readings after subtract.
        rotation = wrap_angles(
            np.array(
                [
                    angles[1],  # yaw
                    angles[0],  # pitch
                    angles[2],  # roll
                ],
                dtype=np.float64,
            )
        )

        # Normalized image-space movement and apparent face
        # size. These are relative measures, not centimeters.
        eye_left = landmarks[33]
        eye_right = landmarks[263]

        eye_distance = np.hypot(
            eye_right.x - eye_left.x,
            eye_right.y - eye_left.y,
        )

        nose = landmarks[1]

        position = np.array(
            [
                nose.x - 0.5,
                0.5 - nose.y,
                eye_distance,
            ],
            dtype=np.float64,
        )

        return rotation, position

    # ----------------------------
    # BLINK UI
    # ----------------------------

    def render_blink(self):
        label = self.latest_blink

        if label == "None":
            self.blink_label.config(
                text="Waiting for blink...",
                fg=MUTED,
            )
        else:
            colors = {
                "Left Eye": ACCENT,
                "Right Eye": PURPLE,
                "Both Eyes": GREEN,
            }

            self.blink_label.config(
                text=label,
                fg=colors.get(label, TEXT),
            )

        for name, widget in self.eye_indicators.items():
            active = name == label
            widget.config(
                bg=ACCENT if active else PANEL_2,
                fg="#06101b" if active else MUTED,
            )

    def _set_openness_bar(self, canvas, label, openness):
        openness = clamp(float(openness), 0.0, 1.0)
        width = max(canvas.winfo_width(), 40)
        fill = max(2, int(width * openness))
        color = GREEN if openness > 0.55 else (
            YELLOW if openness > 0.30 else ORANGE
        )
        canvas.coords(canvas._bar_id, 0, 0, fill, 8)
        canvas.itemconfigure(canvas._bar_id, fill=color)
        label.config(text=f"{int(round(openness * 100)):d}%")

    def render_openness(self):
        if self.blink_result is None:
            left = 1.0
            right = 1.0
        else:
            left = self.blink_result.left_openness
            right = self.blink_result.right_openness

        self._set_openness_bar(
            self.left_open_bar, self.left_open_label, left
        )
        self._set_openness_bar(
            self.right_open_bar, self.right_open_label, right
        )

    def render_gesture(self):
        label = self.latest_gesture
        if label == "None":
            self.gesture_label.config(
                text="Waiting for gesture...",
                fg=MUTED,
            )
            self.header_gesture_label.config(
                text="GESTURE: —",
                fg=MUTED,
            )
        else:
            self.gesture_label.config(
                text=label,
                fg=ACCENT,
            )
            self.header_gesture_label.config(
                text=f"GESTURE: {label}",
                fg=ACCENT,
            )
        self.render_gesture_status()

    def render_gesture_status(self):
        status = self.gesture_engine.status
        colors = {
            "Ready": GREEN,
            "Detecting": YELLOW,
            "Cooldown": ORANGE,
        }
        self.gesture_status_label.config(
            text=status,
            fg=colors.get(status, MUTED),
        )

    def render_gesture_flash(self):
        width = max(self.gesture_flash.winfo_width(), 10)
        active = time.perf_counter() < self.gesture_flash_until
        fill = width if active else 0
        self.gesture_flash.coords(
            self._gesture_flash_bar, 0, 0, fill, 5
        )
        self.gesture_flash.itemconfigure(
            self._gesture_flash_bar,
            fill=ACCENT if active else PANEL,
        )

    def render_hand_gesture(self):
        snapshot = self.hand_gesture_snapshot
        if snapshot is None:
            name, name_fg = "—", MUTED
            hand_text = "—"
            status, status_fg = "NO HAND", MUTED
            agree, window = 0, 0
            recent_text = "—"
        else:
            primary = snapshot.primary
            status = primary.status
            status_colors = {
                "DETECTING": YELLOW,
                "STABLE": GREEN,
                "UNKNOWN": ORANGE,
                "NO HAND": MUTED,
            }
            status_fg = status_colors.get(status, MUTED)
            if primary.named:
                name, name_fg = primary.label, ACCENT
            elif primary.gesture == "UNKNOWN":
                name, name_fg = "UNKNOWN", MUTED
            else:
                name, name_fg = "—", MUTED
            hand_text = primary.hand or "—"
            agree = primary.agree_count
            window = primary.window_count
            if snapshot.recent:
                recent_text = "\n".join(
                    f"{entry.hand:<5} {entry.label}"
                    for entry in snapshot.recent[:5]
                )
            else:
                recent_text = "—"

        self.hand_g_name.config(text=name, fg=name_fg)
        self.hand_g_hand.config(text=hand_text)
        self.hand_g_status.config(text=status, fg=status_fg)
        self.hand_g_recent.config(text=recent_text)

        bar_width = max(self.hand_g_stab_bar.winfo_width(), 40)
        ratio = (agree / window) if window else 0.0
        fill = int(bar_width * ratio) if ratio else 0
        if status == "STABLE":
            color = GREEN
        elif status == "DETECTING":
            color = YELLOW
        else:
            color = "#35506e"
        self.hand_g_stab_bar.coords(
            self._hand_g_stab_rect, 0, 0, fill, 8
        )
        self.hand_g_stab_bar.itemconfigure(
            self._hand_g_stab_rect, fill=color
        )
        if window:
            self.hand_g_stab_label.config(text=f"{agree}/{window}")
        else:
            self.hand_g_stab_label.config(text="—")

    # ----------------------------
    # AXIS VISUALIZATION
    # ----------------------------

    def draw_axes(self):
        canvas = self.axis_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 100)
        height = max(canvas.winfo_height(), 100)

        cx = width / 2
        cy = height / 2

        canvas.create_line(
            0, cy, width, cy,
            fill="#35506e", width=1,
        )
        canvas.create_line(
            cx, 0, cx, height,
            fill="#35506e", width=1,
        )

        for fraction in (0.25, 0.5, 0.75):
            x = width * fraction
            y = height * fraction
            canvas.create_line(
                x, 0, x, height,
                fill="#1a2c41",
            )
            canvas.create_line(
                0, y, width, y,
                fill="#1a2c41",
            )

        canvas.create_text(
            width - 15, cy - 12,
            text="X", fill=ACCENT,
        )
        canvas.create_text(
            cx + 13, 12,
            text="Y", fill=GREEN,
        )

        x = clamp(
            float(self.position[0]) * 3,
            -1, 1,
        )
        y = clamp(
            float(self.position[1]) * 3,
            -1, 1,
        )

        px = cx + x * width * 0.43
        py = cy - y * height * 0.43

        intensity = int(2 + 4 * max(abs(x), abs(y)))

        canvas.create_line(
            cx, cy, px, py,
            fill=ACCENT,
            width=intensity,
        )
        canvas.create_oval(
            px - 7, py - 7,
            px + 7, py + 7,
            fill=ACCENT,
            outline="white",
        )

    # ----------------------------
    # MAIN CAMERA LOOP
    # ----------------------------

    def update_frame(self):
        if not self.running:
            return

        success, frame = self._frames.read()

        if not success:
            self.status_label.config(
                text="● CAMERA ERROR",
                fg=YELLOW,
            )
            self.root.after(100, self.update_frame)
            return

        # Same snapshot as last pass: processing it again looks like the
        # hand stopped, which makes a real move feel sticky.
        if frame is self._seen_frame:
            self.root.after(4, self.update_frame)
            return
        self._seen_frame = frame

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        # Single RGB conversion shared by face mesh and hands.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        # One Hands pass is shared by overlay, gestures, lab, and desktop.
        need_hands = self._hands_needed()
        if need_hands != self.hand_tracker.enabled:
            self.hand_tracker.set_enabled(need_hands)
        tracked_hands = (
            self.hand_tracker.process(rgb) if need_hands else []
        )
        if self.show_hands.get():
            self.hand_tracker.draw(frame, tracked_hands)

        # Widgets reuse the same hand classifier. The gesture panel
        # still draws only when that checkbox is on.
        hand_snapshot = None
        if (
            self.recognize_gestures.get()
            or self.show_widgets.get()
            or self._targets_on
        ):
            hand_snapshot = self.hand_gesture_recognizer.update(tracked_hands)
        if self.recognize_gestures.get() and hand_snapshot is not None:
            self.hand_gesture_snapshot = hand_snapshot
            for hand, gesture in zip(tracked_hands, hand_snapshot.hands):
                draw_hand_gesture_overlay(frame, hand, gesture)
            self.render_hand_gesture()
        # Pointing confirms a widget. While Targets is on, that same pose
        # is aim, so a point should not also toggle a widget.
        if (
            self.show_widgets.get()
            and hand_snapshot is not None
            and not self._targets_on
        ):
            self.head_widgets.handle_hands(hand_snapshot)

        if self._interaction_needed():
            iframe = self.interaction_engine.update(tracked_hands)
            if self._lab_on:
                self.lab.apply(iframe)
            if self._desktop_on:
                self.desktop.apply(iframe, fps=self.fps)
                self.desktop_panel.render(self.desktop.diagnostics())
            draw_index_pointer(frame, iframe)

        self.face_visible = bool(
            result.multi_face_landmarks
        )

        if self.face_visible:
            self.status_label.config(
                text="● TRACKING",
                fg=GREEN,
            )

            face = result.multi_face_landmarks[0]
            landmarks = face.landmark

            # --- BLINK DETECTION ---
            self.blink_result = (
                self.blink_detector.update(landmarks)
            )

            if self.blink_result.detected:
                self.latest_blink = (
                    self.blink_result.latest
                )
                self.render_blink()

            # --- HEAD POSE / POSITION ---
            rotation, position = self.calculate_pose(
                landmarks, width, height
            )

            if rotation is not None:
                self.smoothed_rotation = smooth_angles(
                    self.smoothed_rotation, rotation
                )
                self.smoothed_position = smooth(
                    self.smoothed_position, position
                )

                if self.neutral_rotation is None:
                    self.neutral_rotation = (
                        self.smoothed_rotation.copy()
                    )

                if self.neutral_position is None:
                    self.neutral_position = (
                        self.smoothed_position.copy()
                    )

                self.rotation = wrap_angles(
                    self.smoothed_rotation
                    - self.neutral_rotation
                )

                self.position = (
                    self.smoothed_position
                    - self.neutral_position
                )

            gesture_event = self.gesture_engine.update(
                self.blink_result,
                self.rotation,
            )
            if gesture_event is not None:
                self.latest_gesture = gesture_event.label
                self.gesture_flash_until = (
                    time.perf_counter() + 0.45
                )
                self.render_gesture()
                if self.show_widgets.get():
                    self.head_widgets.handle(gesture_event)
            else:
                self.render_gesture_status()

            # --- OPTIONAL FACE MESH ---
            if self.show_mesh.get():
                for landmark in landmarks:
                    px = int(landmark.x * width)
                    py = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (px, py),
                        1,
                        (80, 210, 250),
                        -1,
                    )

            # --- OPTIONAL EYE SOCKET BOUNDS ---
            if self.show_eyes.get():
                for indices, closed, color in [
                    (
                        BlinkDetector.LEFT_EYE_CONTOUR,
                        self.blink_result.left_closed,
                        (80, 180, 255),
                    ),
                    (
                        BlinkDetector.RIGHT_EYE_CONTOUR,
                        self.blink_result.right_closed,
                        (190, 120, 255),
                    ),
                ]:
                    points = np.array(
                        [
                            (
                                int(landmarks[i].x * width),
                                int(landmarks[i].y * height),
                            )
                            for i in indices
                        ],
                        dtype=np.int32,
                    )

                    cv2.polylines(
                        frame,
                        [points],
                        True,
                        (50, 220, 90) if closed else color,
                        2,
                    )

            # --- OPTIONAL POSE AXES ---
            if self.show_axes.get():
                nose = landmarks[1]
                nx = int(nose.x * width)
                ny = int(nose.y * height)

                yaw, pitch, roll = self.rotation

                length = 80

                cv2.line(
                    frame,
                    (nx, ny),
                    (
                        nx + int(length * np.cos(
                            np.radians(roll)
                        )),
                        ny + int(length * np.sin(
                            np.radians(roll)
                        )),
                    ),
                    (255, 100, 100), 3,
                )

                cv2.line(
                    frame,
                    (nx, ny),
                    (
                        nx + int(yaw * 2),
                        ny - length,
                    ),
                    (80, 230, 120), 3,
                )

                cv2.line(
                    frame,
                    (nx, ny),
                    (
                        nx + int(yaw * 2),
                        ny + int(pitch * 2),
                    ),
                    (100, 150, 255), 3,
                )

            if self.show_widgets.get():
                self.head_widgets.draw(frame, landmarks)

        else:
            self.status_label.config(
                text="● NO FACE",
                fg=YELLOW,
            )
            self.blink_detector.face_lost()
            self.blink_result = None
            self.head_widgets.reset_anchor()
            self.render_gesture_status()

        if self._targets_on:
            landmarks, stable, raw, pinch = self._target_inputs(
                tracked_hands, hand_snapshot
            )
            aspect = width / float(height) if height else 1.0
            target_frame = self.targets.update(
                landmarks,
                time.perf_counter(),
                pointing_stable=stable,
                pointing_raw=raw,
                pinch_started=pinch,
                aspect=aspect,
            )
            draw_target_overlay(frame, target_frame, self._target_draft())
            self.target_panel.render(target_frame)

        self.update_metrics()
        self.draw_axes()
        self.render_openness()
        self.render_gesture_flash()

        # FPS calculation.
        now = time.perf_counter()
        elapsed = max(now - self.last_time, 1e-6)
        self.last_time = now

        instant_fps = 1.0 / elapsed
        self.fps = (
            0.9 * self.fps
            + 0.1 * instant_fps
        ) if self.fps else instant_fps

        self.fps_label.config(
            text=f"FPS: {self.fps:.1f}"
        )

        # Contain-fit inside the video host (preserve aspect, no stretch).
        # Size comes from the host frame, never the image label — that
        # feedback loop is what collapsed the camera column before.
        target_width = max(self.video_frame.winfo_width() - 8, 1)
        target_height = max(self.video_frame.winfo_height() - 8, 1)
        if target_width < 32 or target_height < 32:
            self.root.after(50, self.update_frame)
            return

        scale = min(
            target_width / width,
            target_height / height,
        )
        display = cv2.resize(
            frame,
            (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

        image = Image.fromarray(
            cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        )

        self.video_image = ImageTk.PhotoImage(image)

        self.camera_label.config(
            image=self.video_image,
            text="",
        )

        self.root.after(1, self.update_frame)

    # ----------------------------
    # METRIC UPDATES
    # ----------------------------

    def update_metrics(self):
        yaw, pitch, roll = self.rotation
        x, y, z = self.position

        self.yaw_label.config(
            text=f"{yaw:+.1f}°"
        )
        self.pitch_label.config(
            text=f"{pitch:+.1f}°"
        )
        self.roll_label.config(
            text=f"{roll:+.1f}°"
        )

        self.x_label.config(
            text=f"{x:+.3f}"
        )
        self.y_label.config(
            text=f"{y:+.3f}"
        )
        self.z_label.config(
            text=f"{z:+.3f}"
        )

    def close(self):
        self.running = False
        try:
            self.desktop.disable(reason="CLOSED", notify=False)
        except Exception:
            pass

        self._frames.close()
        if self.cap.isOpened():
            self.cap.release()
        self._frames.join()

        self.hand_tracker.close()
        self.mesh.close()
        self.root.destroy()


if __name__ == "__main__":
    print(
        "AURA starting — Desktop Control is off. Enable it from the camera "
        "bar; press Esc to stop immediately."
    )
    root = tk.Tk()
    app = AuraHUD(root)
    root.mainloop()
