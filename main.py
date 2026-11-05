
import time
import tkinter as tk

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

from aura.core.blink import BlinkDetector
from aura.core.gesture_engine import GestureEngine
from aura.core.hand_tracking import HandTracker
from aura.core.pose import wrap_angles


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


class AuraHUD:
    def __init__(self, root):
        self.root = root
        self.root.title("AURA | Head Tracking HUD")
        self.root.geometry("1460x900")
        self.root.minsize(1180, 760)
        self.root.configure(bg=BG)

        self.running = True

        self.cap = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW,
        )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

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

        self.show_mesh = tk.BooleanVar(value=False)
        self.show_axes = tk.BooleanVar(value=True)
        self.show_hands = tk.BooleanVar(value=False)

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
            font=("Consolas", 15, "bold"),
            width=7,
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

        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(1, weight=1)
        right.grid_rowconfigure(3, weight=1)

        # Camera preview.
        camera_card = self.card(
            left, "LIVE CAMERA",
            "Face / pose / eyes / optional hands"
        )
        camera_card.pack(fill="both", expand=True)

        controls = tk.Frame(camera_card, bg=PANEL)
        controls.pack(side="bottom", fill="x", pady=(6, 0))

        for text, variable in (
            ("Face Mesh", self.show_mesh),
            ("Pose Axes", self.show_axes),
            ("Hand Tracking", self.show_hands),
        ):
            tk.Checkbutton(
                controls,
                text=text,
                variable=variable,
                command=(
                    self._on_hand_toggle
                    if variable is self.show_hands
                    else None
                ),
                bg=PANEL, fg=TEXT,
                selectcolor=PANEL_2,
                activebackground=PANEL,
                activeforeground=TEXT,
            ).pack(side="left", padx=4)

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
        self.camera_label.place(relx=0.5, rely=0.5, anchor="center")

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

    def _on_hand_toggle(self):
        self.hand_tracker.set_enabled(self.show_hands.get())

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

        success, frame = self.cap.read()

        if not success:
            self.status_label.config(
                text="● CAMERA ERROR",
                fg=YELLOW,
            )
            self.root.after(100, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        # Single RGB conversion shared by face mesh and hands.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        hands_on = self.show_hands.get()
        if hands_on != self.hand_tracker.enabled:
            self.hand_tracker.set_enabled(hands_on)
        if hands_on:
            tracked_hands = self.hand_tracker.process(rgb)
            self.hand_tracker.draw(frame, tracked_hands)

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

            # --- EYE MARKERS ---
            for indices, closed, color in [
                (
                    BlinkDetector.LEFT_EYE,
                    self.blink_result.left_closed,
                    (80, 180, 255),
                ),
                (
                    BlinkDetector.RIGHT_EYE,
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

        else:
            self.status_label.config(
                text="● NO FACE",
                fg=YELLOW,
            )
            self.blink_detector.face_lost()
            self.blink_result = None
            self.render_gesture_status()

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

        # Fit the frame inside the fixed video host, never the
        # image label itself (that causes a resize feedback loop).
        target_width = max(self.video_frame.winfo_width() - 4, 1)
        target_height = max(self.video_frame.winfo_height() - 4, 1)
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

        self.root.after(15, self.update_frame)

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

        if self.cap.isOpened():
            self.cap.release()

        self.hand_tracker.close()
        self.mesh.close()
        self.root.destroy()


if __name__ == "__main__":
    print("AURA starting — hand tracking is off until you enable the checkbox.")
    root = tk.Tk()
    app = AuraHUD(root)
    root.mainloop()
