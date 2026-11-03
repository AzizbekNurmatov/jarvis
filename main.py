
import time
import tkinter as tk
from tkinter import ttk

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

from aura.core.blink import BlinkDetector


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


class AuraHUD:
    def __init__(self, root):
        self.root = root
        self.root.title("AURA | Head Tracking HUD")
        self.root.geometry("1460x900")
        self.root.minsize(1100, 720)
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

        self.show_mesh = tk.BooleanVar(value=False)
        self.show_axes = tk.BooleanVar(value=True)

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
            padx=18,
            pady=15,
            highlightbackground="#26354d",
            highlightthickness=1,
        )

        tk.Label(
            outer,
            text=title,
            fg=TEXT,
            bg=PANEL,
            font=("Segoe UI", 12, "bold"),
            anchor="w",
        ).pack(fill="x")

        if subtitle:
            tk.Label(
                outer,
                text=subtitle,
                fg=MUTED,
                bg=PANEL,
                font=("Segoe UI", 9),
                anchor="w",
            ).pack(fill="x", pady=(3, 10))

        return outer

    def metric(self, parent, name, color):
        item = tk.Frame(parent, bg=PANEL)
        item.pack(side="left", expand=True, fill="x")

        tk.Label(
            item,
            text=name,
            fg=MUTED,
            bg=PANEL,
            font=("Segoe UI", 10),
        ).pack()

        label = tk.Label(
            item,
            text="0.0",
            fg=color,
            bg=PANEL,
            font=("Segoe UI", 22, "bold"),
        )
        label.pack(pady=(5, 0))
        return label

    def build_ui(self):
        header = tk.Frame(self.root, bg=BG, padx=24, pady=18)
        header.pack(fill="x")

        tk.Label(
            header,
            text="AURA",
            fg=ACCENT,
            bg=BG,
            font=("Segoe UI", 25, "bold"),
        ).pack(side="left")

        tk.Label(
            header,
            text="  /  SPATIAL HEAD TRACKING",
            fg=MUTED,
            bg=BG,
            font=("Segoe UI", 11),
        ).pack(side="left", pady=(10, 0))

        self.status_label = tk.Label(
            header,
            text="● INITIALIZING",
            fg=YELLOW,
            bg=BG,
            font=("Segoe UI", 11, "bold"),
        )
        self.status_label.pack(side="right")

        body = tk.Frame(self.root, bg=BG, padx=22, pady=8)
        body.pack(fill="both", expand=True)

        body.grid_columnconfigure(0, weight=5)
        body.grid_columnconfigure(1, weight=4)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        left.grid(
            row=0, column=0,
            sticky="nsew", padx=(0, 10)
        )

        right = tk.Frame(body, bg=BG)
        right.grid(
            row=0, column=1,
            sticky="nsew", padx=(10, 0)
        )

        # Camera preview.
        camera_card = self.card(
            left, "LIVE CAMERA",
            "Face Mesh / Pose / Eye Detection"
        )
        camera_card.pack(fill="both", expand=True)

        self.camera_label = tk.Label(
            camera_card,
            bg="#050a14",
            text="Initializing camera...",
            fg=MUTED,
        )
        self.camera_label.pack(
            fill="both", expand=True, pady=(8, 8)
        )

        controls = tk.Frame(camera_card, bg=PANEL)
        controls.pack(fill="x")

        tk.Checkbutton(
            controls,
            text="Face Mesh",
            variable=self.show_mesh,
            bg=PANEL, fg=TEXT,
            selectcolor=PANEL_2,
            activebackground=PANEL,
            activeforeground=TEXT,
        ).pack(side="left", padx=5)

        tk.Checkbutton(
            controls,
            text="Pose Axes",
            variable=self.show_axes,
            bg=PANEL, fg=TEXT,
            selectcolor=PANEL_2,
            activebackground=PANEL,
            activeforeground=TEXT,
        ).pack(side="left", padx=5)

        tk.Button(
            controls,
            text="Calibrate Neutral",
            command=self.calibrate,
            bg=ACCENT,
            fg="#06101b",
            relief="flat",
            padx=16,
            pady=7,
            font=("Segoe UI", 10, "bold"),
        ).pack(side="right", padx=5)

        # Right-side metrics.
        rotation_card = self.card(
            right, "HEAD ROTATION",
            "Relative orientation (degrees)"
        )
        rotation_card.pack(fill="x", pady=(0, 12))

        rotation_row = tk.Frame(rotation_card, bg=PANEL)
        rotation_row.pack(fill="x", pady=(8, 4))

        self.yaw_label = self.metric(
            rotation_row, "YAW", ACCENT
        )
        self.pitch_label = self.metric(
            rotation_row, "PITCH", GREEN
        )
        self.roll_label = self.metric(
            rotation_row, "ROLL", PURPLE
        )

        position_card = self.card(
            right, "HEAD POSITION",
            "Relative to your calibrated neutral position"
        )
        position_card.pack(fill="x", pady=(0, 12))

        position_row = tk.Frame(position_card, bg=PANEL)
        position_row.pack(fill="x", pady=(5, 4))

        self.x_label = self.metric(
            position_row, "X", ACCENT
        )
        self.y_label = self.metric(
            position_row, "Y", GREEN
        )
        self.z_label = self.metric(
            position_row, "Z", PURPLE
        )

        # Two-dimensional live motion graph.
        graph_card = self.card(
            right, "MOVEMENT AXES",
            "Head movement relative to center"
        )
        graph_card.pack(fill="x", pady=(0, 12))

        self.axis_canvas = tk.Canvas(
            graph_card,
            bg="#0c1626",
            height=145,
            highlightthickness=0,
        )
        self.axis_canvas.pack(fill="x", pady=(5, 0))
        self.axis_canvas.bind(
            "<Configure>",
            lambda event: self.draw_axes(),
        )

        # Blink indicator.
        blink_card = self.card(
            right,
            "EYE TRACKING",
            "Most recent blink — no history recorded",
        )
        blink_card.pack(fill="x", pady=(0, 12))

        latest_panel = tk.Frame(
            blink_card,
            bg=PANEL_2,
            padx=16,
            pady=10,
        )
        latest_panel.pack(fill="x")

        tk.Label(
            latest_panel,
            text="LATEST DETECTED BLINK",
            bg=PANEL_2,
            fg=MUTED,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")

        self.blink_label = tk.Label(
            latest_panel,
            text="Waiting for blink...",
            bg=PANEL_2,
            fg=TEXT,
            font=("Segoe UI", 20, "bold"),
        )
        self.blink_label.pack(anchor="w", pady=(5, 0))

        indicators = tk.Frame(blink_card, bg=PANEL)
        indicators.pack(fill="x", pady=(12, 0))

        self.eye_indicators = {}

        for key, title in [
            ("Left Eye", "LEFT"),
            ("Right Eye", "RIGHT"),
            ("Both Eyes", "BOTH"),
        ]:
            indicator = tk.Label(
                indicators,
                text=f"◉  {title}",
                bg=PANEL_2,
                fg=MUTED,
                padx=12,
                pady=10,
                font=("Segoe UI", 10, "bold"),
            )
            indicator.pack(
                side="left",
                fill="x",
                expand=True,
                padx=3,
            )
            self.eye_indicators[key] = indicator

        self.fps_label = tk.Label(
            right,
            text="FPS: --",
            fg=MUTED,
            bg=BG,
            font=("Consolas", 10),
        )
        self.fps_label.pack(anchor="e", pady=5)

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
        self.latest_blink = "None"
        self.render_blink()

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

        rotation = np.array(
            [
                angles[1],  # yaw
                angles[0],  # pitch
                angles[2],  # roll
            ],
            dtype=np.float64,
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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

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
                self.smoothed_rotation = smooth(
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

                self.rotation = (
                    self.smoothed_rotation
                    - self.neutral_rotation
                )

                self.position = (
                    self.smoothed_position
                    - self.neutral_position
                )

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

        self.update_metrics()
        self.draw_axes()

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

        # Resize image to fit camera panel.
        target_width = max(
            self.camera_label.winfo_width(), 320
        )
        target_height = max(
            self.camera_label.winfo_height(), 240
        )

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

        self.mesh.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AuraHUD(root)
    root.mainloop()
