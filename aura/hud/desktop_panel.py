"""Compact Desktop Control status strip (no OS calls)."""

from __future__ import annotations

import tkinter as tk

from aura.core.desktop import DesktopDiagnostics

PANEL = "#151e30"
PANEL_2 = "#1b2940"
TEXT = "#eef4ff"
MUTED = "#8da0ba"
ACCENT = "#38bdf8"
GREEN = "#34d399"
YELLOW = "#fbbf24"
ORANGE = "#fb923c"
OUTLINE = "#26354d"


class DesktopPanel:
    def __init__(self, parent, on_sensitivity=None):
        self.on_sensitivity = on_sensitivity
        self.frame = tk.Frame(
            parent,
            bg=PANEL,
            padx=12,
            pady=8,
            highlightbackground=OUTLINE,
            highlightthickness=1,
        )

        header = tk.Frame(self.frame, bg=PANEL)
        header.pack(fill="x")

        tk.Label(
            header,
            text="DESKTOP CONTROL",
            fg=ORANGE,
            bg=PANEL,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(side="left")

        tk.Label(
            header,
            text="Esc emergency stop",
            fg=MUTED,
            bg=PANEL,
            font=("Segoe UI", 8),
        ).pack(side="left", padx=(10, 0), pady=(1, 0))

        self.status = tk.Label(
            header,
            text="OFF",
            fg=MUTED,
            bg=PANEL,
            font=("Consolas", 9, "bold"),
        )
        self.status.pack(side="right")

        stats = tk.Frame(self.frame, bg=PANEL)
        stats.pack(fill="x", pady=(8, 0))

        self.fps_label = tk.Label(
            stats, text="FPS  --", fg=TEXT, bg=PANEL, font=("Consolas", 9)
        )
        self.fps_label.pack(side="left")

        self.lat_label = tk.Label(
            stats, text="LAT  -- ms", fg=TEXT, bg=PANEL, font=("Consolas", 9)
        )
        self.lat_label.pack(side="left", padx=(16, 0))

        self.screen_label = tk.Label(
            stats, text="", fg=MUTED, bg=PANEL, font=("Consolas", 8)
        )
        self.screen_label.pack(side="left", padx=(16, 0))

        sens_wrap = tk.Frame(self.frame, bg=PANEL)
        sens_wrap.pack(fill="x", pady=(6, 0))

        tk.Label(
            sens_wrap,
            text="SENSITIVITY",
            fg=MUTED,
            bg=PANEL,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="left")

        self.sens_value = tk.Label(
            sens_wrap,
            text="1.00",
            fg=ACCENT,
            bg=PANEL,
            font=("Consolas", 9),
            width=5,
        )
        self.sens_value.pack(side="right")

        self.sens = tk.Scale(
            sens_wrap,
            from_=0.70,
            to=1.40,
            resolution=0.05,
            orient="horizontal",
            bg=PANEL,
            fg=TEXT,
            troughcolor=PANEL_2,
            highlightthickness=0,
            showvalue=0,
            sliderrelief="flat",
            activebackground=ACCENT,
            command=self._on_scale,
        )
        self.sens.set(1.00)
        self.sens.pack(side="left", fill="x", expand=True, padx=(10, 8))

        self.hint = tk.Label(
            self.frame,
            text="Index finger moves the Windows cursor  ·  Short pinch = click  ·  Hold pinch = drag",
            fg=MUTED,
            bg=PANEL,
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.hint.pack(fill="x", pady=(4, 0))

    def _on_scale(self, raw):
        value = float(raw)
        self.sens_value.config(text=f"{value:.2f}")
        if self.on_sensitivity is not None:
            self.on_sensitivity(value)

    def render(self, diag: DesktopDiagnostics):
        colors = {
            "OFF": MUTED,
            "AIMING": ACCENT,
            "PINCH": YELLOW,
            "DRAG": GREEN,
            "LOST": ORANGE,
            "ERROR": ORANGE,
        }
        status = diag.status or "OFF"
        self.status.config(
            text=status,
            fg=colors.get(status, MUTED),
        )
        fps = diag.fps
        self.fps_label.config(
            text=f"FPS  {fps:.1f}" if fps else "FPS  --"
        )
        lat = diag.latency_ms
        self.lat_label.config(
            text=f"LAT  {lat:.0f} ms" if lat else "LAT  -- ms"
        )
        self.screen_label.config(text=diag.screen or "")
