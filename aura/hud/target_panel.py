"""Status strip for image-plane target selection. Display only."""

from __future__ import annotations

import tkinter as tk

from aura.core.targets import (
    STATE_AMBIGUOUS,
    STATE_CALIBRATE,
    STATE_CONFIRMED,
    STATE_LOCKED,
    STATE_LOST,
)

PANEL = "#151e30"
PANEL_2 = "#1b2940"
TEXT = "#eef4ff"
MUTED = "#8da0ba"
ACCENT = "#38bdf8"
GREEN = "#34d399"
YELLOW = "#fbbf24"
ORANGE = "#fb923c"
PURPLE = "#a78bfa"
OUTLINE = "#26354d"


class TargetPanel:
    def __init__(self, parent, on_clear=None):
        self.on_clear = on_clear
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
            text="TARGETS",
            fg=PURPLE,
            bg=PANEL,
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")
        tk.Label(
            header,
            text="2D image aim  ·  not 3D pointing",
            fg=MUTED,
            bg=PANEL,
            font=("Segoe UI", 8),
        ).pack(side="left", padx=(10, 0), pady=(2, 0))

        self.state_label = tk.Label(
            header,
            text="OFF",
            fg=MUTED,
            bg=PANEL,
            font=("Consolas", 9, "bold"),
        )
        self.state_label.pack(side="right")

        stats = tk.Frame(self.frame, bg=PANEL)
        stats.pack(fill="x", pady=(8, 0))
        self.target_label = tk.Label(
            stats, text="AIM  —", fg=TEXT, bg=PANEL, font=("Consolas", 9)
        )
        self.target_label.pack(side="left")
        self.last_label = tk.Label(
            stats, text="LAST  —", fg=MUTED, bg=PANEL, font=("Consolas", 9)
        )
        self.last_label.pack(side="left", padx=(16, 0))

        tools = tk.Frame(self.frame, bg=PANEL)
        tools.pack(fill="x", pady=(8, 0))
        self.name_var = tk.StringVar()
        entry = tk.Entry(
            tools,
            textvariable=self.name_var,
            bg=PANEL_2,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Segoe UI", 9),
            width=18,
        )
        entry.pack(side="left", ipady=3)
        tk.Button(
            tools,
            text="Clear",
            command=self._clear,
            bg=PANEL_2,
            fg=TEXT,
            relief="flat",
            padx=10,
            pady=2,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="left", padx=(8, 0))

        self.hint = tk.Label(
            self.frame,
            text="Drag on the preview to add up to 3 regions. Point until the box locks, then pinch.",
            fg=MUTED,
            bg=PANEL,
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.hint.pack(fill="x", pady=(6, 0))

    def region_name(self) -> str:
        return self.name_var.get().strip()

    def clear_name(self) -> None:
        self.name_var.set("")

    def note(self, text: str) -> None:
        self.hint.config(
            text=text
            or "Drag on the preview to add up to 3 regions. Point until the box locks, then pinch."
        )

    def _clear(self) -> None:
        if self.on_clear is not None:
            self.on_clear()

    def render(self, tframe) -> None:
        colors = {
            STATE_LOCKED: ACCENT,
            STATE_CONFIRMED: GREEN,
            STATE_AMBIGUOUS: YELLOW,
            STATE_LOST: ORANGE,
            STATE_CALIBRATE: MUTED,
        }
        state = tframe.state
        self.state_label.config(text=state, fg=colors.get(state, MUTED))
        if state == STATE_AMBIGUOUS:
            aimed = "AMBIGUOUS"
        elif state == STATE_CALIBRATE:
            aimed = "—"
        else:
            aimed = tframe.aimed_name or "—"
        self.target_label.config(text=f"AIM  {aimed}")
        last = tframe.last_selection.name if tframe.last_selection else "—"
        self.last_label.config(text=f"LAST  {last}")
