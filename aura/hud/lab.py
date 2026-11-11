"""Interaction Lab — pinch-driven HUD widgets (no OS / TARS side effects)."""

from __future__ import annotations

import time
import tkinter as tk

from aura.core.interaction import (
    CURSOR_MOVE,
    PINCH_END,
    PINCH_HOLD,
    PINCH_START,
    InteractionFrame,
)

PANEL = "#151e30"
PANEL_2 = "#1b2940"
TEXT = "#eef4ff"
MUTED = "#8da0ba"
ACCENT = "#38bdf8"
GREEN = "#34d399"
YELLOW = "#fbbf24"
CANVAS_BG = "#0c1626"
OUTLINE = "#26354d"


def _hit(x: float, y: float, box, pad: float = 0.0) -> bool:
    return (
        box["x"] - pad <= x <= box["x"] + box["w"] + pad
        and box["y"] - pad <= y <= box["y"] + box["h"] + pad
    )


class InteractionLab:
    """Dedicated canvas: draggable node, pinch button, and slider."""

    def __init__(self, parent):
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
            text="INTERACTION LAB",
            fg=TEXT,
            bg=PANEL,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(side="left")

        tk.Label(
            header,
            text="Index finger aims  ·  Pinch to grab / click",
            fg=MUTED,
            bg=PANEL,
            font=("Segoe UI", 8),
        ).pack(side="left", padx=(10, 0), pady=(1, 0))

        self.status = tk.Label(
            header,
            text="WAITING FOR HAND",
            fg=MUTED,
            bg=PANEL,
            font=("Consolas", 8),
        )
        self.status.pack(side="right")

        self.canvas = tk.Canvas(
            self.frame,
            bg=CANVAS_BG,
            height=220,
            highlightthickness=0,
        )
        self.canvas.pack(fill="x", pady=(6, 0))
        self.canvas.bind("<Configure>", self._on_resize)

        self._size = (640, 220)
        self._grab = None
        self._button_consumed = False
        self._hover = ""

        self.card = {"x": 28.0, "y": 28.0, "w": 150.0, "h": 70.0}
        self.button = {
            "x": 210.0,
            "y": 36.0,
            "w": 128.0,
            "h": 42.0,
            "clicks": 0,
            "flash_until": 0.0,
        }
        self.slider = {
            "x": 28.0,
            "y": 130.0,
            "w": 420.0,
            "h": 36.0,
            "value": 0.42,
        }

        self._ids = {}
        self._build_items()

    def _on_resize(self, event):
        self._size = (max(event.width, 1), max(event.height, 1))
        self._layout_defaults()
        self._sync_widgets()

    def _layout_defaults(self):
        width, height = self._size
        self.slider["w"] = max(220.0, width - 56.0)
        self.slider["y"] = max(118.0, height - 78.0)
        self.button["x"] = min(self.button["x"], width - self.button["w"] - 16.0)

    def _build_items(self):
        c = self.canvas
        self._ids["card"] = c.create_rectangle(
            0, 0, 1, 1,
            fill=PANEL, outline=OUTLINE, width=2, tags="card",
        )
        self._ids["card_bar"] = c.create_rectangle(
            0, 0, 1, 1, fill=ACCENT, outline="", tags="card",
        )
        self._ids["card_text"] = c.create_text(
            0, 0, text="NODE", fill=TEXT,
            font=("Segoe UI", 11, "bold"), tags="card",
        )
        self._ids["card_sub"] = c.create_text(
            0, 0, text="Pinch and drag", fill=MUTED,
            font=("Segoe UI", 8), tags="card",
        )

        self._ids["btn"] = c.create_rectangle(
            0, 0, 1, 1,
            fill=ACCENT, outline="", tags="btn",
        )
        self._ids["btn_text"] = c.create_text(
            0, 0, text="CLICK  0", fill="#06101b",
            font=("Segoe UI", 10, "bold"), tags="btn",
        )

        self._ids["track"] = c.create_rectangle(
            0, 0, 1, 1, fill=PANEL_2, outline=OUTLINE, tags="slider",
        )
        self._ids["fill"] = c.create_rectangle(
            0, 0, 1, 1, fill=ACCENT, outline="", tags="slider",
        )
        self._ids["knob"] = c.create_oval(
            0, 0, 1, 1, fill=TEXT, outline=ACCENT, width=2, tags="slider",
        )
        self._ids["slider_label"] = c.create_text(
            0, 0, text="LEVEL  42%", fill=MUTED,
            font=("Consolas", 9), anchor="w", tags="slider",
        )

        self._ids["cursor_ring"] = c.create_oval(
            0, 0, 1, 1, outline=ACCENT, width=2, tags="cursor",
        )
        self._ids["cursor_dot"] = c.create_oval(
            0, 0, 1, 1, fill=ACCENT, outline="", tags="cursor",
        )
        c.itemconfigure("cursor", state="hidden")
        self._sync_widgets()

    def reset(self):
        self._grab = None
        self._button_consumed = False
        self._hover = ""
        self.canvas.itemconfigure("cursor", state="hidden")
        self.status.config(text="WAITING FOR HAND", fg=MUTED)
        self._sync_widgets()

    def _pixels(self, nx: float, ny: float) -> tuple:
        width, height = self._size
        return nx * width, ny * height

    def _hit_target(self, px: float, py: float) -> str:
        if _hit(px, py, self.button, pad=4):
            return "button"
        if _hit(px, py, self.card, pad=4):
            return "card"
        if _hit(px, py, self.slider, pad=8):
            return "slider"
        return ""

    def _clamp_card(self):
        width, height = self._size
        self.card["x"] = max(8.0, min(self.card["x"], width - self.card["w"] - 8.0))
        self.card["y"] = max(8.0, min(self.card["y"], height - self.card["h"] - 8.0))

    def _click_button(self):
        self.button["clicks"] += 1
        self.button["flash_until"] = time.perf_counter() + 0.22
        self._button_consumed = True

    def _drag_card(self, px: float, py: float):
        grab = self._grab
        if not grab:
            return
        self.card["x"] = px - grab["dx"]
        self.card["y"] = py - grab["dy"]
        self._clamp_card()

    def _drag_slider(self, px: float):
        left = self.slider["x"] + 10.0
        span = max(1.0, self.slider["w"] - 20.0)
        self.slider["value"] = max(0.0, min(1.0, (px - left) / span))

    def apply(self, iframe: InteractionFrame):
        events = iframe.events if iframe is not None else []
        px = py = 0.0
        visible = bool(iframe and iframe.visible)
        pinching = bool(iframe and iframe.pinching)

        for event in events:
            px, py = self._pixels(event.x, event.y)
            if event.name == CURSOR_MOVE:
                continue
            if event.name == PINCH_START:
                target = self._hit_target(px, py)
                if target == "button" and not self._button_consumed:
                    self._click_button()
                elif target == "card":
                    self._grab = {
                        "kind": "card",
                        "dx": px - self.card["x"],
                        "dy": py - self.card["y"],
                    }
                elif target == "slider":
                    self._grab = {"kind": "slider", "dx": 0.0, "dy": 0.0}
                    self._drag_slider(px)
            elif event.name == PINCH_HOLD and self._grab:
                if self._grab["kind"] == "card":
                    self._drag_card(px, py)
                elif self._grab["kind"] == "slider":
                    self._drag_slider(px)
            elif event.name == PINCH_END:
                self._grab = None
                self._button_consumed = False

        if visible and iframe is not None:
            px, py = self._pixels(iframe.cursor_x, iframe.cursor_y)
            self._hover = self._hit_target(px, py) if not self._grab else (
                self._grab["kind"] if self._grab["kind"] != "slider" else "slider"
            )
            self._place_cursor(px, py, pinching)
            self._update_status(iframe, pinching)
        else:
            self._hover = ""
            self.canvas.itemconfigure("cursor", state="hidden")
            self.status.config(text="WAITING FOR HAND", fg=MUTED)

        self._sync_widgets()

    def _place_cursor(self, px: float, py: float, pinching: bool):
        color = GREEN if pinching else ACCENT
        radius = 9 if pinching else 11
        self.canvas.coords(
            self._ids["cursor_ring"],
            px - radius, py - radius, px + radius, py + radius,
        )
        self.canvas.coords(
            self._ids["cursor_dot"],
            px - 3, py - 3, px + 3, py + 3,
        )
        self.canvas.itemconfigure(self._ids["cursor_ring"], outline=color, state="normal")
        self.canvas.itemconfigure(self._ids["cursor_dot"], fill=color, state="normal")
        self.canvas.tag_raise("cursor")

    def _update_status(self, iframe: InteractionFrame, pinching: bool):
        if self._grab:
            label = "DRAGGING NODE" if self._grab["kind"] == "card" else "ADJUSTING SLIDER"
            color = GREEN
        elif pinching:
            label = "PINCH"
            color = GREEN
        elif self._hover == "button":
            label = "HOVER BUTTON"
            color = YELLOW
        elif self._hover:
            label = f"HOVER {self._hover.upper()}"
            color = YELLOW
        else:
            label = "AIMING"
            color = ACCENT
        hand = iframe.hand or "—"
        self.status.config(
            text=f"{hand}  ·  {iframe.cursor_x:.2f},{iframe.cursor_y:.2f}  ·  {label}",
            fg=color,
        )

    def _sync_widgets(self):
        c = self.canvas
        ids = self._ids
        card = self.card
        hover_card = self._hover == "card" or (self._grab and self._grab["kind"] == "card")
        c.coords(
            ids["card"],
            card["x"], card["y"],
            card["x"] + card["w"], card["y"] + card["h"],
        )
        c.coords(
            ids["card_bar"],
            card["x"], card["y"],
            card["x"] + 6, card["y"] + card["h"],
        )
        c.coords(
            ids["card_text"],
            card["x"] + 22, card["y"] + 26,
        )
        c.coords(
            ids["card_sub"],
            card["x"] + 22, card["y"] + 48,
        )
        c.itemconfigure(
            ids["card"],
            outline=ACCENT if hover_card else OUTLINE,
        )

        btn = self.button
        flashing = time.perf_counter() < btn["flash_until"]
        hover_btn = self._hover == "button"
        if flashing:
            fill = GREEN
        elif hover_btn:
            fill = "#7dd3fc"
        else:
            fill = ACCENT
        c.coords(
            ids["btn"],
            btn["x"], btn["y"],
            btn["x"] + btn["w"], btn["y"] + btn["h"],
        )
        c.coords(
            ids["btn_text"],
            btn["x"] + btn["w"] / 2,
            btn["y"] + btn["h"] / 2,
        )
        c.itemconfigure(ids["btn"], fill=fill)
        c.itemconfigure(
            ids["btn_text"],
            text=f"CLICK  {btn['clicks']}",
        )

        slider = self.slider
        track_y = slider["y"] + slider["h"] / 2
        left = slider["x"] + 10
        right = slider["x"] + slider["w"] - 10
        knob_x = left + slider["value"] * (right - left)
        hover_s = self._hover == "slider" or (
            self._grab and self._grab["kind"] == "slider"
        )
        c.coords(ids["track"], left, track_y - 5, right, track_y + 5)
        c.coords(ids["fill"], left, track_y - 5, knob_x, track_y + 5)
        c.coords(ids["knob"], knob_x - 8, track_y - 8, knob_x + 8, track_y + 8)
        c.coords(ids["slider_label"], left, slider["y"] + 2)
        c.itemconfigure(
            ids["slider_label"],
            text=f"LEVEL  {int(round(slider['value'] * 100)):d}%",
            fill=ACCENT if hover_s else MUTED,
        )
        c.itemconfigure(
            ids["knob"],
            outline=GREEN if hover_s else ACCENT,
        )
