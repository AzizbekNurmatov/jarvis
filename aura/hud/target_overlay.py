"""Draw calibrated regions and the 2D aim ray on the camera frame."""

from __future__ import annotations

from aura.core.targets import (
    STATE_AMBIGUOUS,
    STATE_CONFIRMED,
    STATE_LOCKED,
)


_PANEL = (48, 30, 21)
_ACCENT = (248, 189, 56)
_GREEN = (153, 211, 52)
_YELLOW = (36, 191, 251)
_TEXT = (255, 244, 238)
_MUTED = (186, 160, 141)
_OUTLINE = (77, 53, 38)


def draw_target_overlay(frame, tframe, draft=None) -> None:
    """`draft` is an in-progress box (x, y, w, h) in normalized image space."""
    if frame is None or tframe is None:
        return
    import cv2

    height, width = frame.shape[:2]
    by_id = {region.id: region for region in tframe.regions}
    aimed = tframe.aimed_id
    ambiguous = set(tframe.contenders) if tframe.state == STATE_AMBIGUOUS else set()

    for region in tframe.regions:
        x, y, w, h = _pixels(region.x, region.y, region.w, region.h, width, height)
        if region.id == aimed and tframe.state == STATE_CONFIRMED:
            color, thickness = _GREEN, 3
        elif region.id == aimed and tframe.state == STATE_LOCKED:
            color, thickness = _ACCENT, 3
        elif region.id == aimed:
            color, thickness = _ACCENT, 2
        elif region.id in ambiguous:
            color, thickness = _YELLOW, 2
        else:
            color, thickness = _OUTLINE, 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)
        _label(frame, region.name, x, y)

    if draft is not None:
        x, y, w, h = _pixels(draft[0], draft[1], draft[2], draft[3], width, height)
        cv2.rectangle(frame, (x, y), (x + w, y + h), _ACCENT, 1, cv2.LINE_AA)

    ray = tframe.ray
    if ray is not None:
        x0 = int(round(ray[0] * width))
        y0 = int(round(ray[1] * height))
        x1 = int(round(ray[2] * width))
        y1 = int(round(ray[3] * height))
        cv2.line(frame, (x0, y0), (x1, y1), _ACCENT, 2, cv2.LINE_AA)
        cv2.circle(frame, (x0, y0), 4, _ACCENT, -1, cv2.LINE_AA)

    cv2.putText(
        frame,
        "2D IMAGE AIM",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        _MUTED,
        1,
        cv2.LINE_AA,
    )


def _pixels(x, y, w, h, width, height):
    left = int(round(x * width))
    top = int(round(y * height))
    wide = max(1, int(round(w * width)))
    tall = max(1, int(round(h * height)))
    return left, top, wide, tall


def _label(frame, text, x, y) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    top = max(0, y - th - 8)
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + 6), _PANEL, -1)
    cv2.putText(
        frame,
        text,
        (x + 4, top + th + 2),
        font,
        scale,
        _TEXT,
        1,
        cv2.LINE_AA,
    )
