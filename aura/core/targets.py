"""Image-plane target selection from an index-finger ray.

This is approximate 2D targeting in the selfie-mirrored camera image.
A hit means the finger's on-screen direction crosses a rectangle the user
drew. It is not a 3D ray into the room: there is no depth and no object model.

Gesture recognition stays in hand_gestures. This module only decides which
region is aimed and emits a selection. It does not run an action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import math


AIM_MODE = "image_plane_2d"

STATE_CALIBRATE = "CALIBRATE"
STATE_IDLE = "IDLE"
STATE_AIMING = "AIMING"
STATE_LOCKED = "LOCKED"
STATE_AMBIGUOUS = "AMBIGUOUS"
STATE_LOST = "LOST"
STATE_CONFIRMED = "CONFIRMED"

MAX_TARGETS = 3
_INDEX_MCP = 5
_INDEX_TIP = 8

# Two hits this close, in aspect-corrected image units, are not a choice.
_AMBIGUOUS_GAP = 0.06
_MIN_DIRECTION = 0.035
_MIN_REGION = 0.045


@dataclass(frozen=True)
class TargetRegion:
    id: str
    name: str
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class TargetSelection:
    """One confirmed selection. Consumers such as TARS read this; nothing here acts."""

    target_id: str
    name: str
    timestamp: float
    mode: str = AIM_MODE
    aim_x: float = 0.0
    aim_y: float = 0.0


@dataclass(frozen=True)
class AimHit:
    region: Optional[TargetRegion]
    ambiguous: bool
    contenders: Tuple[TargetRegion, ...]
    ray: Optional[Tuple[float, float, float, float]]
    aim_x: float
    aim_y: float


@dataclass
class TargetFrame:
    mode: str
    state: str
    aimed_id: Optional[str]
    aimed_name: str
    contenders: Tuple[str, ...]
    regions: Tuple[TargetRegion, ...]
    ray: Optional[Tuple[float, float, float, float]]
    last_selection: Optional[TargetSelection]
    # Set only on the update that confirms. Later frames leave this None.
    selection: Optional[TargetSelection]


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def normalize_region(
    x0: float, y0: float, x1: float, y1: float
) -> Optional[Tuple[float, float, float, float]]:
    """Axis-aligned box in normalized image space, or None if it is too small."""
    left = _clamp01(min(x0, x1))
    top = _clamp01(min(y0, y1))
    right = _clamp01(max(x0, x1))
    bottom = _clamp01(max(y0, y1))
    width = right - left
    height = bottom - top
    if width < _MIN_REGION or height < _MIN_REGION:
        return None
    return left, top, width, height


def _point(landmarks, index: int) -> Optional[Tuple[float, float]]:
    if landmarks is None or len(landmarks) <= index:
        return None
    point = landmarks[index]
    return float(point.x), float(point.y)


def _ray_t(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    region: TargetRegion,
    max_t: float,
) -> Optional[float]:
    """Distance along a unit ray to the rectangle, or None on a miss.

    Coordinates are aspect-corrected (x scaled by width/height) so a
    horizontal and a vertical step of the same length compare fairly.
    """
    if region.w <= 0.0 or region.h <= 0.0 or max_t <= 0.0:
        return None
    if region.x <= ox <= region.x + region.w and region.y <= oy <= region.y + region.h:
        return 0.0

    t_near = 0.0
    t_far = float("inf")
    for origin, direction, lo, hi in (
        (ox, dx, region.x, region.x + region.w),
        (oy, dy, region.y, region.y + region.h),
    ):
        if abs(direction) < 1e-8:
            if origin < lo or origin > hi:
                return None
            continue
        inv = 1.0 / direction
        a = (lo - origin) * inv
        b = (hi - origin) * inv
        if a > b:
            a, b = b, a
        t_near = a if a > t_near else t_near
        t_far = b if b < t_far else t_far
        if t_near > t_far:
            return None
    if t_far < 0.0 or t_near > max_t:
        return None
    return t_near if t_near >= 0.0 else 0.0


def resolve_aim(
    landmarks,
    regions: Sequence[TargetRegion],
    aspect: float = 1.0,
    max_ray: float = 0.62,
) -> AimHit:
    """Cast a 2D ray that starts at the index tip and follows knuckle → tip.

    `aspect` is frame width / height. The ray only travels forward from the
    tip, so a region behind the hand is not a target.
    """
    empty = AimHit(None, False, (), None, 0.0, 0.0)
    mcp = _point(landmarks, _INDEX_MCP)
    tip = _point(landmarks, _INDEX_TIP)
    if mcp is None or tip is None or not regions:
        return empty

    aspect = aspect if aspect > 0.05 else 1.0
    sx = (tip[0] - mcp[0]) * aspect
    sy = tip[1] - mcp[1]
    length = math.hypot(sx, sy)
    if length < _MIN_DIRECTION:
        return empty

    ux = sx / length
    uy = sy / length
    ox = tip[0] * aspect
    oy = tip[1]

    def to_image(distance: float) -> Tuple[float, float]:
        return (ox + ux * distance) / aspect, oy + uy * distance

    hits = []
    for region in regions:
        # Region x is scaled into the same space as the ray.
        scaled = TargetRegion(
            region.id,
            region.name,
            region.x * aspect,
            region.y,
            region.w * aspect,
            region.h,
        )
        distance = _ray_t(ox, oy, ux, uy, scaled, max_ray)
        if distance is not None:
            hits.append((distance, region))

    end = max_ray
    if hits:
        hits.sort(key=lambda item: item[0])
        end = hits[0][0] if hits[0][0] > 0.0 else min(max_ray, 0.08)
    x1, y1 = to_image(end)
    ray = (tip[0], tip[1], x1, y1)

    if not hits:
        return AimHit(None, False, (), ray, tip[0], tip[1])

    nearest, winner = hits[0]
    if len(hits) > 1 and (hits[1][0] - nearest) < _AMBIGUOUS_GAP:
        return AimHit(
            None,
            True,
            (hits[0][1], hits[1][1]),
            ray,
            tip[0],
            tip[1],
        )
    hx, hy = to_image(nearest)
    return AimHit(winner, False, (winner,), ray, hx, hy)


class TargetSelector:
    """
    Stable point, then a pinch, selects one calibrated region.

    `TargetFrame.selection` is the only output. Nothing is executed.
    """

    def __init__(
        self,
        stable_s: float = 0.40,
        confirm_cooldown_s: float = 0.85,
        flash_s: float = 0.55,
        max_ray: float = 0.62,
    ):
        self.stable_s = float(stable_s)
        self.confirm_cooldown_s = float(confirm_cooldown_s)
        self.flash_s = float(flash_s)
        self.max_ray = float(max_ray)

        self.regions: list[TargetRegion] = []
        self.latest: Optional[TargetSelection] = None
        self._seq = 1
        self._aim_id: Optional[str] = None
        self._aim_since: Optional[float] = None
        self._locked_id: Optional[str] = None
        self._hit = (0.0, 0.0)
        self._cooldown_until = 0.0
        self._flash_until = 0.0
        self._flash_id: Optional[str] = None

    def add_region(
        self,
        name: str,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> Optional[TargetRegion]:
        if len(self.regions) >= MAX_TARGETS:
            return None
        box = normalize_region(x, y, x + w, y + h)
        if box is None:
            return None
        label = " ".join((name or "").split())
        if not label:
            label = f"Target {len(self.regions) + 1}"
        label = label[:18]
        region = TargetRegion(f"t{self._seq}", label, *box)
        self._seq += 1
        self.regions.append(region)
        return region

    def clear(self) -> None:
        self.regions.clear()
        self.latest = None
        self._reset_aim()
        self._cooldown_until = 0.0
        self._flash_until = 0.0
        self._flash_id = None

    def _reset_aim(self) -> None:
        self._aim_id = None
        self._aim_since = None
        self._locked_id = None

    def _by_id(self, target_id: Optional[str]) -> Optional[TargetRegion]:
        if not target_id:
            return None
        for region in self.regions:
            if region.id == target_id:
                return region
        return None

    def _pack(
        self,
        state: str,
        region: Optional[TargetRegion],
        ray,
        selection: Optional[TargetSelection],
        contenders: Sequence[TargetRegion] = (),
    ) -> TargetFrame:
        return TargetFrame(
            mode=AIM_MODE,
            state=state,
            aimed_id=None if region is None else region.id,
            aimed_name="" if region is None else region.name,
            contenders=tuple(item.id for item in contenders),
            regions=tuple(self.regions),
            ray=ray,
            last_selection=self.latest,
            selection=selection,
        )

    def update(
        self,
        landmarks,
        now: float,
        pointing_stable: bool = False,
        pointing_raw: bool = False,
        pinch_started: bool = False,
        aspect: float = 1.0,
    ) -> TargetFrame:
        now = float(now)
        if not self.regions:
            self._reset_aim()
            return self._pack(STATE_CALIBRATE, None, None, None)

        if landmarks is None:
            self._reset_aim()
            self._flash_until = 0.0
            self._flash_id = None
            return self._pack(STATE_LOST, None, None, None)

        if pinch_started:
            locked = self._by_id(self._locked_id)
            if locked is not None and now >= self._cooldown_until:
                selection = TargetSelection(
                    target_id=locked.id,
                    name=locked.name,
                    timestamp=now,
                    aim_x=self._hit[0],
                    aim_y=self._hit[1],
                )
                self.latest = selection
                self._cooldown_until = now + self.confirm_cooldown_s
                self._flash_until = now + self.flash_s
                self._flash_id = locked.id
                self._reset_aim()
                return self._pack(STATE_CONFIRMED, locked, None, selection)
            self._reset_aim()
            return self._pack(STATE_IDLE, None, None, None)

        if self._flash_until and now < self._flash_until:
            flashed = self._by_id(self._flash_id)
            return self._pack(STATE_CONFIRMED, flashed, None, None)

        if not pointing_raw and not pointing_stable:
            self._reset_aim()
            return self._pack(STATE_IDLE, None, None, None)

        hit = resolve_aim(landmarks, self.regions, aspect=aspect, max_ray=self.max_ray)
        if hit.ambiguous:
            self._reset_aim()
            return self._pack(
                STATE_AMBIGUOUS, None, hit.ray, None, hit.contenders
            )
        if hit.region is None:
            self._reset_aim()
            return self._pack(STATE_AIMING, None, hit.ray, None)

        self._hit = (hit.aim_x, hit.aim_y)
        if hit.region.id != self._aim_id:
            self._aim_id = hit.region.id
            self._aim_since = now
            self._locked_id = None
            return self._pack(STATE_AIMING, hit.region, hit.ray, None)

        held = (
            pointing_stable
            and self._aim_since is not None
            and (now - self._aim_since) >= self.stable_s
        )
        if held:
            self._locked_id = hit.region.id
            return self._pack(STATE_LOCKED, hit.region, hit.ray, None)
        return self._pack(STATE_AIMING, hit.region, hit.ray, None)
