"""Where Develop's photo sits on screen at any zoom: pure geometry, no widgets (so it is tested on its own).

A friend's first test (2026-11-15): "the denoising was efficient, but zooming and dragging is very clunky". The old
view had two states, the whole photo or a rendered 100% window: the wheel jumped straight from one to the other, and
every drag ended with the picture snapping back for the 2 s the next window took to render (with Clean on, on an R5
file). Now the zoom is continuous and the photo moves at once: the whole-photo preview already in memory is drawn
scaled, and a sharp render of just the part on screen (``tile_request``) is laid over it when it is ready.

Units: ``z`` is device pixels per output pixel of the finished photo (1.0 = 100%, one photo pixel per screen pixel
on any display scaling); ``None`` means "fit the whole photo". The centre (cu, cv) is the point of the photo at the
middle of the view, as fractions of the photo. Positions on screen are Qt logical pixels.
"""
from __future__ import annotations

import math

LEVELS = (0.25, 1 / 3, 0.5, 2 / 3, 1.0, 2.0, 3.0, 4.0)   # the zoom steps of the keys and the menu
MAX_ZOOM = 4.0
MARGIN = 18.0                                          # logical pixels round the photo when it is fitted
WHEEL_NOTCH = 120.0                                    # one click of a mouse wheel (Qt's angleDelta)
NOTCH_FACTOR = 2.0 ** 0.5                              # two clicks double the zoom


def label(z: float | None) -> str:
    """How a zoom is written on the button and the chip: Fit, 33%, 100%..."""
    if z is None:
        return "Fit"
    p = z * 100.0
    return f"{p:.0f}%" if p >= 10 else f"{p:.1f}%"


class ZoomState:
    def __init__(self):
        self.ow, self.oh = 0, 0                        # the finished photo's size in output pixels
        self.vw, self.vh = 0.0, 0.0                    # the view's size, logical pixels
        self.dpr = 1.0                                 # device pixels per logical pixel
        self.z: float | None = None
        self.cu, self.cv = 0.5, 0.5

    # ------------------------------------------------------------------------------------ setup
    def set_photo(self, ow: int, oh: int) -> None:
        self.ow, self.oh = int(ow), int(oh)
        self.clamp()

    def set_view(self, vw: float, vh: float, dpr: float = 1.0) -> None:
        self.vw, self.vh, self.dpr = float(vw), float(vh), float(dpr or 1.0)
        if self.z is not None and self.z <= self.fit_z() * 1.0001:
            self.z = None                              # the window grew past this zoom: back to fit
        self.clamp()

    @property
    def ready(self) -> bool:
        return self.ow > 0 and self.oh > 0 and self.vw > 2 * MARGIN and self.vh > 2 * MARGIN

    def fit_z(self) -> float:
        if not self.ready:
            return 1.0
        return min((self.vw - 2 * MARGIN) / self.ow, (self.vh - 2 * MARGIN) / self.oh) * self.dpr

    def current(self) -> float:
        """The zoom now, fit included."""
        return self.fit_z() if self.z is None else self.z

    def max_z(self) -> float:
        return max(MAX_ZOOM, self.fit_z())

    # ------------------------------------------------------------------------------------ where the photo is
    def photo_rect(self) -> tuple[float, float, float, float]:
        """(left, top, width, height) of the whole photo in logical pixels; at a zoom, mostly outside the view."""
        z = self.current()
        pw, ph = self.ow * z / self.dpr, self.oh * z / self.dpr
        if self.z is None:
            return (self.vw - pw) / 2.0, (self.vh - ph) / 2.0, pw, ph
        return self.vw / 2.0 - self.cu * pw, self.vh / 2.0 - self.cv * ph, pw, ph

    def uv_at(self, x: float, y: float) -> tuple[float, float]:
        l, t, w, h = self.photo_rect()
        return (x - l) / w, (y - t) / h

    def pos_of(self, u: float, v: float) -> tuple[float, float]:
        l, t, w, h = self.photo_rect()
        return l + u * w, t + v * h

    def clamp(self) -> None:
        """Keep the photo filling the view: no empty band past an edge once it is bigger than the view, and
        centred along an axis where it is smaller."""
        if self.z is None or not self.ready:
            self.cu, self.cv = 0.5, 0.5
            return
        _, _, pw, ph = self.photo_rect()
        for axis, (size, view) in enumerate(((pw, self.vw), (ph, self.vh))):
            c = self.cu if axis == 0 else self.cv
            if size <= view:
                c = 0.5
            else:
                half = view / 2.0 / size
                c = min(max(c, half), 1.0 - half)
            if axis == 0:
                self.cu = c
            else:
                self.cv = c

    # ------------------------------------------------------------------------------------ changing it
    def zoom_to(self, z: float | None, anchor: tuple[float, float] | None = None) -> None:
        """Zoom to ``z`` (None or anything at or below fit = fit) keeping the photo point under ``anchor`` (a view
        position, logical pixels) where it is. Without an anchor: the middle of the view."""
        if not self.ready:
            return
        ax, ay = anchor if anchor is not None else (self.vw / 2.0, self.vh / 2.0)
        u, v = self.uv_at(ax, ay)
        if z is None or z <= self.fit_z() * 1.0001:
            self.z = None
            self.clamp()
            return
        self.z = float(min(z, self.max_z()))
        pw, ph = self.ow * self.z / self.dpr, self.oh * self.z / self.dpr
        # the point (u, v) stays at (ax, ay): left = ax - u * pw, and left = vw / 2 - cu * pw
        self.cu = (self.vw / 2.0 - (ax - u * pw)) / pw
        self.cv = (self.vh / 2.0 - (ay - v * ph)) / ph
        self.clamp()

    def wheel(self, delta: float, anchor: tuple[float, float]) -> None:
        """A wheel (or touchpad) turn of ``delta`` (Qt angleDelta units): zoom around the pointer. Crossing 100%
        stops at exactly 100% once, so it is easy to land on."""
        if not self.ready or not delta:
            return
        z0 = self.current()
        z1 = z0 * NOTCH_FACTOR ** (delta / WHEEL_NOTCH)
        if (z0 < 1.0 < z1) or (z0 > 1.0 > z1):
            z1 = 1.0
        self.zoom_to(z1, anchor)

    def step(self, direction: int, anchor: tuple[float, float] | None = None) -> None:
        """One level in (+1) or out (-1): Fit, 25%, 33%, 50%, 67%, 100%, 200%, 300%, 400%."""
        z0, zf = self.current(), self.fit_z()
        levels = [lv for lv in LEVELS if lv > zf * 1.001]
        if direction > 0:
            nxt = [lv for lv in levels if lv > z0 * 1.001]
            if nxt:
                self.zoom_to(nxt[0], anchor)
        else:
            lower = [lv for lv in levels if lv < z0 * 0.999]
            self.zoom_to(lower[-1] if lower else None, anchor)

    def pan_by(self, dx: float, dy: float) -> None:
        """Move the photo by (dx, dy) logical pixels (a drag)."""
        if self.z is None:
            return
        _, _, pw, ph = self.photo_rect()
        self.cu -= dx / pw
        self.cv -= dy / ph
        self.clamp()

    def centre_on(self, u: float, v: float) -> None:
        if self.z is None:
            return
        self.cu, self.cv = float(u), float(v)
        self.clamp()

    # ------------------------------------------------------------------------------------ what to render
    def visible_window(self) -> tuple[float, float, float, float]:
        """The part of the photo on screen: (x, y, w, h) in output pixels."""
        l, t, pw, ph = self.photo_rect()
        k = self.ow / pw if pw else 0.0
        x0, y0 = max(0.0, -l * k), max(0.0, -t * k)
        x1, y1 = min(float(self.ow), (self.vw - l) * k), min(float(self.oh), (self.vh - t) * k)
        return x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)

    def tile_request(self, margin: float = 0.08, budget: float = 9.0e6):
        """(x, y, w, h, scale) of the sharp render for this view: what is on screen plus ``margin`` of the view on
        every side (small nudges then need no new render), in whole output pixels, rendered at ``scale`` = the
        zoom up to 100% (above 100% the pixels are enlarged on screen). The margin shrinks if the render would go
        past ``budget`` pixels. None at fit.

        Up to 60% the render is made at no more than 50%, which takes the engine's fast half-size path (with Clean
        on, an R5 view at 52% took 5.2 s at full size and about a fifth of that this way; enlarged by 1.2 at most,
        the difference does not show). Pixel-level judging is done at 100%, which is always full size."""
        if self.z is None or not self.ready:
            return None
        scale = self.z if self.z <= 0.5 else (0.5 if self.z <= 0.6 else min(self.z, 1.0))
        x, y, w, h = self.visible_window()
        if w <= 0 or h <= 0:
            return None
        m = margin
        while True:
            mx, my = w * m, h * m
            x0, y0 = max(0.0, x - mx), max(0.0, y - my)
            x1, y1 = min(float(self.ow), x + w + mx), min(float(self.oh), y + h + my)
            if (x1 - x0) * (y1 - y0) * scale * scale <= budget or m <= 0:
                break
            m = max(0.0, m - 0.02)
        xi, yi = int(math.floor(x0)), int(math.floor(y0))
        return xi, yi, int(math.ceil(x1)) - xi, int(math.ceil(y1)) - yi, scale


def covers(tile_uv, tile_scale: float, window, ow: int, oh: int, scale: float) -> bool:
    """Does a rendered tile ((u0, v0, u1, v1), rendered at tile_scale) hold everything in ``window`` (output px)
    at the wanted scale?"""
    if tile_uv is None or ow <= 0 or oh <= 0:
        return False
    if abs(tile_scale - scale) > 0.01 * max(scale, 1e-6):
        return False
    u0, v0, u1, v1 = tile_uv
    x, y, w, h = window
    eps = 1.0
    return (u0 * ow <= x + eps and v0 * oh <= y + eps and u1 * ow >= x + w - eps and v1 * oh >= y + h - eps)
