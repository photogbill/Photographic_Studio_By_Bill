"""Input curves: how pen pressure (or speed, tilt ...) becomes brush size, opacity or flow.

A curve is a list of (input, output) points in 0..1, joined by a monotone cubic (Fritsch-Carlson), so it never
overshoots and a rising curve always rises. Presets cover the common hands: a light hand wants "Soft", a heavy hand
"Firm".
"""
from __future__ import annotations

import numpy as np

PRESETS = {
    "Linear": [(0, 0), (1, 1)],
    "Soft": [(0, 0), (0.25, 0.5), (1, 1)],           # light pressure already gives a lot (light hands)
    "Firm": [(0, 0), (0.6, 0.3), (1, 1)],            # needs a firm press (heavy hands, fine control at low end)
    "S-curve": [(0, 0), (0.3, 0.15), (0.7, 0.85), (1, 1)],
    "Constant": [(0, 1), (1, 1)],                    # ignore the input
}

PRESET_NOTES = {
    "Linear": ("pens that already feel right", "a light hand gets thin, faint lines"),
    "Soft": ("light hands, quick sketching, sore wrists", "hard to make very thin lines"),
    "Firm": ("heavy hands, fine control of thin lines", "tiring for long sessions"),
    "S-curve": ("a steady middle with gentle ends", "less control at the very ends"),
    "Constant": ("ignoring the input (a mouse, or a fixed size)", "no variation at all"),
}


class Curve:
    def __init__(self, points=None):
        pts = sorted((float(x), float(y)) for x, y in (points or PRESETS["Linear"]))
        if len(pts) < 2:
            pts = [(0.0, pts[0][1] if pts else 0.0), (1.0, pts[0][1] if pts else 1.0)]
        self.points = pts
        self._prep()

    def _prep(self):
        x = np.array([p[0] for p in self.points])
        y = np.array([p[1] for p in self.points])
        h = np.diff(x)
        h[h == 0] = 1e-9
        d = np.diff(y) / h
        m = np.zeros_like(x)
        m[0], m[-1] = d[0], d[-1]
        for i in range(1, len(x) - 1):
            m[i] = 0.0 if d[i - 1] * d[i] <= 0 else 3 * (h[i - 1] + h[i]) / (
                (2 * h[i] + h[i - 1]) / d[i - 1] + (h[i] + 2 * h[i - 1]) / d[i])
        self._x, self._y, self._m = x, y, m

    def __call__(self, v):
        v = np.clip(np.asarray(v, np.float64), 0, 1)
        x, y, m = self._x, self._y, self._m
        i = np.clip(np.searchsorted(x, v, side="right") - 1, 0, len(x) - 2)
        h = x[i + 1] - x[i]
        h = np.where(h == 0, 1e-9, h)
        t = (v - x[i]) / h
        t2, t3 = t * t, t * t * t
        out = ((2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * y[i + 1]
               + (t3 - t2) * h * m[i + 1])
        out = np.clip(out, 0, 1)
        return float(out) if out.ndim == 0 else out

    def to_list(self):
        return [list(p) for p in self.points]

    @classmethod
    def preset(cls, name: str) -> "Curve":
        return cls(PRESETS[name])
