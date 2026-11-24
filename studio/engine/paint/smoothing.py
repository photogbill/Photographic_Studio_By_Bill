"""Stroke smoothing (BUILD_PLAN_PAINT P1, 0.9).

Pen samples go in, smoothed samples come out. Paint only ever draws where the pen has been: there is no prediction
of where it will go (Bill, 2026-11-23: no pen-tip prediction).

Modes (each says what it is best for and weak at, plan 23.1):
- none        every sample as it came
- basic       exponential average
- weighted    average of the last N samples
- stabilizer  the brush lags behind the pen by a distance and catches up when the pen stops (Krita's Stabilizer)
- dynamic     the brush has mass and drag, pulled towards the pen (Krita's Dynamic Brush)
- string      the brush is dragged on a string: it moves only when the pen pulls it tight (Lazy Nezumi style)
- adaptive    Speed-adaptive (the 1-euro filter, Casiez, Roussel and Vogel, CHI 2012): strong smoothing when the pen
              is slow (shaky hands), almost none when it is fast (no lag on quick strokes)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MODES = {
    "none": ("No smoothing", "fast, confident strokes; the truest feel of the pen", "shows every wobble of the hand"),
    "basic": ("Basic", "a light touch of smoothing for everyday sketching", "lags a little on fast strokes"),
    "weighted": ("Weighted", "steady average over the last few samples", "rounds off sharp corners"),
    "stabilizer": ("Stabilizer", "long smooth curves and clean line art (the brush trails the pen)",
                   "the line ends behind the pen until you pause; hard to place small marks"),
    "dynamic": ("Dynamic (mass and drag)", "flowing calligraphic curves with momentum",
                "overshoots tight turns; takes practice"),
    "string": ("Pulled string", "very clean curves and straight-ish lines with full control",
               "the brush only moves once the string is tight, so short marks need a longer string off"),
    "adaptive": ("Speed-adaptive", "steady slow strokes without lag on fast ones; good for shaky hands",
                 "very slow detail is smoothed strongly unless the strength is lowered"),
}


@dataclass
class Sample:
    x: float
    y: float
    pressure: float = 1.0
    t: float = 0.0               # seconds
    tilt_x: float = 0.0          # degrees, -60..60
    tilt_y: float = 0.0
    rotation: float = 0.0        # degrees (pens that report barrel rotation)
    speed: float = 0.0           # pixels per second (filled by the smoother)


class Smoother:
    """``feed(sample)`` returns the smoothed samples to draw now; ``finish()`` returns the tail (catch-up)."""

    def __init__(self, mode: str = "basic", strength: float = 0.5):
        if mode not in MODES:
            raise ValueError(f"unknown smoothing mode: {mode}")
        self.mode = mode
        self.strength = max(0.0, min(1.0, strength))
        self.reset()

    def reset(self):
        self.prev = None
        self.hist: list = []
        self.vx = self.vy = 0.0
        self.dx_f = self.dy_f = 0.0      # 1-euro derivative estimates
        self.last_pen = None

    def _with_speed(self, s: Sample, prev: Sample | None) -> Sample:
        if prev is not None and s.t > prev.t:
            s.speed = math.hypot(s.x - prev.x, s.y - prev.y) / (s.t - prev.t)
        elif prev is not None:
            s.speed = prev.speed
        return s

    def feed(self, s: Sample) -> list:
        m, k = self.mode, self.strength
        pen_prev, self.last_pen = self.last_pen, s
        if self.prev is None:
            self.prev = Sample(**vars(s))
            self.hist = [s]
            return [self.prev]
        p = self.prev
        if m == "none":
            out = Sample(**vars(s))
        elif m == "basic":
            a = 1 - 0.9 * k
            out = Sample(**{**vars(s), "x": p.x + a * (s.x - p.x), "y": p.y + a * (s.y - p.y)})
        elif m == "weighted":
            n = 2 + int(round(k * 18))
            self.hist = (self.hist + [s])[-n:]
            wsum = sum(range(1, len(self.hist) + 1))
            x = sum((i + 1) * q.x for i, q in enumerate(self.hist)) / wsum
            y = sum((i + 1) * q.y for i, q in enumerate(self.hist)) / wsum
            out = Sample(**{**vars(s), "x": x, "y": y})
        elif m == "stabilizer":
            lag = 4 + 60 * k
            d = math.hypot(s.x - p.x, s.y - p.y)
            if d <= lag:
                return []
            f = (d - lag) / d
            out = Sample(**{**vars(s), "x": p.x + f * (s.x - p.x), "y": p.y + f * (s.y - p.y)})
        elif m == "string":
            length = 5 + 80 * k
            d = math.hypot(s.x - p.x, s.y - p.y)
            if d <= length:
                return []
            f = (d - length) / d
            out = Sample(**{**vars(s), "x": p.x + f * (s.x - p.x), "y": p.y + f * (s.y - p.y)})
        elif m == "dynamic":
            dt = max(1e-3, min(0.05, s.t - p.t)) if s.t > p.t else 1 / 200
            mass = 0.02 + 0.5 * k
            drag = 12.0
            ax = ((s.x - p.x) * 400 - drag * self.vx) / (mass * 40)
            ay = ((s.y - p.y) * 400 - drag * self.vy) / (mass * 40)
            self.vx += ax * dt
            self.vy += ay * dt
            out = Sample(**{**vars(s), "x": p.x + self.vx * dt, "y": p.y + self.vy * dt})
        else:  # adaptive: the 1-euro filter on x and y
            dt = (s.t - pen_prev.t) if (pen_prev and s.t > pen_prev.t) else 1 / 200
            min_cut = 0.05 + 4.0 * (1 - k)          # Hz: lower = smoother at rest
            beta = 0.02
            d_cut = 1.0

            def alpha(cut):
                tau = 1 / (2 * math.pi * cut)
                return 1 / (1 + tau / dt)

            dx = (s.x - pen_prev.x) / dt if pen_prev else 0.0
            dy = (s.y - pen_prev.y) / dt if pen_prev else 0.0
            ad = alpha(d_cut)
            self.dx_f += ad * (dx - self.dx_f)
            self.dy_f += ad * (dy - self.dy_f)
            cut = min_cut + beta * math.hypot(self.dx_f, self.dy_f)
            a = alpha(cut)
            out = Sample(**{**vars(s), "x": p.x + a * (s.x - p.x), "y": p.y + a * (s.y - p.y)})
        out = self._with_speed(out, p)
        self.prev = out
        return [out]

    def finish(self) -> list:
        """At pen-up: modes that trail the pen catch up to where it stopped."""
        if self.prev is None or self.last_pen is None:
            return []
        if self.mode in ("stabilizer", "string", "basic", "weighted", "dynamic", "adaptive"):
            s, p = self.last_pen, self.prev
            if math.hypot(s.x - p.x, s.y - p.y) > 0.5:
                out = Sample(**{**vars(s), "pressure": p.pressure})
                self.prev = out
                return [out]
        return []
