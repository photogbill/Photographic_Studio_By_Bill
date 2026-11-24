"""The speed gate (BUILD_PLAN_PAINT 0.5): can the CPU brush keep up with the pen?

A 2,000-dab stroke at 60 px is painted the way a fast pen delivers it (233 reports a second, 2,000 px/s), and the
time for each pen event is measured: the dabs that event adds plus compositing the changed area for the screen.
The gate passes when 95 % of events take under 16 ms (one screen frame).
"""
from __future__ import annotations

import math
import time

import numpy as np

from .brush import BrushSettings, Stroke
from .curves import Curve
from .smoothing import Sample
from .tiles import Document


def run(size: float = 60, dabs: int = 2000, mode: str = "wash", hardness: float = 0.7,
        spacing: float = 0.1, rate_hz: float = 233, speed_px_s: float = 2000) -> dict:
    doc = Document(4000, 4000)
    b = BrushSettings("bench", size=size, hardness=hardness, spacing=spacing, mode=mode, flow=0.7,
                      dynamics={"size": {"pressure": Curve()}, "opacity": {}, "flow": {}})
    st = Stroke(doc.layer, b, (0.2, 0.3, 0.8))
    step = speed_px_s / rate_hz
    gap = size * spacing
    n_events = int(math.ceil(dabs * gap / step)) + 1
    times = []
    cx, cy, R = 2000.0, 2000.0, 1500.0
    for i in range(n_events):
        ang = (i * step) / R
        s = Sample(cx + R * math.cos(ang), cy + R * math.sin(ang), 0.6 + 0.4 * math.sin(i / 20), i / rate_hz)
        t0 = time.perf_counter()
        d = st.add(s)
        if d is not None:
            x0, y0, x1, y1 = d
            doc.composite(x0, y0, x1 - x0, y1 - y0)
        times.append((time.perf_counter() - t0) * 1000)
    st.finish()
    a = np.array(times)
    p95 = float(np.percentile(a, 95))
    return {"size": size, "mode": mode, "dabs": st.dabs, "events": n_events, "median_ms": float(np.median(a)),
            "p95_ms": p95, "max_ms": float(a.max()), "ms_per_dab": float(a.sum() / max(1, st.dabs)),
            "passes": p95 < 16.0}
