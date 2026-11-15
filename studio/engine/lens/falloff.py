"""Lens falloff (vignetting) correction, part of stage 13. OFF by default: Bill likes vignetting (plan 0.4).

Model (lensfun's "pa", the Adobe camera model): recorded = true * (1 + k1 r^2 + k2 r^4 + k3 r^6), r = distance from
the centre / (half-diagonal * norm). The correction divides by that, scaled by ``amount`` in stops (1 = remove all
of it, 0.5 = remove half).

Sources, colour-coded in the UI: the lens wizard (cyan, Phase 7), lensfun (violet), or estimated from the photo
itself (amber). The estimate follows the radial-gradient-symmetry idea of Zheng, Yu, Kang, Lin and Kambhamettu
(CVPR 2008, "Single-image vignetting correction using radial gradient symmetry"): in a natural scene, edges that
get brighter toward the corners are as common as edges that get darker, so the median radial gradient of
log-brightness should be zero at every radius; whatever the median shows is the lens's falloff. It cannot tell
a lens's falloff from light that really falls off in the scene (a candle-lit face, a spotlight), which is one
more reason correction stays off unless the user asks for it.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from ... import log as slog

log = slog.get("engine.falloff")


def model_falloff(model: dict | None, r: np.ndarray) -> np.ndarray:
    """Relative brightness the lens recorded at normalised radius r (1 at the centre)."""
    if not model:
        return np.ones_like(r)
    k = [float(v) for v in model.get("terms", [])] + [0.0, 0.0, 0.0]
    r2 = r * r
    return 1.0 + k[0] * r2 + k[1] * r2 * r2 + k[2] * r2 * r2 * r2


def gain_map(model: dict | None, amount: float, h: int, w: int, origin_xy, full_wh, step: float = 1.0) -> np.ndarray:
    """Per-pixel correction gain for a (region of a) frame. origin_xy: full-res position of pixel (0, 0) of the
    region relative to the frame's top-left; full_wh: full-res frame size; step: full-res pixels per pixel."""
    if not model or abs(amount) < 1e-6:
        return np.ones((h, w), np.float32)
    fw, fh = full_wh
    cx, cy = (fw - 1) / 2.0, (fh - 1) / 2.0
    rn = math.hypot(fw, fh) / 2.0 * float(model.get("norm", 1.0))
    ys = origin_xy[1] + (np.arange(h) + 0.5) * step - 0.5 - cy
    xs = origin_xy[0] + (np.arange(w) + 0.5) * step - 0.5 - cx
    r = np.hypot(ys[:, None], xs[None, :]) / rn
    v = np.maximum(model_falloff(model, r), 0.05)
    return np.exp2(-float(amount) * np.log2(v)).astype(np.float32)


def corner_ev(model: dict | None) -> float:
    """Falloff at the frame corner in stops (negative = darker corners)."""
    if not model:
        return 0.0
    r = 1.0 / float(model.get("norm", 1.0))
    return float(np.log2(max(model_falloff(model, np.array(r)), 1e-3)))


def estimate(Y: np.ndarray, min_samples: int = 20000) -> dict | None:
    """Estimate falloff from a (reduced) linear luminance image of the whole frame. None if it cannot be told."""
    h, w = Y.shape
    if min(h, w) < 200:
        return None
    Z = np.log2(np.maximum(gaussian_filter(Y, 1.0), 1e-6))
    gy, gx = np.gradient(Z)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    dy, dx = yy - cy, xx - cx
    rpx = np.hypot(dy, dx)
    half = math.hypot(h, w) / 2.0
    gr = (gx * dx + gy * dy) / np.maximum(rpx, 1e-6)          # radial derivative, stops per pixel
    usable = (Y > 0.003) & (Y < 0.9) & (rpx > 0.05 * half) & (np.hypot(gx, gy) < 0.08)
    rr = rpx / half
    if usable.sum() < min_samples:
        return None
    bins = np.linspace(0.05, 1.0, 20)
    rc, slope, wts = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = usable & (rr >= a) & (rr < b)
        n = int(m.sum())
        if n < 500:
            continue
        rc.append(0.5 * (a + b))
        slope.append(float(np.median(gr[m])) * half)           # stops per unit normalised radius
        wts.append(math.sqrt(n))
    if len(rc) < 8:
        return None
    rc, slope, wts = np.array(rc), np.array(slope), np.array(wts)
    # d log2 V / dr for log2 V = c1 r^2 + c2 r^4 is 2 c1 r + 4 c2 r^3: weighted least squares
    A = np.stack([2 * rc, 4 * rc ** 3], 1) * wts[:, None]
    c, *_ = np.linalg.lstsq(A, slope * wts, rcond=None)
    resid = slope - np.stack([2 * rc, 4 * rc ** 3], 1) @ c
    corner = float(c[0] + c[1])                                  # log2 V at r = 1
    noise = float(np.sqrt(np.sum((resid * wts) ** 2) / np.sum(wts ** 2)))
    if not (-3.0 < corner < 0.05):                              # lenses darken corners; brighter = scene, not lens
        log.debug("falloff estimate rejected: corner %.2f EV", corner)
        return None
    if abs(corner) < 0.1:
        return None                                              # nothing worth correcting
    # convert log2 V = c1 r^2 + c2 r^4 to the pa polynomial by least squares on V
    r = np.linspace(0, 1, 50)
    V = np.exp2(c[0] * r ** 2 + c[1] * r ** 4)
    B = np.stack([r ** 2, r ** 4, r ** 6], 1)
    k, *_ = np.linalg.lstsq(B, V - 1.0, rcond=None)
    conf = "fair" if noise < 0.15 else "low"
    log.debug("falloff estimate: corner %.2f EV, residual %.3f (%s)", corner, noise, conf)
    return {"model": "pa", "terms": [round(float(v), 6) for v in k], "norm": 1.0,
            "source": "estimated from this photo", "corner_ev": round(corner, 2), "confidence": conf}
