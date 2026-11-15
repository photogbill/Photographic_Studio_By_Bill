"""Gamut compression: bring out-of-gamut and imaginary colours inside a target RGB gamut smoothly.

Why: a camera matrix can turn narrow-band light (violet LEDs, UV fluorescence, lasers, stage lights) into
colours outside the spectral locus, with negative channel values even in ProPhoto. Hard-clipping those
channels shifts hue abruptly and leaves flat patches. This compresses them instead. (Seen on Bill's
UV-lit African violet, 2I2A3558.CR2: the petals land at xy (0.20, -0.05), below the spectral locus.)

Method: the distance-based compression of the ACES Reference Gamut Compression (Academy, 2021).
Per pixel, with ach = max(r, g, b), each channel's distance from the achromatic axis is
d_c = (ach - c) / ach. Distances below a threshold are untouched (in-gamut colours stay exactly as they
were); distances above it are compressed with a smooth power curve so that the limit distance lands on
1.0 (the gamut boundary, channel = 0). Hue is largely preserved; saturation inside the threshold is
untouched. Parameters follow the published ACES RGC defaults.
"""
from __future__ import annotations

import numpy as np

# thresholds and limits for the (r, g, b) distances (ACES RGC 1.0 defaults: cyan, magenta, yellow)
THRESHOLD = np.array([0.815, 0.803, 0.880], np.float32)
LIMIT = np.array([1.147, 1.264, 1.312], np.float32)
POWER = 1.2


def _scale(thr, lim, p):
    return (lim - thr) / np.power(np.power((1.0 - thr) / (lim - thr), -p) - 1.0, 1.0 / p)


def compress_distance(d: np.ndarray, thr, lim, p: float = POWER) -> np.ndarray:
    s = _scale(thr, lim, p)
    x = np.maximum(d - thr, 0) / s
    out = thr + s * x / np.power(1.0 + np.power(x, p), 1.0 / p)
    return np.where(d < thr, d, out).astype(np.float32)


def compress(rgb: np.ndarray, threshold=THRESHOLD, limit=LIMIT, power: float = POWER) -> np.ndarray:
    """Return a gamut-compressed copy of linear RGB (HxWx3 or Nx3). Pixels whose distances are all inside
    the thresholds are returned bit-for-bit unchanged; only the others are computed (fast on normal photos)."""
    a = np.asarray(rgb, np.float32)
    shp = a.shape
    flat = a.reshape(-1, 3)
    out = flat.copy()
    ach = flat.max(axis=1)
    pos = ach > 1e-10
    # a distance beyond the smallest threshold means channel < (1 - thr_min) * ach
    lowest = flat.min(axis=1)
    need = pos & (lowest < (1.0 - float(np.min(threshold))) * ach)
    if not need.any():
        return out.reshape(shp)
    idx = np.nonzero(need)[0]
    sub = flat[idx]
    m = ach[idx][:, None]
    d = (m - sub) / m
    res = sub.copy()
    for c in range(3):
        sel = d[:, c] >= threshold[c]
        if sel.any():
            dc = compress_distance(d[sel, c], threshold[c], limit[c], power)
            res[sel, c] = m[sel, 0] - dc * m[sel, 0]
    out[idx] = res
    return out.reshape(shp)
