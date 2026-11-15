"""Automatic white balance and the neutral eyedropper (plan 9.2).

Auto: the grey-edge hypothesis (van de Weijer, Gevers, Gijsenij, IEEE TIP 16(9), 2007): the average colour of
the image's *edges* (not of the image itself) is grey under a neutral light. It is measured on the raw camera
colours (no white balance applied), with clipped and very dark pixels left out, as a Minkowski p-norm of the
gradients of each channel. It is far less fooled by large single-colour areas (a green lawn, a red wall) than
the grey-world average.

Honest limit: candle and tungsten scenes have a warm cast on purpose; Auto will neutralise it. That is why it is
a button and never a default.

Eyedropper: the clicked spot's raw camera colour is made neutral (average of a small patch).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from ... import log as slog

log = slog.get("engine.autowb")


def camera_rgb_half(image) -> np.ndarray:
    """Half-size raw camera RGB of the camera-crop frame (black-subtracted, 1 = clip, no white balance)."""
    from ..raw import demosaic as dm
    from ..raw import linearize as lin
    raw = image.raw
    lv = image.levels(True)
    cx, cy, cw, ch = raw.crop
    mos = lin.normalise(raw, lv, (cx, cy, cw // 2 * 2, ch // 2 * 2))
    return dm.demosaic(mos, raw.cfa_rgb, "half")


def grey_edge(cam: np.ndarray, p: float = 6.0, sigma: float = 1.0) -> np.ndarray | None:
    """Illuminant colour (R, G, B, sum 1) from a raw camera RGB image, or None if it cannot be told."""
    ok = (cam.max(axis=2) < 0.95) & (cam.min(axis=2) > 0.002)
    ok = gaussian_filter(ok.astype(np.float32), 2.0) > 0.99          # keep away from clipped / black areas
    if ok.mean() < 0.05:
        return None
    e = []
    for c in range(3):
        f = gaussian_filter(cam[..., c], sigma)
        gy, gx = np.gradient(f)
        g = np.hypot(gx, gy)[ok]
        if g.size < 1000:
            return None
        e.append(float(np.mean(g ** p) ** (1.0 / p)))
    e = np.asarray(e)
    if not np.all(e > 0):
        return None
    return e / e.sum()


def auto_multipliers(image) -> tuple[np.ndarray | None, str]:
    cam = camera_rgb_half(image)
    e = grey_edge(cam)
    if e is None:
        mean = cam.reshape(-1, 3)
        mean = mean[(mean.max(1) < 0.95) & (mean.min(1) > 0.002)]
        if mean.size < 300:
            return None, "not enough usable pixels"
        e = mean.mean(0)
        how = "grey world (too few edges)"
    else:
        how = "grey edge"
    m = (1.0 / e) / (1.0 / e).min()
    log.info("auto white balance (%s): multipliers %s", how, np.round(m, 4).tolist())
    return m, how


def patch_multipliers(image, xs: float, ys: float, radius: int = 3) -> np.ndarray | None:
    """Multipliers that make the raw colour around S-frame point (xs, ys) (full-res pixels) neutral."""
    cam = camera_rgb_half(image)
    hx, hy = int(round((xs - 0.5) / 2.0)), int(round((ys - 0.5) / 2.0))
    a, b = max(0, hy - radius), min(cam.shape[0], hy + radius + 1)
    c, d = max(0, hx - radius), min(cam.shape[1], hx + radius + 1)
    patch = cam[a:b, c:d].reshape(-1, 3)
    patch = patch[(patch.max(1) < 0.95)]
    if patch.size == 0 or patch.mean(0).min() <= 1e-5:
        return None
    m = patch.mean(0)
    m = (1.0 / m) / (1.0 / m).min()
    return m
