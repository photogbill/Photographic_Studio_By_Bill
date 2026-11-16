"""Stains and corrosion on scans of old photos and plates (the Plate Lab): unmixed, not painted over.

    "stains": {"on": bool, "amount": 0..150 (%), "sensitivity": 0..100}

Why unmixing
* Foxing on a print, a water stain, the orange-brown corrosion blooms on a daguerreotype-type plate: each is a
  coloured layer lying over the picture. It changes two things, the colour (it is warmer than the picture round
  it) and the brightness. Painting over it (inpainting) throws away whatever of the picture still shows through;
  on a thin stain that is most of it.
* So the stain's colour is used to measure it. On a small copy of the whole photo, in log space:
    warm = log R - log B, gm = log G - (log R + log B) / 2, brightness L = mean of the three logs;
  the local background of warm, gm and L is their median over 8% of the short side (a stain spot is well under
  half that, the picture's own broad colours are not). Inside the stains only, how far brightness sits from its
  background per unit of stain colour is a robust least-squares fit (iteratively reweighted, Huber): the picture
  under the stains is noise to that fit, so a photo whose own colours follow its brightness does not bend it.
* Where a pixel is warmer than its background (a soft weight between two thresholds, Sensitivity moves them),
  the brightness the fit says the stain causes is taken out, and so is the stain's tint. Broad warm areas (skin,
  wood, a sepia tone) are not warmer than their own background, so they stay.
* Honest about loss: in the cores of strong blooms (far warmer than round them) the picture may be gone (on a
  plate the corrosion eats the silver the image is made of). Those places are kept in a map for the overlay
  ("Show what's lost") and are never filled with invented detail.

First tested on Levi L. Hill's Hillotype self-portrait (NMAH PG.003999.32, scan NMAH-AHB2009q10751), and on
synthetic stains laid over a clean portrait (add_stains below; tools/plate_lab_benchmark.py).
No GUI imports (rule 2.2).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

DEFAULTS = {"on": False, "amount": 100.0, "sensitivity": 50.0}
T0, T1 = 0.02, 0.10        # log R/B above the background where the stain weight starts, and its ramp width
LOST = 0.16                # log R/B above the background: a bloom core, where the picture may be gone
WINDOW = 0.08              # the colour background: a median over this share of the short side (> 2x a bloom)
EPS = 1e-4                 # scene-linear floor before the logs (deep shadow is not measured)


def settings(d: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (d or {}).items() if k in DEFAULTS})
    return out


def active(d: dict | None) -> bool:
    return bool(d and d.get("on"))


def _scale(sens: float) -> float:
    """Sensitivity 50 = the measured thresholds; 100 = four times as sensitive; 0 = a quarter."""
    return 2.0 ** ((50.0 - float(sens)) / 25.0)


def axes(rgb: np.ndarray):
    """Brightness and the two colour axes in log space: (L, warm, gm), float32."""
    lg = np.log(np.maximum(rgb.astype(np.float32), 0.0) + EPS)
    L = lg.mean(axis=2)
    warm = lg[..., 0] - lg[..., 2]
    gm = lg[..., 1] - 0.5 * (lg[..., 0] + lg[..., 2])
    return L, warm, gm


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def _local_median(x: np.ndarray, size: float) -> np.ndarray:
    """Median over a square of ``size`` pixels, worked out on a quarter-size copy (it is a slow, smooth field)."""
    h, w = x.shape
    k = 4 if min(h, w) >= 64 else 1
    small = x[: h // k * k, : w // k * k].reshape(h // k, k, w // k, k).mean(axis=(1, 3)) if k > 1 else x
    m = ndi.median_filter(small, size=_odd(size / k), mode="reflect")
    if k > 1:
        zy, zx = h / m.shape[0], w / m.shape[1]
        m = ndi.zoom(m, (zy, zx), order=1, mode="nearest")
        m = np.pad(m, ((0, max(0, h - m.shape[0])), (0, max(0, w - m.shape[1]))), mode="edge")[:h, :w]
    return m.astype(np.float32)


def _masked_blur(x: np.ndarray, m: np.ndarray, s: float) -> np.ndarray:
    num = ndi.gaussian_filter(np.where(m, x, 0.0).astype(np.float32), s)
    den = ndi.gaussian_filter(m.astype(np.float32), s)
    return num / np.maximum(den, 1e-4)


def measure(rgb: np.ndarray, sensitivity: float = 50.0) -> dict:
    """The whole-photo measurement on a small scene-linear copy (long side about 1024): the colour backgrounds,
    the brightness-per-colour fit and the lost map. Everything a block needs to be cleaned the same way at any
    size."""
    h, w = rgb.shape[:2]
    short = float(min(h, w))
    L, warm, gm = axes(rgb)
    Y = rgb.astype(np.float32).mean(axis=2)
    valid = (Y > 0.003) & (Y < 0.98)
    valid = ndi.binary_erosion(valid, iterations=max(1, int(0.01 * short)))
    if valid.sum() < 500:
        valid = np.ones_like(valid)
    win = WINDOW * short
    med_w = _local_median(warm, win)
    med_g = _local_median(gm, win)
    med_L = _local_median(L, win)
    f = _scale(sensitivity)
    dw, dg = warm - med_w, gm - med_g
    # the stains' own colour direction: the weighted median direction of the strongest warm steps, searched only
    # where stains live (orange to yellow-brown, -20 to +50 degrees in the warm / green-magenta plane). A picture's
    # own reds lean magenta (below -35 degrees) and are kept out of both the search and stain_weight's cone.
    th = np.arctan2(dg, dw)
    lo, hi = math.radians(-20.0), math.radians(50.0)
    strong = valid & (np.hypot(dw, dg) > LOST * f) & (th > lo) & (th < hi)
    angle = math.radians(10.0)
    if strong.sum() >= 50:
        a_, w_ = th[strong], np.hypot(dw, dg)[strong]
        o = np.argsort(a_)
        cw = np.cumsum(w_[o])
        angle = float(a_[o][np.searchsorted(cw, 0.5 * cw[-1])])
    swt, p = stain_weight(dw, dg, angle, f)
    sel = valid & (swt > 0.0) & (p < 2.5 * LOST * f)     # opaque cores (no picture left) would only bend the fit
    # the fit: over all the stain pixels together, how far brightness sits from its own background per unit of
    # stain colour (with an intercept). The picture under the stains is noise to this fit, not signal: across many
    # stains its brightness steps average out, while the stain's own step follows its colour step every time.
    X = np.stack([p[sel], np.ones(int(sel.sum()))], -1)
    y = (L - med_L)[sel]
    coef = np.zeros(2, np.float64)
    if X.shape[0] >= 200:
        if X.shape[0] > 200000:                    # a fixed, even subsample (the fit is the same, much faster)
            idx = np.linspace(0, X.shape[0] - 1, 200000).astype(np.int64)
            X, y = X[idx], y[idx]
        wts = np.ones_like(y)
        for _ in range(8):
            coef = np.linalg.lstsq(X * wts[:, None], y * wts, rcond=None)[0]
            r = y - X @ coef
            sc = 1.4826 * float(np.median(np.abs(r))) + 1e-6
            wts = np.sqrt(np.minimum(1.0, 1.5 * sc / np.maximum(np.abs(r), 1e-9)))
    k = float(np.clip(coef[0], -3.0, 3.0))
    core = (p > LOST * f) & (swt > 0) & valid
    core = ndi.binary_opening(core, iterations=1)
    core = ndi.binary_dilation(core, iterations=1)
    return {"med_w": med_w, "med_g": med_g, "coef": [k, 0.0], "angle": angle,
            "lost": core.astype(np.uint8) * 255, "stained": float((swt[valid] > 0.5).mean()),
            "lost_frac": float(core[valid].mean()), "shape": (h, w)}


def stain_weight(dw: np.ndarray, dg: np.ndarray, angle: float, f: float):
    """How much a pixel is stain: its colour step from the background along the stains' own colour direction
    (``angle`` in the warm / green-magenta plane, measured on the photo), ramped between the thresholds, and only
    where the step points that way (within 40 degrees). A real red that leans magenta (lips, a Hillotype's reds)
    points elsewhere and is left alone. Returns (weight 0..1, step along the direction)."""
    ca, sa = math.cos(angle), math.sin(angle)
    p = dw * ca + dg * sa
    r = np.hypot(dw, dg) + 1e-6
    cosang = p / r
    lim = math.cos(math.radians(40.0))
    ang = np.clip((cosang - lim) / (1.0 - lim), 0.0, 1.0)
    return np.clip((p - T0 * f) / (T1 * f), 0.0, 1.0) * ang, p


def clean(rgb: np.ndarray, med_w: np.ndarray, med_g: np.ndarray, coef, d: dict | None,
          step: int = 1, stats: dict | None = None, angle: float = 0.0) -> np.ndarray:
    """Take the stains out of a scene-linear block. ``med_w``/``med_g``: the colour backgrounds sampled at the
    block's pixels; ``coef``: [brightness per unit of stain colour, unused]; ``angle``: the stains' colour
    direction (measure)."""
    st = settings(d)
    amt = float(st["amount"]) / 100.0
    if amt <= 0:
        return rgb
    f = _scale(st["sensitivity"])
    L, warm, gm = axes(rgb)
    dw, dg = warm - med_w, gm - med_g
    sw, p = stain_weight(dw, dg, angle, f)
    sig = max(0.4, 1.0 / max(1, step))
    sw = ndi.gaussian_filter(sw, sig) * amt
    bright = -sw * float(coef[0]) * p                               # the stain's brightness, taken out
    # the stain's tint taken out too, keeping the mean log (the brightness term above is the only one that moves it)
    dR = sw * (dg / 3.0 - dw / 2.0)
    dG = -sw * (2.0 * dg / 3.0)
    dB = sw * (dg / 3.0 + dw / 2.0)
    gain = np.stack([dR, dG, dB], -1) + bright[..., None]
    out = np.maximum(rgb.astype(np.float32), 0.0) + EPS
    out = out * np.exp(np.clip(gain, -4.0, 4.0)) - EPS
    if stats is not None:
        stats["stained"] = float((sw > 0.5 * max(amt, 1e-6)).mean())
    return np.maximum(out, 0.0).astype(np.float32)


# ------------------------------------------------------------------------------------------ test material
def add_stains(rgb: np.ndarray, n: int = 40, seed: int = 0, strength: float = 1.0):
    """Lay synthetic warm stains over a clean scene-linear image, as corrosion blooms and foxing do: round,
    soft-edged, each tinting orange-brown and changing brightness by a stain-specific amount; the strongest
    have opaque cores (the picture gone). Returns (stained, weight 0..1, core mask)."""
    rng = np.random.default_rng(seed)
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wt = np.zeros((h, w), np.float32)
    core = np.zeros((h, w), bool)
    for _ in range(n):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        r = rng.uniform(0.006, 0.02) * min(h, w)
        d = np.hypot(yy - cy, xx - cx) / r
        a = np.clip(1.25 - d, 0, 1) ** 0.7 * rng.uniform(0.5, 1.0)
        wt = np.maximum(wt, a)
        if rng.uniform() < 0.25:
            core |= d < 0.35
    k = strength
    tint = np.array([1.0 + 0.14 * k, 1.0 + 0.01 * k, 1.0 - 0.12 * k], np.float32)     # orange-brown in linear
    bright = 1.0 + 0.20 * k                                                            # blooms read lighter
    gain = 1.0 + wt[..., None] * (tint * bright - 1.0)
    out = np.maximum(rgb.astype(np.float32), 0.0) * gain
    if core.any():                                                                      # opaque cores: no picture
        out[core] = np.array([0.30, 0.17, 0.08], np.float32)
    return out.astype(np.float32), wt, core


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Peak signal to noise ratio (dB) of two images in 0..1 (display values), inside ``mask``."""
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if mask is not None:
        d = d[mask]
    mse = float(d.mean()) if d.size else 0.0
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
