"""Relight: move the light, from the shape of the scene. A Cognitive Adjustment.

    "relight": {"amount": 0..100, "azimuth": -180..180, "elevation": 0..90, "soft": 0..100}

Photoshop's Smart Portrait has a light-direction slider for faces, made by a generative network. Here the light is
moved with a model you can name: the surfaces come from the depth map (local/depth.py: normals from its smoothed
gradients), the light already on the photo is measured from how brightness follows those surfaces (first-order
spherical harmonics, Ramamoorthi and Hanrahan 2001; Basri and Jacobs 2003: a Lambertian scene's shading is close to
a + b . n), and each place is brightened or darkened by the ratio of the new light's shading to the measured one.
So it is a change of light on what is there: no detail, skin or feature is drawn, and texture is untouched (the gain
is smooth). The surfaces are guesses from one photo; at Amount 100 a light from the other side can look painted
on a flat face, which is why Amount starts gentle.

azimuth: where the light comes from, in the photo's plane: 0 = from the right, 90 = from the top, 180 / -180 = from
the left, -90 = from below. elevation: 0 = grazing from that side, 90 = from the camera. soft: how much of the
light is ambient (fills the shadow side).
No GUI imports (rule 2.2).
"""
from __future__ import annotations

import math

import numpy as np

DEFAULTS = {"amount": 0.0, "azimuth": 135.0, "elevation": 35.0, "soft": 40.0}
MAX_RATIO = 3.0


def settings(d: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update({k: float(v) for k, v in (d or {}).items() if k in DEFAULTS})
    return out


def active(d: dict | None) -> bool:
    return settings(d)["amount"] > 1e-6


def normals(D: np.ndarray) -> np.ndarray:
    """Surface normals (h, w, 3; z toward the camera) from a relative depth map (near = 1): smoothed keeping its
    steps, the scene's overall slope taken out so only the shapes of things count, scaled so steep parts tilt
    about 45 degrees; outlines (steps in depth) are left facing the camera."""
    from scipy.ndimage import binary_dilation, gaussian_filter
    from ..detail.finish import guided
    h, w = D.shape
    d = D.astype(np.float32)
    Ds = guided(d, d, max(2, int(round(0.012 * min(h, w)))), 1e-4).astype(np.float64)   # smooth, keeping steps
    gy, gx = np.gradient(Ds - gaussian_filter(Ds, 0.25 * min(h, w)))    # shapes only: the scene's overall tilt
    gy0, gx0 = np.gradient(Ds)                                          # (a floor, a wall) would light everything
    mag = np.hypot(gx0, gy0)
    p90 = max(float(np.percentile(np.hypot(gx, gy), 90)), 1e-9)
    k = 1.0 / p90
    n = np.stack([-gx * k, -gy * k, np.ones_like(gx)], -1)      # nearer = 1, so a slope toward the camera faces it
    step = binary_dilation(mag > 4.0 * p90, iterations=max(1, int(round(0.003 * min(h, w)))))
    n[step] = (0.0, 0.0, 1.0)                    # an outline (a step in depth) is not a surface: left as it is
    return (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32)


def light_vector(azimuth: float, elevation: float) -> np.ndarray:
    a, e = math.radians(azimuth), math.radians(elevation)
    return np.array([math.cos(e) * math.cos(a), -math.cos(e) * math.sin(a), math.sin(e)])   # image y points down


def fit_light(Y: np.ndarray, N: np.ndarray) -> np.ndarray:
    """The light on the photo: brightness ~ c0 + c . n fitted over the mid-tones (robust, Huber-weighted); c is
    scaled down when the fit explains under 30% of the brightness."""
    y = np.asarray(Y, np.float64).reshape(-1)
    n = N.reshape(-1, 3).astype(np.float64)
    lo, hi = np.percentile(y, [5, 97])
    sel = (y > lo) & (y < hi)
    if sel.sum() < 100:
        return np.array([float(np.mean(y)), 0.0, 0.0, 0.0])
    nm = n[sel].mean(axis=0)                     # centred, so c0 is the mean and c only the light's direction
    X = np.column_stack([np.ones(int(sel.sum())), n[sel] - nm])
    t = y[sel]
    wts = np.ones(len(t))
    coef = np.zeros(4)
    for _ in range(4):
        sw = np.sqrt(wts)
        coef, *_ = np.linalg.lstsq(X * sw[:, None], t * sw, rcond=None)
        r = t - X @ coef
        s = 1.4826 * float(np.median(np.abs(r))) + 1e-9
        wts = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-12))
    # How much of the brightness the surfaces explain: in a photo where colour and pattern (albedo) rule, the fit
    # says little about the light, so it is trusted less (the new light is then simply added).
    r2 = 1.0 - float(np.sum(wts * (t - X @ coef) ** 2)) / max(float(np.sum(wts * (t - np.average(t, weights=wts)) ** 2)),
                                                              1e-12)
    coef = coef.copy()
    coef[1:] *= float(np.clip(r2 / 0.3, 0.0, 1.0))
    coef[0] -= float(coef[1:] @ nm)               # back to c0 + c . n
    return coef


def gain(N: np.ndarray, coef: np.ndarray, d: dict | None) -> np.ndarray:
    """The light change as a brightness factor per place (h, w), from normals, the measured light and the
    settings: the new light's shading over the measured one, raised to Amount, smoothed."""
    from scipy.ndimage import gaussian_filter
    s = settings(d)
    amb = 0.15 + 0.7 * s["soft"] / 100.0
    old = coef[0] + N @ coef[1:]
    old = old / max(float(np.percentile(old, 50)), 1e-6)
    lam = light_vector(s["azimuth"], s["elevation"])
    new = amb + (1.0 - amb) * np.maximum(N @ lam, 0.0)
    new = new / max(float(np.percentile(new, 50)), 1e-6)
    lr = np.log(np.clip(new / np.maximum(old, 0.2), 1.0 / MAX_RATIO, MAX_RATIO))
    lr -= float(np.median(lr))                   # the photo's overall brightness stays where it was
    g = np.exp(lr * s["amount"] / 100.0)
    h, w = g.shape
    return gaussian_filter(g, max(1.0, 0.004 * min(h, w))).astype(np.float32)


def direction_name(azimuth: float, elevation: float) -> str:
    if elevation >= 70:
        return "from the front"
    names = ["the right", "the top right", "the top", "the top left", "the left", "the bottom left", "below",
             "the bottom right"]
    return "from " + names[int(round((azimuth % 360) / 45.0)) % 8]
