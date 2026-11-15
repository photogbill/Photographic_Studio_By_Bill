"""Stage 15, complete (plan 9.1): highlights, shadows, whites, blacks, tone curve, Smart lighting, Auto.

Global controls reshape the tone curve itself (so they are fast, exact and can never make halos):
  * shadows / highlights move the tones below / above mid grey along the curve, in stops (smooth bumps in
    log2 space: shadows act between about -10 and 0 EV, highlights between 0 EV and just past the white point)
  * whites moves the white point of the shoulder; blacks deepens (or, positive, lifts into a matte) the black
  * the point curves (master, then red / green / blue) work on display values the way photographers know them
    (0-255 style, sRGB-encoded), drawn through a monotone cubic so a curve never turns back on itself

Smart lighting is the local one: brighter shadows without flattening. An edge-aware base layer of log
luminance (guided filter, He, Sun and Tang, ECCV 2010 / IEEE TPAMI 35(6), 2013) carries the lighting; detail
is what is left. Only the base is lifted (shadows) and eased (highlights), so texture and microcontrast stay,
and strong edges stay sharp in the base, so there is no dark or bright rim.

Auto never changes exposure (Bill's rule: no automatic brightness changes): it only protects highlights that
would clip, sets the white point for flat, dull frames, and sets the black point. A low-key photo stays low-key
(named test).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import uniform_filter

from ... import log as slog

log = slog.get("engine.light")

LUMA = np.array([0.2880402, 0.7118741, 0.0000857], np.float32)    # linear ProPhoto Y weights


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def is_identity(light) -> bool:
    if light is None:
        return True
    return (abs(light.highlights) < 1e-6 and abs(light.shadows) < 1e-6 and abs(light.whites) < 1e-6
            and abs(light.blacks) < 1e-6 and not _has_curve(light.curve) and not _has_param(light))


PARAM_REGIONS = {"shadows": 0.125, "darks": 0.375, "lights": 0.625, "highlights": 0.875}   # centres, display


def _has_param(light) -> bool:
    p = getattr(light, "param", None) or {}
    return any(abs(float(p.get(k, 0))) > 1e-6 for k in PARAM_REGIONS)


def param_fn(param):
    """Parametric curve (four regions of the display range, -100..100 each) as display-linear -> display-linear.
    Bells of half-width 0.25 around each region's centre, tapered to keep pure black and white fixed, and scaled
    down if needed so the curve's slope never drops below 0.25 (always monotone)."""
    if not param:
        return None
    a = {k: float(param.get(k, 0)) / 100.0 for k in PARAM_REGIONS}
    if not any(abs(v) > 1e-6 for v in a.values()):
        return None

    def delta(e):
        d = np.zeros_like(e)
        for k, c in PARAM_REGIONS.items():
            if a[k]:
                u = np.clip((e - c + 0.25) / 0.5, 0.0, 1.0)
                d += 0.12 * a[k] * np.sin(np.pi * u) ** 2
        return d * 4.0 * e * (1.0 - e)
    g = np.linspace(0.0, 1.0, 4001)
    slope = np.gradient(g + delta(g), g)
    k = 1.0
    if slope.min() < 0.25:
        dmin = np.gradient(delta(g), g).min()
        k = (0.75 / -dmin) if dmin < 0 else 1.0

    def f(y):
        e = _enc(y)
        return _dec(np.clip(e + k * delta(e), 0.0, 1.0))
    return f


def _has_curve(curve) -> bool:
    if not curve:
        return False
    for pts in curve.values():
        if pts and any(abs(float(x) - float(y)) > 1e-4 for x, y in pts):
            return True
    return False


# ------------------------------------------------------------------------------------------ global curve
def stop_shift(s: np.ndarray, shadows: float, highlights: float, white: float) -> np.ndarray:
    """New position (stops from mid grey) of tone s after the shadows / highlights controls (-100..100).
    Monotone: the bumps are gentle enough that s' always increases with s."""
    s = np.asarray(s, np.float64)
    out = s.copy()
    if shadows:
        # bump over the shadows: full effect from -6 to -3.5 EV, fading out by mid grey and toward pure black.
        # Ramp widths keep the slope of the shift under 0.7, so the curve stays monotone at +-100.
        w = _smooth((s + 10.0) / 4.0) * (1.0 - _smooth((s + 3.5) / 3.5))
        out = out + 1.6 * (shadows / 100.0) * w
    if highlights:
        top = max(white, 1.0)
        w = _smooth(s / 2.0) * (1.0 - _smooth((s - top) / 2.0))
        out = out + 1.0 * (highlights / 100.0) * w
    return out


def black_white(y: np.ndarray, blacks: float) -> np.ndarray:
    """Display-linear y -> y with the black point moved. Negative deepens, positive lifts (matte)."""
    if blacks < 0:
        bk = -blacks / 100.0 * 0.02
        return np.clip((y - bk) / (1.0 - bk), 0.0, 1.0)
    if blacks > 0:
        bk = blacks / 100.0 * 0.06
        return bk + (1.0 - bk) * y
    return y


def pchip(xs, ys):
    """Monotone cubic interpolant (Fritsch-Carlson) through sorted points; returns f(x)."""
    from scipy.interpolate import PchipInterpolator
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    keep = np.concatenate([[True], np.diff(xs) > 1e-4])
    xs, ys = xs[keep], ys[keep]
    if xs[0] > 0:
        xs, ys = np.concatenate([[0.0], xs]), np.concatenate([[0.0 if ys[0] > 0 else ys[0]], ys])
    if xs[-1] < 1:
        xs, ys = np.concatenate([xs, [1.0]]), np.concatenate([ys, [1.0 if ys[-1] < 1 else ys[-1]]])
    if len(xs) < 2:
        return lambda v: v
    f = PchipInterpolator(xs, np.clip(ys, 0.0, 1.0), extrapolate=True)
    return lambda v: np.clip(f(np.clip(v, 0.0, 1.0)), 0.0, 1.0)


def _enc(y):
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)


def _dec(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def curve_fn(points):
    """A point curve in sRGB-encoded display terms as a display-linear -> display-linear function."""
    if not points:
        return None
    f = pchip([p[0] for p in points], [p[1] for p in points])
    return lambda y: _dec(f(_enc(y)))


def build_light_lut(base_fn, light, mid_in: float, white_stops: float, xs: np.ndarray):
    """(xs, ys_master_or_rgb): compose shadows/highlights (before the base curve, in stops), whites (via
    white_stops already applied by the caller), blacks and point curves (after). Returns ys as a (3, N) array
    when the colour curves differ, else (N,)."""
    xs = np.asarray(xs, np.float64)
    pos = xs > 0
    s = np.zeros_like(xs)
    s[pos] = np.log2(xs[pos] / mid_in)
    s2 = stop_shift(s, light.shadows, light.highlights, white_stops)
    x2 = np.where(pos, mid_in * np.exp2(s2), 0.0)
    y = np.asarray(base_fn(x2.astype(np.float32)), np.float64)
    y = black_white(y, light.blacks)
    pf = param_fn(getattr(light, "param", None))
    if pf is not None:
        y = pf(y)
    curve = light.curve or {}
    m = curve_fn(curve.get("master")) if _has_curve({"m": curve.get("master")}) else None
    if m is not None:
        y = m(y)
    chans = []
    any_c = False
    for c in ("r", "g", "b"):
        f = curve_fn(curve.get(c)) if _has_curve({c: curve.get(c)}) else None
        any_c |= f is not None
        chans.append(f(y) if f is not None else y)
    if any_c:
        return np.stack(chans, 0).astype(np.float32)
    return y.astype(np.float32)


def white_shift(whites: float) -> float:
    """Stops by which the white point moves: + whites = white reached sooner (brighter top end)."""
    return -1.5 * whites / 100.0


# ------------------------------------------------------------------------------------------ Smart lighting
def guided_filter(I: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Self-guided filter (He et al.) with a box of radius r: edge-preserving smoothing in O(N)."""
    size = 2 * r + 1
    mean_I = uniform_filter(I, size, mode="reflect")
    mean_II = uniform_filter(I * I, size, mode="reflect")
    var = np.maximum(mean_II - mean_I * mean_I, 0.0)
    a = var / (var + np.float32(eps))
    b = mean_I - a * mean_I
    return uniform_filter(a, size, mode="reflect") * I + uniform_filter(b, size, mode="reflect")


def smart_radius(eff_scale: float, long_edge_full: float) -> int:
    """Guided-filter radius at this render's resolution: about 1.5% of the long edge."""
    return max(2, int(round(0.015 * long_edge_full * eff_scale)))


def smart_lighting(rgb: np.ndarray, amount: float, mid_in: float, eff_scale: float = 1.0,
                   long_edge_full: float = 6000.0) -> np.ndarray:
    """Lift the lighting of shadows (and ease the brightest areas) while keeping detail. amount 0..100."""
    if amount <= 0:
        return rgb
    k = float(amount) / 100.0
    with slog.stage("light.smart", log, amount=round(amount, 1), mp=round(rgb.shape[0] * rgb.shape[1] / 1e6, 2)):
        Y = rgb @ LUMA if rgb.ndim == 3 else rgb
        floor = np.float32(mid_in * 2.0 ** -14)
        L = np.log2(np.maximum(Y, floor)).astype(np.float32)
        r = smart_radius(eff_scale, long_edge_full)
        base = guided_filter(L, r, 0.25 ** 2)
        s = base - np.float32(math.log2(mid_in))
        # lift up to 1.5 stops where the lighting is 3+ stops below mid grey, fading out by mid grey (midtones and
        # highlights keep their place) and toward pure black; ease the brightest areas a little. The fade keeps the
        # large-scale tones in order (slope of base + lift >= 0.25 at 100%).
        lift = 1.5 * k * _smooth((s + 11.0) / 3.0) * (1.0 - _smooth((s + 3.0) / 3.0))
        ease = -0.5 * k * _smooth((s - 1.5) / 2.0)
        d = (lift + ease).astype(np.float32)
        ratio = np.exp2(d)
        ratio[Y <= floor] = 1.0
        return (rgb * ratio[..., None]).astype(np.float32) if rgb.ndim == 3 else (rgb * ratio).astype(np.float32)


# ------------------------------------------------------------------------------------------ Auto
def auto_tone(Y_scene: np.ndarray, base_fn, mid_in: float) -> dict:
    """Highlights / whites / blacks for a photo (preview-size scene-linear luminance, exposure applied).
    Exposure and shadows are never touched, so a low-key photo stays low-key."""
    y = Y_scene[np.isfinite(Y_scene)].ravel()
    if y.size < 100:
        return {"highlights": 0.0, "whites": 0.0, "blacks": 0.0}
    out = np.asarray(base_fn(np.maximum(y, 0).astype(np.float32)), np.float64)
    clip_frac = float(np.mean(out >= 0.985))
    p998 = float(np.percentile(out, 99.8))
    p0 = float(np.percentile(out, 0.05))
    med = float(np.median(out))
    res = {"highlights": 0.0, "whites": 0.0, "blacks": 0.0}
    if clip_frac > 0.002:                                     # bright areas that would burn: bring them back
        res["highlights"] = -float(np.clip(25.0 + 400.0 * clip_frac, 0.0, 70.0))
    elif p998 < 0.75 and med > 0.12:                          # a flat, dull frame (not a low-key one)
        res["whites"] = float(np.clip((0.92 - p998) * 120.0, 0.0, 50.0))
    if p0 > 0.012:                                            # washed-out blacks
        res["blacks"] = -float(np.clip(p0 * 1500.0, 0.0, 60.0))
    log.debug("auto tone: clip %.4f p99.8 %.3f p0.05 %.4f median %.3f -> %s", clip_frac, p998, p0, med, res)
    return {k: round(v, 1) for k, v in res.items()}
