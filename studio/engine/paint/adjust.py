"""Adjustment layers and filters for Paint (BUILD_PLAN_PAINT P4).

An adjustment layer changes everything below it (inside its group) without touching the paint, and can be hidden,
faded or masked. Filters change a layer's own pixels (undoable). Each says what it is best for and weak at.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .colour import oklab_to_srgb, srgb_to_oklab
from .curves import Curve

ADJUSTMENTS = {
    "curves": ("Curves", "shaping tone: contrast, lifting shadows, taming highlights", "colour shifts (it acts on "
               "all channels together)"),
    "levels": ("Levels", "setting the darkest and lightest points and the mid-tone", "local changes"),
    "hue_sat": ("Hue / saturation / lightness", "shifting colour and saturation evenly to the eye (in OKLCH)",
                "fine selective colour work"),
    "bright_contrast": ("Brightness / contrast", "quick overall changes", "precise tone (use Curves)"),
    "colour_balance": ("Colour balance", "warming or cooling the whole painting", "correcting one colour"),
    "bw": ("Black and white", "checking and painting values", "keeping any colour"),
    "invert": ("Invert", "negatives and checking drawing errors", "nothing else"),
    "posterize": ("Posterize", "flat colour masses and value planning", "smooth gradients"),
    "natural_curve": ("Natural curve (the Studio's)", "contrast that keeps colours as they are: the Studio's Develop "
                      "curve, which sets each pixel's brightness and keeps its colour mix", "per-channel colour "
                      "shifts (use Curves)"),
}
FILTER_MARGIN = 32

FILTERS = {
    "blur": ("Blur (Gaussian)", "softening, depth of field, glows", "detail"),
    "sharpen": ("Sharpen (unsharp mask)", "crisper edges on soft painting", "noise and halos if overdone"),
    "noise": ("Add noise", "grain that breaks up flat digital colour", "clean areas"),
    "motion": ("Motion blur", "speed and movement", "anything still"),
    "median": ("Smooth (median)", "removing speckle while keeping edges", "fine texture"),
    "clean": ("Clean (the Studio's)", "taking noise and grain out of a scanned or photographed drawing, measured "
              "from the picture itself, as the Studio cleans scans", "painted texture you want to keep (it is "
              "measured as noise when it is very fine)"),
}


def defaults(kind):
    return {"curves": {"points": [(0, 0), (1, 1)]}, "levels": {"black": 0.0, "white": 1.0, "gamma": 1.0},
            "hue_sat": {"hue": 0.0, "sat": 1.0, "light": 0.0}, "bright_contrast": {"brightness": 0.0,
                                                                                   "contrast": 0.0},
            "colour_balance": {"warmth": 0.0, "tint": 0.0}, "bw": {}, "invert": {}, "posterize": {"levels": 5},
            "natural_curve": {"points": [(0, 0), (0.25, 0.2), (0.75, 0.8), (1, 1)]}}[kind]


def apply(kind, p, rgb):
    """Apply an adjustment to straight sRGB (..., 3) values 0..1."""
    x = np.clip(rgb, 0, 1).astype(np.float32)
    if kind == "curves":
        return np.clip(Curve(p.get("points", [(0, 0), (1, 1)]))(x), 0, 1).astype(np.float32)
    if kind == "levels":
        b, w = p.get("black", 0.0), max(p.get("white", 1.0), p.get("black", 0.0) + 1e-3)
        return np.clip(np.clip((x - b) / (w - b), 0, 1) ** (1.0 / max(p.get("gamma", 1.0), 0.05)), 0, 1)
    if kind == "hue_sat":
        lab = srgb_to_oklab(x)
        L, a, bb = lab[..., 0], lab[..., 1], lab[..., 2]
        C, H = np.hypot(a, bb) * p.get("sat", 1.0), np.arctan2(bb, a) + np.radians(p.get("hue", 0.0))
        L = np.clip(L + p.get("light", 0.0) * 0.5, 0, 1)
        return oklab_to_srgb(np.stack([L, C * np.cos(H), C * np.sin(H)], -1))
    if kind == "bright_contrast":
        y = x + p.get("brightness", 0.0) * 0.5
        c = p.get("contrast", 0.0)
        return np.clip((y - 0.5) * (1 + c) + 0.5, 0, 1)
    if kind == "colour_balance":
        w_, t = p.get("warmth", 0.0) * 0.15, p.get("tint", 0.0) * 0.15
        return np.clip(x + np.array([w_, -t, -w_], np.float32), 0, 1)
    if kind == "bw":
        v = x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722
        return np.repeat(v[..., None], 3, -1)
    if kind == "invert":
        return 1 - x
    if kind == "natural_curve":                                  # the Studio's own curve, in linear ProPhoto
        from ..color import spaces, tone
        from .colour import linear_to_srgb, srgb_to_linear
        pts = sorted(p.get("points", [(0, 0), (1, 1)]))
        xs = np.array([q[0] for q in pts], np.float32)
        ys = np.array([q[1] for q in pts], np.float32)
        M = spaces.conversion_matrix(spaces.SRGB, spaces.PROPHOTO).astype(np.float32)
        pp = spaces.apply_matrix(np.ascontiguousarray(srgb_to_linear(x).reshape(-1, 1, 3)), M)
        out = tone.apply_curve_rgb(pp, xs, ys, natural=True)
        back = spaces.apply_matrix(np.ascontiguousarray(out.reshape(-1, 1, 3)), np.linalg.inv(M).astype(np.float32))
        return np.clip(linear_to_srgb(np.clip(back, 0, None)), 0, 1).reshape(x.shape).astype(np.float32)
    if kind == "posterize":
        n = max(2, int(p.get("levels", 5)))
        return np.round(x * (n - 1)) / (n - 1)
    raise ValueError(kind)


def filter_pixels(kind, px, amount=1.0, seed=0):
    """A filter on premultiplied RGBA pixels."""
    if kind == "blur":
        s = max(0.3, 4.0 * amount)
        return np.stack([ndimage.gaussian_filter(px[..., c], s) for c in range(4)], -1).astype(np.float32)
    if kind == "sharpen":
        bl = np.stack([ndimage.gaussian_filter(px[..., c], 1.5) for c in range(4)], -1)
        out = px + (px - bl) * amount
        out[..., :3] = np.clip(out[..., :3], 0, np.clip(out[..., 3:4], 0, 1))
        out[..., 3] = np.clip(out[..., 3], 0, 1)
        return out.astype(np.float32)
    if kind == "noise":
        rng = np.random.default_rng(seed)
        n = rng.normal(0, 0.06 * amount, px.shape[:2] + (1,)).astype(np.float32)
        out = px.copy()
        out[..., :3] = np.clip(px[..., :3] + n * px[..., 3:4], 0, px[..., 3:4])
        return out
    if kind == "motion":
        k = max(3, int(16 * amount) | 1)
        return np.stack([ndimage.uniform_filter1d(px[..., c], k, axis=1) for c in range(4)], -1).astype(np.float32)
    if kind == "clean":                                          # the Studio's Clean for scans, on straight colour
        from ..local import scanclean
        a = px[..., 3:4]
        straight = np.where(a > 0, px[..., :3] / np.maximum(a, 1e-8), 0).astype(np.float32)
        cleaned = scanclean.clean(straight, scanclean.measure(straight), luma=amount, chroma=1.5 * amount)
        return np.concatenate([np.clip(cleaned, 0, 1) * a, a], -1).astype(np.float32)
    if kind == "median":
        s = max(3, int(3 + 4 * amount) | 1)
        return np.stack([ndimage.median_filter(px[..., c], s) for c in range(4)], -1).astype(np.float32)
    raise ValueError(kind)
