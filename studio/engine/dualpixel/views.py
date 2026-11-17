"""Views of a Dual Pixel photo for the page, and the files it can make (BUILD_PLAN_TABS 3.5).

Views (all upright uint8 sRGB): the normal picture (A+B), each half developed like the photo, their difference
(amplified 8x around grey: where they disagree is where depth or noise is), the depth map (colour ramp, greyed
where it could not be measured), its confidence, and the noise-versus-detail map. Files: a developed half as a
linear intermediate (what DPRSplit does, built in), a red/cyan anaglyph, a side-by-side pair, a wigglegram GIF
flipping between the two views (optionally exaggerated by warping each half along its offset), and a 16-bit depth
PNG (near = white) for Photoshop or Blender. Every file is named <stem>_DP-..._Studio. No Qt here.
"""
from __future__ import annotations

import os

import numpy as np

from ... import log as slog

log = slog.get("engine.dualpixel.views")

VIEWS = ("A+B", "Half 1", "Half 2", "Difference", "Depth", "Confidence", "Noise vs detail")
VIEW_INFO = {
    "A+B": "The normal photo: both halves together.",
    "Half 1": "Light through one side of the lens only, brightened to match (shading corrected).",
    "Half 2": "Light through the other side.",
    "Difference": "Where the two halves disagree (8x): depth edges and noise light up, things in focus stay grey.",
    "Depth": "Offset between the halves as colour: violet in front, cyan near focus, amber behind. Grey: no texture "
             "to measure.",
    "Confidence": "How far to trust the depth: bright where there is texture, dark on plain areas.",
    "Noise vs detail": "Cyan where both halves agree (real detail), dark where they do not (noise).",
}
NAMES = {"hdr": "DP-HDR", "half1": "DP-A", "half2": "DP-B", "depth": "DP-Depth", "wiggle": "DP-3D",
         "anaglyph": "DP-Anaglyph", "pair": "DP-Pair"}
EXT = {"hdr": ".tif", "half1": ".tif", "half2": ".tif", "depth": ".png", "wiggle": ".gif", "anaglyph": ".jpg",
       "pair": ".jpg"}


def out_name(path: str, kind: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{stem}_{NAMES[kind]}_Studio{EXT[kind]}"


def _to_srgb8(img_pp: np.ndarray) -> np.ndarray:
    from ..io.export import ExportSpec, to_output
    return to_output(np.ascontiguousarray(img_pp, np.float32), ExportSpec(format="jpeg", space="srgb", bits=8))


def _render(image, scale: float) -> np.ndarray:
    """Display-referred linear ProPhoto of an engine Image with the default look (not upright yet: render() turns
    it upright itself)."""
    from ..pipeline.edit import Edit
    from ..pipeline.pipeline import render
    return render(image, Edit(), None, scale)


def developed(dp, gains, which: str, scale: float = 0.25) -> np.ndarray:
    """uint8 sRGB of A+B, half 1 or half 2, developed like the photo, upright."""
    if which == "A+B":
        return _to_srgb8(_render(dp.image, scale))
    img = dp.half_image(1 if which == "Half 1" else 2, gains)
    return _to_srgb8(_render(img, scale))


def _upright(a: np.ndarray, flip: int) -> np.ndarray:
    from ..pipeline.pipeline import apply_orientation
    return np.ascontiguousarray(apply_orientation(a, flip))


def _resize(a: np.ndarray, shape) -> np.ndarray:
    from PIL import Image as PI
    if a.shape[:2] == tuple(shape[:2]):
        return a
    if a.dtype == np.uint8:
        return np.asarray(PI.fromarray(a).resize((shape[1], shape[0]), PI.Resampling.BILINEAR))
    from scipy.ndimage import zoom
    return zoom(a, (shape[0] / a.shape[0], shape[1] / a.shape[1]) + ((1,) if a.ndim == 3 else ()), order=1)


def ramp(t: np.ndarray) -> np.ndarray:
    """0..1 -> violet (near), cyan, green, amber (far): the Studio's palette."""
    from ..stack.focus import _ramp
    return _ramp(t)


def view(dp, gains, name: str, scale: float = 0.25, maps: dict | None = None) -> np.ndarray:
    """uint8 RGB of one of VIEWS, upright, about ``scale`` of full size. ``maps``: {"disp", "conf", "rho"} at
    half size (from depth.py)."""
    maps = maps or {}
    if name in ("A+B", "Half 1", "Half 2"):
        return developed(dp, gains, name, scale)
    if name == "Difference":
        a = developed(dp, gains, "Half 1", scale).astype(np.float32)
        b = developed(dp, gains, "Half 2", scale).astype(np.float32)
        d = (a.mean(2) - b.mean(2)) * 8.0 + 128.0
        return np.repeat(np.clip(d, 0, 255).astype(np.uint8)[..., None], 3, 2)
    h = int(round(dp.shape[0] * scale))
    w = int(round(dp.shape[1] * scale))
    if name == "Depth":
        d, c = maps["disp"], maps["conf"]
        lim = max(0.5, float(np.percentile(np.abs(d[c > 0.3]), 98)) if (c > 0.3).any() else 1.0)
        rgb = ramp(0.5 + 0.5 * np.clip(d / lim, -1, 1))
        k = np.clip(c * 2.0, 0, 1)[..., None]
        rgb = rgb * k + 60.0 * (1 - k)
    elif name == "Confidence":
        c = np.clip(maps["conf"], 0, 1)
        rgb = np.stack([c * 40, c * 230, c * 255], 2)
    elif name == "Noise vs detail":
        r = np.clip(maps["rho"], 0, 1)
        rgb = np.stack([r * 30, r * 220, r * 255], 2)
    else:
        raise ValueError(f"unknown view {name!r}")
    rgb = _resize(np.clip(rgb, 0, 255).astype(np.uint8), (h, w))
    return _upright(rgb, dp.flip)


def export_half(dp, gains, which: int, path: str) -> str:
    """A developed half (scene-linear, shading corrected, brightness of the full photo) as a linear intermediate."""
    from ..io.intermediate import write_intermediate
    from ..pipeline.pipeline import linear_rgb
    from .hdr import hdr_edit
    img = dp.half_image(which, gains)
    rgb = linear_rgb(img, hdr_edit(), None, cache=False)
    del img
    md = dp.image.meta
    meta = {"make": md.make, "model": md.model, "iso": md.iso, "exposure_s": md.exposure_s, "fnumber": md.fnumber,
            "focal_mm": md.focal_mm, "lens": md.lens, "datetime": md.datetime}
    return write_intermediate(path, _upright(rgb, dp.flip), meta, {
        "kind": "dualpixel", "method": f"half{which}", "sources": [os.path.basename(dp.path)], "scale": 1.0})


def _pair(dp, gains, scale: float, boost: float = 1.0, disp=None):
    a = developed(dp, gains, "Half 1", scale)
    b = developed(dp, gains, "Half 2", scale)
    if boost > 1.0 and disp is not None:
        a, b = _exaggerate(a, b, dp, disp, boost - 1.0)
    return a, b


def _exaggerate(a, b, dp, disp, extra: float):
    """Move each half further apart along its measured offset (wigglegram 'boost')."""
    from scipy.ndimage import map_coordinates
    d = _upright(_resize(disp.astype(np.float32), (dp.shape[0] // 2, dp.shape[1] // 2)), dp.flip)
    d = _resize(d, a.shape[:2]) * (a.shape[1] / float(dp.shape[1] if dp.flip not in (5, 6) else dp.shape[0]))
    H, W = a.shape[:2]
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    ax = 0 if dp.flip in (5, 6) else 1                 # the split is along the sensor's long axis
    out = []
    for img, sgn in ((a, 1.0), (b, -1.0)):
        if ax == 1:
            c = [Y, X - sgn * extra * d / 2.0]
        else:
            c = [Y - sgn * extra * d / 2.0, X]
        out.append(np.stack([map_coordinates(img[..., k].astype(np.float32), c, order=1, mode="nearest")
                             for k in range(3)], 2).clip(0, 255).astype(np.uint8))
    return out[0], out[1]


def anaglyph(dp, gains, scale: float = 0.5) -> np.ndarray:
    """Red = half 1, cyan = half 2 (red/cyan glasses)."""
    a, b = _pair(dp, gains, scale)
    out = b.copy()
    out[..., 0] = a.mean(2).astype(np.uint8)
    return out


def side_by_side(dp, gains, scale: float = 0.5) -> np.ndarray:
    a, b = _pair(dp, gains, scale)
    gap = np.zeros((a.shape[0], max(8, a.shape[1] // 50), 3), np.uint8)
    return np.concatenate([a, gap, b], axis=1)


def wigglegram(dp, gains, path: str, long_edge: int = 1200, ms: int = 120, boost: float = 1.0, disp=None) -> str:
    """A GIF flipping between the two views (two frames, looping)."""
    from PIL import Image as PI
    scale = min(1.0, long_edge / float(max(dp.shape)))
    a, b = _pair(dp, gains, scale, boost, disp)
    fa, fb = PI.fromarray(a), PI.fromarray(b)
    tmp = path + ".partial"
    fa.save(tmp, "GIF", save_all=True, append_images=[fb], duration=int(ms), loop=0)
    os.replace(tmp, path)
    return path


def save_jpeg(arr: np.ndarray, path: str, quality: int = 92) -> str:
    from PIL import Image as PI
    tmp = path + ".partial"
    PI.fromarray(arr).save(tmp, "JPEG", quality=quality)
    os.replace(tmp, path)
    return path


def depth_png(disp_full: np.ndarray, conf: np.ndarray, path: str, flip: int = 0, swap: bool = False) -> str:
    """16-bit greyscale PNG of depth, near = white, far = black (low-confidence places keep their smoothed value),
    upright."""
    from PIL import Image as PI
    d = -disp_full if not swap else disp_full           # in front of focus (negative offset) -> bright
    lo, hi = np.percentile(d, [1, 99])
    t = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    a = _upright((t * 65535 + 0.5).astype(np.uint16), flip)
    tmp = path + ".partial"
    PI.fromarray(np.ascontiguousarray(a)).save(tmp, "PNG")    # uint16 2-D -> "I;16" (the mode= argument is going away)
    os.replace(tmp, path)
    return path
