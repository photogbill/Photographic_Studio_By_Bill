"""Stage 15 (basic): exposure and the base tone curve, scene-referred -> display-referred.

The Phase 1 "Studio Standard" curve is defined in log2 space around mid grey:
  * input mid grey ``mid_in`` (raw-relative linear value that should look like 18% grey). Canon meters
    put a mid-grey subject roughly 3 stops below raw clip. Default 0.11, tuned so the R5, 5D IV and 6D
    test files land within about +-0.3 EV of the camera JPEG (docs/RESULTS.md).
  * slope ``contrast`` through mid grey (log-log), giving the camera-like punch.
  * a smooth Hermite shoulder from ``shoulder`` stops above mid grey up to ``white`` stops, where the
    output reaches 1.0 (white). Above that: clipped, like any raw converter at its white point.
  * shadows below mid grey keep the contrast slope down to black (no lifted blacks).
The curve is applied per channel in a Rec. 2020 render space (D50-adapted): saturation behaves like a
camera JPEG, without the extreme hue twists per-channel curves cause in ProPhoto.

"camera_match" fits a monotone curve to the camera's own embedded JPEG by matching luminance quantiles
(no pixel alignment needed). It is the "look like my camera" option and the Phase 1 acceptance yardstick.
"""
from __future__ import annotations

import io
import math

import numpy as np

from . import gamut
from .spaces import PROPHOTO, REC2020, SRGB, conversion_matrix, apply_matrix

MID_OUT = 0.18

# Optional per-body brightness offsets (EV) that make Studio Standard match the camera JPEG's mid grey.
# Used ONLY when the user turns on "Match camera brightness" (off by default). Measured on Bill's files
# (docs/RESULTS.md): R5 from 13 files; 5D Mark IV and 6D from one file each (provisional).
# Calibration will replace this table with per-body measurements.
BODY_BRIGHTNESS_EV = {
    "Canon EOS R5": 0.35,
    "Canon EOS 5D Mark IV": 0.22,
    "Canon EOS 6D": -0.14,
}


def body_brightness_ev(model: str) -> float:
    return float(BODY_BRIGHTNESS_EV.get((model or "").strip(), 0.0))
PP_TO_RENDER = conversion_matrix(PROPHOTO, REC2020).astype(np.float32)
RENDER_TO_PP = conversion_matrix(REC2020, PROPHOTO).astype(np.float32)
PP_LUMA = PROPHOTO.to_xyz_d50[1].astype(np.float32)          # Y weights of linear ProPhoto


def _hermite(t, p0, p1, m0, m1):
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1


TOE_START = -2.0                                              # stops below mid grey where the toe begins
TOE_WIDTH = 1.5                                               # stops over which the slope eases into the toe


def standard_curve(x: np.ndarray, mid_in: float = 0.11, contrast: float = 1.35, shoulder: float = 1.0,
                   white: float = 3.0, white_slope: float = 0.15, toe: float = 0.0) -> np.ndarray:
    """Studio Standard tone curve. x: linear (1.0 = raw clip at 0 EV). Returns display-linear 0..1.
    ``toe`` (-100..100) bends the bottom like film: below TOE_START the log-log slope eases from ``contrast`` to
    contrast x (1 - 0.45 toe/100), so + opens the deepest shadows softly and - crushes them. Always monotone."""
    x = np.asarray(x, np.float32)
    t_white = math.log2(1.0 / MID_OUT)                       # output stops from mid grey to white (2.47)
    shoulder = min(shoulder, t_white / contrast - 0.05, white - 0.25)
    xs = np.maximum(x, 1e-12)
    s = np.log2(xs / mid_in)
    t = contrast * s                                          # straight line through mid grey
    if toe:
        slope_toe = contrast * (1.0 - 0.45 * max(-1.0, min(1.0, toe / 100.0)))
        u = np.maximum((TOE_START - s) / TOE_WIDTH, 0.0)
        integ = np.where(u <= 1.0, u ** 3 - 0.5 * u ** 4, 0.5 + (u - 1.0))   # integral of the smoothstep
        t = t - TOE_WIDTH * (slope_toe - contrast) * integ
    d = white - shoulder
    u = np.clip((s - shoulder) / d, 0.0, 1.0)
    herm = _hermite(u, contrast * shoulder, t_white, contrast * d, white_slope * d)
    t = np.where(s > shoulder, herm, t)
    y = MID_OUT * np.exp2(t)
    y = np.where(s >= white, 1.0, y)
    y = np.where(x <= 0, 0.0, y)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


LUT_LO, LUT_HI, LUT_N = 1e-6, 16.0, 4096


def curve_lut(fn, n: int = LUT_N, max_in: float = LUT_HI):
    """Tabulate a curve on a log-spaced grid (plus zero) for fast application."""
    xs = np.concatenate([[0.0], np.geomspace(LUT_LO, max_in, n - 1)]).astype(np.float32)
    return xs, fn(xs).astype(np.float32)


def _to_log_grid(xs: np.ndarray, ys: np.ndarray):
    """Resample any monotone (xs, ys) curve onto the standard log grid, so it can be applied by indexing."""
    grid = np.geomspace(LUT_LO, LUT_HI, LUT_N - 1)
    return np.interp(grid, xs, ys, left=float(ys[0]), right=float(ys[-1])).astype(np.float32)


def apply_lut(v: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Evaluate the curve at v (any shape) by direct indexing on the log grid: ~5x faster than np.interp.
    Below the grid the curve is a straight line to (0, ys(0)); above it, the last value."""
    g = _to_log_grid(xs, ys)
    y0 = float(np.interp(0.0, xs, ys))
    lo_l, step = np.log2(LUT_LO), (np.log2(LUT_HI) - np.log2(LUT_LO)) / (LUT_N - 2)
    flat = v.reshape(-1)
    u = (np.log2(np.maximum(flat, LUT_LO)) - lo_l) / step
    np.clip(u, 0.0, LUT_N - 2 - 1e-6, out=u)
    i = u.astype(np.int32)
    f = (u - i).astype(np.float32)
    out = g[i] + f * (g[i + 1] - g[i])
    low = flat < LUT_LO
    if low.any():
        out[low] = y0 + (g[0] - y0) * np.maximum(flat[low], 0) / LUT_LO
    return out.reshape(v.shape)


def apply_curve_rgb(img_pp: np.ndarray, xs: np.ndarray, ys: np.ndarray, compress: bool = True) -> np.ndarray:
    """Apply a tone curve per channel in the render space; input and output are linear ProPhoto.
    Out-of-gamut colours are compressed into the render gamut first (gamut.py), so the curve never sees
    negative channels. ``ys`` may be (3, N): one curve each for R, G, B (the point curves of the Light panel)."""
    r = apply_matrix(img_pp, PP_TO_RENDER)
    if compress:
        r = gamut.compress(r)
    np.maximum(r, 0.0, out=r)
    if np.ndim(ys) == 2:
        for c in range(3):
            r[..., c] = apply_lut(np.ascontiguousarray(r[..., c]), xs, ys[c])
    else:
        r = apply_lut(r, xs, ys)
    return apply_matrix(r, RENDER_TO_PP)


def luminance(img_pp: np.ndarray) -> np.ndarray:
    return img_pp.reshape(-1, 3) @ PP_LUMA


# ------------------------------------------------------------------------------------------ camera match
def camera_match_curve(lin_pp: np.ndarray, jpeg_bytes: bytes, n_q: int = 512):
    """Monotone curve (xs, ys) that maps our scene-linear luminance onto the camera JPEG's luminance
    distribution. Both images are reduced to ~1 MP; no alignment is needed (quantile matching)."""
    from PIL import Image
    im = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    im.thumbnail((1400, 1400))
    j = SRGB.decode(np.asarray(im, np.float32) / 255.0).astype(np.float32)
    jl = j.reshape(-1, 3) @ SRGB.to_xyz_d50[1].astype(np.float32)
    step = max(1, int(round(math.sqrt(lin_pp.shape[0] * lin_pp.shape[1] / 1.0e6))))
    ol = luminance(np.ascontiguousarray(lin_pp[::step, ::step]))
    q = np.linspace(0.002, 0.998, n_q)
    xin = np.quantile(ol, q)
    yout = np.quantile(jl, q)
    # enforce strict monotonicity and anchor zero
    xin = np.maximum.accumulate(np.maximum(xin, 1e-7))
    keep = np.concatenate([[True], np.diff(xin) > 1e-9])
    xin, yout = xin[keep], np.maximum.accumulate(yout[keep])
    xs = np.concatenate([[0.0], xin, [xin[-1] * 4]]).astype(np.float32)
    ys = np.concatenate([[0.0], yout, [min(1.0, yout[-1] * 1.02)]]).astype(np.float32)
    return xs, ys


def ev_offset_vs_standard(xs_match: np.ndarray, ys_match: np.ndarray, **std_kw) -> float:
    """How many stops our Standard curve differs from the camera-match curve at mid grey output (0.18).
    Positive = the camera renders brighter than Studio Standard."""
    x_cam = float(np.interp(MID_OUT, ys_match, xs_match))
    xs, ys = curve_lut(lambda v: standard_curve(v, **std_kw))
    x_std = float(np.interp(MID_OUT, ys, xs))
    if x_cam <= 0 or x_std <= 0:
        return float("nan")
    return math.log2(x_std / x_cam)
