"""Small image operations in NumPy and SciPy: resize, affine warp, Gaussian blur.

The Studio does not install OpenCV (THIRD_PARTY.md). A few tools had started calling it anyway (composite, SAM
click-to-select, depth, JPEG repair, the subject picker), which worked on the build machine and failed on every
installed copy. These are drop-in replacements with OpenCV's conventions, so results match what those tools were
tuned on:

* ``resize_linear`` / ``resize_cubic``: pixel centres at ``(i + 0.5) * in / out - 0.5``, edges repeated, no
  anti-aliasing when shrinking (like ``cv2.resize`` with INTER_LINEAR / INTER_CUBIC, cubic a = -0.75).
* ``warp_affine``: ``M`` maps source (x, y) to destination (x, y) like ``cv2.warpAffine``; outside is 0.
* ``gaussian_blur``: sigma in pixels, kernel out to 4 sigma, mirrored edges (BORDER_REFLECT_101).

Each works on 2-D (H, W) and 3-D (H, W, C) float arrays and returns float32.
"""
from __future__ import annotations

import numpy as np


def _axis_linear(n_in: int, n_out: int):
    x = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / float(n_out)) - 0.5
    x = np.clip(x, 0.0, n_in - 1)
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    f = (x - i0).astype(np.float32)
    return i0, i1, f


def resize_linear(a: np.ndarray, w: int, h: int) -> np.ndarray:
    """Bilinear resize to (h, w), as ``cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)``."""
    a = np.asarray(a, np.float32)
    H, W = a.shape[:2]
    w, h = int(w), int(h)
    if (W, H) == (w, h):
        return a.copy()
    y0, y1, fy = _axis_linear(H, h)
    x0, x1, fx = _axis_linear(W, w)
    sh = (-1,) + (1,) * (a.ndim - 1)
    rows = a[y0] * (1.0 - fy).reshape(sh) + a[y1] * fy.reshape(sh)
    sh2 = (1, -1) + (1,) * (a.ndim - 2)
    out = rows[:, x0] * (1.0 - fx).reshape(sh2) + rows[:, x1] * fx.reshape(sh2)
    return np.ascontiguousarray(out, np.float32)


def _cubic_w(t: np.ndarray, a: float = -0.75) -> np.ndarray:
    t = np.abs(t)
    t2, t3 = t * t, t * t * t
    return np.where(t <= 1.0, (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0,
                    np.where(t < 2.0, a * t3 - 5.0 * a * t2 + 8.0 * a * t - 4.0 * a, 0.0))


def _resize_axis_cubic(a: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    n_in = a.shape[axis]
    x = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / float(n_out)) - 0.5
    base = np.floor(x).astype(np.int64)
    f = x - base
    out = None
    for k in range(-1, 3):
        idx = np.clip(base + k, 0, n_in - 1)
        wk = _cubic_w(f - k).astype(np.float32)
        sh = [1] * a.ndim
        sh[axis] = -1
        term = np.take(a, idx, axis=axis) * wk.reshape(sh)
        out = term if out is None else out + term
    return out


def resize_cubic(a: np.ndarray, w: int, h: int) -> np.ndarray:
    """Bicubic resize to (h, w), as ``cv2.resize(..., interpolation=cv2.INTER_CUBIC)`` (a = -0.75)."""
    a = np.asarray(a, np.float32)
    H, W = a.shape[:2]
    if (W, H) == (int(w), int(h)):
        return a.copy()
    out = _resize_axis_cubic(a, int(h), 0)
    out = _resize_axis_cubic(out, int(w), 1)
    return np.ascontiguousarray(out, np.float32)


def warp_affine(a: np.ndarray, M: np.ndarray, w: int, h: int) -> np.ndarray:
    """Bilinear affine warp onto a (h, w) canvas, as ``cv2.warpAffine(a, M, (w, h))`` with a constant 0 border.

    ``M`` (2 x 3) maps source (x, y) to destination (x, y). Only the part of the canvas the warped source can reach
    is computed, so placing a small subject on a large photo stays cheap."""
    from scipy.ndimage import affine_transform
    a = np.asarray(a, np.float32)
    M = np.asarray(M, np.float64)
    A, t = M[:, :2], M[:, 2]
    Ai = np.linalg.inv(A)
    out = np.zeros((int(h), int(w)) + a.shape[2:], np.float32)
    H, W = a.shape[:2]
    corners = np.array([[-1, -1], [W, -1], [-1, H], [W, H]], np.float64)
    dst = corners @ A.T + t
    x0 = int(max(0, np.floor(dst[:, 0].min())))
    x1 = int(min(w, np.ceil(dst[:, 0].max()) + 1))
    y0 = int(max(0, np.floor(dst[:, 1].min())))
    y1 = int(min(h, np.ceil(dst[:, 1].max()) + 1))
    if x1 <= x0 or y1 <= y0:
        return out
    # output (row r, col c) of the window is destination (x0 + c, y0 + r); the source is Ai @ (dst - t), in (x, y).
    # scipy wants (row, col) -> (row, col): swap the axes of the matrix and the offset.
    P = np.array([[Ai[1, 1], Ai[1, 0]], [Ai[0, 1], Ai[0, 0]]])
    o = Ai @ (np.array([x0, y0], np.float64) - t)
    off = np.array([o[1], o[0]])
    win = (y1 - y0, x1 - x0)
    if a.ndim == 2:
        out[y0:y1, x0:x1] = affine_transform(a, P, offset=off, output_shape=win, order=1, mode="grid-constant",
                                             cval=0.0)
    else:
        for c in range(a.shape[2]):
            out[y0:y1, x0:x1, c] = affine_transform(a[..., c], P, offset=off, output_shape=win, order=1,
                                                    mode="grid-constant", cval=0.0)
    return out


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur, sigma in pixels, as ``cv2.GaussianBlur(a, (0, 0), sigma)`` (mirrored edges)."""
    from scipy.ndimage import gaussian_filter
    a = np.asarray(a, np.float32)
    s = (float(sigma), float(sigma)) + (0.0,) * (a.ndim - 2)
    return gaussian_filter(a, s, mode="mirror", truncate=4.0).astype(np.float32)
