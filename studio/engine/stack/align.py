"""Lining frames up: shifts, similarity / affine / perspective fits, warps and local flow (BUILD_PLAN_TABS 1.5).

Every stacking tool stands on this module: Focus Stack (breathing changes the scale a little from frame to frame),
Super-Res Stack (hand shake, rolling shutter, air wobble) and the Dual Pixel Lab (the two halves of one photo).

Conventions, used everywhere below:
  - Arrays are indexed [row, column] = [y, x]; pixel centres sit at whole numbers.
  - A ``Transform`` maps a point of the REFERENCE frame to where the same content lies in the OTHER frame:
    (x', y') = T(x, y). ``warp(img, T)`` therefore samples ``img`` at T(p) for every output pixel p, which puts
    ``img`` into the reference frame.
  - A shift (dy, dx) measured between a reference tile and another tile means: the content at p in the
    reference is found at p + (dy, dx) in the other.
  - Binning by k (an analysis copy) puts the centre of binned pixel i at k * i + (k - 1) / 2 in full pixels;
    ``Transform.rescaled`` and ``Flow.rescaled`` convert between the two.

Registration runs on small luminance copies (``analysis_copy``) and is coarse to fine: one whole-frame phase
correlation at 1/8 size, then tiles measured on the frame already warped by the estimate so far, fitted with
weighted least squares and three rounds of dropping tiles that disagree (something that moved), up to full
analysis size, twice at the last level. Phase correlation's peak fit is biased toward whole pixels; each tile is
re-measured after a Fourier shift by its estimate so far, where the bias is negligible (as lens/auto.py does).
No Qt here (rule 2.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ... import log as slog
from ..parallel import strips

log = slog.get("engine.stack.align")

KINDS = ("shift", "similarity", "affine", "homography")
_NEED = {"shift": 1, "similarity": 2, "affine": 3, "homography": 4}     # tiles a fit needs, at the very least

ALIGN_INFO = {
    "similarity": {"name": "Similarity (default)", "does": "Size, turn and shift: what focus breathing and a "
                   "steady hand do.", "best": "focus brackets on a rail or tripod, hand-held bursts",
                   "weak": "a camera that tilted between frames", "speed": "fast"},
    "affine": {"name": "Affine", "does": "Also allows a slight stretch or shear.",
               "best": "a rail that is not quite square to the subject", "weak": "can follow noise on frames "
               "with little detail", "speed": "fast"},
    "homography": {"name": "Perspective", "does": "Allows a full change of viewpoint on a flat subject.",
                   "best": "a camera that tilted or swung between frames", "weak": "needs detail across the "
                   "whole frame; a deep subject cannot be lined up by one perspective", "speed": "fast"},
    "shift": {"name": "Shift only", "does": "Moves frames, never resizes or turns them.",
              "best": "the moon, telescopes, a camera locked down", "weak": "anything that zoomed or turned",
              "speed": "fastest"},
}


# ------------------------------------------------------------------------------------------ helpers
def luma(rgb: np.ndarray) -> np.ndarray:
    """0.25 R + 0.5 G + 0.25 B: plenty for registration, and the same weights as a 2x2 Bayer bin."""
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        return rgb.astype(np.float32, copy=False)
    return (0.25 * rgb[..., 0] + 0.5 * rgb[..., 1] + 0.25 * rgb[..., 2]).astype(np.float32)


def bin_by(x: np.ndarray, k: int) -> np.ndarray:
    """Average k x k blocks (a cheap, alias-tolerant reduction for analysis copies). Extra rows and columns that
    do not fill a block are dropped."""
    if k <= 1:
        return np.ascontiguousarray(x, np.float32)
    h, w = x.shape[0] // k * k, x.shape[1] // k * k
    x = np.asarray(x[:h, :w], np.float32)
    return np.ascontiguousarray(x.reshape(h // k, k, w // k, k, *x.shape[2:]).mean(axis=(1, 3)), np.float32)


def analysis_copy(lum: np.ndarray, cap: int = 2000) -> tuple[np.ndarray, int]:
    """(copy, k): ``lum`` binned by the smallest integer k that brings its long edge to ``cap`` or under."""
    k = max(1, int(np.ceil(max(lum.shape[:2]) / float(cap))))
    return bin_by(lum, k), k


def noise_sigma(x: np.ndarray) -> float:
    """Pixel noise of a luminance image: the 5-point Laplacian's variance in 16 x 16 blocks, the quietest tenth of
    the blocks (a plain median over the frame counts fine texture as noise on a detailed photo, and a frame full of
    texture would then look too noisy to measure). Corrected for the low pick: exact for pure noise."""
    x = np.asarray(x, np.float32)
    if min(x.shape) < 3:
        return 0.0
    d = (4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1] - x[1:-1, :-2] - x[1:-1, 2:]) / np.float32(np.sqrt(20.0))
    b = 16
    h, w = d.shape[0] // b * b, d.shape[1] // b * b
    if h < b or w < b:
        return float(1.4826 * np.median(np.abs(d - np.median(d))))
    v = (d[:h, :w] ** 2).reshape(h // b, b, w // b, b).mean(axis=(1, 3)).ravel()
    if v.size < 10:
        return float(np.sqrt(np.median(v)))
    return float(np.sqrt(np.percentile(v, 10) / 0.89))


def highpass(x: np.ndarray, sigma: float) -> np.ndarray:
    """x minus its Gaussian blur: detail finer than ``sigma`` pixels, broad shading removed."""
    from scipy.ndimage import gaussian_filter
    x = np.asarray(x, np.float32)
    return (x - gaussian_filter(x, max(1.0, float(sigma)), mode="reflect")).astype(np.float32)


def guided(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter (He, Sun and Tang 2010): smooth ``p`` following the edges of the guide ``I``. Box radius r."""
    from scipy.ndimage import uniform_filter
    size = 2 * int(r) + 1
    I = np.asarray(I, np.float32)
    p = np.asarray(p, np.float32)
    mI = uniform_filter(I, size, mode="reflect")
    mp = uniform_filter(p, size, mode="reflect")
    cov = uniform_filter(I * p, size, mode="reflect") - mI * mp
    var = np.maximum(uniform_filter(I * I, size, mode="reflect") - mI * mI, 0.0)
    a = cov / (var + np.float32(eps))
    b = mp - a * mI
    return (uniform_filter(a, size, mode="reflect") * I + uniform_filter(b, size, mode="reflect")).astype(np.float32)


def weighted_guided(I: np.ndarray, p: np.ndarray, w: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Confidence-weighted guided filter: guided(w * p) / guided(w). Places with little confidence take their
    value from confident neighbours along the guide's edges."""
    num = guided(I, p * w, r, eps)
    den = guided(I, w, r, eps)
    out = num / np.maximum(den, 1e-4)
    weak = den < 1e-3
    if np.any(weak):                                  # nothing confident nearby: plain local average
        from scipy.ndimage import uniform_filter
        size = 4 * int(r) + 1
        a = uniform_filter(p * w, size) / np.maximum(uniform_filter(w, size), 1e-6)
        out[weak] = a[weak]
    return out.astype(np.float32)


# ------------------------------------------------------------------------------------------ phase correlation
def phase_shift(a, b, iters: int = 3):
    """Sub-pixel shift (dy, dx) that maps b onto a, by phase correlation: a(p) is b(p - (dy, dx)), so content at
    p in b lies at p + (dy, dx) in a. The parabolic peak fit is biased toward whole pixels, so the estimate is
    refined: b is moved by the current estimate (Fourier shift) and the small remainder is measured again, where
    the bias is negligible. Returns (dy, dx, peak); the peak (0..1) says how well the two agree."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A = np.fft.fft2(a * win)
    Bf = np.fft.fft2(b * win)
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    ty = tx = 0.0
    peak = 0.0
    h, w = a.shape

    def sub(cm, c0, cp):
        d = cm - 2 * c0 + cp
        return 0.0 if abs(d) < 1e-12 else 0.5 * (cm - cp) / d
    for _ in range(iters):
        B = Bf * np.exp(-2j * np.pi * (fy * ty + fx * tx))
        R = A * np.conj(B)
        R /= np.abs(R) + 1e-9
        c = np.fft.ifft2(R).real
        iy, ix = np.unravel_index(np.argmax(c), c.shape)
        dy = iy + sub(c[(iy - 1) % h, ix], c[iy, ix], c[(iy + 1) % h, ix])
        dx = ix + sub(c[iy, (ix - 1) % w], c[iy, ix], c[iy, (ix + 1) % w])
        dy = dy - h if dy > h / 2 else dy
        dx = dx - w if dx > w / 2 else dx
        ty, tx = ty + dy, tx + dx
        peak = float(c.max())
        if abs(dy) < 0.01 and abs(dx) < 0.01:
            break
    return ty, tx, peak


_WIN: dict = {}


def _hann2(t: int) -> np.ndarray:
    if t not in _WIN:
        _WIN[t] = np.outer(np.hanning(t), np.hanning(t)).astype(np.float32)
    return _WIN[t]


_LOWPASS: dict = {}


def _lowpass(t: int, f0: float) -> np.ndarray:
    """Gaussian weight on the whitened cross-power spectrum: the finest frequencies, where noise outweighs
    detail in a real photo, would otherwise get as much say as the rest and pull the peak down."""
    key = (t, round(f0, 4))
    if key not in _LOWPASS:
        f = np.fft.fftfreq(t)
        r2 = f[:, None] ** 2 + f[None, :] ** 2
        _LOWPASS[key] = np.exp(-r2 / (f0 * f0)).astype(np.float64)
    return _LOWPASS[key]


def _phase_batch(a: np.ndarray, b: np.ndarray, iters: int = 3, f0: float = 0.22):
    """phase_shift for a stack of tiles (N, t, t) at once: (dy, dx, peak), each of length N. The whitened
    spectrum is weighted toward the lower frequencies (``f0`` cycles per pixel), which keeps the peak clear on
    noisy tiles; the iterative re-measurement removes any pull this adds."""
    n, t, _ = a.shape
    win = _hann2(t)
    lp = _lowpass(t, f0) if f0 else None
    a = a - a.mean(axis=(1, 2), keepdims=True)
    b = b - b.mean(axis=(1, 2), keepdims=True)
    A = np.fft.fft2(a * win, axes=(1, 2))
    Bf = np.fft.fft2(b * win, axes=(1, 2))
    f = np.fft.fftfreq(t)
    ty = np.zeros(n)
    tx = np.zeros(n)
    peak = np.zeros(n)
    rows = np.arange(n)
    for _ in range(iters):
        ph = np.exp(-2j * np.pi * (f[None, :, None] * ty[:, None, None] + f[None, None, :] * tx[:, None, None]))
        R = A * np.conj(Bf * ph)
        R /= np.abs(R) + 1e-9
        if lp is not None:
            R *= lp
        c = np.fft.ifft2(R, axes=(1, 2)).real
        if lp is not None:
            c /= lp.mean()                           # peak 1.0 for a perfect match, as without the weight
        flat = c.reshape(n, -1).argmax(axis=1)
        iy, ix = flat // t, flat % t
        c0 = c[rows, iy, ix]
        cym, cyp = c[rows, (iy - 1) % t, ix], c[rows, (iy + 1) % t, ix]
        cxm, cxp = c[rows, iy, (ix - 1) % t], c[rows, iy, (ix + 1) % t]
        dny, dnx = cym - 2 * c0 + cyp, cxm - 2 * c0 + cxp
        sy = np.where(np.abs(dny) < 1e-12, 0.0, 0.5 * (cym - cyp) / np.where(np.abs(dny) < 1e-12, 1.0, dny))
        sx = np.where(np.abs(dnx) < 1e-12, 0.0, 0.5 * (cxm - cxp) / np.where(np.abs(dnx) < 1e-12, 1.0, dnx))
        dy = iy + sy
        dx = ix + sx
        dy = np.where(dy > t / 2, dy - t, dy)
        dx = np.where(dx > t / 2, dx - t, dx)
        ty += dy
        tx += dx
        peak = c0
    return ty, tx, peak


def _bspline_w(u: np.ndarray):
    """Cubic B-spline weights for fractional offsets u (0..1): four taps at floor-1 .. floor+2."""
    u = u[:, None, None]
    u2, u3 = u * u, u * u * u
    return ((1 - u) ** 3 / 6.0, (4 - 6 * u2 + 3 * u3) / 6.0, (1 + 3 * u + 3 * u2 - 3 * u3) / 6.0, u3 / 6.0)


def _sample_tiles(coef: np.ndarray, pad: int, oy, ox, t: int, dy, dx) -> np.ndarray:
    """Tiles (N, t, t) of the image whose cubic-spline coefficients (padded by ``pad``) are ``coef``, taken at
    origins (oy, ox) moved by (dy, dx): exact spline interpolation, one constant shift per tile, done as two
    four-tap passes (far cheaper than map_coordinates on every pixel)."""
    py = np.asarray(oy, np.float64) + dy
    px = np.asarray(ox, np.float64) + dx
    iy, ix = np.floor(py).astype(int), np.floor(px).astype(int)
    fy, fx = (py - iy).astype(np.float32), (px - ix).astype(np.float32)
    view = np.lib.stride_tricks.sliding_window_view(coef, (t + 3, t + 3))
    blk = view[iy - 1 + pad, ix - 1 + pad]
    wy = _bspline_w(fy)
    a = wy[0] * blk[:, 0:t] + wy[1] * blk[:, 1:t + 1] + wy[2] * blk[:, 2:t + 2] + wy[3] * blk[:, 3:t + 3]
    wx = [w.transpose(0, 2, 1) for w in _bspline_w(fx)]
    return (wx[0] * a[:, :, 0:t] + wx[1] * a[:, :, 1:t + 1] + wx[2] * a[:, :, 2:t + 2] +
            wx[3] * a[:, :, 3:t + 3]).astype(np.float32)


def _lk_refine(R: np.ndarray, coef: np.ndarray, pad: int, oy, ox, dy, dx, max_shift: float, iters: int = 8):
    """Lucas-Kanade refinement of per-tile shifts (Gauss-Newton on brightness constancy, symmetric gradients,
    a Hann weight, gain and offset matched per tile so flicker does not matter). Unbiased for small residuals,
    where phase correlation with a window leans toward zero on tiles with strong repeating texture.
    Returns (dy, dx, ncc, aperture): ncc 0..1 says how well the tiles agree once lined up; aperture
    (smaller / larger gradient eigenvalue) is near 0 on a single straight edge, where only one direction can be
    measured."""
    n, t, _ = R.shape
    w = _hann2(t)
    sw = float(w.sum())
    Rz = R - (R * w).sum(axis=(1, 2), keepdims=True) / sw
    Ry, Rx = np.gradient(Rz, axis=(1, 2))
    rr = (w * Rz * Rz).sum(axis=(1, 2))
    dy = np.asarray(dy, np.float64).copy()
    dx = np.asarray(dx, np.float64).copy()
    oy = np.asarray(oy)
    ox = np.asarray(ox)
    hxx = np.zeros(n)
    hxy = np.zeros(n)
    hyy = np.zeros(n)
    idx = np.arange(n)                               # tiles still moving: only those are worked on
    for _ in range(iters):
        I = _sample_tiles(coef, pad, oy[idx], ox[idx], t, dy[idx], dx[idx])
        Iz = I - (I * w).sum(axis=(1, 2), keepdims=True) / sw
        ii = (w * Iz * Iz).sum(axis=(1, 2))
        g = np.sqrt(rr[idx] / np.maximum(ii, 1e-20)).clip(0.5, 2.0)[:, None, None]
        Iz = Iz * g
        Iy, Ix = np.gradient(Iz, axis=(1, 2))
        Gx, Gy = 0.5 * (Ix + Rx[idx]), 0.5 * (Iy + Ry[idx])
        e = Iz - Rz[idx]
        wGx, wGy = w * Gx, w * Gy
        a_xx = (wGx * Gx).sum(axis=(1, 2))
        a_xy = (wGx * Gy).sum(axis=(1, 2))
        a_yy = (wGy * Gy).sum(axis=(1, 2))
        bx = (wGx * e).sum(axis=(1, 2))
        by = (wGy * e).sum(axis=(1, 2))
        hxx[idx], hxy[idx], hyy[idx] = a_xx, a_xy, a_yy
        det = a_xx * a_yy - a_xy * a_xy
        ok = det > 1e-24
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        ddx = -(a_yy * bx - a_xy * by) * inv
        ddy = -(-a_xy * bx + a_xx * by) * inv
        step = np.hypot(ddx, ddy)
        lim = np.minimum(step, 1.0) / np.maximum(step, 1e-12)      # at most a pixel per step
        dx[idx] = np.clip(dx[idx] + np.where(ok, ddx * lim, 0.0), -max_shift, max_shift)
        dy[idx] = np.clip(dy[idx] + np.where(ok, ddy * lim, 0.0), -max_shift, max_shift)
        idx = idx[ok & (step > 0.002)]
        if idx.size == 0:
            break
    I = _sample_tiles(coef, pad, oy, ox, t, dy, dx)
    Iz = I - (I * w).sum(axis=(1, 2), keepdims=True) / sw
    ii = (w * Iz * Iz).sum(axis=(1, 2))
    ncc = (w * Iz * Rz).sum(axis=(1, 2)) / np.sqrt(np.maximum(rr * ii, 1e-30))
    tr = hxx + hyy
    disc = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4 * hxy * hxy, 0.0))
    lmin, lmax = 0.5 * (tr - disc), 0.5 * (tr + disc)
    aperture = lmin / np.maximum(lmax, 1e-30)
    return dy, dx, np.clip(ncc, 0.0, 1.0), aperture


def _grid(n: int, tile: int, step: int) -> np.ndarray:
    """Tile origins along one axis: evenly spread, first at 0, last flush with the far edge."""
    if n <= tile:
        return np.array([0])
    k = max(1, int(round((n - tile) / float(step)))) + 1
    return np.unique(np.round(np.linspace(0, n - tile, k)).astype(int))


def tile_shifts(ref: np.ndarray, img: np.ndarray, tile: int = 128, step: int = 96, max_shift: float = 24,
                clip: float | None = None, noise: float | None = None, min_texture: float = 1.0,
                start=None):
    """Shift of ``img`` against ``ref`` in tiles: (centres N x 2 [y, x], shifts N x 2 [dy, dx], quality N).

    Each tile: phase correlation for the first answer (it finds shifts of many pixels), then Lucas-Kanade for
    the exact sub-pixel value. ``start`` (dy, dx), when given, replaces the phase correlation (the caller knows
    the shift is small). Tiles too flat to measure (gradient energy under ``min_texture`` times what noise alone
    gives: texture weaker than the noise), tiles more than 2% clipped (``clip``: the clip level of the
    luminance), tiles whose answer is out of range and tiles that do not agree once lined up are skipped.
    Quality = correlation of the lined-up tiles, lowered on a lone straight edge (the aperture problem)."""
    from scipy.ndimage import gaussian_filter, spline_filter
    ref = np.asarray(ref, np.float32)
    img = np.asarray(img, np.float32)
    H, W = ref.shape
    tile = int(min(tile, H, W)) // 2 * 2
    if tile < 8:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    ys, xs = _grid(H, tile, step), _grid(W, tile, step)
    oy, ox = np.meshgrid(ys, xs, indexing="ij")
    oy, ox = oy.ravel(), ox.ravel()
    sig = noise_sigma(ref) if noise is None else float(noise)
    floor = min_texture * 4.0 * sig * sig            # mean squared gradient that pure noise gives is 4 sigma^2
    # Broad shading inside a tile (a sky gradient, a vignette) is taken out first: it does not help find a
    # shift and lets a brightness change pass for one. A light blur keeps the gradients out of the pixel noise.
    hs = tile / 12.0
    hr = gaussian_filter(highpass(ref, hs), 0.7)
    hi = gaussian_filter(highpass(img, hs), 0.7)
    pad = int(np.ceil(max_shift)) + 4
    coef = spline_filter(np.pad(hi, pad, mode="reflect"), 3, output=np.float32, mode="mirror")
    view_r = np.lib.stride_tricks.sliding_window_view(hr, (tile, tile))
    view_i = np.lib.stride_tricks.sliding_window_view(hi, (tile, tile))
    view_raw = np.lib.stride_tricks.sliding_window_view(ref, (tile, tile))
    view_rawi = np.lib.stride_tricks.sliding_window_view(img, (tile, tile))
    chunk = max(16, int(2_000_000 // (tile * tile)))
    cen, sh, qq = [], [], []
    for s in range(0, oy.size, chunk):
        by, bx = oy[s:s + chunk], ox[s:s + chunk]
        rawt = view_raw[by, bx]
        gy = np.diff(rawt, axis=1)[:, :, :-1]
        gx = np.diff(rawt, axis=2)[:, :-1, :]
        energy = (gy * gy + gx * gx).mean(axis=(1, 2))
        ok = energy > floor
        if clip is not None:
            ok &= ((rawt >= clip).mean(axis=(1, 2)) < 0.02) & ((view_rawi[by, bx] >= clip).mean(axis=(1, 2)) < 0.02)
        if not np.any(ok):
            continue
        by, bx = by[ok], bx[ok]
        ta = np.ascontiguousarray(view_r[by, bx])
        if start is None:
            tb = np.ascontiguousarray(view_i[by, bx])
            d0y, d0x, _ = _phase_batch(tb, ta, iters=2)
        else:
            d0y = np.full(by.size, float(start[0]))
            d0x = np.full(by.size, float(start[1]))
        inr = (np.abs(d0y) <= max_shift) & (np.abs(d0x) <= max_shift) & np.isfinite(d0y) & np.isfinite(d0x)
        if not np.any(inr):
            continue
        by, bx, ta, d0y, d0x = by[inr], bx[inr], ta[inr], d0y[inr], d0x[inr]
        dy, dx, ncc, ap = _lk_refine(ta, coef, pad, by, bx, d0y, d0x, max_shift)
        q = ncc * np.minimum(1.0, 4.0 * ap)
        good = (np.abs(dy) < max_shift) & (np.abs(dx) < max_shift) & (ncc > 0.2) & np.isfinite(dy)
        cen.append(np.stack([by[good] + (tile - 1) / 2.0, bx[good] + (tile - 1) / 2.0], 1))
        sh.append(np.stack([dy[good], dx[good]], 1))
        qq.append(q[good])
    if not cen:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    cen, sh, qq = np.concatenate(cen), np.concatenate(sh), np.concatenate(qq)
    if qq.size >= 4:                                 # far below the others: a tile fooled by noise or motion
        keep = qq >= 0.3 * float(np.median(qq))
        cen, sh, qq = cen[keep], sh[keep], qq[keep]
    return cen, sh, qq


# ------------------------------------------------------------------------------------------ transforms
@dataclass
class Transform:
    """Maps reference pixel coordinates to image coordinates: [x', y', w'] = matrix @ [x, y, 1]."""
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    kind: str = "shift"
    residual_px: float = 0.0
    tiles_used: int = 0
    outliers: float = 0.0       # share of tiles at the last level that disagree (something moved there)

    @staticmethod
    def identity(kind: str = "shift") -> "Transform":
        return Transform(np.eye(3), kind)

    @staticmethod
    def translation(dy: float, dx: float) -> "Transform":
        m = np.eye(3)
        m[0, 2], m[1, 2] = dx, dy
        return Transform(m, "shift")

    def apply(self, ys, xs):
        """(ys', xs') for arrays of points."""
        m = self.matrix
        ys = np.asarray(ys, np.float64)
        xs = np.asarray(xs, np.float64)
        X = m[0, 0] * xs + m[0, 1] * ys + m[0, 2]
        Y = m[1, 0] * xs + m[1, 1] * ys + m[1, 2]
        if self.kind == "homography":
            Wd = m[2, 0] * xs + m[2, 1] * ys + m[2, 2]
            X, Y = X / Wd, Y / Wd
        return Y, X

    def compose(self, other: "Transform") -> "Transform":
        """self after other: p -> self(other(p))."""
        kinds = [self.kind, other.kind]
        kind = max(kinds, key=KINDS.index)
        return Transform(self.matrix @ other.matrix, kind, self.residual_px, self.tiles_used, self.outliers)

    def inverse(self) -> "Transform":
        return Transform(np.linalg.inv(self.matrix), self.kind, self.residual_px, self.tiles_used, self.outliers)

    def rescaled(self, k: float) -> "Transform":
        """This transform (measured on a copy binned by k) in full-resolution pixel coordinates."""
        if k == 1:
            return Transform(self.matrix.copy(), self.kind, self.residual_px, self.tiles_used, self.outliers)
        o = (k - 1) / 2.0
        A = np.array([[k, 0, o], [0, k, o], [0, 0, 1.0]])
        return Transform(A @ self.matrix @ np.linalg.inv(A), self.kind, self.residual_px * k, self.tiles_used,
                         self.outliers)

    @property
    def shift(self) -> tuple[float, float]:
        """(dy, dx) where the frame centre's content moved, for reports: the translation part."""
        return float(self.matrix[1, 2]), float(self.matrix[0, 2])

    def scale(self) -> float:
        """Magnification (geometric mean of the linear part's singular values)."""
        return float(np.sqrt(abs(np.linalg.det(self.matrix[:2, :2]))))

    def angle_deg(self) -> float:
        m = self.matrix
        return float(np.degrees(np.arctan2(m[1, 0] - m[0, 1], m[0, 0] + m[1, 1])))

    def displacement(self, shape) -> float:
        """Largest movement of a frame corner, in pixels (how far this transform is from doing nothing)."""
        h, w = shape[:2]
        ys = np.array([0, 0, h - 1, h - 1], float)
        xs = np.array([0, w - 1, 0, w - 1], float)
        Y, X = self.apply(ys, xs)
        return float(np.max(np.hypot(Y - ys, X - xs)))

    def to_dict(self) -> dict:
        return {"matrix": np.round(self.matrix, 10).tolist(), "kind": self.kind,
                "residual_px": round(float(self.residual_px), 4), "tiles_used": int(self.tiles_used)}

    @staticmethod
    def from_dict(d: dict) -> "Transform":
        return Transform(np.asarray(d["matrix"], float), d.get("kind", "similarity"),
                         float(d.get("residual_px", 0.0)), int(d.get("tiles_used", 0)))


def _solve(rows: np.ndarray, rhs: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(np.maximum(w, 1e-12))
    sol, *_ = np.linalg.lstsq(rows * sw[:, None], rhs * sw, rcond=None)
    return sol


def _fit_points(p: np.ndarray, q: np.ndarray, w: np.ndarray, kind: str) -> np.ndarray:
    """Matrix taking points p (N x 2, [x, y]) to q, by weighted least squares, on normalised coordinates."""
    m = np.average(p, axis=0, weights=w)
    s = float(np.sqrt(np.average(np.sum((p - m) ** 2, axis=1), weights=w))) or 1.0
    N = np.array([[1 / s, 0, -m[0] / s], [0, 1 / s, -m[1] / s], [0, 0, 1.0]])
    pn = (p - m) / s
    qn = (q - m) / s
    x, y = pn[:, 0], pn[:, 1]
    X, Y = qn[:, 0], qn[:, 1]
    one, zero = np.ones_like(x), np.zeros_like(x)
    ww = np.concatenate([w, w])
    if kind == "shift":
        t = np.average(qn - pn, axis=0, weights=w)
        Mn = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1.0]])
    elif kind == "similarity":
        rows = np.concatenate([np.stack([x, -y, one, zero], 1), np.stack([y, x, zero, one], 1)])
        a, b, tx, ty = _solve(rows, np.concatenate([X, Y]), ww)
        Mn = np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]])
    elif kind == "affine":
        rows = np.concatenate([np.stack([x, y, one, zero, zero, zero], 1),
                               np.stack([zero, zero, zero, x, y, one], 1)])
        a, b, c, d, e, f = _solve(rows, np.concatenate([X, Y]), ww)
        Mn = np.array([[a, b, c], [d, e, f], [0, 0, 1.0]])
    elif kind == "homography":
        rows = np.concatenate([np.stack([x, y, one, zero, zero, zero, -x * X, -y * X], 1),
                               np.stack([zero, zero, zero, x, y, one, -x * Y, -y * Y], 1)])
        h = _solve(rows, np.concatenate([X, Y]), ww)
        Mn = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])
    else:
        raise ValueError(f"kind must be one of {KINDS}")
    M = np.linalg.inv(N) @ Mn @ N
    return M / M[2, 2]


def fit(centres, shifts, quality, kind: str = "similarity", rounds: int = 3) -> Transform:
    """Fit a transform to tile measurements (centres [y, x] in the reference, shifts [dy, dx]): weighted least
    squares, then ``rounds`` rounds of dropping tiles whose residual is over 3x the median (something moved
    there, or the tile was fooled by noise)."""
    centres = np.asarray(centres, float).reshape(-1, 2)
    shifts = np.asarray(shifts, float).reshape(-1, 2)
    w = np.asarray(quality, float).reshape(-1)
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if centres.shape[0] == 0:
        return Transform.identity(kind)
    p = centres[:, ::-1]
    q = p + shifts[:, ::-1]
    need = _NEED[kind]
    if centres.shape[0] < 2 * need:                 # too few tiles for this model: fall back to a plain shift
        kind = "shift" if centres.shape[0] < 2 * _NEED["similarity"] else "similarity"
        need = _NEED[kind]
    keep = np.ones(len(p), bool)
    M = _fit_points(p, q, w, kind)
    # A few rounds of reweighting first (Cauchy weights), so that a quarter of the tiles on something that moved
    # cannot drag the starting fit far enough to look normal to the trimming below.
    for _ in range(6):
        r = _residuals(M, p, q, kind)
        c = max(1.5 * float(np.median(r)), 0.02)
        M = _fit_points(p, q, w / (1.0 + (r / c) ** 2), kind)
    for _ in range(rounds):
        r = _residuals(M, p, q, kind)
        med = float(np.median(r[keep]))
        nk = keep & (r <= max(3.0 * med, 0.02))
        if nk.sum() < 2 * need or nk.sum() == keep.sum():
            break
        keep = nk
        M = _fit_points(p[keep], q[keep], w[keep], kind)
    r = _residuals(M, p[keep], q[keep], kind)
    rms = float(np.sqrt(np.average(r * r, weights=w[keep])))
    return Transform(M, kind, rms, int(keep.sum()))


def _residuals(M, p, q, kind) -> np.ndarray:
    T = Transform(M, kind)
    Y, X = T.apply(p[:, 1], p[:, 0])
    return np.hypot(X - q[:, 0], Y - q[:, 1])


def tile_mask(centres, shifts, transform: Transform) -> np.ndarray:
    """Residual of each tile against a fitted transform, in pixels (for 'did not line up' reports)."""
    centres = np.asarray(centres, float).reshape(-1, 2)
    if centres.shape[0] == 0:
        return np.zeros(0)
    p = centres[:, ::-1]
    q = p + np.asarray(shifts, float).reshape(-1, 2)[:, ::-1]
    return _residuals(transform.matrix, p, q, transform.kind)


# ------------------------------------------------------------------------------------------ registration
def _level_tile(shape, tile: int, step: int) -> tuple[int, int]:
    t = int(min(tile, min(shape) // 2)) // 2 * 2
    t = max(t, 16)
    return t, max(8, int(round(t * step / float(tile))))


def register(ref_luma: np.ndarray, img_luma: np.ndarray, kind: str = "similarity", levels: int = 3,
             tile: int = 128, step: int = 96, clip: float | None = None, init: Transform | None = None,
             cancel=None) -> Transform:
    """Transform taking reference coordinates to ``img_luma`` coordinates (both analysis copies of one size).

    Coarse to fine: at 1/2**levels size one whole-frame phase correlation (skipped when ``init`` is given); then at
    each finer size the image is warped by the estimate so far, tiles are measured on what remains, the model is
    fitted and composed. Two passes at full analysis size."""
    ref = np.asarray(ref_luma, np.float32)
    img = np.asarray(img_luma, np.float32)
    if ref.shape != img.shape:
        h, w = min(ref.shape[0], img.shape[0]), min(ref.shape[1], img.shape[1])
        ref, img = ref[:h, :w], img[:h, :w]
    while levels > 0 and min(ref.shape) // (2 ** levels) < 48:
        levels -= 1
    pr, pi = [ref], [img]
    for _ in range(levels):
        pr.append(bin_by(pr[-1], 2))
        pi.append(bin_by(pi[-1], 2))
    k_top = 2 ** levels
    if init is not None:
        T = Transform(init.matrix.copy(), init.kind)
    else:
        a, b = pi[-1], pr[-1]
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        hs = min(h, w) / 12.0
        dy, dx, _ = phase_shift(highpass(a[:h, :w], hs), highpass(b[:h, :w], hs))
        T = Transform.translation(dy, dx).rescaled(k_top)
        T.kind = "shift"
    noise = [noise_sigma(x) for x in pr]
    final_small = False
    confirmed = False                                # did any tile, at any size, measure something?
    schedule = list(range(levels - 1, -1, -1)) + [0] if levels else [0, 0]
    if init is not None:
        schedule = list(range(levels, -1, -1)) + [0]
    for lv in schedule:
        if cancel is not None and cancel.is_set():
            break
        k = 2 ** lv
        Tl = T
        if k > 1:                                    # full -> level coordinates
            o = (k - 1) / 2.0
            A = np.array([[k, 0, o], [0, k, o], [0, 0, 1.0]])
            Tl = Transform(np.linalg.inv(A) @ T.matrix @ A, T.kind)
        refl, imgl = pr[lv], pi[lv]
        t, s = _level_tile(refl.shape, tile, step)
        warped = warp(imgl, Tl, refl.shape, order=1)
        if lv == 0 and final_small:                  # the estimate is already within a pixel: Lucas-Kanade only
            cen, sh, q = tile_shifts(refl, warped, t, t, max_shift=t / 4.0, clip=clip, noise=noise[lv],
                                     start=(0.0, 0.0))
        else:
            cen, sh, q = tile_shifts(refl, warped, t, s, max_shift=t / 4.0, clip=clip, noise=noise[lv])
        if cen.shape[0] == 0:
            continue
        confirmed = True
        need = _NEED[kind]
        if cen.shape[0] < 2 * need:                 # not enough for the model at this size: refine the shift only
            d = np.average(sh, axis=0, weights=q)
            Tl = Tl.compose(Transform.translation(d[0], d[1]))
        else:
            ys, xs = Tl.apply(cen[:, 0] + sh[:, 0], cen[:, 1] + sh[:, 1])
            target = np.stack([ys, xs], 1)
            Tl = fit(cen, target - cen, q, kind)
            if lv == 0:
                r = tile_mask(cen, target - cen, Tl)
                Tl.outliers = float(np.mean(r > max(1.0, 4.0 * float(np.median(r))))) if r.size else 0.0
        if k > 1:
            o = (k - 1) / 2.0
            A = np.array([[k, 0, o], [0, k, o], [0, 0, 1.0]])
            T = Transform(A @ Tl.matrix @ np.linalg.inv(A), Tl.kind, Tl.residual_px * k, Tl.tiles_used,
                          Tl.outliers)
        else:
            T = Tl
        if lv <= 1 and cen.shape[0] >= 2 * _NEED[kind]:
            final_small = float(np.median(np.hypot(sh[:, 0], sh[:, 1]))) * k < 1.0
    if not confirmed and init is None:
        # Nothing in the frame could be measured (a featureless card, fog, a lens cap): the whole-frame phase
        # correlation then only found the strongest noise peak. Not moving is the honest answer.
        log.debug("register: nothing measurable at any size; keeping the frame where it is")
        return Transform.identity("shift")
    log.debug("register %s: scale %.5f angle %.4f shift %s residual %.3f px (%d tiles)", T.kind, T.scale(),
              T.angle_deg(), np.round(T.shift, 3).tolist(), T.residual_px, T.tiles_used)
    return T


# ------------------------------------------------------------------------------------------ local flow
@dataclass
class Flow:
    """Local displacement left after global registration, on a coarse grid (cheap to keep for 45 MP frames).
    ``dy[i, j], dx[i, j]`` at the point (y0 + i * step, x0 + j * step): content at p in the reference lies at
    p + (dy, dx) in the (globally warped) frame. ``quality`` 0..1 per grid point (0 = filled from neighbours)."""
    dy: np.ndarray
    dx: np.ndarray
    quality: np.ndarray
    y0: float
    x0: float
    step: float

    def at(self, ys, xs):
        """(dy, dx) at arbitrary points, bilinear between grid points."""
        from scipy.ndimage import map_coordinates
        gy = (np.asarray(ys, np.float64) - self.y0) / self.step
        gx = (np.asarray(xs, np.float64) - self.x0) / self.step
        c = [gy, gx]
        return (map_coordinates(self.dy, c, order=1, mode="nearest"),
                map_coordinates(self.dx, c, order=1, mode="nearest"))

    def fields(self, shape):
        """Dense (dy, dx, quality) at every pixel of an image of ``shape``."""
        from scipy.ndimage import map_coordinates
        h, w = shape[:2]
        gy = (np.arange(h) - self.y0) / self.step
        gx = (np.arange(w) - self.x0) / self.step
        Y, X = np.meshgrid(gy, gx, indexing="ij")
        c = [Y, X]
        return tuple(map_coordinates(f, c, order=1, mode="nearest").astype(np.float32)
                     for f in (self.dy, self.dx, self.quality))

    def rescaled(self, k: float) -> "Flow":
        o = (k - 1) / 2.0
        return Flow(self.dy * k, self.dx * k, self.quality, self.y0 * k + o, self.x0 * k + o, self.step * k)

    def magnitude(self) -> float:
        """Median local movement in pixels (for reports)."""
        return float(np.median(np.hypot(self.dy, self.dx)))


def flow(ref_luma: np.ndarray, img_luma: np.ndarray, tile: int = 64, step: int = 32, max_shift: float = 6,
         clip: float | None = None) -> Flow:
    """Dense local displacement of ``img_luma`` (already warped by its global transform) against the reference:
    tile shifts on a fine grid, a 3 x 3 median on the grid (one fooled tile cannot pull its neighbours), tiles
    that could not be measured filled from their neighbours. For hand shake, rolling shutter and air wobble."""
    from scipy.ndimage import gaussian_filter, median_filter
    ref = np.asarray(ref_luma, np.float32)
    img = np.asarray(img_luma, np.float32)
    H, W = ref.shape
    tile = int(min(tile, H, W)) // 2 * 2
    ys, xs = _grid(H, tile, step), _grid(W, tile, step)
    gy, gx = len(ys), len(xs)
    dy = np.full((gy, gx), np.nan)
    dx = np.full((gy, gx), np.nan)
    q = np.zeros((gy, gx))
    cen, sh, qq = tile_shifts(ref, img, tile, step, max_shift, clip=clip)
    if cen.shape[0]:
        iy = np.searchsorted(ys, np.round(cen[:, 0] - (tile - 1) / 2.0).astype(int))
        ix = np.searchsorted(xs, np.round(cen[:, 1] - (tile - 1) / 2.0).astype(int))
        iy, ix = np.clip(iy, 0, gy - 1), np.clip(ix, 0, gx - 1)
        dy[iy, ix], dx[iy, ix], q[iy, ix] = sh[:, 0], sh[:, 1], qq
    valid = np.isfinite(dy)
    if valid.sum() >= 3:                             # median of the valid neighbours (3 x 3)
        big = 1e6
        for f in (dy, dx):
            g = np.where(valid, f, np.nan)
            lo = median_filter(np.where(valid, g, -big), 3, mode="nearest")
            hi = median_filter(np.where(valid, g, big), 3, mode="nearest")
            med = np.where(np.abs(lo) < big / 2, lo, 0.0)
            med = np.where(np.abs(hi) < big / 2, 0.5 * (med + hi), med)
            f[valid] = med[valid]
    wgt = np.where(valid, np.maximum(q, 1e-3), 0.0)
    if not np.any(valid):
        dy[:] = 0.0
        dx[:] = 0.0
    else:
        fy, fx = np.where(valid, dy, 0.0), np.where(valid, dx, 0.0)
        s = 1.0
        while True:                                  # normalised convolution until every point has a value
            den = gaussian_filter(wgt, s, mode="nearest")
            ny = gaussian_filter(fy * wgt, s, mode="nearest") / np.maximum(den, 1e-12)
            nx = gaussian_filter(fx * wgt, s, mode="nearest") / np.maximum(den, 1e-12)
            hole = ~valid & (den > 1e-6)
            dy[hole], dx[hole] = ny[hole], nx[hole]
            if np.all(np.isfinite(dy)) or s > max(gy, gx):
                break
            s *= 2.0
        dy = np.nan_to_num(dy)
        dx = np.nan_to_num(dx)
    c0 = (tile - 1) / 2.0
    stp = float(np.median(np.diff(ys))) if gy > 1 else float(step)
    stx = float(np.median(np.diff(xs))) if gx > 1 else float(step)
    return Flow(dy.astype(np.float32), dx.astype(np.float32), (q * valid).astype(np.float32), ys[0] + c0,
                xs[0] + c0, float(0.5 * (stp + stx)))


# ------------------------------------------------------------------------------------------ warping
def warp(img: np.ndarray, transform: Transform, out_shape=None, order: int = 3, flw: Flow | None = None,
         rows: int = 256) -> np.ndarray:
    """``img`` resampled into the reference frame: out[p] = img[T(p + flow(p))]. Order 3 (cubic spline) for
    results, order 1 for analysis copies. Worked in strips, in parallel; float32 out."""
    from scipy.ndimage import map_coordinates, spline_filter
    img = np.asarray(img, np.float32)
    H, W = (img.shape[:2] if out_shape is None else tuple(out_shape[:2]))
    chans = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]
    if order > 1:
        mode = "mirror"
        chans = [spline_filter(c, order, output=np.float32, mode=mode) for c in chans]
    else:
        mode = "nearest"
    out = np.empty((H, W) if img.ndim == 2 else (H, W, img.shape[2]), np.float32)
    xs = np.arange(W, dtype=np.float64)

    def job(y0, y1):
        Y, X = np.meshgrid(np.arange(y0, y1, dtype=np.float64), xs, indexing="ij")
        if flw is not None:
            fdy, fdx = flw.at(Y, X)
            Y = Y + fdy
            X = X + fdx
        sy, sx = transform.apply(Y, X)
        c = [sy, sx]
        for i, ch in enumerate(chans):
            dst = out[y0:y1] if img.ndim == 2 else out[y0:y1, :, i]
            r = map_coordinates(ch, c, order=order, mode=mode, prefilter=False, output=np.float32)
            dst[...] = r
    strips(H, rows, job)
    return out


def coverage(transform: Transform, src_shape, out_shape) -> np.ndarray:
    """Boolean map of the output pixels whose source point falls inside the source frame (the common area)."""
    H, W = out_shape[:2]
    h, w = src_shape[:2]
    Y, X = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    sy, sx = transform.apply(Y, X)
    return (sy >= 0) & (sy <= h - 1) & (sx >= 0) & (sx <= w - 1)


def common_box(transforms, src_shape, out_shape, margin: int = 2) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1): the largest axis-aligned box of the output where every frame has data, found from the
    frames' corner paths (cheap: no full-size masks). A small margin keeps interpolation away from the edge."""
    H, W = out_shape[:2]
    h, w = src_shape[:2]
    y0, y1, x0, x1 = 0.0, H - 1.0, 0.0, W - 1.0
    for T in transforms:
        inv = T.inverse()
        # where the source frame's edges land in the output: sample each edge densely
        n = 64
        top = inv.apply(np.zeros(n), np.linspace(0, w - 1, n))
        bot = inv.apply(np.full(n, h - 1.0), np.linspace(0, w - 1, n))
        lef = inv.apply(np.linspace(0, h - 1, n), np.zeros(n))
        rig = inv.apply(np.linspace(0, h - 1, n), np.full(n, w - 1.0))
        y0 = max(y0, float(np.max(top[0])))
        y1 = min(y1, float(np.min(bot[0])))
        x0 = max(x0, float(np.max(lef[1])))
        x1 = min(x1, float(np.min(rig[1])))
    iy0, ix0 = int(np.ceil(y0)) + margin, int(np.ceil(x0)) + margin
    iy1, ix1 = int(np.floor(y1)) - margin + 1, int(np.floor(x1)) - margin + 1
    if iy1 - iy0 < 16 or ix1 - ix0 < 16:
        return 0, H, 0, W
    return iy0, iy1, ix0, ix1
