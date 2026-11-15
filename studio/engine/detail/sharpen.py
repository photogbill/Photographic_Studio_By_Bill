"""Stage 12: capture sharpening by deconvolution (BUILD_PLAN.md section 10.5, section 21 change 6).

"Recover lens sharpness": undo part of the blur the lens, the anti-aliasing filter and the demosaic put into
the raw. It works in scene-linear light, on luminance only, before any tone curve, where blur is a physical
convolution and deconvolution is meaningful.

1. The blur is measured from the photo (no lens data needed, so it works for contactless vintage lenses):
   at strong edges, the gradient of the image is compared with the gradient of the same image blurred again
   by a known Gaussian (Zhuo and Sim, Pattern Recognition 44(9), 2011). For a blurred step edge the ratio R
   gives the blur directly: sigma^2 = s0^2 / (R^2 - 1) - sd^2. The sharpest edges in the most detailed tiles
   (the in-focus subject) set the capture blur; defocused areas are not what this stage is for.
2. Deconvolve to a target, not to infinity (section 21 change 6, from the STARRED / Firedec astronomy work):
   the photo is deconvolved only by the kernel that turns the target blur into the measured blur,
   sigma_k^2 = sigma^2 - sigma_t^2. The amount slider moves the target from "as shot" toward 0.5 px.
   A finite target is what keeps ringing away.
3. Richardson-Lucy iterations (the method used on Hubble images before the optics were repaired), run on
   luminance plus a small pedestal so the darkest, noisiest pixels behave.
4. Three safety nets: a noise gate (no sharpening where the local gradient is not clearly above the camera's
   noise at that brightness), a deringing clamp (the result may not overshoot the local 3x3 min / max of the
   original by more than a few percent of the local range: candle flames and catchlights keep clean edges),
   and the luminance change is applied as a bounded ratio to R, G, B so colours do not shift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

from ... import log as slog

log = slog.get("engine.sharpen")

LUMA = np.array([0.2880402, 0.7118741, 0.0000857], np.float32)    # linear ProPhoto Y weights
S0 = 1.0          # re-blur used to measure the blur
SD = 0.5          # derivative-of-Gaussian scale used for gradients
SD_EFF = 0.68     # its effective width on the pixel grid, ratio method (calibrated on synthetic edges)
SD_EFF_P = 0.66   # the same for the profile method
MIN_TARGET = 0.5  # px: the target blur at amount 1 (about the sampling limit)
DARK_SLACK = 0.94 # a dark pixel may end at most 6% (0.09 stop) below its darkest 3x3 neighbour
BRIGHT_SLACK = 0.04


@dataclass
class Blur:
    sigma: float          # measured capture blur (Gaussian sigma, full-resolution pixels)
    edges: int            # edge points used
    spread: float         # interquartile spread of the estimates (px)
    confidence: str       # good | fair | default
    source: str = "measured from this photo"


DEFAULT_BLUR = Blur(sigma=0.9, edges=0, spread=0.0, confidence="default", source="typical value (too few edges)")


# ------------------------------------------------------------------------------------------ measuring blur
_RAMP = {}


def _grad(y: np.ndarray, s: float):
    """Gradient by derivative-of-Gaussian, normalised so a unit ramp reads 1 (at sigma 0.5 the sampled kernel
    alone reads 0.86)."""
    if s not in _RAMP:
        ramp = np.tile(np.arange(64, dtype=np.float32), (16, 1))
        _RAMP[s] = float(gaussian_filter(ramp, s, order=(0, 1))[8, 32])
    k = np.float32(1.0 / _RAMP[s])
    gx = gaussian_filter(y, s, order=(0, 1)) * k
    gy = gaussian_filter(y, s, order=(1, 0)) * k
    return np.hypot(gx, gy), gx, gy


_NOISE_GRAD = None


def _noise_grad_factor() -> float:
    """Typical gradient magnitude (at SD) produced by unit white noise: sets the 'well above noise' threshold."""
    global _NOISE_GRAD
    if _NOISE_GRAD is None:
        n = np.random.default_rng(11).standard_normal((256, 256)).astype(np.float32)
        _NOISE_GRAD = float(np.median(_grad(n, SD)[0]))
    return _NOISE_GRAD


def edge_blur_samples(y: np.ndarray, noise_sigma: float = 0.0, max_points: int = 4000, parts: bool = False):
    """Per-edge-point blur estimates (Gaussian sigma, px) at strong, isolated step edges of a linear luminance tile.

    Two independent estimates must agree at each point: the gradient-ratio method (Zhuo and Sim 2011) and the
    contrast / peak-gradient ratio of the edge profile (a blurred step of contrast C has a peak gradient of
    C / (sqrt(2 pi) sigma)). The profile must have flat plateaus on both sides, which rejects texture, thin lines
    and noise, the things that make the ratio method alone read "too sharp"."""
    from scipy.ndimage import map_coordinates
    y = np.asarray(y, np.float32)
    if min(y.shape) < 32:
        return np.zeros(0)
    g, gx, gy = _grad(y, SD)
    g1, _, _ = _grad(gaussian_filter(y, S0), SD)
    ang = (np.round(np.arctan2(gy, gx) / (np.pi / 4)) % 4).astype(np.int8)
    nms = np.zeros_like(g, bool)
    c = g[1:-1, 1:-1]
    for k, (dy, dx) in enumerate(((0, 1), (1, 1), (1, 0), (1, -1))):
        a = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        b = g[1 - dy:g.shape[0] - 1 - dy, 1 - dx:g.shape[1] - 1 - dx]
        nms[1:-1, 1:-1] |= (ang[1:-1, 1:-1] == k) & (c >= a) & (c >= b)
    thr = max(float(np.percentile(g, 95)), 8.0 * noise_sigma * _noise_grad_factor(), 1e-6)
    ok = nms & (g > thr)
    ok[:10] = ok[-10:] = False
    ok[:, :10] = ok[:, -10:] = False
    ys, xs = np.nonzero(ok)
    if ys.size == 0:
        return np.zeros(0)
    if ys.size > max_points:
        sel = np.argsort(-g[ys, xs])[:max_points]
        ys, xs = ys[sel], xs[sel]
    nx, ny = gx[ys, xs] / g[ys, xs], gy[ys, xs] / g[ys, xs]

    def at(o):                     # mean of 5 samples along the edge (tangent), o px along the normal
        return sum(map_coordinates(y, [ys + o * ny + t * nx, xs + o * nx - t * ny], order=1, mode="nearest")
                   for t in (-2, -1, 0, 1, 2)) / 5.0
    prof = {o: at(o) for o in (-6, -5, -4, -3, 3, 4, 5, 6)}
    left = (prof[-6] + prof[-5] + prof[-4]) / 3.0
    right = (prof[4] + prof[5] + prof[6]) / 3.0
    C = right - left
    tol = 0.2 * C + 2.0 * noise_sigma * math.sqrt(2.0 / 5.0)
    flat = (np.abs(prof[-3] - prof[-6]) < tol) & (np.abs(prof[3] - prof[6]) < tol)
    unclipped = np.maximum.reduce(list(prof.values())) < 0.95
    good = (C > 0) & flat & unclipped
    if not good.any():
        return np.zeros(0)
    gp, g1p, Cg = g[ys, xs][good], g1[ys, xs][good], C[good]
    r = gp / np.maximum(g1p, 1e-12)
    a_ratio = np.where(r > 1.02, S0 * S0 / np.maximum(r * r - 1.0, 1e-6), 25.0)
    a_prof = (Cg / (math.sqrt(2.0 * math.pi) * gp)) ** 2
    if parts:
        return a_ratio, a_prof
    s2_ratio = a_ratio - SD_EFF * SD_EFF
    s2_prof = a_prof - SD_EFF_P * SD_EFF_P
    sr, sp = np.sqrt(np.clip(s2_ratio, 0.0, 25.0)), np.sqrt(np.clip(s2_prof, 0.0, 25.0))
    agree = np.abs(sr - sp) < 0.3 * np.maximum(sr, sp) + 0.15
    est = 0.5 * (sr + sp)
    # below 0.3 px is sharper than a sampled, demosaiced image can be: specular points, defects, aliasing
    return est[agree & (est >= 0.3)]


def summarise_blur(samples: np.ndarray, min_edges: int = 40) -> Blur:
    if samples.size < min_edges:
        return DEFAULT_BLUR
    q25, q50, q75 = np.percentile(samples, [25, 50, 75])
    # the sharper half of the clean step edges = the in-focus subject
    sigma = float(np.clip(np.percentile(samples, 40), 0.35, 3.0))
    conf = "good" if samples.size >= 200 and (q75 - q25) < 0.8 * q50 else "fair"
    return Blur(sigma=round(sigma, 3), edges=int(samples.size), spread=round(float(q75 - q25), 3), confidence=conf)


def select_tiles(y_small: np.ndarray, factor: int, n: int = 8, tile: int = 384) -> list:
    """Pick the n most detailed, unclipped tiles (full-resolution boxes x, y, w, h) from a reduced luminance image
    (``factor`` full-res pixels per small pixel)."""
    t = max(4, tile // factor)
    h, w = y_small.shape
    if h < t or w < t:
        return []
    g, _, _ = _grad(np.sqrt(np.maximum(y_small, 0)), 1.0)
    cand = []
    for y0 in range(0, h - t + 1, t // 2):
        for x0 in range(0, w - t + 1, t // 2):
            blk = y_small[y0:y0 + t, x0:x0 + t]
            if blk.max() > 0.95 or blk.mean() < 0.002:
                continue
            cand.append((float(g[y0:y0 + t, x0:x0 + t].mean()), y0, x0))
    cand.sort(reverse=True)
    out, used = [], []
    for _, y0, x0 in cand:
        if any(abs(y0 - a) < t and abs(x0 - b) < t for a, b in used):
            continue
        used.append((y0, x0))
        out.append((x0 * factor, y0 * factor, tile, tile))
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------------------------------------------ deconvolution
@dataclass
class SharpenParams:
    amount: float = 0.5          # 0 = none, 1 = deconvolve to the 0.5 px target, up to 1.5
    sigma: float = 0.9           # capture blur to undo (px)
    iterations: int = 0          # 0 = automatic (more for bigger kernels)
    threshold: float = 1.0       # noise gate multiplier (higher = only crisp edges are sharpened)


def target_kernel(sigma_blur: float, amount: float) -> float:
    """Gaussian sigma of the kernel that is deconvolved (blur -> target)."""
    amount = float(np.clip(amount, 0.0, 1.5))
    tgt = sigma_blur * (1.0 - min(amount, 1.0)) + MIN_TARGET * min(amount, 1.0)
    if amount > 1.0:
        tgt = max(0.3, MIN_TARGET - (amount - 1.0) * 0.4)
    tgt = min(tgt, sigma_blur)
    return math.sqrt(max(sigma_blur ** 2 - tgt ** 2, 0.0))


def richardson_lucy(y: np.ndarray, sk: float, iterations: int, pedestal: float) -> np.ndarray:
    obs = np.maximum(y, 0.0) + np.float32(pedestal)
    est = obs.copy()
    for _ in range(iterations):
        conv = gaussian_filter(est, sk, truncate=3.0)
        ratio = obs / np.maximum(conv, 1e-12)
        est *= gaussian_filter(ratio, sk, truncate=3.0)
        np.clip(est, 0.0, 256.0, out=est)              # fixed bound: strips must not depend on content
    return est - np.float32(pedestal)


def capture_sharpen(rgb: np.ndarray, p: SharpenParams, noise_sigma_lin=None, rows: int = 512) -> np.ndarray:
    """Deconvolve scene-linear working RGB (H x W x 3). noise_sigma_lin(Y) -> luminance noise std (same units).
    Large frames run in overlapping strips on all cores; the result is the same as one piece."""
    sk = target_kernel(p.sigma, p.amount)
    it = int(p.iterations) if p.iterations > 0 else int(np.clip(round(6 + 8 * sk), 6, 24))
    if sk < 0.2:
        return rgb
    from ..parallel import strips
    with slog.stage("sharpen.deconvolve", log, sigma=round(p.sigma, 3), kernel=round(sk, 3), it=it,
                    mp=round(rgb.shape[0] * rgb.shape[1] / 1e6, 1)):
        h = rgb.shape[0]
        if h <= 2 * rows:
            return _sharpen_block(rgb, p, noise_sigma_lin, sk, it)
        out = np.empty_like(rgb, dtype=np.float32)
        pad = 32

        def job(y0, y1):
            a, b = max(0, y0 - pad), min(h, y1 + pad)
            out[y0:y1] = _sharpen_block(rgb[a:b], p, noise_sigma_lin, sk, it)[y0 - a:y0 - a + (y1 - y0)]
        strips(h, rows, job)
        return out


def _sharpen_block(rgb: np.ndarray, p: SharpenParams, noise_sigma_lin, sk: float, it: int) -> np.ndarray:
    Y = np.maximum(rgb @ LUMA, 0.0).astype(np.float32)
    nfloor = float(noise_sigma_lin(np.float32(0.0))) if noise_sigma_lin else 1e-4
    est = richardson_lucy(Y, sk, it, pedestal=3.0 * nfloor + 1e-5)
    # 1. deringing clamp: no overshoot beyond the local extremes of the original (plus a little)
    #    bright side: a few percent of the local range; dark side: a small fraction of the dark value itself,
    #    so the dark rim next to a candle flame cannot turn black
    lo, hi = minimum_filter(Y, 3), maximum_filter(Y, 3)
    np.clip(est, lo * np.float32(DARK_SLACK) - np.float32(nfloor), hi + np.float32(BRIGHT_SLACK) * (hi - lo), out=est)
    # 2. noise gate: only where the local gradient clearly exceeds the noise at that brightness
    if noise_sigma_lin is not None:
        g, _, _ = _grad(Y, 1.0)
        sn = np.asarray(noise_sigma_lin(gaussian_filter(Y, 1.5)), np.float32)
        snr = g / np.maximum(sn * np.float32(max(p.threshold, 0.05)), 1e-9)
        gate = np.clip((snr - 1.0) / 3.0, 0.0, 1.0)
        gate = gate * gate * (3.0 - 2.0 * gate)
        est = Y + gate * (est - Y)
    # 3. bounded ratio onto the colour channels
    eps = np.float32(max(nfloor, 1e-5))
    ratio = np.clip((est + eps) / (Y + eps), 0.5, 2.0)
    return (rgb * ratio[..., None]).astype(np.float32)
