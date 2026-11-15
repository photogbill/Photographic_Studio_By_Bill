"""Phase 6 finishing tools: Dehaze, Texture and film Grain.

* Dehaze (stage 15, scene-linear, after exposure): the dark-channel prior of He, Sun and Tang (CVPR 2009 /
  IEEE TPAMI 33(12) 2011) with the transmission refined by a guided filter (He, Sun, Tang, TPAMI 2013). The
  airlight is measured once on the whole frame (pipeline.haze_airlight), so a 100% region matches the full render.
  Negative amounts add a soft, even haze.
* Texture (stage 16, scene-linear): fine detail (a few pixels across) lifted or smoothed on log luminance. The
  split between "fine detail" and "edges" is a self-guided filter, so large edges stay out of it (no halos); a noise
  gate keeps it from amplifying noise.
* Grain (end of stage 17, display-referred): film-like grain from a coordinate hash, so it is the same in a 100%
  region and in the full render. Previews show statistically matched grain (what a downscaled full render shows).

Every function returns a new array and changes nothing at its default (amount 0).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter, uniform_filter

from ... import log as slog

log = slog.get("engine.finish")

LUMA = np.array([0.2880, 0.7119, 0.0001], np.float32)     # ProPhoto RGB -> Y


# ------------------------------------------------------------------------------------------ guided filter
def guided(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter (He, Sun, Tang 2013): filter p using guide I, box radius r."""
    size = 2 * r + 1
    mI = uniform_filter(I, size, mode="reflect")
    mp = uniform_filter(p, size, mode="reflect")
    cov = uniform_filter(I * p, size, mode="reflect") - mI * mp
    var = np.maximum(uniform_filter(I * I, size, mode="reflect") - mI * mI, 0.0)
    a = cov / (var + np.float32(eps))
    b = mp - a * mI
    return (uniform_filter(a, size, mode="reflect") * I + uniform_filter(b, size, mode="reflect")).astype(np.float32)


# ------------------------------------------------------------------------------------------ dehaze
def dehaze_patch(eff_scale: float, long_edge_full: float) -> int:
    """Dark-channel patch radius at this render's resolution (0.4% of the long edge at full size)."""
    return max(1, int(round(0.004 * long_edge_full * eff_scale)))


def dehaze_context(long_edge_full: float) -> int:
    r = dehaze_patch(1.0, long_edge_full)
    return 5 * r + 8


def airlight(rgb: np.ndarray) -> np.ndarray:
    """Colour of the haze from a small whole-frame image (scene-linear): the mean of the 0.1% of pixels with the
    brightest dark channel (He et al.), ignoring clipped light sources."""
    dark = minimum_filter(rgb.min(axis=2), size=15, mode="nearest")
    Y = rgb @ LUMA
    ok = Y < np.percentile(Y, 99.9)                          # a lamp or the sun is not the haze
    d = np.where(ok, dark, -1.0)
    n = max(10, int(d.size * 0.001))
    idx = np.argpartition(d.ravel(), -n)[-n:]
    A = rgb.reshape(-1, 3)[idx].mean(axis=0)
    return np.maximum(A, 1e-4).astype(np.float32)


def dehaze(rgb: np.ndarray, amount: float, A: np.ndarray, eff_scale: float = 1.0,
           long_edge_full: float = 6000.0) -> np.ndarray:
    """amount -100..100. + removes haze (more contrast and colour in distant, washed-out areas), - adds haze."""
    if not amount:
        return rgb
    A = np.asarray(A, np.float32).reshape(1, 1, 3)
    with slog.stage("finish.dehaze", log, amount=round(amount, 1), mp=round(rgb.shape[0] * rgb.shape[1] / 1e6, 2)):
        if amount < 0:
            h = np.float32(0.6 * min(1.0, -amount / 100.0))
            return (rgb * (1 - h) + A * h).astype(np.float32)
        w = 0.95 * min(1.0, amount / 100.0)
        r = dehaze_patch(eff_scale, long_edge_full)
        norm = rgb / A
        dark = minimum_filter(norm.min(axis=2), size=2 * r + 1, mode="nearest")
        t = (1.0 - w * dark).astype(np.float32)
        Y = (rgb @ LUMA).astype(np.float32)
        g = np.log2(np.maximum(Y, 1e-6)).astype(np.float32)
        t = guided(g, t, 4 * r, 0.05)
        t = np.clip(t, 0.15, 1.0)[..., None]
        out = (rgb - A) / t + A
        return np.maximum(out, 0.0).astype(np.float32)


# ------------------------------------------------------------------------------------------ texture
def texture_radius(eff_scale: float) -> int:
    return max(1, int(round(2.0 * eff_scale)))


def texture(rgb: np.ndarray, amount: float, eff_scale: float = 1.0, noise_fn=None) -> np.ndarray:
    """amount -100..100. + brings out fine texture (bark, fabric, skin pores), - smooths it (never edges)."""
    if not amount:
        return rgb
    with slog.stage("finish.texture", log, amount=round(amount, 1), mp=round(rgb.shape[0] * rgb.shape[1] / 1e6, 2)):
        Y = np.maximum((rgb @ LUMA).astype(np.float32), 1e-7)
        L = np.log2(Y)
        r = texture_radius(eff_scale)
        base = _self_guided(L, r, 0.2 ** 2)
        d = L - base
        k = float(amount) / 100.0 * min(1.0, 2.0 * eff_scale) ** 0.5
        if k > 0:
            gate = 1.0
            if noise_fn is not None:
                sl = np.asarray(noise_fn(Y), np.float32) / (Y * np.float32(math.log(2)))
                if eff_scale < 1:
                    sl = sl * np.float32(eff_scale)              # a preview pixel averages ~1/scale^2 pixels
                gate = d * d / (d * d + (2.5 * sl) ** 2 + 1e-12)
            L2 = L + np.float32(1.2 * k) * gate * d
        else:
            L2 = L + np.float32(0.85 * max(k, -1.0)) * d
        return (rgb * np.exp2(L2 - L)[..., None]).astype(np.float32)


def _self_guided(L, r, eps):
    size = 2 * r + 1
    m = uniform_filter(L, size, mode="reflect")
    v = np.maximum(uniform_filter(L * L, size, mode="reflect") - m * m, 0.0)
    a = v / (v + np.float32(eps))
    b = m - a * m
    return uniform_filter(a, size, mode="reflect") * L + uniform_filter(b, size, mode="reflect")


# ------------------------------------------------------------------------------------------ grain
def _fmix(h: np.ndarray) -> np.ndarray:
    """MurmurHash3 32-bit finaliser (public domain): scrambles every bit of a uint32 array."""
    h = h ^ (h >> np.uint32(16))
    h = h * np.uint32(0x85EBCA6B)
    h = h ^ (h >> np.uint32(13))
    h = h * np.uint32(0xC2B2AE35)
    return h ^ (h >> np.uint32(16))


def hash_gauss(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Unit Gaussian noise that depends only on integer coordinates and the seed (Box-Muller on two hashes)."""
    with np.errstate(over="ignore"):
        x = (np.asarray(ix, np.int64) + (1 << 20)).astype(np.uint32)
        y = (np.asarray(iy, np.int64) + (1 << 20)).astype(np.uint32)
        s = np.uint32((seed * 0x9E3779B9) & 0xFFFFFFFF)
        h1 = _fmix(x * np.uint32(0x8DA6B343) ^ _fmix(y * np.uint32(0xD8163841) ^ s))
        h2 = _fmix(h1 ^ np.uint32(0x68E31DA4))
    u1 = (h1.astype(np.float64) + 1.0) / 4294967297.0
    u2 = h2.astype(np.float64) / 4294967296.0
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)).astype(np.float32)


def _kernel(sigma: float) -> np.ndarray:
    if sigma < 0.25:
        return np.array([1.0])
    r = int(math.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def grain_field(h: int, w: int, x0: float, y0: float, sx: float, sy: float, size: float, roughness: float,
                seed: int) -> np.ndarray:
    """Grain (unit standard deviation at full size) for an h x w array whose pixel (i, j) sits at full-resolution
    position (x0 + (j + 0.5) / sx - 0.5, y0 + (i + 0.5) / sy - 0.5). size = grain sigma in full-size pixels."""
    full = sx > 0.999 and sy > 0.999
    out = np.zeros((h, w), np.float32)
    octaves = [(1.0 - roughness, size, seed), (roughness, 2.5 * size, seed + 7)]
    norm = math.sqrt(sum(a * a for a, _, _ in octaves)) or 1.0
    for a, sig, sd in octaves:
        if a <= 1e-6:
            continue
        if full:
            sig_px, ratio = sig, 1.0
            ix0, iy0 = int(round(x0)), int(round(y0))
        else:
            sd_full = 0.5 / min(sx, sy)                          # what the downscale averages over
            sig_t = math.sqrt(sig * sig + sd_full * sd_full)
            sig_px, ratio = sig_t * min(sx, sy), sig / sig_t
        k = _kernel(sig_px)
        p = len(k) // 2
        if full:
            jj = np.arange(ix0 - p, ix0 + w + p)
            ii = np.arange(iy0 - p, iy0 + h + p)
        else:
            # preview pixels: one sample per preview pixel, keyed by its full-size position (deterministic)
            jj = np.round(x0 + (np.arange(-p, w + p) + 0.5) / sx - 0.5).astype(np.int64)
            ii = np.round(y0 + (np.arange(-p, h + p) + 0.5) / sy - 0.5).astype(np.int64)
        n = hash_gauss(jj[None, :], ii[:, None], sd)
        if len(k) > 1:
            from scipy.ndimage import convolve1d
            n = convolve1d(convolve1d(n, k, axis=0, mode="nearest"), k, axis=1, mode="nearest")
            n = n / np.float32(np.sum(k * k))                   # back to unit standard deviation
        out += np.float32(a * ratio / norm) * n[p:p + h, p:p + w]
    return out


def grain(rgb: np.ndarray, amount: float, size: float, roughness: float, x0: float, y0: float, sx: float,
          sy: float, seed: int = 1, colour: bool = False) -> np.ndarray:
    """Film grain on display-referred linear RGB. amount 0..100; size 0.5..3 (grain width in full-size pixels);
    roughness 0..100 (clumpier). Strongest in the midtones, none in pure black or white."""
    if amount <= 0:
        return rgb
    with slog.stage("finish.grain", log, amount=round(amount, 1), size=size, mp=round(rgb.size / 3e6, 2)):
        h, w = rgb.shape[:2]
        rough = min(max(roughness / 100.0, 0.0), 1.0)
        a = np.float32(0.25 * amount / 100.0)
        Y = np.clip(rgb @ LUMA, 0.0, 1.0)
        e = Y ** (1 / 2.2)
        wt = (4.0 * e * (1.0 - e)).astype(np.float32)
        if colour:
            out = np.empty_like(rgb)
            for c in range(3):
                g = grain_field(h, w, x0, y0, sx, sy, size, rough, seed + 101 * c)
                out[..., c] = rgb[..., c] * np.exp2(a * 0.8 * g * wt)
            return out.astype(np.float32)
        g = grain_field(h, w, x0, y0, sx, sy, size, rough, seed)
        return (rgb * np.exp2(a * g * wt)[..., None]).astype(np.float32)
