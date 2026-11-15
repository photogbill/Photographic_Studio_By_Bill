"""Gaussian and Laplacian pyramids (Burt and Adelson 1983) with the 5-tap binomial kernel [1 4 6 4 1] / 16.

Written polyphase with NumPy slicing (no zero-insertion, no generic convolution), so one level of a 45 MP
frame costs a few hundred milliseconds. Mirror boundaries. ``down`` blurs and keeps every second sample;
``up`` is the matching expand (zero-insert + 2x the kernel), so ``x == up(down(x)) + laplacian`` exactly.
"""
from __future__ import annotations

import numpy as np


def _down_axis0(x: np.ndarray) -> np.ndarray:
    n = (x.shape[0] + 1) // 2
    p = np.pad(x, ((2, 2),) + ((0, 0),) * (x.ndim - 1), mode="reflect")
    out = p[0:2 * n:2] + p[4:2 * n + 4:2]
    out += 4.0 * (p[1:2 * n + 1:2] + p[3:2 * n + 3:2])
    out += 6.0 * p[2:2 * n + 2:2]
    out *= np.float32(1.0 / 16.0)
    return out


def _up_axis0(x: np.ndarray, n_out: int) -> np.ndarray:
    p = np.pad(x, ((1, 1),) + ((0, 0),) * (x.ndim - 1), mode="reflect")
    n = x.shape[0]
    out = np.empty((2 * n,) + x.shape[1:], np.float32)
    out[0::2] = (p[0:n] + 6.0 * p[1:n + 1] + p[2:n + 2]) * np.float32(1.0 / 8.0)
    out[1::2] = (p[1:n + 1] + p[2:n + 2]) * np.float32(0.5)
    return out[:n_out]


def down(x: np.ndarray) -> np.ndarray:
    """Blur with the binomial kernel and decimate by 2 in both directions (2-D arrays)."""
    x = np.asarray(x, np.float32)
    if x.shape[0] < 3 or x.shape[1] < 3:
        raise ValueError("image too small for another pyramid level")
    return np.ascontiguousarray(_down_axis0(_down_axis0(x).T).T)


def up(x: np.ndarray, shape) -> np.ndarray:
    """Expand to ``shape`` (the shape of the finer level)."""
    x = np.asarray(x, np.float32)
    return np.ascontiguousarray(_up_axis0(_up_axis0(x, shape[0]).T, shape[1]).T)


def max_levels(shape, min_size: int = 8) -> int:
    """How many times an image of this shape can be halved before its short side drops below min_size."""
    n, s = 0, min(shape[:2])
    while s >= 2 * min_size:
        s = (s + 1) // 2
        n += 1
    return n


def gaussian(x: np.ndarray, levels: int) -> list:
    g = [np.asarray(x, np.float32)]
    for _ in range(levels):
        g.append(down(g[-1]))
    return g


def laplacian_from_gaussian(g: list) -> list:
    """Laplacian levels 0..len(g)-2, plus the coarsest Gaussian level as the residual."""
    return [g[i] - up(g[i + 1], g[i].shape) for i in range(len(g) - 1)] + [g[-1]]


def collapse(lap: list) -> np.ndarray:
    x = lap[-1]
    for i in range(len(lap) - 2, -1, -1):
        x = up(x, lap[i].shape) + lap[i]
    return x


_NOISE_FACTORS: list = []


def white_noise_factors(levels: int = 12) -> list:
    """Standard deviation of Gaussian level l relative to level 0 for (slightly correlated) pixel noise, as left
    by a demosaic. Measured once on synthetic noise."""
    global _NOISE_FACTORS
    if len(_NOISE_FACTORS) < levels + 1:
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(7)
        n = gaussian_filter(rng.standard_normal((2048, 2048)).astype(np.float32), 0.6)
        n /= n.std()
        g = gaussian(n, min(levels, max_levels(n.shape, 4)))
        f = [float(x[4:-4, 4:-4].std()) if min(x.shape) > 12 else float(x.std()) for x in g]
        while len(f) < levels + 1:
            f.append(f[-1] * 0.5)
        _NOISE_FACTORS = f
    return _NOISE_FACTORS
