"""A collaborative-filtering denoiser of the BM3D class, in our own code (BUILD_PLAN_NOISE D.2, plan 8.1a).

The idea (Dabov, Foi, Katkovnik, Egiazarian 2007): a photo repeats itself. For every small block, find the blocks
nearby that look most like it, stack them, and transform the stack in three dimensions. What the blocks have in
common (the picture) gathers in a few large coefficients; the noise, which differs from block to block, stays
spread thin. Small coefficients are dropped (pass 1), then the cleaned result is used as a guide for a Wiener
filter of the original (pass 2), and every photosite takes the weighted average of all the blocks that cover it.

It works on one variance-stabilised plane where the noise has strength one. It is the yardstick the other methods
are measured against, not the daily tool: in NumPy it takes minutes on a 45 MP photo. (The ``bm3d`` package is not
GPL-compatible; nothing of it is used.)
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import dct, dctn, idct, idctn

BLOCK = 8                   # block side
STEP = 3                    # distance between reference blocks
SEARCH = 6                  # how far away a similar block may be (photosites), looked for every 2
GROUP = 8                   # blocks in a stack
TILE = 192
PAD = SEARCH + BLOCK
LAMBDA = 2.7                # pass 1: coefficients under this many sigmas are noise

_OFFS = [(dy, dx) for dy in range(-SEARCH, SEARCH + 1, 2) for dx in range(-SEARCH, SEARCH + 1, 2)]
_OFFS.sort(key=lambda o: (o[0] * o[0] + o[1] * o[1]))          # the block itself first
_DY = np.array([o[0] for o in _OFFS])
_DX = np.array([o[1] for o in _OFFS])
_WIN = np.outer(np.kaiser(BLOCK, 2.0), np.kaiser(BLOCK, 2.0)).astype(np.float32)
_CELL = None


def _box(a: np.ndarray) -> np.ndarray:
    """Sum over every BLOCK x BLOCK window (top-left aligned)."""
    c = np.cumsum(a, axis=0, dtype=np.float64)
    c = np.concatenate([c[BLOCK - 1:BLOCK], c[BLOCK:] - c[:-BLOCK]], axis=0)
    c = np.cumsum(c, axis=1)
    return np.concatenate([c[:, BLOCK - 1:BLOCK], c[:, BLOCK:] - c[:, :-BLOCK]], axis=1).astype(np.float32)


def _grid(n: int) -> np.ndarray:
    g = np.arange(0, n, STEP)
    return g if g[-1] == n - 1 else np.append(g, n - 1)


def _groups(guide: np.ndarray):
    """For every reference block of a padded tile: the GROUP most similar blocks nearby (their top-left corners)."""
    hp, wp = guide.shape
    core = guide[SEARCH:hp - SEARCH, SEARCH:wp - SEARCH]
    ry, rx = _grid(core.shape[0] - BLOCK + 1), _grid(core.shape[1] - BLOCK + 1)
    dist = np.empty((len(_OFFS), ry.size, rx.size), np.float32)
    for k, (dy, dx) in enumerate(_OFFS):
        d = core - guide[SEARCH + dy:hp - SEARCH + dy, SEARCH + dx:wp - SEARCH + dx]
        dist[k] = _box(d * d)[np.ix_(ry, rx)]
    dist[0] = -1.0                                              # the block itself is always in its own stack
    pick = np.argpartition(dist, GROUP - 1, axis=0)[:GROUP]
    by = SEARCH + ry[None, :, None] + _DY[pick]
    bx = SEARCH + rx[None, None, :] + _DX[pick]
    return by, bx


def _aggregate(est: np.ndarray, weight: np.ndarray, by: np.ndarray, bx: np.ndarray, shape: tuple) -> np.ndarray:
    hp, wp = shape
    cell = (np.arange(BLOCK)[:, None] * wp + np.arange(BLOCK)[None, :]).ravel()
    flat = ((by * wp + bx)[..., None] + cell).ravel()
    w = (weight[None, :, :, None, None] * _WIN).astype(np.float32)
    num = np.bincount(flat, weights=(est * w).ravel(), minlength=hp * wp)
    den = np.bincount(flat, weights=np.broadcast_to(w, est.shape).ravel(), minlength=hp * wp)
    return (num / np.maximum(den, 1e-9)).reshape(hp, wp).astype(np.float32)


def _forward(blocks: np.ndarray) -> np.ndarray:
    return dct(dctn(blocks, axes=(-2, -1), norm="ortho"), axis=0, norm="ortho")


def _inverse(coef: np.ndarray) -> np.ndarray:
    return idctn(idct(coef, axis=0, norm="ortho"), axes=(-2, -1), norm="ortho")


def _tile(noisy: np.ndarray, nv: float) -> np.ndarray:
    """One padded tile (unit noise scaled by ``nv`` = how much of it to treat as noise)."""
    view = sliding_window_view(noisy, (BLOCK, BLOCK))
    by, bx = _groups(noisy)
    coef = _forward(view[by, bx])
    keep = np.abs(coef) > LAMBDA * np.sqrt(nv)
    keep[0, :, :, 0, 0] = True                                  # a stack's mean brightness is never dropped
    coef *= keep
    weight = 1.0 / np.maximum(keep.sum(axis=(0, 3, 4)), 1).astype(np.float32)
    basic = _aggregate(_inverse(coef), weight, by, bx, noisy.shape)
    # pass 2: the basic result says which blocks are alike and how large each coefficient truly is
    by, bx = _groups(basic)
    cb = _forward(sliding_window_view(basic, (BLOCK, BLOCK))[by, bx])
    wien = cb * cb / (cb * cb + nv)
    weight = 1.0 / np.maximum((wien * wien).sum(axis=(0, 3, 4)), 1e-6).astype(np.float32)
    return _aggregate(_inverse(wien * _forward(view[by, bx])), weight, by, bx, noisy.shape)


def denoise_collab(x: np.ndarray, strength: float, scales: int = 0) -> np.ndarray:
    """Denoise one unit-noise plane. ``strength`` 1 = the measured noise; more treats more of it as noise."""
    if strength <= 0:
        return x
    nv = float(strength)
    h, w = x.shape
    p = np.pad(x.astype(np.float32), PAD, mode="reflect")
    out = np.empty((h, w), np.float32)
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
            t = _tile(p[y0:y1 + 2 * PAD, x0:x1 + 2 * PAD], nv)
            out[y0:y1, x0:x1] = t[PAD:PAD + y1 - y0, PAD:PAD + x1 - x0]
    return out
