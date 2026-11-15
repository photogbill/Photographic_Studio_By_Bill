"""Stage 10: demosaic (raw mosaic -> RGB).

Algorithms (all pure functions on one float32 tile; ``demosaic()`` adds tiling and threads):
  * ``bilinear``  reference / tests only
  * ``mhc``       Malvar, He & Cutler 2004, "High-quality linear interpolation for demosaicing of
                  Bayer-patterned color images" (ICASSP). Fast, clean; used for previews.
  * ``menon``     Menon, Andriani & Calvagno 2007, "Demosaicing with directional filtering and a posteriori
                  decision" (IEEE TIP 16(1)). Directional, with the refining step. The Phase 1 default.
                  Structure follows the BSD-3 colour-demosaicing implementation (see THIRD_PARTY.md).
  * ``half``      2x2 super-pixel (R, mean G, B) at half size. Instant previews, no interpolation.

Plan note (section 3, stage 10): RCD / AMaZE-class quality is the target. RCD is the next algorithm to add;
it must be implemented from its published reference and measured against ``menon`` on the harness.

Conventions: ``mosaic`` is a 2-D float32 array (black-subtracted, white-balanced, 1.0 = clip),
``pattern`` is a 2x2 array of 0/1/2 (R/G/B) giving the colour at (0,0), (0,1), (1,0), (1,1) of this array.
Output is HxWx3 float32 in camera RGB.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import convolve, convolve1d

ALGORITHMS = ("menon", "mhc", "bilinear", "half")


def masks(shape, pattern) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks for R, G and B sites of a mosaic with the given 2x2 pattern."""
    pattern = np.asarray(pattern)
    h, w = shape
    out = []
    for c in range(3):
        m = np.zeros((h, w), dtype=bool)
        for i in (0, 1):
            for j in (0, 1):
                if pattern[i, j] == c:
                    m[i::2, j::2] = True
        out.append(m)
    return out[0], out[1], out[2]


def _check(mosaic, pattern):
    p = np.asarray(pattern)
    if p.shape != (2, 2) or sorted(p.ravel().tolist()) != [0, 1, 1, 2]:
        raise ValueError(f"demosaic needs a 2x2 Bayer pattern with two greens, got {p.tolist()}")
    if mosaic.ndim != 2:
        raise ValueError("mosaic must be 2-D")


# ------------------------------------------------------------------------------------------- bilinear
def bilinear(mosaic: np.ndarray, pattern) -> np.ndarray:
    _check(mosaic, pattern)
    R_m, G_m, B_m = masks(mosaic.shape, pattern)
    k_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
    k_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
    out = np.empty(mosaic.shape + (3,), np.float32)
    out[..., 0] = convolve(mosaic * R_m, k_rb, mode="mirror")
    out[..., 1] = convolve(mosaic * G_m, k_g, mode="mirror")
    out[..., 2] = convolve(mosaic * B_m, k_rb, mode="mirror")
    return out


# ------------------------------------------------------------------------------------------- MHC 2004
_MHC_G = np.array([[0, 0, -1, 0, 0],
                   [0, 0, 2, 0, 0],
                   [-1, 2, 4, 2, -1],
                   [0, 0, 2, 0, 0],
                   [0, 0, -1, 0, 0]], np.float32) / 8
_MHC_ROW = np.array([[0, 0, 0.5, 0, 0],           # colour from horizontal neighbours (at G sites)
                     [0, -1, 0, -1, 0],
                     [-1, 4, 5, 4, -1],
                     [0, -1, 0, -1, 0],
                     [0, 0, 0.5, 0, 0]], np.float32) / 8
_MHC_COL = _MHC_ROW.T.copy()                        # colour from vertical neighbours (at G sites)
_MHC_DIAG = np.array([[0, 0, -1.5, 0, 0],          # R at B sites and B at R sites
                      [0, 2, 0, 2, 0],
                      [-1.5, 0, 6, 0, -1.5],
                      [0, 2, 0, 2, 0],
                      [0, 0, -1.5, 0, 0]], np.float32) / 8


def mhc(mosaic: np.ndarray, pattern) -> np.ndarray:
    _check(mosaic, pattern)
    m = mosaic.astype(np.float32, copy=False)
    R_m, G_m, B_m = masks(m.shape, pattern)
    rows_r = np.any(R_m, axis=1)[:, None]
    rows_b = np.any(B_m, axis=1)[:, None]
    g_interp = convolve(m, _MHC_G, mode="mirror")
    c_row = convolve(m, _MHC_ROW, mode="mirror")
    c_col = convolve(m, _MHC_COL, mode="mirror")
    c_diag = convolve(m, _MHC_DIAG, mode="mirror")
    R = np.where(R_m, m, 0)
    G = np.where(G_m, m, g_interp)
    B = np.where(B_m, m, 0)
    # G site in a red row: R left/right, B above/below. G site in a blue row: the reverse.
    R = np.where(G_m & rows_r, c_row, R)
    R = np.where(G_m & rows_b, c_col, R)
    R = np.where(B_m, c_diag, R)
    B = np.where(G_m & rows_b, c_row, B)
    B = np.where(G_m & rows_r, c_col, B)
    B = np.where(R_m, c_diag, B)
    return np.stack([R, G, B], axis=-1).astype(np.float32)


# ------------------------------------------------------------------------------------------- Menon 2007
def _h(x, k):
    return convolve1d(x, k, axis=-1, mode="mirror")


def _v(x, k):
    return convolve1d(x, k, axis=0, mode="mirror")


_MENON_K = np.array([[0, 0, 1, 0, 1],
                     [0, 0, 0, 1, 0],
                     [0, 0, 3, 0, 3],
                     [0, 0, 0, 1, 0],
                     [0, 0, 1, 0, 1]], np.float32)


def menon(mosaic: np.ndarray, pattern, refine: bool = True) -> np.ndarray:
    _check(mosaic, pattern)
    cfa = mosaic.astype(np.float32, copy=False)
    R_m, G_m, B_m = masks(cfa.shape, pattern)
    h0 = np.array([0.0, 0.5, 0.0, 0.5, 0.0], np.float32)
    h1 = np.array([-0.25, 0.0, 0.5, 0.0, -0.25], np.float32)

    R = np.where(R_m, cfa, 0).astype(np.float32)
    G = np.where(G_m, cfa, 0).astype(np.float32)
    B = np.where(B_m, cfa, 0).astype(np.float32)

    G_H = np.where(G_m, G, _h(cfa, h0) + _h(cfa, h1))
    G_V = np.where(G_m, G, _v(cfa, h0) + _v(cfa, h1))

    C_H = np.where(R_m, R - G_H, np.where(B_m, B - G_H, 0)).astype(np.float32)
    C_V = np.where(R_m, R - G_V, np.where(B_m, B - G_V, 0)).astype(np.float32)

    D_H = np.abs(C_H - np.pad(C_H, ((0, 0), (0, 2)), mode="reflect")[:, 2:])
    D_V = np.abs(C_V - np.pad(C_V, ((0, 2), (0, 0)), mode="reflect")[2:, :])
    del C_H, C_V

    d_H = convolve(D_H, _MENON_K, mode="constant")
    d_V = convolve(D_V, _MENON_K.T, mode="constant")
    del D_H, D_V

    M = d_V >= d_H                       # True = horizontal direction chosen
    G = np.where(M, G_H, G_V).astype(np.float32)
    del d_H, d_V, G_H, G_V

    R_r = np.any(R_m, axis=1)[:, None]   # rows that contain red sites
    B_r = np.any(B_m, axis=1)[:, None]
    kb = np.array([0.5, 0.0, 0.5], np.float32)

    R = np.where(G_m & R_r, G + _h(R, kb) - _h(G, kb), R)
    R = np.where(G_m & B_r, G + _v(R, kb) - _v(G, kb), R)
    B = np.where(G_m & B_r, G + _h(B, kb) - _h(G, kb), B)
    B = np.where(G_m & R_r, G + _v(B, kb) - _v(G, kb), B)

    R = np.where(B_m, np.where(M, B + _h(R, kb) - _h(B, kb), B + _v(R, kb) - _v(B, kb)), R)
    B = np.where(R_m, np.where(M, R + _h(B, kb) - _h(R, kb), R + _v(B, kb) - _v(R, kb)), B)

    if refine:
        R, G, B = _menon_refine(R, G, B, R_m, G_m, B_m, M)
    return np.stack([R, G, B], axis=-1).astype(np.float32)


def _menon_refine(R, G, B, R_m, G_m, B_m, M):
    fir = np.full(3, 1.0 / 3.0, np.float32)
    kb = np.array([0.5, 0.0, 0.5], np.float32)

    # 1. update green at R and B sites from the smoothed colour differences, along the chosen direction
    R_G = R - G
    B_G = B - G
    B_G_m = np.where(B_m, np.where(M, _h(B_G, fir), _v(B_G, fir)), 0)
    R_G_m = np.where(R_m, np.where(M, _h(R_G, fir), _v(R_G, fir)), 0)
    G = np.where(R_m, R - R_G_m, G)
    G = np.where(B_m, B - B_G_m, G)

    # 2. update red and blue at green sites
    R_r = np.any(R_m, axis=1)[:, None]
    R_c = np.any(R_m, axis=0)[None, :]
    B_r = np.any(B_m, axis=1)[:, None]
    B_c = np.any(B_m, axis=0)[None, :]
    R_G = R - G
    B_G = B - G
    R = np.where(G_m & B_r, G + _v(R_G, kb), R)
    R = np.where(G_m & B_c, G + _h(R_G, kb), R)
    B = np.where(G_m & R_r, G + _v(B_G, kb), B)
    B = np.where(G_m & R_c, G + _h(B_G, kb), B)

    # 3. update red at blue sites and blue at red sites
    R_B = R - B
    R_B_m = np.where(B_m, np.where(M, _h(R_B, fir), _v(R_B, fir)), 0)
    R = np.where(B_m, B + R_B_m, R)
    R_B_m = np.where(R_m, np.where(M, _h(R_B, fir), _v(R_B, fir)), 0)
    B = np.where(R_m, R - R_B_m, B)
    return R, G, B


# ------------------------------------------------------------------------------------------- half size
def half(mosaic: np.ndarray, pattern) -> np.ndarray:
    """One RGB pixel per 2x2 quad: (R, mean of the two greens, B). Output is half size."""
    _check(mosaic, pattern)
    p = np.asarray(pattern)
    h, w = (mosaic.shape[0] // 2) * 2, (mosaic.shape[1] // 2) * 2
    m = mosaic[:h, :w]
    quads = {(i, j): m[i::2, j::2] for i in (0, 1) for j in (0, 1)}
    out = np.zeros((h // 2, w // 2, 3), np.float32)
    for (i, j), q in quads.items():
        c = int(p[i, j])
        if c == 1:
            out[..., 1] += 0.5 * q
        else:
            out[..., c] = q
    return out


_FUNCS = {"bilinear": bilinear, "mhc": mhc, "menon": menon}


# ------------------------------------------------------------------------------------------- tiled driver
def demosaic(mosaic: np.ndarray, pattern, algorithm: str = "menon", strip_rows: int = 256, overlap: int = 16,
             workers: int | None = None) -> np.ndarray:
    """Demosaic a whole frame in overlapping horizontal strips, in parallel threads.

    Strips start on even rows so the CFA pattern is the same in every strip. The result is identical to
    processing the frame in one piece, except within ``overlap`` rows of the frame edge where the
    mirrored border differs (tests check this).
    """
    if algorithm == "half":
        return half(mosaic, pattern)
    if algorithm not in _FUNCS:
        raise ValueError(f"unknown demosaic algorithm {algorithm!r}; choose from {ALGORITHMS}")
    fn = _FUNCS[algorithm]
    h, w = mosaic.shape
    strip_rows = max(32, (strip_rows // 2) * 2)
    overlap = max(8, (overlap // 2) * 2)
    if h <= strip_rows + 2 * overlap:
        return fn(mosaic, pattern)
    out = np.empty((h, w, 3), np.float32)
    starts = list(range(0, h, strip_rows))

    def job(y0):
        y1 = min(h, y0 + strip_rows)
        a = max(0, y0 - overlap)
        b = min(h, y1 + overlap)
        res = fn(mosaic[a:b], pattern)              # a is even, so the pattern is unchanged
        out[y0:y1] = res[y0 - a: y0 - a + (y1 - y0)]

    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    if workers <= 1:
        for y0 in starts:
            job(y0)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="demosaic") as ex:
            list(ex.map(job, starts))
    return out
