"""Stage 10: demosaic (raw mosaic -> RGB).

Algorithms (all pure functions on one float32 tile; ``demosaic()`` adds tiling and threads):
  * ``bilinear``  reference / tests only
  * ``mhc``       Malvar, He & Cutler 2004, "High-quality linear interpolation for demosaicing of
                  Bayer-patterned color images" (ICASSP). Fast, clean; used for previews.
  * ``menon``     Menon, Andriani & Calvagno 2007, "Demosaicing with directional filtering and a posteriori
                  decision" (IEEE TIP 16(1)). Directional, with the refining step. The Phase 1 default.
                  Structure follows the BSD-3 colour-demosaicing implementation (see THIRD_PARTY.md).
  * ``half``      2x2 super-pixel (R, mean G, B) at half size. Instant previews, no interpolation.
  * ``rcd``       After RCD, "ratio corrected demosaicing" (L. Sanz Rodriguez, 2017, algorithm description): a
                  vertical / horizontal discrimination refined by its neighbourhood, green from the four neighbours
                  weighted by inverse gradients, then colour differences on the diagonals and at green sites. Our
                  own implementation from the description (plan 23.8, no code copied). Two honest differences:
                  the directional statistics are Hamilton-Adams gradient energies, and the green estimate uses
                  Hamilton-Adams' additive correction instead of RCD's low-pass *ratio*: the ratio version measured
                  1.4 dB worse on the most detailed natural crop (docs/RESULTS.md, session 11). The default from
                  2026-11-15 (better than Menon on every crop measured, clean and noisy).
  * ``lmmse``     Zhang & Wu 2005, "Color demosaicking via directional linear minimum mean square-error
                  estimation" (IEEE TIP 14(12)): horizontal and vertical colour differences, each cleaned by an
                  LMMSE estimate, fused by their error variances. Smooth in noise; on our crops RCD still scored
                  higher with noise added, so it is offered, not chosen automatically (RESULTS, session 11).
  * ``dual``      RCD where there is detail, LMMSE in flat areas, blended by local contrast (darktable's "dual
                  demosaic" idea, our code).

Measured on the harness: see docs/RESULTS.md (session 11).

Conventions: ``mosaic`` is a 2-D float32 array (black-subtracted, white-balanced, 1.0 = clip),
``pattern`` is a 2x2 array of 0/1/2 (R/G/B) giving the colour at (0,0), (0,1), (1,0), (1,1) of this array.
Output is HxWx3 float32 in camera RGB.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import convolve, convolve1d

ALGORITHMS = ("menon", "rcd", "lmmse", "dual", "mhc", "bilinear", "half")
EPS = 1e-5


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


# ------------------------------------------------------------------------------------------- shifted views
def _pad(a: np.ndarray, n: int = 4) -> np.ndarray:
    return np.pad(a, n, mode="reflect")


def _sh(ap: np.ndarray, dy: int, dx: int, n: int = 4) -> np.ndarray:
    """From a padded array: the value at (y + dy, x + dx) for every (y, x) of the unpadded array."""
    h, w = ap.shape[0] - 2 * n, ap.shape[1] - 2 * n
    return ap[n + dy:n + dy + h, n + dx:n + dx + w]


# ------------------------------------------------------------------------------------------- RCD (2017)
def rcd(mosaic: np.ndarray, pattern) -> np.ndarray:
    """Ratio-corrected demosaicing: see the module notes."""
    _check(mosaic, pattern)
    c = np.clip(mosaic.astype(np.float32, copy=False), 0.0, None)
    R_m, G_m, B_m = masks(c.shape, pattern)
    RB_m = ~G_m
    cp = _pad(c)

    def S(dy, dx):
        return _sh(cp, dy, dx)

    # 1. vertical / horizontal discrimination from directional gradient energies, summed along the direction
    gh = (S(0, -1) - S(0, 1)) ** 2 + 0.25 * (2 * c - S(0, -2) - S(0, 2)) ** 2
    gv = (S(-1, 0) - S(1, 0)) ** 2 + 0.25 * (2 * c - S(-2, 0) - S(2, 0)) ** 2
    ghp, gvp = _pad(gh), _pad(gv)
    h_stat = _sh(ghp, 0, -1) + gh + _sh(ghp, 0, 1)
    v_stat = _sh(gvp, -1, 0) + gv + _sh(gvp, 1, 0)
    vh = (v_stat / (v_stat + h_stat + EPS)).astype(np.float32)   # high = more change vertically: go horizontal
    del gh, gv, ghp, gvp, h_stat, v_stat
    vhp = _pad(vh)
    vh_neigh = 0.25 * (_sh(vhp, -1, -1) + _sh(vhp, -1, 1) + _sh(vhp, 1, -1) + _sh(vhp, 1, 1))
    vh_disc = np.where(np.abs(0.5 - vh) < np.abs(0.5 - vh_neigh), vh_neigh, vh)

    # 2. green at red and blue sites: neighbour greens corrected by the centre's curvature, weighted by inverse gradients
    def grad(dy, dx):
        return EPS + np.abs(S(dy, dx) - S(-dy, -dx)) + np.abs(c - S(2 * dy, 2 * dx)) + \
            np.abs(S(dy, dx) - S(3 * dy, 3 * dx)) + np.abs(S(2 * dy, 2 * dx) - S(4 * dy, 4 * dx))

    def est(dy, dx):
        return S(dy, dx) + 0.5 * (c - S(2 * dy, 2 * dx))          # Hamilton-Adams correction (see notes)
    n_g, s_g, w_g, e_g = grad(-1, 0), grad(1, 0), grad(0, -1), grad(0, 1)
    v_est = (s_g * est(-1, 0) + n_g * est(1, 0)) / (n_g + s_g)
    h_est = (e_g * est(0, -1) + w_g * est(0, 1)) / (w_g + e_g)
    del n_g, s_g, w_g, e_g
    G = np.where(RB_m, vh_disc * h_est + (1.0 - vh_disc) * v_est, c).astype(np.float32)
    del v_est, h_est

    # 3a. red at blue sites and blue at red sites, from colour differences on the diagonals
    gp = _pad(G)
    pd = (S(-1, -1) - S(1, 1)) ** 2 + 0.25 * (2 * c - S(-2, -2) - S(2, 2)) ** 2
    qd = (S(-1, 1) - S(1, -1)) ** 2 + 0.25 * (2 * c - S(-2, 2) - S(2, -2)) ** 2
    pdp, qdp = _pad(pd), _pad(qd)
    p_stat = _sh(pdp, -1, -1) + pd + _sh(pdp, 1, 1)
    q_stat = _sh(qdp, -1, 1) + qd + _sh(qdp, 1, -1)
    pq = p_stat / (p_stat + q_stat + EPS)                        # high = more change along P: use Q
    pqp = _pad(pq)
    pq_neigh = 0.25 * (_sh(pqp, -1, -1) + _sh(pqp, -1, 1) + _sh(pqp, 1, -1) + _sh(pqp, 1, 1))
    pq_disc = np.where(np.abs(0.5 - pq) < np.abs(0.5 - pq_neigh), pq_neigh, pq)
    del pd, qd, pdp, qdp, p_stat, q_stat, pq, pqp, pq_neigh
    diff = np.where(RB_m, c - G, 0.0).astype(np.float32)        # (R or B) - G where known
    dp = _pad(diff)

    def dgrad(dy, dx):
        return EPS + np.abs(S(dy, dx) - S(-dy, -dx)) + np.abs(c - S(2 * dy, 2 * dx)) + \
            np.abs(S(dy, dx) - S(3 * dy, 3 * dx))
    nw, se, ne, sw = dgrad(-1, -1), dgrad(1, 1), dgrad(-1, 1), dgrad(1, -1)
    p_est = (se * _sh(dp, -1, -1) + nw * _sh(dp, 1, 1)) / (nw + se)
    q_est = (sw * _sh(dp, -1, 1) + ne * _sh(dp, 1, -1)) / (ne + sw)
    other = G + pq_disc * q_est + (1.0 - pq_disc) * p_est          # the missing one of R / B at B / R sites
    del nw, se, ne, sw, p_est, q_est, pq_disc
    R = np.where(R_m, c, np.where(B_m, other, 0.0)).astype(np.float32)
    B = np.where(B_m, c, np.where(R_m, other, 0.0)).astype(np.float32)
    del other

    # 3b. red and blue at green sites, from orthogonal colour differences
    def gg(dy, dx):
        return EPS + np.abs(_sh(gp, dy, dx) - _sh(gp, -dy, -dx)) + np.abs(G - _sh(gp, 2 * dy, 2 * dx)) + \
            np.abs(_sh(gp, dy, dx) - _sh(gp, 3 * dy, 3 * dx))
    n_g, s_g, w_g, e_g = gg(-1, 0), gg(1, 0), gg(0, -1), gg(0, 1)
    for X in (R, B):
        xp = _pad(X - G)
        v_e = (s_g * _sh(xp, -1, 0) + n_g * _sh(xp, 1, 0)) / (n_g + s_g)
        h_e = (e_g * _sh(xp, 0, -1) + w_g * _sh(xp, 0, 1)) / (w_g + e_g)
        X[G_m] = (G + vh_disc * h_e + (1.0 - vh_disc) * v_e)[G_m]
    return np.stack([R, G, B], axis=-1).astype(np.float32)


# ------------------------------------------------------------------------------------------- LMMSE (2005)
def _lmmse_1d(y: np.ndarray, axis: int):
    """LMMSE estimate of a colour-difference signal along one direction, and its error variance."""
    g = np.array([4, 9, 15, 23, 26, 23, 15, 9, 4], np.float32)
    g /= g.sum()
    f = convolve1d(y, g, axis=axis, mode="mirror")                    # first estimate of the clean signal
    box = np.full(9, 1.0 / 9.0, np.float32)
    mu = convolve1d(f, box, axis=axis, mode="mirror")
    var_s = np.maximum(convolve1d(f * f, box, axis=axis, mode="mirror") - mu * mu, 0.0) + 1e-10
    var_n = convolve1d((y - f) ** 2, box, axis=axis, mode="mirror") + 1e-10
    k = var_s / (var_s + var_n)
    return mu + k * (y - mu), var_s * var_n / (var_s + var_n)


def lmmse(mosaic: np.ndarray, pattern) -> np.ndarray:
    """Directional LMMSE demosaicing (Zhang & Wu 2005): see the module notes."""
    _check(mosaic, pattern)
    c = mosaic.astype(np.float32, copy=False)
    R_m, G_m, B_m = masks(c.shape, pattern)
    k = np.array([-0.25, 0.5, 0.5, 0.5, -0.25], np.float32)           # 1-D Hamilton-Adams interpolation
    ih = convolve1d(c, k, axis=1, mode="mirror")                      # missing colour along the row
    iv = convolve1d(c, k, axis=0, mode="mirror")
    # colour difference G - (R or B): at G sites from the interpolated other colour, elsewhere from interpolated G
    dh = np.where(G_m, c - ih, ih - c).astype(np.float32)
    dv = np.where(G_m, c - iv, iv - c).astype(np.float32)
    xh, eh = _lmmse_1d(dh, 1)
    xv, ev = _lmmse_1d(dv, 0)
    d = (ev * xh + eh * xv) / (eh + ev)                              # fused by error variances
    del dh, dv, xh, xv, eh, ev
    G = np.where(G_m, c, c + d).astype(np.float32)                    # G = (R or B) + (G - colour)
    # red and blue: bilinear interpolation of the colour differences (G - R), (G - B)
    out = []
    for M in (R_m, B_m):
        diff = np.where(M, G - c, 0.0).astype(np.float32)
        w = M.astype(np.float32)
        kk = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], np.float32)
        num = convolve(diff, kk, mode="mirror")
        den = convolve(w, kk, mode="mirror")
        dd = np.where(M, G - c, num / np.maximum(den, 1e-6))
        out.append((G - dd).astype(np.float32))
    return np.stack([out[0], G, out[1]], axis=-1).astype(np.float32)


# ------------------------------------------------------------------------------------------- dual demosaic
DUAL_CONTRAST = 0.06


def detail_mask(rgb: np.ndarray, threshold: float = DUAL_CONTRAST) -> np.ndarray:
    """0 in flat areas, 1 where there is detail: local contrast of the luminance against its local mean."""
    from scipy.ndimage import gaussian_filter
    y = rgb[..., 0] * 0.25 + rgb[..., 1] * 0.5 + rgb[..., 2] * 0.25
    gy, gx = np.gradient(gaussian_filter(y, 0.7))
    con = gaussian_filter(np.hypot(gx, gy), 1.5) / (gaussian_filter(y, 1.5) + 0.01)
    t = np.clip((con - 0.5 * threshold) / (1.5 * threshold), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def dual(mosaic: np.ndarray, pattern) -> np.ndarray:
    """RCD where there is detail, LMMSE in flat areas (no maze patterns in noisy skies and skin)."""
    a = rcd(mosaic, pattern)
    b = lmmse(mosaic, pattern)
    m = detail_mask(a)[..., None]
    return (m * a + (1.0 - m) * b).astype(np.float32)


# ------------------------------------------------------------------------------------------- green equilibration
def green_equilibrate(mosaic: np.ndarray, pattern, radius: int = 4, limit: float = 0.08) -> np.ndarray:
    """Even out the two greens of each 2x2 quad where the sensor reads them slightly differently (maze patterns in
    flat areas, plan 23.8). Only a local *ratio* close to 1 is corrected (within ``limit``), so real detail, which
    makes the two greens differ by more, is left alone. Returns a new mosaic."""
    from scipy.ndimage import uniform_filter
    p = np.asarray(pattern)
    gpos = [(i, j) for i in (0, 1) for j in (0, 1) if p[i, j] == 1]
    if len(gpos) != 2:
        return mosaic
    out = mosaic.astype(np.float32, copy=True)
    (i1, j1), (i2, j2) = gpos
    h = min(out[i1::2, j1::2].shape[0], out[i2::2, j2::2].shape[0])
    w = min(out[i1::2, j1::2].shape[1], out[i2::2, j2::2].shape[1])
    g1, g2 = out[i1::2, j1::2][:h, :w], out[i2::2, j2::2][:h, :w]
    size = 2 * radius + 1
    m1 = uniform_filter(g1, size, mode="mirror")
    m2 = uniform_filter(g2, size, mode="mirror")
    ratio = (m1 + 1e-6) / (m2 + 1e-6)
    ok = (np.abs(ratio - 1.0) < limit) & (m1 > 1e-4) & (m2 > 1e-4)
    s = np.where(ok, np.sqrt(ratio), 1.0).astype(np.float32)
    out[i1::2, j1::2][:h, :w] = g1 / s
    out[i2::2, j2::2][:h, :w] = g2 * s
    return out


_FUNCS = {"bilinear": bilinear, "mhc": mhc, "menon": menon, "rcd": rcd, "lmmse": lmmse, "dual": dual}


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
