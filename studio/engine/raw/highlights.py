"""Stage 9 highlight reconstruction: "rebuild colour" for clipped highlights.

When one colour channel clips (a candle flame's red, a sunlit orange petal) the plain clip turns it white or
yellow. "Rebuild" gives the clipped channels back the colour of the edge of the clipped area:

1. Half-size planes R, G, B of the whole frame after white balance (G = mean of the two greens).
2. Log-chromaticity q_c = log(P_c) - log(mean P) is read on a thin ring just outside the clipped areas, where
   every channel is still valid, and spread inwards by normalised convolution at growing scales (small areas fill
   from close by, large ones from further out).
3. Inside, the brightness comes from the channels that are not clipped (log L = mean over them of log P_u - q_u);
   where every channel is clipped it is the smallest brightness consistent with all of them. Each clipped channel
   becomes exp(q_c + log L), never lower than what was recorded.

The maps are computed once per image and white balance on the whole frame (Image.highlight_maps), so a region
render rebuilds exactly as the full frame does. Our design; the idea of filling clipped channels from the
chromaticity of the surrounding unclipped area is shared with darktable's "inpaint opposed" (no code used).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from ... import log as slog
from ..raw import linearize as lin

log = slog.get("engine.highlights")

CLIP = 0.995                     # a photosite at 99.5% of white is clipped
SIGMAS = (2.0, 6.0, 18.0, 54.0)  # half-size pixels


def _planes(mos: np.ndarray, pat: np.ndarray):
    """Half-size R, G, B planes and per-plane clip flags from a normalised (pre-WB) mosaic."""
    h2, w2 = mos.shape[0] // 2, mos.shape[1] // 2
    P = np.zeros((h2, w2, 3), np.float32)
    K = np.zeros((h2, w2, 3), bool)
    n = np.zeros(3, np.float32)
    for i in (0, 1):
        for j in (0, 1):
            c = int(pat[i, j])
            v = mos[i:2 * h2:2, j:2 * w2:2]
            P[..., c] += v
            K[..., c] |= v >= CLIP
            n[c] += 1
    P /= n
    return P, K


def estimate(P: np.ndarray, K: np.ndarray) -> tuple | None:
    """(y0, x0, est) with est the rebuilt planes inside the bounding box of the clipped areas, or None.
    P: half-size planes after white balance; K: which channels are clipped."""
    A = K.any(axis=2)
    if not A.any():
        return None
    reach = int(3 * SIGMAS[-1])
    ys, xs = np.nonzero(A)
    y0, y1 = max(0, ys.min() - reach), min(A.shape[0], ys.max() + reach + 1)
    x0, x1 = max(0, xs.min() - reach), min(A.shape[1], xs.max() + reach + 1)
    P, K, A = P[y0:y1, x0:x1], K[y0:y1, x0:x1], A[y0:y1, x0:x1]
    A2 = binary_dilation(A, iterations=2)
    # the colour donors: a thin ring just outside the clipped areas, bright enough to belong to the same light
    # (a dark background beside a backlit leaf must not lend it its colour), weighted by brightness
    bright = P.max(axis=2)
    ring = binary_dilation(A2, iterations=4) & ~A2 & (P.min(axis=2) > 1e-4) & (bright > 0.25)
    if ring.sum() < 4:
        log.debug("highlights: no valid edge around the clipped areas; left clipped")
        return None
    logP = np.log(np.maximum(P, 1e-6))
    logM = np.log(np.maximum(P.mean(axis=2), 1e-6))
    w = (ring * np.minimum(bright, 1.0) ** 2).astype(np.float32)
    q = np.zeros_like(P)
    done = np.zeros(A.shape, bool)
    for s in SIGMAS:
        den = gaussian_filter(w, s)
        ok = (den > 0.01) & ~done
        if ok.any():
            for c in range(3):
                num = gaussian_filter(((logP[..., c] - logM) * w).astype(np.float32), s)
                q[..., c][ok] = num[ok] / den[ok]
            done |= ok
        if done[A].all():
            break
    # brightness from the unclipped channels (or, if all clip, the least that fits all of them)
    U = ~K
    nu = U.sum(axis=2)
    lu = np.where(U, logP - q, 0.0).sum(axis=2) / np.maximum(nu, 1)
    lall = (logP - q).max(axis=2)
    logL = np.where(nu > 0, lu, lall)
    est = np.exp(q + logL[..., None]).astype(np.float32)
    est = np.where(K, np.maximum(est, P), P)
    return y0, x0, est.astype(np.float32)


def maps_for(raw, levels, mult) -> tuple | None:
    """Rebuild maps for the whole frame at this white balance (see Image.highlight_maps)."""
    with slog.stage("highlights.estimate", log):
        mos = lin.normalise(raw, levels)
        P, K = _planes(mos, raw.cfa_rgb)
        del mos
        P *= np.asarray(mult, np.float32).reshape(1, 1, 3)
        r = estimate(P, K)
    if r is not None:
        log.info("highlights: rebuilt %.3f%% of the frame", 100.0 * K.any(axis=2).mean())
    return r


def apply(mos: np.ndarray, pat: np.ndarray, mult, maps, y0: int, x0: int) -> np.ndarray:
    """Rebuild the clipped photosites of a white-balanced mosaic region whose top-left photosite is (y0, x0) of the
    full frame (both even). ``maps`` from maps_for."""
    if maps is None:
        return mos
    my, mx, est = maps
    h2, w2 = est.shape[:2]
    for i in (0, 1):
        for j in (0, 1):
            c = int(pat[i, j])
            v = mos[i::2, j::2]
            ky0, kx0 = y0 // 2 - my, x0 // 2 - mx
            ky = np.arange(v.shape[0]) + ky0
            kx = np.arange(v.shape[1]) + kx0
            iy = (ky >= 0) & (ky < h2)
            ix = (kx >= 0) & (kx < w2)
            if not iy.any() or not ix.any():
                continue
            sub = v[np.ix_(iy, ix)]
            e = est[ky[iy]][:, kx[ix], c]
            clipped = sub >= np.float32(CLIP * mult[c])
            sub[clipped] = np.maximum(sub[clipped], e[clipped])
            v[np.ix_(iy, ix)] = sub
    return mos
