"""Night white balance: SGP-LRD (Salient Gray Pixels with Local Reflectance Differences), from RL-AWB.

Ported from the authors' reference code, utils/Algorithm.py of https://github.com/BrianChen1120/RL-AWB:

    Copyright (c) 2026-present, Yuan-Kang Lee, Kuan-Lin Chen
    National Taiwan University, Taiwan; National Yang Ming Chiao Tung University, Taiwan
    Licensed under the Apache License, Version 2.0 (studio/engine/color/LICENSE.apache-2.0)
    Paper: Yuan-Kang Lee, Kuan-Lin Chen, Chia-Che Chang, Yu-Lun Liu, "RL-AWB: Deep Reinforcement Learning for
    Auto White Balance Correction in Low-Light Night-time Scenes", arXiv 2601.05249, 2026.

Changes made here (Apache 2.0, section 4b): vectorised with SciPy filters (the 3 x 3 window sums, counts and
maxima, the 7 x 7 box mean) in place of sliding windows and convolve2d, same arithmetic; takes the Studio's
half-size raw camera RGB (0..1) and scales it to 14-bit counts as the reference does; the dataset loading,
colour-checker masks and error measures removed. And the reinforcement-learning agent that tunes two settings
per photo (the grey-pixel share N and the Minkowski order p) is replaced by a deterministic choice: every setting
on a small grid is run, and the one whose answer sits closest to the middle of all the answers (the medoid) is
used - an unstable setting gives an outlier, a stable one agrees with its neighbours. The spread of the answers is
reported as the confidence.

How SGP-LRD works, in short: grey surfaces have the same local contrast in every colour channel (in log space), so
pixels whose contrast vector points along (1, 1, 1) are candidate greys; at night two filters keep only the
reliable ones (enough variation between channels to be lit, not too far from the image's mean colour); the
brighter ones are trusted more (a weight set by how skewed the brightness is); the light is a Minkowski mean of the
candidates' local averages over their local reflectance (average / maximum in a 3 x 3 window).
No GUI imports (rule 2.2).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, maximum_filter, uniform_filter
from scipy.stats import skew

from ... import log as slog

log = slog.get("engine.nightwb")

COUNTS = 2 ** 14 - 1                 # the reference works on 14-bit raw counts (its thresholds assume them)
LONG_EDGE = 750                      # the reference shrinks a 6000 x 4000 raw by 1/8 (nearest neighbour)
BASE = dict(sigma=0.5, var_th=0.025, color_th=0.35, conf_th=0.9, window=3)   # its LEVI settings
GRID_N = (1.0, 1.25, 1.5, 1.75, 2.0)  # grey-pixel share, percent (the agent's range)
GRID_P = (2.0, 4.0, 7.0, 10.0, 15.0, 22.0, 32.0, 40.0)   # Minkowski order (the agent's range 1-40)


def _deriv_gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    pw = np.arange(1, 51)
    valid = np.nonzero(np.exp(-(pw ** 2) / (2 * sigma * sigma)) > 1e-6)[0]
    width = valid[-1] + 1 if valid.size else 1
    xs = np.arange(-width, width + 1)
    x, y = np.meshgrid(xs, xs)
    k = -x * np.exp(-(x ** 2 + y ** 2) / (2 * sigma * sigma)) / (np.pi * sigma * sigma)
    ax = convolve(img, k, mode="nearest")
    ay = convolve(img, k.T, mode="nearest")
    return np.sqrt(ax ** 2 + ay ** 2)


def _shrink(cam: np.ndarray) -> np.ndarray:
    n = max(1, int(round(max(cam.shape[:2]) / float(LONG_EDGE))))
    return cam[n // 2::n, n // 2::n] if n > 1 else cam


def _prepare(cam: np.ndarray, sigma: float, var_th: float, color_th: float):
    """What does not depend on N or p: the grey angle map and the night filters (reference: gray_index_angular,
    ComputeSaliencyMap)."""
    img = np.clip(_shrink(np.asarray(cam, np.float64)), 0.0, 1.0) * COUNTS
    eps = np.finfo(float).eps
    rr, cc, _ = img.shape
    sat = img.max(axis=2) >= 0.95 * img.max()
    sat = maximum_filter(sat.astype(np.uint8), 3, mode="nearest").astype(bool)
    border = np.zeros((rr, cc), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    excluded = sat | border
    M = [_deriv_gauss(np.log(np.where(img[..., c] == 0, eps, img[..., c])), sigma) for c in range(3)]
    data = np.stack([np.where(m == 0, eps, m) for m in M], -1).reshape(-1, 3)
    dn = data / np.maximum(np.linalg.norm(data, axis=1, keepdims=True), eps)
    ang = np.arccos(np.clip(dn.sum(axis=1) / np.sqrt(3.0), -1.0, 1.0)).reshape(rr, cc)
    flat = (M[0] < eps) & (M[1] < eps) & (M[2] < eps)
    ang[flat] = ang.max()
    pad = np.pad(ang, 3, mode="wrap")                       # convolve2d(..., boundary="wrap"), 7 x 7 mean
    ang = uniform_filter(pad, 7, mode="constant")[3:-3, 3:-3]
    ang[excluded] = ang.max()
    lg = np.log(img + 1.0)                                   # saliency (night filters), whole frame valid
    lvf = np.var(lg, axis=2, ddof=1) > var_th
    Mc = lg.reshape(-1, 3).mean(axis=0)
    dev = np.abs(lg - Mc).max(axis=2) > color_th * Mc.min()
    sal = lvf & ~dev
    return img, ang, lvf, sal


def _confidence(img_masked: np.ndarray, mask: np.ndarray, conf_th: float) -> np.ndarray:
    im = np.clip(img_masked / COUNTS, 0.0, 1.0)
    ow = im.mean(axis=2) * mask
    nz = ow[ow != 0]
    if nz.size == 0:
        sk, m = 0.0, 1.0
    else:
        sk, m = float(skew(nz)), float(nz.mean())
    E = 1.0 if sk > 1.5 else (2.0 if sk > 0.2 else 4.0)
    conf = 1.0 - np.exp(-((ow / m) ** E))
    c2 = np.where(conf < conf_th, 0.0, conf)
    return c2 if c2.sum() > 0 else conf


def _illuminant(img_masked: np.ndarray, mask: np.ndarray, p: float, conf_th: float, window: int) -> np.ndarray:
    """Reference: estimate_illuminant_pixelwise_accelerated (window sums by filters, 'edge' padding)."""
    w = _confidence(img_masked, mask, conf_th)
    n2 = float(window * window)
    num, den = np.zeros(3), np.zeros(3)
    for c in range(3):
        x = img_masked[..., c]
        mx = maximum_filter(x, window, mode="nearest")
        s = uniform_filter(x, window, mode="nearest") * n2
        cnt = uniform_filter((x != 0).astype(np.float64), window, mode="nearest") * n2
        mean = np.divide(s, cnt, out=np.zeros_like(s), where=cnt > 0.5)
        ok = mx > 0
        norm = np.zeros_like(mean)
        norm[ok] = mean[ok] / mx[ok]
        num[c] = np.sum(np.abs((mean * w)[ok]) ** p)
        den[c] = np.sum(np.abs((norm * w)[ok]) ** p)
    e = np.where(den > 0, (num / np.where(den > 0, den, 1.0)) ** (1.0 / p), 0.0)
    nv = np.linalg.norm(e)
    return e / nv if nv > 0 else e


def sgp_lrd(cam: np.ndarray, n_pct: float = 2.0, p: float = 10.0, night: bool = True, _prep=None) -> np.ndarray:
    """The light's colour (R, G, B, unit length) by SGP-LRD with one setting; zeros if it cannot be told."""
    b = BASE
    img, ang, lvf, sal = _prep or _prepare(cam, b["sigma"], b["var_th"], b["color_th"])
    t1 = np.percentile(ang, n_pct)
    t2 = np.percentile(ang, 0.5)
    gp1, gp2 = ang <= t1, ang <= t2
    gps, sgps = (gp2 & lvf, gp1 & sal) if night else (gp2, gp1)
    need = int(np.floor(0.05 * ang.size / 100.0))
    sel = sgps if np.count_nonzero(sgps) >= need else gps
    if not np.any(sel):
        return np.zeros(3)
    return _illuminant(img * sel[..., None], sel.astype(np.float64), p, b["conf_th"], b["window"])


def estimate(cam: np.ndarray, night: bool = True) -> dict | None:
    """Every (N, p) on the grid, and the medoid answer: {"light": (R, G, B) sum 1, "n", "p", "spread_deg",
    "runs"}, or None if no setting could tell."""
    prep = _prepare(cam, BASE["sigma"], BASE["var_th"], BASE["color_th"])
    runs = []
    for n in GRID_N:
        for p in GRID_P:
            e = sgp_lrd(cam, n, p, night, _prep=prep)
            if np.all(e > 0):
                runs.append((n, p, e / e.sum()))
    if not runs:
        return None
    E = np.array([r[2] for r in runs])
    u = E / np.linalg.norm(E, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(u @ u.T, -1.0, 1.0)))
    k = int(np.argmin(np.median(ang, axis=1)))               # closest to all the others
    spread = float(np.median(ang[k]))
    n, p, e = runs[k]
    log.info("night white balance (SGP-LRD): %d settings, picked N=%.2f%% p=%.0f, light %s, spread %.2f deg",
             len(runs), n, p, np.round(e, 4).tolist(), spread)
    return {"light": e, "n": n, "p": p, "spread_deg": spread, "runs": len(runs)}
