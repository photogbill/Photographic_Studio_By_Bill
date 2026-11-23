"""More maps for games from a measured height (BUILD_PLAN_DP3D C): ambient occlusion, cavity and curvature, de-lit
colour (albedo), fine relief, roughness on request, a seam-cut tiling, scale from a coin or two clicks, game sizes
and ORM packing. No Qt here.

Honesty about how each map was made (plan 0.4, rule 1):
  computed from the measured height   ao, cavity, curvature (geometry only: they inherit the height's own labels)
  estimated (amber)                   albedo (the light fitted to the measured shape and divided out), detail (fine
                                      relief from the shading left over), roughness (from the colour's local
                                      variation: nothing in these photos measures shine)
"""
from __future__ import annotations

import math

import numpy as np

from ... import log as slog

log = slog.get("engine.stack.maps")

# map -> (how made, colour of plan 23.1, colour space of the PNG)
HOW = {
    "ao": ("computed from the measured height", "cyan", "linear"),
    "cavity": ("computed from the measured height", "cyan", "linear"),
    "curvature": ("computed from the measured height", "cyan", "linear"),
    "albedo": ("estimated: the light fitted to the measured shape and divided out", "amber", "srgb"),
    "detail": ("estimated: fine relief from the shading left over after de-lighting; never mixed into the height",
               "amber", "linear"),
    "roughness": ("estimated from the colour's local variation (nothing in these photos measures shine)", "amber",
                  "linear"),
    "orm": ("packed: occlusion (R) from the height, roughness (G) estimated or constant, metallic (B) 0", "amber",
            "linear"),
    "gaps": ("measured: places no frame had in focus", "amber", "linear"),
}


def _mode(wrap: bool) -> str:
    return "wrap" if wrap else "nearest"


# ------------------------------------------------------------------------------------------ AO, cavity (C.1)
def _work_size(h: np.ndarray, cap: int = 2048) -> tuple[np.ndarray, float]:
    """(height at most ``cap`` on its long edge, the factor it was shrunk by)."""
    k = max(1, int(math.ceil(max(h.shape) / float(cap))))
    if k == 1:
        return np.asarray(h, np.float32), 1.0
    from .align import bin_by
    return bin_by(np.asarray(h, np.float32), k), float(k)


def _back(m: np.ndarray, shape) -> np.ndarray:
    if m.shape == tuple(shape[:2]):
        return m
    from .surface import _up
    return _up(m, shape)


def ambient_occlusion(height: np.ndarray, slope: float, radius: float | None = None, directions: int = 8,
                      wrap: bool = False, samples: int = 12) -> np.ndarray:
    """Horizon-based ambient occlusion, 0 (closed) .. 1 (open), from a height map (near = high) whose units become
    a slope through ``slope`` (height units per pixel -> rise per pixel, as the normal map's). ``radius`` in pixels
    (default 1/32 of the long edge). For each of 8 directions the highest horizon within the radius is found; the
    occlusion is the mean sine of those horizon angles (Bavoil et al. 2008, as an idea). Worked at most 2048 px on
    the long edge, then brought back to size; wrap-around at the edges when tileable."""
    from scipy.ndimage import shift as nd_shift
    H, W = height.shape
    hs, k = _work_size(height)
    r = float(radius if radius is not None else max(H, W) / 32.0) / k
    r = max(2.0, r)
    s = float(slope) * k                             # rise per working pixel
    occ = np.zeros(hs.shape, np.float32)
    for a in range(directions):
        ang = 2 * math.pi * a / directions
        dy, dx = math.sin(ang), math.cos(ang)
        best = np.zeros(hs.shape, np.float32)
        for t in np.geomspace(1.0, r, samples):
            if wrap:
                sh = np.roll(hs, (int(round(-dy * t)), int(round(-dx * t))), (0, 1))
            else:
                sh = nd_shift(hs, (-dy * t, -dx * t), order=1, mode="nearest", prefilter=False)
            tan = (sh - hs) * np.float32(s / t)
            np.maximum(best, tan, out=best)
        occ += best / np.sqrt(1.0 + best * best)     # sin of the horizon angle
    ao = 1.0 - occ / float(directions)
    return np.clip(_back(ao.astype(np.float32), (H, W)), 0.0, 1.0)


def curvature(height: np.ndarray, slope: float, sigma: float = 2.0, wrap: bool = False) -> np.ndarray:
    """Signed mean curvature (scale-normalised Laplacian of the slope-scaled height): > 0 on ridges and bumps, < 0
    in crevices."""
    from scipy.ndimage import gaussian_laplace
    return (-gaussian_laplace(np.asarray(height, np.float32) * np.float32(slope), sigma, mode=_mode(wrap))
            * np.float32(sigma)).astype(np.float32)


def curvature_png(c: np.ndarray) -> np.ndarray:
    """0..1, 0.5 = flat; the steep end of the picture's own curvature reaches 0 / 1."""
    sc = float(np.percentile(np.abs(c), 99)) or 1e-6
    return np.clip(0.5 + 0.5 * c / sc, 0.0, 1.0).astype(np.float32)


def cavity(height: np.ndarray, slope: float, sigma: float = 1.5, wrap: bool = False) -> np.ndarray:
    """1 = open, darker in the small crevices (negative curvature at a small radius), 0..1."""
    c = curvature(height, slope, sigma, wrap)
    neg = np.maximum(-c, 0.0)
    sc = float(np.percentile(neg, 99.5)) or 1e-6
    return np.clip(1.0 - neg / sc, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------------------------------ de-lighting (C.2)
def _normals_at(height: np.ndarray, slope: float, sigma: float, wrap: bool = False) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    from .surface import normals
    hs = gaussian_filter(np.asarray(height, np.float32), sigma, mode=_mode(wrap)) if sigma > 0 else height
    return normals(hs, slope, True, wrap=wrap)


def fit_light(luma: np.ndarray, nrm: np.ndarray, trust: np.ndarray | None = None) -> dict:
    """One directional light plus ambient fitted to a low-passed luminance and the normals at the same scale:
    L ~ a + b * max(0, n.l), robustly (the darkest 5% and brightest 2% left out: shadows and shine), by a search
    over directions (every 10 degrees round, 10 to 90 degrees up) and least squares for a, b at each. Returns
    {"dir" (x right, y up, z out), "azimuth", "elevation", "a", "b", "r2", "ok"}."""
    L = np.asarray(luma, np.float64)
    m = np.isfinite(L)
    if trust is not None:
        m &= trust
    lo, hi = np.percentile(L[m], [5, 98]) if m.sum() > 100 else (L.min(), L.max())
    m &= (L > lo) & (L < hi)
    step = max(1, int(math.sqrt(m.sum() / 60000.0)))
    sel = np.zeros_like(m)
    sel[::step, ::step] = True
    m &= sel
    if m.sum() < 200:
        return {"ok": False, "r2": 0.0, "dir": [0.0, 0.0, 1.0], "azimuth": 0.0, "elevation": 90.0, "a": 1.0,
                "b": 0.0}
    y = L[m]
    N = nrm[m].astype(np.float64)
    best = None
    var = float(np.var(y)) or 1e-12
    for el in range(10, 91, 10):
        for az in range(0, 360, 10 if el < 90 else 360):
            e, z = math.radians(el), math.radians(az)
            l = np.array([math.cos(e) * math.cos(z), math.cos(e) * math.sin(z), math.sin(e)])
            x = np.maximum(N @ l, 0.0)
            X = np.stack([np.ones_like(x), x], 1)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            if coef[1] <= 0:
                continue
            r2 = 1.0 - float(np.mean((X @ coef - y) ** 2)) / var
            if best is None or r2 > best["r2"]:
                best = {"r2": r2, "dir": l.tolist(), "azimuth": float(az), "elevation": float(el),
                        "a": float(coef[0]), "b": float(coef[1])}
    if best is None:
        return {"ok": False, "r2": 0.0, "dir": [0.0, 0.0, 1.0], "azimuth": 0.0, "elevation": 90.0, "a": 1.0,
                "b": 0.0}
    best["ok"] = best["r2"] > 0.02
    return best


def delight(colour01: np.ndarray, height: np.ndarray, slope: float, trust: np.ndarray | None = None,
            wrap: bool = False, sigma: float | None = None) -> tuple[np.ndarray, dict, np.ndarray]:
    """(albedo 0..1 linear, light fit, shading). The shading is the fitted light on the measured shape at the
    shading's own scale (normals of the height smoothed by ``sigma``, default 1/200 of the long edge): dividing it
    out removes the light the shape caught without touching the texture's fine detail (it is not in the shading).
    ``colour01``: linear RGB 0..1. Where the fit is not trusted (no measured height), the shading is 1."""
    from scipy.ndimage import gaussian_filter
    H, W = height.shape
    sig = float(sigma if sigma is not None else max(2.0, max(H, W) / 200.0))
    hs, k = _work_size(height, 1024)
    cs = colour01
    if k > 1:
        from .align import bin_by
        cs = bin_by(np.asarray(colour01, np.float32), int(k))
        tr_s = bin_by(trust.astype(np.float32), int(k)) > 0.5 if trust is not None else None
    else:
        tr_s = trust
    nrm = _normals_at(hs, slope * k, sig / k, wrap)
    Y = 0.2126 * cs[..., 0] + 0.7152 * cs[..., 1] + 0.0722 * cs[..., 2]
    Ylp = gaussian_filter(Y.astype(np.float32), sig / k, mode=_mode(wrap))
    fit = fit_light(Ylp, nrm, tr_s)
    if not fit["ok"]:
        return np.clip(colour01, 0.0, 1.0).astype(np.float32), fit, np.ones((H, W), np.float32)
    l = np.asarray(fit["dir"], np.float32)
    nfull = _normals_at(height, slope, sig, wrap)
    sh = np.float32(fit["a"]) + np.float32(fit["b"]) * np.maximum(nfull @ l, 0.0)
    sh = sh / max(float(np.mean(sh)), 1e-6)
    if trust is not None:                            # unmeasured places keep their colour (shading 1)
        t = gaussian_filter(trust.astype(np.float32), 3.0)
        sh = t * sh + (1.0 - t)
    sh = np.clip(sh, 0.2, 5.0).astype(np.float32)
    alb = np.clip(colour01 / sh[..., None], 0.0, 1.0).astype(np.float32)
    fit["trusted_share"] = float(trust.mean()) if trust is not None else 1.0
    return alb, fit, sh


def detail_relief(colour01: np.ndarray, shading: np.ndarray, sigma: float = 4.0, wrap: bool = False) -> np.ndarray:
    """Fine relief estimated from what the shading leaves (amber): the high-passed log luminance of the de-lit
    colour, 0.5 = flat, as its own file (never mixed into the measured height)."""
    from scipy.ndimage import gaussian_filter
    Y = 0.2126 * colour01[..., 0] + 0.7152 * colour01[..., 1] + 0.0722 * colour01[..., 2]
    lg = np.log(np.maximum(Y / np.maximum(shading, 1e-3), 1e-4)).astype(np.float32)
    hp = lg - gaussian_filter(lg, sigma, mode=_mode(wrap))
    sc = float(np.percentile(np.abs(hp), 99)) or 1e-6
    return np.clip(0.5 + 0.5 * hp / sc, 0.0, 1.0).astype(np.float32)


def estimated_height(colour01: np.ndarray, sizes=(2.0, 4.0, 8.0, 16.0, 32.0), wrap: bool = False) -> np.ndarray:
    """Height ESTIMATED from one ordinary photo (amber; nothing is measured): the broad light is taken off (log
    luminance minus a very wide blur), then dark is taken as deep, band by band, the finer bands weighing less (a
    crevice is dark at every size, a speck of dirt only at the finest). 0..1, 1 = high. The guess texture tools make;
    the JSON says so wherever it is used."""
    from scipy.ndimage import gaussian_filter
    x = np.asarray(colour01, np.float32)
    Y = x if x.ndim == 2 else 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    lg = np.log(np.maximum(Y, 1e-4)).astype(np.float32)
    m = _mode(wrap)
    lg = lg - gaussian_filter(lg, max(Y.shape) / 8.0, mode=m)
    h = np.zeros_like(lg)
    prev = lg
    for s in sizes:
        blur = gaussian_filter(lg, s, mode=m)
        h += (prev - blur) * (s / sizes[-1]) ** 0.5   # the band between the last size and this one
        prev = blur
    h += (prev - prev.mean()) * 1.0                   # what is left: the widest band
    lo, hi = np.percentile(h, [0.5, 99.5])
    return np.clip((h - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)


def roughness_estimate(albedo01: np.ndarray, wrap: bool = False) -> np.ndarray:
    """Roughness ESTIMATED from the albedo's local variation (amber): busy, varied surfaces read rough (0.95),
    smooth even ones less so (down to 0.55). Nothing in these photos measures shine; say so wherever it is used."""
    from scipy.ndimage import uniform_filter
    Y = 0.2126 * albedo01[..., 0] + 0.7152 * albedo01[..., 1] + 0.0722 * albedo01[..., 2]
    m = uniform_filter(Y, 9, mode=_mode(wrap))
    v = np.sqrt(np.maximum(uniform_filter(Y * Y, 9, mode=_mode(wrap)) - m * m, 0.0)) / np.maximum(m, 1e-3)
    sc = float(np.percentile(v, 95)) or 1e-6
    return np.clip(0.55 + 0.4 * v / sc, 0.55, 0.95).astype(np.float32)


# ------------------------------------------------------------------------------------------ seam-cut tiling (C.5)
def _min_cut(cost: np.ndarray, cyclic_start: int | None = None) -> np.ndarray:
    """A minimum-cost path down the rows of ``cost`` (h, w): one column per row, moving at most one column per row
    (dynamic programming). ``cyclic_start``: the path starts and ends at this column (so a cut across a repeating
    edge meets itself)."""
    h, w = cost.shape
    acc = np.full((h, w), np.inf, np.float64)
    back = np.zeros((h, w), np.int8)
    if cyclic_start is None:
        acc[0] = cost[0]
    else:
        acc[0, cyclic_start] = cost[0, cyclic_start]
    for y in range(1, h):
        prev = acc[y - 1]
        left = np.concatenate([[np.inf], prev[:-1]])
        right = np.concatenate([prev[1:], [np.inf]])
        stack = np.stack([left, prev, right])
        k = np.argmin(stack, axis=0)
        acc[y] = cost[y] + stack[k, np.arange(w)]
        back[y] = (k - 1).astype(np.int8)
    path = np.zeros(h, np.int64)
    path[-1] = int(np.argmin(acc[-1])) if cyclic_start is None else int(cyclic_start)
    for y in range(h - 1, 0, -1):
        path[y - 1] = path[y] + back[y, path[y]]
    return np.clip(path, 0, w - 1)


def cut_weights(guide: np.ndarray, band: float = 0.12, feather: float = 3.0) -> dict:
    """The seam-cut tiling's geometry, worked out once from ``guide`` (the colour or the height, (h, w) or (h, w,
    c)) and applied to every map: per axis, a mask saying which pixels come from the copy offset by half (whose
    picture runs on across that axis's edges) and which from the copy as it is, the boundary being two minimum-
    error cuts through a band around the offset copy's seam, feathered over a few pixels."""
    g = np.asarray(guide, np.float32)
    if g.ndim == 2:
        g = g[..., None]
    out = {}
    for axis in (1, 0):                              # columns first (a vertical cut), then rows
        A0 = g if axis == 1 else np.transpose(g, (1, 0, 2))
        h, w = A0.shape[:2]
        B0 = np.roll(A0, -(w // 2), 1)               # its seam sits at column w - w // 2
        c = w - w // 2
        bw = max(4, int(band * w))
        lo, hi = max(1, c - bw), min(w - 1, c + bw)
        diff = np.sum((A0[:, lo:hi] - B0[:, lo:hi]) ** 2, axis=2)
        mid = c - lo
        cost_l = diff[:, :mid].copy()
        cost_r = diff[:, mid:].copy()
        start_l = int(np.argmin(cost_l[0]))
        start_r = int(np.argmin(cost_r[0]))
        pl = _min_cut(cost_l, start_l) + lo
        pr = _min_cut(cost_r, start_r) + lo + mid
        x = np.arange(w)[None, :]
        inside = (x > pl[:, None]) & (x < pr[:, None])        # between the cuts: the copy as it is (A)
        from scipy.ndimage import gaussian_filter
        m = gaussian_filter(inside.astype(np.float32), feather, mode="wrap")
        mB = 1.0 - m                                  # everywhere else: the offset copy
        out[axis] = mB if axis == 1 else mB.T
    return out


def tile_cut(arr: np.ndarray, cw: dict) -> np.ndarray:
    """One map made tileable with cut_weights' geometry (columns, then rows)."""
    if np.ndim(arr) == 3:
        return np.stack([tile_cut(arr[..., i], cw) for i in range(arr.shape[2])], 2)
    a = np.asarray(arr, np.float32)
    H, W = a.shape
    m1 = cw[1]
    a1 = m1 * np.roll(a, -(W // 2), 1) + (1 - m1) * a
    m0 = cw[0]
    return (m0 * np.roll(a1, -(H // 2), 0) + (1 - m0) * a1).astype(np.float32)


def cut_weights_for(guide: np.ndarray, band: float = 0.12) -> dict:
    """cut_weights for the two passes: the row pass is cut on the picture after the column pass."""
    g = np.asarray(guide, np.float32)
    c1 = cut_weights(g, band)
    g1 = tile_cut(g, {1: c1[1], 0: np.zeros(g.shape[:2], np.float32)})
    c0 = cut_weights(g1, band)
    return {1: c1[1], 0: c0[0]}


# ------------------------------------------------------------------------------------------ scale (C.6 b)
COINS_MM = {"us_quarter": 24.26, "us_penny": 19.05, "us_nickel": 21.21, "us_dime": 17.91, "euro_1": 23.25,
            "euro_2": 25.75, "uk_pound": 23.43, "ca_quarter": 23.88}


def find_coin(rgb01: np.ndarray, coin: str = "us_quarter", r_min_frac: float = 0.01,
              r_max_frac: float = 0.2) -> dict | None:
    """A coin in the picture (the all-in-focus frame): the strongest circle by a Hough vote on the edges' own
    direction (each edge pixel votes along its gradient for centres at every radius), refined to the edge with a
    least-squares circle fit. {"centre": (u, v) 0..1, "radius_px", "mm_per_px", "coin", "score"} or None."""
    from scipy.ndimage import gaussian_filter, maximum_filter, sobel
    Y = np.asarray(0.25 * rgb01[..., 0] + 0.5 * rgb01[..., 1] + 0.25 * rgb01[..., 2], np.float32)
    H, W = Y.shape
    k = max(1, int(math.ceil(max(H, W) / 1024.0)))
    if k > 1:
        from .align import bin_by
        Y = bin_by(Y, k)
    h, w = Y.shape
    g = gaussian_filter(Y, 1.5)
    gx, gy = sobel(g, 1), sobel(g, 0)
    mag = np.hypot(gx, gy)
    thr = float(np.percentile(mag, 97))
    ys, xs = np.nonzero(mag > thr)
    if ys.size < 50:
        return None
    if ys.size > 60000:
        sel = np.random.default_rng(1).choice(ys.size, 60000, replace=False)
        ys, xs = ys[sel], xs[sel]
    ux, uy = gx[ys, xs] / mag[ys, xs], gy[ys, xs] / mag[ys, xs]
    r_lo, r_hi = max(4, int(r_min_frac * min(h, w))), int(r_max_frac * min(h, w))
    radii = np.arange(r_lo, r_hi + 1)
    best = None
    from scipy.ndimage import uniform_filter
    for r in radii:
        acc = np.zeros((h, w), np.float32)
        for sgn in (1, -1):                          # a coin brighter or darker than its surround
            cx = np.rint(xs + sgn * r * ux).astype(int)
            cy = np.rint(ys + sgn * r * uy).astype(int)
            ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
            np.add.at(acc, (cy[ok], cx[ok]), 1.0)
        win = uniform_filter(acc, 5, mode="constant") * 25.0      # the votes landing within 2 px of a centre
        score = win / (2 * math.pi * r)              # as a share of the circumference
        i = int(np.argmax(score))
        sv = float(score.flat[i])
        if best is None or sv > best[0]:
            best = (sv, i // w, i % w, int(r))
    if best is None or best[0] < 0.6:
        return None
    sv, cy0, cx0, r0 = best
    # refine: least-squares circle through the strong edges within 20% of that circle
    d = np.hypot(xs - cx0, ys - cy0)
    m = np.abs(d - r0) < 0.2 * r0
    if m.sum() >= 20:
        X = np.stack([xs[m], ys[m], np.ones(m.sum())], 1).astype(np.float64)
        b = (xs[m] ** 2 + ys[m] ** 2).astype(np.float64)
        sol, *_ = np.linalg.lstsq(X, b, rcond=None)
        cx0, cy0 = sol[0] / 2, sol[1] / 2
        r0 = float(math.sqrt(max(sol[2] + cx0 * cx0 + cy0 * cy0, 1.0)))
    mm = COINS_MM.get(coin)
    if mm is None:
        raise ValueError(f"unknown coin {coin!r}: one of {', '.join(COINS_MM)}")
    r_full = r0 * k
    return {"centre": (float((cx0 + 0.5) * k / W), float((cy0 + 0.5) * k / H)), "radius_px": float(r_full),
            "mm_per_px": float(mm / (2.0 * r_full)), "coin": coin, "coin_mm": mm, "score": round(sv, 3)}


def scale_from_click(u1, v1, u2, v2, mm, W, H) -> float:
    """mm per pixel from two points (0..1 of the picture) a known distance apart."""
    d = math.hypot((u2 - u1) * W, (v2 - v1) * H)
    if d < 2:
        raise ValueError("the two points are the same place")
    return float(mm) / d


def magnification_from_distance(focal_mm: float, distance_m: float) -> float | None:
    """Thin-lens magnification from the focus distance the lens recorded (subject to sensor): m = f / (d - f)
    roughly, with d measured from the sensor (amber: coarse)."""
    d = float(distance_m) * 1000.0
    f = float(focal_mm)
    if not (f > 0 and d > 2 * f):
        return None
    # d = u + v with 1/u + 1/v = 1/f: m = v / u
    disc = d * d - 4 * f * d
    if disc <= 0:
        return None
    u = (d + math.sqrt(disc)) / 2.0
    v = d - u
    return float(v / u)


def view_edge_to_edge(pixel_mm: float, magnification: float, W: int, H: int) -> tuple:
    """How much of the subject the frame covers, mm (width, height): the sensor's size over the magnification."""
    return float(W * pixel_mm / magnification), float(H * pixel_mm / magnification)


# ------------------------------------------------------------------------------------------ game sizes (C.7)
def best_square(label: np.ndarray) -> tuple[int, int, int]:
    """(y0, x0, side) of the largest square, at least 60% of the short edge, whose share of measured places is the
    highest (the measured area's best part)."""
    from scipy.ndimage import uniform_filter
    H, W = label.shape
    side = min(H, W)
    meas = (np.asarray(label) > 0).astype(np.float32)
    best = None
    for frac in (1.0, 0.9, 0.8, 0.7, 0.6):
        s = int(side * frac)
        k = max(1, s // 64)
        small = meas[::k, ::k]
        win = max(1, s // k)
        if win > min(small.shape):
            continue
        m = uniform_filter(small, win, mode="constant")
        hy, hx = win // 2, win // 2
        valid = m[hy:small.shape[0] - (win - hy) + 1, hx:small.shape[1] - (win - hx) + 1]
        if valid.size == 0:
            continue
        i = int(np.argmax(valid))
        sh = float(valid.flat[i]) * (frac ** 0.25)    # prefer larger squares a little
        y0 = (i // valid.shape[1]) * k
        x0 = (i % valid.shape[1]) * k
        if best is None or sh > best[0]:
            best = (sh, y0, x0, s)
    if best is None:
        return 0, 0, side
    _, y0, x0, s = best
    return int(min(y0, H - s)), int(min(x0, W - s)), int(s)


def resize_to(arr: np.ndarray, size: int) -> np.ndarray:
    """A square map resized to size x size (area average when shrinking: no single-pixel detail survives that
    would shimmer in mipmaps)."""
    from PIL import Image
    a = np.asarray(arr, np.float32)
    chans = [a] if a.ndim == 2 else [a[..., i] for i in range(a.shape[2])]
    out = [np.asarray(Image.fromarray(c).resize((size, size), Image.Resampling.BOX if c.shape[0] > size else
                                                 Image.Resampling.BICUBIC), np.float32) for c in chans]
    return out[0] if a.ndim == 2 else np.stack(out, 2)


def _resize_hw(a: np.ndarray, h: int, w: int) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(np.asarray(a, np.float32)).resize((w, h), Image.Resampling.BILINEAR), np.float32)


def low_pass(a: np.ndarray, frac: float = 0.125) -> np.ndarray:
    """The large-scale part of a 2-D map: a Gaussian blur whose sigma is ``frac`` of the long edge, worked out on a
    copy about 256 pixels across (the blur keeps nothing finer than that) and brought back to full size."""
    from scipy.ndimage import gaussian_filter
    a = np.asarray(a, np.float32)
    H, W = a.shape
    k = max(1, int(max(H, W) // 256))
    h2, w2 = max(1, H // k), max(1, W // k)
    small = _resize_hw(a, h2, w2) if k > 1 else a
    s = gaussian_filter(small, sigma=max(0.5, frac * max(h2, w2)), mode="reflect")
    return _resize_hw(s, H, W) if k > 1 else s


def poly_surface(a: np.ndarray, deg: int = 3) -> np.ndarray:
    """The smooth surface (a 2-D polynomial of degree ``deg`` in x and y) that fits a map best, by least squares on a
    copy about 256 pixels across. Unlike a blur it has no edge bias: a plane or a lens's vignetting is followed right
    into the corners, where a texture's seams are."""
    a = np.asarray(a, np.float32)
    H, W = a.shape
    k = max(1, int(max(H, W) // 256))
    h2, w2 = max(2, H // k), max(2, W // k)
    small = _resize_hw(a, h2, w2) if k > 1 else a

    def basis(hh, ww):
        y, x = np.mgrid[0:hh, 0:ww].astype(np.float64)
        y, x = y / max(hh - 1, 1) * 2 - 1, x / max(ww - 1, 1) * 2 - 1
        return np.stack([x ** i * y ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)
    B = basis(h2, w2).reshape(-1, (deg + 1) * (deg + 2) // 2)
    coef, *_ = np.linalg.lstsq(B, small.reshape(-1).astype(np.float64), rcond=None)
    out = np.empty((H, W), np.float32)
    for y0 in range(0, H, 512):                      # full size in strips: the basis of a whole 44 MP map is large
        yy, xx = np.mgrid[y0:min(H, y0 + 512), 0:W].astype(np.float64)
        yy, xx = yy / max(H - 1, 1) * 2 - 1, xx / max(W - 1, 1) * 2 - 1
        bb = np.stack([xx ** i * yy ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)
        out[y0:y0 + 512] = (bb @ coef).astype(np.float32)
    return out


def flatten_colour(col: np.ndarray, frac: float = 0.125) -> tuple[np.ndarray, float]:
    """Before tiling: take out the large-scale brightness changes of a colour picture (0..255 sRGB) - a lens's
    vignetting, light falling off across the frame - on its linear luminance (in stops): first the best smooth
    surface (poly_surface: no edge bias), then broad lumps larger than ``frac`` of the long edge. Detail, cracks
    and colour stay. Returns (picture, how much it changed: the 95th percentile of the correction in stops)."""
    from ..color.spaces import srgb_decode, srgb_encode
    lin = srgb_decode(np.clip(np.asarray(col, np.float32), 0, 255) / 255.0).astype(np.float32)
    L = lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722
    lg = np.log2(np.maximum(L, 1e-5))
    trend = poly_surface(lg)
    r = lg - trend
    trend = trend + low_pass(r, frac)
    corr = np.clip(float(np.median(trend)) - trend, -2.0, 2.0).astype(np.float32)
    out = srgb_encode(np.clip(lin * np.exp2(corr)[..., None], 0, 1)) * 255.0
    stops = float(np.percentile(np.abs(corr), 95))
    return out.astype(np.float32), stops


def flatten_height(h01: np.ndarray, frac: float = 0.125) -> np.ndarray:
    """Before tiling: take out the surface's overall tilt toward the camera and its broad bumps, which can never
    repeat without a step: first the best smooth surface (poly_surface: no edge bias), then lumps larger than
    ``frac`` of the long edge. The finer relief stays, spread again over 0..1 (0.5 and 99.5 percentiles)."""
    h = np.asarray(h01, np.float32)
    d = h - poly_surface(h)
    d = d - low_pass(d, frac)
    lo, hi = np.percentile(d, [0.5, 99.5])
    return np.clip((d - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)


def pack_orm(ao01: np.ndarray, rough01: np.ndarray | None, rough_const: float = 0.9) -> np.ndarray:
    """ORM (occlusion R, roughness G, metallic B), glTF's and Godot's ORMMaterial3D order."""
    r = np.full(ao01.shape, rough_const, np.float32) if rough01 is None else rough01
    return np.stack([ao01, r, np.zeros_like(ao01)], 2).astype(np.float32)
