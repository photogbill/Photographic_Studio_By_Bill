"""Colour match from a reference photo, solved as the Studio's own settings. A Cognitive Adjustment.

Photoshop's Color Transfer / Harmonization filters hand back a changed picture. Here the answer is a set of ordinary
slider values - exposure, contrast, white balance, saturation, and the shadow and highlight colour-grading wheels -
chosen so the photo's colour statistics match the reference's; every one of them can then be moved by hand, and
nothing is painted. Amount scales the change.

What is matched (OKLab, of the photo as displayed; Ottosson 2020):
    median lightness, lightness spread (5th-95th percentile), the mid-tones' mean a and b (the cast), colourfulness
    (the 75th percentile of each colour's distance from that cast, so a cast does not count as saturation), and the
    shadows' and highlights' a and b relative to the mid-tones (split toning).
How it is solved: nine settings for nine numbers. A finite-difference Jacobian from small renders of the real
pipeline (so the tone curve, gamut handling and every other stage are included), then damped Newton steps with
Broyden updates (Broyden 1965; one render per step), the best result kept. Reinhard, Ashikhmin, Gooch and Shirley's
colour transfer (2001) matches the same kind of statistics directly on pixels; here they steer settings instead.
No GUI imports (rule 2.2).
"""
from __future__ import annotations

import math
import time

import numpy as np

PARAMS = ("exposure", "contrast", "wb_red", "wb_blue", "saturation", "shadow_a", "shadow_b", "high_a", "high_b")
STEP = np.array([0.3, 0.08, 0.06, 0.06, 12.0, 0.15, 0.15, 0.15, 0.15])     # finite-difference steps
LIMIT = np.array([3.0, 0.6, 0.8, 0.8, 80.0, 1.0, 1.0, 1.0, 1.0])            # largest change allowed per setting
WEIGHT = np.array([10.0, 5.0, 10.0, 10.0, 20.0, 10.0, 10.0, 10.0, 10.0])   # OKLab units weigh alike (L, a, b)
REG = 0.12
STATS = ("median L", "L spread", "mid a", "mid b", "colourfulness", "shadow a", "shadow b", "highlight a",
         "highlight b")


def stats(rgb_pp: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """The nine colour statistics of a display-referred linear ProPhoto image (optionally weighted per pixel)."""
    from ..color.grade import to_oklab
    lab = to_oklab(np.clip(np.ascontiguousarray(rgb_pp, np.float32), 0.0, None)).reshape(-1, 3).astype(np.float64)
    w = np.ones(len(lab)) if weight is None else np.asarray(weight, np.float64).reshape(-1)
    keep = w > 1e-3
    lab, w = lab[keep], w[keep]
    if len(lab) < 16:
        raise ValueError("too few pixels to measure colour")
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    def pct(x, q):
        o = np.argsort(x)
        c = np.cumsum(w[o])
        return float(x[o][min(np.searchsorted(c, q * c[-1]), len(x) - 1)])

    def mean(x, m):
        ww = w * m
        return float((x * ww).sum() / max(ww.sum(), 1e-9))

    p5, p20, p25, p50, p75, p80, p95 = (pct(L, q) for q in (0.05, 0.2, 0.25, 0.5, 0.75, 0.8, 0.95))
    mid = ((L >= p25) & (L <= p75)).astype(float)
    sh = (L <= p20).astype(float)
    hi = (L >= p80).astype(float)
    am, bm = mean(a, mid), mean(b, mid)
    C = np.hypot(a - am, b - bm)
    return np.array([p50, p95 - p5, am, bm, pct(C, 0.75), mean(a, sh) - am, mean(b, sh) - bm,
                     mean(a, hi) - am, mean(b, hi) - bm])


def _wheel_vec(v) -> np.ndarray:
    """A grading wheel [hue, strength, lum] as a 2-vector in strength / 100 units."""
    if not v:
        return np.zeros(2)
    s = float(v[1]) / 100.0
    return np.array([s * math.cos(math.radians(float(v[0]))), s * math.sin(math.radians(float(v[0])))])


def _wheel_from(vec, old) -> list:
    s = float(min(1.0, np.hypot(*vec)))
    hue = math.degrees(math.atan2(vec[1], vec[0])) % 360.0 if s > 1e-4 else float((old or [0])[0])
    lum = float(old[2]) if old and len(old) >= 3 else 0.0
    return [round(hue, 1), round(100.0 * s, 1), lum]


def apply(edit, p: np.ndarray, base_mult: np.ndarray):
    """The edit with the nine settings moved by ``p`` (in solver units) from where they were."""
    from ..pipeline.edit import Edit
    e = Edit.from_dict(edit.to_dict())
    e.exposure_ev = round(float(edit.exposure_ev + p[0]), 3)
    e.tone.contrast = round(float(np.clip(edit.tone.contrast * math.exp(p[1]), 0.5, 3.0)), 4)
    m = np.asarray(base_mult, float) * np.array([math.exp(p[2]), 1.0, math.exp(p[3])])
    if np.any(np.abs(p[2:4]) > 1e-6):
        e.white_balance.mode = "multipliers"
        e.white_balance.multipliers = [round(float(x / m[1]), 5) for x in m]
    e.colour.saturation = round(float(np.clip(edit.colour.saturation + p[4], -100.0, 100.0)), 1)
    g = dict(edit.colour.grading or {})
    for key, (ia, ib) in (("shadows", (5, 6)), ("highlights", (7, 8))):
        if abs(p[ia]) > 1e-6 or abs(p[ib]) > 1e-6:
            g[key] = _wheel_from(_wheel_vec(g.get(key)) + np.array([p[ia], p[ib]]), g.get(key))
    e.colour.grading = g or None
    return e


def solve(render, edit, target: np.ndarray, base_mult, weight=None, iterations: int = 10, log=None):
    """Settings that give ``render(edit)`` the statistics ``target``. ``render``: Edit -> display-referred linear
    ProPhoto image (small). Returns (p, report) with p in solver units (see apply)."""
    t0 = time.perf_counter()
    target = np.asarray(target, float)

    def resid(p):
        return (stats(render(apply(edit, p, base_mult)), weight) - target) * WEIGHT

    p = np.zeros(len(PARAMS))
    r = resid(p)
    start = float(np.linalg.norm(r))
    J = np.zeros((len(STATS), len(PARAMS)))
    for k in range(len(PARAMS)):                       # one render per setting
        dp = np.zeros(len(PARAMS))
        dp[k] = STEP[k]
        J[:, k] = (resid(dp) - r) / STEP[k]
    best = (float(np.linalg.norm(r)), p.copy())
    renders = 1 + len(PARAMS)
    lam = 1e-3
    R = np.diag(REG / LIMIT)                            # a small pull toward no change: of two answers that
    for it in range(iterations):                       # match equally, the smaller change is kept
        A = J.T @ J + R.T @ R + lam * np.diag(np.maximum(np.diag(J.T @ J), 1e-6))
        step = -np.linalg.solve(A, J.T @ r + R.T @ (R @ p))
        scale = float(np.max(np.abs(step) / (0.5 * LIMIT)))  # no setting moves more than half its limit at once
        if scale > 1.0:
            step /= scale
        pn = np.clip(p + step, -LIMIT, LIMIT)
        rn = resid(pn)
        renders += 1
        dp, dr = pn - p, rn - r
        if float(dp @ dp) > 1e-12:                     # Broyden: the Jacobian learns from every step
            J += np.outer(dr - J @ dp, dp) / float(dp @ dp)
        if np.linalg.norm(rn) ** 2 + np.sum((R @ pn) ** 2) < np.linalg.norm(r) ** 2 + np.sum((R @ p) ** 2):
            p, r = pn, rn
            lam = max(lam / 3.0, 1e-4)
        else:
            lam *= 10.0
        if np.linalg.norm(r) < best[0]:
            best = (float(np.linalg.norm(r)), p.copy())
        if log:
            log.info("colour match step %d: residual %.4f (start %.4f)", it + 1, float(np.linalg.norm(r)), start)
        if np.linalg.norm(r) < 0.02 * start or np.linalg.norm(r) < 0.05:
            break
    p = best[1]
    report = {"start": start, "end": best[0], "renders": renders, "ms": (time.perf_counter() - t0) * 1e3,
              "change": {k: float(v) for k, v in zip(PARAMS, p)}}
    return p, report


def describe(p: np.ndarray) -> str:
    """The change in words, for the status line."""
    parts = []
    if abs(p[0]) >= 0.05:
        parts.append(f"exposure {p[0]:+.2f}")
    if abs(p[1]) >= 0.02:
        parts.append(f"contrast {'up' if p[1] > 0 else 'down'} {abs(math.expm1(p[1])) * 100:.0f}%")
    warm = p[2] - p[3]
    if abs(warm) >= 0.02:
        parts.append("white balance " + ("warmer" if warm > 0 else "cooler"))
    if abs(p[2] + p[3]) >= 0.03:
        parts.append("more magenta" if (p[2] + p[3]) > 0 else "more green")
    if abs(p[4]) >= 2:
        parts.append(f"saturation {p[4]:+.0f}")
    for name, ia in (("shadows", 5), ("highlights", 7)):
        s = float(np.hypot(p[ia], p[ia + 1]))
        if s >= 0.04:
            parts.append(f"{name} tinted ({100 * s:.0f})")
    return ", ".join(parts) if parts else "no change needed"
