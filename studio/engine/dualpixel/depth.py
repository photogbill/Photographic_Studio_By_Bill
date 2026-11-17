"""Depth from the two halves of a Dual Pixel photo, how sure it is, and a noise-versus-detail map
(BUILD_PLAN_TABS 3.3).

The two halves see the scene through opposite sides of the lens. What is in focus lands on the same photosites in
both; what is in front of or behind the focus plane lands slightly apart, left-right along the sensor's long axis,
by an amount that grows with the distance from focus (Wadhwa et al. 2018 use this for synthetic depth of field).
The offset is tiny (a fraction of a pixel to a few pixels), so it is measured with a 1-D search on band-passed,
shading-corrected luminance of each half, and the answer is only trusted where there is texture: on a plain wall
or in the sky it is honestly "do not know", never invented.

Detail versus noise: real detail is the same in both halves (after lining them up), noise is independent in each.
The local correlation of the two halves' fine detail is therefore near 1 on real detail and near 0 on noise (Bill's
eagle file: 0.96 on the feathers, 0.02 in the sky). No Qt here.
"""
from __future__ import annotations

import numpy as np

from ... import log as slog
from ..stack import align as A

log = slog.get("engine.dualpixel.depth")

IN_FOCUS_PX = 0.15                  # |offset| under this (full-size pixels), with confidence: in focus


def halves_luma(dp, gains) -> tuple[np.ndarray, np.ndarray]:
    """Shading-corrected, white-balanced 2 x 2 luminance of each half (half size). Clipped A+B photosites are
    replaced by the block's other values where possible (a clipped half has no useful offset)."""
    m = dp.wb / dp.wb.min()
    h, w = dp.shape[0] // 2 * 2, dp.shape[1] // 2 * 2
    la = np.zeros((h // 2, w // 2), np.float32)
    lb = np.zeros((h // 2, w // 2), np.float32)
    for i in (0, 1):
        for j in (0, 1):
            c = int(dp.pattern[i, j])
            ga = gains[0].plane(i, j, dp.shape)[:h // 2, :w // 2] if gains is not None else 1.0
            gb = gains[1].plane(i, j, dp.shape)[:h // 2, :w // 2] if gains is not None else 1.0
            la += dp.a[i:h:2, j:w:2] * ga * np.float32(m[c])
            lb += dp.b[i:h:2, j:w:2] * gb * np.float32(m[c])
    return la * np.float32(0.5), lb * np.float32(0.5)          # x2 for one half, / 4 for the bin


def _bandpass(x: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return (gaussian_filter(x, 1.0) - gaussian_filter(x, 4.0)).astype(np.float32)


_BPG: list = []


def _bp_gain() -> float:
    if not _BPG:
        n = np.random.default_rng(5).standard_normal((512, 512)).astype(np.float32)
        _BPG.append(float(_bandpass(n)[16:-16, 16:-16].std()))
    return _BPG[0]


def _shift_x(x: np.ndarray, s: float) -> np.ndarray:
    """x moved by s pixels along columns (content at c goes to c + s), linear interpolation, edges held."""
    if s == 0:
        return x
    n = x.shape[1]
    c = np.arange(n, dtype=np.float64) - s
    c0 = np.clip(np.floor(c).astype(int), 0, n - 1)
    c1 = np.clip(c0 + 1, 0, n - 1)
    f = (c - np.floor(c)).astype(np.float32)
    return x[:, c0] * (1 - f) + x[:, c1] * f


def disparity(la: np.ndarray, lb: np.ndarray, tile: int = 48, step: int = 16, max_shift: float = 4.0,
              noise: float | None = None):
    """(disp, conf) at half size: the left-right offset of half 2 against half 1 (content at x in half 1 lies at
    x + disp in half 2), in FULL-SIZE pixels, and how far to trust it (0..1). Normalised cross-correlation at whole
    shifts -max..max (half-size pixels) over each tile, a parabola through the peak for the fraction. conf =
    (peak - second peak, away from the first) x texture / (texture + noise)."""
    from scipy.ndimage import uniform_filter
    A_ = _bandpass(la)
    B_ = _bandpass(lb)
    n_s = int(np.ceil(max_shift))
    shifts = np.arange(-n_s, n_s + 1)
    size = int(tile)
    mA = uniform_filter(A_, size, mode="nearest")
    vA = np.maximum(uniform_filter(A_ * A_, size, mode="nearest") - mA * mA, 0.0)
    scores = []
    for s in shifts:
        Bs = _shift_x(B_, -float(s))                 # half 2 read at x + s
        mB = uniform_filter(Bs, size, mode="nearest")
        vB = np.maximum(uniform_filter(Bs * Bs, size, mode="nearest") - mB * mB, 0.0)
        cov = uniform_filter(A_ * Bs, size, mode="nearest") - mA * mB
        scores.append((cov / np.sqrt(vA * vB + 1e-20))[::step, ::step])
    S = np.stack(scores)                             # (n_shifts, gh, gw)
    k = np.argmax(S, axis=0)
    gh, gw = k.shape
    ii, jj = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    peak = S[k, ii, jj]
    km, kp = np.clip(k - 1, 0, len(shifts) - 1), np.clip(k + 1, 0, len(shifts) - 1)
    sm, sp = S[km, ii, jj], S[kp, ii, jj]
    den = sm - 2 * peak + sp
    frac = np.where((np.abs(den) > 1e-9) & (k > 0) & (k < len(shifts) - 1), 0.5 * (sm - sp) / np.where(
        np.abs(den) > 1e-9, den, 1.0), 0.0)
    d = (shifts[k] + np.clip(frac, -0.5, 0.5)).astype(np.float32)
    far = np.abs(np.arange(len(shifts))[:, None, None] - k[None]) >= 2
    second = np.where(far, S, -1.0).max(axis=0)
    # pixel noise of the noisier half, through the band-pass (measured once on white noise)
    sig = (max(A.noise_sigma(la), A.noise_sigma(lb)) if noise is None else float(noise)) * _bp_gain()
    tex = vA[::step, ::step]
    t = tex / (tex + 4.0 * sig * sig + 1e-20)
    conf = np.clip((peak - second), 0, 1) * t * (k > 0) * (k < len(shifts) - 1)
    conf = np.clip(conf * 2.0, 0, 1).astype(np.float32)
    # Lucas-Kanade along x for the exact fraction (a parabola through correlation scores leans toward whole
    # pixels): half 2 is read at x + d(x), and each tile's remaining offset is solved from the brightness slope.
    d = _refine_1d(A_, B_, d, step, size, max_shift)
    grid = A.Flow(d, d, conf, 0.0, 0.0, float(step))
    dd, _, cc = grid.fields(la.shape)
    return 2.0 * dd, cc


def _refine_1d(A_, B_, d_grid, step: int, size: int, max_shift: float, iters: int = 4):
    from scipy.ndimage import map_coordinates, uniform_filter
    h, w = A_.shape
    d = d_grid.astype(np.float64).copy()
    X = np.arange(w, dtype=np.float64)[None, :]
    Y = np.arange(h, dtype=np.float64)[:, None]
    for _ in range(iters):
        dense = A.Flow(d.astype(np.float32), d.astype(np.float32), np.ones_like(d, np.float32), 0.0, 0.0,
                       float(step)).fields(A_.shape)[0]
        Bw = map_coordinates(B_, [np.broadcast_to(Y, (h, w)), X + dense], order=1, mode="nearest").astype(np.float32)
        gx = np.gradient(Bw, axis=1)
        e = Bw - A_
        num = uniform_filter(gx * e, size, mode="nearest")[::step, ::step]
        den = uniform_filter(gx * gx, size, mode="nearest")[::step, ::step]
        delta = np.where(den > 1e-12, -num / np.maximum(den, 1e-12), 0.0)
        d = np.clip(d + np.clip(delta, -0.5, 0.5), -max_shift, max_shift)
        if float(np.abs(delta).max()) < 0.005:
            break
    return d.astype(np.float32)


def refine(disp: np.ndarray, conf: np.ndarray, guide_luma: np.ndarray, r: int = 24, eps: float = 1e-3) -> np.ndarray:
    """Confidence-weighted guided filter (the A+B luminance as guide): confident offsets spread along the picture's
    edges into places that could not be measured. Same size as the input."""
    g = np.sqrt(np.clip(guide_luma / max(float(np.percentile(guide_luma, 99.5)), 1e-6), 0, 1)).astype(np.float32)
    return A.weighted_guided(g, disp.astype(np.float32), (conf + 1e-4).astype(np.float32), int(r), eps)


def detail_map(la: np.ndarray, lb: np.ndarray, disp: np.ndarray, win: int = 9) -> np.ndarray:
    """Local correlation (0..1) of the two halves' fine detail after lining half 2 up by ``disp``: near 0 is noise,
    near 1 real detail."""
    from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter
    hA = la - gaussian_filter(la, 1.5)
    h, w = lb.shape
    Y, X = np.mgrid[0:h, 0:w].astype(np.float64)
    lbw = map_coordinates(lb, [Y, X + disp / 2.0], order=1, mode="nearest").astype(np.float32)
    hB = lbw - gaussian_filter(lbw, 1.5)
    mA, mB = uniform_filter(hA, win), uniform_filter(hB, win)
    cov = uniform_filter(hA * hB, win) - mA * mB
    vA = np.maximum(uniform_filter(hA * hA, win) - mA * mA, 0)
    vB = np.maximum(uniform_filter(hB * hB, win) - mB * mB, 0)
    return np.clip(cov / np.sqrt(vA * vB + 1e-20), 0, 1).astype(np.float32)


def focus_report(disp: np.ndarray, conf: np.ndarray, swap: bool = False) -> dict:
    """Plain words on focus: share of the frame in focus, and the confident subject's typical offset."""
    c = conf > 0.3
    share_conf = float(c.mean())
    if share_conf < 0.01:
        return {"in_focus_share": 0.0, "offset_px": 0.0, "confident_share": share_conf,
                "sentence": "Too little texture to measure focus from the two halves."}
    d = disp[c] * (-1 if swap else 1)
    inf = float((np.abs(d) < IN_FOCUS_PX).mean())
    # the subject: the confident place nearest to focus that covers at least 5% of what was measured
    med = float(np.median(d[np.abs(d) <= np.quantile(np.abs(d), 0.3)]))
    if abs(med) < IN_FOCUS_PX:
        s = "In focus: nothing to gain from re-aligning the halves."
    else:
        side = "behind" if med > 0 else "in front of"
        s = (f"Offset found (about {abs(med):.1f} px; the sharpest part sits {side} the focus plane): "
             f"micro-refocus could help.")
    return {"in_focus_share": inf * share_conf, "offset_px": med, "confident_share": share_conf, "sentence": s}


def air_steadiness(disp: np.ndarray, conf: np.ndarray) -> dict:
    """Experimental (plan 11.2): the patchy, non-smooth part of the offset map (heat shimmer bends the two views
    differently from place to place). One number (robust spread, px) and the map."""
    from scipy.ndimage import gaussian_filter
    w = (conf + 1e-4).astype(np.float32)
    smooth = gaussian_filter(disp * w, 24) / gaussian_filter(w, 24)
    patchy = (disp - smooth).astype(np.float32)
    c = conf > 0.3
    v = patchy[c]
    s = float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size > 100 else 0.0
    return {"spread_px": s, "map": patchy * (conf > 0.3)}
