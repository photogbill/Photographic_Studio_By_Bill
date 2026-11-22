"""Dense depth for every placed view, measured two ways (BUILD_PLAN_DP3D E.6).

1. **Dual Pixel**: the view's fused depth turned into distance by its solved shape, z = 1 / (m (1 + kappa tau)),
   where it measured something (confidence) and inside the subject.
2. **Multi-view stereo**: a plane sweep against the view's two or three best neighbours (the most shared points at a
   useful angle): for each pixel, inverse-distance hypotheses round its Dual Pixel value (+-10 %: the depth cuts the
   sweep tenfold) or over the view's whole range where Dual Pixel measured nothing; each neighbour's picture warped by
   the hypothesis and compared by normalised cross-correlation in 7 x 7 windows; the best hypothesis refined by a
   parabola.

Fused per pixel by their noise (the Depth Stack's pattern), with an **agreement map**: both and agreeing (cyan), one
of them (amber), disagreeing (red: kept out of the mesh), nothing (no depth). Works at a reduced size (long edge
``WORK``) for speed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter, zoom

from . import geom as G

WORK = 640
N_HYP = 21
SPAN = 0.10
NCC_MIN = 0.55
WIN = 7
LABEL_NONE, LABEL_ONE, LABEL_BOTH, LABEL_RED = 0, 1, 2, 3


@dataclass
class DenseView:
    z: np.ndarray            # distance along the optical axis, world units (NaN where none), working size
    label: np.ndarray        # uint8 LABEL_*
    conf: np.ndarray         # 0..1
    scale: float             # working size / the view's matching size
    f: float                 # focal length at the working size
    cx: float
    cy: float

    def shares(self) -> dict:
        n = float(np.isfinite(self.z).size) or 1.0
        return {k: round(float((self.label == v).sum()) / n, 4) for k, v in
                (("measured_both", LABEL_BOTH), ("measured_one", LABEL_ONE), ("disagree", LABEL_RED))}


def neighbours(sol, i: int, k: int = 3) -> list:
    """The placed views sharing the most points with view i at an angle of 4-60 degrees."""
    if not sol.placed[i]:
        return []
    pts_i = set(sol.obs_point[sol.obs_view == i].tolist())
    Ci = sol.centre(i)
    zi = sol.R[i][2]
    out = []
    for j in range(len(sol.names)):
        if j == i or not sol.placed[j]:
            continue
        ang = np.degrees(np.arccos(np.clip(zi @ sol.R[j][2], -1, 1)))
        if not 4 <= ang <= 60:
            continue
        shared = len(pts_i & set(sol.obs_point[sol.obs_view == j].tolist()))
        if shared >= 8:
            out.append((shared, j, float(np.linalg.norm(sol.centre(j) - Ci))))
    out.sort(reverse=True)
    return [j for _, j, _ in out[:k]]


def _down(a, s, order=1):
    if s == 1.0:
        return np.asarray(a, np.float32)
    return zoom(np.asarray(a, np.float32), s, order=order, grid_mode=True, mode="nearest")


def _ncc(a, b, win=WIN):
    ma, mb = uniform_filter(a, win), uniform_filter(b, win)
    va = uniform_filter(a * a, win) - ma * ma
    vb = uniform_filter(b * b, win) - mb * mb
    cov = uniform_filter(a * b, win) - ma * mb
    return cov / np.sqrt(np.maximum(va * vb, 1e-10)), va


def dual_pixel_depth(sol, view, i: int):
    """(z, valid) at the view's matching size: distance from the view's own measured depth."""
    w = sol.m[i] * (1.0 + sol.kappa[i] * view.tau)
    valid = (view.conf > 0.3) & np.isfinite(w) & (w > 1e-6)
    if view.mask is not None:
        valid &= view.mask
    z = np.where(valid, 1.0 / np.where(valid, w, 1.0), np.nan)
    return z.astype(np.float32), valid


def view_depth(sol, views, i: int, work: int = WORK) -> DenseView:
    v = views[i]
    h, w = v.shape
    s = min(1.0, work / float(max(h, w)))
    f = sol.f[i] * s
    cx, cy = (v.cx + 0.5) * s - 0.5, (v.cy + 0.5) * s - 0.5
    img = _down(v.image, s)
    hh, ww = img.shape
    z_dp_full, valid_full = dual_pixel_depth(sol, v, i)
    z_dp = _down(np.nan_to_num(z_dp_full, nan=0.0), s)
    valid = _down(valid_full.astype(np.float32), s, 0) > 0.5
    z_dp = np.where(valid, z_dp, np.nan)
    mask = _down(v.mask.astype(np.float32), s, 0) > 0.5 if v.mask is not None else np.ones((hh, ww), bool)
    # hypotheses per pixel: round the Dual Pixel value, else over the view's range
    w_dp = np.where(valid, 1.0 / np.where(valid, z_dp, 1.0), np.nan)
    if np.isfinite(w_dp).any():
        lo, hi = np.nanpercentile(w_dp, [1, 99])
    else:
        lo, hi = 0.5, 2.0
    centre = np.where(valid, w_dp, 0.5 * (lo + hi))
    half = np.where(valid, SPAN * w_dp, 0.5 * (hi - lo) * 1.3 + 1e-6)
    deltas = np.linspace(-1.0, 1.0, N_HYP)
    nb = neighbours(sol, i)
    ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float64)
    xn = (xs - cx) / f
    yn = (ys - cy) / f
    R0, t0 = sol.R[i], sol.t[i]
    costs = np.full((N_HYP, hh, ww), np.nan, np.float32)
    texture = None
    if nb:
        nimgs = []
        for j in nb:
            vj = views[j]
            sj = min(1.0, work / float(max(vj.shape)))
            nimgs.append((j, _down(vj.image, sj), sj))
        for k, d in enumerate(deltas):
            wk = centre + d * half
            zk = 1.0 / np.maximum(wk, 1e-9)
            Xc = np.stack([xn * zk, yn * zk, zk], -1)
            X = (Xc - t0) @ R0                              # world = R^T (Xc - t)
            acc = []
            for j, im_j, sj in nimgs:
                Xj = X @ sol.R[j].T + sol.t[j]
                zz = np.where(Xj[..., 2] > 1e-9, Xj[..., 2], 1e-9)
                x, y = Xj[..., 0] / zz, Xj[..., 1] / zz
                dd = 1 + sol.k1 * (x * x + y * y)
                fj = sol.f[j] * sj
                u = fj * x * dd + (views[j].cx + 0.5) * sj - 0.5
                vv = fj * y * dd + (views[j].cy + 0.5) * sj - 0.5
                inside = (Xj[..., 2] > 0) & (u >= 0) & (vv >= 0) & (u <= im_j.shape[1] - 1) & (vv <= im_j.shape[0] - 1)
                warped = map_coordinates(im_j, [vv, u], order=1, mode="nearest").astype(np.float32)
                c, va = _ncc(img, warped)
                if texture is None:
                    texture = va
                acc.append(np.where(inside, 1.0 - c, np.nan))
            a = np.stack(acc)
            # the two best neighbours count (a third may be blocked there)
            a = np.sort(np.where(np.isfinite(a), a, 9.0), 0)[:2]
            costs[k] = np.where(a.min(0) < 9.0, np.mean(np.where(a < 9.0, a, np.nan), 0), np.nan)
    z_mvs = np.full((hh, ww), np.nan, np.float32)
    ncc_best = np.zeros((hh, ww), np.float32)
    if nb:
        cc = np.where(np.isfinite(costs), costs, 9.0)
        kb = np.argmin(cc, 0)
        cb = np.take_along_axis(cc, kb[None], 0)[0]
        interior = (kb > 0) & (kb < N_HYP - 1)
        km = np.clip(kb - 1, 0, N_HYP - 1)
        kp = np.clip(kb + 1, 0, N_HYP - 1)
        cm = np.take_along_axis(cc, km[None], 0)[0]
        cp = np.take_along_axis(cc, kp[None], 0)[0]
        den = cm - 2 * cb + cp
        off = np.where(interior & (den > 1e-6), 0.5 * (cm - cp) / np.where(den > 1e-6, den, 1), 0.0)
        dbest = deltas[kb] + off * (deltas[1] - deltas[0])
        wb = centre + dbest * half
        ncc_best = 1.0 - cb
        tex_ok = texture is not None and True
        good = interior & (ncc_best > NCC_MIN) & (cb < 9.0) & mask & (wb > 0)
        if tex_ok:
            good &= texture > np.percentile(texture, 20)
        z_mvs = np.where(good, 1.0 / np.where(good, wb, 1.0), np.nan).astype(np.float32)
    # fuse: inverse distance, weights by the assumed noise of each (Dual Pixel: its solved depth residual)
    q = sol.quality[i] or {}
    sd_dp = max(float(q.get("depth_rms") or 0.01), 0.003)
    sd_mvs = max(SPAN * 2.0 / (N_HYP - 1) / 2.0, 0.003)
    have_dp = np.isfinite(z_dp)
    have_mvs = np.isfinite(z_mvs)
    both = have_dp & have_mvs
    wd = np.where(have_dp, 1.0 / z_dp, 0.0)
    wm = np.where(have_mvs, 1.0 / z_mvs, 0.0)
    rel = np.where(both, np.abs(wm - wd) / np.maximum(wd, 1e-9), 0.0)
    red = both & (rel > 3.0 * np.hypot(sd_dp, sd_mvs))
    a, b = 1.0 / sd_dp ** 2, 1.0 / sd_mvs ** 2
    wf = np.where(both, (a * wd + b * wm) / (a + b), np.where(have_dp, wd, wm))
    lab = np.full((hh, ww), LABEL_NONE, np.uint8)
    lab[have_dp ^ have_mvs] = LABEL_ONE
    lab[both] = LABEL_BOTH
    lab[red] = LABEL_RED
    keep = (have_dp | have_mvs) & ~red & mask
    z = np.where(keep, 1.0 / np.maximum(wf, 1e-9), np.nan).astype(np.float32)
    conf = np.where(lab == LABEL_BOTH, 1.0, np.where(lab == LABEL_ONE, 0.6, 0.0)).astype(np.float32)
    conf = np.where(keep, conf, 0.0)
    return DenseView(z=z, label=lab, conf=conf, scale=s, f=f, cx=cx, cy=cy)
