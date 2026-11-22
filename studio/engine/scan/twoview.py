"""A pair of views: their relative pose from the matches and from the measured depth, compared
(BUILD_PLAN_DP3D E.3).

* **From the matches**: the essential matrix (8-point, RANSAC) and, in parallel, a homography; when the homography
  explains nearly as many matches (a flat texture, a wall) the pair is called planar and the essential matrix is not
  trusted for its pose (Torr's model-selection idea, GRIC, simplified to the inlier counts).
* **From the depth** (ours): every view's fused depth is affine in inverse distance with its own unknown scale and
  offset (Dual Pixel offsets are affine in 1/z; a stack's step positions are affine in the offsets). After
  normalising the depth per view (tau: 0 at the view's middle depth, 1 over its spread), the inverse distance is
  w = m (1 + kappa tau): one unknown shape number kappa per view (its depth spread as a share of its distance) and a
  scale m that the similarity takes up. For a small grid of kappa pairs, matched points become two 3-D clouds; a
  3-point similarity RANSAC (Umeyama 1991) finds the pose, scored by the reprojection into both views and the depth;
  the best is refined over kappa, the pose and the scale together. It works where the 8-point method is unstable
  (wide angles, few matches) and gives the pose its scale.

Both are tried; the pair keeps the depth pose when both agree ("both agree": cyan), else the one with more inliers
("from the matches" / "from the depth": amber).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from . import geom as G

KGRID = np.geomspace(0.04, 1.6, 9)
THR_PX = 2.5
DEPTH_TOL = 0.05            # relative inverse-distance agreement in the RANSAC
CONF_MIN = 0.3
MIN_INLIERS = 12
AGREE_ROT_DEG = 3.0
AGREE_DIR_DEG = 12.0


@dataclass
class PairResult:
    i: int
    j: int
    matches: np.ndarray                 # (M, 2) feature indices in views i and j
    inliers: np.ndarray                 # (M,) bool
    R: np.ndarray                       # view j relative to view i: X_j = R X_i + t
    t: np.ndarray
    scaled: bool                        # t carries view i's depth units (from the depth) or is a unit direction
    how: str                            # both | matches | depth
    model: str                          # essential | homography
    kappa_i: float | None = None
    kappa_j: float | None = None
    m_j: float | None = None            # view j's inverse-distance scale in view i's units
    agree_deg: float | None = None
    notes: list = field(default_factory=list)

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())

    @property
    def angle(self) -> float:
        return float(np.degrees(np.linalg.norm(G.log_rotation(self.R))))


def sample(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear samples of a map at pixel positions (x, y)."""
    return map_coordinates(np.asarray(img, np.float32), [xy[:, 1], xy[:, 0]], order=1, mode="nearest")


def normalised(view, xy):
    return np.stack([(xy[:, 0] - view.cx) / view.f, (xy[:, 1] - view.cy) / view.f], -1)


def _rays(n):
    return np.concatenate([n, np.ones((len(n), 1))], 1)


def depth_pose(ri, ti, ci, rj, tj, cj, thr, rng, kgrid=KGRID, samples: int = 160, score_max: int = 400):
    """Pose of view j relative to view i from the depth (see the module's text). ri, rj: (M, 3) rays (z = 1); ti, tj:
    normalised depth; ci, cj: confidence. Returns dict(R, t, m_j, kappa_i, kappa_j, inliers) or None."""
    M = len(ri)
    ok = (ci > CONF_MIN) & (cj > CONF_MIN) & np.isfinite(ti) & np.isfinite(tj)
    cand = np.nonzero(ok)[0]
    if cand.size < 6:
        return None
    sc_idx = cand if cand.size <= score_max else rng.choice(cand, score_max, replace=False)
    S = samples
    smp = np.array([rng.choice(cand, 3, replace=False) for _ in range(S)])
    C = len(kgrid)
    best = (-1, None)
    r_i, r_j = ri[sc_idx], rj[sc_idx]
    t_i, t_j = ti[sc_idx], tj[sc_idx]
    wj_all = 1.0 + kgrid[:, None] * tj[None]                       # (C, M)
    Qall = rj[None] / np.maximum(wj_all, 1e-3)[..., None]           # (C, M, 3)
    r_i32, r_j32 = r_i.astype(np.float32), r_j.astype(np.float32)
    Q = Qall[:, sc_idx].astype(np.float32)                           # (C, m, 3)
    wjs = (1.0 + kgrid[:, None] * t_j[None])[:, None, :] > 0.15       # (C, 1, m)
    for a, ki in enumerate(kgrid):
        wi = 1.0 + ki * ti
        P = ri / np.maximum(wi, 1e-3)[:, None]
        Ps = np.broadcast_to(P[smp][None], (C, S, 3, 3))
        Qs = Qall[:, smp]                                            # (C, S, 3, 3)
        s, R, t = G.umeyama(Ps, Qs)
        good = np.isfinite(s) & (s > 0)
        s32 = np.where(good, s, 1.0).astype(np.float32)[..., None]   # (C, S, 1)
        R32 = np.where(good[..., None, None], R, np.eye(3)).astype(np.float32)
        t32 = np.where(good[..., None], t, 0.0).astype(np.float32)
        # P' = s R Q + t, written out by component ((C, S, m) arrays: much faster than einsum here)
        Qx, Qy, Qz = (Q[:, None, :, k] for k in range(3))
        Pq = [s32 * (R32[..., k, 0, None] * Qx + R32[..., k, 1, None] * Qy + R32[..., k, 2, None] * Qz)
              + t32[..., k, None] for k in range(3)]
        Pp = P[sc_idx].astype(np.float32)
        D = [Pp[None, None, :, k] - t32[..., k, None] for k in range(3)]
        Qp = [(R32[..., 0, k, None] * D[0] + R32[..., 1, k, None] * D[1] + R32[..., 2, k, None] * D[2]) / s32
              for k in range(3)]
        with np.errstate(divide="ignore", invalid="ignore"):
            ei = np.hypot(Pq[0] / Pq[2] - r_i32[None, None, :, 0], Pq[1] / Pq[2] - r_i32[None, None, :, 1])
            ej = np.hypot(Qp[0] / Qp[2] - r_j32[None, None, :, 0], Qp[1] / Qp[2] - r_j32[None, None, :, 1])
            di = np.abs(Pq[2] * (1.0 + ki * t_i).astype(np.float32)[None, None] - 1.0)
        inl = (ei < thr) & (ej < thr) & (di < DEPTH_TOL) & (Pq[2] > 0) & (Qp[2] > 0) & \
            ((1.0 + ki * t_i) > 0.15)[None, None] & wjs
        cnt = np.where(good, inl.sum(-1), -1)
        c, k = np.unravel_index(int(np.argmax(cnt)), cnt.shape)
        if cnt[c, k] > best[0]:
            best = (int(cnt[c, k]), (ki, kgrid[c], s[c, k], R[c, k], t[c, k]))
    if best[1] is None or best[0] < 6:
        return None
    ki, kj, s, R, t = best[1]

    def model_errors(par, idx):
        ki_, kj_, w, tt, ls = par[0], par[1], par[2:5], par[5:8], par[8]
        Rm = G.rodrigues(w)
        s_ = np.exp(ls)
        wi_ = 1.0 + ki_ * ti[idx]
        wj_ = 1.0 + kj_ * tj[idx]
        P_ = ri[idx] / np.maximum(wi_, 1e-3)[:, None]
        Q_ = rj[idx] / np.maximum(wj_, 1e-3)[:, None]
        Pq_ = s_ * Q_ @ Rm.T + tt
        Qp_ = (P_ - tt) @ Rm / s_
        e_i = Pq_[:, :2] / np.where(np.abs(Pq_[:, 2:3]) > 1e-9, Pq_[:, 2:3], 1e-9) - ri[idx, :2]
        e_j = Qp_[:, :2] / np.where(np.abs(Qp_[:, 2:3]) > 1e-9, Qp_[:, 2:3], 1e-9) - rj[idx, :2]
        d_i = Pq_[:, 2] * wi_ - 1.0
        d_j = Qp_[:, 2] * wj_ - 1.0
        return e_i, e_j, d_i, d_j, Pq_, Qp_, wi_, wj_

    def inliers_of(par):
        e_i, e_j, d_i, d_j, Pq_, Qp_, wi_, wj_ = model_errors(par, cand)
        m_ = (np.linalg.norm(e_i, axis=1) < thr) & (np.linalg.norm(e_j, axis=1) < thr) & \
            (np.abs(d_i) < DEPTH_TOL) & (Pq_[:, 2] > 0) & (Qp_[:, 2] > 0) & (wi_ > 0.15) & (wj_ > 0.15)
        out = np.zeros(M, bool)
        out[cand[m_]] = True
        return out

    par = np.concatenate([[ki, kj], G.log_rotation(R), t, [np.log(s)]])
    inl = inliers_of(par)
    for _ in range(2):
        idx = np.nonzero(inl)[0]
        if idx.size < 6:
            break

        def res(p):
            e_i, e_j, d_i, d_j = model_errors(p, idx)[:4]
            return np.concatenate([(e_i / thr).ravel(), (e_j / thr).ravel(), d_i / 0.01, d_j / 0.01])
        r = least_squares(res, par, loss="huber", f_scale=1.0, max_nfev=200,
                          bounds=([0.0, 0.0] + [-np.inf] * 7, [5.0, 5.0] + [np.inf] * 7))
        par = r.x
        inl = inliers_of(par)
    Rq = G.rodrigues(par[2:5])
    tq = par[5:8]
    s = float(np.exp(par[8]))
    return {"R": Rq.T, "t": -Rq.T @ tq, "m_j": 1.0 / s, "kappa_i": float(par[0]), "kappa_j": float(par[1]),
            "inliers": inl}


def _dir_angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))


def pair(vi, vj, fi, fj, matches: np.ndarray, i: int, j: int, use_depth: bool = True, thr_px: float = THR_PX,
         rng=None) -> PairResult | None:
    """The relative pose of view j to view i from their matches (and depth), or None when it cannot be found."""
    from ..stack import features as SF
    rng = rng or np.random.default_rng(i * 1000 + j)
    M = len(matches)
    if M < MIN_INLIERS:
        return None
    xi = fi.xy[matches[:, 0]].astype(np.float64)
    xj = fj.xy[matches[:, 1]].astype(np.float64)
    ni, nj = normalised(vi, xi), normalised(vj, xj)
    thr = thr_px / (0.5 * (vi.f + vj.f))
    notes = []
    R_e, t_e, inl_e = G.essential_pose_ransac(ni, nj, thr, rng=rng)
    pe = (R_e, t_e, inl_e) if (R_e is not None and inl_e.sum() >= 8) else None
    _, inl_h = SF.ransac_homography(xi.astype(np.float32), xj.astype(np.float32), thr=thr_px * 1.2, rng=rng)
    n_e = int(pe[2].sum()) if pe else 0
    n_h = int(inl_h.sum()) if inl_h is not None else 0
    pd = None
    if use_depth:
        ti, ci = sample(vi.tau, xi), sample(vi.conf, xi)
        tj, cj = sample(vj.tau, xj), sample(vj.conf, xj)
        pd = depth_pose(_rays(ni), ti, ci, _rays(nj), tj, cj, thr, rng)
    n_d = int(pd["inliers"].sum()) if pd else 0
    # flat: the homography explains (nearly) every match the other models do
    planar = n_h >= 0.95 * max(n_e, n_d, 1) and n_h >= MIN_INLIERS
    agree = None
    if pe and pd:
        agree = max(G.angle_deg(pe[0], pd["R"]), 0.0)
        dir_ = _dir_angle(pe[1], pd["t"])
        both = agree < AGREE_ROT_DEG and dir_ < AGREE_DIR_DEG and not planar
    else:
        both = False
    if both:
        how, R, t, inl, scaled = "both", pd["R"], pd["t"], pd["inliers"] | pe[2], True
    elif planar:
        if pd and n_d >= MIN_INLIERS:
            how, R, t, inl, scaled = "depth", pd["R"], pd["t"], pd["inliers"], True
            notes.append("a flat surface (a homography explains the matches): the pose comes from the depth")
        else:
            return None
    elif pd and 1.4 * n_d >= n_e:
        # the depth's inliers pass a stricter test (the depth must agree too), and the essential matrix is the one
        # that goes wrong on a curved patch seen from two sides: depth wins unless the matches have clearly more
        how, R, t, inl, scaled = "depth", pd["R"], pd["t"], pd["inliers"], True
    elif pe:
        how, R, t, inl, scaled = "matches", pe[0], pe[1], pe[2], False
    else:
        return None
    if inl.sum() < MIN_INLIERS or inl.sum() < 0.12 * M:
        return None
    return PairResult(i, j, matches, inl, R, t, scaled, how, "homography" if planar else "essential",
                      pd["kappa_i"] if (pd and how != "matches") else None,
                      pd["kappa_j"] if (pd and how != "matches") else None,
                      pd["m_j"] if (pd and how != "matches") else None, agree, notes)
