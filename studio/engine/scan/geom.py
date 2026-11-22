"""Geometry for the 3D Scan solver (BUILD_PLAN_DP3D E.3-E.5), NumPy only.

Conventions: a view's pose is world-to-camera, X_cam = R X_world + t (camera x right, y down, z forward); its centre
is C = -R^T t. Normalised image coordinates are x_n = (u - cx) / f, y_n = (v - cy) / f.

* Essential matrix: the normalised 8-point method (Hartley 1997) on calibrated coordinates, RANSAC with every
  hypothesis solved at once, Sampson distance; the pose by the cheirality test.
* Triangulation: the midpoint of N rays (least squares, any number of views at once).
* PnP: a 6-point direct linear transform, RANSAC, refined on the inliers.
* Similarity: Umeyama (PAMI 1991), every hypothesis at once.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


# ------------------------------------------------------------------------------------------ rotations
def rodrigues(w: np.ndarray) -> np.ndarray:
    """Rotation matrices for rotation vectors: (3,) -> (3, 3) or (N, 3) -> (N, 3, 3)."""
    w = np.asarray(w, np.float64)
    one = w.ndim == 1
    w = w.reshape(-1, 3)
    th = np.linalg.norm(w, axis=1)
    k = w / np.maximum(th, 1e-12)[:, None]
    K = np.zeros((len(w), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    R = np.eye(3)[None] + s * K + (1 - c) * (K @ K)
    small = th < 1e-8
    if small.any():
        Ks = np.zeros((small.sum(), 3, 3))
        ws = w[small]
        Ks[:, 0, 1], Ks[:, 0, 2], Ks[:, 1, 2] = -ws[:, 2], ws[:, 1], -ws[:, 0]
        Ks[:, 1, 0], Ks[:, 2, 0], Ks[:, 2, 1] = ws[:, 2], -ws[:, 1], ws[:, 0]
        R[small] = np.eye(3)[None] + Ks
    return R[0] if one else R


def log_rotation(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, np.float64)
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if np.pi - th < 1e-5:
        # near a half turn: the axis from the symmetric part
        A = (R + np.eye(3)) / 2
        ax = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(ax))
        ax = A[:, i] / max(ax[i], 1e-12)
        return ax / np.linalg.norm(ax) * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * th / (2 * np.sin(th))


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(log_rotation(Ra @ Rb.T))))


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def centre(R, t):
    return -np.asarray(R).T @ np.asarray(t)


# ------------------------------------------------------------------------------------------ essential matrix
def _cond(x: np.ndarray):
    m = x.mean(0)
    s = np.sqrt(2) / max(np.sqrt(((x - m) ** 2).sum(1)).mean(), 1e-12)
    T = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
    return T


def _rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rows of the linear system b^T E a = 0 for points a (view 1) and b (view 2), (..., n, 9)."""
    x1, y1 = a[..., 0], a[..., 1]
    x2, y2 = b[..., 0], b[..., 1]
    one = np.ones_like(x1)
    return np.stack([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, one], -1)


def _to_essential(E: np.ndarray) -> np.ndarray:
    U, S, Vt = np.linalg.svd(E)
    s = (S[..., 0] + S[..., 1]) / 2
    D = np.zeros(E.shape[:-2] + (3,))
    D[..., 0] = s
    D[..., 1] = s
    return U @ (D[..., :, None] * Vt)


def eight_point(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    T1, T2 = _cond(x1), _cond(x2)
    a = x1 @ T1[:2, :2].T + T1[:2, 2]
    b = x2 @ T2[:2, :2].T + T2[:2, 2]
    A = _rows(a, b)
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    E = T2.T @ E @ T1
    return _to_essential(E)


def sampson(E: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Sampson distances^2 (normalised units); E (3, 3) or (H, 3, 3) -> (n,) or (H, n)."""
    h1 = np.concatenate([x1, np.ones((len(x1), 1))], 1)
    h2 = np.concatenate([x2, np.ones((len(x2), 1))], 1)
    Ex1 = h1 @ np.swapaxes(E, -1, -2)            # (.., n, 3) = (E h1)^T
    Etx2 = h2 @ E                                 # (E^T h2)^T
    num = np.sum(h2 * Ex1, -1) ** 2
    den = Ex1[..., 0] ** 2 + Ex1[..., 1] ** 2 + Etx2[..., 0] ** 2 + Etx2[..., 1] ** 2
    return num / np.maximum(den, 1e-18)


def essential_ransac(x1: np.ndarray, x2: np.ndarray, thr: float, iters: int = 1024, rng=None, chunk: int = 256):
    """(E, inlier mask) from normalised coordinates; ``thr`` in normalised units (pixels / focal length)."""
    n = len(x1)
    if n < 8:
        return None, np.zeros(n, bool)
    rng = rng or np.random.default_rng(0)
    T1, T2 = _cond(x1), _cond(x2)
    a = x1 @ T1[:2, :2].T + T1[:2, 2]
    b = x2 @ T2[:2, :2].T + T2[:2, 2]
    best_n, best_E = -1, None
    t2 = thr * thr
    for s in range(0, iters, chunk):
        m = min(chunk, iters - s)
        idx = np.array([rng.choice(n, 8, replace=False) for _ in range(m)])
        A = _rows(a[idx], b[idx])                         # (m, 8, 9)
        _, _, Vt = np.linalg.svd(A)
        E = Vt[:, -1].reshape(m, 3, 3)
        E = T2.T[None] @ E @ T1[None]
        E = _to_essential(E)
        d = sampson(E, x1, x2)
        cnt = (d < t2).sum(1)
        k = int(np.argmax(cnt))
        if cnt[k] > best_n:
            best_n, best_E = int(cnt[k]), E[k]
    inl = sampson(best_E, x1, x2) < t2
    for _ in range(2):
        if inl.sum() < 8:
            break
        E = eight_point(x1[inl], x2[inl])
        new = sampson(E, x1, x2) < t2
        if new.sum() >= inl.sum():
            best_E, inl = E, new
        else:
            break
    return best_E, inl


def essential_pose_ransac(x1: np.ndarray, x2: np.ndarray, thr: float, iters: int = 1024, rng=None,
                          keep: int = 40, chunk: int = 256):
    """(R, t unit, inlier mask) of view 2 relative to view 1, or (None, None, mask). The hypotheses are first ranked
    by their Sampson inliers, then the best ``keep`` are judged by inliers that lie IN FRONT of both cameras for
    their best decomposition: on a nearly flat patch several essential matrices fit the matches about equally well,
    and only the true one puts the points in front (the cheirality test, used as the score)."""
    n = len(x1)
    if n < 8:
        return None, None, np.zeros(n, bool)
    rng = rng or np.random.default_rng(0)
    T1, T2 = _cond(x1), _cond(x2)
    a = x1 @ T1[:2, :2].T + T1[:2, 2]
    b = x2 @ T2[:2, :2].T + T2[:2, 2]
    t2 = thr * thr
    pool_E, pool_c = [], []
    for s in range(0, iters, chunk):
        m = min(chunk, iters - s)
        idx = np.array([rng.choice(n, 8, replace=False) for _ in range(m)])
        A = _rows(a[idx], b[idx])
        _, _, Vt = np.linalg.svd(A)
        E = Vt[:, -1].reshape(m, 3, 3)
        E = _to_essential(T2.T[None] @ E @ T1[None])
        cnt = (sampson(E, x1, x2) < t2).sum(1)
        pool_E.append(E)
        pool_c.append(cnt)
    E = np.concatenate(pool_E)
    cnt = np.concatenate(pool_c)
    # local optimisation (LO-RANSAC, Chum, Matas and Kittler 2003): larger samples drawn from the inliers of the
    # best hypotheses. On a curved patch seen from two sides, 8-point samples keep landing on a wrong solution with
    # nearly as many inliers; 20-point samples from the inliers find the true one
    lo_E = []
    for k in np.argsort(-cnt)[:5]:
        inl = np.nonzero(sampson(E[k], x1, x2) < t2)[0]
        if inl.size < 12:
            continue
        kk = min(20, inl.size)
        idx = np.array([rng.choice(inl, kk, replace=False) for _ in range(64)])
        A = _rows(a[idx], b[idx])
        _, _, Vt = np.linalg.svd(A)
        lo_E.append(_to_essential(T2.T[None] @ Vt[:, -1].reshape(-1, 3, 3) @ T1[None]))
    if lo_E:
        lo_E = np.concatenate(lo_E)
        E = np.concatenate([E, lo_E])
        cnt = np.concatenate([cnt, (sampson(lo_E, x1, x2) < t2).sum(1)])
    top = np.argsort(-cnt)[:keep]
    cands = []
    for k in top:
        inl = sampson(E[k], x1, x2) < t2
        if inl.sum() < 8:
            continue
        R, t, front, _ = pose_from_essential(E[k], x1[inl], x2[inl])
        m_ = inl.copy()
        m_[np.nonzero(inl)[0][~front]] = False
        cands.append((int(front.sum()), R, t, m_))
    if not cands:
        return None, None, np.zeros(n, bool)
    cands.sort(key=lambda c: -c[0])
    # the leading distinct candidates refined, judged by their truncated geometric cost (MSAC): the true one fits
    # its points to the noise; a near-tie on counts is a wrong solution that fits them to a pixel or two
    picked = []
    for c in cands:
        if c[0] < 0.8 * cands[0][0]:
            break
        if all(angle_deg(c[1], p[1]) > 2.0 for p in picked):
            picked.append(c)
        if len(picked) >= 4:
            break

    def msac(R, t):
        d = sampson(essential_of(R, t), x1, x2)
        X = triangulate_pair(np.eye(3), np.zeros(3), R, t, x1, x2)
        fr = (X[:, 2] > 0) & ((X @ R.T + t)[:, 2] > 0)
        return float(np.sum(np.where(fr, np.minimum(d, t2), t2))), (d < t2) & fr
    best = None
    for _, R, t, inl in picked:
        for _ in range(2):
            if inl.sum() < 8:
                break
            R, t = refine_essential(R, t, x1[inl], x2[inl], thr)
            cost, inl = msac(R, t)
        cost, inl = msac(R, t)
        if best is None or cost < best[0]:
            best = (cost, R, t, inl)
    return best[1], best[2], best[3]


def refine_essential(R, t, x1, x2, thr: float, iters: int = 60):
    """(R, t unit) refined by least squares on the Sampson distances (Huber at ``thr``): the 8-point solution is
    algebraic and biased; this is the geometric one. The direction of t is kept as two angles."""
    t = t / np.linalg.norm(t)
    # a basis round t: t(a, b) = cos(a) cos(b) t + sin(a) e1 + cos(a) sin(b) e2
    e1 = np.cross(t, [1.0, 0, 0] if abs(t[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(t, e1)

    def tdir(a, b):
        return np.cos(a) * np.cos(b) * t + np.sin(a) * e1 + np.cos(a) * np.sin(b) * e2

    def skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def res(p):
        Rm = rodrigues(p[:3]) @ R
        E = skew(tdir(p[3], p[4])) @ Rm
        return np.sqrt(sampson(E, x1, x2)) / thr
    r = least_squares(res, np.zeros(5), loss="huber", f_scale=1.0, max_nfev=iters * 6)
    return rodrigues(r.x[:3]) @ R, tdir(r.x[3], r.x[4])


def essential_of(R, t):
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return tx @ R


def triangulate_pair(R1, t1, R2, t2, x1, x2) -> np.ndarray:
    """(n, 3) world points by the midpoint of the two rays."""
    C1, C2 = centre(R1, t1), centre(R2, t2)
    d1 = np.concatenate([x1, np.ones((len(x1), 1))], 1) @ R1      # R^T x
    d2 = np.concatenate([x2, np.ones((len(x2), 1))], 1) @ R2
    return midpoint(np.stack([np.broadcast_to(C1, d1.shape), np.broadcast_to(C2, d2.shape)], 1),
                    np.stack([d1, d2], 1))


def midpoint(C: np.ndarray, d: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """The point nearest to rays (C[i, k], d[i, k]) in least squares: C, d (n, k, 3) (NaN rays ignored)."""
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    ok = np.isfinite(d).all(-1) & np.isfinite(C).all(-1)
    if w is None:
        w = np.ones(d.shape[:2])
    w = np.where(ok, w, 0.0)
    d = np.where(ok[..., None], d, 0.0)
    C = np.where(ok[..., None], C, 0.0)
    P = np.eye(3)[None, None] - d[..., :, None] * d[..., None, :]          # (n, k, 3, 3)
    A = np.sum(w[..., None, None] * P, 1)
    b = np.sum(w[..., None] * (P @ C[..., None])[..., 0], 1)
    det = np.linalg.det(A)
    out = np.full((len(A), 3), np.nan)
    good = np.abs(det) > 1e-12
    out[good] = np.linalg.solve(A[good], b[good][..., None])[..., 0]
    return out


def pose_from_essential(E, x1, x2):
    """(R, t (unit), in front mask, points) for the second view relative to the first, by cheirality."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    best = None
    for R in (U @ W @ Vt, U @ W.T @ Vt):
        for t in (U[:, 2], -U[:, 2]):
            X = triangulate_pair(np.eye(3), np.zeros(3), R, t, x1, x2)
            z1 = X[:, 2]
            z2 = (X @ R.T + t)[:, 2]
            front = (z1 > 0) & (z2 > 0) & np.isfinite(z1)
            if best is None or front.sum() > best[2].sum():
                best = (R, t / np.linalg.norm(t), front, X)
    return best


def ray_angle_deg(C1, C2, X) -> np.ndarray:
    a = X - C1
    b = X - C2
    c = np.sum(a * b, -1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def project(R, t, X):
    Xc = X @ np.asarray(R).T + t
    return Xc[:, :2] / Xc[:, 2:3], Xc[:, 2]


# ------------------------------------------------------------------------------------------ PnP
def _pnp_dlt_batch(X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(H, 3, 4) camera matrices from (H, k, 3) points and (H, k, 2) normalised coordinates, k >= 6."""
    H, k, _ = X.shape
    Xh = np.concatenate([X, np.ones((H, k, 1))], -1)
    z = np.zeros((H, k, 4))
    r1 = np.concatenate([Xh, z, -x[..., :1] * Xh], -1)
    r2 = np.concatenate([z, Xh, -x[..., 1:2] * Xh], -1)
    A = np.concatenate([r1, r2], 1)
    _, _, Vt = np.linalg.svd(A)
    return Vt[:, -1].reshape(H, 3, 4)


def refine_pose(R, t, X, x, iters: int = 30):
    def res(p):
        Rm = rodrigues(p[:3])
        xc = X @ Rm.T + p[3:]
        return (xc[:, :2] / np.maximum(xc[:, 2:3], 1e-9) - x).ravel()
    p0 = np.concatenate([log_rotation(R), t])
    r = least_squares(res, p0, loss="huber", f_scale=2e-3, max_nfev=iters * 7)
    return rodrigues(r.x[:3]), r.x[3:]


def pnp_ransac(X: np.ndarray, x: np.ndarray, thr: float, iters: int = 600, rng=None):
    """(R, t, inlier mask) of a camera from world points X (n, 3) and normalised coordinates x (n, 2)."""
    n = len(X)
    if n < 6:
        return None, None, np.zeros(n, bool)
    rng = rng or np.random.default_rng(1)
    idx = np.array([rng.choice(n, 6, replace=False) for _ in range(iters)])
    m = X.mean(0)
    sc = 1.0 / max(np.sqrt(((X - m) ** 2).sum(1)).mean(), 1e-12)
    Xn = (X - m) * sc
    P = _pnp_dlt_batch(Xn[idx], x[idx])
    Xh = np.concatenate([Xn, np.ones((n, 1))], 1)
    proj = np.einsum("hij,nj->hni", P, Xh)
    zz = proj[..., 2]
    err = np.linalg.norm(proj[..., :2] / np.where(np.abs(zz) > 1e-12, zz, 1e-12)[..., None] - x[None], axis=-1)
    cnt = ((err < thr) & (np.sign(zz) == np.sign(np.median(zz, axis=1, keepdims=True)))).sum(1)
    k = int(np.argmax(cnt))
    inl = err[k] < thr
    if inl.sum() < 6:
        return None, None, inl
    R, tn = _rt_from_P(P[k])
    # back to the original world scale: Xn = (X - m) sc  ->  R Xn + tn = sc (R X - R m) + tn
    t = (tn - sc * (R @ m)) / sc
    R, t = refine_pose(R, t, X[inl], x[inl])
    err = np.linalg.norm(project(R, t, X)[0] - x, axis=1)
    z = (X @ R.T + t)[:, 2]
    inl = (err < thr) & (z > 0)
    if inl.sum() >= 6:
        R, t = refine_pose(R, t, X[inl], x[inl])
        err = np.linalg.norm(project(R, t, X)[0] - x, axis=1)
        z = (X @ R.T + t)[:, 2]
        inl = (err < thr) & (z > 0)
    return R, t, inl


def _rt_from_P(P):
    """(R, t) from a DLT camera matrix P = lambda [R | t] (lambda of either sign: a negative one shows as a
    reflection, det < 0, and is undone)."""
    U, S, Vt = np.linalg.svd(P[:, :3])
    R = U @ Vt
    t = P[:, 3] / S.mean()
    if np.linalg.det(R) < 0:
        R, t = -R, -t
    return R, t


# ------------------------------------------------------------------------------------------ similarity
def umeyama(P: np.ndarray, Q: np.ndarray, w: np.ndarray | None = None):
    """(s, R, t) with P ~ s R Q + t for (..., n, 3) arrays (every leading index solved at once)."""
    if w is None:
        w = np.ones(P.shape[:-1])
    w = w / np.maximum(w.sum(-1, keepdims=True), 1e-12)
    mp = np.sum(w[..., None] * P, -2)
    mq = np.sum(w[..., None] * Q, -2)
    Pc = P - mp[..., None, :]
    Qc = Q - mq[..., None, :]
    S = np.einsum("...n,...ni,...nj->...ij", w, Pc, Qc)
    U, D, Vt = np.linalg.svd(S)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    Dm = np.ones(D.shape)
    Dm[..., 2] = d
    R = U @ (Dm[..., :, None] * Vt)
    varq = np.sum(w * np.sum(Qc * Qc, -1), -1)
    s = np.sum(D * Dm, -1) / np.maximum(varq, 1e-18)
    t = mp - s[..., None] * np.einsum("...ij,...j->...i", R, mq)
    return s, R, t
