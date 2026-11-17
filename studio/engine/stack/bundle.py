"""Bundle adjustment for the panorama (BUILD_PLAN_PANO_IR.md 2.4): every frame's rotation and one focal length.

Rotating-camera model: a point seen at pixel x_i in frame i is seen at ``x_j ~ K R_j R_i^T K^-1 x_i`` in frame j, with
``K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]``, one focal length for all frames (same lens and zoom), the principal point
at the centre, and lens distortion already removed when each frame was developed. R is world -> camera.

* **Focal length**: from the file when it records one (focal length and sensor width); adapted vintage lenses record
  nothing, so it is estimated from the pairwise homographies: for the right f, ``K^-1 H K`` is a rotation (Szeliski
  and Shum 1997 give closed forms; here a search over f for the most rotation-like set of pairs, which also works
  when the closed forms fail), shown amber with "type it if you know it".
* **Initial rotations**: the pairwise homographies chained along a maximum spanning tree of the match graph (weights:
  inliers), from the best-connected frame.
* **Solve**: scipy.optimize.least_squares (trf, Huber loss, the sparse Jacobian pattern given), residuals in pixels
  of the analysis copy, both directions of every match; parameters: an axis-angle per frame (the reference fixed) and
  log f. Reports the RMS residual of every pair.
* **Level** (Brown and Lowe's wave correction): the up vector most perpendicular to every frame's x axis, then one
  rotation of the whole panorama so the horizon is level and the middle of the panorama faces forward.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from ... import log as slog

log = slog.get("engine.stack.bundle")


# ------------------------------------------------------------------------------------------- rotations
def rodrigues(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        return np.eye(3) + K
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def log_rotation(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, float)
    c = (np.trace(R) - 1) / 2
    th = math.acos(min(1.0, max(-1.0, c)))
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if abs(th - math.pi) < 1e-6:
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        k = k * np.sign(np.r_[1, np.sign(A[0, 1] or 1), np.sign(A[0, 2] or 1)])
        return k / max(np.linalg.norm(k), 1e-12) * th
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = np.asarray(Ra) @ np.asarray(Rb).T
    return math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(R) - 1) / 2))))


# ------------------------------------------------------------------------------------------- focal length
def _centred(H: np.ndarray, cx: float, cy: float) -> np.ndarray:
    C = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])
    return np.linalg.inv(C) @ H @ C


def rotation_misfit(Hc: np.ndarray, f: float) -> float:
    """How far K^-1 H K is from a rotation (0 = exactly one), for a centred homography."""
    K = np.diag([f, f, 1.0])
    M = np.linalg.inv(K) @ Hc @ K
    d = np.linalg.det(M)
    if not np.isfinite(d) or abs(d) < 1e-12:
        return 1e9
    M = M / np.cbrt(d)
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.sum((s - 1.0) ** 2))


def focal_from_homographies(Hcs: list, weights: list | None = None, width: float = 1000.0) -> float | None:
    """The f (pixels) that makes the pairwise homographies most rotation-like: a coarse log search, then golden
    section. None when nothing is clearly better than the ends of the search (pure translation, a flat scene)."""
    if not Hcs:
        return None
    w = np.asarray(weights if weights is not None else [1.0] * len(Hcs), float)

    def cost(lf):
        f = math.exp(lf)
        e = np.array([rotation_misfit(H, f) for H in Hcs])
        return float(np.sum(w * np.minimum(e, 1.0)) / max(w.sum(), 1e-9))
    grid = np.linspace(math.log(0.25 * width), math.log(30 * width), 90)
    c = np.array([cost(g) for g in grid])
    k = int(np.argmin(c))
    if k == 0 or k == len(grid) - 1:
        return None
    a, b = grid[k - 1], grid[k + 1]
    gr = (math.sqrt(5) - 1) / 2
    x1, x2 = b - gr * (b - a), a + gr * (b - a)
    f1, f2 = cost(x1), cost(x2)
    for _ in range(40):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - gr * (b - a)
            f1 = cost(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + gr * (b - a)
            f2 = cost(x2)
    best = math.exp((a + b) / 2)
    if cost((a + b) / 2) > 0.5 * np.median(c):          # no clear minimum
        return None
    return best


# ------------------------------------------------------------------------------------------- spanning tree
def spanning_tree(n: int, edges: list) -> tuple:
    """Maximum spanning tree (Prim) of the match graph. edges: [(i, j, weight)]. Returns (root, [(parent, child)],
    connected set)."""
    adj = {i: [] for i in range(n)}
    deg = np.zeros(n)
    for i, j, wt in edges:
        adj[i].append((wt, j))
        adj[j].append((wt, i))
        deg[i] += wt
        deg[j] += wt
    if not edges:
        return 0, [], {0}
    root = int(np.argmax(deg))
    seen = {root}
    tree = []
    import heapq
    heap = [(-wt, root, j) for wt, j in adj[root]]
    heapq.heapify(heap)
    while heap:
        nw, p, c = heapq.heappop(heap)
        if c in seen:
            continue
        seen.add(c)
        tree.append((p, c))
        for wt, k in adj[c]:
            if k not in seen:
                heapq.heappush(heap, (-wt, c, k))
    return root, tree, seen


# ------------------------------------------------------------------------------------------- solve
def _K(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def initial_rotations(n: int, pairs: list, f: float, cx: float, cy: float) -> tuple:
    """Rotations from the spanning tree of homographies. pairs: dicts with "i", "j", "H" (j <- i, (x, y))."""
    edges = [(p["i"], p["j"], p["inliers"]) for p in pairs]
    root, tree, seen = spanning_tree(n, edges)
    K = _K(f, cx, cy)
    Ki = np.linalg.inv(K)
    Hs = {}
    for p in pairs:
        Hs[(p["i"], p["j"])] = p["H"]
        Hs[(p["j"], p["i"])] = np.linalg.inv(p["H"])
    R = [None] * n
    R[root] = np.eye(3)
    for parent, child in tree:
        H = Hs[(parent, child)]                       # child <- parent
        Rcp = nearest_rotation(Ki @ H @ K)            # R_child R_parent^T
        R[child] = Rcp @ R[parent]
    return root, R, seen


def undistort(pts_xy, cx, cy, k1: float, rn: float):
    """Observed pixel -> ideal (distortion-free) pixel: ideal = observed * (1 + k1 r^2), r = distance from the centre
    over the half-diagonal ``rn``. k1 > 0 undoes barrel, k1 < 0 pincushion. Shared by every frame (one lens)."""
    if not k1:
        return pts_xy
    u = pts_xy[:, 0] - cx
    v = pts_xy[:, 1] - cy
    g = 1.0 + k1 * (u * u + v * v) / (rn * rn)
    return np.stack([u * g + cx, v * g + cy], 1)


def distort(x, y, cx, cy, k1: float, rn: float, iters: int = 4):
    """Ideal pixel -> observed pixel (the inverse of undistort, by fixed-point iteration). Arrays of any shape."""
    if not k1:
        return x, y
    u, v = x - cx, y - cy
    uo, vo = u, v
    for _ in range(iters):
        g = 1.0 + k1 * (uo * uo + vo * vo) / (rn * rn)
        uo, vo = u / g, v / g
    return uo + cx, vo + cy


def project_between(Ri, Rj, f, cx, cy, pts_xy):
    """Points (x, y) of frame i -> frame j under the rotation model (ideal pixels)."""
    u = pts_xy[:, 0] - cx
    v = pts_xy[:, 1] - cy
    ray = np.stack([u, v, np.full_like(u, f)], 1)
    d = ray @ Ri                                      # R_i^T ray (rows)
    r = d @ Rj.T
    z = np.where(np.abs(r[:, 2]) < 1e-9, 1e-9, r[:, 2])
    return np.stack([f * r[:, 0] / z + cx, f * r[:, 1] / z + cy], 1), r[:, 2] > 0


def adjust(n: int, pairs: list, f0: float, cx: float, cy: float, R0: list, ref: int, solve_f: bool = True,
           loss_scale: float = 2.0, max_nfev: int = 200, solve_k1: bool = False, k10: float = 0.0) -> dict:
    """Refine rotations (and f, and one shared radial distortion k1). Returns {"R": [...], "f": f, "k1": k1,
    "rms": overall, "pair_rms": {(i, j): px}}."""
    idx = [k for k in range(n) if k != ref and R0[k] is not None]
    pos = {k: t for t, k in enumerate(idx)}
    rn = math.hypot(cx, cy)
    nparam = 3 * len(idx) + (1 if solve_f else 0) + (1 if solve_k1 else 0)
    ik1 = nparam - 1 if solve_k1 else None
    ifl = (nparam - 2 if solve_k1 else nparam - 1) if solve_f else None
    x0 = np.zeros(nparam)
    for k in idx:
        x0[3 * pos[k]: 3 * pos[k] + 3] = log_rotation(R0[k])
    if solve_f:
        x0[ifl] = math.log(f0)
    if solve_k1:
        x0[ik1] = k10
    use = [p for p in pairs if R0[p["i"]] is not None and R0[p["j"]] is not None]
    nres = sum(4 * len(p["pa"]) for p in use)
    if nres == 0:
        return {"R": R0, "f": f0, "k1": k10, "rms": 0.0, "pair_rms": {}}

    def unpack(x):
        R = list(R0)
        for k in idx:
            R[k] = rodrigues(x[3 * pos[k]: 3 * pos[k] + 3])
        R[ref] = np.eye(3) if R0[ref] is None else R0[ref]
        f = math.exp(x[ifl]) if solve_f else f0
        k1 = float(x[ik1]) if solve_k1 else k10
        return R, f, k1

    def residuals(x):
        R, f, k1 = unpack(x)
        out = []
        for p in use:
            i, j = p["i"], p["j"]
            a = undistort(p["pa"], cx, cy, k1, rn)
            b = undistort(p["pb"], cx, cy, k1, rn)
            q, _ = project_between(R[i], R[j], f, cx, cy, a)
            out.append((q - b).ravel())
            q2, _ = project_between(R[j], R[i], f, cx, cy, b)
            out.append((q2 - a).ravel())
        return np.concatenate(out)

    S = lil_matrix((nres, nparam), dtype=int)
    row = 0
    for p in use:
        m = 4 * len(p["pa"])
        for k in (p["i"], p["j"]):
            if k in pos:
                c = 3 * pos[k]
                S[row:row + m, c:c + 3] = 1
        if solve_f:
            S[row:row + m, ifl] = 1
        if solve_k1:
            S[row:row + m, ik1] = 1
        row += m
    # two stages: a forgiving loss first (a start that is tens of pixels out must still pull every match), then
    # Huber at a couple of pixels so the few wrong matches that passed RANSAC cannot bend the answer
    r = least_squares(residuals, x0, jac_sparsity=S, method="trf", loss="soft_l1", f_scale=10.0 * loss_scale,
                      x_scale="jac", max_nfev=max_nfev)
    r = least_squares(residuals, r.x, jac_sparsity=S, method="trf", loss="huber", f_scale=loss_scale,
                      x_scale="jac", max_nfev=max_nfev)
    R, f, k1 = unpack(r.x)
    res = residuals(r.x)
    pair_rms = {}
    row = 0
    for p in use:
        m = 4 * len(p["pa"])
        e = res[row:row + m]
        pair_rms[(p["i"], p["j"])] = float(np.sqrt(np.mean(e ** 2)) * math.sqrt(2))   # per point (x and y)
        row += m
    rms = float(np.sqrt(np.mean(res ** 2)) * math.sqrt(2))
    log.info("bundle adjustment: f %.1f px, k1 %.4f, rms %.2f px over %d pairs (%d evaluations)", f, k1, rms,
             len(use), r.nfev)
    return {"R": R, "f": f, "k1": k1, "rms": rms, "pair_rms": pair_rms}


def level(R: list) -> list:
    """Wave correction: the whole panorama turned so the horizon is level and its middle faces forward."""
    ok = [r for r in R if r is not None]
    if len(ok) < 2:
        return R
    X = np.array([r[0] for r in ok])                  # each camera's x axis in world coordinates
    Ydown = np.array([r[1] for r in ok])
    Z = np.array([r[2] for r in ok])
    C = X.T @ X
    w, v = np.linalg.eigh(C)
    up = v[:, 0]
    if np.sum(Ydown @ up) < 0:
        up = -up
    zf = Z.sum(axis=0)
    zf = zf - (zf @ up) * up
    if np.linalg.norm(zf) < 1e-9:
        return R
    zf /= np.linalg.norm(zf)
    xf = np.cross(up, zf)
    W = np.stack([xf, up, zf])                        # new world axes as rows: x right, y down, z forward
    return [None if r is None else r @ W.T for r in R]
