"""The surface of a signed field, and mesh clean-up (BUILD_PLAN_DP3D E.8), NumPy / SciPy only.

* **Surface nets** (Gibson 1998, "Constrained elastic surface nets"; the naive form): one vertex per grid cell the
  surface passes through, at the average of the points where the field crosses zero on the cell's twelve edges; one
  quad per grid edge the surface crosses, joining the four cells round it, split into two triangles. Used in place
  of marching cubes' 256-case table: the same surface, watertight wherever the field is closed, fewer slivers, and
  short enough to check by eye.
* **Largest piece**, **Taubin smoothing** (lambda 0.5, mu -0.53: smooths without shrinking, Taubin 1995),
  **vertex normals**.
Field convention: negative inside, positive outside, zero on the surface.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
_EDGES = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]])


def surface_nets(F: np.ndarray, origin=(0.0, 0.0, 0.0), voxel: float = 1.0):
    """(vertices (n, 3) world, triangles (m, 3), cell index (n, 3) of each vertex) of the zero level of F
    (shape (nx, ny, nz), sampled at origin + voxel * index). Pad F with a positive border to close the surface."""
    F = np.asarray(F, np.float32)
    nx, ny, nz = F.shape
    c = [F[a:nx - 1 + a, b:ny - 1 + b, d:nz - 1 + d] for a, b, d in _CORNERS]
    inside = np.stack([x < 0 for x in c])                       # (8, cx, cy, cz)
    n_in = inside.sum(0)
    active = (n_in > 0) & (n_in < 8)
    idx = np.argwhere(active)                                   # (n, 3) cell indices
    if not len(idx):
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3), int)
    vals = np.stack([x[active] for x in c], 1)                  # (n, 8)
    acc = np.zeros((len(idx), 3))
    cnt = np.zeros(len(idx))
    for a, b in _EDGES:
        va, vb = vals[:, a], vals[:, b]
        cross = (va < 0) != (vb < 0)
        if not cross.any():
            continue
        t = np.where(cross, va / np.where(va - vb == 0, 1e-12, va - vb), 0.0)
        p = _CORNERS[a][None] + t[:, None] * (_CORNERS[b] - _CORNERS[a])[None]
        acc += np.where(cross[:, None], p, 0.0)
        cnt += cross
    pos = idx + acc / np.maximum(cnt, 1)[:, None]
    vid = -np.ones(active.shape, np.int64)
    vid[active] = np.arange(len(idx))
    quads = []
    # edges along axis a with the two other axes (b, c) cyclic: cells round the edge counter-clockwise about +a
    for a in range(3):
        b, cc = (a + 1) % 3, (a + 2) % 3
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a] = slice(0, F.shape[a] - 1)
        sl1[a] = slice(1, F.shape[a])
        for ax in (b, cc):
            sl0[ax] = slice(1, F.shape[ax] - 1)
            sl1[ax] = slice(1, F.shape[ax] - 1)
        s0 = F[tuple(sl0)] < 0
        s1 = F[tuple(sl1)] < 0
        ch = np.argwhere(s0 != s1)
        if not len(ch):
            continue
        inside_low = s0[tuple(ch.T)]
        base = ch.copy()
        base[:, b] += 1
        base[:, cc] += 1                                         # back to grid indices (the slices started at 1)
        cells = []
        for db, dc in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            q = base.copy()
            q[:, b] += db
            q[:, cc] += dc
            cells.append(vid[q[:, 0], q[:, 1], q[:, 2]])
        cells = np.stack(cells, 1)
        ok = (cells >= 0).all(1)
        cells, inside_low = cells[ok], inside_low[ok]
        cells[~inside_low] = cells[~inside_low][:, ::-1]        # outward = from inside to outside
        quads.append(cells)
    if not quads:
        return origin + voxel * pos, np.zeros((0, 3), int), idx
    q = np.concatenate(quads)
    P = pos
    d02 = np.linalg.norm(P[q[:, 0]] - P[q[:, 2]], axis=1)
    d13 = np.linalg.norm(P[q[:, 1]] - P[q[:, 3]], axis=1)
    short02 = d02 <= d13
    tri = np.concatenate([np.where(short02[:, None], q[:, [0, 1, 2]], q[:, [0, 1, 3]]),
                          np.where(short02[:, None], q[:, [0, 2, 3]], q[:, [1, 2, 3]])])
    return np.asarray(origin, float) + voxel * pos, tri.astype(np.int64), idx


def adjacency(n: int, tri: np.ndarray):
    i = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2], tri[:, 1], tri[:, 2], tri[:, 0]])
    j = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0], tri[:, 0], tri[:, 1], tri[:, 2]])
    A = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


def largest_piece(V: np.ndarray, T: np.ndarray, min_share: float = 0.0):
    """Keep the largest connected piece (or every piece with at least ``min_share`` of the triangles)."""
    if not len(T):
        return V, T, np.arange(len(V))
    nc, lab = connected_components(adjacency(len(V), T), directed=False)
    tl = lab[T[:, 0]]
    sizes = np.bincount(tl, minlength=nc)
    keep_l = np.nonzero(sizes >= max(sizes.max() if min_share <= 0 else min_share * len(T), 1))[0]
    keepT = np.isin(tl, keep_l)
    T = T[keepT]
    used = np.unique(T)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[T], used


def taubin(V: np.ndarray, T: np.ndarray, rounds: int = 6, lam: float = 0.5, mu: float = -0.53,
           fixed: np.ndarray | None = None) -> np.ndarray:
    A = adjacency(len(V), T)
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    X = np.asarray(V, np.float64).copy()
    for _ in range(rounds):
        for f in (lam, mu):
            L = (A @ X) / deg[:, None] - X
            step = f * L
            if fixed is not None:
                step[fixed] = 0
            X = X + step
    return X


def vertex_normals(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V, dtype=np.float64)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def remove_unused(V, T, *per_vertex):
    used = np.unique(T)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    return (V[used], remap[T]) + tuple(a[used] for a in per_vertex)
