"""Game-sized meshes and their normal maps (BUILD_PLAN_DP3D E.8).

* **Simplification by quadric clustering** (Lindstrom, "Out-of-core simplification of large polygonal models", 2000):
  vertices are gathered on a grid; each cell's vertex goes where the sum of its vertices' Garland-Heckbert quadrics
  (Garland and Heckbert 1997: squared distance to the planes of the faces round them) is least; faces whose corners
  fall in fewer than three cells disappear. The grid is searched until the triangle count meets the target. Used in
  place of edge-by-edge collapse: the same error measure, all in array operations (no per-edge Python loop), and
  crack-free across texture seams: a cell's position is shared by every chart, its texture coordinates are averaged
  per chart.
* **Normal map baking**: the full mesh and the game mesh share the atlas layout, so each texel's full-detail normal is
  read from the full mesh rasterised in texture space and expressed in the game mesh's tangent frame (OpenGL: green
  up, glTF's convention).
"""
from __future__ import annotations

import numpy as np

from . import raster as RA


def _quadrics(V, T):
    p0, p1, p2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    area = np.linalg.norm(n, axis=1)
    n = n / np.maximum(area, 1e-18)[:, None]
    d = -np.sum(n * p0, 1)
    pl = np.concatenate([n, d[:, None]], 1)                 # (m, 4)
    K = pl[:, :, None] * pl[:, None, :] * area[:, None, None]
    Q = np.zeros((len(V), 4, 4))
    for k in range(3):
        np.add.at(Q, T[:, k], K)
    return Q


def _cluster(V, T, cell, Q, chart=None, uv=None):
    lo = V.min(0)
    key = np.floor((V - lo) / cell).astype(np.int64)
    _, cid = np.unique(key, axis=0, return_inverse=True)
    cid = cid.ravel()
    nc = int(cid.max()) + 1
    Qc = np.zeros((nc, 4, 4))
    np.add.at(Qc, cid, Q)
    A = Qc[:, :3, :3]
    b = Qc[:, :3, 3]
    mean = np.zeros((nc, 3))
    np.add.at(mean, cid, V)
    cnt = np.bincount(cid, minlength=nc)[:, None]
    mean /= np.maximum(cnt, 1)
    pos = mean.copy()
    det = np.abs(np.linalg.det(A))
    scale = np.trace(A, axis1=1, axis2=2) ** 3 + 1e-30
    ok = det > 1e-6 * scale
    if ok.any():
        x = np.linalg.solve(A[ok], -b[ok][..., None])[..., 0]
        # a solution far outside its cell (a flat region) falls back to the mean
        far = np.linalg.norm(x - mean[ok], axis=1) > 1.5 * cell
        x[far] = mean[ok][far]
        pos[ok] = x
    if chart is None:
        newv = cid
        Vn = pos
        uvn = None
        src_cluster = np.arange(nc)
    else:
        pair = cid * (int(chart.max()) + 1) + chart
        upair, newv = np.unique(pair, return_inverse=True)
        newv = newv.ravel()
        src_cluster = upair // (int(chart.max()) + 1)
        Vn = pos[src_cluster]
        uvn = np.zeros((len(upair), 2))
        np.add.at(uvn, newv, uv)
        uvn /= np.maximum(np.bincount(newv, minlength=len(upair)), 1)[:, None]
    Tn = newv[T]
    cl = cid[T]
    good = (cl[:, 0] != cl[:, 1]) & (cl[:, 1] != cl[:, 2]) & (cl[:, 0] != cl[:, 2])
    Tn = Tn[good]
    # duplicates (the same three cells twice) kept once
    s = np.sort(cid[T][good], 1)
    _, first = np.unique(s, axis=0, return_index=True)
    Tn = Tn[np.sort(first)]
    used = np.unique(Tn)
    remap = -np.ones(len(Vn), np.int64)
    remap[used] = np.arange(len(used))
    return Vn[used], remap[Tn], (uvn[used] if uvn is not None else None)


def simplify(V, T, target: int, chart=None, uv=None):
    """(vertices, triangles, uv or None) with about ``target`` triangles (never more than the input)."""
    V = np.asarray(V, np.float64)
    T = np.asarray(T, np.int64)
    if len(T) <= target:
        return V, T, uv
    Q = _quadrics(V, T)
    ext = float(np.max(V.max(0) - V.min(0)))
    lo, hi = ext / 2000.0, ext / 4.0
    best = None
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        Vn, Tn, uvn = _cluster(V, T, mid, Q, chart, uv)
        if len(Tn) > target:
            lo = mid
        else:
            hi = mid
            best = (Vn, Tn, uvn)
        if best is not None and abs(len(best[1]) - target) < 0.05 * target:
            break
    if best is None:
        best = _cluster(V, T, hi, Q, chart, uv)
    return best


def bake_normals(Vf, Tf, uvf, Vl, Tl, uvl, size: int) -> np.ndarray:
    """uint8 (size, size, 3) tangent-space normal map (OpenGL, green up) for the game mesh (Vl, Tl, uvl), carrying
    the full mesh's (Vf, Tf, uvf) normals; both share the atlas layout. Texels no face covers hold flat (128, 128,
    255) and are bled by the caller if wanted."""
    from ..io.gltf import tangents
    from .isosurface import vertex_normals
    Nf = vertex_normals(Vf, Tf)
    Nl = vertex_normals(Vl, Tl)
    Tg = tangents(Vl, Nl, uvl, Tl)
    ff, bf, _ = RA.rasterize(uvf * size - 0.5, np.zeros(len(uvf)), Tf, size, size)
    fl, bl, _ = RA.rasterize(uvl * size - 0.5, np.zeros(len(uvl)), Tl, size, size)
    out = np.zeros((size, size, 3), np.float32)
    out[..., 2] = 1.0
    m = (ff >= 0) & (fl >= 0)
    if m.any():
        nf = (Nf[Tf[ff[m]]] * bf[m][..., None]).sum(1)
        nl = (Nl[Tl[fl[m]]] * bl[m][..., None]).sum(1)
        tg = (Tg[Tl[fl[m]], :3] * bl[m][..., None]).sum(1)
        w = np.sign((Tg[Tl[fl[m]], 3] * bl[m]).sum(1))
        w[w == 0] = 1
        nl /= np.maximum(np.linalg.norm(nl, axis=1, keepdims=True), 1e-12)
        tg = tg - nl * np.sum(tg * nl, 1, keepdims=True)
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
        bt = np.cross(nl, tg) * w[:, None]
        nf /= np.maximum(np.linalg.norm(nf, axis=1, keepdims=True), 1e-12)
        ts = np.stack([np.sum(nf * tg, 1), np.sum(nf * bt, 1), np.sum(nf * nl, 1)], 1)
        ts[:, 2] = np.maximum(ts[:, 2], 0.05)
        ts /= np.linalg.norm(ts, axis=1, keepdims=True)
        out[m] = ts
    return np.clip(np.rint(out * 127.5 + 127.5), 0, 255).astype(np.uint8)
