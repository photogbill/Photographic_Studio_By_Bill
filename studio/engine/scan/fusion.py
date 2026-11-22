"""Volumetric fusion of the views' depth, closed by their silhouettes (BUILD_PLAN_DP3D E.8).

* A **truncated signed distance volume** over the object's box (Curless and Levoy 1996): every voxel is projected into
  every placed view; where the view has a depth there, the voxel takes (measured distance - voxel distance), cut at a
  few voxels, weighted by that depth's confidence.
* The **visual hull** (Laurentini 1994): a voxel that any view sees outside its subject mask is outside the object.
  Where no view measured the surface (the back, the underside), the hull closes the object: inside every silhouette
  it projects into counts as inside. Those parts of the surface are **filled, not measured** (violet).

Field: negative inside, positive outside (``isosurface.surface_nets``). Returns, beside the field, how much each
voxel was measured (the depth weight) so every vertex can say measured (cyan) or filled (violet).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates

from . import geom as G

MAX_VOXELS = 192 ** 3
TRUNC_VOX = 3.0


@dataclass
class Volume:
    field: np.ndarray        # (nx, ny, nz) float32, negative inside
    weight: np.ndarray       # measured weight per voxel
    origin: np.ndarray       # world position of voxel (0, 0, 0)
    voxel: float
    carved: np.ndarray       # bool: outside some silhouette


def object_box(sol, views, margin: float = 0.12):
    """(lo, hi) of the object: the placed points that fall inside a view's subject mask (1-99 %), grown."""
    P = sol.points
    if not len(P):
        raise ValueError("no points were placed")
    inside = np.zeros(len(P), bool)
    for v in range(len(sol.names)):
        if not sol.placed[v] or views[v].mask is None:
            continue
        s = sol.obs_view == v
        xy = np.rint(sol.obs_xy[s]).astype(int)
        m = views[v].mask
        ok = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < m.shape[1]) & (xy[:, 1] < m.shape[0])
        hit = np.zeros(s.sum(), bool)
        hit[ok] = m[xy[ok, 1], xy[ok, 0]]
        inside[sol.obs_point[s][hit]] = True
    Q = P[inside] if inside.sum() > 30 else P
    lo, hi = np.percentile(Q, 1, axis=0), np.percentile(Q, 99, axis=0)
    pad = (hi - lo) * margin
    return lo - pad, hi + pad


def integrate(sol, views, dense: dict, box=None, max_voxels: int = MAX_VOXELS, hull: bool = True,
              progress=None) -> Volume:
    """The fused volume. ``dense``: {view index: dense.DenseView}."""
    lo, hi = box if box is not None else object_box(sol, views)
    ext = hi - lo
    voxel = float((np.prod(ext) / max_voxels) ** (1.0 / 3.0))
    n = np.maximum(np.ceil(ext / voxel).astype(int), 4)
    origin = lo
    trunc = TRUNC_VOX * voxel
    D = np.zeros(int(np.prod(n)), np.float32)
    W = np.zeros_like(D)
    carved = np.zeros(D.shape, bool)
    seen = np.zeros(D.shape, np.int16)
    gx, gy, gz = (origin[k] + voxel * np.arange(n[k]) for k in range(3))
    placed = [v for v in range(len(sol.names)) if sol.placed[v]]
    chunk = 2_000_000
    total = D.size
    for ci, s0 in enumerate(range(0, total, chunk)):
        idx = np.arange(s0, min(s0 + chunk, total))
        ix, rem = np.divmod(idx, n[1] * n[2])
        iy, iz = np.divmod(rem, n[2])
        X = np.stack([gx[ix], gy[iy], gz[iz]], -1)
        for v in placed:
            Xc = X @ sol.R[v].T + sol.t[v]
            z = Xc[:, 2]
            front = z > 1e-6
            zz = np.where(front, z, 1.0)
            x, y = Xc[:, 0] / zz, Xc[:, 1] / zz
            d = 1 + sol.k1 * (x * x + y * y)
            vw = views[v]
            u = sol.f[v] * x * d + vw.cx
            w_ = sol.f[v] * y * d + vw.cy
            h, wd = vw.shape
            inside = front & (u >= 0) & (w_ >= 0) & (u <= wd - 1) & (w_ <= h - 1)
            if not inside.any():
                continue
            seen[idx[inside]] += 1
            if hull and vw.mask is not None:
                mk = vw.mask
                ui = np.clip(np.rint(u).astype(int), 0, wd - 1)
                vi = np.clip(np.rint(w_).astype(int), 0, h - 1)
                out = inside & ~mk[vi, ui]
                carved[idx[out]] = True
            dv = dense.get(v)
            if dv is None:
                continue
            s = dv.scale
            ud = (u + 0.5) * s - 0.5
            vd = (w_ + 0.5) * s - 0.5
            hh, ww = dv.z.shape
            ui = np.clip(np.rint(ud).astype(int), 0, ww - 1)
            vi = np.clip(np.rint(vd).astype(int), 0, hh - 1)
            zm = dv.z[vi, ui]
            cm = dv.conf[vi, ui]
            ok = inside & np.isfinite(zm) & (cm > 0)
            sdf = np.where(ok, zm - z, 0.0)
            ok &= sdf > -trunc
            t = np.clip(sdf / trunc, -1.0, 1.0)
            D[idx[ok]] += (cm[ok] * t[ok]).astype(np.float32)
            W[idx[ok]] += cm[ok].astype(np.float32)
        if progress:
            progress((ci + 1) * chunk / total, "fusing the depth of every view")
    measured = W > 1e-3
    F = np.where(measured, D / np.maximum(W, 1e-6), 0.0)
    if hull:
        # not measured: inside every silhouette it was seen in -> inside (filled); carved or never seen -> outside
        F = np.where(measured, F, np.where(carved | (seen == 0), 1.0, -1.0))
        F = np.where(carved & (F < 0) & ~measured, 1.0, F)
    else:
        F = np.where(measured, F, 1.0)
    return Volume(field=F.reshape(n).astype(np.float32), weight=W.reshape(n), origin=np.asarray(origin, float),
                  voxel=voxel, carved=carved.reshape(n))


def mesh(vol: Volume, smooth: int = 4):
    """(vertices, triangles, measured share per vertex 0..1) of the fused volume: surface nets on the field padded
    with outside, the largest piece, Taubin smoothing."""
    from . import isosurface as IS
    F = np.pad(vol.field, 1, constant_values=1.0)
    Wp = np.pad(vol.weight, 1, constant_values=0.0)
    V, T, cells = IS.surface_nets(F, vol.origin - vol.voxel, vol.voxel)
    if not len(T):
        raise ValueError("the fused volume has no surface")
    # measured: the cell's corners carry depth weight
    wsum = np.zeros(len(cells), np.float32)
    for a, b, c in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)):
        wsum += Wp[cells[:, 0] + a, cells[:, 1] + b, cells[:, 2] + c] > 1e-3
    meas = (wsum / 8.0).astype(np.float32)
    V, T, used = IS.largest_piece(V, T)
    meas = meas[used]
    if smooth:
        V = IS.taubin(V, T, rounds=smooth)
    return V, T, meas
