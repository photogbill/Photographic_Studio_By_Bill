"""A small triangle rasteriser in NumPy (BUILD_PLAN_DP3D E.8, H.3): which triangle is seen at each pixel, with
barycentric weights and depth. Used to decide which view sees each face (texturing), to bake a game mesh's normal map
from the full mesh in texture space, and to draw the 3D Scan page's turntable without OpenGL.

Every triangle's bounding box is enumerated pixel by pixel (all triangles at once, in chunks), tested with edge
functions, and the nearest one kept per pixel by sorting on depth.
"""
from __future__ import annotations

import numpy as np


def rasterize(P: np.ndarray, z: np.ndarray, T: np.ndarray, H: int, W: int, chunk: int = 3_000_000):
    """``P`` (n, 2) pixel positions (x, y; pixel centres at integers), ``z`` (n,) depth (smaller = nearer; use 0
    for a flat texture-space rasterisation), ``T`` (m, 3). Returns (face (H, W) int, -1 = none; bary (H, W, 3);
    depth (H, W))."""
    P = np.asarray(P, np.float64)
    z = np.asarray(z, np.float64)
    T = np.asarray(T, np.int64)
    face = -np.ones(H * W, np.int64)
    bary = np.zeros((H * W, 3), np.float32)
    zbuf = np.full(H * W, np.inf)
    if not len(T):
        return face.reshape(H, W), bary.reshape(H, W, 3), zbuf.reshape(H, W)
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    lo = np.floor(np.minimum(np.minimum(A, B), C)).astype(np.int64)
    hi = np.ceil(np.maximum(np.maximum(A, B), C)).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi[:, 0] = np.minimum(hi[:, 0], W - 1)
    hi[:, 1] = np.minimum(hi[:, 1], H - 1)
    area = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    ok = (hi[:, 0] >= lo[:, 0]) & (hi[:, 1] >= lo[:, 1]) & (np.abs(area) > 1e-12) & np.isfinite(area)
    tris = np.nonzero(ok)[0]
    bw = hi[tris, 0] - lo[tris, 0] + 1
    bh = hi[tris, 1] - lo[tris, 1] + 1
    cnt = bw * bh
    start = 0
    while start < len(tris):
        acc = np.cumsum(cnt[start:])
        end = start + max(1, int(np.searchsorted(acc, chunk, side="right")))
        sel = tris[start:end]
        c = cnt[start:end]
        rep = np.repeat(np.arange(len(sel)), c)
        k = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        bws = np.repeat(bw[start:end], c)
        px = lo[sel[rep], 0] + k % bws
        py = lo[sel[rep], 1] + k // bws
        t = sel[rep]
        a, b, cc = A[t], B[t], C[t]
        ar = area[t]
        w0 = ((b[:, 0] - px) * (cc[:, 1] - py) - (b[:, 1] - py) * (cc[:, 0] - px)) / ar
        w1 = ((cc[:, 0] - px) * (a[:, 1] - py) - (cc[:, 1] - py) * (a[:, 0] - px)) / ar
        w2 = 1.0 - w0 - w1
        eps = -1e-9
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if inside.any():
            t, px, py = t[inside], px[inside], py[inside]
            w = np.stack([w0[inside], w1[inside], w2[inside]], 1)
            zz = w[:, 0] * z[T[t, 0]] + w[:, 1] * z[T[t, 1]] + w[:, 2] * z[T[t, 2]]
            pix = py * W + px
            # the nearest per pixel: this chunk's best, then against what is there
            order = np.lexsort((zz, pix))
            pix, zz, t, w = pix[order], zz[order], t[order], w[order]
            first = np.r_[True, pix[1:] != pix[:-1]]
            pix, zz, t, w = pix[first], zz[first], t[first], w[first]
            better = zz < zbuf[pix]
            pix, zz, t, w = pix[better], zz[better], t[better], w[better]
            zbuf[pix] = zz
            face[pix] = t
            bary[pix] = w
        start = end
    return face.reshape(H, W), bary.reshape(H, W, 3), zbuf.reshape(H, W)


def project(V, R, t, f, cx, cy, k1=0.0):
    """(pixels (n, 2), depth (n,)) of world points in a view."""
    Xc = np.asarray(V) @ np.asarray(R).T + t
    z = Xc[:, 2]
    zz = np.where(z > 1e-9, z, 1e-9)
    x, y = Xc[:, 0] / zz, Xc[:, 1] / zz
    d = 1 + k1 * (x * x + y * y)
    return np.stack([f * x * d + cx, f * y * d + cy], 1), z


def turntable(V, T, colours, angle_deg: float, elev_deg: float = 20.0, size: int = 480, light=(0.4, 0.6, 0.7),
              up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """A shaded picture (size x size x 3 uint8) of a mesh turned by ``angle_deg`` about its up axis: Lambert shading
    times the per-vertex colours (n, 3) 0..1. For the app's model view."""
    V = np.asarray(V, np.float64)
    c = 0.5 * (V.min(0) + V.max(0))
    r = float(np.linalg.norm(V.max(0) - V.min(0))) / 2 or 1.0
    up = np.asarray(up, float)
    up /= np.linalg.norm(up)
    a, e = np.radians(angle_deg), np.radians(elev_deg)
    # a frame with "up" as y: pick any horizontal axis
    h1 = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 0, 1.0])
    h1 /= np.linalg.norm(h1)
    h2 = np.cross(up, h1)
    d = np.cos(e) * (np.cos(a) * h1 + np.sin(a) * h2) + np.sin(e) * up        # from the object to the eye
    eye = c + d * 3.0 * r
    zc = (c - eye) / np.linalg.norm(c - eye)
    xc = np.cross(zc, up)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc])
    t = -R @ eye
    f = size / (2 * np.tan(np.radians(19)))
    P, z = project(V, R, t, f, (size - 1) / 2, (size - 1) / 2)
    face, bary, _ = rasterize(P, z, T, size, size)
    from .isosurface import vertex_normals
    N = vertex_normals(V, T)
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    img = np.full((size, size, 3), 0.12, np.float32)
    m = face >= 0
    if m.any():
        tri = T[face[m]]
        w = bary[m]
        n = (N[tri] * w[..., None]).sum(1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        col = (np.asarray(colours, np.float32)[tri] * w[..., None]).sum(1)
        shade = 0.3 + 0.7 * np.clip(np.abs(n @ L), 0, 1)
        img[m] = col * shade[:, None]
    return np.clip(img * 255, 0, 255).astype(np.uint8)
