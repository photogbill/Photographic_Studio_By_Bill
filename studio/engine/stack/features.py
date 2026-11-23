"""Features for the panorama (BUILD_PLAN_PANO_IR.md 2.3): corners, descriptors, matching and RANSAC, written in NumPy /
SciPy (no OpenCV: it is not installed with the Studio).

Panorama overlaps are 20-60% of a frame, far outside what tile-by-tile registration can search, so frames are joined
by matching distinctive points first:

* **Corners**: Harris (sigma 1.5) on a 1600-px analysis copy of log brightness, high-passed so a smooth sky gradient
  gives none; then adaptive non-maximal suppression (Brown, Szeliski and Winder, CVPR 2005, "MOPS"), which keeps the
  strongest corners spread over the whole frame instead of piled up on the busiest part.
* **Descriptors**: MOPS: 8 x 8 samples spaced 5 px from the image blurred at sigma 2.5, bias and gain normalised (zero
  mean, unit variance). Upright by default (a panorama turns the camera, it rarely rolls it much); oriented at the
  point's smoothed gradient angle when asked (a roll of more than ~15 degrees between frames).
* **Matching**: exact nearest neighbours by matrix products (the same answer as a k-d tree, several times faster
  in 64 dimensions), Lowe's ratio test (0.8) and mutual best matches only.
* **RANSAC**: 4-point homographies (normalised DLT, Hartley), all hypotheses solved at once; the best is re-fitted to
  its inliers. A pair is accepted when inliers > 8 + 0.3 * matches (Brown and Lowe, IJCV 2007, the probabilistic
  check of "Automatic panoramic image stitching using invariant features").
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter

from ... import log as slog

log = slog.get("engine.stack.features")

N_CORNERS = 2000
DESC_SPACING = 5.0
DESC_BLUR = 2.5


def prepare(lum: np.ndarray) -> np.ndarray:
    """Log brightness with the broad shading removed (a sky gradient or vignetting makes no corners)."""
    z = np.log(np.maximum(np.asarray(lum, np.float32), 1e-4))
    return (z - gaussian_filter(z, 24.0)).astype(np.float32)


def corners(img: np.ndarray, n: int = N_CORNERS, sigma: float = 1.5, border: int = 24) -> np.ndarray:
    """(N x 2) corner positions (y, x), strongest first, spread out by ANMS."""
    x = np.asarray(img, np.float32)
    gy, gx = np.gradient(gaussian_filter(x, 0.7))
    a = gaussian_filter(gx * gx, sigma)
    b = gaussian_filter(gy * gy, sigma)
    c = gaussian_filter(gx * gy, sigma)
    R = (a * b - c * c) - 0.04 * (a + b) ** 2
    h, w = R.shape
    peak = (R == maximum_filter(R, size=5)) & (R > 0)
    peak[:border] = peak[-border:] = False
    peak[:, :border] = peak[:, -border:] = False
    ys, xs = np.nonzero(peak)
    if ys.size == 0:
        return np.zeros((0, 2), np.float32)
    s = R[ys, xs]
    thr = np.percentile(s, 30) if s.size > 4 * n else 0.0
    keep = s > thr
    ys, xs, s = ys[keep], xs[keep], s[keep]
    order = np.argsort(-s)[: 5 * n]
    ys, xs, s = ys[order], xs[order], s[order]
    # sub-pixel: the peak of a parabola through the response and its two neighbours, each way
    c0 = R[ys, xs]
    dyn = R[ys + 1, xs] - R[ys - 1, xs]
    dyd = 2 * c0 - R[ys + 1, xs] - R[ys - 1, xs]
    dxn = R[ys, xs + 1] - R[ys, xs - 1]
    dxd = 2 * c0 - R[ys, xs + 1] - R[ys, xs - 1]
    oy = np.clip(np.where(dyd > 1e-12, dyn / (2 * np.maximum(dyd, 1e-12)), 0), -0.5, 0.5)
    ox = np.clip(np.where(dxd > 1e-12, dxn / (2 * np.maximum(dxd, 1e-12)), 0), -0.5, 0.5)
    pts = np.stack([ys + oy, xs + ox], 1).astype(np.float32)
    return anms(pts, s, n)


def anms(pts: np.ndarray, strength: np.ndarray, n: int, c_robust: float = 0.9) -> np.ndarray:
    """Adaptive non-maximal suppression: each point's radius is the distance to the nearest clearly stronger one;
    the n points with the largest radii are kept (strongest first among equals)."""
    m = len(pts)
    if m <= n:
        return pts
    r = np.full(m, np.inf, np.float32)
    s = np.asarray(strength, np.float32)
    for i0 in range(0, m, 512):
        i1 = min(m, i0 + 512)
        d2 = ((pts[i0:i1, None, :] - pts[None, :, :]) ** 2).sum(-1)
        stronger = s[None, :] * c_robust > s[i0:i1, None]
        d2 = np.where(stronger, d2, np.inf)
        r[i0:i1] = np.sqrt(d2.min(axis=1))
    order = np.lexsort((-s, -r))
    return pts[order[:n]]


def describe(img: np.ndarray, pts: np.ndarray, oriented: bool = False) -> tuple:
    """(N x 64 descriptors, angles) for corner points; points too near the edge get NaN rows."""
    x = gaussian_filter(np.asarray(img, np.float32), DESC_BLUR)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 64), np.float32), np.zeros(0, np.float32)
    if oriented:
        gy, gx = np.gradient(gaussian_filter(np.asarray(img, np.float32), 4.5))
        ang = np.arctan2(map_coordinates(gy, pts.T, order=1), map_coordinates(gx, pts.T, order=1))
    else:
        ang = np.zeros(n, np.float32)
    g = (np.arange(8, dtype=np.float32) - 3.5) * DESC_SPACING
    gyy, gxx = np.meshgrid(g, g, indexing="ij")
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    oy = gyy.ravel()[None, :]
    ox = gxx.ravel()[None, :]
    sy = pts[:, 0:1] + sa * ox + ca * oy
    sx = pts[:, 1:2] + ca * ox - sa * oy
    v = map_coordinates(x, [sy.ravel(), sx.ravel()], order=1, mode="nearest").reshape(n, 64)
    mu = v.mean(axis=1, keepdims=True)
    sd = v.std(axis=1, keepdims=True)
    d = (v - mu) / np.maximum(sd, 1e-6)
    flat = sd[:, 0] < 1e-4
    h, w = x.shape
    reach = DESC_SPACING * 3.5 * 1.42 + 1
    out = (pts[:, 0] < reach) | (pts[:, 1] < reach) | (pts[:, 0] > h - 1 - reach) | (pts[:, 1] > w - 1 - reach)
    d[flat | out] = np.nan
    return d.astype(np.float32), ang.astype(np.float32)


def _two_nearest(q: np.ndarray, ref: np.ndarray, rows: int = 1024) -> tuple:
    """Exact two nearest neighbours of every row of ``q`` among the rows of ``ref`` (Euclidean): (distances (N, 2),
    indices (N, 2)). Brute force by matrix products: in 64 dimensions a k-d tree visits nearly every point anyway,
    and the products are many times faster (the same answer as cKDTree.query(k=2))."""
    q = np.asarray(q, np.float32)
    ref = np.asarray(ref, np.float32)
    rn = (ref * ref).sum(1)
    dd = np.empty((len(q), 2), np.float64)
    jj = np.empty((len(q), 2), np.int64)
    for r0 in range(0, len(q), rows):
        a = q[r0:r0 + rows]
        d2 = (a * a).sum(1)[:, None] + rn[None, :] - 2.0 * (a @ ref.T)
        np.maximum(d2, 0.0, out=d2)
        if d2.shape[1] >= 2:
            part = np.argpartition(d2, 1, axis=1)[:, :2]
        else:
            part = np.zeros((len(a), 2), np.int64)
        v = np.take_along_axis(d2, part, 1)
        o = np.argsort(v, axis=1)
        part = np.take_along_axis(part, o, 1)
        v = np.take_along_axis(v, o, 1)
        dd[r0:r0 + rows] = np.sqrt(v)
        jj[r0:r0 + rows] = part
    return dd, jj


def match(d1: np.ndarray, d2: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    """(M x 2) index pairs (i in d1, j in d2): ratio test both ways and mutual best."""
    ok1 = ~np.isnan(d1).any(axis=1)
    ok2 = ~np.isnan(d2).any(axis=1)
    i1, i2 = np.nonzero(ok1)[0], np.nonzero(ok2)[0]
    if len(i1) < 2 or len(i2) < 2:
        return np.zeros((0, 2), np.int64)
    dd, jj = _two_nearest(d1[i1], d2[i2])
    good12 = dd[:, 0] < ratio * dd[:, 1]
    best12 = i2[jj[:, 0]]
    dd2, ii = _two_nearest(d2[i2], d1[i1])
    best21 = np.full(len(d2), -1)
    best21[i2] = np.where(dd2[:, 0] < ratio * dd2[:, 1], i1[ii[:, 0]], -1)
    a = i1[good12]
    b = best12[good12]
    mutual = best21[b] == a
    return np.stack([a[mutual], b[mutual]], 1).astype(np.int64)


# ------------------------------------------------------------------------------------------- homographies
def _normalise(p: np.ndarray):
    c = p.mean(axis=0)
    d = np.sqrt(((p - c) ** 2).sum(1)).mean()
    s = np.sqrt(2) / max(d, 1e-9)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    ph = np.c_[p, np.ones(len(p))] @ T.T
    return ph[:, :2], T


def dlt(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Homography (3x3, x2 ~ H x1) from point pairs in (x, y) order, normalised DLT."""
    a, T1 = _normalise(np.asarray(p1, float))
    b, T2 = _normalise(np.asarray(p2, float))
    n = len(a)
    A = np.zeros((2 * n, 9))
    A[0::2, 0:2] = a
    A[0::2, 2] = 1
    A[0::2, 6:8] = -b[:, 0:1] * a
    A[0::2, 8] = -b[:, 0]
    A[1::2, 3:5] = a
    A[1::2, 5] = 1
    A[1::2, 6:8] = -b[:, 1:2] * a
    A[1::2, 8] = -b[:, 1]
    _, _, vt = np.linalg.svd(A)
    H = vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ H @ T1
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def _dlt_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Many 4-point homographies at once: a, b (K x 4 x 2), already normalised coordinates."""
    K = a.shape[0]
    A = np.zeros((K, 8, 9))
    A[:, 0::2, 0:2] = a
    A[:, 0::2, 2] = 1
    A[:, 0::2, 6:8] = -b[:, :, 0:1] * a
    A[:, 0::2, 8] = -b[:, :, 0]
    A[:, 1::2, 3:5] = a
    A[:, 1::2, 5] = 1
    A[:, 1::2, 6:8] = -b[:, :, 1:2] * a
    A[:, 1::2, 8] = -b[:, :, 1]
    _, _, vt = np.linalg.svd(A)
    return vt[:, -1].reshape(K, 3, 3)


def project(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    ph = np.c_[p, np.ones(len(p))] @ np.asarray(H).T
    return ph[:, :2] / np.where(np.abs(ph[:, 2:3]) < 1e-12, 1e-12, ph[:, 2:3])


def ransac_homography(p1: np.ndarray, p2: np.ndarray, thr: float = 2.0, iters: int = 800, rng=None):
    """(H, inlier mask) with x2 ~ H x1 (points in (x, y) order), or (None, empty mask) when fewer than 4 matches."""
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    n = len(p1)
    if n < 4:
        return None, np.zeros(n, bool)
    rng = rng if rng is not None else np.random.default_rng(0)
    a, T1 = _normalise(p1)
    b, T2 = _normalise(p2)
    idx = np.array([rng.choice(n, 4, replace=False) for _ in range(iters)])
    Hs = _dlt_batch(a[idx], b[idx])
    T2i = np.linalg.inv(T2)
    Hs = T2i[None] @ Hs @ T1[None]
    ph = np.einsum("kij,nj->kni", Hs, np.c_[p1, np.ones(n)])
    w = ph[..., 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    err = np.sqrt(((ph[..., :2] / w - p2[None]) ** 2).sum(-1))
    inl = err < thr
    cnt = inl.sum(axis=1)
    best = int(np.argmax(cnt))
    mask = inl[best]
    if mask.sum() < 4:
        return None, mask
    for _ in range(3):                                   # re-fit to the inliers, then re-score
        H = dlt(p1[mask], p2[mask])
        e = np.sqrt(((project(H, p1) - p2) ** 2).sum(1))
        new = e < thr
        if new.sum() < 4 or np.array_equal(new, mask):
            break
        mask = new
    H = dlt(p1[mask], p2[mask])
    return H, mask


def pair_ok(n_inliers: int, n_matches: int) -> bool:
    """Brown and Lowe 2007: the match set is real when inliers > 8 + 0.3 * matches."""
    return n_inliers > 8 + 0.3 * n_matches


def refine_lk(img_a: np.ndarray, img_b: np.ndarray, pa: np.ndarray, pb: np.ndarray, half: int = 8,
              iters: int = 8) -> tuple:
    """Sub-pixel refinement of matched points: each point of b moved by Lucas-Kanade (translation only) so its patch
    matches the patch around the point in a (both images prepared: log, high-passed). Points (x, y). Returns
    (refined pb, ok mask: converged and moved less than 2 px)."""
    from scipy.ndimage import map_coordinates
    A = np.asarray(img_a, np.float32)
    B = np.asarray(img_b, np.float32)
    g = np.arange(-half, half + 1, dtype=np.float32)
    gy, gx = np.meshgrid(g, g, indexing="ij")
    wgt = np.exp(-(gx ** 2 + gy ** 2) / (2 * (half / 1.5) ** 2)).ravel()[None, :]
    n = len(pa)
    if n == 0:
        return pb.copy(), np.zeros(0, bool)
    ya = pa[:, 1:2] + gy.ravel()[None, :]
    xa = pa[:, 0:1] + gx.ravel()[None, :]
    Ta = map_coordinates(A, [ya.ravel(), xa.ravel()], order=1, mode="nearest").reshape(n, -1)
    Ta = Ta - (Ta * wgt).sum(1, keepdims=True) / wgt.sum()
    By, Bx = np.gradient(B)
    q = pb.astype(np.float64).copy()
    for _ in range(iters):
        yb = q[:, 1:2] + gy.ravel()[None, :]
        xb = q[:, 0:1] + gx.ravel()[None, :]
        cs = [yb.ravel(), xb.ravel()]
        Ib = map_coordinates(B, cs, order=1, mode="nearest").reshape(n, -1)
        Ib = Ib - (Ib * wgt).sum(1, keepdims=True) / wgt.sum()
        Iy = map_coordinates(By, cs, order=1, mode="nearest").reshape(n, -1)
        Ix = map_coordinates(Bx, cs, order=1, mode="nearest").reshape(n, -1)
        e = Ta - Ib
        a11 = (wgt * Ix * Ix).sum(1)
        a12 = (wgt * Ix * Iy).sum(1)
        a22 = (wgt * Iy * Iy).sum(1)
        b1 = (wgt * Ix * e).sum(1)
        b2 = (wgt * Iy * e).sum(1)
        det = a11 * a22 - a12 * a12
        good = det > 1e-12
        dx = np.where(good, (a22 * b1 - a12 * b2) / np.where(good, det, 1), 0)
        dy = np.where(good, (a11 * b2 - a12 * b1) / np.where(good, det, 1), 0)
        q[:, 0] += np.clip(dx, -1, 1)
        q[:, 1] += np.clip(dy, -1, 1)
        if np.max(np.abs(np.r_[dx, dy])) < 0.01:
            break
    moved = np.hypot(*(q - pb).T)
    ok = (moved < 2.0) & good
    return q, ok


class FrameFeatures:
    """Corners (y, x) and descriptors of one analysis copy."""

    def __init__(self, lum: np.ndarray, n: int = N_CORNERS, oriented: bool = False):
        self.shape = lum.shape
        img = prepare(lum)
        self.pts = corners(img, n)
        self.desc, self.ang = describe(img, self.pts, oriented)
        self.img = img


def match_pair(fa: FrameFeatures, fb: FrameFeatures, thr: float = 2.0, rng=None) -> dict | None:
    """Match two frames' features; {"H" (b <- a, (x, y) coords), "pa", "pb" (inlier points (x, y)), "inliers",
    "matches"} when the pair passes the check, else None."""
    m = match(fa.desc, fb.desc)
    if len(m) < 8:
        return None
    pa = fa.pts[m[:, 0]][:, ::-1].astype(float)
    pb = fb.pts[m[:, 1]][:, ::-1].astype(float)
    H, mask = ransac_homography(pa, pb, thr, rng=rng)
    if H is None or not pair_ok(int(mask.sum()), len(m)):
        return None
    a, b = pa[mask], pb[mask]
    if getattr(fa, "img", None) is not None and getattr(fb, "img", None) is not None:
        b2, ok = refine_lk(fa.img, fb.img, a, b)
        if ok.sum() >= max(8, 0.5 * len(a)):
            a, b = a[ok], b2[ok]
            H = dlt(a, b)
    return {"H": H, "pa": a, "pb": b, "inliers": int(len(a)), "matches": int(len(m))}
