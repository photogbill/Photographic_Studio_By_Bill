"""Oriented, scale-normalised features for unordered views (BUILD_PLAN_DP3D E.2), our own NumPy / SciPy code.

Orbiting an object rolls and rescales it between views, so the panorama's upright single-scale MOPS corners are not
enough. This is the scale-invariant feature transform of Lowe (IJCV 2004; the patent expired in 2020):

* **Detection**: a Gaussian scale space (3 levels per octave, sigma 1.6, up to 4 octaves; images smaller than 900 px
  are doubled first), differences of Gaussians, extrema in their 3 x 3 x 3 neighbourhood, refined once to a fraction
  of a pixel and of a level, low-contrast and edge-like points dropped (the Hessian ratio test, r = 10). Inside the
  subject mask (plus a margin) when one is given, so a background that moved between stacks does not vote.
* **Orientation**: a 36-bin histogram of gradient angles round the point (Gaussian weight 1.5 sigma); each peak within
  80 % of the highest gives the point another orientation.
* **Descriptor**: the patch is sampled on a 16 x 16 grid turned to the point's orientation and spaced by its scale, so
  the gradients taken across the samples are already in the point's frame; 4 x 4 cells x 8 angle bins, spread
  between neighbouring cells and bins, Gaussian weighted, normalised, clipped at 0.2, normalised again, then RootSIFT
  (Arandjelovic and Zisserman, CVPR 2012: square root of the L1-normalised histogram), so a dot product is the
  Hellinger kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (binary_dilation, gaussian_filter, map_coordinates, maximum_filter, minimum_filter,
                           zoom)

S_LEVELS = 3
SIGMA0 = 1.6
CONTRAST = 0.04
EDGE_R = 10.0
N_MAX = 4000


@dataclass
class Features:
    xy: np.ndarray          # (N, 2) float32, x right, y down, pixels of the image given
    scale: np.ndarray       # (N,) sigma in those pixels
    angle: np.ndarray       # (N,) radians
    strength: np.ndarray    # (N,) |DoG| at the extremum
    desc: np.ndarray        # (N, 128) float32 RootSIFT (unit length)

    def __len__(self):
        return int(self.xy.shape[0])

    def subset(self, keep):
        return Features(self.xy[keep], self.scale[keep], self.angle[keep], self.strength[keep], self.desc[keep])

    @staticmethod
    def empty():
        z = np.zeros((0,), np.float32)
        return Features(np.zeros((0, 2), np.float32), z, z, z, np.zeros((0, 128), np.float32))


def _normalise(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, np.float32)
    if g.ndim == 3:
        g = 0.2126 * g[..., 0] + 0.7152 * g[..., 1] + 0.0722 * g[..., 2]
    lo, hi = np.percentile(g, [1, 99])
    return np.clip((g - lo) / max(float(hi - lo), 1e-6), -0.2, 1.2).astype(np.float32)


def _pyramid(img: np.ndarray, n_oct: int):
    k = 2.0 ** (1.0 / S_LEVELS)
    sig = [SIGMA0 * k ** i for i in range(S_LEVELS + 3)]
    octs = []
    base = img
    for o in range(n_oct):
        g = [base]
        for i in range(1, S_LEVELS + 3):
            g.append(gaussian_filter(g[-1], np.sqrt(sig[i] ** 2 - sig[i - 1] ** 2), mode="nearest"))
        g = np.stack(g)
        octs.append(g)
        base = g[S_LEVELS][::2, ::2]
        if min(base.shape) < 24:
            break
    return octs, sig


def _orientations(G: np.ndarray, x: np.ndarray, y: np.ndarray, sigma: np.ndarray):
    """(index into the points, angle) for every orientation peak of every point."""
    n = x.size
    if n == 0:
        return np.zeros(0, int), np.zeros(0, np.float32)
    P = 17
    g = (np.arange(P) - (P - 1) / 2.0) / ((P - 1) / 2.0)          # -1..1
    rad = 3.0 * 1.5 * sigma                                        # window radius
    oy, ox = np.meshgrid(g, g, indexing="ij")
    ys = y[:, None, None] + oy[None] * rad[:, None, None]
    xs = x[:, None, None] + ox[None] * rad[:, None, None]
    v = map_coordinates(G, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(n, P, P)
    sp = (2 * rad / (P - 1))[:, None, None]
    gx = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2 * sp)
    gy = (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / (2 * sp)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    gg = g[1:-1]
    wy, wx = np.meshgrid(gg, gg, indexing="ij")
    wgt = np.exp(-(wx ** 2 + wy ** 2) * (3.0 ** 2) / 2.0)          # sigma_w = 1.5 sigma = a third of the radius
    b = (np.floor((ang + np.pi) / (2 * np.pi) * 36).astype(int)) % 36
    idx = (np.arange(n)[:, None, None] * 36 + b).ravel()
    hist = np.bincount(idx, weights=(mag * wgt[None]).ravel(), minlength=n * 36).reshape(n, 36)
    for _ in range(2):
        hist = (np.roll(hist, 1, 1) + hist + np.roll(hist, -1, 1)) / 3.0
    left, right = np.roll(hist, 1, 1), np.roll(hist, -1, 1)
    peak = (hist > left) & (hist >= right) & (hist >= 0.8 * hist.max(1, keepdims=True)) & (hist > 0)
    pi, pb = np.nonzero(peak)
    l_, c_, r_ = left[pi, pb], hist[pi, pb], right[pi, pb]
    den = l_ - 2 * c_ + r_
    off = np.where(np.abs(den) > 1e-12, 0.5 * (l_ - r_) / den, 0.0)
    a = ((pb + 0.5 + off) / 36.0) * 2 * np.pi - np.pi
    return pi, a.astype(np.float32)


def _descriptors(G: np.ndarray, x, y, sigma, angle) -> np.ndarray:
    n = x.size
    if n == 0:
        return np.zeros((0, 128), np.float32)
    P = 16
    half = 3.0 * sigma * 4 / 2.0                                   # 4 cells of 3 sigma: half the window width
    sp = 2 * half / P
    g = (np.arange(-1, P + 1) - (P - 1) / 2.0)                     # 18 samples (one border for the gradients)
    oy, ox = np.meshgrid(g, g, indexing="ij")
    ca, sa = np.cos(angle), np.sin(angle)
    # sample positions: the grid turned by the orientation and scaled by the point's spacing
    dx = (ox[None] * ca[:, None, None] - oy[None] * sa[:, None, None]) * sp[:, None, None]
    dy = (ox[None] * sa[:, None, None] + oy[None] * ca[:, None, None]) * sp[:, None, None]
    v = map_coordinates(G, [(y[:, None, None] + dy).ravel(), (x[:, None, None] + dx).ravel()], order=1,
                        mode="nearest").reshape(n, P + 2, P + 2)
    gx = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) * 0.5
    gy = (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) * 0.5
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), 2 * np.pi)
    c = (np.arange(P) - (P - 1) / 2.0) / 4.0 + 1.5                 # cell coordinate of each sample, 0..3
    cy, cx = np.meshgrid(c, c, indexing="ij")
    w_gauss = np.exp(-((cx - 1.5) ** 2 + (cy - 1.5) ** 2) / (2 * 2.0 ** 2))
    m = mag * w_gauss[None]
    ob = ang / (2 * np.pi) * 8.0
    o0 = np.floor(ob).astype(int)
    fo = ob - o0
    x0 = np.floor(cx).astype(int)
    y0 = np.floor(cy).astype(int)
    fx, fy = cx - x0, cy - y0
    hist = np.zeros(n * 4 * 4 * 8)
    base = (np.arange(n) * 128)[:, None, None]
    for dyc, wy in ((0, 1 - fy), (1, fy)):
        yy = y0 + dyc
        oky = (yy >= 0) & (yy < 4)
        for dxc, wx in ((0, 1 - fx), (1, fx)):
            xx = x0 + dxc
            okc = oky & (xx >= 0) & (xx < 4)
            cell = (np.clip(yy, 0, 3) * 4 + np.clip(xx, 0, 3)) * 8
            for doc, wo in ((0, 1 - fo), (1, fo)):
                ob_ = (o0 + doc) % 8
                wgt = m * (wy * wx * okc)[None] * wo
                idx = base + cell[None] + ob_
                hist += np.bincount(idx.ravel(), weights=wgt.ravel(), minlength=hist.size)
    d = hist.reshape(n, 128)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    d = np.minimum(d, 0.2)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    d /= np.maximum(d.sum(1, keepdims=True), 1e-12)                 # RootSIFT
    return np.sqrt(d).astype(np.float32)


def detect(gray: np.ndarray, mask: np.ndarray | None = None, margin: int = 24, n_max: int = N_MAX,
           n_oct: int = 4) -> Features:
    """Features of one picture (grey or RGB, any range). ``mask``: True where points may be (grown by ``margin``
    pixels)."""
    img = _normalise(gray)
    up = 2.0 if max(img.shape) < 900 else 1.0
    if up > 1:
        img = zoom(img, 2.0, order=1, grid_mode=True, mode="nearest")   # pixel centres: x_up = 2 x + 0.5
        prior = 1.0
    else:
        prior = 0.5
    img = gaussian_filter(img, np.sqrt(max(SIGMA0 ** 2 - prior ** 2, 0.01)), mode="nearest")
    octs, sig = _pyramid(img, n_oct)
    keep_mask = None
    if mask is not None:
        mk = np.asarray(mask, bool)
        if margin > 0:
            mk = binary_dilation(mk, iterations=int(margin))
        keep_mask = mk
    thr = CONTRAST / S_LEVELS
    out = []
    for o, G in enumerate(octs):
        D = G[1:] - G[:-1]                                         # (S+2, h, w)
        mx = maximum_filter(D, size=(3, 3, 3), mode="nearest")
        mn = minimum_filter(D, size=(3, 3, 3), mode="nearest")
        cand = ((D == mx) | (D == mn)) & (np.abs(D) > 0.5 * thr)
        cand[0] = cand[-1] = False
        cand[:, :5] = cand[:, -5:] = False
        cand[:, :, :5] = cand[:, :, -5:] = False
        s_, y_, x_ = np.nonzero(cand)
        if s_.size == 0:
            continue
        # one Newton step in (x, y, s)
        d0 = D[s_, y_, x_]
        dx = 0.5 * (D[s_, y_, x_ + 1] - D[s_, y_, x_ - 1])
        dy = 0.5 * (D[s_, y_ + 1, x_] - D[s_, y_ - 1, x_])
        ds = 0.5 * (D[s_ + 1, y_, x_] - D[s_ - 1, y_, x_])
        dxx = D[s_, y_, x_ + 1] + D[s_, y_, x_ - 1] - 2 * d0
        dyy = D[s_, y_ + 1, x_] + D[s_, y_ - 1, x_] - 2 * d0
        dss = D[s_ + 1, y_, x_] + D[s_ - 1, y_, x_] - 2 * d0
        dxy = 0.25 * (D[s_, y_ + 1, x_ + 1] - D[s_, y_ + 1, x_ - 1] - D[s_, y_ - 1, x_ + 1] + D[s_, y_ - 1, x_ - 1])
        dxs = 0.25 * (D[s_ + 1, y_, x_ + 1] - D[s_ + 1, y_, x_ - 1] - D[s_ - 1, y_, x_ + 1] + D[s_ - 1, y_, x_ - 1])
        dys = 0.25 * (D[s_ + 1, y_ + 1, x_] - D[s_ + 1, y_ - 1, x_] - D[s_ - 1, y_ + 1, x_] + D[s_ - 1, y_ - 1, x_])
        Hm = np.stack([np.stack([dxx, dxy, dxs], -1), np.stack([dxy, dyy, dys], -1),
                       np.stack([dxs, dys, dss], -1)], -2)
        gv = np.stack([dx, dy, ds], -1)
        det = np.linalg.det(Hm)
        ok = np.abs(det) > 1e-12
        offs = np.zeros_like(gv)
        offs[ok] = -np.linalg.solve(Hm[ok], gv[ok][..., None])[..., 0]
        ok &= np.all(np.abs(offs) < 0.6, axis=1)
        val = d0 + 0.5 * np.sum(gv * offs, axis=1)
        tr = dxx + dyy
        dt = dxx * dyy - dxy ** 2
        ok &= (np.abs(val) >= thr) & (dt > 0) & (tr * tr * EDGE_R < (EDGE_R + 1) ** 2 * dt)
        if not ok.any():
            continue
        xo = x_[ok] + offs[ok, 0]
        yo = y_[ok] + offs[ok, 1]
        so = s_[ok] + offs[ok, 2]
        sigma = SIGMA0 * 2.0 ** (so / S_LEVELS)
        lev = np.clip(np.rint(so).astype(int), 0, G.shape[0] - 1)
        strength = np.abs(val[ok])
        scale_img = 2.0 ** o / up
        xi = (xo * 2.0 ** o + 0.5) / up - 0.5                     # back to the given picture's pixel centres
        yi = (yo * 2.0 ** o + 0.5) / up - 0.5
        if keep_mask is not None:
            hh, ww = keep_mask.shape
            inside = keep_mask[np.clip(np.rint(yi).astype(int), 0, hh - 1), np.clip(np.rint(xi).astype(int), 0, ww - 1)]
            xo, yo, sigma, lev, strength, xi, yi = (a[inside] for a in (xo, yo, sigma, lev, strength, xi, yi))
        for L in np.unique(lev):
            sel = lev == L
            pi, ang = _orientations(G[L], xo[sel], yo[sel], sigma[sel])
            if pi.size == 0:
                continue
            dsc = _descriptors(G[L], xo[sel][pi], yo[sel][pi], sigma[sel][pi], ang)
            out.append((np.stack([xi[sel][pi], yi[sel][pi]], -1), sigma[sel][pi] * scale_img, ang,
                        strength[sel][pi], dsc))
    if not out:
        return Features.empty()
    xy = np.concatenate([a[0] for a in out]).astype(np.float32)
    sc = np.concatenate([a[1] for a in out]).astype(np.float32)
    an = np.concatenate([a[2] for a in out]).astype(np.float32)
    st = np.concatenate([a[3] for a in out]).astype(np.float32)
    ds = np.concatenate([a[4] for a in out]).astype(np.float32)
    order = np.argsort(-st)[:n_max]
    return Features(xy[order], sc[order], an[order], st[order], ds[order])


def match(fa: Features, fb: Features, ratio: float = 0.8, chunk: int = 2048) -> np.ndarray:
    """(M, 2) indices: mutual nearest neighbours that pass Lowe's ratio test (RootSIFT: distance^2 = 2 - 2 dot)."""
    if len(fa) < 2 or len(fb) < 2:
        return np.zeros((0, 2), int)
    A, B = fa.desc, fb.desc

    def best2(X, Y):
        i1 = np.empty(len(X), int)
        d1 = np.empty(len(X))
        d2 = np.empty(len(X))
        for s in range(0, len(X), chunk):
            sim = X[s:s + chunk] @ Y.T
            top = np.argpartition(-sim, 1, axis=1)[:, :2]
            v = np.take_along_axis(sim, top, 1)
            o = np.argsort(-v, axis=1)
            top = np.take_along_axis(top, o, 1)
            v = np.take_along_axis(v, o, 1)
            i1[s:s + chunk] = top[:, 0]
            d1[s:s + chunk] = np.sqrt(np.maximum(2 - 2 * v[:, 0], 0))
            d2[s:s + chunk] = np.sqrt(np.maximum(2 - 2 * v[:, 1], 0))
        return i1, d1, d2
    ia, da1, da2 = best2(A, B)
    ib, _, _ = best2(B, A)
    a = np.arange(len(A))
    ok = (da1 < ratio * da2) & (ib[ia] == a)
    return np.stack([a[ok], ia[ok]], -1)


# ------------------------------------------------------------------------------------------ affine simulation (ASIFT)
TILTS = (1.0, 2 ** 0.5, 2.0, 2 ** 1.5)


def simulations(tilts=TILTS) -> list:
    """(tilt, longitude in degrees) pairs: longitudes 72 / t degrees apart over half a turn (Morel and Yu 2009)."""
    out = []
    for t in tilts:
        if t <= 1.0:
            out.append((1.0, 0.0))
            continue
        step = 72.0 / t
        out.extend((t, phi) for phi in np.arange(0.0, 180.0, step))
    return out


def _simulate(img: np.ndarray, t: float, phi: float, mask=None):
    """The picture turned by phi and squeezed by t across (blurred first so it does not alias), and the 2 x 3
    affine taking original (x, y) to the simulated picture's (x, y)."""
    from scipy.ndimage import affine_transform, gaussian_filter1d
    h, w = img.shape
    a = np.radians(phi)
    Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])        # (x, y) -> turned (x, y)
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float) @ Rm.T
    lo = corners.min(0)
    hi = corners.max(0)
    W2, H2 = int(np.ceil(hi[0] - lo[0])) + 1, int(np.ceil(hi[1] - lo[1])) + 1
    # output (y2, x2) -> input (y, x): turned = Rm @ orig - lo  ->  orig = Rm^T (turned + lo)
    Mi = Rm.T
    M_yx = np.array([[Mi[1, 1], Mi[1, 0]], [Mi[0, 1], Mi[0, 0]]])
    off_xy = Mi @ lo
    rot = affine_transform(img, M_yx, offset=(off_xy[1], off_xy[0]), output_shape=(H2, W2), order=1,
                           mode="constant", cval=np.nan)
    valid = np.isfinite(rot)
    rot = np.where(valid, rot, np.nanmean(img))
    mk = None
    if mask is not None:
        mk = affine_transform(np.asarray(mask, np.float32), M_yx, offset=(off_xy[1], off_xy[0]),
                              output_shape=(H2, W2), order=0, mode="constant", cval=0) > 0.5
        mk &= valid
    else:
        mk = valid
    if t > 1.0:
        rot = gaussian_filter1d(rot, 0.8 * np.sqrt(t * t - 1), axis=1, mode="nearest")
        cols = np.arange(0, W2, t)
        rot = map_coordinates(rot, np.meshgrid(np.arange(H2), cols, indexing="ij"), order=1, mode="nearest")
        mk = map_coordinates(mk.astype(np.float32), np.meshgrid(np.arange(H2), cols, indexing="ij"), order=0,
                             mode="constant") > 0.5
    A = np.diag([1.0 / t, 1.0]) @ np.hstack([Rm, -lo[:, None]])               # orig (x, y, 1) -> simulated (x, y)
    return rot.astype(np.float32), mk, A


def detect_affine(gray: np.ndarray, mask: np.ndarray | None = None, margin: int = 24, n_max: int = N_MAX,
                  tilts=TILTS) -> Features:
    """Features from the picture and from simulated tilted views of it (ASIFT), all in the original picture's
    pixels; ``Features.sim`` (added) says which simulation found each one. Matches views up to about 80 degrees
    apart, where plain SIFT stops near 40 (an orbit squeezes the surface across the view)."""
    img = _normalise(gray)
    mk = None
    if mask is not None:
        mk = np.asarray(mask, bool)
        if margin > 0:
            mk = binary_dilation(mk, iterations=int(margin))
    sims = simulations(tilts)
    parts = []
    for si, (t, phi) in enumerate(sims):
        if t == 1.0:
            f = detect(img, mk, margin=0, n_max=n_max)
        else:
            sim, smk, A = _simulate(img, t, phi, mk)
            f = detect(sim, smk if mk is not None else None, margin=0,
                       n_max=max(200, int(n_max / (2 * t))))
            if len(f):
                Ai = np.linalg.inv(np.vstack([A, [0, 0, 1]]))[:2]
                xy = f.xy.astype(np.float64) @ Ai[:, :2].T + Ai[:, 2]
                sc = f.scale * np.sqrt(t)
                f = Features(xy.astype(np.float32), sc.astype(np.float32), f.angle, f.strength, f.desc)
        if len(f):
            parts.append((f, np.full(len(f), si, np.int16)))
    if not parts:
        out = Features.empty()
        out.sim = np.zeros(0, np.int16)
        return out
    out = Features(np.concatenate([p[0].xy for p in parts]), np.concatenate([p[0].scale for p in parts]),
                   np.concatenate([p[0].angle for p in parts]), np.concatenate([p[0].strength for p in parts]),
                   np.concatenate([p[0].desc for p in parts]))
    out.sim = np.concatenate([p[1] for p in parts])
    return out


def extend_affine(gray: np.ndarray, plain: Features, mask: np.ndarray | None = None, margin: int = 24,
                  n_max: int = N_MAX, tilts=TILTS) -> Features:
    """``plain`` (this picture's plain features) followed by the features of its tilted simulations: the plain ones
    keep their indices (matches already made with them stay valid)."""
    img = _normalise(gray)
    mk = None
    if mask is not None:
        mk = np.asarray(mask, bool)
        if margin > 0:
            mk = binary_dilation(mk, iterations=int(margin))
    parts = [(plain, np.zeros(len(plain), np.int16))]
    for si, (t, phi) in enumerate(simulations(tilts)):
        if t == 1.0:
            continue
        sim, smk, A = _simulate(img, t, phi, mk)
        f = detect(sim, smk if mk is not None else None, margin=0, n_max=max(200, int(n_max / (2 * t))))
        if not len(f):
            continue
        Ai = np.linalg.inv(np.vstack([A, [0, 0, 1]]))[:2]
        xy = f.xy.astype(np.float64) @ Ai[:, :2].T + Ai[:, 2]
        parts.append((Features(xy.astype(np.float32), (f.scale * np.sqrt(t)).astype(np.float32), f.angle,
                               f.strength, f.desc), np.full(len(f), si, np.int16)))
    out = Features(np.concatenate([p[0].xy for p in parts]), np.concatenate([p[0].scale for p in parts]),
                   np.concatenate([p[0].angle for p in parts]), np.concatenate([p[0].strength for p in parts]),
                   np.concatenate([p[0].desc for p in parts]))
    out.sim = np.concatenate([p[1] for p in parts])
    return out


def match_affine(fa: Features, fb: Features, ratio: float = 0.8, same_px: float = 4.0, k: int = 8,
                 chunk: int = 1024) -> np.ndarray:
    """Mutual nearest neighbours between two pooled feature sets (several simulations each): the ratio test takes as
    the second neighbour the nearest one more than ``same_px`` from the first in the original picture (the same
    point found in two simulations is not a rival); duplicate matches (the same pair of places) are kept once."""
    if len(fa) < 2 or len(fb) < 2:
        return np.zeros((0, 2), int)

    def nn(X, xyY, Y):
        n = len(X)
        i1 = np.empty(n, int)
        r = np.empty(n)
        kk = min(k, len(Y))
        for s in range(0, n, chunk):
            sim = X[s:s + chunk] @ Y.T
            top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
            v = np.take_along_axis(sim, top, 1)
            o = np.argsort(-v, axis=1)
            top = np.take_along_axis(top, o, 1)
            v = np.take_along_axis(v, o, 1)
            d = np.sqrt(np.maximum(2 - 2 * v, 0))
            far = np.linalg.norm(xyY[top] - xyY[top[:, :1]], axis=2) > same_px
            far[:, 0] = False
            has = far.any(1)
            j2 = np.where(has, np.argmax(far, 1), kk - 1)
            d2 = np.where(has, d[np.arange(len(d)), j2], 2.0)
            i1[s:s + chunk] = top[:, 0]
            r[s:s + chunk] = d[:, 0] / np.maximum(d2, 1e-9)
        return i1, r
    ia, ra = nn(fa.desc, fb.xy, fb.desc)
    ib, rb = nn(fb.desc, fa.xy, fa.desc)
    a = np.arange(len(fa))
    # mutual up to duplicates: b's best match lands on a's point or a duplicate of it (same place)
    back = ib[ia]
    mutual = np.linalg.norm(fa.xy[back] - fa.xy[a], axis=1) <= same_px
    ok = (ra < ratio) & mutual
    m = np.stack([a[ok], ia[ok]], -1)
    if len(m) == 0:
        return m.astype(int)
    key = np.concatenate([np.rint(fa.xy[m[:, 0]] / 2.0), np.rint(fb.xy[m[:, 1]] / 2.0)], 1)
    order = np.argsort(ra[ok])
    _, first = np.unique(key[order], axis=0, return_index=True)
    return m[order[np.sort(first)]].astype(int)
