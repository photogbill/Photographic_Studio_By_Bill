"""Synthetic views for the 3D Scan solver (BUILD_PLAN_DP3D I.2): a bumpy, textured log lying on textured ground,
rendered by a small NumPy ray caster from known camera positions, hand-held: unordered angles, some rolled, the
distance varying by up to 15 %.

Each view also gets what a Dual Pixel focus stack gives the Studio:

* a **fused depth** ``t`` in focus steps, an affine function of inverse distance with its own unknown scale and offset
  per view (t = a / z + b: Dual Pixel offsets are affine in 1/z, and a stack's step positions are affine in the
  offsets), slightly blurred (the measuring windows) and noisy, with its confidence (0 where nothing could be
  measured: the sky and the depth edges);
* a **subject mask** (the log, a little grown, as the measured-depth cut-out gives it).

Truth is kept for the tests: the camera poses (world to camera), the true depth and the surface.
World: z up, metres. Log: axis along x, radius ``R0`` with bumps, ends capped, lying on the ground z = 0.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates

R0 = 0.25            # log radius, m
HALF_L = 0.6         # half its length
BUMP = 0.05          # bump height as a share of the radius
GROUND_R = 4.0       # ground disc radius (the ground texture repeats every 8.5 m: never inside the disc)
LIGHT = np.array([0.45, -0.35, 0.82]) / np.linalg.norm([0.45, -0.35, 0.82])
SKY = 0.62


def _noise_tex(seed: int, n: int = 1024) -> np.ndarray:
    """A multi-scale blob texture in 0..1, periodic (so wrap-around sampling has no seam)."""
    rng = np.random.default_rng(seed)
    t = np.zeros((n, n), np.float32)
    for s, wgt in ((1.5, 0.9), (3.0, 1.0), (7.0, 0.8), (16.0, 0.5)):
        t += wgt * gaussian_filter(rng.standard_normal((n, n)).astype(np.float32), s, mode="wrap") * s ** 0.5
    t = (t - t.mean()) / (3.0 * t.std())
    t = np.tanh(2.0 * t)                       # some edges (corners for the detector)
    return gaussian_filter(((t + 1) / 2).astype(np.float32), 0.8, mode="wrap")


_TEX = {}


def tex(name: str) -> np.ndarray:
    if name not in _TEX:
        _TEX[name] = _noise_tex({"log": 11, "ground": 23, "cap": 37, "cap2": 41, "wall": 51}[name])
    return _TEX[name]


def _sample(t: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    n = t.shape[0]
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return map_coordinates(t, [np.mod(v, n), np.mod(u, n)], order=1, mode="grid-wrap")


def bump(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The log's bumps (-1..1), a smooth function of the angle round it and the position along it."""
    return (0.6 * np.sin(3 * theta + 2.0 * x) * np.cos(7.0 * x) + 0.4 * np.sin(5 * theta - 1.3) * np.sin(4.0 * x + 1))


def radius(theta, x):
    return R0 * (1.0 + BUMP * bump(theta, x))


def look_at(C, target, up=(0, 0, 1), roll_deg=0.0) -> np.ndarray:
    """World-to-camera rotation for a camera at C looking at target (camera x right, y down, z forward)."""
    z = np.asarray(target, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    a = np.radians(roll_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rz @ R


def render(R, C, f, w, h, scene="log", ss=2):
    """(image 0..1, depth z along the optical axis (inf = sky), hit id: 0 sky, 1 ground, 2 log, 3 cap, 4 wall,
    normals (h, w, 3), world points (h, w, 3)). Supersampled ``ss`` x ``ss``."""
    W, H = w * ss, h * ss
    fs = f * ss
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    d_cam = np.stack([(xs + 0.5) / ss - 0.5 - (w - 1) / 2.0, (ys + 0.5) / ss - 0.5 - (h - 1) / 2.0,
                      np.full_like(xs, f)], -1) / 1.0
    d = d_cam @ R                                   # world directions (R is world->camera, so R^T d_cam)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    o = np.asarray(C, float)
    best = np.full((H, W), np.inf)
    hid = np.zeros((H, W), np.int8)
    nrm = np.zeros((H, W, 3))
    alb = np.full((H, W), SKY)
    if scene in ("log", "ground"):
        # ground z = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            tg = -o[2] / d[..., 2]
        P = o + tg[..., None] * d
        ok = (tg > 0) & (P[..., 0] ** 2 + P[..., 1] ** 2 < GROUND_R ** 2)
        best = np.where(ok, tg, best)
        hid[ok] = 1
        nrm[ok] = (0, 0, 1)
        alb = np.where(ok, 0.2 + 0.6 * _sample(tex("ground"), P[..., 0] * 120.0, P[..., 1] * 120.0), alb)
    if scene == "log":
        # the log: (y)^2 + (z - R0)^2 = r(theta, x)^2, solved with the radius refined at the hit (3 rounds)
        dy, dz = d[..., 1], d[..., 2]
        oy, oz = o[1], o[2] - R0
        a = dy * dy + dz * dz
        b = 2 * (oy * dy + oz * dz)
        r = np.full((H, W), R0 * (1 + BUMP))
        tl = np.full((H, W), np.inf)
        for _ in range(4):
            c = oy * oy + oz * oz - r * r
            disc = b * b - 4 * a * c
            with np.errstate(invalid="ignore", divide="ignore"):
                tt = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
            P = o + tt[..., None] * d
            th = np.arctan2(P[..., 2] - R0, P[..., 1])
            r = radius(th, P[..., 0])
            tl = np.where((disc > 0) & (tt > 0), tt, np.inf)
        P = o + tl[..., None] * d
        ok = np.isfinite(tl) & (np.abs(P[..., 0]) <= HALF_L) & (tl < best)
        th = np.arctan2(P[..., 2] - R0, P[..., 1])
        best = np.where(ok, tl, best)
        hid[ok] = 2
        rad = np.stack([np.zeros_like(th), np.cos(th), np.sin(th)], -1)
        nrm[ok] = rad[ok]
        u = (th / (2 * np.pi)) * 1024.0
        v = (P[..., 0] + HALF_L) / (2 * HALF_L) * 1000.0      # under one period: the ends never look alike
        alb = np.where(ok, 0.15 + 0.7 * _sample(tex("log"), u, v), alb)
        # end caps x = +-HALF_L
        for sx in (-1.0, 1.0):
            with np.errstate(divide="ignore", invalid="ignore"):
                tc = (sx * HALF_L - o[0]) / d[..., 0]
            Pc = o + tc[..., None] * d
            rr = np.hypot(Pc[..., 1], Pc[..., 2] - R0)
            okc = (tc > 0) & (rr <= radius(np.arctan2(Pc[..., 2] - R0, Pc[..., 1]), Pc[..., 0])) & (tc < best)
            best = np.where(okc, tc, best)
            hid[okc] = 3
            nrm[okc] = (sx, 0, 0)
            alb = np.where(okc, 0.25 + 0.6 * _sample(tex("cap" if sx > 0 else "cap2"), Pc[..., 1] * 700.0,
                                                      Pc[..., 2] * 700.0), alb)
    if scene == "wall":                              # something else entirely: a textured wall x = 3
        with np.errstate(divide="ignore", invalid="ignore"):
            tw = (3.0 - o[0]) / d[..., 0]
        Pw = o + tw[..., None] * d
        ok = tw > 0
        best = np.where(ok, tw, best)
        hid[ok] = 4
        nrm[ok] = (-1, 0, 0)
        alb = np.where(ok, 0.2 + 0.6 * _sample(tex("wall"), Pw[..., 1] * 300.0, Pw[..., 2] * 300.0), alb)
    shade = np.where(hid > 0, 0.35 + 0.65 * np.clip(nrm @ LIGHT, 0, 1), 1.0)
    img = alb * shade
    P = o + np.where(np.isfinite(best), best, 0)[..., None] * d
    zc = (P - o) @ R[2]                              # depth along the optical axis
    zc = np.where(np.isfinite(best), zc, np.inf)

    def down(a):
        if ss == 1:
            return a
        sh = (h, ss, w, ss) + a.shape[2:]
        return a.reshape(sh).mean(axis=(1, 3))
    zf = np.where(np.isfinite(zc), zc, 0.0)
    fin = down(np.isfinite(zc).astype(float))
    zd = np.where(fin > 0.99, down(zf), np.inf)
    hid_d = hid.reshape(h, ss, w, ss)[:, ss // 2, :, ss // 2]
    return (down(img).astype(np.float32), zd.astype(np.float32), hid_d, down(nrm).astype(np.float32),
            down(np.where(np.isfinite(best)[..., None], P, np.nan)).astype(np.float32))


class SynthView:
    """What the solver gets (image, fused depth, confidence, mask, intrinsics) and the truth."""

    def __init__(self, name, image, t, conf, mask, f, R, C, z, hid, pts, a, b):
        self.name, self.image, self.t, self.conf, self.mask = name, image, t, conf, mask
        self.f = float(f)
        self.R_true, self.C_true = R, np.asarray(C, float)
        self.z_true, self.hid, self.pts_true = z, hid, pts
        self.a, self.b = a, b


def make_view(name, R, C, f=520.0, w=480, h=360, scene="log", rng=None, depth_noise=0.004, blur=1.2):
    rng = rng or np.random.default_rng(0)
    img, z, hid, nrm, pts = render(R, C, f, w, h, scene)
    img = np.clip(img + 0.004 * rng.standard_normal(img.shape), 0, 1).astype(np.float32)
    fin = np.isfinite(z)
    inv = np.where(fin, 1.0 / np.where(fin, z, 1.0), 0.0)
    known = fin.astype(np.float32)
    # the measuring windows blur the depth; depth edges and the sky measure nothing
    inv_b = gaussian_filter(inv * known, blur) / np.maximum(gaussian_filter(known, blur), 1e-6)
    gy, gx = np.gradient(inv)
    edge = np.hypot(gx, gy) > 0.02 * np.nanmax(inv[fin]) if fin.any() else np.zeros_like(fin)
    edge = binary_dilation(edge, iterations=2)
    conf = np.where(fin & ~edge, 1.0, 0.0).astype(np.float32)
    lo, hi = np.percentile(inv[fin], [2, 98]) if fin.any() else (0.3, 0.6)
    a = float(rng.uniform(0.6, 1.8) * 12.0 / max(hi - lo, 1e-6))   # about 12 steps over the view, own scale
    b = float(rng.uniform(-6, 6) - a * (lo + hi) / 2.0)
    t = (a * inv_b + b + depth_noise * 12.0 * rng.standard_normal(inv.shape)).astype(np.float32)
    t = np.where(fin, t, 0.0).astype(np.float32)
    mask = binary_dilation(np.isin(hid, (2, 3)), iterations=2)
    return SynthView(name, img, t, conf, mask, f, R, C, z, hid, pts, a, b)


def ring(n, step_deg=None, seed=0, start=20.0, elev=(18, 32), dist=1.9, jitter=3.0, roll=8.0, w=480, h=360,
         f=520.0, shuffle=True, scene="log", breathing=0.0):
    """n hand-held views round the log (step_deg apart, default 360 / n), returned in a shuffled order."""
    rng = np.random.default_rng(seed)
    step = 360.0 / n if step_deg is None else step_deg
    views = []
    for k in range(n):
        az = np.radians(start + k * step + rng.uniform(-jitter, jitter))
        el = np.radians(rng.uniform(*elev))
        d = dist * rng.uniform(0.87, 1.13)
        target = np.array([0, 0, R0 * 0.9]) + rng.uniform(-0.04, 0.04, 3)
        C = target + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        R = look_at(C, target, roll_deg=rng.uniform(-roll, roll))
        fk = f * (1.0 + rng.uniform(-breathing, breathing))        # focus breathing: each stack its own focal length
        views.append(make_view(f"V{k:02d}", R, C, fk, w, h, scene, rng))
    if shuffle:
        order = rng.permutation(n)
        views = [views[i] for i in order]
    return views


def stray(seed=5, w=480, h=360, f=520.0):
    """A view of something else (a wall far away): shares nothing with the log."""
    rng = np.random.default_rng(seed)
    C = np.array([-2.0, 0.3, 1.0])
    R = look_at(C, [3.0, 0.2, 1.0])
    return make_view("STRAY", R, C, f, w, h, "wall", rng)


def planar_pair(seed=7, w=480, h=360, f=520.0):
    """The ground alone, seen twice from above with a sideways step: a planar texture."""
    rng = np.random.default_rng(seed)
    out = []
    for k, C in enumerate(([0.0, -0.3, 1.6], [0.35, -0.2, 1.55])):
        C = np.asarray(C)
        R = look_at(C, [C[0] * 0.5 + 0.1, 0.15, 0.0], up=(0, 1, 0))
        out.append(make_view(f"P{k}", R, C, f, w, h, "ground", rng))
    return out


def true_surface(n=40000, seed=0):
    """Points on the log's surface (with the bumps) and their outward normals, for the mesh test."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, n)
    x = rng.uniform(-HALF_L, HALF_L, n)
    r = radius(th, x)
    P = np.stack([x, r * np.cos(th), R0 + r * np.sin(th)], -1)
    N = np.stack([np.zeros_like(th), np.cos(th), np.sin(th)], -1)
    return P, N


def to_view(sv, f_marked=None, f_sigma=0.02):
    """The solver's View for a synthetic view (truth stays on the synthetic one). ``f_marked``: the focal length the
    file would say (a macro lens focused close is longer than marked), with its uncertainty ``f_sigma``."""
    from studio.engine.scan.views import View
    return View(name=sv.name, image=sv.image, t=sv.t, conf=sv.conf, mask=sv.mask,
                f=sv.f if f_marked is None else f_marked, f_sigma=f_sigma, lens="synthetic macro",
                colour=np.dstack([np.clip(sv.image * 255, 0, 255).astype(np.uint8)] * 3))
