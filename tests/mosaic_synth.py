"""Synthetic hand-held mosaic sets with known truth, for tests/test_mosaic*.py (BUILD_PLAN_MOSAIC.md section 6).

A ground texture of known size (cracked-asphalt-like: grain at several scales, dark crack lines, blotches), an
optional relief (bumps, a kerb) and an optional crown, rendered through cameras at known poses (height, tilt, roll,
turn), with lens distortion, a different exposure per view, vignetting, noise, an optional moving blob and a shadow.
Each view comes back as a scan.views.View (grey working image, colour, depth near = high, confidence), the way the
Studio's own views are made. No files are written.
"""
from __future__ import annotations

import math

import numpy as np

from studio.engine.stack import mosaic as MO

MM = 0.004                       # ground texture: metres per texel


class Ground:
    def __init__(self, x0=-1.5, x1=5.5, y0=-1.5, y1=1.5, seed=1, relief=None, crown=None):
        rng = np.random.default_rng(seed)
        from scipy.ndimage import gaussian_filter
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        W = int((x1 - x0) / MM)
        H = int((y1 - y0) / MM)
        g = np.zeros((H, W), np.float32)
        for s, a in ((1.0, 0.25), (3.0, 0.35), (9.0, 0.3), (30.0, 0.25)):
            g += a * gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), s)
        g = (g - g.mean()) / (g.std() + 1e-6)
        img = 0.35 + 0.08 * g
        # crack lines
        for _ in range(int(W * H / 4000)):
            x, y = rng.uniform(0, W), rng.uniform(0, H)
            ang = rng.uniform(0, math.pi)
            for t in range(int(rng.uniform(20, 120))):
                ang += rng.normal(0, 0.25)
                x += math.cos(ang) * 1.5
                y += math.sin(ang) * 1.5
                ix, iy = int(x), int(y)
                if 1 <= ix < W - 1 and 1 <= iy < H - 1:
                    img[iy - 1:iy + 1, ix - 1:ix + 1] *= 0.55
        # blotches
        for _ in range(40):
            cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(15, 80)
            yy, xx = np.ogrid[:H, :W]
            img *= 1.0 - 0.15 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)).astype(np.float32)
        self.tex = np.clip(img, 0.02, 0.98).astype(np.float32)
        self.tint = np.array([1.05, 1.0, 0.92], np.float32)
        self.relief = relief           # callable (x, y) -> height (m) above the surface, or None
        self.crown = crown             # (a, b, c) quadratic coefficients, or None

    def z(self, x, y):
        z = MO.surface_z(self.crown, x, y) if self.crown is not None else np.zeros_like(x)
        if self.relief is not None:
            z = z + self.relief(x, y)
        return z

    def sample(self, x, y):
        from scipy.ndimage import map_coordinates
        u = (x - self.x0) / MM - 0.5
        v = (y - self.y0) / MM - 0.5
        g = map_coordinates(self.tex, [v, u], order=1, mode="nearest")
        return g


def bumps(centres, h=0.03, r=0.08):
    def f(x, y):
        z = np.zeros_like(x)
        for cx, cy in centres:
            z = z + h * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r))
        return z
    return f


def kerb(y_edge=0.6, h=0.05):
    def f(x, y):
        return h / (1.0 + np.exp(-(y - y_edge) / 0.004))
    return f


def pose(x, y, height, tilt_deg=0.0, tilt_dir=0.0, yaw=0.0):
    """R (world -> camera), C for a camera at (x, y, height) looking down, turned ``yaw`` and tipped ``tilt_deg``."""
    Rz = MO.rodrigues(np.array([0, 0, yaw]))
    ax = np.array([math.cos(tilt_dir), math.sin(tilt_dir), 0.0])
    Rt = MO.rodrigues(ax * math.radians(tilt_deg))
    Rw = Rt @ Rz                                     # camera -> world rotation applied to the "down" camera
    R = MO.DOWN @ Rw.T
    return R, np.array([x, y, height], float)


def render(ground: Ground, R, C, size=(384, 512), f=560.0, k1=0.0, gain=1.0, vig=0.0, noise=0.004, seed=0,
           blob=None, shadow=None, iters=6):
    """One view: (colour float RGB 0..1, height at each pixel's ground point, ground xy at each pixel)."""
    h, w = size
    L = MO.Lens(f, (w - 1) / 2.0, (h - 1) / 2.0, math.hypot((w - 1) / 2.0, (h - 1) / 2.0))
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    px = np.stack([xs, ys], -1).reshape(-1, 2)
    d = MO.rays(px, R, L, k1)
    # ray against the height field: fixed-point on the height
    dz = d[:, 2]
    lam = (0.0 - C[2]) / dz
    for _ in range(iters):
        X = C + lam[:, None] * d
        zz = ground.z(X[:, 0], X[:, 1])
        lam = (zz - C[2]) / dz
    X = C + lam[:, None] * d
    g = ground.sample(X[:, 0], X[:, 1]).reshape(h, w)
    rgb = g[..., None] * ground.tint[None, None, :]
    if blob is not None:                             # something that moved: a bright disc on the ground
        bx, by, br = blob
        m = ((X[:, 0] - bx) ** 2 + (X[:, 1] - by) ** 2 < br * br).reshape(h, w)
        rgb[m] = np.array([0.9, 0.2, 0.2])
    if shadow is not None:                           # a shadow at the frame's edge (the photographer's)
        sx0, sx1 = shadow
        rgb[:, int(sx0 * w):int(sx1 * w)] *= 0.45
    rr = (((xs - L.cx) ** 2 + (ys - L.cy) ** 2) / (L.rn ** 2))
    rgb = rgb * gain * np.exp(-vig * rr)[..., None]
    rng = np.random.default_rng(seed)
    rgb = rgb + rng.normal(0, noise, rgb.shape)
    hgt = ground.relief(X[:, 0], X[:, 1]).reshape(h, w) if ground.relief is not None else np.zeros((h, w))
    return np.clip(rgb, 0, 1).astype(np.float32), hgt.astype(np.float32), X[:, :2].reshape(h, w, 2)


def to_srgb8(lin):
    a = np.clip(lin, 0, 1)
    s = np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255), 0, 255).astype(np.uint8)


def view_from(name, rgb, hgt, f, focus_m=None, depth=True, dp_scale=1.0, dp_offset=0.0, dp_ramp=(0.0, 0.0),
              dp_noise=0.0, seed=0, f_from="exif"):
    """A scan.views.View from a rendered view. The depth (near = high) is the true height, in arbitrary units per
    view (scale, offset and a ramp across the frame, as real Dual Pixel offsets are), with noise."""
    from studio.engine.scan.views import View
    h, w = rgb.shape[:2]
    col8 = to_srgb8(rgb)
    grey = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
    rng = np.random.default_rng(seed + 99)
    if depth:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = dp_scale * hgt + dp_offset + dp_ramp[0] * xx / w + dp_ramp[1] * yy / h
        t = t + rng.normal(0, dp_noise, t.shape)
        conf = np.ones((h, w), np.float32)
    else:
        t = np.zeros((h, w), np.float32)
        conf = np.zeros((h, w), np.float32)
    meta = {"model": "Canon EOS R5", "focal_mm": 50.0, "lens": "synthetic 50mm"}
    if focus_m is not None:
        meta["focus_distance_m"] = float(focus_m)
    v = View(name=name, image=grey, t=t.astype(np.float32), conf=conf, mask=None, f=float(f), colour=col8,
             path=name, meta=meta, f_from=f_from)
    return v


def strip(ground: Ground, n_per_row=7, rows=2, length=3.0, row_gap=0.55, h_lo=1.0, h_hi=1.6, tilt=8.0,
          size=(384, 512), f=560.0, k1=0.0, seed=3, depth=True, exposure=0.15, vig=0.0, blob=None,
          shadow_views=(), heights=None, serpentine=True, dp_noise=0.0, focus=True, focus_noise=0.0):
    """A hand-held set: rows walked back and forth, each view at a random height, tilt and turn. Returns
    (views, truth) with truth = {"R", "C", "f", "k1"}."""
    rng = np.random.default_rng(seed)
    views, Rs, Cs = [], [], []
    k = 0
    for r in range(rows):
        xs = np.linspace(0.0, length, n_per_row)
        if serpentine and r % 2:
            xs = xs[::-1]
        for x in xs:
            hh = heights[k] if heights is not None else rng.uniform(h_lo, h_hi)
            R, C = pose(x + rng.normal(0, 0.03), r * row_gap + rng.normal(0, 0.03), hh,
                        rng.uniform(0, tilt), rng.uniform(0, 2 * math.pi), rng.normal(0, 0.15))
            gain = math.exp(rng.normal(0, exposure))
            sh = (0.0, 0.18) if k in shadow_views else None
            rgb, hgt, _ = render(ground, R, C, size, f, k1, gain, vig, seed=k, blob=blob if k == 1 else None,
                                 shadow=sh)
            a = R[2, :]
            dist = (0 - C[2]) / a[2]
            fm = abs(dist) * (1 + rng.normal(0, focus_noise)) if focus else None
            dps = rng.uniform(5, 9)
            v = view_from(f"v{k:02d}", rgb, hgt, f, fm, depth, dp_scale=dps,
                          dp_offset=rng.normal(0, 2), dp_ramp=(rng.normal(0, 0.5), rng.normal(0, 0.5)),
                          dp_noise=dp_noise, seed=k)
            v.meta["_dp_scale"] = dps
            views.append(v)
            Rs.append(R)
            Cs.append(C)
            k += 1
    return views, {"R": Rs, "C": Cs, "f": f, "k1": k1}


def align_to_truth(sol: MO.Solution, truth: dict) -> dict:
    """The solved poses in the truth's frame (a similarity in the ground plane: shift, turn, scale) and the errors:
    centre error (m), angle error (degrees), scale factor (m per unit)."""
    idx = [i for i in range(sol.n) if sol.placed[i]]
    A = np.array([sol.C[i] for i in idx])
    B = np.array([truth["C"][i] for i in idx])
    # scale from heights (z is fixed up to scale), then a 2-D similarity for x, y
    s = float(np.median(B[:, 2] / A[:, 2]))
    a2 = A[:, :2] * s
    b2 = B[:, :2]
    ca, cb = a2.mean(0), b2.mean(0)
    U, _, Vt = np.linalg.svd((a2 - ca).T @ (b2 - cb))
    Rot = (U @ Vt).T
    if np.linalg.det(Rot) < 0:
        Vt[-1] *= -1
        Rot = (U @ Vt).T
    t = cb - ca @ Rot.T
    pe = []
    ae = []
    yaw = math.atan2(Rot[1, 0], Rot[0, 0])
    Rz = MO.rodrigues(np.array([0, 0, yaw]))
    for k, i in enumerate(idx):
        c = np.r_[a2[k] @ Rot.T + t, A[k, 2] * s]
        pe.append(float(np.linalg.norm(c - B[k])))
        Rs = sol.R[i] @ Rz.T                          # world(truth) -> camera
        d = MO.log_rotation(Rs @ truth["R"][i].T)
        ae.append(math.degrees(float(np.linalg.norm(d))))
    return {"scale": s, "pos_err_m": float(np.max(pe)), "pos_rms_m": float(np.sqrt(np.mean(np.square(pe)))),
            "ang_err_deg": float(np.max(ae))}
