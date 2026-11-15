"""Stage 13: geometry. Lens distortion, perspective (keystone), straighten, crop, done as ONE resample.

Frames
  S  the camera-crop frame of the sensor, unrotated (what linear_rgb returns), full-resolution pixels.
  D  S turned upright by the camera's orientation flag (what the user sees). All user geometry (straighten,
     perspective, crop) is defined in D, so a sidecar's numbers mean what they look like on screen.
  canvas  D after the warps, same size as D. The crop is a rectangle on the canvas.

Inverse chain for every output pixel (how the resampler finds where to read):
  output -> canvas (+ crop origin) -> undo straighten and perspective (one 3x3 homography)
         -> apply the lens's distortion forward (where the lens actually put that point) -> D -> S.

Distortion models use lensfun's conventions (verified numerically against lensfunpy, docs/RESEARCH.md):
radius 1 = half the short side of a 3:2 frame with this frame's diagonal, times (camera crop / lens crop);
  ptlens: Rd = Ru (a Ru^3 + b Ru^2 + c Ru + 1 - a - b - c)
  poly3:  Rd = Ru (1 - k1 + k1 Ru^2)
  poly5:  Rd = Ru (1 + k1 Ru^2 + k2 Ru^4)
Profile values are frozen into the sidecar when they are first applied, so an edit renders the same forever,
whether or not lensfun (or a newer lensfun database) is installed.

"Constrain to image" zooms in just enough that no empty corners show after the warps (no upsampling: the
output keeps the source's pixel size; the visible area gets smaller instead).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from ... import log as slog

log = slog.get("engine.geometry")

SQRT_3_25 = math.sqrt(3.25)           # half-diagonal / half-short-side of a 3:2 frame
MAX_TILT_DEG = 25.0                   # perspective slider +-100 = this much camera tilt (normal lens)
MANUAL_K = 0.06                       # distortion slider +-100 = poly3 k of -+0.06


# ------------------------------------------------------------------------------------------ distortion
def radial_scale(model: dict | None, r: np.ndarray) -> np.ndarray:
    """Rd / Ru as a function of normalised undistorted radius Ru, for a frozen distortion model dict."""
    if not model:
        return np.ones_like(r)
    m, t = model.get("model"), [float(v) for v in model.get("terms", [])]
    if m == "ptlens" and len(t) >= 3:
        a, b, c = t[:3]
        return a * r ** 3 + b * r ** 2 + c * r + (1.0 - a - b - c)
    if m == "poly3" and len(t) >= 1:
        return 1.0 - t[0] + t[0] * r ** 2
    if m == "poly5" and len(t) >= 2:
        return 1.0 + t[0] * r ** 2 + t[1] * r ** 4
    raise ValueError(f"unknown distortion model {model!r}")


def manual_model(slider: float) -> dict | None:
    """Distortion slider (+ fixes barrel, - fixes pincushion, -100..100) as a poly3 model."""
    if abs(slider) < 1e-6:
        return None
    return {"model": "poly3", "terms": [-MANUAL_K * float(slider) / 100.0], "norm": 1.0, "source": "manual"}


# ------------------------------------------------------------------------------------------ the warp
@dataclass
class GeomParams:
    distortion: dict | None = None       # frozen model {"model", "terms", "norm", "source"}
    distortion_amount: float = 1.0       # fraction of the model's correction applied
    distortion_manual: float = 0.0       # slider, -100..100
    rotate: float = 0.0                  # degrees, + = counter-clockwise
    vertical: float = 0.0                # -100..100: + makes the top wider (fixes leaning-back buildings)
    horizontal: float = 0.0              # -100..100: + makes the right side taller
    crop: tuple | None = None            # (x0, y0, x1, y1), fractions of the constrained frame
    constrain: bool = True


def _orient_size(w, h, flip):
    return (h, w) if flip in (5, 6) else (w, h)


def d_to_s(xd, yd, sw, sh, flip):
    """D-frame coordinates -> S-frame coordinates (pixel centres at integers)."""
    if flip == 3:
        return sw - 1 - xd, sh - 1 - yd
    if flip == 5:                          # np.rot90(S, 1): D[i, j] = S[j, sw - 1 - i]
        return sw - 1 - yd, xd
    if flip == 6:                          # np.rot90(S, -1): D[i, j] = S[sh - 1 - j, i]
        return yd, sh - 1 - xd
    return xd, yd


def s_to_d(xs, ys, sw, sh, flip):
    if flip == 3:
        return sw - 1 - xs, sh - 1 - ys
    if flip == 5:
        return ys, sw - 1 - xs
    if flip == 6:
        return sh - 1 - ys, xs
    return xs, ys


class Warp:
    """Everything stage 13 needs for one image and one set of geometry parameters."""

    def __init__(self, src_w: int, src_h: int, flip: int, g: GeomParams):
        self.sw, self.sh, self.flip, self.g = int(src_w), int(src_h), int(flip), g
        self.wd, self.hd = _orient_size(self.sw, self.sh, self.flip)
        self.cx, self.cy = (self.wd - 1) / 2.0, (self.hd - 1) / 2.0
        diag = math.hypot(self.wd, self.hd)
        self.r_norm = diag / 2.0 / SQRT_3_25
        self.models = []
        if g.distortion and abs(g.distortion_amount) > 1e-6:
            self.models.append((g.distortion, float(g.distortion_amount)))
        mm = manual_model(g.distortion_manual)
        if mm:
            self.models.append((mm, 1.0))
        # homography: content is tilted by the perspective sliders, then rotated (centred coordinates)
        f = diag
        a = math.radians(MAX_TILT_DEG * float(np.clip(g.vertical, -100, 100)) / 100.0)
        b = math.radians(MAX_TILT_DEG * float(np.clip(g.horizontal, -100, 100)) / 100.0)
        rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
        K, Ki = np.diag([f, f, 1.0]), np.diag([1 / f, 1 / f, 1.0])
        H0 = K @ ry @ rx @ Ki
        p = H0 @ np.array([0.0, 0.0, 1.0])
        T = np.array([[1, 0, -p[0] / p[2]], [0, 1, -p[1] / p[2]], [0, 0, 1.0]])
        Hp = T @ H0
        Hp /= Hp[2, 2]
        th = math.radians(float(g.rotate))
        Rt = np.array([[math.cos(th), math.sin(th), 0], [-math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
        self.M = Rt @ Hp                                  # undistorted D (centred) -> canvas (centred)
        self.Minv = np.linalg.inv(self.M)
        self.warped = bool(self.models) or abs(g.rotate) > 1e-6 or abs(g.vertical) > 1e-6 or abs(g.horizontal) > 1e-6
        self.zoom = self._fit_zoom() if (g.constrain and self.warped) else 1.0
        bw, bh = self.wd / self.zoom, self.hd / self.zoom
        bx, by = (self.wd - bw) / 2.0, (self.hd - bh) / 2.0
        c = g.crop or (0.0, 0.0, 1.0, 1.0)
        x0, y0, x1, y1 = (float(np.clip(v, 0.0, 1.0)) for v in c)
        if x1 - x0 < 0.01 or y1 - y0 < 0.01:
            x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
        self.ox = int(round(bx + x0 * bw))
        self.oy = int(round(by + y0 * bh))
        self.out_w = max(1, int(round(bx + x1 * bw)) - self.ox)
        self.out_h = max(1, int(round(by + y1 * bh)) - self.oy)

    # -------------------------------------------------------------------------- properties
    @property
    def identity(self) -> bool:
        return not self.warped and self.ox == 0 and self.oy == 0 and self.out_w == self.wd and self.out_h == self.hd

    @property
    def crop_only(self) -> bool:
        return not self.warped and not self.identity

    @property
    def output_size(self) -> tuple[int, int]:
        return self.out_w, self.out_h

    # -------------------------------------------------------------------------- mapping
    def _distort(self, u, v):
        """Undistorted centred D coords -> distorted (where the lens put them)."""
        for model, amt in self.models:
            rn = self.r_norm * float(model.get("norm", 1.0))
            r = np.hypot(u, v) / rn
            s = radial_scale(model, r)
            if amt != 1.0:
                s = 1.0 + amt * (s - 1.0)
            u, v = u * s, v * s
        return u, v

    def canvas_to_d(self, X, Y):
        """Canvas coordinates (D frame, full res) -> source D coordinates."""
        xc, yc = np.asarray(X, np.float64) - self.cx, np.asarray(Y, np.float64) - self.cy
        Mi = self.Minv
        w = Mi[2, 0] * xc + Mi[2, 1] * yc + Mi[2, 2]
        u = (Mi[0, 0] * xc + Mi[0, 1] * yc + Mi[0, 2]) / w
        v = (Mi[1, 0] * xc + Mi[1, 1] * yc + Mi[1, 2]) / w
        u, v = self._distort(u, v)
        return u + self.cx, v + self.cy

    def output_to_s(self, xo, yo):
        xd, yd = self.canvas_to_d(np.asarray(xo, np.float64) + self.ox, np.asarray(yo, np.float64) + self.oy)
        return d_to_s(xd, yd, self.sw, self.sh, self.flip)

    def s_to_output(self, xs, ys):
        """S-frame pixels -> output pixels (the inverse of output_to_s; the lens distortion is undone by
        fixed-point iteration)."""
        xd, yd = s_to_d(np.asarray(xs, np.float64), np.asarray(ys, np.float64), self.sw, self.sh, self.flip)
        u, v = xd - self.cx, yd - self.cy
        uu, vv = u.copy(), v.copy()
        for _ in range(12):
            du, dv = self._distort(uu, vv)
            uu, vv = uu + (u - du), vv + (v - dv)
        M = self.M
        w = M[2, 0] * uu + M[2, 1] * vv + M[2, 2]
        X = (M[0, 0] * uu + M[0, 1] * vv + M[0, 2]) / w + self.cx - self.ox
        Y = (M[1, 0] * uu + M[1, 1] * vv + M[1, 2]) / w + self.cy - self.oy
        return X, Y

    def _inside(self, X, Y, margin=0.5):
        xd, yd = self.canvas_to_d(X, Y)
        return np.all((xd >= -margin) & (xd <= self.wd - 1 + margin) & (yd >= -margin) & (yd <= self.hd - 1 + margin))

    def _fit_zoom(self) -> float:
        t = np.linspace(0.0, 1.0, 97)

        def ok(z):
            bw, bh = self.wd / z, self.hd / z
            x0, y0 = self.cx - (bw - 1) / 2.0, self.cy - (bh - 1) / 2.0
            xs = np.concatenate([x0 + t * (bw - 1), x0 + t * (bw - 1), np.full_like(t, x0), np.full_like(t, x0 + bw - 1)])
            ys = np.concatenate([np.full_like(t, y0), np.full_like(t, y0 + bh - 1), y0 + t * (bh - 1), y0 + t * (bh - 1)])
            return self._inside(xs, ys)
        if ok(1.0):
            return 1.0
        lo, hi = 1.0, 8.0
        if not ok(hi):
            return hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        return hi

    # -------------------------------------------------------------------------- resampling
    def source_box(self, window=None, scale: float = 1.0, pad: int = 6):
        """S-frame box (x, y, w, h) covering the source pixels needed for an output window (full-res output
        pixels, default the whole output). Clamped to the frame, even-aligned."""
        x0, y0, w, h = window or (0, 0, self.out_w, self.out_h)
        n = 33
        xs = np.concatenate([np.linspace(x0, x0 + w - 1, n)] * 2 + [np.full(n, x0), np.full(n, x0 + w - 1)])
        ys = np.concatenate([np.full(n, y0), np.full(n, y0 + h - 1)] + [np.linspace(y0, y0 + h - 1, n)] * 2)
        gx, gy = np.meshgrid(np.linspace(x0, x0 + w - 1, 9), np.linspace(y0, y0 + h - 1, 9))
        xs, ys = np.concatenate([xs, gx.ravel()]), np.concatenate([ys, gy.ravel()])
        sx, sy = self.output_to_s(xs, ys)
        bx0 = int(max(0, math.floor(sx.min()) - pad)) // 2 * 2
        by0 = int(max(0, math.floor(sy.min()) - pad)) // 2 * 2
        bx1 = int(min(self.sw, math.ceil(sx.max()) + pad + 1))
        by1 = int(min(self.sh, math.ceil(sy.max()) + pad + 1))
        return bx0, by0, max(2, bx1 - bx0), max(2, by1 - by0)

    def resample(self, src: np.ndarray, src_box, src_scale: float, window=None, out_scale: float = 1.0,
                 order: int = 3, rows: int = 256) -> np.ndarray:
        """Resample ``src`` (H x W x C, an S-frame box at ``src_scale`` of full resolution) onto the output grid
        of ``window`` (full-res output pixels) at ``out_scale``."""
        x0, y0, w, h = window or (0, 0, self.out_w, self.out_h)
        ow, oh = max(1, int(round(w * out_scale))), max(1, int(round(h * out_scale)))
        out = np.empty((oh, ow, src.shape[2]), np.float32)
        bx, by = src_box[0], src_box[1]
        jj = (np.arange(ow) + 0.5) / out_scale - 0.5 + x0
        from ..parallel import strips

        def job(r0, r1):
            ii = (np.arange(r0, r1) + 0.5) / out_scale - 0.5 + y0
            X, Y = np.meshgrid(jj, ii)
            sx, sy = self.output_to_s(X, Y)
            # full-res S coords -> pixel coords of the (scaled) source box
            px = (sx - bx + 0.5) * src_scale - 0.5
            py = (sy - by + 0.5) * src_scale - 0.5
            ya, yb = int(max(0, math.floor(py.min()) - 10)), int(min(src.shape[0], math.ceil(py.max()) + 11))
            xa, xb = int(max(0, math.floor(px.min()) - 10)), int(min(src.shape[1], math.ceil(px.max()) + 11))
            if yb <= ya or xb <= xa:
                out[r0:r1] = 0.0
                return
            sub = src[ya:yb, xa:xb]
            coords = [py - ya, px - xa]
            for c in range(src.shape[2]):
                out[r0:r1, :, c] = map_coordinates(sub[..., c], coords, order=order, mode="nearest",
                                                   prefilter=order > 1)
        strips(oh, rows, job)
        return out


def crop_for_aspect(warp_w: float, warp_h: float, aspect: str) -> tuple | None:
    """Centred crop fractions for an aspect preset ('3:2', '4:5', '1:1', 'original', 'free'), given the size of
    the constrained frame. Portrait frames get the portrait version of the ratio."""
    if aspect in ("original", "free", "", None):
        return None
    try:
        a, b = (float(v) for v in aspect.split(":"))
    except ValueError:
        raise ValueError(f"aspect must look like '3:2', got {aspect!r}")
    r = max(a, b) / min(a, b)
    if warp_h > warp_w:
        r = 1.0 / r
    cur = warp_w / warp_h
    if cur > r:                     # too wide: trim the sides
        f = r / cur
        return ((1 - f) / 2, 0.0, (1 + f) / 2, 1.0)
    f = cur / r
    return (0.0, (1 - f) / 2, 1.0, (1 + f) / 2)
