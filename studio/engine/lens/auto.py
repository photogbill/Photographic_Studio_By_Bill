"""Automatic, profile-free lens fixes (section 10.2): the path for contactless vintage glass.

Lateral chromatic aberration (stage 8, on raw planes, before demosaic):
  Lateral CA is a slight difference in magnification between colours: red and blue land a little further from
  (or nearer to) the optical centre than green. It is measured from the photo itself: the frame is cut into
  tiles, each tile's red-to-green and blue-to-green sub-pixel shift is found by phase correlation on edges, the
  radial part of those shifts is fitted with d(r) = k1*r + k3*r^3, and the red and blue planes are resampled by
  that radial map. Needs no lens data at all. Gated: too few edges or an inconsistent fit -> nothing is done.

Purple / green fringing (stage 14, linear RGB after demosaic; on by default):
  Longitudinal CA paints purple (or green) halos on the bright side of high-contrast edges. Pixels are
  desaturated toward their neighbourhood's colour only when all three hold: they sit within a few pixels of a
  strong edge, next to bright, near-clipped light, and their hue is in the fringe band. A purple flower with soft
  edges fails the first two tests and keeps its colour (named test).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, sobel

from ... import log as slog

log = slog.get("engine.lens")


@dataclass
class LateralCA:
    red: tuple            # (k1, k3): radial displacement in plane pixels at normalised radius r = k1 r + k3 r^3
    blue: tuple
    tiles: int            # tiles that contributed
    confidence: str       # good | fair | none
    max_shift_px: float   # largest correction at the corners (full-resolution pixels)


def _planes(mos, pattern):
    p = np.asarray(pattern)
    h, w = (mos.shape[0] // 2) * 2, (mos.shape[1] // 2) * 2
    out = {"G": []}
    for i in (0, 1):
        for j in (0, 1):
            c = int(p[i, j])
            pl = mos[i:h:2, j:w:2]
            if c == 1:
                out["G"].append(pl)
            else:
                out["R" if c == 0 else "B"] = pl
    out["G"] = 0.5 * (out["G"][0] + out["G"][1])
    return out


from ..stack.align import phase_shift as _phase_shift   # noqa: E402  (one copy, in stack/align.py)


def estimate_lateral_ca(mos: np.ndarray, pattern, grid=(9, 13), tile=64) -> LateralCA:
    """Measure lateral CA from a white-balanced normalised mosaic (whole frame)."""
    pl = _planes(mos, pattern)
    G = pl["G"]
    H, W = G.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    rmax = np.hypot(cy, cx)
    if H < 3 * tile or W < 3 * tile:                     # too small to measure: leave it alone
        return LateralCA(red=(0.0, 0.0), blue=(0.0, 0.0), tiles=0, confidence="none", max_shift_px=0.0)
    rows = {"R": [], "B": []}
    used = 0
    for gy in np.linspace(tile, H - tile, grid[0]).astype(int):
        for gx in np.linspace(tile, W - tile, grid[1]).astype(int):
            y0, x0 = gy - tile // 2, gx - tile // 2
            g = G[y0:y0 + tile, x0:x0 + tile]
            e = np.hypot(sobel(g, 0), sobel(g, 1))
            if g.max() > 0.95 or e.mean() < 6 * max(float(np.median(np.abs(g - np.median(g)))) * 0.05, 1e-4):
                continue                                         # flat or clipped tiles say nothing
            r_vec = np.array([gy - cy, gx - cx])
            r = np.hypot(*r_vec)
            if r < 0.25 * rmax:
                continue                                         # CA is ~0 near the centre
            used += 1
            gh = g - gaussian_filter(g, 3)
            for c in ("R", "B"):
                p = pl[c][y0:y0 + tile, x0:x0 + tile]
                p = p * (np.mean(g) / max(np.mean(p), 1e-6))
                dy, dx, peak = _phase_shift(gh, p - gaussian_filter(p, 3))
                if peak < 0.15 or abs(dy) > 3 or abs(dx) > 3:
                    continue
                rows[c].append((r / rmax, dy, dx, r_vec[0] / r, r_vec[1] / r))
    res, sig = {}, {}
    for c in ("R", "B"):
        d0 = np.array(rows[c])
        if len(d0) < 12:
            res[c] = (0.0, 0.0)
            continue
        # remove the constant part first: the colour planes sit a quarter pixel from green on the Bayer grid,
        # and a decentred lens adds its own offset; only the radial, growing part is lateral CA
        dy = d0[:, 1] - np.median(d0[:, 1])
        dx = d0[:, 2] - np.median(d0[:, 2])
        # phase correlation gives green minus colour position; flip it so positive = colour sits further out
        d = np.stack([d0[:, 0], -(dy * d0[:, 3] + dx * d0[:, 4])], 1)
        # Linear magnification model d = k1 r (the dominant lateral-CA term; a cubic term fitted noise on clean
        # frames). Outliers are dropped, then k1 must be clearly non-zero: |k1| > 4 standard errors.
        r_, d_ = d[:, 0], d[:, 1]
        k1 = float((r_ @ d_) / (r_ @ r_))
        res_ = d_ - k1 * r_
        good = np.abs(res_) < 3 * (np.median(np.abs(res_)) * 1.4826 + 1e-3)
        r_, d_ = r_[good], d_[good]
        k1 = float((r_ @ d_) / max(r_ @ r_, 1e-9))
        se = float(np.sqrt(np.sum((d_ - k1 * r_) ** 2) / max(len(r_) - 1, 1)) / np.sqrt(max(r_ @ r_, 1e-9)))
        res[c] = (k1, 0.0) if abs(k1) > 4 * se else (0.0, 0.0)
        sig[c] = abs(k1) / max(se, 1e-9)
    n = min(len(rows["R"]), len(rows["B"]))
    corner = max(abs(sum(res["R"])), abs(sum(res["B"]))) * 2.0       # plane px at r=1 -> full-res px
    conf = "good" if n >= 30 else ("fair" if n >= 12 else "none")
    if corner < 0.15:                                                 # below a sixth of a pixel: not worth it
        conf = "none"
    log.debug("lateral CA: R %s B %s tiles %d corner %.2f px (%s)", np.round(res["R"], 4), np.round(res["B"], 4),
              used, corner, conf)
    return LateralCA(red=res["R"], blue=res["B"], tiles=used, confidence=conf, max_shift_px=corner)


def correct_lateral_ca(mos: np.ndarray, pattern, ca: LateralCA, full_shape=None, origin=(0, 0)) -> np.ndarray:
    """Resample the red and blue planes of a (region of a) mosaic by the measured radial map. ``full_shape`` and
    ``origin`` place the region inside the full visible frame (mosaic pixels) so the optical centre is right."""
    if ca is None or ca.confidence == "none":
        return mos
    p = np.asarray(pattern)
    out = mos.copy()
    fh, fw = full_shape if full_shape else mos.shape
    cy, cx = (fh / 2 - 1) / 2.0, (fw / 2 - 1) / 2.0                 # plane-coordinate centre
    rmax = np.hypot(cy, cx)
    for i in (0, 1):
        for j in (0, 1):
            c = int(p[i, j])
            if c == 1:
                continue
            k1, k3 = ca.red if c == 0 else ca.blue
            plane = mos[i::2, j::2]
            ph, pw = plane.shape
            yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
            Y = yy + origin[0] / 2.0 - cy
            X = xx + origin[1] / 2.0 - cx
            r = np.hypot(Y, X) / rmax
            d = (k1 * r + k3 * r ** 3)                                # how far out the colour landed
            s = np.where(r > 1e-6, d / np.maximum(r * rmax, 1e-6), 0.0)
            src_y, src_x = yy + Y * s, xx + X * s                    # sample where the colour actually is
            out[i::2, j::2] = map_coordinates(plane, [src_y, src_x], order=3, mode="nearest").astype(np.float32)
    return out


# ------------------------------------------------------------------------------------------ fringing
def defringe(rgb: np.ndarray, strength: float = 1.0, radius: float = 4.0) -> np.ndarray:
    """Remove purple (and green) fringes next to bright, high-contrast edges. Linear working RGB in and out."""
    if strength <= 0:
        return rgb
    Y = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
    Ys = gaussian_filter(Y, 1.0)
    edge = np.hypot(sobel(Ys, 0), sobel(Ys, 1)) / np.maximum(Ys + 0.02, 1e-3)
    size = int(2 * radius + 1)
    near_edge = maximum_filter(edge, size=size) > 1.2
    bright = maximum_filter(Y, size=size) > 0.6                     # fringes live next to near-clipped light
    s = np.maximum(rgb.sum(-1), 1e-6)
    r, g, b = rgb[..., 0] / s, rgb[..., 1] / s, rgb[..., 2] / s
    purple = np.clip((0.5 * (r + b) - g) * 6.0, 0, 1) * (b > g) * (r > 0.8 * g)
    green = np.clip((g - 0.5 * (r + b)) * 6.0, 0, 1) * 0.5
    w = (np.maximum(purple, green) * near_edge * bright * strength).astype(np.float32)
    w = gaussian_filter(w, 1.0)
    if not w.any():
        return rgb
    # target colour: the local average chroma of non-fringe neighbours, at this pixel's luminance
    keep = (1.0 - np.clip(w * 2, 0, 1))[..., None]
    nb = np.stack([gaussian_filter(rgb[..., c] * keep[..., 0], radius) for c in range(3)], -1)
    nk = gaussian_filter(keep[..., 0], radius)[..., None]
    nb = nb / np.maximum(nk, 1e-6)
    nY = np.maximum(0.3 * nb[..., 0] + 0.59 * nb[..., 1] + 0.11 * nb[..., 2], 1e-6)
    target = nb * (Y / nY)[..., None]
    target = np.where(nk > 0.05, target, Y[..., None])                # no clean neighbours: go neutral
    out = rgb * (1 - w[..., None]) + target * w[..., None]
    return out.astype(np.float32)
