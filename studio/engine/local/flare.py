"""Automatic lens-flare ghost removal (Bill, 2026-11-16: "add automatic lens flare removal, if it's possible").

Flare ghosts are the chain of soft discs, rings and polygons a bright light throws through a lens: reflections between
lens surfaces, worst in infrared, where the coatings were not made for the light. They have three properties this
module leans on:

* they **add** light (scene + flare, in linear light), smooth inside each ghost: the detail underneath is still there,
  only veiled, so subtracting a smooth estimate of the added light gives the scene back without inventing anything;
* they lie on **one line** through the optical centre, the line from the light (which may be outside the frame)
  through the middle of the picture;
* they are **soft blobs** a few percent of the frame across, brighter than their surroundings by a modest amount.

Detection, on a small copy of log brightness: bright blobs at several sizes (difference of Gaussians); the light
source (the largest clipped area) when it is in the frame; the axis through the centre that the most blob strength
lies along (or the line from the light through the centre); the blobs near that axis are the ghosts. Each is shown
amber (estimated) and can be switched off; the person can also add one by clicking.

Removal, per ghost: the background under the ghost is a membrane (Laplace's equation, like Heal) stretched from the
ghost's surroundings across it, on a smooth copy of the picture; the flare is what the smooth picture has above that
membrane, per colour; that smooth flare is subtracted from the photo (amount 0..100%). Where the flare clipped the
photo completely there is nothing left to recover: those places are reported for Retouch > Fill (which rebuilds them
from their surroundings, with LaMa when it is installed).

Honest limits: veiling glare over the whole frame is Dehaze's and Detail by scale's job; a ghost over a bright busy
area is found less reliably (switch it on by clicking); the membrane assumes the scene under a ghost is smooth at the
ghost's own size, so a ghost straddling a strong edge leaves a faint trace.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter, zoom

from ... import log as slog

log = slog.get("engine.flare")

PROXY_LONG = 900
LUMA = np.array([0.2880, 0.7119, 0.0001], np.float32)
VERSION = 1


def _proxy(rgb: np.ndarray, long_edge: int = PROXY_LONG) -> tuple:
    h, w = rgb.shape[:2]
    k = max(1, int(math.ceil(max(h, w) / float(long_edge))))
    if k > 1:
        hh, ww = h // k * k, w // k * k
        rgb = rgb[:hh, :ww].reshape(hh // k, k, ww // k, k, 3).mean(axis=(1, 3))
    return np.asarray(rgb, np.float32), k


def find_ghosts(rgb: np.ndarray, max_ghosts: int = 12) -> dict:
    """Ghosts in a (whole-frame) scene-linear picture. Returns {"ghosts": [{"x", "y", "r", "strength", "on"}] in
    fractions of the frame (r of the long edge), "axis" (degrees), "source" (x, y fractions or None), "text"}."""
    P, k = _proxy(rgb)
    H, W = P.shape[:2]
    L = np.log(np.maximum(P @ LUMA, 1e-4))
    long = float(max(H, W))
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    clip = (P.max(axis=2) >= 0.97)
    lab, n = label(clip)
    source = None
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        big = int(np.argmax(sizes)) + 1
        if sizes[big - 1] >= max(6, 1e-4 * H * W):
            ys, xs = np.nonzero(lab == big)
            source = (float(xs.mean()), float(ys.mean()))
    blobs = []
    for frac in (0.012, 0.02, 0.032, 0.05, 0.08):
        s = frac * long
        D = gaussian_filter(L, s) - gaussian_filter(L, 2.5 * s)
        spread = 1.4826 * float(np.median(np.abs(D - np.median(D))))
        thr = max(0.06, 5.0 * spread)                # a ghost stands out from the picture's own texture
        peaks = (D == maximum_filter(D, size=int(2 * s) | 1)) & (D > thr)
        m = int(math.ceil(1.5 * s))
        peaks[:m] = peaks[-m:] = False               # the frame edge makes false blobs
        peaks[:, :m] = peaks[:, -m:] = False
        ys, xs = np.nonzero(peaks)
        for y, x in zip(ys, xs):
            if clip[y, x] and source is not None and math.hypot(x - source[0], y - source[1]) < 3 * s:
                continue                             # the light itself, not a ghost
            blobs.append((float(x), float(y), _radius(L, x, y, 1.6 * s), float(D[y, x])))
    if not blobs:
        return {"ghosts": [], "axis": None, "source": _frac(source, W, H), "text": "No flare ghosts found."}
    # merge blobs found at several sizes (keep the strongest in each place)
    blobs.sort(key=lambda b: -b[3])
    kept = []
    for b in blobs:
        if all(math.hypot(b[0] - c[0], b[1] - c[1]) > 0.9 * max(b[2], c[2]) for c in kept):
            kept.append(b)
    # the axis: through the centre (and the light when it is in the frame)
    def dist_to_axis(b, ang):
        dx, dy = b[0] - cx, b[1] - cy
        return abs(-math.sin(ang) * dx + math.cos(ang) * dy)
    if source is not None and math.hypot(source[0] - cx, source[1] - cy) > 0.05 * long:
        ang = math.atan2(source[1] - cy, source[0] - cx)
        tol_extra = 0.03 * long
    else:
        best, ang = -1.0, 0.0
        for a in np.radians(np.arange(0, 180, 1.0)):
            sc = sum(b[3] * max(0.0, 1 - dist_to_axis(b, a) / (0.5 * b[2] + 0.02 * long)) for b in kept)
            if sc > best:
                best, ang = sc, float(a)
        tol_extra = 0.02 * long
    on_axis = [b for b in kept if dist_to_axis(b, ang) < 0.5 * b[2] + tol_extra]
    if len(on_axis) < (1 if source is not None else 2):
        return {"ghosts": [], "axis": math.degrees(ang), "source": _frac(source, W, H),
                "text": "Bright spots found, but none lined up like flare ghosts."}
    on_axis = sorted(on_axis, key=lambda b: -b[3])[:max_ghosts]
    ghosts = [{"x": round(b[0] / W, 5), "y": round(b[1] / H, 5), "r": round(b[2] / long, 5),
               "strength": round(b[3], 3), "on": True} for b in on_axis]
    txt = (f"{len(ghosts)} flare ghost{'s' if len(ghosts) != 1 else ''} found on one line through the centre"
           + (" from the light" if source is not None else "") + " (estimated: check them).")
    log.info("flare: %s", txt)
    return {"ghosts": ghosts, "axis": round(math.degrees(ang), 1), "source": _frac(source, W, H), "text": txt}


def _radius(L: np.ndarray, x: float, y: float, r0: float) -> float:
    """A ghost's radius from its edge: where the ring-averaged brightness falls fastest, between half and three times
    the detector's first guess."""
    H, W = L.shape
    R = int(min(3.2 * r0, max(H, W)))
    x0, x1 = int(max(0, x - R)), int(min(W, x + R + 1))
    y0, y1 = int(max(0, y - R)), int(min(H, y + R + 1))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    sub = gaussian_filter(L, 1.0)[y0:y1, x0:x1]
    rs = np.arange(max(2.0, 0.5 * r0), 3.0 * r0, max(0.5, 0.05 * r0))
    if rs.size < 4:
        return r0
    prof = np.array([sub[(d >= a) & (d < a + max(1.0, 0.1 * r0))].mean() if np.any((d >= a) & (d < a + max(1.0, 0.1 * r0)))
                     else np.nan for a in rs])
    if np.isnan(prof).sum() > len(prof) // 2:
        return r0
    prof = np.where(np.isnan(prof), np.nanmean(prof), prof)
    g = -np.gradient(gaussian_filter(prof, 1.0))
    return float(rs[int(np.argmax(g))] + 0.05 * r0)


def _frac(p, W, H):
    return None if p is None else (round(p[0] / W, 5), round(p[1] / H, 5))


def _membrane_fill(val: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Harmonic interpolation of ``val`` over ``inside`` from its surroundings (retouch.membrane's solver)."""
    from .retouch import membrane
    m = membrane(inside, np.asarray(val, np.float32)[..., None])[..., 0]
    return np.where(inside, m, val).astype(np.float32)


def _nconv(x: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    w = gaussian_filter(m.astype(np.float32), sigma)
    return gaussian_filter(np.where(m, x, 0).astype(np.float32), sigma) / np.maximum(w, 1e-6)


def flare_field(rgb: np.ndarray, ghosts: list) -> tuple:
    """The added light of the switched-on ghosts at proxy size: (field H' x W' x 3, proxy factor k, clipped mask).

    Per ghost: the background is a membrane from the ring well outside the ghost (its values smoothed along the
    ring only, so texture there does not streak inwards); the excess over it is the flare plus the scene's own
    texture; the ghost's support is where the broad excess stands above a third of its peak (so the ghost's own
    edge stays sharp); inside the support the excess is averaged (texture is zero-mean, flare is smooth), outside it
    the flare is zero."""
    P, k = _proxy(rgb)
    H, W = P.shape[:2]
    long = float(max(H, W))
    F = np.zeros_like(P)
    clipped = np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for g in ghosts or []:
        if not g.get("on", True):
            continue
        gx, gy, r = g["x"] * W, g["y"] * H, max(2.0, g["r"] * long)
        R = 1.6 * r
        x0, x1 = int(max(0, gx - 1.3 * R)), int(min(W, gx + 1.3 * R + 1))
        y0, y1 = int(max(0, gy - 1.3 * R)), int(min(H, gy + 1.3 * R + 1))
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue
        sub = P[y0:y1, x0:x1] - F[y0:y1, x0:x1]          # ghosts already measured are taken out first
        d = np.hypot(xx[y0:y1, x0:x1] - gx, yy[y0:y1, x0:x1] - gy)
        inside = d < R
        outside = ~inside
        E = np.zeros_like(sub)
        for c in range(3):
            ring = _nconv(sub[..., c], outside, max(1.0, 0.1 * r))
            bg = _membrane_fill(np.where(outside, ring, 0.0), inside)
            E[..., c] = sub[..., c] - bg
        El = E @ LUMA
        broad = gaussian_filter(El, max(1.0, 0.12 * r))
        peak = float(np.percentile(broad[d < r], 90)) if np.any(d < r) else 0.0
        if peak <= 0:
            continue
        sup = (broad > peak / 6.0) & inside
        from scipy.ndimage import binary_dilation
        sup = binary_dilation(sup, iterations=max(1, int(round(0.08 * r)))) & inside   # take the rim with it
        lab_s, nl = label(sup)
        if nl > 1:                                       # keep the part that holds the ghost's centre
            cyx = (int(min(max(gy - y0, 0), sup.shape[0] - 1)), int(min(max(gx - x0, 0), sup.shape[1] - 1)))
            keep = lab_s[cyx] if lab_s[cyx] else int(np.argmax(np.bincount(lab_s.ravel())[1:]) + 1)
            sup = lab_s == keep
        # the flare follows the ghost's own structure (real ghosts have rings and bright cores) at about a
        # twelfth of its size; texture finer than that is kept
        fl = np.zeros_like(sub)
        for c in range(3):
            fl[..., c] = np.maximum(_nconv(E[..., c], sup, max(1.0, 0.08 * r)), 0.0) * sup
        soft = gaussian_filter(sup.astype(np.float32), 0.8)
        F[y0:y1, x0:x1] += fl * np.clip(soft, 0, 1)[..., None]
        clipped[y0:y1, x0:x1] |= sup & (sub.max(axis=2) >= 0.97)
    return F, k, clipped


def apply(rgb: np.ndarray, F: np.ndarray, k: int, amount: float, origin=(0, 0), step: float = 1.0,
          full_shape=None) -> np.ndarray:
    """Subtract the (proxy-size) flare field from a region of the frame. ``origin`` (x, y) in full-frame pixels of
    the region's first pixel, ``step`` full pixels per region pixel, ``full_shape`` (h, w) of the full frame."""
    if amount <= 0 or not np.any(F):
        return rgb
    h, w = rgb.shape[:2]
    fh, fw = (full_shape or (F.shape[0] * k, F.shape[1] * k))[:2]
    sy = F.shape[0] / float(fh)
    sx = F.shape[1] / float(fw)
    from scipy.ndimage import map_coordinates
    ys = (origin[1] + (np.arange(h) + 0.5) * step) * sy - 0.5
    xs = (origin[0] + (np.arange(w) + 0.5) * step) * sx - 0.5
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    out = rgb.copy()
    a = np.float32(np.clip(amount, 0.0, 1.0))
    for c in range(3):
        f = map_coordinates(F[..., c], [Y, X], order=1, mode="nearest").astype(np.float32)
        out[..., c] = np.maximum(rgb[..., c] - a * f, np.minimum(rgb[..., c], 1e-4))
    return out


def report_clipped(clipped: np.ndarray) -> str:
    n = int(clipped.sum())
    if n == 0:
        return ""
    return (f"Part of a ghost is blown out ({n} px at preview size): nothing is left under it to recover. "
            "Retouch > Fill can rebuild it from its surroundings.")
