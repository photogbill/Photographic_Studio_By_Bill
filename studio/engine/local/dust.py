"""Sensor dust: find it in one photo, learn where it sits on the sensor from several photos (a dust map, private to
this PC), and find the map's spots again in any photo from that camera.

Dust on the sensor's cover glass casts a soft, round, darker shadow that sits in the same place in every photo
from that body. At small apertures (f/11 to f/22) the shadow is small and dark; with the lens open it spreads out
and fades (its blur grows roughly as 1 / f-number), which is why dust is a small-aperture problem.

Finding a spot is blob detection on log brightness: a difference of Gaussians over a pyramid of scales (the
scale-space blob detector, Lindeberg 1998), kept to round blobs (both curvatures positive and similar), to places
smooth enough that the blob stands well above the local texture, and away from deep shadow and clipped light.
One photo cannot tell a dark bird in a clear sky from dust, so single-photo results are shown for review; the map,
built from several different photos, is the dependable route: only blobs that recur in the same sensor position
across different scenes are kept.

The map lives in calibration_local/<camera>/dust_map.json and is never shared (plan rule 2).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter

from ... import log as slog

log = slog.get("engine.dust")

LUMA = np.array([0.2880, 0.7119, 0.0001], np.float32)
SIGMAS = (1.0, 1.5)                    # per pyramid level, in that level's pixels
R_PER_SIGMA = 4.0                      # spot radius / detection scale: a blob of Gaussian width b is found at
                                       # scale b / sqrt(3) and is visible out to ~2b, plus a margin
MAP_NAME = "dust_map.json"


# ------------------------------------------------------------------------------------------ detection
def _down2(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w]
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def find_blobs(Y: np.ndarray, step: int = 1, gx: float = 0.0, gy: float = 0.0, r_min: float = 4.0,
               r_max: float = 60.0, min_contrast: float = 0.08, k_texture: float = 6.0, limit: int = 80,
               min_bright: float = 0.03, max_contrast: float = 0.85) -> list:
    """Dark, round, soft blobs in a luminance image (scene-linear, 1 = clip). Pixel (0, 0) of ``Y`` is the cell at
    full-res S position (gx, gy), pixels ``step`` apart. Returns [{x, y, r, contrast, score}] in full-res S pixels,
    strongest first. ``min_contrast`` is in stops (0.08 = 5% darker), ``k_texture`` how far above the local texture
    and noise (measured in a ring around it) the blob must stand; ``min_bright`` keeps out deep shadow, where noise
    looks like specks and dust does not show anyway; a blob more than ``max_contrast`` stops darker is an object
    (dust is a shadow through a translucent speck, rarely more than 40% darker)."""
    lin = np.maximum(Y, 0.0).astype(np.float32)
    found = []
    lvl = 0
    while True:
        st = step * (2 ** lvl)
        L = None
        for s in SIGMAS:
            r_full = R_PER_SIGMA * s * st
            if r_full < r_min * 0.75 or r_full > r_max * 1.4:
                continue
            if L is None:     # smooth before the log: the log of noisy dark pixels has a dark tail (fake specks)
                L = np.log2(gaussian_filter(lin, 0.8, mode="nearest") + 0.01)
            found += _level(lin, L, s, st, gx, gy, min_contrast, k_texture, min_bright, max_contrast)
        if R_PER_SIGMA * SIGMAS[0] * step * (2 ** (lvl + 1)) > r_max * 1.4 or min(lin.shape) < 32:
            break
        lin = _down2(lin)
        lvl += 1
    found.sort(key=lambda b: -b["score"])
    kept = []
    for b in found:                                          # one blob, one spot (across scales)
        if all(math.hypot(b["x"] - k["x"], b["y"] - k["y"]) > 1.1 * max(b["r"], k["r"]) for k in kept):
            kept.append(b)
        if len(kept) >= limit:
            break
    for k in kept:          # its size: the scale where it is darkest (the score favours coarse scales)
        near = [b for b in found if math.hypot(b["x"] - k["x"], b["y"] - k["y"]) < 0.5 * max(b["r"], k["r"])]
        best = max(near, key=lambda b: b["contrast"]) if near else k
        k["r"], k["contrast"] = best["r"], max(k["contrast"], best["contrast"])
    return kept


def _integral(a: np.ndarray) -> np.ndarray:
    I = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.float64)
    np.cumsum(np.cumsum(a, axis=0, dtype=np.float64), axis=1, out=I[1:, 1:])
    return I


def _box_sum(I: np.ndarray, iy, ix, r: int):
    """Sums (and pixel counts) of boxes of radius r centred on (iy, ix), clipped to the image."""
    h, w = I.shape[0] - 1, I.shape[1] - 1
    y0, y1 = np.clip(iy - r, 0, h), np.clip(iy + r + 1, 0, h)
    x0, x1 = np.clip(ix - r, 0, w), np.clip(ix + r + 1, 0, w)
    return I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0], ((y1 - y0) * (x1 - x0)).astype(np.float64)


def _level(lin, L, s, st, gx, gy, min_contrast, k_texture, min_bright, max_contrast) -> list:
    G1 = gaussian_filter(L, s, mode="nearest")
    G2 = gaussian_filter(L, 3.0 * s, mode="nearest")
    dog = G1 - G2                                             # negative where darker than the surroundings
    size = max(3, int(2 * s + 1) | 1)
    cand = (dog <= minimum_filter(dog, size=size, mode="nearest")) & (-dog > min_contrast)
    iy, ix = np.nonzero(cand)
    if iy.size == 0:
        return []
    if iy.size > 20000:                                       # a busy photo: keep the strongest
        o = np.argsort(dog[iy, ix])[:20000]
        iy, ix = iy[o], ix[o]
    # texture and noise around each candidate: RMS response in a ring (outer box minus inner box), from integral
    # images so only the candidates cost anything
    I2 = _integral(dog * dog)
    so, ao = _box_sum(I2, iy, ix, int(6 * s))
    si, ai = _box_sum(I2, iy, ix, int(1.5 * s))
    tex = np.sqrt(np.maximum((so - si) / np.maximum(ao - ai, 1.0), 0.0))
    d = dog[iy, ix]
    sb, ab = _box_sum(_integral(lin), iy, ix, int(3 * s))
    bright = sb / ab
    keep = (-d > k_texture * tex) & (bright > min_bright) & (-d < max_contrast)
    iy, ix, d, tex = iy[keep], ix[keep], d[keep], tex[keep]
    if iy.size == 0:
        return []
    h, w = lin.shape
    rr = int(3 * s) + 1
    out = []
    off = 0.5 * (st - 1)
    for y, x, dv, t in zip(iy, ix, d, tex):
        y0, y1, x0, x1 = max(0, y - rr), min(h, y + rr + 1), max(0, x - rr), min(w, x + rr + 1)
        if lin[y0:y1, x0:x1].max() > 0.97:                   # next to clipped light: not dust
            continue
        g = G1[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        if g.shape != (3, 3):
            continue
        a = g[1, 2] + g[1, 0] - 2 * g[1, 1]                   # roundness: both curvatures positive and similar
        c = g[2, 1] + g[0, 1] - 2 * g[1, 1]
        bxy = 0.25 * (g[2, 2] + g[0, 0] - g[0, 2] - g[2, 0])
        tr, det = a + c, a * c - bxy * bxy
        disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
        l1, l2 = tr / 2 + disc, tr / 2 - disc
        if not (l2 > 0 and l2 > 0.3 * l1):
            continue
        out.append({"x": float(gx + st * x + off), "y": float(gy + st * y + off), "r": float(R_PER_SIGMA * s * st),
                    "contrast": float(-dv), "score": float(-dv / max(t, 0.004))})
    return out


def blobs_to_spots(blobs: list, sw: int, sh: int, auto: str = "dust") -> list:
    """Heal spots (no source yet) for found blobs."""
    long = float(max(sw, sh))
    return [{"kind": "heal", "pts": [[round(b["x"] / sw, 6), round(b["y"] / sh, 6)]],
             "size": round(max(1.4 * b["r"], b["r"] + 3.0) / long, 6),     # covers the shadow's faint edge
             "feather": 0.35, "opacity": 100, "src": [0.0, 0.0], "enabled": True, "auto": auto} for b in blobs]


# ------------------------------------------------------------------------------------------ the dust map
def map_path(make: str, model: str) -> str:
    from ..calib.profile import local_data_dir
    return os.path.join(local_data_dir(make, model), MAP_NAME)


def load_map(make: str, model: str) -> dict | None:
    p = map_path(make, model)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
        return m if isinstance(m, dict) and isinstance(m.get("spots"), list) else None
    except (OSError, ValueError):
        slog.log_exception(f"dust map {p} unreadable", log)
        return None


def save_map(make: str, model: str, m: dict) -> str:
    p = map_path(make, model)
    tmp = p + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)
    os.replace(tmp, p)
    return p


def cluster(per_photo: list, sw: int, sh: int, min_seen: int) -> list:
    """Blobs (per photo, S fractions + radius fraction) that recur in the same place in at least ``min_seen``
    different photos. Returns [{x, y, size, seen, contrast}]."""
    pts = [(i, b) for i, bl in enumerate(per_photo) for b in bl]
    if not pts:
        return []
    long = float(max(sw, sh))
    xy = np.array([[b["x"] * sw, b["y"] * sh] for _, b in pts])
    rr = np.array([b["size"] * long for _, b in pts])
    n = len(pts)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    for i, j in tree.query_pairs(r=float(max(rr.max(), 4.0))):
        if math.hypot(*(xy[i] - xy[j])) <= max(0.5 * (rr[i] + rr[j]), 0.001 * long):
            parent[find(i)] = find(j)
    groups: dict = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    out = []
    for ks in groups.values():
        seen = len({pts[k][0] for k in ks})
        if seen < min_seen:
            continue
        out.append({"x": round(float(np.median(xy[ks, 0])) / sw, 6), "y": round(float(np.median(xy[ks, 1])) / sh, 6),
                    "size": round(float(np.median(rr[ks])) / long, 6), "seen": seen,
                    "contrast": round(float(np.median([pts[k][1]["contrast"] for k in ks])), 4)})
    out.sort(key=lambda s: (-s["seen"], -s["contrast"]))
    return out


def match_map(m: dict, blobs: list, sw: int, sh: int, fnumber: float | None) -> list:
    """The map's spots that this photo shows: each map spot matched to a blob found near it (of a size that fits
    the aperture). Returns found blobs (full-res S px) in map order."""
    long = float(max(sw, sh))
    nm = m.get("fnumber")
    out = []
    used = set()
    for s in m.get("spots", []):
        x, y, r = s["x"] * sw, s["y"] * sh, s["size"] * long
        if nm and fnumber:
            r_pred = r * min(4.0, max(0.5, float(nm) / float(fnumber)))      # the shadow spreads as the lens opens
            lo, hi = r_pred / 2.2, r_pred * 2.2
        else:
            lo, hi = r / 3.0, r * 4.0
        best, bi = None, None
        for i, b in enumerate(blobs):
            if i in used or not (lo <= b["r"] <= hi):
                continue
            d = math.hypot(b["x"] - x, b["y"] - y)
            if d <= max(1.2 * r, 0.6 * b["r"], 0.002 * long) and (best is None or d < best):
                best, bi = d, i
        if bi is not None:
            used.add(bi)
            out.append(dict(blobs[bi], map_seen=s.get("seen", 0)))
    return out


def new_map(camera: str, spots: list, photos: int, fnumber: float | None) -> dict:
    return {"format": 1, "camera": camera, "made": _dt.date.today().isoformat(), "photos": int(photos),
            "fnumber": round(float(fnumber), 1) if fnumber else None, "spots": spots}
