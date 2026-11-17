"""Night sky: lining frames up on the stars, and a stack that throws out planes, satellites and hot pixels
(BUILD_PLAN_TABS 5.5).

The ground does not move with the stars, so a night-sky burst is lined up on the stars alone: stars are found in
each frame (local peaks well above the sky's noise, round and not clipped, centred to a fraction of a pixel),
matched to the reference frame's stars (a phase correlation of the two star maps drawn as soft dots for the first
guess, then nearest neighbours within 3 px, a similarity fit, twice), and the frames are averaged with a two-pass
sigma clip: anything far from the other frames at that place (a plane's trail, a satellite, a cosmic-ray hit) is
left out, and a map of what was left out is kept to show. Planned, not built: Freeze ground (a separate stack for
the land), light-pollution removal, darks and flats from the camera profile, star-sharpness weighting, drizzle
for sky frames. No Qt here.
"""
from __future__ import annotations

import numpy as np

from ... import log as slog
from . import align as A

log = slog.get("engine.stack.astro")

REJECT_INFO = {
    "sigma": {"name": "Sigma clip (default)", "does": "Leaves out values far from the others at each place.",
              "best": "planes, satellites, hot pixels; 10 frames and up", "weak": "needs enough frames to know "
              "what is normal", "speed": "medium"},
    "winsorized": {"name": "Winsorized", "does": "Pulls far-off values in to the edge of normal instead of "
                   "dropping them.", "best": "fewer frames (5-10); keeps every frame's light",
                   "weak": "a bright trail is softened, not removed", "speed": "medium"},
}
PLANNED = ("Freeze ground (a separate stack for the land), light-pollution removal, darks and flats from your "
           "camera profile, star-sharpness weighting and drizzle for sky frames are planned.")


def _background(luma: np.ndarray, size: int = 31) -> np.ndarray:
    """The sky's smooth glow: a median over ``size`` px, computed on a 4x smaller copy for speed."""
    from scipy.ndimage import median_filter, zoom
    k = 4 if min(luma.shape) > 8 * size else 1
    small = A.bin_by(luma, k)
    bg = median_filter(small, max(3, size // k | 1), mode="nearest")
    if k == 1:
        return bg
    up = zoom(bg, (luma.shape[0] / bg.shape[0], luma.shape[1] / bg.shape[1]), order=1, mode="nearest",
              grid_mode=True)
    return up[:luma.shape[0], :luma.shape[1]]


def stars(luma: np.ndarray, max_n: int = 400, k: float = 5.0, clip: float = 0.98) -> np.ndarray:
    """(N x 3: y, x, flux) of the brightest round, unclipped stars: local maxima over ``k`` sigma of the sky
    background, centred to a fraction of a pixel by an intensity-weighted mean over 5 x 5."""
    from scipy.ndimage import maximum_filter
    L = np.asarray(luma, np.float32)
    r = L - _background(L)
    sig = A.noise_sigma(r) or float(np.std(r)) or 1e-6
    peak = (r == maximum_filter(r, 5, mode="nearest")) & (r > k * sig)
    peak[:3] = peak[-3:] = False
    peak[:, :3] = peak[:, -3:] = False
    ys, xs = np.nonzero(peak)
    if ys.size == 0:
        return np.zeros((0, 3))
    order = np.argsort(-r[ys, xs])[: 4 * max_n]
    ys, xs = ys[order], xs[order]
    out = []
    oy, ox = np.mgrid[-2:3, -2:3]
    for y, x in zip(ys, xs):
        if L[y, x] >= clip:
            continue                                  # clipped: its centre cannot be measured
        w = np.maximum(r[y - 2:y + 3, x - 2:x + 3], 0)
        s = float(w.sum())
        if s <= 0:
            continue
        cy, cx = float((w * oy).sum() / s), float((w * ox).sum() / s)
        vyy = float((w * (oy - cy) ** 2).sum() / s)
        vxx = float((w * (ox - cx) ** 2).sum() / s)
        vxy = float((w * (oy - cy) * (ox - cx)).sum() / s)
        tr, det = vyy + vxx, vyy * vxx - vxy * vxy
        disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
        l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
        if l1 / l2 > 3.0:
            continue                                  # elongated: a trail or two stars together
        out.append((y + cy, x + cx, s))
        if len(out) >= max_n:
            break
    return np.asarray(out, float).reshape(-1, 3)


def _dots(st: np.ndarray, shape, k: int = 2) -> np.ndarray:
    """Star list drawn as soft dots on a k-times smaller canvas (for the first, coarse match)."""
    from scipy.ndimage import gaussian_filter
    h, w = shape[0] // k, shape[1] // k
    img = np.zeros((h, w), np.float32)
    if st.shape[0]:
        y = np.clip(np.round(st[:, 0] / k).astype(int), 0, h - 1)
        x = np.clip(np.round(st[:, 1] / k).astype(int), 0, w - 1)
        np.add.at(img, (y, x), np.sqrt(np.maximum(st[:, 2], 0)).astype(np.float32))
    return gaussian_filter(img, 1.5)


def match(stars_ref: np.ndarray, stars_img: np.ndarray, shape, kind: str = "similarity") -> A.Transform:
    """Transform taking reference coordinates to the frame's, from two star lists (y, x, flux)."""
    from scipy.spatial import cKDTree
    if stars_ref.shape[0] < 3 or stars_img.shape[0] < 3:
        return A.Transform.identity(kind)
    k = 2
    dy, dx, _ = A.phase_shift(_dots(stars_img, shape, k), _dots(stars_ref, shape, k))
    T = A.Transform.translation(dy * k, dx * k)
    tree = cKDTree(stars_img[:, :2])
    for radius in (6.0, 3.0, 3.0):
        Y, X = T.apply(stars_ref[:, 0], stars_ref[:, 1])
        d, j = tree.query(np.stack([Y, X], 1), distance_upper_bound=radius)
        ok = np.isfinite(d)
        if ok.sum() < 3:
            break
        src = stars_ref[ok, :2]
        dst = stars_img[j[ok], :2]
        model = "affine" if (kind != "shift" and ok.sum() >= 30) else ("shift" if kind == "shift" else "similarity")
        w = np.sqrt(np.maximum(stars_ref[ok, 2], 1e-9))
        T = A.fit(src, dst - src, w, model)
    log.debug("stars matched: %d pairs, residual %.3f px", T.tiles_used, T.residual_px)
    return T
