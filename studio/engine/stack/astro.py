"""Night sky: lining frames up on the stars, and a stack that throws out planes, satellites and hot pixels
(BUILD_PLAN_TABS 5.5).

The ground does not move with the stars, so a night-sky burst is lined up on the stars alone: stars are found in
each frame (local peaks well above the sky's noise, round and not clipped, centred to a fraction of a pixel),
matched to the reference frame's stars (a phase correlation of the two star maps drawn as soft dots for the first
guess, then nearest neighbours within 3 px, a similarity fit, twice), and the frames are averaged with a two-pass
sigma clip: anything far from the other frames at that place (a plane's trail, a satellite, a cosmic-ray hit) is
left out, and a map of what was left out is kept to show. With the optional photutils installed, stars are found
by its DAOStarFinder (a fit of each star's profile); ``star_quality`` measures each frame's star width and
elongation, so the reference is the frame with the tightest stars and shaken or trailed frames are left out.
Planned, not built: Freeze ground (a separate stack for the land), light-pollution removal, darks and flats from
the camera profile, drizzle for sky frames. No Qt here.
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
           "camera profile and drizzle for sky frames are planned.")


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


def photutils_ok() -> bool:
    """True when the optional photutils (astropy's star package, BSD-3) is installed."""
    try:
        import photutils.detection  # noqa: F401
        return True
    except Exception:            # noqa: BLE001  (missing or broken optional package: the Studio's own finder)
        return False


def stars(luma: np.ndarray, max_n: int = 400, k: float = 5.0, clip: float = 0.98, method: str = "auto") -> np.ndarray:
    """(N x 3: y, x, flux) of the brightest round, unclipped stars. ``method``: "photutils" (DAOStarFinder: a
    Gaussian fit of each star's profile, the astronomers' standard, when installed), "own" (local maxima over
    ``k`` sigma of the sky, centred by an intensity-weighted mean over 5 x 5) or "auto" (photutils when present)."""
    if method == "photutils" or (method == "auto" and photutils_ok()):
        try:
            return _stars_dao(luma, max_n, k, clip)
        except Exception as e:   # noqa: BLE001  (a photutils fault falls back to the Studio's own finder, logged)
            if method == "photutils":
                raise
            log.info("photutils star finder failed (%s); using the Studio's own", e)
    return _stars_own(luma, max_n, k, clip)


def _stars_dao(luma: np.ndarray, max_n: int, k: float, clip: float) -> np.ndarray:
    from photutils.detection import DAOStarFinder
    L = np.asarray(luma, np.float32)
    r = L - _background(L)
    sig = A.noise_sigma(r) or float(np.std(r)) or 1e-6
    q = star_quality(L, _stars_own(L, 60, k, clip))
    fwhm = float(np.clip(q["fwhm"] if q["n"] >= 3 else 2.5, 1.2, 12.0))
    import inspect
    names = inspect.signature(DAOStarFinder.__init__).parameters
    kw = dict(threshold=k * sig, fwhm=fwhm, exclude_border=True)
    if "sharpness_range" in names:                 # photutils 3 names
        kw.update(sharpness_range=(0.2, 1.0), roundness_range=(-0.6, 0.6), n_brightest=4 * max_n)
    else:                                           # photutils 1-2 names
        kw.update(sharplo=0.2, sharphi=1.0, roundlo=-0.6, roundhi=0.6, brightest=4 * max_n)
    t = DAOStarFinder(**kw)(r)
    if t is None or len(t) == 0:
        return np.zeros((0, 3))
    cols = t.colnames
    xs = np.asarray(t["x_centroid" if "x_centroid" in cols else "xcentroid"], float)
    ys = np.asarray(t["y_centroid" if "y_centroid" in cols else "ycentroid"], float)
    flux = np.asarray(t["flux"], float)
    peak_col = "peak" if "peak" in cols else None
    yi = np.clip(np.round(ys).astype(int), 0, L.shape[0] - 1)
    xi = np.clip(np.round(xs).astype(int), 0, L.shape[1] - 1)
    ok = L[yi, xi] < clip                       # clipped stars: their centre cannot be measured
    if peak_col is None:
        ok &= np.isfinite(flux)
    order = np.argsort(-flux[ok])[:max_n]
    out = np.stack([ys[ok][order], xs[ok][order], np.maximum(flux[ok][order], 1e-9)], 1)
    return out.reshape(-1, 3)


def star_quality(luma: np.ndarray, st: np.ndarray | None = None, box: int = 11) -> dict:
    """{'n', 'fwhm' (px, median), 'elongation' (median long/short axis)} of a frame's stars: a blurred or shaken
    frame has wide stars, a trailed one long stars. Each star is enlarged 4x (cubic) before its second moments are
    taken over the part brighter than a tenth of its peak, so small, undersampled stars (one or two pixels wide,
    as in a binned copy) still measure round when they are round."""
    from scipy.ndimage import label, zoom
    L = np.asarray(luma, np.float32)
    if st is None:
        st = _stars_own(L, 150, max_ratio=None)
    r = L - _background(L)
    h = box // 2
    up = 4
    fw, el = [], []
    for y, x, _ in st[:150]:
        yi, xi = int(round(y)), int(round(x))
        if yi < h or xi < h or yi >= L.shape[0] - h or xi >= L.shape[1] - h:
            continue
        w = r[yi - h:yi + h + 1, xi - h:xi + h + 1]
        rim = np.concatenate([w[0], w[-1], w[1:-1, 0], w[1:-1, -1]])
        w = zoom(w - float(np.median(rim)), up, order=3, mode="nearest")
        pk = float(w.max())
        if pk <= 4.0 * max(float(np.std(rim)), 1e-9):
            continue
        lab, _ = label(w > 0.1 * pk)
        c = lab[w.shape[0] // 2, w.shape[1] // 2]
        if c == 0:
            continue
        w = np.where(lab == c, w, 0.0)                # this star only: a neighbour in the box would stretch it
        s = float(w.sum())
        oy, ox = np.mgrid[0:w.shape[0], 0:w.shape[1]].astype(np.float64) / up
        cy, cx = float((w * oy).sum() / s), float((w * ox).sum() / s)
        vyy = float((w * (oy - cy) ** 2).sum() / s)
        vxx = float((w * (ox - cx) ** 2).sum() / s)
        vxy = float((w * (oy - cy) * (ox - cx)).sum() / s)
        tr, det = vyy + vxx, vyy * vxx - vxy * vxy
        disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
        l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
        # a Gaussian cut at a tenth of its peak keeps 0.76 of its variance (1 - 2 ln10 / (10 - 1) ... in 2-D)
        fw.append(2.3548 * float(np.sqrt(np.sqrt(l1 * l2) / _CUT_VAR)))
        el.append(float(np.sqrt(l1 / l2)))
    if not fw:
        return {"n": 0, "fwhm": 0.0, "elongation": 0.0}
    return {"n": len(fw), "fwhm": float(np.median(fw)), "elongation": float(np.median(el))}


# Share of a 2-D Gaussian's per-axis variance kept when it is cut at a tenth of its peak:
# E[r^2; r < R] / (2 sigma^2) with R^2 = 2 sigma^2 ln 10, over the mass inside R, = 1 - (ln 10) / 9.
_CUT_VAR = 1.0 - np.log(10.0) / 9.0


def _stars_own(luma: np.ndarray, max_n: int = 400, k: float = 5.0, clip: float = 0.98,
               max_ratio: float | None = 3.0) -> np.ndarray:
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
        if max_ratio is not None and l1 / l2 > max_ratio:
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
