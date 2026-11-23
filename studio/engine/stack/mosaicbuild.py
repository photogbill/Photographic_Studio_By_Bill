"""Flat Mosaic, the picture and the height (BUILD_PLAN_MOSAIC.md sections 1.3, 2 and 3): solved views onto one
ground canvas.

    plan(views, sol, params)        everything decided on a small canvas before the full-size pictures are touched:
                                    exposure and vignetting from the overlaps, the views' heights matched to each
                                    other, the relief measured by parallax, seams, ghosts, misfit
    preview(views, sol, params)     a quick stitch from the plan (seconds)
    build(views, sol, params, ...)  the full mosaic, streamed tile by tile through scratch files

Which view wins where (plan 2.2): the nearer view (finer ground detail), then the sharper part of a view (hand-held
tilt puts part of a frame out of focus), then the part nearer a view's centre (less distortion, vignetting and relief
parallax). Seams are then cut where the views agree (seams.py), the overlaps blended across several bands
(pyramid.py), and anything that moved is taken from one view only.

Heights (plan 3): each view's measured depth (Dual Pixel, focus, or both fused: scan.views) is in its own units, with
its own offset and a ramp from the camera's tilt. The views are matched to each other in their overlaps (per view:
scale, offset and ramp; least squares, the reference view fixed), blended with the colour's own seams, and the fitted
ground (plane or gentle curve) taken off: what is left is the relief. The relief's size is then MEASURED from the
mosaic's own geometry: a place that stands above the ground shows up in two overlapping views at slightly different
ground positions, by an amount set by its height and the two cameras' positions (both known from the solve). Fitting
that parallax gives the relief in ground units (mm when the size is known): measured two ways, Dual Pixel shape and
geometric parallax, and the fit's agreement reported. It also says which way is near on this camera.

With the relief in ground units, every view is warped **orthorectified**: each place is lifted by its own height
before it is looked up in the view, so kerbs, potholes and crack walls no longer double at the seams. Where no
height is known, a **local nudge** (align.flow in the overlaps) hides small leftover parallax instead; it hides, it
does not measure, and the page says so. The **misfit map** shows where the views still disagree in position.

No Qt here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ... import log as slog
from . import mosaic as MO

log = slog.get("engine.stack.mosaicbuild")

PLAN_LONG = 2400                 # plan canvas: long edge in pixels
PREVIEW_LONG = 1600
MAX_CANVAS_MP = 400.0
RES_WEIGHT = 0.6                 # priority cost per halving of a view's ground detail: an octave coarser loses
                                 # even to another view's corner (nearer first, then sharper, then centre)
SHARP_WEIGHT = 0.3               # priority cost of a fully soft part of a view
RESOLUTIONS = {"game": 1.0, "fine": 0.5}   # mm per pixel
RELATIVE_FACTOR = {"game": 4.0, "fine": 2.0, "full": 1.0}
MIN_PARALLAX_PX = 0.35           # the relief's predicted parallax must reach this (plan pixels) to be measured
MIN_FIT = 0.3                    # share of the observed parallax the relief must explain to be trusted

RESOLUTION_INFO = {
    "game": {"name": "Game texture, 1 mm per pixel", "does": "Every view brought to 1 mm per pixel on the ground.",
             "best": "Textures for the game: sharp from the game's camera, a 10 m strip is 10,000 pixels.",
             "weak": "Throws away the finest grain a close view saw."},
    "fine": {"name": "Fine, 0.5 mm per pixel", "does": "0.5 mm per pixel on the ground.",
             "best": "Close-up surfaces, decals, anything the camera gets near.",
             "weak": "Four times the pixels of Game texture."},
    "full": {"name": "Full detail", "does": "The views' own detail (the median view's pixel size on the ground).",
             "best": "Keeping everything; making your own sizes later.",
             "weak": "Can be very large: the size is shown before it runs."},
    "custom": {"name": "Custom", "does": "The millimetres per pixel you type.", "best": "A size the game asks for.",
               "weak": "Needs the size known (a ruler or focus distances)."},
}


# ------------------------------------------------------------------------------------------- the canvas
@dataclass
class Canvas:
    res: float                       # ground units per pixel
    theta: float                     # the canvas's x axis in the ground (radians)
    X0: float                        # turned ground x of the left edge
    Y0: float                        # turned ground y of the top edge
    W: int
    H: int

    def to_ground(self, u, v):
        c, s = math.cos(self.theta), math.sin(self.theta)
        xr = self.X0 + (np.asarray(u, float) + 0.5) * self.res
        yr = self.Y0 - (np.asarray(v, float) + 0.5) * self.res
        return c * xr - s * yr, s * xr + c * yr

    def from_ground(self, x, y):
        c, s = math.cos(self.theta), math.sin(self.theta)
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        xr = c * x + s * y
        yr = -s * x + c * y
        return (xr - self.X0) / self.res - 0.5, (self.Y0 - yr) / self.res - 0.5

    def vec(self, gx, gy):
        """A ground displacement -> canvas pixels (du, dv)."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        return (c * gx + s * gy) / self.res, -(-s * gx + c * gy) / self.res

    def resized(self, k: float) -> "Canvas":
        return Canvas(self.res / k, self.theta, self.X0, self.Y0, max(1, int(math.ceil(self.W * k))),
                      max(1, int(math.ceil(self.H * k))))

    def to_dict(self) -> dict:
        return {"res": self.res, "theta": self.theta, "X0": self.X0, "Y0": self.Y0, "W": self.W, "H": self.H}


def full_scale(view) -> float:
    """Full-size colour pixels per working pixel."""
    if view.colour is None:
        return 1.0
    return view.colour.shape[1] / float(view.shape[1])


def native_gsd(sol: MO.Solution, views: list, i: int) -> float:
    """Ground units per full-size colour pixel at view i's centre."""
    return sol.gsd(i, full_scale(views[i]))


def resolution(sol: MO.Solution, views: list, params: MO.MosaicParams) -> tuple:
    """(ground units per pixel, mm per pixel or None, how)."""
    g = [native_gsd(sol, views, i) for i in range(sol.n) if sol.placed[i]]
    med = float(np.median(g))
    mode = params.resolution
    if sol.m_per_unit:
        mpu = sol.m_per_unit
        if mode == "full":
            mm = med * mpu * 1000.0
            return med, mm, f"the views' own detail: {mm:.3f} mm per pixel (the median view)"
        mm = float(params.mm_per_px) if mode == "custom" else RESOLUTIONS.get(mode, 1.0)
        if mm <= 0:
            raise MO.MosaicProblem("The resolution must be above 0 mm per pixel.")
        res = mm / 1000.0 / mpu
        return res, mm, f"{mm:g} mm per pixel" + ("" if mode != "game" else " (game texture)")
    if mode == "custom":
        raise MO.MosaicProblem("A size in millimetres needs the mosaic's scale: mark a ruler, or use a lens that "
                               "records its focus distance.")
    f = RELATIVE_FACTOR.get(mode, 1.0)
    return med * f, None, ("relative: " + ("the views' own detail" if f == 1 else
                                          f"{f:g} times coarser than the views' own detail") +
                           " (no size known, so no millimetres)")


def make_canvas(sol: MO.Solution, views: list, res: float, leave=()) -> Canvas:
    pts, cen = [], []
    for i in range(sol.n):
        if not sol.placed[i] or i in leave:
            continue
        fp = sol.footprint(i, views[i].shape, 16)
        pts.append(fp)
        cen.append(fp.mean(0))
    if not pts:
        raise MO.MosaicProblem("No view could be placed on the ground.")
    P = np.concatenate(pts)
    cen = np.array(cen)
    theta = 0.0
    if len(cen) >= 3:
        d = cen - cen.mean(0)
        ev, evec = np.linalg.eigh(d.T @ d)
        if ev[1] > 2.25 * max(ev[0], 1e-12):         # a strip: its length along the canvas's x axis
            v = evec[:, 1]
            theta = math.atan2(v[1], v[0])
            if math.cos(theta) < 0:                   # keep the reference view's own way up as near as possible
                theta += math.pi
            theta = math.atan2(math.sin(theta), math.cos(theta))
    c, s = math.cos(theta), math.sin(theta)
    xr = c * P[:, 0] + s * P[:, 1]
    yr = -s * P[:, 0] + c * P[:, 1]
    W = int(math.ceil((xr.max() - xr.min()) / res))
    H = int(math.ceil((yr.max() - yr.min()) / res))
    return Canvas(res, theta, float(xr.min()), float(yr.max()), max(W, 1), max(H, 1))


# ------------------------------------------------------------------------------------------- sources
def srgb_to_linear(c8: np.ndarray) -> np.ndarray:
    a = np.asarray(c8, np.float32) / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb8(lin: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(lin, np.float32), 0, 1)
    s = np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


def source(view, kb: int) -> np.ndarray:
    """The view's colour, linear, binned by ``kb`` (box average: no aliasing when shrunk)."""
    from .align import bin_by
    col = view.colour if view.colour is not None else np.repeat(np.asarray(view.image)[..., None] * 255, 3, 2)
    lin = srgb_to_linear(col)
    return bin_by(lin, kb) if kb > 1 else lin


def blur_map(view) -> np.ndarray:
    """0 sharp .. 1 soft, per working pixel: local detail against the view's own sharp parts, smoothed broadly so
    it follows focus falloff (hand-held tilt), not the texture."""
    from scipy.ndimage import gaussian_filter, laplace
    g = np.asarray(view.image, np.float32)
    e = gaussian_filter(laplace(gaussian_filter(g, 1.0)) ** 2, 2.0)
    s = max(4.0, 0.04 * max(g.shape))
    e = gaussian_filter(e, s)
    p90 = float(np.percentile(e, 90)) or 1e-12
    return np.clip(1.0 - e / p90, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------------------------------- heights
@dataclass
class HeightModel:
    """Per view: relief = alpha * (a t + b + c p + d q - S(x, y)): t the view's raw depth, (p, q) its own picture
    position (-0.5 .. 0.5 across the frame: a flat ground seen by a tilted camera gives a depth that is a plane in the
    picture, so each view's offset and ramp live there), S the fitted ground in the same units."""
    coef: dict                        # view -> (a, b, c, d)
    surf: tuple                       # S: (s0, sx, sy) or (s0, sx, sy, sxx, sxy, syy)
    alpha: float | None = None        # relief units -> ground units (None: not measured)
    flipped: bool = False             # the Dual Pixel near sign was the other way round (fixed)
    fit: float | None = None          # share of the observed parallax the relief explains
    max_parallax_px: float = 0.0
    how: str = ""
    colour: str = "violet"

    def S(self, x, y):
        s = self.surf
        z = s[0] + s[1] * x + s[2] * y
        if len(s) > 3:
            z = z + s[3] * x * x + s[4] * x * y + s[5] * y * y
        return z

    def value(self, i: int, t, x, y, p, q):
        a, b, c, d = self.coef[i]
        r = a * t + b + c * p + d * q - self.S(x, y)
        return r * (self.alpha if self.alpha else 1.0)

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "flipped": self.flipped, "fit": self.fit,
                "max_parallax_px": round(self.max_parallax_px, 3), "how": self.how, "colour": self.colour,
                "views": {str(k): [round(float(x), 6) for x in v] for k, v in self.coef.items()}}


def has_height(view) -> bool:
    c = getattr(view, "conf", None)
    return c is not None and float(np.mean(np.asarray(c) > 0.3)) > 0.05


# ------------------------------------------------------------------------------------------- warping
def warp_view(i: int, view, src: np.ndarray, sol: MO.Solution, cv: Canvas, order: int = 1, gain=1.0,
              vig=None, gsd_rel: float = 1.0, blur: np.ndarray | None = None, hfun=None, hmodel=None,
              nudge=None, dtype=np.float32, with_height: bool = False, rows: int = 256):
    """View i warped onto the canvas: a seams.Layer (rgb, valid, r = priority cost, clipped, rho2), with ``h`` and
    ``hc`` (height, confidence) when ``with_height``.

    ``src``: the view's linear colour at any scale (src pixels per working pixel from its width). ``hfun(u, v)``:
    the relief at canvas pixels in ground units (orthorectification). ``nudge``: (du, dv, box) displacement in
    canvas pixels over a box (local nudge). ``hmodel``: the HeightModel for the height values."""
    from scipy.ndimage import map_coordinates, spline_filter
    from .seams import Layer
    hw, ww = view.shape
    L = sol.lenses[i]
    sc = src.shape[1] / float(ww)
    fp = sol.footprint(i, view.shape, 24)
    u, v = cv.from_ground(fp[:, 0], fp[:, 1])
    pad = 3 + (0 if nudge is None else 8)
    x0 = max(0, int(math.floor(u.min())) - pad)
    x1 = min(cv.W, int(math.ceil(u.max())) + pad + 1)
    y0 = max(0, int(math.floor(v.min())) - pad)
    y1 = min(cv.H, int(math.ceil(v.max())) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    bh, bw = y1 - y0, x1 - x0
    out = np.zeros((bh, bw, 3), dtype)
    valid = np.zeros((bh, bw), bool)
    cost = np.zeros((bh, bw), np.float32)
    rho2 = np.zeros((bh, bw), np.float32)
    clipped = np.zeros((bh, bw), bool)
    hh = np.zeros((bh, bw), np.float32) if with_height else None
    hc = np.zeros((bh, bw), np.float32) if with_height else None
    g3 = np.broadcast_to(np.asarray(gain, np.float32), (3,)).astype(np.float32)
    s = np.asarray(src, np.float32)
    clip_src = (s.max(axis=2) >= 0.98).astype(np.float32)
    chans = [s[..., c] for c in range(3)]
    if order > 1:
        chans = [spline_filter(c, order, output=np.float32, mode="mirror") for c in chans]
    tau = getattr(view, "t", None)                    # the raw depth (see solve_heights)
    conf = getattr(view, "conf", None)
    cres = RES_WEIGHT * math.log2(max(gsd_rel, 1e-6))
    for r0 in range(0, bh, rows):
        r1 = min(bh, r0 + rows)
        V, U = np.meshgrid(np.arange(y0 + r0, y0 + r1, dtype=np.float64),
                           np.arange(x0, x1, dtype=np.float64), indexing="ij")
        if nudge is not None:
            du, dv, (ny0, ny1, nx0, nx1) = nudge
            sy = np.clip(V - ny0, 0, du.shape[0] - 1).astype(np.int64)
            sx = np.clip(U - nx0, 0, du.shape[1] - 1).astype(np.int64)
            inside = (V >= ny0) & (V < ny1) & (U >= nx0) & (U < nx1)
            U = U + np.where(inside, du[sy, sx], 0.0)
            V = V + np.where(inside, dv[sy, sx], 0.0)
        gx, gy = cv.to_ground(U, V)
        xy = np.stack([gx, gy], -1)
        h = hfun(U, V) if hfun is not None else None
        px, front = sol.to_view(i, xy, h)
        fx, fy = px[..., 0], px[..., 1]
        ok = front & (fx >= -0.5) & (fx <= ww - 0.5) & (fy >= -0.5) & (fy <= hw - 0.5)
        valid[r0:r1] = ok
        p2 = (((fx - L.cx) ** 2 + (fy - L.cy) ** 2) / (L.rn * L.rn)).astype(np.float32)
        rho2[r0:r1] = p2
        rr = ((fx - L.cx) / ww) ** 2 + ((fy - L.cy) / hw) ** 2
        cst = rr + cres
        cw = [np.clip(fy, 0, hw - 1).ravel(), np.clip(fx, 0, ww - 1).ravel()]
        if blur is not None:
            cst = cst + SHARP_WEIGHT * map_coordinates(blur, cw, order=1, mode="nearest").reshape(fx.shape)
        cost[r0:r1] = cst.astype(np.float32)
        corr = np.float32(1.0) if not vig else np.exp(-(vig[0] * p2 + vig[1] * p2 * p2)).astype(np.float32)
        sxs = (fx + 0.5) * sc - 0.5
        sys_ = (fy + 0.5) * sc - 0.5
        coords = [np.clip(sys_, 0, s.shape[0] - 1).ravel(), np.clip(sxs, 0, s.shape[1] - 1).ravel()]
        for c in range(3):
            sm = map_coordinates(chans[c], coords, order=order, mode="mirror" if order > 1 else "nearest",
                                 prefilter=False, output=np.float32).reshape(fx.shape)
            out[r0:r1, :, c] = np.where(ok, sm * corr * g3[c], 0).astype(dtype)
        cl = map_coordinates(clip_src, coords, order=0, mode="nearest").reshape(fx.shape)
        clipped[r0:r1] = (cl > 0.5) & ok
        if with_height and tau is not None and hmodel is not None and i in hmodel.coef:
            t = map_coordinates(np.asarray(tau, np.float32), cw, order=1, mode="nearest").reshape(fx.shape)
            cc = map_coordinates(np.asarray(conf, np.float32), cw, order=1, mode="nearest").reshape(fx.shape)
            pn, qn = (fx - L.cx) / ww, (fy - L.cy) / hw
            hh[r0:r1] = np.where(ok, hmodel.value(i, t, gx, gy, pn, qn), 0.0).astype(np.float32)
            hc[r0:r1] = np.where(ok, cc, 0.0).astype(np.float32)
    if not valid.any():
        return None
    Ly = Layer(i, (y0, y1, x0, x1), out, valid, cost, clipped)
    Ly.rho2 = rho2
    Ly.h = hh
    Ly.hc = hc
    return Ly


# ------------------------------------------------------------------------------------------- matching heights
def _pairs_of(layers: list):
    from .seams import _region
    for a in range(len(layers)):
        for b in range(a + 1, len(layers)):
            box = _region(layers[a], layers[b])
            if box is not None:
                yield layers[a], layers[b], box


def _smooth_valid(t: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur of ``t`` over its valid pixels only (normalised: nothing outside leaks in)."""
    from scipy.ndimage import gaussian_filter
    if sigma <= 0:
        return t
    v = valid.astype(np.float32)
    num = gaussian_filter(np.where(valid, t, 0).astype(np.float32), sigma)
    den = gaussian_filter(v, sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-6), t).astype(np.float32)


def solve_heights(layers: list, cv: Canvas, ref: int, surface: str, rng=None,
                  blur_px: float = 3.0) -> HeightModel | None:
    """Per view (a, b, c, d) so that a t + b + c p + d q agrees across every overlap. ``t`` is each view's raw depth
    (Dual Pixel offset or focus steps), (p, q) its own picture position (HeightModel).

    Two steps, so the scales cannot collapse and are not pulled small by noise (a plain least-squares fit of one
    noisy depth against another shrinks the answer):
      1. the scales ``a``: in each overlap both depths less a plane (the offsets and ramps), then the scales that
         make the two reliefs agree, found together for all views as the smallest generalised eigenvector
         (a symmetric fit: neither view is "the noisy one"), normalised to a median of 1;
      2. the offsets and ramps (b, c, d) by least squares with the scales fixed, the reference view's held at 0.
    A view whose overlaps hold no relief keeps the median scale."""
    from scipy.linalg import eigh
    from .seams import _crop
    rng = rng if rng is not None else np.random.default_rng(0)
    hl = [L for L in layers if getattr(L, "tau_l", None) is not None]
    if not hl:
        return None
    ids = [L.i for L in hl]
    if ref not in ids:
        ref = ids[0]
    ia = {v: t for t, v in enumerate(ids)}
    n = len(ids)
    samples = []
    Mm = np.zeros((n, n))
    D = np.zeros(n)
    for A, B, box in _pairs_of(hl):
        # smoothed a little (inside each view's valid area only): places a few pixels apart in two views (leftover
        # parallax) must not bias the scales
        va, vb = _crop(A, box, "valid"), _crop(B, box, "valid")
        ta = _smooth_valid(_crop(A, box, "tau_l"), va, blur_px)
        tb = _smooth_valid(_crop(B, box, "tau_l"), vb, blur_px)
        ca, cb = _crop(A, box, "conf_l"), _crop(B, box, "conf_l")
        ok = va & vb & (ca > 0.3) & (cb > 0.3)
        ys, xs = np.nonzero(ok)
        if len(ys) < 50:
            continue
        take = rng.choice(len(ys), min(1500, len(ys)), replace=False)
        ys, xs = ys[take], xs[take]
        w = np.sqrt(ca[ys, xs] * cb[ys, xs]) / math.sqrt(len(ys))
        tA, tB = ta[ys, xs], tb[ys, xs]
        PA_ = (_crop(A, box, "p_l")[ys, xs], _crop(A, box, "q_l")[ys, xs])
        PB_ = (_crop(B, box, "p_l")[ys, xs], _crop(B, box, "q_l")[ys, xs])
        rel = []
        for t, (pp, qq) in ((tA, PA_), (tB, PB_)):
            G = np.stack([np.ones_like(pp), pp, qq], 1)
            cf, *_ = np.linalg.lstsq(G * w[:, None], t * w, rcond=None)
            rel.append(w * (t - G @ cf))
        i, j = ia[A.i], ia[B.i]
        Mm[i, i] += rel[0] @ rel[0]
        Mm[j, j] += rel[1] @ rel[1]
        Mm[i, j] -= rel[0] @ rel[1]
        Mm[j, i] -= rel[0] @ rel[1]
        D[i] += rel[0] @ rel[0]
        D[j] += rel[1] @ rel[1]
        samples.append((A.i, B.i, tA, tB, PA_, PB_, w))
    if not samples:
        return HeightModel(coef={v: (1.0, 0.0, 0.0, 0.0) for v in ids}, surf=(0.0, 0.0, 0.0))
    has = D > 0.02 * max(float(np.median(D[D > 0])) if (D > 0).any() else 1.0, 1e-30)
    a = np.ones(n)
    if has.sum() >= 2:
        sub = np.nonzero(has)[0]
        Ms = Mm[np.ix_(sub, sub)]
        Ds = np.diag(D[sub])
        try:
            vals, vecs = eigh(Ms, Ds)
            v = vecs[:, 0]
            if np.median(v) < 0:
                v = -v
            v = v / max(float(np.median(v[v > 0])) if (v > 0).any() else 1.0, 1e-12)
            a[sub] = np.clip(v, 0.05, 20.0)
        except Exception as exc:                      # noqa: BLE001 - a fault: report, keep the lens's scale
            slog.save_error_report("Mosaic: matching the views' height scales", exc)
    # 2. offsets and ramps with the scales fixed
    others = [v for v in ids if v != ref]
    ib = {v: 3 * t for t, v in enumerate(others)}
    nq = 3 * len(others)
    coef = {}
    if nq:
        rows, rhs, wts = [], [], []
        for i_, j_, tA, tB, PA_, PB_, w in samples:
            R = np.zeros((len(tA), nq))
            b = -(a[ia[i_]] * tA - a[ia[j_]] * tB)
            for v, sgn, (pp, qq) in ((i_, 1.0, PA_), (j_, -1.0, PB_)):
                if v != ref:
                    o = ib[v]
                    R[:, o], R[:, o + 1], R[:, o + 2] = sgn, sgn * pp, sgn * qq
            rows.append(R)
            rhs.append(b)
            wts.append(w)
        Am = np.vstack(rows)
        bm = np.concatenate(rhs)
        wm = np.concatenate(wts)
        w = wm.copy()
        x = np.zeros(nq)
        reg = 1e-9 * np.eye(nq)
        for _ in range(3):
            x, *_ = np.linalg.lstsq(np.vstack([Am * w[:, None], reg]), np.r_[bm * w, np.zeros(nq)], rcond=None)
            r = Am @ x - bm
            sd = 1.4826 * float(np.median(np.abs(r))) + 1e-12
            w = wm * np.minimum(1.0, 2.0 * sd / np.maximum(np.abs(r), 1e-12))
        for v in others:
            o = ib[v]
            coef[v] = (float(a[ia[v]]), float(x[o]), float(x[o + 1]), float(x[o + 2]))
    coef[ref] = (float(a[ia[ref]]), 0.0, 0.0, 0.0)
    return HeightModel(coef=coef, surf=(0.0, 0.0, 0.0))


def fit_ground(hm: HeightModel, layers: list, labels: np.ndarray, cv: Canvas, surface: str, rng=None) -> None:
    """The ground in height units (plane, or plane + quadratic for a gentle curve), fitted to the blended heights."""
    rng = rng if rng is not None else np.random.default_rng(0)
    X, Y, Z, Wt = [], [], [], []
    for L in layers:
        if getattr(L, "tau_l", None) is None or L.i not in hm.coef:
            continue
        y0, y1, x0, x1 = L.box
        mine = (labels[y0:y1, x0:x1] == L.i) & L.valid & (L.conf_l > 0.3)
        ys, xs = np.nonzero(mine)
        if len(ys) < 20:
            continue
        take = rng.choice(len(ys), min(4000, len(ys)), replace=False)
        ys, xs = ys[take], xs[take]
        gx, gy = cv.to_ground(xs + x0, ys + y0)
        a, b, c, d = hm.coef[L.i]
        Z.append(a * L.tau_l[ys, xs] + b + c * L.p_l[ys, xs] + d * L.q_l[ys, xs])
        X.append(gx)
        Y.append(gy)
        Wt.append(L.conf_l[ys, xs])
    if not Z:
        return
    x, y, z, w = map(np.concatenate, (X, Y, Z, Wt))
    cols = [np.ones_like(x), x, y] + ([x * x, x * y, y * y] if surface == "curve" else [])
    A = np.stack(cols, 1)
    ww = np.sqrt(w)
    for _ in range(3):
        s, *_ = np.linalg.lstsq(A * ww[:, None], z * ww, rcond=None)
        r = z - A @ s
        sd = 1.4826 * float(np.median(np.abs(r))) + 1e-12
        ww = np.sqrt(w) * np.minimum(1.0, 2.5 * sd / np.maximum(np.abs(r), 1e-12))
    hm.surf = tuple(float(v) for v in s)


def relief_map(hm: HeightModel, layers: list, labels: np.ndarray, cv: Canvas) -> tuple:
    """The relief (height units, ground fitted off) and its confidence on the canvas, from each place's own view
    (the labels). Unmeasured places are filled from around them (confidence 0)."""
    from .pyramid import fill
    Hm = np.zeros((cv.H, cv.W), np.float32)
    Cm = np.zeros((cv.H, cv.W), np.float32)
    for L in layers:
        if getattr(L, "tau_l", None) is None or L.i not in hm.coef:
            continue
        y0, y1, x0, x1 = L.box
        mine = (labels[y0:y1, x0:x1] == L.i) & L.valid
        V, U = np.mgrid[y0:y1, x0:x1]
        gx, gy = cv.to_ground(U, V)
        a, b, c, d = hm.coef[L.i]
        r = a * L.tau_l + b + c * L.p_l + d * L.q_l - hm.S(gx, gy)
        Hm[y0:y1, x0:x1] = np.where(mine, r, Hm[y0:y1, x0:x1])
        Cm[y0:y1, x0:x1] = np.where(mine, L.conf_l, Cm[y0:y1, x0:x1])
    known = Cm > 0.3
    if known.any() and not known.all():
        Hm = fill(Hm, known)
    return Hm.astype(np.float32), Cm


# ------------------------------------------------------------------------------------------- parallax
def _lum(L, box):
    from .seams import _crop, lum
    return np.log(np.maximum(lum(_crop(L, box, "rgb").astype(np.float32)), 1e-4))


def pair_flows(layers: list, max_shift: float = 6.0) -> list:
    """For every overlapping pair: the local displacement of B against A in their shared box (align.flow), with
    where both are valid. [(A, B, box, Flow, both)]"""
    from .align import flow
    from .pyramid import fill
    from .seams import _crop
    out = []
    for A, B, box in _pairs_of(layers):
        both = _crop(A, box, "valid") & _crop(B, box, "valid")
        if both.sum() < 400 or min(box[1] - box[0], box[3] - box[2]) < 40:
            continue
        la = fill(_lum(A, box), both)
        lb = fill(_lum(B, box), both)
        tile = int(min(48, (min(la.shape) // 2) * 2))
        if tile < 16:
            continue
        try:
            F = flow(la, lb, tile=tile, step=max(8, tile // 2), max_shift=max_shift)
        except Exception as exc:                      # noqa: BLE001 - a fault: report it, go on without this pair
            slog.save_error_report("Mosaic: measuring the overlap", exc)
            continue
        out.append((A, B, box, F, both))
    return out


RELIEF_TILE = 32                 # plan pixels: the window the relief's parallax is measured in (small: bumps)
RELIEF_MAX_SHIFT = 12.0          # plan pixels: the largest parallax looked for


def relief_shifts(layers: list, tile: int = RELIEF_TILE) -> list:
    """For every overlapping pair, raw tile shifts of B against A (align.tile_shifts: phase correlation finds shifts
    of several pixels, Lucas-Kanade the sub-pixel value), with no smoothing across tiles, so a narrow bump keeps its
    full parallax.
    [(A, B, box, centres (N, 2) box pixels [y, x], shifts (N, 2) [dy, dx], quality (N,))]"""
    from .align import tile_shifts
    from .pyramid import fill
    from .seams import _crop
    out = []
    for A, B, box in _pairs_of(layers):
        both = _crop(A, box, "valid") & _crop(B, box, "valid")
        if both.sum() < 400 or min(box[1] - box[0], box[3] - box[2]) < 2 * tile:
            continue
        la = fill(_lum(A, box), both)
        lb = fill(_lum(B, box), both)
        try:
            cen, sh, q = tile_shifts(la, lb, tile=tile, step=tile // 2, max_shift=RELIEF_MAX_SHIFT)
        except Exception as exc:                      # noqa: BLE001 - a fault: report it, go on without this pair
            slog.save_error_report("Mosaic: measuring the relief's parallax", exc)
            continue
        if len(cen) == 0:
            continue
        iy = np.clip(np.rint(cen[:, 0]).astype(int), 0, both.shape[0] - 1)
        ix = np.clip(np.rint(cen[:, 1]).astype(int), 0, both.shape[1] - 1)
        keep = both[iy, ix]
        out.append((A, B, box, cen[keep], sh[keep], q[keep]))
    return out


def measure_relief(hm: HeightModel, shifts: list, Hrel: np.ndarray, sol: MO.Solution, cv: Canvas,
                   tile: int = RELIEF_TILE) -> None:
    """The relief's size from the parallax between overlapping views (module text). Sets hm.alpha (relief units ->
    ground units), hm.flipped, hm.fit, hm.how and hm.colour."""
    from scipy.ndimage import uniform_filter
    if sol.method != "poses":
        hm.how = "relative: plane warps know no camera positions, so the relief's size cannot be measured"
        return
    # a tile's shift is the average movement over the tile: compare it with the relief averaged the same way
    Hs = uniform_filter(np.asarray(Hrel, np.float32), tile, mode="nearest")
    obs, pred, wts = [], [], []
    for A, B, box, cen, sh, q in shifts:
        ok = q > 0.2
        if ok.sum() < 6:
            continue
        U = cen[ok, 1] + box[2]
        V = cen[ok, 0] + box[0]
        X, Y = cv.to_ground(U, V)
        h = Hs[np.clip(np.rint(V).astype(int), 0, cv.H - 1), np.clip(np.rint(U).astype(int), 0, cv.W - 1)]
        g = np.zeros((len(X), 2))
        for Lk, sgn in ((B, 1.0), (A, -1.0)):
            C = sol.C[Lk.i]
            Hk = C[2] - MO.surface_z(sol.q, X, Y)
            g[:, 0] += sgn * (X - C[0]) / Hk
            g[:, 1] += sgn * (Y - C[1]) / Hk
        du, dv = cv.vec(g[:, 0] * h, g[:, 1] * h)
        pred.append(np.stack([dv, du], 1))
        obs.append(sh[ok])
        wts.append(q[ok])
    if not obs:
        hm.how = "relative: the views share too little measured overlap to measure the relief's size"
        return
    o = np.concatenate(obs)
    p = np.concatenate(pred)
    w = np.concatenate(wts)
    # the relief's spread of parallax (centred: a constant offset is the solve's, not the relief's)
    pc = p - np.average(p, axis=0, weights=w)
    oc = o - np.average(o, axis=0, weights=w)
    den = float(np.sum(w[:, None] * pc * pc))
    if den <= 0:
        hm.how = "relative: the ground is too flat to see parallax between the views"
        return
    alpha = float(np.sum(w[:, None] * oc * pc)) / den
    for _ in range(3):                                # robust: points far from the fit lose weight
        r = np.linalg.norm(oc - alpha * pc, axis=1)
        s = 1.4826 * float(np.median(r)) + 1e-6
        ww = w * np.minimum(1.0, 2.5 * s / np.maximum(r, 1e-9))
        den = float(np.sum(ww[:, None] * pc * pc))
        if den <= 0:
            break
        alpha = float(np.sum(ww[:, None] * oc * pc)) / den
    reach = float(np.percentile(np.linalg.norm(alpha * pc, axis=1), 99))
    big = np.linalg.norm(alpha * pc, axis=1) > 0.25 * max(reach, 1e-9)   # where the relief shows: judge the fit
    tot = float(np.sum((w[:, None] * oc * oc)[big])) if big.any() else 0.0
    fit = 1.0 - float(np.sum((w[:, None] * (oc - alpha * pc) ** 2)[big])) / max(tot, 1e-12) if big.any() else 0.0
    hm.max_parallax_px = reach
    hm.fit = round(fit, 3)
    if reach < MIN_PARALLAX_PX:
        hm.how = (f"relative: the relief shows under {MIN_PARALLAX_PX:g} px of parallax between the views (too "
                  "flat, or the views too far apart in height, to measure its size)")
        return
    if fit < MIN_FIT:
        hm.how = (f"relative: the parallax between the views does not follow the measured relief (it explains "
                  f"{100 * max(fit, 0):.0f}%): something moved, or the depth is too noisy here")
        return
    hm.flipped = alpha < 0
    hm.alpha = abs(alpha)
    hm.colour = "cyan" if fit >= 0.6 else "amber"
    hm.how = (f"measured two ways: the views' depth gave the shape, the parallax between overlapping views its size "
              f"(the fit explains {100 * fit:.0f}% of the parallax where the relief shows, up to {reach:.1f} px)")
    if hm.flipped:
        hm.how += ("; the parallax says near and far were the other way round in the depth, so they were swapped "
                   "(this camera's Dual Pixel reading is now known)")
        hm.coef = {k: (-a, -b, -c, -d) for k, (a, b, c, d) in hm.coef.items()}
        hm.surf = tuple(-v for v in hm.surf)


# ------------------------------------------------------------------------------------------- local nudge
def nudge_fields(flows: list, layers: list, max_shift: float = 6.0) -> dict:
    """For each pair, the view that would lose there (higher priority cost) is nudged onto the other inside the
    overlap; the nudge fades out over the overlap's width beyond it. {view: (du, dv, box)} in canvas pixels."""
    from scipy.ndimage import distance_transform_edt
    from .pyramid import fill
    acc = {}
    for A, B, box, F, both in flows:
        from .seams import _crop
        ca = float(np.mean(_crop(A, box, "r")[both]))
        cb = float(np.mean(_crop(B, box, "r")[both]))
        dy, dx, q = F.fields(both.shape)
        if cb >= ca:                                   # B yields: sample B at p + flow
            Y, d_y, d_x = B, dy, dx
        else:                                          # A yields: sample A at p - flow (the flow's inverse, small)
            Y, d_y, d_x = A, -dy, -dx
        y0, y1, x0, x1 = Y.box
        if Y.i not in acc:
            acc[Y.i] = [np.zeros((y1 - y0, x1 - x0), np.float32), np.zeros((y1 - y0, x1 - x0), np.float32),
                        np.zeros((y1 - y0, x1 - x0), np.float32), Y]
        S = acc[Y.i]
        oy, ox = box[0] - y0, box[2] - x0
        sl = (slice(oy, oy + both.shape[0]), slice(ox, ox + both.shape[1]))
        w = both.astype(np.float32)
        S[0][sl] += np.clip(d_x, -max_shift, max_shift) * w
        S[1][sl] += np.clip(d_y, -max_shift, max_shift) * w
        S[2][sl] += w
    out = {}
    for i, (su, sv, sw, Y) in acc.items():
        known = sw > 0
        if not known.any():
            continue
        du = np.where(known, su / np.maximum(sw, 1e-6), 0.0).astype(np.float32)
        dv = np.where(known, sv / np.maximum(sw, 1e-6), 0.0).astype(np.float32)
        du, dv = fill(du, known), fill(dv, known)
        dist = distance_transform_edt(~known)
        width = max(4.0, float(np.sqrt(known.sum()) * 0.5))
        fade = np.exp(-dist / width).astype(np.float32)
        out[i] = (du * fade, dv * fade, Y.box)
    return out


def misfit_map(flows: list, cv: Canvas) -> np.ndarray:
    """How far the views still disagree in position (canvas pixels), the largest over the pairs at each place."""
    M = np.zeros((cv.H, cv.W), np.float32)
    for A, B, box, F, both in flows:
        dy, dx, q = F.fields(both.shape)
        m = np.where(both, np.hypot(dy, dx), 0.0).astype(np.float32)
        y0, y1, x0, x1 = box
        M[y0:y1, x0:x1] = np.maximum(M[y0:y1, x0:x1], m)
    return M


def height_agreement(layers: list, cv: Canvas) -> np.ndarray:
    """|difference| of two views' heights where both measured it (NaN where not compared)."""
    from .seams import _crop
    G = np.full((cv.H, cv.W), np.nan, np.float32)
    for A, B, box in _pairs_of(layers):
        if A.h is None or B.h is None:
            continue
        both = _crop(A, box, "valid") & _crop(B, box, "valid") & (_crop(A, box, "hc") > 0.3) & \
            (_crop(B, box, "hc") > 0.3)
        if not both.any():
            continue
        d = np.abs(_crop(A, box, "h") - _crop(B, box, "h"))
        y0, y1, x0, x1 = box
        sub = G[y0:y1, x0:x1]
        sub[both] = np.fmax(sub[both], d[both])
    return G


# ------------------------------------------------------------------------------------------- the plan
def plan(views: list, sol: MO.Solution, params: MO.MosaicParams, progress=None, cancel=None,
         long_edge: int = PLAN_LONG) -> dict:
    from . import pano as PA
    from . import seams as SE
    from .focus import Cancelled
    say = progress or (lambda *_: None)

    def check():
        if cancel is not None and cancel.is_set():
            raise Cancelled()
    leave = {i for i, v in enumerate(views) if v.name in set(params.leave_out or [])}
    res_out, mm, res_how = resolution(sol, views, params)
    cv_out = make_canvas(sol, views, res_out, leave)
    k = min(1.0, long_edge / float(max(cv_out.W, cv_out.H)))
    cv = cv_out.resized(k)
    use = [i for i in range(sol.n) if sol.placed[i] and i not in leave]
    gsd = {i: native_gsd(sol, views, i) for i in use}
    gmin = min(gsd.values())
    blurs = {}
    for t, i in enumerate(use):
        check()
        blurs[i] = blur_map(views[i])
    want_h = bool(params.height) and any(has_height(views[i]) for i in use)

    def layers_with(hfun=None, nudges=None, gains=None, vig=None, msg="Placing"):
        out = []
        for t, i in enumerate(use):
            check()
            say(0.1 + 0.25 * t / max(len(use), 1), f"{msg} view {t + 1} of {len(use)}")
            kb = max(1, int(cv.res / max(gsd[i], 1e-30) / 1.5))
            src = source(views[i], kb)
            L = warp_view(i, views[i], src, sol, cv, order=1, gain=1.0 if gains is None else gains[i], vig=vig,
                          gsd_rel=gsd[i] / gmin, blur=blurs[i], hfun=hfun,
                          nudge=(nudges or {}).get(i))
            if L is None:
                continue
            if want_h and has_height(views[i]):
                L.tau_l, L.conf_l, L.p_l, L.q_l = _warp_scalar(i, views[i], sol, cv, L, hfun,
                                                                (nudges or {}).get(i))
            else:
                L.tau_l = L.conf_l = None
            out.append(L)
        return out

    layers = layers_with()
    if not layers:
        raise MO.MosaicProblem("No view could be placed on the ground.")
    check()
    say(0.38, "Evening out exposure and vignetting")
    if params.exposure or params.vignette:
        ph = PA.photometric(layers, sol.n, vignette=bool(params.vignette))
        if not params.exposure:
            ph["gains"] = np.ones((sol.n, 3))
        if not params.vignette:
            ph["vig"], ph["corner_ev"] = None, 0.0
    else:
        ph = {"gains": np.ones((sol.n, 3)), "vig": None, "corner_ev": 0.0}

    def apply_ph(lys):
        for L in lys:
            corr = 1.0
            if ph["vig"]:
                c1, c2 = ph["vig"]
                corr = np.exp(-(c1 * L.rho2 + c2 * L.rho2 ** 2))[..., None]
            L.rgb = (L.rgb * corr * ph["gains"][L.i][None, None, :].astype(np.float32)).astype(np.float32)
    apply_ph(layers)
    lab = SE.voronoi(layers, (cv.H, cv.W))
    hm = None
    Hrel = Crel = None
    flows = []
    notes = []
    if want_h:
        say(0.45, "Matching the views' heights")
        hm = solve_heights(layers, cv, sol.ref, sol.surface)
        if hm is not None:
            fit_ground(hm, layers, lab, cv, sol.surface)
            Hrel, Crel = relief_map(hm, layers, lab, cv)
    say(0.52, "Measuring parallax in the overlaps")
    flows = pair_flows(layers)
    ortho = False
    nudges = None
    def ortho_fun(H, a):
        Hg = (H * a).astype(np.float32)

        def hfun(U, V, _H=Hg):
            from scipy.ndimage import map_coordinates
            return map_coordinates(_H, [V, U], order=1, mode="nearest")
        return hfun

    if hm is not None:
        say(0.55, "Measuring the relief's size by parallax")
        shifts_a = relief_shifts(layers)               # the raw parallax, before any correction
        measure_relief(hm, shifts_a, Hrel, sol, cv)
        if hm.alpha:
            Hrel, Crel = relief_map(hm, layers, lab, cv)       # turned the right way up if it was flipped
            ortho = bool(params.ortho)
            if ortho:
                # once more on views lifted by the first answer: there the relief lines up between views, so the
                # views' height scales are matched without leftover parallax biasing them; then the size again
                say(0.58, "Matching the heights again on the lifted views")
                lb = layers_with(hfun=ortho_fun(Hrel, hm.alpha), msg="Lifting")
                apply_ph(lb)
                hm2 = solve_heights(lb, cv, sol.ref, sol.surface)
                if hm2 is not None:
                    lab_b = SE.voronoi(lb, (cv.H, cv.W))
                    fit_ground(hm2, lb, lab_b, cv, sol.surface)
                    if hm.flipped:                     # the depth's sign is known now: keep it
                        hm2.coef = {kk: (-a, -b, -c, -d) for kk, (a, b, c, d) in hm2.coef.items()}
                        hm2.surf = tuple(-v for v in hm2.surf)
                    H2, C2 = relief_map(hm2, lb, lab_b, cv)
                    measure_relief(hm2, shifts_a, H2, sol, cv)
                    if hm2.alpha and not hm2.flipped:
                        hm2.flipped = hm.flipped
                        if hm.flipped:
                            hm2.how += ("; the parallax says near and far were the other way round in the depth, so "
                                        "they were swapped (this camera's Dual Pixel reading is now known)")
                        hm, Hrel, Crel = hm2, H2, C2
    if ortho:
        say(0.62, "Orthorectifying: each place lifted by its own height")
        layers = layers_with(hfun=ortho_fun(Hrel, hm.alpha), vig=None, msg="Orthorectifying")
        apply_ph(layers)
        flows = pair_flows(layers)
    elif params.nudge and flows:
        mags = [float(np.median(np.hypot(F.dy, F.dx))) for _, _, _, F, _ in flows]
        if mags and max(mags) > 0.3:
            say(0.6, "Nudging small leftover parallax in the overlaps")
            nudges = nudge_fields(flows, layers)
            layers = layers_with(nudges=nudges, msg="Nudging")
            apply_ph(layers)
            flows = pair_flows(layers)
            notes.append("A local nudge hid small leftover parallax in the overlaps (it hides it; it does not "
                         "measure it: see the misfit map).")
    check()
    level = {}
    if params.exposure and params.even_out:
        say(0.7, "Levelling the overlaps")
        level = PA.seam_levels(layers)
        for L in layers:
            if L.i in level:
                L.rgb = (L.rgb * np.exp(level[L.i])).astype(np.float32)
    if hm is not None:
        for L in layers:                               # heights in final (ground or relative) units per layer
            if L.tau_l is not None and L.i in hm.coef:
                y0, y1, x0, x1 = L.box
                V, U = np.mgrid[y0:y1, x0:x1]
                gx, gy = cv.to_ground(U, V)
                L.h = hm.value(L.i, L.tau_l, gx, gy, L.p_l, L.q_l).astype(np.float32)
                L.hc = L.conf_l
    say(0.78, "Finding the seams")
    lab = SE.voronoi(layers, (cv.H, cv.W))
    if params.seams == "smart":
        lab = refine_relief(lab, layers, hm)
    ghost = SE.ghosts(layers, (cv.H, cv.W))
    lab, decided = deghost(lab, layers, ghost)
    if decided > 0:
        notes.append(f"Something that moved or a passing shadow was left out by majority ({100 * decided:.1f}% of "
                     "the area, where three or more views saw it): see the ghost map.")
    valid = lab >= 0
    misfit = misfit_map(flows, cv)
    if hm is not None:
        Hrel, Crel = relief_map(hm, layers, lab, cv)
    say(0.9, "Plan ready")
    return {"canvas": cv, "canvas_out": cv_out, "k": k, "layers": layers, "labels": lab, "ghosts": ghost,
            "valid": valid, "gains": ph["gains"], "vig": ph["vig"], "corner_ev": ph["corner_ev"],
            "level": level, "plan_boxes": {L.i: L.box for L in layers}, "hm": hm, "relief": Hrel,
            "relief_conf": Crel, "ortho": ortho, "nudges": nudges, "misfit": misfit, "flows": len(flows),
            "res": res_out, "mm_per_px": mm, "res_how": res_how, "use": [L.i for L in layers], "notes": notes,
            "agreement": height_agreement(layers, cv) if hm is not None else None,
            "overlap": PA.overlap_width(layers, lab) / max(k, 1e-9)}


def _warp_scalar(i, view, sol, cv, L, hfun, nudge):
    """The view's raw depth (t) and confidence on its layer's box (same mapping as the colour)."""
    from scipy.ndimage import map_coordinates
    y0, y1, x0, x1 = L.box
    V, U = np.meshgrid(np.arange(y0, y1, dtype=np.float64), np.arange(x0, x1, dtype=np.float64), indexing="ij")
    if nudge is not None:
        du, dv, (ny0, ny1, nx0, nx1) = nudge
        sy = np.clip(V - ny0, 0, du.shape[0] - 1).astype(np.int64)
        sx = np.clip(U - nx0, 0, du.shape[1] - 1).astype(np.int64)
        inside = (V >= ny0) & (V < ny1) & (U >= nx0) & (U < nx1)
        U = U + np.where(inside, du[sy, sx], 0.0)
        V = V + np.where(inside, dv[sy, sx], 0.0)
    gx, gy = cv.to_ground(U, V)
    h = hfun(U, V) if hfun is not None else None
    px, front = sol.to_view(i, np.stack([gx, gy], -1), h)
    hw, ww = view.shape
    cw = [np.clip(px[..., 1], 0, hw - 1), np.clip(px[..., 0], 0, ww - 1)]
    t = map_coordinates(np.asarray(view.t, np.float32), cw, order=1, mode="nearest")
    c = map_coordinates(np.asarray(view.conf, np.float32), cw, order=1, mode="nearest")
    L_ = sol.lenses[i]
    pn = ((px[..., 0] - L_.cx) / ww).astype(np.float32)
    qn = ((px[..., 1] - L_.cy) / hw).astype(np.float32)
    return (np.where(L.valid, t, 0).astype(np.float32), np.where(L.valid, c, 0).astype(np.float32), pn, qn)


RELIEF_SEAM = 1.5                # seam edge cost at the steepest relief (a cut avoids kerbs and crack walls)


def relief_edge(A, B, box):
    """Extra edge cost for a seam between A and B: the relief's slope there (0 flat .. RELIEF_SEAM steep), from the
    two views' heights, so the cut runs through flat ground rather than along a kerb or a crack wall."""
    from .seams import _crop
    hs = [_crop(L, box, "h") for L in (A, B) if getattr(L, "h", None) is not None]
    if not hs:
        return 0.0
    h = np.mean(hs, axis=0)
    gy, gx = np.gradient(np.asarray(h, np.float32))
    sl = np.hypot(gx, gy)
    p95 = float(np.percentile(sl, 95)) if sl.size else 0.0
    if p95 <= 0:
        return 0.0
    return (RELIEF_SEAM * np.clip(sl / p95, 0, 1)).astype(np.float32)


def refine_relief(labels, layers, hm):
    """Smart seams; with a measured relief the cut also keeps off steep relief (relief_edge)."""
    from . import seams as SE
    if hm is None or not any(getattr(L, "h", None) is not None for L in layers):
        return SE.refine(labels, layers, False)
    return SE.refine(labels, layers, False, edge_extra=relief_edge)


GHOST_MIN = 0.3                  # ghost strength (seams.ghosts) from which a place is decided by majority


def deghost(labels: np.ndarray, layers: list, ghost: np.ndarray, grow: int = 4) -> tuple:
    """Where three or more views overlap and disagree (something moved, a passing shadow, the photographer's own
    shadow at a frame's edge), each place is taken from the view that agrees with the others (nearest the median),
    so the thing is left out rather than cut along a seam. Where only two views see a place there is no majority:
    it stays as the seams chose, and the ghost map shows it. Returns (labels, share of the canvas decided)."""
    from scipy.ndimage import binary_dilation
    from .seams import lum
    m = binary_dilation(ghost >= GHOST_MIN, iterations=grow) & (labels >= 0)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return labels, 0.0
    vals = np.full((len(ys), len(layers)), np.nan, np.float32)
    for t, L in enumerate(layers):
        y0, y1, x0, x1 = L.box
        inside = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
        if not inside.any():
            continue
        yy, xx = ys[inside] - y0, xs[inside] - x0
        ok = L.valid[yy, xx]
        v = np.log(np.maximum(lum(np.asarray(L.rgb[yy, xx], np.float32)), 1e-4))
        col = np.full(inside.sum(), np.nan, np.float32)
        col[ok] = v[ok]
        vals[np.nonzero(inside)[0], t] = col
    cnt = np.isfinite(vals).sum(1)
    use = cnt >= 3
    if not use.any():
        return labels, 0.0
    med = np.nanmedian(vals[use], axis=1)
    d = np.abs(vals[use] - med[:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    best = np.argmin(d, axis=1)
    out = labels.copy()
    ids = np.array([L.i for L in layers], np.int16)
    out[ys[use], xs[use]] = ids[best]
    return out, float(use.sum()) / max(labels.size, 1)


# ------------------------------------------------------------------------------------------- preview and build
def blend_canvas(layers, cv, lab, levels, what="rgb", tile=None):
    """Blend layers into one canvas array (small canvases: one piece)."""
    from . import pano as PA
    from .pyramid import fill
    filled = {}
    for L in layers:
        a = getattr(L, what)
        if a is None:
            continue
        if what == "rgb":
            filled[L.i] = fill(a.astype(np.float32), L.valid)
        else:
            m = L.valid & (L.hc > 0.3)
            st = np.stack([a, L.hc, np.zeros_like(a)], -1).astype(np.float32)
            filled[L.i] = fill(st, m if m.any() else L.valid)
    lys = [L for L in layers if L.i in filled]
    out, _ = PA.blend_region(lys, filled, lambda a, b, c, d: lab[a:b, c:d], 0, cv.H, 0, cv.W, levels)
    return out


def levels_for(overlap_px: float) -> int:
    return int(np.clip(int(math.log2(max(overlap_px, 16) / 4.0)), 2, 7))


def preview(views: list, sol: MO.Solution, params: MO.MosaicParams, progress=None, cancel=None) -> dict:
    """A quick stitch from a plan at PREVIEW_LONG: {"rgb" (linear, cropped), "full", "plan", "crop"}."""
    from . import pano as PA
    sp = plan(views, sol, params, progress, cancel, long_edge=PREVIEW_LONG)
    cv, lab = sp["canvas"], sp["labels"]
    lv = levels_for(sp["overlap"] * sp["k"])
    out = blend_canvas(sp["layers"], cv, lab, lv)
    out[~sp["valid"]] = 0
    cb = PA.crop_box(sp["valid"], params.crop)
    y0, y1, x0, x1 = cb
    return {"rgb": out[y0:y1, x0:x1], "full": out, "plan": sp, "crop": cb}


def build(views: list, sol: MO.Solution, params: MO.MosaicParams, sp: dict | None = None, progress=None,
          cancel=None, tile: int = 1024) -> dict:
    """The full mosaic at the chosen resolution: {"rgb" (linear float32, cropped), "height" (or None), "hconf",
    "agreement", "misfit", "valid", "crop", "canvas", "plan", "labels"}. Streamed through scratch files."""
    from . import pano as PA
    from .focus import Cancelled
    from .frames import Scratch
    from .pyramid import fill, margin
    say = progress or (lambda *_: None)

    def check():
        if cancel is not None and cancel.is_set():
            raise Cancelled()
    if sp is None:
        sp = plan(views, sol, params, lambda f, m: say(0.3 * f, m), cancel)
    cv = sp["canvas_out"]
    if cv.W * cv.H / 1e6 > MAX_CANVAS_MP:
        raise MO.MosaicProblem(f"The mosaic would be {cv.W} x {cv.H} pixels ({cv.W * cv.H / 1e6:.0f} MP): too "
                               "large. Choose a coarser resolution (Game texture is 1 mm per pixel) or leave out "
                               "some views.")
    k = sp["k"]
    lab_s = sp["labels"]
    hm = sp["hm"]

    def labels_at(y0, y1, x0, x1):
        ys = np.clip(((np.arange(y0, y1) + 0.5) * k).astype(np.int64), 0, lab_s.shape[0] - 1)
        xs = np.clip(((np.arange(x0, x1) + 0.5) * k).astype(np.int64), 0, lab_s.shape[1] - 1)
        return lab_s[np.ix_(ys, xs)]
    hfun = _full_height_fun((sp["relief"] * hm.alpha).astype(np.float32), k) if sp["ortho"] else None
    gsd = {i: native_gsd(sol, views, i) for i in sp["use"]}
    gmin = min(gsd.values())
    lv = levels_for(sp["overlap"])
    with_h = hm is not None
    with Scratch("mosaic") as scr:
        layers, filled, hfill = [], {}, {}
        for t, i in enumerate(sp["use"]):
            check()
            say(0.32 + 0.4 * t / max(len(sp["use"]), 1), f"Placing view {t + 1} of {len(sp['use'])} at full size")
            kb = max(1, int(cv.res / max(gsd[i], 1e-30)))
            src = source(views[i], kb)
            nd = None
            if sp["nudges"] and i in sp["nudges"]:
                du, dv, (by0, by1, bx0, bx1) = sp["nudges"][i]
                nd = _nudge_full(du, dv, (by0, by1, bx0, bx1), k)
            L = warp_view(i, views[i], src, sol, cv, order=3, gain=sp["gains"][i], vig=sp["vig"],
                          gsd_rel=gsd[i] / gmin, blur=None, hfun=hfun, hmodel=hm, nudge=nd, dtype=np.float16,
                          with_height=with_h and i in (hm.coef if hm else {}))
            del src
            if L is None:
                continue
            if i in sp["level"]:
                PA.apply_level(L, sp["level"][i], sp["plan_boxes"][i], k, 0.0, 0.0)
            f = scr.array(f"v{i}", L.rgb.shape, np.float16)
            f[...] = fill(L.rgb.astype(np.float32), L.valid).astype(np.float16)
            filled[i] = f
            L.rgb = None
            if with_h and L.h is not None:
                m = L.valid & (L.hc > 0.3)
                hf = scr.array(f"h{i}", L.h.shape + (3,), np.float32)
                hf[...] = fill(np.stack([L.h, L.hc, np.zeros_like(L.h)], -1), m if m.any() else L.valid)
                hfill[i] = hf
                L.h = L.hc = None
            layers.append(L)
        check()
        out = scr.array("canvas", (cv.H, cv.W, 3), np.float32)
        hout = scr.array("hcanvas", (cv.H, cv.W, 3), np.float32) if hfill else None
        M = margin(lv)
        step = max(2 ** lv, tile // (2 ** lv) * (2 ** lv))
        tiles = [(y, x) for y in range(0, cv.H, step) for x in range(0, cv.W, step)]
        hl = [L for L in layers if L.i in hfill]
        for t, (ty, tx) in enumerate(tiles):
            check()
            if t % 4 == 0:
                say(0.74 + 0.2 * t / max(len(tiles), 1), f"Blending: tile {t + 1} of {len(tiles)}")
            y0, y1 = max(0, ty - M), min(cv.H, ty + step + M)
            x0, x1 = max(0, tx - M), min(cv.W, tx + step + M)
            r, labt = PA.blend_region(layers, filled, labels_at, y0, y1, x0, x1, lv)
            r[labt < 0] = 0
            oy, ox = ty - y0, tx - x0
            hh, ww = min(step, cv.H - ty), min(step, cv.W - tx)
            out[ty:ty + hh, tx:tx + ww] = r[oy:oy + hh, ox:ox + ww]
            if hout is not None:
                rh, _ = PA.blend_region(hl, hfill, labels_at, y0, y1, x0, x1, lv)
                hout[ty:ty + hh, tx:tx + ww] = rh[oy:oy + hh, ox:ox + ww]
        # crop on the plan canvas, then in full pixels
        cy0, cy1, cx0, cx1 = PA.crop_box(sp["valid"], params.crop)
        Y0, Y1 = int(math.ceil(cy0 / k)), int(math.floor(cy1 / k))
        X0, X1 = int(math.ceil(cx0 / k)), int(math.floor(cx1 / k))
        if params.crop == "edge":
            Y0, X0 = int(cy0 / k), int(cx0 / k)
            Y1, X1 = int(math.ceil(cy1 / k)), int(math.ceil(cx1 / k))
        Y0, X0 = max(0, Y0), max(0, X0)
        Y1, X1 = min(cv.H, max(Y1, Y0 + 1)), min(cv.W, max(X1, X0 + 1))
        rgb = np.array(out[Y0:Y1, X0:X1])
        height = hconf = None
        if hout is not None:
            height = np.array(hout[Y0:Y1, X0:X1, 0])
            hconf = np.clip(np.array(hout[Y0:Y1, X0:X1, 1]), 0, 1)
    up = lambda a: None if a is None else _plan_to_full(a, k, (Y0, Y1, X0, X1))  # noqa: E731
    valid = up(sp["valid"].astype(np.float32)) > 0.5
    rgb[~valid] = 0
    say(1.0, f"Mosaic {X1 - X0} x {Y1 - Y0}")
    return {"rgb": rgb, "height": height, "hconf": hconf, "valid": valid, "agreement": up(sp["agreement"]),
            "misfit": up(sp["misfit"]), "ghosts": up(sp["ghosts"]), "crop": (Y0, Y1, X0, X1), "canvas": cv,
            "plan": sp, "labels": lab_s}


def _full_height_fun(Hg: np.ndarray, k: float):
    """The relief (ground units, plan canvas) looked up at full-canvas pixels (orthorectification)."""
    def hfun(U, V):
        from scipy.ndimage import map_coordinates
        return map_coordinates(Hg, [(V + 0.5) * k - 0.5, (U + 0.5) * k - 0.5], order=1, mode="nearest")
    return hfun


def _nudge_full(du, dv, box, k):
    """A plan-canvas nudge field (plan pixels) for the full canvas: the same box in full pixels, values in full
    pixels (sampled per full pixel)."""
    from scipy.ndimage import zoom
    by0, by1, bx0, bx1 = box
    Y0, X0 = int(math.floor(by0 / k)), int(math.floor(bx0 / k))
    Y1, X1 = int(math.ceil(by1 / k)), int(math.ceil(bx1 / k))
    z = (max(1, Y1 - Y0) / du.shape[0], max(1, X1 - X0) / du.shape[1])
    dfu = zoom(du, z, order=1, grid_mode=True, mode="nearest") / k
    dfv = zoom(dv, z, order=1, grid_mode=True, mode="nearest") / k
    return dfu.astype(np.float32), dfv.astype(np.float32), (Y0, Y0 + dfu.shape[0], X0, X0 + dfu.shape[1])


def _plan_to_full(a, k, crop):
    """A plan-canvas map brought to the full canvas's cropped box (bilinear; NaN kept where it was NaN)."""
    from scipy.ndimage import map_coordinates
    Y0, Y1, X0, X1 = crop
    ys = (np.arange(Y0, Y1) + 0.5) * k - 0.5
    xs = (np.arange(X0, X1) + 0.5) * k - 0.5
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    a = np.asarray(a, np.float32)
    fin = np.isfinite(a)
    v = map_coordinates(np.nan_to_num(a), [YY, XX], order=1, mode="nearest")
    if not fin.all():
        ok = map_coordinates(fin.astype(np.float32), [YY, XX], order=1, mode="nearest") > 0.5
        v = np.where(ok, v, np.nan)
    return v.astype(np.float32)
