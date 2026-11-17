"""The Panorama tab's engine (BUILD_PLAN_PANO_IR.md section 2): frames in, one stitched panorama out.

    analyse(paths, params)     -> PanoAnalysis: frames developed at a small size, features matched between pairs,
                                  every frame's rotation and one focal length solved (bundle.py), horizon levelled
    layout_view(an, params)    -> each frame's outline on a small canvas, coloured by how well it is tied in
    preview(an, params)        -> a 1/8-size stitch (seconds), the seam labels and where frames disagree
    build(an, params, out)     -> the full panorama, streamed through scratch files on disk, written as a linear
                                  intermediate FIRST_to_LAST_Pano_Studio.tif that opens in Develop

How the stitch is made (each step's choice is on the page with what it is best for and weak at):
* every frame is developed with the reference frame's own edit (the Infrared tab's settings and hot-spot correction
  included), turned upright, warped into the chosen projection (projection.py) with a cubic spline;
* one exposure gain per frame from the overlaps (Brown and Lowe's gain compensation, in linear light);
* seams: each pixel first goes to the frame whose centre is nearest (Voronoi), then every boundary between two frames
  is re-cut through their overlap by dynamic programming (seams.py) where the frames agree, so moving people, water
  and leaves are cut around instead of doubled. The cut also prefers, area by area, the frame that is not clipped and,
  with the flare guard on, the darker of two frames where one is brighter in a soft patch: flare and the infrared hot
  spot only ever add light, and they move with the camera, so a plain blend keeps every frame's flare;
* multi-band blending (Burt and Adelson 1983) across the seams, in tiles that join exactly;
* a crop to the largest rectangle inside the panorama (or keep the whole edge).

Honest limits (said on the page): parallax from hand-held near subjects cannot be blended away; moving water and people
are cut along a seam, not merged; a lens with no focal length in the file needs it typed or estimated; a full 360 needs
overlap all the way round, top and bottom included.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from ... import log as slog
from . import align as A
from . import bundle as BU
from . import features as FE
from . import project as PJ

log = slog.get("engine.stack.pano")

ANALYSIS_LONG = 1600            # analysis copies: long edge in pixels
PREVIEW_SCALE = 0.125
SENSOR_LONG_MM = {"450D": 22.2, "REBEL XSI": 22.2, "KISS X2": 22.2, "R5": 36.0, "5D": 36.0, "6D": 36.0, "R6": 36.0,
                  "1D X": 36.0, "R3": 36.0, "R8": 36.0, "R": 36.0, "7D": 22.3, "90D": 22.3, "80D": 22.3,
                  "R7": 22.3, "R10": 22.3, "R50": 22.3, "77D": 22.3, "800D": 22.3, "M50": 22.3, "60D": 22.3}
PAIR_WARN_PX = 2.0              # a pair whose residual is above this (full-size pixels) is named on the page

METHOD_INFO = {
    "smart": {"name": "Smart seams", "does": "Cuts each overlap where the two frames agree (and, with the flare "
              "guard, keeps the darker frame where one has flare).",
              "best": "Moving people, water, leaves, clouds; flare and the infrared hot spot.",
              "weak": "A seam can still run through something that moved across the whole overlap."},
    "straight": {"name": "Straight (Voronoi)", "does": "Each place from the frame whose centre is nearest.",
                 "best": "Still scenes on a tripod; the most even sharpness.",
                 "weak": "Doubles or cuts anything that moved."},
    "rectangle": {"name": "Crop to a rectangle", "does": "The largest rectangle inside the panorama.",
                  "best": "A finished picture with no empty corners.", "weak": "Loses the panorama's ragged edge."},
    "edge": {"name": "Keep the edge", "does": "Keeps every pixel; the empty border is filled with black.",
             "best": "Filling the corners yourself later, or a 360 panorama.",
             "weak": "Empty corners until you crop or fill them."},
}
WHAT = ("Stitches overlapping photos into one panorama: finds where each frame points, evens out exposure, cuts the "
        "seams where frames agree and blends them.")
SHOOT = ("Overlap each frame by a third, turn the camera rather than walk, manual exposure and focus, and keep "
         "moving things out of the overlaps.")
HONEST = ("Hand-held near subjects shift against the background between frames (parallax): no blend can fix that, "
          "and the seam will show; use a tripod and turn around the lens. Moving water and people are cut along a "
          "seam, not merged. A lens with no focal length in the file needs it typed or estimated. A full 360 needs "
          "overlap all the way round, top and bottom included, or the poles stay empty. Lens flare cannot be taken "
          "out of a single frame; the flare guard only stops it being repeated.")


class PanoProblem(ValueError):
    """The frames cannot make a panorama (said plainly on the page)."""


@dataclass
class PanoParams:
    projection: str = "auto"
    level: bool = True
    focal_px: float = 0.0          # typed focal length in full-size pixels; 0 = from the file or estimated
    focal_mm: float = 0.0          # typed focal length in mm (with the sensor width from the camera); 0 = none
    seams: str = "smart"
    crop: str = "rectangle"
    exposure: bool = True
    flare_guard: bool = True
    leave_out: list = field(default_factory=list)
    blend_levels: int = 0          # 0 = from the overlap width
    seam_level: bool = True        # with exposure: level the broad brightness across every seam (seam_levels)


@dataclass
class PanoAnalysis:
    paths: list
    shape: tuple                    # (h, w) of a full-size upright frame
    k: float                        # full-size pixels per analysis pixel
    f: float                        # focal length, full-size pixels
    focal_source: str               # "file" | "typed" | "estimated" | "guessed"
    R: list                         # world -> camera per frame (None: not placed)
    pairs: list                     # [{"i", "j", "inliers", "matches", "rms"}] (rms in full-size pixels)
    problems: dict                  # path -> why it was left out
    edit: dict                      # the edit every frame is developed with
    flips: list
    small: list = field(default_factory=list)   # 1/4-size upright scene-linear frames (float16), for previews
    small_k: float = 4.0
    rms: float = 0.0
    k1: float = 0.0                 # shared radial distortion (bundle.undistort; over the half-diagonal)

    @property
    def placed(self) -> list:
        return [r is not None for r in self.R]

    @property
    def n(self) -> int:
        return len(self.paths)

    def tie(self, i: int) -> str:
        """How well frame i is tied in: "ok" (several good pairs), "warn" (one, or a weak one), "dropped"."""
        if self.R[i] is None:
            return "dropped"
        mine = [p for p in self.pairs if i in (p["i"], p["j"])]
        good = [p for p in mine if p["rms"] <= PAIR_WARN_PX and p["inliers"] >= 30]
        return "ok" if good else "warn"


# ------------------------------------------------------------------------------------------- the layout
def exif_focal_px(meta, long_px: int) -> float:
    """Focal length in pixels from the file (focal length in mm and the sensor's long side), or 0."""
    fmm = float(getattr(meta, "focal_mm", 0) or 0)
    if fmm <= 0:
        return 0.0
    model = str(getattr(meta, "model", "") or "").upper()
    sensor = 0.0
    for key, mm in sorted(SENSOR_LONG_MM.items(), key=lambda kv: -len(kv[0])):
        if key in model.split() or model.endswith(" " + key) or f" {key} " in f" {model} ":
            sensor = mm
            break
    if sensor <= 0:
        return 0.0
    return fmm / sensor * long_px


def candidate_pairs(n: int, near: int = 6) -> list:
    if n <= 12:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    return [(i, j) for i in range(n) for j in range(i + 1, min(n, i + near + 1))]


def solve_layout(lums: list, f_hint: float = 0.0, solve_f: bool = True, do_level: bool = True,
                 progress=None, cancel=None) -> dict:
    """Rotations and focal length from analysis copies (all one size). ``f_hint`` in analysis pixels (0: estimate).
    Returns {"R", "f", "focal_source", "pairs" (with H, pa, pb), "rms", "ref"}."""
    from .focus import Cancelled
    n = len(lums)
    h, w = lums[0].shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    say = progress or (lambda *_: None)
    feats = []
    for i, L in enumerate(lums):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        say(0.1 + 0.3 * i / n, f"Finding points in frame {i + 1} of {n}")
        feats.append(FE.FrameFeatures(L))
    rng = np.random.default_rng(0)
    tried, pairs = set(), []

    def try_pairs(cands):
        for t, (i, j) in enumerate(cands):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            if (i, j) in tried:
                continue
            tried.add((i, j))
            m = FE.match_pair(feats[i], feats[j], rng=rng)
            if m is not None:
                m.update(i=i, j=j)
                pairs.append(m)
    say(0.45, "Matching frames")
    try_pairs(candidate_pairs(n))
    if not pairs:
        raise PanoProblem("No two frames share enough detail to be joined. Do they overlap by about a third?")
    if f_hint > 0:
        f0, src = float(f_hint), "file"
    else:
        Hcs = [BU._centred(p["H"], cx, cy) for p in pairs]
        f0 = BU.focal_from_homographies(Hcs, [p["inliers"] for p in pairs], width=max(w, h))
        src = "estimated"
        if f0 is None:
            f0, src = 1.2 * max(w, h), "guessed"
    say(0.6, "Solving where each frame points")
    starts = [f0] if (src == "file" or not solve_f) else [f0, 0.8 * f0, 1.25 * f0]
    sol, ref, seen = None, 0, set()
    for fs in starts:                                 # f is weakly fixed by a long lens: try a few starts
        r0, R0, sn = BU.initial_rotations(n, pairs, fs, cx, cy)
        s1 = BU.adjust(n, pairs, fs, cx, cy, R0, r0, solve_f=solve_f and src != "file")
        if sol is None or s1["rms"] < sol["rms"] - 1e-6:
            sol, ref, seen = s1, r0, sn
        if sol["rms"] < 0.5 and n > 20:
            break
    # then one shared lens distortion (vintage and adapted lenses have no profile), continued from the best
    # rotation-only answer: f and distortion trade off on a long lens, and this start is the one that held (Bill's
    # 135 mm Iraq set: 136 mm from this start)
    if len(pairs) >= 2:
        sk = BU.adjust(n, pairs, sol["f"], cx, cy, sol["R"], ref, solve_f=solve_f and src != "file", solve_k1=True)
        if sk["rms"] < 0.97 * sol["rms"] and abs(sk["k1"]) < 0.15:
            sol = sk
    if n > 12:                                        # second pass: frames the first solve says overlap
        more = []
        fov = 2 * math.atan(0.5 * max(w, h) / sol["f"])
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in tried or sol["R"][i] is None or sol["R"][j] is None:
                    continue
                if BU.angle_deg(sol["R"][i], sol["R"][j]) < math.degrees(0.8 * fov):
                    more.append((i, j))
        if more:
            say(0.7, "Matching frames the layout says overlap")
            try_pairs(more)
            ref, R0, seen = BU.initial_rotations(n, pairs, sol["f"], cx, cy)
            k1 = float(sol.get("k1", 0.0))
            sol = BU.adjust(n, pairs, sol["f"], cx, cy, R0, ref, solve_f=solve_f and src != "file",
                            solve_k1=bool(k1), k10=k1)
    R = sol["R"]
    R = [R[i] if i in seen else None for i in range(n)]
    if do_level:
        R = BU.level(R)
    for p in pairs:
        p["rms"] = sol["pair_rms"].get((p["i"], p["j"]), 0.0)
    return {"R": R, "f": sol["f"], "k1": float(sol.get("k1", 0.0)), "focal_source": src, "pairs": pairs,
            "rms": sol["rms"], "ref": ref, "solve_f": solve_f and src != "file"}


def refine_layout(sol: dict, bigs: list, s: float, solve_f: bool = True, cancel=None) -> dict:
    """The matches moved to sub-pixel accuracy on copies ``s`` times larger (Lucas-Kanade on each match's patch), then
    the bundle adjusted again there: the layout's precision at a finer scale than the features were found at."""
    from .focus import Cancelled
    if s <= 1.0 or not bigs:
        return sol
    h, w = bigs[0].shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    prep = {}
    pairs = []
    for p in sol["pairs"]:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        i, j = p["i"], p["j"]
        for k in (i, j):
            if k not in prep:
                prep[k] = FE.prepare(np.asarray(bigs[k], np.float32))
        a = (p["pa"] + 0.5) * s - 0.5
        b = (p["pb"] + 0.5) * s - 0.5
        b2, ok = FE.refine_lk(prep[i], prep[j], a, b, half=10)
        if ok.sum() < 8:
            continue
        pairs.append(dict(p, pa=a[ok], pb=b2[ok]))
    if not pairs:
        return sol
    R0 = list(sol["R"])
    ref = sol.get("ref", 0)
    k1 = float(sol.get("k1", 0.0))
    # rotations only: the focal length and distortion stay as the whole-frame solve found them (on real sets the
    # finer matches include things that moved, smoke and clouds, which would pull f along the flat valley it lies in)
    out = BU.adjust(len(R0), pairs, sol["f"] * s, cx, cy, R0, ref, solve_f=False, solve_k1=False, k10=k1)
    if out["rms"] / s >= 0.98 * sol["rms"]:
        return sol
    res = dict(sol, R=out["R"], rms=out["rms"] / s)
    for p in sol["pairs"]:
        p["rms"] = out["pair_rms"].get((p["i"], p["j"]), p.get("rms", 0.0) * s) / s
    return res


def pano_edit(ref_path: str):
    """The edit every frame is developed with: the reference frame's own raw-stage settings (white balance, denoise,
    demosaic, lens fixes, infrared and its hot spot) and nothing that belongs to one photo (geometry, crop, masks,
    retouch, portrait, composite)."""
    from .. import api
    from ..pipeline.edit import Edit, Geometry
    try:
        e = api.load_edit(ref_path)
    except (OSError, ValueError):
        e = Edit()
    if not os.path.exists(api.sidecar_path(ref_path)):
        from .frames import stack_edit
        try:
            e = stack_edit(api.open(ref_path))
        except Exception:                             # noqa: BLE001 - a scan has no raw white balance
            e = Edit()
    e.geometry = Geometry()
    e.masks = e.retouch = e.portrait = e.composite = None
    e.sharpen.enabled = False
    return e


def _upright(rgb: np.ndarray, flip: int) -> np.ndarray:
    from ..pipeline.pipeline import apply_orientation
    return np.ascontiguousarray(apply_orientation(rgb, flip))


def analyse(paths: list, params: PanoParams | None = None, progress=None, cancel=None) -> PanoAnalysis:
    from .. import api
    from ..pipeline.pipeline import linear_rgb
    from .focus import Cancelled
    params = params or PanoParams()
    say = progress or (lambda *_: None)
    use = [p for p in paths if p not in set(params.leave_out or [])]
    if len(use) < 2:
        raise PanoProblem("A panorama needs at least two frames.")
    edit = pano_edit(use[0])
    lums, small, flips, problems, keep = [], [], [], {}, []
    shape = None
    meta0 = None
    for i, p in enumerate(use):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        say(0.02 + 0.08 * i / len(use), f"Developing frame {i + 1} of {len(use)}")
        try:
            img = api.open(p)
            rgb = linear_rgb(img, edit, half_size=True, cache=False)
            flip = int(getattr(img.raw, "flip", 0) or 0)
            full = img.output_size                      # (w, h) upright
            if meta0 is None:
                meta0 = img.meta
            img.drop_cache()
        except Exception as err:                        # noqa: BLE001 - one bad frame is left out and named
            if not slog.expected_problem(err):
                slog.log_exception("panorama: a frame could not be developed", log)
                slog.save_error_report("Panorama: developing a frame")
            problems[p] = f"could not be read ({str(err)[:60]})"
            continue
        up = _upright(rgb, flip)
        fs = (full[1], full[0])
        if shape is None:
            shape = fs
        elif fs != shape:
            problems[p] = "a different size or orientation from the first frame"
            continue
        keep.append(p)
        flips.append(flip)
        small.append(A.bin_by(up, 2).astype(np.float16))
        lums.append(A.luma(up))
    if len(keep) < 2:
        raise PanoProblem("Fewer than two frames could be used.")
    k_half = shape[1] / float(lums[0].shape[1])
    lum_a = []
    kk = max(1, int(math.ceil(max(lums[0].shape) / float(ANALYSIS_LONG))))
    for L in lums:
        lum_a.append(A.bin_by(L, kk))
    k = k_half * kk
    f_hint = 0.0
    src_note = ""
    if params.focal_px > 0:
        f_hint, src_note = params.focal_px / k, "typed"
    elif params.focal_mm > 0 and meta0 is not None:
        class _M:
            focal_mm = params.focal_mm
            model = getattr(meta0, "model", "")
        fp = exif_focal_px(_M, max(shape))
        if fp > 0:
            f_hint, src_note = fp / k, "typed"
    if f_hint <= 0 and meta0 is not None:
        fp = exif_focal_px(meta0, max(shape))
        if fp > 0:
            f_hint = fp / k
    sol = solve_layout(lum_a, f_hint, solve_f=not src_note, do_level=False, progress=progress, cancel=cancel)
    if kk > 1:                                        # finish on the half-size copies (kk times finer)
        say(0.85, "Refining the layout at a finer scale")
        sol = refine_layout(sol, lums, float(kk), solve_f=sol.get("solve_f", True), cancel=cancel)
    if params.level:
        sol["R"] = BU.level(sol["R"])
    src = src_note or sol["focal_source"]
    pairs = [{"i": p["i"], "j": p["j"], "inliers": p["inliers"], "matches": p["matches"], "rms": p["rms"] * k}
             for p in sol["pairs"]]
    for i, r in enumerate(sol["R"]):
        if r is None:
            problems[keep[i]] = "no overlap found with the other frames"
    an = PanoAnalysis(paths=keep, shape=shape, k=k, f=sol["f"] * k, focal_source=src, R=sol["R"], pairs=pairs,
                      problems=problems, edit=edit.to_dict(), flips=flips, small=small,
                      small_k=shape[1] / float(small[0].shape[1]), rms=sol["rms"] * k, k1=sol.get("k1", 0.0))
    log.info("panorama layout: %d frames placed of %d, f %.0f px (%s), rms %.2f px", sum(an.placed), len(keep),
             an.f, src, an.rms)
    say(1.0, f"{sum(an.placed)} of {len(use)} frames placed")
    return an


def resolved_projection(an: PanoAnalysis, params: PanoParams) -> str:
    if params.projection in PJ.PROJECTIONS:
        return params.projection
    eh, ev = PJ.extents_deg(an.R, an.f, an.shape[1], an.shape[0])
    return PJ.auto_projection(eh, ev)


def layout_view(an: PanoAnalysis, params: PanoParams, long_edge: int = 900) -> dict:
    """Each frame's outline on a small canvas (in canvas pixels) with its tie-in colour, and the canvas size."""
    proj = resolved_projection(an, params)
    h, w = an.shape
    Rs = an.R
    if not any(r is not None for r in Rs):
        return {"size": (long_edge, long_edge // 2), "outlines": [], "projection": proj}
    u0, v0, W, H = PJ.canvas_box(Rs, an.f, w, h, proj, an.f)
    s = long_edge / float(max(W, H))
    out = []
    for i, R in enumerate(Rs):
        if R is None:
            out.append({"i": i, "pts": [], "tie": "dropped"})
            continue
        d = PJ.frame_outline(R, an.f, w, h)
        u, v, ok = PJ.to_canvas(d, proj, an.f)
        pts = [((uu - u0) * s, (vv - v0) * s) for uu, vv, o in zip(u, v, ok) if o]
        out.append({"i": i, "pts": pts, "tie": an.tie(i)})
    return {"size": (int(W * s), int(H * s)), "outlines": out, "projection": proj}


# ------------------------------------------------------------------------------------------- the canvas
@dataclass
class Canvas:
    projection: str
    scale: float                    # canvas pixels per radian
    u0: int
    v0: int
    W: int
    H: int

    def resized(self, k: float) -> "Canvas":
        return Canvas(self.projection, self.scale * k, int(math.floor(self.u0 * k)), int(math.floor(self.v0 * k)),
                      max(1, int(math.ceil(self.W * k))), max(1, int(math.ceil(self.H * k))))


MAX_CANVAS_MP = 400.0


def make_canvas(an: PanoAnalysis, params: PanoParams, factor: float = 1.0) -> Canvas:
    proj = resolved_projection(an, params)
    h, w = an.shape
    s = an.f * factor
    u0, v0, W, H = PJ.canvas_box(an.R, an.f, w, h, proj, s)
    return Canvas(proj, s, u0, v0, W, H)


def warp_layer(i: int, rgb: np.ndarray, frame_scale: float, an: PanoAnalysis, cv: Canvas, order: int = 1,
               gain=1.0, dtype=np.float32, vig=None):
    """Frame i (upright scene-linear, ``frame_scale`` of full size) warped into the canvas: a seams.Layer with its
    pixels, validity, distance from the frame's own centre (0 centre .. ~0.5 corners) and where it is clipped."""
    from scipy.ndimage import map_coordinates, spline_filter
    from .seams import Layer
    R = an.R[i]
    hf, wf = rgb.shape[:2]
    f = an.f * frame_scale
    cx, cy = (wf - 1) / 2.0, (hf - 1) / 2.0
    rn = math.hypot(cx, cy)
    d = PJ.frame_outline(R, f, wf, hf, 40)
    u, v, ok = PJ.to_canvas(d, cv.projection, cv.scale)
    if not ok.any():
        return None
    x0 = max(0, int(math.floor(u[ok].min() - cv.u0)) - 2)
    x1 = min(cv.W, int(math.ceil(u[ok].max() - cv.u0)) + 3)
    y0 = max(0, int(math.floor(v[ok].min() - cv.v0)) - 2)
    y1 = min(cv.H, int(math.ceil(v[ok].max() - cv.v0)) + 3)
    if x1 <= x0 or y1 <= y0:
        return None
    bh, bw = y1 - y0, x1 - x0
    out = np.zeros((bh, bw, 3), dtype)
    valid = np.zeros((bh, bw), bool)
    rr = np.zeros((bh, bw), np.float32)
    rho2 = np.zeros((bh, bw), np.float32)
    clipped = np.zeros((bh, bw), bool)
    g3 = np.broadcast_to(np.asarray(gain, np.float32), (3,)).astype(np.float32)
    src = np.asarray(rgb, np.float32)
    clip_src = (src.max(axis=2) >= 0.98).astype(np.float32)
    chans = [src[..., c] for c in range(3)]
    if order > 1:
        chans = [spline_filter(c, order, output=np.float32, mode="mirror") for c in chans]
    xs = np.arange(x0, x1, dtype=np.float64) + cv.u0
    for r0 in range(0, bh, 256):
        r1 = min(bh, r0 + 256)
        vv, uu = np.meshgrid(np.arange(y0 + r0, y0 + r1, dtype=np.float64) + cv.v0, xs, indexing="ij")
        dd = PJ.from_canvas(uu, vv, cv.projection, cv.scale)
        fx, fy, front = PJ.to_camera(dd, R, f, cx, cy)
        fx, fy = BU.distort(fx, fy, cx, cy, an.k1, rn)
        ok = front & (fx >= -0.5) & (fx <= wf - 0.5) & (fy >= -0.5) & (fy <= hf - 0.5)
        valid[r0:r1] = ok
        rr[r0:r1] = (((fx - cx) / wf) ** 2 + ((fy - cy) / hf) ** 2).astype(np.float32)
        p2 = (((fx - cx) ** 2 + (fy - cy) ** 2) / (rn * rn)).astype(np.float32)
        rho2[r0:r1] = p2
        corr = np.float32(1.0) if not vig else np.exp(-(vig[0] * p2 + vig[1] * p2 * p2)).astype(np.float32)
        coords = [np.clip(fy, 0, hf - 1).ravel(), np.clip(fx, 0, wf - 1).ravel()]
        for c in range(3):
            s = map_coordinates(chans[c], coords, order=order, mode="mirror" if order > 1 else "nearest",
                                prefilter=False, output=np.float32).reshape(fx.shape)
            out[r0:r1, :, c] = np.where(ok, s * corr * g3[c], 0).astype(dtype)
        cl = map_coordinates(clip_src, coords, order=0, mode="nearest").reshape(fx.shape)
        clipped[r0:r1] = (cl > 0.5) & ok
    L = Layer(i, (y0, y1, x0, x1), out, valid, rr, clipped)
    L.rho2 = rho2
    return L


def exposure_gains(layers: list, n: int) -> np.ndarray:
    """One gain per frame from the overlaps (log means of unclipped, exposed pixels both frames share), least
    squares with a weak pull toward 1 (Brown and Lowe's gain compensation); normalised to a geometric mean of 1."""
    from .seams import _crop, _region, lum
    rows, rhs, wts = [], [], []
    idx = {L.i: t for t, L in enumerate(layers)}
    for a in range(len(layers)):
        for b in range(a + 1, len(layers)):
            A, B = layers[a], layers[b]
            box = _region(A, B)
            if box is None:
                continue
            la = lum(_crop(A, box, "rgb").astype(np.float32))
            lb = lum(_crop(B, box, "rgb").astype(np.float32))
            m = _crop(A, box, "valid") & _crop(B, box, "valid") & ~_crop(A, box, "clipped") & \
                ~_crop(B, box, "clipped") & (la > 0.004) & (lb > 0.004) & (la < 0.9) & (lb < 0.9)
            k = int(m.sum())
            if k < 50:
                continue
            r = float(np.log(np.median(la[m])) - np.log(np.median(lb[m])))
            row = np.zeros(len(layers))
            row[idx[A.i]], row[idx[B.i]] = 1.0, -1.0
            rows.append(row)
            rhs.append(-r)
            wts.append(math.sqrt(k))
    g = np.ones(n)
    if not rows:
        return g
    Am = np.array(rows) * np.array(wts)[:, None]
    bm = np.array(rhs) * np.array(wts)
    lam = 0.05 * float(np.mean(wts))
    Am = np.vstack([Am, lam * np.eye(len(layers))])
    bm = np.r_[bm, np.zeros(len(layers))]
    a, *_ = np.linalg.lstsq(Am, bm, rcond=None)
    a -= a.mean()
    for L, v in zip(layers, a):
        g[L.i] = float(np.exp(v))
    return g


def photometric(layers: list, n: int, vignette: bool = True, samples: int = 4000, rng=None) -> dict:
    """Exposure gains per frame and channel, and one vignetting curve for the lens (log V = c1 rho^2 + c2 rho^4,
    rho over the half-diagonal), from pixel pairs in the overlaps: the same scene point seen by two frames must come
    out equal once each frame's falloff and gain are undone (Goldman and Chen, ICCV 2005, linear in the log domain).
    Returns {"gains": (n x 3), "vig": (c1, c2) or None, "corner_ev"}."""
    from .seams import _crop, _region, lum
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = {L.i: t for t, L in enumerate(layers)}
    m = len(layers)
    rowsL, rhsL, rowsC = [], [], {c: [] for c in range(3)}
    for a in range(m):
        for b in range(a + 1, m):
            A, B = layers[a], layers[b]
            box = _region(A, B)
            if box is None:
                continue
            ra = _crop(A, box, "rgb").astype(np.float32)
            rb = _crop(B, box, "rgb").astype(np.float32)
            la, lb = lum(ra), lum(rb)
            ok = _crop(A, box, "valid") & _crop(B, box, "valid") & ~_crop(A, box, "clipped") & \
                ~_crop(B, box, "clipped") & (la > 0.004) & (lb > 0.004) & (la < 0.9) & (lb < 0.9)
            ys, xs = np.nonzero(ok)
            if len(ys) < 50:
                continue
            take = rng.choice(len(ys), min(samples, len(ys)), replace=False)
            ys, xs = ys[take], xs[take]
            pa = _crop(A, box, "rho2")[ys, xs]
            pb = _crop(B, box, "rho2")[ys, xs]
            w = 1.0 / math.sqrt(len(ys))
            # log la - (c1 pa + c2 pa^2) + ga = log lb - (c1 pb + c2 pb^2) + gb
            k = len(ys)
            R = np.zeros((k, 2 + m))
            R[:, 0] = -(pa - pb)
            R[:, 1] = -(pa ** 2 - pb ** 2)
            R[:, 2 + idx[A.i]] = 1.0
            R[:, 2 + idx[B.i]] = -1.0
            rowsL.append(R * w)
            rhsL.append((np.log(lb[ys, xs]) - np.log(la[ys, xs])) * w)
            for c in range(3):
                va = np.maximum(ra[ys, xs, c], 1e-4)
                vb = np.maximum(rb[ys, xs, c], 1e-4)
                rowsC[c].append((idx[A.i], idx[B.i], np.log(vb) - np.log(va), pa, pb, w))
    gains = np.ones((n, 3))
    if not rowsL:
        return {"gains": gains, "vig": None, "corner_ev": 0.0}
    Am = np.vstack(rowsL)
    bm = np.concatenate(rhsL)
    lam = 0.02
    if not vignette:
        Am[:, :2] = 0
    prior = np.zeros((m + 2, 2 + m))
    prior[0, 0] = prior[1, 1] = 0.01
    prior[2:, 2:] = lam * np.eye(m)
    sol, *_ = np.linalg.lstsq(np.vstack([Am, prior]), np.r_[bm, np.zeros(m + 2)], rcond=None)
    c1, c2 = (float(sol[0]), float(sol[1])) if vignette else (0.0, 0.0)
    corner = -(c1 + c2) / math.log(2)                  # stops darker at the corner, as the lens recorded it
    if vignette and not (-0.05 < corner < 3.0):        # a lens darkens its corners; anything else is the scene
        c1 = c2 = 0.0
        corner = 0.0
    # per-channel gains with the vignetting fixed
    for c in range(3):
        rows, rhs = [], []
        for ia, ib, d, pa, pb, w in rowsC[c]:
            k = len(d)
            R = np.zeros((k, m))
            R[:, ia], R[:, ib] = 1.0, -1.0
            rows.append(R * w)
            rhs.append((d + (c1 * (pa - pb) + c2 * (pa ** 2 - pb ** 2))) * w)
        A2 = np.vstack(rows + [lam * np.eye(m)])
        b2 = np.concatenate(rhs + [np.zeros(m)])
        g, *_ = np.linalg.lstsq(A2, b2, rcond=None)
        for L, v in zip(layers, g):
            gains[L.i, c] = v
    gains = gains - gains[[L.i for L in layers]].mean()  # geometric mean 1 over all channels and frames
    gains = np.exp(gains)
    return {"gains": gains, "vig": (c1, c2) if (c1 or c2) else None, "corner_ev": round(corner, 2)}


def _levels(an: PanoAnalysis, factor: float, params: PanoParams, overlap_px: float | None = None) -> int:
    if params.blend_levels:
        return int(params.blend_levels)
    if overlap_px:                                     # broad tones blend across a quarter of the overlap
        return int(np.clip(int(math.log2(max(overlap_px, 16) / 4.0)), 2, 8))
    side = min(an.shape) * factor
    return int(np.clip(int(math.log2(max(side, 32) / 40.0)), 2, 6))


def overlap_width(layers: list, labels: np.ndarray) -> float:
    """Typical width of the overlaps (canvas pixels): each pair's shared area over the length of its boundary."""
    from .seams import _crop, _region
    ws = []
    for a in range(len(layers)):
        for b in range(a + 1, len(layers)):
            A, B = layers[a], layers[b]
            box = _region(A, B)
            if box is None:
                continue
            both = _crop(A, box, "valid") & _crop(B, box, "valid")
            area = float(both.sum())
            if area < 50:
                continue
            ys, xs = np.nonzero(both)
            length = max(np.ptp(ys), np.ptp(xs)) + 1
            ws.append(area / length)
    return float(np.median(ws)) if ws else 0.0


LEVEL_SIGMA = 6.0       # plan-canvas pixels: the broad brightness each side of a seam is compared at this blur
LEVEL_LIMIT = 0.7       # a seam is levelled by at most a factor of 2 (the rest is a real difference, left to the blend)
LEVEL_GRID = 200        # the levelling fields are solved on a grid this many cells across the canvas, then smoothed
LEVEL_SMOOTH = 3.0      # how strongly a frame's field must be smooth from cell to cell, against agreeing in overlaps
LEVEL_ANCHOR = 1e-3     # a faint pull of every field toward 0
LEVEL_MOVED = 0.18      # a difference with edges this strong (log) is something that moved: not levelled


def seam_levels(layers: list) -> dict:
    """Overlap levelling: a smooth log-gain field for every frame (per channel, over its box) so that wherever frames
    overlap they agree in their broad brightness and colour. Global gains and the vignetting curve even out what is
    the same everywhere; what is local (a flare veil that grows toward the sun in one frame, a hot spot, haze or light
    that changed between frames) still differs, and a multi-band blend of two frames whose broad brightness slopes
    opposite ways leaves a step or a dark crease along the seam, wherever the seam is put.

    All frames are solved together, on one coarse grid over the canvas, by least squares: in every overlap cell the
    frames should agree (log_i + F_i = log_j + F_j, weighted by how much of the cell they share), every field should be
    smooth (neighbouring cells alike), and a faint pull keeps each field near 0 (the global gains already set the
    overall level). Solving all pairs at once keeps three-frame overlaps consistent, which pairwise corrections are
    not. Something that moved (a difference with edges) is left out of the agreement (the seam cuts around it) and
    the smoothness fills in there. Only the broad part is touched, so fine detail never moves. Gradient-domain
    stitching does the same in spirit (Levin, Zomet, Peleg, Weiss, ECCV 2004). Returns {frame index: (h, w, 3)
    float32 log gains over the frame's box}."""
    from scipy import sparse
    from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates
    from scipy.sparse.linalg import splu
    from .pyramid import fill
    from .seams import _region, core_mask
    if len(layers) < 2:
        return {}
    logs, ok, fine = {}, {}, {}
    for L in layers:
        v = (L.valid & ~L.clipped).astype(np.float32)
        den = gaussian_filter(v, LEVEL_SIGMA)
        num = np.stack([gaussian_filter(np.asarray(L.rgb[..., c], np.float32) * v, LEVEL_SIGMA)
                        for c in range(3)], -1)
        logs[L.i] = np.log(np.maximum(num / np.maximum(den, 1e-6)[..., None], 1e-5)).astype(np.float32)
        ok[L.i] = (den > 0.5) & core_mask(L)
        fine[L.i] = np.log(np.maximum(np.asarray(L.rgb, np.float32).mean(-1), 1e-4))
    H = max(L.box[1] for L in layers)
    W = max(L.box[3] for L in layers)
    s = max(1, int(math.ceil(max(H, W) / float(LEVEL_GRID))))
    hc, wc = -(-H // s), -(-W // s)

    def to_cells(a, box):
        """Block sums of a (h, w[, c]) array that sits at canvas box, on the canvas grid: (hc, wc[, c])."""
        y0, y1, x0, x1 = box
        full = np.zeros((hc * s, wc * s) + a.shape[2:], np.float64)
        full[y0:y1, x0:x1] = a
        return full.reshape((hc, s, wc, s) + a.shape[2:]).sum(axis=(1, 3))
    # the unknowns: one per frame per cell where the frame covers at least half the cell
    idx, n = {}, 0
    for L in layers:
        cov = to_cells(L.valid.astype(np.float64), L.box) / float(s * s)
        m = -np.ones((hc, wc), np.int64)
        have = cov >= 0.5
        m[have] = np.arange(n, n + int(have.sum()))
        n += int(have.sum())
        idx[L.i] = m
    rows, cols, vals, wts, rhs = [], [], [], [], []
    eq = 0
    for a in range(len(layers)):
        for b in range(a + 1, len(layers)):
            A_, B_ = layers[a], layers[b]
            box = _region(A_, B_)
            if box is None:
                continue
            y0, y1, x0, x1 = box
            sa = (slice(y0 - A_.box[0], y1 - A_.box[0]), slice(x0 - A_.box[2], x1 - A_.box[2]))
            sb = (slice(y0 - B_.box[0], y1 - B_.box[0]), slice(x0 - B_.box[2], x1 - B_.box[2]))
            both = ok[A_.i][sa] & ok[B_.i][sb]
            if both.sum() < 20:
                continue
            d = logs[A_.i][sa] - logs[B_.i][sb]
            res = (fine[A_.i][sa] - fine[B_.i][sb]) - d.mean(-1)
            moved = gaussian_filter(np.abs(np.where(both, res, 0.0)), 2.0) > LEVEL_MOVED
            w = (both & ~binary_dilation(moved, iterations=int(2 * LEVEL_SIGMA))).astype(np.float64)
            if w.sum() < 20:
                continue
            Wc = to_cells(w, box)
            Dc = to_cells(np.clip(d, -LEVEL_LIMIT, LEVEL_LIMIT) * w[..., None], box)
            ia, ib = idx[A_.i], idx[B_.i]
            sel = (Wc > 0.5 * s * s) & (ia >= 0) & (ib >= 0)
            cy, cx = np.nonzero(sel)
            k = len(cy)
            if k == 0:
                continue
            e = np.arange(eq, eq + k)
            rows += [e, e]
            cols += [ia[cy, cx], ib[cy, cx]]
            vals += [np.ones(k), -np.ones(k)]
            wts.append(Wc[cy, cx] / float(s * s))
            rhs.append(-Dc[cy, cx] / Wc[cy, cx][:, None])          # F_a - F_b = -(log_a - log_b)
            eq += k
    if eq == 0:
        return {}
    for L in layers:                                               # smoothness inside each frame
        m = idx[L.i]
        for oy, ox in ((0, 1), (1, 0)):
            p, q = m[:hc - oy, :wc - ox], m[oy:, ox:]
            sel = (p >= 0) & (q >= 0)
            k = int(sel.sum())
            if k == 0:
                continue
            e = np.arange(eq, eq + k)
            rows += [e, e]
            cols += [p[sel], q[sel]]
            vals += [np.ones(k), -np.ones(k)]
            wts.append(np.full(k, LEVEL_SMOOTH))
            rhs.append(np.zeros((k, 3)))
            eq += k
    e = np.arange(eq, eq + n)                                      # a faint pull to 0
    rows.append(e)
    cols.append(np.arange(n))
    vals.append(np.ones(n))
    wts.append(np.full(n, LEVEL_ANCHOR))
    rhs.append(np.zeros((n, 3)))
    eq += n
    Am = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(eq, n))
    wv = np.concatenate(wts)
    bv = np.concatenate(rhs, axis=0)
    AtW = Am.T.multiply(wv[None, :]).tocsr()
    lu = splu((AtW @ Am).tocsc())
    sol = np.stack([lu.solve(AtW @ bv[:, c]) for c in range(3)], -1)
    out = {}
    for L in layers:
        m = idx[L.i]
        Fg = np.zeros((hc, wc, 3), np.float32)
        have = m >= 0
        Fg[have] = sol[m[have]]
        Fg = fill(Fg, have)                                         # beyond the frame's cells: carried smoothly
        y0, y1, x0, x1 = L.box
        gy = (np.arange(y0, y1) + 0.5) / s - 0.5
        gx = (np.arange(x0, x1) + 0.5) / s - 0.5
        GY, GX = np.meshgrid(gy, gx, indexing="ij")
        F = np.stack([map_coordinates(Fg[..., c], [GY, GX], order=1, mode="nearest") for c in range(3)], -1)
        if s > 1:
            F = gaussian_filter(F, (0.5 * s, 0.5 * s, 0))
        out[L.i] = F.astype(np.float32)
    return out


def apply_level(L, F: np.ndarray, plan_box, k: float, du: float, dv: float, rows: int = 256) -> None:
    """Multiply a full-canvas layer by its seam-levelling field, solved on the plan canvas (``k`` plan pixels per
    canvas pixel, offsets ``du``, ``dv`` as in build's labels_at)."""
    from scipy.ndimage import map_coordinates
    y0, y1, x0, x1 = L.box
    xs = (np.arange(x0, x1) + 0.5) * k - du - 0.5 - plan_box[2]
    for r0 in range(0, y1 - y0, rows):
        r1 = min(y1 - y0, r0 + rows)
        ys = (np.arange(y0 + r0, y0 + r1) + 0.5) * k - dv - 0.5 - plan_box[0]
        GY, GX = np.meshgrid(ys, xs, indexing="ij")
        g = np.stack([map_coordinates(F[..., c], [GY, GX], order=1, mode="nearest") for c in range(3)], -1)
        L.rgb[r0:r1] = (np.asarray(L.rgb[r0:r1], np.float32) * np.exp(g)).astype(L.rgb.dtype)


def plan_seams(an: PanoAnalysis, params: PanoParams, factor: float, progress=None, cancel=None) -> dict:
    """Layers at ``factor`` of full size from the small frames, exposure gains, labels (seams), ghosts and the valid
    area: everything decided before the full-size frames are touched."""
    from . import seams as SE
    from .focus import Cancelled
    say = progress or (lambda *_: None)
    cv = make_canvas(an, params, factor)
    leave = set(params.leave_out or [])
    use = [i for i in range(an.n) if an.R[i] is not None and an.paths[i] not in leave]
    layers = []
    for t, i in enumerate(use):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        say(0.1 + 0.4 * t / max(len(use), 1), f"Placing frame {t + 1} of {len(use)}")
        L = warp_layer(i, np.asarray(an.small[i], np.float32), 1.0 / an.small_k, an, cv, order=1)
        if L is not None:
            layers.append(L)
    if not layers:
        raise PanoProblem("No frame could be placed on the canvas.")
    ph = photometric(layers, an.n, vignette=params.exposure) if params.exposure else \
        {"gains": np.ones((an.n, 3)), "vig": None, "corner_ev": 0.0}
    gains = ph["gains"]
    for L in layers:
        corr = 1.0
        if ph["vig"]:
            c1, c2 = ph["vig"]
            corr = np.exp(-(c1 * L.rho2 + c2 * L.rho2 ** 2))[..., None]
        L.rgb = (L.rgb * corr * gains[L.i][None, None, :].astype(np.float32)).astype(np.float32)
    say(0.6, "Finding the seams")
    level = {}
    if params.exposure and params.seam_level:
        say(0.55, "Levelling the overlaps")
        level = seam_levels(layers)
        for L in layers:
            if L.i in level:
                L.rgb = (L.rgb * np.exp(level[L.i])).astype(np.float32)
    lab = SE.voronoi(layers, (cv.H, cv.W))
    if params.seams == "smart":
        lab = SE.refine(lab, layers, params.flare_guard)
    ghost = SE.ghosts(layers, (cv.H, cv.W))
    valid = lab >= 0
    return {"canvas": cv, "layers": layers, "gains": gains, "labels": lab, "ghosts": ghost, "valid": valid,
            "level": level, "plan_boxes": {L.i: L.box for L in layers},
            "factor": factor, "vig": ph["vig"], "corner_ev": ph["corner_ev"],
            "overlap": overlap_width(layers, lab) / max(factor, 1e-9)}


def crop_box(valid: np.ndarray, mode: str) -> tuple:
    from .seams import largest_rectangle
    if mode == "edge":
        ys, xs = np.nonzero(valid)
        return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    k = max(1, int(math.ceil(max(valid.shape) / 1200.0)))
    small = valid[::k, ::k] if k > 1 else valid
    from scipy.ndimage import binary_erosion
    small = binary_erosion(small, iterations=1)
    y0, y1, x0, x1 = largest_rectangle(small)
    return y0 * k, y1 * k, x0 * k, x1 * k


def _region_image(L, y0, y1, x0, x1, filled: np.ndarray):
    """The part (y0:y1, x0:x1) of the canvas from a layer's filled pixels, the edge carried outward beyond its box."""
    by0, by1, bx0, bx1 = L.box
    iy0, iy1 = max(y0, by0), min(y1, by1)
    ix0, ix1 = max(x0, bx0), min(x1, bx1)
    if iy1 <= iy0 or ix1 <= ix0:
        return None
    part = np.asarray(filled[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0], np.float32)
    pad = ((iy0 - y0, y1 - iy1), (ix0 - x0, x1 - ix1), (0, 0))
    return np.pad(part, pad, mode="edge")


def blend_region(layers: list, filled: dict, labels_at, y0: int, y1: int, x0: int, x1: int, levels: int):
    """Multi-band blend of the canvas window (y0:y1, x0:x1). ``labels_at(y0, y1, x0, x1)`` gives the labels there."""
    from .pyramid import blend
    lab = labels_at(y0, y1, x0, x1)
    imgs, masks = [], []
    for L in layers:
        if not np.any(lab == L.i):
            continue
        im = _region_image(L, y0, y1, x0, x1, filled[L.i])
        if im is None:
            continue
        imgs.append(im)
        masks.append((lab == L.i).astype(np.float32))
    if not imgs:
        return np.zeros((y1 - y0, x1 - x0, 3), np.float32), lab
    return blend(imgs, masks, levels), lab


def preview(an: PanoAnalysis, params: PanoParams, progress=None, cancel=None) -> dict:
    """A 1/8-size stitch: {"rgb" (scene-linear, cropped), "full" (uncropped), "labels", "ghosts", "crop", "gains",
    "canvas", "projection", "report"}."""
    from .pyramid import fill
    say = progress or (lambda *_: None)
    sp = plan_seams(an, params, PREVIEW_SCALE, progress, cancel)
    cv, layers, lab = sp["canvas"], sp["layers"], sp["labels"]
    say(0.8, "Blending")
    filled = {L.i: fill(L.rgb.astype(np.float32), L.valid) for L in layers}
    lv = _levels(an, PREVIEW_SCALE, params, sp["overlap"] * PREVIEW_SCALE)
    out, _ = blend_region(layers, filled, lambda a, b, c, d: lab[a:b, c:d], 0, cv.H, 0, cv.W, lv)
    out[~sp["valid"]] = 0
    cb = crop_box(sp["valid"], params.crop)
    y0, y1, x0, x1 = cb
    say(1.0, "Preview ready")
    return {"rgb": out[y0:y1, x0:x1], "full": out, "labels": lab, "ghosts": sp["ghosts"], "crop": cb,
            "gains": sp["gains"], "canvas": cv, "projection": cv.projection, "report": report(an, sp)}


def report(an: PanoAnalysis, sp: dict | None = None) -> dict:
    bad = [p for p in an.pairs if p["rms"] > PAIR_WARN_PX]
    txt = [f"{sum(an.placed)} of {an.n} frames placed, focal length {an.f * 1.0:.0f} px ({an.focal_source})."]
    if an.k1:
        txt.append(f"Lens distortion measured from the overlaps: k1 {an.k1:+.4f}.")
    if bad:
        txt.append("Parallax or a moving subject between: " + ", ".join(
            f"{os.path.basename(an.paths[p['i']])} and {os.path.basename(an.paths[p['j']])} ({p['rms']:.1f} px)"
            for p in bad[:6]) + ".")
    if sp is not None and len(sp["gains"]):
        g = [float(np.mean(sp["gains"][i])) for i in range(an.n) if an.R[i] is not None]
        if g:
            txt.append(f"Exposure evened out: gains {min(g):.2f} to {max(g):.2f}.")
        if sp.get("vig"):
            txt.append(f"Lens vignetting measured from the overlaps and evened out ({sp['corner_ev']:.1f} stop at "
                       "the corners).")
    return {"text": " ".join(txt), "bad_pairs": bad, "colour": "cyan" if not bad else "amber"}


def build(an: PanoAnalysis, params: PanoParams, out_path: str, progress=None, cancel=None, factor: float = 1.0,
          tile: int = 1024) -> dict:
    """The full panorama (``factor`` < 1 builds it smaller), streamed: every frame developed once at full size,
    warped (cubic spline) into its own scratch file, then blended tile by tile into the canvas file, cropped and
    written as a linear intermediate. Returns {"path", "size", "text", ...}."""
    from .. import api
    from ..io.intermediate import write_intermediate
    from ..pipeline.edit import Edit
    from ..pipeline.pipeline import linear_rgb
    from .focus import Cancelled
    from .frames import Scratch
    from .pyramid import fill, margin
    say = progress or (lambda *_: None)

    def check():
        if cancel is not None and cancel.is_set():
            raise Cancelled()
    seam_factor = min(0.25, factor, 3000.0 / max(make_canvas(an, params, factor).W,
                                                 make_canvas(an, params, factor).H, 1) * factor)
    sp = plan_seams(an, params, seam_factor, lambda f, m: say(0.05 * f, m), cancel)
    lab_s = sp["labels"]
    cv = make_canvas(an, params, factor)
    if cv.W * cv.H / 1e6 > MAX_CANVAS_MP:
        raise PanoProblem(f"The panorama would be {cv.W} x {cv.H} pixels ({cv.W * cv.H / 1e6:.0f} MP): too large. "
                          "Build it smaller (Build at 50%) or leave out some frames.")
    k = sp["canvas"].scale / cv.scale                    # seam canvas pixels per full canvas pixel
    du = sp["canvas"].u0 - cv.u0 * k
    dv = sp["canvas"].v0 - cv.v0 * k

    def labels_at(y0, y1, x0, x1):
        ys = np.clip(((np.arange(y0, y1) + 0.5) * k - dv - 0.5 + 0.5).astype(np.int64), 0, lab_s.shape[0] - 1)
        xs = np.clip(((np.arange(x0, x1) + 0.5) * k - du - 0.5 + 0.5).astype(np.int64), 0, lab_s.shape[1] - 1)
        return lab_s[np.ix_(ys, xs)]
    edit = Edit.from_dict(an.edit)
    lv = _levels(an, factor, params, sp["overlap"] * factor)
    with Scratch("pano") as scr:
        layers, filled = [], {}
        use = [L.i for L in sp["layers"]]
        for t, i in enumerate(use):
            check()
            say(0.08 + 0.55 * t / max(len(use), 1), f"Developing and placing frame {t + 1} of {len(use)}")
            img = api.open(an.paths[i])
            rgb = linear_rgb(img, edit, half_size=factor <= 0.5, cache=False)
            img.drop_cache()
            del img
            rgb = _upright(rgb, an.flips[i])
            fs = rgb.shape[1] / float(an.shape[1])
            L = warp_layer(i, rgb, fs, an, cv, order=3, gain=sp["gains"][i], dtype=np.float16, vig=sp["vig"])
            del rgb
            if L is None:
                continue
            if i in sp.get("level", {}):
                apply_level(L, sp["level"][i], sp["plan_boxes"][i], k, du, dv)
            f = scr.array(f"frame{i}", L.rgb.shape, np.float16)
            f[...] = fill(L.rgb.astype(np.float32), L.valid).astype(np.float16)
            filled[i] = f
            L.rgb = None
            layers.append(L)
        check()
        out = scr.array("canvas", (cv.H, cv.W, 3), np.float32)
        M = margin(lv)
        step = max(2 ** lv, tile // (2 ** lv) * (2 ** lv))
        tiles = [(y, x) for y in range(0, cv.H, step) for x in range(0, cv.W, step)]
        for t, (ty, tx) in enumerate(tiles):
            check()
            if t % 4 == 0:
                say(0.65 + 0.3 * t / max(len(tiles), 1), f"Blending: tile {t + 1} of {len(tiles)}")
            y0, y1 = max(0, ty - M), min(cv.H, ty + step + M)
            x0, x1 = max(0, tx - M), min(cv.W, tx + step + M)
            res, lab = blend_region(layers, filled, labels_at, y0, y1, x0, x1, lv)
            res[lab < 0] = 0
            oy, ox = ty - y0, tx - x0
            hh, ww = min(step, cv.H - ty), min(step, cv.W - tx)
            out[ty:ty + hh, tx:tx + ww] = res[oy:oy + hh, ox:ox + ww]
        # crop on the seam canvas, then in full-canvas pixels
        cy0, cy1, cx0, cx1 = crop_box(sp["valid"], params.crop)
        Y0 = int(math.ceil((cy0 + dv) / k)) if params.crop != "edge" else max(0, int((cy0 + dv) / k))
        Y1 = int(math.floor((cy1 + dv) / k)) if params.crop != "edge" else min(cv.H, int(math.ceil((cy1 + dv) / k)))
        X0 = int(math.ceil((cx0 + du) / k)) if params.crop != "edge" else max(0, int((cx0 + du) / k))
        X1 = int(math.floor((cx1 + du) / k)) if params.crop != "edge" else min(cv.W, int(math.ceil((cx1 + du) / k)))
        Y0, X0 = max(0, Y0), max(0, X0)
        Y1, X1 = min(cv.H, max(Y1, Y0 + 1)), min(cv.W, max(X1, X0 + 1))
        check()
        say(0.96, "Writing the panorama")
        info = {"tool": "Panorama", "frames": [os.path.basename(p) for i, p in enumerate(an.paths) if i in use],
                "projection": cv.projection, "focal_px": round(an.f, 1), "focal_source": an.focal_source,
                "seams": params.seams, "flare_guard": params.flare_guard, "crop": params.crop,
                "k1": round(an.k1, 6), "scale": factor}
        meta = None
        try:
            from .focus import _meta_of
            meta = _meta_of(api.open(an.paths[use[0]]))
        except Exception:                                 # noqa: BLE001 - the metadata is a courtesy
            meta = None
        path = write_intermediate(out_path, out[Y0:Y1, X0:X1], meta, info)
    rep = report(an, sp)
    rep.update(path=path, size=(X1 - X0, Y1 - Y0), canvas=(cv.W, cv.H), crop_canvas=(Y0, Y1, X0, X1))
    rep["text"] = f"Saved {os.path.basename(path)} ({X1 - X0} x {Y1 - Y0}). " + rep["text"]
    say(1.0, rep["text"])
    log.info("panorama built: %s %dx%d", path, X1 - X0, Y1 - Y0)
    return rep
