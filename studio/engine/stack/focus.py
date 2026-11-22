"""Focus Stack: a focus bracket merged into one photo sharp from front to back (BUILD_PLAN_TABS section 4).

Pass A (analysis, half size): every frame is developed once at half size with the same edit (frames.stack_edit)
and kept on disk as float16; the frames are lined up neighbour to neighbour (focus breathing changes the scale a
little from frame to frame), put in focus order, evened out in brightness, and each place's sharpness is measured
in every frame with the energy pure noise would give taken off (noise must not look like focus). That gives a
depth map (which frame is sharpest where), its confidence, and the weights the merges need.

Merges, all streaming one frame at a time (preview from the half-size frames on disk; the full-size build decodes
each frame once more):
  depth    each place from the frame where it is sharpest, the two frames either side blended (soft joins);
           places with no detail in any frame (sky, a plain wall) average every frame, which lowers their noise
  average  frames blended by how sharp each is
  pyramid  the strongest detail at every size from any frame (Laplacian pyramid, Burt and Adelson 1983): the
           choice is made on local energy of the luminance, the same for all three colours (no fringes), and
           where no frame has detail above its noise the coefficients are averaged instead of picked (a plain
           per-pixel maximum is what raises noise); the coarsest level is the depth result's (honest colour)
  auto     the pyramid where frames overlap in sharpness (hairs, bristles), the depth result elsewhere
Halo guard (all methods): every output pixel is held within the lowest and highest value any lined-up frame had
in a 5 x 5 neighbourhood of that place: nothing can be brighter or darker than every source, which removes the
halos at depth edges and the pyramid's overshoot.

The result is written as a linear intermediate (io/intermediate.py) and opens in Develop. No Qt here.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading
from dataclasses import asdict, dataclass, field

import numpy as np

from ... import log as slog
from . import align as A
from . import frames as F

log = slog.get("engine.stack.focus")

METHOD_INFO = {
    "auto": {"name": "Auto (default)", "does": "Finest detail where things overlap, sharpest frame elsewhere.",
             "best": "most stacks", "weak": "slowest", "speed": "slow"},
    "depth": {"name": "Sharpest frame per place", "does": "Takes each place from the frame where it is sharpest, "
              "with soft joins.", "best": "simple subjects, clean colour, lowest noise",
              "weak": "hairs and bristles that cross in front of each other", "speed": "fast"},
    "pyramid": {"name": "Finest detail everywhere", "does": "Keeps the strongest detail at every size from any "
                "frame.", "best": "hairs, bristles, overlapping detail",
                "weak": "can raise noise and glare; colours slightly flatter", "speed": "medium"},
    "average": {"name": "Weighted blend", "does": "Blends frames by how sharp each is.",
                "best": "smooth subjects, lowest noise, no hard joins", "weak": "a little softer than the others",
                "speed": "fast"},
    "measured": {"name": "Measured depth (Dual Pixel)", "does": "Takes each place from the frames whose plane of "
                 "focus the measured depth puts it on, not from contrast; the finest-detail pick only where two "
                 "frames are both in focus.", "best": "Dual Pixel stacks: no halos at depth edges (a blurred bright "
                 "edge never wins as 'sharp'); the default for them",
                 "weak": "needs Dual Pixel frames (otherwise works like Auto); plain areas fall back to sharpness",
                 "speed": "slow"},
}
METHODS = tuple(METHOD_INFO)

HONEST = ("Works on a subject that held still. Something that moved between frames (wind on a leaf) shows as a "
          "ghost there: untick the frames where it moved, or use the single frame that has it sharp.")
WHAT_TO_SHOOT = ("A focus bracket: same framing, same exposure, focus stepped from near to far. Tripod or rail. "
                 "5 to 100 frames.")
MAX_GAIN_STOPS = 0.3
DP_TILE, DP_STEP = 16, 4     # Dual Pixel offsets in a stack: much finer than one photo's (48 / 16): the frames
                             # average the noise, and a small window follows a curved surface (docs/RESULTS.md)
PICK_LEVELS = 99             # pyramid: sizes (from the finest) where the strongest detail is picked; coarser
                             # ones come from the depth result (all sizes pick: best on the synthetic tests)
RESIDUAL_WARN_PX = 1.5


class Cancelled(Exception):
    """The person pressed Cancel."""


@dataclass
class FocusParams:
    method: str = "auto"
    denoise_first: bool = False
    align: str = "similarity"
    equalise: bool = True
    radius: float = 1.0          # detail size: scales the sharpness filters
    smooth: float = 1.0          # smoothness of joins: scales the depth map's edge-aware smoothing
    halo_guard: bool = True
    order: str = "auto"          # auto (by magnification, else capture order) | capture | reverse
    local: bool = True           # local flow after the global line-up (hand-held stacks; harmless on a tripod)
    retouch: list = field(default_factory=list)   # [{"frame": i, "points": [[u, v], ...], "radius": r}] (0..1)
    dual_pixel: str = "auto"     # auto: when every frame is a Dual Pixel RAW, each frame's two halves are measured
                                 # too and depth is measured two ways (Depth view, surfaces); off: never
    auto_measured: bool = True   # Auto on a Dual Pixel stack merges from the measured depth (method "measured")
    moving_single: bool = True   # take moving parts from one frame: where the measures disagree (something moved
                                 # between frames), the place comes from the single frame that has it sharpest
                                 # instead of a blend of frames (a double leaf); needs Dual Pixel frames

    def analysis_key(self) -> tuple:
        """Settings that change the analysis (anything else only changes the merge)."""
        return (self.denoise_first, self.align, self.equalise, round(float(self.radius), 3), self.order,
                bool(self.local), self.dual_pixel)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "FocusParams":
        return FocusParams(**{k: v for k, v in (d or {}).items() if k in FocusParams.__dataclass_fields__})


@dataclass
class FrameReport:
    name: str
    path: str
    used: bool = True
    status: str = "ok"           # ok (cyan) | warn (amber) | dropped (red)
    reason: str = ""
    residual_px: float = 0.0
    gain_stops: float = 0.0
    scale: float = 1.0
    moved_share: float = 0.0


@dataclass
class FocusAnalysis:
    paths: list                  # in focus order (only frames that are used)
    reports: list                # FrameReport per frame, same order as ``paths``, then any dropped ones
    ref: int                     # index (in ``paths``) of the reference frame
    t_half: list                 # Transform per frame, reference half-size coords -> frame half-size coords
    t_full: list                 # the same at full size
    f_half: list                 # local flow per frame (reference coordinates, half size), or None
    f_full: list
    gains: list
    box_full: tuple              # (y0, y1, x0, x1) of the common area, full size, camera-crop frame
    box_half: tuple
    shape_full: tuple            # (h, w) of a full-size frame (camera crop)
    flip: int
    edit: dict
    meta: dict
    params: FocusParams
    aligned: np.ndarray          # memmap (N, bh, bw, 3) float16: lined-up half-size frames, gains applied
    energy: np.ndarray           # memmap (N, bh, bw) float16: noise-floored sharpness per frame
    smax: np.ndarray             # (bh, bw) float32
    ssec: np.ndarray
    idx: np.ndarray              # (bh, bw) int16: sharpest frame
    noise_e: np.ndarray          # (bh, bw) float32: energy that noise alone gives at the reference's brightness
    guide: np.ndarray            # (bh, bw) float32: perceptual luminance of the reference (edges for smoothing)
    scratch: F.Scratch
    notes: list = field(default_factory=list)
    cropped_share: float = 0.0
    ref_noise: tuple = (0.0, 0.0)
    _depth: dict = field(default_factory=dict)
    dual_pixel: bool = False     # every frame's Dual Pixel offsets were measured (dp_disp, dp_conf)
    dp_disp: np.ndarray | None = None   # memmap (N, bh, bw) float16: offset between the halves, full-size px of the
                                        # reference, lined up like the frames (depthfuse.py)
    dp_conf: np.ndarray | None = None   # memmap (N, bh, bw) float16: 0..1
    dp_notes: list = field(default_factory=list)
    dp_frames: list = field(default_factory=list)   # per frame (focus order): its Dual Pixel offsets were measured
    from_cache: bool = False                        # the analysis came from the analysis cache (A.4)

    @property
    def n(self) -> int:
        return len(self.paths)

    def close(self):
        self.scratch.close()

    # ------------------------------------------------------------------ maps (half size, box frame)
    def presence(self) -> np.ndarray:
        """0..1: how sure we are that some frame has real detail here (0 = flat in every frame)."""
        from scipy.ndimage import gaussian_filter
        p = np.clip(self.smax / (2.0 * self.noise_e + 1e-12), 0.0, 1.0)
        return gaussian_filter(p.astype(np.float32), 2.0)

    def confidence(self) -> np.ndarray:
        return np.clip((self.smax - self.ssec) / (self.smax + self.noise_e + 1e-12), 0.0, 1.0).astype(np.float32)

    def depth(self, smooth: float | None = None) -> np.ndarray:
        """Fractional depth (0 .. N-1, which frame is sharpest), smoothed along the picture's edges so places with
        little confidence take their depth from confident neighbours."""
        smooth = float(self.params.smooth if smooth is None else smooth)
        key = round(smooth, 3)
        if key not in self._depth:
            d0 = self.idx.astype(np.float32)
            if smooth <= 0:
                d = d0
            else:
                w = (self.confidence() * self.presence() + 1e-4).astype(np.float32)
                r = max(2, int(round(8 * smooth)))          # 16 full-size pixels at smooth 1
                d = A.weighted_guided(self.guide, d0, w, r, 1e-3)
            self._depth = {key: np.clip(d, 0, self.n - 1).astype(np.float32)}
        return self._depth[key]

    def mix(self) -> np.ndarray:
        """auto: 1 where frames overlap in sharpness (use the pyramid), 0 elsewhere (use the depth result)."""
        from scipy.ndimage import gaussian_filter
        r = self.ssec / (self.smax + self.noise_e + 1e-12)
        m = _smoothstep(0.55, 0.8, r) * self.presence()
        return np.clip(gaussian_filter(m.astype(np.float32), 4.0), 0.0, 1.0)


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------------------ sharpness
_NOISE_C: dict = {}


def _energy_raw(L: np.ndarray, radius: float) -> np.ndarray:
    """Scale-normalised Laplacian-of-Gaussian energy at two sizes, smoothed: how much fine detail is here."""
    from scipy.ndimage import gaussian_filter, gaussian_laplace
    s1, s2 = 1.0 * radius, 2.0 * radius
    a = gaussian_laplace(L, s1) * np.float32(s1 * s1)
    e = a * a
    b = gaussian_laplace(L, s2) * np.float32(s2 * s2)
    e += b * b
    return gaussian_filter(e, 4.0 * radius).astype(np.float32)


def _noise_factor(radius: float) -> float:
    """Energy that unit white noise gives through _energy_raw (measured once per size)."""
    key = round(float(radius), 3)
    if key not in _NOISE_C:
        n = np.random.default_rng(11).standard_normal((512, 512)).astype(np.float32)
        e = _energy_raw(n, radius)
        m = int(16 * radius) + 8
        _NOISE_C[key] = float(e[m:-m, m:-m].mean())
    return _NOISE_C[key]


def noise_curve(L: np.ndarray) -> tuple[float, float]:
    """(a, b): pixel noise variance of a luminance image as a * L + b, measured from the image itself (median of
    the squared fine differences in brightness bins; most of a focus-bracket frame is out of focus, so the
    median sees noise, not detail)."""
    d = (4 * L[1:-1, 1:-1] - L[:-2, 1:-1] - L[2:, 1:-1] - L[1:-1, :-2] - L[1:-1, 2:]) / np.float32(np.sqrt(20.0))
    v = L[1:-1, 1:-1]
    k = max(1, int(np.sqrt(v.size / 400_000)))
    d, v = d[::k, ::k].ravel(), v[::k, ::k].ravel()
    ok = (v > 0.002) & (v < 0.95)
    if ok.sum() < 1000:
        s = float(1.4826 * np.median(np.abs(d)))
        return 0.0, s * s
    d, v = d[ok], v[ok]
    qs = np.quantile(v, np.linspace(0, 1, 13))
    xs, ys = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (v >= lo) & (v <= hi)
        if m.sum() > 200:
            xs.append(float(np.median(v[m])))
            ys.append(float(np.median(d[m] ** 2)) / 0.4549)     # median of chi-square(1) = 0.4549
    if len(xs) < 3:
        s = float(np.median(ys)) if ys else 1e-6
        return 0.0, s
    a, b = np.polyfit(xs, ys, 1)
    if a < 0:
        a, b = 0.0, float(np.median(ys))
    return float(max(a, 0.0)), float(max(b, 1e-10))


def sharpness(L: np.ndarray, radius: float, noise=None) -> tuple[np.ndarray, np.ndarray]:
    """(S, Nf): sharpness with the noise energy taken off (floored at 0), and the noise energy itself."""
    from scipy.ndimage import gaussian_filter
    a, b = noise if noise is not None else noise_curve(L)
    e = _energy_raw(L, radius)
    Ls = gaussian_filter(L, 4.0 * radius)
    nf = (np.float32(_noise_factor(radius)) * (np.float32(a) * np.maximum(Ls, 0) + np.float32(b))).astype(np.float32)
    return np.maximum(e - 1.5 * nf, 0.0).astype(np.float32), nf


# ------------------------------------------------------------------------------------------ helpers
def _tick(progress, f, msg):
    if progress is not None:
        try:
            progress(float(min(max(f, 0.0), 1.0)), msg)
        except Exception:                            # noqa: BLE001  (a progress display must never stop the job)
            pass


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _with_origin(T: A.Transform, y0: float, x0: float) -> A.Transform:
    """T applied to box coordinates: p_box -> T(p_box + origin)."""
    return T.compose(A.Transform.translation(y0, x0))


def _flow_in_box(fl, y0: float, x0: float):
    """A reference-frame flow for warping into box coordinates (p_box + origin)."""
    if fl is None:
        return None
    return A.Flow(fl.dy, fl.dx, fl.quality, fl.y0 - y0, fl.x0 - x0, fl.step)


def _even_box(box, shape):
    y0, y1, x0, x1 = box
    y0, x0 = (y0 + 1) // 2 * 2, (x0 + 1) // 2 * 2
    y1, x1 = y0 + (y1 - y0) // 2 * 2, x0 + (x1 - x0) // 2 * 2
    return int(y0), int(min(y1, shape[0] // 2 * 2)), int(x0), int(min(x1, shape[1] // 2 * 2))


def _meta_of(img) -> dict:
    md = img.meta
    return {"make": md.make, "model": md.model, "iso": md.iso, "exposure_s": md.exposure_s, "fnumber": md.fnumber,
            "focal_mm": md.focal_mm, "lens": md.lens, "datetime": md.datetime,
            "focus_distance_m": getattr(md, "focus_distance_m", None)}


def _up2(m: np.ndarray, shape) -> np.ndarray:
    """A half-size map brought to full size (bilinear, pixel areas aligned: full (Y, X) sits at half
    ((Y - 0.5) / 2, (X - 0.5) / 2))."""
    from scipy.ndimage import map_coordinates
    H, W = shape[:2]
    out = np.empty((H, W) + m.shape[2:], np.float32)
    xs = (np.arange(W, dtype=np.float64) - 0.5) / 2.0
    chans = [m] if m.ndim == 2 else [m[..., c] for c in range(m.shape[2])]
    for y in range(0, H, 512):
        ys = (np.arange(y, min(H, y + 512), dtype=np.float64) - 0.5) / 2.0
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        for c, ch in enumerate(chans):
            r = map_coordinates(np.asarray(ch, np.float32), [Y, X], order=1, mode="nearest")
            if m.ndim == 2:
                out[y:y + 512] = r
            else:
                out[y:y + 512, :, c] = r
    return out


# ------------------------------------------------------------------------------------------ pass A
def analyse(paths, params: FocusParams | None = None, progress=None, cancel=None, cache: bool = False,
            workers: int | None = None) -> FocusAnalysis:
    """Pass A at half size: develop, line up, order, even out, measure sharpness. Raises ValueError with a plain
    reason when the frames cannot make a stack, Cancelled on Cancel. Close the result (``.close()``) when done.

    ``cache``: reuse an earlier analysis of the same frames with the same settings (BUILD_PLAN_DP3D A.4), and keep
    this one for next time. ``workers``: cap on frames developed at once (default: frames.stack_workers())."""
    params = params or FocusParams()
    infos = F.probe(paths)
    msgs, block = F.check(infos)
    if block:
        raise F.StackProblem(" ".join(msgs))
    usable = [f for f in infos if f.ok]
    dropped = [FrameReport(f.name, f.path, False, "dropped", f.problem) for f in infos if not f.ok]
    if params.order == "reverse":
        usable = usable[::-1]
    key = None
    if cache:
        from . import anacache as AC
        key = AC.key_for([f.path for f in usable], params.analysis_key())
        if AC.has(key):
            hit = _from_cache(key, params, msgs, dropped)
            if hit is not None:
                _tick(progress, 1.0, "Analysis from the cache")
                log.info("focus analysis from the cache (%s): %d frames", key, hit.n)
                return hit
    dp_flags = [params.dual_pixel != "off" and F.is_dual(f.path) for f in usable]
    use_dp = len(usable) >= 2 and sum(dp_flags) >= 2
    if not use_dp:
        dp_flags = [False] * len(usable)
    n = len(usable)
    m = n // 2
    if key is not None:
        from . import anacache as AC
        scratch = AC.new_entry(key)
    else:
        scratch = F.Scratch("focus")
    try:
        _tick(progress, 0.0, "Opening the reference frame")
        ref_img = F.open_frame(usable[m].path)
        edit = F.stack_edit(ref_img, params.denoise_first)
        flip = int(getattr(ref_img.raw, "flip", 0))
        meta = _meta_of(ref_img)
        del ref_img
        dstate = _dual_state(meta, use_dp)
        dev = _Developer(usable, dp_flags, edit, dstate, scratch, n)
        # the reference first (its size decides the arrays, and a Dual Pixel stack fits its shading on it), then
        # the rest, several at once on a PC with the cores and memory for it
        order_dev = [m] + [i for i in range(n) if i != m]
        nw = F.stack_workers(limit=workers) if use_dp else 1
        done = 0
        for i, ok in dev.run(order_dev, nw, cancel):
            done += 1
            _tick(progress, 0.05 + 0.4 * done / n, f"Developing frame {done} of {n} at half size" +
                  (" and measuring its Dual Pixel halves" if dp_flags[i] else ""))
        _check(cancel)
        dropped += dev.dropped
        keep = [usable[i] for i in range(n) if dev.ok[i]]
        slot = [i for i in range(n) if dev.ok[i]]
        lum = [dev.lum[i] for i in slot]
        kdp = [dp_flags[i] and dev.has_dp[i] for i in slot]
        raw_mm, shape_h = dev.raw_mm, dev.shape_h
        if len(keep) < 2:
            raise F.StackProblem("Fewer than two frames could be developed: " +
                                 "; ".join(f"{r.name} ({r.reason})" for r in dropped))
        n = len(keep)
        m = n // 2
        k_a = max(1, int(np.ceil(max(shape_h) / 2000.0)))
        kind = params.align
        use_dp = use_dp and sum(kdp) >= 2

        # ---- line up neighbour to neighbour, in capture order first
        _tick(progress, 0.47, "Lining the frames up")
        order = list(range(n))
        T_ana, res, outl, fl_ana, pairs = _chain(lum, order, kind, cancel)
        scales = [T.scale() for T in T_ana]
        if params.order == "auto" and n >= 3:
            sp = max(scales) - min(scales)
            by_scale = sorted(range(n), key=lambda j: scales[j])
            if scales[0] > scales[-1]:
                by_scale = by_scale[::-1]                # keep the direction the frames were shot in
            if sp > 5e-4 and by_scale != order:
                log.info("focus: frames put in focus order by magnification (spread %.3f%%)", 100 * sp)
                order = by_scale
                T_ana, res, outl, fl_ana, pairs = _chain(lum, order, kind, cancel)
                scales = [T.scale() for T in T_ana]
        if params.local:                             # the order is settled: line up locally too
            _tick(progress, 0.48, "Lining the frames up locally")
            T_ana, res, outl, fl_ana, pairs = _chain(lum, order, kind, cancel, True, pairs)
        keep = [keep[j] for j in order]
        slot = [slot[j] for j in order]
        kdp = [kdp[j] for j in order]
        lum = [lum[j] for j in order]
        perm = order
        m = n // 2
        t_half = [T.rescaled(k_a) for T in T_ana]
        t_full = [T.rescaled(2) for T in t_half]
        f_half = [f.rescaled(k_a) if f is not None else None for f in fl_ana]
        f_full = [f.rescaled(2) if f is not None else None for f in f_half]
        shape_f = (shape_h[0] * 2, shape_h[1] * 2)
        box = _even_box(A.common_box(t_full, shape_f, shape_f, margin=4), shape_f)
        box_h = (box[0] // 2, box[1] // 2, box[2] // 2, box[3] // 2)
        bh, bw = box_h[1] - box_h[0], box_h[3] - box_h[2]
        cropped = 1.0 - (bh * bw) / float(shape_h[0] * shape_h[1])

        # ---- warp each half-size frame into the reference's frame, gains, sharpness
        energy = scratch.array("energy", (n, bh, bw), np.float16)
        aligned = scratch.array("aligned", (n, bh, bw, 3), np.float16)
        dpd = scratch.array("dp_disp", (n, bh, bw), np.float16) if use_dp else None
        dpc = scratch.array("dp_conf", (n, bh, bw), np.float16) if use_dp else None
        dp_notes: list = list(dev.notes)
        smax = np.zeros((bh, bw), np.float32)
        ssec = np.zeros((bh, bw), np.float32)
        idx = np.zeros((bh, bw), np.int16)
        reports = []
        gains = [1.0] * n
        sequence = [m] + [j for j in range(n) if j != m]  # the reference first: the others are evened out to it
        ref_lp = None
        noise_e = None
        guide = None
        ref_noise = None
        for step, j in enumerate(sequence):
            _check(cancel)
            _tick(progress, 0.5 + 0.45 * step / n, f"Measuring sharpness: frame {step + 1} of {n}")
            src = np.asarray(raw_mm[slot[j]], np.float32)
            al = A.warp(src, _with_origin(t_half[j], box_h[0], box_h[2]), (bh, bw), order=3,
                        flw=_flow_in_box(f_half[j], box_h[0], box_h[2]))
            del src
            L = A.luma(al)
            g = 1.0
            clamped = False
            if params.equalise:
                from scipy.ndimage import gaussian_filter
                lp = gaussian_filter(L, 16.0)
                if ref_lp is None:
                    ref_lp = lp
                else:
                    ok = (lp > 0.01) & (lp < 0.9) & (ref_lp > 0.01) & (ref_lp < 0.9)
                    if ok.sum() > 100:
                        g = float(_trimmed(ref_lp[ok]) / max(_trimmed(lp[ok]), 1e-9))
                    lim = 2.0 ** MAX_GAIN_STOPS
                    clamped = g > lim or g < 1 / lim
                    g = float(np.clip(g, 1 / lim, lim))
                    if abs(g - 1.0) > 1e-4:
                        al *= np.float32(g)
                        L *= np.float32(g)
            gains[j] = g
            aligned[j] = al.astype(np.float16)         # slot j = frame j in focus order
            del al
            nc = noise_curve(L)
            S, nf = sharpness(L, params.radius, nc)
            if j == m:
                ref_noise = nc
                noise_e = nf
                guide = np.sqrt(np.clip(L / max(float(np.percentile(L, 99.5)), 1e-6), 0, 1)).astype(np.float32)
            energy[j] = S.astype(np.float16)
            if use_dp:
                if kdp[j]:
                    _warp_offsets(dev, slot[j], t_half[j], f_half[j], box_h, (bh, bw), dpd, dpc, j)
                else:
                    dpd[j] = np.float16(0)
                    dpc[j] = np.float16(0)
            better = S > smax
            ssec = np.where(better, smax, np.maximum(ssec, S))
            idx = np.where(better, np.int16(j), idx)
            smax = np.where(better, S, smax)
            del S, L
            rep = FrameReport(keep[j].name, keep[j].path)
            rep.residual_px = float(res[j] * k_a * 2)
            rep.moved_share = float(outl[j])
            rep.gain_stops = float(np.log2(g))
            rep.scale = float(t_full[j].scale())
            why = []
            if j != m and rep.residual_px > RESIDUAL_WARN_PX:
                why.append(f"did not line up well ({rep.residual_px:.1f} px)")
            if rep.moved_share > 0.03:
                why.append(f"something moved in about {100 * rep.moved_share:.0f}% of the frame")
            if clamped:
                why.append(f"much {'darker' if g > 1 else 'brighter'} than the others (evened out by "
                           f"{MAX_GAIN_STOPS} stop only)")
            if why:
                rep.status, rep.reason = "warn", "; ".join(why)
            reports.append((j, rep))
        aligned.flush()
        dev.release()
        reports = [r for _, r in sorted(reports, key=lambda t: t[0])] + dropped
        notes = list(msgs)
        n_dp = int(sum(kdp))
        if use_dp:
            dpd.flush()
            dpc.flush()
            if n_dp == n:
                notes.append("Every frame is a Dual Pixel RAW: depth is measured two ways (Depth view).")
            else:
                notes.append(f"Dual Pixel on {n_dp} of {n} frames: depth is measured two ways where those frames "
                             f"are in focus, from focus alone elsewhere.")
            if dstate is not None:
                dp_notes.extend(_shading_notes(dstate))
        if cropped > 0.005:
            notes.append(f"The edges move between frames: {100 * cropped:.1f}% of the frame is cropped so every "
                         f"place has every frame.")
        an = FocusAnalysis(paths=[k.path for k in keep], reports=reports, ref=m, t_half=t_half, t_full=t_full,
                           f_half=f_half, f_full=f_full, gains=gains, box_full=box, box_half=box_h, shape_full=shape_f,
                           flip=flip, edit=edit.to_dict(), meta=meta, params=FocusParams.from_dict(params.to_dict()),
                           aligned=aligned, energy=energy, smax=smax, ssec=ssec, idx=idx, noise_e=noise_e,
                           guide=guide, scratch=scratch, notes=notes, cropped_share=cropped, ref_noise=ref_noise,
                           dual_pixel=bool(use_dp), dp_disp=dpd, dp_conf=dpc, dp_notes=dp_notes)
        an.dp_frames = [bool(x) for x in kdp]
        if dstate is not None and use_dp:
            _remember_shading(meta, dstate)
        if key is not None:
            _to_cache(an, key)
        _tick(progress, 1.0, "Analysis done")
        log.info("focus analysis: %d frames, order %s, box %s, cropped %.1f%%, gains %s, dual pixel %d", n,
                 "by magnification" if perm != list(range(n)) else "capture", box, 100 * cropped,
                 np.round(np.log2(gains), 3).tolist(), n_dp)
        return an
    except BaseException:
        if hasattr(scratch, "discard"):
            scratch.discard()
        else:
            scratch.close()
        raise


class _Developer:
    """Pass A's first part: every frame developed at half size (and, for Dual Pixel frames, its halves measured
    from the same decode), the reference first, the rest several at once when the PC allows (threads: NumPy, SciPy
    and LibRaw do the work outside Python's lock; every frame writes only its own slot of the arrays on disk)."""

    def __init__(self, usable, dp_flags, edit, dstate, scratch, n):
        self.usable, self.dp_flags, self.edit, self.dstate, self.scratch = usable, dp_flags, edit, dstate, scratch
        self.n = n
        self.ok = [False] * n
        self.has_dp = [False] * n
        self.lum = [None] * n
        self.dropped: list = []
        self.notes: list = []
        self.raw_mm = None
        self.dp_d = None
        self.dp_c = None
        self.shape_h = None
        import threading
        self._lock = threading.Lock()

    def _one(self, i: int):
        fi = self.usable[i]
        la = lb = None
        try:
            if self.dp_flags[i]:
                try:
                    rgb, la, lb = F.develop_and_halves(fi.path, self.edit, self.dstate)
                except Exception as e:               # noqa: BLE001  (the halves failed: the frame still counts)
                    if isinstance(e, Cancelled):
                        raise
                    self._dp_failed(fi, e)
                    rgb = F.scene_rgb(fi.path, self.edit, True)
            else:
                rgb = F.scene_rgb(fi.path, self.edit, True)
        except Exception as e:                       # noqa: BLE001  (an unreadable frame is left out, named)
            log.info("focus: %s left out (%s)", fi.name, e)
            with self._lock:
                self.dropped.append(FrameReport(fi.name, fi.path, False, "dropped", _short(e)))
            return i, False
        with self._lock:
            if self.shape_h is None:
                self.shape_h = rgb.shape[:2]
                self.raw_mm = self.scratch.array("half", (self.n,) + rgb.shape, np.float16)
                if any(self.dp_flags):
                    self.dp_d = self.scratch.array("dp_raw_d", (self.n,) + rgb.shape[:2], np.float16)
                    self.dp_c = self.scratch.array("dp_raw_c", (self.n,) + rgb.shape[:2], np.float16)
        if rgb.shape[:2] != self.shape_h:
            with self._lock:
                self.dropped.append(FrameReport(fi.name, fi.path, False, "dropped",
                                                "a different size from the others"))
            return i, False
        self.raw_mm[i] = rgb.astype(np.float16)
        ana, _ = A.analysis_copy(A.luma(rgb), 2000)
        del rgb
        if la is not None:
            try:
                from ..dualpixel import depth as DEP
                if la.shape != tuple(self.shape_h):
                    raise F.StackProblem(f"its halves are {la.shape[1]} x {la.shape[0]}, the frame "
                                         f"{self.shape_h[1]} x {self.shape_h[0]}")
                d, c = DEP.disparity(la, lb, DP_TILE, DP_STEP)
                self.dp_d[i] = d.astype(np.float16)
                self.dp_c[i] = np.clip(c, 0.0, 1.0).astype(np.float16)
                self.has_dp[i] = True
            except Exception as e:                   # noqa: BLE001  (one frame without offsets: the rest work)
                self._dp_failed(fi, e)
        self.lum[i] = ana
        self.ok[i] = True
        return i, True

    def _dp_failed(self, fi, e):
        with self._lock:
            self.notes.append(f"{fi.name}: its Dual Pixel halves could not be measured ({_short(e)}); depth there "
                              f"comes from the other frames.")
        if not (slog.expected_problem(e) or isinstance(e, F.StackProblem)):
            slog.log_exception(f"Dual Pixel halves of {fi.name}", log)
            slog.save_error_report(f"Focus Stack: measuring the Dual Pixel halves of {fi.name}")
        else:
            log.info("focus: dual pixel halves of %s not measured (%s)", fi.name, e)

    def run(self, order: list, workers: int, cancel):
        """Yields (frame index, developed?) as frames finish (in the calling thread: progress and Cancel there)."""
        first, rest = order[0], order[1:]
        _check(cancel)
        yield self._one(first)
        if workers <= 1 or len(rest) <= 1:
            for i in rest:
                _check(cancel)
                yield self._one(i)
            return
        from concurrent.futures import ThreadPoolExecutor, as_completed
        ex = ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="focus-dev")
        try:
            futs = [ex.submit(self._guarded, i, cancel) for i in rest]
            for f in as_completed(futs):
                r = f.result()
                if r is None:
                    continue
                yield r
                _check(cancel)
        finally:
            ex.shutdown(wait=True, cancel_futures=True)

    def _guarded(self, i, cancel):
        if cancel is not None and cancel.is_set():
            return None
        return self._one(i)

    def release(self):
        for a in (self.raw_mm, self.dp_d, self.dp_c):
            if a is not None:
                self.scratch.drop(a)
        self.raw_mm = self.dp_d = self.dp_c = None


def _warp_offsets(dev: _Developer, i: int, t_half, f_half, box_h, shape_box, dpd, dpc, j: int):
    """Frame j's Dual Pixel offset and confidence (measured on its own halves in pass A's first part), lined up
    exactly like the frame (the offsets turned into the reference's pixels: divided by the frame's
    magnification)."""
    d = np.asarray(dev.dp_d[i], np.float32)
    c = np.asarray(dev.dp_c[i], np.float32)
    T = _with_origin(t_half, box_h[0], box_h[2])
    fl = _flow_in_box(f_half, box_h[0], box_h[2])
    dw = A.warp(d, T, shape_box, order=1, flw=fl) / np.float32(max(t_half.scale(), 1e-6))
    cw = A.warp(c, T, shape_box, order=1, flw=fl)
    dpd[j] = dw.astype(np.float16)
    dpc[j] = np.clip(cw, 0.0, 1.0).astype(np.float16)


def _dual_state(meta: dict, use_dp: bool):
    """The shading shared by the stack's Dual Pixel frames: from the camera's Dual Pixel profile when it has this
    lens setting (fitted on an earlier stack: cyan), else fitted on the reference frame."""
    if not use_dp:
        return None
    try:
        from ..dualpixel import dpprofile
        prof = dpprofile.find(meta.get("model") or "")
        key = dpprofile.lens_key(meta.get("lens"), meta.get("focal_mm"), meta.get("fnumber"))
        if prof is not None and key in prof.shading:
            e = prof.shading[key]
            shp = tuple(e.get("shape", ()))
            g = prof.gains_for(key, shp)
            if g is not None:
                return F.DualState(g, float(e.get("level_ratio") or 2.0),
                                   f"the camera's Dual Pixel profile ({prof.origin}, {e.get('date', '')})")
    except Exception:                                # noqa: BLE001  (a profile is a speed-up only)
        slog.log_exception("dual pixel shading from the profile", log)
    return F.DualState()


def _shading_notes(st) -> list:
    out = []
    if st.refits:
        out.append(f"Dual Pixel shading: fitted once ({st.source}) and reused; {len(st.refits)} frame(s) drifted "
                   f"more than {100 * F.DRIFT:.0f}% in brightness balance and got their own fit.")
    if st.ratios:
        r = np.asarray(st.ratios)
        log.info("dual pixel level ratios across the stack: %.4f .. %.4f (spread %.2f%%), shading %s", r.min(),
                 r.max(), 100 * (r.max() - r.min()) / max(r.mean(), 1e-9), st.source)
    return out


def _remember_shading(meta: dict, st):
    """A shading fitted here is saved to this PC's Dual Pixel profile for the lens setting (A.3.3: later stacks
    with the same lens, focal length and aperture skip the fit)."""
    if not st.fitted_here or st.gains is None:
        return
    try:
        from ..dualpixel import dpprofile
        model = meta.get("model") or ""
        if not model:
            return
        prof = dpprofile.local_or_new(model)
        key = dpprofile.lens_key(meta.get("lens"), meta.get("focal_mm"), meta.get("fnumber"))
        prof.remember_gains(key, st.gains, st.level_ratio or 2.0, 1)
        prof.save()
    except OSError as e:                             # a read-only folder: only a speed-up is lost
        log.info("dual pixel shading not saved to the profile (%s)", e)
    except Exception:                                # noqa: BLE001
        slog.log_exception("saving the dual pixel shading", log)


# ------------------------------------------------------------------------------------------ the analysis cache
def _to_cache(an: FocusAnalysis, key: str):
    """Keep this analysis in the cache (A.4): the big arrays are already in the cache folder; the small fields go
    in small.npz and meta.json."""
    from . import anacache as AC
    try:
        small = {"smax": an.smax, "ssec": an.ssec, "idx": an.idx, "noise_e": an.noise_e, "guide": an.guide,
                 "gains": np.asarray(an.gains, np.float64), "ref_noise": np.asarray(an.ref_noise, np.float64),
                 "dp_frames": np.asarray(getattr(an, "dp_frames", []), bool)}
        for nm, ts in (("t_half", an.t_half), ("t_full", an.t_full)):
            small[nm] = np.stack([t.matrix for t in ts])
        tmeta = [{"kind": t.kind, "residual_px": float(t.residual_px), "tiles_used": int(t.tiles_used or 0),
                  "outliers": float(getattr(t, "outliers", 0.0) or 0.0)} for t in an.t_half]
        for nm, fl in (("f_half", an.f_half), ("f_full", an.f_full)):
            for j, f in enumerate(fl):
                if f is not None:
                    small[f"{nm}_{j}_dy"], small[f"{nm}_{j}_dx"], small[f"{nm}_{j}_q"] = f.dy, f.dx, f.quality
                    small[f"{nm}_{j}_geo"] = np.array([f.y0, f.x0, f.step], np.float64)
        meta = {"paths": an.paths, "reports": [asdict(r) for r in an.reports], "ref": an.ref,
                "box_full": list(an.box_full), "box_half": list(an.box_half), "shape_full": list(an.shape_full),
                "flip": an.flip, "edit": an.edit, "meta": an.meta, "params": an.params.to_dict(),
                "notes": an.notes, "cropped_share": an.cropped_share, "dual_pixel": an.dual_pixel,
                "dp_notes": an.dp_notes, "transform_meta": tmeta}
        AC.finish(an.scratch, key, json.loads(json.dumps(meta, default=_plainj)), small)
    except Exception:                                # noqa: BLE001  (the cache is a speed-up; the analysis stands)
        slog.log_exception("keeping the focus analysis in the cache", log)


def _plainj(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _from_cache(key: str, params: FocusParams, msgs: list, dropped: list):
    from . import anacache as AC
    got = AC.load(key)
    if got is None:
        return None
    sc, meta, small = got
    try:
        tm = meta.get("transform_meta") or []

        def transforms(nm):
            out = []
            for j, M in enumerate(small[nm]):
                t = tm[j] if j < len(tm) else {}
                out.append(A.Transform(np.asarray(M, np.float64), t.get("kind", params.align),
                                       t.get("residual_px", 0.0), t.get("tiles_used", 0), t.get("outliers", 0.0)))
            return out

        def flows(nm, n):
            out = []
            for j in range(n):
                if f"{nm}_{j}_dy" in small:
                    g = small[f"{nm}_{j}_geo"]
                    out.append(A.Flow(small[f"{nm}_{j}_dy"], small[f"{nm}_{j}_dx"], small[f"{nm}_{j}_q"],
                                      float(g[0]), float(g[1]), float(g[2])))
                else:
                    out.append(None)
            return out
        n = len(meta["paths"])
        dp = bool(meta.get("dual_pixel"))
        reps = [FrameReport(**r) for r in meta["reports"]]
        an = FocusAnalysis(paths=list(meta["paths"]), reports=reps, ref=int(meta["ref"]), t_half=transforms("t_half"),
                           t_full=transforms("t_full"), f_half=flows("f_half", n), f_full=flows("f_full", n),
                           gains=[float(g) for g in small["gains"]], box_full=tuple(meta["box_full"]),
                           box_half=tuple(meta["box_half"]), shape_full=tuple(meta["shape_full"]),
                           flip=int(meta["flip"]), edit=meta["edit"], meta=meta["meta"],
                           params=FocusParams.from_dict(params.to_dict()), aligned=sc.open("aligned"),
                           energy=sc.open("energy"), smax=small["smax"], ssec=small["ssec"], idx=small["idx"],
                           noise_e=small["noise_e"], guide=small["guide"], scratch=sc,
                           notes=list(meta.get("notes", [])) + ["From the analysis cache (the same frames and "
                                                                "settings were analysed before)."],
                           cropped_share=float(meta.get("cropped_share", 0.0)),
                           ref_noise=tuple(float(x) for x in small["ref_noise"]), dual_pixel=dp,
                           dp_disp=sc.open("dp_disp") if dp else None, dp_conf=sc.open("dp_conf") if dp else None,
                           dp_notes=list(meta.get("dp_notes", [])))
        an.dp_frames = [bool(x) for x in small.get("dp_frames", np.zeros(0, bool))]
        an.from_cache = True
        return an
    except Exception:                                # noqa: BLE001  (an unreadable entry: analyse again)
        slog.log_exception(f"analysis cache entry {key}", log)
        sc.close()
        return None


def _trimmed(v: np.ndarray) -> float:
    lo, hi = np.percentile(v, [10, 90])
    s = v[(v >= lo) & (v <= hi)]
    return float(s.mean()) if s.size else float(np.mean(v))


def _short(e: Exception) -> str:
    t = str(e).strip()
    return (t[:160] + "...") if len(t) > 160 else (t or type(e).__name__)


def _chain(lum: list, order: list, kind: str, cancel, local: bool = False, pairs: list | None = None):
    """Register each frame to its neighbour in ``order``, compose to the middle frame. Returns (transforms from
    the reference to each frame, residuals, outlier shares, local flows or None), indexed in ``order`` positions.
    ``local``: after each pair's global fit, local flow takes out what one transform cannot (a hand-held macro
    changes perspective between frames); the flows are added up along the chain."""
    n = len(order)
    m = n // 2
    pair = []
    pflow = []
    for k, (a, b) in enumerate(zip(order[:-1], order[1:])):
        _check(cancel)
        P = pairs[k] if pairs is not None else A.register(lum[a], lum[b], kind=kind, cancel=cancel)
        P = A.Transform(P.matrix.copy(), P.kind, P.residual_px, P.tiles_used, P.outliers)
        if local:
            wb = A.warp(lum[b], P, lum[a].shape, order=1)
            fl = A.flow(lum[a], wb, 64, 32, max_shift=6)
            after = A.warp(lum[b], P, lum[a].shape, order=1, flw=fl)
            c, sh, _ = A.tile_shifts(lum[a], after, 96, 96, 4, start=(0.0, 0.0))
            if c.shape[0]:                           # what is left after the local flow, for the report
                P.residual_px = float(np.sqrt(np.median(np.sum(sh * sh, axis=1))))
            pflow.append(fl)
        pair.append(P)
    T = [None] * n
    T[m] = A.Transform.identity(kind)
    for j in range(m + 1, n):
        T[j] = pair[j - 1].compose(T[j - 1])         # ref -> j-1 -> j
    for j in range(m - 1, -1, -1):
        T[j] = pair[j].inverse().compose(T[j + 1])   # ref -> j+1 -> j
    res = [0.0] * n
    outl = [0.0] * n
    for j in range(n):
        near = [pair[k] for k in (j - 1, j) if 0 <= k < n - 1]
        res[j] = float(min(p.residual_px for p in near)) if near else 0.0
        outl[j] = float(min(getattr(p, "outliers", 0.0) for p in near)) if near else 0.0
    res[m] = 0.0
    flows = [None] * n
    if local and pflow:
        g = pflow[0]
        zero = np.zeros_like(g.dy)
        D = [None] * n
        D[m] = (zero, zero)
        for j in range(m + 1, n):
            D[j] = (D[j - 1][0] + pflow[j - 1].dy, D[j - 1][1] + pflow[j - 1].dx)
        for j in range(m - 1, -1, -1):
            D[j] = (D[j + 1][0] - pflow[j].dy, D[j + 1][1] - pflow[j].dx)
        for j in range(n):
            if j != m:
                flows[j] = A.Flow(D[j][0].astype(np.float32), D[j][1].astype(np.float32), g.quality, g.y0, g.x0,
                                  g.step)
    return T, res, outl, flows, pair


# ------------------------------------------------------------------------------------------ merging
class _Source:
    """Lined-up frames at one resolution, one at a time, with the per-frame maps the merges need."""

    def __init__(self, an: FocusAnalysis, full: bool, edit=None, scratch=None):
        self.an = an
        self.full = full
        self.edit = edit
        self.scratch = scratch
        if full:
            y0, y1, x0, x1 = an.box_full
        else:
            y0, y1, x0, x1 = 0, an.box_half[1] - an.box_half[0], 0, an.box_half[3] - an.box_half[2]
        self.shape = (y1 - y0, x1 - x0)
        self._cache = {}

    def frame(self, j: int) -> np.ndarray:
        an = self.an
        if not self.full:
            return np.asarray(an.aligned[j], np.float32)
        rgb = F.scene_rgb(an.paths[j], self.edit, False)
        y0, y1, x0, x1 = an.box_full
        al = A.warp(rgb, _with_origin(an.t_full[j], y0, x0), self.shape, order=3,
                    flw=_flow_in_box(an.f_full[j], y0, x0))
        del rgb
        if abs(an.gains[j] - 1.0) > 1e-6:
            al *= np.float32(an.gains[j])
        return al

    def halfmap(self, m: np.ndarray) -> np.ndarray:
        """A half-size map at this resolution."""
        if not self.full:
            return m
        return _up2(m, self.shape)

    def alloc(self, name, shape, dtype=np.float32):
        if self.full and self.scratch is not None:
            return self.scratch.array(name, shape, dtype)
        return np.zeros(shape, dtype)


def _weights_depth(an: FocusAnalysis, params: FocusParams):
    D = an.depth(params.smooth)
    P = an.presence()
    n = an.n

    def w(j):
        tent = np.maximum(0.0, 1.0 - np.abs(D - j))
        return (P * tent + (1.0 - P) / n).astype(np.float32)
    return w


def _weights_average(an: FocusAnalysis):
    floor = (0.05 * float(np.mean(an.smax)) + float(np.mean(an.noise_e))) ** 2
    tot = np.zeros(an.smax.shape, np.float32)
    for j in range(an.n):
        e = np.asarray(an.energy[j], np.float32)
        tot += e * e + np.float32(floor)

    def w(j):
        e = np.asarray(an.energy[j], np.float32)
        return ((e * e + np.float32(floor)) / tot).astype(np.float32)
    return w


def resolve_method(an: FocusAnalysis, params: FocusParams) -> str:
    """The method a merge really uses: Auto on a Dual Pixel stack is Measured depth (B.1); Measured depth without
    Dual Pixel frames is Auto."""
    m = params.method if params.method in METHODS else "auto"
    dp = bool(getattr(an, "dual_pixel", False))
    if m == "auto" and dp and params.auto_measured:
        return "measured"
    if m == "measured" and not dp:
        return "auto"
    return m


def _weights_measured(an: FocusAnalysis, params: FocusParams, res):
    """Measured depth (B.1): each place's weight goes to the two frames whose planes of focus (res.positions) lie
    either side of its measured depth, shared linearly between them (a tent in focus positions); where the depth was
    filled in rather than measured, the sharpness weights of 'Sharpest frame per place' take over smoothly."""
    from scipy.ndimage import gaussian_filter
    base = _weights_depth(an, params)
    q = np.asarray(res.positions, np.float64)
    order = np.argsort(q)                            # a frame out of focus order still finds its neighbours
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    z = np.asarray(res.depth, np.float32)
    cm = np.where(res.label > 0, np.clip(2.0 * res.confidence, 0.0, 1.0), 0.0).astype(np.float32)
    cm = gaussian_filter(cm, 1.5)
    n = an.n

    def w(j):
        r = int(rank[j])
        qj = float(q[j])
        t = np.zeros(z.shape, np.float32)
        if r > 0:
            ql = float(q[order[r - 1]])
            sel = (z >= ql) & (z <= qj)
            t[sel] = (z[sel] - ql) / max(qj - ql, 1e-6)
        else:
            t[z <= qj] = 1.0
        if r < n - 1:
            qr = float(q[order[r + 1]])
            sel = (z > qj) & (z <= qr)
            t[sel] = (qr - z[sel]) / max(qr - qj, 1e-6)
        else:
            t[z > qj] = 1.0
        return (cm * t + (1.0 - cm) * base(j)).astype(np.float32)
    return w


def _mix_measured(an: FocusAnalysis, res) -> np.ndarray:
    """The finest-detail pick only where the measured depth puts a place within the depth of field of two frames
    (hairs crossing in front of each other); elsewhere the measured frames alone."""
    from scipy.ndimage import gaussian_filter
    mix = an.mix()
    if res.dof_half is None:
        return mix
    z = np.asarray(res.depth, np.float32)
    cnt = np.zeros(z.shape, np.float32)
    for x in res.positions:
        cnt += (np.abs(z - np.float32(x)) < np.float32(res.dof_half)).astype(np.float32)
    both = gaussian_filter((cnt >= 2).astype(np.float32), 2.0)
    filled = (res.label == 0).astype(np.float32)
    return np.clip(mix * np.maximum(both, filled), 0.0, 1.0).astype(np.float32)


def moving_frames(an: FocusAnalysis, res, feather: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """(K, mask): for the places where the measures disagree (B.2: something moved between frames), the one frame
    each such region takes it from (the frame most often sharpest inside it; -1 elsewhere), and a 0..1 mask: 1 over
    the regions (the agreement map's red places, grown a little), falling to 0 over ``feather`` pixels outside them.
    K is grown into that band too, so the feathered edge blends one frame with the merge (never with nothing)."""
    from scipy.ndimage import binary_dilation, distance_transform_edt, label as nd_label
    from .depthfuse import DISAGREE_STEPS
    ag = np.nan_to_num(np.asarray(res.agreement, np.float32), nan=0.0)
    red = binary_dilation(ag > DISAGREE_STEPS, iterations=3)
    K = np.full(red.shape, -1, np.int16)
    if not red.any():
        return K, np.zeros(red.shape, np.float32)
    lab, nreg = nd_label(red)
    idx = np.asarray(an.idx, np.int32)
    pres = an.presence() > 0.3
    for r in range(1, nreg + 1):
        m = lab == r
        if m.sum() < 16:
            continue
        vals = idx[m & pres] if (m & pres).sum() >= 8 else idx[m]
        K[m] = int(np.bincount(vals, minlength=an.n).argmax())
    if not (K >= 0).any():
        return K, np.zeros(red.shape, np.float32)
    dist, (iy, ix) = distance_transform_edt(K < 0, return_indices=True)
    grown = np.where(dist <= feather, K[iy, ix], -1).astype(np.int16)
    t = np.clip(1.0 - dist / max(feather, 1e-6), 0.0, 1.0)
    mask = (t * t * (3 - 2 * t)).astype(np.float32)
    return grown, mask


def _levels(shape, full: bool) -> int:
    from ..detail.pyramid import max_levels
    return max(1, min(6 + (1 if full else 0), max_levels(shape, 8)))


def merge(an: FocusAnalysis, params: FocusParams, full: bool = False, progress=None, cancel=None,
          scratch: F.Scratch | None = None, edit=None) -> np.ndarray:
    """The merged result in the box frame (scene-linear, not yet upright), at half size (from the frames on disk)
    or full size (decoding each frame once). One pass over the frames for every method."""
    from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
    from ..detail import pyramid as pyr
    method = resolve_method(an, params)
    src = _Source(an, full, edit, scratch)
    H, W = src.shape
    n = an.n
    res = None
    if method == "measured" or (params.moving_single and getattr(an, "dual_pixel", False)):
        from . import depthfuse as DFU
        res = DFU.fuse(an, params.smooth)
        if res.method != "focus+dual_pixel":
            res = None
            method = "auto" if method == "measured" else method
    need_pyr = method in ("pyramid", "auto", "measured")
    if method == "average":
        wfun = _weights_average(an)
    elif method == "measured":
        wfun = _weights_measured(an, params, res)
    else:
        wfun = _weights_depth(an, params)
    K = kmask = None
    if res is not None and params.moving_single:
        K, kmask = moving_frames(an, res)
        if not (K >= 0).any():
            K = kmask = None
    one = src.alloc("one", (H, W, 3)) if K is not None else None
    Kf = None
    if K is not None:                                # frame numbers: nearest, never blended between two frames
        Kf = K.astype(np.float32)
        if src.full:
            Kf = np.repeat(np.repeat(Kf, 2, 0), 2, 1)
            Kf = np.pad(Kf, ((0, max(0, H - Kf.shape[0])), (0, max(0, W - Kf.shape[1]))), mode="edge")[:H, :W]
    acc = src.alloc("acc", (H, W, 3))
    wsum = src.alloc("wsum", (H, W))
    if params.halo_guard:
        lo = src.alloc("lo", (H, W, 3), np.float16)
        hi = src.alloc("hi", (H, W, 3), np.float16)
        lo[...] = np.float16(np.inf)
        hi[...] = np.float16(-np.inf)
    if need_pyr:
        nl = _levels((H, W), full)
        shapes = [(H, W)]
        for _ in range(nl):
            shapes.append(((shapes[-1][0] + 1) // 2, (shapes[-1][1] + 1) // 2))
        best = [src.alloc(f"best{l}", shapes[l] + (3,), np.float16 if full else np.float32) for l in range(nl)]
        bscore = [src.alloc(f"bsc{l}", shapes[l], np.float16 if full else np.float32) for l in range(nl)]
        msum = [src.alloc(f"msum{l}", shapes[l] + (3,), np.float16 if full else np.float32) for l in range(nl)]
        for l in range(nl):
            bscore[l][...] = -1.0
    strokes = _strokes_by_frame(unflip_strokes(params.retouch, an.flip), n)
    if need_pyr:
        lap_f = _lap_noise_factors(nl)
        na, nb = an.ref_noise
        if full:                                     # a half-size pixel averages four: full size has 4x the variance
            na, nb = 4.0 * na, 4.0 * nb
    for j in range(n):
        _check(cancel)
        _tick(progress, 0.05 + 0.85 * j / n, f"{'Building' if full else 'Previewing'}: frame {j + 1} of {n}")
        f = src.frame(j)
        w = src.halfmap(wfun(j))
        for y in range(0, H, 1024):
            acc[y:y + 1024] += f[y:y + 1024] * w[y:y + 1024, :, None]
            wsum[y:y + 1024] += w[y:y + 1024]
        del w
        if one is not None:
            for y in range(0, H, 1024):
                sel = np.abs(Kf[y:y + 1024] - j) < 0.5
                if sel.any():
                    blk = np.asarray(one[y:y + 1024], np.float32)
                    blk[sel] = f[y:y + 1024][sel]
                    one[y:y + 1024] = blk
        if params.halo_guard:
            for c in range(3):
                lo[..., c] = np.minimum(lo[..., c], minimum_filter(f[..., c], 5, mode="nearest"))
                hi[..., c] = np.maximum(hi[..., c], maximum_filter(f[..., c], 5, mode="nearest"))
        if need_pyr:
            g = [f[..., c] for c in range(3)]
            for l in range(nl):
                g1 = [pyr.down(x) for x in g]
                lap = [x - pyr.up(y, x.shape) for x, y in zip(g, g1)]
                lum = 0.25 * lap[0] + 0.5 * lap[1] + 0.25 * lap[2]
                glum = 0.25 * g1[0] + 0.5 * g1[1] + 0.25 * g1[2]
                # noise at this size and brightness, from the measured noise curve: the same yardstick for every
                # frame (a frame's own spread at this size would count its detail as noise)
                sig = np.float32(lap_f[l]) * np.sqrt(np.maximum(np.float32(na) * pyr.up(glum, lum.shape) +
                                                                np.float32(nb), 1e-12))
                score = uniform_filter(np.abs(lum), 5, mode="nearest") / sig
                del glum, sig
                take = score > np.asarray(bscore[l], np.float32)
                bs = np.asarray(bscore[l], np.float32)
                bscore[l][...] = np.where(take, score, bs)
                for c in range(3):
                    b = np.asarray(best[l][..., c], np.float32)
                    best[l][..., c] = np.where(take, lap[c], b)
                    msum[l][..., c] = np.asarray(msum[l][..., c], np.float32) + lap[c]
                del lap, lum, score, take
                g = g1
            del g
        if j in strokes:
            strokes[j] = (strokes[j], f if not full else None)
        del f
    out = src.alloc("out", (H, W, 3))
    for y in range(0, H, 1024):
        out[y:y + 1024] = acc[y:y + 1024] / np.maximum(wsum[y:y + 1024], 1e-12)[..., None]
    if need_pyr:
        _tick(progress, 0.92, "Putting the detail together")
        mx = src.halfmap(an.mix()) if method == "auto" else (src.halfmap(_mix_measured(an, res))
                                                              if method == "measured" else None)
        pick_levels = PICK_LEVELS + (1 if full else 0)
        # coarsest level: the depth result's, so colour and brightness stay honest
        for c in range(3):
            gd = [np.asarray(out[..., c], np.float32)]
            for _ in range(nl):
                gd.append(pyr.down(gd[-1]))
            lapc = []
            for l in range(nl):
                if l >= pick_levels:                 # coarse sizes cannot tell sharp from soft: depth result's
                    lapc.append(gd[l] - pyr.up(gd[l + 1], gd[l].shape))
                    continue
                b = np.asarray(best[l][..., c], np.float32)
                mean = np.asarray(msum[l][..., c], np.float32) / np.float32(n)
                sc = np.asarray(bscore[l], np.float32)
                wd = _smoothstep(1.3, 2.0, sc).astype(np.float32)   # detail above its noise: pick; else average
                lapc.append(wd * b + (1 - wd) * mean)
            lapc.append(gd[-1])
            pc = pyr.collapse(lapc)
            del lapc, gd
            if mx is not None:
                out[..., c] = mx * pc + (1 - mx) * np.asarray(out[..., c], np.float32)
            else:
                out[..., c] = pc
            del pc
    if one is not None:                              # moving parts from one frame (B.2)
        km = src.halfmap(kmask)
        for y in range(0, H, 1024):
            mk = km[y:y + 1024, :, None]
            out[y:y + 1024] = mk * np.asarray(one[y:y + 1024], np.float32) + (1 - mk) * np.asarray(
                out[y:y + 1024], np.float32)
    if params.halo_guard:
        for y in range(0, H, 1024):
            out[y:y + 1024] = np.clip(out[y:y + 1024], np.asarray(lo[y:y + 1024], np.float32),
                                      np.asarray(hi[y:y + 1024], np.float32))
    if strokes:
        _apply_strokes(out, strokes, src, full)
    return out


_LAPF: list = []


def _lap_noise_factors(levels: int) -> list:
    """Standard deviation of each Laplacian level for unit white pixel noise (measured once)."""
    global _LAPF
    if len(_LAPF) < levels:
        from ..detail import pyramid as pyr
        n = np.random.default_rng(3).standard_normal((1024, 1024)).astype(np.float32)
        g = pyr.gaussian(n, min(8, pyr.max_levels(n.shape, 8)))
        lap = pyr.laplacian_from_gaussian(g)
        f = [float(x[8:-8, 8:-8].std()) if min(x.shape) > 24 else float(x.std()) for x in lap[:-1]]
        while len(f) < levels:
            f.append(f[-1] * 0.5)
        _LAPF = f
    return _LAPF


def _robust_sigma(x: np.ndarray) -> float:
    s = x[::4, ::4] if x.size > 1_000_000 else x
    return float(1.4826 * np.median(np.abs(s - np.median(s))))


# ------------------------------------------------------------------------------------------ retouch from a frame
def unflip_strokes(retouch: list, flip: int) -> list:
    """Strokes drawn on the upright result, in the frame the merge works in (before turning upright)."""
    def conv(u, v):
        if flip == 3:
            return 1 - u, 1 - v
        if flip == 6:
            return v, 1 - u
        if flip == 5:
            return 1 - v, u
        return u, v
    out = []
    for s in retouch or []:
        d = dict(s)
        d["points"] = [list(conv(float(u), float(v))) for u, v in s.get("points", [])]
        out.append(d)
    return out


def _strokes_by_frame(retouch: list, n: int) -> dict:
    out: dict = {}
    for s in retouch or []:
        try:
            j = int(s["frame"])
            if 0 <= j < n and s.get("points"):
                out.setdefault(j, []).append(s)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def stroke_mask(strokes: list, shape) -> np.ndarray:
    """Soft mask (0..1) of brush strokes given in result-relative coordinates (u, v in 0..1, radius as a share of
    the long edge), in the box frame of ``shape`` (not upright: the caller converts upright strokes)."""
    from scipy.ndimage import gaussian_filter
    H, W = shape[:2]
    m = np.zeros((H, W), np.float32)
    yy, xx = np.ogrid[:H, :W]
    for s in strokes:
        r = max(1.0, float(s.get("radius", 0.02)) * max(H, W))
        for u, v in s.get("points", []):
            cx, cy = float(u) * (W - 1), float(v) * (H - 1)
            y0, y1 = int(max(0, cy - r - 2)), int(min(H, cy + r + 3))
            x0, x1 = int(max(0, cx - r - 2)), int(min(W, cx + r + 3))
            sub = (yy[y0:y1] - cy) ** 2 + (xx[:, x0:x1] - cx) ** 2 <= r * r
            m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], sub)
    return np.clip(gaussian_filter(m, max(1.0, 0.25 * np.sqrt(H * W) / 200.0)), 0, 1)


def _apply_strokes(out, strokes: dict, src: _Source, full: bool):
    for j, item in strokes.items():
        sl, f = item if isinstance(item, tuple) else (item, None)
        if f is None:
            f = src.frame(j)
        msk = stroke_mask(sl, out.shape)
        for y in range(0, out.shape[0], 1024):
            mk = msk[y:y + 1024, :, None]
            out[y:y + 1024] = mk * f[y:y + 1024] + (1 - mk) * np.asarray(out[y:y + 1024], np.float32)
        del f


def source_patch(an: FocusAnalysis, frame_index: int, box=None) -> np.ndarray:
    """One lined-up source frame's pixels (half size, box frame, upright), for the retouch brush's view."""
    from ..pipeline.pipeline import apply_orientation
    a = np.asarray(an.aligned[frame_index], np.float32)
    if box is not None:
        y0, y1, x0, x1 = box
        a = a[y0:y1, x0:x1]
    return np.ascontiguousarray(apply_orientation(a, an.flip))


# ------------------------------------------------------------------------------------------ preview / build
def preview(an: FocusAnalysis, params: FocusParams | None = None, progress=None, cancel=None) -> np.ndarray:
    """Half-size result, scene-linear ProPhoto, upright."""
    from ..pipeline.pipeline import apply_orientation
    params = params or an.params
    out = merge(an, params, False, progress, cancel)
    return np.ascontiguousarray(apply_orientation(out, an.flip))


def view_map(an: FocusAnalysis, name: str) -> np.ndarray:
    """uint8 RGB views over the result, upright: 'depth' (which frame each place came from, colour ramp; grey
    where no frame has detail), 'confidence'."""
    from ..pipeline.pipeline import apply_orientation
    if name == "depth":
        d = an.depth() / max(1, an.n - 1)
        rgb = _ramp(d)
        p = an.presence()[..., None]
        grey = np.full_like(rgb, 60.0)
        rgb = rgb * p + grey * (1 - p)
    elif name == "confidence":
        c = np.clip(an.confidence() * an.presence(), 0, 1)
        rgb = np.stack([c * 40, c * 230, c * 255], 2)
    else:
        raise ValueError(name)
    return np.ascontiguousarray(apply_orientation(np.clip(rgb, 0, 255).astype(np.uint8), an.flip))


def _ramp(t: np.ndarray) -> np.ndarray:
    """Near (0) -> far (1): violet, cyan, green, amber (the Studio's palette)."""
    stops = np.array([[150, 90, 255], [0, 220, 255], [60, 230, 120], [255, 190, 40]], np.float32)
    x = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(stops) - 2)
    f = (x - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def build(an: FocusAnalysis, params: FocusParams | None, out_path: str, progress=None, cancel=None) -> str:
    """Pass B at full size, one frame in memory at a time; writes the linear intermediate and the sidecar's
    "stack" block. Returns the path written."""
    from ..io.intermediate import write_intermediate
    from ..pipeline.edit import Edit
    from ..pipeline.pipeline import apply_orientation
    params = params or an.params
    edit = Edit.from_dict(an.edit)
    with F.Scratch("focusbuild") as sc:
        out = merge(an, params, True, progress, cancel, sc, edit)
        _check(cancel)
        _tick(progress, 0.95, "Writing the result")
        up = apply_orientation(out, an.flip)
        info = {"kind": "focus", "sources": [os.path.basename(p) for p in an.paths], "method": params.method,
                "params": params.to_dict(), "scale": 1.0, "frames": an.n}
        write_intermediate(out_path, up, an.meta, info)
        del up, out
    extra = write_depth_sidecar(out_path, an, params)
    write_stack_block(out_path, "focus", an.paths, params.to_dict(), extra)
    _tick(progress, 1.0, "Done")
    return out_path


def depth_sidecar_path(picture_path: str) -> str:
    """Where a picture's measured depth is kept: NAME.depth.npz beside it (BUILD_PLAN_DP3D B.7)."""
    return os.path.splitext(picture_path)[0] + ".depth.npz"


def write_depth_sidecar(out_path: str, an: FocusAnalysis, params: FocusParams | None = None) -> dict:
    """The stack's fused depth travels with its result (B.7): NAME.depth.npz beside it (half size, upright, near =
    1 over the measured range, float16; its confidence; how each place was measured), so Develop can select by
    measured distance, relight with measured normals and set the depth of field. Returns the sidecar's "depth"
    block ({} when there is no depth to keep)."""
    from . import depthfuse as DFU
    from . import surface as SURF
    from .depthmask import strict_label
    from ..pipeline.pipeline import apply_orientation
    try:
        res = DFU.fuse(an, (params or an.params).smooth)
        near_high, how, measured, _ = SURF.decide_near(res, str(an.meta.get("model") or ""), SURF.SurfaceParams())
        h = res.depth if near_high else -res.depth
        lo, hi = SURF.height_range(h, res.label)
        d01 = np.clip((h - lo) / (hi - lo), 0.0, 1.0)
        p = depth_sidecar_path(out_path)
        tmp = p + ".partial.npz"
        np.savez_compressed(tmp, depth=np.ascontiguousarray(apply_orientation(d01, an.flip)).astype(np.float16),
                            confidence=np.ascontiguousarray(apply_orientation(res.confidence, an.flip))
                            .astype(np.float16),
                            label=np.ascontiguousarray(apply_orientation(res.label, an.flip)).astype(np.uint8),
                            label_strict=np.ascontiguousarray(apply_orientation(strict_label(res), an.flip))
                            .astype(np.uint8),
                            range_steps=np.float32(hi - lo))
        os.replace(tmp, p)
        return {"depth": {"file": os.path.basename(p), "source": "measured", "method": res.method,
                          "near_end": how, "near_measured": bool(measured), "units": res.units,
                          "range": float(hi - lo), "coverage": res.shares()}}
    except OSError as e:                             # a read-only folder: the result itself is what matters
        log.info("depth sidecar for %s not written (%s)", os.path.basename(out_path), e)
    except Exception:                                # noqa: BLE001  (the stack's result stands without it)
        slog.log_exception("depth sidecar", log)
        slog.save_error_report(f"Focus Stack: keeping the depth of {os.path.basename(out_path)}")
    return {}


def write_stack_block(out_path: str, kind: str, sources: list, params: dict, extra: dict | None = None):
    """The result's own sidecar remembers how it was made, so 'Build again' can refill the page."""
    from .. import api
    try:
        doc = api.load_doc(out_path) or {}
        doc["stack"] = {"kind": kind, "sources": [os.path.abspath(p) for p in sources], "params": params,
                        "date": _dt.datetime.now().isoformat(timespec="seconds"), **(extra or {})}
        api.write_doc(out_path, doc)
    except OSError as e:                             # a read-only folder: the result itself is what matters
        log.info("stack sidecar for %s not written (%s)", os.path.basename(out_path), e)


_lock = threading.Lock()
