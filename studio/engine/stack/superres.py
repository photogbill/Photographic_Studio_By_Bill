"""Super-Res Stack: a burst lined up and merged into a larger or cleaner photo (BUILD_PLAN_TABS section 5).

Analysis (all modes): each frame's raw mosaic is read once (int16, 90 MB for 45 MP; on disk for a big burst), a
2 x 2 luminance bin of it is the analysis copy. Frames are ranked by sharpness (the sharpest is the reference;
blurred ones are dropped), lined up to the reference (similarity for hand-held and clean bursts, shift only for
the moon, on the stars for the night sky), local flow takes out what is left (hand shake, rolling shutter, air
wobble), and a robustness map per frame says where it agrees with the reference: something that moved, or a tile
that could not be lined up, gets weight near zero there and that place falls back to the frames that agree (in
the end to the reference alone).

Merges:
  average     each frame developed as usual, lined up, a weighted average (noise falls as 1 / sqrt(N))
  drizzle     every raw photosite dropped onto a finer grid at its measured position, no demosaic (Fruchter
              and Hook 2002): real extra detail when the frames landed on different sub-pixel positions. Square
              drops of pixfrac input pixels, exact area overlaps, done in output bands with np.bincount so memory
              stays bounded; colours a place never received are filled from their neighbours
  sigma       (night sky) two-pass sigma clip: values far from the others at a place are left out
  winsorized  (night sky) far values pulled in to the edge of normal instead
The lens's lateral colour fringing, measured on the reference frame, is corrected inside the drizzle by moving
the red and blue photosites to where their light belongs.

Results are written as linear intermediates (io/intermediate.py) and open in Develop. No Qt here.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ... import log as slog
from . import align as A
from . import astro
from . import frames as F
from .focus import Cancelled, _check, _short, _tick, noise_curve, write_stack_block

log = slog.get("engine.stack.superres")

MODE_INFO = {
    "handheld": {"name": "Handheld burst", "does": "Lines up a hand-held burst and builds a larger, cleaner photo.",
                 "best": "a subject that held still; 12-30 frames",
                 "weak": "anything that moved falls back to one frame there"},
    "clean": {"name": "Cleaner, same size", "does": "Averages the burst for less noise at the normal size.",
              "best": "low light without a tripod; 4 frames and up", "weak": "adds no detail"},
    "moon": {"name": "Moon and planets", "does": "Keeps the sharpest frames, undoes the air's wobble, builds larger.",
             "best": "50 frames and up at a fast shutter speed", "weak": "needs many frames; thin cloud ruins it"},
    "sky": {"name": "Night sky", "does": "Lines up on the stars and averages, removing planes and satellites.",
            "best": "10 frames and up on a tripod or tracker", "weak": "the ground blurs (Freeze ground comes later)"},
}
MODES = tuple(MODE_INFO)
MERGE_INFO = {
    "drizzle": {"name": "Drizzle (from the raw data)", "does": "Drops every raw photosite onto a finer grid at its "
                "measured position: no demosaic.", "best": "real extra detail when the frames landed on "
                "different sub-pixel positions", "weak": "needs 12+ frames for 2x, 6+ for 1.5x; slower; raw files "
                "only", "speed": "slow"},
    "average": {"name": "Robust average", "does": "Develops each frame, lines it up, averages what agrees.",
                "best": "cleaning noise; few frames; anything that moved", "weak": "no extra detail",
                "speed": "medium"},
    "sigma": dict(astro.REJECT_INFO["sigma"]),
    "winsorized": dict(astro.REJECT_INFO["winsorized"]),
}
MERGES_FOR = {"handheld": ("drizzle", "average"), "clean": ("average", "drizzle"), "moon": ("drizzle", "average"),
              "sky": ("sigma", "winsorized", "average")}
MODE_DEFAULTS = {
    "handheld": dict(merge="drizzle", scale=2.0, keep_best=1.0, local=True, reject=3.0, sharpen=0.3),
    "clean": dict(merge="average", scale=1.0, keep_best=1.0, local=True, reject=3.0, sharpen=0.0),
    "moon": dict(merge="drizzle", scale=2.0, keep_best=0.25, local=True, reject=4.0, sharpen=1.0),
    "sky": dict(merge="sigma", scale=1.0, keep_best=1.0, local=False, reject=3.0, sharpen=0.0),
}
WHAT_TO_SHOOT = {
    "handheld": "Hand-held burst: 12 to 30 frames of something still, at a fast enough shutter speed, holding the "
                "camera as you normally would (the small movements are what add detail).",
    "clean": "Any burst of 4 frames or more, hand-held or on a tripod, of a scene that holds still.",
    "moon": "Moon or planets: 50 frames or more at a fast shutter speed, ISO as low as that allows, long lens.",
    "sky": "Night sky: 10 frames or more on a tripod or tracker, same exposure; the stars are lined up, not the "
           "ground.",
}
HONEST = ("The gain depends on a sharp lens and on real movement between frames: a hand-held burst has it, a "
          "locked tripod does not (that still cleans noise). 2x is a realistic ceiling. Where something moved, "
          "that place falls back to a single frame.")
BURST_ADVICE = ("For a burst, expose for the highlights (slightly under): averaging cleans the shadows. This is the "
                "opposite of the single-photo Canon rule (expose a little over), because here the shadows get "
                "many frames' worth of light.")
NO_MOVEMENT = ("These frames did not move between shots: stacking will clean noise but cannot add detail")
SCALES = (1.0, 1.5, 2.0)


@dataclass
class SuperParams:
    mode: str = "handheld"
    merge: str = "drizzle"
    scale: float = 2.0
    keep_best: float = 1.0       # share of the sharp-enough frames kept (moon: the sharpest quarter)
    pixfrac: float = 0.7         # drizzle drop size, in input pixels
    reject: float = 3.0          # robustness: how far from the reference (in noise sigmas) still counts
    local: bool = True           # local flow after the global line-up
    sharpen: float = 0.3         # finish: handheld light unsharp of the finest scale; moon starlet sharpening
    crop: tuple | None = None    # (y0, y1, x0, x1) of the reference frame (camera-crop pixels) to build; None = all
    clip_k: float = 3.0          # sky rejection, in standard deviations
    lens_ca: bool = True         # correct the lens's colour fringing inside the drizzle

    @staticmethod
    def for_mode(mode: str) -> "SuperParams":
        p = SuperParams(mode=mode)
        for k, v in MODE_DEFAULTS.get(mode, {}).items():
            setattr(p, k, v)
        return p

    def analysis_key(self) -> tuple:
        return (self.mode, bool(self.local), round(float(self.reject), 3))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["crop"] = list(self.crop) if self.crop else None
        return d

    @staticmethod
    def from_dict(d: dict) -> "SuperParams":
        p = SuperParams(**{k: v for k, v in (d or {}).items() if k in SuperParams.__dataclass_fields__})
        if p.crop is not None:
            p.crop = tuple(int(v) for v in p.crop)
        return p


@dataclass
class FrameReport:
    name: str
    path: str
    used: bool = True
    status: str = "ok"           # ok | warn | dropped
    reason: str = ""
    sharpness: float = 0.0
    rank: int = 0                # 1 = sharpest
    offset: tuple = (0.0, 0.0)   # (dy, dx) of the frame centre against the reference, full-size pixels
    rejected: float = 0.0        # share of the frame given little weight (moved, or did not line up)


@dataclass
class BurstAnalysis:
    paths: list                  # usable frames, capture order
    reports: list                # FrameReport per path (same order), then the files left out
    ref: int
    t_full: list                 # Transform per frame (reference -> frame), full-size camera-crop pixels
    flows: list                  # Flow per frame at full size, or None
    robust: np.ndarray           # memmap (N, ha, wa) float16: weight 0..1 per frame, reference frame coordinates
    sharp: list
    k_ana: int                   # full-size pixels per analysis pixel
    shape_full: tuple
    raw: bool
    mosaics: list                # int16 mosaics (RAM or memmap), raw bursts only
    infos: list                  # RawInfo per frame (raw bursts)
    mult: np.ndarray             # reference white balance (min 1)
    flip: int
    meta: dict
    edit: dict
    params: SuperParams
    ref_ana: np.ndarray
    offsets_bg: list             # sky: background offset per frame (luminance units, subtracted before rejection)
    frac: list                   # sub-pixel parts of each kept frame's offset (fy, fx)
    moved: bool                  # frames landed on different sub-pixel positions
    sharp_centre: tuple          # (y, x) full size: the sharpest place (default preview window)
    ca: object = None
    scratch: object = None
    notes: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.paths)

    def close(self):
        if self.scratch is not None:
            self.scratch.close()

    def kept(self, params: SuperParams | None = None) -> list:
        """Indices of the frames used: not dropped, then the ``keep_best`` share of the sharpest (the reference
        always)."""
        params = params or self.params
        ok = [i for i, r in enumerate(self.reports[:self.n]) if r.status != "dropped"]
        ok.sort(key=lambda i: -self.sharp[i])
        k = max(1, int(round(len(ok) * float(np.clip(params.keep_best, 0.01, 1.0)))))
        out = ok[:k]
        if self.ref not in out:
            out = [self.ref] + out[:-1]
        return sorted(out)

    def rejected_view(self, params: SuperParams | None = None) -> np.ndarray:
        """0..1 at analysis size: the share of the kept frames' weight left out at each place (1 = only the
        reference there)."""
        ks = [k for k in self.kept(params) if k != self.ref]
        if not ks:
            return np.zeros(self.ref_ana.shape, np.float32)
        acc = np.zeros(self.robust.shape[1:], np.float32)
        for k in ks:
            acc += 1.0 - np.asarray(self.robust[k], np.float32)
        return acc / len(ks)


# ------------------------------------------------------------------------------------------ analysis helpers
def _bin_luma(mos16: np.ndarray, pattern, mult) -> np.ndarray:
    """2 x 2 luminance of a white-balanced int16 mosaic: (R + 2 G + B) / 4, half size."""
    h, w = mos16.shape[0] // 2 * 2, mos16.shape[1] // 2 * 2
    out = np.zeros((h // 2, w // 2), np.float32)
    sc = np.float32(1.0 / F.INT16_SCALE)
    for i in (0, 1):
        for j in (0, 1):
            out += mos16[i:h:2, j:w:2].astype(np.float32) * (sc * np.float32(mult[int(pattern[i, j])]))
    out *= np.float32(0.25)
    return out


def _lap(L):
    return (4 * L[1:-1, 1:-1] - L[:-2, 1:-1] - L[2:, 1:-1] - L[1:-1, :-2] - L[1:-1, 2:])


def _disc(L: np.ndarray) -> np.ndarray:
    """The moon / planet: the bright disc (with a margin)."""
    from scipy.ndimage import binary_dilation
    t = 0.35 * float(np.percentile(L, 99.7))
    m = L > t
    return binary_dilation(m, iterations=4) if m.any() else np.ones(L.shape, bool)


def _sharpness(L: np.ndarray, noise, mode: str) -> float:
    a, b = noise
    lp = _lap(L)
    if mode == "moon":
        reg = _disc(L)[1:-1, 1:-1]
    else:
        h, w = lp.shape
        reg = np.zeros(lp.shape, bool)
        reg[h // 10:h - h // 10, w // 10:w - w // 10] = True
    v = float(np.var(lp[reg])) if reg.any() else float(np.var(lp))
    nv = 20.0 * (a * float(np.mean(np.maximum(L[1:-1, 1:-1][reg], 0))) + b) if reg.any() else 20.0 * b
    return v / max(nv, 1e-12)


def _centroid(L: np.ndarray):
    m = _disc(L)
    w = np.where(m, np.maximum(L, 0), 0)
    s = float(w.sum()) or 1.0
    y, x = np.mgrid[0:L.shape[0], 0:L.shape[1]]
    return float((w * y).sum() / s), float((w * x).sum() / s)


def _robust_map(ref: np.ndarray, img_w: np.ndarray, noise, reject: float) -> np.ndarray:
    """exp(-(d / (reject * sigma))^2): d the 5 x 5 smoothed difference from the reference, sigma the noise of that
    difference at this brightness plus what a 0.5 px misalignment of an edge would give (so edges are not thrown
    out for sub-pixel reasons). Moving things and places that did not line up get weight near 0."""
    from scipy.ndimage import uniform_filter
    a, b = noise
    d = uniform_filter(img_w - ref, 5, mode="nearest")
    sd = np.sqrt(2.0 * (np.float32(a) * np.maximum(ref, 0) + np.float32(b))) / 5.0
    gy, gx = np.gradient(ref)
    alias = 0.5 * uniform_filter(np.hypot(gy, gx), 5, mode="nearest")
    return np.exp(-(d / (np.float32(reject) * (sd + alias) + 1e-9)) ** 2).astype(np.float32)


def _frac_spread(fr: list) -> float:
    """Largest distance (in pixels, on the unit circle of sub-pixel positions) of any frame from the burst's mean
    sub-pixel position, over both axes."""
    if len(fr) < 2:
        return 0.0
    a = np.asarray(fr, float)
    worst = 0.0
    for ax in (0, 1):
        th = 2 * np.pi * a[:, ax]
        mean = np.angle(np.mean(np.exp(1j * th)))
        d = np.abs(np.angle(np.exp(1j * (th - mean)))) / (2 * np.pi)
        worst = max(worst, float(d.max()))
    return worst


# ------------------------------------------------------------------------------------------ analysis
def analyse(paths, params: SuperParams | None = None, progress=None, cancel=None) -> BurstAnalysis:
    """Rank, line up, local flow and robustness, at analysis size. Raises ValueError with a plain reason when the
    frames cannot make a stack, Cancelled on Cancel. Close the result when done."""
    params = params or SuperParams()
    infos = F.probe(paths)
    msgs, block = F.check(infos)
    if block:
        raise F.StackProblem(" ".join(msgs))
    usable = [f for f in infos if f.ok]
    left_out = [FrameReport(f.name, f.path, False, "dropped", f.problem) for f in infos if not f.ok]
    raw = all(f.is_raw for f in usable)
    n0 = len(usable)
    scratch = F.Scratch("superres")
    try:
        mos_list, rinfo, lum, keep = [], [], [], []
        mult0 = None
        edit = None
        meta = {}
        flip = 0
        shape = None
        memmap_mosaics = False
        for i, fi in enumerate(usable):
            _check(cancel)
            _tick(progress, 0.02 + 0.45 * i / n0, f"Reading frame {i + 1} of {n0}")
            try:
                if raw:
                    mos, ri = F.mosaic(fi.path)
                    if mult0 is None:
                        mult0 = ri.wb / ri.wb.min()
                        memmap_mosaics = n0 * mos.nbytes > 1.5e9
                        flip, meta = ri.flip, dict(ri.meta)
                    if shape is not None and mos.shape != shape:
                        raise ValueError("a different size from the others")
                    L = _bin_luma(mos, ri.pattern, mult0)
                    if memmap_mosaics:
                        mm = scratch.array(f"mos{len(keep):04d}", mos.shape, np.int16)
                        mm[...] = mos
                        mm.flush()
                        mos = mm
                    shape = mos.shape
                    mos_list.append(mos)
                    rinfo.append(ri)
                else:
                    if edit is None:
                        ref_img = F.open_frame(fi.path)
                        edit = F.stack_edit(ref_img, False)
                        flip = int(getattr(ref_img.raw, "flip", 0))
                        md = ref_img.meta
                        meta = {"make": md.make, "model": md.model, "iso": md.iso, "exposure_s": md.exposure_s,
                                "fnumber": md.fnumber, "focal_mm": md.focal_mm, "lens": md.lens,
                                "datetime": md.datetime}
                        del ref_img
                    rgb = F.scene_rgb(fi.path, edit, True)
                    if shape is not None and (rgb.shape[0] * 2, rgb.shape[1] * 2) != shape:
                        raise ValueError("a different size from the others")
                    shape = (rgb.shape[0] * 2, rgb.shape[1] * 2)
                    L = A.luma(rgb)
                    del rgb
            except Cancelled:
                raise
            except Exception as e:                   # noqa: BLE001  (an unreadable frame is left out, named)
                log.info("superres: %s left out (%s)", fi.name, e)
                left_out.append(FrameReport(fi.name, fi.path, False, "dropped", _short(e)))
                continue
            ana, k_a = A.analysis_copy(L, 2000)
            del L
            lum.append(ana)
            keep.append(fi)
        n = len(keep)
        if n < 2:
            raise F.StackProblem("Fewer than two frames could be read: " +
                             "; ".join(f"{r.name} ({r.reason})" for r in left_out))
        if raw and edit is None:
            ref_img = F.open_frame(keep[0].path)
            edit = F.stack_edit(ref_img, False)
            del ref_img
        k_a = max(1, int(np.ceil(max(shape[0] // 2, shape[1] // 2) / 2000.0)))
        K = 2 * k_a                                  # full-size pixels per analysis pixel
        mode = params.mode if params.mode in MODES else "handheld"

        # ---- rank by sharpness, reference = sharpest, drop the blurred
        noise = [noise_curve(L) for L in lum]
        # one noise yardstick for every frame (the same ISO): a frame's own estimate drops when it is blurred,
        # which would make a blurred frame look sharp
        common = (float(np.median([z[0] for z in noise])), float(np.median([z[1] for z in noise])))
        sharp = [_sharpness(L, common, mode) for L in lum]
        ref = int(np.argmax(sharp))
        med = float(np.median(sharp))
        order = sorted(range(n), key=lambda i: -sharp[i])
        reports = [FrameReport(f.name, f.path, sharpness=float(sharp[i]), rank=order.index(i) + 1)
                   for i, f in enumerate(keep)]
        for i in range(n):
            if i != ref and sharp[i] < 0.6 * med:
                reports[i].status, reports[i].used = "dropped", False
                reports[i].reason = "blurred (hand shake or the air) compared with the others"
        if raw:
            mult0 = rinfo[ref].wb / rinfo[ref].wb.min()
            if any(abs(np.log(ri.wb / ri.wb.min() / mult0)).max() > 0.02 for ri in rinfo):
                lum = [_bin_luma(m, ri.pattern, mult0) for m, ri in zip(mos_list, rinfo)]   # one balance for all
                lum = [A.analysis_copy(L, 2000)[0] for L in lum]
            flip, meta = rinfo[ref].flip, dict(rinfo[ref].meta)

        # ---- line up to the reference
        ref_ana = lum[ref]
        ha, wa = ref_ana.shape
        robust = scratch.array("robust", (n, ha, wa), np.float16)
        robust[ref] = 1.0
        t_full = [A.Transform.identity() for _ in range(n)]
        flows = [None] * n
        offsets_bg = [0.0] * n
        st_ref = astro.stars(ref_ana) if mode == "sky" else None
        c_ref = _centroid(ref_ana) if mode == "moon" else None
        todo = [i for i in range(n) if i != ref and reports[i].status != "dropped"]
        for step, i in enumerate(todo):
            _check(cancel)
            _tick(progress, 0.5 + 0.45 * step / max(1, len(todo)), f"Lining up frame {step + 1} of {len(todo)}")
            L = lum[i]
            if mode == "sky":
                T = astro.match(st_ref, astro.stars(L), L.shape)
            elif mode == "moon":
                cy, cx = _centroid(L)
                init = A.Transform.translation(cy - c_ref[0], cx - c_ref[1])
                T = A.register(ref_ana, L, kind="shift", init=init, cancel=cancel)
            else:
                T = A.register(ref_ana, L, kind="similarity", cancel=cancel)
            Lw = A.warp(L, T, ref_ana.shape, order=1)
            fl = None
            if params.local and mode in ("handheld", "moon", "clean"):
                tile, stp = (48, 24) if mode == "moon" else (64, 32)
                fl = A.flow(ref_ana, Lw, tile, stp, max_shift=8 if mode == "moon" else 6)
                Lw = A.warp(L, T, ref_ana.shape, order=1, flw=fl)
            if mode == "sky":
                offsets_bg[i] = float(np.median(Lw[8:-8, 8:-8] - ref_ana[8:-8, 8:-8]))
                r = np.ones(ref_ana.shape, np.float32)
            else:
                r = _robust_map(ref_ana, Lw, noise[ref], params.reject)
            cov = A.coverage(T, L.shape, ref_ana.shape)
            r = np.where(cov, r, 0.0)
            robust[i] = r.astype(np.float16)
            reports[i].rejected = float(np.mean(r[cov] < 0.5)) if cov.any() else 1.0
            t_full[i] = T.rescaled(k_a).rescaled(2)
            flows[i] = fl.rescaled(k_a).rescaled(2) if fl is not None else None
            c = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)
            Y, X = t_full[i].apply(np.array([c[0]]), np.array([c[1]]))
            reports[i].offset = (float(Y[0] - c[0]), float(X[0] - c[1]))
            if T.residual_px * K > 2.0 or T.tiles_used == 0:
                reports[i].status = "warn"
                reports[i].reason = "did not line up well" if T.tiles_used else "could not be lined up"
            elif reports[i].rejected > 0.25:
                reports[i].status = "warn"
                reports[i].reason = f"something moved: {100 * reports[i].rejected:.0f}% of it falls back to the " \
                                    f"other frames"
        robust.flush()
        an = BurstAnalysis(paths=[k.path for k in keep], reports=reports + left_out, ref=ref, t_full=t_full,
                           flows=flows, robust=robust, sharp=sharp, k_ana=K, shape_full=shape, raw=raw,
                           mosaics=mos_list, infos=rinfo, mult=np.asarray(mult0 if raw else np.ones(3)), flip=flip,
                           meta=meta, edit=edit.to_dict(), params=SuperParams.from_dict(params.to_dict()),
                           ref_ana=ref_ana, offsets_bg=offsets_bg, frac=[], moved=True,
                           sharp_centre=_sharpest_place(ref_ana, K), scratch=scratch, notes=list(msgs))
        _coverage(an, params)
        if raw and params.lens_ca:
            an.ca = _lens_ca(keep[ref].path)
        _tick(progress, 1.0, "Analysis done")
        log.info("superres analysis (%s): %d frames, reference %s, %d dropped, moved %s", mode, n,
                 keep[ref].name, sum(r.status == "dropped" for r in reports), an.moved)
        return an
    except BaseException:
        scratch.close()
        raise


def _coverage(an: BurstAnalysis, params: SuperParams):
    """Sub-pixel parts of each kept frame's offset, and whether they spread (anything to gain)."""
    fr = []
    for k in an.kept(params):
        dy, dx = an.reports[k].offset
        fr.append((float(dy % 1.0), float(dx % 1.0)))
    an.frac = fr
    an.moved = _frac_spread(fr) >= 0.1
    if not an.moved and NO_MOVEMENT not in " ".join(an.notes):
        an.notes.append(NO_MOVEMENT + ".")


def _sharpest_place(L: np.ndarray, K: int) -> tuple:
    from scipy.ndimage import uniform_filter
    e = uniform_filter(np.pad(np.abs(_lap(L)), 1), max(9, min(L.shape) // 8), mode="nearest")
    m = max(4, min(L.shape) // 10)
    e[:m] = e[-m:] = 0
    e[:, :m] = e[:, -m:] = 0
    y, x = np.unravel_index(int(np.argmax(e)), e.shape)
    return (float(y * K + (K - 1) / 2.0), float(x * K + (K - 1) / 2.0))


def _lens_ca(path: str):
    try:
        img = F.open_frame(path)
        ca = img.lateral_ca()
        return ca if getattr(ca, "confidence", "none") != "none" else None
    except Exception as e:                           # noqa: BLE001  (no correction is the safe default)
        log.info("lens CA for the drizzle not measured (%s)", e)
        return None


def output_info(an: BurstAnalysis, params: SuperParams) -> dict:
    """Size, disk and time of a build, said before it starts."""
    y0, y1, x0, x1 = _box(an, params)
    s = float(params.scale) if params.merge == "drizzle" and an.raw else 1.0
    w, h = int(round((x1 - x0) * s)), int(round((y1 - y0) * s))
    if an.flip in (5, 6):
        w, h = h, w
    nk = len(an.kept(params))
    mp_in = (x1 - x0) * (y1 - y0) / 1e6
    per = {"drizzle": 0.45, "average": 0.18, "sigma": 0.36, "winsorized": 0.36}.get(params.merge, 0.2)
    secs = nk * mp_in * per * (s * s if params.merge == "drizzle" else 1.0) ** 0.5
    return {"w": w, "h": h, "mp": w * h / 1e6, "gb": w * h * 12 * 0.8 / 1e9, "minutes": max(0.1, secs / 60.0),
            "frames": nk, "scale": s}


def describe_output(info: dict) -> str:
    sc = f"{info['scale']:g}x" if info["scale"] > 1 else "Same size"
    return (f"{sc}: {info['w']:,} x {info['h']:,} px, about {info['gb']:.1f} GB on disk, about "
            f"{max(1, round(info['minutes']))} minute{'s' if round(info['minutes']) != 1 else ''} "
            f"({info['frames']} frames).")


def frame_warnings(an: BurstAnalysis, params: SuperParams) -> list:
    """Plain-language warnings before a build (not refusals)."""
    out = []
    nk = len(an.kept(params))
    if params.merge == "drizzle" and an.raw:
        need = 12 if params.scale >= 2 else 6
        if nk < need:
            out.append(f"{nk} frames is few for {params.scale:g}x drizzle ({need} or more fill every colour): "
                       f"some places will be filled from their neighbours. "
                       f"{'1.5x' if params.scale >= 2 else 'Same size'} would suit these frames better.")
        if not an.moved:
            out.append(NO_MOVEMENT + ": Cleaner, same size is the better choice.")
    if params.merge == "drizzle" and not an.raw:
        out.append("Drizzle needs raw files: these frames are developed files, so the robust average is used.")
    return out


# ------------------------------------------------------------------------------------------ merging
def _box(an: BurstAnalysis, params: SuperParams, window=None) -> tuple:
    H, W = an.shape_full
    b = window if window is not None else params.crop
    if b is None:
        return 0, H, 0, W
    y0, y1, x0, x1 = (int(v) for v in b)
    y0, x0 = max(0, y0) // 2 * 2, max(0, x0) // 2 * 2
    y1, x1 = min(H, y1), min(W, x1)
    if y1 - y0 < 16 or x1 - x0 < 16:
        raise F.StackProblem("The crop is too small: draw a bigger box.")
    return y0, y1, x0, x1


def default_window(an: BurstAnalysis, params: SuperParams, out_px: int = 768) -> tuple:
    """The input box whose result is about ``out_px`` square, around the sharpest place."""
    s = float(params.scale) if params.merge == "drizzle" and an.raw else 1.0
    half = int(out_px / s / 2) // 2 * 2
    H, W = an.shape_full
    cy, cx = an.sharp_centre
    y0 = int(np.clip(cy - half, 0, max(0, H - 2 * half))) // 2 * 2
    x0 = int(np.clip(cx - half, 0, max(0, W - 2 * half))) // 2 * 2
    return y0, min(H, y0 + 2 * half), x0, min(W, x0 + 2 * half)


def _robust_at(an: BurstAnalysis, k: int, ys, xs) -> np.ndarray:
    from scipy.ndimage import map_coordinates
    K = an.k_ana
    c = [(np.asarray(ys, np.float64) - (K - 1) / 2.0) / K, (np.asarray(xs, np.float64) - (K - 1) / 2.0) / K]
    return map_coordinates(np.asarray(an.robust[k], np.float32), c, order=1, mode="nearest")


def _sharp_weight(an: BurstAnalysis, k: int) -> float:
    return float(np.clip(an.sharp[k] / max(an.sharp[an.ref], 1e-12), 0.3, 1.0))


def _shifted_flow(fl, y0, x0):
    if fl is None:
        return None
    return A.Flow(fl.dy, fl.dx, fl.quality, fl.y0 - y0, fl.x0 - x0, fl.step)


def _average(an: BurstAnalysis, params: SuperParams, box, alloc, progress, cancel, p0=0.0, p1=1.0):
    """Robust average (and the sky's sigma clip / winsorized), scale 1, one frame in memory at a time."""
    from ..pipeline.edit import Edit
    from scipy.ndimage import binary_erosion
    edit = Edit.from_dict(an.edit)
    y0, y1, x0, x1 = box
    H, W = y1 - y0, x1 - x0
    kept = an.kept(params)
    H0, W0 = an.shape_full
    mode = params.merge if params.merge in ("sigma", "winsorized") else "average"
    passes = 2 if mode != "average" else 1
    acc = alloc("acc", (H, W, 3))
    wsum = alloc("wsum", (H, W))
    sq = alloc("sq", (H, W)) if passes == 2 else None
    rej = alloc("rej", (H, W)) if passes == 2 else None
    mean = None
    std = None
    total = passes * len(kept)
    done = 0
    for ps in range(passes):
        if ps == 1:
            mean = alloc("mean", (H, W, 3))
            std = alloc("std", (H, W))
            for y in range(0, H, 1024):
                w = np.maximum(wsum[y:y + 1024], 1e-9)
                mean[y:y + 1024] = acc[y:y + 1024] / w[..., None]
                m_l = A.luma(np.asarray(mean[y:y + 1024]))
                std[y:y + 1024] = np.sqrt(np.maximum(sq[y:y + 1024] / w - m_l * m_l, 0.0))
            acc[...] = 0
            wsum[...] = 0
        for k in kept:
            _check(cancel)
            _tick(progress, p0 + (p1 - p0) * done / total,
                  f"{'Averaging' if ps == 0 else 'Leaving out what does not fit'}: frame {done % len(kept) + 1} "
                  f"of {len(kept)}")
            done += 1
            T = an.t_full[k]
            fl = an.flows[k]
            mg = 16 + (int(np.ceil(np.max(np.hypot(fl.dy, fl.dx)))) if fl is not None else 0)
            ys, xs = T.apply(np.array([y0, y0, y1, y1], float), np.array([x0, x1, x0, x1], float))
            ry0 = int(max(0, np.floor(ys.min()) - mg)) // 2 * 2
            rx0 = int(max(0, np.floor(xs.min()) - mg)) // 2 * 2
            ry1 = int(min(H0, np.ceil(ys.max()) + mg))
            rx1 = int(min(W0, np.ceil(xs.max()) + mg))
            if ry1 - ry0 < 8 or rx1 - rx0 < 8:
                continue
            whole = (ry0, ry1, rx0, rx1) == (0, H0, 0, W0)
            region = None if whole else (rx0, ry0, rx1 - rx0, ry1 - ry0)
            try:
                rgb = F.scene_rgb(an.paths[k], edit, False, region)
            except Cancelled:
                raise
            except Exception as e:                   # noqa: BLE001  (a frame that fails now is skipped, named)
                log.warning("superres: frame %s skipped in the merge (%s)", os.path.basename(an.paths[k]), e)
                continue
            Tl = A.Transform.translation(-ry0, -rx0).compose(T).compose(A.Transform.translation(y0, x0))
            al = A.warp(rgb, Tl, (H, W), order=3, flw=_shifted_flow(fl, y0, x0))
            cov = binary_erosion(A.coverage(Tl, rgb.shape, (H, W)), iterations=2)
            del rgb
            if an.offsets_bg[k]:
                al -= np.float32(an.offsets_bg[k])
            wk = _sharp_weight(an, k) if mode == "average" else 1.0
            for y in range(0, H, 512):
                yy = np.arange(y + y0, min(y1, y + y0 + 512), dtype=np.float64)
                Yg, Xg = np.meshgrid(yy, np.arange(x0, x1, dtype=np.float64), indexing="ij")
                r = _robust_at(an, k, Yg, Xg).astype(np.float32) if mode == "average" else \
                    np.ones(Yg.shape, np.float32)
                w = r * np.float32(wk) * cov[y:y + 512]
                f = al[y:y + 512]
                if ps == 0:
                    acc[y:y + 512] += f * w[..., None]
                    wsum[y:y + 512] += w
                    if sq is not None:
                        lf = A.luma(f)
                        sq[y:y + 512] += w * lf * lf
                else:
                    m = np.asarray(mean[y:y + 512])
                    s = np.asarray(std[y:y + 512])
                    lim = np.float32(params.clip_k) * s + 1e-6
                    if mode == "sigma":
                        ok = np.abs(A.luma(f) - A.luma(m)) <= lim
                        w = w * ok
                        rej[y:y + 512] += (~ok) & (w == 0) & cov[y:y + 512]
                        acc[y:y + 512] += f * w[..., None]
                    else:
                        fc = np.clip(f, m - lim[..., None], m + lim[..., None])
                        rej[y:y + 512] += np.any(fc != f, axis=2) & cov[y:y + 512]
                        acc[y:y + 512] += fc * w[..., None]
                    wsum[y:y + 512] += w
            del al
    out = alloc("out", (H, W, 3))
    for y in range(0, H, 1024):
        w = wsum[y:y + 1024]
        good = w > 1e-6
        o = acc[y:y + 1024] / np.maximum(w, 1e-9)[..., None]
        if mean is not None:                         # everything left out at a place: the plain mean there
            o = np.where(good[..., None], o, np.asarray(mean[y:y + 1024]))
        out[y:y + 1024] = o
    extra = {}
    if rej is not None:
        extra["rejected"] = np.asarray(rej) / max(1, len(kept))
    return out, extra


def _ca_move(qy, qx, c, an: BurstAnalysis, info):
    """Red / blue photosite positions moved to where their light belongs (the lens's lateral colour fringing)."""
    ca = an.ca
    if ca is None or c == 1:
        return qy, qx
    k1, k3 = ca.red if c == 0 else ca.blue
    vh, vw = info.visible_shape
    oy, ox = info.crop_origin
    cy, cx = (vh - 1) / 2.0 - oy, (vw - 1) / 2.0 - ox
    rmax = np.hypot((vh - 1) / 2.0, (vw - 1) / 2.0)
    Y, X = qy - cy, qx - cx
    rr = np.hypot(Y, X)
    r = rr / rmax
    d = 2.0 * (k1 * r + k3 * r ** 3)                  # plane pixels -> full-size pixels
    s = np.where(rr > 1e-6, d / np.maximum(rr, 1e-6), 0.0)
    return qy - Y * s, qx - X * s


def _boxsum(x: np.ndarray, r: int) -> np.ndarray:
    """Sum over a (2r+1)^2 window, as explicit shifted sums in a fixed order (the same answer whatever the size of
    the array around it: drizzle bands must give the same pixels as one piece)."""
    if r <= 0:
        return x.copy()
    p = np.pad(x, r, mode="edge")
    h, w = x.shape
    t = np.zeros((h + 2 * r, w), np.float32)
    for d in range(2 * r + 1):
        t += p[:, d:d + w]
    o = np.zeros((h, w), np.float32)
    for d in range(2 * r + 1):
        o += t[d:d + h]
    return o


def _drizzle(an: BurstAnalysis, params: SuperParams, box, alloc, progress, cancel, band_rows: int = 512,
             p0=0.0, p1=1.0):
    """Drizzle from the raw mosaics into output bands. Returns (camera-to-ProPhoto RGB, extra)."""
    from ..color import camera as cam
    from ..color.spaces import apply_matrix
    y0, y1, x0, x1 = box
    s = float(params.scale)
    Hout, Wout = int(round((y1 - y0) * s)), int(round((x1 - x0) * s))
    a = max(0.3, float(params.pixfrac) * s)
    K = int(np.ceil(a)) + 1
    mult = np.asarray(an.mult, np.float64)
    kept = an.kept(params)
    nk = len(kept)
    info0 = an.infos[an.ref]
    M = cam.camera_to_working(info0.cam_xyz).astype(np.float32)
    share = {0: 0.25, 1: 0.5, 2: 0.25}
    floor = [0.08 * nk * share[c] / (s * s) for c in range(3)]
    mg = 8
    out = alloc("drizzle", (Hout, Wout, 3))
    filled = 0
    H0, W0 = an.shape_full
    inv = {k: an.t_full[k].inverse() for k in kept}
    fmax = {k: (float(np.max(np.hypot(an.flows[k].dy, an.flows[k].dx))) if an.flows[k] is not None else 0.0)
            for k in kept}
    wk = {k: _sharp_weight(an, k) for k in kept}
    nb = (Hout + band_rows - 1) // band_rows
    for bi, b0 in enumerate(range(0, Hout, band_rows)):
        _check(cancel)
        _tick(progress, p0 + (p1 - p0) * bi / nb, f"Drizzling: part {bi + 1} of {nb}")
        b1 = min(Hout, b0 + band_rows)
        a0, a1 = max(0, b0 - mg), min(Hout, b1 + mg)
        rows = a1 - a0
        num = np.zeros((3, rows * Wout), np.float32)
        den = np.zeros((3, rows * Wout), np.float32)
        ry0 = (a0 - a - 1 + 0.5) / s - 0.5 + y0
        ry1 = (a1 + a + 1 + 0.5) / s - 0.5 + y0
        for k in kept:
            T = an.t_full[k]
            fy, _ = T.apply(np.array([ry0, ry0, ry1, ry1]), np.array([x0 - 2.0, x1 + 2.0, x0 - 2.0, x1 + 2.0]))
            fm = fmax[k] + 6
            fr0 = int(max(0, np.floor(fy.min() - fm))) // 2 * 2
            fr1 = int(min(H0, np.ceil(fy.max() + fm) + 2))
            if fr1 - fr0 < 2:
                continue
            m = an.mosaics[k][fr0:fr1].astype(np.float32) * np.float32(1.0 / F.INT16_SCALE)
            info = an.infos[k]
            for c in range(3):
                idxs, ws, vs = [], [], []
                for i in (0, 1):
                    for j in (0, 1):
                        if int(info.pattern[i, j]) != c:
                            continue
                        pl = m[i::2, j::2]
                        ph, pw = pl.shape
                        qy = np.repeat((fr0 + i + 2.0 * np.arange(ph))[:, None], pw, 1).ravel()
                        qx = np.tile(j + 2.0 * np.arange(pw), ph)
                        vals = pl.ravel()
                        qy, qx = _ca_move(qy, qx, c, an, info)
                        py0, px0 = inv[k].apply(qy, qx)
                        py, px = py0, px0
                        fl = an.flows[k]
                        if fl is not None:
                            for _ in range(2):                # p + d(p) = T^-1(q): two fixed-point steps
                                dy, dx = fl.at(py, px)
                                py, px = py0 - dy, px0 - dx
                        del py0, px0
                        # output coordinates of the whole result (never relative to the band: the same photosite
                        # must get the same weights whatever the band size)
                        oy = s * (py - y0 + 0.5) - 0.5
                        ox = s * (px - x0 + 0.5) - 0.5
                        sel = (oy > a0 - a) & (oy < a1 - 1 + a) & (ox > -a) & (ox < Wout - 1 + a)
                        if not sel.any():
                            continue
                        oy, ox, py, px, vals = oy[sel], ox[sel], py[sel], px[sel], vals[sel]
                        v = np.minimum(vals.astype(np.float64) * mult[c], 1.0)
                        r = _robust_at(an, k, py, px) * wk[k]
                        lo_y = oy - a / 2.0
                        lo_x = ox - a / 2.0
                        by0 = np.floor(lo_y + 0.5).astype(np.int64)
                        bx0 = np.floor(lo_x + 0.5).astype(np.int64)
                        for ty in range(K):
                            iy = by0 + ty
                            ovy = np.minimum(oy + a / 2.0, iy + 0.5) - np.maximum(lo_y, iy - 0.5)
                            for tx in range(K):
                                ix = bx0 + tx
                                ovx = np.minimum(ox + a / 2.0, ix + 0.5) - np.maximum(lo_x, ix - 0.5)
                                w = np.maximum(ovy, 0) * np.maximum(ovx, 0) * (r / (a * a))
                                ok = (iy >= a0) & (iy < a1) & (ix >= 0) & (ix < Wout) & (w > 0)
                                idxs.append((iy[ok] - a0) * Wout + ix[ok])
                                ws.append(w[ok])
                                vs.append(v[ok])
                if not idxs:
                    continue
                idx = np.concatenate(idxs)
                w = np.concatenate(ws)
                v = np.concatenate(vs)
                del idxs, ws, vs
                num[c] += np.bincount(idx, w * v, minlength=rows * Wout).astype(np.float32)
                den[c] += np.bincount(idx, w, minlength=rows * Wout).astype(np.float32)
                del idx, w, v
            del m
        # values, holes filled from neighbours (3 x 3, 5 x 5, ... 15 x 15: band margins are 8 rows)
        rgb = np.empty((rows, Wout, 3), np.float32)
        for c in range(3):
            n_ = num[c].reshape(rows, Wout)
            d_ = den[c].reshape(rows, Wout)
            val = n_ / np.maximum(d_, 1e-12)
            hole = d_ < floor[c]
            for r_ in (1, 2, 3, 5, 7):
                if not hole.any():
                    break
                sn, sd = _boxsum(n_, r_), _boxsum(d_, r_)
                fill = hole & (sd >= floor[c])
                val[fill] = sn[fill] / sd[fill]
                hole &= ~fill
            if hole.any():                           # nothing anywhere near: nearest value (very rare)
                from scipy.ndimage import distance_transform_edt
                _, (iy, ix) = distance_transform_edt(hole, return_indices=True)
                val = val[iy, ix]
            filled += int((d_[b0 - a0:b1 - a0] < floor[c]).sum())
            rgb[..., c] = val
        out[b0:b1] = apply_matrix(np.ascontiguousarray(rgb[b0 - a0:b1 - a0]), M)
        del num, den, rgb
    share_filled = filled / max(1.0, 3.0 * Hout * Wout)
    return out, {"filled_share": share_filled}


def _finish(img: np.ndarray, mode: str, amount: float, rows: int = 512):
    """Moon: starlet sharpening of the luminance (gains 1.6 / 1.3 / 1.1 on the three finest scales, times the
    slider). Hand-held: a light lift of the finest scale. Coefficients near the noise are left alone. In place,
    in bands (a big result is never copied whole)."""
    if amount <= 0 or mode not in ("moon", "handheld"):
        return img
    from ..raw.denoise import starlet, starlet_noise_sigma
    gains = (1.6, 1.3, 1.1) if mode == "moon" else (2.0, 1.0, 1.0)
    g = [1.0 + amount * (x - 1.0) for x in gains]
    H = img.shape[0]
    sample = A.luma(np.asarray(img[::max(1, H // 512)]))
    sig = A.noise_sigma(sample) / max(starlet_noise_sigma(3)[0] * 2.0, 1e-6) if sample.size else 0.0
    nsig = starlet_noise_sigma(3)
    pad = 32
    head = None
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        lo, hi = max(0, y0 - pad), min(H, y1 + pad)
        blk = np.asarray(img[lo:hi], np.float32).copy()
        if head is not None and lo < y0:
            blk[:y0 - lo] = head[-(y0 - lo):]         # original rows above (the band before was overwritten)
        new_head = np.asarray(img[max(0, y1 - pad):y1], np.float32).copy()
        Y = np.maximum(A.luma(blk), 1e-6)
        bands = starlet(Y, 3)
        Yn = Y.copy()
        for j in range(3):
            w = bands[j]
            t = 3.0 * sig * nsig[j]
            keep = 1.0 - np.exp(-(w / max(t, 1e-9)) ** 2)
            Yn += np.float32(g[j] - 1.0) * w * keep
        ratio = np.clip(np.maximum(Yn, 0) / Y, 0.5, 2.0)[..., None]
        img[y0:y1] = (blk * ratio)[y0 - lo:y1 - lo]
        head = new_head
    return img


def _defringe_inplace(img: np.ndarray, rows: int = 512):
    from ..lens.auto import defringe
    H = img.shape[0]
    pad = 16
    head = None
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        lo, hi = max(0, y0 - pad), min(H, y1 + pad)
        blk = np.asarray(img[lo:hi], np.float32).copy()
        if head is not None and lo < y0:
            blk[:y0 - lo] = head[-(y0 - lo):]
        new_head = np.asarray(img[max(0, y1 - pad):y1], np.float32).copy()
        img[y0:y1] = defringe(blk, 1.0)[y0 - lo:y1 - lo]
        head = new_head
    return img


def merge(an: BurstAnalysis, params: SuperParams, box=None, alloc=None, progress=None, cancel=None,
          band_rows: int = 512):
    """The merged result of ``box`` (input pixels of the reference frame; None = the params' crop or the whole
    frame), not yet upright. Returns (rgb scene-linear ProPhoto, extra)."""
    if alloc is None:
        def alloc(name, shape, dtype=np.float32):
            return np.zeros(shape, dtype)
    box = _box(an, params, box)
    method = params.merge
    if method == "drizzle" and not an.raw:
        method = "average"
    if method == "drizzle":
        out, extra = _drizzle(an, params, box, alloc, progress, cancel, band_rows, 0.0, 0.9)
        _defringe_inplace(out)
    else:
        p = SuperParams.from_dict(params.to_dict())
        p.merge = method
        out, extra = _average(an, p, box, alloc, progress, cancel, 0.0, 0.9)
    _finish(out, params.mode, float(params.sharpen))
    return out, extra


def preview(an: BurstAnalysis, params: SuperParams | None = None, box=None, progress=None, cancel=None):
    """A 100% window of the result (default: about 768 px square around the sharpest place), built the real
    way. Returns (result upright, best single frame of the same window at the same size upright, extra)."""
    from ..pipeline.pipeline import apply_orientation
    params = params or an.params
    win = box if box is not None else default_window(an, params)
    out, extra = merge(an, params, win, None, progress, cancel)
    ref = reference_window(an, params, win, out.shape[:2])
    return (np.ascontiguousarray(apply_orientation(out, an.flip)),
            np.ascontiguousarray(apply_orientation(ref, an.flip)), extra)


def reference_window(an: BurstAnalysis, params: SuperParams, win, out_shape) -> np.ndarray:
    """The reference frame alone, developed as usual and enlarged (bicubic) to the result's size: what B shows."""
    from ..pipeline.edit import Edit
    edit = Edit.from_dict(an.edit)
    y0, y1, x0, x1 = _box(an, params, win)
    rgb = F.scene_rgb(an.paths[an.ref], edit, False, (x0, y0, x1 - x0, y1 - y0))
    if rgb.shape[:2] == tuple(out_shape):
        return rgb
    s = out_shape[0] / float(rgb.shape[0])
    T = A.Transform(np.array([[1 / s, 0, (1 / s - 1) / 2.0], [0, 1 / s, (1 / s - 1) / 2.0], [0, 0, 1.0]]), "affine")
    return A.warp(rgb, T, out_shape, order=3)


def build(an: BurstAnalysis, params: SuperParams | None, out_path: str, progress=None, cancel=None,
          band_rows: int = 512) -> str:
    """The full result (whole frame or the crop), written as a linear intermediate with the sidecar's "stack"
    block. Returns the path."""
    from ..io.intermediate import write_intermediate
    from ..pipeline.pipeline import apply_orientation
    params = params or an.params
    with F.Scratch("srbuild") as sc:
        out, extra = merge(an, params, None, sc.array, progress, cancel, band_rows)
        _check(cancel)
        _tick(progress, 0.95, "Writing the result")
        scale = float(params.scale) if (params.merge == "drizzle" and an.raw) else 1.0
        info = {"kind": "superres", "mode": params.mode, "method": params.merge if an.raw or
                params.merge != "drizzle" else "average", "scale": scale, "frames": len(an.kept(params)),
                "sources": [os.path.basename(an.paths[k]) for k in an.kept(params)], "params": params.to_dict()}
        write_intermediate(out_path, apply_orientation(out, an.flip), an.meta, info)
        del out
    write_stack_block(out_path, "superres", [an.paths[k] for k in an.kept(params)], params.to_dict())
    _tick(progress, 1.0, "Done")
    return out_path


def result_tag(params: SuperParams, raw: bool = True) -> str:
    """Name tag (1.3): SuperRes2x, SuperRes15x, Stack (noise only), Moon2x, Sky."""
    if params.mode == "sky":
        return "Sky"
    s = float(params.scale) if (params.merge == "drizzle" and raw) else 1.0
    if s <= 1.0:
        return "Stack"
    st = f"{s:g}".replace(".", "")
    return f"Moon{st}x" if params.mode == "moon" else f"SuperRes{st}x"
