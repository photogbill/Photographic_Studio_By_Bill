"""Depth Stack: depth measured two ways in a focus stack, fused into one map, and where the two agree
(BUILD_PLAN_DEPTH_STACK B).

Two independent measures of how far each place is:
  focus        which frame of the stack is sharpest here (Focus Stack's own sharpness measure, with the noise
               energy taken off), to a fraction of a frame by a three-point Gaussian fit through the sharpness of
               the best frame and its neighbours. Strong anywhere with texture, over the whole stack; only as fine
               as the focus step, and it cannot say on which side of a frame's plane a place lies.
  Dual Pixel   each frame's signed left-right offset between its two halves (dualpixel.depth.disparity): zero on
               that frame's plane of focus, growing on either side, to a fraction of a pixel. Fine and signed, but
               only trustworthy near the frame's own focus (a few pixels of offset) and silent on plain areas.

The stack ties them together without any lens data: where a place is confidently measured in two neighbouring
frames, the change of its offset from one frame to the next is one focus step, in pixels of offset. Every pair of
neighbours gets its own step (a hand-held stack does not step evenly); the median over the stack is ``s`` and the
spread is reported. Frame i's plane then sits at position q_i (in steps, q_0 = 0), and every frame near a place
gives its own estimate of that place's position, t_i = q_i - d_i / s. Those votes and the focus estimate are
averaged with weights from their own measured precision: the Dual Pixel noise is measured from the step
calibration itself (two frames' offsets at the same place differ by one step plus noise), the focus noise from how
far focus strays from Dual Pixel where both are sure. Measure two ways and trust the agreement (Bill's rule from his
RF work).

The agreement map is how far the votes at a place stray from the fused answer (RMS, in steps): the focus estimate
and every nearby frame's own Dual Pixel vote, so it shows focus against Dual Pixel and the frames against each other.
A large value is a finding (a leaf that moved, a reflection, a bad frame), never averaged away silently: it lowers
the confidence there and is shown and saved as its own map.

Labels (plan 23.1 colours): 2 = measured both ways (cyan), 1 = one way (amber), 0 = neither: filled in smoothly from
the measured places around it (violet), with confidence 0 so nothing filled is passed off as measured. The picture's
brightness never decides a height, measured or filled (_finish says why).

Without Dual Pixel the same result comes from focus alone (positions = frame numbers: uneven steps cannot be seen).
One Dual Pixel frame on its own gives its offset map (``from_dual_pixel``). No Qt here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ... import log as slog
from . import align as A

log = slog.get("engine.stack.depthfuse")

DP_LIMIT_PX = 6.0           # |offset| (full-size px) past which a frame's offset is not trusted (the search is +-8)
DP_CONF_MIN = 0.3           # a confident offset (for measuring the focus step)
NEAR_STEPS = 1.25           # Dual Pixel votes come from frames within about this many steps of the place
FOCUS_MIN = 0.15            # focus quality under this: focus has nothing to say here
DP_MIN = 0.2                # summed Dual Pixel weight under this: Dual Pixel has nothing to say here
DISAGREE_STEPS = 0.5        # agreement map: votes straying farther than this (steps, RMS) disagree (shown red)
COMPARE_DP = 1.0            # without focus, Dual Pixel votes worth this much (about two frames) can be compared
HALF_CONF_STEPS = 0.25      # confidence 0.5 = an expected error of a quarter of a step
MIN_PAIR_SAMPLES = 400      # places a pair of frames needs in common to measure its own step
MAX_STEP_SPREAD = 1.0       # the pairs' own steps straying more than this (a share of s) from each other: the Dual
                            # Pixel step cannot be trusted (too few places measured in both frames of each pair)
LABEL_FILLED, LABEL_ONE, LABEL_BOTH = 0, 1, 2

METHOD_INFO = {
    "focus+dual_pixel": {
        "name": "Focus and Dual Pixel, fused",
        "does": "Which frame is sharpest, refined by every frame's own Dual Pixel offset; the stack measures its "
                "own focus step.",
        "best": "textured surfaces shot wide open (bark, stone, moss): fine, signed height and a map of where the "
                "two measures agree",
        "weak": "plain areas (nothing to measure: filled in and marked); anything that moved between frames "
                "(shown in the agreement map)"},
    "focus": {
        "name": "Depth from focus",
        "does": "Which frame is sharpest at each place, to a fraction of a frame.",
        "best": "any focus stack, any camera",
        "weak": "only as fine as the focus step; uneven hand-held steps cannot be seen; plain areas are filled in"},
    "dual_pixel": {
        "name": "Dual Pixel offset (one frame)",
        "does": "The signed offset between the photo's two halves.",
        "best": "one Dual Pixel photo, near its plane of focus",
        "weak": "shallow: only a few pixels of offset either side of focus are trustworthy; plain areas filled in"},
    "estimated": {
        "name": "Estimated from one ordinary photo",
        "does": "Nothing is measured: dark is taken as deep at several sizes (the guess texture tools make), after "
                "the broad light is taken off.",
        "best": "flat textures shot straight on in soft, even light (bark, stone, plaster) when there is no focus "
                "stack or Dual Pixel photo",
        "weak": "colour and shadow are mistaken for shape (a dark stain becomes a pit, a pale stone a bump); the "
                "broad shape of the surface is lost; every map made from it is amber"},
    "mosaic": {
        "name": "Flat Mosaic",
        "does": "Many hand-held views of a surface stitched on its ground plane: each view's own depth (Dual Pixel, "
                "focus, or both) matched to its neighbours', the ground taken off, and the relief's size measured "
                "from the parallax between overlapping views.",
        "best": "roads, paths, floors and walls larger than one frame: a texture that does not repeat for metres",
        "weak": "plain areas (nothing to measure: filled in and marked); without overlapping relief the size of the "
                "relief stays relative; something that moved shows in the agreement map"},
}


@dataclass
class DepthResult:
    """A measured depth map (half size, the stack's box frame, not yet upright) and how it was made."""
    depth: np.ndarray                    # float32: position in focus steps (frame i's plane at positions[i]);
                                         # grows with frame order. One Dual Pixel frame: -offset in px
    confidence: np.ndarray               # 0..1 (0 where filled in)
    label: np.ndarray                    # uint8: LABEL_BOTH / LABEL_ONE / LABEL_FILLED
    agreement: np.ndarray                # float32 steps (RMS); NaN where the two were not both there to compare
    focus_depth: np.ndarray | None       # focus alone (steps), before smoothing
    focus_quality: np.ndarray | None     # 0..1
    dp_depth: np.ndarray | None          # Dual Pixel alone (steps), NaN where it has nothing to say
    method: str                          # focus+dual_pixel | focus | dual_pixel | estimated
    units: str                           # steps | frames | dual pixel px | estimated
    positions: list = field(default_factory=list)
    step_px: float | None = None         # s: offset change per focus step, full-size px, signed
    step_spread: float | None = None     # spread of the pairs' own steps, as a share of s
    pair_steps: list = field(default_factory=list)
    sigma_dp: float | None = None        # measured Dual Pixel noise of one frame's vote, steps (at typical conf)
    sigma_focus: float | None = None     # measured focus noise, steps (at typical quality)
    notes: list = field(default_factory=list)
    dof_half: float | None = None        # half a frame's depth of field, steps (sharpness above half its peak)
    gap: np.ndarray | None = None        # bool: measured places sharp in no frame (BUILD_PLAN_DP3D B.3)
    coach: list = field(default_factory=list)   # plain sentences: where the focus jumped, what to do next time

    def shares(self) -> dict:
        lab = self.label
        n = float(lab.size) or 1.0
        ag = self.agreement
        cmp_ = np.isfinite(ag)
        if self.method == "estimated":           # one ordinary photo: nothing measured, everything estimated
            return {"measured_both": 0.0, "measured_one": 0.0, "filled": 0.0, "disagree": 0.0, "estimated": 1.0}
        out = {"measured_both": round(float((lab == LABEL_BOTH).sum()) / n, 4),
               "measured_one": round(float((lab == LABEL_ONE).sum()) / n, 4),
               "filled": round(float((lab == LABEL_FILLED).sum()) / n, 4),
               "disagree": round(float((cmp_ & (np.nan_to_num(ag) > DISAGREE_STEPS)).sum()) / n, 4)}
        if self.gap is not None:
            out["gap"] = round(float(self.gap.sum()) / n, 4)
        return out

    def agreement_stats(self) -> dict:
        ag = self.agreement[np.isfinite(self.agreement)]
        if ag.size < 100:
            return {"available": False}
        return {"available": True, "median_steps": round(float(np.median(ag)), 3),
                "p90_steps": round(float(np.percentile(ag, 90)), 3),
                "compared_share": round(float(ag.size) / self.agreement.size, 4)}


# ------------------------------------------------------------------------------------------ focus
def focus_peak(energy, idx: np.ndarray, noise_e: np.ndarray, presence: np.ndarray | None = None):
    """(k, quality): the sharpest frame to a fraction (three-point Gaussian fit on the sharpness of the best frame
    and its two neighbours) and how clearly the place has one focus (1 - the sharpness two frames away / the
    peak, times ``presence``: is there detail here at all). ``energy`` is (N, h, w), read one frame at a time."""
    n = int(energy.shape[0])
    idx = np.asarray(idx).astype(np.int32)
    floor = (0.25 * np.asarray(noise_e, np.float32) + np.float32(1e-12))
    nan = np.float32(np.nan)
    e = {o: np.full(idx.shape, nan, np.float32) for o in (-2, -1, 0, 1, 2)}
    for j in range(n):
        ej = np.asarray(energy[j], np.float32) + floor
        for o in (-2, -1, 0, 1, 2):                  # e[o] = sharpness of frame idx + o
            m = idx == (j - o)
            if m.any():
                e[o][m] = ej[m]
    l0 = np.log(e[0])
    lm, lp = np.log(e[-1]), np.log(e[1])
    den = lm - 2.0 * l0 + lp
    ok = np.isfinite(den) & (den < -1e-6)
    off = np.where(ok, 0.5 * (lm - lp) / np.where(ok, den, -1.0), 0.0)
    k = idx.astype(np.float32) + np.clip(np.nan_to_num(off), -0.5, 0.5).astype(np.float32)
    far = np.fmax(e[-2], e[2])                       # NaN only where both are off the stack's ends
    near = np.fmax(e[-1], e[1])
    far = np.where(np.isfinite(far), far, near)
    q = np.clip(1.0 - np.nan_to_num(far / e[0], nan=1.0), 0.0, 1.0).astype(np.float32)
    if n >= 3:
        # sharpest in the first or last frame is not a peak until the sharpness falls away from it as steeply as
        # it does beside a real peak: a place whose focus lies beyond the stack (the far background, a blurred
        # leaf in front) is sharpest at an end frame without being in focus there, and is not measured
        edge = (idx == 0) | (idx == n - 1)
        r_side = near / e[0]
        inner = ~edge & (q > 0.3) & np.isfinite(r_side)
        r_in = float(np.median(r_side[inner])) if inner.sum() > 200 else 0.7
        r_edge = np.where(idx == 0, e[1], e[-1]) / e[0]
        ok = np.clip((1.25 * r_in - np.nan_to_num(r_edge, nan=1.0)) / max(0.25 * r_in, 1e-3), 0.0, 1.0)
        q = np.where(edge, q * ok, q).astype(np.float32)
    if presence is not None:
        q *= np.asarray(presence, np.float32)
    return k, q


# ------------------------------------------------------------------------------------------ the focus step
def calibrate_steps(D, C, n: int, k: np.ndarray | None = None, kq: np.ndarray | None = None,
                    limit: float = DP_LIMIT_PX, have: list | None = None) -> dict:
    """The offset change from each frame to the next (one focus step, full-size px), from places confidently
    measured in both and within the trustworthy range; where focus knows the place it must lie near that pair.
    ``have``: the frames that have Dual Pixel offsets (a mixed stack, BUILD_PLAN_DP3D A.6: neighbouring Dual Pixel
    frames are paired across the ordinary ones between them, and the change is shared out over the frames it
    spans). Returns {"s", "pairs": [{pair, to, span, step_px, spread_px, samples}], "sigma_pair_px", "spread",
    "conf_mean"}; s is None when too few places could be measured. step_px is per frame of the span."""
    rng = np.random.default_rng(7)
    have = list(range(n)) if have is None else list(have)
    pairs, pooled, resid, confs = [], [], [], []
    for a, b in zip(have[:-1], have[1:]):
        span = b - a
        d0, d1 = np.asarray(D[a], np.float32), np.asarray(D[b], np.float32)
        c0, c1 = np.asarray(C[a], np.float32), np.asarray(C[b], np.float32)
        m = (c0 > DP_CONF_MIN) & (c1 > DP_CONF_MIN) & (np.abs(d0) < limit) & (np.abs(d1) < limit)
        if k is not None:
            known = kq >= FOCUS_MIN if kq is not None else np.ones(k.shape, bool)
            m &= ~known | ((k > a - 1.0) & (k < b + 1.0))
        diff = (d1 - d0)[m]
        cm = np.minimum(c0, c1)[m]
        if diff.size > 200_000:
            sel = rng.choice(diff.size, 200_000, replace=False)
            diff, cm = diff[sel], cm[sel]
        entry = {"pair": a, "to": b, "span": span}
        if diff.size < MIN_PAIR_SAMPLES:
            pairs.append(dict(entry, step_px=None, spread_px=None, samples=int(diff.size)))
            continue
        med = float(np.median(diff))
        mad = float(1.4826 * np.median(np.abs(diff - med)))
        pairs.append(dict(entry, step_px=med / span, spread_px=mad, samples=int(diff.size)))
        pooled.append(diff / span)
        resid.append(diff - med)
        confs.append(cm)
    if not pooled:
        return {"s": None, "pairs": pairs, "sigma_pair_px": None, "spread": None, "conf_mean": None}
    allp = np.concatenate(pooled)
    measured = [p["step_px"] for p in pairs if p["step_px"] is not None]
    # one step = the typical step between neighbours (median of the pairs' own steps); with fewer than three
    # pairs measured, the median over every place measured
    s = float(np.median(measured)) if len(measured) >= 3 else float(np.median(allp))
    if abs(s) < 1e-3:
        return {"s": None, "pairs": pairs, "sigma_pair_px": None, "spread": None, "conf_mean": None}
    r = np.concatenate(resid)
    sig = float(1.4826 * np.median(np.abs(r)))
    steps = np.array([p["step_px"] for p in pairs if p["step_px"] is not None])
    spread = float(1.4826 * np.median(np.abs(steps - np.median(steps))) / abs(s)) if steps.size >= 3 else \
        float(np.ptp(steps) / abs(s)) if steps.size == 2 else 0.0
    return {"s": s, "pairs": pairs, "sigma_pair_px": sig, "spread": spread,
            "conf_mean": float(np.mean(np.concatenate(confs)))}


def positions_from(cal: dict, n: int) -> tuple[list, list]:
    """(q, notes): each frame's plane of focus in steps (q_0 = 0) from the pairs' own steps; a pair that could not
    be measured takes the stack's median step. Frames without Dual Pixel (a mixed stack) sit evenly between the
    Dual Pixel frames either side of them, one step per frame beyond the last."""
    s = cal["s"]
    pairs = cal["pairs"]
    q: list = [None] * n
    first = pairs[0]["pair"] if pairs else 0
    q[first] = 0.0
    notes = []
    missing, back = [], []
    for p in pairs:
        a = p["pair"]
        b = p.get("to", a + 1)
        span = p.get("span", b - a)
        st = p["step_px"]
        if st is None:
            st = s
            missing.append(a)
        elif np.sign(st) != np.sign(s):
            back.append(a)
        q[b] = q[a] + span * st / s
        for j in range(a + 1, b):                    # ordinary frames between two Dual Pixel frames
            q[j] = q[a] + (j - a) * (q[b] - q[a]) / span
    known = [i for i in range(n) if q[i] is not None]
    for i in range(n):
        if q[i] is None:
            if i < known[0]:
                q[i] = q[known[0]] - (known[0] - i)
            else:
                q[i] = q[known[-1]] + (i - known[-1])
    if q[0] != 0.0:                                  # positions count from the first frame
        q0 = q[0]
        q = [x - q0 for x in q]
    if missing:
        notes.append(f"Dual Pixel could not measure the step between {len(missing)} pair(s) of frames (too little "
                     f"texture in focus in both): the stack's typical step was used there.")
    if back:
        notes.append("The focus went back on itself between frames " + ", ".join(f"{i + 1} and {i + 2}" for i in
                     back[:4]) + (" ..." if len(back) > 4 else "") + ": those frames are out of focus order.")
    return [float(x) for x in q], notes


# ------------------------------------------------------------------------------------------ Dual Pixel votes
def _dp_votes(D, C, q, s: float, center: np.ndarray | None, ref: np.ndarray | None = None,
              limit: float = DP_LIMIT_PX):
    """One pass over the frames: (mean of the votes t_i = q_i - d_i / s, summed weight, RMS of the votes about
    ``ref`` or None). Weights: each frame's confidence times a window around ``center`` (frames near the place);
    with ``center`` None, frames are preferred where their offset is small (near their own focus)."""
    sh = np.asarray(D[0]).shape
    num = np.zeros(sh, np.float32)
    den = np.zeros(sh, np.float32)
    sq = np.zeros(sh, np.float32) if ref is not None else None
    inv = np.float32(1.0 / s)
    for i in range(len(q)):
        d = np.asarray(D[i], np.float32)
        c = np.asarray(C[i], np.float32)
        t = np.float32(q[i]) - d * inv
        if center is None:
            g = np.exp(-np.square(d * inv / np.float32(NEAR_STEPS)))
        else:
            g = np.exp(-np.square((np.float32(q[i]) - center) / np.float32(NEAR_STEPS)))
        w = np.where((np.abs(d) < limit) & (c > 0.05), c * g, 0.0).astype(np.float32)
        num += w * t
        den += w
        if sq is not None:
            sq += w * np.square(t - ref)
    mean = num / np.maximum(den, 1e-12)
    rms = np.sqrt(sq / np.maximum(den, 1e-12)) if sq is not None else None
    return mean.astype(np.float32), den, rms


def supported(m: np.ndarray, quality: np.ndarray | None = None, size: int = 9, share: float = 0.3,
              min_area: float = 2e-4, keep_quality: float = 0.45) -> np.ndarray:
    """A place counts as measured only with measured neighbours (at least ``share`` of a size x size window), and
    only in a measured region of at least ``min_area`` of the picture unless that region is clearly measured (mean
    ``quality`` of ``keep_quality`` or more: one in-focus blade of grass). An out-of-focus background throws up
    scattered clumps that just pass (noise, tiny highlights): on Bill's fly stack 2000 of them, quality about 0.2,
    against 0.76 on the fly and the leaves; a fill would copy them as bumps."""
    from scipy.ndimage import label, mean as lmean, uniform_filter
    m = m & (uniform_filter(m.astype(np.float32), size, mode="nearest") >= share)
    if not m.any():
        return m
    lab, n = label(m)
    if n <= 1:
        return m
    sizes = np.bincount(lab.ravel())[1:]
    keep = sizes >= min_area * m.size
    if quality is not None:
        q = np.asarray(lmean(np.asarray(quality, np.float32), lab, np.arange(1, n + 1)))
        keep |= q >= keep_quality
    return np.concatenate([[False], keep])[lab]


# ------------------------------------------------------------------------------------------ filling, smoothing
def push_pull(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Values for the places with no weight, from the weighted values around them at ever coarser sizes (push-pull,
    Gortler et al. 1996): near a measured place the fill follows it closely, deep in a hole it is the broad mean."""
    v = np.asarray(v, np.float32)
    w = np.clip(np.asarray(w, np.float32), 0.0, None)
    levels = [(v * w, w)]
    while min(levels[-1][0].shape) > 4:
        num, den = levels[-1]
        h, ww = num.shape[0] // 2 * 2, num.shape[1] // 2 * 2
        if h < 2 or ww < 2:
            break
        levels.append((A.bin_by(num, 2) * 4.0, A.bin_by(den, 2) * 4.0))
    num, den = levels[-1]
    tot = float(den.sum())
    val = np.where(den > 0, num / np.maximum(den, 1e-12), (float(num.sum()) / tot) if tot > 0 else 0.0)
    for num, den in reversed(levels[:-1]):
        up = _upsample(val, num.shape)
        a = np.clip(den, 0.0, 1.0)
        val = a * (num / np.maximum(den, 1e-12)) + (1.0 - a) * up
    return val.astype(np.float32)


def _upsample(x: np.ndarray, shape) -> np.ndarray:
    from scipy.ndimage import zoom
    h, w = shape
    out = zoom(x, (h / x.shape[0], w / x.shape[1]), order=1, mode="nearest", grid_mode=True)
    if out.shape != (h, w):                          # rounding: pad or cut the last row / column
        out = np.pad(out, ((0, max(0, h - out.shape[0])), (0, max(0, w - out.shape[1]))), mode="edge")[:h, :w]
    return out.astype(np.float32)


def _finish(raw: np.ndarray, weight: np.ndarray, known: np.ndarray, guide: np.ndarray, smooth: float) -> np.ndarray:
    """Measured places keep their measured depth (only a light, weight-aware smoothing of a couple of pixels takes
    the speckle out); holes are filled from the measured places around them alone (push-pull: close to a measured
    edge the fill follows it, deep in a hole it is the broad mean), then smoothed. The picture's brightness is never
    let in: an edge-aware filter guided by the picture copies its pattern into what it smooths, so a dark knot would
    read as a hole, and an out-of-focus background's bright and dark blobs would become bumps (seen on Bill's fly
    stack). ``guide`` is kept for the signature; filled places are marked as filled by the caller."""
    from scipy.ndimage import gaussian_filter
    w = np.where(known, weight, 0.0).astype(np.float32)
    top = float(np.percentile(w[known], 95)) if known.any() else 1.0
    wn = np.clip(w / max(top, 1e-12), 0.0, 1.0)
    filled = np.where(known, raw, push_pull(np.where(known, raw, 0.0), wn)).astype(np.float32)
    if smooth <= 0:
        return filled
    sg = max(0.5, 1.5 * smooth)
    wk = (wn + 1e-3).astype(np.float32)
    meas = gaussian_filter(filled * wk, sg) / gaussian_filter(wk, sg)
    if known.all():
        return meas.astype(np.float32)
    fill = gaussian_filter(filled, max(2.0, 4.0 * smooth))
    # measured places: measured; filled places: the smooth fill; a soft seam of a few pixels between them
    k = np.clip(gaussian_filter(known.astype(np.float32), 2.0) * 2.0, 0.0, 1.0)
    k = np.where(known, np.maximum(k, 0.5), np.minimum(k, 0.5))
    return (k * meas + (1.0 - k) * fill).astype(np.float32)


# ------------------------------------------------------------------------------------------ the fusion
def fuse_maps(energy, idx, noise_e, presence, guide, D=None, C=None, smooth: float = 1.0) -> DepthResult:
    """The fused depth from a focus stack's sharpness (``energy`` (N, h, w), ``idx`` the sharpest frame, the
    noise energy, ``presence``) and, when given, each frame's Dual Pixel offset ``D`` (N, h, w; full-size px in
    the reference's frame) and its confidence ``C``. All arrays half size, the stack's box frame."""
    n = int(energy.shape[0])
    k, kq = focus_peak(energy, idx, noise_e, presence)
    has_f = supported(kq >= FOCUS_MIN, kq)
    notes: list = []
    if D is None or C is None:
        # focus alone has no second measure to check a place against: a place far from its measured neighbours
        # (an edge sharpest in the wrong frame) is dropped and filled instead
        known = has_f & ~_specks(k, has_f, sigma=4.0, k=3.0)
        depth = _finish(k, kq, known, guide, smooth)
        conf = np.where(known, np.clip(kq, 0, 1), 0.0).astype(np.float32)
        lab = np.where(known, LABEL_ONE, LABEL_FILLED).astype(np.uint8)
        return DepthResult(depth=depth, confidence=conf, label=lab, agreement=np.full(k.shape, np.nan, np.float32),
                           focus_depth=k, focus_quality=kq, dp_depth=None, method="focus", units="frames",
                           positions=[float(i) for i in range(n)], notes=notes)

    have = [i for i in range(n) if float(np.asarray(C[i], np.float32).max()) > 0.0]
    cal = calibrate_steps(D, C, n, k, kq, have=have if len(have) < n else None)
    s = cal["s"]
    if s is not None and (cal["spread"] or 0.0) > MAX_STEP_SPREAD:
        notes.append(f"Dual Pixel: the step between frames came out too uneven to trust ({100 * cal['spread']:.0f}% "
                     f"spread: too few places sharp in both frames of each pair), so depth comes from focus alone.")
        s = None
        cal = dict(cal, s=None)
    if s is None:
        if not notes:
            notes.append("Dual Pixel: too little texture in focus to measure the step between frames, so depth "
                         "comes from focus alone.")
        r = fuse_maps(energy, idx, noise_e, presence, guide, None, None, smooth)
        r.notes = notes + r.notes
        r.pair_steps = cal["pairs"]
        return r
    q, qn = positions_from(cal, n)
    notes += qn
    qa = np.asarray(q, np.float32)
    mono = bool(np.all(np.diff(qa) > 0))
    t_k = np.interp(k.ravel(), np.arange(n, dtype=np.float32), qa if mono else np.arange(n, dtype=np.float32)) \
        .reshape(k.shape).astype(np.float32)

    # Dual Pixel on its own: start from the frames whose offsets are small (near their own focus), then once more
    # around that answer
    t0, w0, _ = _dp_votes(D, C, q, s, None)
    t_dp, w_dp, _ = _dp_votes(D, C, q, s, t0)
    has_d = supported(w_dp >= DP_MIN, np.clip(w_dp / 2.0, 0.0, 1.0))

    # precision of each measure, from the stack itself
    sig_d = (cal["sigma_pair_px"] or 0.05) / np.sqrt(2.0) / abs(s)       # one frame's vote, steps
    sig_d = float(np.clip(sig_d, 0.01, 2.0))
    c_bar = float(cal["conf_mean"] or 0.5)
    both = has_f & has_d & (kq > 0.5 * float(np.median(kq[has_f])) if has_f.any() else has_f)
    if both.sum() > 200:
        res = (t_k - t_dp)[both]
        # an RMS held at 3 x the robust spread: focus errors have long tails (texture, uneven blur) that a median
        # alone does not see, and an underrated focus noise lets focus pull the answer off
        mad = float(1.4826 * np.median(np.abs(res - np.median(res))))
        sr = float(np.sqrt(np.mean(np.minimum(np.square(res), (3.0 * max(mad, 1e-6)) ** 2))))
        sig_k = float(np.sqrt(max(sr * sr - sig_d * sig_d / max(float(np.median(w_dp[both])) / c_bar, 1.0),
                                  0.01 ** 2)))
        q_bar = float(np.median(kq[both]))
    else:
        sig_k, q_bar = 0.5, float(np.median(kq[has_f])) if has_f.any() else 0.5
    sig_k = float(np.clip(sig_k, 0.01, 3.0))

    def combine(center):
        t_d, w_d, _ = _dp_votes(D, C, q, s, center)
        W_d = np.where(w_d >= DP_MIN, w_d / np.float32(sig_d * sig_d * c_bar), 0.0).astype(np.float32)
        W_k = np.where(has_f, kq / np.float32(sig_k * sig_k * q_bar), 0.0).astype(np.float32)
        W = W_d + W_k
        t = (W_d * t_d + W_k * t_k) / np.maximum(W, 1e-12)
        return t.astype(np.float32), W, W_d

    center = np.where(has_f, t_k, t_dp).astype(np.float32)
    t1, _, _ = combine(center)
    fused, W, W_d = combine(t1)
    known = has_f | has_d
    fused = np.where(known, fused, np.where(has_f, t_k, t_dp)).astype(np.float32)

    # agreement: how far every vote here strays from the answer (RMS, steps): the focus estimate and each nearby
    # frame's own Dual Pixel vote. Focus against Dual Pixel, and the frames against each other: a leaf that moved
    # gives focus two peaks (focus then says it does not know) and the frames' votes contradict each other
    _, w_a, rms = _dp_votes(D, C, q, s, fused, ref=fused)
    wk = np.where(has_f, kq, 0.0).astype(np.float32)
    w_a = np.where(w_a >= DP_MIN, w_a, 0.0).astype(np.float32)
    sq = w_a * np.square(np.nan_to_num(rms)) + wk * np.square(t_k - fused)
    agree = np.sqrt(sq / np.maximum(w_a + wk, 1e-12)).astype(np.float32)
    cmp_ = (has_f & has_d) | (w_a >= COMPARE_DP)
    agree = np.where(cmp_, agree, np.nan).astype(np.float32)

    half = np.float32(1.0 / HALF_CONF_STEPS ** 2)
    conf = W / (W + half)
    conf = np.where(cmp_, conf * np.exp(-np.square(np.nan_to_num(agree) / np.float32(DISAGREE_STEPS))), conf)
    conf = np.where(known, conf, 0.0).astype(np.float32)
    lab = np.full(k.shape, LABEL_FILLED, np.uint8)
    lab[has_f ^ has_d] = LABEL_ONE
    lab[has_f & has_d] = LABEL_BOTH
    depth = _finish(fused, W, known, guide, smooth)
    st = cal["pairs"]
    log.info("depth fusion: %d frames, s %.3f px/step (spread %.1f%%), sigma dp %.3f focus %.3f steps, both %.1f%% "
             "one %.1f%% filled %.1f%%", n, s, 100 * (cal["spread"] or 0), sig_d, sig_k,
             100 * float((lab == 2).mean()), 100 * float((lab == 1).mean()), 100 * float((lab == 0).mean()))
    return DepthResult(depth=depth, confidence=conf, label=lab, agreement=agree, focus_depth=t_k, focus_quality=kq,
                       dp_depth=np.where(has_d, t_dp, np.nan).astype(np.float32), method="focus+dual_pixel",
                       units="steps", positions=[round(float(x), 4) for x in q], step_px=float(s),
                       step_spread=cal["spread"], pair_steps=st, sigma_dp=sig_d, sigma_focus=sig_k, notes=notes)


# ------------------------------------------------------------------------------------------ gaps (B.3)
HALF_PEAK = float(np.sqrt(2.0 * np.log(2.0)))   # a Gaussian falls to half its peak at 1.18 sigma
HALF_DISC_SEP = 8.0 / (3.0 * np.pi)              # the two half discs' centroids are 0.849 R apart (R: blur radius)
SHARP_BLUR_PX = 1.0                              # sharp at the pixel level: a blur disc of radius 1 full-size pixel


def focus_width(energy, idx: np.ndarray, noise_e: np.ndarray, kq: np.ndarray, q_min: float = 0.5) -> float | None:
    """How fast sharpness falls away from a place's best frame, in frames: the median sigma of the three-point
    Gaussian fit (focus_peak) over places with a clear focus. None when too few places have one."""
    n = int(energy.shape[0])
    if n < 3:
        return None
    idx = np.asarray(idx).astype(np.int32)
    inner = (idx > 0) & (idx < n - 1) & (np.asarray(kq) >= q_min)
    if inner.sum() < 200:
        return None
    floor = (0.25 * np.asarray(noise_e, np.float32) + np.float32(1e-12))
    nan = np.float32(np.nan)
    e = {o: np.full(idx.shape, nan, np.float32) for o in (-1, 0, 1)}
    for j in range(n):
        ej = np.asarray(energy[j], np.float32) + floor
        for o in (-1, 0, 1):
            m = inner & (idx == (j - o))
            if m.any():
                e[o][m] = ej[m]
    den = np.log(e[-1]) - 2.0 * np.log(e[0]) + np.log(e[1])
    ok = inner & np.isfinite(den) & (den < -1e-4)
    if ok.sum() < 200:
        return None
    sig = np.sqrt(-1.0 / den[ok])                    # log of a Gaussian: second difference -1 / sigma^2
    return float(np.median(sig))


def gaps(depth: np.ndarray, label: np.ndarray, positions: list, dof_half: float) -> tuple[np.ndarray, list]:
    """(gap map, coach sentences). A measured place farther than half a depth of field from every frame's plane
    was sharp in no frame: a gap. The coach names the pairs of frames the focus jumped between."""
    q = np.asarray(positions, np.float32)
    dist = np.full(depth.shape, np.inf, np.float32)
    for x in q:
        dist = np.minimum(dist, np.abs(depth - x))
    gap = (label > LABEL_FILLED) & (dist > np.float32(dof_half))
    coach = []
    jumps = [(i, float(q[i + 1] - q[i])) for i in range(len(q) - 1) if abs(q[i + 1] - q[i]) > 2.0 * dof_half]
    for i, j in jumps[:4]:
        lo, hi = min(q[i], q[i + 1]), max(q[i], q[i + 1])
        share = float((gap & (depth > lo) & (depth < hi)).mean())
        if share < 0.002:
            continue
        coach.append(f"Between frames {i + 1} and {i + 2} the focus jumped {abs(j):.1f} steps: places at that depth "
                     f"(about {100 * share:.0f}% of the picture) are a little soft. Rock a little slower there next "
                     f"time.")
    if len(jumps) > 4:
        coach.append(f"The focus jumped in {len(jumps)} places in all.")
    return gap, coach


def _add_gaps(res: DepthResult, energy, idx, noise_e):
    """Fill in res.dof_half, res.gap and res.coach (B.3). With Dual Pixel the depth of field comes from optics: one
    step moves the offset by s, so the blur disc grows by s / 0.849 pixels of radius per step (the half discs'
    centroids are 0.849 R apart), and a place is sharp within SHARP_BLUR_PX of radius. Without Dual Pixel, from how
    fast sharpness falls away from a place's best frame (half its peak)."""
    if not res.positions or len(res.positions) < 2:
        return res
    if res.step_px:
        res.dof_half = float(SHARP_BLUR_PX * HALF_DISC_SEP / abs(res.step_px))
    else:
        sig = focus_width(energy, idx, noise_e, res.focus_quality if res.focus_quality is not None else
                          np.ones_like(res.depth))
        if sig is None:
            return res
        res.dof_half = float(HALF_PEAK * sig)
    res.gap, res.coach = gaps(res.depth, res.label, res.positions, res.dof_half)
    if res.method == "focus":
        res.coach.append("Without Dual Pixel the frames' real focus positions cannot be seen: the steps are taken "
                         "as even, so gaps from an uneven hand only show where they are large.")
    return res


def fuse(an, smooth: float | None = None) -> DepthResult:
    """The fused depth of a Focus Stack analysis (focus.FocusAnalysis): Dual Pixel offsets used when the analysis
    kept them. Cached on the analysis (the same analysis and smoothness give the same answer)."""
    smooth = float(an.params.smooth if smooth is None else smooth)
    key = ("fused", round(smooth, 3))
    hit = an._depth.get(key)
    if hit is not None:
        return hit
    D = getattr(an, "dp_disp", None)
    C = getattr(an, "dp_conf", None)
    res = fuse_maps(an.energy, an.idx, an.noise_e, an.presence(), an.guide, D, C, smooth)
    res.notes = list(getattr(an, "dp_notes", []) or []) + res.notes
    try:
        _add_gaps(res, an.energy, an.idx, an.noise_e)
    except Exception:                                # noqa: BLE001  (the coach is advice; the depth stands)
        slog.log_exception("focus gaps", log)
    an._depth[key] = res
    return res


def _specks(v: np.ndarray, known: np.ndarray, sigma: float = 6.0, k: float = 4.0) -> np.ndarray:
    """Measured places far from their measured neighbourhood (more than k robust spreads from the local weighted
    mean): an offset measured where the window straddled a plain area, typically along the edge of the sky. With
    only one measure there is nothing else to check it against."""
    from scipy.ndimage import gaussian_filter
    w = known.astype(np.float32)
    x = np.where(known, v, 0.0).astype(np.float32)
    den = np.maximum(gaussian_filter(w, sigma), 1e-6)
    mu = gaussian_filter(x * w, sigma) / den
    var = np.maximum(gaussian_filter(x * x * w, sigma) / den - mu * mu, 0.0)
    r = np.abs(v - mu)[known]
    if r.size < 100:
        return np.zeros_like(known)
    floor = float(1.4826 * np.median(r))
    return known & (np.abs(v - mu) > k * np.maximum(np.sqrt(var), floor))


def from_dual_pixel(disp: np.ndarray, conf: np.ndarray, guide_luma: np.ndarray, smooth: float = 1.0,
                    edge: int = 12) -> DepthResult:
    """One Dual Pixel photo: depth = -offset (px; the sign of near is the camera's own, see dualpixel.depth),
    filled and smoothed like the stack's. Half size, the camera crop's frame. ``edge``: half the measuring window
    (half-size px; disparity's default window is 48 wide at half size, so its offsets reach about 24 px)."""
    d = np.asarray(disp, np.float32)
    c = np.asarray(conf, np.float32)
    known = (c >= DP_MIN) & (np.abs(d) < DP_LIMIT_PX)   # an offset pinned at the end of the search is not one
    raw = -d
    # an offset measured with its window half over a plain area (the edge of the sky) leans toward whatever the
    # texture inside the window does: with only one measure there is nothing to check it against, so places within
    # half a window of an area that could not be measured are filled from inside instead, and stray specks dropped
    if known.any() and not known.all():
        from scipy.ndimage import binary_erosion
        known = binary_erosion(known, np.ones((2 * edge + 1, 2 * edge + 1), bool), border_value=1)
    known &= ~_specks(raw, known)
    if known.any():                                  # measured places measured; the rest filled from them only
        g = np.asarray(guide_luma, np.float32)
        g = np.sqrt(np.clip(g / max(float(np.percentile(g, 99.5)), 1e-6), 0, 1)).astype(np.float32)
        depth = _finish(np.where(known, raw, 0.0).astype(np.float32), np.where(known, c, 0.0), known, g, smooth)
    else:
        depth = np.zeros_like(d)
    lab = np.where(known, LABEL_ONE, LABEL_FILLED).astype(np.uint8)
    return DepthResult(depth=depth.astype(np.float32), confidence=np.where(known, c, 0.0).astype(np.float32),
                       label=lab, agreement=np.full(d.shape, np.nan, np.float32), focus_depth=None,
                       focus_quality=None, dp_depth=np.where(known, raw, np.nan).astype(np.float32),
                       method="dual_pixel", units="dual pixel px")
