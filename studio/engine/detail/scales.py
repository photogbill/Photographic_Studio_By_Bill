"""Stage 16, revised: "Detail by scale" (BUILD_PLAN.md section 23.11).

Photographers argue endlessly over what microcontrast is. Bill's ruling: it is the richness of tonal
transitions, "more shades of shadow", and it shows when a photo is turned to black and white. The other
meanings are not argued about, they are offered: every method ships with its own on/off switch and its own
Amount, and any number of them can be on at once.

    method              what it does                                              scale (6000 px frame)
    ------------------  --------------------------------------------------------  ---------------------
    veil                takes off (or adds) a uniform stray-light haze floor       whole frame
    separation          spreads crowded tones apart, region by region              no scale boost
    broad               large shapes, overall punch                                100 px and up
    tonal (default)     medium scales up, the finest scale slightly relaxed        8-90 px (+ 1-4 px down)
    local               the first engine: medium-scale differences made larger     8-90 px (a bump)
    crisp               the finest scale up (negative: gentle softening)           1-4 px

Rules for combining (23.11)
---------------------------
1. Fixed order, whatever order the sliders were moved in: veil, then tonal separation, then the scale methods.
2. One shared budget per scale. Methods that touch the same scale add into one gain for that scale; the edge
   guard, the noise protection and the no-tone-reversal limit apply to the TOTAL. Stacking three methods can
   never produce halos or amplify noise that one method alone could not.
3. ``plan`` reports, per band, what each method contributed and whether the total hit a guard (the panel's meter).

How each part works
-------------------
* Medium (tonal, local): fast local Laplacian filter on log2 luminance (Paris et al. 2011; Aubry et al. 2014).
  ``local`` is the engine in microcontrast.py, unchanged. ``tonal`` uses the same filter with the paper's
  smooth detail remapping,

        f(d) = d + A * d * exp(-d^2 / (2 * 0.5^2))                   d in stops from the local tone,

  which lifts small differences by (1 + A), fades out by 1.5 stops (edges are untouched) and whose slope never
  drops below 1 - 0.446 A, so tones cannot swap order while A <= 2.24. The first engine's taper stops at 1.25
  and, on its coarse grid of reference tones, delivers about a third of that; this is what lets the effect be
  plainly visible at screen size. n is the noise protection, applied per pyramid level: where the level holds
  nothing above twice its own noise (and never below 0.02 stops), nothing is changed. The shared budget per
  pyramid level is
  0.8 * A_local + 0.446 * A_tonal <= 1; a stack that asks for more is scaled back together.
* Fine (crisp, and tonal's relaxing): gain on the finest Laplacian coefficients, with the same noise toe as the
  medium band and an edge taper (coefficients above 1.5 stops are edges and are left alone).
* Broad and tonal separation and veil are measured on a small copy of the WHOLE frame (the "proxy") and then
  sampled at each pixel's place on the sensor, so a 100% view and the export always agree and a region render
  needs no extra context.
    - broad: the tonal filter on the proxy's coarse levels: a smooth brightness field.
    - separation: per region (a grid of tiles), tones are pushed away from the centre of whatever tones are
      crowded around them in that region, so a crowd spreads apart while its centre stays put (flat areas keep
      their brightness, smooth gradients are untouched), blended between tiles.
    - veil: stray light sets a grey floor that dark tones cannot go below, so they merge (Chambers / Zeiss).
      The floor is the colour of the frame's darkest 0.5%; removing it, with mid grey held, IS shadow
      microcontrast. A negative amount adds a haze instead (the "Soft glow" look).

Colour: every change is a luminance ratio applied to R, G and B in linear light, so hue and saturation stay.
"""
from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates, maximum_filter, uniform_filter

from ... import log as slog
from ..parallel import workers
from . import microcontrast as MC
from . import pyramid as P

log = slog.get("engine.scales")

LUMA = MC.LUMA
# The "lightness" that drives every new method. ProPhoto luminance gives blue a weight of 0.0001, so a violet
# flower counts as ten stops below mid grey and would get nothing; these weights still equal luminance for
# every neutral tone. (The first engine keeps true luminance, so old edits render as they did.)
TONE_W = np.array([0.25, 0.60, 0.15], np.float32)
REF_LONG_EDGE = MC.REF_LONG_EDGE
K_TONAL = 2.0               # remap amplitude of "tonal" at 100% (small differences about x2.8; the monotone limit is 2.24)
K_BROAD = 1.2
SIGMA = 0.5                 # stops: differences from the local tone up to about here are shades; 3x is an edge
SLOPE_COST = 2.0 * math.exp(-1.5)      # 0.446: the remap's slope never drops below 1 - SLOPE_COST * amplitude
T_MIN = 0.02                # stops: coefficients below this are never treated as detail, noise model or not
SOFT_MAX = 0.25             # most a negative medium / broad amount may take off
TONE_STEP = 0.25            # stops between the reference tones of the tonal filter (half of SIGMA: nearly all of the gain arrives)
EDGE_GUARD = 0.3            # band coefficients above this fraction of the range count as edges
K_CRISP = 1.5               # fine-coefficient gain of "crisp" at 100%
FINE_RELAX = 0.25           # how much "tonal" at 100% relaxes the finest scale
NEG = 0.8                   # a slider at -50% multiplies its band by 1 - 0.5 * NEG
R_TONAL = 3.0 * SIGMA       # stops from the local tone: beyond this the remap changes nothing
R_FINE = 1.5                # stops: fine coefficients above this are edges
FINE_MIN, FINE_MAX = -0.5, 2.0
PROXY_LONG = 1024           # long edge of the whole-frame copy used by veil, separation and broad
SEP_BINS, SEP_LO, SEP_HI = 64, -12.0, 4.0     # tone histogram: stops relative to mid grey
SEP_TILES = 8               # tiles along the long edge
SEP_K = 1.0                 # at 100% a crowd of similar tones spreads to about twice its width
SEP_WINDOW = 0.75           # stops: the window that decides which tones are "crowded together"
SEP_PRIOR = 0.3             # weight of "no crowd here" against the tile's own tones
SEP_REACH = 0.5             # stops: tones further than about this from a crowd's centre are pushed less
VEIL_REMOVE = 0.9           # fraction of the measured floor removed at 100%
VEIL_ADD = 0.08             # haze added at -100%, as a fraction of mid grey

# The panel's texts (rule 23.1: what it does / best for / weak at), shared by the app, the CLI and the tests.
METHODS = {
    "tonal": dict(name="Tonal", lo=0, hi=100, colour="cyan",
                  does="Medium scales up, the finest scale slightly relaxed. Bill's meaning of microcontrast.",
                  best="Modelling, shades of shadow, black and white.",
                  weak="Can look soft on subjects that need bite."),
    "local": dict(name="Local contrast", lo=-50, hi=100, colour="violet",
                  does="The first engine: medium-scale differences made larger, edges left alone.",
                  best="General depth.", weak="Subtle; weak in noisy shadows."),
    "crisp": dict(name="Crisp", lo=-50, hi=100, colour="amber",
                  does="The finest scale up. Negative softens gently. Judge it at 100%.",
                  best="Fur, feathers, fabric, lenses that are a little soft.",
                  weak="Raises grain; can look etched."),
    "separation": dict(name="Tonal separation", lo=0, hi=100, colour="green",
                       does="Spreads crowded tones apart region by region. No scale is boosted.",
                       best="Flat, dull files; dark-on-dark subjects.",
                       weak="Can look processed if pushed."),
    "veil": dict(name="Veil removal", lo=-100, hi=100, colour="rose",
                 does="Takes off the stray-light haze floor so dark tones separate. Negative adds a soft haze.",
                 best="Backlit and hazy scenes, uncoated lenses.",
                 weak="Nothing to gain on clean modern lenses."),
    "broad": dict(name="Broad", lo=-50, hi=100, colour="blue",
                  does="Large shapes, overall punch.", best="Flat light.",
                  weak="Deepens shadows; watch clipping."),
}
ORDER = ("veil", "separation", "broad", "tonal", "local", "crisp")        # the fixed order of rule 1

# Presets are just saved combinations of the method sliders (percent).
PRESETS = {
    "Tonal": dict(tonal=60),
    "Crisp": dict(crisp=60),
    "Both": dict(tonal=55, crisp=45, broad=20),
    "Soft glow": dict(tonal=25, crisp=-40, veil=-35),
}


@dataclass
class ScaleParams:
    """Amounts as fractions (the interface shows percent). 0 = that method is off."""
    tonal: float = 0.0           # 0..1
    local: float = 0.0           # -0.5..1 (the first engine's amount when positive)
    crisp: float = 0.0           # -0.5..1
    separation: float = 0.0      # 0..1
    veil: float = 0.0            # -1..1: + removes haze, - adds it
    broad: float = 0.0           # -0.5..1
    shadows: float = 1.0         # relative strength by tone, shared by every method (shadows first)
    midtones: float = 0.7
    highlights: float = 0.3
    scale: float = 0.0           # "local" only: -1 finer .. +1 broader
    range_ev: float = 2.0        # "local" only

    def clamped(self) -> "ScaleParams":
        c = lambda v, lo, hi: float(min(max(float(v or 0.0), lo), hi))
        return ScaleParams(c(self.tonal, 0, 1), c(self.local, -0.5, 1), c(self.crisp, -0.5, 1),
                           c(self.separation, 0, 1), c(self.veil, -1, 1), c(self.broad, -0.5, 1),
                           c(self.shadows, 0, 1), c(self.midtones, 0, 1), c(self.highlights, 0, 1),
                           c(self.scale, -1, 1), float(max(self.range_ev, 0.1)))

    @property
    def only_local(self) -> bool:
        """Nothing but the first engine with a positive amount: rendered by the unchanged old code path, so a
        sidecar written before this panel existed renders bit for bit as it did."""
        return self.local > 1e-4 and not any(abs(getattr(self, k)) > 1e-4 for k in
                                              ("tonal", "crisp", "separation", "veil", "broad"))

    @property
    def scales_active(self) -> bool:
        return (any(abs(getattr(self, k)) > 1e-4 for k in ("tonal", "local", "crisp", "broad"))
                and max(self.shadows, self.midtones, self.highlights) > 1e-4)

    @property
    def needs_proxy(self) -> bool:
        return any(abs(getattr(self, k)) > 1e-4 for k in ("separation", "veil", "broad"))

    @property
    def active(self) -> bool:
        return self.scales_active or self.needs_proxy

    @property
    def beyond_veil(self) -> bool:
        """Anything to do after the veil step (which the pipeline runs first, before any shadow lift)."""
        return self.scales_active or abs(self.separation) > 1e-4 or abs(self.broad) > 1e-4

    def legacy(self) -> MC.MCParams:
        return MC.MCParams(amount=max(self.local, 0.0), shadows=self.shadows, midtones=self.midtones,
                           highlights=self.highlights, scale=self.scale, range_ev=self.range_ev)


# ------------------------------------------------------------------------------------------- simple level
def from_simple(amount: float, balance: float) -> dict:
    """Simple level: Amount 0..1 and Balance -1 (Tonal) .. 0 (both) .. +1 (Crisp) -> method amounts.
    Fine is never driven up without Medium unless the user goes all the way to Crisp (the bands are coupled)."""
    a, t = float(min(max(amount, 0.0), 1.0)), float(min(max(balance, -1.0), 1.0))
    return {"tonal": round(a * (1.0 - max(t, 0.0)), 4), "crisp": round(a * (1.0 + min(t, 0.0)), 4)}


def to_simple(tonal: float, crisp: float) -> tuple[float, float]:
    """The Amount and Balance that show a pair of method amounts (inverse of from_simple)."""
    tonal, crisp = max(float(tonal), 0.0), max(float(crisp), 0.0)
    a = max(tonal, crisp)
    if a < 1e-6:
        return 0.0, -0.5
    return a, (crisp / tonal - 1.0) if tonal >= crisp else (1.0 - tonal / crisp)


# --------------------------------------------------------------------------------------------- band plan
def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def band_weights(lg):
    """(fine, medium, broad) weight of features 2**lg pixels across on a 6000 px frame: fine up to about 4 px,
    medium about 8-90 px, broad from about 100 px."""
    lg = np.asarray(lg, np.float64)
    fine = np.clip(3.0 - lg, 0.0, 1.0)
    med = _smooth(lg - 2.0) * (1.0 - _smooth((lg - 6.3) / 1.2))
    broad = _smooth((lg - 6.5) / 1.1)
    return fine, med, broad


def _level_lg(n_levels: int, eff_scale: float, long_edge_full: float) -> np.ndarray:
    """log2 of the feature size (half a period, in pixels of a 6000 px frame) that each Laplacian level of an
    image rendered at ``eff_scale`` holds: level 0 holds 2 px features, level 1 holds 4 px, and so on."""
    norm = REF_LONG_EDGE / max(long_edge_full, 1.0)
    return np.arange(n_levels) + 1.0 + math.log2(norm / max(eff_scale, 1e-6))


def tone_weight(p: ScaleParams, s) -> np.ndarray:
    """0..1 strength at tone s (stops from mid grey): the shadows / midtones / highlights weighting."""
    sh, mid, hi = MC.tone_bands(s)
    return (p.shadows * sh + p.midtones * mid + p.highlights * hi) / max(1.0, max(p.shadows, p.midtones,
                                                                                 p.highlights))


def plan(p: ScaleParams, eff_scale: float = 1.0, long_edge_full: float = REF_LONG_EDGE, n_levels: int = 14) -> dict:
    """Per pyramid level of an image rendered at ``eff_scale``: the fine gain, the tonal remap amplitude and the
    weights of the first engine, after the shared budget; plus the meter's numbers per band."""
    p = p.clamped()
    lg = _level_lg(n_levels, eff_scale, long_edge_full)
    wf, wm, _ = band_weights(lg)
    neg = lambda v: v if v >= 0 else v * NEG
    # fine: crisp up (or down), tonal relaxes it; one total, clamped
    gf_parts = {"crisp": K_CRISP * neg(p.crisp) if p.crisp >= 0 else neg(p.crisp), "tonal": -FINE_RELAX * p.tonal}
    gf_raw = sum(gf_parts.values())
    gf_tot = float(min(max(gf_raw, FINE_MIN), FINE_MAX))
    gf = gf_tot * wf
    # medium: tonal (new remap) + local (first engine); negative local is a plain reduction of the tonal remap
    a_tonal = (K_TONAL * p.tonal + min(p.local, 0.0) * NEG) * wm
    w_leg = MC.level_weights(p.legacy(), eff_scale, long_edge_full, n_levels) if p.local > 1e-4 else np.zeros(n_levels)
    a_leg = max(p.local, 0.0) * MC.A_MAX * w_leg
    use = 0.8 * a_leg + np.maximum(a_tonal, 0.0) * SLOPE_COST
    k = np.where(use > 1.0, 1.0 / np.maximum(use, 1e-9), 1.0)
    a_tonal = np.where(a_tonal > 0, a_tonal * k, np.maximum(a_tonal, -SOFT_MAX))
    w_leg = w_leg * k
    med_parts = {"tonal": K_TONAL * p.tonal, "local": (MC.A_MAX if p.local > 0 else NEG) * p.local}
    med_use = 0.8 * max(p.local, 0.0) * MC.A_MAX + K_TONAL * p.tonal * SLOPE_COST
    a_broad = float(min(max(K_BROAD * p.broad if p.broad >= 0 else neg(p.broad), -SOFT_MAX), 0.95 / SLOPE_COST))
    bands = {
        "fine": {"parts": gf_parts, "total": gf_tot, "limited": abs(gf_raw - gf_tot) > 1e-6},
        "medium": {"parts": med_parts, "total": float(sum(med_parts.values()) / max(med_use, 1.0)),
                   "limited": bool(med_use > 1.0 + 1e-6)},
        "broad": {"parts": {"broad": a_broad}, "total": a_broad, "limited": False},
    }
    return {"gf": gf, "a_tonal": a_tonal, "w_leg": w_leg, "a_broad": a_broad, "bands": bands}


def meter(p: ScaleParams) -> dict:
    """What the panel's meter draws: per band, each method's share (as a fraction: +1.0 = small tonal differences
    in the shadows doubled), the total after the guards, and whether a guard limited it."""
    b = plan(p)["bands"]
    for band in b.values():
        scale = band["total"] / sum(band["parts"].values()) if abs(sum(band["parts"].values())) > 1e-9 else 1.0
        band["parts"] = {k: round(float(v * (scale if band["limited"] else 1.0)), 4)
                         for k, v in band["parts"].items() if abs(v) > 1e-6}
        band["total"] = round(float(band["total"]), 4)
    return b


def context(p: ScaleParams, long_edge_full: float) -> tuple[int, int]:
    """(pad, align) in full-resolution pixels that a region render needs (medium band only: broad, separation
    and veil come from the whole-frame proxy and need none)."""
    p = p.clamped()
    if p.only_local:
        return MC.context(p.legacy(), long_edge_full)
    if not p.scales_active:
        return 0, 2
    pl = plan(p, 1.0, long_edge_full, 14)
    on = np.nonzero((np.abs(pl["a_tonal"]) > 0.02) | (pl["w_leg"] > 0))[0]
    if on.size == 0:
        return 16, 8                       # fine band only
    align = int(2 ** (int(on[-1]) + 1))
    return 2 * align, align


# -------------------------------------------------------------------------------------------- medium band
def _remap_tonal(img, gamma, amount):
    """d -> d + amount * d * exp(-d^2 / (2 s^2)): the smooth detail remapping of the fast local Laplacian paper
    (Aubry et al. 2014), s = SIGMA stops. It has no feature narrower than the reference-tone grid, so a plain
    gradient comes out as it went in. (A noise toe inside the remap puts a notch at d = 0 that the grid cannot
    follow: it rippled smooth gradients by 1%. The noise protection is applied to the coefficients instead.)"""
    d = img - np.float32(gamma)
    b = np.exp(d * d * np.float32(-0.5 / (SIGMA * SIGMA)))
    b *= d
    b *= np.float32(amount)
    b += img
    return b


def _tone_lut(fn, mid_log2: float):
    """fn(tone) sampled an eighth of a stop apart, with the index function for an array of log2 tones."""
    tg = np.arange(-16.0, 8.01, 0.125)
    lut = np.asarray(fn(tg + mid_log2), np.float32)
    return lut, lambda L: np.clip((L - np.float32(mid_log2) + 16.0) * 8.0 + 0.5, 0.0, len(tg) - 1.0).astype(np.int32)


def tonal_delta(L: np.ndarray, a_levels: np.ndarray, tone_w, mid_log2: float, noise_at=None) -> np.ndarray:
    """Change in log2 luminance from the tonal remap, with per-level amplitudes ``a_levels`` (0 = finest level of
    L). tone_w(s) -> 0..1 strength at tone s (stops from mid grey). noise_at(level) -> function giving the noise
    (stops, at pixel level of L, after the denoiser) for a log2 tone. The filter result is linear in the
    amplitude, so one set of remapped pyramids at unit amplitude serves every level's own amplitude.

    The reference tones sit a quarter of a stop apart and are filtered in parallel.

    Noise protection: at every level, a coefficient below twice that level's noise (and never below T_MIN) is
    not detail, and nothing is changed there. A clean gradient has no coefficients at all, so it is left
    exactly alone."""
    on = np.nonzero(np.abs(a_levels) > 1e-4)[0]
    if on.size == 0:
        return np.zeros_like(L)
    l0, l1 = int(on[0]), int(on[-1])
    nlev = min(l1 + 1, P.max_levels(L.shape, 4))
    if nlev <= l0:
        return np.zeros_like(L)
    l1 = min(l1, nlev - 1)
    G = P.gaussian(L, nlev)
    step = TONE_STEP
    lo = max(min(float(G[lv].min()) for lv in range(l0, l1 + 1)), mid_log2 - 16.0)
    hi = min(max(float(G[lv].max()) for lv in range(l0, l1 + 1)), mid_log2 + 8.0)
    gam = mid_log2 + step * np.arange(int(math.floor((lo - mid_log2) / step)) - 1,
                                      int(math.ceil((hi - mid_log2) / step)) + 2)
    A = np.asarray(tone_w(gam - mid_log2), np.float64)
    sign = 1.0 if float(np.sum(a_levels[l0:l1 + 1])) >= 0 else -1.0
    I0 = G[l0]
    n = l1 - l0 + 1
    lap_in = [G[lv] - P.up(G[lv + 1], G[lv].shape) for lv in range(l0, l1 + 1)]
    Gc = [np.clip(G[lv], np.float32(gam[0]), np.float32(gam[-1])) for lv in range(l0, l1 + 1)]
    out = [np.zeros_like(x) for x in lap_in]                # sum over reference tones of weight * (filtered - input)
    lock = threading.Lock()

    def one(gi):
        g = float(gam[gi])
        a = float(A[gi])
        if a <= 1e-4:
            return                          # this tone is left as it is: it adds nothing to the change
        ws = [np.maximum(1.0 - np.abs(Gc[j] - np.float32(g)) / np.float32(step), 0.0) for j in range(n)]
        if not any(w.any() for w in ws):
            return
        GR = P.gaussian(_remap_tonal(I0, g, sign * a), n)
        parts = [ws[j] * (GR[j] - P.up(GR[j + 1], GR[j].shape) - lap_in[j]) if ws[j].any() else None
                 for j in range(n)]
        with lock:
            for j in range(n):
                if parts[j] is not None:
                    out[j] += parts[j]

    nw = workers()
    if nw <= 1 or len(gam) <= 1:
        for gi in range(len(gam)):
            one(gi)
    else:
        with ThreadPoolExecutor(max_workers=nw, thread_name_prefix="tones") as ex:
            for f in [ex.submit(one, gi) for gi in range(len(gam))]:
                f.result()
    nf = P.white_noise_factors()
    D = None
    for lv in range(l1, -1, -1):
        D = np.zeros_like(G[lv]) if D is None else P.up(D, G[lv].shape)
        if l0 <= lv <= l1 and abs(a_levels[lv]) > 1e-4:
            j = lv - l0
            mag = maximum_filter(np.abs(lap_in[j]), size=3)
            # edge guard: near a big tonal edge at this scale (and its skirts, two samples either side) nothing
            # is changed, so the lift never spills from an edge into the flat areas beside it
            big = maximum_filter(mag, size=3) / np.float32(EDGE_GUARD * R_TONAL)
            big *= big
            big *= big
            big += 1.0
            d = out[j] / big
            if sign > 0:                                    # softening is not held back by noise
                t = np.float32(T_MIN)
                if noise_at is not None:
                    lut, idx = _tone_lut(noise_at(lv), mid_log2)
                    t = np.maximum(np.float32(2.0 * nf[lv]) * lut[idx(G[lv])], np.float32(T_MIN))
                mag *= mag
                mag *= mag
                d *= mag / (mag + t ** 4)
            D += np.float32(sign * a_levels[lv]) * d
    return D


# ---------------------------------------------------------------------------------------------- fine band
def fine_delta(L: np.ndarray, gf: np.ndarray, tone_w, mid_log2: float, noise_log2=None) -> np.ndarray:
    """Change in log2 luminance from the gain on the finest Laplacian levels (``gf`` per level, 0 = finest)."""
    on = np.nonzero(np.abs(gf) > 1e-4)[0]
    if on.size == 0:
        return np.zeros_like(L)
    l1 = min(int(on[-1]), P.max_levels(L.shape, 4) - 1)
    if l1 < 0:
        return np.zeros_like(L)
    G = P.gaussian(L, min(l1 + 3, P.max_levels(L.shape, 4)))        # two more levels for the edge guard
    nf = P.white_noise_factors()
    tg = np.arange(-16.0, 8.01, 0.125)                      # tone look-up, an eighth of a stop apart
    tw_lut = np.asarray(tone_w(tg), np.float32)
    nz_lut = np.asarray(noise_log2(tg + mid_log2), np.float32) ** 4 if noise_log2 is not None else None
    D = None
    for lv in range(l1, -1, -1):
        D = np.zeros_like(G[lv]) if D is None else P.up(D, G[lv].shape)
        g = float(gf[lv])
        if abs(g) <= 1e-4:
            continue
        base = P.up(G[lv + 1], G[lv].shape)
        t4 = np.float32((2.0 * nf[lv]) ** 4)
        guard = None
        if g > 0:
            # Crisp is for texture. An edge and its skirts are left to capture sharpening (no overshoot), and so
            # is anything riding on a bigger structure at the next two coarser scales (the slope of a candle
            # flame is not texture, and must not grow a dark ring).
            guard = maximum_filter(np.abs(G[lv] - base), size=5)
            for up_lv in (lv + 1, lv + 2):
                if up_lv + 1 < len(G):
                    cb = maximum_filter(np.abs(G[up_lv] - P.up(G[up_lv + 1], G[up_lv].shape)), size=3)
                    for q in range(up_lv, lv, -1):
                        cb = P.up(cb, G[q - 1].shape)
                    np.maximum(guard, cb, out=guard)
            guard *= np.float32(1.0 / (EDGE_GUARD * R_FINE))
            guard *= guard
            guard *= guard
            guard += 1.0
        for y0 in range(0, base.shape[0], 512):             # in strips: a 45 MP frame must not need gigabytes
            b = base[y0:y0 + 512]
            c = G[lv][y0:y0 + 512] - b
            i0 = np.clip((b - np.float32(mid_log2) + 16.0) * 8.0 + 0.5, 0.0, len(tg) - 1.0).astype(np.int32)
            ac = np.abs(c)
            e = np.maximum(1.0 - ac * np.float32(1.0 / R_FINE), 0.0)
            e *= e
            if g > 0 and nz_lut is not None:                # the same noise protection as the medium band
                ac *= ac
                ac *= ac
                e *= ac / (ac + t4 * nz_lut[i0] + np.float32(1e-20))
            e *= tw_lut[i0]
            e *= c
            if guard is not None:
                e /= guard[y0:y0 + 512]
            D[y0:y0 + 512] += np.float32(g) * e
    return D


# ------------------------------------------------------------------------------- whole-frame proxy fields
@dataclass
class Fields:
    """What veil, tonal separation and broad measured on the whole-frame proxy (pre-exposure scene light)."""
    veil: np.ndarray | None = None          # RGB removed (+) or added (-), in post-exposure units
    veil_gain: float = 1.0                  # keeps mid grey where it was
    sep: np.ndarray | None = None           # [ty, tx, bin] offsets in stops
    broad: np.ndarray | None = None         # [py, px] offsets in stops
    shape: tuple = (1, 1)                   # proxy (h, w)
    report: dict | None = None


def _soft_sub(x, v):
    """x - v for x >> v, sliding smoothly to 0 at 0 (monotone, never negative)."""
    v = np.maximum(np.asarray(v, np.float32), 1e-12)
    return x - v * (1.0 - np.exp(-np.maximum(x, 0.0) / v))


def veil_floor(proxy: np.ndarray) -> np.ndarray:
    """Colour of the stray-light floor: the mean of the darkest 0.5% of the (lightly smoothed) proxy."""
    sm = uniform_filter(proxy, size=(3, 3, 1))
    Y = sm @ TONE_W
    thr = np.percentile(Y, 0.5)
    sel = sm[Y <= thr]
    return np.maximum(sel.mean(axis=0), 0.0).astype(np.float32) if sel.size else np.zeros(3, np.float32)


def _separation_table(S: np.ndarray, p: ScaleParams, noise_stops=None) -> np.ndarray:
    """Per tile, the tone offset (stops) by tone bin. S: proxy tones in stops from mid grey.

    For every tone L the tile's histogram gives m(L), the centre of the tones crowded around L (a Gaussian
    window 0.75 stops wide: the mean-shift step). Tones are pushed away from that centre,

        offset(L) = k * d * exp(-d^2 / (2 * 0.5^2)),      d = L - m(L),

    so a crowd of similar tones spreads to about (1 + k) times its width, its centre stays exactly where it was
    (a flat wall or sky does not change brightness, and neighbouring tiles agree on it: no halos), tones more
    than a stop from any crowd are left alone, and evenly spread tones (a smooth gradient) are not touched."""
    h, w = S.shape
    t = max(8, int(round(max(h, w) / SEP_TILES)))
    ny, nx = max(1, int(round(h / t))), max(1, int(round(w / t)))
    edges = np.linspace(SEP_LO, SEP_HI, SEP_BINS + 1)
    bw = edges[1] - edges[0]
    nodes = edges                                            # the table is sampled at the bin edges
    tw = np.asarray(tone_weight(p, nodes), np.float64)
    if noise_stops is not None:              # where a pixel's noise is a fifth of a stop, the push is halved
        tw = tw / (1.0 + (np.asarray(noise_stops(nodes), np.float64) / 0.2) ** 2)
    k = SEP_K * p.separation * tw
    centres = 0.5 * (edges[:-1] + edges[1:])
    kern = np.exp(-0.5 * ((nodes[:, None] - centres[None, :]) / SEP_WINDOW) ** 2)      # [node, bin]
    ksum = kern.sum(axis=1)
    table = np.zeros((ny, nx, SEP_BINS + 1), np.float32)
    ys = np.linspace(0, h, ny + 1).round().astype(int)
    xs = np.linspace(0, w, nx + 1).round().astype(int)
    for j in range(ny):
        for i in range(nx):
            # each tile reads a neighbourhood half a tile wider than itself, so neighbours agree more
            y0, y1 = max(0, ys[j] - t // 2), min(h, ys[j + 1] + t // 2)
            x0, x1 = max(0, xs[i] - t // 2), min(w, xs[i + 1] + t // 2)
            v = S[y0:y1, x0:x1].ravel()
            v = np.clip(v, SEP_LO, SEP_HI - 1e-6).astype(np.float64)
            hist = np.histogram(v, bins=edges)[0].astype(np.float64)
            tot = hist.sum()
            if tot <= 0:
                continue
            sums = np.histogram(v, bins=edges, weights=v)[0]             # the true tones, not the bin centres
            prior = SEP_PRIOR * tot / SEP_BINS                # thinly spread tones are no crowd: they stay put
            den = kern @ hist + prior * ksum
            m = (kern @ sums + prior * ksum * nodes) / np.maximum(den, 1e-30)
            d = nodes - m
            T = nodes + k * d * np.exp(-0.5 * (d / SEP_REACH) ** 2)
            for q in range(1, len(T)):                       # tones never swap order (slope at least 0.25)
                if T[q] < T[q - 1] + 0.25 * bw:
                    T[q] = T[q - 1] + 0.25 * bw
            table[j, i] = T - nodes
    return table


def measure_fields(proxy: np.ndarray, p: ScaleParams, gain: float, mid: float, noise_sigma_lin=None) -> Fields:
    """Veil floor, separation table and broad field from the whole-frame proxy (scene-linear, BEFORE exposure;
    ``gain`` = 2**exposure, ``mid`` = the post-exposure value rendered as mid grey)."""
    p = p.clamped()
    f = Fields(shape=proxy.shape[:2], report={})
    work = proxy.astype(np.float32) * np.float32(gain)
    if abs(p.veil) > 1e-4:
        if p.veil > 0:
            F = veil_floor(work)
            fy = float(F @ TONE_W)
            cap = 0.25 * mid                 # a floor brighter than this is a dark subject, not stray light
            if fy > cap:
                F = F * np.float32(cap / fy)
                fy = cap
            v = np.float32(VEIL_REMOVE * p.veil) * F
            f.report["veil_floor_ev"] = round(math.log2(max(fy, 1e-9) / mid), 2)
        else:
            mean = work.reshape(-1, 3).mean(axis=0)
            col = mean / max(float(mean @ TONE_W), 1e-9)
            v = -np.float32(VEIL_ADD * -p.veil * mid) * col.astype(np.float32)
        f.veil = v.astype(np.float32)
        vy = float(v @ TONE_W)
        after = float(_soft_sub(np.float32(mid), np.float32(vy))) if vy > 0 else mid - vy
        f.veil_gain = float(mid / max(after, 1e-9))
        work = apply_veil(work, f)
    Y = np.maximum(work @ TONE_W, np.float32(mid * 2.0 ** -14))
    S = np.log2(Y / np.float32(mid)).astype(np.float32)
    if p.separation > 1e-4:
        ns = None
        if noise_sigma_lin is not None:
            ns = lambda s: np.asarray(noise_sigma_lin(mid * np.exp2(s)), np.float64) / (mid * np.exp2(s) * math.log(2.0))
        f.sep = _separation_table(S, p, ns)
        S = S + sample_separation(f, S, *np.meshgrid(np.arange(S.shape[1], dtype=np.float32),
                                                      np.arange(S.shape[0], dtype=np.float32)))
    if abs(p.broad) > 1e-4:
        a = plan(p)["a_broad"]
        lg = np.arange(12) + 1.0 + math.log2(REF_LONG_EDGE / max(S.shape))
        wb = band_weights(lg)[2] * a
        wb[np.abs(wb) < 0.05 * abs(a)] = 0.0
        pad = int(min(256, min(S.shape) - 1))                # continued beyond the frame, as in apply()
        f.broad = tonal_delta(_extend(S, pad), wb, lambda s: tone_weight(p, s), 0.0)[
            pad:pad + S.shape[0], pad:pad + S.shape[1]].astype(np.float32)
    return f


def apply_veil(rgb: np.ndarray, f: Fields) -> np.ndarray:
    if f.veil is None:
        return rgb
    if float(f.veil @ TONE_W) > 0:
        out = _soft_sub(rgb, f.veil)
    else:
        out = rgb - f.veil
    out *= np.float32(f.veil_gain)
    return out


def sample_separation(f: Fields, S: np.ndarray, px, py) -> np.ndarray:
    """Tone offset (stops) for pixels with tone S (stops from mid grey) at proxy positions (px, py)."""
    ny, nx, nb = f.sep.shape
    h, w = f.shape
    ty = (np.asarray(py, np.float32) + 0.5) / h * ny - 0.5
    tx = (np.asarray(px, np.float32) + 0.5) / w * nx - 0.5
    b = (np.clip(S, SEP_LO, SEP_HI) - SEP_LO) * ((nb - 1) / (SEP_HI - SEP_LO))
    return map_coordinates(f.sep, [np.clip(ty, 0, ny - 1), np.clip(tx, 0, nx - 1), b], order=1,
                           mode="nearest", prefilter=False)


def sample_broad(f: Fields, px, py) -> np.ndarray:
    return map_coordinates(f.broad, [np.asarray(py, np.float32), np.asarray(px, np.float32)], order=1,
                           mode="nearest", prefilter=False)


# -------------------------------------------------------------------------------------------------- apply
def apply(rgb: np.ndarray, p: ScaleParams, mid_in: float, noise_sigma_lin=None, eff_scale: float = 1.0,
          long_edge_full: float = REF_LONG_EDGE, fields: Fields | None = None, proxy_xy=None,
          veil_done: bool = False) -> np.ndarray:
    """Detail by scale on scene-referred linear ProPhoto (after exposure). Returns a new array.

    fields + proxy_xy = (px, py): the whole-frame measurements and each pixel's position on the proxy (needed
    when veil, separation or broad is on). Without them the image itself is taken as the whole frame.
    """
    p = p.clamped()
    if not p.active:
        return rgb
    if p.only_local:
        return MC.apply(rgb, p.legacy(), mid_in, noise_sigma_lin, eff_scale, long_edge_full)
    h, w = rgb.shape[:2]
    if p.needs_proxy and fields is None:
        k = min(1.0, PROXY_LONG / max(h, w))
        prox = rgb if k >= 1 else _shrink(rgb, k)
        fields = measure_fields(prox, p, 1.0, mid_in, noise_sigma_lin)
        gx, gy = np.meshgrid((np.arange(w, dtype=np.float32) + 0.5) * (prox.shape[1] / w) - 0.5,
                             (np.arange(h, dtype=np.float32) + 0.5) * (prox.shape[0] / h) - 0.5)
        proxy_xy = (gx, gy)
    out = rgb
    with slog.stage("detail_by_scale", log, mp=round(h * w / 1e6, 2),
                    on=",".join(k for k in ORDER if abs(getattr(p, k)) > 1e-4)):
        if fields is not None and fields.veil is not None and not veil_done:        # 1. veil
            out = apply_veil(out.astype(np.float32, copy=True), fields)
        Y = out @ TONE_W
        floor = np.float32(mid_in * 2.0 ** -14)
        L = np.log2(np.maximum(Y, floor)).astype(np.float32)
        mid_log2 = math.log2(mid_in)
        D = np.zeros_like(L)
        if fields is not None and fields.sep is not None:                           # 2. tonal separation
            D += sample_separation(fields, L - np.float32(mid_log2), *proxy_xy)
        if fields is not None and fields.broad is not None:                         # 3. scale methods
            D += sample_broad(fields, *proxy_xy)
        if p.scales_active:
            res = min(1.0, float(eff_scale))

            resid = list(getattr(noise_sigma_lin, "resid_levels", None) or [1.0])
            shift = math.log2(1.0 / res)                    # image level 0 is this full-resolution level

            def left(level):                                # fraction of the noise the denoiser left at a level
                return float(np.interp(level + shift, np.arange(len(resid) + 1), resid + [1.0]))

            def noise_at(level):
                def noise_log2(gam):
                    y = np.exp2(np.asarray(gam, np.float64))
                    if noise_sigma_lin is None:
                        return np.zeros_like(y)
                    return np.asarray(noise_sigma_lin(y), np.float64) * (res * left(level)) / (y * math.log(2.0))
                return noise_log2
            nl = P.max_levels(L.shape, 4)
            pl = plan(p, eff_scale, long_edge_full, nl + 1)
            tw = lambda s: tone_weight(p, s)
            Ls = L + D if (fields is not None and (fields.sep is not None or fields.broad is not None)) else L
            # Beyond the frame the picture is continued by point reflection (2 * edge - mirror), so a gradient
            # that runs off the frame keeps running. A plain mirror turns it into a crease at the border, which
            # the filter then "brings out" as a dark or bright band along the frame edge.
            on_all = np.nonzero((np.abs(pl["a_tonal"]) > 1e-4) | (pl["w_leg"] > 0) | (np.abs(pl["gf"]) > 1e-4))[0]
            pad = int(min(2 ** (int(on_all[-1]) + 1), min(h, w) - 1)) if on_all.size else 0
            Lp = _extend(Ls, pad)
            Dp = np.zeros_like(Lp)
            if np.any(np.abs(pl["a_tonal"]) > 1e-4):
                Dp += tonal_delta(Lp, pl["a_tonal"], tw, mid_log2, noise_at if noise_sigma_lin is not None else None)
            on = np.nonzero(pl["w_leg"] > 0)[0]
            if on.size:
                Dp += MC.local_laplacian_delta(Lp, p.legacy(), mid_log2, pl["w_leg"], noise_at(int(on[0])))
            if np.any(np.abs(pl["gf"]) > 1e-4):
                Dp += fine_delta(Lp, pl["gf"], tw, mid_log2, noise_at(0))
            D += Dp[pad:pad + h, pad:pad + w] if pad else Dp
        ratio = np.exp2(np.clip(D, -3.0, 3.0)).astype(np.float32)
        ratio[Y <= floor] = 1.0
        out = out * ratio[..., None]
    return out.astype(np.float32)


def _extend(L: np.ndarray, pad: int) -> np.ndarray:
    """L continued ``pad`` pixels beyond every edge by point reflection (the value at the edge is the centre)."""
    return np.pad(L, pad, mode="reflect", reflect_type="odd") if pad > 0 else L


def _shrink(rgb: np.ndarray, k: float) -> np.ndarray:
    """Area-average reduction by an integer factor near 1 / k (the proxy needs no better)."""
    n = max(1, int(round(1.0 / k)))
    h, w = (rgb.shape[0] // n) * n, (rgb.shape[1] // n) * n
    return rgb[:h, :w].reshape(h // n, n, w // n, n, 3).mean(axis=(1, 3)).astype(np.float32)


# -------------------------------------------------------------------------------------------- the meter
def balance_ratio(Y: np.ndarray, eff_scale: float = 1.0, long_edge_full: float = REF_LONG_EDGE) -> float:
    """The balance number of 23.11: medium-scale tonal variation (features about 13 px across on a 6000 px
    frame) over pixel-level contrast (about 3 px), measured on display lightness (cube root of luminance). Heliar 1.35,
    Canon 180 macro 1.21, Sigma 135 Art 1.06 on Bill's same-leaf pictures: higher reads as modelled, lower as
    etched."""
    V = np.cbrt(np.maximum(np.asarray(Y, np.float32), 0.0))
    n = P.max_levels(V.shape, 8)
    lg = _level_lg(n, eff_scale, long_edge_full)
    i_f, i_m = int(np.argmin(np.abs(lg - 1.72))), int(np.argmin(np.abs(lg - 3.7)))
    if i_m <= i_f or i_m >= n:
        return 0.0
    G = P.gaussian(V, i_m + 1)
    lap = lambda i: G[i] - P.up(G[i + 1], G[i].shape)
    fine, med = float(np.sqrt(np.mean(lap(i_f) ** 2))), float(np.sqrt(np.mean(lap(i_m) ** 2)))
    return med / fine if fine > 1e-9 else 0.0
