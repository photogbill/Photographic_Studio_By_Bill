"""Stage 16: microcontrast, done properly (BUILD_PLAN.md section 9.3).

Bill's definition: more distinguishable shades inside shadows and midtones, "the variations of shadow",
"more dynamic range within exposure". Not edge sharpening, not halo "clarity". His reference is a soft 1920s
Heliar beating sharp modern lenses 19 times out of 20 on the same leaf: so it is about small tonal differences
at medium spatial scales, not about fine detail.

How it works
------------
A local Laplacian filter (Paris, Hasinoff, Kautz 2011) on scene-referred log2 luminance, evaluated with the
fast algorithm of Aubry, Paris, Hasinoff, Kautz and Durand (ACM TOG 33(5), 2014): the image is remapped
around a set of reference tones, each remapped image gets a Laplacian pyramid, and every output coefficient is
interpolated from the remapped pyramids whose reference tone is closest to the local tone there.

The remapping function is what makes it "shades of shadow" rather than clarity:

    f(d) = d * (1 + A * n(|d|) * e(|d|))         d = pixel tone minus the local reference tone, in stops

  * e(x) = (1 - (x / range)^2)^2 below ``range`` (default 2 stops), 0 above: tonal differences smaller than the
    range are spread apart; bigger ones (real edges) are left exactly as they are, so no halos can form. (1 stop
    was tried first: real photos vary by more than that within 30 px, and the effect was only +6%; 2 stops gives
    +20-27% more medium-scale tonal variation on a real R5 frame, docs/RESULTS.md.)
  * n(x) = x^2 / (x^2 + t^2): a toe at the noise level t of that tone (from the camera noise model), so
    differences that are only noise are not amplified. Deep, noisy shadows get less, never more noise.
  * A depends on the local tone: separate strengths for shadows, midtones and highlights (shadows first).
  * f is monotone for A <= 1.25, so tones never swap order (no gradient reversal).

Only medium spatial scales are touched (default: features about 8-90 px on a 6000 px wide frame); the finest
levels (pixel detail, grain, noise) and the coarsest (overall tone) pass through unchanged. The scale band is
defined in full-resolution pixels, so the fit-to-screen preview and the 100% export match.

Colour: the luminance change is applied as a ratio to R, G and B in linear light, so hue and saturation stay put.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

from ... import log as slog
from . import pyramid as P

log = slog.get("engine.microcontrast")

A_MAX = 1.2                 # f stays monotone up to 1.25
REF_LONG_EDGE = 6000.0      # scale band is defined for a 6000 px long edge and follows the sensor size
EDGE_GUARD = 0.25           # band coefficients above a quarter of the range count as edges
LUMA = np.array([0.2880402, 0.7118741, 0.0000857], np.float32)    # linear ProPhoto Y weights


@dataclass
class MCParams:
    amount: float = 0.0          # 0..1 (UI 0-100 %)
    shadows: float = 1.0         # relative strength by tone
    midtones: float = 0.7
    highlights: float = 0.3
    scale: float = 0.0           # -1 = finer (about 4-45 px) .. +1 = broader (about 16-180 px)
    range_ev: float = 2.0        # tonal differences up to this many stops are "shades"; bigger ones are edges

    @property
    def active(self) -> bool:
        return self.amount > 1e-4 and max(self.shadows, self.midtones, self.highlights) > 1e-4


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def tone_bands(s):
    """Weights (shadows, midtones, highlights) at tone s (stops relative to mid grey). They sum to 1 between the
    black and white tapers; pure black and blown highlights get nothing (there is no tonal detail there)."""
    s = np.asarray(s, np.float64)
    sh = 1.0 - _smooth((s + 2.5) / 2.0)          # 1 below -2.5 EV, 0 above -0.5 EV
    hi = _smooth((s - 0.5) / 1.5)                # 0 below +0.5 EV, 1 above +2 EV
    mid = np.clip(1.0 - sh - hi, 0.0, 1.0)
    taper = _smooth((s + 10.0) / 3.0) * (1.0 - _smooth((s - 3.0) / 1.0))
    return sh * taper, mid * taper, hi * taper


def tone_strength(p: MCParams, s) -> np.ndarray:
    sh, mid, hi = tone_bands(s)
    return p.amount * A_MAX * (p.shadows * sh + p.midtones * mid + p.highlights * hi) / max(
        1.0, max(p.shadows, p.midtones, p.highlights))


def level_weights(p: MCParams, eff_scale: float, long_edge_full: float, n_levels: int) -> np.ndarray:
    """Weight of each pyramid level of the image being processed (rendered at ``eff_scale`` of full size)."""
    centre = 4.5 + 1.5 * float(np.clip(p.scale, -1.0, 1.0))           # log2 of the feature size, ref frame
    norm = REF_LONG_EDGE / max(long_edge_full, 1.0)
    w = np.zeros(n_levels)
    for lvl in range(n_levels):
        size = (2.0 ** lvl) / max(eff_scale, 1e-6) * norm             # feature size in reference pixels
        w[lvl] = math.exp(-0.5 * ((math.log2(size) - centre) / 1.3) ** 2)
    w[w < 0.05] = 0.0
    return w


def remap(img: np.ndarray, gamma: float, amount: float, rng: float, toe: float) -> np.ndarray:
    d = img - np.float32(gamma)
    ad2 = d * d
    e = np.maximum(1.0 - ad2 * np.float32(1.0 / (rng * rng)), 0.0)
    e *= e
    n = ad2 / (ad2 + np.float32(toe * toe + 1e-12))
    return img + np.float32(amount) * d * n * e


def local_laplacian_delta(L: np.ndarray, p: MCParams, mid_log2: float, weights: np.ndarray, noise_log2,
                          step: float | None = None) -> np.ndarray:
    """Change in log2 luminance produced by the tone-dependent fast local Laplacian filter.

    L: log2 luminance (H x W). weights: per pyramid level (0 = finest). noise_log2(gamma) -> noise standard
    deviation in stops at pixel level for tone gamma (vectorised), already scaled for this render's resolution.
    """
    band = np.nonzero(weights)[0]
    if band.size == 0:
        return np.zeros_like(L)
    l0, l1 = int(band[0]), int(band[-1])
    nlev = min(l1 + 1, P.max_levels(L.shape, 4))
    if nlev <= l0:
        return np.zeros_like(L)
    l1 = min(l1, nlev - 1)
    G = P.gaussian(L, nlev)                       # G[0..nlev]
    rng = float(max(p.range_ev, 0.1))
    step = float(step or 0.5 * rng)
    # reference tones on a fixed grid anchored at mid grey, so a region render uses the same tones as the full one
    lo = min(float(G[lv].min()) for lv in range(l0, l1 + 1))
    hi = min(max(float(G[lv].max()) for lv in range(l0, l1 + 1)), mid_log2 + 8.0)
    lo = max(lo, mid_log2 - 16.0)
    k0 = int(math.floor((lo - mid_log2) / step)) - 1
    k1 = int(math.ceil((hi - mid_log2) / step)) + 1
    I0 = G[l0]
    nf = P.white_noise_factors()
    out = [np.zeros_like(G[lv]) for lv in range(l0, l1 + 1)]
    wsum = [np.zeros_like(G[lv]) for lv in range(l0, l1 + 1)]
    gam = mid_log2 + step * np.arange(k0, k1 + 1)
    A = tone_strength(p, gam - mid_log2)
    toe = 1.5 * np.asarray(noise_log2(gam), np.float64) * nf[l0]
    Gc = [np.clip(G[lv], np.float32(gam[0]), np.float32(gam[-1])) for lv in range(l0, l1 + 1)]
    for gi, g in enumerate(gam):
        need = any(np.any(np.abs(Gc[j] - g) < step) for j in range(l1 - l0 + 1))
        if not need:
            continue
        a = float(A[gi]) if toe[gi] < 0.9 * rng else float(A[gi]) * max(0.0, 1.0 - (toe[gi] - 0.9 * rng) / (0.6 * rng))
        R = remap(I0, g, a, rng, float(toe[gi])) if a > 1e-4 else I0
        GR = P.gaussian(R, l1 - l0 + 1)
        for j in range(l1 - l0 + 1):
            lv = l0 + j
            w = np.maximum(1.0 - np.abs(Gc[j] - np.float32(g)) / np.float32(step), 0.0)
            if not w.any():
                continue
            LR = GR[j] - P.up(GR[j + 1], GR[j].shape)
            out[j] += w * LR
            wsum[j] += w
    # Delta pyramid: weight * (filtered - original) on the band levels, nothing elsewhere
    D = None
    for lv in range(l1, -1, -1):
        if D is None:
            D = np.zeros_like(G[lv])
        else:
            D = P.up(D, G[lv].shape)
        if l0 <= lv <= l1 and weights[lv] > 0:
            j = lv - l0
            Lin = G[lv] - P.up(G[lv + 1], G[lv].shape)
            filt = np.where(wsum[j] > 1e-6, out[j] / np.maximum(wsum[j], 1e-6), Lin)
            # Edge guard: where this scale holds a big tonal edge (a band coefficient approaching the range), leave
            # it alone. The local Laplacian filter alone already never changes the edge itself, but it slightly
            # brightens the bright side / darkens the dark side next to strong edges; this removes that too.
            big = maximum_filter(np.abs(Lin), size=3) / np.float32(EDGE_GUARD * rng)
            guard = 1.0 / (1.0 + big ** 4)
            D += np.float32(weights[lv]) * guard * (filt - Lin)
    return D


def apply(rgb: np.ndarray, p: MCParams, mid_in: float, noise_sigma_lin=None, eff_scale: float = 1.0,
          long_edge_full: float = 6000.0) -> np.ndarray:
    """Microcontrast on scene-referred linear ProPhoto (after exposure). Returns a new array.

    mid_in: the scene value the tone curve renders as mid grey (its log2 anchors the tone bands).
    noise_sigma_lin(Y) -> noise standard deviation of luminance at full resolution, same units as rgb.
    eff_scale: resolution of rgb relative to the full-size render (previews are smaller).
    """
    if not p.active:
        return rgb
    Y = rgb @ LUMA if rgb.ndim == 3 else rgb
    floor = np.float32(mid_in * 2.0 ** -14)
    L = np.log2(np.maximum(Y, floor)).astype(np.float32)
    nl = P.max_levels(L.shape, 4)
    w = level_weights(p, eff_scale, long_edge_full, nl + 1)
    mid_log2 = math.log2(mid_in)
    res = min(1.0, float(eff_scale))                                  # smaller renders average noise away

    def noise_log2(gam):
        y = np.exp2(np.asarray(gam, np.float64))
        if noise_sigma_lin is None:
            return np.zeros_like(y)
        return np.asarray(noise_sigma_lin(y), np.float64) * res / (y * math.log(2.0))
    with slog.stage("microcontrast", log, amount=round(p.amount, 3), levels=int(np.count_nonzero(w)),
                    mp=round(L.size / 1e6, 2)):
        D = local_laplacian_delta(L, p, mid_log2, w, noise_log2)
        ratio = np.exp2(np.clip(D, -3.0, 3.0)).astype(np.float32)
        ratio[Y <= floor] = 1.0
        out = rgb * ratio[..., None] if rgb.ndim == 3 else rgb * ratio
    return out.astype(np.float32)


def context(p: MCParams, long_edge_full: float) -> tuple[int, int]:
    """(pad, align): full-resolution pixels of context a region render needs, and the multiple its origin must
    sit on so its pyramid lines up with the full frame's."""
    w = level_weights(p, 1.0, long_edge_full, 14)
    band = np.nonzero(w)[0]
    if band.size == 0:
        return 0, 2
    align = int(2 ** (int(band[-1]) + 1))
    return 2 * align, align
