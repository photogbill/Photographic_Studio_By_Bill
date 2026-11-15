"""Auto HDR on a single raw (BUILD_PLAN.md 23.6, 23.12): one button, then a strength. It never runs by itself.

A raw file usually holds more range than a screen shows: shadows that print as black, highlights that roll off
into white. Auto HDR measures how much range this photo has, decides how many stops must be made room for, and
then opens the shadows and holds the highlights WITHOUT flattening local contrast.

How it works
------------
Everything is decided on a small copy of the whole frame (the proxy of detail/scales.py, about 1024 px), so the
100% view and the export agree and a region render needs no context:

1. **Measure** (the auto part): robust percentiles of the scene's tones in stops from mid grey. The shadows need
   ``-3.5 - p2`` stops of lift for the 2nd percentile to become visible; the highlights need ``p99.5 - 2`` stops
   of pull to sit under the tone curve's shoulder. The strength decides what share of each is given.
2. **Compress** with one of three engines (rule 23.1: each wins on different photos, the user can choose):
     * ``fusion``    exposure fusion (Mertens, Kautz, Van Reeth 2007) of five virtual exposures made from the
                     scene-referred data, as HDR+ does (Hasinoff et al. 2016). Each exposure is weighted by how
                     well exposed it is at each place and the weights are blended across scales.
     * ``laplacian`` local Laplacian tone compression (Paris et al. 2011; fast form of Aubry et al. 2014):
                     differences up to a stop are detail and keep their size, larger ones are compressed.
     * ``retinex``   Multi-Scale Retinex (Jobson, Rahman, Woodell, NASA Langley 1997; the theory is Land's): the
                     lighting, estimated as the blurred picture at three scales, is divided out part of the way.
   Each gives a brightness change in stops per proxy pixel; the last two are scaled so the darkest areas get the
   lift and the brightest the pull that step 1 asked for.
3. **Noise-aware lift** (idea: Retinex with a noise-aware shadow-up function, arXiv 1811.03280): the lift is
   limited where the signal is mostly noise, using the calibrated noise model and what the denoiser leaves.
   Lifting noise only shows noise.
4. **Edge guard**: an engine's change can drift inside an evenly lit area beside a big edge (a halo). So the
   change is re-expressed as one value per level of *lighting* (an edge-keeping average of the tones) and read
   back from the lighting: every evenly lit area gets one change, edge to edge, and tones stay in order.
5. **Edge-aware upsampling** (fast guided filter, He and Sun 2015): on the proxy the change is modelled as
   a * tone + b in every neighbourhood; a and b are smooth, so they can be enlarged to any size and evaluated
   with the full-resolution tones. The change then follows the photo's own edges at full size too.

Colour: the change is a luminance ratio applied to R, G and B in linear light, so lifted shadows keep their
colour (they do not go grey) and held highlights keep their hue.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter

from ... import log as slog
from ..detail import pyramid as P

log = slog.get("engine.hdr")

TONE_W = np.array([0.25, 0.60, 0.15], np.float32)       # as detail/scales.py: blue must not count as black
SHADOW_FLOOR = -3.5         # stops from mid grey: tones below this print close to black
HIGHLIGHT_CEIL = 2.0        # stops from mid grey: tones above this are in the shoulder
MAX_LIFT, MAX_PULL = 4.0, 3.0
MIN_LIFT, MIN_PULL = 0.5, 0.3           # what "some range to make room for" means for a photo that already fits

# name -> (share of the needed lift, share of the needed pull), and the texts beside each choice (rule 23.1)
STRENGTHS = {
    "light": dict(name="Light", lift=0.35, pull=0.8,
                  does="Holds the highlights, opens shadows a little. Looks like a careful single exposure."),
    "medium": dict(name="Medium", lift=0.6, pull=0.9, does="Balanced. The usual choice."),
    "heavy": dict(name="Heavy", lift=1.0, pull=1.0,
                  does="Everything visible: flatter overall, strong local contrast. Can look processed."),
    "highlights": dict(name="Highlights only", lift=0.0, pull=1.0,
                       does="Tames bright areas, leaves the shadows exactly as shot."),
    "lowkey": dict(name="Low-key", lift=0.5, pull=1.0,
                   does="For candle and night work: protects flames and lamps, keeps blacks black, lifts only "
                        "the lower mid-tones."),
    "shadows": dict(name="Shadows only", lift=0.8, pull=0.0,
                    does="Opens shadows, leaves highlights as shot (backlit subjects)."),
}
ENGINES = {
    "fusion": dict(name="Exposure fusion (recommended)",
                   best="Natural results on most photos; cannot make halos by design.",
                   weak="Gentler than the others on very deep shadows.", speed="Fast."),
    "laplacian": dict(name="Local Laplacian",
                      best="Keeps fine and medium detail exactly while the big differences shrink.",
                      weak="Can look flat when pushed hard.", speed="Fast."),
    "retinex": dict(name="Multi-Scale Retinex (NASA)",
                    best="Pulling detail out of very deep shadows.",
                    weak="Can halo at hard edges and grey the lighting out when pushed.", speed="Fast."),
}


@dataclass
class HDRParams:
    strength: str = "medium"
    amount: float = 1.0          # 0..1.5 (the interface shows 0-150%)
    engine: str = "fusion"

    @property
    def active(self) -> bool:
        return self.amount > 1e-4 and self.strength in STRENGTHS


@dataclass
class HDRFields:
    """The brightness change as a * tone + b (stops; tone = log2 of the lightness over mid grey), on the proxy."""
    a: np.ndarray
    b: np.ndarray
    shape: tuple
    report: dict


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def scene_range(S: np.ndarray) -> dict:
    """Robust shadow and highlight percentiles of the proxy's tones (stops from mid grey)."""
    sm = uniform_filter(S, 3)
    lo, hi = np.percentile(sm, [2.0, 99.5])
    return {"lo": float(lo), "hi": float(hi), "range": float(hi - lo),
            "need_lift": float(np.clip(SHADOW_FLOOR - lo, 0.0, MAX_LIFT)),
            "need_pull": float(np.clip(hi - HIGHLIGHT_CEIL, 0.0, MAX_PULL))}


def plan(S: np.ndarray, p: HDRParams) -> dict:
    """How many stops of lift and pull this strength gives this scene."""
    r = scene_range(S)
    st = STRENGTHS[p.strength]
    r["lift"] = float(p.amount * st["lift"] * max(r["need_lift"], MIN_LIFT)) if st["lift"] else 0.0
    r["pull"] = float(p.amount * st["pull"] * max(r["need_pull"], MIN_PULL)) if st["pull"] else 0.0
    r["fits"] = r["need_lift"] <= 0.0 and r["need_pull"] <= 0.0
    return r


# ------------------------------------------------------------------------------------------------ engines
def _fusion(S: np.ndarray, lift: float, pull: float) -> np.ndarray:
    """Exposure fusion of five virtual exposures of S (stops from mid grey). Returns the change in stops."""
    evs = [-pull, -0.5 * pull, 0.0, 0.5 * lift, lift]
    evs = sorted(set(round(e, 4) for e in evs))
    enc = lambda s: 1.0 / (1.0 + np.exp2(-(s + 0.0) * 0.6))            # a soft display encoding: mid grey = 0.5
    n = max(1, P.max_levels(S.shape, 4))
    V = [enc(S + e).astype(np.float32) for e in evs]
    W = [np.exp(-0.5 * ((v - 0.5) / 0.15) ** 2).astype(np.float32) + 1e-6 for v in V]
    tot = sum(W)
    acc = None
    for v, w in zip(V, W):
        gw = P.gaussian(w / tot, n)
        lv = P.laplacian_from_gaussian(P.gaussian(v, n))
        part = [a * b for a, b in zip(lv, gw)]
        acc = part if acc is None else [x + y for x, y in zip(acc, part)]
    fused = np.clip(P.collapse(acc), 1e-4, 1.0 - 1e-4)
    s_out = np.log2(fused / (1.0 - fused)) / 0.6
    return (s_out - S).astype(np.float32)


def _laplacian(S: np.ndarray, beta: float, sigma: float = 1.0, step: float = 0.5) -> np.ndarray:
    """Local Laplacian tone compression: differences beyond ``sigma`` stops shrink toward slope ``beta``,
    smaller ones keep their size. Returns the change in stops (before anchoring)."""
    n = max(1, P.max_levels(S.shape, 4))
    G = P.gaussian(S, n)
    gam = np.arange(math.floor(float(S.min()) / step) - 1, math.ceil(float(S.max()) / step) + 2) * step
    out = [np.zeros_like(G[i]) for i in range(n)]
    for g in gam:
        ws = [np.maximum(1.0 - np.abs(G[i] - np.float32(g)) / np.float32(step), 0.0) for i in range(n)]
        if not any(w.any() for w in ws):
            continue
        d = S - np.float32(g)
        R = S - np.float32(1.0 - beta) * (d - np.float32(sigma) * np.tanh(d / np.float32(sigma)))
        GR = P.gaussian(R, n)
        for i in range(n):
            if ws[i].any():
                out[i] += ws[i] * (GR[i] - P.up(GR[i + 1], GR[i].shape))
    res = G[n]
    res = float(res.mean()) + np.float32(beta) * (res - float(res.mean()))      # the coarsest level shrinks too
    return (P.collapse(out + [res]) - S).astype(np.float32)


def _retinex(S: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Multi-Scale Retinex: minus a share of the lighting estimated at three scales (2%, 8%, 30% of the long
    edge). The classic uses Gaussian surrounds, which is where its halos at hard edges come from; here each
    surround is an edge-keeping (guided) average, our edge guard. Returns the change in stops (before anchoring)."""
    from .light import guided_filter
    long_edge = max(S.shape)
    surround = sum(guided_filter(S, max(2, int(round(f * long_edge))), 0.6 ** 2) for f in (0.02, 0.08, 0.30)) / 3.0
    return (-k * (surround - float(np.median(surround)))).astype(np.float32)


def _anchor(D: np.ndarray, S: np.ndarray, lift: float, pull: float) -> np.ndarray:
    """Scale and shift an engine's change so the darkest 5% get ``lift`` and the brightest 1% get ``-pull``.
    With only one side asked for, the other side is pinned and the change is kept on the asked side only."""
    lo_t, hi_t = np.percentile(S, [5.0, 99.0])
    d_lo = float(D[S <= lo_t].mean())
    d_hi = float(D[S >= hi_t].mean())
    if abs(d_lo - d_hi) < 1e-6:
        return np.zeros_like(D)
    al = (lift + pull) / (d_lo - d_hi)
    out = al * (D - d_hi) - pull
    return out.astype(np.float32)


def _by_lighting(D: np.ndarray, B: np.ndarray, amount: float = 1.0, step: float = 0.25) -> np.ndarray:
    """The edge guard shared by every engine. An engine's change can drift inside an evenly lit area next to a big
    edge (that drift is a halo: exposure fusion and Retinex both do it). So the change is re-expressed as a
    function of the lighting B alone, the average change the engine gave to each level of lighting, and
    evaluated on B, which keeps the photo's edges: every evenly lit area gets one change, edge to edge. The
    function is kept in tone order (lighting + change never decreases)."""
    lo, hi = float(B.min()), float(B.max())
    n = max(2, int(math.ceil((hi - lo) / step)) + 1)
    pos = (B - lo) / step
    i0 = np.clip(pos.astype(np.int32), 0, n - 2)
    fr = (pos - i0).astype(np.float64)
    wsum = np.bincount(i0.ravel(), (1 - fr).ravel(), n) + np.bincount(i0.ravel() + 1, fr.ravel(), n)
    dsum = (np.bincount(i0.ravel(), ((1 - fr) * D).ravel(), n) + np.bincount(i0.ravel() + 1, (fr * D).ravel(), n))
    have = wsum > 1e-9
    nodes = lo + step * np.arange(n)
    F = np.interp(nodes, nodes[have], dsum[have] / wsum[have])
    w = np.where(have, wsum, 0.0)
    if n >= 5:                                              # a smooth curve, populous tones counting most
        ws = uniform_filter(w, 5, mode="nearest")
        ok = ws > 1e-6 * float(w.max())                     # (levels of lighting nobody has keep the interpolation)
        F = np.where(ok, uniform_filter(F * w, 5, mode="nearest") / np.maximum(ws, 1e-30), F)
    T = nodes + amount * F                                  # Amount scales the change
    for q in range(1, n):                                   # tone order, slope at least 0.25
        if T[q] < T[q - 1] + 0.25 * step:
            T[q] = T[q - 1] + 0.25 * step
    return np.interp(B, nodes, T - nodes).astype(np.float32)


# ---------------------------------------------------------------------------------------------- the field
def measure(proxy: np.ndarray, p: HDRParams, gain: float, mid: float, noise_sigma_lin=None) -> HDRFields:
    """Auto HDR's brightness field from the whole-frame proxy (scene-linear, BEFORE exposure; ``gain`` =
    2**exposure; ``mid`` = the post-exposure value rendered as mid grey; noise_sigma_lin(Y) = pixel noise of the
    post-exposure image, with ``resid_levels`` when a denoiser ran)."""
    Y = np.maximum(proxy.astype(np.float32) @ TONE_W * np.float32(gain), np.float32(mid * 2.0 ** -14))
    S = np.log2(Y / np.float32(mid)).astype(np.float32)
    pl = plan(S, p)
    lift, pull = pl["lift"], pl["pull"]
    amt = max(p.amount, 1e-6)
    l1, p1 = lift / amt, pull / amt                         # the strength at 100%; Amount scales the result
    from .light import guided_filter
    B = guided_filter(S, max(2, int(round(0.03 * max(S.shape)))), 0.35 ** 2)       # the lighting: edges kept
    with slog.stage("hdr." + p.engine, log, strength=p.strength, lift=round(lift, 2), pull=round(pull, 2)):
        if lift <= 1e-4 and pull <= 1e-4:
            D = np.zeros_like(S)
        elif p.engine == "laplacian":
            beta = float(np.clip(1.0 - (l1 + p1) / max(pl["range"], 1.0), 0.3, 1.0))
            D = _anchor(_laplacian(S, beta), S, l1, p1)
        elif p.engine == "retinex":
            D = _anchor(_retinex(S), S, l1, p1)
        else:
            D = _fusion(S, l1, p1)
        D = _by_lighting(D, B, amt)                         # edge guard: one change per level of lighting
        # each side only does what it was asked to: no pull = highlights exactly as shot, and the reverse
        D = np.clip(D, -pull if pull > 1e-4 else 0.0, lift if lift > 1e-4 else 0.0)
        if lift <= 1e-4:
            D = np.minimum(D, 0.0) * _smooth((B + 1.0) / 1.5)           # shadows exactly as shot
        if pull <= 1e-4:
            D = np.maximum(D, 0.0) * (1.0 - _smooth((B - 0.0) / 1.5))   # highlights exactly as shot
        if p.strength == "lowkey":                                      # blacks stay black; only lower mid-tones rise
            win = _smooth((B + 6.5) / 2.0) * (1.0 - _smooth((B + 1.5) / 1.5))
            D = np.where(D > 0, D * win, D)
        limited = 0.0
        if noise_sigma_lin is not None and lift > 1e-4:                 # 3. noise-aware lift
            resid = float((getattr(noise_sigma_lin, "resid_levels", None) or [1.0])[0])
            yv = mid * np.exp2(B.astype(np.float64))
            snr = yv / np.maximum(np.asarray(noise_sigma_lin(yv), np.float64) * resid, 1e-12)
            keep = (snr * snr / (snr * snr + 4.0)).astype(np.float32)   # SNR 2 keeps half of the lift
            pos = np.maximum(D, 0.0)
            if float(pos.sum()) > 0:
                limited = float(1.0 - (pos * keep).sum() / pos.sum())
            D = np.where(D > 0, D * keep, D)
    # 4. edge-aware model D ~ a * S + b in every neighbourhood (guided filter with S as the guide)
    r = max(2, int(round(0.012 * max(S.shape))))
    size = 2 * r + 1
    mS, mD = uniform_filter(S, size, mode="reflect"), uniform_filter(D, size, mode="reflect")
    cov = uniform_filter(S * D, size, mode="reflect") - mS * mD
    var = np.maximum(uniform_filter(S * S, size, mode="reflect") - mS * mS, 0.0)
    a = cov / (var + np.float32(0.35 ** 2))
    a = np.clip(a, -0.75, 0.25)             # tones never swap order: the local slope stays at least 0.25
    b = mD - a * mS
    a = uniform_filter(a, size, mode="reflect").astype(np.float32)
    b = uniform_filter(b, size, mode="reflect").astype(np.float32)
    rep = dict(pl, engine=p.engine, strength=p.strength, noise_limited=round(limited, 3),
               lifted=round(max(0.0, float(np.percentile(D, 99.5))), 2),
               pulled=round(max(0.0, float(-np.percentile(D, 0.5))), 2))
    return HDRFields(a=a, b=b, shape=S.shape, report=rep)


def apply(rgb: np.ndarray, f: HDRFields, mid: float, px, py) -> np.ndarray:
    """Apply the field to a scene-linear image (after exposure) whose pixels sit at proxy positions (px, py)."""
    Y = np.maximum(rgb @ TONE_W, np.float32(mid * 2.0 ** -14))
    S = np.log2(Y / np.float32(mid))
    co = [np.asarray(py, np.float32), np.asarray(px, np.float32)]
    a = map_coordinates(f.a, co, order=1, mode="nearest", prefilter=False)
    b = map_coordinates(f.b, co, order=1, mode="nearest", prefilter=False)
    D = np.clip(a * S + b, -MAX_PULL, MAX_LIFT)
    return (rgb * np.exp2(D)[..., None]).astype(np.float32)


def apply_whole(rgb: np.ndarray, p: HDRParams, mid: float, noise_sigma_lin=None):
    """Convenience for tests and tools: the image itself is the whole frame. Returns (image, report)."""
    from ..detail.scales import PROXY_LONG, _shrink
    h, w = rgb.shape[:2]
    k = min(1.0, PROXY_LONG / max(h, w))
    prox = rgb if k >= 1 else _shrink(rgb, k)
    f = measure(prox, p, 1.0, mid, noise_sigma_lin)
    gx, gy = np.meshgrid((np.arange(w, dtype=np.float32) + 0.5) * (prox.shape[1] / w) - 0.5,
                         (np.arange(h, dtype=np.float32) + 0.5) * (prox.shape[0] / h) - 0.5)
    return apply(rgb, f, mid, gx, gy), f.report
