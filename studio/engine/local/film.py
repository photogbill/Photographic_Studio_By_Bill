"""Film negatives -> positives, for scans and for negatives copied with a camera (plan: old photo restoration).

A negative records light as density: where the scene was bright the film is dark. Density is
D = -log10(T), T the fraction of light the film lets through. The film base itself (and, on colour film, the
orange mask) has a density of its own, the same everywhere, so what carries the picture is the density above
the base,

    d = log10(T_base / T)            per channel,

which is proportional to the scene's log exposure (the film's characteristic curve, straight over most of its
range, Hurter and Driffield 1890). Colour film has three layers with their own speed and contrast, so the three
channels are put on a common footing by their own density range in this frame: the thinnest part of the picture
(the deepest scene shadow) and the densest (the brightest highlight) line up in all three channels, which takes
out the orange mask and the layers' different contrasts in one move. The same idea underlies darktable's
negadoctor (Aurélien Pierre, after Kodak's Cineon printing densities) and the hand method of darkroom colour
printers: neutral shadows and neutral highlights.

The positive is written as scene-linear light for the Studio's own tone curve, which then plays the part of the
printing paper:

    n = (d - lo) / (hi - lo)                    0 at the frame's thinnest picture area, 1 at its densest
    E = WHITE * 2 ** (stops * (n - 1))          the densest area reaches the curve's white

Everything that depends on the whole frame (film base, lo, hi, stops) is measured once and kept in the edit, so a
zoomed region renders exactly like the whole photo and an old sidecar renders the same after an update.
"""
from __future__ import annotations

import math

import numpy as np

from ... import log as slog

log = slog.get("engine.film")

KINDS = ("colour", "bw")
WHITE = 0.88                    # scene-linear value the Studio Standard curve shows as white (0.11 x 2^3)
EPS = 1e-6
GAMMA = {"colour": 0.60, "bw": 0.70}     # typical negative contrast (density per decade of exposure)
P_LO, P_HI = 0.2, 99.6                    # percentiles of the picture's density taken as its range


def active(film: dict | None) -> bool:
    return bool(film) and film.get("kind") in KINDS


def needs_measure(film: dict | None) -> bool:
    return active(film) and not all(k in film for k in ("base", "lo", "hi", "stops"))


def measure(T: np.ndarray, kind: str, base=None) -> dict:
    """Film base and density range from a small copy of the whole negative (linear light let through, any
    scale). ``base``: the film base picked by the user (else the clearest unclipped areas). Returns the
    parameters to keep in the edit."""
    if kind not in KINDS:
        raise ValueError(f"film kind {kind!r}")
    T = np.asarray(T, np.float32)
    h, w = T.shape[:2]
    top = float(np.percentile(T.max(axis=2), 99.9))
    ok = T.max(axis=2) < 0.98 * max(top, EPS)          # clipped light (an empty holder, a gap) is not film
    if base is None:
        # the film base is the clearest film: the rebate between frames or, cropped tight, the deepest shadows
        # (base plus fog). The brightest few hundredths of the unclipped pixels, per channel.
        flat = T[ok] if ok.sum() > 100 else T.reshape(-1, 3)
        base = [float(np.percentile(flat[:, c], 99.7)) for c in range(3)]
    base = [max(float(b), EPS) for b in base]
    # the picture's range from the middle 90% (holder edges and rebate out of it)
    y0, y1, x0, x1 = int(0.05 * h), int(0.95 * h), int(0.05 * w), int(0.95 * w)
    mid = T[y0:y1, x0:x1].reshape(-1, 3)
    okm = ok[y0:y1, x0:x1].reshape(-1)
    if okm.sum() > 100:
        mid = mid[okm]
    d = np.log10(np.asarray(base, np.float32)[None, :] / np.maximum(mid, EPS))
    lo = [float(np.percentile(d[:, c], P_LO)) for c in range(3)]
    hi = [float(np.percentile(d[:, c], P_HI)) for c in range(3)]
    hi = [max(h_, l_ + 0.05) for l_, h_ in zip(lo, hi)]
    rng = float(np.mean([h_ - l_ for l_, h_ in zip(lo, hi)]))
    stops = float(np.clip(rng / GAMMA[kind] / math.log10(2.0), 4.0, 12.0))
    out = {"kind": kind, "base": [round(b, 7) for b in base], "lo": [round(v, 5) for v in lo],
           "hi": [round(v, 5) for v in hi], "stops": round(stops, 3)}
    log.info("film %s: base %s, density %s..%s, %.1f stops", kind, out["base"], out["lo"], out["hi"], stops)
    return out


def positive(T: np.ndarray, film: dict, scene: bool = True) -> np.ndarray:
    """Linear light through the negative (h, w, 3) -> scene-linear positive for the Studio's tone curve."""
    base = np.asarray(film["base"], np.float32)
    lo = np.asarray(film["lo"], np.float32)
    hi = np.asarray(film["hi"], np.float32)
    stops = float(film.get("stops", 7.0)) * float(film.get("range", 1.0))
    d = np.log10(base / np.maximum(np.asarray(T, np.float32), EPS))
    n = (d - lo) / np.maximum(hi - lo, 0.05)
    if film.get("kind") == "bw":
        n = np.repeat(n.mean(axis=2, keepdims=True), 3, axis=2)
    E = np.float32(WHITE) * np.exp2(np.float32(stops) * (n - 1.0))
    return np.ascontiguousarray(E, np.float32)


def negative_from(E: np.ndarray, kind: str = "colour", base=(0.72, 0.42, 0.24), gamma=(0.55, 0.60, 0.65),
                  range_stops: float = 9.0) -> np.ndarray:
    """A synthetic negative of a scene-linear picture (for tests): orange base, per-layer contrast."""
    E = np.maximum(np.asarray(E, np.float32), 1e-6)
    ref = float(np.percentile(E, 99.5))
    logE = np.log10(E / ref)
    logE = np.maximum(logE, -range_stops * math.log10(2.0))
    if kind == "bw":
        logE = np.repeat(logE.mean(axis=2, keepdims=True), 3, axis=2)
        base, gamma = (0.80, 0.78, 0.76), (0.70, 0.70, 0.70)
    d = (logE + range_stops * math.log10(2.0)) * np.asarray(gamma, np.float32)
    return (np.asarray(base, np.float32) * np.power(10.0, -d)).astype(np.float32)
