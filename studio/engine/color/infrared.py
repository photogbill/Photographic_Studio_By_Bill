"""Infrared and full-spectrum raws (BUILD_PLAN_PANO_IR.md section 1): the colour science of the Infrared tab.

Infrared and full-spectrum raws break normal colour: the camera's white balance cannot go far enough and its colour
matrix was made for visible light. Everything here acts on the raw itself, through ``edit.infrared`` and
``edit.white_balance`` (multipliers), so Develop continues on the same raw with every tool afterwards.

    edit.infrared = {
      "on": True, "filter": "850", "guess": {"key": "850", "confidence": 0.93, "rg": .., "bg": .., ...},
      "profile": "identity",          # identity (camera RGB taken as it is) | visible (the camera matrix)
      "mixer": [1,0,0, 0,1,0, 0,0,1], # rows = output R, G, B from white-balanced camera R, G, B (linear light)
      "preset": "none", "strength": 1.0,
      "mono_raw": False,              # 850 nm and deeper: develop without demosaicing (mono_mosaic)
      "bw": {...},                    # black and white from the channels (default_bw), kept apart from the colour side
      "palette": None,                # {"name", "amount"}: colour by brightness for a black and white IR photo (added)
      "hotspot": None,                # {"model": {...}, "amount": 0..1} (lens/hotspot.py)
      "wb_source": "foliage",         # foliage | click_foliage | neutral | measured:<name> | as_shot | hand
    }

The matrix step (pipeline "camera_matrix" stage, ``apply_camera``):

* ``identity``: white-balanced camera RGB is taken as linear light with sRGB primaries and converted to ProPhoto with
  one fixed matrix ``P = spaces.conversion_matrix(SRGB, PROPHOTO)`` (Bradford D65 -> D50), so a channel swap looks the
  way it does in Photoshop on a linear file and white stays white (P maps (1, 1, 1) to (1, 1, 1)). No visible-light
  colour science is pretended for light the camera matrix was never made for.
* ``visible``: the camera matrix first (``S = P^-1 @ camera_to_working``: camera -> linear sRGB-like), then the mixer.
  For a full-spectrum camera with an IR-cut filter on the lens, which behaves like an ordinary camera.
* Colour side: ``M = P @ Mix @ S``. Black and white side (``bw["on"]``): per channel exposure, contrast, curve and
  separation, then one mono mix, R = G = B.

Why black and white for 850 nm: on Bill's converted 450D at 850 nm the three CFA dyes pass the same light and the raw
is monochrome to within 1.5% (measured 2026-11-16, plan 0.3); what is left is a level-dependent sensor gain difference,
not colour. ``mono_mosaic`` removes it and develops every photosite as a luminance sample: full resolution, no colour
noise, no demosaic pattern. ``colour_meter`` checks any raw for usable colour against the two greens of each quad,
which see identical light.

The filter families other than 850 nm come from a synthetic model of a converted camera (Gaussian dyes that all turn
transparent past 800 nm, long-pass filters, foliage / sky / skin / grey reflectances; it lives in
tests/test_infrared.py) until real 590 / 665 / 720 nm frames arrive, so the guess is always shown as estimated (amber).
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter

from ... import log as slog

log = slog.get("engine.infrared")

VERSION = 1

# -------------------------------------------------------------------------------------------------- filters
FILTERS = {
    "full": {"label": "Full spectrum (no filter)",
             "best": "A converted camera with no filter: visible and infrared together; with an IR-cut filter on the "
                     "lens it behaves like a normal camera.",
             "weak": "The raw is pink-red; white balance needs a neutral in the frame, and colours are never quite "
                     "those of a normal camera."},
    "550": {"label": "530-550 nm",
            "best": "The strongest false colour of any IR filter: swap red and blue, white balance on foliage.",
            "weak": "Colour can look garish; sky and skin take the most care."},
    "590": {"label": "590 nm",
            "best": "Classic false colour: gold foliage and blue sky after the swap.",
            "weak": "Blue needs a large gain, so the blue channel is noisy: Clean helps."},
    "665": {"label": "650-665 nm",
            "best": "Less colour than 590: cream foliage, soft blue sky after the swap.",
            "weak": "Colour is subtle; low saturation looks best."},
    "720": {"label": "720 nm",
            "best": "Mostly brightness with a faint colour: lovely in black and white or with a gentle swap.",
            "weak": "Little colour to work with: strong swaps turn muddy."},
    "850": {"label": "850 nm and deeper",
            "best": "Black and white: the raw is monochrome (all three dyes pass the same light), so it can be "
                    "developed without demosaicing at full resolution.",
            "weak": "No colour is recorded: any colour you see is added afterwards."},
    "unknown": {"label": "Not sure",
                "best": "Starts from the camera's own colours with infrared white balance; pick a filter when you know.",
                "weak": "Nothing is tuned to your filter."},
}
FILTER_ORDER = ("full", "550", "590", "665", "720", "850", "unknown")

# Family signatures (raw R/G, raw B/G, spread (std) of log R/G, of log B/G over the exposed frame) of a daylight scene
# (sky above, foliage below, a skin patch and a grey card) through each filter, from the synthetic converted-camera
# model in tests/test_infrared.py (which checks these numbers); the model's blue dye opens to infrared later than the
# green (about 830 nm against 790 nm), which matches both Bill's 850 nm frames (B/G 0.99) and his 2010 colour-infrared
# frames (B/G 0.63-0.73). 850 nm was measured on Bill's 450D (plan 0.3).
FAMILY = {
    "full": (1.544, 0.734, 0.129, 0.144),
    "550": (1.700, 0.723, 0.102, 0.240),
    "590": (1.744, 0.767, 0.198, 0.097),
    "665": (1.612, 0.783, 0.144, 0.036),
    "720": (1.410, 0.783, 0.068, 0.035),
    "850": (1.000, 1.000, 0.015, 0.016),
}
MONO_TOL = 0.05              # 850 rule: R/G and B/G within 5% of 1 ...
MONO_SPREAD = 0.03           # ... and the chroma spread under 0.03 (the 850 frames measured 0.015)
METER_LIMIT = 3.0            # colour meter: under this many times the two-green control = no usable colour
                             # (Bill's 850 nm frames measure 0.5-2.0; his 2010 colour-IR frames 3.8-6.4)

# -------------------------------------------------------------------------------------------------- presets
PRESETS = ("none", "swap_rb", "blue_sky_590", "goldie", "aerochrome_like", "mono_best", "mono_red", "candle_glow")
PRESET_INFO = {
    "none": {"label": "Camera channels",
             "best": "The raw's own channels after infrared white balance: the honest starting point.",
             "weak": "Most filters look brown or blue-grey this way."},
    "swap_rb": {"label": "Swap red and blue",
                "best": "The standard infrared swap: blue skies from 550-720 nm filters.",
                "weak": "Foliage can turn cyan-blue if white balance is not set on foliage first."},
    "blue_sky_590": {"label": "Blue sky (590)",
                     "best": "590 nm: the swap plus a little green into blue for a deeper sky.",
                     "weak": "Overdoes the sky with 720 nm and deeper."},
    "goldie": {"label": "Goldie",
               "best": "Gold foliage, blue sky: the swap with blue's saturation reduced.",
               "weak": "Needs a filter that still records colour (550-665 nm)."},
    "aerochrome_like": {"label": "Colour infrared film look",
                        "best": "Infrared to red, red to green, green to blue: red foliage like colour-infrared film. "
                                "Best on full spectrum or 590 nm with foliage in the frame.",
                        "weak": "A look, not a measurement; with 720 nm and deeper there is too little colour."},
    "mono_best": {"label": "Black and white: best channel",
                  "best": "The channel with the most detail for its noise, measured on this photo.",
                  "weak": "Ignores the other channels' tonal differences."},
    "mono_red": {"label": "Black and white: red channel",
                 "best": "The red channel alone: the most light with 590-720 nm filters, softer infrared glow.",
                 "weak": "Red also records visible red, so skies stay lighter than in the blue channel; a quarter "
                         "of the photosites: a little noisier than an even mix."},
    "candle_glow": {"label": "Candle glow (added colour)",
                    "best": "Black and white with warm highlights, for candlelit infrared portraits.",
                    "weak": "The warmth is added colour: an 850 nm raw records none (shown amber)."},
}
# preset -> (mixer rows, saturation of the result in camera terms, black and white?)
_SWAP = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
_PRESET_MIX = {
    "none": (np.eye(3), 1.0, False),
    "swap_rb": (np.array(_SWAP, float), 1.0, False),
    "blue_sky_590": (np.array([[0, 0, 1], [0, 1, 0], [0.85, 0.15, 0]], float), 1.0, False),
    "goldie": (np.array([[0.1, 0, 0.9], [0, 1, 0], [0.6, 0.2, 0.2]], float), 0.85, False),
    "aerochrome_like": (np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float), 1.15, False),
    "mono_best": (np.eye(3), 1.0, True),
    "mono_red": (np.eye(3), 1.0, True),
    "candle_glow": (np.eye(3), 1.0, True),
}
CANDLE_GRADING = {"highlights": [65.0, 35.0, 0.0], "midtones": [55.0, 12.0, 0.0], "balance": -10.0,
                  "blending": 50.0}

# black-and-white presets (weights for R, G, B after channel tone)
BW_PRESETS = ("even", "dark_sky", "soft_skin", "separation")
BW_PRESET_INFO = {
    "even": {"label": "Even", "best": "A plain average of the three channels: the least noise.",
             "weak": "Uses none of the channels' differences."},
    "dark_sky": {"label": "Dark sky, bright foliage",
                 "best": "590-720 nm: the blue photosites see almost only infrared (their dye blocks visible red), "
                         "so blue up and red down: skies go dark, foliage glows.",
                 "weak": "Raises noise; at 850 nm the channels are the same and nothing changes."},
    "soft_skin": {"label": "Soft skin",
                  "best": "Weights toward the channel with the least texture: smooth skin in portraits.",
                  "weak": "Can soften fine detail elsewhere too."},
    "separation": {"label": "Most separation",
                   "best": "Weights solved for the most local contrast for their noise (estimated, amber).",
                   "weak": "A measurement on this photo: can pick a noisier mix than you would like."},
}

# -------------------------------------------------------------------------------------------------- palettes
# colour by brightness (sRGB stops); for black and white IR photos, always labelled as added colour
PALETTES = {
    "ironbow": [(0.0, (0.0, 0.0, 0.05)), (0.25, (0.33, 0.0, 0.55)), (0.5, (0.85, 0.15, 0.25)),
                (0.75, (1.0, 0.6, 0.0)), (1.0, (1.0, 1.0, 0.85))],
    "inferno": [(0.0, (0.0, 0.0, 0.02)), (0.25, (0.34, 0.06, 0.43)), (0.5, (0.73, 0.21, 0.33)),
                (0.75, (0.98, 0.55, 0.04)), (1.0, (0.99, 1.0, 0.64))],
    "platinum": [(0.0, (0.06, 0.04, 0.03)), (0.5, (0.52, 0.45, 0.38)), (1.0, (0.98, 0.96, 0.92))],
    "cyanotype": [(0.0, (0.0, 0.07, 0.2)), (0.5, (0.18, 0.42, 0.66)), (1.0, (0.92, 0.96, 1.0))],
    "selenium": [(0.0, (0.05, 0.02, 0.04)), (0.5, (0.45, 0.4, 0.42)), (1.0, (0.97, 0.96, 0.95))],
}
PALETTE_INFO = {
    "ironbow": "A thermal-camera look (added colour): dark purples through red and orange to white.",
    "inferno": "A scientific heat map (added colour), even steps of brightness.",
    "platinum": "A warm platinum / palladium print tone (added colour).",
    "cyanotype": "The blue of a cyanotype print (added colour).",
    "selenium": "A cool selenium-toned print (added colour).",
}


# ================================================================================================== settings
def default_bw() -> dict:
    """Black and white from the channels (plan 1.5c): per-channel tone, then one mono mix. Defaults change nothing
    but the conversion to black and white."""
    return {"on": False, "ev": [0.0, 0.0, 0.0], "contrast": [0.0, 0.0, 0.0], "sep": [0.0, 0.0, 0.0],
            "curve": [None, None, None], "mix": [1 / 3, 1 / 3, 1 / 3], "keep": True, "preset": "even", "recipe": ""}


def default_ir(filter_key: str = "unknown", guess: dict | None = None) -> dict:
    """A fresh infrared edit with the treatment the filter suggests (plan 0.4: 850 opens in black and white)."""
    k = filter_key if filter_key in FILTERS else "unknown"
    preset = {"full": "none", "550": "swap_rb", "590": "blue_sky_590", "665": "goldie", "720": "swap_rb",
              "850": "mono_best", "unknown": "none"}[k]
    ir = {"on": True, "filter": k, "guess": guess, "profile": "identity", "mixer": np.eye(3).ravel().tolist(),
          "preset": preset, "strength": 1.0, "mono_raw": k == "850", "bw": default_bw(), "palette": None,
          "hotspot": None, "wb_source": "foliage", "v": VERSION}
    if preset not in ("mono_best", "mono_red", "candle_glow"):
        ir["mixer"] = mixer(preset).ravel().tolist()
    else:
        ir["bw"]["on"] = True
        ir["bw"]["preset"] = "even"
    return ir


def is_on(ir) -> bool:
    return bool(ir) and bool(ir.get("on"))


def normalise(ir: dict | None) -> dict | None:
    """Fill keys missing from an older or hand-written infrared edit (never changes values that are there)."""
    if not ir:
        return ir
    base = default_ir(ir.get("filter", "unknown"))
    out = dict(base)
    out.update({k: v for k, v in ir.items() if v is not None or k in ("guess", "palette", "hotspot")})
    bw = default_bw()
    bw.update(ir.get("bw") or {})
    out["bw"] = bw
    return out


# ================================================================================================== matrices
def _P() -> np.ndarray:
    """Linear 'camera taken as sRGB primaries' -> linear ProPhoto (D50, Bradford). Maps (1, 1, 1) to (1, 1, 1)."""
    from .spaces import PROPHOTO, SRGB, conversion_matrix
    return conversion_matrix(SRGB, PROPHOTO)


def _saturation(s: float) -> np.ndarray:
    """3x3 that scales colour around the channel average (s = 1: unchanged, 0: grey)."""
    avg = np.full((3, 3), 1.0 / 3.0)
    return (1.0 - s) * avg + s * np.eye(3)


def mixer(preset: str, strength: float = 1.0) -> np.ndarray:
    """The 3x3 mixer of a colour preset at a strength (0 = identity, 1 = the preset)."""
    m, sat, _ = _PRESET_MIX.get(preset, _PRESET_MIX["none"])
    M = _saturation(sat) @ np.asarray(m, float)
    s = float(np.clip(strength, 0.0, 1.5))
    return (1.0 - s) * np.eye(3) + s * M


def row_sums(mix) -> list:
    """Row sums of a mixer (a row that does not sum to 1 changes that channel's brightness)."""
    return [round(float(v), 3) for v in np.asarray(mix, float).reshape(3, 3).sum(axis=1)]


def _working_from_camera(ir: dict, raw) -> np.ndarray:
    """S: white-balanced camera RGB -> the linear RGB the mixer works on."""
    if ir.get("profile") == "visible" and raw is not None and getattr(raw, "cam_xyz", None) is not None:
        from .camera import camera_to_working
        try:
            return np.linalg.inv(_P()) @ camera_to_working(raw.cam_xyz)
        except (ValueError, np.linalg.LinAlgError):
            log.warning("infrared: camera matrix unusable; taking camera channels as they are")
    return np.eye(3)


def mono_weights(bw: dict) -> np.ndarray:
    w = np.asarray(bw.get("mix", [1 / 3] * 3), float)[:3]
    if bw.get("keep", True):
        s = float(w.sum())
        if abs(s) > 1e-6:
            w = w / s
    return w


def has_channel_tone(bw: dict) -> bool:
    if not bw or not bw.get("on"):
        return False
    return (any(abs(float(v)) > 1e-6 for v in bw.get("ev", [0, 0, 0]))
            or any(abs(float(v)) > 1e-6 for v in bw.get("contrast", [0, 0, 0]))
            or any(abs(float(v)) > 1e-6 for v in bw.get("sep", [0, 0, 0]))
            or any(c for c in (bw.get("curve") or [None] * 3)))


def camera_to_working(ir: dict, raw) -> np.ndarray:
    """The whole linear part of the infrared matrix step: white-balanced camera RGB -> linear ProPhoto. With black
    and white on (and no per-channel tone) R = G = B = the mono mix."""
    S = _working_from_camera(ir, raw)
    bw = ir.get("bw") or {}
    if bw.get("on"):
        w = mono_weights(bw)
        return np.ones((3, 1)) @ (w[None, :] @ S)
    mix = np.asarray(ir.get("mixer") or np.eye(3).ravel(), float).reshape(3, 3)
    return _P() @ mix @ S


# -------------------------------------------------------------------------------------------- channel tone
_PIVOT = 0.18 ** (1 / 2.2)


def _contrast_curve(e: np.ndarray, t: float) -> np.ndarray:
    """Contrast in the 1/2.2 encoding around mid grey: 0 -> 0, pivot -> pivot, 1 -> 1, slope 2^t at the pivot."""
    g = 2.0 ** float(np.clip(t, -1.0, 1.0))
    p = _PIVOT
    lo = p * np.power(np.clip(e / p, 0.0, None), g)
    hi = 1.0 - (1.0 - p) * np.power(np.clip((1.0 - e) / (1.0 - p), 0.0, None), g)
    return np.where(e < p, lo, hi)


def _curve_fn(points):
    from scipy.interpolate import PchipInterpolator
    pts = sorted((float(x), float(y)) for x, y in points)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if xs[0] > 0:
        xs, ys = np.r_[0.0, xs], np.r_[0.0, ys]
    if xs[-1] < 1:
        xs, ys = np.r_[xs, 1.0], np.r_[ys, 1.0]
    keep = np.r_[True, np.diff(xs) > 1e-6]
    return PchipInterpolator(xs[keep], ys[keep], extrapolate=True)


def channel_tone(rgb: np.ndarray, bw: dict) -> np.ndarray:
    """Per raw channel (after white balance, before the mix): exposure (stops), contrast (-100..100), a small curve
    (points in the 1/2.2 encoding) and separation (-100..100: how much of what this channel sees differently from
    the other two is kept; -100 = none, +100 = twice). Returns a new float32 array."""
    out = np.empty(rgb.shape, np.float32)
    ev = bw.get("ev", [0, 0, 0])
    ct = bw.get("contrast", [0, 0, 0])
    cv = bw.get("curve") or [None] * 3
    for c in range(3):
        x = rgb[..., c].astype(np.float32) * np.float32(2.0 ** float(ev[c]))
        if abs(float(ct[c])) > 1e-6 or cv[c]:
            pos = np.maximum(x, 0.0)
            inside = pos <= 1.0
            e = np.power(np.minimum(pos, 1.0), 1 / 2.2)
            if abs(float(ct[c])) > 1e-6:
                e = _contrast_curve(e, float(ct[c]) / 100.0)
            if cv[c]:
                e = np.clip(_curve_fn(cv[c])(e), 0.0, 1.0)
            x = np.where(inside, np.power(np.clip(e, 0.0, 1.0), 2.2), x).astype(np.float32)
        out[..., c] = x
    sep = [float(v) / 100.0 for v in bw.get("sep", [0, 0, 0])]
    if any(abs(v) > 1e-6 for v in sep):
        avg = out.mean(axis=2)
        for c in range(3):
            if abs(sep[c]) > 1e-6:
                out[..., c] = avg + np.float32(1.0 + sep[c]) * (out[..., c] - avg)
    return out


def apply_camera(rgb: np.ndarray, ir: dict, raw, strips=None) -> np.ndarray:
    """The infrared matrix step on a white-balanced camera RGB array (in place where it can be)."""
    from .spaces import apply_matrix
    bw = ir.get("bw") or {}
    run = strips or (lambda img, fn: fn(img))
    if has_channel_tone(bw):
        S = _working_from_camera(ir, raw).astype(np.float32)
        w = mono_weights(bw).astype(np.float32)

        def job(s):
            if not np.allclose(S, np.eye(3)):
                apply_matrix(s, S, out=s)
            t = channel_tone(s, bw)
            m = t @ w
            s[...] = m[..., None]
        run(rgb, job)
        return rgb
    M = camera_to_working(ir, raw).astype(np.float32)
    run(rgb, lambda s: apply_matrix(s, M, out=s))
    return rgb


def mono_of(cam_rgb: np.ndarray, bw: dict) -> np.ndarray:
    """The black and white result (one channel) of white-balanced camera RGB under these channel settings."""
    t = channel_tone(cam_rgb, bw) if has_channel_tone(bw) else cam_rgb.astype(np.float32)
    return t @ mono_weights(bw).astype(np.float32)


# ================================================================================================== measuring
def _planes(mos: np.ndarray, pat) -> dict:
    """The four CFA planes of a mosaic by colour: {'R', 'G1', 'G2', 'B'} (half size each)."""
    p = np.asarray(pat)
    h, w = mos.shape[0] // 2 * 2, mos.shape[1] // 2 * 2
    out, greens = {}, []
    for i in (0, 1):
        for j in (0, 1):
            c = int(p[i, j])
            q = mos[i:h:2, j:w:2]
            if c == 1 or c == 3:
                greens.append(q)
            elif c == 0:
                out["R"] = q
            else:
                out["B"] = q
    out["G1"], out["G2"] = greens[0], greens[1]
    return out


def _bin(x: np.ndarray, k: int) -> np.ndarray:
    h, w = x.shape[0] // k * k, x.shape[1] // k * k
    return x[:h, :w].reshape(h // k, k, w // k, k, *x.shape[2:]).mean(axis=(1, 3))


def channel_stats(cam: np.ndarray) -> dict:
    """Raw R/G, B/G, the spread of their logs and the log-correlation, from half-size raw camera RGB (no white
    balance), in 8x8 bins. The brighter half of the frame below clipping gives ``rg``, ``bg`` and their robust spread
    (plan 0.3's measurement, for the 850 rule); every exposed bin (above the darkest fifth) gives ``rg_all``,
    ``bg_all`` and the plain spread across materials (``std_rg``, ``std_bg``), which tell the other filters apart."""
    b = _bin(np.asarray(cam, np.float32), 8)
    R, G, B = b[..., 0], b[..., 1], b[..., 2]
    ok = (b.max(axis=2) < 0.9) & (b.min(axis=2) > 2e-4)
    if ok.sum() < 20:
        ok = b.max(axis=2) < 0.98
    if ok.sum() < 4:
        return {"rg": 1.0, "bg": 1.0, "spread_rg": 0.0, "spread_bg": 0.0, "corr": 0.0, "n": 0,
                "rg_all": 1.0, "bg_all": 1.0, "std_rg": 0.0, "std_bg": 0.0}
    lum = G[ok]
    bright = ok.copy()
    bright[ok] = lum >= np.median(lum)
    lr = np.log(R[bright] / G[bright])
    lb = np.log(B[bright] / G[bright])
    mr, mb = float(np.median(lr)), float(np.median(lb))
    sr = float(1.4826 * np.median(np.abs(lr - mr)))
    sb = float(1.4826 * np.median(np.abs(lb - mb)))
    lg = np.log(G[bright])
    corr = 0.0
    if lr.size > 3 and np.std(lg) > 1e-9:
        cr = np.corrcoef(np.log(R[bright]), lg)[0, 1]
        cb = np.corrcoef(np.log(B[bright]), lg)[0, 1]
        corr = float(np.nanmin([cr, cb]))
    wide = ok.copy()
    wide[ok] = lum > np.percentile(lum, 20)
    la = np.log(R[wide] / G[wide])
    lba = np.log(B[wide] / G[wide])
    return {"rg": float(math.exp(mr)), "bg": float(math.exp(mb)), "spread_rg": sr, "spread_bg": sb, "corr": corr,
            "n": int(bright.sum()), "rg_all": float(math.exp(np.median(la))), "bg_all": float(math.exp(np.median(lba))),
            "std_rg": float(np.std(la)), "std_bg": float(np.std(lba))}


def guess_from_stats(st: dict, meter: float | None = None) -> dict:
    """The filter a raw was most likely shot through, from channel_stats (and the colour meter as a guard)."""
    rg, bg = st["rg"], st["bg"]
    spread = max(st["spread_rg"], st["spread_bg"])
    out = dict(st)
    if abs(rg - 1) < MONO_TOL and abs(bg - 1) < MONO_TOL and spread < MONO_SPREAD:
        conf = 0.95 - 4.0 * max(abs(rg - 1), abs(bg - 1)) - 5.0 * spread
        if meter is not None and meter >= METER_LIMIT:            # a 720 frame that happens to look grey: caught
            out.update(key="720", confidence=0.5, caught=True)
        else:
            out.update(key="850", confidence=round(float(np.clip(conf, 0.5, 0.95)), 2))
        out["sentence"] = sentence(out)
        return out
    if meter is not None and meter < METER_LIMIT and abs(rg - 1) < 0.15 and abs(bg - 1) < 0.15:
        # the channel ratios drift across the frame (mixed light, flare) but the meter finds no colour
        out.update(key="850", confidence=0.6, mixed=True)
        out["sentence"] = sentence(out)
        return out
    best, dbest, d2 = "unknown", 1e9, 1e9
    arg, abg = st.get("rg_all", rg), st.get("bg_all", bg)
    srg, sbg = st.get("std_rg", st["spread_rg"]), st.get("std_bg", st["spread_bg"])
    for k, (frg, fbg, fsr, fsb) in FAMILY.items():
        if k == "850":
            continue
        d = math.hypot(math.log(arg / frg) / 0.06, math.log(abg / fbg) / 0.04) \
            + math.hypot((srg - fsr) / 0.04, (sbg - fsb) / 0.04)
        if d < dbest:
            best, dbest, d2 = k, d, dbest
        elif d < d2:
            d2 = d
    if dbest > 8.0:
        out.update(key="unknown", confidence=0.2)
    else:
        margin = (d2 - dbest) / max(d2, 1e-6)
        out.update(key=best, confidence=round(float(np.clip(0.35 + 0.5 * margin - 0.04 * dbest, 0.2, 0.85)), 2))
    out["sentence"] = sentence(out)
    return out


def sentence(g: dict) -> str:
    lab = FILTERS.get(g.get("key", "unknown"), FILTERS["unknown"])["label"]
    nums = f"raw R/G {g.get('rg', 0):.2f}, B/G {g.get('bg', 0):.2f}"
    if g.get("mixed"):
        return f"{nums}: the channels drift across the frame, but the colour meter finds no colour: 850 nm or deeper."
    if g.get("caught"):
        return f"{nums}: looks grey, but the colour meter finds colour: probably 720 nm."
    if g.get("key") == "unknown":
        return f"{nums}: does not match any filter family. Pick the filter you used."
    return f"{nums}: looks like {lab}."


def guess_filter(image, meter: float | None = None) -> dict:
    """{"key", "confidence", "rg", "bg", "spread_rg", "spread_bg", "corr", "sentence"} for an opened photo."""
    from .autowb import camera_rgb_half
    with slog.stage("infrared.guess", log):
        st = channel_stats(camera_rgb_half(image))
    g = guess_from_stats(st, meter)
    log.info("infrared guess: %s (%.2f) R/G %.3f B/G %.3f spread %.3f/%.3f", g["key"], g["confidence"], g["rg"],
             g["bg"], g["spread_rg"], g["spread_bg"])
    return g


# ------------------------------------------------------------------------------------------- colour meter
def _nconv(x: np.ndarray, ok: np.ndarray, sigma: float) -> np.ndarray:
    w = gaussian_filter(ok.astype(np.float32), sigma)
    return gaussian_filter(np.where(ok, x, 0).astype(np.float32), sigma) / np.maximum(w, 1e-6)


def meter_from_planes(P: dict, sigma: float = 6.0) -> dict:
    """"Is there any colour in this raw?" (plan 1.5b). The one dimension of colour a monochrome raw cannot have,
    h = log G - (log R + log B) / 2 (G = mean of the two greens; both centred on the quad, so first-order edge terms
    cancel), against the two greens of each quad, which see identical light: c = (log G1 - log G2) scaled to the
    same noise. The brightness trend of h is removed (a level-dependent gain is not colour), strong edges are left
    out, both are smoothed alike. Returns {"ratio", "colour": bool, "text"}: under METER_LIMIT, no usable colour."""
    R, G1, G2, B = (np.asarray(P[k], np.float32) for k in ("R", "G1", "G2", "B"))
    G = 0.5 * (G1 + G2)
    floor = max(float(np.percentile(G, 5)), 1e-4)
    ok = (np.minimum(np.minimum(R, B), np.minimum(G1, G2)) > floor * 0.5) & \
         (np.maximum(np.maximum(R, B), np.maximum(G1, G2)) < 0.9)
    if ok.mean() < 0.02:
        return {"ratio": float("nan"), "colour": False, "text": "Too dark or too bright to measure colour."}
    eps = np.float32(1e-6)
    lG = np.log(np.maximum(G, eps))
    h = lG - 0.5 * (np.log(np.maximum(R, eps)) + np.log(np.maximum(B, eps)))
    # the control: each G1 against the mean of its four diagonal G2 neighbours (centred on it, so like h only
    # second-order edge terms are left), scaled to the noise h would have
    G2p = np.pad(G2, ((1, 0), (0, 1)), mode="edge")
    G2d = 0.25 * (G2p[1:, :-1] + G2p[1:, 1:] + G2p[:-1, :-1] + G2p[:-1, 1:])
    c = np.log(np.maximum(G1, eps)) - np.log(np.maximum(G2d, eps))
    mR, mB, mG = (float(np.median(x[ok])) for x in (R, B, G))
    k = math.sqrt(max((0.5 + 0.25 * (mG / max(mR, 1e-6) + mG / max(mB, 1e-6))) / 1.25, 1e-6))
    c = c * np.float32(k)
    # brightness trend: a smooth (quadratic in log G) robust fit, so a level-dependent gain goes but a material's
    # colour, which is not a smooth function of brightness, stays
    v, hv = lG[ok].astype(np.float64), h[ok].astype(np.float64)
    if v.size > 200 and np.ptp(v) > 1e-3:
        sub = slice(None, None, max(1, v.size // 200000))
        vv, hh = v[sub], hv[sub]
        wts = np.ones_like(vv)
        coef = np.zeros(3)
        for _ in range(4):                       # iteratively reweighted (Tukey-like) for robustness
            coef = np.polyfit(vv, hh, 2, w=wts)
            r = hh - np.polyval(coef, vv)
            sc = 1.4826 * np.median(np.abs(r)) + 1e-9
            wts = np.clip(1.0 - (r / (4.0 * sc)) ** 2, 0.0, 1.0) ** 2 + 1e-6
        h = h - np.polyval(coef, lG).astype(np.float32)
    else:
        h = h - np.float32(np.median(hv))
    c = c - np.float32(np.median(c[ok]))
    gm = np.hypot(*np.gradient(gaussian_filter(lG, 1.0)))
    ok &= gm < np.percentile(gm[ok], 75)
    hs = _nconv(h, ok, sigma)
    cs = _nconv(c, ok, sigma)
    sh = float(1.4826 * np.median(np.abs(hs[ok] - np.median(hs[ok]))))
    sc = float(1.4826 * np.median(np.abs(cs[ok] - np.median(cs[ok]))))
    ratio = sh / max(sc, 1e-9)
    colour = ratio >= METER_LIMIT
    text = (f"Colour meter {ratio:.1f}x the two-green control: there is colour to work with." if colour else
            f"Colour meter {ratio:.1f}x the two-green control: no usable colour, this raw is black and white.")
    return {"ratio": round(ratio, 2), "colour": bool(colour), "text": text}


def colour_meter(image) -> dict:
    from ..raw import linearize as lin
    raw = image.raw
    if getattr(raw, "is_scan", False):
        return {"ratio": float("nan"), "colour": True, "text": "Not a raw: the meter needs the camera's mosaic."}
    lv = image.levels(True)
    cx, cy, cw, ch = raw.crop
    x0, y0 = cx // 2 * 2, cy // 2 * 2
    with slog.stage("infrared.colour_meter", log):
        mos = lin.normalise(raw, lv, (x0, y0, cw // 2 * 2, ch // 2 * 2))
        r = meter_from_planes(_planes(mos, raw.cfa_rgb))
    log.info("infrared colour meter: %s", r["text"])
    return r


# ------------------------------------------------------------------------------------------- white balance
def _mult_from_ratios(rg: float, bg: float) -> np.ndarray:
    m = np.array([1.0 / max(rg, 1e-6), 1.0, 1.0 / max(bg, 1e-6)])
    return m / m.min()


def foliage_multipliers_cam(cam: np.ndarray, box=None) -> tuple[np.ndarray | None, dict]:
    """White balance that makes foliage neutral, from half-size raw camera RGB: the top 40% of brightness (below
    clipping) whose local texture is moderate (not flat sky, not specular glints); median R/G and B/G of those.
    ``box`` (x0, y0, x1, y1 fractions) limits the search. Returns (multipliers or None, report)."""
    a = np.asarray(cam, np.float32)
    if box is not None:
        h, w = a.shape[:2]
        x0, y0, x1, y1 = box
        a = a[int(y0 * h):max(int(y1 * h), int(y0 * h) + 2), int(x0 * w):max(int(x1 * w), int(x0 * w) + 2)]
    lum = a.mean(axis=2)
    ok = (a.max(axis=2) < 0.95) & (a.min(axis=2) > 1e-4)
    if ok.sum() < 50:
        return None, {"text": "Not enough exposed area to find foliage."}
    ll = np.log(np.maximum(lum, 1e-5))
    var = np.maximum(uniform_filter(ll * ll, 7) - uniform_filter(ll, 7) ** 2, 0.0)
    t_l = np.percentile(lum[ok], 60)
    sel = ok & (lum >= t_l)
    if sel.sum() > 100:
        v = var[sel]
        lo, hi = np.percentile(v, 25), np.percentile(v, 85)
        sel &= (var >= lo) & (var <= hi)
    if sel.sum() < 20:
        return None, {"text": "No foliage-like area found."}
    rg = float(np.median(a[..., 0][sel] / np.maximum(a[..., 1][sel], 1e-6)))
    bg = float(np.median(a[..., 2][sel] / np.maximum(a[..., 1][sel], 1e-6)))
    m = _mult_from_ratios(rg, bg)
    return m, {"text": f"Foliage white balance from {100.0 * sel.mean():.0f}% of the photo (estimated).",
               "fraction": float(sel.mean()), "rg": rg, "bg": bg}


def foliage_multipliers(image, box=None) -> tuple[np.ndarray | None, dict]:
    from .autowb import camera_rgb_half
    return foliage_multipliers_cam(camera_rgb_half(image), box)


def foliage_at(image, xs: float, ys: float, radius: int = 10) -> np.ndarray | None:
    """Click foliage: the median colour of a patch around S-frame point (xs, ys) (full-size pixels) made neutral.
    A median over a larger patch than the neutral eyedropper, because foliage is textured."""
    from .autowb import camera_rgb_half
    cam = camera_rgb_half(image)
    hx, hy = int(round((xs - 0.5) / 2.0)), int(round((ys - 0.5) / 2.0))
    a, b = max(0, hy - radius), min(cam.shape[0], hy + radius + 1)
    c, d = max(0, hx - radius), min(cam.shape[1], hx + radius + 1)
    p = cam[a:b, c:d].reshape(-1, 3)
    p = p[(p.max(1) < 0.95) & (p.min(1) > 1e-5)]
    if p.shape[0] < 4:
        return None
    rg = float(np.median(p[:, 0] / p[:, 1]))
    bg = float(np.median(p[:, 2] / p[:, 1]))
    return _mult_from_ratios(rg, bg)


def gain_warning(mult) -> str:
    """Infrared white balance costs noise: say so once a channel gain passes 4x."""
    m = np.asarray(mult, float)
    g = float(m.max() / m.min())
    if g > 4.0:
        return f"One channel is multiplied {g:.1f}x: Clean (noise reduction) is suggested."
    return ""


# ------------------------------------------------------------------------------------------- best channel
def best_channel(cam_wb: np.ndarray) -> tuple[int, list]:
    """The channel with the most detail for its noise (noise-weighted gradient energy) of a white-balanced
    camera RGB image. Returns (index, scores)."""
    from ..stack.align import noise_sigma
    scores = []
    for c in range(3):
        x = np.asarray(cam_wb[..., c], np.float32)
        gy, gx = np.gradient(gaussian_filter(x, 0.7))
        e = float(np.mean(gx * gx + gy * gy))
        s = noise_sigma(x)
        scores.append(e / max(s * s, 1e-12))
    return int(np.argmax(scores)), [round(v, 2) for v in scores]


def bw_preset_weights(preset: str, cam_wb: np.ndarray | None = None) -> np.ndarray:
    """Mono mix weights for a black and white preset (plan 1.5c)."""
    if preset == "dark_sky":
        return np.array([-0.3, 0.0, 1.3])
    if preset in ("soft_skin", "separation") and cam_wb is not None:
        from ..stack.align import noise_sigma
        a = np.asarray(cam_wb, np.float32)
        if preset == "soft_skin":                 # least texture for its brightness
            tex = []
            for c in range(3):
                x = a[..., c]
                hp = x - gaussian_filter(x, 2.0)
                tex.append(float(np.median(np.abs(hp))) / max(float(np.median(x)), 1e-6))
            w = np.zeros(3)
            w[int(np.argmin(tex))] = 0.7
            return w + 0.1
        # most separation: maximise local contrast for its noise over a small grid of weights
        best, bw = -1.0, np.array([1 / 3] * 3)
        lum = a.mean(axis=2)
        for wr in np.linspace(-0.5, 1.5, 9):
            for wb in np.linspace(-0.5, 1.5, 9):
                w = np.array([wr, 1.0 - wr - wb, wb])
                if np.abs(w).sum() > 2.5:
                    continue
                m = a @ w.astype(np.float32)
                k = float(np.median(m)) / max(float(np.median(lum)), 1e-6)
                if k < 0.3:
                    continue
                lc = float(np.std(m - gaussian_filter(m, 6.0))) / max(k, 1e-6)
                n = noise_sigma(m) / max(k, 1e-6)
                score = lc / max(n, 1e-9)
                if score > best:
                    best, bw = score, w
        return bw
    return np.array([1 / 3] * 3)


# ------------------------------------------------------------------------------------------- noise meter
def noise_factor(cam_wb_full: np.ndarray, bw: dict) -> dict:
    """How much a black and white mix multiplies the noise against the plain average, measured on a demosaiced
    full-size crop at the same brightness (plan 1.5c: the textbook sqrt(sum w^2) is far too pessimistic, because
    demosaicing makes the channels' noise correlated)."""
    from ..stack.align import noise_sigma
    a = np.asarray(cam_wb_full, np.float32)
    avg = a.mean(axis=2)
    mono = mono_of(a, bw)
    ma, mm = float(np.median(avg)), float(np.median(mono))
    if ma <= 1e-6:
        return {"factor": 1.0, "level": 0.0, "colour": "cyan", "text": "Too dark to measure the noise."}
    k = mm / ma
    if k < 0.05:
        return {"factor": float("nan"), "level": k, "colour": "amber",
                "text": "This mix is nearly black: the noise cannot be compared."}
    f = noise_sigma(mono) / max(k * noise_sigma(avg), 1e-12)
    col = "cyan" if f <= 1.5 else "amber"
    if f > 2.0:
        txt = f"This mix multiplies the noise by {f:.1f} (at the same brightness)."
    else:
        txt = f"Noise x{f:.1f} against an even mix (at the same brightness)."
    return {"factor": round(float(f), 2), "level": round(k, 3), "colour": col, "text": txt}


def sample_crops(image, edit, size: int = 384, n: int = 2) -> np.ndarray:
    """White-balanced, demosaiced (the edit's algorithm) camera RGB of a few full-size crops, side by side: what the
    noise meter measures on."""
    from ..raw import demosaic as dm
    from ..raw import linearize as lin
    from ..pipeline.pipeline import wb_multipliers
    raw = image.raw
    lv = image.levels(edit.use_optical_black)
    cx, cy, cw, ch = raw.crop
    mult = wb_multipliers(image, edit)
    s = min(size, cw // 2 * 2, ch // 2 * 2)
    spots = [(0.5, 0.5), (0.3, 0.35), (0.7, 0.65)][:max(1, n)]
    out = []
    for fx, fy in spots:
        x = (cx + int(fx * cw) - s // 2) // 2 * 2
        y = (cy + int(fy * ch) - s // 2) // 2 * 2
        x = int(np.clip(x, cx // 2 * 2, cx + cw - s))
        y = int(np.clip(y, cy // 2 * 2, cy + ch - s))
        mos = lin.normalise(raw, lv, (x, y, s, s))
        pat = raw.cfa_rgb
        for i in (0, 1):
            for j in (0, 1):
                mos[i::2, j::2] *= np.float32(mult[pat[i, j]])
        np.minimum(mos, 1.0, out=mos)
        algo = edit.demosaic if edit.demosaic != "half" else "rcd"
        out.append(dm.demosaic(mos, pat, algo))
    return np.concatenate(out, axis=1)


def noise_cost(image, edit) -> dict:
    ir = normalise(edit.infrared) or default_ir()
    with slog.stage("infrared.noise_meter", log):
        crops = sample_crops(image, edit)
        return noise_factor(crops, ir["bw"])


# ================================================================================================== mono raw
MONO_SIGMA = 16.0            # plane pixels: the smooth local ratio of each CFA position to the greens


def mono_gain_maps(mos: np.ndarray, pat, sigma: float = MONO_SIGMA) -> np.ndarray:
    """Per CFA position (2 x 2 x H/2 x W/2), a smooth local gain relative to the mean of the two greens, measured
    where nothing is clipped or very dark (ratio of smoothed sums: robust to noise). Dividing by it brings the
    four photosites of a quad to the same gain, the level-dependent part included."""
    p = np.asarray(pat)
    h, w = mos.shape[0] // 2 * 2, mos.shape[1] // 2 * 2
    q = {(i, j): mos[i:h:2, j:w:2].astype(np.float32) for i in (0, 1) for j in (0, 1)}
    greens = [q[(i, j)] for i in (0, 1) for j in (0, 1) if int(p[i, j]) in (1, 3)]
    G = 0.5 * (greens[0] + greens[1])
    allq = np.stack(list(q.values()))
    floor = max(float(np.percentile(G, 2)), 1e-5)
    ok = (allq.min(axis=0) > 0.5 * floor) & (allq.max(axis=0) < 0.95)
    if ok.mean() < 0.01:
        return np.ones((2, 2) + G.shape, np.float32)
    okf = ok.astype(np.float32)
    den = gaussian_filter(G * okf, sigma)
    gains = np.ones((2, 2) + G.shape, np.float32)
    for (i, j), x in q.items():
        num = gaussian_filter(x * okf, sigma)
        g = num / np.maximum(den, 1e-9)
        wt = gaussian_filter(okf, sigma)
        glob = float((x[ok]).sum() / max(G[ok].sum(), 1e-9))
        g = np.where(wt > 0.05, g, glob)                 # where nothing was measurable, the frame's own ratio
        # the level-dependent part is a few percent (1.2% on Bill's 850 frames); a larger local swing is detail
        # that the CFA sampled out of phase (aliasing), not gain, so the local map stays within 5% of the frame's
        gains[i, j] = np.clip(g, glob * 0.95, glob * 1.05)
    return gains


def fix_isolated(mono: np.ndarray, k: float = 6.0, ab: tuple | None = None) -> tuple[np.ndarray, int]:
    """Hot and dead photosites in a monochrome mosaic (a demosaic hides them; developing every photosite does not):
    a value further than k noise-sigmas from the median of its 8 neighbours and outside their whole range is
    replaced by that median. ``ab``: the noise model sigma^2 = a * level + b (the same for every region of a photo,
    so a region render fixes exactly what the whole frame does); without it noise is measured from the array."""
    x = mono.astype(np.float32)
    fp = np.ones((3, 3), bool)
    fp[1, 1] = False
    med = median_filter(x, footprint=fp, mode="reflect")
    d = x - med
    from scipy.ndimage import maximum_filter, minimum_filter
    nmax = maximum_filter(x, footprint=fp, mode="reflect")
    nmin = minimum_filter(x, footprint=fp, mode="reflect")
    lev = np.maximum(med, 0.0)
    if ab is not None:
        sig = np.sqrt(np.maximum(np.float32(ab[0]) * lev + np.float32(ab[1]), 1e-10)).astype(np.float32)
    else:
        edges = np.quantile(lev, np.linspace(0, 1, 9))
        sig = np.zeros_like(x)
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (lev >= a) & (lev <= b)
            if sel.any():
                s = 1.4826 * float(np.median(np.abs(d[sel] - np.median(d[sel]))))
                sig[sel] = max(s, 1e-5)
    spread = nmax - nmin                      # a hot pixel stands out from neighbours that agree with each other;
    bad = ((d > k * sig) & (x > nmax + 0.5 * k * sig) & (d > 2.0 * spread)) | \
          ((d < -k * sig) & (x < nmin - 0.5 * k * sig) & (-d > 2.0 * spread))     # fine detail does not
    n = int(bad.sum())
    if n:
        x[bad] = med[bad]
    return x, n


def mono_mosaic(mos: np.ndarray, pat, half: bool = False, gains: np.ndarray | None = None,
                noise_ab: tuple | None = None) -> np.ndarray:
    """850 nm and deeper (plan 1.5b): the four CFA positions brought to the same gain, isolated hot / dead
    photosites fixed, and every photosite used as a luminance sample. Returns HxWx3 with R = G = B (half size when
    ``half``). ``gains`` (from mono_gain_maps on the whole frame, sliced to this mosaic) keep region renders equal to
    the full frame."""
    h, w = mos.shape[0] // 2 * 2, mos.shape[1] // 2 * 2
    m = mos[:h, :w].astype(np.float32, copy=True)
    g = gains if gains is not None else mono_gain_maps(m, pat)
    for i in (0, 1):
        for j in (0, 1):
            m[i::2, j::2] /= g[i, j][:m[i::2, j::2].shape[0], :m[i::2, j::2].shape[1]]
    if half:
        mono = 0.25 * (m[0::2, 0::2] + m[0::2, 1::2] + m[1::2, 0::2] + m[1::2, 1::2])
    else:
        mono, n = fix_isolated(m, ab=noise_ab)
        if n:
            log.debug("mono raw: %d isolated photosites fixed", n)
    return np.repeat(mono[..., None], 3, axis=2).astype(np.float32)


# ================================================================================================== palettes
def _palette_lut(name: str, n: int = 1024) -> np.ndarray:
    """Palette stops (sRGB) -> n x 3 linear-light ProPhoto values along encoded brightness 0..1."""
    from .spaces import PROPHOTO, SRGB, conversion_matrix, srgb_decode
    stops = PALETTES.get(name) or PALETTES["platinum"]
    pos = np.array([s[0] for s in stops])
    col = np.array([s[1] for s in stops], float)
    t = np.linspace(0, 1, n)
    rgb = np.stack([np.interp(t, pos, col[:, c]) for c in range(3)], -1)
    lin = srgb_decode(rgb)
    return (lin @ conversion_matrix(SRGB, PROPHOTO).T).astype(np.float32)


def apply_palette(out: np.ndarray, palette: dict) -> np.ndarray:
    """Colour by brightness on the finished (display-referred, linear ProPhoto) picture: added colour."""
    from .spaces import srgb_encode
    if not palette or float(palette.get("amount", 0)) <= 0:
        return out
    lut = _palette_lut(palette.get("name", "platinum"))
    a = np.float32(np.clip(float(palette.get("amount", 1.0)), 0.0, 1.0))
    Y = out @ np.array([0.2880, 0.7119, 0.0001], np.float32)
    e = np.clip(srgb_encode(np.clip(Y, 0.0, 1.0)), 0.0, 1.0)
    idx = np.clip((e * (lut.shape[0] - 1)).astype(np.int32), 0, lut.shape[0] - 1)
    col = lut[idx]
    return (out + a * (col - out)).astype(np.float32)


# ================================================================================================== saved data
def _safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9 _.-]+", "_", str(name)).strip(" .")
    return s[:80] or "unnamed"


def filter_wb_dir() -> Path:
    from ... import paths
    d = paths.get("CALIBRATION_LOCAL") / "infrared"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_filter_wb(name: str, mult, camera: str = "", filter_key: str = "") -> Path:
    """'Measure this filter': a grey card's multipliers saved by filter name on this PC (cyan)."""
    p = filter_wb_dir() / (_safe(name) + ".json")
    m = np.asarray(mult, float)
    m = m / m.min()
    p.write_text(json.dumps({"name": name, "multipliers": [round(float(v), 6) for v in m], "camera": camera,
                             "filter": filter_key, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "v": 1},
                            indent=2), encoding="utf-8")
    return p


def load_filter_wbs(camera: str | None = None) -> list:
    out = []
    try:
        for p in sorted(filter_wb_dir().glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if camera and d.get("camera") and d["camera"] != camera:
                continue
            out.append(d)
    except OSError:
        pass
    return out


def recipe_dir() -> Path:
    from ... import paths
    d = paths.get("SETTINGS") / "ir_recipes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_recipe(name: str, bw: dict, camera: str = "", filter_key: str = "") -> Path:
    """'Save as B&W recipe': the channel settings by name on this PC (beside Looks, never on C:\\)."""
    keep = {k: bw.get(k) for k in ("ev", "contrast", "sep", "curve", "mix", "keep", "preset")}
    p = recipe_dir() / (_safe(name) + ".json")
    p.write_text(json.dumps({"name": name, "bw": keep, "camera": camera, "filter": filter_key, "v": 1}, indent=2),
                 encoding="utf-8")
    remember_last(name, camera, filter_key)
    return p


def load_recipes(filter_key: str | None = None) -> list:
    out = []
    for p in sorted(recipe_dir().glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if filter_key and d.get("filter") and d["filter"] != filter_key:
            continue
        out.append(d)
    return out


def apply_recipe(bw: dict, recipe: dict) -> dict:
    out = dict(bw)
    for k, v in (recipe.get("bw") or {}).items():
        out[k] = json.loads(json.dumps(v))
    out["on"] = True
    out["recipe"] = recipe.get("name", "")
    return out


def _last_path() -> Path:
    return recipe_dir() / "_last.json"


def remember_last(name: str, camera: str, filter_key: str) -> None:
    try:
        d = json.loads(_last_path().read_text(encoding="utf-8")) if _last_path().is_file() else {}
    except (OSError, ValueError):
        d = {}
    d[f"{camera}|{filter_key}"] = name
    _last_path().write_text(json.dumps(d, indent=2), encoding="utf-8")


def last_recipe(camera: str, filter_key: str) -> dict | None:
    try:
        d = json.loads(_last_path().read_text(encoding="utf-8")) if _last_path().is_file() else {}
    except (OSError, ValueError):
        return None
    name = d.get(f"{camera}|{filter_key}")
    if not name:
        return None
    for r in load_recipes():
        if r.get("name") == name:
            return r
    return None


# ================================================================================================== the tab's actions
def camera_half_wb(image, edit) -> np.ndarray:
    """Half-size white-balanced camera RGB (no matrix): what the presets, thumbnails and best-channel test use."""
    from .autowb import camera_rgb_half
    from ..pipeline.pipeline import wb_multipliers
    cam = camera_rgb_half(image)
    m = wb_multipliers(image, edit).astype(np.float32)
    return np.minimum(cam * m[None, None, :], 1.0)


def channel_thumbs(cam_wb: np.ndarray, long_edge: int = 120) -> list:
    """The three channels each as its own greyscale picture (uint8), like Photoshop's channels palette, on one
    shared brightness scale so their differences show."""
    a = np.asarray(cam_wb, np.float32)
    k = max(1, int(math.ceil(max(a.shape[:2]) / float(long_edge))))
    a = _bin(a, k) if k > 1 else a
    top = max(float(np.percentile(a, 99.5)), 1e-6)
    out = []
    for c in range(3):
        v = np.clip(a[..., c] / top, 0.0, 1.0) ** (1 / 2.2)
        out.append((v * 255 + 0.5).astype(np.uint8))
    return out


def measured_wb_for(camera: str, filter_key: str) -> dict | None:
    for d in load_filter_wbs(camera):
        if d.get("filter") == filter_key:
            return d
    return None


def start(image, edit, filter_key: str | None = None) -> tuple:
    """Turn the Infrared treatment on for a photo: the colour meter, the filter guess, white balance (the filter's
    measured white balance for this camera when there is one, cyan; else foliage, amber; else the camera's own) and
    the filter's default look. Returns (new Edit, report dict)."""
    from ..pipeline.edit import Edit, WhiteBalance
    meter = colour_meter(image)
    guess = guess_filter(image, meter.get("ratio"))
    key = filter_key or guess["key"]
    e = Edit.from_dict(edit.to_dict())
    ir = default_ir(key, guess)
    if key == "850" or not meter.get("colour", True):
        ir["mono_raw"] = key == "850"
    camera = getattr(image.meta, "model", "") or ""
    rep = {"meter": meter, "guess": guess, "wb": ""}
    meas = measured_wb_for(camera, key)
    if meas:
        e.white_balance = WhiteBalance(mode="multipliers", multipliers=list(meas["multipliers"]))
        ir["wb_source"] = f"measured:{meas['name']}"
        rep["wb"] = f"White balance measured for this filter ({meas['name']})."
        rep["wb_colour"] = "cyan"
    else:
        m, r = foliage_multipliers(image)
        if m is not None:
            e.white_balance = WhiteBalance(mode="multipliers", multipliers=[round(float(v), 5) for v in m])
            ir["wb_source"] = "foliage"
            rep["wb"] = r["text"]
        else:
            ir["wb_source"] = "as_shot"
            rep["wb"] = "No foliage found: the camera's own white balance is used. Click something neutral."
        rep["wb_colour"] = "amber"
    e.infrared = ir
    e = apply_preset(image, e, ir["preset"], 1.0)
    last = last_recipe(camera, key)
    if last and e.infrared["bw"].get("on"):
        e.infrared["bw"] = apply_recipe(e.infrared["bw"], last)
        rep["recipe"] = last.get("name", "")
    gw = gain_warning(wb_from(e))
    if gw:
        rep["noise"] = gw
    return e, rep


def wb_from(edit) -> np.ndarray:
    wb = edit.white_balance
    if wb.mode == "multipliers" and wb.multipliers:
        m = np.asarray(wb.multipliers, float)
        return m / m.min()
    return np.ones(3)


def apply_preset(image, edit, preset: str, strength: float = 1.0):
    """A look: a colour mixer, or black and white from the channels (mono_best measures the photo). Returns a new
    Edit. ``candle_glow`` also warms the highlights in Colour grading (added colour)."""
    from ..pipeline.edit import Edit
    e = Edit.from_dict(edit.to_dict())
    ir = normalise(e.infrared) or default_ir()
    ir["preset"] = preset
    ir["strength"] = float(strength)
    bw = dict(ir["bw"])
    if preset in ("mono_best", "mono_red", "candle_glow"):
        w = [1.0, 0.0, 0.0]
        if preset in ("mono_best", "candle_glow") and image is not None:
            try:
                k, _ = best_channel(camera_half_wb(image, e))
                w = [0.0, 0.0, 0.0]
                w[k] = 1.0
            except Exception:                    # noqa: BLE001 - the even mix is a safe answer
                slog.log_exception("infrared: best channel failed", log)
                slog.save_error_report("Infrared: measuring the best channel")
                w = [1 / 3] * 3
        if preset in ("mono_best", "candle_glow") and ir.get("filter") == "850":
            w = [1 / 3] * 3                      # at 850 the channels hold the same picture: the even mix is quietest
        bw.update(on=True, mix=w, preset="even" if w == [1 / 3] * 3 else "", keep=True)
        if preset == "candle_glow":
            g = dict(e.colour.grading or {})
            g.update(json.loads(json.dumps(CANDLE_GRADING)))
            e.colour.grading = g
    else:
        bw["on"] = False
        ir["mixer"] = [round(float(v), 6) for v in mixer(preset, strength).ravel()]
    ir["bw"] = bw
    e.infrared = ir
    return e


def ir_part(edit) -> dict:
    """What 'Apply to the shoot' and Paste carry: the infrared settings and the white balance."""
    from dataclasses import asdict
    return {"infrared": json.loads(json.dumps(edit.infrared)), "white_balance": asdict(edit.white_balance),
            "grading": json.loads(json.dumps(edit.colour.grading)) if (edit.infrared or {}).get("preset") ==
            "candle_glow" else None}


def paste_ir(edit, part: dict):
    from ..pipeline.edit import Edit, WhiteBalance
    e = Edit.from_dict(edit.to_dict())
    e.infrared = json.loads(json.dumps(part["infrared"]))
    if e.infrared:
        e.infrared["guess"] = None                  # the guess was made for the other photo
    e.white_balance = WhiteBalance(**{k: v for k, v in part["white_balance"].items()
                                      if k in WhiteBalance.__dataclass_fields__})
    if part.get("grading"):
        e.colour.grading = json.loads(json.dumps(part["grading"]))
    return e


def turn_off(edit):
    """Back to normal colour: the infrared settings are kept (switched off) so turning it on again loses nothing."""
    from ..pipeline.edit import Edit
    e = Edit.from_dict(edit.to_dict())
    if e.infrared:
        e.infrared = dict(e.infrared, on=False)
    return e
