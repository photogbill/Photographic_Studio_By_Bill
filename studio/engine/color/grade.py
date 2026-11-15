"""Stage 17: colour (plan 9.2). Saturation, vibrance, HSL, the DxO-style colour wheel, colour grading, black and
white with a colour mixer, and .cube look LUTs.

All of it works in OKLab / OKLCh (Bjorn Ottosson, 2020): a perceptual space where changing hue or chroma keeps the
perceived lightness, and lightness changes keep the hue (no blue turning purple when it is darkened, the classic
HSL problem). It runs on display-referred linear ProPhoto after the tone curve, so the colour you click on the
screen is the colour that gets changed. (Plan section 3 places stage 17 before the curve; on screen-picked
colours this order is the honest one. Recorded in docs/PROGRESS.md.)

Neutral greys are never tinted by hue controls: every hue-selective weight is multiplied by a chroma gate, so a
grey has no hue to select.
"""
from __future__ import annotations

import math
import os

import numpy as np

from ... import log as slog
from .spaces import PROPHOTO, SRGB, conversion_matrix

log = slog.get("engine.colour")

# OKLab (Ottosson 2020): linear sRGB -> LMS, then cube root, then Lab
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
PP_TO_SRGB = conversion_matrix(PROPHOTO, SRGB)
TO_LMS = (_M1 @ PP_TO_SRGB).astype(np.float32)
FROM_LMS = np.linalg.inv(TO_LMS).astype(np.float32)
M2 = _M2.astype(np.float32)
M2_INV = np.linalg.inv(_M2).astype(np.float32)

# Hue bands for HSL and the B&W mixer (OKLCh hue angles of the photographer's named colours, degrees)
BANDS = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
BAND_HUES = [25.0, 60.0, 100.0, 140.0, 195.0, 255.0, 295.0, 335.0]
SKIN_HUE = 55.0              # skin tones sit around here in OKLCh (orange-red), protected by vibrance


def to_oklab(rgb_pp: np.ndarray) -> np.ndarray:
    lms = rgb_pp.reshape(-1, 3) @ TO_LMS.T
    return (np.cbrt(lms) @ M2.T).reshape(rgb_pp.shape).astype(np.float32)


def from_oklab(lab: np.ndarray) -> np.ndarray:
    lms_ = lab.reshape(-1, 3) @ M2_INV.T
    return ((lms_ ** 3) @ FROM_LMS.T).reshape(lab.shape).astype(np.float32)


def hue_dist(h, c):
    """Signed smallest angle h - c in degrees."""
    return (np.asarray(h) - c + 180.0) % 360.0 - 180.0


def band_weights(h: np.ndarray) -> np.ndarray:
    """(8, ...) smooth weights of each named hue band; they sum to 1 around the wheel."""
    h = np.asarray(h, np.float32)
    n = len(BAND_HUES)
    w = np.zeros((n,) + h.shape, np.float32)
    for i in range(n):
        c0, c1 = BAND_HUES[i], BAND_HUES[(i + 1) % n]
        span = (c1 - c0) % 360.0
        t = ((h - c0) % 360.0) / span
        inside = t < 1.0                                          # the next band takes t = 1
        rise = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))      # 0 at c0 -> 1 at c1
        w[i] += np.where(inside, 1.0 - rise, 0.0)
        w[(i + 1) % n] += np.where(inside, rise, 0.0)
    return w


def range_weight(h: np.ndarray, centre: float, width: float, falloff: float) -> np.ndarray:
    """DxO-style colour-wheel range: full strength within width/2 of the centre, cosine falloff beyond."""
    d = np.abs(hue_dist(h, centre))
    half = max(width, 0.0) / 2.0
    f = max(falloff, 1e-3)
    t = np.clip((d - half) / f, 0.0, 1.0)
    return (0.5 + 0.5 * np.cos(np.pi * t)).astype(np.float32)


def chroma_gate(C: np.ndarray) -> np.ndarray:
    return (C / (C + np.float32(0.02))).astype(np.float32)


def tonal_weights(L: np.ndarray, balance: float, blending: float):
    """Shadows / midtones / highlights weights from OKLab lightness for colour grading."""
    pivot = 0.5 + 0.25 * (balance / 100.0)
    soft = 0.12 + 0.28 * (blending / 100.0)
    sh = 1.0 / (1.0 + np.exp((L - (pivot - 0.15)) / (soft * 0.35)))
    hi = 1.0 / (1.0 + np.exp(-(L - (pivot + 0.15)) / (soft * 0.35)))
    mid = np.clip(1.0 - sh - hi, 0.0, 1.0)
    return sh.astype(np.float32), mid.astype(np.float32), hi.astype(np.float32)


# Black & white filter presets: what a coloured filter on the lens does to black-and-white film (bw_mix values).
BW_FILTERS = {
    "Red filter": {"red": 60, "orange": 45, "yellow": 20, "green": -30, "aqua": -45, "blue": -60, "purple": -20,
                   "magenta": 30},
    "Orange filter": {"red": 40, "orange": 40, "yellow": 25, "green": -15, "aqua": -30, "blue": -40, "purple": -15,
                      "magenta": 15},
    "Yellow filter": {"red": 15, "orange": 25, "yellow": 30, "green": 5, "aqua": -15, "blue": -25, "purple": -10,
                      "magenta": 0},
    "Green filter": {"red": -30, "orange": -10, "yellow": 20, "green": 45, "aqua": 20, "blue": -10, "purple": -20,
                     "magenta": -25},
    "Blue filter": {"red": -40, "orange": -35, "yellow": -20, "green": -5, "aqua": 25, "blue": 50, "purple": 25,
                    "magenta": 0},
}


def colourfulness(srgb8: np.ndarray) -> float:
    """Hasler and Susstrunk (2003) colourfulness M of an sRGB 8-bit image: about 15 slightly, 33 moderately,
    45 averagely, 59 quite, 82 highly colourful."""
    a = np.asarray(srgb8, np.float64).reshape(-1, 3)
    rg = a[:, 0] - a[:, 1]
    yb = 0.5 * (a[:, 0] + a[:, 1]) - a[:, 2]
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


AUTO_COLOUR_TARGET = 45.0


def auto_vibrance(m: float) -> float:
    """Vibrance that brings a dull photo toward 'averagely colourful'. Never lowers colour, at most +35."""
    if m <= 1e-6 or m >= AUTO_COLOUR_TARGET:
        return 0.0
    return float(min(35.0, 60.0 * (AUTO_COLOUR_TARGET / m - 1.0)))


def _wheel_active(v) -> bool:
    """A grading wheel [hue, strength, lum] does something when it has a tint or a lightness change."""
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return False
    return abs(float(v[1])) > 1e-6 or (len(v) >= 3 and abs(float(v[2])) > 1e-6)


def is_identity(c) -> bool:
    if c is None:
        return True
    if abs(c.saturation) > 1e-6 or abs(c.vibrance) > 1e-6 or c.bw or (c.lut and c.lut_amount > 0):
        return False
    if c.hsl and any(abs(float(v)) > 1e-6 for vals in c.hsl.values() for v in vals):
        return False
    if c.ranges and any(abs(r.get("shift", 0)) > 1e-6 or abs(r.get("sat", 0)) > 1e-6 or abs(r.get("lum", 0)) > 1e-6
                        for r in c.ranges):
        return False
    if c.grading and any(_wheel_active(v) for k, v in c.grading.items()
                         if k in ("shadows", "midtones", "highlights", "global")):
        return False
    return True


def apply(rgb: np.ndarray, c) -> np.ndarray:
    """Colour stage on display-referred linear ProPhoto (H x W x 3). Returns a new array."""
    if is_identity(c):
        return rgb
    with slog.stage("colour", log, mp=round(rgb.shape[0] * rgb.shape[1] / 1e6, 2)):
        lab = to_oklab(rgb)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        C = np.hypot(a, b)
        h = np.degrees(np.arctan2(b, a)) % 360.0
        gate = chroma_gate(C)
        dh = np.zeros_like(C)
        kC = np.ones_like(C)
        dL = np.zeros_like(C)
        # HSL (8 bands): hue shift up to +-30 deg, saturation x0..x2, lightness +-0.15 (chroma-gated)
        if c.hsl:
            W = band_weights(h)
            for i, name in enumerate(BANDS):
                v = c.hsl.get(name)
                if not v:
                    continue
                hh, ss, ll = (float(x) for x in (list(v) + [0, 0, 0])[:3])
                wi = W[i] * gate
                if hh:
                    dh += wi * 0.30 * hh
                if ss:
                    kC *= 1.0 + wi * ss / 100.0
                if ll:
                    dL += wi * 0.15 * ll / 100.0
        # colour wheel ranges (DxO style)
        for r in (c.ranges or []):
            w = range_weight(h, float(r.get("hue", 0)), float(r.get("width", 40)), float(r.get("falloff", 25))) * gate
            if r.get("shift"):
                dh += w * float(r["shift"])
            if r.get("sat"):
                kC *= 1.0 + w * float(r["sat"]) / 100.0
            if r.get("lum"):
                dL += w * 0.15 * float(r["lum"]) / 100.0
        # global saturation and vibrance (vibrance: more for muted colours, less for skin)
        if c.saturation:
            kC *= 1.0 + c.saturation / 100.0
        if c.vibrance:
            muted = np.clip(1.0 - C / 0.25, 0.0, 1.0) ** 2
            skin = 1.0 - 0.7 * range_weight(h, SKIN_HUE, 30.0, 25.0) * gate
            kC *= 1.0 + (c.vibrance / 100.0) * muted * (skin if c.vibrance > 0 else 1.0)
        kC = np.maximum(kC, 0.0)
        C2 = C * kC
        h2 = np.radians(h + dh)
        L2 = L + dL
        if c.bw:
            mix = c.bw_mix or {}
            if any(abs(float(mix.get(n, 0))) > 1e-6 for n in BANDS):
                W = band_weights(h)
                for i, name in enumerate(BANDS):
                    m = float(mix.get(name, 0))
                    if m:
                        L2 = L2 + W[i] * np.clip(C / 0.15, 0.0, 1.0) * 0.25 * m / 100.0
            a2 = np.zeros_like(L2)
            b2 = np.zeros_like(L2)
        else:
            a2, b2 = C2 * np.cos(h2), C2 * np.sin(h2)
        # colour grading: tint shadows / midtones / highlights (OKLab a, b offsets), after B&W so it split-tones
        g = c.grading or {}
        wheels = [(k, g.get(k)) for k in ("shadows", "midtones", "highlights", "global")]
        if any(_wheel_active(v) for _, v in wheels):
            sh, mid, hi = tonal_weights(L2, float(g.get("balance", 0)), float(g.get("blending", 50)))
            tw = {"shadows": sh, "midtones": mid, "highlights": hi, "global": 1.0}
            for k, v in wheels:
                if not _wheel_active(v):
                    continue
                hue, strength = math.radians(float(v[0])), float(v[1]) / 100.0
                if strength:
                    amt = 0.06 * strength * tw[k]
                    a2 = a2 + amt * math.cos(hue)
                    b2 = b2 + amt * math.sin(hue)
                if len(v) >= 3 and float(v[2]):
                    L2 = L2 + 0.1 * float(v[2]) / 100.0 * tw[k]
        out = from_oklab(np.stack([np.maximum(L2, 0.0), a2, b2], -1))
        if c.lut and c.lut_amount > 0:
            out = apply_lut(out, load_lut(c.lut), float(c.lut_amount))
    return out


# ------------------------------------------------------------------------------------------ .cube LUTs
_LUTS: dict = {}


def lut_dirs() -> list:
    from ... import paths
    return [str(paths.PROFILES / "looks" / "luts"), str(paths.get("SETTINGS") / "luts")]


def find_lut(name: str) -> str | None:
    if os.path.isfile(name):
        return name
    for d in lut_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def list_luts() -> list:
    out = []
    for d in lut_dirs():
        if os.path.isdir(d):
            out += sorted(f for f in os.listdir(d) if f.lower().endswith(".cube"))
    return out


def parse_cube(text: str) -> np.ndarray:
    """Adobe / Resolve .cube 3D LUT -> array (N, N, N, 3) indexed [r, g, b]. Domain must be 0..1."""
    size, rows, dmin, dmax = None, [], (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    for line in text.splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        k = t.split()
        if k[0].upper() == "LUT_3D_SIZE":
            size = int(k[1])
        elif k[0].upper() == "DOMAIN_MIN":
            dmin = tuple(float(v) for v in k[1:4])
        elif k[0].upper() == "DOMAIN_MAX":
            dmax = tuple(float(v) for v in k[1:4])
        elif k[0].upper() in ("TITLE", "LUT_1D_SIZE"):
            if k[0].upper() == "LUT_1D_SIZE":
                raise ValueError("1D LUTs are not supported (use a 3D .cube)")
        else:
            try:
                rows.append([float(v) for v in k[:3]])
            except ValueError:
                continue
    if not size or len(rows) != size ** 3:
        raise ValueError(f"not a valid 3D .cube LUT (size {size}, {len(rows)} entries)")
    if dmin != (0.0, 0.0, 0.0) or dmax != (1.0, 1.0, 1.0):
        raise ValueError("only .cube LUTs with a 0..1 domain are supported")
    # .cube order: red changes fastest
    return np.asarray(rows, np.float32).reshape(size, size, size, 3).transpose(2, 1, 0, 3).copy()


def load_lut(name: str) -> np.ndarray:
    p = find_lut(name)
    if p is None:
        raise FileNotFoundError(f"LUT {name!r} not found in {lut_dirs()}")
    if p not in _LUTS:
        with open(p, encoding="utf-8", errors="replace") as fh:
            _LUTS[p] = parse_cube(fh.read())
    return _LUTS[p]


def apply_lut(rgb_pp: np.ndarray, lut: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Apply a look LUT that expects sRGB-encoded values (the usual for photo looks), trilinear.
    The LUT only sees the in-gamut part of each colour; its change is added back on top, so colours outside
    sRGB (ProPhoto holds plenty) keep their extra saturation instead of being clipped by the look."""
    from scipy.ndimage import map_coordinates
    M = conversion_matrix(PROPHOTO, SRGB).astype(np.float32)
    Mi = np.linalg.inv(M).astype(np.float32)
    s = rgb_pp.reshape(-1, 3) @ M.T
    sc = np.clip(s, 0.0, 1.0)
    e = SRGB.encode(sc).astype(np.float32)
    n = lut.shape[0]
    coords = (e * (n - 1)).T
    out = np.stack([map_coordinates(lut[..., k], coords, order=1, mode="nearest") for k in range(3)], -1)
    lin = SRGB.decode(np.clip(out, 0.0, 1.0)).astype(np.float32)
    res = ((s + (lin - sc)) @ Mi.T).reshape(rgb_pp.shape)
    if amount < 1.0:
        res = rgb_pp + np.float32(amount) * (res - rgb_pp)
    return res.astype(np.float32)
