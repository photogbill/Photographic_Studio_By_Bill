"""Colour helpers for Paint: HSV and hex, the colour picker (Alt + click), and the blend spaces paint mixes in.

Blend spaces (BUILD_PLAN_PAINT 0.5, 0.9), each saying what it is best for and weak at:
- classic  mixes the stored (gamma-encoded) values, as Photoshop, Krita and Clip Studio do: what painters expect
- light    mixes in linear light: glows and coloured light mix like real light, edges of soft brushes look brighter
- oklab    mixes in Oklab (Bjorn Ottosson, 2020): even lightness, so blends of two hues do not dip into grey mud
- pigment  mixes like paint: Kubelka-Munk theory over 38 wavelengths, ported from spectral.js (Ronald van Wijnen,
           MIT; data in spectral_data.py). Each colour becomes a reflectance spectrum, the spectra are mixed through
           their absorption / scattering ratio K/S weighted by concentration (amount squared times luminance, as
           spectral.js does), and the mix is turned back into a colour. So blue and yellow make green. Mixbox is
           NOT used (its licence forbids it); this is our own code on MIT data.
- lpm      Lorentz Pigment Mixing, adapted (Ivkovic et al., IntechOpen 2025, DOI 10.5772/acrt.20250065; their code is
           GPL-3.0, github.com/RatkoIV/LPM, ported here). Their published code is an image filter: each channel is
           pushed towards the next (R->G, G->B, B->R) by a Lorentz curve, multiplied, blended 30 % with the original,
           brightened and its darks deepened. Used as a mixer it would change a colour mixed with itself and never
           reach the two end colours, so here the filter acts only on the middle of a mix: its strength is zero at
           both ends, grows with how different the two colours are, and keeps the plain mix's lightness. Not
           physics, judged by eye (as the paper does); our Kubelka-Munk pigment mode (P2) is the one that mixes
           like paint.
"""
from __future__ import annotations

import colorsys

import numpy as np

from .tiles import Document

SPACES = {
    "classic": ("Classic", "what painters expect from Photoshop, Krita and Clip Studio", "two hues mix through grey"),
    "light": ("Light (linear)", "glows, fire and coloured light that mix like real light",
              "dark colours spread further at a brush's soft edge"),
    "oklab": ("Oklab (even lightness)", "smooth blends between hues without a muddy middle, even gradients",
              "it is not paint: blue and yellow give a clean mid colour, not green (pigment mixing comes in P2)"),
    "pigment": ("Pigment (Kubelka-Munk)", "mixing like real paint: blue and yellow make green, mixes darken and dull "
                "the way pigments do", "slower on big brushes; dark colours win mixes, as real pigments do; a model "
                "of paint, not measured tube colours"),
    "lpm": ("LPM (Lorentz, adapted)", "a touch more saturation where two different colours meet, so blends stay "
            "a little livelier than Classic", "the effect is small (about 1 % on most mixes); not physics, judged "
            "only by eye"),
}


def hsv_to_rgb(h: float, s: float, v: float) -> tuple:
    return colorsys.hsv_to_rgb(h % 1.0, max(0, min(1, s)), max(0, min(1, v)))


def rgb_to_hsv(r: float, g: float, b: float) -> tuple:
    return colorsys.rgb_to_hsv(r, g, b)


def to_hex(rgb) -> str:
    return "#" + "".join(f"{int(round(max(0, min(1, c)) * 255)):02x}" for c in rgb[:3])


def from_hex(s: str) -> tuple:
    s = s.lstrip("#")
    if len(s) != 6:
        raise ValueError("a colour is # and six hex digits")
    return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def luminance(rgb) -> float:
    """Rec. 709 luma of an sRGB colour (the value-only view)."""
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def srgb_to_linear(v):
    v = np.asarray(v, np.float32)
    return np.where(v <= 0.04045, v / 12.92, ((np.maximum(v, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(v):
    v = np.clip(np.asarray(v, np.float32), 0, None)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055).astype(np.float32)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
_M1i = np.linalg.inv(_M1).astype(np.float32)
_M2i = np.linalg.inv(_M2).astype(np.float32)


def srgb_to_oklab(rgb):
    lms = srgb_to_linear(rgb) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_srgb(lab):
    lms = (np.asarray(lab, np.float32) @ _M2i.T) ** 3
    return np.clip(linear_to_srgb(lms @ _M1i.T), 0, 1)


def mix(a, b, t, space: str = "classic"):
    """Straight sRGB colours ``a`` and ``b`` mixed by ``t`` (0 = a, 1 = b) in a blend space."""
    if space == "light":
        return linear_to_srgb(srgb_to_linear(a) * (1 - t) + srgb_to_linear(b) * t)
    if space == "oklab":
        return oklab_to_srgb(srgb_to_oklab(a) * (1 - t) + srgb_to_oklab(b) * t)
    if space == "lpm":
        return lpm_mix(a, b, t)
    if space == "pigment":
        return pigment_mix(a, b, t)
    return np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t


def lpm_filter(c, x):
    """The LPM transform from the published code, per pixel: ``c`` straight sRGB (..., 3), ``x`` the position on the
    Lorentz curve (0..1; in the paper the picture's row, here the mix amount)."""
    c = np.clip(np.asarray(c, np.float32), 0, 1)
    R, G, B = c[..., 0], c[..., 1], c[..., 2]
    x = np.asarray(x, np.float32)
    x = x[..., 0] if x.ndim == c.ndim else x
    L = 1.0 / (1.0 + (x - 0.5) ** 2)                        # the Lorentz curve
    sf = 0.05                                              # scaleFactor: minimal colour change
    RG = (1 - sf) * R + sf * (R * (1 - L) + G * L)
    GB = (1 - sf) * G + sf * (G * (1 - L) + B * L)
    BR = (1 - sf) * B + sf * (B * (1 - L) + R * L)
    enh = np.stack([RG * (1 - GB), GB * (1 - BR), BR * (1 - RG)], -1)   # each channel's dominance
    f = 0.7 * c + 0.3 * enh                                # maxDifference 0.3
    f = np.power(np.maximum(f * 1.4, 0), 0.8)              # brighten, gamma 0.8
    f = np.where(f < 0.5, f * 0.9, f)                      # deepen the darks
    return np.minimum(f, 1.0).astype(np.float32)


def lpm_mix(a, b, t):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    t = np.asarray(t, np.float32)
    m = a * (1 - t) + b * t                                # the plain (Classic) mix
    f = lpm_filter(m, t)
    lum = np.array([0.2126, 0.7152, 0.0722], np.float32)
    lm, lf = (m * lum).sum(-1, keepdims=True), (f * lum).sum(-1, keepdims=True)
    f = np.clip(f * (lm / np.maximum(lf, 1e-6)), 0, 1)     # keep the plain mix's lightness
    diff = np.minimum(1.0, np.abs(a - b).max(-1, keepdims=True) / 0.5)
    w = 4 * t * (1 - t) * diff                             # zero at both ends and for a colour with itself
    return np.clip(m + w * (f - m), 0, 1).astype(np.float32)


def over(base: np.ndarray, colour, a: np.ndarray, space: str = "classic") -> np.ndarray:
    """Paint of ``colour`` (straight sRGB) at alpha ``a`` over ``base`` (premultiplied RGBA), mixed in a blend space."""
    colour = np.asarray(colour, np.float32)
    if space == "classic":
        return np.concatenate([colour * a, a], -1) + base * (1 - a)
    ba = base[..., 3:4]
    oa = a + ba * (1 - a)
    bc = base[..., :3] / np.maximum(ba, 1e-8)
    t = a / np.maximum(oa, 1e-8)                      # the paint's share of the visible colour
    c = mix(bc, colour if space == "pigment" else np.broadcast_to(colour, bc.shape), t, space)
    c = np.where(ba > 0, c, colour)
    return np.concatenate([c * oa, oa], -1).astype(np.float32)


def pick(doc: Document, x: float, y: float, radius: int = 0, layer_only: bool = False):
    """The colour under (x, y): the whole picture (default) or the active layer only, averaged over a small square
    when ``radius`` > 0. Returns (r, g, b) straight, or None where there is no paint (layer only)."""
    x0, y0 = int(np.floor(x)) - radius, int(np.floor(y)) - radius
    n = 2 * radius + 1
    px = doc.layer.read(x0, y0, n, n) if layer_only else doc.composite(x0, y0, n, n)
    a = px[..., 3].sum()
    if a <= 1e-6:
        return None
    return tuple(float(v) for v in px[..., :3].sum(axis=(0, 1)) / a)


# ---------------------------------------------------------------- OKLCH and gamuts (the wide-gamut picker)
# The canvas stores sRGB. The wide-gamut picker works in OKLCH (lightness, chroma, hue: even steps to the eye) and
# shows where sRGB and Display P3 end, so the artist sees which colours the canvas can hold; a colour outside sRGB is
# brought inside at the same lightness and hue by lowering chroma (never by clipping channels, which shifts hue).
_LIN_SRGB_TO_P3 = np.array([[0.8224621, 0.1775380, 0.0000000], [0.0331941, 0.9668058, 0.0000000],
                            [0.0170827, 0.0723974, 0.9105199]], np.float64)


def oklch_to_oklab(L, C, h_deg):
    h = np.radians(h_deg)
    return np.stack([np.asarray(L, np.float64) + 0 * h, C * np.cos(h), C * np.sin(h)], -1)


def oklab_to_linear_srgb(lab):
    lms = (np.asarray(lab, np.float64) @ _M2i.T.astype(np.float64)) ** 3
    return lms @ _M1i.T.astype(np.float64)


def srgb_to_oklch(rgb):
    lab = np.asarray(srgb_to_oklab(np.asarray(rgb, np.float32)), np.float64)
    return lab[..., 0], np.hypot(lab[..., 1], lab[..., 2]), np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360


def _lin_srgb_to(gamut):
    if gamut == "p3":
        return _LIN_SRGB_TO_P3
    from ..color import spaces as S
    key = {"displayp3": "displayp3", "adobergb": "adobergb", "prophoto": "prophoto"}.get(gamut)
    return None if key is None else S.conversion_matrix(S.SRGB, S.SPACES[key]).astype(np.float64)


def in_gamut(L, C, h_deg, gamut: str = "srgb", eps: float = 1e-4):
    """True where an OKLCH colour fits in sRGB, Display P3 ("p3" / "displayp3"), Adobe RGB or ProPhoto."""
    lin = oklab_to_linear_srgb(oklch_to_oklab(L, C, h_deg))
    M = _lin_srgb_to(gamut)
    if M is not None:
        lin = lin @ M.T
    return np.all((lin >= -eps) & (lin <= 1 + eps), axis=-1)


def max_chroma(L, h_deg, gamut: str = "srgb", iters: int = 18):
    """The largest chroma at this lightness and hue that still fits the gamut (bisection)."""
    lo = np.zeros(np.broadcast(np.asarray(L), np.asarray(h_deg)).shape)
    hi = np.full_like(lo, 0.9)
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = in_gamut(L, mid, h_deg, gamut)
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    return lo


def oklch_to_space(L, C, h_deg, space: str):
    """An OKLCH colour as stored values of a working space, chroma lowered to fit that space (linear: sRGB's
    primaries, so chroma is limited to sRGB's, but values above 1 are not clipped)."""
    if space in ("srgb", "linear"):
        if space == "srgb":
            return oklch_to_srgb(L, C, h_deg)
        C = np.minimum(C, max_chroma(L, h_deg, "srgb"))
        return tuple(float(v) for v in np.clip(oklab_to_linear_srgb(oklch_to_oklab(L, C, h_deg)), 0, None))
    C = np.minimum(C, max_chroma(L, h_deg, space))
    lin = oklab_to_linear_srgb(oklch_to_oklab(L, C, h_deg)) @ _lin_srgb_to(space).T
    from ..color import spaces as S
    return tuple(float(v) for v in np.clip(S.SPACES[space].encode(np.clip(lin, 0, 1)), 0, 1))


def oklch_to_srgb(L, C, h_deg):
    """An OKLCH colour as canvas sRGB; outside sRGB the chroma is lowered to fit (same lightness and hue)."""
    C = np.minimum(C, max_chroma(L, h_deg, "srgb"))
    return tuple(float(v) for v in np.clip(linear_to_srgb(oklab_to_linear_srgb(oklch_to_oklab(L, C, h_deg))), 0, 1))


# ---------------------------------------------------------------- the triangle picker's maths (as Krita / MyPaint)
def triangle_corners(cx, cy, r, hue):
    """The HSV triangle inside a hue ring of radius r: (pure hue, white, black) corners, pure hue pointing at the
    hue's angle on the ring."""
    import math
    a = hue * 2 * math.pi
    pts = []
    for k in range(3):
        t = a + k * 2 * math.pi / 3
        pts.append((cx + r * math.cos(t), cy - r * math.sin(t)))
    return pts


def triangle_to_sv(x, y, corners):
    """Point in the triangle -> (saturation, value), clamped to the triangle (barycentric)."""
    (hx, hy), (wx, wy), (bx, by) = corners
    d = (wy - by) * (hx - bx) + (bx - wx) * (hy - by)
    a = ((wy - by) * (x - bx) + (bx - wx) * (y - by)) / d          # weight of the pure hue
    b = ((by - hy) * (x - bx) + (hx - bx) * (y - by)) / d          # weight of white
    a, b = max(0.0, a), max(0.0, b)
    if a + b > 1:
        a, b = a / (a + b), b / (a + b)
    v = a + b                                                       # black weight = 1 - v
    s = a / v if v > 1e-6 else 0.0
    return min(1.0, s), min(1.0, v)


def sv_to_triangle(s, v, corners):
    (hx, hy), (wx, wy), (bx, by) = corners
    a, b = s * v, v * (1 - s)
    c = 1 - a - b
    return a * hx + b * wx + c * bx, a * hy + b * wy + c * by


# ---------------------------------------------------------------- pigment mixing (Kubelka-Munk, from spectral.js)
_SPEC = None


def _spec():
    global _SPEC
    if _SPEC is None:
        from . import spectral_data as sd
        _SPEC = (np.array([sd.BASE_SPECTRA[k] for k in "WCMYRGB"], np.float64),
                 np.array(sd.CMF, np.float64), np.array(sd.XYZ_RGB, np.float64))
    return _SPEC


def srgb_to_reflectance(rgb):
    """Straight sRGB (..., 3) -> reflectance spectra (..., 38): white, then the secondary and primary parts."""
    base = _spec()[0]
    lin = srgb_to_linear(np.clip(rgb, 0, 1)).astype(np.float64)
    w = lin.min(-1, keepdims=True)
    l = lin - w
    r, g, b = l[..., 0], l[..., 1], l[..., 2]
    parts = np.stack([w[..., 0], np.minimum(g, b), np.minimum(r, b), np.minimum(r, g),
                      np.maximum(0, np.minimum(r - b, r - g)), np.maximum(0, np.minimum(g - b, g - r)),
                      np.maximum(0, np.minimum(b - g, b - r))], -1)
    return np.maximum(np.finfo(np.float64).eps, parts @ base)


def reflectance_to_srgb(R):
    _, cmf, xyz_rgb = _spec()
    lin = (R @ cmf.T) @ xyz_rgb.T
    return np.clip(linear_to_srgb(np.clip(lin, 0, None).astype(np.float32)), 0, 1)


_PIG_CACHE: dict = {}


def _pig_parts(rgb):
    """K/S spectra (N, 38) and luminance (N,) of straight sRGB colours (N, 3), float32."""
    cmf = _spec()[1]
    R = srgb_to_reflectance(rgb)
    ks = ((1 - R) ** 2 / (2 * R)).astype(np.float32)
    y = np.maximum(1e-12, R @ cmf[1]).astype(np.float32)
    return ks, y


def _pig_matrix():
    if "M" not in _PIG_CACHE:
        _, cmf, xyz_rgb = _spec()
        _PIG_CACHE["M"] = (xyz_rgb @ cmf).T.astype(np.float32)      # (38, 3): reflectance -> linear sRGB
    return _PIG_CACHE["M"]


def pigment_mix(a, b, t):
    """Straight sRGB ``a`` and ``b`` mixed as paint by ``t`` (0 = a, 1 = b): Kubelka-Munk as spectral.js does it.
    Fast path for painting: only pixels with 0 < t < 1 are computed, a single brush colour's spectrum is cached,
    and each distinct canvas colour is turned into a spectrum once."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    shape = np.broadcast_shapes(a.shape, b.shape)
    t = np.asarray(t, np.float32)
    tt = np.broadcast_to(t[..., 0] if (t.ndim and t.shape[-1] == 1) else t, shape[:-1]).reshape(-1)
    A = np.broadcast_to(a, shape).reshape(-1, 3)
    B = b.reshape(1, 3) if b.ndim == 1 else np.broadcast_to(b, shape).reshape(-1, 3)
    out = np.array(A, np.float32)
    full = tt >= 1
    if full.any():
        out[full] = B[full] if B.shape[0] == out.shape[0] else b
    idx = np.nonzero((tt > 0) & (tt < 1))[0]
    if not len(idx):
        return out.reshape(shape)
    Ai, ti = A[idx], tt[idx]
    Bi = b.reshape(1, 3) if b.ndim == 1 else B[idx]
    q = np.round(Ai * 1023).astype(np.int64)                         # each distinct canvas colour once
    key = (q[:, 0] << 20) | (q[:, 1] << 10) | q[:, 2]
    uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    ksa_u, ya_u = _pig_parts(Ai[first])
    one_brush = Bi.shape[0] == 1 or bool((Bi == Bi[0]).all())
    if one_brush:                                          # one brush colour: cached
        k = tuple(np.round(Bi[0], 5))
        hit = _PIG_CACHE.get(k)
        if hit is None:
            hit = _pig_parts(Bi[:1])
            if len(_PIG_CACHE) > 256:
                m = _PIG_CACHE.pop("M", None)
                _PIG_CACHE.clear()
                if m is not None:
                    _PIG_CACHE["M"] = m
            _PIG_CACHE[k] = hit
        ksb, yb = hit[0], hit[1]
        # the mix depends only on (canvas colour, paint amount t): each distinct pair, t in 1/2048 steps, is
        # computed once. (Not the K/S weight: strong dark pigments act at tiny weights, so rounding the weight
        # would lose a light touch of them; the mix is smooth in t.)
        tq = np.round(ti * 2048).astype(np.int64)
        pk, pfirst, pinv = np.unique(inv.astype(np.int64) * 2049 + tq, return_index=True, return_inverse=True)
        cu = inv[pfirst]
        tu = (tq[pfirst] / 2048.0).astype(np.float32)
        ca, cb = (1 - tu) ** 2 * ya_u[cu], tu ** 2 * yb[0]           # concentration: amount squared x luminance
        wu = (cb / np.maximum(ca + cb, 1e-12))[:, None]
        ks = ksa_u[cu] + wu * (ksb - ksa_u[cu])
        R = 1 + ks - np.sqrt(ks * ks + 2 * ks)                        # K/S back to reflectance
        lin = R @ _pig_matrix()
        out[idx] = np.clip(linear_to_srgb(np.clip(lin, 0, None)), 0, 1)[pinv]
        return out.reshape(shape)
    ksb, yb = _pig_parts(Bi)
    ksa, ya = ksa_u[inv], ya_u[inv]
    ca, cb = (1 - ti) ** 2 * ya, ti ** 2 * yb
    w = (cb / np.maximum(ca + cb, 1e-12))[:, None]
    ks = ksa + w * (ksb - ksa)
    R = 1 + ks - np.sqrt(ks * ks + 2 * ks)
    lin = R @ _pig_matrix()
    out[idx] = np.clip(linear_to_srgb(np.clip(lin, 0, None)), 0, 1)
    return out.reshape(shape)
