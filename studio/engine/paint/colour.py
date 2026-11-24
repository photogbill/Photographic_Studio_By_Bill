"""Colour helpers for Paint: HSV and hex, the colour picker (Alt + click), and the blend spaces paint mixes in.

Blend spaces (BUILD_PLAN_PAINT 0.5, 0.9), each saying what it is best for and weak at:
- classic  mixes the stored (gamma-encoded) values, as Photoshop, Krita and Clip Studio do: what painters expect
- light    mixes in linear light: glows and coloured light mix like real light, edges of soft brushes look brighter
- oklab    mixes in Oklab (Bjorn Ottosson, 2020): even lightness, so blends of two hues do not dip into grey mud
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
    return np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t


def over(base: np.ndarray, colour, a: np.ndarray, space: str = "classic") -> np.ndarray:
    """Paint of ``colour`` (straight sRGB) at alpha ``a`` over ``base`` (premultiplied RGBA), mixed in a blend space."""
    colour = np.asarray(colour, np.float32)
    if space == "classic":
        return np.concatenate([colour * a, a], -1) + base * (1 - a)
    ba = base[..., 3:4]
    oa = a + ba * (1 - a)
    bc = base[..., :3] / np.maximum(ba, 1e-8)
    t = a / np.maximum(oa, 1e-8)                      # the paint's share of the visible colour
    c = mix(bc, np.broadcast_to(colour, bc.shape), t, space)
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
