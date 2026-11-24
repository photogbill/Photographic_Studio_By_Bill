"""Working colour spaces for Paint (BUILD_PLAN_PAINT P5).

A painting's pixels are stored in its working space (``Document.space``): what the brush lays down, what layers
blend in and what is saved. sRGB is the default; wider spaces hold more saturated colours (Display P3, Adobe RGB,
ProPhoto); "linear" is a floating-point, scene-linear canvas (HDR: values above 1 are kept, light adds up like
light, and EXR files keep everything). The screen shows every space correctly through the Studio's display path.

Each space says what it is best for and weak at (plan 23.1).
"""
from __future__ import annotations

import numpy as np

from ..color import spaces as S

SPACES = {
    "srgb": ("sRGB", "the web, phones and most printers; what every program assumes", "the fewest colours"),
    "displayp3": ("Display P3", "richer reds and greens on modern screens and phones", "programs that ignore colour "
                  "profiles show it dull"),
    "adobergb": ("Adobe RGB", "print: more cyans and greens that inkjet printers can reach", "screens that cannot "
                 "show it; dull in programs that ignore profiles"),
    "prophoto": ("ProPhoto RGB", "keeping every colour through heavy editing", "8-bit files band; many of its colours "
                 "no screen or printer can show"),
    "linear": ("Linear light (HDR, floating point)", "glows, fire and light that add up like light, values above "
               "white, EXR for 3D and games", "colours mix differently from what painters expect; soft brushes "
               "spread darker"),
}


def _rgbspace(space):
    return S.SPACES["srgb" if space == "linear" else space]


def to_linear(px_straight, space):
    """Stored values -> linear light in that space's primaries."""
    if space == "linear":
        return np.asarray(px_straight, np.float32)
    return _rgbspace(space).decode(np.clip(px_straight, 0, None)).astype(np.float32)


def from_linear(lin, space):
    if space == "linear":
        return np.asarray(lin, np.float32)
    return _rgbspace(space).encode(np.clip(lin, 0, None)).astype(np.float32)


def matrix(src, dst):
    return S.conversion_matrix(_rgbspace(src), _rgbspace(dst)).astype(np.float32)


def convert(rgb, src, dst):
    """Straight colours from one working space to another (out-of-gamut values are kept; callers clip)."""
    if src == dst:
        return np.asarray(rgb, np.float32)
    lin = to_linear(rgb, src)
    out = S.apply_matrix(np.ascontiguousarray(lin.reshape(-1, 1, 3)), matrix(src, dst)).reshape(np.shape(rgb))
    return from_linear(out, dst)


def from_srgb(rgb, space):
    if space == "srgb":
        return tuple(float(v) for v in rgb[:3])
    return tuple(float(v) for v in np.clip(convert(np.asarray(rgb[:3], np.float32), "srgb", space), 0, None))


def to_srgb(rgb, space):
    if space == "srgb":
        return tuple(float(v) for v in rgb[:3])
    return tuple(float(v) for v in np.clip(convert(np.asarray(rgb[:3], np.float32), space, "srgb"), 0, 1))


def to_prophoto_linear(px_straight, space):
    """For the screen and the Studio's engine: linear ProPhoto."""
    lin = to_linear(px_straight, space)
    if space == "prophoto":
        return lin
    shp = lin.shape
    return S.apply_matrix(np.ascontiguousarray(lin.reshape(-1, 1, 3)), S.conversion_matrix(
        _rgbspace(space), S.PROPHOTO).astype(np.float32)).reshape(shp)


def convert_document(doc, dst):
    """Convert every layer's pixels (and fill colours) to another working space, as one change of the document."""
    src = doc.space
    if src == dst:
        return
    for lay in doc.layers:
        for k, t in list(lay.tiles.items()):
            a = t[..., 3:4]
            straight = np.where(a > 0, t[..., :3] / np.maximum(a, 1e-8), 0)
            c = convert(straight, src, dst)
            if dst != "linear":
                c = np.clip(c, 0, 1)
            lay.tiles[k] = np.concatenate([c * a, a], -1).astype(np.float32)
        if lay.kind == "fill" and lay.adj and "colour" in lay.adj:
            lay.adj["colour"] = list(np.clip(convert(np.asarray(lay.adj["colour"], np.float32), src, dst), 0, None))
    if doc.background is not None:
        doc.background = tuple(float(v) for v in np.clip(convert(np.asarray(doc.background, np.float32), src, dst),
                                                         0, None))
    doc.space = dst
    doc.dirty = True
