"""Fill and gradient (BUILD_PLAN_PAINT P1).

Fill: the region around a point whose colour is within a tolerance of the clicked colour, read from the whole
picture, the active layer, or a chosen reference layer (fill flats on one layer while reading the line art on
another). **Gap closing** keeps the fill inside line art that has small breaks: the lines are thickened by the gap
size before the region is found, then the region grows back up to the lines (the same idea as Krita's and Clip
Studio's "close gap"). **Grow** pushes the fill a pixel or two under the lines' soft edges, so no pale halo shows.

Gradient: linear or radial, from the painting colour to the second colour or to see-through, mixed in one of the
blend spaces (colour.py): Classic, Light or Oklab.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .colour import mix
from .tiles import Document, Layer

FILL_SOURCES = {
    "picture": ("The whole picture", "filling what you see, across all layers", "fills stop at paint on any layer"),
    "layer": ("This layer only", "filling inside shapes painted on this layer", "ignores line art on other layers"),
    "reference": ("A reference layer", "flats on their own layer while the line art sits on another",
                  "pick the line-art layer as the reference first"),
}

GAP_NOTE = ("Gap closing", "line art with small breaks in it: the fill stays inside",
            "very thin spaces narrower than the gap may not be filled; lower it for fine detail")


def region(src: np.ndarray, x: int, y: int, tolerance: float = 0.1, gap: int = 0, grow: int = 1) -> np.ndarray:
    """The fill region (bool, src's size) around (x, y) in ``src`` (premultiplied RGBA). Empty if (x, y) is outside."""
    h, w = src.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), bool)
    a = src[..., 3:4]
    straight = np.concatenate([np.where(a > 0, src[..., :3] / np.maximum(a, 1e-8), 0), a], -1)
    seed = straight[y, x]
    similar = np.abs(straight - seed).max(axis=-1) <= tolerance + 1e-6
    if gap > 0:
        barrier = ndimage.binary_dilation(~similar, iterations=int(gap))
        open_ = similar & ~barrier
        if open_[y, x]:
            lab, _ = ndimage.label(open_)
            core = lab == lab[y, x]
            # grow back up to the lines, but only through similar pixels, and no further than the gap
            reg = ndimage.binary_dilation(core, iterations=int(gap) + 1, mask=similar)
        else:                                   # the click landed in a gap-sized space: fill it without closing
            lab, _ = ndimage.label(similar)
            reg = lab == lab[y, x]
    else:
        lab, _ = ndimage.label(similar)
        reg = lab == lab[y, x]
    if grow > 0:
        reg = ndimage.binary_dilation(reg, iterations=int(grow))
    return reg


def source_pixels(doc: Document, source: str = "picture", reference: int | None = None) -> np.ndarray:
    if source == "layer":
        return doc.layer.read(0, 0, doc.width, doc.height)
    if source == "reference" and reference is not None and 0 <= reference < len(doc.layers):
        return doc.layers[reference].read(0, 0, doc.width, doc.height)
    return doc.composite(background=True)


def fill(doc: Document, x: float, y: float, colour, tolerance: float = 0.1, gap: int = 0, grow: int = 1,
         source: str = "picture", reference: int | None = None, opacity: float = 1.0, undo=None):
    """Fill on the active layer. Returns the changed rectangle (x0, y0, x1, y1) or None."""
    lay = doc.paint_layer()
    reg = region(source_pixels(doc, source, reference), int(np.floor(x)), int(np.floor(y)), tolerance, gap, grow)
    if not reg.any():
        return None
    ys, xs = np.nonzero(reg)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    m = reg[y0:y1, x0:x1, None].astype(np.float32) * opacity
    base = lay.read(x0, y0, x1 - x0, y1 - y0)
    src = np.concatenate([np.asarray(colour[:3], np.float32) * m, m], -1)
    out = src + base * (1 - m)
    if lay.alpha_lock:
        ba = base[..., 3:4]
        out = np.concatenate([out[..., :3] / np.maximum(out[..., 3:4], 1e-8) * ba, ba], -1)
    if undo is not None:
        undo.begin("Fill")
    lay.write(x0, y0, out.astype(np.float32), on_touch=undo.touch if undo is not None else None)
    if undo is not None:
        undo.end()
    return x0, y0, x1, y1


GRADIENT_KINDS = {
    "linear": ("Linear", "skies, light falling across a wall, backgrounds", "straight bands only"),
    "radial": ("Radial", "glows, spotlights, vignettes", "always round"),
}


def gradient_pixels(w: int, h: int, p0, p1, c0, c1, kind: str = "linear", space: str = "oklab") -> np.ndarray:
    """A gradient (premultiplied RGBA, h x w). ``c0`` / ``c1`` are (r, g, b, a) straight."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if kind == "radial":
        r = max(1e-6, float(np.hypot(dx, dy)))
        t = np.hypot(xx - p0[0], yy - p0[1]) / r
    else:
        L2 = max(1e-6, dx * dx + dy * dy)
        t = ((xx - p0[0]) * dx + (yy - p0[1]) * dy) / L2
    t = np.clip(t, 0, 1)[..., None]
    c0, c1 = np.asarray(c0, np.float32), np.asarray(c1, np.float32)
    a = c0[3] + (c1[3] - c0[3]) * t
    # when one end is see-through its colour is the other end's (no grey fringe towards transparent)
    rgb0 = c0[:3] if c0[3] > 0 else c1[:3]
    rgb1 = c1[:3] if c1[3] > 0 else c0[:3]
    rgb = mix(np.broadcast_to(rgb0, (h, w, 3)), np.broadcast_to(rgb1, (h, w, 3)), t, space)
    return np.concatenate([rgb * a, a], -1).astype(np.float32)


def gradient(doc: Document, p0, p1, c0, c1, kind: str = "linear", space: str = "oklab", undo=None):
    """Lay a gradient over the whole active layer (painted on top of what is there)."""
    lay: Layer = doc.paint_layer()
    g = gradient_pixels(doc.width, doc.height, p0, p1, c0, c1, kind, space)
    base = lay.read(0, 0, doc.width, doc.height)
    out = g + base * (1 - g[..., 3:4])
    if lay.alpha_lock:
        ba = base[..., 3:4]
        out = np.concatenate([out[..., :3] / np.maximum(out[..., 3:4], 1e-8) * ba, ba], -1)
    if undo is not None:
        undo.begin("Gradient")
    lay.write(0, 0, out.astype(np.float32), on_touch=undo.touch if undo is not None else None)
    if undo is not None:
        undo.end()
    return 0, 0, doc.width, doc.height
