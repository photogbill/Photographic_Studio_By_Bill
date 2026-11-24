"""Selections for Paint (BUILD_PLAN_PAINT P4): a soft mask the size of the canvas (0 = outside, 1 = inside), so
feathered edges and painted selections are the same thing. Every way of making one, and every way of changing it,
each saying what it is best for and weak at.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .colour import srgb_to_oklab

MODES = {"replace": "Replace", "add": "Add (Shift)", "subtract": "Subtract (Alt)", "intersect": "Intersect"}

TOOLS = {
    "rect": ("Rectangle", "boxes, panels and straight crops", "anything round or irregular"),
    "ellipse": ("Ellipse", "round shapes: eyes, plates, moons", "irregular shapes"),
    "lasso": ("Lasso", "free-hand outlines around anything", "precise straight edges"),
    "polygon": ("Polygon", "straight-edged shapes, click by click (double-click closes)", "curves"),
    "wand": ("Magic wand", "areas of one colour, like a flat-coloured shape", "soft gradients (it stops early)"),
    "colour": ("Colour range", "every place in the picture with a similar colour", "colours that also appear "
               "where you do not want them"),
    "subject": ("Subject (click a thing)", "a figure, an object, an animal: click it and the Studio's SAM model "
                "outlines it", "fine hair and see-through things; it needs the SAM model (it comes with the Studio)"),
}


class SelectProblem(ValueError):
    """Said to the person in plain words (not a fault)."""


_SAM_CACHE: dict = {}


def subject(doc, x, y, key=None):
    """Click-to-select with the Studio's SAM 2.1 (local/sam.py), on the visible picture. The picture's embedding is
    kept for ``key`` (for example the undo position), so further clicks on an unchanged picture are quick."""
    from ..local import sam
    size = sam.choose(None)
    if size is None:
        raise SelectProblem("Subject select needs the SAM model in the Studio's models folder (the Studio comes with "
                            "SAM 2.1 Tiny; Convert Models.bat makes Large).")
    px = doc.composite(background=True)
    a = px[..., 3:4]
    rgb = np.clip(np.where(a > 0, px[..., :3] / np.maximum(a, 1e-8), 1.0), 0, 1).astype(np.float32)
    hit = _SAM_CACHE.get("feats") if key is not None and _SAM_CACHE.get("key") == (key, size) else None
    if hit is None:
        hit, _ = sam.encode(rgb, size)
        _SAM_CACHE.update(key=(key, size), feats=hit)
    logits, _score = sam.decode(hit, size, [(x / doc.width, y / doc.height)])
    return (sam.bitmap(logits, doc.height, doc.width).astype(np.float32) / 255.0)


def combine(old, new, mode="replace"):
    if old is None or mode == "replace":
        return new
    if mode == "add":
        return np.maximum(old, new)
    if mode == "subtract":
        return np.clip(old - new, 0, 1)
    if mode == "intersect":
        return np.minimum(old, new)
    raise ValueError(mode)


def rect(w, h, x0, y0, x1, y1):
    m = np.zeros((h, w), np.float32)
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1
    return m


def polygon(w, h, pts, k: int = 4):
    """Lasso / polygon / ellipse outline filled, edges anti-aliased (k x k supersampling)."""
    from PIL import Image, ImageDraw
    if len(pts) < 3:
        return np.zeros((h, w), np.float32)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0 = max(0, int(np.floor(min(xs))) - 1), max(0, int(np.floor(min(ys))) - 1)
    x1, y1 = min(w, int(np.ceil(max(xs))) + 2), min(h, int(np.ceil(max(ys))) + 2)
    m = np.zeros((h, w), np.float32)
    if x1 <= x0 or y1 <= y0:
        return m
    im = Image.new("L", ((x1 - x0) * k, (y1 - y0) * k), 0)
    ImageDraw.Draw(im).polygon([((x - x0) * k, (y - y0) * k) for x, y in pts], fill=255)
    m[y0:y1, x0:x1] = np.asarray(im, np.float32).reshape(y1 - y0, k, x1 - x0, k).mean(axis=(1, 3)) / 255
    return m


def ellipse(w, h, x0, y0, x1, y1):
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2
    t = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    return polygon(w, h, list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t))))


def wand(doc, x, y, tolerance=0.1, source="picture"):
    from .fill import region, source_pixels
    return region(source_pixels(doc, source, None), int(x), int(y), tolerance, 0, 0).astype(np.float32)


def colour_range(doc, rgb, fuzziness=0.08):
    """Soft: full inside the fuzziness in Oklab, fading out over the same distance again."""
    px = doc.composite()
    a = px[..., 3:4]
    straight = np.where(a > 0, px[..., :3] / np.maximum(a, 1e-8), 0)
    d = np.linalg.norm(srgb_to_oklab(straight) - srgb_to_oklab(np.asarray(rgb, np.float32)), axis=-1)
    return np.clip(2.0 - d / max(fuzziness, 1e-4), 0, 1).astype(np.float32)


def from_layer(doc, lay):
    return lay.read(0, 0, doc.width, doc.height)[..., 3].copy()


def feather(m, radius):
    return np.clip(ndimage.gaussian_filter(m, max(0.1, radius / 2)), 0, 1).astype(np.float32)


def grow(m, px):
    if px <= 0:
        return m
    d = ndimage.distance_transform_edt(m < 0.5)
    return np.maximum(m, np.clip(px + 0.5 - d, 0, 1)).astype(np.float32)


def shrink(m, px):
    return (1 - grow(1 - m, px)).astype(np.float32)


def invert(m):
    return (1 - m).astype(np.float32)


def bounds(m, thr=0.004):
    ys, xs = np.nonzero(m > thr)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
