"""OpenRaster (.ora): the open layered format Krita, GIMP and MyPaint all read (BUILD_PLAN_PAINT P1).

An .ora file is a zip: ``mimetype`` (stored first, uncompressed), ``stack.xml`` (the layers, top first),
``data/*.png`` (each layer, cropped to its painted area, with an x / y offset), ``mergedimage.png`` and
``Thumbnails/thumbnail.png``. Layers are 8-bit PNGs (what every reader understands). The paper colour is kept as a
``studio-background`` attribute that other programs ignore.

A painting is the artist's own file: it is saved where she chooses, under the name she gives (plan 0.4.7). Saving
writes a ``.partial`` file first and swaps it in, so a crash mid-save never leaves half a painting.
"""
from __future__ import annotations

import io
import os
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .tiles import TILE, Document, Layer

OPS = {"normal": "svg:src-over", "multiply": "svg:multiply", "screen": "svg:screen", "overlay": "svg:overlay",
       "add": "svg:plus", "darken": "svg:darken", "lighten": "svg:lighten"}
OPS_BACK = {v: k for k, v in OPS.items()}


def _to_png(px_premult: np.ndarray) -> bytes:
    a = px_premult[..., 3:4]
    rgb = np.where(a > 0, px_premult[..., :3] / np.maximum(a, 1e-8), 0)
    rgba = np.concatenate([rgb, a], axis=-1)
    im = Image.fromarray((np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")
    buf = io.BytesIO()
    im.save(buf, "PNG", compress_level=6)
    return buf.getvalue()


def _from_png(data: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(data)).convert("RGBA")
    f = np.asarray(im, np.float32) / 255.0
    f[..., :3] *= f[..., 3:4]
    return f


def save(doc: Document, path: str) -> str:
    """Write ``doc`` as OpenRaster to ``path``. Returns the path."""
    root = ET.Element("image", {"version": "0.0.6", "w": str(doc.width), "h": str(doc.height),
                                "xres": "72", "yres": "72"})
    if doc.background is not None:
        root.set("studio-background", ",".join(f"{c:.6f}" for c in doc.background))
    stack = ET.SubElement(root, "stack")
    files = []
    for i, lay in reversed(list(enumerate(doc.layers))):      # stack.xml lists the top layer first
        b = lay.bounds()
        if b is None:
            x0 = y0 = 0
            px = np.zeros((1, 1, 4), np.float32)
        else:
            x0, y0, x1, y1 = b
            px = lay.read(x0, y0, x1 - x0, y1 - y0)
            ys, xs = np.nonzero(px[..., 3] > 0)
            if len(xs):
                px = px[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                x0, y0 = x0 + int(xs.min()), y0 + int(ys.min())
            else:
                x0 = y0 = 0
                px = np.zeros((1, 1, 4), np.float32)
        src = f"data/layer{i:03d}.png"
        files.append((src, _to_png(px)))
        ET.SubElement(stack, "layer", {
            "name": lay.name, "src": src, "x": str(x0), "y": str(y0), "opacity": f"{lay.opacity:.4f}",
            "visibility": "visible" if lay.visible else "hidden", "composite-op": OPS.get(lay.blend, "svg:src-over"),
            "edit-locked": "true" if lay.locked else "false", "alpha-preserve": "true" if lay.alpha_lock else "false",
            "selected": "true" if i == doc.active else "false"})
    merged = doc.composite()
    tmp = path + ".partial"
    with zipfile.ZipFile(tmp, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "image/openraster", compress_type=zipfile.ZIP_STORED)
        z.writestr("stack.xml", ET.tostring(root, encoding="utf-8", xml_declaration=True),
                   compress_type=zipfile.ZIP_DEFLATED)
        for name, data in files:
            z.writestr(name, data, compress_type=zipfile.ZIP_STORED)
        z.writestr("mergedimage.png", _to_png(merged), compress_type=zipfile.ZIP_STORED)
        th = Image.open(io.BytesIO(_to_png(merged)))
        th.thumbnail((256, 256))
        buf = io.BytesIO()
        th.save(buf, "PNG")
        z.writestr("Thumbnails/thumbnail.png", buf.getvalue(), compress_type=zipfile.ZIP_STORED)
    os.replace(tmp, path)
    return path


def load(path: str) -> Document:
    """Open an OpenRaster file (ours, Krita's, GIMP's or MyPaint's). Groups are flattened into their layers in order
    (groups come with P4); unknown attributes are ignored."""
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("stack.xml"))
        w, h = int(root.get("w")), int(root.get("h"))
        bg = root.get("studio-background")
        background = tuple(float(v) for v in bg.split(",")) if bg else None
        layers = []
        for el in root.iter("layer"):                   # document order: top first
            lay = Layer(el.get("name", "Layer"))
            lay.opacity = float(el.get("opacity", "1"))
            lay.visible = el.get("visibility", "visible") != "hidden"
            lay.blend = OPS_BACK.get(el.get("composite-op", "svg:src-over"), "normal")
            lay.locked = el.get("edit-locked", "false") == "true"
            lay.alpha_lock = el.get("alpha-preserve", "false") == "true"
            src = el.get("src")
            if src and src in z.namelist():
                px = _from_png(z.read(src))
                if px[..., 3].any():
                    lay.write(int(float(el.get("x", "0"))), int(float(el.get("y", "0"))), px)
            lay.selected = el.get("selected") == "true"
            layers.append(lay)
    layers.reverse()                                     # our list is bottom first
    if not layers:
        layers = [Layer("Layer 1")]
    active = next((i for i, l in enumerate(layers) if getattr(l, "selected", False)), len(layers) - 1)
    if background is None:
        background = (1.0, 1.0, 1.0) if not _opaque_bottom(layers[0], w, h) else None
    doc = Document(w, h, background, layers, active, path)
    return doc


def _opaque_bottom(lay: Layer, w: int, h: int) -> bool:
    """A file from another program whose bottom layer already covers the canvas needs no paper under it."""
    return len(lay.tiles) >= ((w + TILE - 1) // TILE) * ((h + TILE - 1) // TILE) and all(
        t[..., 3].min() > 0.99 for t in lay.tiles.values())
