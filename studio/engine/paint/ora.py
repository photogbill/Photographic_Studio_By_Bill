"""OpenRaster (.ora): the open layered format Krita, GIMP and MyPaint all read (BUILD_PLAN_PAINT P1).

An .ora file is a zip: ``mimetype`` (stored first, uncompressed), ``stack.xml`` (the layers, top first),
``data/*.png`` (each layer, cropped to its painted area, with an x / y offset), ``mergedimage.png`` and
``Thumbnails/thumbnail.png``. Groups are nested ``<stack>`` elements (the OpenRaster way; Krita, GIMP and MyPaint
read and write them). A pass-through group is written ``isolation="auto"`` with Normal and full opacity (what the
spec reads as not isolated) plus our own ``studio-pass-through``, so a faded pass-through group survives a round trip
through our own files; an isolated group is ``isolation="isolate"``. Layers are 8-bit PNGs (what every reader understands). The paper colour is kept as a
``studio-background`` attribute that other programs ignore.

A painting is the artist's own file: it is saved where she chooses, under the name she gives (plan 0.4.7). Saving
writes a ``.partial`` file first and swaps it in, so a crash mid-save never leaves half a painting.
"""
from __future__ import annotations

import io
import os
import struct
import zipfile
import zlib
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


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def _to_png16(px_premult: np.ndarray) -> bytes:
    """16-bit RGBA PNG (straight alpha), each row filtered 'Up' (vectorised, and compresses smooth paint well)."""
    a = px_premult[..., 3:4]
    rgb = np.where(a > 0, px_premult[..., :3] / np.maximum(a, 1e-8), 0)
    v = np.clip(np.rint(np.concatenate([rgb, a], -1) * 65535), 0, 65535).astype(">u2")
    h, w = v.shape[:2]
    rows = np.ascontiguousarray(v).reshape(h, -1).view(np.uint8)
    up = rows.copy()
    up[1:] = rows[1:] - rows[:-1]                       # uint8 arithmetic wraps, as the PNG filter does
    raw = np.concatenate([np.full((h, 1), 2, np.uint8), up], axis=1)
    raw[0, 0] = 0                                       # first row: no filter
    raw[0, 1:] = rows[0]
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 6, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + _chunk(b"IEND", b""))


def _decode_png16(data: bytes):
    """Our own decoder for 16-bit PNGs: RGBA / RGB / grey (+alpha), filters None, Sub and Up (what our writer and
    most others use for smooth images). Returns premultiplied float RGBA, or None for anything else (then Pillow
    reads it as 8-bit)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos, idat, hdr = 8, [], None
    while pos + 8 <= len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
        pos += 12 + n
    if hdr is None:
        return None
    w, h, depth, ctype, _, _, interlace = hdr
    chans = {0: 1, 2: 3, 4: 2, 6: 4}.get(ctype)
    if depth != 16 or chans is None or interlace:
        return None
    bpp = 2 * chans
    stride = w * bpp
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8)
    if raw.size != h * (stride + 1):
        return None
    raw = raw.reshape(h, stride + 1)
    filt, rows = raw[:, 0], raw[:, 1:].copy()
    if not set(np.unique(filt).tolist()) <= {0, 1, 2}:
        return None
    for y in range(h):
        f = filt[y]
        if f == 1:                                       # Sub: running sum along the row, per byte position
            rows[y] = np.cumsum(rows[y].reshape(w, bpp).astype(np.uint64), axis=0).astype(np.uint8).reshape(-1)
        elif f == 2 and y > 0:                           # Up
            rows[y] = rows[y] + rows[y - 1]
    v = rows.view(">u2").reshape(h, w, chans).astype(np.float32) / 65535.0
    if chans == 1:
        v = np.concatenate([v, v, v, np.ones_like(v)], -1)
    elif chans == 2:
        v = np.concatenate([v[..., :1]] * 3 + [v[..., 1:]], -1)
    elif chans == 3:
        v = np.concatenate([v, np.ones_like(v[..., :1])], -1)
    v[..., :3] *= v[..., 3:4]
    return v


def _png_bits(data: bytes) -> int:
    return data[24] if len(data) > 25 and data[:8] == b"\x89PNG\r\n\x1a\n" else 8


def _from_png(data: bytes) -> np.ndarray:
    if _png_bits(data) == 16:
        v = _decode_png16(data)
        if v is not None:
            return v
    im = Image.open(io.BytesIO(data)).convert("RGBA")
    f = np.asarray(im, np.float32) / 255.0
    f[..., :3] *= f[..., 3:4]
    return f


def save(doc: Document, path: str, bits: int | None = None) -> str:
    """Write ``doc`` as OpenRaster to ``path`` with layers of ``bits`` (default: the painting's own). Returns the
    path."""
    bits = doc.bits if bits is None else bits
    enc = _to_png16 if bits == 16 else _to_png
    root = ET.Element("image", {"version": "0.0.6", "w": str(doc.width), "h": str(doc.height),
                                "xres": "72", "yres": "72"})
    if doc.background is not None:
        root.set("studio-background", ",".join(f"{c:.6f}" for c in doc.background))
    if doc.space != "srgb":
        root.set("studio-space", doc.space)
    stack = ET.SubElement(root, "stack")
    files = []
    kids = doc._kids()

    def put(parent_el, i):
        lay = doc.layers[i]
        common = {"name": lay.name, "opacity": f"{lay.opacity:.4f}",
                  "visibility": "visible" if lay.visible else "hidden",
                  "edit-locked": "true" if lay.locked else "false",
                  "selected": "true" if i == doc.active else "false"}
        if lay.is_group:
            passing = lay.blend == "pass"
            el = ET.SubElement(parent_el, "stack", {
                **common, "composite-op": "svg:src-over" if passing else OPS.get(lay.blend, "svg:src-over"),
                "isolation": "auto" if passing else "isolate"})
            if passing:
                el.set("studio-pass-through", "true")
            if lay.collapsed:
                el.set("studio-collapsed", "true")
            for j in reversed(kids.get(lay.uid, [])):         # top first
                put(el, j)
            return
        if lay.is_generated:                            # fill / file layers: their pixels, for other programs
            full = lay.generated(0, 0, doc.width, doc.height, doc.width, doc.height)
            tmp = Layer("x")
            tmp.write(0, 0, full)
            x0, y0, px = _cropped(tmp)
        else:
            x0, y0, px = _cropped(lay)
        src = f"data/layer{i:03d}.png"
        files.append((src, enc(px)))
        el = ET.SubElement(parent_el, "layer", {
            **common, "src": src, "x": str(x0), "y": str(y0), "composite-op": OPS.get(lay.blend, "svg:src-over"),
            "alpha-preserve": "true" if lay.alpha_lock else "false"})
        if lay.medium:
            el.set("studio-medium", lay.medium)
        if lay.clip:
            el.set("studio-clip", "true")
        if lay.kind in ("adjust", "filter", "fill", "file") and lay.adj:
            import json
            el.set("studio-adjust", json.dumps(lay.adj))
            el.set("studio-kind", lay.kind)
        if lay.mask is not None:                           # layer mask: a 16-bit grey picture over the canvas
            mv = lay.mask_values(0, 0, doc.width, doc.height)
            mpx = np.concatenate([np.repeat(mv[..., None], 3, -1), np.ones_like(mv[..., None])], -1)
            msrc = f"data/mask{i:03d}.png"
            files.append((msrc, _to_png16(mpx)))
            el.set("studio-mask", msrc)
        if lay.height:                                     # impasto: thickness 0..4 as a 16-bit grey picture
            ks = list(lay.height)
            hx0, hy0 = min(k[0] for k in ks) * TILE, min(k[1] for k in ks) * TILE
            hx1, hy1 = (max(k[0] for k in ks) + 1) * TILE, (max(k[1] for k in ks) + 1) * TILE
            from .tiles import read_tiles
            hv = read_tiles(lay.height, hx0, hy0, hx1 - hx0, hy1 - hy0, 1)[..., 0] / 4.0
            hpx = np.concatenate([np.repeat(hv[..., None], 3, -1), np.ones_like(hv[..., None])], -1)
            hsrc = f"data/height{i:03d}.png"
            files.append((hsrc, _to_png16(hpx)))
            el.set("studio-height", hsrc)
            el.set("studio-height-x", str(hx0))
            el.set("studio-height-y", str(hy0))

    for i in reversed(kids.get(None, [])):                    # stack.xml lists the top layer first
        put(stack, i)
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


def _cropped(lay: Layer):
    b = lay.bounds()
    if b is None:
        return 0, 0, np.zeros((1, 1, 4), np.float32)
    x0, y0, x1, y1 = b
    px = lay.read(x0, y0, x1 - x0, y1 - y0)
    ys, xs = np.nonzero(px[..., 3] > 0)
    if not len(xs):
        return 0, 0, np.zeros((1, 1, 4), np.float32)
    return x0 + int(xs.min()), y0 + int(ys.min()), px[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def load(path: str) -> Document:
    """Open an OpenRaster file (ours, Krita's, GIMP's or MyPaint's), groups included; unknown attributes are
    ignored."""
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("stack.xml"))
        w, h = int(root.get("w")), int(root.get("h"))
        bg = root.get("studio-background")
        background = tuple(float(v) for v in bg.split(",")) if bg else None
        layers: list = []                               # built bottom first, each group after its contents
        depth = [8]
        top = root.find("stack")

        def common(el, lay):
            lay.opacity = float(el.get("opacity", "1"))
            lay.visible = el.get("visibility", "visible") != "hidden"
            lay.locked = el.get("edit-locked", "false") == "true"
            lay.selected = el.get("selected") == "true"

        def walk(stack_el, parent):
            for el in reversed(list(stack_el)):          # document order is top first
                if el.tag == "stack":
                    g = Layer(el.get("name", "Group"), kind="group", parent=parent)
                    common(el, g)
                    op = el.get("composite-op", "svg:src-over")
                    iso = el.get("isolation", "auto")
                    if el.get("studio-pass-through") == "true" or (
                            iso != "isolate" and op == "svg:src-over" and g.opacity >= 0.999):
                        g.blend = "pass"
                    else:
                        g.blend = OPS_BACK.get(op, "normal")
                    g.collapsed = el.get("studio-collapsed") == "true"
                    walk(el, g.uid)
                    layers.append(g)
                elif el.tag == "layer":
                    lay = Layer(el.get("name", "Layer"), parent=parent)
                    common(el, lay)
                    lay.blend = OPS_BACK.get(el.get("composite-op", "svg:src-over"), "normal")
                    lay.alpha_lock = el.get("alpha-preserve", "false") == "true"
                    lay.medium = el.get("studio-medium", "")
                    lay.clip = el.get("studio-clip") == "true"
                    if el.get("studio-adjust"):
                        import json
                        lay.kind, lay.adj = el.get("studio-kind", "adjust"), json.loads(el.get("studio-adjust"))
                        if lay.is_generated or lay.is_adjust:
                            lay.tiles = {}                     # made again from the setting, not the stored pixels
                    ms = el.get("studio-mask")
                    if ms and ms in z.namelist():
                        mv = _from_png(z.read(ms))[..., 0]
                        lay.mask = {}
                        for ty in range(0, mv.shape[0], TILE):
                            for tx in range(0, mv.shape[1], TILE):
                                t = np.ones((TILE, TILE), np.float32)
                                sub = mv[ty:ty + TILE, tx:tx + TILE]
                                t[:sub.shape[0], :sub.shape[1]] = sub
                                if (t < 0.9999).any():
                                    lay.mask[(tx // TILE, ty // TILE)] = t
                    hs = el.get("studio-height")
                    if hs and hs in z.namelist():
                        hv = _from_png(z.read(hs))[..., 0] * 4.0
                        hx0, hy0 = int(el.get("studio-height-x", 0)), int(el.get("studio-height-y", 0))
                        for ty in range(0, hv.shape[0], TILE):
                            for tx in range(0, hv.shape[1], TILE):
                                t = np.zeros((TILE, TILE), np.float32)
                                sub = hv[ty:ty + TILE, tx:tx + TILE]
                                t[:sub.shape[0], :sub.shape[1]] = sub
                                if t.any():
                                    lay.height[((hx0 + tx) // TILE, (hy0 + ty) // TILE)] = t
                    src = el.get("src")
                    if src and src in z.namelist():
                        data = z.read(src)
                        depth[0] = max(depth[0], _png_bits(data))
                        px = _from_png(data)
                        if px[..., 3].any():
                            lay.write(int(float(el.get("x", "0"))), int(float(el.get("y", "0"))), px)
                    layers.append(lay)
        if top is not None:
            walk(top, None)
    if not any(not l.is_group for l in layers):
        layers.append(Layer("Layer 1"))
    if not layers:
        layers = [Layer("Layer 1")]
    active = next((i for i, l in enumerate(layers) if getattr(l, "selected", False)), len(layers) - 1)
    if background is None:
        background = (1.0, 1.0, 1.0) if not _opaque_bottom(layers[0], w, h) else None
    doc = Document(w, h, background, layers, active, path)
    doc.bits = 16 if depth[0] == 16 else 8
    doc.space = root.get("studio-space", "srgb")
    doc.check()
    return doc


def _opaque_bottom(lay: Layer, w: int, h: int) -> bool:
    """A file from another program whose bottom layer already covers the canvas needs no paper under it."""
    return len(lay.tiles) >= ((w + TILE - 1) // TILE) * ((h + TILE - 1) // TILE) and all(
        t[..., 3].min() > 0.99 for t in lay.tiles.values())
