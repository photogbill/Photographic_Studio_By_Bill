"""Opening Krita's .kra files (BUILD_PLAN_PAINT P5), so her old paintings come along.

A .kra is a zip: ``maindoc.xml`` describes the layers (paint and group layers, names, opacity, visibility, blend
mode, offsets), and each paint layer's pixels are in Krita's tiled format: 64 x 64 tiles, each LZF-compressed, the
channels stored one after another (blue, green, red, alpha for 8-bit RGBA). Read here: 8-bit and 16-bit RGBA paint
layers and groups. Anything else (vector, filter, fill, file and clone layers, masks, other colour spaces) is left
out and said; if no layer can be read, the flattened picture Krita stores (``mergedimage.png``) is opened instead.
Written from Krita's published source behaviour; no real .kra from her has been tried yet.
"""
from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

from .ora import _from_png
from .tiles import Document, Layer

OPS = {"normal": "normal", "multiply": "multiply", "screen": "screen", "overlay": "overlay", "add": "add",
       "linear_dodge": "add", "darken": "darken", "lighten": "lighten"}


def lzf_decompress(data: bytes, out_len: int) -> bytes:
    """LZF (as liblzf): literal runs and back references."""
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        c = data[i]
        i += 1
        if c < 32:                                          # literal run of c + 1 bytes
            out += data[i:i + c + 1]
            i += c + 1
        else:                                               # back reference
            ln = c >> 5
            if ln == 7:
                ln += data[i]
                i += 1
            ref = len(out) - ((c & 0x1F) << 8) - data[i] - 1
            i += 1
            ln += 2
            if ref < 0:
                raise ValueError("bad LZF data")
            if ref + ln <= len(out):
                out += out[ref:ref + ln]
            else:
                for k in range(ln):
                    out.append(out[ref + k])
        if len(out) > out_len:
            break
    return bytes(out[:out_len])


def _read_tiles(data: bytes):
    """Krita's tiled layer data -> (pixels premultiplied RGBA float, x0, y0) covering all tiles, or None."""
    f = io.BytesIO(data)
    head = {}
    while True:
        line = f.readline().decode("ascii", "replace").strip()
        if not line:
            return None
        key, _, val = line.partition(" ")
        head[key] = val
        if key == "DATA":
            break
    tw, th, ps = int(head.get("TILEWIDTH", 64)), int(head.get("TILEHEIGHT", 64)), int(head.get("PIXELSIZE", 4))
    if ps not in (4, 8):
        return None
    tiles = []
    for _ in range(int(head["DATA"])):
        x, y, comp, size = f.readline().decode("ascii").strip().split(",")
        raw = f.read(int(size))
        n = tw * th * ps
        if raw[:1] == b"\x01":
            px = lzf_decompress(raw[1:], n)
        else:
            px = raw[1:1 + n]
        tiles.append((int(x), int(y), px))
    if not tiles:
        return None
    xs, ys = [t[0] for t in tiles], [t[1] for t in tiles]
    x0, y0 = min(xs), min(ys)
    W, H = max(xs) + tw - x0, max(ys) + th - y0
    out = np.zeros((H, W, 4), np.float32)
    bpc = ps // 4
    for x, y, px in tiles:
        a = np.frombuffer(px.ljust(tw * th * ps, b"\0"), np.uint8 if bpc == 1 else "<u2").astype(np.float32)
        a = a.reshape(4, th, tw) / (255.0 if bpc == 1 else 65535.0)       # planar: B, G, R, A
        b, g, r, al = a
        out[y - y0:y - y0 + th, x - x0:x - x0 + tw] = np.stack([r * al, g * al, b * al, al], -1)
    return out, x0, y0


def load(path: str):
    """(Document, notes): the painting, and a list of plain-words notes about anything left out."""
    notes = []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        root = ET.fromstring(z.read("maindoc.xml"))
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        image = root.find(f"{ns}IMAGE")
        if image is None:
            raise ValueError("this .kra has no image in it")
        w, h = int(image.get("width")), int(image.get("height"))
        docname = image.get("name", "")
        cs = image.get("colorspacename", "RGBA")
        layers: list = []

        def walk(el, parent):
            items = list(el.findall(f"{ns}layer"))
            for le in reversed(items):                      # maindoc lists the top layer first
                kind = le.get("nodetype", "")
                name = le.get("name", "Layer")
                lay = Layer(name, opacity=int(le.get("opacity", 255)) / 255, visible=le.get("visible", "1") == "1",
                            locked=le.get("locked", "0") == "1", blend=OPS.get(le.get("compositeop", "normal"),
                                                                                  "normal"), parent=parent)
                if kind == "grouplayer":
                    lay.kind, lay.blend = "group", "pass" if le.get("passthrough", "0") == "1" else lay.blend
                    lay.collapsed = le.get("collapsed", "0") == "1"
                    sub = le.find(f"{ns}layers")
                    if sub is not None:
                        walk(sub, lay.uid)
                    layers.append(lay)
                    continue
                if kind != "paintlayer":
                    notes.append(f"\"{name}\" ({kind or 'unknown'}) was left out: Paint cannot read that kind of "
                                 "Krita layer yet.")
                    continue
                if le.get("colorspacename", cs) not in ("RGBA", "RGBA16"):
                    notes.append(f"\"{name}\" uses {le.get('colorspacename')} and was left out.")
                    continue
                fn = f"{docname}/layers/{le.get('filename')}"
                if fn in names:
                    got = _read_tiles(z.read(fn))
                    if got is not None:
                        px, tx, ty = got
                        lay.write(tx + int(le.get("x", 0)), ty + int(le.get("y", 0)), px)
                    lay.prune()
                if le.find(f"{ns}masks") is not None:
                    notes.append(f"\"{name}\" had masks in Krita; they were left out.")
                layers.append(lay)
        walk(image.find(f"{ns}layers"), None)
        if not any(not l.is_group for l in layers) and "mergedimage.png" in names:
            lay = Layer("Picture (flattened)")
            lay.write(0, 0, _from_png(z.read("mergedimage.png")))
            layers = [lay]
            notes.append("No layer could be read, so the flattened picture was opened.")
    if not layers:
        layers = [Layer("Layer 1")]
    doc = Document(w, h, None, layers, len(layers) - 1, None)
    doc.check()
    return doc, notes
