"""Photoshop .psd open and save (BUILD_PLAN_PAINT P5), written from Adobe's published file format: RGB, 8 or 16
bits, layers with names, opacity, visibility, blend modes, clipping, groups (section dividers) and layer masks;
raw, RLE (PackBits) and ZIP channel data (read); the flattened picture for programs that only read that; the
colour profile (image resource 1039) both ways, so a Display P3 or Adobe RGB painting stays itself; .psb (large
documents) opened. Not read or written: adjustment and text layers (their pixels are kept as plain layers where
Photoshop stored them), smart objects, layer styles, CMYK / Lab files; .psb is not written.
"""
from __future__ import annotations

import os
import struct

import numpy as np

from .tiles import TILE, Document, Layer

KEYS = {"normal": b"norm", "multiply": b"mul ", "screen": b"scrn", "overlay": b"over", "add": b"lddg",
        "darken": b"dark", "lighten": b"lite", "pass": b"pass"}
KEYS_BACK = {v: k for k, v in KEYS.items()}
KEYS_BACK[b"lgD "] = "add"
_BIG_KEYS = {b"LMsk", b"Lr16", b"Lr32", b"Layr", b"Mt16", b"Mt32", b"Mtrn", b"Alph", b"FMsk", b"lnk2", b"FEid",
             b"FXid", b"PxSD"}
_ICC_NAMES = {"srgb": "sRGB", "displayp3": "Display P3", "adobergb": "Adobe RGB", "prophoto": "ProPhoto"}


class PSDProblem(ValueError):
    pass


# ---------------------------------------------------------------- writing
def _pascal(name: str) -> bytes:
    b = name.encode("latin-1", "replace")[:255]
    out = bytes([len(b)]) + b
    return out + b"\0" * ((4 - len(out) % 4) % 4)


def _unicode_block(name: str) -> bytes:
    u = name.encode("utf-16-be")
    data = struct.pack(">I", len(name)) + u
    data += b"\0" * ((4 - len(data) % 4) % 4)
    return b"8BIMluni" + struct.pack(">I", len(data)) + data


def _straight8(px, space="srgb"):
    a = px[..., 3:4]
    rgb = np.where(a > 0, px[..., :3] / np.maximum(a, 1e-8), 0)
    if space == "linear":                                    # other programs expect encoded values
        from .space import convert
        rgb = convert(rgb, "linear", "srgb")
    return (np.clip(np.concatenate([rgb, a], -1), 0, 1) * 255 + 0.5).astype(np.uint8)


def save(doc: Document, path: str) -> str:
    """Write ``doc`` as an 8-bit RGB .psd (atomic)."""
    recs, chans = [], []
    order = []                                               # bottom first, with group dividers before contents
    kids = doc._kids()

    def walk(parent):
        for i in kids.get(parent, []):
            lay = doc.layers[i]
            if lay.is_group:
                order.append(("end", lay))
                walk(lay.uid)
                order.append(("group", lay))
            else:
                order.append(("layer", lay))
    walk(None)
    for kind, lay in order:
        extra = b""
        src = lay
        if kind == "layer" and lay.is_generated:              # fill / file layers: their pixels
            src = Layer("x")
            src.write(0, 0, lay.generated(0, 0, doc.width, doc.height, doc.width, doc.height))
        if kind == "layer" and src.tiles and not lay.is_adjust:
            b = src.bounds()
            x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(doc.width, b[2]), min(doc.height, b[3])
            if x1 <= x0 or y1 <= y0:
                x0 = y0 = 0
                x1 = y1 = 1
            px = _straight8(src.read(x0, y0, x1 - x0, y1 - y0), doc.space)
        else:
            x0 = y0 = 0
            x1 = y1 = 0
            px = np.zeros((0, 0, 4), np.uint8)
        planes = [(-1, px[..., 3]), (0, px[..., 0]), (1, px[..., 1]), (2, px[..., 2])]
        mask_rec = b""
        if kind == "layer" and lay.mask is not None:
            mv = (np.clip(lay.mask_values(0, 0, doc.width, doc.height), 0, 1) * 255 + 0.5).astype(np.uint8)
            planes.append((-2, mv))
            mask_rec = struct.pack(">iiiiBB2x", 0, 0, doc.height, doc.width, 255, 0)
        rec = struct.pack(">iiiiH", y0, x0, y1, x1, len(planes))
        for cid, plane in planes:
            rec += struct.pack(">hI", cid, 2 + plane.size)
        blend = KEYS.get(lay.blend if kind != "end" else "normal", b"norm")
        flags = 0 if lay.visible else 2
        rec += b"8BIM" + blend + struct.pack(">BBBB", int(round(lay.opacity * 255)), 1 if lay.clip else 0, flags, 0)
        name = "</Layer group>" if kind == "end" else lay.name
        extra += struct.pack(">I", len(mask_rec)) + mask_rec + struct.pack(">I", 0) + _pascal(name)
        extra += _unicode_block(name)
        if kind in ("group", "end"):
            t = 3 if kind == "end" else (2 if lay.collapsed else 1)
            data = struct.pack(">I", t) + (b"8BIM" + KEYS.get(lay.blend, b"pass") if kind == "group" else b"")
            extra += b"8BIMlsct" + struct.pack(">I", len(data)) + data
        rec += struct.pack(">I", len(extra)) + extra
        recs.append(rec)
        chans.append(b"".join(struct.pack(">H", 0) + p.tobytes() for _, p in planes))
    info = struct.pack(">h", len(recs)) + b"".join(recs) + b"".join(chans)
    if len(info) % 2:
        info += b"\0"
    layer_info = struct.pack(">I", len(info)) + info
    lmi = layer_info + struct.pack(">I", 0)
    merged = _straight8(doc.composite(background=True), doc.space)
    if doc.background is None:
        a = merged[..., 3:4].astype(np.float32) / 255
        merged = (merged[..., :3] * a + 255 * (1 - a) + 0.5).astype(np.uint8)
    else:
        merged = merged[..., :3]
    out = b"8BPS" + struct.pack(">H6xHIIHH", 1, 3, doc.height, doc.width, 8, 3)
    from ..color import icc as _icc
    prof = _icc.profile_bytes("srgb" if doc.space == "linear" else doc.space)
    res = b"8BIM" + struct.pack(">H", 1039) + b"\0\0" + struct.pack(">I", len(prof)) + prof + (b"\0" * (len(prof) % 2))
    out += struct.pack(">I", 0) + struct.pack(">I", len(res)) + res + struct.pack(">I", len(lmi)) + lmi
    out += struct.pack(">H", 0) + b"".join(np.ascontiguousarray(merged[..., c]).tobytes() for c in range(3))
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        f.write(out)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- reading
class _R:
    def __init__(self, d, p=0):
        self.d, self.p = d, p

    def take(self, n):
        if self.p + n > len(self.d):
            raise PSDProblem("the file ends too early")
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def skip_block(self):
        """Read a 4-byte length and step over that many bytes (not ``p += u()``: that would lose the 4 bytes)."""
        n = self.u(">I")[0]
        self.p += n

    def u(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(n))


def _rle_rows(r, rows, width, big=False):
    counts = [r.u(">I" if big else ">H")[0] for _ in range(rows)]
    out = bytearray()
    for c in counts:
        end, row = r.p + c, bytearray()
        while r.p < end:
            n = r.take(1)[0]
            if n > 127:
                n -= 256
            if n < 0:
                if n != -128:
                    row += r.take(1) * (1 - n)
            else:
                row += r.take(n + 1)
        out += bytes(row[:width]).ljust(width, b"\0")
    return bytes(out)


def _channel(r, h, w, depth, length=None, big=False):
    start = r.p
    comp = r.u(">H")[0]
    bpp = depth // 8
    if h <= 0 or w <= 0:
        return None
    if comp == 0:
        raw = r.take(h * w * bpp)
    elif comp == 1:
        raw = _rle_rows(r, h, w * bpp, big)
    elif comp in (2, 3):                                  # ZIP, without / with row prediction
        import zlib
        n = (length - 2) if length else len(r.d) - r.p
        raw = zlib.decompress(r.take(n))
        if comp == 3:
            a = np.frombuffer(raw, np.uint8 if bpp == 1 else ">u2").reshape(h, w).astype(np.int64)
            a = np.cumsum(a, axis=1) & (0xFF if bpp == 1 else 0xFFFF)
            raw = a.astype(np.uint8 if bpp == 1 else ">u2").tobytes()
    else:
        raise PSDProblem(f"this .psd uses a channel compression ({comp}) Paint does not know")
    a = np.frombuffer(raw, np.uint8 if bpp == 1 else ">u2").reshape(h, w).astype(np.float32)
    return a / (255.0 if bpp == 1 else 65535.0)


def load(path: str) -> Document:
    d = open(path, "rb").read()
    r = _R(d)
    if r.take(4) != b"8BPS":
        raise PSDProblem("not a Photoshop file")
    version, nch, h, w, depth, mode = r.u(">H6xHIIHH")
    if version not in (1, 2):
        raise PSDProblem("not a Photoshop file this Paint knows")
    big = version == 2                                       # .psb: some lengths are 8 bytes
    if mode != 3 or depth not in (8, 16):
        raise PSDProblem("only RGB files of 8 or 16 bits are read (this one is a different colour mode or depth)")
    r.skip_block()                                           # colour mode data
    ir_len = r.u(">I")[0]                                    # image resources: the colour profile (1039)
    ir_end, icc = r.p + ir_len, None
    while r.p + 12 <= ir_end:
        if r.take(4) != b"8BIM":
            break
        rid = r.u(">H")[0]
        nl = r.take(1)[0]
        r.p += nl + ((nl + 1) % 2)
        ln = r.u(">I")[0]
        blk = r.take(ln)
        r.p += ln % 2
        if rid == 1039:
            icc = blk
    r.p = ir_end
    lmi_len = r.u(">Q" if big else ">I")[0]
    lmi_end = r.p + lmi_len
    layers_out = []
    if lmi_len:
        li_len = r.u(">Q" if big else ">I")[0]
        if li_len:
            count = abs(r.u(">h")[0])
            infos = []
            for _ in range(count):
                top, left, bottom, right, n = r.u(">iiiiH")
                ch = [r.u(">hQ" if big else ">hI") for _ in range(n)]
                sig, key = r.take(4), r.take(4)
                opacity, clip, flags, _ = r.u(">BBBB")
                xl = r.u(">I")[0]
                xend = r.p + xl
                mlen = r.u(">I")[0]
                mrect = r.u(">iiii") if mlen >= 16 else None
                r.p += mlen - (16 if mrect else 0)
                r.skip_block()                               # blending ranges
                nl = r.take(1)[0]
                name = r.take(nl).decode("latin-1", "replace")
                r.p += (4 - (nl + 1) % 4) % 4
                section = None
                while r.p + 12 <= xend:
                    if r.take(4) not in (b"8BIM", b"8B64"):
                        break
                    k = r.take(4)
                    ln = r.u(">Q" if (big and k in _BIG_KEYS) else ">I")[0]
                    blk = r.take(ln)
                    if k == b"luni":
                        n_ = struct.unpack(">I", blk[:4])[0]
                        name = blk[4:4 + 2 * n_].decode("utf-16-be", "replace")
                    elif k in (b"lsct", b"lsdk"):
                        section = struct.unpack(">I", blk[:4])[0]
                        if len(blk) >= 12:
                            key = blk[8:12]
                r.p = xend
                infos.append(dict(rect=(top, left, bottom, right), ch=ch, key=key, opacity=opacity, clip=clip,
                                  hidden=bool(flags & 2), name=name, section=section, mrect=mrect))
            for info in infos:                               # channel image data, in the same order
                top, left, bottom, right = info["rect"]
                planes = {}
                for cid, ln in info["ch"]:
                    start = r.p
                    if cid == -2 and info["mrect"]:
                        mt, ml, mb, mr = info["mrect"]
                        planes[cid] = (_channel(r, mb - mt, mr - ml, depth, ln, big), mt, ml)
                    else:
                        planes[cid] = _channel(r, bottom - top, right - left, depth, ln, big)
                    r.p = start + ln
                info["planes"] = planes
            stack = [None]
            pending = []
            for info in infos:                               # bottom first
                sec = info["section"]
                if sec == 3:                                 # the bottom of a group: its contents follow
                    pending.append([])
                    continue
                if sec in (1, 2):
                    g = Layer(info["name"], kind="group", opacity=info["opacity"] / 255, visible=not info["hidden"],
                              blend=KEYS_BACK.get(info["key"], "pass"), collapsed=sec == 2)
                    contents = pending.pop() if pending else []
                    for c in contents:
                        if c.parent is None:
                            c.parent = g.uid
                    (pending[-1] if pending else layers_out).extend(contents + [g])
                    continue
                lay = Layer(info["name"], opacity=info["opacity"] / 255, visible=not info["hidden"],
                            blend=KEYS_BACK.get(info["key"], "normal"), clip=bool(info["clip"]))
                p = info["planes"]
                if all(k in p and p[k] is not None for k in (0, 1, 2)):
                    a = p.get(-1)
                    a = np.ones_like(p[0]) if a is None else a
                    px = np.stack([p[0] * a, p[1] * a, p[2] * a, a], -1).astype(np.float32)
                    lay.write(left, top, px)
                if -2 in p and p[-2][0] is not None:
                    mv, mt, ml = p[-2]
                    full = np.ones((h, w), np.float32)
                    y0, x0 = max(0, mt), max(0, ml)
                    y1, x1 = min(h, mt + mv.shape[0]), min(w, ml + mv.shape[1])
                    if y1 > y0 and x1 > x0:
                        full[y0:y1, x0:x1] = mv[y0 - mt:y1 - mt, x0 - ml:x1 - ml]
                    from .edit import add_mask
                    add_mask(lay, full, Document(w, h))
                (pending[-1] if pending else layers_out).append(lay)
            while pending:
                layers_out.extend(pending.pop())
    r.p = lmi_end
    if not layers_out:                                       # no layers: the flattened picture as one layer
        comp = r.u(">H")[0]
        n = min(nch, 4)
        planes = []
        if comp == 0:
            for _ in range(n):
                planes.append(np.frombuffer(r.take(h * w * depth // 8), np.uint8 if depth == 8 else ">u2")
                              .reshape(h, w).astype(np.float32) / (255.0 if depth == 8 else 65535.0))
        else:
            raise PSDProblem("this .psd has no layers and a packed picture; save it with layers to open it here")
        lay = Layer("Background")
        a = planes[3] if n > 3 else np.ones((h, w), np.float32)
        lay.write(0, 0, np.stack([planes[0] * a, planes[1] * a, planes[2] * a, a], -1).astype(np.float32))
        layers_out = [lay]
    doc = Document(w, h, None, layers_out, len(layers_out) - 1, None)
    doc.check()
    if not any(not l.is_group for l in doc.layers):
        doc.layers.append(Layer("Layer 1"))
    doc.space = space_from_icc(icc)
    return doc


def space_from_icc(icc):
    """Our working space for an embedded profile, by its description (unknown or none: sRGB)."""
    if not icc:
        return "srgb"
    text = icc.decode("latin-1", "replace").lower()
    for key, words in (("displayp3", ("display p3", "p3")), ("adobergb", ("adobe rgb", "adobergb", "1998")),
                       ("prophoto", ("prophoto", "romm"))):
        if any(w in text for w in words):
            return key
    return "srgb"
