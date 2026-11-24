"""OpenEXR files for Paint (BUILD_PLAN_PAINT P5): floating-point, scene-linear pictures for 3D, games and HDR.

Written from the OpenEXR file layout (openexr.com, "OpenEXR File Layout"): scan-line files, channels R, G, B, A
as 32-bit floats, written uncompressed; read uncompressed, ZIP (16 lines per block) and ZIPS (one line) files,
half or float channels. Not read: PIZ, PXR24, B44, DWA compression, tiled and multi-part files (said plainly).
Pixels are straight (not premultiplied) linear light here; EXR stores premultiplied, so A is used to convert.
"""
from __future__ import annotations

import os
import struct
import zlib

import numpy as np


class EXRProblem(ValueError):
    pass


def _attr(name, typ, data):
    return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(data)) + data


def save(path, rgba_premult_linear: np.ndarray) -> str:
    """Write a float RGBA EXR (premultiplied, linear), atomic."""
    px = np.asarray(rgba_premult_linear, np.float32)
    h, w = px.shape[:2]
    chans = ["A", "B", "G", "R"]                               # EXR lists channels in alphabetical order
    idx = {"R": 0, "G": 1, "B": 2, "A": 3}
    chl = b"".join(c.encode() + b"\0" + struct.pack("<iB3xii", 2, 0, 1, 1) for c in chans) + b"\0"
    head = b"\x76\x2f\x31\x01" + struct.pack("<I", 2)
    head += _attr("channels", "chlist", chl)
    head += _attr("compression", "compression", b"\0")
    box = struct.pack("<iiii", 0, 0, w - 1, h - 1)
    head += _attr("dataWindow", "box2i", box) + _attr("displayWindow", "box2i", box)
    head += _attr("lineOrder", "lineOrder", b"\0")
    head += _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    head += _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0, 0))
    head += _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    from ... import __version__
    from ..io.export import OWN_MARK
    head += _attr("software", "string", f"{OWN_MARK} {__version__}".encode()) + b"\0"
    line_bytes = w * 4 * 4
    table_at = len(head)
    first = table_at + 8 * h
    offsets = struct.pack(f"<{h}Q", *[first + y * (8 + line_bytes) for y in range(h)])
    body = bytearray()
    for y in range(h):
        body += struct.pack("<ii", y, line_bytes)
        for c in chans:
            body += px[y, :, idx[c]].astype("<f4").tobytes()
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        f.write(head + offsets + bytes(body))
    os.replace(tmp, path)
    return path


def _undo_zip(raw: bytes) -> bytes:
    """OpenEXR's ZIP: deflate, then a delta predictor, then the bytes were split into even / odd halves."""
    t = np.frombuffer(zlib.decompress(raw), np.uint8).astype(np.int32)
    t = (np.cumsum(np.concatenate([t[:1], t[1:] - 128])) & 0xFF).astype(np.uint8)
    n = len(t)
    half = (n + 1) // 2
    out = np.empty(n, np.uint8)
    out[0::2] = t[:half]
    out[1::2] = t[half:]
    return out.tobytes()


def load(path) -> np.ndarray:
    """(h, w, 4) float32 premultiplied linear RGBA."""
    d = open(path, "rb").read()
    if d[:4] != b"\x76\x2f\x31\x01":
        raise EXRProblem("not an OpenEXR file")
    flags = struct.unpack("<I", d[4:8])[0]
    if flags & 0x200 or flags & 0x1000:
        raise EXRProblem("tiled or multi-part EXR files are not read yet")
    p, attrs = 8, {}
    while d[p] != 0:
        e = d.index(b"\0", p)
        name = d[p:e].decode()
        e2 = d.index(b"\0", e + 1)
        typ = d[e + 1:e2].decode()
        n = struct.unpack("<i", d[e2 + 1:e2 + 5])[0]
        attrs[name] = (typ, d[e2 + 5:e2 + 5 + n])
        p = e2 + 5 + n
    p += 1
    chans = []
    cl = attrs["channels"][1]
    q = 0
    while cl[q] != 0:
        e = cl.index(b"\0", q)
        nm = cl[q:e].decode()
        ptype = struct.unpack("<i", cl[e + 1:e + 5])[0]
        chans.append((nm, ptype))
        q = e + 17
    comp = attrs["compression"][1][0]
    lines_per = {0: 1, 2: 1, 3: 16}.get(comp)
    if lines_per is None:
        raise EXRProblem("this EXR uses a compression Paint cannot read yet (PIZ, PXR24, B44 or DWA); save it with "
                         "ZIP or no compression")
    x0, y0, x1, y1 = struct.unpack("<iiii", attrs["dataWindow"][1])
    w, h = x1 - x0 + 1, y1 - y0 + 1
    nblocks = (h + lines_per - 1) // lines_per
    offs = struct.unpack(f"<{nblocks}Q", d[p:p + 8 * nblocks])
    sizes = {1: 2, 2: 4}
    out = {nm: np.zeros((h, w), np.float32) for nm, _ in chans}
    for off in offs:
        yb, ln = struct.unpack("<ii", d[off:off + 8])
        raw = d[off + 8:off + 8 + ln]
        nl = min(lines_per, y1 - yb + 1)
        need = sum(w * sizes.get(pt, 4) for _, pt in chans) * nl
        data = raw if (comp == 0 or len(raw) == need) else _undo_zip(raw)
        pos = 0
        for row in range(nl):
            for nm, pt in chans:
                n = w * sizes.get(pt, 4)
                dt = "<f2" if pt == 1 else ("<f4" if pt == 2 else "<u4")
                out[nm][yb - y0 + row] = np.frombuffer(data[pos:pos + n], dt).astype(np.float32)
                pos += n
    one = np.ones((h, w), np.float32)
    return np.stack([out.get("R", one * 0), out.get("G", one * 0), out.get("B", one * 0), out.get("A", one)], -1)
