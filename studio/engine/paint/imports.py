"""Bringing other programs' brushes into Paint (BUILD_PLAN_PAINT P2).

- GIMP ``.gbr`` (a tip) and ``.gih`` (an image hose: several tips, one picked at random per dab).
- Photoshop ``.abr``: the sampled tips of versions 1 and 2 and of versions 6 to 10 (the "8BIM samp" section, raw or
  RLE-packed), following the layout GIMP's own reader uses. Photoshop's other brush settings (dynamics, textures)
  live in a separate descriptor section and are not read; every tip becomes a brush with its spacing.
- Krita ``.kpp``: the preset's XML, kept in the PNG. Engine, size, spacing, softness, opacity, flow and pressure for
  size / opacity / flow are mapped; Krita's other engines become the nearest of ours and say so; tips stored as
  separate files in Krita's resource folders cannot come along (the round tip is used).
- Clip Studio ``.sut`` (a SQLite file) and ``.sutg``: the tip pictures stored inside (PNG) are taken out; Clip
  Studio's own settings are not read.
Readers are written from the formats' published or reverse-engineered layouts and checked on synthetic files built
the same way; real files from each program have not been tried yet.
"""
from __future__ import annotations

import io
import os
import re
import sqlite3
import struct
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .brush import BrushSettings
from .curves import Curve

EXTS = (".abr", ".gbr", ".gih", ".kpp", ".sut", ".sutg")


class ImportProblem(ValueError):
    """The file is not one we can read; the message says why in plain words."""


def _brush(name, tip, spacing=0.1, source="", tips=None):
    lin = Curve()
    b = BrushSettings(name, size=float(max(tip.shape)) if tip is not None else 30.0, min_size=0.15, hardness=0.8,
                      spacing=max(0.02, min(2.0, spacing)), tip=tip,
                      dynamics={"size": {"pressure": lin}, "opacity": {}, "flow": {"pressure": lin}},
                      best_for=f"a tip brought in from {source}", weak_at="only the tip came across; set the "
                      "dynamics in Edit brush", tags="Imported")
    if tips:
        b.tips = tips
    return b


def _norm_tip(a):
    a = np.asarray(a, np.float32)
    if a.max() <= 0:
        raise ImportProblem("the brush tip is empty")
    return a / a.max()


# ---------------------------------------------------------------- GIMP
def _read_gbr(data: bytes, pos: int = 0):
    """One GIMP brush record at ``pos``: (name, tip 0..1, spacing fraction, next position)."""
    if len(data) - pos < 20:
        raise ImportProblem("too short to be a GIMP brush")
    hsize, version, w, h, depth = struct.unpack(">IIIII", data[pos:pos + 20])
    if version == 1:
        spacing, name_at = 25, pos + 20
    elif version in (2, 3):
        if data[pos + 20:pos + 24] != b"GIMP":
            raise ImportProblem("not a GIMP brush (no GIMP mark)")
        spacing = struct.unpack(">I", data[pos + 24:pos + 28])[0]
        name_at = pos + 28
    else:
        raise ImportProblem(f"GIMP brush version {version} is not known")
    name = data[name_at:pos + hsize].split(b"\0")[0].decode("utf-8", "replace") or "GIMP brush"
    n = w * h * depth
    px = np.frombuffer(data[pos + hsize:pos + hsize + n], np.uint8)
    if px.size != n or depth not in (1, 4):
        raise ImportProblem("the GIMP brush's picture is cut short")
    px = px.reshape(h, w, depth)
    tip = px[..., 0] if depth == 1 else px[..., 3]          # colour brushes: the alpha is the shape
    return name, tip.astype(np.float32) / 255.0, spacing / 100.0, pos + hsize + n


def read_gbr(path) -> list:
    data = Path(path).read_bytes()
    name, tip, spacing, _ = _read_gbr(data)
    return [_brush(name, _norm_tip(tip), spacing, "GIMP (.gbr)")]


def read_gih(path) -> list:
    data = Path(path).read_bytes()
    nl1 = data.find(b"\n")
    nl2 = data.find(b"\n", nl1 + 1)
    if nl1 < 0 or nl2 < 0:
        raise ImportProblem("not a GIMP image hose (.gih)")
    name = data[:nl1].decode("utf-8", "replace").strip() or Path(path).stem
    pos, tips, spacing = nl2 + 1, [], 0.25
    while pos < len(data) - 20:
        _, tip, spacing, pos = _read_gbr(data, pos)
        tips.append(_norm_tip(tip))
    if not tips:
        raise ImportProblem("the image hose has no tips in it")
    return [_brush(name, tips[0], spacing, "GIMP (.gih)", tips=tips)]


# ---------------------------------------------------------------- Photoshop
class _R:
    def __init__(self, data):
        self.d, self.p = data, 0

    def take(self, n):
        if self.p + n > len(self.d):
            raise ImportProblem("the file ends too early")
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def u8(self):
        return self.take(1)[0]

    def u16(self):
        return struct.unpack(">H", self.take(2))[0]

    def u32(self):
        return struct.unpack(">I", self.take(4))[0]

    def i32(self):
        return struct.unpack(">i", self.take(4))[0]


def _unpack_bits(r: _R, height: int, row_bytes: int) -> bytes:
    """Photoshop's RLE: one length per row, then PackBits for each row."""
    lens = [r.u16() for _ in range(height)]
    out = bytearray()
    for L in lens:
        end, row = r.p + L, bytearray()
        while r.p < end:
            n = r.u8()
            if n > 127:
                n -= 256
            if n < 0:
                if n == -128:
                    continue
                row += bytes([r.u8()]) * (1 - n)
            else:
                row += r.take(n + 1)
        out += row[:row_bytes].ljust(row_bytes, b"\0")
    return bytes(out)


def _abr_sample(r: _R, top, left, bottom, right, depth, compress):
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0 or w > 8000 or h > 8000:
        raise ImportProblem("a brush tip in the file has an impossible size")
    bpp = max(1, depth // 8)
    raw = _unpack_bits(r, h, w * bpp) if compress else r.take(w * h * bpp)
    a = np.frombuffer(raw, np.uint8 if bpp == 1 else ">u2").reshape(h, w).astype(np.float32)
    return a / (255.0 if bpp == 1 else 65535.0)


def read_abr(path) -> list:
    data = Path(path).read_bytes()
    r = _R(data)
    version = r.u16()
    stem = Path(path).stem
    out = []
    if version in (1, 2):
        count = r.u16()
        for k in range(count):
            btype, size = r.u16(), r.u32()
            nxt = r.p + size
            if btype == 2:
                r.u32()                                            # misc
                spacing = r.u16()
                name = f"{stem} {k + 1}"
                if version == 2:
                    n = r.u32()
                    name = r.take(2 * n).decode("utf-16-be", "replace").rstrip("\0") or name
                r.u8()                                             # anti-aliasing
                r.take(8)                                          # short bounds
                top, left, bottom, right = r.i32(), r.i32(), r.i32(), r.i32()
                depth, compress = r.u16(), r.u8()
                tip = _abr_sample(r, top, left, bottom, right, depth, compress)
                out.append(_brush(name, _norm_tip(tip), spacing / 100.0, "Photoshop (.abr)"))
            r.p = nxt
    elif version in (6, 7, 10):
        sub = r.u16()
        while True:                                                # find the 8BIM samp section
            if r.p + 12 > len(data) or r.take(4) != b"8BIM":
                raise ImportProblem("no brush tips found in this Photoshop file")
            key, size = r.take(4), r.u32()
            if key == b"samp":
                end = r.p + size
                break
            r.p += size
        k = 0
        while r.p < end - 4:
            bsize = r.u32()
            nxt = r.p + bsize + ((4 - bsize % 4) % 4)
            r.p += 47 if sub == 1 else 301
            top, left, bottom, right = r.i32(), r.i32(), r.i32(), r.i32()
            depth, compress = r.u16(), r.u8()
            k += 1
            tip = _abr_sample(r, top, left, bottom, right, depth, compress)
            out.append(_brush(f"{stem} {k}", _norm_tip(tip), 0.25, "Photoshop (.abr)"))
            r.p = nxt
    else:
        raise ImportProblem(f"Photoshop brush version {version} is not known")
    if not out:
        raise ImportProblem("no sampled brush tips in this Photoshop file (computed round brushes are not read)")
    return out


# ---------------------------------------------------------------- Krita
_KRITA_ENGINES = {"paintbrush": "wash", "colorsmudge": "smudge", "spraybrush": "spray", "hatchingbrush": "hatching",
                  "sketchbrush": "sketch", "tangentnormal": "normal", "duplicate": "clone", "chalkbrush": "chalk"}


def read_kpp(path) -> list:
    from PIL import Image
    try:
        im = Image.open(path)
        im.load()
    except OSError as err:
        raise ImportProblem(f"not a Krita preset (.kpp): {err}") from None
    xml = im.text.get("preset") if hasattr(im, "text") else None
    if not xml:
        raise ImportProblem("this .kpp has no Krita preset inside")
    root = ET.fromstring(xml)
    params = {p.get("name"): (p.text or "") for p in root.iter("param")}
    engine = root.get("paintopid", "paintbrush")
    name = root.get("name") or Path(path).stem
    b = BrushSettings(name, mode=_KRITA_ENGINES.get(engine, "wash"), tags="Imported",
                      dynamics={"size": {}, "opacity": {}, "flow": {}})
    note = "" if engine in _KRITA_ENGINES else f" Krita's {engine} engine became plain painting."
    bd = params.get("brush_definition", "")
    if bd.strip().startswith("<"):
        brush = ET.fromstring(bd)
        b.spacing = float(brush.get("spacing", b.spacing))
        b.angle = float(brush.get("angle", 0)) * (180 / np.pi if abs(float(brush.get("angle", 0))) <= 6.3 else 1)
        mg = brush.find("MaskGenerator")
        if mg is not None:
            b.size = float(mg.get("diameter", b.size))
            b.roundness = max(0.05, min(1.0, float(mg.get("ratio", 1))))
            b.hardness = max(0.0, min(1.0, (float(mg.get("hfade", 1)) + float(mg.get("vfade", 1))) / 2))
        if brush.get("type") not in (None, "auto_brush"):
            note += " Its picture tip lives in Krita's resource folder and could not come along (a round tip is used)."
    for key, attr in (("OpacityValue", "opacity"), ("FlowValue", "flow")):
        try:
            setattr(b, attr, max(0.01, min(1.0, float(params[key]))))
        except (KeyError, ValueError):
            pass
    for key, target in (("PressureSize", "size"), ("PressureOpacity", "opacity"), ("PressureFlow", "flow")):
        if params.get(key, "").lower() == "true":
            b.dynamics[target]["pressure"] = Curve()
    b.best_for = f"a Krita preset ({engine})"
    b.weak_at = ("Krita's finer settings (sensor curves, textures, masked brushes) do not come across." + note).strip()
    return [b]


# ---------------------------------------------------------------- Clip Studio
_PNG = b"\x89PNG\r\n\x1a\n"


def _pngs_in(blob: bytes):
    i = blob.find(_PNG)
    while i >= 0:
        j = blob.find(b"IEND", i)
        if j < 0:
            return
        yield blob[i:j + 8]
        i = blob.find(_PNG, j)


def _tip_from_png(png: bytes):
    from PIL import Image
    im = Image.open(io.BytesIO(png))
    if "A" in im.getbands():
        a = np.asarray(im.getchannel("A"), np.float32) / 255.0
        if a.min() < 0.99:
            return a
    return 1.0 - np.asarray(im.convert("L"), np.float32) / 255.0      # dark paints


def _sut_db(data: bytes, stem: str):
    import tempfile
    tips, name = [], stem
    fd, tmp = tempfile.mkstemp(suffix=".sut")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        con = sqlite3.connect(tmp)
        try:
            tables = [t for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            for t in tables:
                cols = [c[1] for c in con.execute(f'PRAGMA table_info("{t}")')]
                if t.lower() == "node" and "NodeName" in cols:
                    row = con.execute(f'SELECT NodeName FROM "{t}" WHERE NodeName IS NOT NULL LIMIT 1').fetchone()
                    if row and row[0]:
                        name = str(row[0])
                for c in cols:
                    if "thumb" in c.lower():
                        continue                                    # thumbnails are not tips
                    for (v,) in con.execute(f'SELECT "{c}" FROM "{t}"'):
                        if isinstance(v, (bytes, memoryview)):
                            for png in _pngs_in(bytes(v)):
                                try:
                                    tips.append(_norm_tip(_tip_from_png(png)))
                                except (OSError, ImportProblem):
                                    pass
        finally:
            con.close()
    finally:
        os.remove(tmp)
    return name, tips


def read_sut(path) -> list:
    p = Path(path)
    data = p.read_bytes()
    found = []
    if data[:16].startswith(b"SQLite format 3"):
        found.append(_sut_db(data, p.stem))
    elif zipfile.is_zipfile(p):                                     # a group: brushes inside
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                d = z.read(n)
                if d.startswith(b"SQLite format 3"):
                    found.append(_sut_db(d, Path(n).stem))
    else:                                                           # last try: tips stored anywhere in the file
        tips = [_norm_tip(_tip_from_png(png)) for png in _pngs_in(data)]
        found.append((p.stem, tips))
    out = []
    for name, tips in found:
        if tips:
            out.append(_brush(name, tips[0], 0.1, "Clip Studio", tips=tips if len(tips) > 1 else None))
    if not out:
        raise ImportProblem("no brush tip pictures found in this Clip Studio file")
    return out


def import_brushes(path) -> list:
    """Every brush in a file from another program, as Paint brushes. Raises ImportProblem with the reason."""
    ext = Path(path).suffix.lower()
    try:
        if ext == ".gbr":
            return read_gbr(path)
        if ext == ".gih":
            return read_gih(path)
        if ext == ".abr":
            return read_abr(path)
        if ext == ".kpp":
            return read_kpp(path)
        if ext in (".sut", ".sutg"):
            return read_sut(path)
    except (struct.error, ET.ParseError, sqlite3.Error, zipfile.BadZipFile) as err:
        raise ImportProblem(f"the file could not be read ({err})") from None
    raise ImportProblem(f"{ext} brushes are not known (Paint reads {', '.join(EXTS)} and MyPaint .myb)")
