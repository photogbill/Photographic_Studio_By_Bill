"""Minimal, dependency-free TIFF / EXIF / Canon MakerNote reader for CR2 and CR3.

Why our own: LibRaw does not expose everything the plan needs (the true ISO when EXIF says 65535,
Canon ShotInfo, sensor temperature, the masked-border geometry in SensorInfo), and CR3 keeps its
metadata in ISO-BMFF boxes that general EXIF libraries do not open.

CR2: a TIFF file. IFD0 -> ExifIFD (0x8769) -> MakerNote (0x927C). Canon MakerNote is a bare IFD whose
     offsets are relative to the start of the TIFF file.
CR3: ISO-BMFF. moov/uuid(85c0b687...) holds CMT1 (IFD0 as TIFF), CMT2 (ExifIFD as TIFF),
     CMT3 (Canon MakerNote as TIFF), CMT4 (GPS, never read). Each CMTn is a complete little TIFF.

Values are returned as Python numbers, tuples or strings. Nothing here logs values (serials stay private).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
TYPE_FMT = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 11: "f", 12: "d", 13: "I"}

EXIF_IFD = 0x8769
GPS_IFD = 0x8825
MAKERNOTE = 0x927C


@dataclass
class IFD:
    tags: dict = field(default_factory=dict)        # tag -> value
    types: dict = field(default_factory=dict)       # tag -> TIFF type
    offsets: dict = field(default_factory=dict)     # tag -> absolute data offset (for MakerNote)


class TiffReader:
    """Reads IFDs from a bytes buffer holding a TIFF stream (base = start of the TIFF header)."""

    def __init__(self, buf: bytes, base: int = 0):
        self.buf = buf
        self.base = base
        bo = buf[base:base + 2]
        if bo == b"II":
            self.e = "<"
        elif bo == b"MM":
            self.e = ">"
        else:
            raise ValueError("not a TIFF stream")
        magic = struct.unpack_from(self.e + "H", buf, base + 2)[0]
        if magic not in (42, 0x4F52, 0x5352, 0x55):   # 42 = TIFF; Olympus ORF variants; 0x55 = Panasonic RW2
            raise ValueError("bad TIFF magic %r" % magic)
        self.ifd0_offset = struct.unpack_from(self.e + "I", buf, base + 4)[0]

    def read_ifd(self, offset: int, rel_base: int | None = None, max_entries: int = 2000) -> tuple[IFD, int]:
        """Read one IFD at ``offset`` (relative to rel_base, default self.base). Returns (ifd, next_offset)."""
        rb = self.base if rel_base is None else rel_base
        buf, e = self.buf, self.e
        pos = rb + offset
        if pos + 2 > len(buf):
            raise ValueError("IFD offset outside buffer")
        n = struct.unpack_from(e + "H", buf, pos)[0]
        if n == 0 or n > max_entries:
            raise ValueError("implausible IFD entry count %d" % n)
        ifd = IFD()
        for i in range(n):
            p = pos + 2 + 12 * i
            if p + 12 > len(buf):
                break
            tag, typ, count = struct.unpack_from(e + "HHI", buf, p)
            size = TYPE_SIZES.get(typ)
            if size is None:
                continue
            nbytes = size * count
            if nbytes <= 4:
                dpos = p + 8
            else:
                dpos = rb + struct.unpack_from(e + "I", buf, p + 8)[0]
            ifd.types[tag] = typ
            ifd.offsets[tag] = dpos
            if dpos + nbytes > len(buf):
                continue
            ifd.tags[tag] = self._value(typ, count, dpos)
        nxt_pos = pos + 2 + 12 * n
        nxt = struct.unpack_from(e + "I", buf, nxt_pos)[0] if nxt_pos + 4 <= len(buf) else 0
        return ifd, nxt

    def _value(self, typ: int, count: int, dpos: int):
        buf, e = self.buf, self.e
        if typ == 2:
            raw = buf[dpos:dpos + count]
            return raw.split(b"\x00", 1)[0].decode("latin-1", "replace").strip()
        if typ == 7:
            if count > 64:
                return None          # big undefined blobs (MakerNote etc.) are reached via offsets
            return bytes(buf[dpos:dpos + count])
        if typ in (5, 10):
            fmt = e + ("II" if typ == 5 else "ii") * count
            v = struct.unpack_from(fmt, buf, dpos)
            vals = tuple((v[2 * i] / v[2 * i + 1]) if v[2 * i + 1] else 0.0 for i in range(count))
            return vals[0] if count == 1 else vals
        fmt = TYPE_FMT[typ]
        if count > 4096:
            count = 4096
        v = struct.unpack_from(e + fmt * count, buf, dpos)
        return v[0] if count == 1 else tuple(v)


# ----------------------------------------------------------------------------------------- containers
def _boxes(buf: bytes, start: int, end: int):
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[i:i + 8])
        hdr = 8
        if size == 1:
            if i + 16 > end:
                break
            size = struct.unpack(">Q", buf[i + 8:i + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - i
        if size < hdr:
            break
        yield typ, i + hdr, min(i + size, end)
        i += size


CANON_UUID = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")


def cr3_cmt_blobs(buf: bytes) -> dict:
    """Return {b'CMT1': bytes, ...} from the head of a CR3 file."""
    out = {}
    for typ, s, e in _boxes(buf, 0, len(buf)):
        if typ != b"moov":
            continue
        for t2, s2, e2 in _boxes(buf, s, e):
            if t2 == b"uuid" and buf[s2:s2 + 16] == CANON_UUID:
                for t3, s3, e3 in _boxes(buf, s2 + 16, e2):
                    if t3 in (b"CMT1", b"CMT2", b"CMT3"):          # CMT4 is GPS: never read
                        out[t3] = buf[s3:e3]
    return out


@dataclass
class TiffMeta:
    ifd0: dict = field(default_factory=dict)
    exif: dict = field(default_factory=dict)
    makernote: dict = field(default_factory=dict)
    container: str = ""


def _exif_block(buf: bytes) -> bytes:
    """The TIFF-structured EXIF block of a JPEG (APP1 "Exif") or PNG (eXIf chunk); b"" when there is none."""
    if buf[:2] == b"\xff\xd8":
        i = 2
        while i + 4 <= len(buf) and buf[i] == 0xFF:
            marker = buf[i + 1]
            if marker in (0xD9, 0xDA):              # end of image / start of scan: no EXIF before the pixels
                break
            n = struct.unpack_from(">H", buf, i + 2)[0]
            if marker == 0xE1 and buf[i + 4:i + 10] == b"Exif\0\0":
                return buf[i + 10:i + 2 + n]
            i += 2 + n
        return b""
    i = 8
    while i + 8 <= len(buf):
        n = struct.unpack_from(">I", buf, i)[0]
        kind = buf[i + 4:i + 8]
        if kind == b"eXIf":
            return buf[i + 8:i + 8 + n]
        if kind in (b"IDAT", b"IEND"):
            break
        i += 12 + n
    return b""


def read_tiff_meta(path: str, head_bytes: int = 24 * 1024 * 1024) -> TiffMeta:
    """Read IFD0, ExifIFD and the Canon MakerNote from a CR2/CR3 (or any TIFF-based raw)."""
    with open(path, "rb") as fh:
        buf = fh.read(head_bytes)
    meta = TiffMeta()
    if buf[:2] == b"\xff\xd8" or buf[:8] == b"\x89PNG\r\n\x1a\n":     # a scan: EXIF inside JPEG APP1 / PNG eXIf
        meta.container = "jpeg" if buf[:2] == b"\xff\xd8" else "png"
        buf = _exif_block(buf)
        if not buf:
            return meta
    elif buf[4:8] == b"ftyp":
        meta.container = "isobmff"
        blobs = cr3_cmt_blobs(buf)
        if b"CMT1" in blobs:
            r = TiffReader(blobs[b"CMT1"])
            meta.ifd0 = r.read_ifd(r.ifd0_offset)[0].tags
        if b"CMT2" in blobs:
            r = TiffReader(blobs[b"CMT2"])
            meta.exif = r.read_ifd(r.ifd0_offset)[0].tags
        if b"CMT3" in blobs:
            r = TiffReader(blobs[b"CMT3"])
            meta.makernote = r.read_ifd(r.ifd0_offset)[0].tags
        return meta
    meta.container = "tiff"
    r = TiffReader(buf)
    ifd0, _ = r.read_ifd(r.ifd0_offset)
    meta.ifd0 = ifd0.tags
    if EXIF_IFD in ifd0.tags:
        exif, _ = r.read_ifd(int(ifd0.tags[EXIF_IFD]))
        meta.exif = exif.tags
        if MAKERNOTE in exif.offsets:
            mn_pos = exif.offsets[MAKERNOTE]
            make = str(ifd0.tags.get(0x010F, ""))
            try:
                if make.upper().startswith("CANON"):
                    # Canon: bare IFD at the MakerNote position, offsets relative to the TIFF start.
                    mn, _ = r.read_ifd(mn_pos - r.base, rel_base=r.base)
                    meta.makernote = mn.tags
            except (ValueError, struct.error):
                meta.makernote = {}
    return meta
