"""Write ICC v2 matrix/TRC display profiles for our output spaces.

Writing our own keeps the numbers identical to spaces.py and avoids shipping third-party profile files.
Profiles are named "... compatible" (e.g. "Adobe RGB (1998) compatible"): Adobe RGB is a trademark, the
colour space itself is free to implement. Colour-managed software identifies the space by its contents.

Layout: 128-byte header, tag table, tags (desc, cprt, wtpt, rXYZ, gXYZ, bXYZ, rTRC, gTRC, bTRC).
Primaries are stored adapted to D50 with Bradford, as the ICC spec requires.
"""
from __future__ import annotations

import datetime as _dt
import struct
from functools import lru_cache

import numpy as np

from .spaces import RGBSpace, SPACES, srgb_decode, xy_to_XYZ

COPYRIGHT = "No copyright. Made by Photographic Studio By Bill (GPL-3.0). Free to use."


def _s15f16(v: float) -> bytes:
    return struct.pack(">i", int(round(v * 65536.0)))


def _xyz_tag(xyz) -> bytes:
    return b"XYZ " + b"\0" * 4 + b"".join(_s15f16(v) for v in xyz)


def _desc_tag(text: str) -> bytes:
    a = text.encode("ascii", "replace") + b"\0"
    body = b"desc" + b"\0" * 4 + struct.pack(">I", len(a)) + a
    body += struct.pack(">II", 0, 0)            # unicode language code, unicode count
    body += struct.pack(">HB", 0, 0) + b"\0" * 67   # scriptcode code, count, 67-byte buffer
    return body


def _text_tag(text: str) -> bytes:
    return b"text" + b"\0" * 4 + text.encode("ascii", "replace") + b"\0"


def _curv_tag(space: RGBSpace) -> bytes:
    if space.trc == "gamma":
        g = int(round(space.gamma * 256.0))
        return b"curv" + b"\0" * 4 + struct.pack(">IH", 1, g)
    # sRGB-like: 1024-entry table of the decode curve (encoded value -> linear)
    n = 1024
    x = np.linspace(0, 1, n)
    y = np.clip(np.round(srgb_decode(x) * 65535.0), 0, 65535).astype(">u2")
    return b"curv" + b"\0" * 4 + struct.pack(">I", n) + y.tobytes()


def _pad4(b: bytes) -> bytes:
    return b + b"\0" * ((4 - len(b) % 4) % 4)


@lru_cache(maxsize=None)
def profile_bytes(key: str) -> bytes:
    """ICC profile for one of the spaces in spaces.SPACES (by key: 'adobergb', 'srgb', 'prophoto', ...)."""
    space = SPACES[key]
    M = space.to_xyz_d50
    trc = _curv_tag(space)
    tags = [
        (b"desc", _desc_tag(space.name)),
        (b"cprt", _text_tag(COPYRIGHT)),
        (b"wtpt", _xyz_tag(xy_to_XYZ(space.white_xy))),
        (b"rXYZ", _xyz_tag(M[:, 0])),
        (b"gXYZ", _xyz_tag(M[:, 1])),
        (b"bXYZ", _xyz_tag(M[:, 2])),
        (b"rTRC", trc),
        (b"gTRC", trc),
        (b"bTRC", trc),
    ]
    table_len = 4 + 12 * len(tags)
    offset = 128 + table_len
    table = struct.pack(">I", len(tags))
    data = b""
    shared = {}
    for sig, body in tags:
        if body in shared:                       # the three TRC tags share one copy
            off, size = shared[body]
        else:
            off, size = offset + len(data), len(body)
            data += _pad4(body)
            shared[body] = (off, size)
        table += sig + struct.pack(">II", off, size)
    total = 128 + table_len + len(data)
    now = _dt.datetime(2026, 1, 1)               # fixed date: identical bytes every build (deterministic export)
    header = struct.pack(">I", total)
    header += b"\0\0\0\0"                        # preferred CMM: none
    header += struct.pack(">I", 0x02100000)      # version 2.1
    header += b"mntr" + b"RGB " + b"XYZ "
    header += struct.pack(">6H", now.year, now.month, now.day, now.hour, now.minute, now.second)
    header += b"acsp" + b"\0\0\0\0"              # platform: none
    header += struct.pack(">I", 0)               # flags
    header += b"\0" * 4 + b"\0" * 4              # manufacturer, model
    header += b"\0" * 8                          # attributes
    header += struct.pack(">I", 0)               # rendering intent: perceptual
    header += b"".join(_s15f16(v) for v in (0.9642, 1.0, 0.8249))   # PCS illuminant D50
    header += b"\0" * 4                          # creator
    header += b"\0" * 16                         # profile ID (v2: zero)
    header += b"\0" * 28                         # reserved
    assert len(header) == 128
    out = header + table + data
    assert len(out) == total
    return out
