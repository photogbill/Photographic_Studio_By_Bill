"""The linear intermediate: how a stacking tool's result goes back into Develop (BUILD_PLAN_TABS 1.2).

A focus stack, a super-resolution merge or a Dual Pixel highlight recovery is written as a 32-bit float TIFF of
scene-linear ProPhoto RGB (1.0 = the sensor clip of a neutral, values above 1.0 kept), already upright, with a
linear ProPhoto profile embedded, the Studio's Software mark (so it counts as the Studio's own file and a later
build may replace it) and a JSON description that says what it is. Develop opens it through the scan path, but
the scan path's "undo the Standard tone curve" step is skipped (``ScanImage.scene_linear``): the pixels already
are scene light, so every Develop tool after demosaicing applies to them as it does to a raw.

    {"mark": "studio-intermediate-1", "kind": "focus" | "superres" | "dualpixel", "sources": [...],
     "method": ..., "params": {...}, "scale": 1.0 | 1.5 | 2.0}
"""
from __future__ import annotations

import json
import os

import numpy as np

from ... import __version__, log as slog

log = slog.get("engine.intermediate")

MARK = "studio-intermediate-1"
KINDS = ("focus", "superres", "dualpixel")
LABELS = {"focus": "Focus stack", "superres": "Super-res", "dualpixel": "Dual Pixel"}


def write_intermediate(path: str, rgb_scene: np.ndarray, meta: dict | None, info: dict) -> str:
    """Write ``rgb_scene`` (float32 H x W x 3, scene-linear ProPhoto, upright) as a 32-bit float TIFF, zlib
    compressed (BigTIFF past 4 GB), through ``path + '.partial'`` so no half-written file ever carries the real
    name. ``info`` goes into the description with the mark; ``meta`` (camera, lens, exposure) into the XMP.
    Returns the path."""
    import tifffile
    from ..color import icc
    from .export import xmp_packet
    rgb = np.asarray(rgb_scene)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"an intermediate is H x W x 3, not {rgb.shape}")
    d = dict(info or {})
    d["mark"] = MARK
    d.setdefault("kind", "focus")
    d.setdefault("scale", 1.0)
    desc = json.dumps(d, default=_jsonable, separators=(",", ":"))
    meta = slog.scrub(dict(meta or {}))
    prof = icc.profile_bytes("prophoto", linear=True)
    xmp = xmp_packet(meta)
    extratags = [(34675, "B", len(prof), prof, True), (700, "B", len(xmp), xmp, True)]
    for code, key in ((271, "make"), (272, "model")):
        if meta.get(key):
            extratags.append((code, "s", 0, str(meta[key]), True))
    big = rgb.shape[0] * rgb.shape[1] * 12 > 3_900_000_000
    tmp = path + ".partial"
    try:
        prev = preview_srgb(rgb)
        with slog.stage("intermediate.write", log, file=os.path.basename(path), w=rgb.shape[1], h=rgb.shape[0]):
            with tifffile.TiffWriter(tmp, bigtiff=big) as tw:
                tw.write(_tiles(rgb), shape=rgb.shape, dtype=np.float32, photometric="rgb", planarconfig="contig",
                         compression="zlib", description=desc, tile=(TILE, TILE),
                         software=f"Photographic Studio By Bill {__version__}", metadata=None,
                         extratags=extratags, datetime=meta.get("datetime") or None)
                # page 2: a small sRGB preview (reduced-resolution subfile) for the Library's thumbnails, so a
                # 180 MP float file is never read whole just to draw a card
                tw.write(prev, photometric="rgb", subfiletype=1, compression="zlib", metadata=None)
        os.replace(tmp, path)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    log.info("intermediate %s written (%s, %d x %d)", os.path.basename(path), d.get("kind"), rgb.shape[1],
             rgb.shape[0])
    return path


PREVIEW_LONG = 1600


def preview_srgb(rgb: np.ndarray, long_edge: int = PREVIEW_LONG) -> np.ndarray:
    """uint8 sRGB of a scene-linear ProPhoto image with the default look (the Standard curve), long edge about
    ``long_edge`` (strided: a memory-mapped image is not read whole)."""
    from ..color import spaces as sp
    from .scanfile import scene_to_display
    k = max(1, int(np.ceil(max(rgb.shape[:2]) / float(long_edge))))
    small = np.ascontiguousarray(rgb[::k, ::k], np.float32)
    disp = scene_to_display(np.nan_to_num(small, nan=0.0, posinf=1.0, neginf=0.0))
    s = sp.srgb_encode(np.clip(sp.apply_matrix(disp, sp.conversion_matrix(sp.PROPHOTO, sp.SRGB)), 0.0, 1.0))
    return (s * 255.0 + 0.5).astype(np.uint8)


def read_preview(path: str) -> np.ndarray | None:
    """The small sRGB preview page of an intermediate, or None."""
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            if len(tf.pages) < 2:
                return None
            a = tf.pages[1].asarray()
        return a if a.dtype == np.uint8 and a.ndim == 3 else None
    except Exception:                                    # noqa: BLE001
        return None


TILE = 256


def _tiles(rgb: np.ndarray):
    """256 x 256 float32 tiles in row-major order, edge tiles padded: a memory-mapped result is streamed to the
    file a band at a time, never copied whole."""
    h, w = rgb.shape[:2]
    for y in range(0, h, TILE):
        band = np.ascontiguousarray(rgb[y:y + TILE], np.float32)
        for x in range(0, w, TILE):
            t = band[:, x:x + TILE]
            if t.shape[0] != TILE or t.shape[1] != TILE:
                p = np.zeros((TILE, TILE, 3), np.float32)
                p[:t.shape[0], :t.shape[1]] = t
                t = p
            yield np.ascontiguousarray(t)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_info(path: str) -> dict | None:
    """The description of an intermediate, or None for any other file (never raises)."""
    if not str(path).lower().endswith((".tif", ".tiff")):
        return None
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            tag = tf.pages[0].tags.get(270)
            if tag is None:
                return None
            v = tag.value
            if isinstance(v, bytes):
                v = v.decode("utf-8", "replace")
            if MARK not in v:
                return None
            d = json.loads(v)
        return d if isinstance(d, dict) and d.get("mark") == MARK else None
    except Exception:                                    # noqa: BLE001  (not ours, unreadable: not an intermediate)
        return None


def label(info: dict | None) -> str:
    """Short chip text for the Library: 'Focus stack', 'Super-res 2x', 'Dual Pixel HDR'."""
    if not info:
        return ""
    k = info.get("kind", "")
    if k == "superres":
        sc = float(info.get("scale", 1.0) or 1.0)
        mode = info.get("mode", "")
        if mode == "sky":
            return "Night sky stack"
        if sc > 1.01:
            return f"Super-res {sc:g}x"
        return "Burst stack"
    if k == "dualpixel":
        m = info.get("method", "")
        return {"hdr": "Dual Pixel HDR", "half1": "Dual Pixel half", "half2": "Dual Pixel half"}.get(m, "Dual Pixel")
    return LABELS.get(k, "Stack")
