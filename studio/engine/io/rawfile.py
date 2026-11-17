"""Stage 1: decode. Opens a raw file with LibRaw (rawpy) and keeps everything the later stages need.

What is kept:
  * the full sensor mosaic as uint16, including the masked (optical-black) borders
  * the visible-area geometry and the camera's default crop (what the camera JPEG shows)
  * the 2x2 CFA pattern for the visible area, as LibRaw channel indices and as R/G/B
  * LibRaw black levels, white level and the camera's linear maximum per channel
  * as-shot and daylight white balance, the camera matrix (XYZ -> camera, D65)
  * metadata with the true ISO (metadata.py)

The raw file is never modified (rule 0.3.4).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ... import log as slog
from .metadata import Metadata, read_metadata

log = slog.get("engine.rawfile")

RAW_EXTENSIONS = (".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf", ".orf", ".rw2", ".pef",
                  ".dng", ".x3f", ".iiq", ".3fr", ".erf", ".mrw", ".srw", ".rwl", ".kdc", ".dcr", ".mos")


class UnsupportedRaw(Exception):
    """The file opened but is not something the Phase 1 engine can develop (e.g. not a Bayer sensor)."""


@dataclass
class RawImage:
    path: str
    meta: Metadata
    mosaic: np.ndarray                 # uint16, full sensor incl. masked borders
    top: int                           # visible area offset in the full sensor
    left: int
    height: int                        # visible area size
    width: int
    crop: tuple                        # camera default crop (x, y, w, h) relative to the visible area
    cfa_idx: np.ndarray                # 2x2 LibRaw channel indices (0..3) at visible-area origin
    cfa_rgb: np.ndarray                # 2x2 of 0/1/2 (R/G/B) at visible-area origin
    color_desc: str
    black_libraw: np.ndarray           # per LibRaw channel index (4)
    white_level: float
    linear_max: np.ndarray             # per LibRaw channel index (4); 0 = unknown
    wb_as_shot: np.ndarray             # 3 multipliers R, G, B with G = 1
    wb_daylight: np.ndarray
    cam_xyz: np.ndarray                # 3x3, XYZ (D65) -> camera
    flip: int                          # LibRaw flip code: 0, 3 (180), 5 (90 CCW), 6 (90 CW)
    is_bayer: bool = True
    notes: list = field(default_factory=list)

    # -------------------------------------------------------------------------------------------- views
    @property
    def visible(self) -> np.ndarray:
        return self.mosaic[self.top:self.top + self.height, self.left:self.left + self.width]

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    def black_for_positions(self, black4: np.ndarray) -> np.ndarray:
        """2x2 array of black levels for the CFA positions, given per-channel-index black levels."""
        return np.asarray(black4, np.float32)[self.cfa_idx]

    def clip_level(self) -> float:
        """The raw value at which the sensor stops being linear (camera linear max if known)."""
        lm = [v for v in self.linear_max if v > 0]
        return float(min(min(lm), self.white_level)) if lm else float(self.white_level)

    def embedded_jpeg(self) -> bytes | None:
        """Largest embedded JPEG preview (for the Library and the camera-JPEG comparison)."""
        import rawpy
        try:
            with rawpy.imread(self.path) as r:
                th = r.extract_thumb()
            if th.format == rawpy.ThumbFormat.JPEG:
                return bytes(th.data)
        except Exception as e:                   # some files have no preview; that's fine
            log.debug("no embedded JPEG in %s: %s", self.path, e)
        return None


def _cfa_from_colors(colors: np.ndarray, desc: str) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(colors[:2, :2], dtype=np.int64)
    letters = {"R": 0, "G": 1, "B": 2}
    rgb = np.vectorize(lambda c: letters.get(desc[c] if c < len(desc) else "?", -1))(idx)
    return idx, rgb.astype(np.int64)


def open_raw(path: str) -> RawImage:
    """Decode a raw file. Raises FileNotFoundError, or UnsupportedRaw with a plain-language reason (a layout the
    engine does not develop yet, a format the decoder lacks, a damaged file)."""
    import rawpy
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    name = os.path.basename(path)
    try:
        return _open_raw(path)
    except UnsupportedRaw:
        raise
    except rawpy.LibRawFileUnsupportedError as e:
        raise UnsupportedRaw(f"{name}: the raw decoder (LibRaw {_libraw_version()}) does not read this file. It "
                             f"may be from a camera newer than the decoder, in a compressed format it lacks (some "
                             f"Nikon High Efficiency raws), cut short when it was copied, or not a raw file at "
                             f"all.") from e
    except (rawpy.LibRawIOError, rawpy.LibRawDataError) as e:
        raise UnsupportedRaw(f"{name}: the file is damaged or incomplete (the raw decoder stopped: "
                             f"{_libraw_text(e)}). Copy it from the card again if you still have it.") from e
    except rawpy.LibRawError as e:
        raise UnsupportedRaw(f"{name}: the raw decoder could not read it ({_libraw_text(e)}).") from e
    except getattr(rawpy, "NotSupportedError", ()) as e:      # not one flat mosaic: sRAW / mRAW, Foveon, linear DNG
        raise UnsupportedRaw(f"{name}: not a Bayer mosaic ({_libraw_text(e)}). The Studio develops Bayer sensors "
                             f"so far; sRAW / mRAW, Foveon and already-developed DNGs are planned.") from e


def _libraw_version() -> str:
    import rawpy
    return ".".join(str(v) for v in getattr(rawpy, "libraw_version", ()) or ()) or "?"


def _libraw_text(e: Exception) -> str:
    """LibRaw's message as text (rawpy hands it over as bytes: "b'Input/output error'")."""
    a = e.args[0] if e.args else e
    if isinstance(a, bytes):
        return a.decode("utf-8", "replace")
    return str(a)


def _other(r) -> dict:
    """What LibRaw itself read about the shot (used only where our own metadata reader found nothing: raw
    containers it does not parse, such as Fujifilm RAF). Field names changed between rawpy versions."""
    o = getattr(r, "other", None)
    if o is None:
        return {}

    def first(*names):
        for n in names:
            v = getattr(o, n, None)
            if v:
                return v
        return None
    lens = getattr(r, "lens", None)
    return {"iso": first("iso_speed"), "shutter": first("shutter_speed", "shutter"), "aperture": first("aperture"),
            "focal": first("focal_length", "focal_len"), "timestamp": first("timestamp"),
            "lens": str(getattr(lens, "model", "") or "").strip() if lens is not None else ""}


def _fill_from_libraw(meta: Metadata, other: dict) -> None:
    """Fill what our reader left empty from LibRaw's reading; never replaces a value our reader found."""
    try:
        if not meta.exposure_s and other.get("shutter") and float(other["shutter"]) > 0:
            meta.exposure_s = float(other["shutter"])
        if not meta.fnumber and other.get("aperture") and float(other["aperture"]) > 0:
            meta.fnumber = float(other["aperture"])
        if not meta.focal_mm and other.get("focal") and float(other["focal"]) > 0:
            meta.focal_mm = float(other["focal"])
        if not meta.lens and other.get("lens"):
            meta.lens = other["lens"]
        ts = other.get("timestamp")
        if not meta.datetime and ts:
            import datetime as _dt
            if not isinstance(ts, _dt.datetime):
                ts = _dt.datetime.fromtimestamp(float(ts))
            if 1990 <= ts.year <= 2100:
                meta.datetime = ts.strftime("%Y:%m:%d %H:%M:%S")
    except (TypeError, ValueError, OverflowError, OSError):
        pass


def _open_raw(path: str) -> RawImage:
    import rawpy
    with slog.stage("decode", log, file=os.path.basename(path)):
        with rawpy.imread(path) as r:
            s = r.sizes
            flat = getattr(getattr(rawpy, "RawType", None), "Flat", None)
            if flat is not None and getattr(r, "raw_type", flat) != flat:
                # not one mosaic but full colour at every pixel: Canon sRAW / mRAW, Sigma Foveon, linear DNGs
                raise UnsupportedRaw(f"{os.path.basename(path)}: this file holds full colour at every pixel (Canon "
                                     f"sRAW / mRAW, Sigma Foveon, or a DNG another program already developed), "
                                     f"not a sensor mosaic. The Studio develops Bayer mosaics so far; these are "
                                     f"planned.")
            mosaic = np.array(r.raw_image, dtype=np.uint16, copy=True)
            colors_vis = np.array(r.raw_colors_visible[:4, :4])
            desc = r.color_desc.decode("ascii", "replace") if isinstance(r.color_desc, bytes) else str(r.color_desc)
            black = np.array(r.black_level_per_channel, dtype=np.float64)
            white = float(r.white_level)
            try:
                linmax = np.array(r.camera_white_level_per_channel, dtype=np.float64)
            except Exception:
                linmax = np.zeros(4)
            if linmax is None or np.size(linmax) != 4:
                linmax = np.zeros(4)
            wb = list(r.camera_whitebalance)
            day = list(r.daylight_whitebalance)
            cam_xyz = np.array(r.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
            pattern = np.array(r.raw_pattern)
            other = _other(r)
            libraw_iso = float(other["iso"]) if other.get("iso") else None
            flip = int(s.flip)
            crop = (int(s.crop_left_margin) - int(s.left_margin), int(s.crop_top_margin) - int(s.top_margin),
                    int(s.crop_width), int(s.crop_height))
            top, left, h, w = int(s.top_margin), int(s.left_margin), int(s.height), int(s.width)
    meta = read_metadata(path, libraw_iso)
    _fill_from_libraw(meta, other)

    notes = []
    is_bayer = pattern.shape == (2, 2) and len(desc) >= 3 and sorted(desc[:3]) == ["B", "G", "R"]
    cfa_idx, cfa_rgb = _cfa_from_colors(colors_vis, desc)
    if not is_bayer or sorted(cfa_rgb.ravel().tolist()) != [0, 1, 1, 2]:
        xtrans = pattern.shape == (6, 6)
        raise UnsupportedRaw(f"{os.path.basename(path)}: " +
                             ("a Fujifilm X-Trans sensor. The Studio develops Bayer sensors so far; X-Trans is "
                              "planned." if xtrans else
                              f"not a Bayer sensor (pattern {pattern.tolist()}, colours {desc!r}). The Studio "
                              f"develops Bayer sensors so far; other layouts (Foveon, sRAW / mRAW, already-"
                              f"developed DNGs) are planned."))
    if mosaic.shape[0] < top + h or mosaic.shape[1] < left + w:
        raise UnsupportedRaw("visible area lies outside the raw buffer")

    # Camera default crop, clamped to the visible area and aligned to the CFA (even offsets).
    cx, cy, cw, ch = crop
    if cw <= 0 or ch <= 0:
        cx, cy, cw, ch = 0, 0, w, h
    cx, cy = max(0, cx), max(0, cy)
    cw, ch = min(cw, w - cx), min(ch, h - cy)
    if cx % 2:
        cx, cw = cx + 1, cw - 1
    if cy % 2:
        cy, ch = cy + 1, ch - 1
    if not np.isfinite(cam_xyz).all() or abs(np.linalg.det(cam_xyz)) < 1e-9:
        notes.append("No colour matrix for this camera: colours use a generic matrix.")
        cam_xyz = np.array([[0.7, -0.1, -0.1], [-0.45, 1.25, 0.2], [-0.09, 0.2, 0.58]])   # typical CMOS
    from ..color.camera import normalise_wb
    wb3 = normalise_wb(wb)
    if not (np.isfinite(wb3).all() and wb3.min() > 0):
        wb3 = normalise_wb(day) if sum(day[:3]) > 0 else np.array([2.0, 1.0, 1.5])
        notes.append("No as-shot white balance in the file: daylight used.")
    day3 = normalise_wb(day) if sum(day[:3]) > 0 else wb3.copy()

    img = RawImage(path=path, meta=meta, mosaic=mosaic, top=top, left=left, height=h, width=w,
                   crop=(cx, cy, cw, ch), cfa_idx=cfa_idx, cfa_rgb=cfa_rgb, color_desc=desc,
                   black_libraw=black, white_level=white, linear_max=linmax, wb_as_shot=wb3, wb_daylight=day3,
                   cam_xyz=cam_xyz, flip=flip, is_bayer=True, notes=notes)
    log.info("opened %s: %s %s, %d x %d (%.1f MP), %s", os.path.basename(path), meta.model or "unknown camera",
             meta.iso_label, w, h, img.megapixels, "flip %d" % flip if flip else "upright")
    return img
