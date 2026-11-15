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

import io
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
    """Decode a raw file. Raises FileNotFoundError, UnsupportedRaw, or rawpy errors for corrupt files."""
    import rawpy
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with slog.stage("decode", log, file=os.path.basename(path)):
        with rawpy.imread(path) as r:
            s = r.sizes
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
            libraw_iso = float(r.other.iso_speed) if getattr(r, "other", None) else None
            flip = int(s.flip)
            crop = (int(s.crop_left_margin) - int(s.left_margin), int(s.crop_top_margin) - int(s.top_margin),
                    int(s.crop_width), int(s.crop_height))
            top, left, h, w = int(s.top_margin), int(s.left_margin), int(s.height), int(s.width)
    meta = read_metadata(path, libraw_iso)

    notes = []
    is_bayer = pattern.shape == (2, 2) and len(desc) >= 3 and sorted(desc[:3]) == ["B", "G", "R"]
    cfa_idx, cfa_rgb = _cfa_from_colors(colors_vis, desc)
    if not is_bayer or sorted(cfa_rgb.ravel().tolist()) != [0, 1, 1, 2]:
        raise UnsupportedRaw(f"{os.path.basename(path)}: not a 2x2 Bayer sensor (pattern {pattern.tolist()}, "
                             f"colours {desc!r}). X-Trans and other layouts come in Phase 14.")
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
