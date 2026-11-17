"""Scans: TIFF, JPEG and PNG files (a flatbed or film scanner, an old print copied by a phone, an exported file).

A scan is not a raw. It is an image someone has already turned into a picture, with a tone curve and a colour
space baked in. The Studio's tools all work on scene-linear light, so a scan is brought in like this:

  1. decode the pixels as stored (8 or 16 bit, grey or RGB; float TIFFs are taken as linear) and turn them the
     right way up from the EXIF / TIFF orientation;
  2. undo the file's transfer curve and colour space with its embedded ICC profile (matrix/TRC and grey profiles
     are read here field by field; a profile we cannot read, or none at all, means sRGB, which is what a file
     without a profile is taken to be everywhere) -> display-linear ProPhoto RGB, 1.0 = the file's white;
  3. (pipeline.py) undo the Studio Standard tone curve, so a default edit shows the scan exactly as it is and every
     slider (exposure, contrast, highlights...) acts on it as it does on a raw. A film negative skips this step:
     its display-linear values are light let through the film, which the negative stage turns into a positive.

The file is never modified (rule 0.3.4). Edits live in NAME.tif.psb.json beside it, as for raws.
"""
from __future__ import annotations

import io
import os
import struct
from dataclasses import dataclass, field

import numpy as np

from ... import log as slog
from ..color import spaces as sp
from .metadata import Metadata, read_metadata

log = slog.get("engine.scanfile")

SCAN_EXTENSIONS = (".tif", ".tiff", ".jpg", ".jpeg", ".png")
THUMB_LONG = 1600                       # long edge of the thumbnail the Library shows (the raws' embedded JPEG)


class UnsupportedScan(Exception):
    """The file opened but holds something the Studio cannot develop (CMYK, a palette with transparency...)."""


def is_scan(path: str) -> bool:
    return str(path).lower().endswith(SCAN_EXTENSIONS)


# ------------------------------------------------------------------------------------------ ICC profiles
def _s15(b: bytes, o: int) -> float:
    return struct.unpack_from(">i", b, o)[0] / 65536.0


def _curve(b: bytes, o: int):
    """A 'curv' or 'para' tag -> function on 0..1 arrays (encoded -> linear)."""
    sig = b[o:o + 4]
    if sig == b"curv":
        n = struct.unpack_from(">I", b, o + 8)[0]
        if n == 0:
            return lambda v: v
        if n == 1:
            g = struct.unpack_from(">H", b, o + 12)[0] / 256.0
            return lambda v: np.power(np.maximum(v, 0.0), g)
        tab = np.frombuffer(b, ">u2", n, o + 12).astype(np.float64) / 65535.0
        xs = np.linspace(0.0, 1.0, n)
        return lambda v: np.interp(v, xs, tab)
    if sig == b"para":
        ft = struct.unpack_from(">H", b, o + 8)[0]
        npar = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(ft)
        if npar is None:
            raise ValueError(f"para function type {ft}")
        p = [_s15(b, o + 12 + 4 * i) for i in range(npar)]
        g = p[0]
        if ft == 0:
            return lambda v: np.power(np.maximum(v, 0.0), g)
        if ft == 1:
            a, bb = p[1], p[2]
            return lambda v: np.where(v >= -bb / a, np.power(np.maximum(a * v + bb, 0.0), g), 0.0)
        if ft == 2:
            a, bb, c = p[1], p[2], p[3]
            return lambda v: np.where(v >= -bb / a, np.power(np.maximum(a * v + bb, 0.0), g) + c, c)
        if ft == 3:
            a, bb, c, d = p[1], p[2], p[3], p[4]
            return lambda v: np.where(v >= d, np.power(np.maximum(a * v + bb, 0.0), g), c * v)
        a, bb, c, d, e, f = p[1:7]
        return lambda v: np.where(v >= d, np.power(np.maximum(a * v + bb, 0.0), g) + e, c * v + f)
    raise ValueError(f"tone curve type {sig!r}")


def _desc(b: bytes, o: int) -> str:
    sig = b[o:o + 4]
    try:
        if sig == b"desc":
            n = struct.unpack_from(">I", b, o + 8)[0]
            return b[o + 12:o + 12 + n].split(b"\0")[0].decode("latin-1", "replace")
        if sig == b"mluc":
            n, size = struct.unpack_from(">II", b, o + 8)
            if n:
                ln, off = struct.unpack_from(">II", b, o + 16 + 4)
                return b[o + off:o + off + ln].decode("utf-16-be", "replace")
    except Exception:
        pass
    return ""


@dataclass
class Profile:
    """What the Studio needs from a file's colour profile: per-channel decode curves and RGB -> XYZ (D50)."""
    name: str
    curves: list                        # 1 (grey) or 3 functions, encoded 0..1 -> linear
    to_xyz_d50: np.ndarray | None       # 3x3 for RGB; None for grey
    source: str = "icc"                 # icc | assumed


def parse_icc(b: bytes) -> Profile:
    """Matrix/TRC RGB and grey-TRC ICC profiles (every scanner, camera and editor profile in practice). Raises
    ValueError for anything else (LUT-based profiles), and the caller falls back to sRGB."""
    if len(b) < 132 or b[36:40] != b"acsp":
        raise ValueError("not an ICC profile")
    space = b[16:20]
    n = struct.unpack_from(">I", b, 128)[0]
    tags = {}
    for i in range(n):
        sig, off, size = struct.unpack_from(">4sII", b, 132 + 12 * i)
        tags[sig] = off
    name = _desc(b, tags[b"desc"]) if b"desc" in tags else ""
    if space == b"GRAY":
        if b"kTRC" not in tags:
            raise ValueError("grey profile without kTRC")
        return Profile(name or "grey", [_curve(b, tags[b"kTRC"])], None)
    if space != b"RGB ":
        raise ValueError(f"colour space {space!r}")
    need = (b"rXYZ", b"gXYZ", b"bXYZ", b"rTRC", b"gTRC", b"bTRC")
    if not all(t in tags for t in need):
        raise ValueError("not a matrix/TRC profile")
    cols = [[_s15(b, tags[t] + 8 + 4 * k) for k in range(3)] for t in need[:3]]
    M = np.array(cols, float).T
    return Profile(name or "embedded", [_curve(b, tags[t]) for t in need[3:]], M)


def _space_profile(space: sp.RGBSpace, source: str = "assumed") -> Profile:
    return Profile(space.name, [space.decode] * 3, space.to_xyz_d50, source)


def profile_for(icc: bytes | None, grey: bool) -> Profile:
    if icc:
        try:
            p = parse_icc(icc)
            if grey and p.to_xyz_d50 is not None:
                log.info("grey image with an RGB profile (%s): its green curve is used", p.name)
                p = Profile(p.name, [p.curves[1]], None)
            elif not grey and p.to_xyz_d50 is None:
                log.info("RGB image with a grey profile (%s): the curve applies to all three", p.name)
                p = Profile(p.name, p.curves * 3, sp.SRGB.to_xyz_d50)
            return p
        except Exception as e:
            log.warning("embedded colour profile unreadable here (%s); taking the file as sRGB", e)
    if grey:
        return Profile("sRGB grey", [sp.srgb_decode], None, "assumed")
    return _space_profile(sp.SRGB)


# ------------------------------------------------------------------------------------------ decoding
_ORIENT = {1: (0, False), 2: (0, True), 3: (2, False), 4: (2, True), 5: (3, True), 6: (3, False), 7: (1, True),
           8: (1, False)}                   # EXIF orientation -> (np.rot90 turns counter-clockwise, mirror first)


def _upright(a: np.ndarray, orientation: int) -> np.ndarray:
    k, mirror = _ORIENT.get(int(orientation or 1), (0, False))
    if mirror:
        a = a[:, ::-1]
    if k:
        a = np.rot90(a, k)
    return np.ascontiguousarray(a)


def _read_tiff(path: str):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        a = page.asarray()
        icc = None
        orient = 1
        tag = page.tags.get(34675)
        if tag is not None:
            icc = bytes(tag.value)
        t = page.tags.get(274)
        if t is not None:
            orient = int(t.value)
        photometric = int(getattr(page.photometric, "value", page.photometric))
        extra = page.extrasamples or ()
    if photometric == 5:
        raise UnsupportedScan("CMYK TIFF: save it as RGB first")
    if photometric == 3:                        # palette: tifffile gives indices; expand with the colour map
        raise UnsupportedScan("palette TIFF: save it as RGB or greyscale first")
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[2] not in (3, 4):
        a = np.moveaxis(a, 0, -1)               # planar storage
    if a.ndim == 3 and a.shape[2] > 3:
        a = a[..., :3]                          # alpha / extra samples are not part of the picture
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if photometric == 0 and a.dtype.kind in "ui":   # min-is-white
        a = np.iinfo(a.dtype).max - a
    del extra
    return a, icc, orient


def _read_pil(path: str):
    from PIL import Image as PI
    with PI.open(path) as im:
        icc = im.info.get("icc_profile")
        orient = 1
        try:
            orient = int(im.getexif().get(274, 1) or 1)
        except Exception:
            pass
        mode = im.mode
        if mode in ("I;16", "I;16B", "I;16L"):
            a = np.asarray(im, np.uint16)
        elif mode == "I":
            a = np.clip(np.asarray(im, np.int64), 0, 65535).astype(np.uint16)
        elif mode in ("L", "LA"):
            a = np.asarray(im.convert("L"), np.uint8)
        elif mode == "CMYK":
            raise UnsupportedScan("CMYK JPEG: save it as RGB first")
        elif mode == "F":
            a = np.asarray(im, np.float32)
        else:
            a = np.asarray(im.convert("RGB"), np.uint8)
    return a, icc, orient


@dataclass
class ScanImage:
    """A scan in the shape the pipeline knows from RawImage (crop, flip, white balance, meta...)."""
    path: str
    meta: Metadata
    data: np.ndarray                    # (h, w) or (h, w, 3), uint8 / uint16 / float32, upright
    profile: Profile
    height: int
    width: int
    crop: tuple                         # whole image: (0, 0, w, h)
    bits: int
    grey: bool                          # stored as one channel
    flip: int = 0                       # already upright
    top: int = 0
    left: int = 0
    is_scan: bool = True
    is_bayer: bool = False
    color_desc: str = "RGB"
    cfa_rgb: np.ndarray = field(default_factory=lambda: np.array([[0, 1], [1, 2]]))
    wb_as_shot: np.ndarray = field(default_factory=lambda: np.ones(3))
    wb_daylight: np.ndarray = field(default_factory=lambda: np.ones(3))
    cam_xyz: np.ndarray = field(default_factory=lambda: np.linalg.inv(sp.PROPHOTO.to_xyz_d50))
    notes: list = field(default_factory=list)
    scene_linear: bool = False          # a stacking tool's result (io/intermediate.py): already scene light
    _luts: list | None = None

    # The "camera" of a scan is linear ProPhoto (D50): the file's white is (1, 1, 1), so the as-shot balance is
    # one and the Temperature slider starts near 5000 K; moving it acts as it does on a raw.

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    @property
    def mono(self) -> bool:
        """Black and white: stored as grey, or RGB whose channels agree (a B&W print scanned in colour)."""
        if self.grey:
            return True
        k = max(1, int(np.sqrt(self.height * self.width / 250_000)))
        s = self.data[::k, ::k].astype(np.float32)
        top = float(np.iinfo(self.data.dtype).max) if self.data.dtype.kind in "ui" else 1.0
        spread = (s.max(axis=2) - s.min(axis=2)) / top
        return float(np.percentile(spread, 99)) < 0.06

    def _lut(self):
        """Per channel: stored value -> display-linear (integer files only)."""
        if self._luts is None and self.data.dtype.kind in "ui":
            top = np.iinfo(self.data.dtype).max
            xs = np.arange(top + 1, dtype=np.float64) / top
            self._luts = [np.asarray(f(xs), np.float32) for f in self.profile.curves]
        return self._luts

    def display_linear(self, box=None, half: bool = False) -> np.ndarray:
        """Display-linear ProPhoto RGB (1.0 = the file's white) of a box (x, y, w, h); half = 2x2 averages."""
        x, y, w, h = box if box is not None else (0, 0, self.width, self.height)
        a = self.data[y:y + h, x:x + w]
        luts = self._lut()
        if a.ndim == 2:
            g = luts[0][a] if luts is not None else np.asarray(self.profile.curves[0](a.astype(np.float32)),
                                                                 np.float32)
            if half:
                g = _half(g[..., None])[..., 0]
            return np.repeat(g[..., None], 3, axis=2)       # grey (D50) = equal ProPhoto channels
        if luts is not None:
            lin = np.empty(a.shape, np.float32)
            for c in range(3):
                lin[..., c] = luts[c][a[..., c]]
        else:
            lin = np.stack([np.asarray(self.profile.curves[c](a[..., c].astype(np.float32)), np.float32)
                            for c in range(3)], axis=2)
        if half:
            lin = _half(lin)
        M = np.linalg.inv(sp.PROPHOTO.to_xyz_d50) @ self.profile.to_xyz_d50
        return sp.apply_matrix(lin, M)

    def embedded_jpeg(self) -> bytes | None:
        """A small sRGB JPEG of the scan, standing in for a raw's embedded preview (Library thumbnails)."""
        from PIL import Image as PI
        k = max(1, int(np.ceil(max(self.width, self.height) / THUMB_LONG)))
        lin = self.display_linear()[::k, ::k] if k > 1 else self.display_linear()
        if self.scene_linear:                       # scene light: put the default look on, as Develop shows it
            lin = scene_to_display(np.ascontiguousarray(lin))
        M = sp.conversion_matrix(sp.PROPHOTO, sp.SRGB)
        s = sp.srgb_encode(np.clip(sp.apply_matrix(np.ascontiguousarray(lin), M), 0.0, 1.0))
        buf = io.BytesIO()
        PI.fromarray((s * 255.0 + 0.5).astype(np.uint8)).save(buf, "JPEG", quality=88)
        return buf.getvalue()


def scene_to_display(rgb_pp: np.ndarray) -> np.ndarray:
    """Scene-linear ProPhoto -> display-linear ProPhoto through the default Studio Standard curve (per channel in
    the render space, as Develop applies it): the inverse of pipeline.scan_to_scene. For thumbnails of the
    stacking tools' results, which would look flat and dark shown as they are stored."""
    from ..color import tone as T
    from ..pipeline.edit import Tone
    t = Tone()
    r = sp.apply_matrix(np.ascontiguousarray(rgb_pp, np.float32), T.PP_TO_RENDER)
    r = np.asarray(T.standard_curve(np.maximum(r, 0.0), mid_in=t.mid_grey, contrast=t.contrast,
                                    shoulder=t.shoulder, white=t.white, toe=t.toe), np.float32)
    return sp.apply_matrix(np.clip(r, 0.0, 1.0), T.RENDER_TO_PP)


def _half(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w]
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def open_scan(path: str) -> ScanImage:
    """Decode a scan. Raises FileNotFoundError, UnsupportedScan, or the decoder's error for a broken file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    low = path.lower()
    with slog.stage("decode.scan", log, file=os.path.basename(path)):
        a, icc, orient = _read_tiff(path) if low.endswith((".tif", ".tiff")) else _read_pil(path)
        if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] != 3):
            raise UnsupportedScan(f"{os.path.basename(path)}: {a.shape} samples per pixel")
        if a.dtype.kind == "f":
            a = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        elif a.dtype not in (np.uint8, np.uint16):
            if a.dtype.kind in "ui":
                a = (a.astype(np.float64) / np.iinfo(a.dtype).max * 65535.0 + 0.5).astype(np.uint16)
            else:
                a = a.astype(np.float32)
        a = _upright(a, orient)
        grey = a.ndim == 2
        prof = profile_for(icc, grey)
        if a.dtype.kind == "f":                 # float TIFFs are linear light already
            prof = Profile(prof.name + " (linear float)", [lambda v: v] * (1 if grey else 3),
                           prof.to_xyz_d50, prof.source)
    try:
        meta = read_metadata(path, None)
    except Exception as e:
        log.warning("metadata: %s (%s); left empty", os.path.basename(path), e)
        meta = Metadata()
    meta.orientation = 1                        # the pixels are upright now
    h, w = a.shape[:2]
    bits = 32 if a.dtype.kind == "f" else (16 if a.dtype == np.uint16 else 8)
    img = ScanImage(path=path, meta=meta, data=a, profile=prof, height=h, width=w, crop=(0, 0, w, h), bits=bits,
                    grey=grey)
    if a.dtype.kind == "f" and low.endswith((".tif", ".tiff")):
        from .intermediate import read_info
        info = read_info(path)
        if info is not None:
            img.scene_linear = True                 # pipeline._scan_linear keeps the light as it is
            # exactly linear ProPhoto (the embedded profile says so, rounded to 1/65536 per primary)
            img.profile = Profile("ProPhoto RGB (linear)", [lambda v: v] * (1 if grey else 3), sp.PROPHOTO.to_xyz_d50,
                                  "icc")
            img.notes.append(f"Result of the Studio's {info.get('kind', 'stacking')} tool: scene-linear light.")
    log.info("scan %s: %dx%d, %d-bit %s, profile %s (%s)", os.path.basename(path), w, h, bits,
             "grey" if grey else "RGB", prof.name, prof.source)
    return img


def scan_preview(path: str, max_side: int | None = None) -> np.ndarray:
    """An upright sRGB uint8 preview of a scan, reduced quickly (JPEG draft mode, strided TIFFs) and colour
    managed with the file's profile: the Library thumbnail and the Review view's "file preview"."""
    from PIL import Image as PI
    low = path.lower()
    if low.endswith((".tif", ".tiff")):
        from .intermediate import read_info, read_preview
        if read_info(path) is not None:             # a stack result: it carries its own small preview
            pv = read_preview(path)
            if pv is not None:
                if max_side and max(pv.shape[:2]) > max_side:
                    im = PI.fromarray(pv)
                    im.thumbnail((max_side, max_side), PI.Resampling.BILINEAR)
                    pv = np.asarray(im)
                return np.ascontiguousarray(pv)
    if low.endswith((".jpg", ".jpeg")) and max_side:
        with PI.open(path) as im:
            icc = im.info.get("icc_profile")
            try:
                orient = int(im.getexif().get(274, 1) or 1)
            except Exception:
                orient = 1
            im.draft(im.mode if im.mode in ("L", "RGB") else "RGB", (max_side, max_side))
            a = np.asarray(im.convert("L") if im.mode == "L" else im.convert("RGB"))
    else:
        a, icc, orient = _read_tiff(path) if low.endswith((".tif", ".tiff")) else _read_pil(path)
    if max_side:
        k = max(1, int(max(a.shape[:2]) // max_side))
        a = a[::k, ::k]
    a = _upright(a, orient)
    grey = a.ndim == 2
    prof = profile_for(icc, grey)
    scene = False
    if a.dtype.kind == "f":
        v = np.maximum(a.astype(np.float32), 0.0)
        lin = v if grey else v
        from .intermediate import read_info
        scene = (not grey) and read_info(path) is not None
        if scene:                                  # a stack result: show it with the default look, as Develop does
            lin = scene_to_display(np.ascontiguousarray(lin))   # (lin is ProPhoto: its profile is linear ProPhoto)
            prof = Profile("ProPhoto RGB", [lambda x: x] * 3, sp.PROPHOTO.to_xyz_d50, "icc")
        else:
            lin = np.clip(lin, 0.0, 1.0)
    else:
        v = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
        lin = np.asarray(prof.curves[0](v), np.float32) if grey else \
            np.stack([np.asarray(prof.curves[c](v[..., c]), np.float32) for c in range(3)], axis=2)
    if grey:
        out = sp.srgb_encode(np.clip(lin, 0.0, 1.0))
        out = np.repeat(out[..., None], 3, axis=2)
    else:
        M = np.linalg.inv(sp.SRGB.to_xyz_d50) @ prof.to_xyz_d50
        out = sp.srgb_encode(np.clip(sp.apply_matrix(np.ascontiguousarray(lin), M), 0.0, 1.0))
    out = (out * 255.0 + 0.5).astype(np.uint8)
    if max_side and max(out.shape[:2]) > max_side:
        im = PI.fromarray(out)
        im.thumbnail((max_side, max_side), PI.Resampling.BILINEAR)
        out = np.asarray(im)
    return np.ascontiguousarray(out)
