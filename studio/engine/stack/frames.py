"""A set of frames for the stacking tools: probing them, one shared edit, streaming them (BUILD_PLAN_TABS 1.4).

Frames are never all in memory as RGB at once. A 45 MP frame is 540 MB as float32 RGB; the tools decode one
frame, take what they need (an analysis copy, a contribution to a running sum) and let it go. Raw mosaics for the
Super-Res drizzle are kept as int16 (90 MB a frame) and, for a large set, in a memory-mapped file under the
app's cache folder (``Scratch``), deleted when the job ends.

Every frame gets the same development (``stack_edit``): the white balance fixed to the reference frame's
as-shot multipliers, the lens's colour fringing corrected, no sharpening, a neutral highlight clip. Same settings
on every frame is what makes them comparable. No Qt here (rule 2.2).
"""
from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass, field

import numpy as np

from ... import log as slog

log = slog.get("engine.stack.frames")

INT16_SCALE = 30000.0          # stored mosaic value = normalised value * this (1.0 = sensor clip)


class StackProblem(ValueError):
    """These frames cannot make a stack, said in plain words (different cameras, too few readable frames...).
    Not a fault in the Studio: no error report."""


@dataclass
class FrameInfo:
    path: str
    name: str
    width: int = 0              # 0 = not known until the frame is opened
    height: int = 0
    camera: str = ""
    iso: float | None = None
    exposure_s: float | None = None
    fnumber: float | None = None
    focal_mm: float | None = None
    taken: str = ""             # 'YYYY:MM:DD HH:MM:SS' or ''
    flip: int = 0
    ok: bool = True
    problem: str = ""
    is_raw: bool = True
    focus_mode: str = ""        # what the camera says: one-shot | servo | ai-focus | manual | "" (not recorded)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _is_raw(path: str) -> bool:
    from ..io.rawfile import RAW_EXTENSIONS
    return str(path).lower().endswith(tuple(RAW_EXTENSIONS))


def supported(path: str) -> bool:
    """A file the stacking tools can take: a raw, or a TIFF / JPEG / PNG (developed frames, scans)."""
    from ..io.scanfile import SCAN_EXTENSIONS
    return _is_raw(path) or str(path).lower().endswith(SCAN_EXTENSIONS)


def probe(paths) -> list[FrameInfo]:
    """Metadata of every frame (512 KB read per file, no decoding), sorted by capture time then name. Files the
    tools cannot take are kept in the list with ok=False and the reason, so the page can show them."""
    from ..io.metadata import read_metadata
    out = []
    for p in paths:
        p = os.fspath(p)
        name = os.path.basename(p)
        fi = FrameInfo(path=p, name=name, is_raw=_is_raw(p))
        if not os.path.isfile(p):
            fi.ok, fi.problem = False, "file not found"
        elif not supported(p):
            fi.ok, fi.problem = False, "not a raw, TIFF, JPEG or PNG"
        else:
            try:
                md = read_metadata(p, None, head_bytes=512 * 1024)
                fi.camera = " ".join(x for x in (md.make, md.model) if x).strip()
                fi.iso, fi.exposure_s, fi.fnumber, fi.focal_mm = md.iso, md.exposure_s, md.fnumber, md.focal_mm
                fi.taken = (md.datetime or "").strip("\x00 ")
                fi.focus_mode = md.focus_mode
                si = md.sensor_info or {}
                if si.get("right") and si.get("bottom"):
                    fi.width = int(si["right"]) - int(si.get("left", 0)) + 1
                    fi.height = int(si["bottom"]) - int(si.get("top", 0)) + 1
            except Exception as e:                       # noqa: BLE001  (metadata is a nicety; the decode decides)
                log.info("probe %s: metadata unreadable (%s)", name, e)
        out.append(fi)
    out.sort(key=lambda f: (f.taken or "~", f.name.lower()))
    return out


def check(frames: list[FrameInfo]) -> tuple[list[str], bool]:
    """(plain-language warnings, blocking). Different cameras or sizes block a build; different exposure, ISO or
    aperture only warn (a focus stack and a burst expect the same, but it can still work)."""
    good = [f for f in frames if f.ok]
    msgs: list[str] = []
    block = False
    bad = [f for f in frames if not f.ok]
    if bad:
        msgs.append(f"{len(bad)} file(s) left out: " + "; ".join(f"{f.name} ({f.problem})" for f in bad[:4]) +
                    (" ..." if len(bad) > 4 else ""))
    if len(good) < 2:
        msgs.append("Add at least two frames.")
        return msgs, True
    cams = {f.camera for f in good if f.camera}
    if len(cams) > 1:
        msgs.append("These frames come from different cameras (" + ", ".join(sorted(cams)) + "): a stack needs "
                    "one camera.")
        block = True
    sizes = {(f.width, f.height) for f in good if f.width and f.height}
    if len(sizes) > 1:
        msgs.append("These frames are not all the same size: a stack needs one size.")
        block = True
    kinds = {f.is_raw for f in good}
    if len(kinds) > 1:
        msgs.append("Raws and developed files (TIFF / JPEG) are mixed: use one kind.")
        block = True

    def spread(vals, ratio):
        v = [x for x in vals if x]
        return len(v) > 1 and max(v) / min(v) > ratio
    if spread([f.exposure_s for f in good], 1.12):
        msgs.append("The shutter speed changes between frames: brightness will be evened out, but noise differs.")
    if spread([f.iso for f in good], 1.12):
        msgs.append("The ISO changes between frames: brightness will be evened out, but noise differs.")
    if spread([f.fnumber for f in good], 1.06):
        msgs.append("The aperture changes between frames: depth of field and brightness differ.")
    return msgs, block


def stack_edit(reference, denoise: bool = False):
    """ONE edit for every frame of a stack, from the reference frame (an engine Image): white balance fixed to the
    reference's as-shot multipliers, lateral CA and fringe correction on, no sharpening, a neutral clip, the
    vignette left as it is, denoise on or off as asked."""
    from ..pipeline.edit import Edit
    e = Edit()
    e.white_balance.mode = "multipliers"
    m = np.asarray(reference.raw.wb_as_shot, float)[:3]
    e.white_balance.multipliers = (m / m[1]).tolist()
    e.lens.auto_ca = True
    e.lens.defringe = True
    e.lens.vignette = False
    e.sharpen.enabled = False
    e.highlight_mode = "clip"
    e.denoise.enabled = bool(denoise)
    return e


def scene_rgb(path: str, edit, half: bool, region=None) -> np.ndarray:
    """Stages 1-12 of one frame: scene-linear ProPhoto (1.0 = sensor clip of a neutral), camera-crop frame, not yet
    turned upright; ``region`` (x, y, w, h, camera-crop pixels, even origin) develops only that part. The opened
    Image is dropped at once (frames are streamed, never kept)."""
    from .. import api
    from ..pipeline.pipeline import linear_rgb
    img = api.open(path)
    try:
        return np.ascontiguousarray(linear_rgb(img, edit, region, half_size=half, cache=False), np.float32)
    finally:
        img.drop_cache()
        del img


def open_frame(path: str):
    """The engine Image of one frame (for the reference: its white balance, flip, noise model)."""
    from .. import api
    return api.open(path)


def is_dual(path: str) -> bool:
    """True when the frame is a Dual Pixel RAW (a CR3 is told from its box structure alone, cheaply; a CR2 needs
    one decode). Anything else, or anything unreadable: False (the frame is then used as an ordinary frame)."""
    if not str(path).lower().endswith((".cr3", ".cr2")):
        return False
    from ..dualpixel.decode import is_dual_pixel
    try:
        return bool(is_dual_pixel(path))
    except Exception as e:                           # noqa: BLE001  (not readable as Dual Pixel: an ordinary frame)
        log.info("is_dual %s: %s", os.path.basename(path), e)
        return False


def dual_halves(path: str) -> tuple[np.ndarray, np.ndarray]:
    """The two halves' luminance of a Dual Pixel RAW frame (half size, camera-crop frame, shading corrected: the
    same frame as scene_rgb(path, edit, half=True)). Both frames are decoded and let go at once."""
    from ..dualpixel import decode as DD, depth as DEP
    dp = DD.open_dual(path)
    try:
        return DEP.halves_luma(dp, DD.shading(dp))
    finally:
        try:
            dp.image.drop_cache()
        except Exception:                            # noqa: BLE001  (only a cache)
            pass
        del dp


@dataclass
class RawInfo:
    """What the drizzle needs from a frame besides its mosaic."""
    pattern: np.ndarray                      # 2x2 of 0/1/2 (R/G/B) at the camera crop's origin
    cam_xyz: np.ndarray
    flip: int
    wb: np.ndarray                           # as-shot multipliers R, G, B (G = 1)
    noise_a: np.ndarray                      # noise variance = a * signal + b per CFA position (normalised units)
    noise_b: np.ndarray
    shape: tuple = (0, 0)                    # (h, w) of the camera crop
    meta: dict = field(default_factory=dict)
    crop_origin: tuple = (0, 0)              # (y, x) of the camera crop in the visible area
    visible_shape: tuple = (0, 0)            # (h, w) of the visible area (the lens's centre is its centre)


def mosaic(path: str, noise: bool = False) -> tuple[np.ndarray, RawInfo]:
    """The normalised mosaic of a raw frame (black subtracted, 1.0 = clip), cropped to the camera crop, as int16
    (value * INT16_SCALE), with the pattern, colour matrix, orientation, white balance and (``noise``) the noise
    model (a second or two per frame: only asked for when needed)."""
    from .. import api
    from ..raw import linearize as lin
    img = api.open(path)
    raw = img.raw
    if getattr(raw, "is_scan", False):
        raise ValueError(f"{os.path.basename(path)} is not a raw file: drizzle needs raw frames")
    lv = img.levels(True)
    cx, cy, cw, ch = raw.crop
    cw, ch = cw // 2 * 2, ch // 2 * 2
    mos = lin.normalise(raw, lv, (cx, cy, cw, ch))
    out = np.empty(mos.shape, np.int16)
    for y in range(0, mos.shape[0], 512):
        blk = np.clip(mos[y:y + 512] * np.float32(INT16_SCALE), -32767, 32767)
        out[y:y + 512] = np.rint(blk).astype(np.int16)
    del mos
    a, b = np.full((2, 2), 2e-4), np.full((2, 2), 1e-6)
    if noise:
        try:
            nm = img.noise_model(True)
            a = np.asarray(nm.a, float).reshape(2, 2)
            b = np.asarray(nm.b, float).reshape(2, 2)
        except Exception as e:                       # noqa: BLE001  (a default model still lets the merge run)
            log.info("noise model for %s unavailable (%s): a default is used", os.path.basename(path), e)
    md = raw.meta
    info = RawInfo(pattern=np.asarray(raw.cfa_rgb).copy(), cam_xyz=np.asarray(raw.cam_xyz, float).copy(),
                   flip=int(raw.flip), wb=np.asarray(raw.wb_as_shot, float).copy(), noise_a=a, noise_b=b,
                   shape=out.shape, meta={"make": md.make, "model": md.model, "iso": md.iso,
                                          "exposure_s": md.exposure_s, "fnumber": md.fnumber,
                                          "focal_mm": md.focal_mm, "lens": md.lens, "datetime": md.datetime},
                   crop_origin=(int(cy), int(cx)), visible_shape=(int(raw.height), int(raw.width)))
    del img
    return out, info


def from_int16(m: np.ndarray) -> np.ndarray:
    return m.astype(np.float32) * np.float32(1.0 / INT16_SCALE)


class Scratch:
    """A private folder under the app's cache for one job's memory-mapped arrays; removed by ``close()`` (also on
    error or Cancel: use it as a context manager)."""

    def __init__(self, tag: str = "stack"):
        from ... import paths
        base = paths.temp_dir()
        base.mkdir(parents=True, exist_ok=True)       # temp_dir() does not create it
        self.dir = os.path.join(str(base), f"{tag}_{uuid.uuid4().hex[:10]}")
        os.makedirs(self.dir, exist_ok=True)
        self._maps: list = []

    def array(self, name: str, shape, dtype=np.float32) -> np.ndarray:
        p = os.path.join(self.dir, name + ".dat")
        a = np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=tuple(shape))
        self._maps.append(a)
        return a

    def drop(self, a: np.ndarray):
        """Release one array early and delete its file (frees disk while the job goes on)."""
        try:
            fn = a.filename
            a._mmap.close()                           # noqa: SLF001
            self._maps = [x for x in self._maps if x is not a]
            os.remove(fn)
        except Exception:                             # noqa: BLE001  (removed with the folder at the end anyway)
            pass

    def close(self):
        for a in self._maps:
            try:
                a._mmap.close()                       # noqa: SLF001  (release the file before removing it)
            except Exception:                         # noqa: BLE001
                pass
        self._maps = []
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def default_out_dir(first_source: str) -> tuple[str, str]:
    """(folder, note): beside the first source frame; if that folder cannot be written, the app's
    outputs/stacks folder, and a note that says so."""
    from ... import paths
    d = os.path.dirname(os.path.abspath(first_source))
    if os.path.isdir(d) and os.access(d, os.W_OK):
        return d, ""
    alt = paths.get("OUTPUTS") / "stacks"
    alt.mkdir(parents=True, exist_ok=True)
    return str(alt), f"The photos' folder cannot be written; the result goes to {alt}."


def result_base(paths_: list) -> str:
    """'<first stem>_to_<last stem>' (one frame: '<stem>'): the start of a stack result's name."""
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths_]
    return stems[0] if len(stems) == 1 or stems[0] == stems[-1] else f"{stems[0]}_to_{stems[-1]}"


def result_name(paths_: list, tag: str, ext: str = ".tif") -> str:
    """'<first stem>_to_<last stem>_<tag>_Studio.tif' (one frame: '<stem>_<tag>_Studio.tif')."""
    return f"{result_base(paths_)}_{tag}_Studio{ext}"
