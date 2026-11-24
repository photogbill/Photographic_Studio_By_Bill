"""FITS: the file format of astronomy (dedicated astro cameras, Siril, PixInsight, DeepSkyStacker, NASA archives).

Read through astropy (BSD-3, optional: ``tools/ensure_env.bat`` adds it once; the Studio starts without it). A FITS
file opens like a scan, as scene-linear light (pipeline._scan_linear keeps it as it is):

- one plane: greyscale; with a ``BAYERPAT`` card (a one-shot colour astro camera's raw): demosaiced (RCD);
- three planes (NAXIS3 = 3): RGB, linear sRGB primaries (what Siril and PixInsight store);
- whole numbers are scaled by their range (16-bit 0..65535 -> 0..1); floats above 1.5 (DeepSkyStacker writes
  0..65535) are scaled the same way; 0..1 floats are kept exactly;
- FITS rows run bottom-up unless ``ROWORDER = 'TOP-DOWN'``; they are turned upright here.

Written (``write``) as 32-bit float, planes first (NAXIS3 = 3 for colour), bottom-up with ``ROWORDER``, the
camera's details in the standard cards (EXPTIME, ISOSPEED, INSTRUME, DATE-OBS, FOCALLEN) and ``CREATOR`` carrying
the Studio's mark, so ``export.is_own_export`` knows its own files. No Qt here.
"""
from __future__ import annotations

import os

import numpy as np

from ... import log as slog

log = slog.get("engine.io.fits")

FITS_EXTENSIONS = (".fits", ".fit", ".fts")
MISSING = ("FITS files need the optional astronomy package (astropy). Run Setup again with the internet on, or "
           "restart the Studio, and it is added once by itself.")
_PAT = {"R": 0, "G": 1, "B": 2}


def available() -> bool:
    try:
        import astropy.io.fits  # noqa: F401
        return True
    except Exception:            # noqa: BLE001  (missing or broken optional package: FITS is simply not offered)
        return False


def is_fits(path: str) -> bool:
    return str(path).lower().endswith(FITS_EXTENSIONS)


class FitsProblem(Exception):
    """A FITS file the Studio cannot show (no picture in it, a cube, astropy missing). Said where it happens."""


def _first_image(hdul):
    for hdu in hdul:
        if getattr(hdu, "data", None) is not None and getattr(hdu, "is_image", False) and hdu.data.ndim >= 2:
            return hdu
    raise FitsProblem("no picture in this FITS file (only tables)")


def _scale(a: np.ndarray, own: bool = False) -> tuple[np.ndarray, float]:
    """float32 with the file's white at 1.0, and the divisor used. Floats are kept as they are when the file is
    the Studio's own or most of the picture is under 1.5 (values above 1 are real light, not a 0..65535 scale)."""
    if a.dtype.kind in "ui":
        top = float(np.iinfo(a.dtype).max) if a.dtype.itemsize <= 2 else float(max(a.max(), 1))
        return a.astype(np.float32) / top, top
    a = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    v = a.ravel()[:: max(1, a.size // 200_000)]
    hi = float(np.percentile(v, 99.99)) if v.size else 0.0
    if not own and v.size and float(np.median(v)) > 1.5:
        top = 65535.0 if hi <= 65535.0 * 1.01 else float(a.max())
        return a / top, top
    return a, 1.0


def bayer_pattern(header) -> np.ndarray | None:
    """The 2x2 CFA pattern (0 R, 1 G, 2 B) of a one-shot colour camera's raw FITS, offsets applied, or None."""
    p = str(header.get("BAYERPAT", "") or "").strip().upper()
    if len(p) != 4 or any(c not in _PAT for c in p) or sorted(p) != ["B", "G", "G", "R"]:
        return None
    m = np.array([[_PAT[p[0]], _PAT[p[1]]], [_PAT[p[2]], _PAT[p[3]]]])
    dy, dx = int(header.get("YBAYROFF", 0) or 0) % 2, int(header.get("XBAYROFF", 0) or 0) % 2
    return np.roll(m, (-dy, -dx), axis=(0, 1))


def read(path: str) -> tuple[np.ndarray, dict, dict]:
    """(pixels, header cards, info): float32 upright (h, w) or (h, w, 3), the white at 1.0 (scene-linear).
    info: {'bayer': pattern or None, 'scale': divisor, 'planes': n}. Raises FitsProblem, FileNotFoundError."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if not available():
        raise FitsProblem(MISSING)
    from astropy.io import fits as F
    with slog.stage("decode.fits", log, file=os.path.basename(path)):
        with F.open(path, memmap=False) as hdul:
            hdu = _first_image(hdul)
            a = np.asarray(hdu.data)
            hdr = hdu.header
            cards = {k: hdr[k] for k in hdr.keys() if k and k not in ("COMMENT", "HISTORY", "")}
        top_down = str(cards.get("ROWORDER", "BOTTOM-UP")).strip().upper() == "TOP-DOWN"
        if a.ndim == 3:
            if a.shape[0] == 1:
                a = a[0]
            elif a.shape[0] == 3:
                a = np.moveaxis(a, 0, -1)
            elif a.shape[-1] == 3:
                pass
            else:
                raise FitsProblem(f"{a.shape[0]} planes: a data cube, not a photo (open one plane in Siril)")
        elif a.ndim != 2:
            raise FitsProblem(f"{a.ndim}-dimensional data, not a photo")
        if a.dtype.byteorder not in ("=", "|"):
            a = a.astype(a.dtype.newbyteorder("="))
        if not top_down:
            a = a[::-1]                                  # FITS row 0 is the bottom of the picture
        from .export import OWN_MARK
        a, top = _scale(np.ascontiguousarray(a), str(cards.get("CREATOR", "")).startswith(OWN_MARK))
        bayer = bayer_pattern(cards) if a.ndim == 2 else None
        if bayer is not None:
            if not top_down and a.shape[0] % 2 == 0:     # turned upright: an even height swaps the pattern rows
                bayer = np.roll(bayer, 1, axis=0)
            from ..raw import demosaic as D
            a = np.maximum(D.demosaic(np.maximum(a, 0.0), bayer, "rcd"), 0.0).astype(np.float32)
    log.info("fits %s: %s, %s, scale 1/%g%s", os.path.basename(path), "x".join(map(str, a.shape)), a.dtype, top,
             f", CFA {bayer.tolist()}" if bayer is not None else "")
    return np.ascontiguousarray(a, np.float32), cards, {"bayer": bayer, "scale": top,
                                                         "planes": 1 if a.ndim == 2 else 3}


def metadata(cards: dict):
    """The Studio's Metadata from the standard FITS cards (what astro cameras and capture programs write)."""
    from .metadata import Metadata
    m = Metadata()
    m.model = str(cards.get("INSTRUME", "") or "").strip()
    m.make = str(cards.get("CAMMAKER", "") or "").strip()
    for k in ("EXPTIME", "EXPOSURE"):
        if cards.get(k) not in (None, ""):
            try:
                m.exposure_s = float(cards[k])
                break
            except (TypeError, ValueError):
                pass
    for k in ("ISOSPEED", "ISO"):
        if cards.get(k) not in (None, ""):
            try:
                m.iso = float(cards[k])
                m.iso_source = "FITS"
                break
            except (TypeError, ValueError):
                pass
    try:
        if cards.get("FOCALLEN") not in (None, ""):
            m.focal_mm = float(cards["FOCALLEN"])
    except (TypeError, ValueError):
        pass
    try:
        if cards.get("FOCRATIO") not in (None, ""):
            m.fnumber = float(cards["FOCRATIO"])
    except (TypeError, ValueError):
        pass
    d = str(cards.get("DATE-OBS", "") or "").strip()
    if d:
        m.datetime = d.replace("T", " ")[:19].replace("-", ":", 2)
    m.lens = str(cards.get("TELESCOP", "") or "").strip()
    try:
        if cards.get("CCD-TEMP") not in (None, ""):
            m.camera_temperature_c = float(cards["CCD-TEMP"])
    except (TypeError, ValueError):
        pass
    return m


def autostretch(lum: np.ndarray, target: float = 0.25) -> tuple[float, float]:
    """(black, midtone) of a screen stretch that puts the sky background at ``target``: the black point 2.8 MADs
    below the median, the midtone balance solved for the median (the usual astronomy screen stretch)."""
    v = np.asarray(lum, np.float32).ravel()
    if v.size > 400_000:
        v = v[:: v.size // 400_000]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    black = max(0.0, med - 2.8 * mad)
    x = min(max((med - black) / max(1.0 - black, 1e-9), 1e-6), 0.999)
    # MTF(x, m) = (m - 1) x / ((2m - 1) x - m) = target  ->  m
    m = x * (target - 1.0) / (2.0 * x * target - x - target)
    return black, float(min(max(m, 1e-4), 0.9999))


def stretch(a: np.ndarray, black: float, m: float) -> np.ndarray:
    x = np.clip((a - black) / max(1.0 - black, 1e-9), 0.0, 1.0)
    return ((m - 1.0) * x / ((2.0 * m - 1.0) * x - m)).astype(np.float32)


def preview(path: str, max_side: int | None = None) -> np.ndarray:
    """An upright sRGB uint8 preview with the astronomy screen stretch (linear astro data is nearly black shown as
    it is): the Library card and Review's file preview."""
    from ..color import spaces as sp
    a, _, _ = read(path)
    if max_side:
        k = max(1, int(max(a.shape[:2]) // max_side))
        a = a[::k, ::k]
    lum = a if a.ndim == 2 else a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    black, m = autostretch(lum)
    s = stretch(a, black, m)
    if s.ndim == 2:
        s = np.repeat(s[..., None], 3, axis=2)
    out = (np.clip(sp.srgb_encode(np.clip(s, 0.0, 1.0)), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    if max_side and max(out.shape[:2]) > max_side:
        from PIL import Image as PI
        im = PI.fromarray(out)
        im.thumbnail((max_side, max_side), PI.Resampling.BILINEAR)
        out = np.asarray(im)
    return np.ascontiguousarray(out)


def size(path: str) -> tuple[int, int]:
    """(width, height) from the header alone."""
    if not available():
        raise FitsProblem(MISSING)
    from astropy.io import fits as F
    with F.open(path, memmap=True) as hdul:
        hdu = _first_image(hdul)
        h = hdu.header
        return int(h["NAXIS1"]), int(h["NAXIS2"])


def is_own(path: str) -> bool:
    if not available():
        return False
    from astropy.io import fits as F
    from .export import OWN_MARK
    try:
        with F.open(path, memmap=True) as hdul:
            return str(hdul[0].header.get("CREATOR", "")).startswith(OWN_MARK)
    except Exception:            # noqa: BLE001  (somebody else's or broken file: not ours)
        return False


def write(path: str, arr: np.ndarray, meta: dict | None = None, extra: dict | None = None) -> str:
    """32-bit float FITS of (h, w) or (h, w, 3) linear values, written through NAME.partial. Returns ``path``."""
    if not available():
        raise FitsProblem(MISSING)
    from astropy.io import fits as F
    from ... import __version__
    from .export import OWN_MARK
    a = np.asarray(arr, np.float32)
    if a.ndim == 3:
        data = np.ascontiguousarray(np.moveaxis(a[::-1], -1, 0))
    else:
        data = np.ascontiguousarray(a[::-1])
    hdu = F.PrimaryHDU(data)
    h = hdu.header
    h["ROWORDER"] = ("BOTTOM-UP", "FITS order: first row is the bottom")
    h["CREATOR"] = (f"{OWN_MARK} {__version__}"[:68], "software that wrote this file")
    meta = meta or {}
    if meta.get("model"):
        h["INSTRUME"] = str(meta.get("model"))[:68]
    if meta.get("make"):
        h["CAMMAKER"] = str(meta.get("make"))[:68]
    if meta.get("exposure_s"):
        h["EXPTIME"] = (float(meta["exposure_s"]), "[s] exposure of one frame")
    if meta.get("iso"):
        h["ISOSPEED"] = float(meta["iso"])
    if meta.get("focal_mm"):
        h["FOCALLEN"] = (float(meta["focal_mm"]), "[mm]")
    if meta.get("fnumber"):
        h["FOCRATIO"] = float(meta["fnumber"])
    if meta.get("lens"):
        h["TELESCOP"] = str(meta["lens"])[:68]
    dt = str(meta.get("datetime") or "")
    if len(dt) >= 19:
        h["DATE-OBS"] = dt[:10].replace(":", "-") + "T" + dt[11:19]
    for k, v in (extra or {}).items():
        try:
            h[str(k)[:8].upper()] = v
        except (ValueError, TypeError) as e:
            log.info("fits: card %s left out (%s)", k, e)
    tmp = path + ".partial"
    hdu.writeto(tmp, overwrite=True, output_verify="silentfix")
    os.replace(tmp, path)
    log.info("wrote FITS %s (%s)", os.path.basename(path), "x".join(map(str, a.shape)))
    return path
