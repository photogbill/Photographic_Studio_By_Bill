"""Exporting a painting as one flat picture (BUILD_PLAN_PAINT P1): PNG, TIFF, JPEG and WebP, with a colour profile.

Every export carries its ICC profile (engine/color/icc.py) and the Studio's mark, is named ``NAME_Studio.ext`` and
never replaces a file the Studio did not write (io/export.py: ``free_target``, ``is_own_export``).

The canvas holds sRGB. Choosing a wider profile converts the colours into it exactly (same colours, a different
description): useful when a print shop or a layout program asks for Adobe RGB or Display P3. It cannot add colours
the canvas never had.
"""
from __future__ import annotations

import os

import numpy as np

from ... import __version__
from ..color import icc
from ..color.spaces import SPACES, SRGB, apply_matrix, conversion_matrix, srgb_decode
from ..io.export import OWN_MARK, write_png_any
from .tiles import Document

# key -> (name, extension, best for, weak at)
FORMATS = {
    "png": ("PNG", ".png", "sharing with see-through areas kept, exact colours, no loss", "large files"),
    "tiff": ("TIFF", ".tif", "print shops and layout programs; 16-bit keeps smooth gradients", "large files; "
             "browsers do not show it"),
    "jpeg": ("JPEG", ".jpg", "the web and phones, everything opens it", "no see-through (the paper colour, or "
             "white, fills it) and a little loss"),
    "webp": ("WebP", ".webp", "the web: smaller than JPEG at the same look, and keeps see-through areas",
             "some older programs and printers cannot open it; 8-bit only"),
    "exr": ("OpenEXR", ".exr", "3D, games and HDR: floating point, linear light, values above white kept",
            "photo programs and the web do not show it; big files"),
}
# key -> (name, best for, weak at)
PROFILES = {
    "srgb": ("sRGB", "the web, phones, most printers: what every program assumes", "the narrowest of the four"),
    "displayp3": ("Display P3", "Apple screens and modern phones", "programs that ignore profiles show it dull"),
    "adobergb": ("Adobe RGB", "print shops that ask for it", "programs that ignore profiles show it dull"),
    "prophoto": ("ProPhoto RGB", "keeping every colour through later editing (use 16-bit)", "8-bit files band; "
                 "programs that ignore profiles show it very dull"),
}


def flat_pixels(doc: Document, space: str = "srgb", alpha: bool = True, light=None) -> np.ndarray:
    """The picture as straight (not premultiplied) float RGBA 0..1 encoded in ``space``; ``light`` lights the paint's
    relief (impasto) as the canvas does."""
    px = doc.composite(background=True, light=light)
    a = px[..., 3:4]
    rgb = np.where(a > 0, px[..., :3] / np.maximum(a, 1e-8), 0).astype(np.float32)
    if doc.space != "srgb":                                  # the painting's own working space -> sRGB first
        from .space import convert
        rgb = np.clip(convert(rgb, doc.space, "srgb"), 0, 1)
    if not alpha:                                       # see-through areas over white (JPEG)
        rgb = rgb * a + (1 - a)
        a = np.ones_like(a)
    if space != "srgb":
        dst = SPACES[space]
        lin = srgb_decode(np.clip(rgb, 0, 1)).astype(np.float32)
        lin = apply_matrix(lin, conversion_matrix(SRGB, dst).astype(np.float32))
        rgb = dst.encode(np.clip(lin, 0, 1)).astype(np.float32)
    return np.concatenate([np.clip(rgb, 0, 1), a], -1)


def _quant(f, bits):
    if bits == 16:
        return np.clip(np.rint(f * 65535), 0, 65535).astype(np.uint16)
    return np.clip(np.rint(f * 255), 0, 255).astype(np.uint8)


def export(doc: Document, path: str, fmt: str = "png", space: str = "srgb", bits: int = 8, quality: int = 92,
           lossless: bool = False, light=None) -> str:
    """Write the flat picture to ``path`` (written through ``.partial`` then swapped in). Returns the path."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {tuple(FORMATS)}")
    if space not in PROFILES:
        raise ValueError(f"profile must be one of {tuple(PROFILES)}")
    if fmt == "exr":
        from . import exr
        from .space import convert
        px = doc.composite(background=True, light=light)
        a = px[..., 3:4]
        rgb = np.where(a > 0, px[..., :3] / np.maximum(a, 1e-8), 0)
        lin = convert(rgb, doc.space, "linear")             # linear sRGB primaries, values above 1 kept
        return exr.save(path, np.concatenate([lin * a, a], -1).astype(np.float32))
    if fmt in ("jpeg", "webp"):
        bits = 8
    prof = icc.profile_bytes(space)
    keep_alpha = fmt != "jpeg"
    f = flat_pixels(doc, space, keep_alpha, light)
    if not keep_alpha or f[..., 3].min() >= 1.0:
        f = f[..., :3]                                   # fully opaque: no alpha channel needed
    arr = _quant(f, bits)
    meta = {"software": f"{OWN_MARK} {__version__}"}
    if fmt == "png":
        return write_png_any(path, arr, prof, meta)
    tmp = path + ".partial"
    if fmt == "tiff":
        import tifffile
        extra = [(34675, "B", len(prof), prof, True)]
        kw = {"extrasamples": ["unassalpha"]} if arr.shape[-1] == 4 else {}
        tifffile.imwrite(tmp, arr, photometric="rgb", planarconfig="contig", software=meta["software"],
                         resolution=(300, 300), resolutionunit="INCH", extratags=extra, metadata=None,
                         compression="zlib", **kw)
    else:
        from PIL import Image
        im = Image.fromarray(arr)
        exif = Image.Exif()
        exif[305] = meta["software"]
        if fmt == "jpeg":
            im.save(tmp, "JPEG", quality=int(quality), subsampling=0, optimize=True, icc_profile=prof,
                    exif=exif.tobytes())
        else:
            im.save(tmp, "WEBP", quality=int(quality), lossless=bool(lossless), method=4, icc_profile=prof,
                    exif=exif.tobytes())
    os.replace(tmp, path)
    return path
