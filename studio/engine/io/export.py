"""Stage 21: output. Colour-space conversion, encoding, quantisation, metadata, write.

Formats: 16-bit TIFF (default), 8/16-bit PNG, JPEG. Spaces: Adobe RGB (default), sRGB, ProPhoto,
Display P3. The matching ICC profile (icc.py) is always embedded.

Metadata policy (section 15): camera make/model, date, exposure, the true ISO and the lens (the user's tag for
contactless glass) are written (EXIF in JPEG, XMP in TIFF and PNG); serial numbers and GPS are never written
(we never read them in the first place).
"""
from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from ... import __version__, log as slog
from ..color import gamut, icc
from ..color.spaces import PROPHOTO, SPACES, apply_matrix, conversion_matrix

log = slog.get("engine.export")

FORMATS = ("tiff", "png", "jpeg")
EXT = {"tiff": ".tif", "png": ".png", "jpeg": ".jpg"}


@dataclass
class ExportSpec:
    format: str = "tiff"            # tiff | png | jpeg
    space: str = "adobergb"         # adobergb | srgb | prophoto | displayp3
    bits: int = 16                  # 16 or 8 (JPEG is always 8)
    quality: int = 95               # JPEG quality
    long_edge: int | None = None    # resize so the long edge is this many pixels
    compression: str = "none"       # TIFF: none | zip
    suffix: str = "_Studio"         # added to the photo's own name: IMG_0001.CR2 -> IMG_0001_Studio.tif
    gamut_compress: bool = True     # compress out-of-gamut colours smoothly instead of hard clipping
    sharpen_for: str = "none"       # output sharpening: none | screen | glossy | matte (after any resize)
    sharpen_amount: str = "standard"   # low | standard | high

    def validate(self):
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}")
        if self.space not in SPACES or self.space == "rec2020":
            raise ValueError("space must be adobergb, srgb, prophoto or displayp3")
        if self.format == "jpeg":
            self.bits = 8
        if self.bits not in (8, 16):
            raise ValueError("bits must be 8 or 16")
        if self.sharpen_for not in OUTPUT_SHARPEN:
            raise ValueError(f"sharpen_for must be one of {tuple(OUTPUT_SHARPEN)}")
        if self.sharpen_amount not in SHARPEN_AMOUNT:
            raise ValueError(f"sharpen_amount must be one of {tuple(SHARPEN_AMOUNT)}")


# Output sharpening for the medium: (radius in output pixels, strength). Prints assume about 300 ppi.
OUTPUT_SHARPEN = {"none": (0.0, 0.0), "screen": (0.6, 0.45), "glossy": (0.9, 0.65), "matte": (1.2, 0.9)}
SHARPEN_AMOUNT = {"low": 0.6, "standard": 1.0, "high": 1.5}


def output_sharpen(img_pp: np.ndarray, spec: "ExportSpec") -> np.ndarray:
    """Unsharp mask on log luminance (no colour fringes), sized for the medium; differences are clamped to half a
    stop so edges never get bright or dark rims. Display-referred linear ProPhoto in and out."""
    r, k = OUTPUT_SHARPEN.get(spec.sharpen_for, (0.0, 0.0))
    if r <= 0 or k <= 0:
        return img_pp
    from scipy.ndimage import gaussian_filter
    k *= SHARPEN_AMOUNT.get(spec.sharpen_amount, 1.0)
    with slog.stage("output.sharpen", log, medium=spec.sharpen_for, amount=spec.sharpen_amount):
        Y = np.maximum(img_pp @ np.array([0.2880, 0.7119, 0.0001], np.float32), 1e-6)
        L = np.log2(Y)
        d = np.clip(L - gaussian_filter(L, r), -0.5, 0.5)
        return (img_pp * np.exp2(np.float32(k) * d)[..., None]).astype(np.float32)


def to_output(img_pp: np.ndarray, spec: ExportSpec, dither_seed: int = 0) -> np.ndarray:
    """Display-referred linear ProPhoto -> encoded integer array in the output space."""
    space = SPACES[spec.space]
    M = conversion_matrix(PROPHOTO, space).astype(np.float32)
    h = img_pp.shape[0]
    dtype = np.uint16 if spec.bits == 16 else np.uint8
    maxv = 65535.0 if spec.bits == 16 else 255.0
    out = np.empty(img_pp.shape, dtype)
    rng = np.random.default_rng(dither_seed)     # seeded: exports are deterministic
    for y in range(0, h, 512):
        s = apply_matrix(img_pp[y:y + 512], M)
        if spec.gamut_compress:
            s = gamut.compress(s)
        np.clip(s, 0.0, 1.0, out=s)
        e = space.encode(s).astype(np.float32) * np.float32(maxv)
        if spec.bits == 8:                        # triangular dither hides banding in smooth skies
            e += (rng.random(e.shape, np.float32) - rng.random(e.shape, np.float32))
        out[y:y + 512] = np.clip(np.rint(e), 0, maxv).astype(dtype)
    return out


def _rational(v: float, den: int = 10000) -> tuple:
    from fractions import Fraction
    f = Fraction(float(v)).limit_denominator(den)
    return f.numerator, f.denominator


def _xml(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def xmp_packet(meta: dict) -> bytes:
    """XMP for TIFF (tag 700) and PNG (iTXt): camera, exposure, true ISO and lens - what Photoshop, Lightroom and
    ExifTool read. EXIF 2.3 PhotographicSensitivity carries ISOs above 65535 (ISO 102400 stays 102400)."""
    attrs = [f'xmp:CreatorTool="Photographic Studio By Bill {_xml(__version__)}"']
    if meta.get("make"):
        attrs.append(f'tiff:Make="{_xml(meta["make"])}"')
    if meta.get("model"):
        attrs.append(f'tiff:Model="{_xml(meta["model"])}"')
    if meta.get("exposure_s"):
        n, d = _rational(meta["exposure_s"], 100000)
        attrs.append(f'exif:ExposureTime="{n}/{d}"')
    if meta.get("fnumber"):
        n, d = _rational(meta["fnumber"], 100)
        attrs.append(f'exif:FNumber="{n}/{d}"')
    if meta.get("focal_mm"):
        n, d = _rational(meta["focal_mm"], 100)
        attrs.append(f'exif:FocalLength="{n}/{d}"')
    if meta.get("lens"):
        attrs.append(f'aux:Lens="{_xml(meta["lens"])}"')
        attrs.append(f'exifEX:LensModel="{_xml(meta["lens"])}"')
    iso = meta.get("iso")
    seq = ""
    if iso:
        attrs.append(f'exifEX:PhotographicSensitivity="{int(round(iso))}"')
        seq = (f"<exif:ISOSpeedRatings><rdf:Seq><rdf:li>{min(int(round(iso)), 65535)}</rdf:li></rdf:Seq>"
               f"</exif:ISOSpeedRatings>")
    body = "\n    ".join(attrs)
    x = ('<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
         '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
         '  <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/"'
         ' xmlns:tiff="http://ns.adobe.com/tiff/1.0/" xmlns:exif="http://ns.adobe.com/exif/1.0/"'
         ' xmlns:exifEX="http://cipa.jp/exif/1.0/" xmlns:aux="http://ns.adobe.com/exif/1.0/aux/"\n    '
         f'{body}>\n   {seq}\n  </rdf:Description>\n </rdf:RDF>\n</x:xmpmeta>\n<?xpacket end="w"?>')
    return x.encode("utf-8")


def write_tiff(path: str, arr: np.ndarray, icc_bytes: bytes, meta: dict, compression: str = "none"):
    import tifffile
    xmp = xmp_packet(meta)
    extratags = [(34675, "B", len(icc_bytes), icc_bytes, True), (700, "B", len(xmp), xmp, True)]
    for code, key in ((271, "make"), (272, "model")):
        v = meta.get(key)
        if v:
            extratags.append((code, "s", 0, str(v), True))
    kw = {}
    if compression == "zip":
        kw["compression"] = "zlib"
    tifffile.imwrite(path, arr, photometric="rgb", planarconfig="contig", software=f"Photographic Studio By Bill "
                     f"{__version__}", resolution=(300, 300), resolutionunit="INCH", extratags=extratags,
                     datetime=meta.get("datetime") or None, metadata=None, **kw)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: str, arr: np.ndarray, icc_bytes: bytes, profile_name: str = "ICC profile",
              meta: dict | None = None):
    """8- or 16-bit RGB PNG with an embedded ICC profile (iCCP). Our own writer: Pillow cannot write
    16-bit RGB PNGs."""
    h, w, c = arr.shape
    bit_depth = 16 if arr.dtype == np.uint16 else 8
    rows = arr.astype(">u2" if bit_depth == 16 else np.uint8).reshape(h, -1).view(np.uint8)
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rows], axis=1)      # filter type 0 per row
    name = profile_name.encode("latin-1", "replace")[:79]
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bit_depth, 2, 0, 0, 0)))
        fh.write(_png_chunk(b"iCCP", name + b"\0\0" + zlib.compress(icc_bytes, 9)))
        if meta is not None:                      # XMP as an uncompressed international text chunk
            fh.write(_png_chunk(b"iTXt", b"XML:com.adobe.xmp\0\0\0\0\0" + xmp_packet(meta)))
        fh.write(_png_chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)))
        fh.write(_png_chunk(b"IEND", b""))


def write_jpeg(path: str, arr: np.ndarray, icc_bytes: bytes, quality: int, meta: dict):
    from PIL import Image
    im = Image.fromarray(arr)
    exif = Image.Exif()
    if meta.get("make"):
        exif[271] = str(meta["make"])
    if meta.get("model"):
        exif[272] = str(meta["model"])
    exif[305] = f"Photographic Studio By Bill {__version__}"
    ex = exif.get_ifd(0x8769)
    if meta.get("exposure_s"):
        ex[0x829A] = _rational(meta["exposure_s"], 100000)
    if meta.get("fnumber"):
        ex[0x829D] = _rational(meta["fnumber"], 100)
    if meta.get("focal_mm"):
        ex[0x920A] = _rational(meta["focal_mm"], 100)
    if meta.get("iso"):
        iso = int(round(meta["iso"]))
        ex[0x8827] = min(iso, 65535)
        if iso > 65535:                           # EXIF 2.3: sensitivity type 3 = recommended exposure index
            ex[0x8830] = 3
            ex[0x8832] = iso
    if meta.get("lens"):
        ex[0xA434] = str(meta["lens"])
    im.save(path, "JPEG", quality=int(quality), subsampling=0, optimize=True, icc_profile=icc_bytes,
            exif=exif.tobytes())


OWN_MARK = "Photographic Studio By Bill"       # written into every export (TIFF/EXIF Software, XMP CreatorTool)


def is_own_export(path: str) -> bool:
    """True when the file at ``path`` is one the Studio exported earlier (it carries the Studio's Software /
    CreatorTool mark). Anything unreadable or unmarked counts as somebody else's file."""
    low = str(path).lower()
    try:
        if low.endswith((".tif", ".tiff")):
            import tifffile
            with tifffile.TiffFile(path) as tf:
                tag = tf.pages[0].tags.get(305)
                return tag is not None and str(tag.value).startswith(OWN_MARK)
        if low.endswith((".jpg", ".jpeg")):
            from PIL import Image
            with Image.open(path) as im:
                return str(im.getexif().get(305) or "").startswith(OWN_MARK)
        if low.endswith(".png"):
            with open(path, "rb") as fh:
                if fh.read(8) != b"\x89PNG\r\n\x1a\n":
                    return False
                while True:                       # our writer puts the XMP text chunk before the pixels
                    head = fh.read(8)
                    if len(head) < 8:
                        return False
                    n, tag = struct.unpack(">I", head[:4])[0], head[4:]
                    if tag in (b"IDAT", b"IEND"):
                        return False
                    if tag == b"iTXt" and n < (1 << 20):
                        return OWN_MARK.encode() in fh.read(n)
                    fh.seek(n + 4, os.SEEK_CUR)
    except Exception:                             # noqa: BLE001  (unreadable: not ours, so never replaced)
        return False
    return False


def _same_file(a: str, b: str) -> bool:
    try:
        if os.path.exists(a) and os.path.exists(b):
            return os.path.samefile(a, b)
    except OSError:
        pass
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def export_name(source_path: str, spec: "ExportSpec", extra: str = "") -> str:
    """The file name of an export: the photo's own name with ``_Studio`` added (Bill's rule, 2026-11-16: every
    export has its own name, never the original's), then the format's extension. IMG_0001.CR2 -> IMG_0001_Studio.tif;
    a virtual copy puts its name in between (``extra``): IMG_0001_Dark_one_Studio.tif. A file that already ends in
    _Studio (an earlier export opened again) is not given the ending twice."""
    stem = os.path.splitext(os.path.basename(source_path))[0]
    if extra:
        stem += "_" + extra
    if spec.suffix and not stem.lower().endswith(spec.suffix.lower()):
        stem += spec.suffix
    return stem + EXT[spec.format]


def free_target(path: str, source: str | None = None, taken: set | None = None) -> str:
    """The name an export can be written under without destroying anything (rule 0.3.4: originals are never
    changed). ``path`` itself when nothing is there, or when what is there is an earlier export by the Studio
    (exporting again replaces it). Never replaced: the photo being exported (an earlier export opened and exported
    again into its own folder), a file already written in this batch (``taken``: two photos with the same name),
    and any file the Studio did not write. Those exports go beside it as NAME-2.ext, NAME-3.ext ... ``taken``
    (a set) is updated with the name chosen. With export_name's _Studio ending this is the safety net, not the
    usual case."""
    def key(p):
        return os.path.normcase(os.path.abspath(p))

    def usable(p):
        if taken is not None and key(p) in taken:
            return False
        if not os.path.lexists(p):
            return True
        if source and _same_file(p, source):
            return False
        return os.path.isfile(p) and is_own_export(p)

    stem, ext = os.path.splitext(path)
    out, n = path, 1
    while not usable(out):
        n += 1
        out = f"{stem}-{n}{ext}"
        if n > 9999:
            raise OSError(f"no free file name for {path}")
    if out != path:
        log.info("export: %s is kept as it is (not a Studio export, or already written in this batch); writing "
                 "%s", os.path.basename(path), os.path.basename(out))
    if taken is not None:
        taken.add(key(out))
    return out


def write(path: str, img_pp: np.ndarray, spec: ExportSpec, meta: dict | None = None) -> str:
    """Convert and write. ``img_pp`` is display-referred linear ProPhoto (from render()). Returns the path."""
    spec.validate()
    meta = slog.scrub(meta or {})
    with slog.stage("output.convert", log, space=spec.space, bits=spec.bits):
        arr = to_output(img_pp, spec)
    icc_bytes = icc.profile_bytes(spec.space)
    tmp = path + ".partial"
    with slog.stage("output.write", log, format=spec.format, file=os.path.basename(path)):
        if spec.format == "tiff":
            write_tiff(tmp, arr, icc_bytes, meta, spec.compression)
        elif spec.format == "png":
            write_png(tmp, arr, icc_bytes, SPACES[spec.space].name, meta)
        else:
            write_jpeg(tmp, arr, icc_bytes, spec.quality, meta)
        os.replace(tmp, path)                   # never leave a half-written file under the real name
    log.info("wrote %s (%d x %d, %d-bit %s, %s)", path, arr.shape[1], arr.shape[0], spec.bits,
             spec.format.upper(), SPACES[spec.space].name)
    return path
