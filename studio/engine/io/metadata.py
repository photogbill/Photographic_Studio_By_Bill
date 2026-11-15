"""Camera metadata the engine needs, with the TRUE ISO (BUILD_PLAN.md section 4).

Canon bodies write ISO 65535 in the standard EXIF ISO field for any ISO above 51200, because that field
is a 16-bit number (confirmed on the 5D Mark IV at ISO 102400). The real ISO lives in:
  * EXIF 2.3 32-bit fields: RecommendedExposureIndex (0x8832), ISOSpeed (0x8833),
    StandardOutputSensitivity (0x8831)
  * Canon MakerNote ShotInfo (tag 0x0004): BaseISO and AutoISO
  * LibRaw's own reading (it decodes the maker notes too)
65535, 32767, 0 and negative values are treated as "invalid, look elsewhere". Every source is kept so
disagreements show up in the log. Serial numbers are read by nobody: they are not in this structure.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from ... import log as slog
from .tiffmeta import read_tiff_meta

log = slog.get("engine.metadata")

INVALID_ISO = {65535, 32767, 65534}

# Standard ISO values (third stops) plus Canon expanded settings. Used to snap maker-note values.
STANDARD_ISOS = [25, 32, 40, 50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600,
                 2000, 2500, 3200, 4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600, 32000, 40000,
                 51200, 64000, 80000, 102400, 128000, 160000, 204800, 256000, 320000, 409600, 512000, 640000,
                 819200, 1024000, 1638400, 3276800, 4096000]

# EXIF tags
T_MAKE, T_MODEL, T_ORIENT, T_DATETIME = 0x010F, 0x0110, 0x0112, 0x0132
T_EXPOSURE, T_FNUMBER, T_ISO, T_SENS_TYPE = 0x829A, 0x829D, 0x8827, 0x8830
T_SOS, T_REI, T_ISOSPEED = 0x8831, 0x8832, 0x8833
T_DTO, T_FOCAL, T_LENSMODEL, T_LENSMAKE = 0x9003, 0x920A, 0xA434, 0xA433
T_EXPBIAS = 0x9204
# Canon MakerNote tags
C_CAMSET, C_SHOTINFO, C_LENSMODEL, C_SENSORINFO = 0x0001, 0x0004, 0x0095, 0x00E0


@dataclass
class Metadata:
    make: str = ""
    model: str = ""
    iso: float | None = None              # the true ISO
    iso_source: str = ""                  # where it came from
    iso_candidates: dict = field(default_factory=dict)
    exposure_s: float | None = None
    fnumber: float | None = None
    focal_mm: float | None = None
    lens: str = ""
    datetime: str = ""
    exposure_bias_ev: float | None = None
    orientation: int = 1
    camera_temperature_c: float | None = None
    sensor_info: dict = field(default_factory=dict)    # Canon SensorInfo borders (geometry, not identity)
    manual_lens: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def iso_label(self) -> str:
        return f"ISO {int(round(self.iso))}" if self.iso else "ISO ?"


def _first(v):
    if isinstance(v, tuple):
        return v[0] if v else None
    return v


def valid_iso(v) -> float | None:
    """Return v as a float if it is a believable ISO, else None."""
    v = _first(v)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f <= 0 or int(f) in INVALID_ISO or f > 10_000_000:
        return None
    return f


def snap_iso(v: float, tol_stops: float = 1.0 / 6.0) -> float:
    """Snap a computed ISO to the nearest standard third-stop value if it is within tol_stops."""
    best = min(STANDARD_ISOS, key=lambda s: abs(math.log2(v / s)))
    return float(best) if abs(math.log2(v / best)) <= tol_stops else float(round(v))


def canon_shotinfo_iso(shotinfo) -> float | None:
    """True ISO from Canon ShotInfo: BaseISO * AutoISO / 100 (ExifTool's definition).
    AutoISO = 2^(v1/32)*100 ; BaseISO = 2^(v2/32)*100/32 ; values are signed 16-bit."""
    if not isinstance(shotinfo, tuple) or len(shotinfo) < 3:
        return None

    def s16(x):
        return x - 65536 if x >= 32768 else x
    auto_raw, base_raw = s16(shotinfo[1]), s16(shotinfo[2])
    if base_raw == 0:
        return None
    base = 2.0 ** (base_raw / 32.0) * 100.0 / 32.0
    auto = 2.0 ** (auto_raw / 32.0) * 100.0
    iso = base * auto / 100.0
    if not (10 <= iso <= 10_000_000):
        return None
    return snap_iso(iso)


def canon_temperature(shotinfo) -> float | None:
    """Canon ShotInfo index 12 is CameraTemperature (+128 offset) on bodies that record it."""
    if isinstance(shotinfo, tuple) and len(shotinfo) > 12:
        v = shotinfo[12]
        if 0 < v < 256:
            t = v - 128.0
            if -40 <= t <= 90:
                return t
    return None


def resolve_iso(candidates: dict) -> tuple[float | None, str]:
    """Pick the true ISO from the candidates dict {source: value}. Order of trust:
    1. EXIF ISOSpeedRatings when valid (it is exact when it is not saturated)
    2. EXIF 32-bit fields (ISOSpeed, RecommendedExposureIndex, StandardOutputSensitivity)
    3. Canon MakerNote ShotInfo
    4. LibRaw's value
    """
    order = ["exif_iso", "exif_isospeed", "exif_rei", "exif_sos", "canon_shotinfo", "libraw"]
    for k in order:
        v = valid_iso(candidates.get(k))
        if v is not None:
            return v, k
    return None, ""


def read_metadata(path: str, libraw_iso: float | None = None) -> Metadata:
    """Read camera metadata. Never raises for a readable file: unknown fields stay empty."""
    md = Metadata()
    try:
        tm = read_tiff_meta(path)
    except Exception as e:                       # unknown container: LibRaw values only
        log.warning("metadata: could not parse TIFF structure of %s (%s); using LibRaw values only", path, e)
        tm = None
    cands: dict = {}
    if tm is not None:
        i0, ex, mn = tm.ifd0, tm.exif, tm.makernote
        md.make = str(i0.get(T_MAKE, "") or "").strip()
        md.model = str(i0.get(T_MODEL, "") or "").strip()
        o = _first(i0.get(T_ORIENT))
        md.orientation = int(o) if isinstance(o, int) and 1 <= o <= 8 else 1
        md.datetime = str(ex.get(T_DTO, "") or i0.get(T_DATETIME, "") or "")
        e = _first(ex.get(T_EXPOSURE))
        md.exposure_s = float(e) if isinstance(e, (int, float)) and e > 0 else None
        f = _first(ex.get(T_FNUMBER))
        md.fnumber = float(f) if isinstance(f, (int, float)) and f > 0 else None
        md.manual_lens = isinstance(f, (int, float)) and f == 0
        fl = _first(ex.get(T_FOCAL))
        md.focal_mm = float(fl) if isinstance(fl, (int, float)) and fl > 0 else None
        b = _first(ex.get(T_EXPBIAS))
        md.exposure_bias_ev = float(b) if isinstance(b, (int, float)) else None
        md.lens = str(ex.get(T_LENSMODEL, "") or mn.get(C_LENSMODEL, "") or "").strip()
        cands["exif_iso"] = _first(ex.get(T_ISO))
        cands["exif_isospeed"] = _first(ex.get(T_ISOSPEED))
        cands["exif_rei"] = _first(ex.get(T_REI))
        cands["exif_sos"] = _first(ex.get(T_SOS))
        si = mn.get(C_SHOTINFO)
        cands["canon_shotinfo"] = canon_shotinfo_iso(si)
        md.camera_temperature_c = canon_temperature(si)
        sens = mn.get(C_SENSORINFO)
        if isinstance(sens, tuple) and len(sens) >= 13:
            md.sensor_info = dict(width=sens[1], height=sens[2], left=sens[5], top=sens[6], right=sens[7],
                                  bottom=sens[8], mask_left=sens[9], mask_top=sens[10], mask_right=sens[11],
                                  mask_bottom=sens[12])
    if libraw_iso:
        cands["libraw"] = libraw_iso
    md.iso_candidates = {k: v for k, v in cands.items() if v is not None}
    md.iso, md.iso_source = resolve_iso(cands)
    # Cross-check: warn when the trusted value disagrees with the maker note by more than 1/6 stop.
    chk = valid_iso(cands.get("canon_shotinfo")) or valid_iso(cands.get("libraw"))
    if md.iso and chk and abs(math.log2(md.iso / chk)) > 1.0 / 6.0:
        log.warning("ISO sources disagree for %s: %s", path, md.iso_candidates)
    log.debug("metadata %s: %s %s %s (from %s) %s s f/%s", path, md.make, md.model, md.iso_label, md.iso_source,
              md.exposure_s, md.fnumber)
    return md
