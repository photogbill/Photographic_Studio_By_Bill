"""Camera profiles (section 7): what calibration measures, per ISO, and how the pipeline uses it.

Where profiles live:
  * calibration_local/profiles/   profiles made on this PC (private by default, never committed)
  * profiles/cameras/             shared profiles (CC BY-SA 4.0), downloaded or contributed on purpose
A profile is numbers only: no images, no serial numbers, no hot-pixel or PRNU maps (those stay in
calibration_local/<model>/ and are never shared, rule 0.3.2).

Tiers (rule 0.3.11): a profile made on this body is cyan, a shared one violet, an image estimate amber.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from ... import __version__, log as slog, paths
from .noise import NoiseModel

log = slog.get("engine.profile")
SCHEMA = 1
HOT_ELECTRONS = 100.0       # a HOT photosite collects at least this many electrons with no light: a bright dot at
#                             any ISO. In electrons, so the same photosites are "hot" whatever ISO they are seen at
HOT_FRACTION = 0.01         # ...of the range above black, where the camera's gain is not known (a Quick profile)


def hot_limit_dn(range_dn: float, gain_e_per_dn: float | None) -> float:
    """How far above black (raw levels) a photosite must stand with no light to be called hot."""
    if gain_e_per_dn:
        return min(HOT_ELECTRONS / float(gain_e_per_dn), 0.9 * float(range_dn))
    return HOT_FRACTION * float(range_dn)


@dataclass
class IsoEntry:
    iso: float
    frames: int = 0
    black_dn: list | None = None           # 2x2 per CFA position (visible-area origin)
    read_noise_dn: list | None = None      # 2x2 temporal read noise
    row_banding_dn: float | None = None    # true row-pattern amplitude (white-noise part removed)
    col_banding_dn: float | None = None
    gain_e_per_dn: float | None = None     # from flat pairs (photon transfer); None in a Quick profile
    gain_pairs: int = 0
    full_well_e: float | None = None
    dark_current: list = field(default_factory=list)   # [{exposure_s, temp_c, frames, excess_dn, noise_dn, ...}]
    hot_pixels: int = 0                    # count only; the map itself stays local. HOT: at least 1% of the
    #                                        range above black with no light (a bright dot in any photo)
    warm_pixels: int = 0                   # stand out of the noise at this ISO but below that (healed too)
    by_temp: list = field(default_factory=list)        # short darks per camera temperature: [{temp_c, frames,
    #                                                    paired, read_noise_dn 2x2, row_banding_dn, black_dn}]
    thermal: dict | None = None            # long exposures: {var_dn2_per_s, at_temp_c, doubling_c, points}
    read_noise_paired: bool = False        # read noise from two frames compared (else one frame: includes pattern)
    black_offset_dn: list | None = None    # 2x2: the picture area's black minus the masked border's (a photo takes
    #                                        its black from the border; this is what that misses)
    fixed_pattern_dn: float | None = None  # per-photosite pattern that is the same in every frame (centre crop)
    neighbour_corr: float | None = None    # how much neighbouring photosites' noise agrees (0 = independent)
    excess_kurtosis: float | None = None   # tails of the noise (0 = a bell curve)
    digital_gain: float = 1.0              # > 1: the raw values are stretched in the camera (a pushed ISO)
    pushed: bool = False                   # this ISO is a lower analogue gain pushed digitally


@dataclass
class CameraProfile:
    make: str
    model: str
    method: str = "quick"                  # quick (dark frames) | full (darks + flat pairs) | complete (+ Part G)
    isos: list = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    created: str = ""
    app_version: str = __version__
    schema: int = SCHEMA
    license: str = "CC BY-SA 4.0"
    origin: str = "local"                  # local (made on this PC) | shared

    # ------------------------------------------------------------------------------------------ lookup
    def gain_at(self, iso: float | None) -> float | None:
        """Electrons per raw level at this ISO: measured there, or from the sensor's gain x ISO (which is one
        number for all its real gains) where only other ISOs have flat pairs. None for a Quick profile."""
        if not iso:
            return None
        e = self.entry_for(iso, tol_stops=0.01)
        if e is not None and e.gain_e_per_dn:
            return float(e.gain_e_per_dn)
        gx = [x.gain_e_per_dn * x.iso for x in self.isos if x.gain_e_per_dn and not x.pushed]
        return float(np.median(gx)) / float(iso) if gx else None

    def entry_for(self, iso: float | None, tol_stops: float = 1.0 / 3.0) -> IsoEntry | None:
        if not iso or not self.isos:
            return None
        best = min(self.isos, key=lambda e: abs(math.log2(e.iso / iso)))
        return best if abs(math.log2(best.iso / iso)) <= tol_stops + 1e-9 else None

    def read_noise_at(self, e: IsoEntry, temp_c: float | None) -> np.ndarray:
        """Read noise (2x2 DN) at the camera temperature the photo was taken at, when frames near it were
        measured as pairs (within 6 degrees C); else the profile's overall figure."""
        rn = np.asarray(e.read_noise_dn, float)
        if temp_c is None or not e.by_temp:
            return rn
        near = [t for t in e.by_temp if t.get("paired") and t.get("temp_c") is not None
                and abs(t["temp_c"] - temp_c) <= 6.0]
        if not near:
            return rn
        return np.asarray(min(near, key=lambda t: abs(t["temp_c"] - temp_c))["read_noise_dn"], float)

    @staticmethod
    def thermal_variance(e: IsoEntry, temp_c: float | None, exposure_s: float | None) -> float:
        """Extra noise variance (DN^2) a long exposure adds, from the long dark frames; 0 when not measured."""
        th = e.thermal
        if not th or not exposure_s or exposure_s < 1.0:
            return 0.0
        rate = float(th.get("var_dn2_per_s") or 0.0)
        if th.get("doubling_c") and temp_c is not None and th.get("at_temp_c") is not None:
            rate *= 2.0 ** (min(max(temp_c - th["at_temp_c"], -25.0), 25.0) / th["doubling_c"])
        return max(rate, 0.0) * float(exposure_s)

    def noise_model(self, iso: float | None, range_dn: np.ndarray, temp_c: float | None = None,
                    exposure_s: float | None = None) -> NoiseModel | None:
        """Calibrated noise model for this ISO, or None. range_dn: 2x2 (white - black) in DN. With the photo's
        camera temperature and shutter speed, the read noise is the one measured near that temperature and a
        long exposure gets its measured extra noise."""
        e = self.entry_for(iso)
        if e is None or e.read_noise_dn is None:
            return None
        rng = np.asarray(range_dn, float)
        var = self.read_noise_at(e, temp_c) ** 2 + self.thermal_variance(e, temp_c, exposure_s)
        b = var / rng ** 2
        if e.gain_e_per_dn:
            a = 1.0 / (e.gain_e_per_dn * rng)
            src = "calibrated" if self.origin == "local" else "shared"
            return NoiseModel(a=a, b=b, source=src, confidence="good", iso=iso)
        return NoiseModel(a=np.full((2, 2), np.nan), b=b, source="calibrated-read-noise", confidence="fair", iso=iso)

    # ------------------------------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CameraProfile":
        d = dict(d)
        isos = [IsoEntry(**{k: v for k, v in e.items() if k in IsoEntry.__dataclass_fields__})
                for e in d.pop("isos", [])]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(isos=sorted(isos, key=lambda e: e.iso), **known)

    def save(self, path: str | None = None) -> str:
        path = path or local_profile_path(self.make, self.model)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.created = self.created or _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        tmp = path + ".partial"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(slog.scrub(self.to_dict()), fh, indent=2)
        os.replace(tmp, path)
        _CACHE.clear()
        return path

    @classmethod
    def load(cls, path: str, origin: str | None = None) -> "CameraProfile":
        with open(path, encoding="utf-8") as fh:
            p = cls.from_dict(json.load(fh))
        if origin:
            p.origin = origin
        return p


def _slug(make: str, model: str) -> str:
    name = model if model.lower().startswith(make.lower()) else f"{make} {model}"
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "unknown_camera"


def local_profile_path(make: str, model: str) -> str:
    return str(paths.get("CALIBRATION_LOCAL") / "profiles" / (_slug(make, model) + ".json"))


def shared_profile_path(make: str, model: str) -> str:
    return str(paths.PROFILES / "cameras" / (_slug(make, model) + ".json"))


def local_data_dir(make: str, model: str) -> str:
    """Private per-body data (hot-pixel map, darks). Never shared."""
    p = paths.get("CALIBRATION_LOCAL") / _slug(make, model)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


_CACHE: dict = {}


def find_profile(make: str, model: str) -> CameraProfile | None:
    """This PC's own profile first (cyan), then a shared one (violet)."""
    key = (make, model)
    if key not in _CACHE:
        prof = None
        for path, origin in ((local_profile_path(make, model), "local"), (shared_profile_path(make, model), "shared")):
            if os.path.isfile(path):
                try:
                    prof = CameraProfile.load(path, origin)
                    break
                except Exception:
                    slog.log_exception(f"profile {path} unreadable", log)
        _CACHE[key] = prof
    return _CACHE[key]
