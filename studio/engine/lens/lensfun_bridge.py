"""Bridge to lensfun (LGPL-3.0 library, CC BY-SA 3.0 database) through lensfunpy (MIT): profiles for modern lenses
with electronic contacts (plan section 10.1, source 3, violet tier).

lensfun is optional. Without it everything still works; the Lens panel just says no database is installed.
Only the numbers are taken from lensfun (distortion and vignetting terms at the photo's focal length and
aperture); the Studio applies them itself (geometry.py, falloff.py), and they are frozen into the sidecar, so an
edit made with lensfun renders the same on a PC without it.

Contactless vintage lenses are never in lensfun: they take the measured-from-the-photo path (amber) or the lens
wizard (cyan, Phase 7).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ... import log as slog

log = slog.get("engine.lensfun")

_DB = None
_ERR = None


def available() -> bool:
    return _db() is not None


def status() -> str:
    if available():
        try:
            import lensfunpy
            return f"lensfun database installed (lensfunpy {lensfunpy.__version__})"
        except Exception:
            return "lensfun database installed"
    return "lensfun is not installed (modern-lens profiles unavailable; everything else works)"


def _db():
    global _DB, _ERR
    if _DB is None and _ERR is None:
        try:
            import lensfunpy
            _DB = lensfunpy.Database()
        except Exception as e:                       # not installed, or a broken install: degrade quietly
            _ERR = e
            log.info("lensfun unavailable: %s", e)
    return _DB


def normalise_lens_name(make: str, name: str) -> str:
    """Canon writes 'EF180mm f/3.5L Macro USM'; lensfun calls it 'Canon EF 180mm f/3.5L Macro USM'."""
    n = (name or "").strip()
    n = re.sub(r"^(EF-S|EF-M|EF|RF-S|RF)(?=\d)", r"\1 ", n)
    if (make or "").lower().startswith("canon") and re.match(r"^(EF|RF)", n):
        n = "Canon " + n
    return n


def _numbers(s: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", s or ""))


@dataclass
class LensMatch:
    model: str
    maker: str
    score: int
    crop_factor: float
    camera_crop: float
    lens: object
    camera: object


def find_lens(make: str, model: str, lens_name: str, focal_mm: float | None = None) -> LensMatch | None:
    """Find the lensfun entry for a photo. Strict: the focal lengths and f-number in the names must agree and the
    photo's focal length must lie in the lens's range, otherwise no match (a wrong profile is worse than none)."""
    db = _db()
    if db is None or not lens_name:
        return None
    try:
        cams = db.find_cameras(make, model) or []
        if not cams:
            log.info("lensfun: camera %s %s not in the database", make, model)
            return None
        cam = cams[0]
        want = normalise_lens_name(make, lens_name)
        found = list(db.find_lenses(cam, None, want) or [])
        if not found:
            found = list(db.find_lenses(cam, None, want, True) or [])
        nums = _numbers(want)
        for ln in found:
            if nums and not nums <= _numbers(ln.model) | _numbers(str(ln.max_aperture)):
                continue
            if focal_mm and not (ln.min_focal - 0.5 <= focal_mm <= (ln.max_focal or ln.min_focal) + 0.5):
                continue
            return LensMatch(model=ln.model, maker=ln.maker, score=int(ln.score), crop_factor=float(ln.crop_factor),
                             camera_crop=float(cam.crop_factor), lens=ln, camera=cam)
    except Exception:
        slog.log_exception("lensfun lookup failed", log)
    return None


def profile_for(make, model, lens_name, focal_mm, fnumber, distance_m: float = 1000.0) -> dict:
    """{'lens': name, 'distortion': model dict or None, 'vignetting': model dict or None} at this focal length and
    aperture. Empty dict when there is no match."""
    m = find_lens(make, model, lens_name, focal_mm)
    if m is None:
        return {}
    import lensfunpy as lf
    out = {"lens": m.model, "distortion": None, "vignetting": None}
    norm = m.camera_crop / max(m.crop_factor, 1e-6)
    src = f"lensfun: {m.model}"
    f = float(focal_mm or m.lens.min_focal)
    try:
        d = m.lens.interpolate_distortion(f)
        if d is not None:
            names = {lf.DistortionModel.PTLENS: "ptlens", lf.DistortionModel.POLY3: "poly3",
                     lf.DistortionModel.POLY5: "poly5"}
            name = names.get(d.model)
            if name:
                k = {"ptlens": 3, "poly3": 1, "poly5": 2}[name]
                out["distortion"] = {"model": name, "terms": [round(float(t), 8) for t in list(d.terms)[:k]],
                                     "norm": round(norm, 6), "source": src, "focal": f}
    except Exception:
        slog.log_exception("lensfun distortion", log)
    try:
        if fnumber:
            v = m.lens.interpolate_vignetting(f, float(fnumber), float(distance_m))
            if v is not None and v.model == lf.VignettingModel.PA:
                out["vignetting"] = {"model": "pa", "terms": [round(float(t), 8) for t in list(v.terms)[:3]],
                                     "norm": round(norm, 6), "source": src, "focal": f, "aperture": float(fnumber)}
    except Exception:
        slog.log_exception("lensfun vignetting", log)
    return out
