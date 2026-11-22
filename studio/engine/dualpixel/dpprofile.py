"""What the Studio has learned about one camera's Dual Pixel halves (BUILD_PLAN_DP3D A.1, A.3.3, C.6 d).

Numbers only, never pictures (rule 0.3.2), so a profile can be shared like a camera's noise profile:
  near     which way is near: the sign of the offset of a place IN FRONT OF the plane of focus, how it was found
           (the vote of A.1 on receding scenes), when, and the votes themselves
  shading  each half's shading (GainField coefficients) per lens setting (lens, focal length, aperture): measured
           once on a stack and reused, so later stacks skip the fit
  scale    offset per unit of defocus per lens setting (C.6 d): turns Dual Pixel offsets into millimetres

Where (the same tiers as camera profiles, plan 23.1):
  calibration_local/profiles/<camera>_dualpixel.json   learned on this PC from your own frames (cyan)
  profiles/cameras/<camera>_dualpixel.json             shared (violet)
The Studio's built-in table (depth.NEAR_SIGN) comes after both. No Qt here.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import asdict, dataclass, field

from ... import __version__, log as slog, paths

log = slog.get("engine.dualpixel.profile")
SCHEMA = 1


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", " ".join(str(model or "").split())).strip("_") or "unknown_camera"


def local_path(model: str) -> str:
    return str(paths.get("CALIBRATION_LOCAL") / "profiles" / (_slug(model) + "_dualpixel.json"))


def shared_path(model: str) -> str:
    return str(paths.PROFILES / "cameras" / (_slug(model) + "_dualpixel.json"))


def lens_key(lens: str, focal_mm, fnumber) -> str:
    """One key per lens setting: 'Makro-Planar 2/50|50|2.0' (a blank lens name is kept blank: same body, same
    focal length and aperture is still a useful key for a chipped lens that does not say its name)."""
    f = "" if focal_mm in (None, 0) else f"{float(focal_mm):g}"
    a = "" if fnumber in (None, 0) else f"{float(fnumber):.1f}"
    return f"{str(lens or '').strip()}|{f}|{a}"


@dataclass
class DualPixelProfile:
    model: str
    near_sign: int | None = None            # +1 / -1: sign of the offset of a place in front of focus
    near_how: str = ""                      # plain words: how it was found
    near_date: str = ""
    near_votes: list = field(default_factory=list)   # [{stack, frames, vote, strength}]
    shading: dict = field(default_factory=dict)      # lens_key -> {"a": {"i,j": [10..15 coef]}, "b": ..., "shape",
    #                                                  "date", "frames", "level_ratio"}
    scale: dict = field(default_factory=dict)        # lens_key -> {"px_per_inv_mm" or "px_per_mm", "how", "date"}
    origin: str = "local"                   # local (measured on this PC, cyan) | shared (violet)
    schema: int = SCHEMA
    app_version: str = __version__
    license: str = "CC BY-SA 4.0"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("origin", None)
        return d

    @classmethod
    def from_dict(cls, d: dict, origin: str = "local") -> "DualPixelProfile":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__ and k != "origin"}
        p = cls(**known)
        p.origin = origin
        return p

    def save(self, path: str | None = None) -> str:
        path = path or local_path(self.model)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".partial"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(slog.scrub(self.to_dict()), fh, indent=2)
        os.replace(tmp, path)
        _CACHE.clear()
        return path

    # ------------------------------------------------------------------ shading (A.3.3)
    def gains_for(self, key: str, shape):
        """The two GainFields saved for this lens setting and sensor size, or None."""
        from .decode import GainField
        e = self.shading.get(key)
        if not e or tuple(e.get("shape", ())) != tuple(shape):
            return None
        try:
            ga = {tuple(int(t) for t in k.split(",")): v for k, v in e["a"].items()}
            gb = {tuple(int(t) for t in k.split(",")): v for k, v in e["b"].items()}
            src = "measured" if self.origin == "local" else "shared"
            return GainField(ga, tuple(shape), src), GainField(gb, tuple(shape), src)
        except (KeyError, ValueError, TypeError, AttributeError):
            return None

    def remember_gains(self, key: str, gains, level_ratio: float, frames: int = 1):
        ga, gb = gains
        self.shading[key] = {"a": {f"{i},{j}": list(map(float, v)) for (i, j), v in ga.coef.items()},
                             "b": {f"{i},{j}": list(map(float, v)) for (i, j), v in gb.coef.items()},
                             "shape": list(ga.shape), "date": _dt.date.today().isoformat(), "frames": int(frames),
                             "level_ratio": round(float(level_ratio), 4)}


_CACHE: dict = {}


def find(model: str) -> DualPixelProfile | None:
    """This PC's own profile first (cyan), then a shared one (violet); None when neither exists."""
    key = _slug(model)
    if key not in _CACHE:
        prof = None
        for path, origin in ((local_path(model), "local"), (shared_path(model), "shared")):
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as fh:
                        prof = DualPixelProfile.from_dict(json.load(fh), origin)
                    break
                except (OSError, ValueError) as e:
                    log.warning("dual pixel profile %s unreadable (%s)", path, e)
        _CACHE[key] = prof
    return _CACHE[key]


def local_or_new(model: str) -> DualPixelProfile:
    """This PC's own profile to add to (a shared one is copied, keeping its facts, so what is learned here is
    saved as this PC's)."""
    p = find(model)
    if p is None:
        return DualPixelProfile(model=" ".join(str(model or "").split()))
    if p.origin != "local":
        q = DualPixelProfile.from_dict(p.to_dict(), "local")
        return q
    return p


def clear_cache():
    _CACHE.clear()
