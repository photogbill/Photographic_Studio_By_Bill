"""Looks (presets), copy / paste / sync of settings, and change descriptions for the history list (plan 5).

A look is a small, shareable ``.psblook`` file: plain JSON holding part of an edit. Applying it changes only what
it names, so a look never overwrites exposure, white balance, crop or the lens tag unless it says so.

  {"format": 1, "name": "Warm film", "about": "...", "author": "...", "settings": {"light": {...}, "colour": {...}}}

Starter looks live in profiles/looks (shared, CC BY-SA); your own in settings/looks (private).
"""
from __future__ import annotations

import copy
import json
import os
import re

from ... import log as slog
from ... import paths

log = slog.get("engine.looks")

EXT = ".psblook"

# Copy / paste groups: what the photographer thinks of, mapped to Edit fields
GROUPS = {
    "light": ("Light (exposure, contrast, highlights, shadows, curves, clipped highlights)",
              ["exposure_ev", "tone", "light", "hdr", "highlight_mode", "glow_profile"]),
    "white_balance": ("White balance", ["white_balance"]),
    "colour": ("Colour (saturation, HSL, colour wheel, grading, B&W)", ["colour"]),
    "detail": ("Detail (detail by scale, sharpening)", ["microcontrast", "detail", "sharpen"]),
    "clean": ("Clean (noise, banding)", ["denoise", "banding"]),
    "lens": ("Lens fixes (fringing, distortion, vignetting)", ["lens"]),
    "shape": ("Shape (straighten, perspective, crop)", ["geometry"]),
    "masks": ("Masks (local adjustments)", ["masks"]),
    "retouch": ("Retouch (heal, clone, dust spots)", ["retouch"]),
    "portrait": ("Portrait sliders (the faces and body sculpting stay with each photo)", ["portrait"]),
    "lens_tag": ("Lens tag", []),
}
DEFAULT_COPY = ["light", "white_balance", "colour", "detail", "clean", "lens"]     # not shape, not the tag


def copy_groups(src, dst, groups) -> object:
    """A new Edit: ``dst`` with the chosen groups taken from ``src``."""
    from ..pipeline.edit import Edit
    s, d = src.to_dict(), dst.to_dict()
    for g in groups:
        if g not in GROUPS:
            raise ValueError(f"unknown settings group {g!r}")
        for key in GROUPS[g][1]:
            if key == "lens":
                tag = d["lens"].get("tag", "")
                d["lens"] = copy.deepcopy(s["lens"])
                d["lens"]["tag"] = tag
            elif key == "portrait":
                d["portrait"] = _portrait_sliders(s.get("portrait"), d.get("portrait"))
            else:
                d[key] = copy.deepcopy(s[key])
        if g == "lens_tag":
            d["lens"]["tag"] = s["lens"].get("tag", "")
    return Edit.from_dict(d)


def _portrait_sliders(src: dict | None, dst: dict | None) -> dict | None:
    """The portrait sliders of ``src`` with the faces and sculpting of ``dst`` (those belong to one photo)."""
    from ..local.portrait import FACE_KEYS
    out = copy.deepcopy(dst) if dst else {}
    for k in FACE_KEYS:
        out[k] = float((src or {}).get(k, 0) or 0)
    return out or None


# ------------------------------------------------------------------------------------------ looks
def _deep_merge(base: dict, part: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in part.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("hsl", "grading", "bw_mix", "curve"):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# A look never changes brightness on its own (Bill's rule): these are dropped from every look, shared or saved.
BRIGHTNESS_KEYS = {"exposure_ev": None, "tone": ("mid_grey", "match_camera_brightness"),
                   "hdr": None}       # Auto HDR lifts and holds brightness: a look never carries it


def _strip(settings: dict) -> dict:
    s = copy.deepcopy(settings)
    s.pop("stage_versions", None)
    for k, sub in BRIGHTNESS_KEYS.items():
        if sub is None:
            s.pop(k, None)
        elif isinstance(s.get(k), dict):
            for x in sub:
                s[k].pop(x, None)
    if isinstance(s.get("lens"), dict):
        s["lens"].pop("tag", None)
    s.pop("retouch", None)                   # spots belong to one photo, never to a look
    s.pop("glow_profile", None)              # a lens's own measurement, not a look
    if isinstance(s.get("portrait"), dict):  # faces and sculpting belong to one photo: a look keeps the sliders only
        from ..local.portrait import FACE_KEYS
        s["portrait"] = {k: float(s["portrait"].get(k, 0) or 0) for k in FACE_KEYS}
    elif "portrait" in s:
        s.pop("portrait")
    return s


def apply_look(edit, look: dict):
    """Return a new Edit with the look's settings laid over ``edit``. Exposure, the brightness-matching switches
    and the lens tag are never taken from a look."""
    from ..pipeline.edit import Edit
    return Edit.from_dict(_deep_merge(edit.to_dict(), _strip(look.get("settings", {}))))


def look_dirs() -> list:
    return [("shared", str(paths.PROFILES / "looks")), ("yours", str(paths.get("SETTINGS") / "looks"))]


def list_looks() -> list:
    """[{"name", "about", "path", "source"}] sorted: shared first, then yours."""
    out = []
    for source, d in look_dirs():
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(EXT):
                p = os.path.join(d, f)
                try:
                    lk = load_look(p)
                    out.append({"name": lk.get("name") or os.path.splitext(f)[0], "about": lk.get("about", ""),
                                "path": p, "source": source})
                except Exception:
                    slog.log_exception(f"look {p} unreadable", log)
    return out


def load_look(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        lk = json.load(fh)
    if not isinstance(lk, dict) or not isinstance(lk.get("settings"), dict):
        raise ValueError(f"{path} is not a look file")
    return lk


def find_look(name: str) -> dict | None:
    for lk in list_looks():
        if lk["name"].lower() == name.lower():
            return load_look(lk["path"])
    return None


def look_from_edit(edit, name: str, groups, about: str = "", author: str = "") -> dict:
    d = edit.to_dict()
    settings = {}
    for g in groups:
        for key in GROUPS[g][1]:
            settings[key] = copy.deepcopy(d[key])
    return {"format": 1, "name": name, "about": about, "author": author, "settings": _strip(settings)}


def save_look(look: dict) -> str:
    d = paths.get("SETTINGS") / "looks"
    d.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\- ]+", "", look["name"]).strip() or "look"
    p = d / (safe + EXT)
    tmp = str(p) + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(look, fh, indent=2)
    os.replace(tmp, p)
    return str(p)


# ------------------------------------------------------------------------------------------ history labels
_NAMES = {"exposure_ev": "Exposure", "light": "Light", "tone": "Tone", "colour": "Colour", "white_balance":
          "White balance", "microcontrast": "Local contrast", "detail": "Detail by scale", "hdr": "Auto HDR", "sharpen": "Sharpening", "denoise": "Noise",
          "banding": "Banding", "lens": "Lens", "geometry": "Shape", "demosaic": "Demosaic",
          "masks": "Masks", "retouch": "Retouch", "finish": "Finish", "profile": "Camera profile", "highlight_mode": "Clipped highlights",
          "glow_profile": "Lens glow profile", "portrait": "Portrait", "film": "Film", "halftone": "Printing dots", "stains": "Stains & corrosion", "scratches": "Dust & scratches",
          "colourize": "Colourize", "depth": "Depth haze", "jpeg": "JPEG repair", "relight": "Relight", "composite": "Composite"}


_SUB = {"ranges": "colour wheel", "grading": "colour grading", "hsl": "HSL", "bw_mix": "B&W mix", "curve": "tone curve",
        "bw": "black & white", "lut": "LUT", "lut_amount": "LUT amount", "smart": "Smart lighting"}


def describe_change(old, new) -> str:
    """A short label for what changed between two edits ("Light: shadows +25", "Colour: colour wheel changed")."""
    a, b = old.to_dict(), new.to_dict()
    changed = [k for k in b if k != "stage_versions" and a.get(k) != b.get(k)]
    if not changed:
        return "No change"
    if len(changed) > 1:
        return ", ".join(_NAMES.get(k, k) for k in changed[:3]) + ("..." if len(changed) > 3 else "")
    k = changed[0]
    if k == "masks":
        return _describe_masks(a.get("masks") or [], b.get("masks") or [])
    if k == "retouch":
        return _describe_retouch(a.get("retouch") or [], b.get("retouch") or [])
    if k == "portrait":
        return _describe_portrait(a.get("portrait") or {}, b.get("portrait") or {})
    if k == "film":
        f = b.get("film") or {}
        if not f:
            return "Film: positive"
        if (a.get("film") or {}).get("kind") != f.get("kind"):
            return "Film: " + ("colour negative" if f.get("kind") == "colour" else "black and white negative")
    if k in ("halftone", "stains", "scratches", "colourize", "jpeg") and not (b.get(k) or {}).get("on") and (a.get(k) or {}).get("on"):
        return f"{_NAMES.get(k, k)}: off"
    if b[k] is None:
        return _NAMES.get(k, k)
    if isinstance(b[k], dict):
        sub = [s for s in b[k] if (a.get(k) or {}).get(s) != b[k][s]]
        if len(sub) == 1:
            v = b[k][sub[0]]
            name = _SUB.get(sub[0], sub[0].replace("_", " "))
            if isinstance(v, bool):
                val = "on" if v else "off"
            elif isinstance(v, (int, float)):
                val = f"{v:+g}"
            else:
                val = "changed"
            return f"{_NAMES.get(k, k)}: {name} {val}"
        return f"{_NAMES.get(k, k)}: {len(sub)} settings"
    v = b[k]
    return f"{_NAMES.get(k, k)} {v:+g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else _NAMES.get(k, k)


def _describe_masks(old: list, new: list) -> str:
    if len(new) > len(old):
        return f"Mask added: {new[-1].get('name', 'mask')}"
    if len(new) < len(old):
        return "Mask deleted"
    for mo, mn in zip(old, new):
        if mo == mn:
            continue
        name = mn.get("name", "mask")
        if mo.get("shapes") != mn.get("shapes"):
            return f"{name}: shape" + (" added" if len(mn.get("shapes", [])) > len(mo.get("shapes", [])) else " changed")
        ao, an = mo.get("adjust") or {}, mn.get("adjust") or {}
        diff = [x for x in an if ao.get(x) != an.get(x)]
        if len(diff) == 1:
            v = an[diff[0]]
            return f"{name}: {diff[0]} {v:+g}" if isinstance(v, (int, float)) else f"{name}: {diff[0]}"
        for key, txt in (("enabled", "on/off"), ("invert", "inverted"), ("opacity", "amount"), ("name", "renamed")):
            if mo.get(key) != mn.get(key):
                return f"{name}: {txt}"
        return f"{name}: {len(diff)} settings"
    return "Masks"


def _describe_retouch(old: list, new: list) -> str:
    kinds = {"heal": "heal", "clone": "clone", "redeye": "red eye", "pupil": "pupil"}
    if len(new) > len(old):
        added = new[len(old):]
        if len(added) > 1:
            src = {s.get("auto") for s in added}
            what = "dust spots" if src <= {"dust", "map"} else "spots"
            return f"Retouch: {len(added)} {what} added"
        return f"Retouch: {kinds.get(added[0].get('kind'), 'spot')} spot added"
    if len(new) < len(old):
        n = len(old) - len(new)
        return "Retouch: spot removed" if n == 1 else f"Retouch: {n} spots removed"
    return "Retouch: spot changed"


_PORTRAIT = {"younger": "younger", "lines": "lines", "under_eyes": "under eyes", "smile_lines": "smile lines",
             "even_skin": "even skin", "jawline": "jawline", "double_chin": "double chin", "eyes": "eyes",
             "teeth": "teeth"}


def _describe_portrait(old: dict, new: dict) -> str:
    fo, fn = old.get("faces") or [], new.get("faces") or []
    if fo != fn:
        if len(fn) != len(fo):
            return f"Portrait: {len(fn)} face{'s' if len(fn) != 1 else ''}"
        return "Portrait: face changed"
    so, sn = old.get("sculpt") or [], new.get("sculpt") or []
    if so != sn:
        if len(sn) > len(so):
            k = sn[-1]
            if k.get("kind") == "shape":
                return f"Sculpt: {SHAPE_NAMES.get(k.get('tool'), 'shape')} added"
            return f"Sculpt: {k.get('kind', 'stroke')}"
        if len(sn) < len(so):
            return "Sculpt: removed" if len(sn) else "Sculpting cleared"
        return "Sculpt: shape changed"
    diff = [k for k in _PORTRAIT if old.get(k) != new.get(k)]
    if len(diff) == 1:
        v = new.get(diff[0]) or 0
        return f"Portrait: {_PORTRAIT[diff[0]]} {float(v):+g}"
    return "Portrait" if not diff else f"Portrait: {len(diff)} settings"


SHAPE_NAMES = {"breast": "breast lift & size", "butt": "butt improvement", "waist": "slim waist",
               "tummy": "slim tummy", "arm": "slim arm", "thigh": "slim thigh"}
