"""Lens identity and tagging (plan sections 6.2 and 10.4).

Contactless (adapted, vintage) lenses write nothing into the raw file, so the photographer picks the lens once
per shoot and it is stored in each photo's sidecar (``edit.lens.tag``) and written into exported files.

Two lists, merged by name:
  * profiles/lenses/manual_lenses.json  - a shared starter list (CC BY-SA, in the repository)
  * settings/my_lenses.json             - the user's own additions (private, never shared)
A lens marked ``character`` is one whose softness or glow is the point (Heliar, Laack, Petzval...): automatic
"Recover lens sharpness" stays off for it unless the user turns it on (plan 10.5).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from ... import log as slog
from ... import paths

log = slog.get("engine.lenslib")

SHARED_FILE = "manual_lenses.json"
MINE_FILE = "my_lenses.json"


@dataclass
class LensEntry:
    name: str
    maker: str = ""
    focal_mm: float | None = None
    max_aperture: float | None = None
    character: bool = False
    contacts: bool = False            # True = the camera records it itself (no tagging needed)
    notes: str = ""

    def label(self) -> str:
        return self.name + ("  (character lens)" if self.character else "")


def _read(path) -> list:
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        rows = d.get("lenses", []) if isinstance(d, dict) else d
        return [LensEntry(**{k: v for k, v in r.items() if k in LensEntry.__dataclass_fields__}) for r in rows
                if isinstance(r, dict) and r.get("name")]
    except FileNotFoundError:
        return []
    except Exception:
        slog.log_exception(f"lens list {path} unreadable", log)
        return []


def shared_path() -> str:
    return os.path.join(str(paths.PROFILES), "lenses", SHARED_FILE)


def mine_path() -> str:
    return os.path.join(str(paths.get("SETTINGS")), MINE_FILE)


def list_lenses() -> list:
    """Shared starter list plus the user's own lenses (the user's entry wins on a name clash), sorted by name."""
    out = {e.name.lower(): e for e in _read(shared_path())}
    for e in _read(mine_path()):
        out[e.name.lower()] = e
    return sorted(out.values(), key=lambda e: e.name.lower())


def get(name: str) -> LensEntry | None:
    if not name:
        return None
    for e in list_lenses():
        if e.name.lower() == name.strip().lower():
            return e
    return None


def add_lens(entry: LensEntry) -> str:
    """Add or replace one of the user's own lenses. Returns the file written."""
    if not entry.name.strip():
        raise ValueError("a lens needs a name")
    p = mine_path()
    rows = {e.name.lower(): e for e in _read(p)}
    rows[entry.name.strip().lower()] = entry
    doc = {"format": 1, "about": "Your own lenses for tagging contactless glass. Private: never shared.",
           "lenses": [asdict(e) for e in sorted(rows.values(), key=lambda e: e.name.lower())]}
    tmp = p + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, p)
    return p


def effective_lens(meta, edit) -> tuple[str, str]:
    """(lens name, where it came from): the user's tag wins, then the camera's record, else unknown."""
    tag = (getattr(edit.lens, "tag", "") or "").strip()
    if tag:
        return tag, "tagged by you"
    if getattr(meta, "lens", ""):
        return meta.lens, "recorded by the camera"
    if getattr(meta, "manual_lens", False) or not getattr(meta, "fnumber", None):
        return "", "manual lens, not tagged yet"
    return "", "unknown"


def is_character(name: str) -> bool:
    e = get(name)
    return bool(e and e.character)
