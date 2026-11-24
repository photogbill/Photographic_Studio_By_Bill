"""Keys for Paint, and the pen-key mapping screen's model (BUILD_PLAN_PAINT P1, 0.2).

Tablet express keys and pen buttons reach Paint as ordinary keys: the tablet's own driver (Huion's, or
OpenTabletDriver on Linux) is set to send a key, and here that key is given an action. So any tablet works, and
each action can have up to two keys (one on the keyboard, one on the tablet). Kept in ``paint/keys.json`` beside the
app (studio/paths.py), never in the user profile.

Key names are Qt's portable text form ("Ctrl+Z", "F13", "Shift+B"). Holding Space to pan and the pen's barrel button
for the pop-up palette are fixed and not listed here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# id -> (name, what it does, default keys)
ACTIONS = {
    "undo": ("Undo", "take back the last step", ["Ctrl+Z"]),
    "redo": ("Redo", "put the step back (follows the branch last used)", ["Ctrl+Shift+Z", "Ctrl+Y"]),
    "brush": ("Brush tool", "paint with the selected brush", ["B"]),
    "fill": ("Fill tool", "fill an area with the painting colour", ["F"]),
    "gradient": ("Gradient tool", "drag to lay a gradient", ["G"]),
    "eraser": ("Eraser on / off", "the brush rubs out instead of painting", ["E"]),
    "swap": ("Swap colours", "painting colour and second colour change places", ["X"]),
    "smaller": ("Brush smaller", "brush size down a step", ["["]),
    "bigger": ("Brush bigger", "brush size up a step", ["]"]),
    "popup": ("Pop-up palette", "brushes, colours and tools at the pointer (as right click)", []),
    "pick": ("Pick colour here", "take the colour under the pointer", []),
    "canvas_only": ("Canvas only", "hide the panels to paint on the whole screen", ["Tab"]),
    "rotate_left": ("Rotate view left", "turn the canvas 15 degrees left (the painting is not changed)", ["4"]),
    "rotate_right": ("Rotate view right", "turn the canvas 15 degrees right", ["6"]),
    "rotate_reset": ("Straighten view", "the canvas upright again", ["5"]),
    "mirror": ("Mirror view", "see the painting flipped, to catch drawing errors", ["M"]),
    "fit": ("Fit to window", "the whole painting in view", ["0"]),
    "zoom_100": ("Actual pixels", "zoom to 100 %", ["1"]),
    "zoom_in": ("Zoom in", "", ["+", "="]),
    "zoom_out": ("Zoom out", "", ["-"]),
    "new_layer": ("New layer", "a new layer above the selected one", ["Ctrl+Shift+N"]),
    "values": ("Values view on / off", "the painting in greys, to check values", ["V"]),
    "save": ("Save", "save the painting (.ora)", ["Ctrl+S"]),
    "save_as": ("Save as", "save under a new name", ["Ctrl+Shift+S"]),
    "open": ("Open", "open a painting", ["Ctrl+O"]),
}
MAX_KEYS = 2


def defaults() -> dict:
    return {a: list(v[2]) for a, v in ACTIONS.items()}


def keys_file() -> Path:
    from ... import paths
    d = paths.get("PAINT")
    d.mkdir(parents=True, exist_ok=True)
    return d / "keys.json"


def load(path: str | Path | None = None) -> dict:
    """The saved key map, with any action missing from the file at its default (so new actions get keys)."""
    m = defaults()
    p = Path(path) if path else keys_file()
    try:
        saved = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return m
    except (OSError, ValueError):
        return m                                         # unreadable: defaults (the file is left for her to see)
    for a, keys in saved.get("keys", {}).items():
        if a in m and isinstance(keys, list):
            m[a] = [str(k) for k in keys if k][:MAX_KEYS]
    return m


def save(m: dict, path: str | Path | None = None) -> Path:
    p = Path(path) if path else keys_file()
    tmp = str(p) + ".partial"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "keys": m}, f, indent=1)
    os.replace(tmp, p)
    return p


def assign(m: dict, action: str, key: str, slot: int = 0) -> str | None:
    """Give ``key`` to ``action`` in ``slot`` (0 or 1). A key does one thing only: if another action had it, it is
    taken from that action, whose name is returned so the screen can say so."""
    taken_from = None
    for a, keys in m.items():
        if a != action and key in keys:
            keys.remove(key)
            taken_from = a
    keys = [k for k in m.get(action, []) if k != key]
    while len(keys) <= slot:
        keys.append("")
    keys[slot] = key
    m[action] = [k for k in keys if k][:MAX_KEYS]
    return taken_from


def clear(m: dict, action: str) -> None:
    m[action] = []


def action_for(m: dict, key: str) -> str | None:
    return next((a for a, keys in m.items() if key in keys), None)


def conflicts(m: dict) -> list:
    """Keys given to more than one action (a hand-edited file); empty when all is well."""
    seen, out = {}, []
    for a, keys in m.items():
        for k in keys:
            if k in seen:
                out.append((k, seen[k], a))
            seen[k] = a
    return out
