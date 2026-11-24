"""Swatch sets for Paint (BUILD_PLAN_PAINT P1): save, load, and a palette taken from the painting.

Files are GIMP palettes (``.gpl``, plain text), which Krita, GIMP, Inkscape and MyPaint all read and write; Krita's
own ``.kpl`` sets are read too. Sets live in ``paint/swatches`` beside the app (studio/paths.py), never in the user
profile. Three sets come with Paint, each built from how painters really worked (the Brush Atlas, plan 0.8).

Palette from the painting: k-means in Oklab (even steps to the eye) over the visible picture, so the swatches are
the colour masses she actually painted, sorted dark to light. Best for: keeping a painting's colours consistent, or
picking up a reference's palette. Weak at: tiny accents (a small bright spot can be outvoted by large masses; raise
the count to catch it).
"""
from __future__ import annotations

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .colour import from_hex, oklab_to_srgb, srgb_to_oklab, to_hex
from .tiles import Document


@dataclass
class SwatchSet:
    name: str = "Swatches"
    colours: list = field(default_factory=list)     # [(r, g, b) sRGB 0..1]
    names: list = field(default_factory=list)       # one per colour ("" when none)
    path: str | None = None
    note: str = ""                                   # what the set is for

    def add(self, rgb, name: str = "") -> None:
        self.colours.append(tuple(float(c) for c in rgb[:3]))
        self.names.append(name)

    def remove(self, i: int) -> None:
        del self.colours[i]
        del self.names[i]


def _h(s):
    return from_hex(s)


BUILT_IN = [
    SwatchSet("Zorn palette", [_h("#e8e2d0"), _h("#c9962f"), _h("#b5352a"), _h("#1d1b1a")],
              ["Titanium white", "Yellow ochre", "Vermilion", "Ivory black"],
              note="Anders Zorn's four colours: flesh and portraits with only warm earths; black reads as a cool "
                   "blue beside them. Weak at: true greens and blues."),
    SwatchSet("Earth palette", [_h("#efe9db"), _h("#c9962f"), _h("#8f4a2a"), _h("#6b3a24"), _h("#4b3a2a"),
                                _h("#5b6b3a"), _h("#34404f"), _h("#1f1c1a")],
              ["White", "Yellow ochre", "Burnt sienna", "Burnt umber", "Raw umber", "Green earth", "Payne's grey",
               "Lamp black"],
              note="The old masters' earth pigments: landscapes, underpaintings, sober portraits. Weak at: bright "
                   "saturated colour."),
    SwatchSet("Value steps (10)", [tuple([v / 9] * 3) for v in range(10)], [f"Value {v}" for v in range(10)],
              note="Ten even greys for value studies and grisaille. Even in stored value, not in lightness."),
]


def swatch_dir() -> Path:
    from ... import paths
    d = paths.get("PAINT") / "swatches"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- files
def save_gpl(sw: SwatchSet, path: str) -> str:
    """Write a GIMP palette (atomic: a ``.partial`` file swapped in)."""
    lines = ["GIMP Palette", f"Name: {sw.name}", "Columns: 8"]
    if sw.note:
        lines.append(f"# {sw.note}")
    lines.append("#")
    for rgb, n in zip(sw.colours, sw.names + [""] * (len(sw.colours) - len(sw.names))):
        r, g, b = (int(round(max(0, min(1, c)) * 255)) for c in rgb)
        lines.append(f"{r:3d} {g:3d} {b:3d}\t{n or to_hex(rgb)}")
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    sw.path = path
    return path


def load(path: str) -> SwatchSet:
    """Read a ``.gpl`` (GIMP, Krita, Inkscape, MyPaint) or ``.kpl`` (Krita) swatch set."""
    if path.lower().endswith(".kpl"):
        return _load_kpl(path)
    sw = SwatchSet(Path(path).stem, path=path)
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
        if first != "GIMP Palette":
            raise ValueError("not a GIMP palette (.gpl): the first line should read 'GIMP Palette'")
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("Name:"):
                sw.name = line[5:].strip() or sw.name
                continue
            if line.startswith("# ") and not sw.colours and not sw.note:
                sw.note = line[2:].strip()
                continue
            m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s*(.*)", line)
            if m and not line.startswith("Columns"):
                sw.add(tuple(int(m.group(i)) / 255 for i in (1, 2, 3)), m.group(4).strip())
    return sw


def _load_kpl(path: str) -> SwatchSet:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("colorset.xml"))
    sw = SwatchSet(root.get("name") or Path(path).stem, path=path)
    for entry in root.iter("ColorSetEntry"):
        rgb = entry.find("RGB")
        if rgb is None:
            srgb = entry.find("sRGB")
            rgb = srgb
        if rgb is None:
            continue                                     # CMYK / Lab entries: skipped
        sw.add((float(rgb.get("r", 0)), float(rgb.get("g", 0)), float(rgb.get("b", 0))), entry.get("name", ""))
    return sw


def saved_sets() -> list:
    """The built-in sets, then every set in paint/swatches (unreadable files are skipped and said in the log)."""
    out = [SwatchSet(s.name, list(s.colours), list(s.names), None, s.note) for s in BUILT_IN]
    for p in sorted(swatch_dir().glob("*")):
        if p.suffix.lower() in (".gpl", ".kpl"):
            try:
                out.append(load(str(p)))
            except (OSError, ValueError, KeyError, ET.ParseError, zipfile.BadZipFile) as err:
                import logging
                logging.getLogger("studio.paint").info("swatch set %s skipped: %s", p.name, err)
    return out


def safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip() or "Swatches"


# ---------------------------------------------------------------- palette from the painting
def from_painting(doc: Document, count: int = 8, samples: int = 40000, seed: int = 0) -> SwatchSet:
    """The ``count`` main colour masses of the visible picture (k-means in Oklab), dark to light."""
    px = doc.composite(background=True)
    a = px[..., 3:4]
    rgb = np.where(a > 0, px[..., :3] / np.maximum(a, 1e-8), 0).reshape(-1, 3)
    keep = a.reshape(-1) > 0.5
    rgb = rgb[keep]
    rng = np.random.default_rng(seed)
    if len(rgb) > samples:
        rgb = rgb[rng.choice(len(rgb), samples, replace=False)]
    lab = srgb_to_oklab(rgb.astype(np.float32)).astype(np.float64)
    uniq = np.unique(np.round(lab, 3), axis=0)
    k = max(1, min(count, len(uniq)))
    # k-means++ start, then Lloyd steps
    cent = [lab[rng.integers(len(lab))]]
    for _ in range(1, k):
        d = np.min(((lab[:, None] - np.array(cent)[None]) ** 2).sum(-1), axis=1)
        if d.sum() <= 0:
            break
        cent.append(lab[rng.choice(len(lab), p=d / d.sum())])
    cent = np.array(cent)
    for _ in range(25):
        lbl = np.argmin(((lab[:, None] - cent[None]) ** 2).sum(-1), axis=1)
        new = np.array([lab[lbl == j].mean(0) if (lbl == j).any() else cent[j] for j in range(len(cent))])
        if np.allclose(new, cent, atol=1e-6):
            break
        cent = new
    cent = cent[np.argsort(cent[:, 0])]
    sw = SwatchSet("From the painting", note="The painting's main colour masses, dark to light.")
    for c in cent:
        rgb_c = oklab_to_srgb(c.astype(np.float32))
        sw.add(rgb_c, to_hex(rgb_c))
    return sw
