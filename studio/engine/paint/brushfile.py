"""Brushes on disk and the brush editor's live preview stroke (BUILD_PLAN_PAINT P2).

Saved brushes live in ``paint/brushes`` beside the app (studio/paths.py) as ``NAME.json``, with ``NAME_tip.png`` and
``NAME_grain.png`` when the brush has a picture tip or a picture grain. MyPaint brushes (.myb) in the same folder are
read by the MyPaint engine.

The preview is a real stroke painted by the real engine on a small canvas: an S-curve whose pressure rises from a
light touch to full and eases off again, so size, opacity, grain and jitter can all be seen changing along it (Moko's
idea). Smudge brushes are given colour bands to pull; erasers are given paint to remove.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np

from .brush import BrushSettings, make_stroke
from .curves import Curve
from .smoothing import Sample
from .tiles import Document

_SKIP = {"tip", "grain_tile", "myb", "dynamics", "tips"}


def to_dict(b: BrushSettings) -> dict:
    d = {k: v for k, v in vars(b).items() if k not in _SKIP}
    d["dynamics"] = {t: {i: c.to_list() for i, c in ins.items()} for t, ins in b.dynamics.items()}
    d["version"] = 1
    return d


def from_dict(d: dict) -> BrushSettings:
    known = set(BrushSettings.__dataclass_fields__)
    kw = {k: v for k, v in d.items() if k in known and k not in _SKIP}
    b = BrushSettings(**kw)
    b.dynamics = {t: {i: Curve(pts) for i, pts in ins.items()} for t, ins in d.get("dynamics", {}).items()}
    for t in ("size", "opacity", "flow"):
        b.dynamics.setdefault(t, {})
    return b


def brush_dir() -> Path:
    from ... import paths
    d = paths.get("PAINT") / "brushes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip() or "Brush"


def _png(path, a):
    from PIL import Image
    Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def _read_png(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), np.float32) / 255.0


def save(b: BrushSettings, folder: str | Path | None = None) -> Path:
    """Write the brush (atomic). A picture tip or grain goes beside it as a PNG."""
    folder = Path(folder) if folder else brush_dir()
    stem = safe_name(b.name)
    d = to_dict(b)
    if b.tip is not None:
        _png(folder / f"{stem}_tip.png", b.tip)
        d["tip_file"] = f"{stem}_tip.png"
    if b.grain_tile is not None:
        _png(folder / f"{stem}_grain.png", b.grain_tile)
        d["grain_file"] = f"{stem}_grain.png"
    if b.tips:
        d["tip_files"] = []
        for k, t in enumerate(b.tips):
            _png(folder / f"{stem}_tip{k + 1}.png", t)
            d["tip_files"].append(f"{stem}_tip{k + 1}.png")
    p = folder / f"{stem}.json"
    tmp = str(p) + ".partial"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, p)
    return p


def load(path: str | Path) -> BrushSettings:
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    b = from_dict(d)
    if d.get("tip_file") and (p.parent / d["tip_file"]).is_file():
        b.tip = _read_png(p.parent / d["tip_file"])
    b.tips = [_read_png(p.parent / f) for f in d.get("tip_files", []) if (p.parent / f).is_file()]
    if d.get("grain_file") and (p.parent / d["grain_file"]).is_file():
        b.grain_tile = _read_png(p.parent / d["grain_file"])
    return b


def saved(folder: str | Path | None = None) -> list:
    """Every saved brush (unreadable files are skipped and said in the log)."""
    import logging
    folder = Path(folder) if folder else brush_dir()
    out = []
    for p in sorted(folder.glob("*.json")):
        try:
            out.append(load(p))
        except (OSError, ValueError, TypeError, KeyError) as err:
            logging.getLogger("studio.paint").info("brush %s skipped: %s", p.name, err)
    return out


# ---------------------------------------------------------------- the live preview
def preview_samples(w: int, h: int) -> list:
    """An S-curve across the strip; pressure light -> full -> easing off, speed varying, a little tilt."""
    n = 140
    pts = []
    for i in range(n):
        u = i / (n - 1)
        x = 0.07 * w + u * 0.86 * w
        y = h / 2 + math.sin(u * 2 * math.pi) * h * 0.22
        p = min(1.0, 0.08 + 1.6 * u) if u < 0.55 else max(0.25, 1.0 - (u - 0.55) * 1.6)
        pts.append(Sample(x, y, p, u * 0.6, 20 * u, 0.0, 0.0, 600 + 2400 * u))
    return pts


def preview(b: BrushSettings, w: int = 360, h: int = 110, colour=(0.12, 0.14, 0.22),
            paper=(0.97, 0.95, 0.90), seed: int = 0) -> np.ndarray:
    """The brush painted by the engine on a small strip: (h, w, 3) uint8 sRGB."""
    doc = Document(w, h, background=tuple(paper))
    lay = doc.layer
    if b.mode == "smudge" or b.eraser or (b.mode == "mypaint" and b.myb is not None
                                          and b.myb.base.get("smudge", 0) > 0):
        bands = [(0.85, 0.25, 0.2), (0.95, 0.8, 0.2), (0.2, 0.45, 0.8), (0.25, 0.6, 0.35)]
        px = np.zeros((h, w, 4), np.float32)
        for k, c in enumerate(bands):
            px[:, k * w // 4:(k + 1) * w // 4, :3] = c
        px[..., 3] = 1.0
        lay.write(0, 0, px)
    st = make_stroke(lay, b, colour, None, seed=seed, sample=lay.read)
    for s in preview_samples(w, h):
        st.add(s)
    st.finish()
    img = doc.composite()
    return (np.clip(img[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
