"""Autosave and crash recovery (BUILD_PLAN_PAINT 0.4.2: never lose a painting).

Every open painting is written once a minute (when it changed) to ``paint/recovery`` beside the app
(studio/paths.py: never C:\\ or the user profile), as ``<session>.ora`` plus a small ``<session>.json`` saying what it
was. A clean close removes them. Anything still there when Paint starts was left by a crash or a power cut, and
Paint offers it back.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from ... import paths
from . import ora
from .tiles import Document


def recovery_dir() -> Path:
    d = paths.get("PAINT") / "recovery"
    d.mkdir(parents=True, exist_ok=True)
    return d


class AutoSaver:
    def __init__(self, doc: Document, title: str = "Untitled"):
        self.doc, self.title = doc, title
        self.session = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.last = 0.0
        self.generation = 0          # bumped by the canvas when the painting changes
        self.saved_generation = 0

    def changed(self) -> None:
        self.generation += 1

    def due(self, every_s: float = 60.0) -> bool:
        return self.generation != self.saved_generation and time.time() - self.last >= every_s

    def save_now(self) -> Path:
        d = recovery_dir()
        target = d / f"{self.session}.ora"
        ora.save(self.doc, str(target))
        meta = {"session": self.session, "title": self.title, "original": self.doc.path,
                "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "width": self.doc.width, "height": self.doc.height}
        tmp = d / f"{self.session}.json.partial"
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(tmp, d / f"{self.session}.json")
        self.last = time.time()
        self.saved_generation = self.generation
        return target

    def close(self) -> None:
        """The painting was closed on purpose: its recovery copy is no longer needed."""
        for ext in (".ora", ".json"):
            try:
                (recovery_dir() / f"{self.session}{ext}").unlink()
            except FileNotFoundError:
                pass


def recoverable() -> list[dict]:
    """Recovery copies left behind (newest first): each a dict with title, original, saved and the .ora path."""
    out = []
    for js in recovery_dir().glob("*.json"):
        o = js.with_suffix(".ora")
        if not o.exists():
            continue
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        meta["ora"] = str(o)
        out.append(meta)
    return sorted(out, key=lambda m: m.get("saved", ""), reverse=True)


def recover(meta: dict) -> Document:
    doc = ora.load(meta["ora"])
    doc.path = meta.get("original")
    doc.dirty = True
    return doc


def discard(meta: dict) -> None:
    for p in (meta["ora"], meta["ora"][:-4] + ".json"):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
