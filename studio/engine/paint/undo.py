"""Undo and redo by changed tiles (BUILD_PLAN_PAINT 0.5, 0.4.2).

A stroke records each tile's content just before it first changes; undo swaps those back, redo swaps them again, so
both are bit-exact. Layer changes (add, delete, move, properties) are kept as whole-layer-list snapshots, which are
cheap because tiles are shared, not copied. The history is bounded by a memory budget, oldest steps dropped first.
"""
from __future__ import annotations

import copy

import numpy as np

from .tiles import Document, Layer


class _TileStep:
    def __init__(self, name: str):
        self.name = name
        self.before: dict = {}     # (layer uid, tile key) -> tile copy or None (tile did not exist)
        self.after: dict = {}

    def nbytes(self) -> int:
        return sum(t.nbytes for t in list(self.before.values()) + list(self.after.values()) if t is not None)


class _LayersStep:
    def __init__(self, name: str, before: list, active_before: int):
        self.name = name
        self.before, self.active_before = before, active_before
        self.after, self.active_after = None, None

    def nbytes(self) -> int:
        return 1024


def _snapshot_layers(doc: Document) -> list:
    """Layer list with properties copied and tile dicts copied shallowly (tile arrays shared)."""
    out = []
    for lay in doc.layers:
        c = copy.copy(lay)
        c.tiles = dict(lay.tiles)
        out.append(c)
    return out


def _find(doc: Document, uid: int) -> Layer | None:
    for lay in doc.layers:
        if lay.uid == uid:
            return lay
    return None


class UndoStack:
    def __init__(self, doc: Document, budget_mb: float = 1024):
        self.doc = doc
        self.budget = int(budget_mb * 1024 * 1024)
        self.done: list = []
        self.undone: list = []
        self._open: _TileStep | None = None

    # ---- strokes
    def begin(self, name: str = "Stroke") -> None:
        self._open = _TileStep(name)

    def touch(self, layer: Layer, key) -> None:
        """Called by the brush before a tile first changes in the open step."""
        st = self._open
        if st is None:
            return
        k = (layer.uid, key)
        if k not in st.before:
            t = layer.tiles.get(key)
            st.before[k] = None if t is None else t.copy()

    def end(self) -> None:
        st, self._open = self._open, None
        if st is None or not st.before:
            return
        for (uid, key) in st.before:
            lay = _find(self.doc, uid)
            t = lay.tiles.get(key) if lay else None
            st.after[(uid, key)] = None if t is None else t.copy()
        self._push(st)

    # ---- layer changes
    def layers_changing(self, name: str):
        """Call before changing the layer list or a layer's properties; then ``layers_changed()`` after."""
        self._pending = _LayersStep(name, _snapshot_layers(self.doc), self.doc.active)

    def layers_changed(self) -> None:
        st = getattr(self, "_pending", None)
        if st is None:
            return
        st.after, st.active_after = _snapshot_layers(self.doc), self.doc.active
        self._pending = None
        self._push(st)

    # ---- undo / redo
    def can_undo(self) -> bool:
        return bool(self.done)

    def can_redo(self) -> bool:
        return bool(self.undone)

    def undo(self) -> str | None:
        if not self.done:
            return None
        st = self.done.pop()
        self._apply(st, back=True)
        self.undone.append(st)
        self.doc.dirty = True
        return st.name

    def redo(self) -> str | None:
        if not self.undone:
            return None
        st = self.undone.pop()
        self._apply(st, back=False)
        self.done.append(st)
        self.doc.dirty = True
        return st.name

    def _apply(self, st, back: bool) -> None:
        if isinstance(st, _TileStep):
            src = st.before if back else st.after
            for (uid, key), t in src.items():
                lay = _find(self.doc, uid)
                if lay is None:
                    continue
                if t is None:
                    lay.tiles.pop(key, None)
                else:
                    lay.tiles[key] = t.copy()
        else:
            layers = st.before if back else st.after
            self.doc.layers = _snapshot_layers(Document(layers=layers))
            self.doc.active = min(st.active_before if back else st.active_after, len(self.doc.layers) - 1)

    def _push(self, st) -> None:
        self.done.append(st)
        self.undone.clear()
        total = sum(s.nbytes() for s in self.done)
        while len(self.done) > 1 and total > self.budget:
            total -= self.done.pop(0).nbytes()

    def history(self) -> list[str]:
        return [s.name for s in self.done]


def tiles_equal(a: Layer, b: dict) -> bool:
    """Test helper: a layer's tiles equal a dict of tiles exactly."""
    if set(a.tiles) != set(b):
        return False
    return all(np.array_equal(a.tiles[k], b[k]) for k in b)
