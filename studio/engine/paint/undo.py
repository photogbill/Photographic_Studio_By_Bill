"""Undo and redo by changed tiles, kept as a tree: branching undo (BUILD_PLAN_PAINT 0.5, 0.4.2, 0.9).

A stroke records each tile's content just before it first changes and just after; undo swaps the old tiles back,
redo the new ones, so both are bit-exact. Layer changes (add, delete, move, properties) are whole-layer-list
snapshots, cheap because the tile arrays are shared.

Unlike other painting programs, undoing a few steps and then painting does not throw the undone steps away: the new
work starts a **branch**, and ``jump()`` goes to any step on any branch (the History panel). So two lighting or
colour ideas can both be kept and compared. The tree is bounded by a memory budget: branches away from where you are
go first, then the oldest steps.
"""
from __future__ import annotations

import copy
import itertools
import time

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


class Node:
    _ids = itertools.count(1)

    def __init__(self, step, parent):
        self.id = next(Node._ids)
        self.step, self.parent = step, parent
        self.children: list = []
        self.last_child = None        # the branch redo follows
        self.time = time.time()

    @property
    def name(self) -> str:
        return self.step.name if self.step is not None else "Start"


def _snapshot_layers(doc: Document) -> list:
    """Layer list with properties copied and tile dicts copied shallowly (tile arrays shared)."""
    out = []
    for lay in doc.layers:
        c = copy.copy(lay)
        c.tiles = dict(lay.tiles)
        c.height = dict(lay.height)
        c.mask = None if lay.mask is None else dict(lay.mask)
        c.adj = None if lay.adj is None else copy.deepcopy(lay.adj)
        c.wet = None
        out.append(c)
    return out


def _slot(lay: Layer, key):
    """Undo keys are (tx, ty) for colour tiles and ("h", tx, ty) for thickness (impasto) tiles."""
    if len(key) == 3 and key[0] == "h":
        return lay.height, key[1:]
    if len(key) == 3 and key[0] == "m":
        if lay.mask is None:
            lay.mask = {}
        return lay.mask, key[1:]
    return lay.tiles, key


def _find(doc: Document, uid: int) -> Layer | None:
    for lay in doc.layers:
        if lay.uid == uid:
            return lay
    return None


class UndoStack:
    def __init__(self, doc: Document, budget_mb: float = 1024):
        self.doc = doc
        self.budget = int(budget_mb * 1024 * 1024)
        self.root = Node(None, None)
        self.current = self.root
        self._open: _TileStep | None = None
        self._pending = None

    # ---- strokes
    def begin(self, name: str = "Stroke") -> None:
        if isinstance(self.current.step, _TileStep):
            self._settle(self.current.step)
        self._open = _TileStep(name)

    def touch(self, layer: Layer, key) -> None:
        """Called by the brush before a tile first changes in the open step."""
        st = self._open
        if st is None:
            return
        k = (layer.uid, key)
        if k not in st.before:
            d, kk = _slot(layer, key)
            t = d.get(kk)
            st.before[k] = None if t is None else t.copy()

    def end(self) -> None:
        st, self._open = self._open, None
        if st is None or not st.before:
            return
        for (uid, key) in st.before:
            lay = _find(self.doc, uid)
            t = _slot(lay, key)[0].get(_slot(lay, key)[1]) if lay else None
            st.after[(uid, key)] = None if t is None else t.copy()
        st.wet = any(l.wet is not None for l in self.doc.layers)
        self._push(st)

    def amend(self, layer: Layer, key, old) -> None:
        """Watercolour drying after a stroke ended: changes go into the latest stroke's step, so undoing that stroke
        takes back the drying too. ``old`` is the tile before this change (None: there was none)."""
        st = self._open if self._open is not None else self.current.step
        if not isinstance(st, _TileStep):
            return
        k = (layer.uid, key)
        if k not in st.before:
            st.before[k] = None if old is None else old.copy()
        if st is not self._open:
            st.__dict__.setdefault("stale", set()).add(k)     # its "after" is copied when it is needed

    def _settle(self, st) -> None:
        """Copy the current tiles as the "after" of a step whose tiles drying changed after it ended."""
        stale = getattr(st, "stale", None)
        if not stale:
            return
        for (uid, key) in stale:
            lay = _find(self.doc, uid)
            if lay is None:
                continue
            d, kk = _slot(lay, key)
            t = d.get(kk)
            st.after[(uid, key)] = None if t is None else t.copy()
        stale.clear()

    # ---- layer changes
    def layers_changing(self, name: str):
        """Call before changing the layer list or a layer's properties; then ``layers_changed()`` after."""
        self.doc.dry_all()
        if isinstance(self.current.step, _TileStep):
            self._settle(self.current.step)
        self._pending = _LayersStep(name, _snapshot_layers(self.doc), self.doc.active)
        self._pending.doc_before = (self.doc.space, self.doc.background)

    def layers_changed(self) -> None:
        st, self._pending = self._pending, None
        if st is None:
            return
        st.after, st.active_after = _snapshot_layers(self.doc), self.doc.active
        st.doc_after = (self.doc.space, self.doc.background)
        self._push(st)

    # ---- undo / redo / jump
    def can_undo(self) -> bool:
        return self.current is not self.root

    def can_redo(self) -> bool:
        return self.current.last_child is not None

    def undo(self) -> str | None:
        self.doc.dry_all()
        for l in self.doc.layers:
            l.wet = None
        if isinstance(self.current.step, _TileStep):
            self._settle(self.current.step)
        if self.current is self.root:
            return None
        n = self.current
        self._apply(n.step, back=True)
        self.current = n.parent
        self.current.last_child = n
        self.doc.dirty = True
        return n.name

    def redo(self) -> str | None:
        self.doc.dry_all()
        n = self.current.last_child
        if n is None:
            return None
        self._apply(n.step, back=False)
        self.current = n
        self.doc.dirty = True
        return n.name

    def jump(self, target: Node) -> None:
        """Go to any step on any branch: undo up to where the two paths meet, then redo down to ``target``."""
        up = set(id(n) for n in self.path(target))
        while id(self.current) not in up:
            self.undo()
        down = []
        n = target
        while n is not self.current:
            down.append(n)
            n = n.parent
        for n in reversed(down):
            self.current.last_child = n
            self.redo()

    def path(self, node: Node | None = None) -> list:
        """The steps from the start to ``node`` (default: where you are), the start included."""
        n, out = node or self.current, []
        while n is not None:
            out.append(n)
            n = n.parent
        return out[::-1]

    def _apply(self, st, back: bool) -> None:
        if isinstance(st, _TileStep):
            src = st.before if back else st.after
            for (uid, key), t in src.items():
                lay = _find(self.doc, uid)
                if lay is None:
                    continue
                d, kk = _slot(lay, key)
                if t is None:
                    d.pop(kk, None)
                else:
                    d[kk] = t.copy()
        else:
            layers = st.before if back else st.after
            self.doc.layers = _snapshot_layers(Document(layers=layers))
            state = getattr(st, "doc_before" if back else "doc_after", None)
            if state is not None:                              # the working space and paper go back too
                self.doc.space, self.doc.background = state
            self.doc.active = min(st.active_before if back else st.active_after, len(self.doc.layers) - 1)

    def _push(self, st) -> None:
        n = Node(st, self.current)
        self.current.children.append(n)
        self.current.last_child = n
        self.current = n
        self._trim()

    # ---- memory budget
    def _nodes(self, start=None):
        stack = [start or self.root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children)

    def nbytes(self) -> int:
        return sum(n.step.nbytes() for n in self._nodes() if n.step is not None)

    def _trim(self) -> None:
        total = self.nbytes()
        on_path = set(id(n) for n in self.path())
        while total > self.budget:
            # first the oldest branch that is not on the way to where you are
            off = [n for n in self._nodes() if id(n) not in on_path and n.parent is not None
                   and id(n.parent) in on_path]
            if off:
                victim = min(off, key=lambda n: n.time)
                total -= sum(m.step.nbytes() for m in self._nodes(victim))
                victim.parent.children.remove(victim)
                if victim.parent.last_child is victim:
                    victim.parent.last_child = None
                continue
            # then the oldest step: its child becomes the new start
            if not self.root.children or self.root is self.current or self.current.parent is self.root:
                break                                    # the last step can always be undone
            child = self.root.children[0]
            total -= child.step.nbytes() if child.step is not None else 0
            child.step, child.parent = None, None
            self.root = child
            on_path = set(id(n) for n in self.path())

    # ---- side by side: two branches at once
    def picture_at(self, node: Node, max_side: int = 1200):
        """The visible picture at any step (premultiplied RGBA, shrunk by whole-pixel averaging so its longer side
        is at most ``max_side``), leaving the painting exactly where it was."""
        here, dirty = self.current, self.doc.dirty
        try:
            self.jump(node)
            img = self.doc.composite()
        finally:
            self.jump(here)
            self.doc.dirty = dirty
        f = max(1, -(-max(img.shape[:2]) // max_side))
        if f > 1:
            h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
            img = img[:h, :w].reshape(h // f, f, w // f, f, 4).mean(axis=(1, 3))
        return img

    def other_tip(self) -> Node | None:
        """The newest step at the end of a branch you are not on: the other idea, for side by side."""
        on_path = set(id(n) for n in self.path())
        tips = [n for n in self._nodes() if not n.children and id(n) not in on_path]
        if not tips:                                        # no other branch: the end of your own redo path
            n = self.current
            while n.last_child is not None:
                n = n.last_child
            return n if n is not self.current else None
        return max(tips, key=lambda n: n.time)

    # ---- for the History panel
    def history(self) -> list[str]:
        """Names of the steps from the start to where you are."""
        return [n.name for n in self.path()[1:]]

    def tree(self) -> list:
        """Every step for the History panel, depth first, newest branch last: (node, depth, branch_start)."""
        out = []
        stack = [(self.root, 0, False)]                  # no recursion: a long painting has thousands of steps
        while stack:
            n, depth, branch = stack.pop()
            out.append((n, depth, branch))
            for i, c in reversed(list(enumerate(n.children))):
                stack.append((c, depth + (1 if i > 0 else 0), i > 0))
        return out


def tiles_equal(a: Layer, b: dict) -> bool:
    """Test helper: a layer's tiles equal a dict of tiles exactly."""
    if set(a.tiles) != set(b):
        return False
    return all(np.array_equal(a.tiles[k], b[k]) for k in b)
