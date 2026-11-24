"""The tiled canvas (BUILD_PLAN_PAINT 0.5).

Layer groups: ``Document.layers`` stays one flat list, bottom first, so undo snapshots, the brush and the fill need
nothing new. Each layer names its group in ``parent`` (a uid, None at the top level), and a group sits directly
ABOVE its contents in the list (post-order), so a group and everything in it is one unbroken run of the list.

Each layer is a sparse dict of 256 x 256 float32 RGBA tiles, premultiplied, values 0..1 in the canvas's encoded
(sRGB) space. A tile that was never painted does not exist, so a huge canvas costs only what was touched, and undo
stores only the tiles a stroke changed (undo.py).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

TILE = 256

# Blend modes: key -> (name, best for, weak at). Every mode the person can choose says what it is for (plan 23.1).
BLEND_MODES = {
    "normal": ("Normal", "ordinary painting: the layer covers what is below by its opacity",
               "nothing; it is the default"),
    "multiply": ("Multiply", "shadows, line art over colour, glazes that only darken",
                 "it can never lighten; whites vanish"),
    "screen": ("Screen", "glows and light that only brightens",
               "it can never darken; blacks vanish"),
    "overlay": ("Overlay", "adding contrast and texture over a painted base",
                "strong on mid tones, weak near black and white"),
    "add": ("Add (light)", "fire, sparks and light that piles up",
            "clips to white quickly"),
    "darken": ("Darken", "keeping the darker of the two, e.g. ink over paper colour",
               "hard colour edges where the two cross"),
    "lighten": ("Lighten", "keeping the lighter of the two, e.g. highlights on a dark base",
                "hard colour edges where the two cross"),
}

# Groups have one more mode. Groups only.
GROUP_BLEND = {
    "pass": ("Pass-through", "groups used only to keep layers tidy: each layer inside blends with the whole picture "
             "below, exactly as if there were no group", "the group's own blend mode does nothing; group opacity fades "
             "the group's effect as a whole"),
}
GROUP_NOTE = {
    "isolated": "Isolated: the layers inside are put together first, then the result is laid on the picture with the "
                "group's opacity and blend mode. Best for: a Multiply group of shadows, fading a whole figure. Weak "
                "at: a Multiply or Screen layer inside only affects the layers in the same group.",
}


class LayerLocked(Exception):
    """The layer is locked (or is a group with nothing to paint on); the person is told and nothing is painted."""


_ids = itertools.count(1)


def tile_key(x: float, y: float) -> tuple[int, int]:
    return int(np.floor(x / TILE)), int(np.floor(y / TILE))


def empty_tile() -> np.ndarray:
    return np.zeros((TILE, TILE, 4), np.float32)


@dataclass
class Layer:
    name: str = "Layer"
    opacity: float = 1.0
    visible: bool = True
    locked: bool = False          # no painting at all
    alpha_lock: bool = False      # paint changes colour only, never where the layer is see-through
    blend: str = "normal"
    tiles: dict = field(default_factory=dict)
    uid: int = field(default_factory=lambda: next(_ids))
    kind: str = "paint"           # "paint" or "group"
    parent: int | None = None     # uid of the group this layer is in; None = top level
    collapsed: bool = False       # groups: contents hidden in the Layers panel (a view setting, not undone)

    @property
    def is_group(self) -> bool:
        return self.kind == "group"

    def tile(self, key, create: bool = False):
        t = self.tiles.get(key)
        if t is None and create:
            t = empty_tile()
            self.tiles[key] = t
        return t

    def read(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """The layer's pixels in a rectangle (premultiplied RGBA, float32); see-through where never painted."""
        out = np.zeros((h, w, 4), np.float32)
        for (tx, ty), t in self._tiles_in(x0, y0, w, h):
            ox, oy = tx * TILE, ty * TILE
            ax0, ay0 = max(x0, ox), max(y0, oy)
            ax1, ay1 = min(x0 + w, ox + TILE), min(y0 + h, oy + TILE)
            out[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = t[ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox]
        return out

    def write(self, x0: int, y0: int, px: np.ndarray, on_touch=None) -> None:
        """Write premultiplied RGBA pixels at (x0, y0). ``on_touch(layer, key)`` is called before a tile first changes
        (undo keeps its old content)."""
        h, w = px.shape[:2]
        for ty in range(y0 // TILE, (y0 + h - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x0 + w - 1) // TILE + 1):
                ox, oy = tx * TILE, ty * TILE
                ax0, ay0 = max(x0, ox), max(y0, oy)
                ax1, ay1 = min(x0 + w, ox + TILE), min(y0 + h, oy + TILE)
                src = px[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
                if (tx, ty) not in self.tiles and not src[..., 3].any():
                    continue                              # nothing to store in a tile that does not exist
                if on_touch:
                    on_touch(self, (tx, ty))
                self.tile((tx, ty), True)[ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox] = src

    def _tiles_in(self, x0, y0, w, h):
        for ty in range(int(np.floor(y0 / TILE)), int(np.floor((y0 + h - 1) / TILE)) + 1):
            for tx in range(int(np.floor(x0 / TILE)), int(np.floor((x0 + w - 1) / TILE)) + 1):
                t = self.tiles.get((tx, ty))
                if t is not None:
                    yield (tx, ty), t

    def bounds(self):
        """(x0, y0, x1, y1) of the painted tiles, or None when the layer is empty."""
        if not self.tiles:
            return None
        xs = [k[0] for k in self.tiles]
        ys = [k[1] for k in self.tiles]
        return min(xs) * TILE, min(ys) * TILE, (max(xs) + 1) * TILE, (max(ys) + 1) * TILE

    def prune(self) -> None:
        """Drop tiles that became fully see-through (an erased area costs nothing)."""
        for k in [k for k, t in self.tiles.items() if not t[..., 3].any()]:
            del self.tiles[k]


@dataclass
class Document:
    width: int = 2000
    height: int = 1500
    background: tuple = (1.0, 1.0, 1.0)      # paper colour (sRGB 0..1); None is see-through
    layers: list = field(default_factory=list)
    active: int = 0
    path: str | None = None                   # the artist's own file (NAME.ora), when saved
    dirty: bool = False
    bits: int = 8                             # layer depth in the .ora: 8 (every reader) or 16 (smoother glazes)

    def __post_init__(self):
        if not self.layers:
            self.layers.append(Layer("Layer 1"))

    @property
    def layer(self) -> Layer:
        return self.layers[self.active]

    # ---- the tree
    def index_of(self, lay) -> int:
        for i, l in enumerate(self.layers):
            if l is lay or l.uid == getattr(lay, "uid", lay):
                return i
        raise ValueError("layer not in this painting")

    def by_uid(self, uid) -> Layer | None:
        return next((l for l in self.layers if l.uid == uid), None)

    def parent_of(self, lay: Layer) -> Layer | None:
        return None if lay.parent is None else self.by_uid(lay.parent)

    def ancestors(self, lay: Layer) -> list:
        out, p = [], self.parent_of(lay)
        while p is not None:
            out.append(p)
            p = self.parent_of(p)
        return out

    def depth(self, lay: Layer) -> int:
        return len(self.ancestors(lay))

    def start_of(self, i: int) -> int:
        """First list index of node ``i`` and everything inside it (``i`` itself for a plain layer)."""
        lay = self.layers[i]
        if not lay.is_group:
            return i
        s = i
        while s > 0 and lay in self.ancestors(self.layers[s - 1]):
            s -= 1
        return s

    def children(self, parent_uid=None) -> list:
        """List indices of the layers directly inside ``parent_uid`` (None = top level), bottom first."""
        return [i for i, l in enumerate(self.layers) if l.parent == parent_uid]

    def shown(self, lay: Layer) -> bool:
        """Visible, and every group it is in is visible."""
        return lay.visible and all(g.visible for g in self.ancestors(lay))

    def locked_by(self, lay: Layer) -> Layer | None:
        """The layer or group whose lock stops painting on ``lay``, or None."""
        if lay.locked:
            return lay
        return next((g for g in self.ancestors(lay) if g.locked), None)

    def paint_layer(self) -> Layer:
        """The layer a stroke, fill or gradient goes to. A selected group passes the stroke to the top paint layer
        inside it (which becomes the selected layer); a locked layer or group raises LayerLocked with what to do."""
        lay = self.layer
        if lay.is_group:
            s = self.start_of(self.active)
            inside = [j for j in range(self.active - 1, s - 1, -1) if not self.layers[j].is_group]
            if not inside:
                raise LayerLocked(f"\"{lay.name}\" is a group with no layer in it. Add a layer inside it (the + "
                                  "button) to paint.")
            pick = next((j for j in inside if self.locked_by(self.layers[j]) is None), inside[0])
            self.active = pick
            lay = self.layers[pick]
        by = self.locked_by(lay)
        if by is lay:
            raise LayerLocked(f"The layer \"{lay.name}\" is locked. Unlock it in the Layers panel to paint on it.")
        if by is not None:
            raise LayerLocked(f"The layer \"{lay.name}\" is inside the locked group \"{by.name}\". Unlock the "
                              "group in the Layers panel to paint on it.")
        return lay

    def check(self) -> None:
        """Repair a tree that does not hold together (a file from elsewhere, a fault): unknown parents go to the top
        level, and every group's contents are moved directly below it."""
        uids = {l.uid for l in self.layers}
        for l in self.layers:
            if l.parent not in uids or l.parent == l.uid:
                l.parent = None
        order: list = []

        def walk(parent):
            for l in [l for l in self.layers if l.parent == parent]:
                if l.is_group:
                    walk(l.uid)
                order.append(l)
        walk(None)
        seen = {id(l) for l in order}
        order += [l for l in self.layers if id(l) not in seen]        # cycles: kept, at the top level
        for l in order:
            if id(l) not in seen:
                l.parent = None
        cur = self.layers[self.active] if 0 <= self.active < len(self.layers) else None
        self.layers[:] = order
        self.active = order.index(cur) if cur in order else len(order) - 1

    # ---- changing the tree
    def add_layer(self, name: str | None = None, index: int | None = None) -> Layer:
        """A new paint layer above the selected one (inside a selected group: at the top of it)."""
        n_paint = sum(1 for l in self.layers if not l.is_group)
        lay = Layer(name or f"Layer {n_paint + 1}")
        self._insert(lay, index)
        return lay

    def add_group(self, name: str | None = None, wrap: bool = True) -> Layer:
        """A new group. ``wrap``: the selected layer (or group) goes inside it, so selecting a layer and pressing
        Group is all it takes; otherwise an empty group above the selected layer."""
        n = sum(1 for l in self.layers if l.is_group)
        g = Layer(name or f"Group {n + 1}", kind="group", blend="pass")
        if wrap and self.layers:
            i = self.active
            g.parent = self.layers[i].parent
            self.layers[i].parent = g.uid
            self.layers.insert(i + 1, g)
            self.active = i + 1
        else:
            self._insert(g, None)
        self.dirty = True
        return g

    def _insert(self, lay: Layer, index: int | None) -> None:
        if index is not None:
            self.layers.insert(index, lay)
            self.active = index
        elif not self.layers:
            self.layers.append(lay)
            self.active = 0
        else:
            cur = self.layer
            if cur.is_group:                      # top of the group = directly below the group itself
                lay.parent = cur.uid
                i = self.active
            else:
                lay.parent = cur.parent
                i = self.active + 1
            self.layers.insert(i, lay)
            self.active = i
        self.dirty = True

    def delete(self, i: int | None = None) -> bool:
        """Delete a layer, or a group with everything in it. False (nothing done) if no paint layer would be left."""
        i = self.active if i is None else i
        s = self.start_of(i)
        rest = self.layers[:s] + self.layers[i + 1:]
        if not any(not l.is_group for l in rest):
            return False
        self.layers[:] = rest
        self.active = max(0, min(s - 1 if s > 0 else 0, len(rest) - 1))
        self.dirty = True
        return True

    def move(self, d: int, i: int | None = None) -> bool:
        """Move a layer (with everything in it) one place up (d=1) or down (d=-1). Moving up past an open group goes
        into it at the bottom; down past an open group goes in at the top; past the edge of its own group it comes
        out. Closed groups are stepped over as one. False if it cannot move."""
        i = self.active if i is None else i
        s, lay = self.start_of(i), self.layers[i]
        block = self.layers[s:i + 1]
        L = self.layers
        if d > 0:
            if i + 1 >= len(L):
                return False
            nxt = L[i + 1]
            if nxt.uid == lay.parent:                         # top of its group: come out above the group
                L[s:i + 2] = [nxt] + block
                lay.parent = nxt.parent
                self.active = i + 1
            else:
                j = next(k for k in range(i + 1, len(L)) if L[k].parent == lay.parent)   # sibling above
                sib = L[j]
                if sib.is_group and not sib.collapsed:        # in at the bottom: no reordering needed
                    lay.parent = sib.uid
                else:
                    L[s:j + 1] = L[i + 1:j + 1] + block
                    self.active = j
        else:
            if s == 0:
                return False
            prev = L[s - 1]
            if prev.parent == lay.parent:                     # sibling below
                if prev.is_group and not prev.collapsed:      # in at the top: just below the group itself
                    L[s - 1:i + 1] = block + [prev]
                    lay.parent = prev.uid
                    self.active = i - 1
                else:
                    ps = self.start_of(s - 1)
                    L[ps:i + 1] = block + L[ps:s]
                    self.active = ps + (i - s)
            elif lay.parent is not None:                      # bottom of its group: come out below the group
                g = self.parent_of(lay)
                gs = self.start_of(self.index_of(g))
                L[gs:i + 1] = block + L[gs:s]
                lay.parent = g.parent
                self.active = gs + (i - s)
            else:
                return False
        self.dirty = True
        return True

    def ungroup(self, i: int | None = None) -> bool:
        """Remove a group, keeping what was in it where it was. Isolated groups with opacity or a blend mode change
        how the picture looks when removed; the panel says so."""
        i = self.active if i is None else i
        g = self.layers[i]
        if not g.is_group:
            return False
        for l in self.layers:
            if l.parent == g.uid:
                l.parent = g.parent
        del self.layers[i]
        self.active = max(0, i - 1)
        self.dirty = True
        return True

    def flatten_group(self, i: int | None = None) -> Layer | None:
        """Merge a group into one paint layer that looks the same (isolated result, the group's opacity and mode
        kept; pass-through becomes Normal). Returns the new layer."""
        i = self.active if i is None else i
        g = self.layers[i]
        if not g.is_group:
            return None
        s = self.start_of(i)
        keys = set()
        for l in self.layers[s:i]:
            if not l.is_group:
                keys |= set(l.tiles)
        flat = Layer(g.name, opacity=g.opacity, visible=g.visible, locked=g.locked,
                     blend="normal" if g.blend == "pass" else g.blend, parent=g.parent)
        kids = self._kids()
        for (tx, ty) in keys:
            t = self._level(np.zeros((TILE, TILE, 4), np.float32), g.uid, tx * TILE, ty * TILE, TILE, TILE, kids,
                            force_visible=True)
            if t[..., 3].any():
                flat.tiles[(tx, ty)] = t.astype(np.float32)
        self.layers[s:i + 1] = [flat]
        self.active = s
        self.dirty = True
        return flat

    def merge_down(self, i: int | None = None) -> bool:
        """Merge a paint layer into the paint layer directly below it in the same group. False if there is none."""
        i = self.active if i is None else i
        top = self.layers[i]
        if top.is_group or i == 0:
            return False
        low = self.layers[i - 1]
        if low.is_group or low.parent != top.parent:
            return False
        for key in set(top.tiles) | set(low.tiles):
            a = low.tiles.get(key)
            a = np.zeros((TILE, TILE, 4), np.float32) if a is None else a
            b = top.tiles.get(key)
            a = a * low.opacity
            if b is not None and top.visible:
                a = blend(a, b * top.opacity, top.blend)
            low.tiles[key] = a.astype(np.float32)
        low.opacity = 1.0
        low.prune()
        del self.layers[i]
        self.active = i - 1
        self.dirty = True
        return True

    # ---- the picture
    def _kids(self) -> dict:
        kids: dict = {}
        for i, l in enumerate(self.layers):
            kids.setdefault(l.parent, []).append(i)
        return kids

    def _level(self, out, parent, x0, y0, w, h, kids, force_visible=False):
        for i in kids.get(parent, ()):
            lay = self.layers[i]
            if (not lay.visible and not force_visible) or lay.opacity <= 0:
                continue
            if lay.is_group:
                if lay.uid not in kids:
                    continue
                if lay.blend == "pass":                       # inside layers blend with the picture below
                    res = self._level(out.copy(), lay.uid, x0, y0, w, h, kids)
                    out = res if lay.opacity >= 1 else out + (res - out) * lay.opacity
                else:                                         # isolated
                    g = self._level(np.zeros_like(out), lay.uid, x0, y0, w, h, kids)
                    if lay.opacity < 1:
                        g = g * lay.opacity
                    out = blend(out, g, lay.blend)
                continue
            if not lay.tiles:
                continue
            src = lay.read(x0, y0, w, h)
            if lay.opacity < 1:
                src = src * lay.opacity
            out = blend(out, src, lay.blend)
        return out

    def composite(self, x0: int = 0, y0: int = 0, w: int | None = None, h: int | None = None,
                  background: bool = True) -> np.ndarray:
        """The visible picture in a rectangle: premultiplied RGBA float32 (opaque where there is paper)."""
        w = self.width if w is None else w
        h = self.height if h is None else h
        out = np.zeros((h, w, 4), np.float32)
        if background and self.background is not None:
            out[..., :3] = self.background
            out[..., 3] = 1.0
        return self._level(out, None, x0, y0, w, h, self._kids())


def blend(dst: np.ndarray, src: np.ndarray, mode: str = "normal") -> np.ndarray:
    """``src`` over ``dst`` (both premultiplied RGBA) with a separable blend mode (W3C compositing formulas)."""
    sa = src[..., 3:4]
    da = dst[..., 3:4]
    if mode == "normal":
        return src + dst * (1 - sa)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(sa > 0, src[..., :3] / np.maximum(sa, 1e-8), 0)
        dc = np.where(da > 0, dst[..., :3] / np.maximum(da, 1e-8), 0)
    if mode == "multiply":
        b = sc * dc
    elif mode == "screen":
        b = sc + dc - sc * dc
    elif mode == "overlay":
        b = np.where(dc <= 0.5, 2 * sc * dc, 1 - 2 * (1 - sc) * (1 - dc))
    elif mode == "add":
        b = np.minimum(sc + dc, 1)
    elif mode == "darken":
        b = np.minimum(sc, dc)
    elif mode == "lighten":
        b = np.maximum(sc, dc)
    else:                                                 # "pass" on a flattened group, or unknown: Normal
        return src + dst * (1 - sa)
    rgb = (1 - da) * src[..., :3] + (1 - sa) * dst[..., :3] + sa * da * b
    return np.concatenate([rgb, sa + da * (1 - sa)], axis=-1).astype(np.float32)
