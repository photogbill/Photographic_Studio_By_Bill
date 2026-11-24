"""Editing with selections, masks, transforms and filters (BUILD_PLAN_PAINT P4). Every change goes through the
undo stack as one step. ``sel`` is a canvas-sized soft mask (select.py) or None for the whole layer.
"""
from __future__ import annotations

import numpy as np

from . import adjust, select, transform
from .tiles import TILE, Document, Layer


def _rect(doc, lay, sel):
    if sel is not None:
        b = select.bounds(sel)
        return b
    b = lay.bounds()
    if b is None:
        return None
    x0, y0, x1, y1 = b
    return max(0, x0), max(0, y0), min(doc.width, x1), min(doc.height, y1)


def _put(lay, x0, y0, px, undo, name):
    if undo is not None:
        undo.begin(name)
    lay.write(x0, y0, px.astype(np.float32), on_touch=undo.touch if undo is not None else None)
    lay.prune()
    if undo is not None:
        undo.end()


def clear(doc: Document, lay: Layer, sel, undo=None):
    """Delete what is inside the selection (soft edges fade)."""
    r = _rect(doc, lay, sel)
    if r is None:
        return None
    x0, y0, x1, y1 = r
    px = lay.read(x0, y0, x1 - x0, y1 - y0)
    k = 1.0 if sel is None else (1 - sel[y0:y1, x0:x1, None])
    _put(lay, x0, y0, px * k, undo, "Delete")
    return r


def fill_selection(doc, lay, sel, colour, undo=None):
    if sel is None:
        sel = np.ones((doc.height, doc.width), np.float32)
    r = select.bounds(sel)
    if r is None:
        return None
    x0, y0, x1, y1 = r
    m = sel[y0:y1, x0:x1, None]
    base = lay.read(x0, y0, x1 - x0, y1 - y0)
    out = np.concatenate([np.asarray(colour[:3], np.float32) * m, m], -1) + base * (1 - m)
    _put(lay, x0, y0, out, undo, "Fill selection")
    return r


def copy(doc, lay, sel):
    """The pixels inside the selection: (premultiplied RGBA, x0, y0), or None."""
    r = _rect(doc, lay, sel)
    if r is None:
        return None
    x0, y0, x1, y1 = r
    px = lay.read(x0, y0, x1 - x0, y1 - y0)
    if sel is not None:
        px = px * sel[y0:y1, x0:x1, None]
    return px, x0, y0


def paste(doc, clip, undo=None, name="Pasted"):
    """A new layer above the selected one holding the copied pixels, where they came from."""
    px, x0, y0 = clip
    if undo is not None:
        undo.layers_changing("Paste")
    lay = doc.add_layer(name)
    lay.write(x0, y0, px.astype(np.float32))
    if undo is not None:
        undo.layers_changed()
    return lay


def filter_layer(doc, lay, kind, amount=1.0, sel=None, undo=None):
    r = _rect(doc, lay, sel)
    if r is None:
        return None
    x0, y0, x1, y1 = r
    pad = 24
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(doc.width, x1 + pad), min(doc.height, y1 + pad)
    px = lay.read(X0, Y0, X1 - X0, Y1 - Y0)
    out = adjust.filter_pixels(kind, px, amount)
    if sel is not None:
        k = sel[Y0:Y1, X0:X1, None]
        out = px + (out - px) * k
    _put(lay, X0, Y0, out, undo, adjust.FILTERS[kind][0])
    return X0, Y0, X1, Y1


class TransformSession:
    """Transform the selected part of a layer (or all of it), always from the pixels kept at the start."""

    def __init__(self, doc: Document, lay: Layer, sel=None):
        self.doc, self.lay, self.sel = doc, lay, sel
        self.before = {k: t.copy() for k, t in lay.tiles.items()}
        full = lay.read(0, 0, doc.width, doc.height)
        k = np.ones((doc.height, doc.width, 1), np.float32) if sel is None else sel[..., None]
        self.moving = full * k                               # what moves
        self.staying = full * (1 - k)                        # what stays
        b = select.bounds(self.moving[..., 3])
        self.box = b or (0, 0, doc.width, doc.height)
        self.liquify = None

    def corners(self):
        x0, y0, x1, y1 = self.box
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def _show(self, moved):
        from .tiles import blend
        out = blend(self.staying, moved, "normal")
        self.lay.tiles = {}
        self.lay.write(0, 0, out.astype(np.float32))
        self.lay.prune()

    def free(self, **kw):
        x0, y0, x1, y1 = self.box
        kw.setdefault("cx", (x0 + x1) / 2)
        kw.setdefault("cy", (y0 + y1) / 2)
        M = transform.affine(**kw)
        self._show(transform.warp_with_matrix(self.moving, M, self.moving.shape[:2]))
        return M

    def perspective(self, dst):
        H = transform.homography(self.corners(), dst)
        self._show(transform.warp_with_matrix(self.moving, H, self.moving.shape[:2]))
        return H

    def warp(self, offsets, n=None):
        """``offsets``: (n, n, 2) movements of an n x n grid laid over the box."""
        x0, y0, x1, y1 = self.box
        n = np.asarray(offsets).shape[0] if n is None else n
        h, w = self.moving.shape[:2]
        gx = np.linspace(0, w, n)
        gy = np.linspace(0, h, n)
        src = np.stack(np.meshgrid(gx, gy), -1)
        inside = ((src[..., 0] >= x0 - 1) & (src[..., 0] <= x1 + 1) & (src[..., 1] >= y0 - 1) &
                  (src[..., 1] <= y1 + 1))
        dst = src + np.asarray(offsets, np.float64) * (1 if inside.any() else 1)
        self._show(transform.grid_warp(self.moving, src, dst))

    def mesh(self, offsets):
        """A dense grid (n x n from the offsets' shape)."""
        self.warp(offsets, n=np.asarray(offsets).shape[0])

    def cage(self, src_cage, dst_cage):
        """What is inside the cage moves with it; the rest of the layer stays."""
        from .select import polygon
        from .tiles import blend
        h, w = self.moving.shape[:2]
        inside = polygon(w, h, [tuple(p) for p in src_cage])[..., None]
        moved = transform.cage_warp(self.moving, src_cage, dst_cage)
        stays = blend(self.staying, self.moving * (1 - inside), "normal")
        out = blend(stays, moved, "normal")
        self.lay.tiles = {}
        self.lay.write(0, 0, out.astype(np.float32))
        self.lay.prune()

    def liquify_push(self, x, y, mx, my, radius, strength=0.6):
        if self.liquify is None:
            self.liquify = transform.Liquify(self.moving)
        r = self.liquify.push(x, y, mx, my, radius, strength)
        if r is not None:
            self._show(self.liquify.result())
        return r

    def commit(self, undo=None, name="Transform"):
        """One undo step from before the transform to now."""
        if undo is None:
            return
        after = self.lay.tiles
        self.lay.tiles = dict(self.before)
        undo.begin(name)
        for k in set(self.before) | set(after):
            undo.touch(self.lay, k)
        self.lay.tiles = after
        undo.end()

    def cancel(self):
        self.lay.tiles = dict(self.before)


# ---- masks
def add_mask(lay: Layer, sel=None, doc: Document | None = None):
    """A mask that shows everything, or what the selection selects."""
    lay.mask = {}
    if sel is not None and doc is not None:
        H, W = sel.shape
        for ty in range(0, H, TILE):
            for tx in range(0, W, TILE):
                t = np.ones((TILE, TILE), np.float32)
                sub = sel[ty:ty + TILE, tx:tx + TILE]
                t[:sub.shape[0], :sub.shape[1]] = sub
                lay.mask[(tx // TILE, ty // TILE)] = t
    return lay.mask


def apply_mask(lay: Layer):
    """Bake the mask into the paint and remove it."""
    if not lay.mask:
        lay.mask = None
        return
    for k, t in list(lay.tiles.items()):
        m = lay.mask.get(k)
        if m is not None:
            lay.tiles[k] = t * m[..., None]
    lay.mask = None
    lay.prune()
