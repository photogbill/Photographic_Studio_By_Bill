"""The tiled canvas (BUILD_PLAN_PAINT 0.5).

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

    def __post_init__(self):
        if not self.layers:
            self.layers.append(Layer("Layer 1"))

    @property
    def layer(self) -> Layer:
        return self.layers[self.active]

    def add_layer(self, name: str | None = None, index: int | None = None) -> Layer:
        lay = Layer(name or f"Layer {len(self.layers) + 1}")
        i = self.active + 1 if index is None else index
        self.layers.insert(i, lay)
        self.active = i
        self.dirty = True
        return lay

    def composite(self, x0: int = 0, y0: int = 0, w: int | None = None, h: int | None = None,
                  background: bool = True) -> np.ndarray:
        """The visible picture in a rectangle: premultiplied RGBA float32 (opaque where there is paper)."""
        w = self.width if w is None else w
        h = self.height if h is None else h
        out = np.zeros((h, w, 4), np.float32)
        if background and self.background is not None:
            out[..., :3] = self.background
            out[..., 3] = 1.0
        for lay in self.layers:                    # index 0 is the bottom layer
            if not lay.visible or lay.opacity <= 0 or not lay.tiles:
                continue
            src = lay.read(x0, y0, w, h)
            if lay.opacity < 1:
                src = src * lay.opacity
            out = blend(out, src, lay.blend)
        return out


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
    else:
        raise ValueError(f"unknown blend mode: {mode}")
    rgb = (1 - da) * src[..., :3] + (1 - sa) * dst[..., :3] + sa * da * b
    return np.concatenate([rgb, sa + da * (1 - sa)], axis=-1).astype(np.float32)
