"""Paper grain for Paint's brushes (BUILD_PLAN_PAINT P2; paper tooth after Sousa & Buchanan 1999, Van Haevre).

A grain is a tileable height map (0 = the bottom of a valley in the paper, 1 = the top of a ridge), fixed to the
canvas like real paper: strokes move over it, it does not move with them. A light touch only catches the ridges; more
pressure pushes the medium down into the valleys, as pencil, charcoal and pastel do. All grains are made here from
seeded noise (no picture files), so they are the same on every PC, and any picture can be used as a grain too.

Each grain says what it is best for and weak at (plan 23.1).
"""
from __future__ import annotations

import numpy as np

SIDE = 256          # the grain tile, in canvas pixels at scale 1

GRAINS = {
    "cold_press": ("Cold-press paper", "watercolour and gouache texture, soft pencil",
                   "too even for rough charcoal"),
    "rough": ("Rough charcoal paper", "charcoal, pastel and conté: strong broken tooth",
              "fine detail breaks up"),
    "fine": ("Fine grain", "a light tooth for pencil and ink washes, barely there at full pressure",
             "hardly visible on hard, opaque brushes"),
    "canvas": ("Canvas weave", "oil and acrylic dry-brush over a woven canvas", "the weave repeats at large scales"),
    "laid": ("Laid paper", "the lines of handmade laid paper under chalk and pencil", "obviously directional"),
}


def _noise(seed, sigma):
    """Tileable smooth noise: white noise filtered in the frequency domain (wraps around at the edges)."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((SIDE, SIDE))
    f = np.fft.fftfreq(SIDE)
    fx, fy = np.meshgrid(f, f)
    g = np.exp(-(fx ** 2 + fy ** 2) * (2 * np.pi * sigma) ** 2 / 2)
    n = np.real(np.fft.ifft2(np.fft.fft2(w) * g))
    n -= n.min()
    return n / max(n.max(), 1e-9)


def _make(kind: str) -> np.ndarray:
    if kind == "cold_press":
        h = 0.6 * _noise(11, 2.5) + 0.4 * _noise(12, 6.0)
    elif kind == "rough":
        h = 0.45 * _noise(21, 1.5) + 0.55 * _noise(22, 4.0)
        h = h ** 1.6
    elif kind == "fine":
        h = 0.8 * _noise(31, 0.8) + 0.2 * _noise(32, 3.0)
    elif kind == "canvas":
        y, x = np.mgrid[0:SIDE, 0:SIDE] / SIDE * 2 * np.pi * 32        # 32 threads across the tile: it wraps
        weave = 0.5 + 0.25 * np.sin(x) * np.sign(np.sin(y / 2)) + 0.25 * np.sin(y) * np.sign(np.sin(x / 2))
        h = 0.75 * weave + 0.25 * _noise(41, 1.0)
    elif kind == "laid":
        y = np.mgrid[0:SIDE, 0:SIDE][0] / SIDE * 2 * np.pi * 40
        h = 0.6 * (0.5 + 0.5 * np.cos(y)) ** 0.5 + 0.4 * _noise(51, 1.5)
    else:
        raise ValueError(f"unknown grain: {kind}")
    h = h - h.min()
    return (h / max(h.max(), 1e-9)).astype(np.float32)


_cache: dict = {}


def grain(kind: str) -> np.ndarray:
    """The grain tile (SIDE x SIDE float32, 0..1), made once."""
    g = _cache.get(kind)
    if g is None:
        g = _cache[kind] = _make(kind)
    return g


def from_picture(gray: np.ndarray) -> np.ndarray:
    """Any picture (2-D 0..1, light = ridge) as a grain tile."""
    from PIL import Image
    im = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8)).resize((SIDE, SIDE), Image.BILINEAR)
    g = np.asarray(im, np.float32) / 255.0
    g -= g.min()
    return g / max(g.max(), 1e-9)


def patch(tile: np.ndarray, x0: int, y0: int, w: int, h: int, scale: float = 1.0) -> np.ndarray:
    """The grain under a canvas rectangle, the tile repeated and scaled (nearest sample: cheap, stays sharp)."""
    s = max(0.1, scale)
    xs = (np.floor((np.arange(x0, x0 + w)) / s).astype(np.int64)) % SIDE
    ys = (np.floor((np.arange(y0, y0 + h)) / s).astype(np.int64)) % SIDE
    return tile[np.ix_(ys, xs)]


def tooth(height: np.ndarray, pressure: float, strength: float, pressure_fills: bool = True) -> np.ndarray:
    """How much medium the paper takes at each pixel (0..1). The medium reaches down to a level set by pressure (a
    light touch only the ridges, full pressure most of the valleys); ``strength`` 0 = no grain, 1 = full."""
    depth = (0.25 + 0.65 * pressure) if pressure_fills else 0.6
    level = 1.0 - depth                                  # ridges above this level take paint
    t = np.clip((height - level) / 0.18 + 0.5, 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    return (1.0 - strength) + strength * t
