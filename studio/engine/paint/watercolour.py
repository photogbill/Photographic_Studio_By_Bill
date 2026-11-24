"""Watercolour that flows and dries (BUILD_PLAN_PAINT P3), after Curtis et al., "Computer-Generated Watercolor"
(SIGGRAPH 1997) and Van Laerhoven & Van Reeth (2005): a grid of water and pigment on paper, not polygons (Adobe's
polygon watercolour patents, US 8,917,282 / 283, are kept clear of).

A watercolour layer keeps, while it is wet, three things over a working box around the wet paint:
- water ``W`` (how wet the paper is),
- pigment suspended in the water ``G`` and pigment settled on the paper ``D``, both as optical density per
  channel (a brush colour c carries density -ln c, so a full-strength stroke dries to exactly c).
Each step: water evens out inside the wet area (it never runs onto dry paper); suspended pigment spreads within the
wet area and is carried towards the edges, where water evaporates fastest (the dark edges of a dried wash), and away
from fresh water dropped into damp paint (back-runs, "blooms"); some pigment settles, more in the paper's valleys
(granulation); water evaporates; where the paper dries, the rest settles. A brush with clean water loosens settled
pigment: most is carried off by the brush, the rest goes back into the water (lifting).

The layer shows transmittance exp(-(D + G)) through Multiply, so glazes darken as layered watercolour does, and
Krita and GIMP show the layer the same way. Drying runs in steps (the canvas calls ``step`` on a timer); undo, layer
changes and saving finish it at once (``force_dry``).

Speed: water, suspended pigment and their flow are simulated at half resolution (``scale`` 2); settled pigment and
the paper's grain stay at full resolution, so edges and granulation keep their detail.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from . import texture as _tex
from .tiles import TILE, Layer, read_tiles

PAD = 48


def _down(a, f):
    """Block mean by f (the full-resolution box is always a multiple of f)."""
    if f == 1:
        return a
    h, w = a.shape[0] // f, a.shape[1] // f
    return a[:h * f, :w * f].reshape(h, f, w, f, *a.shape[2:]).mean(axis=(1, 3))


def _up(a, f):
    if f == 1:
        return a
    return np.repeat(np.repeat(a, f, 0), f, 1)


class WetSim:
    def __init__(self, layer: Layer, grain: str = "cold_press", granulation: float = 0.5,
                 edge_darkening: float = 0.6, dry_speed: float = 1.0, scale: int = 2):
        self.layer = layer
        self.f = max(1, int(scale))
        self.box = None                       # (x0, y0, w, h) full-resolution canvas pixels, multiples of f
        self.W = self.G = self.S = None       # half resolution: water, suspended pigment, this stroke's coverage
        self.D = None                         # full resolution: settled pigment
        self.grain = grain
        self.granulation = granulation
        self.edge = edge_darkening
        self.dry_speed = dry_speed
        self.undo = None
        self.steps = 0
        self._noted: set = set()
        self._noted_step = None

    @property
    def active(self) -> bool:
        return self.W is not None and bool((self.W > 1e-4).any())

    def new_stroke(self):
        if self.S is not None:
            self.S[:] = 0

    # ---- the working box
    def _ensure(self, x0, y0, x1, y1):
        f = self.f
        x0, y0, x1, y1 = x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD
        if self.box is not None:
            bx, by, bw, bh = self.box
            if x0 >= bx and y0 >= by and x1 <= bx + bw and y1 <= by + bh:
                return
            x0, y0 = min(x0, bx), min(y0, by)
            x1, y1 = max(x1, bx + bw), max(y1, by + bh)
        x0, y0 = (x0 // f) * f, (y0 // f) * f
        x1, y1 = -(-x1 // f) * f, -(-y1 // f) * f
        w, h = x1 - x0, y1 - y0
        T = read_tiles(self.layer.tiles, x0, y0, w, h, 4)
        eff = 1.0 - T[..., 3:4] + T[..., :3]                  # what Multiply shows: 1 - a + a * colour
        D = -np.log(np.clip(eff, 1e-4, 1.0)).astype(np.float32)
        W = np.zeros((h // f, w // f), np.float32)
        G = np.zeros((h // f, w // f, 3), np.float32)
        S = np.zeros((h // f, w // f), np.float32)
        if self.box is not None:                              # keep what was wet
            bx, by, bw, bh = self.box
            hs = (slice((by - y0) // f, (by - y0 + bh) // f), slice((bx - x0) // f, (bx - x0 + bw) // f))
            W[hs], G[hs], S[hs] = self.W, self.G, self.S
            D[by - y0:by - y0 + bh, bx - x0:bx - x0 + bw] = self.D
        self.box, self.W, self.G, self.S, self.D = (x0, y0, w, h), W, G, S, D
        tile = _tex.grain(self.grain) if self.grain in _tex.GRAINS else _tex.grain("cold_press")
        self.paper = _tex.patch(tile, x0, y0, w, h)           # full resolution
        self.paper_lo = _down(self.paper, f)

    # ---- the brush puts water and pigment down
    def add(self, m: np.ndarray, x0: int, y0: int, water: float, density, lift: float = 0.0):
        """One dab: mask ``m`` at (x0, y0) adds ``water`` and pigment of optical ``density`` (3,); ``lift`` > 0
        loosens settled pigment (most leaves with the brush)."""
        f = self.f
        h, w = m.shape
        self._ensure(x0, y0, x0 + w, y0 + h)
        bx, by, _, _ = self.box
        # the dab on the half-resolution grid
        lx0, ly0 = (x0 - bx) // f, (y0 - by) // f
        lx1, ly1 = -(-(x0 - bx + w) // f), -(-(y0 - by + h) // f)
        full = np.zeros(((ly1 - ly0) * f, (lx1 - lx0) * f), np.float32)
        ox, oy = x0 - bx - lx0 * f, y0 - by - ly0 * f
        full[oy:oy + h, ox:ox + w] = m
        ml = _down(full, f)
        sl = (slice(ly0, ly1), slice(lx0, lx1))
        W, G, S = self.W[sl], self.G[sl], self.S[sl]
        inc = ml * (1.0 - S)                                  # like a wash: one stroke lays its colour once
        S += inc
        W += inc * water * (1.2 - np.minimum(W, 1.0))
        np.minimum(W, 2.0, out=W)
        G += inc[..., None] * np.asarray(density, np.float32)[None, None, :]
        if lift > 0:
            fsl = (slice(ly0 * f, ly1 * f), slice(lx0 * f, lx1 * f))
            D = self.D[fsl]
            up = D * (_up(inc, f)[..., None] * lift)
            D -= up
            G += _down(up, f) * 0.35
        self._write(x0, y0, x0 + w, y0 + h)

    # ---- flow and drying
    def step(self, n: int = 1):
        if not self.active:
            return None
        lo = None
        for _ in range(n):
            r = self._step()
            if r is not None:
                lo = r if lo is None else (min(lo[0], r[0]), min(lo[1], r[1]), max(lo[2], r[2]), max(lo[3], r[3]))
        bx, by, bw, bh = self.box
        if not self.active:
            self._settle_all()
            return self.box_rect()
        if lo is None:
            return None
        f = self.f
        rect = (bx + lo[0] * f, by + lo[1] * f, bx + lo[2] * f, by + lo[3] * f)
        self._write(*rect)
        return rect

    def box_rect(self):
        bx, by, bw, bh = self.box
        return bx, by, bx + bw, by + bh

    def _step(self):
        """One step over the part still wet (with a margin): it gets cheaper as the wash dries."""
        wet = self.W > 1e-4
        rows, cols = np.nonzero(wet.any(1))[0], np.nonzero(wet.any(0))[0]
        if not len(rows):
            return None
        m = 8
        y0, y1 = max(0, rows[0] - m), min(self.W.shape[0], rows[-1] + m + 1)
        x0, x1 = max(0, cols[0] - m), min(self.W.shape[1], cols[-1] + m + 1)
        f = self.f
        W, G = self.W[y0:y1, x0:x1], self.G[y0:y1, x0:x1]
        D = self.D[y0 * f:y1 * f, x0 * f:x1 * f]
        self._step_on(W, G, D, self.paper_lo[y0:y1, x0:x1], self.paper[y0 * f:y1 * f, x0 * f:x1 * f])
        return x0, y0, x1, y1

    def _step_on(self, W, G, D, paper_lo, paper):
        f = self.f
        M = (W > 1e-4).astype(np.float32)
        self.steps += 1
        cnt = uniform_filter(M, 3)
        Wa = uniform_filter(W * M, 3) / np.maximum(cnt, 1e-6)          # water evens out inside the wet area
        W[:] = (W + 0.5 * (Wa - W)) * M
        Ga = uniform_filter(G * M[..., None], (3, 3, 1)) / np.maximum(cnt, 1e-6)[..., None]
        G += 0.08 * (Ga - G) * M[..., None]                             # pigment spreads within it
        # a steady drift outwards near the edge (fastest evaporation there: the dark edges) and away from fresh
        # water (back-runs); distances in half-resolution pixels, so the drift is halved to keep the same look
        P = uniform_filter(M, max(5, 15 // f))
        Ws = uniform_filter(W, max(3, 7 // f))
        px, py = np.gradient(P, axis=1), np.gradient(P, axis=0)
        mag = np.sqrt(px * px + py * py)
        band = mag > 1e-3
        ux = np.where(band, px / np.maximum(mag, 1e-6), 0.0)
        uy = np.where(band, py / np.maximum(mag, 1e-6), 0.0)
        wx, wy = np.gradient(Ws, axis=1), np.gradient(Ws, axis=0)
        vx = -(0.22 / f * self.edge * ux + 0.8 * wx) * M
        vy = -(0.22 / f * self.edge * uy + 0.8 * wy) * M
        self._advect(G, vx, vy, M)
        # settling: more as the water thins; granulation at full resolution from the paper's height
        thin = np.clip(1.0 - W / 0.6, 0.0, 1.0)
        rate = np.clip((0.004 + 0.03 * thin) * M, 0, 1)
        dep = G * rate[..., None]
        G -= dep
        gran = self._gran(paper)
        D += _up(dep, f) * gran[..., None]
        W -= (0.012 * self.dry_speed) * (1.0 + 1.5 * (1.0 - P)) * M     # evaporation, fastest at the edges
        dry = (W <= 0) & (M > 0)
        if dry.any():
            g = np.where(dry[..., None], G, 0.0)
            D += _up(g, f) * gran[..., None]
            G[dry] = 0
            W[dry] = 0

    def _gran(self, paper):
        """Granulation: settling weighted towards the paper's valleys, normalised within each f x f block, so it
        moves pigment about and never makes or loses any."""
        gran = 1.0 + self.granulation * 1.6 * (0.5 - paper)
        return (gran / np.maximum(_up(_down(gran, self.f), self.f), 1e-6)).astype(np.float32)

    @staticmethod
    def _advect(G, vx, vy, M):
        """Upwind transport of pigment between wet pixels only (conserves pigment)."""
        for axis, v in ((1, vx), (0, vy)):
            vf = 0.5 * (v + np.roll(v, -1, axis))
            vf = np.clip(vf, -0.24, 0.24) * (M * np.roll(M, -1, axis))
            Gn = np.roll(G, -1, axis)
            flux = np.where((vf > 0)[..., None], G, Gn) * vf[..., None]
            if axis == 1:
                flux[:, -1] = 0
            else:
                flux[-1] = 0
            G -= flux
            G += np.roll(flux, 1, axis)

    def _settle_all(self):
        self.D += _up(self.G, self.f) * self._gran(self.paper)[..., None]
        self.G[:] = 0
        self.W[:] = 0
        bx, by, bw, bh = self.box
        self._write(bx, by, bx + bw, by + bh)

    def force_dry(self):
        if self.W is None:
            return None
        self._settle_all()
        r = self.box_rect()
        self.layer.wet = None
        return r

    # ---- what the layer shows
    def _write(self, x0, y0, x1, y1):
        bx, by, bw, bh = self.box
        f = self.f
        x0, y0 = max(x0, bx), max(y0, by)
        x1, y1 = min(x1, bx + bw), min(y1, by + bh)
        x0, y0 = bx + ((x0 - bx) // f) * f, by + ((y0 - by) // f) * f
        x1, y1 = bx + -(-(x1 - bx) // f) * f, by + -(-(y1 - by) // f) * f
        g = _up(self.G[(y0 - by) // f:(y1 - by) // f, (x0 - bx) // f:(x1 - bx) // f], f)
        tot = self.D[y0 - by:y1 - by, x0 - bx:x1 - bx] + g
        a = np.clip(tot.sum(-1) * 60.0, 0, 1)[..., None]
        px = np.concatenate([np.exp(-tot) * a, a], -1).astype(np.float32)   # premultiplied
        lay = self.layer
        for ty in range(y0 // TILE, (y1 - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x1 - 1) // TILE + 1):
                ox, oy = tx * TILE, ty * TILE
                ax0, ay0 = max(x0, ox), max(y0, oy)
                ax1, ay1 = min(x1, ox + TILE), min(y1, oy + TILE)
                src = px[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
                t = lay.tiles.get((tx, ty))
                if t is None and not src[..., 3].any():
                    continue
                if self.undo is not None:
                    target = self.undo._open if self.undo._open is not None else self.undo.current.step
                    if target is not self._noted_step:
                        self._noted, self._noted_step = set(), target
                    if (lay.uid, (tx, ty)) not in self._noted:
                        self.undo.amend(lay, (tx, ty), t)              # its state before this change
                        self._noted.add((lay.uid, (tx, ty)))
                lay.tile((tx, ty), True)[ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox] = src


def density_of(colour, strength: float = 1.0) -> np.ndarray:
    """The optical density a brush colour carries: full strength dries to exactly that colour."""
    c = np.clip(np.asarray(colour[:3], np.float32), 0.02, 1.0)
    return -np.log(c) * strength
