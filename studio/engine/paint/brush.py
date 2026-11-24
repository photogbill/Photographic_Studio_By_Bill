"""The pixel brush engine (BUILD_PLAN_PAINT P1).

A stroke is a row of dabs. Pen samples (already smoothed, smoothing.py) are joined by quadratic curves through the
midpoints (no straight segments between samples), and dabs are placed along them at a spacing that is a fraction of
the dab's size. Each dab is a round tip (hardness sets how soft its edge is) or an image tip, placed with sub-pixel
accuracy. Size, opacity and flow each follow the pen through editable curves (curves.py): pressure, speed, tilt,
random and fade.

Painting modes (each says what it is best for and weak at):
- wash      dabs pile up within one stroke but never past the stroke's opacity (how Krita and Photoshop paint by
            default): an even tone however often you go back over the same place in one stroke
- buildup   every dab adds on top of the last, like a marker: overlaps get darker
- airbrush  build-up, and paint keeps flowing while the pen rests
- smudge    the brush carries paint it picks up from the canvas (as Krita's Color Smudge, our own code): "smear"
            carries the pixels themselves along the stroke, "dull" carries one averaged colour; smudge length sets
            how far the paint is carried, colour rate how much of the brush's own colour is mixed in (an oily,
            loaded brush), and it reads this layer only or every layer (then paint from below is pulled up onto
            this layer)

The eraser is the same engine taking paint away (also the pen's eraser end). Alpha lock keeps the layer's
see-through areas see-through. Strokes write through ``Layer.write``-style tile access and tell the undo stack
before each tile first changes.
"""
from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field

import numpy as np

from .colour import over
from .curves import Curve
from .smoothing import Sample
from .tiles import TILE, Layer

MODES = {
    "wash": ("Wash", "even tones: going over the same place in one stroke never goes past the stroke's opacity",
             "a single stroke cannot be built darker; lift the pen to add a layer of tone"),
    "buildup": ("Build-up", "markers and glazes: overlaps get darker, like real ink",
                "uneven tone where dabs overlap at low spacing"),
    "airbrush": ("Airbrush", "soft shading: paint keeps flowing while the pen rests",
                 "easy to over-darken if you pause"),
    "smudge": ("Smudge", "blending edges, pulling paint along, mixing on the canvas like oils",
               "slower on very large brushes; a long smudge length drags paint far from where it was"),
    "mypaint": ("MyPaint engine", "MyPaint's own brushes (.myb), with all their curves: charcoal, inks, smudgy "
                "brushes and the rest of MyPaint's packs", "a few MyPaint features are not ported yet (spectral paint "
                "mode, smudge buckets, directional offsets), so some brushes look a little different"),
}
SMUDGE_KINDS = {
    "smear": ("Smear", "pulling shapes and detail along the stroke, like a finger in wet paint",
              "can repeat a pattern (the carried pixels) at low spacing"),
    "dull": ("Dull", "blending two colours into a soft, even mix", "loses texture and detail"),
}

INPUTS = ("pressure", "speed", "tilt", "random", "fade")
INPUT_NOTES = {
    "pressure": "how hard the pen presses (every tablet)",
    "speed": "how fast the pen moves (slow = 0, 3000 px/s or faster = 1)",
    "tilt": "how far the pen leans (upright = 0, flat = 1); pens without tilt stay at 0",
    "random": "a new random value for every dab",
    "fade": "distance along the stroke, 0 at the start to 1 at the fade length",
}


from .tiles import LayerLocked  # noqa: E402,F401  (kept here: older code imports it from brush)


@dataclass
class BrushSettings:
    name: str = "Round"
    size: float = 20.0           # diameter in canvas pixels at full input
    min_size: float = 0.0        # fraction of size at zero input (keeps a thin line at the lightest touch)
    hardness: float = 0.8        # 1 = hard edge, 0 = very soft
    opacity: float = 1.0
    flow: float = 1.0
    spacing: float = 0.1         # fraction of the dab's diameter
    mode: str = "wash"
    airbrush_rate: float = 60.0  # dabs per second while resting (airbrush)
    fade_length: float = 1000.0  # pixels, for the "fade" input
    roundness: float = 1.0       # 1 = circle, less = ellipse
    angle: float = 0.0           # degrees, for elliptical and image tips
    scatter: float = 0.0         # fraction of size, random sideways offset
    tip: np.ndarray | None = None   # image tip: 2-D alpha 0..1, None = round
    eraser: bool = False
    space: str = "classic"         # blend space: classic, light or oklab (colour.SPACES)
    smudge_kind: str = "smear"     # smudge mode: smear | dull (SMUDGE_KINDS)
    smudge_length: float = 0.8     # 0..1: how much of the carried paint stays with the brush per dab
    colour_rate: float = 0.0       # 0..1: how much of the brush colour is mixed into the carried paint per dab
    smudge_all: bool = False       # read every layer (True) or this layer only
    myb: object = None             # mode "mypaint": the MyPaint brush (mypaint.MyPaintBrush)
    dynamics: dict = field(default_factory=lambda: {"size": {"pressure": Curve()}, "opacity": {}, "flow": {}})
    best_for: str = ""
    weak_at: str = ""

    def copy(self) -> "BrushSettings":
        d = dict(vars(self))
        d["myb"] = self.myb
        d["dynamics"] = {k: dict(v) for k, v in self.dynamics.items()}
        return BrushSettings(**d)


def presets() -> list[BrushSettings]:
    """The starting brushes. Each says what it is best for and weak at (plan 23.1); the Brush Atlas (plan 0.8) grows
    from here in P2-P3."""
    lin = Curve.preset("Linear")
    soft = Curve.preset("Soft")
    return [
        BrushSettings("Pencil (HB)", size=4, min_size=0.25, hardness=0.9, opacity=0.85, flow=0.9, spacing=0.08,
                      dynamics={"size": {"pressure": soft}, "opacity": {"pressure": lin}, "flow": {}},
                      best_for="sketching and construction lines; pressure changes both width and darkness",
                      weak_at="large areas of tone (no paper grain yet: that comes with natural media, P3)"),
        BrushSettings("Ink pen", size=6, min_size=0.1, hardness=1.0, opacity=1.0, flow=1.0, spacing=0.05,
                      dynamics={"size": {"pressure": lin}, "opacity": {}, "flow": {}},
                      best_for="clean line art that swells and tapers with pressure",
                      weak_at="soft shading; pair it with a smoothing mode for long curves"),
        BrushSettings("Round brush", size=30, min_size=0.15, hardness=0.75, opacity=1.0, flow=0.6, spacing=0.08,
                      dynamics={"size": {"pressure": lin}, "opacity": {}, "flow": {"pressure": lin}},
                      best_for="everyday painting: blocking in colour and forms",
                      weak_at="no bristle texture or colour pick-up (smudge and bristles come in P2-P3)"),
        BrushSettings("Soft airbrush", size=120, hardness=0.0, opacity=1.0, flow=0.08, spacing=0.05,
                      mode="airbrush", dynamics={"size": {}, "opacity": {}, "flow": {"pressure": lin}},
                      best_for="gradual shading, glows and soft transitions",
                      weak_at="crisp edges; paint keeps flowing if you rest the pen"),
        BrushSettings("Marker", size=24, hardness=0.95, opacity=0.5, flow=0.5, spacing=0.06, mode="buildup",
                      dynamics={"size": {}, "opacity": {}, "flow": {}},
                      best_for="flat colour that darkens where strokes overlap, like alcohol markers",
                      weak_at="smooth even fills (overlaps show)"),
        BrushSettings("Flat wash", size=60, hardness=0.6, opacity=0.4, flow=1.0, spacing=0.06, roundness=0.35,
                      angle=30, dynamics={"size": {"pressure": Curve([(0, 0.4), (1, 1)])}, "opacity": {}, "flow": {}},
                      best_for="broad flat strokes and even washes of tone",
                      weak_at="the angle is fixed (follows tilt and direction later)"),
        BrushSettings("Gouache (pigment)", size=34, min_size=0.2, hardness=0.7, opacity=0.9, flow=0.5, spacing=0.06,
                      space="pigment", dynamics={"size": {"pressure": lin}, "opacity": {}, "flow": {"pressure": lin}},
                      best_for="colours that mix like real paint where strokes overlap: blue over yellow goes green",
                      weak_at="bright clean overlays (pigment mixes darken, as real paint does); a little slower"),
        BrushSettings("Smudge (smear)", size=40, hardness=0.4, opacity=1.0, flow=0.9, spacing=0.05, mode="smudge",
                      smudge_kind="smear", smudge_length=0.9, colour_rate=0.0,
                      dynamics={"size": {"pressure": Curve([(0, 0.3), (1, 1)])}, "opacity": {"pressure": lin},
                                "flow": {}},
                      best_for="pulling paint and shapes along, softening hair and fur, wet-into-wet edges",
                      weak_at="flat areas of one colour (there is nothing to pull)"),
        BrushSettings("Blender (dull)", size=50, hardness=0.3, opacity=0.8, flow=0.8, spacing=0.05, mode="smudge",
                      smudge_kind="dull", smudge_length=0.6, colour_rate=0.0,
                      dynamics={"size": {}, "opacity": {"pressure": lin}, "flow": {}},
                      best_for="blending two colours where they meet into a soft even mix",
                      weak_at="keeps no texture or detail"),
        BrushSettings("Oil (loaded brush)", size=36, min_size=0.3, hardness=0.6, opacity=1.0, flow=1.0,
                      spacing=0.05, mode="smudge", smudge_kind="dull", smudge_length=0.5, colour_rate=0.25,
                      dynamics={"size": {"pressure": lin}, "opacity": {}, "flow": {}},
                      best_for="painting that mixes with the wet paint under it, as oils do; the colour runs out "
                               "along the stroke",
                      weak_at="pure clean colour (it always picks up what is under it; lift and paint again)"),
        BrushSettings("Eraser (hard)", size=24, hardness=0.95, eraser=True, spacing=0.06,
                      dynamics={"size": {"pressure": Curve([(0, 0.3), (1, 1)])}, "opacity": {}, "flow": {}},
                      best_for="clean removal and cutting edges back", weak_at="soft fades"),
        BrushSettings("Eraser (soft)", size=90, hardness=0.0, flow=0.4, eraser=True, spacing=0.06,
                      dynamics={"size": {}, "opacity": {}, "flow": {"pressure": Curve()}},
                      best_for="fading paint out gently", weak_at="precise edges"),
    ]


# ------------------------------------------------------------------------------------------------- dab masks
_cache: dict = {}


def dab_mask(diameter: float, hardness: float, fx: float, fy: float, roundness: float = 1.0, angle: float = 0.0,
             tip: np.ndarray | None = None):
    """A dab's alpha (2-D float32) and its top-left offset, for a dab centred at sub-pixel offset (fx, fy) in 0..1.
    Cached per quarter-pixel size and position, so a stroke reuses a handful of masks."""
    d = max(0.5, diameter)
    q = (round(d * 4) / 4, round(hardness, 2), round(fx * 4) / 4, round(fy * 4) / 4, round(roundness, 2),
         round(angle) % 180, id(tip) if tip is not None else 0)
    hit = _cache.get(q)
    if hit is not None:
        return hit
    d, fx, fy = q[0], q[2], q[3]
    r = d / 2
    n = int(math.ceil(r + 1.5)) * 2 + 1
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    dx = xx - (c + fx - 0.5)
    dy = yy - (c + fy - 0.5)
    if tip is not None:
        from PIL import Image
        side = max(1, int(round(d)))
        img = Image.fromarray((np.clip(tip, 0, 1) * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR)
        if angle:
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
        t = np.asarray(img, np.float32) / 255.0
        m = np.zeros((n, n), np.float32)
        o = c - side // 2
        m[o:o + side, o:o + side] = t[:max(0, n - o), :max(0, n - o)]
    else:
        if roundness < 1 or angle:
            a = math.radians(angle)
            u = dx * math.cos(a) + dy * math.sin(a)
            v = (-dx * math.sin(a) + dy * math.cos(a)) / max(roundness, 0.05)
            dist = np.sqrt(u * u + v * v)
        else:
            dist = np.sqrt(dx * dx + dy * dy)
        h = min(max(hardness, 0.0), 1.0)
        rr = dist / max(r, 1e-6)
        if h >= 1:
            m = np.ones_like(rr)
        else:
            t = np.clip((rr - h) / (1 - h), 0, 1)
            m = 1 - t * t * (3 - 2 * t)                    # smoothstep falloff from the hard core to the rim
        m = m * np.clip(r + 0.5 - dist, 0, 1)              # anti-aliased rim (thin brushes stay smooth)
    m = m.astype(np.float32)
    res = (m, -c)
    if len(_cache) > 4000:
        _cache.clear()
    _cache[q] = res
    return res


# ------------------------------------------------------------------------------------------------- stroke
class Stroke:
    """One stroke on one layer. ``add(sample)`` paints and returns the changed rectangle (x0, y0, x1, y1) or None;
    ``finish()`` ends the stroke."""

    def __init__(self, layer: Layer, brush: BrushSettings, colour=(0, 0, 0), undo=None, eraser: bool | None = None,
                 seed: int | None = None, sample=None):
        if layer.is_group:
            raise LayerLocked(f"\"{layer.name}\" is a group. Pick a layer inside it to paint.")
        if layer.locked:
            raise LayerLocked(f"The layer \"{layer.name}\" is locked. Unlock it in the Layers panel to paint on it.")
        self.layer, self.b = layer, brush
        self.colour = np.asarray(colour[:3], np.float32)
        self.undo = undo
        self.eraser = brush.eraser if eraser is None else eraser
        self.rng = _random.Random(seed)
        self.pts: list[Sample] = []
        self.left = 0.0                  # distance carried to the next dab
        self.travel = 0.0                # stroke length so far (fade input)
        self.orig: dict = {}             # tile key -> content before the stroke (None = no tile)
        self.mask: dict = {}             # wash: tile key -> accumulated stroke alpha
        self.last_t = None
        self.dabs = 0
        self._dirty = None
        # smudge: where the paint is read (``sample(x0, y0, w, h)`` -> premultiplied RGBA; default: this layer) and
        # what the brush carries (one colour for "dull", a patch of pixels for "smear")
        self.sample = sample if (sample is not None and brush.smudge_all) else layer.read
        self.carried = None

    # ---- inputs
    def _inputs(self, s: Sample) -> dict:
        tilt = min(1.0, math.hypot(s.tilt_x, s.tilt_y) / 60.0)
        return {"pressure": s.pressure, "speed": min(1.0, s.speed / 3000.0), "tilt": tilt,
                "random": self.rng.random(), "fade": min(1.0, self.travel / max(1.0, self.b.fade_length))}

    def _value(self, target: str, base: float, inp: dict) -> float:
        v = base
        for name, curve in self.b.dynamics.get(target, {}).items():
            v *= curve(inp.get(name, 1.0))
        return v

    def dab_size(self, s: Sample) -> float:
        inp = self._inputs(s)
        f = self._value("size", 1.0, inp)
        return self.b.size * (self.b.min_size + (1 - self.b.min_size) * f)

    # ---- placing dabs
    def add(self, s: Sample):
        self._dirty = None
        pts = self.pts
        pts.append(s)
        if len(pts) == 1:
            self._dab(s)
        elif len(pts) == 2:
            pass                                             # wait for the third point to curve the first piece
        else:
            a, b, c = pts[-3], pts[-2], pts[-1]
            start = a if len(pts) == 3 else _mid(a, b)
            self._curve(start, b, _mid(b, c))
        if self.b.mode == "airbrush":
            self._rest(s)
        return self._dirty

    def finish(self):
        self._dirty = None
        pts = self.pts
        if len(pts) == 2:
            self._line(pts[0], pts[1])
        elif len(pts) > 2:
            self._line(_mid(pts[-2], pts[-1]), pts[-1])
        if self.b.mode == "wash":
            self.mask.clear()
        self.layer.prune()
        return self._dirty

    def _rest(self, s: Sample):
        if self.last_t is None:
            self.last_t = s.t
            return
        n = int((s.t - self.last_t) * self.b.airbrush_rate)
        if n > 0:
            self.last_t += n / self.b.airbrush_rate
            for _ in range(min(n, 30)):
                self._dab(s)

    def _line(self, p: Sample, q: Sample):
        self._curve(p, _mid(p, q), q)

    def _curve(self, p0: Sample, p1: Sample, p2: Sample):
        """Dabs along the quadratic Bezier p0 - p1 - p2 at the brush's spacing."""
        length = (math.hypot(p1.x - p0.x, p1.y - p0.y) + math.hypot(p2.x - p1.x, p2.y - p1.y))
        if length < 1e-6:
            return
        steps = max(2, int(length * 2))
        prev = p0
        acc = self.left                  # distance since the last dab
        for i in range(1, steps + 1):
            t = i / steps
            cur = _bez(p0, p1, p2, t)
            seg = math.hypot(cur.x - prev.x, cur.y - prev.y)
            self.travel += seg
            acc += seg
            while True:
                gap = max(0.5, self.b.spacing * max(1.0, self.dab_size(cur)))
                if acc < gap:
                    break
                acc -= gap
                f = 1 - acc / seg if seg > 0 else 1
                self._dab(_lerp(prev, cur, f))
            prev = cur
        self.left = acc

    # ---- one dab
    def _dab(self, s: Sample):
        inp = self._inputs(s)
        size = self.b.size * (self.b.min_size + (1 - self.b.min_size) * self._value("size", 1.0, inp))
        if size < 0.3:
            return
        opacity = min(1.0, self._value("opacity", self.b.opacity, inp))
        flow = min(1.0, self._value("flow", self.b.flow, inp))
        x, y = s.x, s.y
        if self.b.scatter:
            x += (self.rng.random() - 0.5) * 2 * self.b.scatter * size
            y += (self.rng.random() - 0.5) * 2 * self.b.scatter * size
        ix, iy = math.floor(x), math.floor(y)
        m, off = dab_mask(size, self.b.hardness, x - ix, y - iy, self.b.roundness, self.b.angle, self.b.tip)
        x0, y0 = ix + off, iy + off
        self.dabs += 1
        if self.b.mode == "smudge" and not self.eraser:
            self._smudge(m, x0, y0, opacity * flow)
        else:
            self._stamp(m, x0, y0, opacity, flow)

    def _smudge(self, m: np.ndarray, x0: int, y0: int, strength: float):
        """One smudge dab: lay the carried paint down through the mask, then pick up what is here."""
        h, w = m.shape
        here = self.sample(x0, y0, w, h)                     # read before laying anything down
        b = self.b
        brush_c = np.concatenate([self.colour, [1.0]]).astype(np.float32)
        if b.smudge_kind == "dull":
            mw = m[..., None]
            tot = float(m.sum())
            pick = (here * mw).sum(axis=(0, 1)) / tot if tot > 0 else np.zeros(4, np.float32)
            if self.carried is None or self.carried.shape != (4,):
                self.carried = pick
            carried = self.carried
        else:
            if self.carried is None or self.carried.shape != here.shape:
                self.carried = here.copy()
            carried = self.carried
        if b.colour_rate > 0:
            carried = carried * (1 - b.colour_rate) + brush_c * b.colour_rate
        self._stamp_paint(m, x0, y0, strength, carried)
        # what stays on the brush: the carried paint mixed with what was just picked up
        k = min(max(b.smudge_length, 0.0), 1.0)
        if b.smudge_kind == "dull":
            self.carried = carried * k + pick * (1 - k)
        else:
            self.carried = carried * k + here * (1 - k)

    def _stamp_paint(self, m, x0, y0, strength, paint):
        """Mix ``paint`` (a premultiplied colour, or a patch the size of the mask) into the layer through the
        mask: out = lerp(layer, paint, mask * strength)."""
        h, w = m.shape
        lay = self.layer
        for ty in range(y0 // TILE, (y0 + h - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x0 + w - 1) // TILE + 1):
                key = (tx, ty)
                ox, oy = tx * TILE, ty * TILE
                ax0, ay0 = max(x0, ox), max(y0, oy)
                ax1, ay1 = min(x0 + w, ox + TILE), min(y0 + h, oy + TILE)
                mm = m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
                if not mm.any():
                    continue
                p = paint if paint.ndim == 1 else paint[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
                tile = lay.tiles.get(key)
                if tile is None and (lay.alpha_lock or (paint.ndim == 1 and paint[3] <= 0)
                                     or (paint.ndim == 3 and not p[..., 3].any())):
                    continue                                 # nothing to lay down here
                if key not in self.orig:
                    if self.undo is not None:
                        self.undo.touch(lay, key)
                    self.orig[key] = None if tile is None else tile.copy()
                if tile is None:
                    tile = lay.tile(key, True)
                sl = (slice(ay0 - oy, ay1 - oy), slice(ax0 - ox, ax1 - ox))
                a = (mm * strength)[..., None]
                base = tile[sl]
                out = base * (1 - a) + p * a
                if lay.alpha_lock:
                    ba = base[..., 3:4]
                    straight = out[..., :3] / np.maximum(out[..., 3:4], 1e-8)
                    out = np.concatenate([straight * ba, ba], axis=-1)
                tile[sl] = out.astype(np.float32)
                self._grow(ax0, ay0, ax1, ay1)

    def _stamp(self, m: np.ndarray, x0: int, y0: int, opacity: float, flow: float):
        h, w = m.shape
        lay = self.layer
        mode = "buildup" if self.b.mode == "airbrush" else self.b.mode
        for ty in range(y0 // TILE, (y0 + h - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x0 + w - 1) // TILE + 1):
                key = (tx, ty)
                ox, oy = tx * TILE, ty * TILE
                ax0, ay0 = max(x0, ox), max(y0, oy)
                ax1, ay1 = min(x0 + w, ox + TILE), min(y0 + h, oy + TILE)
                mm = m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
                if not mm.any():
                    continue
                tile = lay.tiles.get(key)
                if tile is None and (self.eraser or lay.alpha_lock):
                    continue                                 # nothing there to erase or recolour
                if key not in self.orig:
                    if self.undo is not None:
                        self.undo.touch(lay, key)
                    self.orig[key] = None if tile is None else tile.copy()
                if tile is None:
                    tile = lay.tile(key, True)
                sl = (slice(ay0 - oy, ay1 - oy), slice(ax0 - ox, ax1 - ox))
                a = (mm * flow)[..., None]
                if mode == "wash":
                    S = self.mask.get(key)
                    if S is None:
                        S = self.mask[key] = np.zeros((TILE, TILE, 1), np.float32)
                    s = S[sl]
                    s += a * np.maximum(opacity - s, 0)
                    base = self.orig[key]
                    base = np.zeros((ay1 - ay0, ax1 - ax0, 4), np.float32) if base is None else base[sl]
                    nz = mm > 0                              # only where this dab lands: the rest is unchanged
                    region = tile[sl]
                    region[nz] = self._apply(base[nz], s[nz])
                else:
                    nz = mm > 0
                    region = tile[sl]
                    region[nz] = self._apply(region[nz], (a * opacity)[nz])
                self._grow(ax0, ay0, ax1, ay1)

    def _apply(self, base: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self.eraser:
            return base * (1 - a)
        out = over(base, self.colour, a, self.b.space)
        if self.layer.alpha_lock:
            ba = base[..., 3:4]
            straight = out[..., :3] / np.maximum(out[..., 3:4], 1e-8)
            out = np.concatenate([straight * ba, ba], axis=-1)
        return out

    def _grow(self, x0, y0, x1, y1):
        d = self._dirty
        self._dirty = (x0, y0, x1, y1) if d is None else (min(d[0], x0), min(d[1], y0), max(d[2], x1),
                                                          max(d[3], y1))


def _mid(a: Sample, b: Sample) -> Sample:
    return _lerp(a, b, 0.5)


def _lerp(a: Sample, b: Sample, t: float) -> Sample:
    return Sample(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.pressure + (b.pressure - a.pressure) * t,
                  a.t + (b.t - a.t) * t, a.tilt_x + (b.tilt_x - a.tilt_x) * t, a.tilt_y + (b.tilt_y - a.tilt_y) * t,
                  a.rotation + (b.rotation - a.rotation) * t, a.speed + (b.speed - a.speed) * t)


def _bez(p0: Sample, p1: Sample, p2: Sample, t: float) -> Sample:
    return _lerp(_lerp(p0, p1, t), _lerp(p1, p2, t), t)


def make_stroke(layer: Layer, brush: BrushSettings, colour=(0, 0, 0), undo=None, eraser=None, seed=None,
                sample=None):
    """The right stroke for the brush: the pixel engine, or the MyPaint engine for MyPaint brushes. The Size,
    Opacity and Hardness sliders drive a MyPaint brush's own radius, opaque and hardness."""
    if brush.mode == "mypaint" and brush.myb is not None:
        from .mypaint import MyPaintStroke
        st = MyPaintStroke(layer, brush.myb, colour, undo, eraser=eraser or brush.eraser, seed=seed, sample=sample)
        st.base["radius_logarithmic"] = math.log(max(0.4, brush.size / 2))
        st.base["opaque"] = brush.opacity
        st.base["hardness"] = brush.hardness
        return st
    return Stroke(layer, brush, colour, undo, eraser, seed=seed, sample=sample)


def from_myb(myb) -> BrushSettings:
    """A MyPaint brush as an entry in the brush list."""
    b = myb.base
    return BrushSettings(myb.name, size=float(2 * math.exp(b["radius_logarithmic"])), hardness=min(1, b["hardness"]),
                         opacity=min(1.0, b["opaque"]), mode="mypaint", myb=myb, eraser=b["eraser"] >= 1.0,
                         best_for=(myb.comment.strip() or "a MyPaint brush") + " (MyPaint engine)",
                         weak_at=MODES["mypaint"][2])


def paint_stroke(layer: Layer, brush: BrushSettings, samples, colour=(0, 0, 0), undo=None, eraser=None,
                 sample=None) -> Stroke:
    """Paint a whole stroke from a list of Samples (used by tests, the benchmark and scripts)."""
    if undo is not None:
        undo.begin("Erase" if (brush.eraser if eraser is None else eraser) else "Stroke")
    st = make_stroke(layer, brush, colour, undo, eraser, sample=sample)
    for s in samples:
        st.add(s)
    st.finish()
    if undo is not None:
        undo.end()
    return st


def load_tip(path: str, invert: bool | None = None) -> np.ndarray:
    """An image tip from any picture (plan 0.8: "any object as a brush"): greys become how much paint each pixel
    puts down. Dark marks on a light ground (a scan of ink, a photo of a leaf) are turned round so the dark parts
    paint; a picture with see-through areas uses its alpha. Cropped to the mark and made square."""
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        t = np.asarray(im.convert("RGBA"), np.float32)[..., 3] / 255.0
    else:
        g = np.asarray(im.convert("L"), np.float32) / 255.0
        if invert is None:
            invert = float(np.median(g)) > 0.5                  # light ground: the marks are dark
        t = 1 - g if invert else g
    ys, xs = np.nonzero(t > 0.02)
    if len(xs):
        t = t[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = t.shape
    n = max(h, w)
    sq = np.zeros((n, n), np.float32)
    sq[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = t
    if n > 512:                                                 # tips are resampled per size anyway
        sq = np.asarray(Image.fromarray((sq * 255).astype(np.uint8)).resize((512, 512), Image.LANCZOS),
                        np.float32) / 255.0
    return sq
