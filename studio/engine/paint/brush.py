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
from . import texture as _tex

MODES = {
    "wash": ("Wash", "even tones: going over the same place in one stroke never goes past the stroke's opacity",
             "a single stroke cannot be built darker; lift the pen to add a layer of tone"),
    "buildup": ("Build-up", "markers and glazes: overlaps get darker, like real ink",
                "uneven tone where dabs overlap at low spacing"),
    "airbrush": ("Airbrush", "soft shading: paint keeps flowing while the pen rests",
                 "easy to over-darken if you pause"),
    "smudge": ("Smudge", "blending edges, pulling paint along, mixing on the canvas like oils",
               "slower on very large brushes; a long smudge length drags paint far from where it was"),
    "spray": ("Spray", "speckle, grit, stars, foliage texture: many small particles inside the brush",
              "clean shapes; it is random by design"),
    "chalk": ("Chalk", "chalk and crayon: each dab drops random pixels, fewer at a light touch",
              "smooth tone; pair it with paper grain for real chalk"),
    "sketch": ("Sketch", "loose sketching: thin lines join the pen to earlier points of the stroke (web lines)",
               "precise line art; dense scribbles fill in quickly"),
    "hatching": ("Hatching", "pen-and-ink shading: parallel lines fixed to the paper, thicker with pressure, "
                 "cross-hatched when you press hard", "soft gradients; the lines are always straight"),
    "shape": ("Shape (filled lasso)", "flat colour shapes: draw the outline and it is filled when you lift the pen",
              "nothing shows until the pen lifts; no pressure along the edge"),
    "clone": ("Clone", "copying one part of the picture onto another (Ctrl+click sets where to copy from)",
              "it copies, it does not blend: obvious repeats if overused"),
    "normal": ("Tangent normal", "painting normal maps for games and 3D: the colour follows the pen's tilt (or the "
               "stroke's direction with a mouse)", "only makes sense on a layer meant as a normal map"),
    "watercolour": ("Watercolour", "transparent washes that flow and dry on the paper: dark edges, blooms where "
                    "water meets damp paint, granulation, lifting with clean water", "opaque cover and quick "
                    "corrections; it needs its own layer (a new layer becomes a watercolour layer)"),
    "bristle": ("Bristles", "oil, acrylic and gouache strokes: each bristle carries its own paint, picks up colour "
                "from the canvas and runs dry, so strokes streak and break up as real ones do",
                "smooth flat colour; slower than a plain brush"),
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
    # texture (P2): paper grain, a second tip, more dabs per place, colour jitter
    grain: str = ""                # texture.GRAINS key, "picture" (grain_tile), or "" for none
    grain_strength: float = 0.0    # 0 = no grain, 1 = full tooth
    grain_scale: float = 1.0       # size of the grain on the canvas
    grain_pressure: bool = True    # more pressure pushes the medium into the paper's valleys
    grain_tile: np.ndarray | None = None
    dual: bool = False             # a second tip breaks up the first (as Photoshop's Dual Brush)
    dual_size: float = 0.25        # the second tip's size as a fraction of the dab
    dual_count: int = 4            # second-tip marks per dab
    dual_scatter: float = 0.5      # how far they spread inside the dab (fraction of its size)
    count: int = 1                 # dabs placed at each spot (with scatter: a spray of dabs)
    jitter_h: float = 0.0          # colour jitter: hue, saturation and value, 0..1
    jitter_s: float = 0.0
    jitter_v: float = 0.0
    jitter_per_stroke: bool = False   # one jittered colour for the whole stroke instead of every dab
    # engines (P2)
    particle_size: float = 0.06    # spray: one particle as a share of the brush size
    hatch_angle: float = 45.0      # hatching: line angle in degrees
    hatch_spacing: float = 6.0     # hatching: distance between lines in canvas pixels
    crosshatch: float = 0.75       # hatching: pressure above which a second set of lines crosses (1 = never)
    sketch_reach: float = 3.0      # sketch: how far back lines reach, as a multiple of the size
    sketch_density: float = 0.35   # sketch: share of earlier points joined
    normal_tilt: float = 45.0      # tangent normal: the tilt used when the pen reports none (degrees)
    tips: list = field(default_factory=list)   # picture tips picked at random per dab (GIMP image hoses)
    # natural media (P3)
    water: float = 0.6             # watercolour: how much water each dab puts down
    pigment: float = 0.5           # watercolour: how strong the colour is (1 = dries to the colour itself)
    lift: float = 0.0              # watercolour: clean water lifts settled pigment (0..1)
    granulation: float = 0.5       # watercolour: pigment settling in the paper's valleys
    edge_darkening: float = 0.6    # watercolour: pigment carried to the edges as they dry
    impasto: float = 0.0           # paint thickness each dab adds (the relief); 0 = flat
    bristles: int = 24             # bristle engine: how many
    paint_load: float = 1.0        # bristle engine: how long the paint lasts (larger = longer strokes)
    pickup: float = 0.3            # bristle engine: how much colour the bristles pick up from the canvas
    tags: str = ""                 # the brush list's category
    dynamics: dict = field(default_factory=lambda: {"size": {"pressure": Curve()}, "opacity": {}, "flow": {}})
    best_for: str = ""
    weak_at: str = ""

    def copy(self) -> "BrushSettings":
        d = dict(vars(self))
        d["myb"] = self.myb
        d["grain_tile"] = self.grain_tile
        d["tips"] = list(self.tips)
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
        BrushSettings("Charcoal (rough paper)", size=22, min_size=0.4, hardness=0.7, opacity=0.9, flow=0.8,
                      spacing=0.06, roundness=0.7, angle=20, grain="rough", grain_strength=0.85,
                      dynamics={"size": {"pressure": soft}, "opacity": {"pressure": lin}, "flow": {}},
                      best_for="charcoal drawing: a light touch catches only the paper's ridges, pressing harder "
                               "fills the valleys", weak_at="smooth even tone (that is what the grain is for)"),
        BrushSettings("Soft pastel", size=40, min_size=0.5, hardness=0.5, opacity=0.85, flow=0.7, spacing=0.06,
                      grain="cold_press", grain_strength=0.7, dual=True, dual_size=0.35, dual_count=5,
                      dual_scatter=0.6, jitter_v=0.04,
                      dynamics={"size": {"pressure": lin}, "opacity": {"pressure": lin}, "flow": {}},
                      best_for="broad soft colour with broken edges and paper showing through",
                      weak_at="crisp lines and fine detail"),
        BrushSettings("Dry brush (canvas)", size=50, min_size=0.3, hardness=0.8, opacity=1.0, flow=0.9,
                      spacing=0.05, roundness=0.4, angle=-15, grain="canvas", grain_strength=0.9,
                      grain_pressure=True, dynamics={"size": {"pressure": lin}, "opacity": {}, "flow": {}},
                      best_for="dry, scumbled strokes over a canvas weave, as oils dragged with little paint",
                      weak_at="wet, smooth blending; the weave can repeat on huge strokes"),
        BrushSettings("Spray", size=90, hardness=0.9, opacity=1.0, flow=1.0, spacing=0.25, mode="buildup",
                      count=6, scatter=1.0, dual=True, dual_size=0.06, dual_count=6, dual_scatter=1.0,
                      dynamics={"size": {}, "opacity": {}, "flow": {"pressure": lin}},
                      best_for="spray-can speckle, grit, stars and foliage texture",
                      weak_at="clean shapes; it is random by design"),
        BrushSettings("Leaves (colour jitter)", size=36, min_size=0.5, hardness=0.85, opacity=1.0, flow=1.0,
                      spacing=0.7, roundness=0.45, scatter=0.8, jitter_h=0.08, jitter_s=0.15, jitter_v=0.2,
                      dynamics={"size": {"pressure": lin, "random": Curve([(0, 0.6), (1, 1)])}, "opacity": {},
                                "flow": {}},
                      best_for="foliage, flowers and crowds of small marks that each differ a little in colour",
                      weak_at="flat colour (turn the jitter down)"),
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
                 seed: int | None = None, sample=None, clone=None, clip=None):
        if layer.is_group:
            raise LayerLocked(f"\"{layer.name}\" is a group. Pick a layer inside it to paint.")
        if layer.locked:
            raise LayerLocked(f"The layer \"{layer.name}\" is locked. Unlock it in the Layers panel to paint on it.")
        self.layer, self.b = layer, brush
        wet_brush = brush.mode == "watercolour" and not (brush.eraser if eraser is None else eraser)
        if wet_brush and layer.medium != "watercolour":
            if layer.tiles:
                raise LayerLocked("Watercolour needs a layer of its own. Add a new layer: it becomes a watercolour "
                                  "layer with the first watercolour stroke.")
            layer.medium, layer.blend = "watercolour", "multiply"
        elif layer.medium == "watercolour" and not wet_brush and not (brush.eraser if eraser is None else eraser):
            raise LayerLocked(f"\"{layer.name}\" is a watercolour layer. Use a watercolour brush here, or paint on "
                              "another layer.")
        if layer.medium == "watercolour" and not wet_brush and layer.wet is not None:
            layer.wet.force_dry()                            # erasing works on dry paint
        self.colour = np.asarray(colour[:3], np.float32)
        self.base_colour = self.colour.copy()
        self._jit = None
        self.np_rng = np.random.default_rng(seed)
        self.clip = clip                     # a selection (canvas-sized soft mask): paint only inside it
        self.hist: list = []                 # sketch: earlier points of the stroke
        self.last_xy = None                  # tangent normal and bristles: the last dab, for the direction
        self.bristle = None                  # bristle engine: per-bristle offsets, colours and paint left
        self.carried_h = None                # smudge: carried paint thickness
        self.orig_h: dict = {}               # impasto: thickness at the stroke's start, per tile
        self._wet_started = False            # watercolour: this stroke has begun in the simulation
        self.clone = clone                   # clone: (dx, dy) from where the pen is to where paint is copied from
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
        # clone copies from this layer, or from the whole picture when "read every layer" is ticked
        self.clone_sample = sample if (sample is not None and brush.smudge_all) else layer.read
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
        if self.b.mode == "shape":
            return None                                      # filled when the pen lifts
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
        if self.b.mode == "shape":
            self._fill_shape()
            self.layer.prune()
            return self._dirty
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
        b = self.b
        if b.jitter_h or b.jitter_s or b.jitter_v:
            if not b.jitter_per_stroke or self._jit is None:
                self._jit = self._jittered()
            self.colour = self._jit
        if b.mode == "normal":
            self.colour = self._normal_colour(s)
        if b.mode == "spray":
            self._spray(s, size, opacity, flow)
            return
        if b.mode == "bristle" and not self.eraser:
            self._bristle_dab(s, size, opacity, flow, inp["pressure"])
            return
        for _ in range(max(1, int(b.count))):
            x, y = s.x, s.y
            if b.scatter:
                x += (self.rng.random() - 0.5) * 2 * b.scatter * size
                y += (self.rng.random() - 0.5) * 2 * b.scatter * size
            ix, iy = math.floor(x), math.floor(y)
            tip = self.rng.choice(b.tips) if b.tips else b.tip
            m, off = dab_mask(size, b.hardness, x - ix, y - iy, b.roundness, b.angle, tip)
            x0, y0 = ix + off, iy + off
            if b.mode == "chalk":
                keep = 0.2 + 0.65 * inp["pressure"]
                m = m * (self.np_rng.random(m.shape) < keep)
            elif b.mode == "hatching":
                m = m * self._hatch(x0, y0, m.shape, inp["pressure"])
            if b.dual:
                m = m * self._dual_mask(m.shape[0], size)
            if b.grain and b.grain_strength > 0:
                tile = b.grain_tile if b.grain == "picture" and b.grain_tile is not None else (
                    _tex.grain(b.grain) if b.grain in _tex.GRAINS else None)
                if tile is not None:
                    hgt = _tex.patch(tile, x0, y0, m.shape[1], m.shape[0], b.grain_scale)
                    m = m * _tex.tooth(hgt, inp["pressure"], b.grain_strength, b.grain_pressure)
            self.dabs += 1
            if b.mode == "smudge" and not self.eraser:
                self._smudge(m, x0, y0, opacity * flow)
            elif b.mode == "watercolour" and not self.eraser:
                self._wet(m, x0, y0, opacity * flow)
            elif b.mode == "clone" and not self.eraser:
                if self.clone is None:
                    raise LayerLocked("Ctrl+click where to copy from first, then paint where it should go.")
                dx, dy = self.clone
                patch = self.clone_sample(x0 + int(round(dx)), y0 + int(round(dy)), m.shape[1], m.shape[0])
                self._stamp_paint(m, x0, y0, opacity * flow, patch)
            else:
                self._stamp(m, x0, y0, opacity, flow)
            if self.eraser and self.layer.height:
                self._stamp_height(m, x0, y0, opacity * flow, mode="erase")
            elif b.impasto > 0 and b.mode not in ("smudge", "watercolour", "clone"):
                hm, _ = dab_mask(size, min(b.hardness, 0.25), x - ix, y - iy, b.roundness, b.angle, tip)
                if hm.shape == m.shape:                       # a rounded ridge, as thick paint stands
                    hm = hm * (m > 0)
                else:
                    hm = m
                self._stamp_height(hm, x0, y0, b.impasto * opacity * flow)
            if b.mode == "sketch":
                self._sketch_lines(s.x, s.y, size, opacity * flow)

    # ---- engines
    def _wet(self, m, x0, y0, strength):
        from .watercolour import WetSim, density_of
        lay, b = self.layer, self.b
        if lay.wet is None:
            lay.wet = WetSim(lay, b.grain if b.grain else "cold_press", b.granulation, b.edge_darkening)
        sim = lay.wet
        if not self._wet_started:
            sim.new_stroke()
            self._wet_started = True
        sim.undo = self.undo
        sim.granulation, sim.edge = b.granulation, b.edge_darkening
        dens = density_of(self.colour, b.pigment * strength) if b.pigment > 0 else np.zeros(3, np.float32)
        sim.add(m, x0, y0, b.water * max(0.2, strength), dens, b.lift)
        h, w = m.shape
        self._grow(x0, y0, x0 + w, y0 + h)

    def _bristle_dab(self, s, size, opacity, flow, pressure):
        """Each bristle lays its own small dab of its own paint, picks up colour from the canvas, and runs dry."""
        b = self.b
        if self.bristle is None:
            n = max(3, int(b.bristles))
            r = np.sqrt(self.np_rng.random(n))
            a = self.np_rng.random(n) * 2 * np.pi
            self.bristle = {"off": np.stack([r * np.cos(a), r * np.sin(a) * max(0.05, b.roundness)], -1),
                            "thick": 0.7 + 0.6 * self.np_rng.random(n),
                            "col": np.tile(self.colour, (n, 1)).astype(np.float32),
                            "load": np.full(n, 1.0, np.float32)}
        br = self.bristle
        if self.last_xy is not None and (s.x, s.y) != self.last_xy:
            self._dir = math.atan2(s.y - self.last_xy[1], s.x - self.last_xy[0])
        self.last_xy = (s.x, s.y)
        ang = getattr(self, "_dir", 0.0) + math.pi / 2 + math.radians(b.angle)   # bristles lie across the stroke
        ca, sa = math.cos(ang), math.sin(ang)
        spread = size / 2 * (0.45 + 0.55 * pressure)                             # bristles splay as you press
        d = max(1.0, size * 0.11)
        use = max(0.0003, 0.004 / max(0.05, b.paint_load))
        R = int(math.ceil(spread + d + 3))
        cx, cy = math.floor(s.x), math.floor(s.y)
        X0, Y0, N = cx - R, cy - R, 2 * R + 1
        here = self.sample(X0, Y0, N, N) if b.pickup > 0 else None
        A = np.zeros((N, N), np.float32)                     # coverage of all bristles in this dab
        C = np.zeros((N, N, 3), np.float32)                  # their colour, weighted by coverage
        H = np.zeros((N, N), np.float32)                     # their thickness (furrows)
        Wt = np.zeros((N, N), np.float32)                    # the sum of the weights, for the mean colour
        any_paint = False
        offs = br["off"]
        xs = s.x + (offs[:, 0] * ca - offs[:, 1] * sa) * spread
        ys = s.y + (offs[:, 0] * sa + offs[:, 1] * ca) * spread
        live = br["load"] > 0.02
        if here is not None and live.any():                  # every bristle picks up at once (one mix)
            ix_ = np.clip(np.floor(xs).astype(int) - X0, 0, N - 1)
            iy_ = np.clip(np.floor(ys).astype(int) - Y0, 0, N - 1)
            px = here[iy_, ix_]
            take = live & (px[:, 3] > 0.05)
            if take.any():
                got = px[take, :3] / px[take, 3:4]
                k = (b.pickup * px[take, 3] * 0.25)[:, None]
                if b.space == "pigment":
                    from .colour import pigment_mix
                    br["col"][take] = pigment_mix(br["col"][take], got.astype(np.float32), k)
                else:
                    br["col"][take] = br["col"][take] * (1 - k) + got * k
        for i in np.nonzero(live)[0]:
            load = float(br["load"][i])
            x, y = float(xs[i]), float(ys[i])
            ix, iy = math.floor(x), math.floor(y)
            m, off = dab_mask(d * br["thick"][i], 0.6, x - ix, y - iy)
            lx, ly = ix + off - X0, iy + off - Y0
            h_, w_ = m.shape
            if lx < 0 or ly < 0 or lx + w_ > N or ly + h_ > N:
                continue
            strength = opacity * flow * min(1.0, load ** 0.5)
            a = m * strength
            sub = A[ly:ly + h_, lx:lx + w_]
            C[ly:ly + h_, lx:lx + w_] += a[..., None] * br["col"][i]
            Wt[ly:ly + h_, lx:lx + w_] += a
            A[ly:ly + h_, lx:lx + w_] = sub + a * (1 - sub)
            if b.impasto > 0:
                hs = H[ly:ly + h_, lx:lx + w_]
                np.maximum(hs, m * b.impasto * strength, out=hs)
            br["load"][i] = load - use * (0.5 + strength)
            self.dabs += 1
            any_paint = True
        if not any_paint:
            return
        colour = np.clip(C / np.maximum(Wt, 1e-6)[..., None], 0, 1)
        paint = np.concatenate([colour, np.ones((N, N, 1), np.float32)], -1)
        self._stamp_paint(A, X0, Y0, 1.0, paint)
        if b.impasto > 0 and H.any():
            self._stamp_height(H, X0, Y0, 1.0)

    def _clipped(self, mm, x0, y0):
        """The dab limited to the selection (outside the canvas counts as unselected)."""
        c = self.clip
        if c is None:
            return mm
        h, w = mm.shape
        H, W = c.shape
        out = np.zeros_like(mm)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = mm[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] * \
                c[sy0:sy1, sx0:sx1]
        return out

    def _stamp_height(self, m, x0, y0, amount, mode="add", patch=None):
        """Paint thickness (impasto): add, or (smudge) mix towards a carried patch of thickness."""
        h, w = m.shape
        lay = self.layer
        for ty in range(y0 // TILE, (y0 + h - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x0 + w - 1) // TILE + 1):
                ox, oy = tx * TILE, ty * TILE
                ax0, ay0 = max(x0, ox), max(y0, oy)
                ax1, ay1 = min(x0 + w, ox + TILE), min(y0 + h, oy + TILE)
                mm = self._clipped(m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0], ax0, ay0)
                if not mm.any():
                    continue
                t = lay.height.get((tx, ty))
                if t is None and mode != "add":
                    if patch is None or not patch.any():
                        continue
                if self.undo is not None:
                    self.undo.touch(lay, ("h", tx, ty))
                if t is None:
                    t = lay.height[(tx, ty)] = np.zeros((TILE, TILE), np.float32)
                sl = (slice(ay0 - oy, ay1 - oy), slice(ax0 - ox, ax1 - ox))
                if mode == "add":                            # like a wash: one stroke raises it once
                    base = self.orig_h.get((tx, ty))
                    if base is None:
                        base = self.orig_h[(tx, ty)] = t.copy()
                    t[sl] = np.minimum(np.maximum(t[sl], base[sl] + mm * amount), 4.0)
                elif mode == "erase":
                    t[sl] *= 1 - mm * amount
                else:
                    p = patch[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
                    a = mm * amount
                    t[sl] = t[sl] * (1 - a) + p * a

    def _spray(self, s, size, opacity, flow):
        b = self.b
        r = size / 2
        n = max(1, int(b.count) * 4)
        for _ in range(n):
            while True:                                         # a point in the disc, denser in the middle
                u, v = self.rng.gauss(0, 0.45), self.rng.gauss(0, 0.45)
                if u * u + v * v <= 1:
                    break
            x, y = s.x + u * r, s.y + v * r
            d = max(0.8, size * b.particle_size * (0.5 + self.rng.random()))
            ix, iy = math.floor(x), math.floor(y)
            m, off = dab_mask(d, b.hardness, x - ix, y - iy)
            self.dabs += 1
            self._stamp(m, ix + off, iy + off, opacity, flow)

    def _hatch(self, x0, y0, shape, pressure):
        """Parallel lines fixed to the canvas; thicker with pressure, a second set across above ``crosshatch``."""
        b = self.b
        h, w = shape
        yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float32)
        sp = max(1.5, b.hatch_spacing)
        width = 0.12 + 0.5 * pressure                           # share of the spacing that is ink

        def lines(angle):
            a = math.radians(angle)
            d = (xx * math.sin(a) - yy * math.cos(a)) / sp
            f = np.abs(d - np.round(d)) * sp                    # distance to the nearest line, in pixels
            return np.clip(width * sp / 2 + 0.5 - f, 0, 1)
        out = lines(b.hatch_angle)
        if pressure > b.crosshatch:
            out = np.maximum(out, lines(b.hatch_angle + 90))
        return out

    def _sketch_lines(self, x, y, size, strength):
        b = self.b
        reach = max(4.0, size * b.sketch_reach)
        hist = self.hist
        made = 0
        pool = hist[:-3]
        if len(pool) > 40:
            pool = self.rng.sample(pool, 40)              # earlier points from anywhere in the stroke
        for (px, py) in pool:
            d = math.hypot(px - x, py - y)
            if d < max(2.0, size * 1.5) or d > reach or self.rng.random() > b.sketch_density:
                continue
            self._ink_line(x, y, px, py, max(0.6, size * 0.08), strength * 0.6 * (1 - d / reach))
            made += 1
            if made >= 6:
                break
        hist.append((x, y))
        if len(hist) > 400:
            del hist[:100]

    def _ink_line(self, x1, y1, x2, y2, width, alpha):
        """An anti-aliased line as a mask, laid down like a dab."""
        x0, y0 = int(math.floor(min(x1, x2) - width - 1)), int(math.floor(min(y1, y2) - width - 1))
        w = int(math.ceil(abs(x2 - x1) + 2 * width + 3))
        h = int(math.ceil(abs(y2 - y1) + 2 * width + 3))
        yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float32) + 0.5
        dx, dy = x2 - x1, y2 - y1
        L2 = max(dx * dx + dy * dy, 1e-6)
        t = np.clip(((xx - x1) * dx + (yy - y1) * dy) / L2, 0, 1)
        dist = np.hypot(xx - (x1 + t * dx), yy - (y1 + t * dy))
        m = np.clip(width / 2 + 0.5 - dist, 0, 1).astype(np.float32) * alpha
        if m.any():
            self._stamp(m, x0, y0, 1.0, 1.0)

    def _normal_colour(self, s):
        """Tangent-space normal from the pen's tilt (X right, Y up, Z out of the canvas); without tilt, the stroke's
        direction at the brush's set tilt."""
        tilt = math.hypot(s.tilt_x, s.tilt_y)
        if tilt > 0.5:
            az = math.atan2(-s.tilt_y, s.tilt_x)
        else:
            px, py = self.last_xy if self.last_xy else (s.x, s.y)
            az = math.atan2(-(s.y - py), s.x - px) if (s.x, s.y) != (px, py) else 0.0
            tilt = self.b.normal_tilt
        self.last_xy = (s.x, s.y)
        e = math.radians(90.0 - min(tilt, 89.0))
        n = (math.cos(e) * math.cos(az), math.cos(e) * math.sin(az), math.sin(e))
        return np.asarray([n[0] * 0.5 + 0.5, n[1] * 0.5 + 0.5, n[2] * 0.5 + 0.5], np.float32)

    def _fill_shape(self):
        """Shape mode: the outline drawn so far, filled (4x supersampled edges)."""
        pts = [(p.x, p.y) for p in self.pts]
        if len(pts) < 3:
            return
        from PIL import Image, ImageDraw
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, y0 = int(math.floor(min(xs))) - 1, int(math.floor(min(ys))) - 1
        w, h = int(math.ceil(max(xs))) - x0 + 2, int(math.ceil(max(ys))) - y0 + 2
        k = 4
        im = Image.new("L", (w * k, h * k), 0)
        ImageDraw.Draw(im).polygon([((x - x0) * k, (y - y0) * k) for x, y in pts], fill=255)
        m = np.asarray(im, np.float32).reshape(h, k, w, k).mean(axis=(1, 3)) / 255.0
        self.dabs += 1
        self._stamp(m.astype(np.float32), x0, y0, self.b.opacity, 1.0)

    def _jittered(self):
        """The stroke's colour moved at random in hue, saturation and value by the jitter amounts."""
        import colorsys
        b = self.b
        h, s_, v = colorsys.rgb_to_hsv(*[float(c) for c in self.base_colour])
        h = (h + (self.rng.random() - 0.5) * b.jitter_h) % 1.0
        s_ = min(1.0, max(0.0, s_ + (self.rng.random() - 0.5) * 2 * b.jitter_s))
        v = min(1.0, max(0.0, v + (self.rng.random() - 0.5) * 2 * b.jitter_v))
        return np.asarray(colorsys.hsv_to_rgb(h, s_, v), np.float32)

    def _dual_mask(self, n: int, size: float) -> np.ndarray:
        """Small marks of a second round tip scattered inside the dab; where none falls, the dab paints nothing."""
        b = self.b
        out = np.zeros((n, n), np.float32)
        d2 = max(1.0, size * b.dual_size)
        c = n // 2
        for _ in range(max(1, int(b.dual_count))):
            ox = (self.rng.random() - 0.5) * 2 * b.dual_scatter * size / 2
            oy = (self.rng.random() - 0.5) * 2 * b.dual_scatter * size / 2
            px, py = c + ox, c + oy
            ix, iy = math.floor(px), math.floor(py)
            m2, off = dab_mask(d2, 0.6, px - ix, py - iy)
            x0, y0 = ix + off, iy + off
            ax0, ay0 = max(0, x0), max(0, y0)
            ax1, ay1 = min(n, x0 + m2.shape[1]), min(n, y0 + m2.shape[0])
            if ax1 > ax0 and ay1 > ay0:
                sub = m2[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
                np.maximum(out[ay0:ay1, ax0:ax1], sub, out=out[ay0:ay1, ax0:ax1])
        return out

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
        k = min(max(b.smudge_length, 0.0), 1.0)
        if self.layer.height or b.impasto > 0:              # knives and smudges push the paint's thickness too
            from .tiles import read_tiles
            here_h = read_tiles(self.layer.height, x0, y0, w, h, 1)[..., 0]
            if self.carried_h is None or self.carried_h.shape != here_h.shape:
                self.carried_h = here_h + b.impasto
            self._stamp_height(m, x0, y0, strength, mode="mix", patch=self.carried_h)
            self.carried_h = self.carried_h * k + here_h * (1 - k)
        # what stays on the brush: the carried paint mixed with what was just picked up
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
                mm = self._clipped(m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0], ax0, ay0)
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
        mode = "buildup" if self.b.mode in ("airbrush", "spray", "sketch", "chalk") else self.b.mode
        for ty in range(y0 // TILE, (y0 + h - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x0 + w - 1) // TILE + 1):
                key = (tx, ty)
                ox, oy = tx * TILE, ty * TILE
                ax0, ay0 = max(x0, ox), max(y0, oy)
                ax1, ay1 = min(x0 + w, ox + TILE), min(y0 + h, oy + TILE)
                mm = self._clipped(m[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0], ax0, ay0)
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
                sample=None, clone=None, clip=None):
    """The right stroke for the brush: the pixel engine, or the MyPaint engine for MyPaint brushes. The Size,
    Opacity and Hardness sliders drive a MyPaint brush's own radius, opaque and hardness."""
    if brush.mode == "mypaint" and brush.myb is not None:
        from .mypaint import MyPaintStroke
        st = MyPaintStroke(layer, brush.myb, colour, undo, eraser=eraser or brush.eraser, seed=seed, sample=sample)
        st.base["radius_logarithmic"] = math.log(max(0.4, brush.size / 2))
        st.base["opaque"] = brush.opacity
        st.base["hardness"] = brush.hardness
        return st
    return Stroke(layer, brush, colour, undo, eraser, seed=seed, sample=sample, clone=clone, clip=clip)


def from_myb(myb) -> BrushSettings:
    """A MyPaint brush as an entry in the brush list."""
    b = myb.base
    return BrushSettings(myb.name, size=float(2 * math.exp(b["radius_logarithmic"])), hardness=min(1, b["hardness"]),
                         opacity=min(1.0, b["opaque"]), mode="mypaint", myb=myb, eraser=b["eraser"] >= 1.0,
                         tags="MyPaint",
                         best_for=(myb.comment.strip() or "a MyPaint brush") + " (MyPaint engine)",
                         weak_at=MODES["mypaint"][2])


def paint_stroke(layer: Layer, brush: BrushSettings, samples, colour=(0, 0, 0), undo=None, eraser=None,
                 sample=None, clip=None) -> Stroke:
    """Paint a whole stroke from a list of Samples (used by tests, the benchmark and scripts)."""
    if undo is not None:
        undo.begin("Erase" if (brush.eraser if eraser is None else eraser) else "Stroke")
    st = make_stroke(layer, brush, colour, undo, eraser, sample=sample, clip=clip)
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
