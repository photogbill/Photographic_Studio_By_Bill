"""Drawing aids for Paint (BUILD_PLAN_PAINT P6): assistants that guide the stroke, symmetry and seamless tiles,
colour harmonies, gamut masks and a colour-blindness preview. Each says what it is best for and weak at.

Assistants work as in Krita: the stroke's first few pixels say which assistant (and which of its directions) the
artist means; from then on every pen sample is pulled onto it, by the snapping strength (1 = exactly on it).
Nothing is predicted: only where the pen has been is moved sideways onto the guide.
"""
from __future__ import annotations

import math

import numpy as np

ASSISTANTS = {
    "ruler": ("Ruler", "straight lines between two points you set", "lines in other directions"),
    "parallel": ("Parallel ruler", "many lines in one direction: hatching, rain, wood grain", "lines that converge"),
    "vanishing": ("Vanishing point", "lines that all meet at one point", "anything not aimed at the point"),
    "perspective": ("Perspective (1, 2 or 3 points)", "buildings and rooms: each line goes to the vanishing point "
                    "it starts towards, or stays upright / level", "curved things"),
    "ellipse": ("Ellipse", "wheels, plates, round things seen at an angle", "other curves"),
    "concentric": ("Concentric ellipses", "rings, ripples, the rims of cups at any size", "straight lines"),
    "fisheye": ("Fisheye (curvilinear)", "five-point / fisheye perspective: lines bend through two points on the "
                "horizon", "ordinary perspective"),
    "spline": ("Curve (spline)", "one smooth curve you shape with four points: a road, a spine, a river",
               "anything else"),
}


def _proj_line(px, py, ax, ay, dx, dy):
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    t = (px - ax) * ux + (py - ay) * uy
    return ax + t * ux, ay + t * uy


def _proj_ellipse(px, py, cx, cy, rx, ry, ang, scale=1.0):
    """Nearest point on an ellipse (by angle of the point in the ellipse's own frame: close and smooth)."""
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    x, y = px - cx, py - cy
    u, v = x * ca + y * sa, -x * sa + y * ca
    t = math.atan2(v / max(ry * scale, 1e-6), u / max(rx * scale, 1e-6))
    u2, v2 = rx * scale * math.cos(t), ry * scale * math.sin(t)
    return cx + u2 * ca - v2 * sa, cy + u2 * sa + v2 * ca


def _ellipse_scale(px, py, cx, cy, rx, ry, ang):
    a = math.radians(ang)
    x, y = px - cx, py - cy
    u, v = x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a)
    return math.sqrt((u / max(rx, 1e-6)) ** 2 + (v / max(ry, 1e-6)) ** 2)


def _circle_through(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return None
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d
    return ux, uy, math.hypot(x1 - ux, y1 - uy)


class Assistant:
    """``kind`` from ASSISTANTS; ``points`` its handles in canvas pixels:
    ruler / parallel: two points; vanishing: one; perspective: one to three vanishing points; ellipse /
    concentric: centre, end of the first axis, a point on the second axis; fisheye: the two horizon points;
    spline: four control points."""

    def __init__(self, kind, points):
        self.kind, self.points = kind, [tuple(map(float, p)) for p in points]

    def variants(self, sx, sy):
        """The guides this assistant offers for a stroke starting at (sx, sy): functions p -> snapped p."""
        k, P = self.kind, self.points
        if k == "ruler":
            (ax, ay), (bx, by) = P[:2]
            return [lambda x, y: _proj_line(x, y, ax, ay, bx - ax, by - ay)]
        if k == "parallel":
            (ax, ay), (bx, by) = P[:2]
            return [lambda x, y: _proj_line(x, y, sx, sy, bx - ax, by - ay)]
        if k == "vanishing":
            vx, vy = P[0]
            return [lambda x, y: _proj_line(x, y, sx, sy, vx - sx, vy - sy)]
        if k == "perspective":
            out = [(lambda vx, vy: (lambda x, y: _proj_line(x, y, sx, sy, vx - sx, vy - sy)))(vx, vy)
                   for vx, vy in P]
            if len(P) < 3:                                       # one and two points: uprights stay upright
                out.append(lambda x, y: (sx, y))
            if len(P) < 2:                                       # one point: and level lines stay level
                out.append(lambda x, y: (x, sy))
            return out
        if k in ("ellipse", "concentric"):
            (cx, cy), (ax, ay), (bx, by) = P[:3]
            rx = math.hypot(ax - cx, ay - cy)
            ang = math.degrees(math.atan2(ay - cy, ax - cx))
            a = math.radians(ang)
            ry = abs(-(bx - cx) * math.sin(a) + (by - cy) * math.cos(a)) or rx
            sc = 1.0 if k == "ellipse" else _ellipse_scale(sx, sy, cx, cy, rx, ry, ang)
            return [lambda x, y: _proj_ellipse(x, y, cx, cy, rx, ry, ang, sc)]
        if k == "fisheye":
            (ax, ay), (bx, by) = P[:2]
            c = _circle_through((ax, ay), (bx, by), (sx, sy))
            if c is None:                                       # on the horizon itself: the horizon line
                return [lambda x, y: _proj_line(x, y, ax, ay, bx - ax, by - ay)]
            ux, uy, r = c
            return [lambda x, y: (ux + (x - ux) * r / max(math.hypot(x - ux, y - uy), 1e-6),
                                  uy + (y - uy) * r / max(math.hypot(x - ux, y - uy), 1e-6))]
        if k == "spline":
            p = np.asarray(P[:4], np.float64)
            t = np.linspace(0, 1, 400)[:, None]
            curve = ((1 - t) ** 3) * p[0] + 3 * ((1 - t) ** 2) * t * p[1] + 3 * (1 - t) * t * t * p[2] + t ** 3 * p[3]

            def onto(x, y, curve=curve):
                i = int(np.argmin(((curve - (x, y)) ** 2).sum(1)))
                return float(curve[i, 0]), float(curve[i, 1])
            return [onto]
        raise ValueError(k)


class Guide:
    """One stroke's guide: chosen from the first ``decide`` pixels of movement, then held."""

    def __init__(self, assistants, strength=1.0, decide=8.0):
        self.assistants, self.strength, self.decide = list(assistants), float(strength), decide
        self.start = None
        self.fn = None
        self.early = []

    def __call__(self, x, y):
        if not self.assistants or self.strength <= 0:
            return x, y
        if self.start is None:
            self.start = (x, y)
            return x, y
        if self.fn is None:
            self.early.append((x, y))
            if math.hypot(x - self.start[0], y - self.start[1]) < self.decide:
                return x, y
            sx, sy = self.start
            best = None
            for a in self.assistants:                           # the guide the stroke already follows best
                for f in a.variants(sx, sy):
                    err = sum(math.hypot(px - f(px, py)[0], py - f(px, py)[1]) for px, py in self.early)
                    if best is None or err < best[0]:
                        best = (err, f)
            self.fn = best[1]
        gx, gy = self.fn(x, y)
        s = self.strength
        return x + (gx - x) * s, y + (gy - y) * s


# ---------------------------------------------------------------- symmetry and wrap-around
SYMMETRY = {
    "none": ("None", "", ""),
    "x": ("Mirror left-right", "faces, figures, vases: both sides at once", "anything lop-sided on purpose"),
    "y": ("Mirror top-bottom", "reflections in water, tiles", "most subjects"),
    "xy": ("Mirror both ways", "ornaments and four-fold patterns", "anything else"),
    "radial": ("Radial", "wheels, flowers, rose windows: N copies turned about the centre", "lettering"),
    "mandala": ("Mandala", "radial with mirroring: snowflakes, mandalas", "anything that must not mirror"),
    "wrap": ("Seamless tile (wrap-around)", "textures that repeat without a seam: paint across one edge and it comes "
             "in at the other", "single pictures"),
}


def symmetry_maps(kind, w, h, cx=None, cy=None, spokes=6):
    """Functions (x, y) -> (x, y) for every extra copy of the stroke (the original is not included)."""
    cx = w / 2 if cx is None else cx
    cy = h / 2 if cy is None else cy
    mx = lambda x, y: (2 * cx - x, y)
    my = lambda x, y: (x, 2 * cy - y)
    if kind == "x":
        return [mx]
    if kind == "y":
        return [my]
    if kind == "xy":
        return [mx, my, lambda x, y: (2 * cx - x, 2 * cy - y)]
    if kind in ("radial", "mandala"):
        out = []
        n = max(2, int(spokes))
        for k in range(1, n):
            a = 2 * math.pi * k / n
            ca, sa = math.cos(a), math.sin(a)
            out.append((lambda ca, sa: lambda x, y: (cx + (x - cx) * ca - (y - cy) * sa,
                                                     cy + (x - cx) * sa + (y - cy) * ca))(ca, sa))
        if kind == "mandala":
            out += [(lambda f: lambda x, y: f(*mx(x, y)))(f) for f in [lambda x, y: (x, y)] + out]
        return out
    if kind == "wrap":
        return [(lambda dx, dy: lambda x, y: (x + dx, y + dy))(dx, dy)
                for dx in (-w, 0, w) for dy in (-h, 0, h) if (dx, dy) != (0, 0)]
    return []


# ---------------------------------------------------------------- colour harmonies and gamut masks (in OKLCH)
HARMONIES = {
    "complementary": ("Complementary", [0, 180], "the strongest contrast of hue", "can look harsh"),
    "analogous": ("Analogous", [-30, 0, 30], "calm, natural colour: landscapes, light", "low contrast"),
    "triadic": ("Triadic", [0, 120, 240], "lively, balanced colour", "busy if all three are strong"),
    "split": ("Split complementary", [0, 150, 210], "contrast that is easier than complementary", "less punch"),
    "tetradic": ("Tetradic (square)", [0, 90, 180, 270], "rich schemes with many colours", "hard to balance"),
}

GAMUT_MASKS = {
    "none": ("No mask", [], "", ""),
    "triad": ("Triad", [(0, 40), (120, 40), (240, 40)], "a three-colour harmony that still feels unified",
              "colours outside the three sectors"),
    "complementary": ("Complementary", [(0, 60), (180, 60)], "warm against cool, light against shadow", "greens "
                      "and purples between them"),
    "atmospheric": ("Atmospheric (one wide sector)", [(0, 110)], "unified light: a sunset, a misty morning",
                    "strong contrasts"),
    "split": ("Split complementary", [(0, 50), (150, 40), (210, 40)], "contrast with harmony", "the rest of the "
              "wheel"),
}


def harmony(rgb, kind):
    """Colours in harmony with ``rgb`` (sRGB), same lightness and chroma, hues turned in OKLCH."""
    from .colour import oklch_to_srgb, srgb_to_oklch
    L, C, H = (float(v) for v in srgb_to_oklch(np.asarray(rgb[:3], np.float32)))
    return [oklch_to_srgb(L, C, (H + d) % 360) for d in HARMONIES[kind][1]]


def hue_allowed(h, mask, rotation=0.0):
    sectors = GAMUT_MASKS[mask][1]
    if not sectors:
        return True
    return any(abs(((h - (c + rotation)) + 180) % 360 - 180) <= w / 2 + 0.01 for c, w in sectors)


def apply_mask(rgb, mask, rotation=0.0):
    """Keep a colour inside the gamut mask: its hue is moved to the nearest edge of an open sector (lightness and
    chroma kept); greys pass."""
    from .colour import oklch_to_srgb, srgb_to_oklch
    if mask == "none":
        return tuple(rgb[:3])
    L, C, H = (float(v) for v in srgb_to_oklch(np.asarray(rgb[:3], np.float32)))
    if C < 0.02 or hue_allowed(H, mask, rotation):
        return tuple(float(v) for v in rgb[:3])
    best = None
    for c, w in GAMUT_MASKS[mask][1]:
        for edge in (c + rotation - w / 2, c + rotation + w / 2):
            d = abs(((H - edge) + 180) % 360 - 180)
            if best is None or d < best[0]:
                best = (d, edge % 360)
    return oklch_to_srgb(L, C, best[1])


# ---------------------------------------------------------------- colour-blindness preview (Machado et al. 2009)
VISION = {
    "normal": ("Normal vision", "", ""),
    "protan": ("Protanopia (no red cones)", "checking a painting reads for about 1 in 100 men", "simulated, severity 1"),
    "deutan": ("Deuteranopia (no green cones)", "the most common colour blindness (about 1 in 16 men, all kinds)",
               "simulated, severity 1"),
    "tritan": ("Tritanopia (no blue cones)", "rare blue-yellow colour blindness", "simulated, severity 1"),
    "achroma": ("Achromatopsia (values only)", "seeing values only, like the Values button", "colour is gone"),
}
_MACHADO = {   # linear RGB, severity 1.0 (Machado, Oliveira & Fernandes, IEEE TVCG 2009)
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]], np.float32),
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]], np.float32),
    "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]], np.float32),
}


def simulate_vision(rgb, kind):
    """Straight sRGB (..., 3) as seen with a colour-vision deficiency (for the view only)."""
    from .colour import linear_to_srgb, srgb_to_linear
    if kind == "normal":
        return rgb
    lin = srgb_to_linear(np.clip(rgb, 0, 1))
    if kind == "achroma":
        y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        return np.repeat(linear_to_srgb(y)[..., None], 3, -1)
    return np.clip(linear_to_srgb(np.clip(lin @ _MACHADO[kind].T, 0, 1)), 0, 1)


# ---------------------------------------------------------------- one pen stroke drawn as several
def _union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


class GuidedStroke:
    """Feeds one pen stroke through a Guide (assistants) and into one stroke per symmetry copy; all copies go into
    the same undo step. ``factory()`` makes a fresh brush stroke. With wrap-around, a copy is drawn only while it is
    on (or near) the canvas, so painting across an edge comes in at the other side without painting off the canvas.
    Pen tilt and barrel rotation are turned and mirrored with each copy, so flat and angled tips match."""

    def __init__(self, factory, maps=(), guide=None, wrap=None, margin=0.0):
        self.factory, self.guide = factory, guide
        self.maps = [None] + list(maps)                         # None: the stroke itself
        self.wrap, self.margin = wrap, float(margin)            # wrap: (width, height) of the canvas
        self.strokes = [factory()] + [None] * (len(self.maps) - 1)
        self.done = []

    @property
    def orig(self):
        out = {}
        for s in [s for s in self.strokes if s is not None] + self.done:
            out.update(getattr(s, "orig", {}))
        return out

    def _place(self, f, s):
        from dataclasses import replace
        x, y = f(s.x, s.y)
        ex = f(s.x + 1.0, s.y)
        ey = f(s.x, s.y + 1.0)
        jx, jy = (ex[0] - x, ex[1] - y), (ey[0] - x, ey[1] - y)
        det = jx[0] * jy[1] - jx[1] * jy[0]
        tx = s.tilt_x * jx[0] + s.tilt_y * jy[0]
        ty = s.tilt_x * jx[1] + s.tilt_y * jy[1]
        turn = math.degrees(math.atan2(jx[1], jx[0]))
        rot = (-s.rotation if det < 0 else s.rotation) + turn
        return replace(s, x=x, y=y, tilt_x=tx, tilt_y=ty, rotation=rot)

    def add(self, s):
        if self.guide is not None:
            from dataclasses import replace
            gx, gy = self.guide(s.x, s.y)
            s = replace(s, x=gx, y=gy)
        rect = None
        for i, f in enumerate(self.maps):
            p = s if f is None else self._place(f, s)
            if self.wrap is not None and f is not None:
                w, h, m = self.wrap[0], self.wrap[1], self.margin
                if not (-m <= p.x <= w + m and -m <= p.y <= h + m):
                    if self.strokes[i] is not None:             # left the canvas: this copy ends here
                        rect = _union(rect, self.strokes[i].finish())
                        self.done.append(self.strokes[i])
                        self.strokes[i] = None
                    continue
            if self.strokes[i] is None:
                self.strokes[i] = self.factory()
            rect = _union(rect, self.strokes[i].add(p))
        return rect

    def finish(self):
        rect = None
        for st in self.strokes:
            if st is not None:
                rect = _union(rect, st.finish())
        return rect
