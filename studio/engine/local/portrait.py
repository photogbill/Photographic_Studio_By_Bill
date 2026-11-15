"""Portrait retouching (plan section 9.7): blemishes, lines, under-eye shadows, smile lines, even skin, eyes, teeth,
the jawline, and subtle body sculpting. Everything is stored in the edit as plain data and computed at render time.

    "portrait": {
      "faces": [{"pts": [[x, y], ... 468], "conf": 0..1, "how": "found" | "eyes", "enabled": true}],
      "younger": 0..100,         one slider for all of the below, each kept at least this far along
      "lines": 0..100,           wrinkles and fine lines (forehead, between the brows, crow's feet, under the eyes)
      "under_eyes": 0..100,      dark circles and hollows under the eyes
      "smile_lines": 0..100,     the folds from nose to mouth and mouth to chin
      "even_skin": 0..100,       blotches and uneven colour (the face only)
      "jawline": 0..100,         firms the jaw below the cheeks (a small warp)
      "double_chin": 0..100,     lifts the soft area under the jaw toward it (a small warp), deepens the shadow
                                 just under the jawline where there is one, and softens the crease. Not part of
                                 Younger: it reshapes, so it is only ever asked for on its own
      "eyes": 0..100,            brighter whites, fewer red veins, crisper iris
      "teeth": 0..100,           whiter (towards the eye whites' colour, so a warm photo stays warm), a little brighter
      "iris_light", "iris_clarity", "iris_colour", "ring", "catch": 0..100   the eye tools (local/eyes.py), each
                                 needing the irises found in the face entry ("irises"); not part of Younger
      "eye_colour": {"preset": "blue" | "green" | ... | "custom", "hue", "sat", "light", "amount": 0..100,
                     "which": "both" | "right" | "left"}
      "sculpt": [...]            body sculpting: brush strokes and shape tools (see below)
    }

Points are fractions of the S frame (camera crop before rotation), like masks and spots, so the face stays where
it is when the photo is straightened or cropped.

How the face work is done, and why
* What makes a face read older is mostly light and shade at particular sizes: lines are narrow dark grooves,
  hollows and folds are wider soft shadows, age spots and redness are blotches of colour, and the eye whites and
  teeth yellow. So the face's brightness is split into layers by size with edge-aware (guided) filters (He, Sun and
  Tang 2013), the same family Smooth skin uses: the finest layer (pores, fine hair) is never touched, the line layer
  has its dark grooves filled in (bright ridges only eased), and the fold layer has its shadows lifted part way, so
  the face keeps its shape. That is the approach the retouching literature describes as natural, as opposed to
  blurring (AutoRetouch, Shafaei, Little and Schmidt 2021; Liang et al. 2014 region-aware beautification; Batool
  and Chellappa 2014 on wrinkle modelling). Zones come from the 468-point face mesh, and every effect fades out at
  the zone edges.
* The changes are worked out once per face on a copy of the face (at most ~800 pixels across) and kept as a smooth
  gain map attached to the photo, so a 100% view and the export get exactly the same change, the full-resolution
  pores and texture are untouched, and moving exposure does not redo the work.
* Blemishes are found by blob detection (dark spots and red spots, at the sizes of real blemishes, standing well
  above the skin's own texture), and each becomes a Heal spot in Retouch (amber, removable one by one). Very dark,
  well-defined spots are left alone: they are usually moles or beauty marks, part of the person's face.

Body sculpting is a warp of the S frame, applied after retouching, in the order the strokes were made:
    {"kind": "push", "pts": [[x, y], ...], "vec": [[dx, dy], ...], "r": radius (fraction of the long edge)}
    {"kind": "shrink" | "grow", "pts": [[x, y], ...], "r": .., "amount": 0..1}
    {"kind": "shape", "tool": "breast" | "butt" | "waist" | "tummy" | "arm" | "thigh", "c": [x, y],
     "e1": [ex, ey] (direction of the long axis, S pixels), "a": half length, "b": half width (fractions of the
     long edge), "up": [ux, uy] (the photo's up, in the S frame), "lift": -100..100, "size": -100..100,
     "slim": -100..100, "enabled": true}
Every warp is capped (a push moves at most 30% of its radius, a lift 14% of the shape, a slim 12% of the width) and
fades smoothly to nothing, so it stays believable; each is inverted exactly (fixed-point iteration), so pixels are
sampled, never smeared.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter

from ... import log as slog
from . import faces as fz

log = slog.get("engine.portrait")

LUMA = np.array([0.2880, 0.7119, 0.0001], np.float32)
FACE_KEYS = ("younger", "lines", "under_eyes", "smile_lines", "even_skin", "jawline", "double_chin", "eyes", "teeth",
             "iris_light", "iris_clarity", "iris_colour", "ring", "catch")
WORK_FACE_PX = 640                    # the face's width on the copy the changes are worked out on (at most)
SHAPE_TOOLS = ("breast", "butt", "waist", "tummy", "arm", "thigh")
LIFT_TOOLS = ("breast", "butt")
JAW_LINE = list(fz.JAW_L) + [fz.CHIN] + list(fz.JAW_R)[::-1]


# ------------------------------------------------------------------------------------------ settings
def amounts(p: dict | None) -> dict:
    """Effective 0..1 amount of each face effect: its own slider, or the Younger slider's share, whichever is more."""
    p = p or {}
    y = min(max(float(p.get("younger", 0)), 0.0), 100.0) / 100.0
    share = {"lines": 1.0, "under_eyes": 0.85, "smile_lines": 0.6, "even_skin": 0.4, "jawline": 0.5, "eyes": 0.4,
             "teeth": 0.3, "double_chin": 0.0, "iris_light": 0.0, "iris_clarity": 0.0, "iris_colour": 0.0,
             "ring": 0.0, "catch": 0.0}
    out = {}
    for k, s in share.items():
        v = min(max(float(p.get(k, 0)), 0.0), 100.0) / 100.0
        out[k] = max(v, y * s)
    return out


def faces_of(p: dict | None) -> list:
    return [f for f in (p or {}).get("faces") or [] if f.get("enabled", True) and len(f.get("pts") or []) == 468]


def face_active(p: dict | None) -> bool:
    from .eyes import colour_settings
    a = amounts(p)
    return bool(faces_of(p)) and (any(a[k] > 1e-4 for k in a if k != "jawline") or colour_settings(p) is not None)


def sculpt_active(p: dict | None) -> bool:
    return bool(warps(p, 4096, 4096))          # a nominal size: the double-chin warp skips faces too small to see


# ------------------------------------------------------------------------------------------ a face's geometry
class Face:
    """One stored face in full-resolution S-frame pixels."""

    def __init__(self, f: dict, sw: int, sh: int):
        self.P = np.asarray(f["pts"], float).reshape(468, 2) * np.array([sw, sh], float)
        self.W = max(fz.size_of(self.P), 4.0)
        self.er, self.el = fz.eye_centres(self.P)
        self.iod = max(float(np.hypot(*(self.el - self.er))), 1.0)
        mid = 0.5 * (self.er + self.el)
        e = (self.el - self.er) / self.iod
        n = np.array([e[1], -e[0]])
        if np.dot(mid - self.P[fz.CHIN], n) < 0:
            n = -n
        self.up = n                                   # towards the forehead
        self.x0, self.y0 = self.P.min(axis=0)
        self.x1, self.y1 = self.P.max(axis=0)
        from .eyes import iris_list
        self.irises = iris_list(f, sw, sh)

    def window(self, margin: float, sw: int, sh: int):
        m = margin * self.W
        return (max(0, int(self.x0 - m)), max(0, int(self.y0 - m)), min(sw, int(self.x1 + m) + 1),
                min(sh, int(self.y1 + m) + 1))


# ------------------------------------------------------------------------------------------ zones
def _poly(draw, pts, fill):
    draw.polygon([(float(x), float(y)) for x, y in pts], fill=fill)


def zones(face: Face) -> dict:
    """The face's zones as soft maps on a small grid in S pixels: {"x0", "y0", "q", name: array}."""
    from PIL import Image as PI, ImageDraw
    W = face.W
    q = W / 200.0
    m = 0.6 * W
    x0, y0 = face.x0 - m, face.y0 - m
    nw, nh = int((face.x1 - face.x0 + 2 * m) / q) + 2, int((face.y1 - face.y0 + 2 * m) / q) + 2
    P = (face.P - np.array([x0, y0])) / q
    up = face.up * (0.22 * W / q)

    def canvas():
        im = PI.new("L", (nw, nh), 0)
        return im, ImageDraw.Draw(im)

    def arr(im, blur):
        a = np.asarray(im, np.float32) / 255.0
        return gaussian_filter(a, blur / q) if blur > 0 else a

    def dilated(idx, d):
        im, dr = canvas()
        _poly(dr, P[idx], 255)
        pts = [(float(x), float(y)) for x, y in P[idx]]
        dr.line(pts + [pts[0]], fill=255, width=max(1, int(round(2 * d / q))), joint="curve")
        return arr(im, 0)

    upper = [127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356]
    im, dr = canvas()
    _poly(dr, P[fz.FACE_OVAL], 255)
    oval = arr(im, 0)
    im, dr = canvas()
    _poly(dr, np.concatenate([P[upper], (P[upper] + up)[::-1]]), 255)
    ext = arr(im, 0) * (1.0 - oval)
    feat = np.maximum.reduce([dilated(fz.RIGHT_EYE, 0.035 * W), dilated(fz.LEFT_EYE, 0.035 * W),
                              dilated(fz.RIGHT_BROW, 0.025 * W), dilated(fz.LEFT_BROW, 0.025 * W),
                              dilated(fz.LIPS_OUTER, 0.02 * W)])
    im, dr = canvas()
    for a, b in fz.NOSTRILS:
        c = 0.5 * (P[a] + P[b])
        r = 0.65 * float(np.hypot(*(P[a] - P[b])))
        dr.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=255)
    feat = np.maximum(feat, arr(im, 0))
    skin_hard = np.clip(np.maximum(oval, ext) - feat, 0.0, 1.0)
    skin = gaussian_filter(skin_hard, 0.015 * W / q)
    # lines: everywhere on the skin a little, fully where lines form
    im, dr = canvas()
    brow_top = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
    _poly(dr, np.concatenate([P[brow_top] - up * 0.15, (P[upper[2:-2]] + up)[::-1]]), 255)     # forehead
    for c, rx, ry in ((P[9], 0.09, 0.12), (P[168], 0.07, 0.08)):                                   # between brows
        dr.ellipse([c[0] - rx * W / q, c[1] - ry * W / q, c[0] + rx * W / q, c[1] + ry * W / q], fill=255)
    for corner, inner in ((33, 133), (263, 362)):                                                  # crow's feet
        d = P[corner] - P[inner]
        d = d / (np.hypot(*d) + 1e-6)
        c = P[corner] + d * 0.09 * W / q
        r = 0.11 * W / q
        dr.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=255)
    under_r = [33, 7, 163, 144, 145, 153, 154, 155, 133, 121, 120, 119, 118, 117, 111]
    under_l = [263, 249, 390, 373, 374, 380, 381, 382, 362, 350, 349, 348, 347, 346, 340]
    _poly(dr, P[under_r], 255)
    _poly(dr, P[under_l], 255)
    special = arr(im, 0)
    peri = np.clip(dilated(fz.LIPS_OUTER, 0.06 * W) - dilated(fz.LIPS_OUTER, 0.012 * W), 0, 1) * 0.8
    near = np.maximum.reduce([dilated(fz.RIGHT_EYE, 0.02 * W), dilated(fz.LEFT_EYE, 0.02 * W),
                              dilated(fz.RIGHT_BROW, 0.012 * W), dilated(fz.LEFT_BROW, 0.012 * W),
                              dilated(fz.LIPS_OUTER, 0.01 * W)])
    skin_close = gaussian_filter(np.clip(np.maximum(oval, ext) - near, 0.0, 1.0), 0.008 * W / q)
    lines = skin_close * np.maximum(0.45, gaussian_filter(np.maximum(special, peri), 0.02 * W / q))
    # folds: under the eyes, nose to mouth, mouth to chin
    im, dr = canvas()
    _poly(dr, P[under_r], 255)
    _poly(dr, P[under_l], 255)
    for idx in (under_r, under_l):                    # the hollow reaches a little below the mesh's line
        low = P[idx[9:]]
        _poly(dr, np.concatenate([low, (low - up * 0.27)[::-1]]), 255)
    under = gaussian_filter(np.clip(arr(im, 0) - dilated(fz.RIGHT_EYE, 0.012 * W) - dilated(fz.LEFT_EYE, 0.012 * W),
                                    0, 1), 0.025 * W / q)
    im, dr = canvas()
    centre = P[fz.NOSE_TIP]
    for wing, cheek, mouth, jaw in ((fz.NOSE_WING_R, 206, fz.MOUTH_R, 149), (fz.NOSE_WING_L, 426, fz.MOUTH_L, 378)):
        out = P[mouth] + (P[mouth] - centre) * 0.12
        path = [P[wing], P[cheek], out]
        dr.line([(float(x), float(y)) for x, y in path], fill=255, width=max(1, int(0.07 * W / q)), joint="curve")
        low = out + (P[jaw] - out) * 0.55
        dr.line([(float(out[0]), float(out[1])), (float(low[0]), float(low[1]))], fill=200,
                width=max(1, int(0.06 * W / q)))
    smile = gaussian_filter(arr(im, 0), 0.025 * W / q) * gaussian_filter(skin_hard, 0.01 * W / q)
    im, dr = canvas()
    _poly(dr, P[fz.RIGHT_EYE], 255)
    _poly(dr, P[fz.LEFT_EYE], 255)
    eye = arr(im, 0.004 * W)
    im, dr = canvas()
    _poly(dr, P[fz.LIPS_INNER], 255)
    teeth = arr(im, 0.004 * W)
    im, dr = canvas()
    wr, wl = P[fz.NOSE_WING_R], P[fz.NOSE_WING_L]          # the lower nose: wings, nostrils and their creases
    c = 0.5 * (wr + wl) + 0.15 * (P[fz.NOSE_TIP] - 0.5 * (wr + wl))
    rx = 0.62 * float(np.hypot(*(wr - wl)))
    dr.ellipse([c[0] - rx, c[1] - 0.75 * rx, c[0] + rx, c[1] + 0.75 * rx], fill=255)
    nose = arr(im, 0)
    im, dr = canvas()                                     # the face's outer edge: hair and background meet it
    pts = [(float(x), float(y)) for x, y in P[fz.FACE_OVAL]]
    dr.line(pts + [pts[0]], fill=255, width=max(1, int(round(0.10 * W / q))), joint="curve")
    rim = arr(im, 0)
    blem = np.clip(oval - np.maximum.reduce([dilated(fz.RIGHT_EYE, 0.08 * W), dilated(fz.LEFT_EYE, 0.08 * W),
                                             dilated(fz.RIGHT_BROW, 0.05 * W), dilated(fz.LEFT_BROW, 0.05 * W),
                                             dilated(fz.LIPS_OUTER, 0.035 * W), feat, nose, rim]), 0, 1)
    im, dr = canvas()
    for i in (50, 280, 205, 425):
        c, r = P[i], 0.06 * W / q
        dr.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=255)
    cheek = arr(im, 0) * (skin_hard > 0.5)
    # double chin: the band under the jaw where the crease forms (from 6% to 30% of the face below the jaw line),
    # and a thin band just under the jaw (its shadow), fading towards the jaw angles
    dn = -face.up * (W / q)
    jaw = P[JAW_LINE]
    im, dr = canvas()
    _poly(dr, np.concatenate([jaw + 0.06 * dn, (jaw + 0.30 * dn)[::-1]]), 255)
    chin = arr(im, 0.03 * W)
    im, dr = canvas()
    mid = P[JAW_LINE[2:-2]] + 0.025 * dn
    dr.line([(float(x), float(y)) for x, y in mid], fill=255, width=max(1, int(round(0.05 * W / q))), joint="curve")
    jawshade = arr(im, 0.015 * W)
    return {"x0": x0, "y0": y0, "q": q, "skin": skin, "lines": lines, "under": under, "smile": smile, "eye": eye,
            "teeth": teeth, "blem": blem, "cheek": cheek, "ext": ext, "chin": chin, "jawshade": jawshade}


def sample_zone(z: dict, name: str, X, Y) -> np.ndarray:
    """A zone at S-pixel positions (bilinear, zero outside)."""
    a = z[name]
    fy = (np.asarray(Y, float) - z["y0"]) / z["q"] - 0.5
    fx = (np.asarray(X, float) - z["x0"]) / z["q"] - 0.5
    return map_coordinates(a, [fy, fx], order=1, mode="constant", cval=0.0).astype(np.float32)


# ------------------------------------------------------------------------------------------ the face gain map
def _down(img: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return img
    h, w = img.shape[0] // k * k, img.shape[1] // k * k
    return img[:h, :w].reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3)).astype(np.float32)


def fetch_plan(face: Face, sw: int, sh: int):
    """(S box, half-size?, extra downsampling) for the copy of a face the gain map is worked out on."""
    box = face.window(0.6, sw, sh)
    half = face.W >= 1200
    step = 2 if half else 1
    k = max(1, int(math.ceil(face.W / step / WORK_FACE_PX)))
    return box, half, k


def _skin_gate(lab: np.ndarray, L: np.ndarray, cheek: np.ndarray, s_col: float = 0.07,
               s_lum: float = 1.8) -> np.ndarray:
    """How much each pixel looks like this face's own cheeks (colour direction and brightness): keeps hair and
    glasses out of the forehead and the edges of the face, in any light (candle light included)."""
    sel = cheek > 0.5
    if sel.sum() < 20:
        return np.ones(L.shape, np.float32)
    Lab_L = np.maximum(lab[..., 0], 1e-3)
    ra, rb = lab[..., 1] / Lab_L, lab[..., 2] / Lab_L
    ma, mb, mL = float(np.median(ra[sel])), float(np.median(rb[sel])), float(np.median(L[sel]))
    d2 = ((ra - ma) ** 2 + (rb - mb) ** 2) / s_col ** 2 + ((L - mL) / s_lum) ** 2
    return np.exp(-0.5 * d2).astype(np.float32)


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; wide ones are done on a smaller copy (the result is smooth, so nothing is lost)."""
    if sigma <= 6.0:
        return gaussian_filter(a, sigma)
    f = max(2, int(sigma / 3.0))
    h, w = a.shape
    hh, ww = -(-h // f) * f, -(-w // f) * f
    pad = np.pad(a, ((0, hh - h), (0, ww - w)), mode="edge")
    small = pad.reshape(hh // f, f, ww // f, f).mean(axis=(1, 3))
    small = gaussian_filter(small, sigma / f)
    yy = (np.arange(h) + 0.5) / f - 0.5
    xx = (np.arange(w) + 0.5) / f - 0.5
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return map_coordinates(small, [Y, X], order=1, mode="nearest").astype(a.dtype)


def _line_footprints(n: int, k: int = 6) -> list:
    """Straight-line structuring elements of length n at k angles."""
    out = []
    c = (n - 1) / 2.0
    for i in range(k):
        a = math.pi * i / k
        fp = np.zeros((n, n), bool)
        for t in np.linspace(-c, c, 2 * n):
            fp[int(round(c + t * math.sin(a))), int(round(c + t * math.cos(a)))] = True
        out.append(fp)
    return out


def line_layers(fine: np.ndarray, n: int):
    """(dark, bright) elongated detail of a log-brightness image: grooves and ridges narrower than n pixels but
    longer than they are wide. A grey-level closing along a line fills a dark groove when the line crosses it and
    leaves it when the line runs along it, so (largest closing over 6 angles) - (smallest) is the depth of the
    grooves alone: round dark spots (freckles, moles, pores) fill at every angle and cancel out, so they stay
    (Serra 1982, mathematical morphology; oriented line operators as in Soille and Talbot 2001). The same with
    openings for bright ridges."""
    from scipy.ndimage import grey_closing, grey_opening
    cmax = cmin = omax = omin = None
    for fp in _line_footprints(n):
        c = grey_closing(fine, footprint=fp)
        o = grey_opening(fine, footprint=fp)
        cmax = c if cmax is None else np.maximum(cmax, c)
        cmin = c if cmin is None else np.minimum(cmin, c)
        omax = o if omax is None else np.maximum(omax, o)
        omin = o if omin is None else np.minimum(omin, o)
    sm = 0.7
    dark = -gaussian_filter(cmax - cmin, sm)            # <= 0: how deep the groove is
    bright = np.clip(gaussian_filter(omax - omin, sm), 0.0, 0.3)
    return dark.astype(np.float32), bright.astype(np.float32)


def face_gain(rgb: np.ndarray, gx: float, gy: float, q: float, face: Face, amt: dict, colour: dict | None = None
              ) -> np.ndarray:
    """The gain map (h, w, 3) for one face, on a scene-linear copy ``rgb`` whose pixel (0, 0) is centred at S
    (gx + (q - 1) / 2, gy + (q - 1) / 2), pixels q S-pixels apart. Multiply the photo by it (sampled at each
    pixel's S position) to apply the face effects."""
    from ..color.grade import from_oklab, to_oklab
    from .masks import _smooth_skin
    h, w = rgb.shape[:2]
    z = zones(face)
    c = 0.5 * (q - 1)
    X = gx + q * np.arange(w)[None, :] + c
    Y = gy + q * np.arange(h)[:, None] + c
    X, Y = np.broadcast_to(X, (h, w)), np.broadcast_to(Y, (h, w))
    Z = {k: sample_zone(z, k, X, Y) for k in ("skin", "lines", "under", "smile", "eye", "teeth", "cheek", "ext",
                                              "chin", "jawshade")}
    Wk = face.W / q
    pos = np.maximum(rgb, 0.0)
    Yl = np.maximum(pos @ LUMA, 1e-6)
    L = np.log2(Yl).astype(np.float32)
    lab = to_oklab(pos)
    gate = _skin_gate(lab, L, Z["cheek"])               # inside the mesh: lenient (one side of a candle-lit face
    strict = _skin_gate(lab, L, Z["cheek"], 0.05, 1.0)  # can be stops darker); above it: only forehead skin, no hair
    gate = (0.5 + 0.5 * gate) * (1.0 - Z["ext"]) + strict * Z["ext"]
    for k in ("skin", "lines", "under", "smile", "chin", "jawshade"):
        Z[k] = Z[k] * gate
    out = rgb.astype(np.float32).copy()
    # even skin first (the blotch layer), then light and shade, then colour
    if amt.get("even_skin", 0) > 0:
        rm = max(2, int(round(0.03 * Wk)))
        ev = _smooth_skin(out, 100.0, rm)
        out += (np.float32(0.8 * amt["even_skin"]) * Z["skin"])[..., None] * (ev - out)
    # Lines are narrow dark grooves: a grey-level closing (Serra 1982; the same idea as DullRazor's hair removal,
    # Lee et al. 1997) fills every dark valley narrower than ~3% of the face and leaves anything wider, and
    # anything bright, alone, even right next to strong edges, where an edge-aware blur would keep the groove.
    # The finest layer (pores, the skin's grain) is taken out first with a tiny blur and never touched.
    from scipy.ndimage import grey_closing
    fine = gaussian_filter(L, max(0.6, 0.0025 * Wk)).astype(np.float32)
    nl = max(3, int(round(0.04 * Wk)) | 1)
    Dl_dark, Dl_bright = line_layers(fine, nl)
    # Folds and hollows are wide soft shadows: the same at about 12% of the face, on a smoother copy, so only the
    # broad shading is lifted (part way: the face keeps its shape)
    base2 = gaussian_filter(L, max(1.0, 0.008 * Wk)).astype(np.float32)
    nf = max(5, int(round(0.12 * Wk)) | 1)
    Df = np.minimum(base2 - _blur(grey_closing(base2, size=(nf, nf)), nf / 3.0), 0.0)
    g1 = fine
    g2 = _blur(L, max(1.0, 0.01 * Wk))
    dL = np.zeros_like(L)
    al, au, asm = amt.get("lines", 0), amt.get("under_eyes", 0), amt.get("smile_lines", 0)
    kl = np.maximum(al * Z["lines"], 0.7 * au * Z["under"])
    if kl.max() > 0:
        dL -= kl * (1.0 * Dl_dark + 0.2 * Dl_bright)
    if au > 0:
        dL -= np.float32(0.75 * au) * Z["under"] * np.minimum(Df - np.minimum(Dl_dark, 0.0) * 0.0, 0.0)
    if asm > 0:
        dL -= np.float32(0.6 * asm) * Z["smile"] * Df
    adc = amt.get("double_chin", 0)
    if adc > 0:
        # the crease under the chin: its shadow lifted part way (the warp in warps() does the shape)
        dL -= np.float32(0.7 * adc) * Z["chin"] * Df
        # the jaw's own shadow deepened a little, only where there already is one (darker than its surroundings),
        # so flat, frontal light never gets a painted-on line
        sd = np.clip(_blur(base2, max(2.0, 0.08 * Wk)) - base2, 0.0, 1.5)
        dL -= np.float32(0.2 * adc) * Z["jawshade"] * np.clip(sd / 0.6, 0.0, 1.0)
    dL = np.clip(dL, -0.3, 0.8)
    out *= np.exp2(dL)[..., None]
    lab2 = to_oklab(np.maximum(out, 0.0))
    changed = False
    if au > 0:                                       # the bluish / purple cast of dark circles
        Ll = np.maximum(lab2[..., 0], 1e-3)
        for ch in (1, 2):
            rat = lab2[..., ch] / Ll
            c2, c3 = _blur(rat, max(1.0, 0.015 * Wk)), _blur(rat, max(2.0, 0.05 * Wk))
            lab2[..., ch] = (rat - np.float32(0.7 * au) * Z["under"] * (c2 - c3)) * Ll
        changed = True
    ae, at = amt.get("eyes", 0), amt.get("teeth", 0)
    b_white = None
    if (ae > 0 or at > 0) and (Z["eye"] > 0.5).sum() > 12:
        sel = Z["eye"] > 0.5
        Le = L[sel]
        p50, p90 = float(np.percentile(Le, 50)), float(np.percentile(Le, 90))
        wwhite = np.clip((L - p50) / max(p90 - p50, 0.1), 0.0, 1.0) * Z["eye"]
        wsel = wwhite > 0.5
        if wsel.sum() > 4:
            b_white = float(np.median(lab2[..., 2][wsel] / np.maximum(lab2[..., 0][wsel], 1e-3)))
        if ae > 0:
            Ll = np.maximum(lab2[..., 0], 1e-3)
            if wsel.sum() > 4:
                a_ref = float(np.median(lab2[..., 1][wsel] / Ll[wsel]))
                ra = lab2[..., 1] / Ll
                lab2[..., 1] = (ra - np.float32(0.7 * ae) * wwhite * np.maximum(ra - a_ref, 0.0)) * Ll
            lab2[..., 0] *= np.exp2(np.float32(0.35 * ae) * wwhite / 3.0)       # OkLab L ~ cube root of light
            iris = (1.0 - wwhite / np.maximum(Z["eye"], 1e-3)) * Z["eye"]
            lab2[..., 0] *= np.exp2(np.float32(0.45 * ae) * iris * np.clip(g1 - g2, -0.3, 0.3) / 3.0)
            changed = True
    if at > 0 and (Z["teeth"] > 0.5).sum() > 12:
        sel = Z["teeth"] > 0.5
        Lt = L[sel]
        p40, p85 = float(np.percentile(Lt, 40)), float(np.percentile(Lt, 85))
        wt = np.clip((L - p40) / max(p85 - p40, 0.1), 0.0, 1.0) * Z["teeth"]
        Ll = np.maximum(lab2[..., 0], 1e-3)
        rb = lab2[..., 2] / Ll
        ref = b_white if b_white is not None else float(np.percentile(rb[sel], 20))
        lab2[..., 2] = (rb - np.float32(0.75 * at) * wt * np.maximum(rb - ref, 0.0)) * Ll
        lab2[..., 0] *= np.exp2(np.float32(0.2 * at) * wt / 3.0)
        changed = True
    from . import eyes as ey
    wide = None                                      # where an eye colour may move channels far (brown to blue)
    if face.irises and (ey.eye_active(None, amt) or colour is not None):
        M = ey.eye_masks(face.irises, X, Y, Z["eye"], lab2[..., 0], q,
                         {"r": face.P[fz.RIGHT_EYE], "l": face.P[fz.LEFT_EYE]})
        wz = (Z["eye"] > 0.5) & (M["disc"] < 0.2)
        white_L = float(np.percentile(lab2[..., 0][wz], 80)) if wz.sum() > 6 else None
        r_work = float(np.median([ir[3] for ir in face.irises])) / q
        if ey.apply_eyes(lab2, M, amt, colour, white_L, r_work):
            changed = True
        if colour is not None:
            wide = M["disc"]
    if changed:
        out = from_oklab(lab2)
    G = np.ones_like(out)
    ok = rgb > 1e-6
    out = np.maximum(out, 0.0)
    G[ok] = out[ok] / rgb[ok]
    if wide is None:
        G = np.clip(G, 0.3, 3.0)
    else:
        G = np.clip(G, (0.3 - 0.28 * wide)[..., None], (3.0 + 27.0 * wide)[..., None])
    t = np.minimum.reduce([np.arange(h)[:, None] + np.zeros((1, w)), (h - 1 - np.arange(h))[:, None] + np.zeros((1, w)),
                           np.arange(w)[None, :] + np.zeros((h, 1)), (w - 1 - np.arange(w))[None, :] + np.zeros((h, 1))])
    taper = np.clip(t / 4.0, 0.0, 1.0).astype(np.float32)[..., None]
    return (1.0 + taper * (G - 1.0)).astype(np.float32)


def paste_gain(base: np.ndarray, bgx: float, bgy: float, step: int, gain) -> None:
    """Multiply a block (pixel (0, 0) is the cell at S (bgx, bgy), ``step`` apart) by a face gain map
    (G, gx, gy, q)."""
    G, gx, gy, q = gain
    h, w = base.shape[:2]
    cb, cw = 0.5 * (step - 1), 0.5 * (q - 1)
    xs = bgx + step * np.arange(w) + cb
    ys = bgy + step * np.arange(h) + cb
    fx = (xs - gx - cw) / q
    fy = (ys - gy - cw) / q
    jx = np.nonzero((fx > -1) & (fx < G.shape[1]))[0]
    iy = np.nonzero((fy > -1) & (fy < G.shape[0]))[0]
    if jx.size == 0 or iy.size == 0:
        return
    sl = (slice(iy[0], iy[-1] + 1), slice(jx[0], jx[-1] + 1))
    FY, FX = np.meshgrid(fy[sl[0]], fx[sl[1]], indexing="ij")
    for ch in range(3):
        g = map_coordinates(G[..., ch], [FY, FX], order=1, mode="constant", cval=1.0)
        base[sl + (ch,)] *= g


# ------------------------------------------------------------------------------------------ blemishes
def find_blemishes(rgb: np.ndarray, gx: float, gy: float, step: int, face: Face, limit: int = 60):
    """Blemishes on one face, in a scene-linear block (pixel (0, 0) is the cell at S (gx, gy), ``step`` apart).
    Returns (blobs [{x, y, r, dark, red, score}] in S pixels, number of probable moles left alone)."""
    from ..color.grade import to_oklab
    h, w = rgb.shape[:2]
    z = zones(face)
    c = 0.5 * (step - 1)
    X = np.broadcast_to(gx + step * np.arange(w)[None, :] + c, (h, w))
    Y = np.broadcast_to(gy + step * np.arange(h)[:, None] + c, (h, w))
    ok_zone = sample_zone(z, "blem", X, Y)
    pos = np.maximum(rgb, 0.0)
    lab = to_oklab(pos)
    L = np.log2(gaussian_filter(np.maximum(pos @ LUMA, 0.0), 0.7) + 1e-4).astype(np.float32)
    A = gaussian_filter(lab[..., 1] / np.maximum(lab[..., 0], 1e-3), 0.7).astype(np.float32)
    gate = _skin_gate(lab, L, sample_zone(z, "cheek", X, Y))
    ok = (ok_zone > 0.5) & (gate > 0.25)
    if ok.sum() < 50:
        return [], 0
    Wk = face.W / step
    sig = max(0.8, 0.003 * Wk)
    cands = []
    from .dust import _box_sum, _integral
    IL = _integral(L)
    while sig <= 0.02 * Wk:
        G1d, G1r = gaussian_filter(L, sig), gaussian_filter(A, sig)
        Dd = G1d - gaussian_filter(L, 2.5 * sig)
        Dr = G1r - gaussian_filter(A, 2.5 * sig)
        td = 1.4826 * float(np.median(np.abs(Dd[ok] - np.median(Dd[ok])))) + 1e-4
        tr = 1.4826 * float(np.median(np.abs(Dr[ok] - np.median(Dr[ok])))) + 1e-5
        dark, red = -Dd, Dr
        sd, sr = dark / (5.0 * td) * (dark > 0.05), red / (5.0 * tr) * (red > 0.012)
        score = np.maximum(sd, sr)
        score[~ok] = 0
        size = max(3, int(2 * sig + 1) | 1)
        peak = (score >= maximum_filter(score, size=size)) & (score > 1.0)
        iy, ix = np.nonzero(peak)
        if iy.size > 3000:
            o = np.argsort(-score[iy, ix])[:3000]
            iy, ix = iy[o], ix[o]
        if iy.size:
            # the skin right around each candidate must be quiet (a ring): hair, lashes and creases are not
            ro, ri = max(2, int(round(4.0 * sig))), max(1, int(round(1.8 * sig)))
            sums = {}
            for nm, D in (("d", Dd), ("r", Dr)):
                I2 = _integral(D * D)
                so, ao = _box_sum(I2, iy, ix, ro)
                si, ai = _box_sum(I2, iy, ix, ri)
                sums[nm] = np.sqrt(np.maximum((so - si) / np.maximum(ao - ai, 1.0), 0.0))
        for n, (y, x) in enumerate(zip(iy, ix)):
            use_red = sr[y, x] > sd[y, x]
            resp = red[y, x] if use_red else dark[y, x]
            ring = sums["r"][n] if use_red else sums["d"][n]
            if resp < 3.0 * ring:
                continue
            G = G1r if use_red else G1d
            g = G[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            if g.shape != (3, 3):
                continue
            a = g[1, 2] + g[1, 0] - 2 * g[1, 1]               # roundness: both curvatures the same sign, similar
            cc = g[2, 1] + g[0, 1] - 2 * g[1, 1]
            bxy = 0.25 * (g[2, 2] + g[0, 0] - g[0, 2] - g[2, 0])
            if use_red:
                a, cc, bxy = -a, -cc, -bxy
            tr_, det = a + cc, a * cc - bxy * bxy
            disc = math.sqrt(max(tr_ * tr_ / 4 - det, 0.0))
            l1, l2 = tr_ / 2 + disc, tr_ / 2 - disc
            if not (l2 > 0 and l2 > 0.35 * l1):
                continue
            # how much darker the middle is than the skin around it (a mole is dark through and through)
            r_in, r_out = max(1, int(round(2.5 * sig))), max(2, int(round(4.0 * sig)))
            so_, ao_ = _box_sum(IL, np.array([y]), np.array([x]), r_out)
            si_, ai_ = _box_sum(IL, np.array([y]), np.array([x]), r_in)
            ring_mean = float((so_ - si_)[0] / max((ao_ - ai_)[0], 1.0))
            depth = ring_mean - float(L[y, x])
            cands.append((float(score[y, x]), int(y), int(x), sig, depth, float(red[y, x]),
                          float(red[y, x] / tr)))
        sig *= 1.35
    cands.sort(key=lambda t: -t[0])
    kept, moles = [], 0
    for s, y, x, sg, dk, rd, rz in cands:
        rho = 1.5 * sg
        if any(math.hypot(x - k[2], y - k[1]) < 1.2 * max(rho, 1.5 * k[3]) for k in kept):
            continue
        if dk > 0.75 and (rz < 2.5 or rd < 0.008):     # very dark and not red: a mole or beauty mark, keep it
            moles += 1
            kept.append((s, y, x, sg, dk, rd, rz, True))
            continue
        kept.append((s, y, x, sg, dk, rd, rz, False))
        if sum(1 for k in kept if not k[7]) >= limit:
            break
    blobs = [{"x": float(gx + step * x + c), "y": float(gy + step * y + c), "r": float(1.5 * sg * step),
              "dark": dk, "red": rd, "score": s} for s, y, x, sg, dk, rd, rz, mole in kept if not mole]
    return blobs, moles


def feature_avoid(face: Face) -> list:
    """Eyes, brows, lips and nostrils as circles (S pixels) that a heal source must not take from."""
    out = []
    for idx in (fz.RIGHT_EYE, fz.LEFT_EYE, fz.RIGHT_BROW, fz.LEFT_BROW, fz.LIPS_OUTER):
        p = face.P[idx]
        c = p.mean(axis=0)
        out.append((float(c[0]), float(c[1]), float(np.max(np.hypot(*(p - c).T))) * 1.1))
    for a, b in fz.NOSTRILS:
        c = 0.5 * (face.P[a] + face.P[b])
        out.append((float(c[0]), float(c[1]), float(np.hypot(*(face.P[a] - face.P[b])))))
    return out


# ------------------------------------------------------------------------------------------ body sculpting
def _bump(t, T: float = 1.0):
    u = np.clip(1.0 - (t / T) ** 2, 0.0, None)
    return u * u


def warps(p: dict | None, sw: int, sh: int, skin: list | None = None) -> list:
    """Every warp of the edit, in order, as primitives in S pixels: jawline and double chin first (attached to the
    face), then the sculpting strokes and shapes. ``skin``: per face (in faces_of order), the map from chin_skin()
    that keeps the double-chin warp on skin; without it the warp moves whatever is under the jaw."""
    p = p or {}
    long = float(max(sw, sh))
    out = []
    jaw = amounts(p).get("jawline", 0.0)
    if jaw > 1e-4:
        for f in faces_of(p):
            fc = Face(f, sw, sh)
            centre = fc.P[fz.NOSE_TIP]
            for side in (fz.JAW_R, fz.JAW_L):
                for i in side[1:5]:                      # the jowl part of the jaw line
                    pt = fc.P[i]
                    v = centre - pt
                    v = v / (np.hypot(*v) + 1e-9)
                    d = (0.75 * v + 0.45 * fc.up)
                    d = d / (np.hypot(*d) + 1e-9) * 0.022 * fc.W * jaw
                    out.append(("push", float(pt[0]), float(pt[1]), 0.12 * fc.W, float(d[0]), float(d[1])))
    dc = amounts(p).get("double_chin", 0.0)
    if dc > 1e-4:
        for i, f in enumerate(faces_of(p)):
            tk = _tuck(Face(f, sw, sh), dc, skin[i] if skin and i < len(skin) else None)
            if tk is not None:
                out.append(tk)
    for s in p.get("sculpt") or []:
        if not s.get("enabled", True):
            continue
        k = s.get("kind")
        try:
            if k == "push":
                R = max(float(s.get("r", 0.05)) * long, 2.0)
                for (x, y), (dx, dy) in zip(s.get("pts") or [], s.get("vec") or []):
                    D = np.array([float(dx) * sw, float(dy) * sh])
                    n = float(np.hypot(*D))
                    if n > 0.3 * R:
                        D *= 0.3 * R / n
                    if n > 1e-3:
                        out.append(("push", float(x) * sw, float(y) * sh, R, float(D[0]), float(D[1])))
            elif k in ("shrink", "grow"):
                R = max(float(s.get("r", 0.05)) * long, 2.0)
                a = min(max(float(s.get("amount", 0.5)), 0.0), 1.0) * 0.12
                a = a if k == "shrink" else -a
                for x, y in s.get("pts") or []:
                    out.append(("scale", float(x) * sw, float(y) * sh, R, a))
            elif k == "shape" and s.get("tool") in SHAPE_TOOLS:
                vals = [float(s.get(v, 0)) for v in ("lift", "size", "slim")]
                if not any(abs(v) > 1e-3 for v in vals):
                    continue
                e1 = np.asarray(s.get("e1", [1, 0]), float)
                e1 = e1 / (np.hypot(*e1) + 1e-12)
                up = np.asarray(s.get("up", [0, -1]), float)
                up = up / (np.hypot(*up) + 1e-12)
                c = s.get("c", [0.5, 0.5])
                out.append(("shape", s["tool"], float(c[0]) * sw, float(c[1]) * sh, e1, float(s.get("a", 0.05)) * long,
                            float(s.get("b", 0.04)) * long, up) + tuple(np.clip(vals, -100, 100) / 100.0))
        except (TypeError, ValueError, KeyError):
            continue
    return out


def chin_skin(rgb: np.ndarray, gx: float, gy: float, q: float, face: Face) -> np.ndarray:
    """How much of each spot round one face is skin like its cheeks (0..1), on the face copy's grid, smooth enough
    to weight a warp without tearing it. The double-chin warp is multiplied by it: skin under
    the jaw moves, a collar, a scarf or a beard right under the chin stays where it is."""
    from ..color.grade import to_oklab
    z = zones(face)
    h, w = rgb.shape[:2]
    c = 0.5 * (q - 1)
    X = np.broadcast_to(gx + q * np.arange(w)[None, :] + c, (h, w))
    Y = np.broadcast_to(gy + q * np.arange(h)[:, None] + c, (h, w))
    pos = np.maximum(rgb, 0.0)
    L = np.log2(np.maximum(pos @ LUMA, 1e-6)).astype(np.float32)
    g = _skin_gate(to_oklab(pos), L, sample_zone(z, "cheek", X, Y), 0.07, 1.2)
    # pulled in from its edges first (4% of the face), so the weight is nearly 0 where skin meets a collar
    # and the collar's edge stays put; then smoothed (6%), so the warp's slope stays well under one
    from scipy.ndimage import minimum_filter
    g = minimum_filter(g, size=max(3, int(round(0.08 * face.W / q)) | 1))
    return gaussian_filter(g, 0.06 * face.W / q).astype(np.float32)


def _tuck(fc: Face, amount: float, skin=None):
    """The double-chin warp of one face: the soft area under the jaw drawn up towards it. Nothing moves at the jaw
    line itself or above it (the face keeps its length and its jaw); the most movement, 4.5% of the face width at
    100, is 16% of the face below the jaw, where the second chin's lower edge sits, and it fades out by 32%. The
    profile (sin squared) never folds (steepest slope under half a pixel per pixel), and ``skin`` (chin_skin:
    smooth, so the sum stays well under one) keeps it on skin."""
    e1 = (fc.el - fc.er) / fc.iod
    up = fc.up
    c = fc.P[fz.CHIN]
    J = fc.P[JAW_LINE] - c
    ja, jv = J @ e1, J @ up
    o = np.argsort(ja)
    ja, jv = ja[o], jv[o]
    A = 0.95 * float(min(-ja[0], ja[-1]))
    if A < 2.0:
        return None
    return ("tuck", float(c[0]), float(c[1]), e1.astype(float), up.astype(float), ja, jv, A, 0.32 * fc.W,
            0.045 * fc.W * float(amount), skin)


def _disp(w, X, Y):
    """Forward displacement (dx, dy) of one warp at S positions."""
    if w[0] == "tuck":
        _, cx, cy, e1, up, ja, jv, A, S, m, sk = w
        DX, DY = X - cx, Y - cy
        a = DX * e1[0] + DY * e1[1]
        s = np.interp(a, ja, jv) - (DX * up[0] + DY * up[1])     # how far below the jaw line
        x = np.clip(s / S, 0.0, 1.0)
        t = np.clip((A - np.abs(a)) / (0.35 * A), 0.0, 1.0)
        f = m * (t * t * (3.0 - 2.0 * t)) * np.sin(np.pi * x) ** 2
        if sk is not None:
            M, kx, ky, q = sk
            c0 = 0.5 * (q - 1)
            f = f * map_coordinates(M, [(np.asarray(Y) - ky - c0) / q, (np.asarray(X) - kx - c0) / q], order=1,
                                    mode="nearest")
        return f * up[0], f * up[1]
    if w[0] == "push":
        _, cx, cy, R, dx, dy = w
        f = _bump(np.hypot(X - cx, Y - cy) / R)
        return dx * f, dy * f
    if w[0] == "scale":
        _, cx, cy, R, a = w
        f = _bump(np.hypot(X - cx, Y - cy) / R)
        return -a * (X - cx) * f, -a * (Y - cy) * f
    _, tool, cx, cy, e1, A, B, up, lift, size, slim = w
    e2 = np.array([-e1[1], e1[0]])
    DX, DY = X - cx, Y - cy
    a = DX * e1[0] + DY * e1[1]
    b = DX * e2[0] + DY * e2[1]
    dx = np.zeros_like(DX)
    dy = np.zeros_like(DY)
    if tool in LIFT_TOOLS:
        R = max(A, B)
        t = np.sqrt((a / A) ** 2 + (b / B) ** 2)
        f = _bump(t, 1.6)
        if lift:
            v = (DX * up[0] + DY * up[1]) / R
            wl = np.clip(0.6 - 0.4 * v, 0.2, 1.0)       # the lower part rises more than the top
            m = lift * 0.14 * R * f * wl
            dx, dy = dx + m * up[0], dy + m * up[1]
        if size:
            dx, dy = dx + size * 0.10 * DX * f, dy + size * 0.10 * DY * f
        if slim:
            m = -slim * 0.10 * b * f
            dx, dy = dx + m * e2[0], dy + m * e2[1]
    else:
        g = _bump(a / A)
        hh = _bump(b / B, 2.0)
        m = -(slim - 0.5 * size) * 0.12 * b * g * hh
        dx, dy = dx + m * e2[0], dy + m * e2[1]
        if lift:
            m = lift * 0.08 * A * g * hh
            dx, dy = dx + m * up[0], dy + m * up[1]
    return dx, dy


def _reach(w) -> tuple:
    """(x0, y0, x1, y1) S box a warp can change, and how far it moves anything (pixels)."""
    if w[0] == "tuck":
        _, cx, cy, e1, up, ja, jv, A, S, m = w[:10]
        corners = [(a, v) for a in (-A, A) for v in (float(jv.min()) - S, float(jv.max()))]
        xs = [cx + a * e1[0] + v * up[0] for a, v in corners]
        ys = [cy + a * e1[1] + v * up[1] for a, v in corners]
        return (min(xs), min(ys), max(xs), max(ys)), m
    if w[0] in ("push", "scale"):
        cx, cy, R = w[1], w[2], w[3]
        mv = math.hypot(w[4], w[5]) if w[0] == "push" else abs(w[4]) * R * 0.4
        return (cx - R, cy - R, cx + R, cy + R), mv
    _, tool, cx, cy, e1, A, B, up, lift, size, slim = w
    if tool in LIFT_TOOLS:
        r = 1.6 * max(A, B)
        mv = (abs(lift) * 0.14 + abs(size) * 0.07 + abs(slim) * 0.07) * max(A, B) * 1.6
    else:
        r = math.hypot(A, 2.0 * B)
        mv = (abs(slim) + abs(size)) * 0.12 * B * 1.2 + abs(lift) * 0.08 * A
    return (cx - r, cy - r, cx + r, cy + r), mv


def margin(ws: list) -> float:
    """The farthest any warp moves a pixel (S pixels): how much context a region render needs."""
    return max([_reach(w)[1] for w in ws] + [0.0])


def source_of(ws: list, X, Y, iters: int = 4):
    """Where each output position takes its pixel from: every warp inverted, last first."""
    X = np.asarray(X, float).copy()
    Y = np.asarray(Y, float).copy()
    for w in reversed(ws):
        (x0, y0, x1, y1), mv = _reach(w)
        sel = (X > x0 - mv) & (X < x1 + mv) & (Y > y0 - mv) & (Y < y1 + mv)
        if not sel.any():
            continue
        px, py = X[sel], Y[sel]
        sx, sy = px.copy(), py.copy()
        for _ in range(iters):                         # solve s + d(s) = p
            dx, dy = _disp(w, sx, sy)
            sx, sy = px - dx, py - dy
        X[sel], Y[sel] = sx, sy
    return X, Y


def _cubic_sample(img: np.ndarray, fy: np.ndarray, fx: np.ndarray) -> np.ndarray:
    """Catmull-Rom sampling (local, so a block and the whole photo give the same pixels), edges clamped."""
    h, w = img.shape[:2]
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    ty, tx = (fy - y0)[..., None], (fx - x0)[..., None]

    def wts(t):
        t2, t3 = t * t, t * t * t
        return (-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2)
    wy, wx = wts(ty), wts(tx)
    out = np.zeros(fy.shape + (img.shape[2],), np.float32)
    for i in range(4):
        yi = np.clip(y0 + i - 1, 0, h - 1)
        row = np.zeros_like(out)
        for j in range(4):
            xi = np.clip(x0 + j - 1, 0, w - 1)
            row += wx[j] * img[yi, xi]
        out += wy[i] * row
    return out


def apply_warps(base: np.ndarray, bgx: float, bgy: float, step: int, ws: list, cubic: bool = True) -> np.ndarray:
    """The block after the warps (pixel (0, 0) is the cell at S (bgx, bgy), ``step`` apart). Only the parts the warps
    reach are resampled."""
    if not ws:
        return base
    h, w = base.shape[:2]
    c = 0.5 * (step - 1)
    xs = bgx + step * np.arange(w) + c
    ys = bgy + step * np.arange(h) + c
    boxes = [_reach(wp) for wp in ws]
    x0 = min(b[0][0] - b[1] for b in boxes)
    y0 = min(b[0][1] - b[1] for b in boxes)
    x1 = max(b[0][2] + b[1] for b in boxes)
    y1 = max(b[0][3] + b[1] for b in boxes)
    jx = np.nonzero((xs >= x0) & (xs <= x1))[0]
    iy = np.nonzero((ys >= y0) & (ys <= y1))[0]
    if jx.size == 0 or iy.size == 0:
        return base
    out = base.copy()
    for a in range(iy[0], iy[-1] + 1, 256):
        b = min(iy[-1] + 1, a + 256)
        X, Y = np.meshgrid(xs[jx[0]:jx[-1] + 1], ys[a:b])
        SX, SY = source_of(ws, X, Y)
        moved = (np.abs(SX - X) > 0.01 * step) | (np.abs(SY - Y) > 0.01 * step)
        if not moved.any():
            continue
        fx = (SX[moved] - bgx - c) / step
        fy = (SY[moved] - bgy - c) / step
        if cubic:
            vals = _cubic_sample(base, fy, fx)
        else:
            vals = np.stack([map_coordinates(base[..., k], [fy, fx], order=1, mode="nearest") for k in range(3)], -1)
        blk = out[a:b, jx[0]:jx[-1] + 1]
        blk[moved] = vals
    return out
