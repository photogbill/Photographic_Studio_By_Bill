"""Phase 9 (first part): local adjustments with masks (plan section 9.5). No neural networks.

A mask is stored in the edit as plain data and computed at render time, so it is exact at any size and a 100%
region matches the full render. Shapes are attached to the photo: their coordinates are fractions of the camera-crop
frame before rotation (the "S frame"), so straightening or cropping later does not move them off their subject.

    {"name": "Mask 1", "enabled": true, "invert": false, "opacity": 100,
     "shapes": [{"type": "radial", "op": "add", ...}, ...],
     "adjust": {"exposure": 0.5, "saturation": -20, ...}}

Shape types:
  linear      x0, y0 (full effect) -> x1, y1 (no effect)
  radial      cx, cy, rx, ry (fractions of the long edge), angle (degrees), feather 0..1
  brush       strokes: [{"pts": [[x, y], ...], "size": radius as a fraction of the long edge, "feather": 0..1,
              "flow": 0..1, "erase": bool, "ref": [L, a, b] or null}], auto: stay on the colour under the brush
  luminance   lo, hi, soft (screen brightness 0..1)
  colour      hue (OKLCh degrees), width, soft (degrees), chroma (minimum colourfulness 0..1)
  object      w, h, data (zlib + base64 bitmap of the S frame at w x h), fg / bg seed points: "click an object",
              GrabCut-style colour models on a small render, refined with a guided filter
  skin        skin tones (OKLCh hue, colourfulness and lightness band around typical skin), refined to edges
  person      w, h, data (person probability, as for object), people (33 body points each, S fractions + how
              visible), faces (face outlines from Find faces), part: person | background | skin | face | hair |
              clothes | arms | legs | torso (local/people.py), refined to edges
Shapes combine in order: add (union), subtract, intersect.

Adjustments (each -100..100 unless noted): exposure (EV, -4..4), contrast, highlights, shadows, whites, blacks,
temperature, tint, saturation, vibrance, texture, clarity, dehaze, sharpness, noise (0..100), smooth_skin (0..100,
with skin_size -100..100 for the size of what counts as a blemish).

Smooth skin follows what the retouching literature found works: split the skin into layers by size with
edge-aware (guided) filters, soften only the middle layer (blotches, uneven tone, small wrinkles) and keep the
finest layer (pores, fine hair), because plain blurring makes skin look plastic (AutoRetouch, Shafaei, Little and
Schmidt, WACV 2021; region-aware beautification, Liang et al. 2014). Uneven colour (redness) is evened out in the
same middle layer. It is a mask adjustment, so it stays where the skin is (use the Skin mask, alone or combined
with a brush); inside the mask it also eases off on colours far from skin (eyes, brows, lips).
"""
from __future__ import annotations

import base64
import math
import zlib

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from ... import log as slog

log = slog.get("engine.masks")

LUMA = np.array([0.2880, 0.7119, 0.0001], np.float32)
SCENE_KEYS = ("exposure", "temperature", "tint", "texture", "clarity", "dehaze", "smooth_skin")
TONE_KEYS = ("contrast", "highlights", "shadows", "whites", "blacks")
COLOUR_KEYS = ("saturation", "vibrance")
DETAIL_KEYS = ("sharpness", "noise")
ALL_KEYS = SCENE_KEYS + TONE_KEYS + COLOUR_KEYS + DETAIL_KEYS


def active(masks) -> list:
    """The masks that change something."""
    out = []
    for m in masks or []:
        if not m.get("enabled", True) or not m.get("shapes") or float(m.get("opacity", 100)) <= 0:
            continue
        adj = m.get("adjust") or {}
        if any(abs(float(adj.get(k, 0))) > 1e-6 for k in ALL_KEYS):
            out.append(m)
    return out


def uses(masks, keys) -> bool:
    return any(abs(float((m.get("adjust") or {}).get(k, 0))) > 1e-6 for m in active(masks) for k in keys)


# ------------------------------------------------------------------------------------------ bitmaps
def encode_bitmap(a: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.asarray(a, np.uint8).tobytes(), 9)).decode("ascii")


def decode_bitmap(data: str, w: int, h: int) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(data)), np.uint8).reshape(h, w)


# ------------------------------------------------------------------------------------------ the pixels' context
class Ctx:
    """What shapes need to know about the pixels of one render: S-frame positions (pixels), the S frame size, the
    render scale, scene-linear RGB (after the global scene stages) and the global tone curve for screen brightness."""

    def __init__(self, rgb, xs, ys, sw, sh, scale, lut_fn):
        self.rgb, self.xs, self.ys, self.sw, self.sh, self.scale = rgb, xs, ys, float(sw), float(sh), float(scale)
        self.long = float(max(sw, sh))
        self._lut = lut_fn
        self._ld = self._lab = self._guide = None

    @property
    def lum_display(self) -> np.ndarray:
        """Screen brightness 0..1 (sRGB-like encoding of the global tone curve applied to luminance)."""
        if self._ld is None:
            y = np.maximum(self.rgb @ LUMA, 0.0)
            d = np.clip(self._lut(y), 0.0, 1.0)
            self._ld = np.where(d <= 0.0031308, 12.92 * d, 1.055 * np.power(d, 1 / 2.4) - 0.055).astype(np.float32)
        return self._ld

    @property
    def lab(self) -> np.ndarray:
        if self._lab is None:
            from ..color.grade import to_oklab
            self._lab = to_oklab(np.maximum(self.rgb, 0.0))
        return self._lab

    def features(self) -> np.ndarray:
        """(screen brightness, a / L, b / L): what "the same colour" means for auto-masking (exposure-proof)."""
        lab = self.lab
        L = np.maximum(lab[..., 0], 1e-3)
        return np.stack([self.lum_display, lab[..., 1] / L, lab[..., 2] / L], -1).astype(np.float32)

    @property
    def guide(self) -> np.ndarray:
        if self._guide is None:
            self._guide = np.log2(np.maximum(self.rgb @ LUMA, 1e-6)).astype(np.float32)
        return self._guide


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------------------------------ shapes
def shape_alpha(sh: dict, c: Ctx) -> np.ndarray:
    t = sh.get("type")
    if t == "linear":
        x0, y0 = float(sh["x0"]) * c.sw, float(sh["y0"]) * c.sh
        x1, y1 = float(sh["x1"]) * c.sw, float(sh["y1"]) * c.sh
        dx, dy = x1 - x0, y1 - y0
        n = dx * dx + dy * dy
        if n < 1e-9:
            return np.zeros(c.xs.shape, np.float32)
        p = ((c.xs - x0) * dx + (c.ys - y0) * dy) / n
        return (1.0 - _smooth(p)).astype(np.float32)
    if t == "radial":
        cx, cy = float(sh["cx"]) * c.sw, float(sh["cy"]) * c.sh
        rx, ry = max(float(sh["rx"]) * c.long, 1.0), max(float(sh["ry"]) * c.long, 1.0)
        a = math.radians(float(sh.get("angle", 0.0)))
        u = (c.xs - cx) * math.cos(a) + (c.ys - cy) * math.sin(a)
        v = -(c.xs - cx) * math.sin(a) + (c.ys - cy) * math.cos(a)
        d = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
        f = min(max(float(sh.get("feather", 0.5)), 0.01), 1.0)
        return (1.0 - _smooth((d - (1.0 - f)) / f)).astype(np.float32)
    if t == "brush":
        return _brush(sh, c)
    if t == "luminance":
        L = c.lum_display
        lo, hi, s = float(sh.get("lo", 0.0)), float(sh.get("hi", 1.0)), max(float(sh.get("soft", 0.1)), 1e-3)
        return (_smooth((L - (lo - s)) / s) * (1.0 - _smooth((L - hi) / s))).astype(np.float32)
    if t == "colour":
        from ..color.grade import range_weight
        lab = c.lab
        C = np.hypot(lab[..., 1], lab[..., 2])
        h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0
        w = range_weight(h, float(sh.get("hue", 0)), float(sh.get("width", 30)), float(sh.get("soft", 20)))
        cmin = float(sh.get("chroma", 0.03))
        return (w * _smooth((C - 0.5 * cmin) / max(cmin, 1e-3))).astype(np.float32)
    if t == "skin":
        return refine(skin_weight(c.lab, float(sh.get("hue", SKIN_HUE)), float(sh.get("width", 40.0))), c, 4.0)
    if t == "person":
        from .people import part_alpha
        return part_alpha(sh, c)
    if t == "object":
        bm = decode_bitmap(sh["data"], int(sh["w"]), int(sh["h"])).astype(np.float32) / 255.0
        fy = c.ys / c.sh * bm.shape[0] - 0.5
        fx = c.xs / c.sw * bm.shape[1] - 0.5
        a = map_coordinates(bm, [fy, fx], order=1, mode="nearest").astype(np.float32)
        return refine(a, c)
    raise ValueError(f"unknown mask shape {t!r}")


def refine(a: np.ndarray, c: Ctx, r_full: float = 6.0) -> np.ndarray:
    """Snap a soft mask to the photo's edges (guided filter on log luminance)."""
    from ..detail.finish import guided
    r = max(1, int(round(r_full * c.scale)))
    return np.clip(guided(c.guide, a.astype(np.float32), r, 0.02), 0.0, 1.0).astype(np.float32)


def _seg_dist(xs, ys, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    n = dx * dx + dy * dy
    if n < 1e-9:
        return np.hypot(xs - ax, ys - ay)
    t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / n, 0.0, 1.0)
    return np.hypot(xs - (ax + t * dx), ys - (ay + t * dy))


def _brush(sh: dict, c: Ctx) -> np.ndarray:
    H, W = c.xs.shape
    out = np.zeros((H, W), np.float32)
    x_axis, y_axis = c.xs[0, :], c.ys[:, 0]
    regular = np.ndim(c.xs) == 2 and np.allclose(c.xs, x_axis[None, :]) and np.allclose(c.ys, y_axis[:, None])
    feats = c.features() if sh.get("auto") else None
    for st in sh.get("strokes", []):
        pts = np.asarray(st.get("pts", []), float)
        if pts.size == 0:
            continue
        px, py = pts[:, 0] * c.sw, pts[:, 1] * c.sh
        R = max(float(st.get("size", 0.03)) * c.long, 1.0)
        f = min(max(float(st.get("feather", 0.5)), 0.01), 1.0)
        flow = min(max(float(st.get("flow", 1.0)), 0.0), 1.0)
        bx0, bx1, by0, by1 = px.min() - R, px.max() + R, py.min() - R, py.max() + R
        if regular:
            jx = np.nonzero((x_axis >= bx0) & (x_axis <= bx1))[0]
            iy = np.nonzero((y_axis >= by0) & (y_axis <= by1))[0]
            if jx.size == 0 or iy.size == 0:
                continue
            sl = (slice(iy[0], iy[-1] + 1), slice(jx[0], jx[-1] + 1))
        else:
            sl = (slice(None), slice(None))
        X, Y = c.xs[sl], c.ys[sl]
        d = np.full(X.shape, np.inf)
        if len(px) == 1:
            d = np.hypot(X - px[0], Y - py[0])
        for k in range(len(px) - 1):
            d = np.minimum(d, _seg_dist(X, Y, px[k], py[k], px[k + 1], py[k + 1]))
        a = (1.0 - _smooth((d / R - (1.0 - f)) / f)) * flow
        ref = st.get("ref")
        if feats is not None and ref is not None:
            F = feats[sl]
            dl = (F[..., 0] - float(ref[0])) / 0.12
            da = (F[..., 1] - float(ref[1])) / 0.15
            db = (F[..., 2] - float(ref[2])) / 0.15
            a = a * np.exp(-0.5 * (dl * dl + da * da + db * db))
        a = a.astype(np.float32)
        if st.get("erase"):
            out[sl] *= 1.0 - a
        else:
            out[sl] = np.maximum(out[sl], a)
    if sh.get("auto"):
        out = refine(out, c, 3.0)
    return out


def mask_alpha(m: dict, c: Ctx) -> np.ndarray:
    a = None
    for sh in m.get("shapes", []):
        s = shape_alpha(sh, c)
        if a is None:
            a = s
            continue
        op = sh.get("op", "add")
        if op == "subtract":
            a = a * (1.0 - s)
        elif op == "intersect":
            a = a * s
        else:
            a = a + s - a * s
    if a is None:
        a = np.zeros(c.xs.shape, np.float32)
    if m.get("invert"):
        a = 1.0 - a
    return np.clip(a * float(m.get("opacity", 100)) / 100.0, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------------------------------ skin
SKIN_HUE = 55.0                     # OKLCh degrees (grade.SKIN_HUE): skin of every complexion sits near here


def skin_weight(lab: np.ndarray, hue: float = SKIN_HUE, width: float = 40.0) -> np.ndarray:
    """How skin-like each pixel's colour is (0..1): hue near skin, moderately colourful, not black or white."""
    from ..color.grade import range_weight
    L = lab[..., 0]
    C = np.hypot(lab[..., 1], lab[..., 2])
    h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0
    w = range_weight(h, hue, width, 25.0)
    w = w * _smooth((C - 0.012) / 0.02) * (1.0 - _smooth((C - 0.20) / 0.08))
    w = w * _smooth((L - 0.15) / 0.12) * (1.0 - _smooth((L - 0.95) / 0.04))
    return w.astype(np.float32)


def skin_radius(scale: float, long_full: float, size: float = 0.0) -> int:
    """Blemish scale (box radius, this render's pixels): 0.6% of the long edge at size 0, x2 per +50. A box this
    wide takes out blotches up to about a third of it across (a head-and-shoulders portrait's spots and redness)."""
    return max(2, int(round(0.006 * long_full * 2.0 ** (float(size) / 50.0) * scale)))


SKIN_STRIP = 384                    # rows per strip (memory: a 45 MP export stays a few hundred MB per strip)


def smooth_skin(rgb: np.ndarray, amount: float, scale: float = 1.0, long_full: float = 6000.0,
                size: float = 0.0) -> np.ndarray:
    """amount 0..100. Softens blotches, uneven tone and colour while keeping pores and edges (scene-linear).
    Worked in overlapping strips, each with enough rows around it that the result is the same as in one piece."""
    if amount <= 0:
        return rgb
    rm = skin_radius(scale, long_full, size)
    pad = 2 * rm + 4                                # two box filters of radius rm reach this far
    h = rgb.shape[0]
    if h <= SKIN_STRIP + 2 * pad:
        return _smooth_skin(rgb, amount, rm)
    from ..parallel import strips, workers
    out = np.empty_like(rgb, dtype=np.float32)

    def job(y0, y1):
        a0, a1 = max(0, y0 - pad), min(h, y1 + pad)
        out[y0:y1] = _smooth_skin(rgb[a0:a1], amount, rm)[y0 - a0:y1 - a0]
    strips(h, SKIN_STRIP, job, min(4, workers()))         # at most 4 strips at once: each needs ~0.5 GB at 45 MP
    return out


class _GuideStats:
    """Box statistics of the guide at one radius, shared by every guided filter that uses it (He, Sun and Tang's
    guided filter, with the guide's mean and variance computed once instead of once per filtered channel)."""

    def __init__(self, I: np.ndarray, r: int, eps: float):
        from scipy.ndimage import uniform_filter
        self.uf = lambda x: uniform_filter(x, 2 * r + 1, mode="reflect")
        self.I = I
        self.eps = np.float32(eps)
        self.mI = self.uf(I)
        self.den = np.maximum(self.uf(I * I) - self.mI * self.mI, 0.0) + self.eps

    def apply(self, p: np.ndarray | None = None) -> np.ndarray:
        """The guided filter of p (None = the guide itself)."""
        if p is None:
            a = 1.0 - self.eps / self.den                  # cov = var when p is the guide
            b = self.mI * (1.0 - a)
        else:
            mp = self.uf(p)
            a = (self.uf(self.I * p) - self.mI * mp) / self.den
            b = mp - a * self.mI
        return (self.uf(a.astype(np.float32)) * self.I + self.uf(b.astype(np.float32))).astype(np.float32)


def _smooth_skin(rgb: np.ndarray, amount: float, rm: int) -> np.ndarray:
    from ..color.grade import from_oklab, to_oklab
    s = np.float32(min(amount, 100.0) / 100.0)
    rf = max(1, int(round(rm / 8.0)))
    Y = np.maximum(rgb @ LUMA, 1e-7)
    L = np.log2(Y).astype(np.float32)
    fine = _GuideStats(L, rf, 0.10 ** 2)            # pores and fine hair out, edges in
    mid = _GuideStats(L, rm, 0.20 ** 2)             # blotches out too; eyes, brows, lips stay (strong edges)
    A, B = fine.apply(), mid.apply()
    lab = to_oklab(np.maximum(rgb, 0.0))
    gate = 0.3 + 0.7 * skin_weight(lab)             # ease off on colours far from skin
    del lab
    k = s * gate
    out = rgb * np.exp2(-k * (A - B))[..., None]
    del A, B
    lab2 = to_oklab(np.maximum(out, 0.0))
    for ch in (1, 2):                               # even out blotchy colour in the same middle layer
        c = np.ascontiguousarray(lab2[..., ch])
        lab2[..., ch] = c - 0.8 * k * (fine.apply(c) - mid.apply(c))
    res = from_oklab(lab2)
    return np.where(rgb < 0, rgb, res).astype(np.float32)        # out-of-gamut negatives stay as they were


# ------------------------------------------------------------------------------------------ adjustments
def apply_scene(rgb: np.ndarray, m: dict, a: np.ndarray, c: Ctx, airlight=None) -> np.ndarray:
    """Exposure, white balance, Texture, Clarity and Dehaze inside the mask (scene-linear)."""
    adj = m.get("adjust") or {}
    A = a[..., None]
    ev = float(adj.get("exposure", 0))
    t, ti = float(adj.get("temperature", 0)) / 100.0, float(adj.get("tint", 0)) / 100.0
    if ev or t or ti:
        g = np.ones(a.shape + (3,), np.float32)
        if ev:
            g *= np.exp2(np.float32(ev) * a)[..., None]
        if t or ti:
            g[..., 0] *= np.exp2(np.float32(0.35 * t + 0.1 * ti) * a)
            g[..., 1] *= np.exp2(np.float32(-0.25 * ti) * a)
            g[..., 2] *= np.exp2(np.float32(-0.35 * t + 0.1 * ti) * a)
        rgb = rgb * g
    from ..detail import finish as F
    v = float(adj.get("smooth_skin", 0))
    if v > 0:                       # only where the mask is (a brushed face is a small part of the photo)
        ys, xs = np.nonzero(a > 0.01)                      # below 1% of the effect nothing visible changes
        if ys.size:
            size = float(adj.get("skin_size", 0))
            pad = 2 * skin_radius(c.scale, c.long, size) + 4
            y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + 1 + pad)
            x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + 1 + pad)
            sub = rgb[y0:y1, x0:x1]
            sm = smooth_skin(sub, v, c.scale, c.long, size)
            rgb = rgb.copy()
            rgb[y0:y1, x0:x1] = sub + A[y0:y1, x0:x1] * (sm - sub)
    for key, fn in (("texture", lambda x, v: F.texture(x, v, c.scale)),
                    ("clarity", lambda x, v: clarity(x, v, c.scale)),
                    ("dehaze", lambda x, v: F.dehaze(x, v, airlight, c.scale, c.long) if airlight is not None
                     else x)):
        v = float(adj.get(key, 0))
        if v:
            rgb = rgb + A * (fn(rgb, v) - rgb)
    return rgb.astype(np.float32)


def clarity_radius(scale: float) -> int:
    return max(2, int(round(15.0 * scale)))


def clarity(rgb: np.ndarray, amount: float, scale: float = 1.0) -> np.ndarray:
    """Midtone contrast at a medium scale (like 'Clarity'), edge-aware so big edges get no halo."""
    from ..detail.finish import _self_guided
    if not amount:
        return rgb
    Y = np.maximum(rgb @ LUMA, 1e-7)
    L = np.log2(Y)
    base = _self_guided(L, clarity_radius(scale), 0.35 ** 2)
    d = L - base
    k = np.float32(0.8 * amount / 100.0)
    mid = np.exp(-0.5 * ((L - np.log2(0.18)) / 2.5) ** 2).astype(np.float32)      # midtones most
    return (rgb * np.exp2(k * mid * d)[..., None]).astype(np.float32)


def tone_light(m: dict, base_light, base_contrast: float):
    """(Light, contrast) for this mask's tone controls, laid over the global ones."""
    adj = m.get("adjust") or {}
    L = type(base_light)(**{k: getattr(base_light, k) for k in base_light.__dataclass_fields__})
    for k in ("highlights", "shadows", "whites", "blacks"):
        v = float(adj.get(k, 0))
        if v:
            setattr(L, k, float(np.clip(getattr(L, k) + v, -100, 100)))
    con = base_contrast * (1.0 + 0.4 * float(adj.get("contrast", 0)) / 100.0)
    return L, max(0.6, min(2.4, con))


def colour_of(m: dict):
    from ..pipeline.edit import Colour
    adj = m.get("adjust") or {}
    return Colour(saturation=float(adj.get("saturation", 0)), vibrance=float(adj.get("vibrance", 0)))


def apply_detail(rgb: np.ndarray, m: dict, a: np.ndarray, scale: float) -> np.ndarray:
    """Sharpness (+ crisper, - softer) and noise smoothing inside the mask (display-referred)."""
    adj = m.get("adjust") or {}
    s, n = float(adj.get("sharpness", 0)), float(adj.get("noise", 0))
    if not s and not n:
        return rgb
    Y = np.maximum(rgb @ LUMA, 1e-6)
    L = np.log2(Y)
    L2 = L
    if s > 0:
        r = max(0.6, 1.0 * scale)
        L2 = L2 + np.float32(1.2 * s / 100.0) * np.clip(L - gaussian_filter(L, r), -0.5, 0.5)
    elif s < 0:
        r = max(0.8, 2.5 * scale)
        L2 = L2 + np.float32(-s / 100.0) * (gaussian_filter(L, r) - L)
    out = rgb * np.exp2(L2 - L)[..., None]
    if n > 0:
        from ..detail.finish import guided
        r = max(1, int(round(2 * scale)))
        g = np.log2(np.maximum(out @ LUMA, 1e-6)).astype(np.float32)
        sm = np.stack([guided(g, out[..., k].astype(np.float32), r, 0.004) for k in range(3)], -1)
        out = out + np.float32(n / 100.0) * (sm - out)
    A = a[..., None]
    return (rgb + A * (out - rgb)).astype(np.float32)


# ------------------------------------------------------------------------------------------ click an object
def _hist_model(f: np.ndarray, w: np.ndarray, lo, hi, bins=(10, 16, 16)) -> np.ndarray:
    """Smoothed, normalised 3-D colour histogram of features f (N, 3) with weights w."""
    edges = [np.linspace(lo[k], hi[k], bins[k] + 1) for k in range(3)]
    H, _ = np.histogramdd(f, bins=edges, weights=w)
    H = gaussian_filter(H, 0.8) + 1e-6
    return H / H.sum()


def _lookup(H: np.ndarray, f: np.ndarray, lo, hi) -> np.ndarray:
    idx = []
    for k in range(3):
        n = H.shape[k]
        i = np.floor((f[:, k] - lo[k]) / max(hi[k] - lo[k], 1e-9) * n).astype(int)
        idx.append(np.clip(i, 0, n - 1))
    return H[idx[0], idx[1], idx[2]]


def object_bitmap(feat: np.ndarray, fg: list, bg: list, iterations: int = 4) -> np.ndarray:
    """'Click an object' on a small image. feat: (h, w, 3) features (screen brightness, a/L, b/L); fg / bg: seed
    points as (x, y) fractions. GrabCut-style (Rother, Kolmogorov, Blake 2004) but with colour histograms and an
    edge-aware guided filter instead of graph cuts: the object's colours are learnt around the clicks, the rest
    from everything else, each pixel is scored by which it resembles more, the score is smoothed along the photo's
    edges, and only the connected part holding a click is kept. Repeated a few times with the models re-learnt
    from the current answer. Brightness counts less than colour, so shading across a leaf does not split it."""
    from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, label
    from ..detail.finish import guided
    h, w = feat.shape[:2]
    f = np.stack([gaussian_filter(feat[..., k].astype(np.float64), 1.0) for k in range(3)], -1)
    f[..., 0] *= 0.8
    dark = feat[..., 0] < 0.08                       # colour is noise in near-black areas: call it neutral there
    f[..., 1][dark] *= feat[..., 0][dark] / 0.08
    f[..., 2][dark] *= feat[..., 0][dark] / 0.08
    flat = f.reshape(-1, 3)
    lo = np.quantile(flat, 0.002, axis=0) - 1e-3
    hi = np.quantile(flat, 0.998, axis=0) + 1e-3
    yy, xx = np.mgrid[0:h, 0:w]
    r = max(3.0, 0.025 * max(h, w))

    def discs(pts, rad):
        m = np.zeros((h, w), bool)
        for x, y in pts:
            m |= (xx - (x * w - 0.5)) ** 2 + (yy - (y * h - 0.5)) ** 2 <= rad * rad
        return m
    if not fg:
        return np.zeros((h, w), np.uint8)
    seed_f, seed_b = discs(fg, r), discs(bg, r)
    guide = f[..., 0].astype(np.float32) * 2.0 + 0.5 * f[..., 1].astype(np.float32)
    mask = seed_f.copy()
    for it in range(iterations):
        inner = (binary_erosion(mask, iterations=2) if it else mask) | seed_f
        outer = ~binary_dilation(mask, iterations=3) | seed_b
        wf = inner.ravel().astype(np.float64)
        wb = outer.ravel().astype(np.float64)
        Hf = _hist_model(flat, wf, lo, hi)
        Hb = _hist_model(flat, wb, lo, hi)
        llr = np.log(_lookup(Hf, flat, lo, hi)) - np.log(_lookup(Hb, flat, lo, hi))
        llr = np.clip(llr.reshape(h, w), -6, 6).astype(np.float32)
        llr[seed_f] = 6.0
        llr[seed_b] = -6.0
        sm = guided(guide, llr, max(2, int(round(0.008 * max(h, w)))), 0.02)
        new = sm > 0.0
        holes = binary_fill_holes(new) & ~new          # fill only small holes (specks), not a fly on the leaf
        hl, nh = label(holes)
        if nh:
            sizes = np.bincount(hl.ravel())
            small = np.nonzero(sizes < 0.001 * h * w)[0]
            new |= np.isin(hl, small[small > 0])
        lab, n = label(new)
        keep = {int(v) for v in np.unique(lab[seed_f])} - {0}
        new = np.isin(lab, list(keep)) if keep else seed_f
        if np.array_equal(new, mask):
            break
        mask = new
    # the score is averaged across the edge, which leaves the answer a pixel or two short of it: grow back up to
    # 2 pixels where the score is only mildly against (render-time refinement then snaps to the real edge)
    mask = mask | (binary_dilation(mask, iterations=2) & (sm > -2.0))
    return (mask * 255).astype(np.uint8)
