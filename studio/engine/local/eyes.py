"""Eyes (plan 9.7 follow-up, session 11): the iris found in each eye, and the tools that need it: eye colour, iris
brightness, clarity and colour strength, the dark outer ring of the iris, catchlights, and automatic red-eye and
blown-pupil fixes. No GUI imports (rule 2.2).

Finding the iris
* Each eye gets its own crop from the face mesh, as MediaPipe Iris does: centred between the eye's corners (mesh
  points 33 / 133 and 362 / 263), square, 2.3 x the corner distance, turned level; the person's left eye is mirrored
  so both look like a right eye to the model.
* MediaPipe Iris (Ablavatski, Vakunov, Grishchenko, Raveendran, Zhdanovich, "Real-time Pupil Tracking from Monocular
  Video for Digital Puppetry", 2020; Apache 2.0, models/faces/iris_landmark.tflite) gives the iris centre and four
  points on its edge, run by tinynet on this PC.
* Without the model, or when its answer is implausible: a classical fit, the dark disc of the size the eye predicts
  (the human iris is about 11.7 mm across in nearly every adult, the eye opening about 28-30 mm, so the iris radius
  is about 0.2 x the corner distance), scored by how much darker it is than the white around it and how strong its
  edge is, inside the eyelids.
* Each found iris is stored in its face's entry in the edit, as camera-frame fractions like the face points:
      "irises": [{"side": "r" | "l", "c": [x, y], "e": [x, y] (a point on the edge), "how": "model" | "fit"}]

The tools (all 0 by default, subtle at 100; worked out in the face's gain map, portrait.face_gain)
* iris_light, iris_clarity, iris_colour: brightness, local contrast (the fibres), colour strength of the iris
  (never the pupil or the catchlights).
* ring: darkens the outer edge of the iris (the limbal ring), a classic retouch: a prominent ring reads as young
  and healthy (Peshek, Semmaknejad, Hoffman, Foley, "Preliminary evidence that the limbal ring influences facial
  attractiveness", Evolutionary Psychology 2011).
* catch: brightens the catchlights already there (nothing is painted in).
* eye colour: a new hue and colour strength for the iris, its lightness moved toward the colour's typical lightness
  relative to the eye whites (so a shadowed eye stays shadowed), the fibres' texture kept (the change is a smooth
  multiplier), the pupil and catchlights untouched; one eye or both.
* Red eye: red pupils inside the found irises become Red eye spots (amber, removable one by one). Pupils the flash
  blew out (white or golden, no red left to pull down) become Pupil spots, which rebuild a dark pupil and keep the
  catchlight; the same spot fixes the green / yellow / white glow in animals' eyes (clicked by hand).
"""
from __future__ import annotations

import math
import os
import threading

import numpy as np
from scipy.ndimage import gaussian_filter, label as nd_label

from ... import log as slog
from . import faces as fz

log = slog.get("engine.eyes")

IRIS_MODEL = "iris_landmark.tflite"
EYE_SCALE = 2.3                       # the eye crop: 2.3 x the corner distance (as MediaPipe Iris)
IRIS_RATIO = 0.20                     # iris radius / corner distance, typical
RATIO_OK = (0.12, 0.32)               # plausible range for the model's answer
CORNERS = {"r": (33, 133), "l": (362, 263)}
EYE_PTS = {"r": fz.RIGHT_EYE, "l": fz.LEFT_EYE}
EYE_KEYS = ("iris_light", "iris_clarity", "iris_colour", "ring", "catch", "eye_colour")
# Eye colours: (hue in OKLCh degrees, colour strength C / L, lightness relative to the eye whites)
COLOURS = {
    "blue": (245.0, 0.20, 0.62), "green": (150.0, 0.16, 0.55), "hazel": (85.0, 0.15, 0.50),
    "grey": (235.0, 0.04, 0.62), "amber": (75.0, 0.24, 0.55), "brown": (55.0, 0.15, 0.38),
    "violet": (300.0, 0.17, 0.55),
}
_lock = threading.Lock()
_model = None


# ------------------------------------------------------------------------------------------ the model
def model_path() -> str:
    return os.path.join(fz.model_dir(), IRIS_MODEL)


def model_available() -> bool:
    return os.path.isfile(model_path())


def _iris_model():
    global _model
    with _lock:
        if _model is None:
            from .tinynet import Model
            _model = Model(model_path())
        return _model


# ------------------------------------------------------------------------------------------ finding the iris
def eye_roi(P: np.ndarray, side: str):
    """The eye's crop (cx, cy, size, angle) in the image the points are in."""
    a, b = P[CORNERS[side][0]], P[CORNERS[side][1]]
    if a[0] > b[0]:
        a, b = b, a
    d = float(np.hypot(*(b - a)))
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    c = 0.5 * (a + b)
    return float(c[0]), float(c[1]), max(d * EYE_SCALE, 4.0), ang, d


def _polygon_mask(poly: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Inside test for a closed polygon (even-odd), on arrays of points."""
    inside = np.zeros(X.shape, bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > Y) != (y2 > Y)
        xint = (x2 - x1) * (Y - y1) / ((y2 - y1) + 1e-12) + x1
        inside ^= cond & (X < xint)
    return inside


def _model_iris(img: np.ndarray, roi, side: str):
    """(centre (2,), radius) in image pixels from the iris model, or None."""
    cx, cy, size, ang, _ = roi
    c = fz.crop(img, (cx, cy, size, ang), 64)
    best = []
    for cc in (c, fz._normalise_crop(c)):
        x = cc[:, ::-1] if side == "l" else cc
        out = _iris_model()(np.ascontiguousarray(x)[None])
        ir = next(v for k, v in out.items() if v.size == 15).reshape(5, 3)[:, :2].astype(float)
        if side == "l":
            ir[:, 0] = 64.0 - ir[:, 0]
        X, Y = fz._roi_points((cx, cy, size, ang), ir[:, 0] / 64.0, ir[:, 1] / 64.0)
        pts = np.stack([X, Y], -1)
        r = float(np.mean(np.hypot(*(pts[1:] - pts[0]).T)))
        best.append((pts[0], r))
    return best


def _refine(img: np.ndarray, P: np.ndarray, side: str, roi, c, r):
    """A seed (the mesh's or the model's iris) finished by the edge search close to it: the search window is
    +-0.15 r on each axis, so anything it returns inside 0.25 r is taken, with the size kept to 0.85-1.3 of the
    seed's. Otherwise the seed stands."""
    ref = _fit_iris(img, P, side, roi, near=(c, r))
    if ref is not None and np.hypot(*(ref[0] - c)) < 0.25 * r and 0.85 < ref[1] / r < 1.3:
        return ref[0], ref[1]
    return c, r


def _fit_iris(img: np.ndarray, P: np.ndarray, side: str, roi, near=None):
    """The classical fit: the dark disc of the predicted size inside the eyelids. (centre, radius, score).
    ``near`` = (centre, radius) from the model: search only close to it (the model finds where the iris is, the
    edge decides how big it is)."""
    cx, cy, size, ang, d = roi
    n = 96
    c = fz.crop(img, (cx, cy, size, ang), n)
    Y = (c @ np.array([0.2126, 0.7152, 0.0722], np.float32)).astype(np.float32)
    g = np.arange(n) + 0.5
    U, V = np.meshgrid(g, g)
    px = size / n                                         # image pixels per crop pixel
    # the eyelid polygon in crop pixels
    E = P[EYE_PTS[side]]
    ca, sa = math.cos(ang), math.sin(ang)
    du, dv = E[:, 0] - cx, E[:, 1] - cy
    poly = np.stack([(ca * du + sa * dv) / px + n / 2, (-sa * du + ca * dv) / px + n / 2], -1)
    lid = _polygon_mask(poly, U, V)
    if lid.sum() < 20:
        return None
    gy, gx = np.gradient(gaussian_filter(Y, 0.8))
    best = None
    rr = d / px
    ys, xs = np.nonzero(lid)
    radii = np.linspace(0.15 * rr, 0.27 * rr, 7)
    oys = range(int(ys.min()), int(ys.max()) + 1, 2)
    oxs = range(int(xs.min()), int(xs.max()) + 1, 2)
    if near is not None:                                  # around the model's answer, finer steps
        (nx0, ny0), nr = near
        du0, dv0 = nx0 - cx, ny0 - cy
        u0 = (ca * du0 + sa * dv0) / px + n / 2
        v0 = (-sa * du0 + ca * dv0) / px + n / 2
        r0 = nr / px
        radii = np.linspace(0.9 * r0, 1.25 * r0, 8)
        step = max(1, int(round(0.08 * r0)))
        oys = range(int(v0 - 0.15 * r0), int(v0 + 0.15 * r0) + 1, step)
        oxs = range(int(u0 - 0.15 * r0), int(u0 + 0.15 * r0) + 1, step)
    for r in radii:
        for oy in oys:
            for ox in oxs:
                if not (0 <= oy < n and 0 <= ox < n) or not lid[oy, ox]:
                    continue
                dist = np.hypot(U - (ox + 0.5), V - (oy + 0.5))
                disc = (dist < r) & lid
                ring = (dist >= r * 1.15) & (dist < r * 1.6) & lid
                if disc.sum() < 6 or ring.sum() < 6:
                    continue
                contrast = float(Y[ring].mean() - Y[disc].mean())
                edge = (np.abs(dist - r) < 1.0) & lid
                if edge.sum() > 3:
                    nx, ny = (U - ox - 0.5) / (dist + 1e-6), (V - oy - 0.5) / (dist + 1e-6)
                    radial = float(np.mean((gx * nx + gy * ny)[edge]))
                else:
                    radial = 0.0
                score = contrast + 2.0 * max(radial, 0.0) * r
                if best is None or score > best[0]:
                    best = (score, ox + 0.5, oy + 0.5, r)
    if best is None or best[0] <= 0:
        return None
    _, ox, oy, r = best
    X, Yp = fz._roi_points((cx, cy, size, ang), ox / n, oy / n)
    return np.array([X, Yp], float), r * px, best[0]


def find_irises(img: np.ndarray, P: np.ndarray, iris10: np.ndarray | None = None) -> list:
    """Both irises of one face: [{"side", "c": (x, y), "r": radius, "how"}] in the image's pixels. ``P`` is the
    468-point mesh in the same image; ``iris10`` the newer mesh's own 10 iris points, when it gave them (they are
    tried first: "mesh"), then the iris model ("model"), then the classical fit ("fit"). The mesh's and the model's
    answers are both finished by the edge search, which sets the size."""
    out = []
    for side in ("r", "l"):
        roi = eye_roi(P, side)
        d = roi[4]
        if d < 6:
            continue
        lid = P[EYE_PTS[side]]
        res = None
        if iris10 is not None and len(iris10) >= 10:
            q = np.asarray(iris10[:5] if side == "r" else iris10[5:10], float)
            c = q[0]
            r = float(np.mean(np.hypot(*(q[1:] - c).T)))
            if RATIO_OK[0] <= r / d <= RATIO_OK[1] and _polygon_mask(lid, np.array([[c[0]]]), np.array([[c[1]]]))[0, 0]:
                c, r = _refine(img, P, side, roi, c, r)
                res = (c, r, "mesh")
        if res is None and model_available():
            try:
                cands = _model_iris(img, roi, side)
                good = [(c, r) for c, r in cands if RATIO_OK[0] <= r / d <= RATIO_OK[1] and
                        _polygon_mask(lid, np.array([[c[0]]]), np.array([[c[1]]]))[0, 0]]
                if good:
                    c = np.mean([g[0] for g in good], axis=0)
                    r = float(np.mean([g[1] for g in good]))
                    c, r = _refine(img, P, side, roi, c, r)
                    res = (c, r, "model")
            except Exception as e:                       # noqa: BLE001  (the fit below still works)
                log.warning("iris model failed: %s", e)
        if res is None:
            fit = _fit_iris(img, P, side, roi)
            if fit is not None:
                res = (fit[0], fit[1], "fit")
        if res is not None:
            out.append({"side": side, "c": (float(res[0][0]), float(res[0][1])), "r": float(res[1]),
                        "how": res[2]})
    log.info("irises: %s", ", ".join(f"{e['side']} r={e['r']:.1f}px ({e['how']})" for e in out) or "none")
    return out


# ------------------------------------------------------------------------------------------ settings
def colour_settings(p: dict | None) -> dict | None:
    """The eye-colour target {"hue", "sat", "light", "which"} or None when off."""
    ec = (p or {}).get("eye_colour") or {}
    amt = min(max(float(ec.get("amount", 0) or 0), 0.0), 100.0)
    preset = str(ec.get("preset", "") or "")
    if amt <= 0 or not preset or preset == "none":
        return None
    if preset == "custom":
        h, s, t = float(ec.get("hue", 245.0)), float(ec.get("sat", 0.18)), float(ec.get("light", 0.55))
    elif preset in COLOURS:
        h, s, t = COLOURS[preset]
    else:
        return None
    return {"hue": h, "sat": max(0.0, min(0.4, s)), "light": max(0.15, min(0.95, t)),
            "which": str(ec.get("which", "both") or "both"), "amount": amt / 100.0}


def custom_from_srgb(rgb8) -> tuple:
    """An sRGB colour picked by the photographer -> (hue, colour strength, lightness) for a custom eye colour."""
    from ..color.grade import to_oklab
    from ..color.spaces import PROPHOTO, SRGB, conversion_matrix
    c = np.asarray(rgb8, np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    pp = conversion_matrix(SRGB, PROPHOTO) @ lin                      # the pipeline works in linear ProPhoto
    lab = to_oklab(pp.astype(np.float32).reshape(1, 1, 3))[0, 0]
    L = max(float(lab[0]), 1e-3)
    return (math.degrees(math.atan2(float(lab[2]), float(lab[1]))) % 360.0,
            float(np.hypot(lab[1], lab[2])) / L, L)


def eye_active(p: dict | None, amt: dict) -> bool:
    return any(amt.get(k, 0) > 1e-4 for k in EYE_KEYS if k != "eye_colour") or colour_settings(p) is not None


# ------------------------------------------------------------------------------------------ masks on the work grid
def iris_list(f: dict, sw: int, sh: int) -> list:
    """A stored face's irises in S pixels: [(side, cx, cy, r, how)]."""
    out = []
    for e in f.get("irises") or []:
        try:
            cx, cy = float(e["c"][0]) * sw, float(e["c"][1]) * sh
            ex, ey = float(e["e"][0]) * sw, float(e["e"][1]) * sh
        except (KeyError, TypeError, IndexError, ValueError):
            continue
        r = math.hypot(ex - cx, ey - cy)
        if r > 0.5:
            out.append((str(e.get("side", "r")), cx, cy, r, str(e.get("how", "model"))))
    return out


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def eye_masks(irises: list, X: np.ndarray, Y: np.ndarray, eye_zone: np.ndarray, lab_L: np.ndarray, q: float,
              lids: dict | None = None) -> dict:
    """Soft masks on the work grid: iris (without pupil and catchlights), pupil, ring, catch, and per side.
    ``lids``: {side: (16, 2) eyelid outline in S pixels} from the face mesh, so nothing leaks onto the eyelids."""
    h, w = X.shape
    m = {k: np.zeros((h, w), np.float32) for k in ("iris", "pupil", "ring", "catch", "iris_r", "iris_l", "disc")}
    for side, cx, cy, r, _ in irises:
        d = np.hypot(X - cx, Y - cy) / max(r, 1e-3)
        feather = min(0.25, 1.5 * q / max(r, 1e-3))
        disc = (1.0 - _smoothstep((d - (1.0 - feather)) / (1.5 * feather))).astype(np.float32)  # edge: no old rim
        if lids and side in lids:
            near = d < 1.3
            open_ = np.zeros((h, w), np.float32)
            open_[near] = _polygon_mask(lids[side], X[near], Y[near])
            disc *= gaussian_filter(open_, 0.6)               # the eyelids cover part of the iris
        else:
            disc *= np.clip((eye_zone - 0.35) * 3.0, 0.0, 1.0)
        if disc.max() <= 0:
            continue
        inner = disc > 0.5
        if inner.sum() < 4:
            continue
        Li = lab_L[inner]
        med = float(np.median(Li))
        # the pupil: the dark core (soft), within 0.7 of the radius
        dark = _smoothstep((0.62 * med - lab_L) / max(0.25 * med, 1e-4)) * (d < 0.7)
        # catchlights: much brighter than the iris
        bright = _smoothstep((lab_L - 1.6 * max(med, 1e-4)) / max(0.6 * med, 1e-4)) * (d < 1.0)
        pupil = (dark * disc).astype(np.float32)
        catch = (bright * disc).astype(np.float32)
        iris = (disc * (1.0 - pupil) * (1.0 - catch)).astype(np.float32)
        ring = (_smoothstep((d - 0.72) / 0.18) * disc * (1.0 - catch)).astype(np.float32)
        m["iris"] = np.maximum(m["iris"], iris)
        m["pupil"] = np.maximum(m["pupil"], pupil)
        m["catch"] = np.maximum(m["catch"], catch)
        m["ring"] = np.maximum(m["ring"], ring)
        m["disc"] = np.maximum(m["disc"], disc)
        m["iris_" + ("r" if side == "r" else "l")] = np.maximum(m["iris_" + ("r" if side == "r" else "l")], iris)
    return m


def apply_eyes(lab2: np.ndarray, M: dict, amt: dict, colour: dict | None, white_L: float | None, r_work: float):
    """The eye tools on an OKLab work copy (in place). ``white_L``: the eye whites' median OKLab lightness."""
    L = lab2[..., 0]
    changed = False
    iris = M["iris"]
    if iris.max() <= 0:
        return False
    a_light, a_clar, a_col = amt.get("iris_light", 0), amt.get("iris_clarity", 0), amt.get("iris_colour", 0)
    a_ring, a_catch = amt.get("ring", 0), amt.get("catch", 0)
    if a_clar > 0:                                   # local contrast measured within the iris only (normalised
        lg = np.log2(np.maximum(L, 1e-4))            # convolution), so the white and the pupil never pull it
        sg = max(1.0, 0.18 * r_work)
        wm = gaussian_filter(iris, sg)
        mean = gaussian_filter(lg * iris, sg) / np.maximum(wm, 1e-4)
        det = np.clip(lg - mean, -0.25, 0.25) * (wm > 0.05)
        k = np.float32(1.2 * a_clar)
        sel = iris > 0.5
        if sel.sum() > 4:                            # keep the iris's average brightness (light ~ L cubed)
            det = det - float(np.log2(np.mean(np.exp2(3.0 * k * det[sel])))) / (3.0 * float(k))
        L *= np.exp2(k * iris * det)
        changed = True
    if a_light > 0:
        L *= np.exp2(np.float32(0.5 * a_light / 3.0) * iris)               # up to half a stop (OKLab L ~ cube root)
        changed = True
    if a_ring > 0:
        L *= np.exp2(-np.float32(0.6 * a_ring / 3.0) * M["ring"])
        changed = True
    if a_catch > 0:
        L *= np.exp2(np.float32(0.6 * a_catch / 3.0) * M["catch"])
        changed = True
    if a_col > 0:
        k = (1.0 + np.float32(0.8 * a_col) * iris)
        lab2[..., 1] *= k
        lab2[..., 2] *= k
        changed = True
    if colour is not None:
        which = colour["which"]
        sel = M["iris"] if which == "both" else M["iris_r" if which == "right" else "iris_l"]
        kc = (np.float32(colour["amount"]) * sel).astype(np.float32)
        if kc.max() > 0:
            inside = sel > 0.5
            ra, rb = lab2[..., 1] / np.maximum(L, 1e-4), lab2[..., 2] / np.maximum(L, 1e-4)
            if white_L and inside.sum() > 3:                 # lightness first, toward the colour's typical value
                med = float(np.median(L[inside]))
                f = min(1.35, max(0.75, colour["light"] * white_L / max(med, 1e-4)))
                L *= np.power(np.float32(f), kc)
            Ll = np.maximum(L, 1e-4)
            h = math.radians(colour["hue"])
            ta, tb = colour["sat"] * math.cos(h), colour["sat"] * math.sin(h)
            # move the iris's average colour to the target, keep half of its own variation (the fibres)
            ma, mb = (float(np.median(ra[inside])), float(np.median(rb[inside]))) if inside.sum() > 3 else (0.0, 0.0)
            lab2[..., 1] = (ra + kc * (ta - ma) - 0.5 * kc * (ra - ma)) * Ll
            lab2[..., 2] = (rb + kc * (tb - mb) - 0.5 * kc * (rb - mb)) * Ll
            changed = True
    return changed


# ------------------------------------------------------------------------------------------ red eye and blown pupils
def find_red_pupils(rgb: np.ndarray, gx: float, gy: float, q: float, irises: list, lids: dict | None = None) -> list:
    """Look inside each iris (scene-linear work copy) for a red pupil or a blown-out one. ``lids``: {side: eyelid
    outline in S pixels}, so the skin over the iris never counts as iris. Returns [{"kind": "redeye" | "pupil", "x",
    "y", "r"}] in S pixels."""
    h, w = rgb.shape[:2]
    c = 0.5 * (q - 1)
    X = gx + q * np.arange(w)[None, :] + c
    Y = gy + q * np.arange(h)[:, None] + c
    X, Y = np.broadcast_to(X, (h, w)), np.broadcast_to(Y, (h, w))
    out = []
    pos = np.maximum(rgb, 0.0)
    R, G, B = pos[..., 0], pos[..., 1], pos[..., 2]
    lum = pos @ np.array([0.2880, 0.7119, 0.0001], np.float32)
    for side, cx, cy, r, _ in irises:
        d = np.hypot(X - cx, Y - cy) / max(r, 1e-3)
        open_ = np.ones((h, w), bool)
        if lids and side in lids:
            near = d < 1.6
            open_ = np.zeros((h, w), bool)
            open_[near] = _polygon_mask(lids[side], X[near], Y[near])
        disc = (d < 0.95) & open_
        if disc.sum() < 6:
            continue
        ring = (d > 1.1) & (d < 1.5) & open_
        white = float(np.percentile(lum[ring], 80)) if ring.sum() > 6 else float(np.percentile(lum[disc], 95))
        outer = (d > 0.75) & (d < 0.95) & open_
        iris_lum = float(np.median(lum[outer])) if outer.sum() > 4 else 0.0
        # a red pupil glows: strongly saturated red (a brown iris is red-dominant too, but nowhere near this) and
        # brighter than the iris around it
        red = (R / np.maximum(0.5 * (G + B), 1e-5) > 5.0) & (lum > 1.2 * iris_lum) & (d < 0.8) & open_
        blown = (lum > 0.8 * max(white, 1e-4)) & (np.max(pos, axis=-1) / np.maximum(np.min(pos, axis=-1), 1e-5) < 1.6) \
            & (d < 0.6) & open_
        for kind, mask in (("redeye", red), ("pupil", blown)):
            lab, n = nd_label(mask)
            if n == 0:
                continue
            sizes = np.bincount(lab.ravel())[1:]
            k = int(np.argmax(sizes)) + 1
            area_frac = sizes[k - 1] / max(float(disc.sum()), 1.0)
            if kind == "redeye" and area_frac < 0.06:
                continue
            if kind == "pupil" and area_frac < 0.10:          # a catchlight is smaller than this
                continue
            sel = lab == k
            px, py = float(X[sel].mean()), float(Y[sel].mean())
            rad = max(math.sqrt(sizes[k - 1] * q * q / math.pi) * 1.25, 0.35 * r)
            out.append({"kind": kind, "x": px, "y": py, "r": min(rad, 0.95 * r), "side": side})
            break
    return out


def pupil_fix(D: np.ndarray, inside: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """A blown pupil or an animal's eye glow rebuilt (retouch spot kind "pupil"): a dark, neutral pupil with the
    catchlight kept. ``D`` scene-linear patch; ``inside`` the spot; ``ring`` just outside it (the iris)."""
    lum = np.maximum(D, 0.0) @ np.array([0.2880, 0.7119, 0.0001], np.float32)
    ref = float(np.percentile(lum[ring], 25)) if ring.sum() > 6 else float(np.percentile(lum[inside], 5))
    dark = max(0.12 * ref, 1e-4)
    out = np.where(inside[..., None], np.float32(dark), D).astype(np.float32)
    # keep the catchlight: the brightest compact core, or a small disc at the brightest point
    li = np.where(inside, lum, 0.0)
    top = li.max()
    if top > 0:
        core = li > 0.92 * top
        area = core.sum() / max(float(inside.sum()), 1.0)
        if area > 0.08:                                      # uniformly blown: a small catchlight at the peak
            yy, xx = np.unravel_index(int(np.argmax(li)), li.shape)
            rr = max(1.0, 0.15 * math.sqrt(inside.sum() / math.pi))
            Yg, Xg = np.mgrid[0:li.shape[0], 0:li.shape[1]]
            core = np.hypot(Xg - xx, Yg - yy) < rr
        core = gaussian_filter(core.astype(np.float32), 0.7)
        out = out + core[..., None] * (D - out)
    return out.astype(np.float32)
