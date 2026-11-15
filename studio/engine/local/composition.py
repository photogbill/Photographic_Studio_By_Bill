"""Composition: the overlays of the Crop tool and suggested crops, by method.

Overlays (``OVERLAYS``) are drawn on the crop frame (or the whole photo). Each is plain geometry in fractions of the
frame (0..1 across and down); the ones that are not symmetric can be turned (``ORIENTATIONS``: mirrored left-right
and / or top-bottom). Where a method's geometry depends on the frame's shape (rabatment, the 45-degree diagonals,
dynamic symmetry, the golden triangles) it is worked out in pixels, so a 45-degree line really is 45 degrees.

    Rule of thirds        lines at 1/3 and 2/3
    Golden ratio          the phi grid: lines at 0.382 and 0.618
    Golden spiral         the Fibonacci (golden) spiral and its squares, stretched to the frame (as Lightroom does)
    Golden triangles      one diagonal and the perpendiculars to it from the other two corners
    Diagonals             the baroque (lower left to upper right) and sinister (upper left to lower right) diagonals
    Diagonal method       45-degree lines from each corner (Edwin Westhoff's diagonal method)
    Dynamic symmetry      Jay Hambidge's armature: the diagonals and their reciprocals
    Harmonic armature     diagonals, the diamond of the side midpoints, corners to the far midpoints, centre lines
    Rabatment             the squares on the two short sides (each long side "folded" onto the frame)
    Centre                centre lines, for symmetry
    Radial                spokes from the centre, for radial compositions
    Grid                  an 8 x 8 grid, for straightening and placing

Suggested crops (``suggest``) are built from what the Studio knows about the photo (``subject``): faces (eyes, head,
which way the face is turned), people (33 body points from the Person finder) and, when there is nobody, the part of
the photo that stands out (spectral-residual saliency, Hou and Zhang 2007) and is in focus. Each candidate puts the
main point (the eyes, or the subject's centre) on one of the method's strong points and is scored on the rules
portrait and fine-art photographers use: the face and head stay whole with some headroom; there is more space in
front of a face than behind it (lead room); no edge cuts through a joint (knees, elbows, wrists, ankles) or the neck;
nobody is half in the frame; and a larger crop is better than a smaller one when all else is equal (resolution).
No crop model with a licence that fits this project was found (the public ones are trained on research-only data),
so this is rules, not a learnt taste.
"""
from __future__ import annotations

import math

import numpy as np

PHI = (1.0 + 5.0 ** 0.5) / 2.0

OVERLAYS = [
    ("thirds", "Rule of thirds"),
    ("phi", "Golden ratio (phi grid)"),
    ("spiral", "Golden spiral"),
    ("triangles", "Golden triangles"),
    ("diagonals", "Diagonals (baroque and sinister)"),
    ("diag45", "Diagonal method (45 degrees)"),
    ("dynamic", "Dynamic symmetry"),
    ("armature", "Harmonic armature"),
    ("rabatment", "Rabatment"),
    ("centre", "Centre (symmetry)"),
    ("radial", "Radial"),
    ("grid", "Grid"),
]
NAMES = dict(OVERLAYS)
ORIENTATIONS = {"spiral": 4, "triangles": 2}          # how many ways each can be turned (others look the same)
METHODS = [k for k, _ in OVERLAYS if k not in ("radial", "grid")]   # the ones that say where a subject goes


# ------------------------------------------------------------------------------------------ geometry helpers
def _orient(pts, o: int):
    """Mirror points in the unit square: bit 0 left-right, bit 1 top-bottom."""
    out = []
    for x, y in pts:
        if o & 1:
            x = 1.0 - x
        if o & 2:
            y = 1.0 - y
        out.append((float(x), float(y)))
    return out


def _clip_ray(p, d, a: float):
    """Where a ray from p (pixels, frame a x 1) in direction d leaves the frame."""
    ts = []
    for k, lim in ((0, 0.0), (0, a), (1, 0.0), (1, 1.0)):
        if abs(d[k]) > 1e-12:
            t = (lim - p[k]) / d[k]
            if t > 1e-9:
                q = (p[0] + t * d[0], p[1] + t * d[1])
                if -1e-9 <= q[0] <= a + 1e-9 and -1e-9 <= q[1] <= 1 + 1e-9:
                    ts.append(t)
    t = min(ts) if ts else 0.0
    return (p[0] + t * d[0], p[1] + t * d[1])


def _foot(c, p0, p1):
    """Foot of the perpendicular from c onto the line p0-p1 (pixels)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t = ((c[0] - p0[0]) * dx + (c[1] - p0[1]) * dy) / (dx * dx + dy * dy)
    return (p0[0] + t * dx, p0[1] + t * dy)


def _norm(pts, a: float):
    return [(x / a, y) for x, y in pts]


def _spiral():
    """The golden spiral in a golden rectangle (PHI x 1): its quarter arcs and squares, and where it converges."""
    x, y, w, h = 0.0, 0.0, PHI, 1.0
    arcs, squares = [], []
    for i in range(40):                                # 11 squares are drawn; the rest only find the eye
        d = i % 4
        if d == 0:
            s = h
            sq, cen, t0 = (x, y, s, s), (x + s, y + s), 180
            x, w = x + s, w - s
        elif d == 1:
            s = w
            sq, cen, t0 = (x, y, s, s), (x, y + s), 270
            y, h = y + s, h - s
        elif d == 2:
            s = h
            sq, cen, t0 = (x + w - s, y, s, s), (x + w - s, y), 0
            w = w - s
        else:
            s = w
            sq, cen, t0 = (x, y + h - s, s, s), (x + s, y + h - s), 90
            h = h - s
        if i < 11:
            ts = np.radians(t0 + np.linspace(0.0, 90.0, 13))
            arcs.extend([(cen[0] + s * math.cos(t), cen[1] + s * math.sin(t)) for t in ts][(1 if arcs else 0):])
            squares.append(sq)
    eye = (x + w / 2.0, y + h / 2.0)
    return arcs, squares, eye


_SPIRAL = _spiral()


# ------------------------------------------------------------------------------------------ overlays
def lines(name: str, aspect: float, orient: int = 0) -> list:
    """The overlay as polylines [[(x, y), ...], ...] in fractions of a frame ``aspect`` (width / height) wide."""
    a = max(float(aspect), 1e-3)
    o = orient % ORIENTATIONS.get(name, 1)
    L = []
    if name == "thirds":
        for k in (1, 2):
            L += [[(k / 3, 0.0), (k / 3, 1.0)], [(0.0, k / 3), (1.0, k / 3)]]
    elif name == "phi":
        g = 1.0 / PHI ** 2
        for v in (g, 1.0 - g):
            L += [[(v, 0.0), (v, 1.0)], [(0.0, v), (1.0, v)]]
    elif name == "spiral":
        arcs, squares, _ = _SPIRAL
        L.append(_orient([(px / PHI, py) for px, py in arcs], o))
        for sx, sy, s, _s in squares[:7]:
            L.append(_orient([(sx / PHI, sy), ((sx + s) / PHI, sy), ((sx + s) / PHI, sy + s), (sx / PHI, sy + s),
                              (sx / PHI, sy)], o))
    elif name == "triangles":
        c0, c1, e0, e1 = ((0.0, 1.0), (a, 0.0), (0.0, 0.0), (a, 1.0)) if o == 0 else \
                         ((0.0, 0.0), (a, 1.0), (a, 0.0), (0.0, 1.0))
        L.append(_norm([c0, c1], a))
        L.append(_norm([e0, _foot(e0, c0, c1)], a))
        L.append(_norm([e1, _foot(e1, c0, c1)], a))
    elif name == "diagonals":
        L += [[(0.0, 1.0), (1.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)]]
    elif name == "diag45":
        for c, d in (((0.0, 0.0), (1, 1)), ((0.0, 1.0), (1, -1)), ((a, 0.0), (-1, 1)), ((a, 1.0), (-1, -1))):
            L.append(_norm([c, _clip_ray(c, d, a)], a))
    elif name == "dynamic":
        d1, d2 = ((0.0, 1.0), (a, 0.0)), ((0.0, 0.0), (a, 1.0))
        L += [_norm(list(d1), a), _norm(list(d2), a)]
        for c, (p0, p1) in (((0.0, 0.0), d1), ((a, 1.0), d1), ((a, 0.0), d2), ((0.0, 1.0), d2)):
            f = _foot(c, p0, p1)
            L.append(_norm([c, _clip_ray(c, (f[0] - c[0], f[1] - c[1]), a)], a))
    elif name == "armature":
        L += [[(0.0, 0.0), (1.0, 1.0)], [(0.0, 1.0), (1.0, 0.0)],
              [(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5), (0.5, 0.0)],
              [(0.5, 0.0), (0.5, 1.0)], [(0.0, 0.5), (1.0, 0.5)]]
        for c, m1, m2 in (((0, 0), (1, 0.5), (0.5, 1)), ((1, 0), (0, 0.5), (0.5, 1)),
                          ((0, 1), (1, 0.5), (0.5, 0)), ((1, 1), (0, 0.5), (0.5, 0))):
            L += [[c, m1], [c, m2]]
    elif name == "rabatment":
        if a >= 1.0:
            for v in (1.0 / a, 1.0 - 1.0 / a):
                L.append([(v, 0.0), (v, 1.0)])
        else:
            for v in (a, 1.0 - a):
                L.append([(0.0, v), (1.0, v)])
    elif name == "centre":
        L += [[(0.5, 0.0), (0.5, 1.0)], [(0.0, 0.5), (1.0, 0.5)]]
    elif name == "radial":
        c = (a / 2.0, 0.5)
        for k in range(16):
            t = 2.0 * math.pi * k / 16.0
            L.append(_norm([c, _clip_ray(c, (math.cos(t), math.sin(t)), a)], a))
    elif name == "grid":
        for k in range(1, 8):
            L += [[(k / 8, 0.0), (k / 8, 1.0)], [(0.0, k / 8), (1.0, k / 8)]]
    return [[(float(x), float(y)) for x, y in pl] for pl in L]


def points(name: str, aspect: float, orient: int = 0) -> list:
    """The method's strong points (where a subject is placed), in fractions of the frame."""
    a = max(float(aspect), 1e-3)
    o = orient % ORIENTATIONS.get(name, 1)
    thirds = [(x, y) for x in (1 / 3, 2 / 3) for y in (1 / 3, 2 / 3)]
    if name in ("thirds", "diagonals"):
        return thirds
    if name == "phi":
        g = 1.0 / PHI ** 2
        return [(x, y) for x in (g, 1 - g) for y in (g, 1 - g)]
    if name == "spiral":
        return _orient([(_SPIRAL[2][0] / PHI, _SPIRAL[2][1])], o)
    if name == "triangles":
        c0, c1, e0, e1 = ((0.0, 1.0), (a, 0.0), (0.0, 0.0), (a, 1.0)) if o == 0 else \
                         ((0.0, 0.0), (a, 1.0), (a, 0.0), (0.0, 1.0))
        return _norm([_foot(e0, c0, c1), _foot(e1, c0, c1)], a)
    if name == "diag45":
        h = min(0.5, a / 2.0)
        return [(h / a, 0.5), (1.0 - h / a, 0.5)] if a >= 1 else [(0.5, h * 1), (0.5, 1 - h)]
    if name == "dynamic":
        d1, d2 = ((0.0, 1.0), (a, 0.0)), ((0.0, 0.0), (a, 1.0))
        return _norm([_foot((0.0, 0.0), *d1), _foot((a, 1.0), *d1), _foot((a, 0.0), *d2), _foot((0.0, 1.0), *d2)], a)
    if name == "armature":
        return thirds + [(0.5, 0.5)]
    if name == "rabatment":
        if a >= 1.0:
            return [(v, y) for v in (1.0 / a, 1.0 - 1.0 / a) for y in (1 / 3, 2 / 3)]
        return [(x, v) for v in (a, 1.0 - a) for x in (1 / 3, 2 / 3)]
    if name in ("centre", "radial"):
        return [(0.5, 0.5)]
    return []


# ------------------------------------------------------------------------------------------ suggestions
JOINTS = (13, 14, 15, 16, 25, 26, 27, 28)              # BlazePose: elbows, wrists, knees, ankles


def _rect_for(ratio: float, frame_aspect: float, s: float):
    """(cw, ch): a crop of this pixel ratio at scale s of the largest that fits, in fractions of the frame."""
    if ratio >= frame_aspect:
        cw, ch = 1.0, frame_aspect / ratio
    else:
        cw, ch = ratio / frame_aspect, 1.0
    return cw * s, ch * s


def _where(p, crop):
    x0, y0, x1, y1 = crop
    return ((p[0] - x0) / (x1 - x0), (p[1] - y0) / (y1 - y0))


def _outside(box, crop) -> float:
    """Fraction of a box's area outside the crop."""
    bx0, by0, bx1, by1 = box
    x0, y0, x1, y1 = crop
    area = max(1e-9, (bx1 - bx0) * (by1 - by0))
    ix = max(0.0, min(bx1, x1) - max(bx0, x0))
    iy = max(0.0, min(by1, y1) - max(by0, y0))
    return 1.0 - ix * iy / area


def score(crop, subject: dict, method: str, aspect: float, orient: int = 0) -> tuple:
    """(penalty, reasons) of one crop for one method: lower is better."""
    x0, y0, x1, y1 = crop
    cw, ch = x1 - x0, y1 - y0
    main = subject["main"]
    q = _where(main, crop)
    pts = points(method, aspect, orient)
    if not pts:
        return 99.0, []
    d = min(math.hypot(q[0] - px, (q[1] - py)) for px, py in pts)
    pen = 4.0 * d
    why = []
    for box in subject.get("keep", []):                                   # faces / heads must stay whole
        out = _outside(box, crop)
        if out > 0.001:
            pen += 8.0 + 10.0 * out
            why.append("cuts a face")
    for box in subject.get("others", []):                                 # nobody half in the frame
        out = _outside(box, crop)
        if 0.05 < out < 0.95:
            pen += 2.0
            why.append("cuts someone at the edge")
    sb = subject.get("box")
    if sb is not None:
        pen += 3.0 * _outside(sb, crop)
    top = subject.get("head_top")
    if top is not None:
        room = (top - y0) / ch
        if room < 0.02:
            pen += 3.0
        elif room > 0.4 and subject.get("portrait"):
            pen += 1.5 * (room - 0.4)
    g = subject.get("gaze", 0.0)                                         # >0: the face is turned to the right
    if abs(g) > 0.15:
        behind = (q[0] - 0.5) * math.copysign(1.0, g)
        if behind > 0.02:
            pen += 2.5 * behind
            why.append("little space in front of the face")
    for (jx, jy) in subject.get("joints", []):
        jq = _where((jx, jy), crop)
        for edge in (jq[0], 1.0 - jq[0], jq[1], 1.0 - jq[1]):
            if -0.01 < edge < 0.035:
                pen += 1.5
                why.append("an edge cuts at a joint")
                break
    for (nx, ny) in subject.get("necks", []):
        nq = _where((nx, ny), crop)
        if 0.0 < nq[0] < 1.0 and (abs(nq[1]) < 0.04 or abs(1.0 - nq[1]) < 0.04):
            pen += 2.0
            why.append("an edge cuts at the neck")
    pen -= 0.6 * math.sqrt(max(cw * ch, 1e-9))                          # larger is better, all else equal
    return pen, why


def _place(main, pt, cw, ch):
    x0 = min(max(main[0] - pt[0] * cw, 0.0), 1.0 - cw)
    y0 = min(max(main[1] - pt[1] * ch, 0.0), 1.0 - ch)
    return (x0, y0, x0 + cw, y0 + ch)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _place_name(q) -> str:
    h = "left" if q[0] < 0.45 else ("right" if q[0] > 0.55 else "centre")
    v = "upper" if q[1] < 0.45 else ("lower" if q[1] > 0.55 else "middle")
    return "the centre" if h == "centre" and v == "middle" else f"the {v} {h}" if h != "centre" else f"the {v} centre"


def suggest(subject: dict, frame_aspect: float, ratio: float, method: str | None = None, n: int = 3) -> list:
    """Up to ``n`` different crops [{"crop": (x0, y0, x1, y1), "method", "orient", "score", "text"}] in fractions of
    the frame, best first. ``ratio``: the crop's width / height in pixels. ``method``: one of METHODS, or None for
    all of them."""
    methods = [method] if method in METHODS else METHODS
    cands = []
    for m in methods:
        for o in range(ORIENTATIONS.get(m, 1)):
            for s in (1.0, 0.86, 0.74, 0.64, 0.55, 0.47):
                cw, ch = _rect_for(ratio, frame_aspect, s)
                aspect = ratio
                for pt in points(m, aspect, o) or [(0.5, 0.5)]:
                    crop = _place(subject["main"], pt, cw, ch)
                    pen, why = score(crop, subject, m, aspect, o)
                    cands.append((pen, crop, m, o, s, why))
    cands.sort(key=lambda c: c[0])
    out = []
    for pen, crop, m, o, s, why in cands:
        if any(_iou(crop, c["crop"]) > 0.72 for c in out):
            continue
        q = _where(subject["main"], crop)
        size = "the whole frame" if s > 0.95 else ("a loose crop" if s > 0.7 else ("a medium crop" if s > 0.56
                                                                                   else "a tight crop"))
        what = subject.get("what", "the subject")
        text = f"{NAMES[m]}: {what} on {_place_name(q)}, {size}"
        if why:
            text += f" (but {why[0]})"
        out.append({"crop": tuple(round(float(v), 5) for v in crop), "method": m, "orient": o,
                    "score": round(float(pen), 3), "text": text})
        if len(out) >= n:
            break
    return out


def saliency(rgb8: np.ndarray) -> np.ndarray:
    """Where a small photo (uint8 or 0..1 RGB) stands out and is sharp, 0..1, same size. Spectral residual (Hou and
    Zhang, CVPR 2007) on a 64-pixel copy, times local sharpness (so a blurred background counts less), times
    colour contrast with the photo's average, with a gentle pull towards the centre."""
    from scipy.ndimage import gaussian_filter, laplace, uniform_filter, zoom
    a = np.asarray(rgb8, np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    h, w = a.shape[:2]
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    k = 64.0 / max(h, w)
    small = zoom(lum, k, order=1)
    F = np.fft.fft2(small)
    logamp = np.log(np.abs(F) + 1e-6)
    resid = logamp - uniform_filter(logamp, 3, mode="nearest")
    sr = np.abs(np.fft.ifft2(np.exp(resid + 1j * np.angle(F)))) ** 2
    sr = gaussian_filter(sr, 2.5)
    sr = zoom(sr, (h / sr.shape[0], w / sr.shape[1]), order=1)[:h, :w]
    if sr.shape != (h, w):
        sr = np.pad(sr, ((0, h - sr.shape[0]), (0, w - sr.shape[1])), mode="edge")
    sharp = gaussian_filter(np.abs(laplace(gaussian_filter(lum, 0.8))), max(h, w) / 60.0)
    mean = a.reshape(-1, 3).mean(0)
    col = gaussian_filter(np.sqrt(((a - mean) ** 2).sum(-1)), max(h, w) / 80.0)

    def nrm(x):
        lo, hi = np.percentile(x, 2), np.percentile(x, 99.5)
        return np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    yy, xx = np.mgrid[0:h, 0:w]
    centre = np.exp(-(((xx / w - 0.5) / 0.45) ** 2 + ((yy / h - 0.5) / 0.45) ** 2))
    m = nrm(sr) * (0.35 + nrm(sharp)) * (0.5 + nrm(col)) * (0.75 + 0.25 * centre)
    return nrm(gaussian_filter(m, max(h, w) / 50.0)).astype(np.float32)


def salient_subject(rgb8: np.ndarray) -> dict:
    """A subject from saliency alone (no face, no person): its weighted centre and the box of its strongest part."""
    s = saliency(rgb8)
    h, w = s.shape
    from scipy.ndimage import label
    mask = s > 0.5
    lab, n = label(mask)
    if n:
        iy, ix = np.unravel_index(int(np.argmax(s)), s.shape)
        keep = lab == lab[iy, ix]
    else:
        keep = s >= s.max() * 0.9
    ys, xs = np.nonzero(keep)
    wts = s[keep]
    cx, cy = float((xs * wts).sum() / wts.sum() + 0.5) / w, float((ys * wts).sum() / wts.sum() + 0.5) / h
    box = (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)
    return {"main": (cx, cy), "box": box, "what": "the subject"}
