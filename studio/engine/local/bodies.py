"""Body parts found, for body sculpting (BUILD_PLAN_PORTRAIT.md H.3).

Find bodies stores, on the photo's "portrait", what the Person finder saw (local/people.py, pipeline.find_people):

    "bodies": {"w", "h", "data": person mask on a small grid of the S frame (masks.encode_bitmap),
               "people": [[[x, y, how sure], ... 33 BlazePose points], ...] (fractions of the S frame)}

From that, at any time, ``find_parts`` works out each part of each person in the S frame: a centre line (from one
end ``a`` to the other ``b``) and how far the body's outline is from it on each side. The outline is measured where
the person mask ends, looking across the centre line at several places along the part ("measured", cyan); where it
cannot be seen (an arm pressed to the side, the thighs touching, a part in loose clothing) it is a share of the
shoulder width instead ("estimated", amber). A scan across the torso stops where an arm starts, so a hanging arm is
not counted in the waist.

Parts: upper arm, forearm, thigh and calf on each side; neck, shoulders, waist (the narrowest place between ribs and
hips), hips; tummy (side-on only: just the front edge moves); chest and seat (these place the existing Breast and Butt
lift shapes, since a lift is not a slim). Whether the person faces the camera or is side-on is read from how wide
the shoulders are against the torso's length (the points' 2D layout; the stored points carry no depth).

A part becomes a sculpt item that carries its own geometry, so the edit renders the same way later:

    {"kind": "part", "part": "thigh", "name": "Thigh (their left)", "slim": -100..100, "enabled": true,
     "geo": [{"a": [x, y], "b": [x, y], "wl": .., "wr": .. (fractions of the long edge), "fl": 0..1, "fr": 0..1,
              "t0": .., "t1": ..}, ...],      one per side when "Both sides" is on
     "guard": [[x, y, r], ...]}              faces and hands: never moved

    {"kind": "legs", "h": [x, y] hips, "f": [x, y] feet, "w": half width (long edge), "amount": 0..100}

portrait.warps turns these into warps ("outline", "stretch"). Everything here is plain geometry on stored data; no
model runs at render time.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import map_coordinates

from ... import log as slog
from . import people as pp

log = slog.get("engine.bodies")

MEASURED, ESTIMATED = "measured", "estimated"
SLIM_PARTS = ("upper_arm", "forearm", "thigh", "calf", "neck", "shoulders", "waist", "hips", "tummy")
LIFT_PARTS = ("chest", "seat")
PART_NAMES = {"upper_arm": "Upper arm", "forearm": "Forearm", "thigh": "Thigh", "calf": "Calf", "neck": "Neck",
              "shoulders": "Shoulders", "waist": "Waist", "hips": "Hips", "tummy": "Tummy", "chest": "Chest",
              "seat": "Seat"}
# limb: (part, joint a, joint b, default half width as a share of the shoulder width)
LIMBS = (("upper_arm", pp.L_SH, pp.L_EL, 0.13, "L"), ("upper_arm", pp.R_SH, pp.R_EL, 0.13, "R"),
         ("forearm", pp.L_EL, pp.L_WR, 0.10, "L"), ("forearm", pp.R_EL, pp.R_WR, 0.10, "R"),
         ("thigh", pp.L_HIP, pp.L_KN, 0.20, "L"), ("thigh", pp.R_HIP, pp.R_KN, 0.20, "R"),
         ("calf", pp.L_KN, pp.L_AN, 0.13, "L"), ("calf", pp.R_KN, pp.R_AN, 0.13, "R"))
SLIM_CAP = 0.12            # a slim of 100 moves the outline 12% of its distance from the centre line
LEGS_CAP = 0.06            # longer legs at 100: 6% longer


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def decode_mask(store: dict) -> np.ndarray:
    from .masks import decode_bitmap
    return decode_bitmap(store["data"], int(store["w"]), int(store["h"])).astype(np.float32) / 255.0


class _Mask:
    """The person mask, sampled at S pixels."""

    def __init__(self, m: np.ndarray, sw: int, sh: int):
        self.m, self.sw, self.sh = m, sw, sh

    def at(self, X, Y):
        h, w = self.m.shape
        fx = (np.asarray(X, float) + 0.5) / self.sw * w - 0.5
        fy = (np.asarray(Y, float) + 0.5) / self.sh * h - 0.5
        return map_coordinates(self.m, [fy, fx], order=1, mode="constant", cval=0.0)


def _seg_dist(X, Y, a, b):
    d = b - a
    n = float(d @ d)
    t = np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / max(n, 1e-9), 0.0, 1.0)
    return np.hypot(X - (a[0] + t * d[0]), Y - (a[1] + t * d[1]))


class Body:
    """One person's points in S pixels, with the frame the parts are measured in."""

    def __init__(self, P, sw: int, sh: int, index: int = 0):
        self.P = np.asarray(P, float).reshape(33, 3)
        self.sw, self.sh, self.index = sw, sh, index
        self.long = float(max(sw, sh))
        self.pts = self.P[:, :2] * np.array([sw, sh], float)
        self.vis = self.P[:, 2]
        up, shw, head, hw = pp._person_frame(self.P, sw, sh)
        if self.ok(pp.L_SH, pp.R_SH, pp.L_HIP, pp.R_HIP):    # side-on the shoulders look narrow: scale by the torso
            sc = self.pts[[pp.L_SH, pp.R_SH]].mean(axis=0)
            T = float(np.hypot(*(sc - self.pts[[pp.L_HIP, pp.R_HIP]].mean(axis=0))))
            shw = max(shw, 0.7 * T)
        self.up, self.shw, self.head, self.hw = up, shw, head, max(hw, 0.3 * shw)

    def ok(self, *idx) -> bool:
        return all(self.vis[i] > pp.VISIBLE for i in idx)

    def arm_tubes(self):
        """(a, b, radius) of each visible upper arm and forearm (S pixels): a torso scan stops where these start."""
        out = []
        for part, i, j, r, _ in LIMBS[:4]:
            if self.ok(i, j):
                out.append((self.pts[i], self.pts[j], r * self.shw))
        return out

    def view(self) -> tuple[str, float]:
        """("front" | "turned" | "side" | "unknown", which way the person faces across the torso: +1 toward the
        centre line's left normal, -1 the other way, 0 when facing the camera)."""
        if not self.ok(pp.L_SH, pp.R_SH):
            return "unknown", 0.0
        sc = self.pts[[pp.L_SH, pp.R_SH]].mean(axis=0)
        width = float(np.hypot(*(self.pts[pp.L_SH] - self.pts[pp.R_SH])))
        if self.ok(pp.L_HIP, pp.R_HIP):
            length = float(np.hypot(*(sc - self.pts[[pp.L_HIP, pp.R_HIP]].mean(axis=0))))
        else:
            length = 1.6 * max(width, self.hw * 1.6)
        r = width / max(length, 1e-6)
        kind = "front" if r > 0.55 else ("side" if r < 0.35 else "turned")
        d = self.up                                   # centre line direction (hips to shoulders)
        n = np.array([-d[1], d[0]])
        off = float((self.pts[pp.NOSE] - sc) @ n) / max(self.hw, 1e-6) if self.vis[pp.NOSE] > 0.3 else 0.0
        face = 0.0 if kind == "front" and abs(off) < 0.25 else float(np.sign(off))
        return kind, face


def _scan(mask: _Mask, p, n, maxd: float, step: float, block=None):
    """How far from ``p`` along ``n`` the person mask ends (S pixels), or (maxd, False) when it does not end within
    maxd (or p is outside the person). ``block``: tubes [(a, b, r)] that also end the scan (arms)."""
    d = np.arange(0.0, maxd + step, step)
    X, Y = p[0] + n[0] * d, p[1] + n[1] * d
    m = mask.at(X, Y)
    if m[0] < 0.5:
        return maxd, False
    inside = m >= 0.5
    if block:
        for a, b, r in block:
            inside &= ~(_seg_dist(X, Y, a, b) < 0.9 * r)
        if not inside[0]:
            return maxd, False
    out = np.nonzero(~inside)[0]
    if out.size == 0:
        return maxd, False
    k = int(out[0])
    if m[k] < 0.5 and k > 0:                          # between two samples: where it crosses one half
        f = (m[k - 1] - 0.5) / max(m[k - 1] - m[k], 1e-6)
        return float(d[k - 1] + f * step), True
    return float(d[k]), True


def _edges(mask, a, b, ts, default: float, maxd: float, step: float, block=None):
    """Median distance from the centre line a->b to the outline on its left and right, over positions ts (0..1);
    each side MEASURED when most positions saw the edge, else the default (ESTIMATED)."""
    d = b - a
    L = float(np.hypot(*d))
    d = d / max(L, 1e-9)
    n = np.array([-d[1], d[0]])
    res = []
    for sgn in (1.0, -1.0):
        got = []
        for t in ts:
            p = a + d * L * t
            v, ok = _scan(mask, p, sgn * n, maxd, step, block)
            if ok:
                got.append(v)
        if len(got) >= max(1, (len(ts) + 1) // 2):
            res.append((float(np.median(got)), MEASURED))
        else:
            res.append((default, ESTIMATED))
    return res


def _in_frame(body, a, b, t0, t1) -> bool:
    """The middle of the part's working stretch is in the photo (a hip line below the frame's edge is not a part)."""
    m = a + (b - a) * 0.5 * (t0 + t1)
    return 0.0 <= m[0] < body.sw and 0.0 <= m[1] < body.sh


def _part(body, part, side, a, b, edges, t0, t1, view="", fl=1.0, fr=1.0, note=""):
    sw, sh, long = body.sw, body.sh, body.long
    (wl, hl), (wr, hr) = edges
    sname = {"L": " (their left)", "R": " (their right)"}.get(side, "")
    pid = f"p{body.index}.{part}" + (f".{side}" if side else "")
    return {"id": pid, "person": body.index, "part": part, "side": side, "name": PART_NAMES[part] + sname,
            "a": [round(float(a[0]) / sw, 6), round(float(a[1]) / sh, 6)],
            "b": [round(float(b[0]) / sw, 6), round(float(b[1]) / sh, 6)],
            "wl": round(wl / long, 6), "wr": round(wr / long, 6), "how_l": hl, "how_r": hr,
            "fl": float(fl), "fr": float(fr), "t0": float(t0), "t1": float(t1), "view": view, "note": note}


def find_parts(people: list, mask: np.ndarray, sw: int, sh: int) -> list:
    """Every part of every person found (see the module notes), in S-frame fractions."""
    M = _Mask(np.asarray(mask, np.float32), sw, sh)
    step = max(1.0, max(sw, sh) / float(max(mask.shape)) * 0.5)
    out = []
    for i, P in enumerate(people or []):
        try:
            out.extend(_parts_of(Body(P, sw, sh, i), M, step))
        except (ValueError, IndexError, FloatingPointError):
            log.warning("bodies: person %d could not be measured", i, exc_info=True)
    log.info("bodies: %d part(s) on %d person(s)", len(out), len(people or []))
    return out


def _parts_of(body: Body, M: _Mask, step: float) -> list:
    return [q for q in _all_parts(body, M, step) if _in_frame(body, np.asarray(q["a"]) * [body.sw, body.sh],
                                                               np.asarray(q["b"]) * [body.sw, body.sh],
                                                               q["t0"], q["t1"])]


def _all_parts(body: Body, M: _Mask, step: float) -> list:
    out = []
    shw = body.shw
    view, face = body.view()
    ts = (0.3, 0.4, 0.5, 0.6, 0.7)
    for part, i, j, r, side in LIMBS:
        if not body.ok(i, j):
            continue
        a, b = body.pts[i], body.pts[j]
        if np.hypot(*(b - a)) < 4 * step:
            continue
        e = _edges(M, a, b, ts, r * shw, 2.4 * r * shw, step)
        out.append(_part(body, part, side, a, b, e, 0.22, 0.78, view))
    if not body.ok(pp.L_SH, pp.R_SH):
        return out
    sc = body.pts[[pp.L_SH, pp.R_SH]].mean(axis=0)
    u = body.up
    tubes = body.arm_tubes()
    # neck: from the shoulder line to the chin (about 60% of the way up to the ears)
    hc = sc + 0.6 * (body.head - sc)
    if np.hypot(*(hc - sc)) > 4 * step:
        e = _edges(M, sc, hc, (0.45, 0.6, 0.75), 0.17 * shw, 0.5 * shw, step)
        out.append(_part(body, "neck", "", sc, hc, e, 0.35, 0.8, view))
    # shoulders: the outline at the shoulder line, on a short stretch of the torso's centre line
    if body.ok(pp.L_HIP, pp.R_HIP):
        hipc = body.pts[[pp.L_HIP, pp.R_HIP]].mean(axis=0)
        T = float(np.hypot(*(sc - hipc)))
    else:
        hipc, T = None, 1.6 * shw
    sa, sb = sc - u * 0.32 * T, sc + u * 0.10 * T
    e = _edges(M, sa, sb, (0.6, 0.7, 0.8), 0.62 * shw, 1.0 * shw, step)
    out.append(_part(body, "shoulders", "", sa, sb, e, 0.45, 0.8, view))
    if hipc is None:
        return out
    # the torso, measured every 5% from the hips (0) to the shoulders (1), arms excluded
    tq = np.arange(-0.1, 0.71, 0.05)
    widths = []
    d = (sc - hipc) / T
    n = np.array([-d[1], d[0]])
    for t in tq:
        p = hipc + d * T * t
        l, okl = _scan(M, p, n, 1.0 * shw, step, tubes)
        r_, okr = _scan(M, p, -n, 1.0 * shw, step, tubes)
        widths.append((l, okl, r_, okr))
    meas = [(t, w) for t, w in zip(tq, widths) if 0.15 <= t <= 0.6 and w[1] and w[3]]
    if meas:
        tw, (wl, _, wr, _) = min(meas, key=lambda q: q[1][0] + q[1][2])
        we = ((wl, MEASURED), (wr, MEASURED))
    else:
        tw = 0.4
        we = ((0.38 * shw, ESTIMATED), (0.38 * shw, ESTIMATED))
    note = "" if view == "front" else "Waist works best facing the camera."
    wa, wb = hipc + d * T * (tw - 0.2), hipc + d * T * (tw + 0.2)
    out.append(_part(body, "waist", "", wa, wb, we, 0.35, 0.65, view, note=note))
    ha, hb = hipc - d * T * 0.3, hipc + d * T * 0.14
    e = _edges(M, ha, hb, (0.45, 0.55, 0.65), 0.45 * shw, 1.1 * shw, step, tubes)
    out.append(_part(body, "hips", "", ha, hb, e, 0.35, 0.75, view))
    if view in ("side", "turned") and face != 0.0:
        ta, tb = hipc + d * T * (tw - 0.28), hipc + d * T * (tw + 0.12)
        e = _edges(M, ta, tb, (0.4, 0.5, 0.6), 0.35 * shw, 1.0 * shw, step, tubes)
        fl, fr = (1.0, 0.0) if face > 0 else (0.0, 1.0)
        out.append(_part(body, "tummy", "", ta, tb, e, 0.3, 0.7, view, fl, fr))
        ca, cb = hipc + d * T * 0.55, hipc + d * T * 0.9
        e = _edges(M, ca, cb, (0.4, 0.5, 0.6), 0.35 * shw, 1.0 * shw, step, tubes)
        out.append(_part(body, "chest", "", ca, cb, e, 0.3, 0.7, view, fl, fr))
        sa2, sb2 = hipc - d * T * 0.25, hipc + d * T * 0.12
        e = _edges(M, sa2, sb2, (0.4, 0.5, 0.6), 0.4 * shw, 1.0 * shw, step, tubes)
        out.append(_part(body, "seat", "", sa2, sb2, e, 0.3, 0.7, view, fr, fl))
    elif view == "front":
        ca, cb = hipc + d * T * 0.55, hipc + d * T * 0.9
        e = _edges(M, ca, cb, (0.4, 0.5, 0.6), 0.4 * shw, 1.0 * shw, step, tubes)
        out.append(_part(body, "chest", "", ca, cb, e, 0.3, 0.7, view))
    return out


# ------------------------------------------------------------------------------------------ for the screen
def outline(part: dict, sw: int, sh: int, n: int = 12) -> np.ndarray:
    """The part's outline (S pixels, closed polygon): its edges along the stretch the warp works on, tapering
    toward the joints, so the person sees what a click will change."""
    a = np.asarray(part["a"], float) * [sw, sh]
    b = np.asarray(part["b"], float) * [sw, sh]
    long = float(max(sw, sh))
    d = b - a
    L = float(np.hypot(*d))
    d = d / max(L, 1e-9)
    nv = np.array([-d[1], d[0]])
    t = np.linspace(0.0, 1.0, n)
    t0, t1 = float(part.get("t0", 0.2)), float(part.get("t1", 0.8))
    k = np.where(t < t0, 0.75 + 0.25 * _smooth(t / max(t0, 1e-6)),
                 np.where(t > t1, 0.75 + 0.25 * _smooth((1 - t) / max(1 - t1, 1e-6)), 1.0))
    wl, wr = float(part["wl"]) * long, float(part["wr"]) * long
    left = a[None] + (t * L)[:, None] * d + (k * wl)[:, None] * nv
    right = a[None] + (t * L)[:, None] * d - (k * wr)[:, None] * nv
    return np.vstack([left, right[::-1]])


def how(part: dict) -> str:
    return MEASURED if part.get("how_l") == MEASURED and part.get("how_r") == MEASURED else ESTIMATED


def _inside(poly: np.ndarray, x: float, y: float) -> bool:
    """Even-odd point in polygon."""
    xs, ys = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    cross = ((ys > y) != (yj > y)) & (x < (xj - xs) * (y - ys) / np.where(yj - ys == 0, 1e-12, yj - ys) + xs)
    return bool(np.count_nonzero(cross) % 2)


def part_at(parts: list, x: float, y: float, sw: int, sh: int):
    """The smallest part whose outline holds S-fraction point (x, y), or None."""
    best, area = None, None
    for p in parts:
        poly = outline(p, sw, sh)
        if _inside(poly, x * sw, y * sh):
            ar = 0.5 * abs(float(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1))))
            if area is None or ar < area:
                best, area = p, ar
    return best


# ------------------------------------------------------------------------------------------ sculpt items
def _geo(p: dict) -> dict:
    return {k: p[k] for k in ("a", "b", "wl", "wr", "fl", "fr", "t0", "t1")}


def _partner(part: dict, parts: list):
    if not part.get("side"):
        return None
    other = {"L": "R", "R": "L"}[part["side"]]
    for q in parts:
        if q["person"] == part["person"] and q["part"] == part["part"] and q.get("side") == other:
            return q
    return None


def guards(people: list, person: int, sw: int, sh: int) -> list:
    """Circles (S fractions, radius as a share of the long edge) that body warps never move: the face and hands."""
    if not (0 <= person < len(people or [])):
        return []
    b = Body(people[person], sw, sh, person)
    long = b.long
    out = [[round(float(b.head[0]) / sw, 6), round(float(b.head[1]) / sh, 6), round(1.15 * b.hw / long, 6)]]
    for wr, ix in ((pp.L_WR, pp.L_IDX), (pp.R_WR, pp.R_IDX)):
        if b.ok(wr):
            q = b.pts[ix] if b.ok(ix) else b.pts[wr]
            c = 0.5 * (b.pts[wr] + q)
            out.append([round(float(c[0]) / sw, 6), round(float(c[1]) / sh, 6), round(0.17 * b.shw / long, 6)])
    return out


def item_for(part: dict, parts: list, people: list, sw: int, sh: int, mirror: bool = False, slim: float = 30.0):
    """A sculpt item for a found part (slim parts only; chest and seat go through lift_shapes)."""
    geo = [_geo(part)]
    name = part["name"]
    if mirror:
        q = _partner(part, parts)
        if q is not None:
            geo.append(_geo(q))
            name = PART_NAMES[part["part"]] + " (both)"
    return {"kind": "part", "part": part["part"], "id": part["id"] + (".both" if len(geo) > 1 else ""),
            "name": name, "slim": float(slim), "geo": geo, "guard": guards(people, part["person"], sw, sh),
            "how": how(part), "enabled": True}


def lift_shapes(part: dict, people: list, sw: int, sh: int) -> list:
    """The Breast (chest) or Butt (seat) lift shapes placed from a found part: two side by side when the person
    faces the camera, the front (or back) one when side-on. Placed, not measured: amber."""
    long = float(max(sw, sh))
    a = np.asarray(part["a"], float) * [sw, sh]
    b = np.asarray(part["b"], float) * [sw, sh]
    d = b - a
    L = float(np.hypot(*d))
    d = d / max(L, 1e-9)
    nv = np.array([-d[1], d[0]])
    wl, wr = float(part["wl"]) * long, float(part["wr"]) * long
    tool = "breast" if part["part"] == "chest" else "butt"
    c0 = a + d * L * 0.5
    if part.get("fl", 1.0) and part.get("fr", 1.0):              # facing the camera: one each side
        r = 0.32 * (wl + wr)
        centres = [c0 + nv * 0.48 * wl, c0 - nv * 0.48 * wr]
    else:
        side = 1.0 if part.get("fl", 1.0) else -1.0
        w = wl if side > 0 else wr
        r = 0.55 * w
        centres = [c0 + side * nv * 0.6 * w]
    out = []
    for c in centres:
        out.append({"kind": "shape", "tool": tool, "c": [round(float(c[0]) / sw, 6), round(float(c[1]) / sh, 6)],
                    "e1": [round(float(nv[0]), 5), round(float(nv[1]), 5)], "a": round(r / long, 6),
                    "b": round(0.85 * r / long, 6), "up": [round(float(d[0]), 5), round(float(d[1]), 5)],
                    "lift": 30, "size": 0, "slim": 0, "enabled": True, "placed": True})
    return out


def legs_item(people: list, person: int, sw: int, sh: int, amount: float = 30.0):
    """Longer legs for one person: needs the hips and at least one ankle in view. None otherwise."""
    if not (0 <= person < len(people or [])):
        return None
    b = Body(people[person], sw, sh, person)
    if not b.ok(pp.L_HIP, pp.R_HIP):
        return None
    ank = [b.pts[k] for k in (pp.L_AN, pp.R_AN) if b.ok(k)]
    if not ank:
        return None
    hc = b.pts[[pp.L_HIP, pp.R_HIP]].mean(axis=0)
    fc = np.mean(ank, axis=0)
    if np.hypot(*(fc - hc)) < 8:
        return None
    return {"kind": "legs", "person": person, "name": "Longer legs",
            "h": [round(float(hc[0]) / sw, 6), round(float(hc[1]) / sh, 6)],
            "f": [round(float(fc[0]) / sw, 6), round(float(fc[1]) / sh, 6)],
            "w": round(1.1 * b.shw / b.long, 6), "amount": float(amount), "enabled": True}


def manual_part(a_s, b_s, mask: np.ndarray | None, sw: int, sh: int) -> dict:
    """A part the person drew by dragging along a limb (S fractions): its edges measured from the person mask when
    there is one, else a third of its length each side (estimated)."""
    a = np.asarray(a_s, float) * [sw, sh]
    b = np.asarray(b_s, float) * [sw, sh]
    L = float(np.hypot(*(b - a)))
    long = float(max(sw, sh))
    default = 0.3 * L
    if mask is not None:
        M = _Mask(np.asarray(mask, np.float32), sw, sh)
        step = max(1.0, long / float(max(mask.shape)) * 0.5)
        e = _edges(M, a, b, (0.3, 0.4, 0.5, 0.6, 0.7), default, 1.2 * L, step)
    else:
        e = ((default, ESTIMATED), (default, ESTIMATED))
    (wl, hl), (wr, hr) = e
    return {"id": "drawn", "person": -1, "part": "drawn", "side": "", "name": "Drawn part",
            "a": [round(float(a_s[0]), 6), round(float(a_s[1]), 6)], "b": [round(float(b_s[0]), 6), round(float(b_s[1]), 6)],
            "wl": round(wl / long, 6), "wr": round(wr / long, 6), "how_l": hl, "how_r": hr, "fl": 1.0, "fr": 1.0,
            "t0": 0.2, "t1": 0.8, "view": "", "note": ""}


PART_NAMES["drawn"] = "Drawn part"
