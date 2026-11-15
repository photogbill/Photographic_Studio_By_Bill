"""Straight lines in a photo: automatic straighten (level) and profile-free distortion measurement.

Line support regions in the spirit of LSD (von Gioi, Jakubowicz, Morel, Randall, IEEE TPAMI 2010): thin edges
(non-maximum suppression of the gradient) are grouped by edge orientation into connected regions; each region is
fitted with a line. Long, thin regions are line segments.

Distortion, plumb-line method (Devernay and Faugeras, "Straight lines have to be straight", Machine Vision and
Applications 13(1), 2001): find the radial distortion that makes the long segments straightest. Plan 10.2 rule:
never guess distortion on organic subjects. So the answer is "none" unless there are several long segments, the
correction makes them clearly straighter, they agree on the direction of the distortion, and they end up
genuinely straight (sub-pixel residual).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, label, sobel

from ... import log as slog
from .geometry import SQRT_3_25

log = slog.get("engine.lines")


def segments(Y: np.ndarray, min_len: float = 40.0, max_segments: int = 400) -> list:
    """Line segments in a luminance image. Each: dict(points=(N,2) x,y float, angle_deg (math convention, y up,
    -90..90], length px, rms px)."""
    Z = np.sqrt(np.maximum(np.asarray(Y, np.float32), 0.0))
    Z = gaussian_filter(Z, 1.0)
    gx, gy = sobel(Z, 1), sobel(Z, 0)
    mag = np.hypot(gx, gy)
    h, w = Z.shape
    thr = max(float(np.percentile(mag, 90)), 4.0 * float(np.median(mag)) + 1e-6)
    q = (np.round(np.arctan2(gy, gx) / (np.pi / 4)) % 4).astype(np.int8)
    nms = np.zeros((h, w), bool)
    c = mag[1:-1, 1:-1]
    for k, (dy, dx) in enumerate(((0, 1), (1, 1), (1, 0), (1, -1))):
        a = mag[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        b = mag[1 - dy:h - 1 - dy, 1 - dx:w - 1 - dx]
        nms[1:-1, 1:-1] |= (q[1:-1, 1:-1] == k) & (c > a) & (c >= b)
    edge = nms & (mag > thr)
    edge[:3] = edge[-3:] = False
    edge[:, :3] = edge[:, -3:] = False
    # sub-pixel edge position: parabola through the gradient magnitude across the edge
    ey, ex = np.nonzero(edge)
    nx, ny = gx[ey, ex], gy[ey, ex]
    nn = np.maximum(np.hypot(nx, ny), 1e-12)
    nx, ny = nx / nn, ny / nn
    from scipy.ndimage import map_coordinates
    m0 = mag[ey, ex]
    mp = map_coordinates(mag, [ey + ny, ex + nx], order=1)
    mm = map_coordinates(mag, [ey - ny, ex - nx], order=1)
    den = mp - 2 * m0 + mm
    off = np.where(np.abs(den) > 1e-12, 0.5 * (mm - mp) / np.where(np.abs(den) > 1e-12, den, 1.0), 0.0)
    off = np.clip(off, -0.5, 0.5)
    subx = np.zeros((h, w), np.float64)
    suby = np.zeros((h, w), np.float64)
    subx[ey, ex] = ex + off * nx
    suby[ey, ex] = ey + off * ny
    # edge direction (along the edge), 0..180
    ori = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0
    used = np.zeros((h, w), bool)
    cands = []
    width = 22.5
    for b0 in np.arange(0.0, 180.0, width / 2):
        d = np.abs(((ori - b0 - width / 2) + 90.0) % 180.0 - 90.0)
        m = edge & (d <= width / 2)
        lab, n = label(m, structure=np.ones((3, 3)))
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        objs = find_objects(lab)
        for idx in np.nonzero(sizes >= max(8, min_len * 0.4))[0]:
            if idx == 0:
                continue
            sl = objs[idx - 1]
            yy, xx = np.nonzero(lab[sl] == idx)
            cands.append((int(sizes[idx]), yy + sl[0].start, xx + sl[1].start))
    cands.sort(key=lambda t: -t[0])
    out = []
    for size, ys, xs in cands:
        if used[ys, xs].mean() > 0.5:
            continue
        P = np.stack([subx[ys, xs], suby[ys, xs]], 1)
        sg = _seg(P)
        if sg["length"] < min_len * 0.5 or sg["rms"] > max(1.5, 0.012 * sg["length"]) or size < 0.6 * sg["length"]:
            continue
        used[ys, xs] = True
        out.append(sg)
        if len(out) >= max_segments * 3:
            break
    out = [sg for sg in merge_collinear(out) if sg["length"] >= min_len]
    out.sort(key=lambda s: -s["length"])
    return out[:max_segments]


def _seg(P: np.ndarray) -> dict:
    mu = P.mean(0)
    _, _, Vt = np.linalg.svd(P - mu, full_matrices=False)
    dirv = Vt[0]
    t = (P - mu) @ dirv
    ang = math.degrees(math.atan2(-dirv[1], dirv[0]))
    ang = (ang + 90.0) % 180.0 - 90.0
    i0, i1 = int(np.argmin(t)), int(np.argmax(t))
    return {"points": P, "angle_deg": ang, "length": float(t.max() - t.min()),
            "rms": float(np.sqrt(np.mean(((P - mu) @ Vt[1]) ** 2))), "ends": (P[i0], P[i1])}


def merge_collinear(segs: list, max_gap: float = 14.0, max_angle: float = 4.0, max_offset: float = 2.5) -> list:
    """Join pieces of one line that were split at junctions (grid crossings, window bars) or by slight curvature."""
    segs = list(segs)
    changed = True
    while changed:
        changed = False
        segs.sort(key=lambda s: -s["length"])
        i = 0
        while i < len(segs):
            a = segs[i]
            j = i + 1
            while j < len(segs):
                b = segs[j]
                da = abs(((a["angle_deg"] - b["angle_deg"]) + 90.0) % 180.0 - 90.0)
                if da > max_angle:
                    j += 1
                    continue
                gap = min(np.hypot(*(ea - eb)) for ea in a["ends"] for eb in b["ends"])
                if gap > max_gap:
                    j += 1
                    continue
                # b's nearest end must lie on a's line (continuation, not a parallel neighbour)
                ea = min(a["ends"], key=lambda e: min(np.hypot(*(e - eb)) for eb in b["ends"]))
                eb = min(b["ends"], key=lambda e: np.hypot(*(e - ea)))
                th = math.radians(a["angle_deg"])
                nrm = np.array([math.sin(th), math.cos(th)])          # unit normal (image coords, y down)
                if abs(float((eb - ea) @ nrm)) > max_offset + 0.15 * gap:
                    j += 1
                    continue
                m = _seg(np.concatenate([a["points"], b["points"]]))
                if m["rms"] > max(1.5, 0.012 * m["length"]):
                    j += 1
                    continue
                segs[i] = a = m
                segs.pop(j)
                changed = True
            i += 1
    return segs


def auto_level(segs: list, frame_wh, max_tilt: float = 12.0) -> dict | None:
    """Rotation (degrees, + = counter-clockwise, the Edit convention) that levels the photo, from near-horizontal
    and near-vertical segments. None when the photo has no trustworthy lines."""
    if not segs:
        return None
    devs, wts = [], []
    for s in segs:
        a = s["angle_deg"]
        d = a if abs(a) <= max_tilt else ((a - 90.0) if a > 0 else (a + 90.0)) if abs(abs(a) - 90.0) <= max_tilt else None
        if d is None:
            continue
        devs.append(d)
        wts.append(s["length"])
    if not devs:
        return None
    devs, wts = np.array(devs), np.array(wts)
    o = np.argsort(devs)
    cw = np.cumsum(wts[o])
    med = float(devs[o][np.searchsorted(cw, cw[-1] / 2.0)])
    mad = float(np.sum(wts * np.abs(devs - med)) / np.sum(wts))
    total = float(wts.sum())
    need = 0.35 * (frame_wh[0] + frame_wh[1])
    if total < need or mad > 1.5:
        log.debug("auto level: not trustworthy (length %.0f of %.0f needed, spread %.2f deg)", total, need, mad)
        return None
    return {"rotate": round(-med, 2), "lines": int(len(devs)), "spread_deg": round(mad, 2),
            "confidence": "good" if mad < 0.6 and total > 2 * need else "fair"}


def _undistort_points(P, k, cx, cy, rn):
    """Invert Rd = Ru (1 - k + k Ru^2) for points P (x, y) by Newton steps."""
    d = (P - [cx, cy]) / rn
    rd = np.hypot(d[:, 0], d[:, 1])
    ru = rd.copy()
    for _ in range(8):
        f = ru * (1 - k + k * ru * ru) - rd
        fp = 1 - k + 3 * k * ru * ru
        ru = np.clip(ru - f / np.where(np.abs(fp) < 1e-3, 1e-3, fp), 0.0, 4.0)
    s = np.where(rd > 1e-9, ru / np.maximum(rd, 1e-9), 1.0)
    return d * s[:, None] * rn + [cx, cy]


def _valid_k(k, rmax):
    """The poly3 model must stay monotone out to the corners (no fold-over)."""
    return 1 - k + 3 * k * rmax * rmax > 0.2 and 1 - k > 0.2


def _line_cost(P):
    mu = P.mean(0)
    _, S, _ = np.linalg.svd(P - mu, full_matrices=False)
    return float(S[1] ** 2)


def measure_distortion(segs: list, frame_wh, min_segments: int = 4) -> dict | None:
    """Plumb-line estimate of poly3 distortion (Edit convention: the lens's own distortion; k < 0 = barrel)."""
    w, h = frame_wh
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rn = math.hypot(w, h) / 2.0 / SQRT_3_25
    short = min(w, h)
    long_segs = []
    for s in segs:
        if s["length"] < 0.12 * short:
            continue
        P = s["points"]
        # a segment through the centre says nothing about radial distortion
        mid = P.mean(0)
        if math.hypot(mid[0] - cx, mid[1] - cy) < 0.15 * short:
            continue
        if len(P) > 600:
            P = P[np.linspace(0, len(P) - 1, 600).astype(int)]
        long_segs.append(P)
    if len(long_segs) < min_segments:
        return None
    rmax = math.hypot(w, h) / 2.0 / rn * 1.05
    ks = np.array([k for k in np.linspace(-0.12, 0.12, 49) if _valid_k(k, rmax)])

    def total(k):
        if not _valid_k(k, rmax):
            return float("inf")
        return sum(_line_cost(_undistort_points(P, k, cx, cy, rn)) for P in long_segs)
    costs = np.array([total(k) for k in ks])
    i = int(np.argmin(costs))
    a, b = ks[max(0, i - 1)], ks[min(len(ks) - 1, i + 1)]
    for _ in range(30):                                         # golden-section refine
        m1, m2 = a + 0.382 * (b - a), a + 0.618 * (b - a)
        a, b = (a, m2) if total(m1) < total(m2) else (m1, b)
    k = 0.5 * (a + b)
    c0, ck = total(0.0), total(k)
    if c0 <= 0:
        return None
    gain = 1.0 - ck / c0
    # per-segment agreement on the sign of k, and genuine straightness after correction
    agree, wsum, resid = 0.0, 0.0, []
    for P in long_segs:
        cpos, cneg = _line_cost(_undistort_points(P, abs(k), cx, cy, rn)), _line_cost(_undistort_points(P, -abs(k), cx, cy, rn))
        better_sign = 1.0 if cpos < cneg else -1.0
        wt = len(P)
        agree += wt * (better_sign == np.sign(k))
        wsum += wt
        resid.append(math.sqrt(_line_cost(_undistort_points(P, k, cx, cy, rn)) / len(P)))
    agree /= max(wsum, 1)
    rmed = float(np.median(resid))
    ok = abs(k) >= 0.004 and gain >= 0.35 and agree >= 0.7 and rmed <= 0.8
    log.debug("plumb-line: k=%.4f gain=%.2f agree=%.2f resid=%.2f px segments=%d -> %s", k, gain, agree, rmed,
              len(long_segs), "use" if ok else "reject")
    if not ok:
        return None
    return {"model": "poly3", "terms": [round(float(k), 5)], "norm": 1.0,
            "source": "measured from straight lines in this photo", "lines": len(long_segs),
            "confidence": "good" if gain > 0.6 and len(long_segs) >= 8 else "fair"}
