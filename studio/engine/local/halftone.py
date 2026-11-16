"""Printing dots (halftone) on scans of printed photos: a newspaper cutting, a yearbook, a magazine page, a book
plate. A Cognitive Adjustment; no model.

    "halftone": {"on": bool, "amount": 0..100 (%)}

How
* A halftone screen is a regular grid of dots, so in the photo's spectrum it is a handful of sharp peaks: the
  screen's frequency in its two directions (a black and white screen is usually turned 45 degrees; colour prints
  have one screen per ink, each at its own angle) and their harmonics. The picture itself has a smooth spectrum.
* measure: on a full-resolution crop of the middle of the photo (up to 1024 x 1024, each colour channel and the
  luminance), the log power spectrum against its own median at each radius; local maxima standing more than
  15 dB above it, between 1/25 and 1/2 cycle per pixel, are the screen. Nothing found, nothing is changed.
* clean: each block (with context round it, mirrored at the edges) has soft notches cut at those frequencies
  and at their mirror images, the same in every block, so a 100% view matches the export. Only the screen's own
  frequencies go; the picture's detail at other frequencies stays.
* Where the preview is smaller than the screen's own period the dots are already averaged away: those notches
  are skipped at that scale.
No GUI imports (rule 2.2).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

DEFAULTS = {"on": False, "amount": 100.0}
CROP = 1024                 # side of the measured crop (full-resolution pixels)
FMIN, FMAX = 1.0 / 25.0, 0.5
PROMINENCE_DB = 15.0
SIGMA = 0.012               # notch width (cycles per pixel); its kernel is ~13 px wide, inside the context
MAX_PEAKS = 24


def settings(d: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (d or {}).items() if k in DEFAULTS})
    return out


def active(d: dict | None) -> bool:
    return bool(d and d.get("on"))


def pad_for() -> int:
    return 48


def _peaks_one(x: np.ndarray) -> list:
    """Screen peaks of one channel: [(fx, fy, prominence dB)] in cycles per pixel, half plane (fy > 0 or fy = 0 and
    fx > 0)."""
    h, w = x.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2((x - x.mean()) * win))
    P = np.log(np.abs(F) ** 2 + 1e-12)
    fy = (np.arange(h) - h // 2) / float(h)
    fx = (np.arange(w) - w // 2) / float(w)
    FX, FY = np.meshgrid(fx, fy)
    R = np.hypot(FX, FY)
    rb = np.minimum((R * min(h, w)).astype(np.int64), min(h, w))
    order = np.argsort(rb, axis=None)
    rbs, Ps = rb.ravel()[order], P.ravel()[order]
    edges = np.searchsorted(rbs, np.arange(rbs.max() + 2))
    med = np.zeros(rbs.max() + 1, np.float64)
    for i in range(rbs.max() + 1):
        seg = Ps[edges[i]:edges[i + 1]]
        med[i] = np.median(seg) if seg.size else 0.0
    prom = (P - med[rb]) * (10.0 / math.log(10.0))            # dB above the spectrum at that radius
    peak = (P == ndi.maximum_filter(P, size=5)) & (prom > PROMINENCE_DB) & (R >= FMIN) & (R <= FMAX)
    half = (FY > 0) | ((FY == 0) & (FX > 0))
    ys, xs = np.nonzero(peak & half)
    out = []
    for y, xx in zip(ys, xs):
        y0, y1, x0, x1 = max(0, y - 1), min(h, y + 2), max(0, xx - 1), min(w, xx + 2)
        wgt = np.exp(P[y0:y1, x0:x1] - P[y, xx])                # centroid of the peak (sub-bin position)
        cy = float((wgt * FY[y0:y1, x0:x1]).sum() / wgt.sum())
        cx = float((wgt * FX[y0:y1, x0:x1]).sum() / wgt.sum())
        out.append((cx, cy, float(prom[y, xx])))
    return out


def measure(rgb: np.ndarray) -> dict:
    """The screen's peaks from a full-resolution scene-linear crop: {"peaks": [[fx, fy], ...], "period": px or
    None, "angle": degrees or None, "strength": dB}."""
    a = np.log(np.maximum(rgb.astype(np.float32), 0.0) + 1e-3)
    chans = [a[..., c] for c in range(3)] + [a.mean(axis=2)]
    found = []
    for ch in chans:
        found.extend(_peaks_one(ch))
    found.sort(key=lambda p: -p[2])
    cand = []
    for fx, fy, pr in found:                                     # one entry per frequency (channels agree)
        if all(math.hypot(fx - q[0], fy - q[1]) > 0.008 for q in cand):
            cand.append((fx, fy, pr))
    none = {"peaks": [], "period": None, "angle": None, "strength": 0.0}
    # a screen is a lattice: two fundamentals of about the same frequency at right angles (a dot screen) and
    # their sums and multiples. Peaks off every lattice are the picture's own (a fence, a fabric) and are left.
    best = None
    for i, (ax, ay, ap) in enumerate(cand[:12]):
        if ap < PROMINENCE_DB + 5:
            continue
        ra = math.hypot(ax, ay)
        for bx, by, bp in cand[:24]:
            rb = math.hypot(bx, by)
            if abs(rb - ra) > 0.06 * ra:
                continue
            cosang = abs(ax * bx + ay * by) / (ra * rb)
            if cosang > math.sin(math.radians(4.0)):            # not within 4 degrees of a right angle
                continue
            lat = _on_lattice(cand, (ax, ay), (bx, by))
            score = sum(p[2] for p in lat)
            if best is None or score > best[0]:
                best = (score, lat, (ax, ay), ap)
    if best is None or len(best[1]) < 4:
        return none
    lat = sorted(best[1], key=lambda p: -p[2])[:MAX_PEAKS]
    ax, ay = best[2]
    ang = math.degrees(math.atan2(ay, ax)) % 90.0
    return {"peaks": [[p[0], p[1]] for p in lat], "period": 1.0 / math.hypot(ax, ay), "angle": ang,
            "strength": best[3]}


def _on_lattice(cand, f1, f2, tol: float = 0.006) -> list:
    """The candidate peaks lying on n f1 + m f2 (small integers), either sign (a peak and its mirror are one)."""
    out = []
    for fx, fy, pr in cand:
        ok = False
        for n in range(-4, 5):
            for m in range(-4, 5):
                if n == 0 and m == 0:
                    continue
                gx, gy = n * f1[0] + m * f2[0], n * f1[1] + m * f2[1]
                if min(math.hypot(fx - gx, fy - gy), math.hypot(fx + gx, fy + gy)) < tol:
                    ok = True
                    break
            if ok:
                break
        if ok:
            out.append((fx, fy, pr))
    return out


def notch_mask(h: int, w: int, peaks, step: int, amount: float) -> np.ndarray:
    """The multiplier for an h x w block's unshifted FFT: soft notches at every peak and its mirror image."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    M = np.ones((h, w), np.float32)
    s = SIGMA * step
    for px, py in peaks:
        qx, qy = px * step, py * step                            # cycles per block pixel
        if math.hypot(qx, qy) >= 0.5:                            # finer than this scale's pixels: already averaged
            continue
        for sx, sy in ((qx, qy), (-qx, -qy)):
            d2 = (fx - sx) ** 2 + (fy - sy) ** 2
            M *= (1.0 - amount * np.exp(-d2 / (2.0 * s * s))).astype(np.float32)
    return M


def clean(rgb: np.ndarray, peaks, d: dict | None, step: int = 1) -> np.ndarray:
    """Cut the screen's notches out of a scene-linear block (each channel, in log space so the dots' contrast is
    treated the same in shadow and light)."""
    st = settings(d)
    amt = float(np.clip(st["amount"] / 100.0, 0.0, 1.0))
    if amt <= 0 or not peaks:
        return rgb
    pad = 16
    a = np.log(np.maximum(rgb.astype(np.float32), 0.0) + 1e-3)
    a = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w = a.shape[:2]
    M = notch_mask(h, w, peaks, step, amt)
    out = np.empty_like(a)
    for c in range(3):
        out[..., c] = np.real(np.fft.ifft2(np.fft.fft2(a[..., c]) * M))
    out = out[pad:-pad, pad:-pad]
    return np.maximum(np.exp(out) - 1e-3, 0.0).astype(np.float32)


def screen(img: np.ndarray, period: float = 6.0, angle: float = 45.0, contrast: float = 0.6) -> np.ndarray:
    """Test material: print a scene-linear picture with a rotated dot screen (a cosine dot pattern), as a halftone
    reproduction looks once scanned slightly out of focus."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = math.radians(angle)
    u = (xx * math.cos(t) + yy * math.sin(t)) * (2 * math.pi / period)
    v = (-xx * math.sin(t) + yy * math.cos(t)) * (2 * math.pi / period)
    dots = 0.5 + 0.25 * (np.cos(u) + np.cos(v))                  # 0..1, a dot grid
    g = 1.0 + contrast * (dots - 0.5) * 2.0
    return np.maximum(img * g[..., None] if img.ndim == 3 else img * g, 0.0).astype(np.float32)
