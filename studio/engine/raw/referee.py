"""The Dual Pixel referee (BUILD_PLAN_NOISE H.1, H.2): a photo that carries its own second opinion.

A Dual Pixel RAW file holds the photo twice over: the light through the left half of the lens and the light
through the right half, taken in the same instant. Where the picture is in focus the two halves show the same
scene, and each has **its own photon noise**. That gives two things no single frame can:

H.1  **The noise, measured on this very photo.** The difference of the halves holds no picture, only noise. Its
     strength is compared with what the camera's profile predicts (1.00 = the profile is right for this photo).

H.2  **A referee for Clean's strength.** One half is cleaned and compared with the *other, untouched* half. The
     other half's noise knows nothing of the first half's, so the comparison is honest: the strength with the
     least error against the other half is the strength with the least error against the true scene (the idea of
     Noise2Noise, Lehtinen et al. 2018, used here to judge instead of to train). The share of fine detail kept is
     measured the same way. Nothing is assumed about the noise except that the halves do not share it, so this
     works where the single-photo estimate (raw/auto_strength.py) cannot: where neighbouring photosites share
     noise.

Three cautions, each handled here and said in the result:
  * **Out of focus, the halves do not show the same thing** (that is how the camera focuses). Blocks where the
    halves disagree by more than noise are left out; a photo with nothing in focus gives no verdict.
  * **Each half is darker toward one side** (opposite sides). Slow differences are taken out before comparing.
  * **The file stores the sum and one half**, so the other half is a subtraction and the halves share the
    camera's read noise, in opposite directions. Only places where photon noise is at least four times the read
    noise are used; a dark photo gives no verdict.
And one limit: a half holds half the light, so the verdict is reached at about 0.7 of the photo's own signal to
noise. The strength it gives is used as it is.

Works in the variance-stabilised luminance / colour planes, like Clean itself. No Qt.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from ... import log as slog
from ..calib.noise import anscombe
from . import auto_strength as AS
from . import denoise as DN

log = slog.get("engine.referee")

PIECE = 384                 # mosaic photosites per side of a piece (192 per colour plane)
SHADE_SIGMA = 10.0          # plane pixels: differences slower than this are shading, not noise
BLOCK = 16
AGREE = 1.8                 # a block's mid-band mismatch above this times the quiet blocks' is "not the same scene"
PHOTON_OVER_READ = 4.0
MIN_USED = 0.12             # share of the pieces' area that must agree for a verdict
MAX_LINK = 0.08             # the halves may share this much of their noise at most (0 = each half's is its own)


def second_frame(image):
    """(full-sensor mosaic of frame 2 as uint16, its black level per CFA position) for a Dual Pixel photo, or None
    for a normal raw. Read once per opened photo (a second or two)."""
    key = ("dual pixel frame 2",)
    with image._lock:
        if key in image._memo:
            return image._memo[key]
    out = None
    path = getattr(image.raw, "path", "") or ""
    try:
        from ..dualpixel import decode as DP
        if path.lower().endswith(".cr3") and DP.is_dual_pixel(path):
            m, black = DP._cr3_second_frame(path)
            if m.shape == image.raw.mosaic.shape:
                out = (m, np.asarray(image.raw.black_for_positions(black), np.float32).reshape(2, 2))
    except Exception:                                         # noqa: BLE001  (a normal raw, or an unreadable frame)
        slog.log_exception(f"second frame of {path} could not be read; the single-photo estimate is used", log)
    with image._lock:
        image._memo[key] = out
    return out


def halves(image, frame2, box: tuple, use_optical_black: bool = True) -> tuple:
    """(sum, half 1, half 2) of a region, normalised like the photo (black out, 1.0 = clip), not floored.
    ``box``: (x0, y0, w, h) in visible-area coordinates, x0 and y0 even."""
    from . import linearize as lin
    raw = image.raw
    lv = image.levels(use_optical_black)
    x0, y0, w, h = box
    ab = lin.normalise(raw, lv, box)
    m, black = frame2
    hv = m[raw.top + y0:raw.top + y0 + h, raw.left + x0:raw.left + x0 + w].astype(np.float32)
    a = np.empty_like(hv)
    for i in (0, 1):
        for j in (0, 1):
            a[i::2, j::2] = (hv[i::2, j::2] - black[i, j]) / np.float32(lv.range[i, j])
    return ab, a, ab - a


def _planes(mos2: np.ndarray, pattern, model) -> tuple:
    """Y, C1, C2 of one half scaled to the photo's brightness. A half of n electrons scaled by two has twice the
    photo's photon gain; its read noise is not known exactly (see the module text), so only bright places count."""
    roles = DN.plane_roles(pattern)
    pl = {n: anscombe(np.maximum(mos2[i::2, j::2], 0.0), 2.0 * float(model.a[i, j]), 2.0 * float(model.b[i, j]))
          for n, (i, j) in roles.items()}
    Y, C1, C2, _ = DN.to_yc(pl["R"], pl["G1"], pl["G2"], pl["B"])
    return Y, C1, C2


_BAND_NOISE = None


def _band(x: np.ndarray) -> np.ndarray:
    return gaussian_filter(x, 1.5) - gaussian_filter(x, 4.0)


def _band_noise() -> float:
    global _BAND_NOISE
    if _BAND_NOISE is None:
        n = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
        _BAND_NOISE = float(np.mean(_band(n) ** 2))
    return _BAND_NOISE


def prepare(ab: np.ndarray, a: np.ndarray, b: np.ndarray, pattern, model) -> dict | None:
    """One piece made ready for judging: the planes of both halves, half 2 brought to half 1's slow shading, and
    the weight of every place (1 where the halves show the same scene and photon noise rules, 0 elsewhere)."""
    roles = DN.plane_roles(pattern)
    ya, c1a, c2a = _planes(2.0 * a, pattern, model)
    yb, c1b, c2b = _planes(2.0 * b, pattern, model)
    lp = lambda v: gaussian_filter(v, SHADE_SIGMA)            # noqa: E731
    gain = np.clip(lp(yb) / np.maximum(lp(ya), 1e-3), 0.4, 2.5)        # luminance: shading multiplies
    off1, off2 = lp(c1b - c1a), lp(c2b - c2a)                           # colour planes are differences: it adds
    # where photon noise rules, and nothing is clipped
    photon = sum(float(model.a[i, j]) * uniform_filter(np.maximum(ab[i::2, j::2], 0.0), BLOCK) for i, j in roles.values())
    read = sum(float(model.b[i, j]) for i, j in roles.values())
    top = np.maximum.reduce([uniform_filter(ab[i::2, j::2], 5) for i, j in roles.values()])
    ok = (photon >= PHOTON_OVER_READ * read) & (top < 0.85)
    # where the halves show the same scene: the mid band of their difference is at the level of the quiet blocks
    m = uniform_filter(_band(ya * gain - yb) ** 2, BLOCK)
    if not ok.any():
        return None
    quiet = float(np.percentile(m[ok], 25))
    agree = ok & (m <= AGREE * max(quiet, 1e-6) * 1.35)       # (the quartile sits about 1.35 below a flat mean)
    w = agree.astype(np.float32)
    return dict(ya=ya, yb=yb, c1a=c1a, c1b=c1b, c2a=c2a, c2b=c2b, gain=gain, off1=off1, off2=off2, w=w,
                used=float(w.mean()), bright=float(ok.mean()), quiet=quiet / (2.0 * _band_noise()))


def noise_link(pieces: list) -> float | None:
    """Do the two halves share noise in this photo? Measured where the photo is flat (nothing fine in either
    half): there the finest bands of the two halves agree only if their noise does. 0 = each half's noise is its
    own; -0.2 = a fifth of it is shared in opposite directions (the read noise of a dark photo: the file stores
    the sum and one half). A positive figure is faint detail both halves see, not noise. None when the pieces
    hold no flat place to measure it in."""
    num = n = 0.0
    for p in pieces:
        g = p["gain"]
        fa, fb = AS.fine(p["ya"]), AS.fine(p["yb"])
        ea, eb = uniform_filter(fa * fa, BLOCK), uniform_filter(fb * fb, BLOCK)
        flat = (p["w"] > 0.5) & (ea <= 1.15 * AS.FINE_NOISE) & (eb <= 1.15 * AS.FINE_NOISE)
        if int(flat.sum()) < 4 * BLOCK * BLOCK:
            continue
        num += float((fa * fb)[flat].sum())
        n += float(flat.sum())
    return None if n <= 0 else round(num / (AS.FINE_NOISE * n), 3)


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float((x * w).sum() / max(float(w.sum()), 1e-6))


def noise_check(pieces: list) -> dict | None:
    """H.1: this photo's own noise against the profile. ``ratio``: measured noise strength over predicted
    (1.00 = the profile fits this photo; 1.10 = the photo is 10% noisier than the profile says)."""
    num = den = 0.0
    for p in pieces:
        d = AS.fine(p["ya"] * p["gain"] - p["yb"])
        num += float((d * d * p["w"]).sum())
        den += float(((1.0 + p["gain"] ** 2) * AS.FINE_NOISE * p["w"]).sum())
    if den <= 0:
        return None
    return dict(ratio=round(float(np.sqrt(num / den)), 3), photosites=int(sum(float(p["w"].sum()) for p in pieces)))


def _judge(fn, strength: float, scales: int, mine: np.ndarray, other: np.ndarray, w: np.ndarray, fix) -> dict:
    """Clean one half's plane and hold it against the other half's. error: mean squared difference from the other
    half, less that half's own noise (the true error, plus a constant that does not depend on the strength)."""
    f = fix(fn(mine, strength, scales))
    fo = AS.fine(other)
    return dict(risk=_wmean((f - other) ** 2, w) - 1.0,
                texture=_wmean(AS.fine(fix(mine)) * fo, w),              # the other half's noise is not in `mine`:
                along=_wmean(AS.fine(f) * fo, w))                        # no noise term to take off


def choose(fn, pieces: list, scales: int, kind: str = "luma", candidates=AS.CANDIDATES,
           weight: float = AS.DETAIL_WEIGHT) -> dict:
    """The strength for luminance ("luma") or colour ("chroma") planes, judged by the other half. Same result
    shape as auto_strength.choose, same detail-first cost, so the two can be compared and shown alike."""
    table, kept_table, costs = [], [], []
    for s in candidates:
        ms = []
        for p in pieces:
            if kind == "luma":
                g = p["gain"]
                ms.append(_judge(fn, s, scales, p["ya"], p["yb"], p["w"], lambda v, g=g: v * g))
            else:
                for mine, other, off in ((p["c1a"], p["c1b"], p["off1"]), (p["c2a"], p["c2b"], p["off2"])):
                    ms.append(_judge(fn, s, scales, mine, other, p["w"], lambda v, off=off: v + off))
        table.append((float(s), round(float(np.mean([m["risk"] for m in ms])), 4)))
        k = AS._kept(ms)
        kept_table.append((float(s), None if k is None else round(k, 3)))
        costs.append(float(np.mean([AS.cost(m, weight) for m in ms])))
    best = min(costs)
    j = next(i for i, c in enumerate(costs) if c <= best + AS.NEAR_BEST * max(abs(best), 0.05) + 1e-4)
    return dict(strength=table[j][0], risk=table[j][1], kept=kept_table[j][1], best=min(r for _, r in table),
                table=table, kept_table=kept_table)


def judge_photo(image, edit, boxes: list, method: str) -> dict | None:
    """The whole referee for one photo: None when it is not a Dual Pixel file, {"note": why} when it is but gives
    no verdict, else {luma, chroma, kept, noise_ratio, used, ...}."""
    frame2 = second_frame(image)
    if frame2 is None:
        return None
    model = image.noise_model(edit.use_optical_black)
    pieces = []
    for box in boxes:
        ab, a, b = halves(image, frame2, box, edit.use_optical_black)
        p = prepare(ab, a, b, image.raw.cfa_rgb, model)
        if p is not None:
            pieces.append(p)
    used = float(np.mean([p["used"] for p in pieces])) if pieces else 0.0
    bright = float(np.mean([p["bright"] for p in pieces])) if pieces else 0.0
    if used < MIN_USED:
        why = ("it is too dark for the two halves to be compared fairly (they share the camera's read noise)"
               if bright < 2 * MIN_USED else "too little of it is in focus for the two halves to show the same scene")
        log.info("referee: no verdict (%s; %.0f%% bright enough, %.0f%% agreeing)", why, 100 * bright, 100 * used)
        return dict(note=f"This is a Dual Pixel photo, but {why}; the single-photo estimate is used instead.")
    link = noise_link(pieces)
    if link is not None and link < -MAX_LINK:
        log.info("referee: no verdict, the halves share %.0f%% of their noise", 100 * link)
        return dict(note="This is a Dual Pixel photo, but its two halves share part of their noise here (as they "
                         "do where the photo is dark), so one cannot judge the other; the single-photo estimate "
                         "is used instead.")
    fn, fc = DN.method_fns(method)
    slow = method in ("mppca", "collab")
    lu = choose(fn, pieces[:3] if slow else pieces, 4, "luma", AS.CANDIDATES_SLOW if slow else AS.CANDIDATES)
    ch = choose(fc, pieces, 6, "chroma")
    check = noise_check(pieces) or {}
    measured = getattr(model, "source", "") in ("calibrated", "shared")   # only then is there a profile to check
    log.info("referee (%s): luminance %.2f, colour %.2f; %.0f%% of %d pieces used; this photo's noise is %.2f x "
             "the %s model's; noise shared between the halves: %s", method, lu["strength"], ch["strength"],
             100 * used, len(pieces), check.get("ratio", 0), getattr(model, "source", "?"), link)
    return dict(luma=lu["strength"], chroma=ch["strength"], kept=lu["kept"], luma_table=lu["table"],
                chroma_table=ch["table"], kept_table=lu["kept_table"], kind="referee", pieces=len(pieces),
                used=round(used, 3), noise_ratio=check.get("ratio") if measured else None,
                noise_ratio_estimated_model=None if measured else check.get("ratio"), link=link)
