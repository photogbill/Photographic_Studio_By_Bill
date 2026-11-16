"""Automatic white balance and the neutral eyedropper (plan 9.2).

Auto: the grey-edge hypothesis (van de Weijer, Gevers, Gijsenij, IEEE TIP 16(9), 2007): the average colour of
the image's *edges* (not of the image itself) is grey under a neutral light. It is measured on the raw camera
colours (no white balance applied), with clipped and very dark pixels left out, as a Minkowski p-norm of the
gradients of each channel. It is far less fooled by large single-colour areas (a green lawn, a red wall) than
the grey-world average.

Honest limit: candle and tungsten scenes have a warm cast on purpose; Auto will neutralise it. That is why it is
a button and never a default.

Second method (plan 23.9), offered beside it: the **Grayness Index** (Qian, Kamarainen, Nikkanen, Matas, CVPR
2019). Learning-free: under one light, a grey surface has the same local contrast in every colour channel, so
the Laplacian of log(R) - log(intensity) and of log(B) - log(intensity) is zero there. The pixels where it is
closest to zero (and where there is contrast at all) are the ones most likely to be grey; their mean colour is
the light. When those pixels fall into two clearly different colours, the photo has two light sources (candle
and window), and both are reported.

Third method: **Night (SGP-LRD)** from RL-AWB (Lee, Chen, Chang, Liu, 2026; Apache 2.0; color/nightwb.py), made for
low light: salient grey pixels kept by two night filters, weighted by brightness, a Minkowski mean of local
reflectance; its two settings picked per photo by agreement across a grid instead of the paper's learned agent.
Tested here on synthetic casts: much better than grey edge on a scene with nothing grey (coffee: 2.0 against 7.4
degrees), worse than the grayness index where there is real grey (astronaut: 5.6 against 0.1); it still answers in
very dark frames where the other two give up. Real night raws decide; the photo's own estimates are all shown.

All the estimates are always computed; how well they agree is the confidence.

**Keep the mood**: 100% makes the light fully neutral; less corrects only part of the way from the as-shot
balance (people prefer not-quite-neutral, and a candle should stay warm). The default for a warm light in a
low-light photo is 60%, so Auto no longer kills candlelight.

Eyedropper: the clicked spot's raw camera colour is made neutral (average of a small patch).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, uniform_filter

from ... import log as slog

log = slog.get("engine.autowb")


def camera_rgb_half(image) -> np.ndarray:
    """Half-size raw camera RGB of the camera-crop frame (black-subtracted, 1 = clip, no white balance)."""
    from ..raw import demosaic as dm
    from ..raw import linearize as lin
    raw = image.raw
    lv = image.levels(True)
    cx, cy, cw, ch = raw.crop
    mos = lin.normalise(raw, lv, (cx, cy, cw // 2 * 2, ch // 2 * 2))
    return dm.demosaic(mos, raw.cfa_rgb, "half")


def grey_edge(cam: np.ndarray, p: float = 6.0, sigma: float = 1.0) -> np.ndarray | None:
    """Illuminant colour (R, G, B, sum 1) from a raw camera RGB image, or None if it cannot be told."""
    ok = (cam.max(axis=2) < 0.95) & (cam.min(axis=2) > 0.002)
    ok = gaussian_filter(ok.astype(np.float32), 2.0) > 0.99          # keep away from clipped / black areas
    if ok.mean() < 0.05:
        return None
    e = []
    for c in range(3):
        f = gaussian_filter(cam[..., c], sigma)
        gy, gx = np.gradient(f)
        g = np.hypot(gx, gy)[ok]
        if g.size < 1000:
            return None
        e.append(float(np.mean(g ** p) ** (1.0 / p)))
    e = np.asarray(e)
    if not np.all(e > 0):
        return None
    return e / e.sum()


# What each method is good and bad at (rule 23.1), shown beside the Auto white balance method picker.
METHODS = {
    "grey_edge": dict(name="Grey edge", best="Photos with plenty of edges and mixed colours.",
                      weak="One dominant colour with coloured edges (autumn leaves) can pull it."),
    "grayness": dict(name="Grayness index", best="Finds the truly grey bits; reports a second light source.",
                     weak="Needs something close to neutral somewhere in the frame."),
    "night": dict(name="Night (SGP-LRD)", best="Low light and night scenes, and scenes with nothing grey in them "
                                               "(a brown room, a lamp-lit street).",
                  weak="Bright scenes with real grey surfaces: the grayness index is better there."),
}


def _shrink_to(cam: np.ndarray, long_edge: int = 900) -> np.ndarray:
    n = max(1, int(max(cam.shape[:2]) // long_edge))
    if n <= 1:
        return cam
    h, w = (cam.shape[0] // n) * n, (cam.shape[1] // n) * n
    return cam[:h, :w].reshape(h // n, n, w // n, n, 3).mean(axis=(1, 3))


def grayness_index(cam: np.ndarray, top: float = 0.001, sigma: float = 1.2):
    """(illuminant (R, G, B, sum 1), lights) from a raw camera RGB image, or (None, []) if it cannot be told.
    ``lights``: one or two (illuminant, share) entries: two when the greyest pixels split into two colours."""
    cam = _shrink_to(np.asarray(cam, np.float32))
    ok = (cam.max(axis=2) < 0.95) & (cam.min(axis=2) > 0.003)
    if ok.mean() < 0.05:
        return None, []
    lg = np.log(np.maximum(cam, 1e-6))
    ln = np.log(np.maximum(cam.mean(axis=2), 1e-6))
    dr = gaussian_laplace(lg[..., 0] - ln, sigma)
    db = gaussian_laplace(lg[..., 2] - ln, sigma)
    gi = np.sqrt(dr * dr + db * db)
    contrast = np.max(np.abs(np.stack([gaussian_laplace(lg[..., c], sigma) for c in range(3)], axis=2)), axis=2)
    gi[contrast < 1e-3] = np.inf                     # flat areas carry no cue: every colour looks "grey" there
    gi = uniform_filter(np.where(np.isfinite(gi), gi, 10.0), 7)
    gi[~ok] = np.inf
    gi[contrast < 1e-3] = np.inf
    valid = np.isfinite(gi)
    n = int(max(50, top * gi.size))
    if int(valid.sum()) < 2 * n:
        return None, []
    thr = np.partition(gi[valid], n)[n]
    px = cam[valid & (gi <= thr)].astype(np.float64)
    px = px / px.sum(axis=1, keepdims=True)          # chromaticity of each grey candidate
    e = px.mean(axis=0)
    lights = [(e / e.sum(), 1.0)]
    # Two lights? Take a wider set of grey candidates (the greyest of all may all sit under one lamp), split them
    # into two colours (2-means on chromaticity) and see how far apart and how evenly shared they are.
    n2 = min(int(valid.sum()) // 2, 10 * n)
    thr2 = np.partition(gi[valid], n2)[n2]
    wide = cam[valid & (gi <= thr2)].astype(np.float64)
    wide = wide / wide.sum(axis=1, keepdims=True)
    c = np.stack([wide[np.argmin(wide[:, 0])], wide[np.argmax(wide[:, 0])]])
    split = True
    for _ in range(12):
        lab = np.argmin(((wide[:, None, :] - c[None]) ** 2).sum(axis=2), axis=1)
        if lab.min() == lab.max():
            split = False
            break
        c = np.stack([wide[lab == k].mean(axis=0) for k in (0, 1)])
    if split:
        share = float((lab == 0).mean())
        spread = max(float(np.sqrt(((wide[lab == k] - c[k]) ** 2).sum(axis=1).mean())) for k in (0, 1))
        apart = float(np.linalg.norm(c[0] - c[1]))
        if 0.2 <= share <= 0.8 and angle_deg(c[0], c[1]) > 6.0 and apart > 3.0 * spread:
            lights = [(c[0] / c[0].sum(), share), (c[1] / c[1].sum(), 1.0 - share)]
    return e / e.sum(), lights


def angle_deg(a, b) -> float:
    """Angle between two light colours (the usual white-balance error measure)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30), -1.0, 1.0))))


def _mult(e) -> np.ndarray:
    return (1.0 / np.asarray(e, np.float64)) / (1.0 / np.asarray(e, np.float64)).min()


def estimates(image) -> dict:
    """Both estimates, their agreement and a confidence, for the Auto white balance panel.
    {"grey_edge": mult | None, "grayness": mult | None, "lights": [(mult, share), ...], "agreement_deg",
     "confidence": "high" | "medium" | "low"}"""
    from . import nightwb
    cam = camera_rgb_half(image)
    ge = grey_edge(cam)
    gi, lights = grayness_index(cam)
    try:
        nt = nightwb.estimate(cam)
    except Exception:
        slog.log_exception("night white balance failed; left out", log)
        nt = None
    ni = nt["light"] if nt else None
    out = {"grey_edge": _mult(ge) if ge is not None else None, "grayness": _mult(gi) if gi is not None else None,
           "night": _mult(ni) if ni is not None else None, "night_spread_deg": nt["spread_deg"] if nt else None,
           "lights": [(_mult(l), s) for l, s in lights], "agreement_deg": None, "confidence": "low"}
    have = [e for e in (ge, gi, ni) if e is not None]
    if len(have) >= 2:                               # the widest disagreement between any two of them
        a = max(angle_deg(x, y) for i, x in enumerate(have) for y in have[i + 1:])
        out["agreement_deg"] = round(a, 1)
        out["confidence"] = "high" if a < 3.0 else ("medium" if a < 7.0 else "low")
    log.info("auto white balance estimates: grey edge %s, grayness %s, night %s, agreement %s deg, %d light(s)",
             None if ge is None else np.round(out["grey_edge"], 3).tolist(),
             None if gi is None else np.round(out["grayness"], 3).tolist(),
             None if ni is None else np.round(out["night"], 3).tolist(), out["agreement_deg"], len(lights))
    return out


def keep_mood(auto_mult, as_shot_mult, amount: float) -> np.ndarray:
    """Part-way correction: amount 1 = the estimate (fully neutral), 0 = as shot. Interpolated in log space, so
    half-way is half the correction in every channel; green stays the reference."""
    a, s = np.asarray(auto_mult, np.float64), np.asarray(as_shot_mult, np.float64)
    a, s = a / a[1], s / s[1]
    k = float(np.clip(amount, 0.0, 1.0))
    m = np.exp(k * np.log(a) + (1.0 - k) * np.log(s))
    return m / m.min()


def default_mood(temperature_k: float, iso: float | None) -> float:
    """How far Auto corrects by default: 60% for a warm light (below 3500 K) in a low-light photo (ISO 1600 and
    up), where the warmth is the picture; 90% otherwise (not-quite-neutral is what people prefer)."""
    if temperature_k < 3500.0 and (iso or 0) >= 1600:
        return 0.6
    return 0.9


def in_band(cam_xyz, mult, tint_limit: float = 60.0):
    """Keep an estimate inside a sensible band around the daylight / tungsten curve: a tint beyond the limit is
    pulled back to it (an estimate far off the curve is a coloured subject, not a light)."""
    from . import camera as cam
    T, tint = cam.temp_tint_from_multipliers(cam_xyz, mult)
    T2, tint2 = float(np.clip(T, 1800.0, 20000.0)), float(np.clip(tint, -tint_limit, tint_limit))
    if abs(T2 - T) < 1e-6 and abs(tint2 - tint) < 1e-6:
        return np.asarray(mult, np.float64), False
    m = cam.multipliers_for_white(cam_xyz, cam.temp_tint_to_xy(T2, tint2))
    return np.asarray(m, np.float64) / np.min(m), True


def auto_multipliers(image, method: str = "grey_edge") -> tuple[np.ndarray | None, str]:
    cam = camera_rgb_half(image)
    if method == "night":
        from . import nightwb
        nt = nightwb.estimate(cam)
        if nt is not None:
            m = _mult(nt["light"])
            log.info("auto white balance (night): multipliers %s", np.round(m, 4).tolist())
            return m, "night (SGP-LRD)"
    if method == "grayness":
        gi, lights = grayness_index(cam)
        if gi is not None:
            m = _mult(gi)
            how = "grayness index" + (", two light sources found" if len(lights) > 1 else "")
            log.info("auto white balance (%s): multipliers %s", how, np.round(m, 4).tolist())
            return m, how
    e = grey_edge(cam)
    if e is None:
        mean = cam.reshape(-1, 3)
        mean = mean[(mean.max(1) < 0.95) & (mean.min(1) > 0.002)]
        if mean.size < 300:
            return None, "not enough usable pixels"
        e = mean.mean(0)
        how = "grey world (too few edges)"
    else:
        how = "grey edge"
    m = (1.0 / e) / (1.0 / e).min()
    log.info("auto white balance (%s): multipliers %s", how, np.round(m, 4).tolist())
    return m, how


def patch_multipliers(image, xs: float, ys: float, radius: int = 3) -> np.ndarray | None:
    """Multipliers that make the raw colour around S-frame point (xs, ys) (full-res pixels) neutral."""
    cam = camera_rgb_half(image)
    hx, hy = int(round((xs - 0.5) / 2.0)), int(round((ys - 0.5) / 2.0))
    a, b = max(0, hy - radius), min(cam.shape[0], hy + radius + 1)
    c, d = max(0, hx - radius), min(cam.shape[1], hx + radius + 1)
    patch = cam[a:b, c:d].reshape(-1, 3)
    patch = patch[(patch.max(1) < 0.95)]
    if patch.size == 0 or patch.mean(0).min() <= 1e-5:
        return None
    m = patch.mean(0)
    m = (1.0 / m) / (1.0 / m).min()
    return m
