"""The infrared hot spot (BUILD_PLAN_PANO_IR.md 1.5): a bright patch in the middle of the frame that many lenses show
in infrared, from reflections inside the lens and off the sensor at wavelengths its coatings were not made for.

Model: on top of the ordinary falloff, a radial Gaussian bump, recorded = true * g(r) with
``g(r) = 1 + a * exp(-(r / s)^2)`` in radius normalised to the half-diagonal, centred at (cx, cy) (fractions of the
frame; the optical centre by default, an off-centre spot allowed). ``stops = log2(1 + a)``.

Measured from one flat frame (sky, a wall: the person says so) by fitting the bump to the radial profile of
log-brightness after a smooth falloff term is taken out (amber: one photo); or learnt from several photos of one lens
and filter (the median of the frames, each normalised by its own large-scale level, like Learn dust map): cyan from
five photos on. Correction is OFF by default, like vignetting; amount 0..1.

Honest limit: on a single textured photo the scene itself can look like a bump (a sunlit middle); the fit is checked
against a plain falloff and refused when the bump does not explain the profile clearly better.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

from ... import log as slog

log = slog.get("engine.hotspot")


def bump(model: dict, r: np.ndarray) -> np.ndarray:
    a = float(model.get("a", 0.0))
    s = max(float(model.get("s", 0.35)), 1e-3)
    return 1.0 + a * np.exp(-(r / s) ** 2)


def gain_map(model: dict | None, amount: float, h: int, w: int, origin_xy, full_wh, step: float = 1.0) -> np.ndarray:
    """Per-pixel correction gain for a (region of a) frame (same arguments as falloff.gain_map)."""
    if not model or abs(amount) < 1e-6:
        return np.ones((h, w), np.float32)
    fw, fh = full_wh
    cx = float(model.get("cx", 0.5)) * fw - 0.5
    cy = float(model.get("cy", 0.5)) * fh - 0.5
    rn = math.hypot(fw, fh) / 2.0
    ys = origin_xy[1] + (np.arange(h) + 0.5) * step - 0.5 - cy
    xs = origin_xy[0] + (np.arange(w) + 0.5) * step - 0.5 - cx
    r = np.hypot(ys[:, None], xs[None, :]) / rn
    g = np.maximum(bump(model, r), 1e-3)
    return np.exp2(-float(amount) * np.log2(g)).astype(np.float32)


def _radial(Y: np.ndarray, cx: float, cy: float, nb: int = 40):
    h, w = Y.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - (cy * h - 0.5), xx - (cx * w - 0.5)) / (math.hypot(h, w) / 2.0)
    Z = np.log(np.maximum(Y, 1e-6))
    ok = (Y > 1e-3) & (Y < 0.95)
    bins = np.linspace(0, r.max() + 1e-6, nb + 1)
    rc, med, n = [], [], []
    idx = np.digitize(r[ok], bins) - 1
    zv = Z[ok]
    for b in range(nb):
        sel = idx == b
        if sel.sum() > 30:
            rc.append(0.5 * (bins[b] + bins[b + 1]))
            med.append(float(np.median(zv[sel])))
            n.append(int(sel.sum()))
    return np.array(rc), np.array(med), np.array(n)


def fit_profile(Y: np.ndarray, centre=(0.5, 0.5), textured: bool = False) -> dict | None:
    """Fit the bump (with a smooth falloff c1 r^2 + c2 r^4 in log) to a linear luminance frame. None when no bump
    explains the profile clearly better than falloff alone."""
    Y = np.asarray(Y, np.float32)
    k = max(1, int(max(Y.shape) // 600))
    if k > 1:
        h, w = Y.shape[0] // k * k, Y.shape[1] // k * k
        Y = Y[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    Y = gaussian_filter(Y, 1.0)
    cx, cy = centre
    rc, z, n = _radial(Y, cx, cy)
    if rc.size < 10:
        return None
    wts = np.sqrt(n / n.max())

    def res_full(p):
        lev, a, s, c1 = p
        return wts * (lev + np.log1p(max(a, -0.9) * np.exp(-(rc / max(s, 1e-3)) ** 2)) + c1 * rc ** 2 - z)

    def res_fall(p):
        lev, c1, c2 = p
        return wts * (lev + c1 * rc ** 2 + c2 * rc ** 4 - z)

    # the falloff alone gets the more flexible model (r^2 and r^4), so a bump has to earn its place; with the bump,
    # one falloff term that only darkens toward the corners (a lens never brightens them), and a bump no wider than
    # 0.7 of the half-diagonal: wider, it and the falloff could stand in for each other
    f0 = least_squares(res_fall, [z[0], 0.0, 0.0], loss="soft_l1", f_scale=0.05)
    best = None
    for s0 in (0.2, 0.35, 0.55):
        try:
            r = least_squares(res_full, [z[-1], 0.2, s0, 0.0], loss="soft_l1", f_scale=0.05,
                              bounds=([-np.inf, 0.0, 0.05, -5.0], [np.inf, 4.0, 0.7, 0.0]))
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    lev, a, s, c1 = best.x
    gain = f0.cost / max(best.cost, 1e-12)
    stops = math.log2(1.0 + a)
    need = 1.8 if textured else 1.6
    if stops < 0.05 or gain < need:
        log.debug("hot spot: none (%.3f stop, fit gain %.2f)", stops, gain)
        return None
    return {"a": round(float(a), 5), "s": round(float(s), 4), "cx": float(cx), "cy": float(cy),
            "stops": round(stops, 3), "fit_gain": round(float(gain), 2), "frames": 1,
            "source": "estimated from one photo"}


def measure_image(image, edit) -> dict | None:
    """Measure the hot spot of an opened photo (the whole frame, scene-linear luminance, no hot-spot correction)."""
    from ..pipeline.edit import Edit
    from ..pipeline.pipeline import linear_rgb
    e = Edit.from_dict(edit.to_dict())
    if e.infrared:
        e.infrared = dict(e.infrared, hotspot=None)
    e.lens.vignette = False
    with slog.stage("hotspot.measure", log):
        rgb = linear_rgb(image, e, half_size=True, cache=False)
        Y = rgb @ np.array([0.2880, 0.7119, 0.0001], np.float32)
        m = fit_profile(Y, textured=True)
    if m:
        log.info("hot spot measured: %.2f stop, width %.2f", m["stops"], m["s"])
    return m


def learn(frames_luma: list) -> dict | None:
    """Learn from several photos of one lens and filter: each frame divided by its own large-scale level (scene
    shapes cancel in the median), then the bump fitted to the median frame (cyan from five frames)."""
    stack = []
    shape = None
    for Y in frames_luma:
        Y = np.asarray(Y, np.float32)
        if shape is None:
            shape = Y.shape
        if Y.shape != shape:
            continue
        ok = (Y > 1e-3) & (Y < 0.95)
        lev = float(np.median(Y[ok])) if np.any(ok) else 1.0
        stack.append(np.where(ok, Y * (0.25 / max(lev, 1e-6)), 0.0).astype(np.float32))   # same level, clipped out
    if not stack:
        return None
    med = np.median(np.stack(stack), axis=0)
    m = fit_profile(med, textured=len(stack) < 3)
    if m:
        m["frames"] = len(stack)
        m["source"] = f"learnt from {len(stack)} photos"
    return m


def learn_from_files(paths_: list, progress=None) -> dict:
    """Library > Tools > Learn IR hot spot: each photo developed with its own edit (no hot-spot correction, no
    vignette correction), its luminance at a small size, then ``learn``. Returns {"ok", "model", "photos",
    "skipped", "lens", "camera"}."""
    from .. import api
    from ..pipeline.pipeline import linear_rgb
    lum, skipped, lens, camera, shape = [], [], "", "", None
    for i, p in enumerate(paths_):
        try:
            img = api.open(p)
            e = api.load_edit(p)
            if e.infrared:
                e.infrared = dict(e.infrared, hotspot=None)
            e.lens.vignette = False
            rgb = linear_rgb(img, e, half_size=True, cache=False)
            Y = rgb @ np.array([0.2880, 0.7119, 0.0001], np.float32)
            k = max(1, int(max(Y.shape) // 600))
            h, w = Y.shape[0] // k * k, Y.shape[1] // k * k
            Y = Y[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
            if shape is None:
                shape = Y.shape
            if Y.shape != shape:
                skipped.append((p, "a different size or orientation"))
                continue
            lum.append(Y)
            lens = lens or (e.lens.tag or getattr(img.meta, "lens", "") or "")
            camera = camera or (getattr(img.meta, "model", "") or "")
        except Exception as err:                    # noqa: BLE001 - one unreadable photo does not end the learning
            from ... import log as _slog
            if not _slog.expected_problem(err):
                _slog.save_error_report("Learn IR hot spot: developing one photo")
            skipped.append((p, str(err)[:80]))
        if progress:
            progress((i + 1) / max(len(paths_), 1), f"Photo {i + 1} of {len(paths_)}")
    if len(lum) < 3:
        return {"ok": False, "reason": "fewer than 3 photos could be used", "skipped": skipped}
    m = learn(lum)
    if m is None:
        return {"ok": False, "reason": "no hot spot found (the photos do not share a bright centre)",
                "skipped": skipped, "photos": len(lum)}
    return {"ok": True, "model": m, "photos": len(lum), "skipped": skipped, "lens": lens, "camera": camera}


# ------------------------------------------------------------------------------------------- saved per lens + filter
def profile_dir() -> Path:
    from ... import paths
    d = paths.get("CALIBRATION_LOCAL") / "hotspot"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_profile(name: str, model: dict, lens: str = "", filter_key: str = "", camera: str = "") -> Path:
    safe = re.sub(r"[^A-Za-z0-9 _.-]+", "_", name).strip(" .")[:80] or "hotspot"
    p = profile_dir() / (safe + ".json")
    p.write_text(json.dumps({"name": name, "model": model, "lens": lens, "filter": filter_key, "camera": camera,
                             "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "v": 1}, indent=2), encoding="utf-8")
    return p


def load_profiles() -> list:
    out = []
    try:
        for p in sorted(profile_dir().glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    except OSError:
        pass
    return out
