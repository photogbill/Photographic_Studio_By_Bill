"""Composite: a subject from another photo placed into this one. A Cognitive Adjustment.

    "composite": [{"src": path, "mask": {"w", "h", "data"}, "fg": [...], "bg": [...], "model": "sam2.1-tiny",
                   "x": 0.5, "y": 0.55, "size": 0.5, "rotate": 0.0, "flip": False,
                   "match": None | {"p": [...], "mult": [...], "amount": 1.0}, "grain": True, "feather": 1.0,
                   "enabled": True, "name": "..."}]

The subject is cut out of its own photo with click-to-select (SAM 2.1 or the classic method; the outline is kept as
a bitmap in that photo's sensor frame, so it follows the photo's own crop and rotation), developed with that
photo's own settings, and laid onto this photo's finished picture:
    x, y      where its middle goes (fractions of this photo as shown)
    size      its height as a fraction of this photo's height
    rotate    degrees, + = counter-clockwise; flip mirrors it
    match     a colour match of the SOURCE photo's settings to this photo (local/colourmatch.py), solved on the two
              whole photos: the subject is re-developed under this scene's light and colour mood, as editable
              settings, rather than recoloured after the fact
    grain     this photo's noise measured and the missing part added to the subject (a cleaner subject on a grainy
              photo is the giveaway Photoshop's Harmonization leaves alone)
    feather   edge softness in pixels at full size, on top of the outline's own soft edge
Nothing is generated: every pixel of the subject is from its photo. The layers are drawn last, on the finished
picture, in every view (whole, zoomed, export) at that view's resolution.
No GUI imports (rule 2.2).
"""
from __future__ import annotations

import json
import math
import os
import threading

import numpy as np

from ... import log as slog

log = slog.get("engine.composite")
_lock = threading.Lock()
_SOURCES: dict = {}                 # path -> (file mtime, Image), the last few source photos
_SUBJECTS: dict = {}                # key -> (rgb, alpha) renders of subjects, the last few
MAX_SOURCES, MAX_SUBJECTS = 3, 6


def layers(edit) -> list:
    return [L for L in (getattr(edit, "composite", None) or []) if L.get("enabled", True) and L.get("mask")
            and L.get("src")]


def active(edit) -> bool:
    return bool(layers(edit))


def _source(path: str):
    from .. import api
    mt = os.path.getmtime(path)
    with _lock:
        hit = _SOURCES.get(path)
        if hit and hit[0] == mt:
            return hit[1]
    img = api.open(path)
    with _lock:
        _SOURCES[path] = (mt, img)
        while len(_SOURCES) > MAX_SOURCES:
            _SOURCES.pop(next(iter(_SOURCES)))
    return img


def source_edit(layer: dict):
    """The source photo's own settings, with the layer's colour match applied."""
    from .. import api
    from .edit import Edit
    from .pipeline import colour_match_apply
    try:
        e = api.load_edit(layer["src"])
    except Exception:
        e = Edit()
    e.composite = None                                  # a subject never brings its own subjects along
    m = layer.get("match")
    if m and m.get("p") and float(m.get("amount", 1.0)):
        e = colour_match_apply(e, m["p"], m["mult"], float(m.get("amount", 1.0)))
    return e


def _sidecar_mtime(path: str) -> float:
    from .. import api
    try:
        return os.path.getmtime(api.sidecar_path(path))
    except OSError:
        return 0.0


def _key(layer: dict, extra) -> str:
    core = {k: layer.get(k) for k in ("src", "mask", "match")}
    return json.dumps([core, os.path.getmtime(layer["src"]), _sidecar_mtime(layer["src"]), extra], sort_keys=True,
                      default=str)


def subject(layer: dict, scale: float):
    """(rgb, alpha) of the subject in its photo as shown, cropped to the subject, at ``scale`` of that photo; and
    the crop's box in those pixels."""
    from .pipeline import render, render_mask
    k = _key(layer, round(float(scale), 3))
    with _lock:
        hit = _SUBJECTS.get(k)
    if hit is not None:
        return hit
    src = _source(layer["src"])
    e = source_edit(layer)
    with slog.stage("composite.subject", log, src=os.path.basename(layer["src"]), scale=round(scale, 3)):
        shape = {"type": "object", "op": "add", "w": int(layer["mask"]["w"]), "h": int(layer["mask"]["h"]),
                 "data": layer["mask"]["data"]}
        em = type(e).from_dict(e.to_dict())
        em.masks = [{"name": "subject", "enabled": True, "adjust": {}, "shapes": [shape]}]
        a = render_mask(src, em, 0, scale)
        rgb = render(src, e, scale=scale)
        if rgb.shape[:2] != a.shape[:2]:
            from ..imgops import resize_linear
            a = resize_linear(a, rgb.shape[1], rgb.shape[0])
        t = np.clip((a - 0.25) / 0.5, 0.0, 1.0)            # a cut-out is solid inside: only the edge is soft
        a = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        ys, xs = np.nonzero(a > 0.02)
        if len(ys) == 0:
            res = (None, None, None)
        else:
            m = 3
            y0, y1 = max(0, ys.min() - m), min(a.shape[0], ys.max() + 1 + m)
            x0, x1 = max(0, xs.min() - m), min(a.shape[1], xs.max() + 1 + m)
            res = (np.ascontiguousarray(rgb[y0:y1, x0:x1]), np.ascontiguousarray(a[y0:y1, x0:x1]),
                   (x0, y0, x1 - x0, y1 - y0))
    with _lock:
        _SUBJECTS[k] = res
        while len(_SUBJECTS) > MAX_SUBJECTS:
            _SUBJECTS.pop(next(iter(_SUBJECTS)))
    return res


def subject_extent(layer: dict) -> tuple[float, float]:
    """The subject's height and width as fractions of its photo as shown (from a small render)."""
    from .pipeline import output_size
    rgb, a, box = subject(layer, 0.15)
    if a is None:
        return 0.0, 0.0
    src = _source(layer["src"])
    ow, oh = output_size(src, source_edit(layer))
    ys, xs = np.nonzero(a > 0.5)
    if len(ys) == 0:
        ys, xs = np.nonzero(a > 0.02)
    sh = (ys.max() - ys.min() + 1) / (0.15 * oh)
    sw = (xs.max() - xs.min() + 1) / (0.15 * ow)
    return float(sh), float(sw)


def noise_sigma(rgb: np.ndarray, alpha: np.ndarray | None = None) -> float:
    """Noise of a picture (display-linear), as a relative brightness change: the MAD of a fine Laplacian of
    log brightness, over its flattest half (and inside alpha when given)."""
    from scipy.ndimage import laplace, uniform_filter
    y = np.log(np.maximum(rgb.mean(axis=2).astype(np.float64), 1e-4))
    lap = laplace(y) / math.sqrt(20.0)
    act = uniform_filter(np.abs(lap), 7)
    sel = np.ones(y.shape, bool) if alpha is None else alpha > 0.9
    sel &= np.exp(y) > 0.02
    if sel.sum() < 64:
        return 0.0
    flat = act <= np.percentile(act[sel], 50)
    v = lap[sel & flat]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 32 else 0.0


def apply(image, edit, out: np.ndarray, window=None) -> np.ndarray:
    """Lay the edit's subjects onto a finished picture ``out`` (display-linear ProPhoto) that shows ``window``
    (x, y, w, h in full-size output pixels; None = the whole photo)."""
    from ..imgops import gaussian_blur, warp_affine
    from .pipeline import output_size
    Ls = layers(edit)
    if not Ls:
        return out
    ow, oh = output_size(image, edit)
    wx, wy, ww, wh = window if window is not None else (0, 0, ow, oh)
    oh_px, ow_px = out.shape[:2]
    sc = ow_px / float(ww)                                  # output pixels per full-size pixel
    res = out.copy()
    dest_sigma = None
    for i, L in enumerate(Ls):
        try:
            hfrac, _ = subject_extent(L)
            if hfrac <= 0:
                continue
            src = _source(L["src"])
            _, src_oh = output_size(src, source_edit(L))
            target_h = float(L.get("size", 0.5)) * oh * sc        # subject height in this view's pixels
            s = float(np.clip(target_h / max(hfrac * src_oh, 1.0) * 1.15, 0.02, 1.0))
            s = round(s * 40.0) / 40.0 if s > 0.05 else round(s, 3)   # few sizes, so renders are reused
            rgb, a, box = subject(L, s)
            if rgb is None:
                continue
            ys, xs = np.nonzero(a > 0.5)
            if len(ys) == 0:
                continue
            cy, cx = 0.5 * (ys.min() + ys.max()), 0.5 * (xs.min() + xs.max())
            k = target_h / float(ys.max() - ys.min() + 1)
            th = math.radians(float(L.get("rotate", 0.0)))
            fl = -1.0 if L.get("flip") else 1.0
            c, sn = math.cos(th), math.sin(th)
            A = k * np.array([[c * fl, sn], [-sn * fl, c]])        # + = counter-clockwise on screen
            dx = (float(L.get("x", 0.5)) * ow - wx) * sc
            dy = (float(L.get("y", 0.55)) * oh - wy) * sc
            t = np.array([dx, dy]) - A @ np.array([cx, cy])
            M = np.hstack([A, t[:, None]]).astype(np.float64)
            sub = warp_affine(rgb, M, ow_px, oh_px)
            al = warp_affine(a, M, ow_px, oh_px)
            if not np.any(al > 1e-3):
                continue
            fe = float(L.get("feather", 1.0)) * sc
            if fe > 0.3:
                al = gaussian_blur(al, fe) * np.clip(al * 4.0, 0, 1) ** 0.5   # soften, never grow
            if L.get("grain", True):
                if dest_sigma is None:
                    dest_sigma = noise_sigma(out)
                sub_sigma = noise_sigma(sub, al)
                extra = math.sqrt(max(dest_sigma ** 2 - sub_sigma ** 2, 0.0))
                if extra > 1e-4:
                    rng = np.random.default_rng(1234 + i + int(wx) * 7 + int(wy) * 13)
                    n = rng.standard_normal(al.shape).astype(np.float32) * extra
                    sub = sub * np.exp(n)[..., None]
            al3 = np.clip(al, 0.0, 1.0)[..., None]
            res = res * (1.0 - al3) + sub * al3
        except Exception:
            slog.log_exception(f"composite layer {i + 1} failed; left out", log)
    return res.astype(np.float32)
