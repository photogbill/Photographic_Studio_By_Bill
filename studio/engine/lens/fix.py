"""The "Fix lens" button (plan 10.2): every automatic lens fix in one call, with a report of what was done and
where each number came from. Colour tiers follow rule 0.3.11: cyan = calibrated (lens wizard, Phase 7),
violet = shared database (lensfun), amber = estimated from this photo, dim = nothing to do.

Everything measured is frozen into the returned Edit, so the sidecar renders the same forever. Vignette
correction is measured but stays OFF unless the user already had it on (Bill likes vignetting).
"""
from __future__ import annotations

import copy
import math

import numpy as np

from ... import log as slog
from . import falloff, geometry, library, lines, lensfun_bridge

log = slog.get("engine.fixlens")

CYAN, VIOLET, AMBER, DIM = "cyan", "violet", "amber", "dim"


def preview_luminance(image, edit=None, long_edge: int = 1600):
    """Upright (as displayed) linear luminance of the whole frame, about ``long_edge`` pixels, for measurements.
    Returns (Y, full_res_pixels_per_pixel)."""
    from ..pipeline.edit import Edit
    from ..pipeline.pipeline import apply_orientation, linear_rgb, resize
    e = Edit(demosaic="half")
    if edit is not None:
        e.white_balance = copy.deepcopy(edit.white_balance)
    e.lens.defringe = False
    rgb = linear_rgb(image, e, None, half_size=True, cache=False)
    k = min(1.0, long_edge / max(rgb.shape[:2]))
    if k < 1.0:
        rgb = resize(rgb, max(8, round(rgb.shape[1] * k)), max(8, round(rgb.shape[0] * k)))
    Y = rgb @ np.array([0.2880402, 0.7118741, 0.0000857], np.float32)
    step = 2.0 / k
    return np.ascontiguousarray(apply_orientation(Y, image.raw.flip)), step


def _undistort(P, model, cx, cy, r_norm):
    """Remove a frozen distortion model from points (x, y) by Newton steps on Rd = Ru * s(Ru)."""
    if not model:
        return P
    rn = r_norm * float(model.get("norm", 1.0))
    d = (P - [cx, cy]) / rn
    rd = np.hypot(d[:, 0], d[:, 1])
    ru = rd.copy()
    for _ in range(10):
        h = 1e-4
        f = ru * geometry.radial_scale(model, ru) - rd
        fp = ((ru + h) * geometry.radial_scale(model, ru + h) - (ru - h) * geometry.radial_scale(model, ru - h)) / (2 * h)
        ru = np.clip(ru - f / np.where(np.abs(fp) < 1e-3, 1e-3, fp), 0.0, 4.0)
    s = np.where(rd > 1e-9, ru / np.maximum(rd, 1e-9), 1.0)
    return d * s[:, None] * rn + [cx, cy]


def level_from_segments(segs, wh, model=None) -> dict | None:
    w, h = wh
    if model:
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        r_norm = math.hypot(w, h) / 2.0 / geometry.SQRT_3_25
        segs = [lines._seg(_undistort(s["points"], model, cx, cy, r_norm)) for s in segs]
    return lines.auto_level(segs, wh)


def auto_straighten(image, edit) -> dict | None:
    """Rotation that levels the photo from its straight lines (after removing known lens distortion)."""
    Y, _ = preview_luminance(image, edit)
    segs = lines.segments(Y)
    return level_from_segments(segs, (Y.shape[1], Y.shape[0]), edit.geometry.distortion)


def fix_lens(image, edit):
    """Return (new_edit, report). report: list of {"what", "text", "tier"} for the result panel."""
    e = copy.deepcopy(edit)
    rep = []
    meta = image.meta
    e.lens.auto_ca = True
    e.lens.defringe = True
    e.lens.defringe_strength = max(e.lens.defringe_strength, 1.0)
    # 1. lateral colour: always measured from the photo
    ca = image.lateral_ca()
    if ca.confidence == "none":
        rep.append({"what": "Lateral colour", "text": "none measurable in this photo", "tier": DIM})
    else:
        rep.append({"what": "Lateral colour", "text": f"{ca.max_shift_px:.1f} px at the corners, corrected "
                    f"(measured from this photo, {ca.confidence})", "tier": AMBER})
    rep.append({"what": "Purple / green fringes", "text": "removal on (only next to bright, high-contrast edges)",
                "tier": AMBER})
    # 2. which lens
    name, how = library.effective_lens(meta, e)
    if name:
        rep.append({"what": "Lens", "text": f"{name} ({how})", "tier": DIM})
    else:
        rep.append({"what": "Lens", "text": "manual lens, not tagged: tag it in the Library to remember it",
                    "tier": DIM})
    prof = {}
    if meta.lens and not e.lens.tag:                 # the camera recorded it: a database profile may exist
        prof = lensfun_bridge.profile_for(meta.make, meta.model, meta.lens, meta.focal_mm, meta.fnumber)
    elif e.lens.tag:                                 # a tagged manual lens that lensfun happens to know (some do)
        ent = library.get(e.lens.tag)
        if ent and ent.focal_mm:
            prof = lensfun_bridge.profile_for(meta.make, meta.model, ent.name, ent.focal_mm, meta.fnumber)
    Y = None
    segs = None
    # 3. distortion: profile first, else straight lines, else leave alone (never guess on organic subjects)
    if prof.get("distortion"):
        e.geometry.distortion = prof["distortion"]
        e.geometry.distortion_amount = 1.0
        rep.append({"what": "Distortion", "text": f"corrected from the lensfun profile for {prof['lens']}",
                    "tier": VIOLET})
    else:
        Y, _ = preview_luminance(image, edit)
        segs = lines.segments(Y)
        d = lines.measure_distortion(segs, (Y.shape[1], Y.shape[0]))
        if d:
            e.geometry.distortion = d
            e.geometry.distortion_amount = 1.0
            kind = "barrel" if d["terms"][0] < 0 else "pincushion"
            rep.append({"what": "Distortion", "text": f"{kind} distortion measured from {d['lines']} straight lines "
                        f"and corrected ({d['confidence']})", "tier": AMBER})
        else:
            if not lensfun_bridge.available() and meta.lens:
                why = "lensfun is not installed, and there are not enough straight lines to measure it"
            elif not meta.lens:
                why = "no profile for a manual lens, and not enough straight lines to measure it"
            else:
                why = "no profile for this lens, and not enough straight lines to measure it"
            rep.append({"what": "Distortion", "text": f"left alone: {why} (right for organic subjects)",
                        "tier": DIM})
    # 4. falloff: measured, correction stays as the user had it (off by default)
    vm = prof.get("vignetting")
    tier = VIOLET
    if not vm:
        if Y is None:
            Y, _ = preview_luminance(image, edit)
        vm = falloff.estimate(Y)
        tier = AMBER
    if vm:
        e.lens.vignette_model = vm
        state = "corrected" if e.lens.vignette else "correction is off (keeps your lens's character); tick it to apply"
        src = "lensfun profile" if tier == VIOLET else "estimated from this photo"
        rep.append({"what": "Vignetting", "text": f"{falloff.corner_ev(vm):+.1f} EV at the corners ({src}); {state}",
                    "tier": tier})
    else:
        rep.append({"what": "Vignetting", "text": "none measurable", "tier": DIM})
    log.info("fix lens: %s", "; ".join(f"{r['what']}: {r['text']}" for r in rep))
    return e, rep
