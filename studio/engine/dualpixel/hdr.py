"""Dual Pixel highlight recovery: about one extra stop in the highlights (BUILD_PLAN_TABS 3.4, plan 23.5).

A+B saturates when the two photodiodes together reach the sensor's clip; one half alone gets about half the light,
so where A+B is clipped half 1 often is not. Recovery develops A+B and half 1 (doubled, shading corrected) with
exactly the same pipeline, and blends half 1 in only where A+B nears its clip:
    w = smoothstep(lo, hi, the highest photosite of A+B nearby / clip)
    result = (1 - w) A+B + w (2 x half 1),   w = 0 where half 1 is itself clipped (3 px feathered edge)
Where the halves are offset (out of focus) half 1 is first moved by half the local offset so the recovered light
sits where the picture is. Below ``lo`` of the clip nothing changes: those pixels are the normal development,
bit for bit. The shadows gain nothing (the normal picture already holds all the light). The result is
scene-linear with values above 1.0, written as a linear intermediate: Develop's Highlights slider and tone curve
bring the recovered range into view. No Qt here.
"""
from __future__ import annotations

import numpy as np

from ... import log as slog

log = slog.get("engine.dualpixel.hdr")

HONEST = ("About one extra stop in the highlights, nothing extra in the shadows: the normal picture already holds "
          "all the light.")


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def hdr_edit(edit=None):
    """The edit both developments use: the photo's own, with no highlight clip (the recovered range must survive)
    and no sharpening (it is the photo's job later, in Develop)."""
    from ..pipeline.edit import Edit
    e = Edit.from_dict((edit or Edit()).to_dict())
    e.highlight_mode = "none"
    e.sharpen.enabled = False
    return e


def weight_map(dp, lo: float = 0.80, hi: float = 0.97, feather: float = 3.0) -> np.ndarray:
    """Per-pixel blend weight (camera crop, full size): near A+B's clip, and only where half 1 has room."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    near = maximum_filter(dp.ab, 3, mode="nearest")
    w = _smoothstep(lo, hi, near).astype(np.float32)
    bad = maximum_filter(dp.clipped_a.astype(np.uint8), 3, mode="nearest").astype(np.float32)
    ok = 1.0 - np.clip(gaussian_filter(bad, feather / 2.0, mode="nearest") * 2.0, 0, 1)
    ok[dp.clipped_a] = 0.0
    return (w * ok).astype(np.float32)


def recover(dp, gains, disp=None, lo: float = 0.80, hi: float = 0.97, edit=None, strength: float = 1.0):
    """(rgb scene-linear ProPhoto, camera-crop frame, not yet upright; report). ``disp``: the offset map (full-size
    pixels, half size, from depth.disparity / refine) to move half 1 into place; ``strength`` 0..1 scales the
    blend weight."""
    from scipy.ndimage import map_coordinates
    from ..pipeline.pipeline import linear_rgb
    e = hdr_edit(edit)
    ab_rgb = linear_rgb(dp.image, e, None, cache=False)
    w = weight_map(dp, lo, hi) * np.float32(np.clip(strength, 0.0, 1.0))
    if not np.any(w > 0):
        return ab_rgb, {"stops": 0.0, "recovered_share": 0.0, "clipped_both_share": float(
            (dp.clipped_ab & dp.clipped_a).mean()), "sentence": "No highlights near the clip: nothing to recover."}
    h1 = dp.half_image(1, gains)
    half_rgb = linear_rgb(h1, e, None, cache=False)
    del h1
    if disp is not None:                            # move half 1 by half the local offset (into the A+B position)
        from ..stack.focus import _up2
        d = _up2(disp, half_rgb.shape[:2]) if disp.shape != half_rgb.shape[:2] else disp
        moving = np.abs(d) > 0.25
        if moving.any():
            H, W = half_rgb.shape[:2]
            for y0 in range(0, H, 512):
                y1 = min(H, y0 + 512)
                if not moving[y0:y1].any():
                    continue
                Y, X = np.mgrid[y0:y1, 0:W].astype(np.float64)
                Xs = X - np.where(moving[y0:y1], d[y0:y1] / 2.0, 0.0)
                for c in range(3):
                    half_rgb[y0:y1, :, c] = map_coordinates(half_rgb[..., c], [Y, Xs], order=1, mode="nearest")
    out = ab_rgb
    wc = w[..., None]
    sel = w > 0
    out[sel] = ((1 - wc) * ab_rgb + wc * half_rgb)[sel]
    # 1.0 is the sensor's clip of a neutral: how far past it the recovered light reaches (the brightest 0.1%)
    peak = float(np.percentile(np.max(out[sel], axis=1), 99.9)) if sel.any() else 1.0
    stops = float(np.clip(np.log2(max(peak, 1.0)), 0.0, 1.2))
    rec = float(((w > 0.5)).mean())
    both = float((dp.clipped_ab & dp.clipped_a).mean())
    report = {"stops": stops, "recovered_share": rec, "clipped_both_share": both,
              "sentence": (f"Recovered up to +{stops:.1f} stop in {100 * rec:.1f}% of the frame"
                           + (f"; {100 * both:.1f}% is clipped in both halves" if both > 0.001 else "") + ".")}
    log.info("dual pixel highlights: %s", report["sentence"])
    return out, report
