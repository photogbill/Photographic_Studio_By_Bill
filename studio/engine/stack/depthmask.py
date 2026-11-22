"""Subject masks from measured depth (BUILD_PLAN_DP3D C.9).

Outdoors there is no plain backdrop, so cutting a leaf or a stone out by colour fails: the subject and the litter
behind it are the same browns. The measured depth separates them: the subject is a connected region of measured
places that are near each other in depth, and a focus stack shot wide open measures nothing in the bokeh behind
(places where nothing was measured are background by definition). The region the photographer clicks (or, without
a click, the largest near one) is the subject; holes inside it that could not be measured (a plain patch on a leaf)
are filled; its outline is then snapped to the colour edges (a guided filter on the picture), so the edge follows
the picture while the depth decides what is in and what is out. No Qt here.
"""
from __future__ import annotations

import os

import numpy as np

from ... import log as slog

log = slog.get("engine.stack.depthmask")

JUMP = 0.06               # a depth step (share of the measured range) that separates two regions
MIN_REGION = 0.002        # regions smaller than this share of the picture are not subjects


def strict_label(res) -> np.ndarray:
    """The places a subject may be made of: in a focus stack, places that were sharp in some frame (a clear focus
    peak) or measured both ways; a lone Dual Pixel offset in the bokeh, or a focus estimate without a peak, is not
    enough (seen on Bill's bramble stack: scattered measured specks in the blur joined leaves to the background).
    One Dual Pixel photo: its own measured places."""
    from . import depthfuse as DF
    lab = np.asarray(res.label)
    if res.focus_quality is None:
        return lab.copy()
    keep = (np.asarray(res.focus_quality) >= 0.3) | (lab == DF.LABEL_BOTH)
    keep = DF.supported(keep & (lab > 0), np.asarray(res.focus_quality))
    return np.where(keep, lab, 0).astype(np.uint8)


def fill_small_holes(m: np.ndarray, max_share: float = 0.15) -> np.ndarray:
    """Holes inside a region filled only when small against the region (a plain patch on a leaf), never a big
    enclosed piece of background between leaflets."""
    from scipy.ndimage import binary_fill_holes, label as nd_label
    filled = binary_fill_holes(m)
    holes = filled & ~m
    if not holes.any():
        return m
    lab, n = nd_label(holes)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    small = np.zeros(n + 1, bool)
    small[1:] = sizes[1:] <= max_share * max(int(m.sum()), 1)
    return m | small[lab]


def regions(depth01: np.ndarray, label: np.ndarray, jump: float = JUMP) -> tuple[np.ndarray, int]:
    """Connected regions of measured places with no depth step bigger than ``jump`` between neighbours."""
    from scipy.ndimage import gaussian_filter, label as nd_label
    d = gaussian_filter(np.asarray(depth01, np.float32), 1.0)
    gy, gx = np.gradient(d)
    edge = np.hypot(gx, gy) > jump / 3.0              # a step spread over about three pixels
    meas = (np.asarray(label) > 0) & ~edge
    lab, n = nd_label(meas)
    return lab, int(n)


def subject(depth01: np.ndarray, label: np.ndarray, guide: np.ndarray | None = None, click=None,
            jump: float = JUMP) -> tuple[np.ndarray, dict]:
    """(alpha 0..1 at the depth's size, info). ``click``: (u, v) 0..1 on the subject; without one, the region with
    the most area, weighted toward the near ones. ``guide``: the picture's luminance at the depth's size (snaps the
    outline to its edges)."""
    from scipy.ndimage import binary_closing, distance_transform_edt
    lab, n = regions(depth01, label, jump)
    H, W = lab.shape
    if n == 0:
        return np.zeros((H, W), np.float32), {"found": False, "why": "nothing was measured in this picture"}
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    means = np.bincount(lab.ravel(), weights=np.asarray(depth01, np.float64).ravel(), minlength=n + 1) / \
        np.maximum(sizes, 1)
    pick = None
    how = ""
    if click is not None:
        u, v = float(click[0]), float(click[1])
        y, x = int(np.clip(v * H, 0, H - 1)), int(np.clip(u * W, 0, W - 1))
        if lab[y, x] > 0:
            pick = int(lab[y, x])
        else:                                         # the nearest measured region to the click
            dist, (iy, ix) = distance_transform_edt(lab == 0, return_indices=True)
            if dist[y, x] < 0.05 * max(H, W):
                pick = int(lab[iy[y, x], ix[y, x]])
        how = "the region you clicked"
    if pick is None:
        ok = sizes[1:] >= MIN_REGION * H * W
        if not ok.any():
            return np.zeros((H, W), np.float32), {"found": False, "why": "no measured region is big enough"}
        score = sizes[1:] * (0.5 + means[1:]) * ok
        pick = int(np.argmax(score)) + 1
        how = "the largest near region (no click given)"
    m = lab == pick
    med = float(np.median(np.asarray(depth01)[m]))
    # measured neighbours at the subject's depth join it (a region split by a crease), farther ones stay out
    near_same = (np.asarray(label) > 0) & (np.abs(np.asarray(depth01) - med) < 2.5 * jump)
    grow = m.copy()
    for _ in range(3):
        from scipy.ndimage import binary_dilation
        grow = binary_dilation(grow, iterations=4) & (near_same | m)
    m = fill_small_holes(binary_closing(grow, iterations=2))
    alpha = m.astype(np.float32)
    if guide is not None:
        from ..detail.finish import guided
        g = np.asarray(guide, np.float32)
        g = np.log(np.maximum(g, 1e-4))
        g = (g - g.min()) / max(float(np.ptp(g)), 1e-6)
        r = max(2, int(round(max(H, W) / 300.0)))
        alpha = np.clip(guided(g, alpha, r, 0.01), 0.0, 1.0).astype(np.float32)
    info = {"found": True, "how": how, "share": round(float(alpha.mean()), 4), "depth": round(med, 4),
            "regions": n}
    return alpha, info


def measured_depth_for(path: str, image=None):
    """(depth01 upright near = 1, label upright, how) for a picture: its NAME.depth.npz (a focus stack's result,
    B.7), or a Dual Pixel RAW measured now. None when it has neither."""
    from .focus import depth_sidecar_path
    side = depth_sidecar_path(path)
    if os.path.isfile(side):
        with np.load(side) as z:
            lab = z["label_strict"] if "label_strict" in z.files else z["label"]
            return z["depth"].astype(np.float32), lab.astype(np.uint8), "the focus stack's measured depth"
    from ..dualpixel import decode as DD
    if str(path).lower().endswith((".cr3", ".cr2")) and DD.is_dual_pixel(path):
        from ..dualpixel import depth as DEP
        from ..pipeline.pipeline import apply_orientation
        from . import depthfuse as DF
        from . import surface as SURF
        dp = DD.open_dual(path)
        g = DD.shading(dp)
        la, lb = DEP.halves_luma(dp, g)
        d, c = DEP.disparity(la, lb)
        res = DF.from_dual_pixel(d, c, la + lb)
        near_high, *_ = SURF.decide_near(res, dp.image.meta.model, SURF.SurfaceParams())
        h = res.depth if near_high else -res.depth
        lo, hi = SURF.height_range(h, res.label)
        d01 = np.clip((h - lo) / (hi - lo), 0, 1)
        return (np.ascontiguousarray(apply_orientation(d01, dp.flip)).astype(np.float32),
                np.ascontiguousarray(apply_orientation(res.label, dp.flip)).astype(np.uint8),
                "the photo's Dual Pixel halves")
    return None


def subject_shape(img, edit, click=None) -> dict:
    """A mask shape (masks.py "object": a bitmap snapped to the picture's edges) of the subject cut out by its
    measured depth. Raises ValueError (said plainly) when the picture has no measured depth."""
    from ..local import masks as mk
    from ..pipeline.pipeline import _srgb_preview
    path = str(getattr(img.raw, "path", "") or "")
    got = measured_depth_for(path, img)
    if got is None:
        raise ValueError("this picture has no measured depth: use a focus stack's result (it keeps its depth) or a "
                         "Dual Pixel RAW")
    d01, label, how = got
    base, _ = _srgb_preview(img, edit, max(d01.shape))
    from scipy.ndimage import zoom
    Y = 0.25 * base[..., 0] + 0.5 * base[..., 1] + 0.25 * base[..., 2]
    Yg = zoom(Y, (d01.shape[0] / Y.shape[0], d01.shape[1] / Y.shape[1]), order=1, grid_mode=True,
              mode="nearest")[:d01.shape[0], :d01.shape[1]]
    if Yg.shape != d01.shape:
        Yg = np.pad(Yg, ((0, d01.shape[0] - Yg.shape[0]), (0, d01.shape[1] - Yg.shape[1])), mode="edge")
    alpha, info = subject(d01, label, Yg, click)
    if not info.get("found"):
        raise ValueError(f"no subject found by measured depth: {info.get('why', '')}")
    return {"type": "object", "op": "add", "w": int(alpha.shape[1]), "h": int(alpha.shape[0]),
            "data": mk.encode_bitmap(np.round(alpha * 255).astype(np.uint8)), "source": "measured depth",
            "how": how + "; " + info["how"], "share": info["share"]}
