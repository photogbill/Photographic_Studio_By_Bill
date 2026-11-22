"""One view for the 3D Scan solver (BUILD_PLAN_DP3D E.1): one focus stack (or one Dual Pixel photo).

What the solver needs from it: the all-in-focus picture (grey for matching, colour for texturing), the fused depth
``t`` with its confidence on the same pixel grid, larger = nearer (the camera's near sign applied), the subject mask,
and the camera (focal length in pixels and the principal point). The depth is normalised per view to ``tau`` (0 at
the view's middle depth, 1 over its 10th-90th percentile spread inside the mask), so one shape number per view
(kappa) turns it into inverse distance up to scale: w = m (1 + kappa tau).

Real views are made from a stack's result (``from_result``: NAME_Studio.tif and NAME.depth.npz beside it, both
written by Focus Stack), from a stack's frames (``from_stack``: analysed through the analysis cache), or from one
Dual Pixel photo (``from_dual``). The working size is a long edge of 2048 for matching (the depth and masks
resized to it); the texture uses the full-size colour later.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ... import log as slog

log = slog.get("engine.scan.views")

WORK_EDGE = 2048


@dataclass
class View:
    name: str
    image: np.ndarray                     # float32 grey 0..1 (h, w) for matching
    t: np.ndarray                         # fused depth (h, w), larger = nearer; any units
    conf: np.ndarray                      # 0..1 (h, w); 0 where nothing was measured
    mask: np.ndarray | None               # subject (h, w) bool, or None (the whole frame)
    f: float                              # focal length, pixels of this working size
    cx: float = -1.0
    cy: float = -1.0
    colour: np.ndarray | None = None      # uint8 RGB (H, W, 3), any size (scaled to the working size by ``scale``)
    path: str = ""                        # the result / photo it came from
    meta: dict = field(default_factory=dict)
    tau: np.ndarray | None = None
    t_ref: float = 0.0
    t_scale: float = 1.0
    f_from: str = "exif"
    f_sigma: float = 0.03                 # how far the true focal length may be from f (log share): wide in macro
    lens: str = ""                        # views of the same lens are tied together (focus breathing is smooth)

    def __post_init__(self):
        h, w = self.image.shape[:2]
        if self.cx < 0:
            self.cx = (w - 1) / 2.0
        if self.cy < 0:
            self.cy = (h - 1) / 2.0
        self.normalise_depth()

    @property
    def shape(self):
        return self.image.shape[:2]

    def normalise_depth(self):
        t = np.asarray(self.t, np.float32)
        good = (np.asarray(self.conf) > 0.5) & np.isfinite(t)
        if self.mask is not None and (good & self.mask).sum() > 200:
            good &= self.mask
        v = t[good] if good.sum() > 50 else t[np.isfinite(t)]
        if v.size == 0:
            self.t_ref, self.t_scale = 0.0, 1.0
        else:
            lo, mid, hi = np.percentile(v, [10, 50, 90])
            self.t_ref = float(mid)
            self.t_scale = float(max(hi - lo, 1e-6))
        self.tau = ((t - self.t_ref) / self.t_scale).astype(np.float32)


def focal_px(focal_mm: float | None, model: str, width_px: int, full_width_px: int) -> float | None:
    """Focal length in pixels of a picture ``width_px`` wide, from the lens's focal length and the camera's pixel
    size (``surface.PIXEL_MM``) at its full width."""
    from ..stack.surface import pixel_mm
    pmm = pixel_mm(model or "")
    if not focal_mm or not pmm:
        return None
    return float(focal_mm) / pmm * (width_px / float(full_width_px))


def _resize(a, shape, order=1):
    from scipy.ndimage import zoom
    a = np.asarray(a, np.float32)
    if a.shape[:2] == tuple(shape):
        return a
    z = (shape[0] / a.shape[0], shape[1] / a.shape[1]) + ((1,) if a.ndim == 3 else ())
    out = zoom(a, z, order=order, grid_mode=True, mode="nearest")
    return out[:shape[0], :shape[1]]


def from_arrays(name: str, colour8: np.ndarray, depth_near_high: np.ndarray, conf: np.ndarray, label=None,
                focal_mm=None, model: str = "", path: str = "", meta: dict | None = None,
                work_edge: int = WORK_EDGE, mask: np.ndarray | None = None) -> View:
    """A view from a full-size colour picture and its depth maps (any size: resized to the working size)."""
    H, W = colour8.shape[:2]
    s = min(1.0, work_edge / float(max(H, W)))
    h, w = int(round(H * s)), int(round(W * s))
    col = _resize(colour8, (h, w)) / 255.0
    grey = (0.2126 * col[..., 0] + 0.7152 * col[..., 1] + 0.0722 * col[..., 2]).astype(np.float32)
    t = _resize(depth_near_high, (h, w))
    c = np.clip(_resize(conf, (h, w)), 0, 1)
    if mask is None and label is not None:
        from ..stack import depthmask as DM
        lab = _resize(np.asarray(label, np.float32), (h, w), order=0)
        try:
            mask = DM.subject(np.clip((t - np.nanmin(t)) / max(float(np.nanmax(t) - np.nanmin(t)), 1e-6), 0, 1),
                              lab, grey, None) > 0.5
        except Exception as exc:                     # noqa: BLE001 - no subject found: the whole frame
            log.info("view %s: no subject mask (%s)", name, exc)
            mask = None
    elif mask is not None:
        mask = _resize(np.asarray(mask, np.float32), (h, w), order=0) > 0.5
    f = focal_px(focal_mm, model, w, W) if focal_mm else None
    f_from = "exif"
    f_sigma = 0.35
    md = dict(meta or {})
    if f is None:
        f = 1.2 * max(h, w)                         # a normal lens's view; the bundle adjustment refines it
        f_from = "guessed (no focal length or pixel size): found by the solver"
        f_sigma = 0.6
    else:
        # focus breathing: a lens focused close sits further from the sensor, so the picture's focal length is the
        # image distance v = f (1 + magnification), up to 2 f at 1:1. Known from the recorded focus distance;
        # otherwise wide (the solver finds it, tied across stacks of the same lens)
        d = md.get("focus_distance_m")
        if d and float(d) * 1000.0 > 1.5 * float(focal_mm):
            from ..stack.maps import magnification_from_distance
            mg = magnification_from_distance(float(focal_mm), float(d))
            if mg:
                f *= 1.0 + mg
                f_from = f"exif and the recorded focus distance (magnification {mg:.2f})"
                f_sigma = 0.05
        else:
            f_from = "exif; the magnification is not recorded, so the focal length is found by the solver"
    v = View(name=name, image=grey, t=t, conf=c, mask=mask, f=float(f), colour=colour8, path=path,
             meta=md, f_from=f_from, f_sigma=f_sigma, lens=f"{md.get('lens', '')} {focal_mm or ''}".strip())
    return v


def from_result(path: str, work_edge: int = WORK_EDGE) -> View:
    """A view from a Focus Stack result (NAME_Studio.tif / .png) and the NAME.depth.npz beside it."""
    from ..stack import focus as FS
    from ..pipeline.pipeline import measured_depth
    side = FS.depth_sidecar_path(path)
    if not os.path.exists(side):
        raise FileNotFoundError(f"{os.path.basename(path)} has no measured depth beside it ({os.path.basename(side)})")
    from PIL import Image as PILImage
    im = PILImage.open(path)
    col = np.asarray(im.convert("RGB"))
    z = np.load(side, allow_pickle=False)
    depth = np.asarray(z["depth"], np.float32)       # near = high (written so by Focus Stack)
    conf = np.asarray(z["confidence"], np.float32)
    lab = np.asarray(z["label_strict"] if "label_strict" in z.files else z["label"], np.float32)
    meta = {}
    try:                                             # the result carries the camera's EXIF (written by the Studio)
        from ..io.metadata import read_metadata
        md = read_metadata(path)
        meta = {"make": md.make, "model": md.model, "lens": md.lens, "focal_mm": md.focal_mm,
                "fnumber": md.fnumber, "focus_distance_m": getattr(md, "focus_distance_m", None)}
    except Exception as exc:                         # noqa: BLE001 - no EXIF: the focal length is found by the solver
        log.info("view %s: no camera data (%s)", os.path.basename(path), exc)
    _ = measured_depth                                # (Develop reads the same sidecar)
    return from_arrays(os.path.splitext(os.path.basename(path))[0], col, depth, conf, lab,
                       meta.get("focal_mm"), meta.get("model", ""), path, meta, work_edge)


def from_analysis(an, name: str, progress=None, cancel=None, work_edge: int = WORK_EDGE) -> View:
    """A view from a Focus Stack analysis (the analysis cache keeps it): the fused depth turned near = high and
    upright, its strict measured label for the subject mask, and the all-in-focus merge at full size.

    The fused depth is affine in inverse distance when Dual Pixel gave the steps (positions measured in Dual Pixel
    offset, which is affine in 1 / z); with focus alone the steps are frame numbers, near enough the same over a
    macro view's shallow range (the solver's depth shape kappa absorbs the difference to first order)."""
    from ..pipeline.pipeline import apply_orientation
    from ..stack import depthfuse as DF
    from ..stack import surface as SURF
    from ..stack.depthmask import strict_label
    res = DF.fuse(an)
    near_high, *_ = SURF.decide_near(res, str(an.meta.get("model") or ""), SURF.SurfaceParams())
    h = res.depth if near_high else -res.depth
    depth = np.ascontiguousarray(apply_orientation(h, an.flip)).astype(np.float32)
    conf = np.ascontiguousarray(apply_orientation(res.confidence, an.flip)).astype(np.float32)
    lab = np.ascontiguousarray(apply_orientation(strict_label(res), an.flip)).astype(np.float32)
    colour = SURF.stack_colour(an, "auto", progress, cancel)
    meta = dict(an.meta)
    meta["frames"] = [os.path.basename(q) for q in an.paths]
    meta["dual_pixel"] = bool(an.dual_pixel)
    return from_arrays(name, colour, depth, conf, lab, meta.get("focal_mm"), str(meta.get("model") or ""),
                       an.paths[0] if an.paths else "", meta, work_edge)


def from_stack(paths: list, name: str, focus_params=None, workers=None, cache: bool = True, progress=None,
               cancel=None) -> View:
    from ..stack import focus as FS
    p = focus_params or FS.FocusParams()
    an = FS.analyse(paths, p, progress, cancel, cache=cache, workers=workers)
    try:
        return from_analysis(an, name, progress, cancel)
    finally:
        an.close()


def from_dual(path: str) -> View:
    """A view from one Dual Pixel photo: its own offsets as the depth (a shallow measure: best near its focus)."""
    from ..dualpixel import decode as DD
    from ..dualpixel import depth as DEP
    from ..pipeline.edit import Edit
    from ..pipeline.pipeline import apply_orientation, render
    from ..stack import depthfuse as DF
    from ..stack import surface as SURF
    dp = DD.open_dual(path)
    try:
        gains = DD.shading(dp)
        la, lb = DEP.halves_luma(dp, gains)
        disp, conf = DEP.disparity(la, lb)
        res = DF.from_dual_pixel(disp, conf, la + lb)
        md = dp.image.meta
        near_high, *_ = SURF.decide_near(res, md.model or "", SURF.SurfaceParams())
        h = res.depth if near_high else -res.depth
        colour = SURF.srgb8(render(dp.image, Edit()))
        meta = {"make": md.make, "model": md.model, "lens": md.lens, "focal_mm": md.focal_mm,
                "fnumber": md.fnumber, "focus_distance_m": getattr(md, "focus_distance_m", None)}
        return from_arrays(os.path.splitext(os.path.basename(path))[0], colour,
                           np.ascontiguousarray(apply_orientation(h, dp.flip)),
                           np.ascontiguousarray(apply_orientation(res.confidence, dp.flip)),
                           np.ascontiguousarray(apply_orientation(res.label.astype(np.float32), dp.flip)),
                           md.focal_mm, md.model or "", path, meta)
    finally:
        try:
            dp.image.drop_cache()
        except Exception:                            # noqa: BLE001
            pass
