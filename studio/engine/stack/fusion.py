"""Visible + infrared fusion (BUILD_PLAN_PANO_IR.md 1.7): one visible frame and one infrared frame of the same scene.

Built and tested on synthetic pairs (Bill has no visible / IR pair yet). Alignment across spectra cannot match
brightness (foliage is dark in visible light and bright in infrared) but edges are where they are in both, so the two
frames are registered on their **gradient magnitudes** (``align.register`` with a homography), then a local flow on the
same pair takes up small parallax.

Modes (METHOD_INFO):
* **IR detail, visible colour**: OKLab lightness from the infrared frame (blend amount), a/b from the visible one.
* **Colour infrared (CIR)**: R = infrared, G = visible red, B = visible green: remote sensing's standard false colour,
  the colour-infrared film look made from real data, not guessed.
* **NDVI map**: (IR - red) / (IR + red), the vegetation index satellites use, with a colour ramp; absolute only when
  both frames were white-balanced on the same grey card (cyan), otherwise relative (amber).

Output: ``FIRST_to_LAST_IRfusion_Studio.tif`` (a linear intermediate that opens in Develop); NDVI also as a 16-bit
greyscale PNG for other programs.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import gaussian_filter

from ... import log as slog
from . import align

log = slog.get("engine.fusion")

METHOD_INFO = {
    "detail": {"name": "IR detail, visible colour",
               "does": "Brightness and detail from the infrared frame, colour from the visible one.",
               "best": "Cutting haze; glowing foliage with true sky colour.",
               "weak": "Things infrared sees through (some fabrics, thin smoke) change their tones."},
    "cir": {"name": "Colour infrared (CIR)",
            "does": "Red = infrared, green = visible red, blue = visible green.",
            "best": "The colour-infrared film look from real data: red foliage, the remote-sensing standard.",
            "weak": "Skin and sky take odd colours; it is a false-colour map, not a portrait look."},
    "ndvi": {"name": "NDVI map (vegetation)",
             "does": "(IR - red) / (IR + red): how much living plants reflect infrared over red, as a colour ramp.",
             "best": "Showing healthy vegetation, crops and gardens like a satellite.",
             "weak": "Absolute only when both frames were white balanced on the same grey card; otherwise relative."},
}
HOW_TO = ("How to shoot a pair: tripod, focus set for the infrared frame, one frame with an IR-cut (visible) filter "
          "and one with the 720 or 850 nm filter. Two infrared frames (720 + 850 nm) work too: their difference is "
          "real spectral information.")


class FusionProblem(ValueError):
    """The two frames cannot be fused (sizes, no overlap)."""


def gradient_image(lum: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Edge strength of log brightness: the same in both spectra where the brightness itself is not."""
    z = np.log(np.maximum(gaussian_filter(np.asarray(lum, np.float32), sigma), 1e-5))
    gy, gx = np.gradient(z)
    g = np.hypot(gx, gy)
    return (g / max(float(np.percentile(g, 99)), 1e-6)).astype(np.float32)


def align_pair(vis_lum: np.ndarray, ir_lum: np.ndarray, local: bool = True, cancel=None):
    """(Transform, Flow or None) taking visible-frame coordinates to infrared-frame coordinates, from the gradient
    images (analysis copies of one size)."""
    gv, gi = gradient_image(vis_lum), gradient_image(ir_lum)
    T = align.register(gv, gi, kind="homography", levels=3, cancel=cancel)
    flw = None
    if local:
        wi = align.warp(gi, T, gv.shape, order=1)
        try:
            flw = align.flow(gv, wi, tile=64, step=32, max_shift=4)
        except Exception:                          # noqa: BLE001 - the global fit alone is still a good answer
            slog.log_exception("fusion: local flow failed", log)
            slog.save_error_report("Fusion: local alignment")
            flw = None
    return T, flw


def ndvi(ir_red: np.ndarray, red: np.ndarray) -> np.ndarray:
    return ((ir_red - red) / np.maximum(ir_red + red, 1e-6)).astype(np.float32)


def ndvi_ramp(v: np.ndarray) -> np.ndarray:
    """-1..1 -> a brown-yellow-green ramp (scene-linear-ish, for the intermediate)."""
    stops = np.array([[-1.0, 0.05, 0.05, 0.25], [0.0, 0.45, 0.35, 0.2], [0.25, 0.85, 0.8, 0.2],
                      [0.5, 0.25, 0.65, 0.15], [1.0, 0.02, 0.3, 0.05]])
    out = np.stack([np.interp(v, stops[:, 0], stops[:, c]) for c in (1, 2, 3)], -1)
    return (out ** 2.2 * 0.18 / 0.2).astype(np.float32)


def fuse(vis_rgb: np.ndarray, ir_rgb_aligned: np.ndarray, mode: str, amount: float = 1.0) -> tuple:
    """Fuse a visible frame (scene-linear ProPhoto) with an infrared frame already in its coordinates. Returns
    (result HxWx3, extra) where extra holds the NDVI array for "ndvi"."""
    vis = np.asarray(vis_rgb, np.float32)
    ir = np.asarray(ir_rgb_aligned, np.float32)
    ir_l = ir.mean(axis=2)
    if mode == "cir":
        return np.stack([ir_l, vis[..., 0], vis[..., 1]], -1).astype(np.float32), None
    if mode == "ndvi":
        v = ndvi(ir_l, vis[..., 0])
        return ndvi_ramp(v), v
    from ..color.grade import from_oklab, to_oklab
    lab_v = to_oklab(np.maximum(vis, 0))
    lab_i = to_oklab(np.repeat(np.maximum(ir_l, 0)[..., None], 3, axis=2))
    a = float(np.clip(amount, 0.0, 1.0))
    lab = lab_v.copy()
    lab[..., 0] = (1 - a) * lab_v[..., 0] + a * lab_i[..., 0]
    return np.maximum(from_oklab(lab), 0).astype(np.float32), None


def build(vis_path: str, ir_path: str, mode: str, out_path: str, progress=None, cancel=None,
          amount: float = 1.0) -> dict:
    """Develop both frames (their own sidecar edits, so the infrared frame keeps its Infrared tab settings), align,
    fuse, write the intermediate (and the NDVI PNG). Returns {"path", "text", "colour", ...}."""
    from .. import api
    from ..io.intermediate import write_intermediate
    from ..pipeline.pipeline import linear_rgb
    from .focus import Cancelled
    say = progress or (lambda *_: None)

    def check():
        if cancel is not None and cancel.is_set():
            raise Cancelled()
    say(0.05, "Developing the visible frame...")
    iv = api.open(vis_path)
    ev = api.load_edit(vis_path)
    vis = np.ascontiguousarray(linear_rgb(iv, ev, cache=False), np.float32)
    check()
    say(0.3, "Developing the infrared frame...")
    ii = api.open(ir_path)
    ei = api.load_edit(ir_path)
    ir = np.ascontiguousarray(linear_rgb(ii, ei, cache=False), np.float32)
    check()
    if abs(vis.shape[0] / vis.shape[1] - ir.shape[0] / ir.shape[1]) > 0.02:
        raise FusionProblem("The two frames have different shapes (one is turned or cropped): they cannot be "
                            "lined up.")
    say(0.5, "Lining up the frames on their edges...")
    lv, k = align.analysis_copy(align.luma(vis), 1600)
    li = align.bin_by(align.luma(ir), max(1, int(round(k * ir.shape[1] / vis.shape[1]))))
    li = li[:lv.shape[0], :lv.shape[1]]
    if li.shape != lv.shape:
        pad = np.zeros(lv.shape, np.float32)
        pad[:li.shape[0], :li.shape[1]] = li
        li = pad
    T, flw = align_pair(lv, li, cancel=cancel)
    check()
    s = vis.shape[1] / float(lv.shape[1])
    Tf = T.rescaled(s)
    flf = flw.rescaled(s) if flw is not None else None
    say(0.7, "Fusing...")
    ir_al = align.warp(ir, Tf, vis.shape, order=3, flw=flf)
    out, extra = fuse(vis, ir_al, mode, amount)
    check()
    absolute = (ev.white_balance.mode == "multipliers" and ei.white_balance.mode == "multipliers")
    info = {"tool": "Infrared fusion", "mode": mode, "visible": os.path.basename(vis_path),
            "infrared": os.path.basename(ir_path), "transform": Tf.to_dict()}
    say(0.85, "Writing...")
    path = write_intermediate(out_path, out, None, info)
    rep = {"path": path, "mode": mode, "colour": "cyan" if (mode != "ndvi" or absolute) else "amber"}
    if extra is not None:
        png = os.path.splitext(path)[0] + "_NDVI.png"
        from PIL import Image as PI
        v16 = np.round((np.clip(extra, -1, 1) + 1.0) * 0.5 * 65535).astype(np.uint16)
        PI.fromarray(v16).save(png)
        rep["ndvi_png"] = png
        rep["text"] = (f"Saved {os.path.basename(path)} and {os.path.basename(png)} (NDVI, 16-bit: 0 = -1, "
                       f"65535 = +1)." + ("" if absolute else " Relative values: white balance both frames on the "
                                          "same grey card for absolute NDVI."))
    else:
        rep["text"] = f"Saved {os.path.basename(path)}."
    say(1.0, rep["text"])
    log.info("fusion %s: %s", mode, path)
    return rep
