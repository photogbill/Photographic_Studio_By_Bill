"""Black and white -> colour with DDColor (Kang et al., ICCV 2023, "DDColor: Towards Photo-Realistic Image
Colorization via Dual Decoders"; Apache License 2.0), run on this PC by onnxnp.py from Qualcomm's ONNX export of
the "paper tiny" model (ConvNeXt-T encoder, pixel and colour decoders), converted by tools/convert_onnx_model.py
(8-bit weights: 53 MB; colour within 0.15-0.65 Lab units of the float model on average, invisible side by side).

What the model does: from the brightness of a photo, at its own small size (256 pixels), it predicts the two
colour coordinates of CIE Lab (a: green-red, b: blue-yellow) for every pixel.

How the Studio uses it, so the photo keeps everything it had:

  * only colour is added: each pixel keeps its exact brightness (CIE L*) of the photo as shown with the Studio's
    standard curve; the colour is put in scene-linear light by undoing that curve per channel, so every later
    slider (exposure, contrast, colour...) acts on the colourized photo as on any other;
  * the small colour map is brought up to full size edge-aware (guided filter, He et al. 2010, the full-size
    brightness as the guide), so colour stops at the edges of the photo's own shapes;
  * the colours are a plausible guess. The interface says so.

The model file lives in models/colour/ddcolor.npz and ships with the Studio (see models/colour/NOTICE).
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
from scipy import ndimage as ndi

from ... import log as slog
from ... import paths
from ..color import spaces as sp

log = slog.get("engine.colourize")

MODEL = "ddcolor.npz"
DEFAULTS = {"on": False, "amount": 1.0, "hints": None, "warm": 0.0, "tint": 0.0}
_LOCK = threading.Lock()
_MODEL = None


def settings(d: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (d or {}).items() if k in DEFAULTS})
    return out


def hints_of(d: dict | None) -> list:
    """The colour hints (a part of the photo clicked and given a colour) that have their area worked out."""
    return [h for h in ((d or {}).get("hints") or []) if h.get("data") and h.get("rgb")]


def active(d: dict | None) -> bool:
    """Anything to do: the automatic colour (needs the DDColor model) or a hint (needs nothing but the click)."""
    return bool(d) and ((bool(d.get("on")) and available()) or bool(hints_of(d)))


def model_dir() -> str:
    return os.path.join(str(paths.APP_ROOT), "models", "colour")


def model_path() -> str:
    return os.path.join(model_dir(), MODEL)


def available() -> bool:
    return os.path.isfile(model_path())


def model():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            from .onnxnp import Graph
            with slog.stage("colourize.load", log):
                _MODEL = Graph.load(model_path())
            log.info("DDColor input %s, outputs %s", _MODEL.input_shape, _MODEL.outputs)
        return _MODEL


# ------------------------------------------------------------------------------------------ Lab (D65, sRGB)
_D65 = sp.xy_to_XYZ(sp.D65_XY)
_SRGB_TO_XYZ = sp.SRGB.to_xyz                           # linear sRGB -> XYZ (D65)
_XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ)


def _f(t):
    d = 6.0 / 29.0
    return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)


def _finv(f):
    d = 6.0 / 29.0
    return np.where(f > d, f ** 3, 3 * d * d * (f - 4.0 / 29.0))


def L_of_Y(Y):
    """CIE L* (0..100) of a relative luminance (white = 1)."""
    return 116.0 * _f(np.maximum(Y, 0.0)) - 16.0


def lab_to_linear_srgb(L, a, b):
    fy = (L + 16.0) / 116.0
    X = _D65[0] * _finv(fy + a / 500.0)
    Y = _D65[1] * _finv(fy)
    Z = _D65[2] * _finv(fy - b / 200.0)
    xyz = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    return xyz @ _XYZ_TO_SRGB.T.astype(np.float32)


# ------------------------------------------------------------------------------------------ prediction
def predict(L: np.ndarray) -> tuple[np.ndarray, float]:
    """Lab a, b (h, w, 2) at the model's own size for a picture's L* (any size). Returns (ab, milliseconds)."""
    from PIL import Image as PI
    m = model()
    shp = tuple(m.input_shape)
    nchw = len(shp) == 4 and shp[1] == 3
    H, W = (shp[2], shp[3]) if nchw else (shp[1], shp[2])
    Ls = np.asarray(PI.fromarray(np.ascontiguousarray(L, np.float32)).resize((W, H), PI.Resampling.BICUBIC))
    grey = sp.srgb_encode(np.clip(lab_to_linear_srgb(Ls, 0.0 * Ls, 0.0 * Ls), 0.0, 1.0))    # L with no colour
    x = np.transpose(grey, (2, 0, 1))[None] if nchw else grey[None]
    t = time.perf_counter()
    out = m(np.ascontiguousarray(x, np.float32))
    ms = 1000.0 * (time.perf_counter() - t)
    ab = np.asarray(next(iter(out.values())), np.float32)
    ab = ab[0]
    if ab.shape[0] == 2 and ab.shape[-1] != 2:
        ab = np.transpose(ab, (1, 2, 0))
    return np.ascontiguousarray(ab), ms


# ------------------------------------------------------------------------------------------ applying it
def guided(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter (He, Sun and Tang 2010): ``p`` smoothed so its edges follow the guide ``I``."""
    k = 2 * r + 1
    mI = ndi.uniform_filter(I, k, mode="reflect")
    mp = ndi.uniform_filter(p, k, mode="reflect")
    cov = ndi.uniform_filter(I * p, k, mode="reflect") - mI * mp
    var = ndi.uniform_filter(I * I, k, mode="reflect") - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return ndi.uniform_filter(a, k, mode="reflect") * I + ndi.uniform_filter(b, k, mode="reflect")


def pad_for(map_shape, full_shape) -> int:
    """Context (full-resolution pixels) a block needs for the edge-aware upsampling."""
    k = max(full_shape) / float(max(map_shape))
    return int(np.ceil(2.5 * max(2.0, k))) + 8


def even_tones(L: np.ndarray) -> np.ndarray:
    """What the colour model is shown: a faded or badly exposed photo's L* stretched to a normal range (0.5 and
    99.5 percentiles to L* 6 and 94). Adobe's own advice for its colourizer is to fix brightness and contrast
    first; here it is done for you, and only for the model's eyes: the photo keeps its own brightness."""
    lo, hi = np.percentile(L, [0.5, 99.5])
    if hi - lo < 5.0 or (lo <= 8.0 and hi >= 92.0):
        return L
    return np.clip((L - lo) / (hi - lo) * 88.0 + 6.0, 0.0, 100.0).astype(np.float32)


def display_target(rgb: np.ndarray, ab_block: np.ndarray | None, amount: float, step: int, k_map: float,
                   to_display, warm: float = 0.0, tint: float = 0.0):
    """The colour a scene-linear block should show, as display-linear ProPhoto, with each pixel's brightness as
    shown by the standard curve. ``ab_block``: the model's a, b at the block's pixels (h, w, 2), or None for no
    automatic colour (grey). Returns (target, display luminance, L*/100 as the edge guide)."""
    LUMA_PP = np.array([0.2880, 0.7119, 0.0001], np.float32)
    Y = (rgb.reshape(-1, 3) @ LUMA_PP).reshape(rgb.shape[:2])
    Yd = to_display(Y).astype(np.float32)
    L = L_of_Y(Yd).astype(np.float32)
    Ln = L / 100.0
    if ab_block is None or amount <= 0:
        return np.repeat(Yd[..., None], 3, axis=2), Yd, Ln
    r = max(1, int(round(1.5 * k_map / max(1, step))))
    # near black and near white no colour fits (it would leave the gamut and change the brightness): taper it
    t = np.clip(np.minimum(L, 100.0 - L) / 10.0, 0.0, 1.0)
    taper = (t * t * (3 - 2 * t)).astype(np.float32) * np.float32(amount)
    a = guided(Ln, ab_block[..., 0], r, 1e-4) * taper
    b = guided(Ln, ab_block[..., 1], r, 1e-4) * taper
    if warm or tint:                       # colour balance of the suggested colour: up to 12 a*/b* units each way
        tb = (t * t * (3 - 2 * t)).astype(np.float32)
        a = a + np.float32(0.12 * tint) * tb
        b = b + np.float32(0.12 * warm) * tb
    lin = lab_to_linear_srgb(L, a, b)
    disp_pp = sp.apply_matrix(np.ascontiguousarray(lin, np.float32), sp.conversion_matrix(sp.SRGB, sp.PROPHOTO))
    return np.maximum(disp_pp, 0.0), Yd, Ln


def material(rgb01, Yd: np.ndarray) -> np.ndarray:
    """A picked colour (sRGB 0..1) laid on light and shade: the colour's chromaticity at each pixel's own
    display brightness, as a dyed cloth or a painted wall looks under the light falling on it (reflectance times
    illumination: the shading stays, the colour is the material's). Pulled towards grey only as far as needed to
    stay inside the gamut, so the brightness never changes. Display-linear ProPhoto (h, w, 3)."""
    # worked in the render space (linear Rec. 2020), where the tone curve is applied channel by channel and
    # anything above 1 would be cut, changing the brightness
    lin = sp.srgb_decode(np.clip(np.asarray(rgb01, np.float64), 0.0, 1.0))
    rr = sp.conversion_matrix(sp.SRGB, sp.REC2020) @ lin
    luma = sp.REC2020.to_xyz[1]
    y = float(rr @ luma)
    c = (rr / y if y > 1e-6 else np.ones(3)).astype(np.float32)          # unit-luminance colour
    Y = Yd[..., None]
    full = Y * c                                                         # the colour at this brightness
    over = np.max(full, axis=2, keepdims=True)
    t = np.where(over > 1.0, (1.0 - Y) / np.maximum(over - Y, 1e-6), 1.0)
    t = np.clip(t, 0.0, 1.0)
    out = sp.apply_matrix(np.ascontiguousarray(Y + t * (full - Y), np.float32),
                          sp.conversion_matrix(sp.REC2020, sp.PROPHOTO))
    # the Studio measures brightness in ProPhoto (D50): match it there exactly
    LUMA_PP = np.array([0.2880, 0.7119, 0.0001], np.float32)
    yo = (out.reshape(-1, 3) @ LUMA_PP).reshape(Yd.shape)
    return (out * (Yd / np.maximum(yo, 1e-6))[..., None]).astype(np.float32)


def blend_hint(target: np.ndarray, Yd: np.ndarray, guide: np.ndarray, alpha: np.ndarray, rgb01, r: int) -> np.ndarray:
    """Lay one hint's colour into the target where its area is (alpha 0..1, coarse), its edge snapped to the
    photo's own edges (guided filter on brightness)."""
    a = np.clip(guided(guide, alpha.astype(np.float32), max(1, r), 1e-3), 0.0, 1.0)[..., None]
    return target + a * (material(rgb01, Yd) - target)


# ------------------------------------------------------------------------------------------ a part's area
def geodesic_area(L: np.ndarray, fg: list, bg: list | None = None, edge: float = 0.08) -> np.ndarray:
    """The part of a picture under the clicks, for a colour hint: everything reachable from a click without
    crossing a sharp edge in brightness (uint8 0/255, the picture's size).

    Distances are geodesic (Dijkstra on the pixel grid, scipy's csgraph): stepping between neighbours costs a
    little plus the square of their brightness difference, scaled so a step of ``edge`` (L*/100) costs one. The
    square is the point: a fold or shadow changes brightness gradually and costs almost nothing however deep it
    is, while the sharp edge of a collar, a sleeve against a wall or a badge stops the spread. Colour plays no part,
    so the blotches an automatic colourizer leaves on a garment do not cut it up, which is what makes Photoshop's
    colour points spread so unevenly. A pixel belongs to the part when it is nearer the part's clicks than one
    edge crossing, and nearer them than any Alt-click."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    h, w = L.shape
    Ls = ndi.gaussian_filter(np.asarray(L, np.float64), 1.2)            # film grain is not an edge
    lam = 1.0 / (edge * edge)
    eps = 0.6 / max(h, w)                                               # crossing the whole photo costs 0.6
    idx = np.arange(h * w).reshape(h, w)
    rows, cols, vals = [], [], []
    for dy, dx, k in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
        a = idx[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)] if dx >= 0 else idx[0:h - dy, 1:w]
        b = idx[max(0, dy):, max(0, dx):] if dx >= 0 else idx[dy:, 0:w - 1]
        a, b = a[:b.shape[0], :b.shape[1]], b[:a.shape[0], :a.shape[1]]
        la, lb = Ls.reshape(-1)[a.reshape(-1)], Ls.reshape(-1)[b.reshape(-1)]
        c = eps * k + lam * (la - lb) ** 2
        rows += [a.reshape(-1), b.reshape(-1)]
        cols += [b.reshape(-1), a.reshape(-1)]
        vals += [c, c]
    G = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w)).tocsr()

    def seeds(pts):
        out = []
        for x, y in pts or []:
            out.append(int(np.clip(round(y * h - 0.5), 0, h - 1)) * w + int(np.clip(round(x * w - 0.5), 0, w - 1)))
        return out
    f = seeds(fg)
    if not f:
        return np.zeros((h, w), np.uint8)
    df = dijkstra(G, directed=False, indices=f, min_only=True).reshape(h, w)
    inside = df < 1.0
    b = seeds(bg)
    if b:
        db = dijkstra(G, directed=False, indices=b, min_only=True).reshape(h, w)
        inside &= df < db
    lab, n = ndi.label(inside)
    keep = {int(lab.reshape(-1)[i]) for i in f} - {0}
    inside = np.isin(lab, list(keep))
    holes = ndi.binary_fill_holes(inside) & ~inside                     # specks inside the part join it
    hl, nh = ndi.label(holes)
    if nh:
        sizes = np.bincount(hl.ravel())
        small = np.nonzero(sizes < 0.0005 * h * w)[0]
        inside |= np.isin(hl, small[small > 0])
    return (inside * 255).astype(np.uint8)
