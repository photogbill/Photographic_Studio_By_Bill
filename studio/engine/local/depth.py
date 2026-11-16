"""Depth from one photo: Depth Anything V2 Small (Yang, Kang, Huang, Zhao, Xu, Feng, Zhao, NeurIPS 2024; Apache 2.0,
models/depth/NOTICE), run by the Studio's own NumPy runner (onnxnp.py). A Cognitive Adjustment.

    "depth": {"haze": -100..100}       atmospheric haze added (+) or taken away (-) by distance

What it gives
* A relative depth map of the whole photo (near = 1, far = 0), measured once per photo on a 518 x 518 copy (the
  model's own size, the photo squeezed to it; the map is stretched back) and kept with the photo.
* Depth masks (masks.py, shape "depth"): select by distance, from and to, snapped to the photo's edges.
* Depth haze: air between the camera and the scene, I = J t + A (1 - t) with t = exp(-k * farness) (Koschmieder);
  adding it pushes the distance back, taking it away (inverting the same model, limited) clears distant haze.
* The map itself, shown in colour on request.
Honest limits (the model's own, as Adobe lists for its Depth Blur): glass, mirrors and water confuse it, and a photo
with no clear near and far gives a flat map. It is relative depth: an order, not metres.
No GUI imports (rule 2.2).
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

MODEL = "depth_anything_v2_small.npz"
SIZE = 518
DEFAULTS = {"haze": 0.0}
_lock = threading.Lock()
_graph = None


def settings(d: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (d or {}).items() if k in DEFAULTS})
    return out


def active(d: dict | None) -> bool:
    return abs(float(settings(d)["haze"])) > 1e-6


def model_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "depth")


def model_path() -> str:
    return os.path.abspath(os.path.join(model_dir(), MODEL))


def available() -> bool:
    return os.path.exists(model_path())


def model():
    global _graph
    with _lock:
        if _graph is None:
            from .onnxnp import Graph
            _graph = Graph.load(model_path())
        return _graph


def predict(srgb01: np.ndarray) -> tuple[np.ndarray, float]:
    """Relative depth (near = 1, far = 0) of a display-encoded sRGB image (h, w, 3, 0..1), at its own size;
    (depth, milliseconds)."""
    import cv2
    h, w = srgb01.shape[:2]
    img = cv2.resize(np.ascontiguousarray(srgb01, np.float32), (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    x = ((img - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32))
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)
    t = time.perf_counter()
    d = model()(x)["predicted_depth"][0]
    ms = (time.perf_counter() - t) * 1000.0
    d = cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = np.percentile(d, [1.0, 99.0])
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32), ms


def haze(rgb: np.ndarray, D: np.ndarray, amount: float, airlight) -> np.ndarray:
    """Add (+) or take away (-) haze by distance on a scene-linear block. ``D``: depth (near = 1) at its pixels;
    ``amount``: -100..100; ``airlight``: the haze's colour and brightness (3,), scene-linear."""
    far = np.clip(1.0 - D, 0.0, 1.0)[..., None]
    A = np.asarray(airlight, np.float32)[None, None, :]
    k = abs(float(amount)) / 100.0 * 1.6
    t = np.exp(-k * far * far * 2.0)                       # nearby barely touched, the distance most
    if amount > 0:
        out = rgb * t + A * (1.0 - t)
    else:
        t = np.maximum(t, 0.35)                            # never divide distant haze into noise
        out = (rgb - A * (1.0 - t)) / t
    return np.maximum(out, 0.0).astype(np.float32)


def colour_map(D: np.ndarray) -> np.ndarray:
    """The depth map for the eye (RGBA uint8): warm near, cool far (the 'turbo' look, approximated)."""
    t = np.clip(D, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0, 1)
    rgba = np.stack([r, g, b, np.full_like(t, 0.85)], -1)
    return (rgba * 255).astype(np.uint8)
