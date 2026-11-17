"""Click to select: Segment Anything 2.1 (Ravi, Gabeur, Hu, Hu, Ryali, Ma, Khedr, Radle, Rolland, Gustafson, Mintun,
Pan, Alwala, Carion, Wu, Girshick, Dollar, Feichtenhofer, "SAM 2: Segment Anything in Images and Videos", Meta FAIR
2024; Apache 2.0, models/sam/NOTICE), run by the Studio's own NumPy runner (onnxnp.py). A Cognitive Adjustment.

Two sizes, chosen in Masks > Click-to-select (only the installed ones are offered):
    tiny   Hiera-Tiny, 39 M parameters: about 10 s to look at a photo on a desktop CPU, then each click is instant.
    large  Hiera-Large, 224 M parameters: finer edges on hair, leaves and thin parts; several times slower to look.
Either one looks at the photo once (an "image embedding", kept with the photo), then each click (+ add, Alt not
this) is a fraction of a second. The answer is the model's outline of the thing under the clicks: nothing is
painted or invented, it only says where the mask goes. Object masks and Colour a part use it; without a model the
Studio's own colour-and-edge method (masks.object_bitmap) is used.

How it is run (as Meta's SAM2ImagePredictor): the photo, as shown, squeezed to 1024 x 1024, ImageNet normalisation;
clicks in that square's pixels; three candidate outlines come back with the model's own quality score each, and the
best scored is used. The outline is 256 x 256 logits, stretched to the photo and turned into a soft edge.
No GUI imports (rule 2.2).
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

SIZE = 1024
SIZES = {"tiny": "Tiny (fast)", "large": "Large (finest edges)"}
_FILES = {"tiny": ("sam2.1_tiny_encoder.npz", "sam2.1_tiny_decoder.npz"),
          "large": ("sam2.1_large_encoder.npz", "sam2.1_large_decoder.npz")}
FEATS = ("image_embed", "high_res_feats_0", "high_res_feats_1")
_lock = threading.Lock()
_graphs: dict = {}


def model_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "sam"))


def files(size: str) -> tuple[str, str]:
    enc, dec = _FILES[size]
    return os.path.join(model_dir(), enc), os.path.join(model_dir(), dec)


def installed() -> list[str]:
    """The sizes whose files are in models/sam, smallest first."""
    return [s for s in _FILES if all(os.path.exists(p) for p in files(s))]


def available(size: str | None = None) -> bool:
    return bool(installed()) if size is None else size in installed()


def choose(pref: str | None) -> str | None:
    """The size to use for a preference ("tiny", "large", "classic" or None = the smallest installed): None means
    the Studio's own method."""
    if pref == "classic":
        return None
    have = installed()
    if pref in have:
        return pref
    return have[0] if have else None


def _graph(size: str, which: int):
    from .onnxnp import Graph
    key = (size, which)
    with _lock:
        if key not in _graphs:
            if which == 0:                                 # one encoder in memory at a time (Large is big)
                for k in [k for k in _graphs if k[1] == 0]:
                    del _graphs[k]
            _graphs[key] = Graph.load(files(size)[which])
        return _graphs[key]


def encode(srgb01: np.ndarray, size: str) -> tuple[dict, float]:
    """Look at a display-encoded sRGB photo (h, w, 3, 0..1): the image embedding (three feature maps) and
    milliseconds."""
    from ..imgops import resize_linear
    img = resize_linear(srgb01, SIZE, SIZE)
    x = (img - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)
    t = time.perf_counter()
    out = _graph(size, 0)(x)
    return {k: out[k] for k in FEATS}, (time.perf_counter() - t) * 1000.0


def decode(feats: dict, size: str, fg: list, bg: list | None = None) -> tuple[np.ndarray, float]:
    """The outline for the clicks (fg: on the thing, bg: not it; (x, y) fractions of the photo): the best scored of
    the model's three candidates as 256 x 256 logits (> 0 = inside), and its score."""
    pts = [(float(x) * SIZE, float(y) * SIZE) for x, y in fg] + [(float(x) * SIZE, float(y) * SIZE)
                                                                  for x, y in (bg or [])]
    lab = [1.0] * len(fg) + [0.0] * len(bg or [])
    out = _graph(size, 1)(point_coords=np.array([pts], np.float32), point_labels=np.array([lab], np.float32),
                          mask_input=np.zeros((1, 1, 256, 256), np.float32),
                          has_mask_input=np.zeros(1, np.float32), **feats)
    iou = out["iou_predictions"][0]
    k = int(np.argmax(iou))
    return out["masks"][0, k].astype(np.float32), float(iou[k])


def bitmap(logits: np.ndarray, h: int, w: int) -> np.ndarray:
    """The outline stretched to (h, w) as a soft-edged uint8 mask (255 inside)."""
    from ..imgops import resize_linear
    up = resize_linear(logits, w, h)
    return np.round(255.0 / (1.0 + np.exp(-np.clip(up, -30, 30)))).astype(np.uint8)
