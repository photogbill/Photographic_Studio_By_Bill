"""The scratch detector of "Bringing Old Photos Back to Life" (Wan et al., CVPR 2020; Microsoft, MIT licence),
run here in NumPy: no PyTorch, nothing to install.

It is a U-Net (Ronneberger et al. 2015) with blur-pooled downsampling (Zhang 2019, "Making convolutional networks
shift-invariant again"), trained by its authors on real old photographs with hand-marked scratches, cracks and
dust. Given a photo in grey, it gives each pixel the probability of being damage. It knows what scratches on old
prints look like, which a filter does not: a white line across a face is damage, the white line of a collar is
not. Its authors run it with the short side at 256 pixels, where it was trained; scratches.py uses its map as the
judge of which marks are damage, and its own full-resolution search for exactly where each mark's edges are.

The weights (models/scratches/scratch_detector.npz) are Microsoft's, converted by
tools/convert_scratch_detector.py: batch normalisation folded into each convolution, stored as 8-bit integers with
one scale per filter (38 MB instead of 151; the damage map agrees with the float model on 99.9% of pixels).
"""
from __future__ import annotations

import os

import numpy as np

from ... import log as slog
from ... import paths

log = slog.get("engine.scratchnet")

MODEL = "scratch_detector.npz"
DEPTH = 4
NET_SHORT = 256                         # the short side the authors trained and run it at


def model_path() -> str:
    return os.path.join(str(paths.APP_ROOT), "models", "scratches", MODEL)


def available() -> bool:
    return os.path.isfile(model_path())


# ------------------------------------------------------------------------------------------ conversion
def convert(sd: dict) -> dict:
    """Microsoft's PyTorch state dict (name -> float array) -> the arrays this runner reads: each convolution as
    (kh, kw, cin, cout) with its batch normalisation folded in."""
    out = {}

    def conv(prefix, bn=None, name=None):
        w = np.asarray(sd[prefix + ".weight"], np.float64)            # (cout, cin, kh, kw)
        b = np.asarray(sd.get(prefix + ".bias", np.zeros(w.shape[0])), np.float64)
        if bn is not None:
            g, beta = np.asarray(sd[bn + ".weight"], np.float64), np.asarray(sd[bn + ".bias"], np.float64)
            m, v = np.asarray(sd[bn + ".running_mean"], np.float64), np.asarray(sd[bn + ".running_var"], np.float64)
            k = g / np.sqrt(v + 1e-5)
            w = w * k[:, None, None, None]
            b = (b - m) * k + beta
        w = np.transpose(w, (2, 3, 1, 0))                                # (kh, kw, cin, cout)
        scale = np.abs(w).reshape(-1, w.shape[-1]).max(axis=0) / 127.0 + 1e-12
        out[name + ".wq"] = np.clip(np.round(w / scale), -127, 127).astype(np.int8)    # 8 bits per weight,
        out[name + ".ws"] = scale.astype(np.float32)                                     # one scale per filter
        out[name + ".b"] = b.astype(np.float32)

    conv("first.1", None, "first")
    for i in range(DEPTH):
        conv(f"down_sample.{i}.1", f"down_sample.{i}.2", f"ds{i}")
        conv(f"down_path.{i}.block.1", f"down_path.{i}.block.2", f"dp{i}a")
        conv(f"down_path.{i}.block.5", f"down_path.{i}.block.6", f"dp{i}b")
        conv(f"up_path.{i}.up.2", None, f"up{i}")
        conv(f"up_path.{i}.conv_block.block.1", f"up_path.{i}.conv_block.block.2", f"up{i}a")
        conv(f"up_path.{i}.conv_block.block.5", f"up_path.{i}.conv_block.block.6", f"up{i}b")
    conv("last.1", None, "last")
    return out


# ------------------------------------------------------------------------------------------ layers (HWC float32)
def _reflect(x, p):
    return np.pad(x, ((p, p), (p, p), (0, 0)), mode="reflect")


def _conv(x, w, b, act=True):
    """Valid convolution of a reflect-padded (h, w, cin) array with (kh, kw, cin, cout) weights, + bias,
    LeakyReLU(0.2). Tap by tap: one matrix product per kernel position, strip by strip to bound memory."""
    kh, kw, cin, cout = w.shape
    p = kh // 2
    xp = _reflect(x, p)
    H, W = x.shape[:2]
    out = np.empty((H, W, cout), np.float32)
    wm = w.astype(np.float32).reshape(kh * kw * cin, cout)
    rows = max(1, int(6e6 // max(1, W * kh * kw * cin)))
    for y0 in range(0, H, rows):            # im2col per strip: one large matrix product (BLAS) per strip
        y1 = min(H, y0 + rows)
        cols = np.concatenate([xp[y0 + i:y1 + i, j:j + W] for i in range(kh) for j in range(kw)], axis=2)
        out[y0:y1] = (cols.reshape(-1, kh * kw * cin) @ wm).reshape(y1 - y0, W, cout)
    out += b
    if act:
        np.maximum(out, 0.2 * out, out=out)
    return out


def _blurpool(x):
    """Anti-aliased stride-2 downsampling: [1 2 1] x [1 2 1] / 16, reflect padding (Zhang 2019)."""
    xp = _reflect(x, 1)
    k = (1.0, 2.0, 1.0)
    H, W = x.shape[:2]
    oh, ow = (H + 1) // 2, (W + 1) // 2
    acc = np.zeros((oh, ow, x.shape[2]), np.float32)
    for i in range(3):
        for j in range(3):
            acc += (k[i] * k[j] / 16.0) * xp[i:i + 2 * oh:2, j:j + 2 * ow:2]
    return acc


def _up2(x):
    """Bilinear x2, align_corners=False (PyTorch's nn.Upsample(scale_factor=2, mode='bilinear'))."""
    def axis(n):
        src = (np.arange(2 * n) + 0.5) / 2.0 - 0.5
        src = np.clip(src, 0, n - 1)
        i0 = np.floor(src).astype(np.int64)
        i1 = np.minimum(i0 + 1, n - 1)
        f = (src - i0).astype(np.float32)
        return i0, i1, f
    H, W = x.shape[:2]
    y0, y1, fy = axis(H)
    x0, x1, fx = axis(W)
    a = x[y0] * (1 - fy)[:, None, None] + x[y1] * fy[:, None, None]
    return a[:, x0] * (1 - fx)[None, :, None] + a[:, x1] * fx[None, :, None]


class ScratchNet:
    def __init__(self, weights: dict):
        self.w = weights

    @classmethod
    def load(cls, path: str | None = None) -> "ScratchNet":
        p = path or model_path()
        w = {}
        with np.load(p) as z:
            for k in z.files:
                if k.endswith(".wq"):
                    w[k[:-3] + ".w"] = (z[k].astype(np.float32) * z[k[:-3] + ".ws"]).astype(np.float32)
                elif not k.endswith(".ws"):
                    w[k] = z[k]
        return cls(w)

    def _c(self, x, name, act=True):
        return _conv(x, self.w[name + ".w"], self.w[name + ".b"], act)

    def probability(self, grey: np.ndarray) -> np.ndarray:
        """Damage probability (h, w) for a grey picture (h, w) in 0..1 as displayed (sRGB-encoded), each side a
        multiple of 16 (the four poolings)."""
        H, W = grey.shape
        if H % 16 or W % 16:
            raise ValueError(f"scratch detector input {W}x{H}: sides must be multiples of 16")
        x = ((np.asarray(grey, np.float32) - 0.5) / 0.5)[..., None]
        x = self._c(x, "first")
        skips = []
        for i in range(DEPTH):
            skips.append(x)
            x = _blurpool(self._c(x, f"ds{i}"))
            x = self._c(self._c(x, f"dp{i}a"), f"dp{i}b")
        for i in range(DEPTH):
            up = self._c(_up2(x), f"up{i}", act=False)
            x = np.concatenate([up, skips[-i - 1]], axis=2)
            x = self._c(self._c(x, f"up{i}a"), f"up{i}b")
        logit = self._c(x, "last", act=False)[..., 0]
        return (1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))).astype(np.float32)


_NET = None


def net() -> ScratchNet | None:
    """The detector, loaded once; None when the model file is not installed."""
    global _NET
    if _NET is None and available():
        with slog.stage("scratchnet.load", log):
            _NET = ScratchNet.load()
    return _NET


def short_for(h: int, w: int) -> int:
    """The scale to run it at: its own 256 for small pictures; 384 from 1000 pixels up, where thin scratches
    would otherwise fall below a pixel (measured: more of a thin scratch is seen and fewer false marks made)."""
    return NET_SHORT if min(h, w) < 1000 else 384


def damage_map(grey: np.ndarray, short: int = NET_SHORT) -> np.ndarray:
    """Probability of damage over a whole grey picture (display-encoded 0..1), at the detector's own scale (short
    side ``short``, rounded to multiples of 16), returned at that scale."""
    from PIL import Image as PI
    n = net()
    if n is None:
        raise FileNotFoundError(model_path())
    H, W = grey.shape
    if H <= W:
        h = short
        w = int(round(W * short / H / 16.0)) * 16
    else:
        w = short
        h = int(round(H * short / W / 16.0)) * 16
    h, w = max(16, int(round(h / 16.0)) * 16), max(16, w)
    small = np.asarray(PI.fromarray(np.ascontiguousarray(grey, np.float32)).resize((w, h), PI.Resampling.BICUBIC))
    with slog.stage("scratchnet.run", log, w=w, h=h):
        return n.probability(np.clip(small, 0.0, 1.0))
