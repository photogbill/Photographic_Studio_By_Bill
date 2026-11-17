"""JPEG repair inside the file's own quantisation intervals. A Cognitive Adjustment.

A JPEG stores each 8 x 8 block as DCT coefficients rounded to multiples of the file's quantisation table. So the
picture before compression had, in every block, coefficients somewhere inside [q (k - 1/2), q (k + 1/2)] - the
file says no more than that. The usual decoder shows the middle of every interval, which is where the blocks and
the ringing round edges come from. Here we pick, among all the pictures the file allows, the smoothest one: total
variation (Huber-smoothed) is lowered by gradient steps, and after every step each block is put back inside its
intervals (projection onto convex sets; Zakhor 1992, Yang, Galatsanos and Katsaggelos 1993; total variation as the
prior as in Alter, Durand and Froment 2005 and Bredies and Holler 2012). Every result is a picture that compresses
to exactly the same file: nothing is invented, only the decoder's guess inside the rounding is improved.

Colour is done in the file's own YCbCr: brightness at full size with the brightness table, the two colour planes at
their stored (usually halved) size with the colour table, then brought back to full size. Output keeps 16 bits.
No GUI imports (rule 2.2).
"""
from __future__ import annotations

import time

import numpy as np

ITERATIONS = 20                              # more smooths fine texture at good qualities (tested)
EPS = 2.0                                    # Huber smoothing of total variation, in 8-bit levels
TAU = 0.22                                   # gradient step (stable below EPS / 8)
SAFE = 0.48                                  # half-width of each interval used, in steps of the table
_STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                      14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                      92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float).reshape(8, 8)


def _dct_matrix() -> np.ndarray:
    k = np.arange(8)[:, None]
    n = np.arange(8)[None, :]
    C = np.cos(np.pi * (2 * n + 1) * k / 16.0) * np.sqrt(2.0 / 8.0)
    C[0] /= np.sqrt(2.0)
    return C


_C = _dct_matrix().astype(np.float32)


def info(path: str) -> dict | None:
    """The JPEG's quantisation tables (8 x 8, natural order), components and sampling, or None for other files."""
    from PIL import Image as PI
    try:
        with PI.open(path) as im:
            if im.format != "JPEG" or not getattr(im, "quantization", None):
                return None
            q = {int(k): np.asarray(list(v), float).reshape(8, 8) for k, v in im.quantization.items()}
            layers = [tuple(int(x) for x in L) for L in (getattr(im, "layer", None) or [])]
            if im.mode not in ("L", "RGB") or not layers:
                return None
            return {"q": q, "layers": layers, "mode": im.mode, "size": im.size, "quality": quality(q[0])}
    except Exception:
        return None


def quality(q_luma: np.ndarray) -> int:
    """The IJG quality setting closest to a brightness table (for the report; other encoders scale differently)."""
    s = float(np.mean(np.asarray(q_luma, float) / _STD_LUMA)) * 100.0
    qual = (200.0 - s) / 2.0 if s <= 100.0 else 5000.0 / s
    return int(np.clip(round(qual), 1, 100))


def _blocks(x: np.ndarray) -> np.ndarray:
    h, w = x.shape
    return x.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)


def _unblocks(b: np.ndarray) -> np.ndarray:
    nh, nw = b.shape[:2]
    return b.transpose(0, 2, 1, 3).reshape(nh * 8, nw * 8)


def bdct(x: np.ndarray) -> np.ndarray:
    return _C @ _blocks(x) @ _C.T


def bidct(c: np.ndarray) -> np.ndarray:
    return _unblocks(_C.T @ c @ _C)


def _tv_grad(x: np.ndarray) -> np.ndarray:
    dx = np.diff(x, axis=1, append=x[:, -1:])
    dy = np.diff(x, axis=0, append=x[-1:, :])
    n = np.sqrt(dx * dx + dy * dy + EPS * EPS)
    px, py = dx / n, dy / n
    div = px - np.pad(px, ((0, 0), (1, 0)))[:, :-1] + py - np.pad(py, ((1, 0), (0, 0)))[:-1, :]
    return -div


def repair_plane(p: np.ndarray, q: np.ndarray, iterations: int = ITERATIONS) -> np.ndarray:
    """One plane (0..255 floats, as decoded) repaired inside the intervals of table ``q`` (8 x 8)."""
    h, w = p.shape
    H, W = -(-h // 8) * 8, -(-w // 8) * 8
    x = np.pad(np.asarray(p, np.float32), ((0, H - h), (0, W - w)), mode="edge")
    q = np.asarray(q, np.float32)
    c = bdct(x - 128.0)
    k = np.round(c / q)
    lo, hi = (k - SAFE) * q, (k + SAFE) * q          # a hair inside, so the file's own rounding is kept exactly
    x = bidct(np.clip(c, lo, hi)) + 128.0
    for _ in range(iterations):
        x = x - TAU * _tv_grad(x)
        x = bidct(np.clip(bdct(x - 128.0), lo, hi)) + 128.0
    return x[:h, :w]


def consistent(p: np.ndarray, q: np.ndarray, ref: np.ndarray) -> float:
    """Share of the coefficients of ``p`` that round to the same values as those of ``ref`` (the decoded file)."""
    h, w = p.shape
    H, W = h // 8 * 8, w // 8 * 8
    a = np.round(bdct(np.asarray(p, np.float64)[:H, :W] - 128.0) / q)
    b = np.round(bdct(np.asarray(ref, np.float64)[:H, :W] - 128.0) / q)
    return float((a == b).mean())


def repair_file(path: str, iterations: int = ITERATIONS) -> tuple[np.ndarray, dict]:
    """The JPEG at ``path`` decoded and repaired, as stored (not yet turned upright): uint16 (h, w) or (h, w, 3),
    and a report."""
    from PIL import Image as PI

    from ..imgops import resize_linear
    t = time.perf_counter()
    inf = info(path)
    if inf is None:
        raise ValueError("not a JPEG the repair can read")
    q = inf["q"]
    with PI.open(path) as im:
        if inf["mode"] == "L":
            y = np.asarray(im, np.float64)
            out = repair_plane(y, q[inf["layers"][0][3]], iterations)
            res = np.clip(out, 0, 255)
        else:
            im.draft("YCbCr", im.size)
            ycc = np.asarray(im.convert("YCbCr") if im.mode != "YCbCr" else im, np.float64)
            hmax = max(L[1] for L in inf["layers"])
            vmax = max(L[2] for L in inf["layers"])
            h, w = ycc.shape[:2]
            planes = []
            for c, L in enumerate(inf["layers"][:3]):
                fx, fy = hmax // L[1], vmax // L[2]
                pl = ycc[..., c]
                if fx > 1 or fy > 1:                     # the colour planes at their stored size
                    sh, sw = -(-h // fy), -(-w // fx)
                    pp = np.pad(pl, ((0, sh * fy - h), (0, sw * fx - w)), mode="edge")
                    small = pp.reshape(sh, fy, sw, fx).mean(axis=(1, 3))
                    fixed = repair_plane(small, q[L[3]], iterations)
                    pl = resize_linear(fixed, sw * fx, sh * fy)[:h, :w].astype(np.float64)
                else:
                    pl = repair_plane(pl, q[L[3]], iterations)
                planes.append(pl)
            Y, Cb, Cr = planes[0], planes[1] - 128.0, planes[2] - 128.0
            res = np.clip(np.stack([Y + 1.402 * Cr, Y - 0.344136 * Cb - 0.714136 * Cr, Y + 1.772 * Cb], -1), 0, 255)
    rep = {"quality": inf["quality"], "ms": (time.perf_counter() - t) * 1e3, "iterations": iterations}
    return np.round(res * (65535.0 / 255.0)).astype(np.uint16), rep


def blockiness(y: np.ndarray) -> float:
    """Mean step across 8-pixel block edges over the mean step elsewhere (1 = no blocks)."""
    y = np.asarray(y, np.float64)
    dx = np.abs(np.diff(y, axis=1))
    edge = dx[:, 7::8].mean()
    cols = np.ones(dx.shape[1], bool)
    cols[7::8] = False
    return float(edge / max(dx[:, cols].mean(), 1e-9))
