"""Multi-band blending for the panorama (Burt and Adelson 1983, "A multiresolution spline with application to image
mosaics"), on the Studio's own pyramids (detail/pyramid.py, the 5-tap binomial kernel the Focus Stack uses).

Each frame's Laplacian pyramid is weighted by the Gaussian pyramid of its label mask and the weights are normalised
level by level, so fine detail switches sharply at the seam while broad tones blend over a wide band: no visible step
in a sky, no ghosting of fine detail. ``fill`` first extends every frame past its own edge (push-pull), so a frame's
edge never shows up as a false detail near the seam.

Tiled: ``blend_tile`` works on a tile plus a margin of ``margin(levels)`` pixels and tiles whose origins are multiples of
2^levels join exactly (the test checks tiles against one-piece blending).
"""
from __future__ import annotations

import numpy as np

from ..detail import pyramid as P


def margin(levels: int) -> int:
    return 6 * (2 ** levels)


def fill(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Push-pull: invalid pixels get a smooth extension of the valid ones (coarse averages of valid pixels only)."""
    v = valid.astype(np.float32)
    if v.all():
        return img.astype(np.float32)
    x = img.astype(np.float32) * (v[..., None] if img.ndim == 3 else v)
    stack = [(x, v)]
    while min(stack[-1][1].shape) > 2 and stack[-1][1].min() <= 0:
        xs, vs = stack[-1]
        h, w = vs.shape
        hh, ww = (h + 1) // 2, (w + 1) // 2
        pad = ((0, hh * 2 - h), (0, ww * 2 - w))
        vp = np.pad(vs, pad)
        xp = np.pad(xs, pad + (((0, 0),) if xs.ndim == 3 else ()))
        v2 = vp.reshape(hh, 2, ww, 2).sum(axis=(1, 3))
        x2 = xp.reshape((hh, 2, ww, 2) + xs.shape[2:]).sum(axis=(1, 3))
        stack.append((x2, v2))
    xs, vs = stack[-1]
    cur = xs / np.maximum(vs, 1e-12)[..., None] if xs.ndim == 3 else xs / np.maximum(vs, 1e-12)
    for xs, vs in reversed(stack[:-1]):
        h, w = vs.shape
        up = np.repeat(np.repeat(cur, 2, axis=0), 2, axis=1)[:h, :w]
        mean = xs / np.maximum(vs, 1e-12)[..., None] if xs.ndim == 3 else xs / np.maximum(vs, 1e-12)
        has = vs > 0
        cur = np.where(has[..., None] if xs.ndim == 3 else has, mean, up)
    out = img.astype(np.float32).copy()
    m = ~valid
    out[m] = cur[m]
    return out


def blend(images: list, masks: list, levels: int) -> np.ndarray:
    """Blend same-size images (H x W x 3, already filled past their edges) by soft masks (H x W, summing to about 1
    where anything is valid). Returns H x W x 3."""
    H, W = masks[0].shape
    lv = max(1, min(levels, P.max_levels((H, W), 8)))
    acc = None
    wacc = None
    for img, m in zip(images, masks):
        if not np.any(m):
            continue
        gm = P.gaussian(m.astype(np.float32), lv)
        chans = []
        for c in range(3):
            g = P.gaussian(np.asarray(img[..., c], np.float32), lv)
            chans.append(P.laplacian_from_gaussian(g))
        if acc is None:
            acc = [[np.zeros_like(chans[c][l]) for l in range(lv + 1)] for c in range(3)]
            wacc = [np.zeros_like(gm[l]) for l in range(lv + 1)]
        for l in range(lv + 1):
            wacc[l] += gm[l]
            for c in range(3):
                acc[c][l] += gm[l] * chans[c][l]
    if acc is None:
        return np.zeros((H, W, 3), np.float32)
    out = np.empty((H, W, 3), np.float32)
    for c in range(3):
        lap = [acc[c][l] / np.maximum(wacc[l], 1e-6) for l in range(lv + 1)]
        out[..., c] = P.collapse(lap)
    return out
