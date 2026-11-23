"""Flatten before tiling (engine/stack/maps.py flatten_colour, flatten_height; surface.write with tileable): a
vignetted, tilted texture must lose the vignetting and the tilt and keep its fine detail, so the copies meet."""
from __future__ import annotations

import numpy as np

from studio.engine.stack import maps as MP


def _texture(n=512, seed=1):
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    detail = gaussian_filter(rng.standard_normal((n, n)).astype(np.float32), 2.0)
    detail /= np.abs(detail).max()
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    vign = 1.0 - 0.6 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) * 2      # corners about 0.6 stop darker... and more
    return detail, vign, x, y


def test_flatten_colour_removes_vignetting_keeps_detail():
    detail, vign, _, _ = _texture()
    base = 0.25 + 0.05 * detail                                   # linear
    from studio.engine.color.spaces import srgb_encode, srgb_decode
    col = np.repeat((srgb_encode(base * vign) * 255)[..., None], 3, 2).astype(np.float32)
    out, stops = MP.flatten_colour(col)
    lin = srgb_decode(out[..., 0] / 255.0)
    n = lin.shape[0]
    corner, centre = lin[:n // 8, :n // 8].mean(), lin[7 * n // 16:9 * n // 16, 7 * n // 16:9 * n // 16].mean()
    before = (base * vign)[:n // 8, :n // 8].mean() / (base * vign)[7 * n // 16:9 * n // 16, 7 * n // 16:9 * n // 16].mean()
    assert before < 0.75                                          # the test picture really is vignetted
    assert abs(corner / centre - 1) < 0.06                         # evened out
    assert stops > 0.3
    hp = lambda a: a - MP.low_pass(a, 0.02)                        # noqa: E731  fine detail
    assert np.corrcoef(hp(lin).ravel(), hp(base).ravel())[0, 1] > 0.95


def test_flatten_height_removes_tilt_keeps_relief():
    detail, _, x, y = _texture(seed=2)
    h = 0.6 * x + 0.3 * y + 0.05 * detail                          # tilted ground, small relief
    h = (h - h.min()) / (h.max() - h.min())
    out = MP.flatten_height(h)
    assert out.min() >= 0 and out.max() <= 1
    n = out.shape[0]
    q = [out[:n // 2, :n // 2].mean(), out[:n // 2, n // 2:].mean(), out[n // 2:, :n // 2].mean(), out[n // 2:, n // 2:].mean()]
    assert max(q) - min(q) < 0.05                                  # no quadrant higher than another
    assert np.corrcoef(out.ravel(), detail.ravel())[0, 1] > 0.9     # the relief is what is left


def test_tiled_flattened_height_meets_at_the_seams():
    detail, _, x, y = _texture(seed=3)
    h = 0.7 * x + 0.03 * detail
    raw = (h - h.min()) / (h.max() - h.min())
    flat = MP.flatten_height(h)
    step = lambda a: abs(float((a[:, 0] - a[:, -1]).mean()))       # noqa: E731  the systematic step at the seam
    assert step(raw) > 0.8 and step(flat) < 0.05
