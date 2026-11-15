"""Smooth skin (mask adjustment) and the Skin mask."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from conftest import synthetic_raw
from studio import engine as E
from studio.engine.color.grade import to_oklab
from studio.engine.color.spaces import PROPHOTO, SRGB, conversion_matrix
from studio.engine.local import masks as mk
from studio.engine.pipeline.pipeline import Image


def _face(seed=5, h=600, w=600):
    """Log-luminance layers of a synthetic skin patch: shading, blotches, pores, and an 'eye' (a strong edge)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    shade = np.log2(0.3) + 0.3 * np.sin(xx / 300.0)
    blot = np.zeros((h, w))
    for _ in range(25):
        cx, cy = rng.uniform(50, w - 50, 2)
        r = rng.uniform(5, 10)
        blot += -0.25 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    pores = 0.06 * gaussian_filter(rng.standard_normal((h, w)), 0.8) / 0.25
    eye = gaussian_filter(np.where(((xx - 300) / 80.0) ** 2 + ((yy - 300) / 35.0) ** 2 < 1, -1.6, 0.0), 1.0)
    far = np.hypot((xx - 300) / 80.0, (yy - 300) / 35.0) > 1.6
    return shade, blot, pores, eye, far


def test_smooth_skin_softens_blotches_keeps_pores_and_edges():
    shade, blot, pores, eye, far = _face()
    Y = 2.0 ** (shade + blot + pores + eye)
    rgb = np.stack([Y * 1.25, Y, Y * 0.75], -1).astype(np.float32)          # a skin colour
    out = mk.smooth_skin(rgb, 100, 1.0, 6000.0)
    Li = np.log2(np.maximum(rgb @ mk.LUMA, 1e-7))
    Lo = np.log2(np.maximum(out @ mk.LUMA, 1e-7))
    kept = np.polyfit(blot[far], (Lo - Li)[far] + blot[far], 1)[0]
    assert kept < 0.5                                                        # most of the blotchiness gone
    pore_in = np.corrcoef(pores[far], (Li - gaussian_filter(Li, 2.0))[far])[0, 1]
    pore_out = np.polyfit((Li - gaussian_filter(Li, 2.0))[far], (Lo - gaussian_filter(Lo, 2.0))[far], 1)[0]
    assert pore_in > 0.8 and pore_out > 0.75                                 # pores (fine texture) kept
    assert (Lo[300, 300] - Lo[300, 450]) / (Li[300, 300] - Li[300, 450]) > 0.9   # the eye keeps its contrast
    assert np.array_equal(mk.smooth_skin(rgb, 0, 1.0, 6000.0), rgb)


def test_skin_weight_covers_every_complexion_but_not_eyes_or_hair():
    M = conversion_matrix(SRGB, PROPHOTO)

    def w(srgb8):
        c = np.asarray(srgb8, float) / 255.0
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return float(mk.skin_weight(to_oklab((lin @ M.T).astype(np.float32)[None, None]))[0, 0])
    for skin in ((234, 192, 170), (198, 134, 100), (110, 70, 50), (60, 40, 32)):
        assert w(skin) > 0.9
    assert w((230, 225, 220)) < 0.1             # white of the eye
    assert w((30, 25, 22)) < 0.1                # black hair
    assert w((60, 110, 200)) < 0.05             # a blue shirt


def test_skin_mask_with_smooth_skin_renders_and_region_matches():
    shade, blot, pores, eye, _ = _face(h=240, w=360)
    Y = 2.0 ** (shade + blot + pores)
    sc = np.stack([Y * 1.25, Y, Y * 0.75], -1).astype(np.float32)
    sc[:, :60] = [0.05, 0.12, 0.3]                                           # a blue area: not skin
    img = Image(synthetic_raw(np.clip(sc, 0, 0.9)))
    m = {"name": "Skin", "shapes": [{"type": "skin"}], "adjust": {"smooth_skin": 80}}
    e = E.Edit(masks=[m])
    a = E.render_mask(img, e, 0, scale=1.0)
    assert a[120, 200] > 0.8 and a[120, 20] < 0.1
    full = E.render(img, e)
    plain = E.render(img, E.Edit())
    assert np.abs(full - plain)[:, :40].max() < 1e-3                         # outside the skin: untouched
    assert np.abs(full - plain)[:, 100:].max() > 0.01                        # skin: changed
    part = E.render(img, e, region=(150, 60, 120, 100))
    assert np.abs(part - full[60:160, 150:270]).max() < 3e-3
