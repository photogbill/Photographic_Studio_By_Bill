"""Phase 5: automatic lens fixes without a profile (section 10.2) and the named defringe test."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from conftest import mosaic_of, scene
from studio.engine.lens import auto as A

PAT = np.array([[0, 1], [1, 2]])


def _ca_scene(kr=0.003, kb=-0.002, h=512, w=768):
    img = scene(h, w)
    y, x = np.mgrid[0:h, 0:w]
    img = img * (0.6 + 0.4 * ((np.sin(x / 7.0) * np.sin(y / 9.0)) > 0))[..., None]
    cy, cx = (h - 1) / 2, (w - 1) / 2

    def mag(ch, k):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        return map_coordinates(ch, [cy + (yy - cy) / (1 + k), cx + (xx - cx) / (1 + k)], order=3, mode="nearest")
    src = img.copy()
    src[..., 0], src[..., 2] = mag(img[..., 0], kr), mag(img[..., 2], kb)
    return mosaic_of(src), np.hypot(cy, cx) / 2


def test_lateral_ca_is_measured_and_mostly_removed():
    mos, rmax_plane = _ca_scene()
    ca = A.estimate_lateral_ca(mos, PAT)
    assert ca.confidence != "none"
    r = 0.7
    assert 0.003 * r * rmax_plane * 0.7 < ca.red[0] * r + ca.red[1] * r ** 3 < 0.003 * r * rmax_plane * 1.3
    fixed = A.correct_lateral_ca(mos, PAT, ca)
    ca2 = A.estimate_lateral_ca(fixed, PAT)
    before = abs(ca.red[0] * r + ca.red[1] * r ** 3)
    after = abs(ca2.red[0] * r + ca2.red[1] * r ** 3)
    assert after < 0.3 * before


def test_no_ca_means_no_correction():
    mos = mosaic_of(scene(512, 768))
    assert A.estimate_lateral_ca(mos, PAT).confidence == "none"


def test_purple_fringe_removed_next_to_bright_edges():
    img = np.full((120, 160, 3), 0.15, np.float32)
    img[:, 80:] = 0.98                                   # a bright, near-clipped area (window, candle)
    img[:, 76:80] = np.array([0.5, 0.15, 0.7])           # purple halo on the edge
    out = A.defringe(img)
    px = out[60, 78]
    assert (0.5 * (px[0] + px[2]) - px[1]) < 0.4 * (0.5 * (0.5 + 0.7) - 0.15)


def test_named_purple_flower_survives():
    """Named test (section 17): a purple flower with soft edges keeps its colour."""
    y, x = np.mgrid[0:200, 0:200]
    soft = np.clip(1.2 - np.hypot(y - 100, x - 100) / 60.0, 0, 1)[..., None]
    img = (0.12 + soft * (np.array([0.45, 0.12, 0.55]) - 0.12)).astype(np.float32)
    out = A.defringe(img)
    assert np.allclose(out, img, atol=1e-4)
