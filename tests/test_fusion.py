"""Visible + infrared fusion (BUILD_PLAN_PANO_IR.md 1.7), on synthetic pairs: alignment across spectra on edges,
the colour-infrared channels, and NDVI against the model's own values."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from studio.engine.stack import align
from studio.engine.stack import fusion as FU


def _scene(h=480, w=640, seed=4):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    lab = np.zeros((h, w), np.int32)                 # 0 foliage, 1 sky, 2 building
    lab[: h // 3] = 1
    lab[h // 2:, w // 2: w // 2 + 120] = 2
    blobs = gaussian_filter(rng.normal(0, 1, (h, w)), 3) > 0.02
    tex = 1 + 0.3 * gaussian_filter(rng.normal(0, 1, (h, w)), 1.2)
    # visible: foliage dark green, sky blue, building grey; IR: foliage bright, sky dark, building grey
    vis_r = np.choose(lab, [0.04, 0.10, 0.25]) * tex
    vis_g = np.choose(lab, [0.08, 0.18, 0.25]) * tex
    vis_b = np.choose(lab, [0.03, 0.45, 0.25]) * tex
    ir = np.choose(lab, [0.55, 0.06, 0.28]) * tex
    vis_r[(lab == 0) & blobs] *= 0.7
    ir[(lab == 0) & blobs] *= 0.8
    vis = np.stack([vis_r, vis_g, vis_b], -1).astype(np.float32)
    irr = np.repeat(ir[..., None], 3, axis=2).astype(np.float32)
    return vis, irr, lab


def _warp_by(img, H):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # the IR frame's pixel q shows the visible frame's point H^-1 q
    Hi = np.linalg.inv(H)
    X = Hi[0, 0] * xx + Hi[0, 1] * yy + Hi[0, 2]
    Y = Hi[1, 0] * xx + Hi[1, 1] * yy + Hi[1, 2]
    W = Hi[2, 0] * xx + Hi[2, 1] * yy + Hi[2, 2]
    return np.stack([map_coordinates(img[..., c], [Y / W, X / W], order=3, mode="reflect")
                     for c in range(img.shape[2])], -1).astype(np.float32)


def test_alignment_across_spectra_within_a_fifth_of_a_pixel():
    vis, ir, _ = _scene()
    a = np.deg2rad(0.6)
    H = np.array([[np.cos(a), -np.sin(a), 6.3], [np.sin(a), np.cos(a), -4.1], [1.5e-6, -1e-6, 1.0]])
    ir_moved = _warp_by(ir, H)
    T, flw = FU.align_pair(align.luma(vis), align.luma(ir_moved), local=False)
    ys, xs = np.mgrid[60:420:60, 80:560:80].astype(float)
    ty, tx = T.apply(ys, xs)
    W = H[2, 0] * xs + H[2, 1] * ys + H[2, 2]
    hx = (H[0, 0] * xs + H[0, 1] * ys + H[0, 2]) / W
    hy = (H[1, 0] * xs + H[1, 1] * ys + H[1, 2]) / W
    err = np.hypot(ty - hy, tx - hx)
    assert err.max() < 0.2, err.max()


def test_cir_channels_are_the_inputs_and_ndvi_matches_the_model():
    vis, ir, lab = _scene()
    out, _ = FU.fuse(vis, ir, "cir")
    assert np.allclose(out[..., 0], ir.mean(axis=2)) and np.allclose(out[..., 1], vis[..., 0])
    assert np.allclose(out[..., 2], vis[..., 1])
    _, v = FU.fuse(vis, ir, "ndvi")
    want = {0: (0.55 - 0.04) / (0.55 + 0.04), 1: (0.06 - 0.10) / (0.06 + 0.10)}
    for k, val in want.items():
        sel = (lab == k)
        if k == 0:
            sel &= np.abs(vis[..., 0] / np.maximum(ir[..., 0], 1e-6) - 0.04 / 0.55) < 1e-3
        assert abs(float(np.median(v[sel])) - val) < 0.02, (k, float(np.median(v[sel])), val)


def test_detail_mode_takes_lightness_from_infrared_and_colour_from_visible():
    from studio.engine.color.grade import to_oklab
    vis, ir, lab = _scene()
    out, _ = FU.fuse(vis, ir, "detail", 1.0)
    L_out = to_oklab(out)[..., 0]
    L_ir = to_oklab(ir)[..., 0]
    assert np.abs(L_out - L_ir).mean() < 0.01
    hue_v = np.arctan2(to_oklab(vis)[..., 2], to_oklab(vis)[..., 1])
    hue_o = np.arctan2(to_oklab(out)[..., 2], to_oklab(out)[..., 1])
    sky = lab == 1
    assert np.median(np.abs(np.angle(np.exp(1j * (hue_o[sky] - hue_v[sky]))))) < 0.05


def test_every_mode_says_what_it_is_for():
    for k, d in FU.METHOD_INFO.items():
        assert d["name"] and d["best"] and d["weak"] and d["does"]
