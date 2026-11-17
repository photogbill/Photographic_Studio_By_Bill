"""Panorama features (BUILD_PLAN_PANO_IR.md 2.3): pairs found on the synthetic sphere, outliers rejected."""
from __future__ import annotations

import numpy as np

from pano_synth import render, rot, world
from studio.engine.stack import features as FE


def _K(f, w, h):
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])


def test_matching_two_overlapping_frames_finds_the_true_homography():
    tex = world()
    f, w, h = 600.0, 640, 480
    R1, R2 = rot(0, 1, 0.5), rot(24, -1, -0.5)
    a, b = render(tex, R1, f, w, h), render(tex, R2, f, w, h)
    fa, fb = FE.FrameFeatures(a), FE.FrameFeatures(b)
    assert len(fa.pts) > 500
    m = FE.match_pair(fa, fb)
    assert m is not None and m["inliers"] > 50
    K = _K(f, w, h)
    Ht = K @ R2 @ R1.T @ np.linalg.inv(K)
    q = FE.project(m["H"], m["pa"])
    qt = FE.project(Ht, m["pa"])
    assert np.median(np.hypot(*(q - qt).T)) < 0.5


def test_ransac_rejects_planted_outliers():
    rng = np.random.default_rng(1)
    H = np.array([[1.02, 0.03, 40.0], [-0.02, 0.99, -12.0], [1e-5, -2e-5, 1.0]])
    p1 = rng.uniform(0, 600, (200, 2))
    p2 = FE.project(H, p1) + rng.normal(0, 0.3, (200, 2))
    bad = rng.choice(200, 60, replace=False)
    p2[bad] = rng.uniform(0, 600, (60, 2))
    He, mask = FE.ransac_homography(p1, p2, thr=2.0, rng=rng)
    assert mask[bad].sum() <= 2
    good = np.setdiff1d(np.arange(200), bad)
    assert mask[good].mean() > 0.95
    assert np.abs(FE.project(He, p1[good]) - FE.project(H, p1[good])).max() < 1.0


def test_unrelated_frames_do_not_pair():
    tex = world(seed=0)
    tex2 = world(seed=7)
    f, w, h = 600.0, 640, 480
    a = render(tex, rot(0), f, w, h)
    b = render(tex2, rot(150), f, w, h)
    assert FE.match_pair(FE.FrameFeatures(a), FE.FrameFeatures(b)) is None
    assert FE.pair_ok(50, 100) and not FE.pair_ok(20, 100)
