"""Lens-flare ghost removal: synthetic ghosts with known added light on a textured scene, and Bill's IR frame."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from conftest import ROOT
from studio.engine.local import flare as FL


def _scene(h=600, w=900, seed=1):
    rng = np.random.default_rng(seed)
    base = 0.18 * np.exp(0.5 * gaussian_filter(rng.normal(0, 1, (h, w)), 6) * 6)
    tex = 1 + 0.15 * gaussian_filter(rng.normal(0, 1, (h, w)), 1.0) * 3
    s = np.clip(base * tex, 0.01, 0.8).astype(np.float32)
    return np.stack([s, s * 0.95, s * 0.85], -1)


def _ghosts(shape, centre, pts, radii, amp):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    F = np.zeros((h, w, 3), np.float32)
    for (x, y), r, a in zip(pts, radii, amp):
        d = np.hypot(xx - x, yy - y)
        disc = np.clip((r - d) / (0.15 * r), 0, 1)
        F += disc[..., None] * np.array([a, a * 0.8, a * 1.1], np.float32)
    return F


def test_ghosts_on_one_line_are_found_and_removed_keeping_detail():
    S = _scene()
    h, w = S.shape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    ang = math.radians(35)
    pts = [(cx + t * math.cos(ang), cy + t * math.sin(ang)) for t in (-260, -120, 90, 230)]
    F = _ghosts(S.shape, (cx, cy), pts, [40, 22, 30, 55], [0.10, 0.18, 0.12, 0.08])
    I = S + F
    found = FL.find_ghosts(I)
    assert len(found["ghosts"]) >= 3, found
    assert abs(((found["axis"] - 35) + 90) % 180 - 90) < 6
    field, k, clipped = FL.flare_field(I, found["ghosts"])
    out = FL.apply(I, field, k, 1.0, full_shape=(h, w))
    before = np.abs(I - S).mean()
    after = np.abs(out - S).mean()
    assert after < 0.35 * before, (before, after)
    # detail is kept: the fine texture inside a ghost still matches the scene's
    x, y = int(pts[1][0]), int(pts[1][1])
    win = (slice(y - 12, y + 12), slice(x - 12, x + 12))
    hp = lambda a: a[..., 1] - gaussian_filter(a[..., 1], 2)
    c = np.corrcoef(hp(out)[win].ravel(), hp(S)[win].ravel())[0, 1]
    assert c > 0.95


def test_nothing_found_on_a_clean_scene():
    S = _scene(seed=4)
    assert FL.find_ghosts(S)["ghosts"] == [] or len(FL.find_ghosts(S)["ghosts"]) <= 1


def test_region_apply_matches_whole_frame():
    S = _scene()
    h, w = S.shape[:2]
    F = _ghosts(S.shape, None, [(300, 200)], [50], [0.15])
    I = S + F
    field, k, _ = FL.flare_field(I, [{"x": 300 / w, "y": 200 / h, "r": 50 / max(h, w), "on": True}])
    whole = FL.apply(I, field, k, 1.0, full_shape=(h, w))
    part = FL.apply(I[150:260, 240:380], field, k, 1.0, origin=(240, 150), full_shape=(h, w))
    assert np.allclose(part, whole[150:260, 240:380], atol=1e-6)


REAL = ROOT / "test_raws" / "panoramic raw ir" / "IMG_2944.CR2"


@pytest.mark.skipif(not REAL.is_file(), reason="Bill's IR flare frame is not in test_raws")
def test_bills_ir_flare_chain_is_found():
    from studio import engine as E
    from studio.engine.pipeline.pipeline import linear_rgb
    im = E.open(str(REAL))
    rgb = linear_rgb(im, E.Edit(), half_size=True, cache=False)
    found = FL.find_ghosts(rgb)
    assert len(found["ghosts"]) >= 2 and found["source"] is not None
