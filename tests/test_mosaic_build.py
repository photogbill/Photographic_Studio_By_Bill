"""Flat Mosaic, the picture and the height (BUILD_PLAN_MOSAIC.md M1-M3; plan section 6 tests 3, 5, 6 and the nudge):
synthetic hand-held sets (tests/mosaic_synth.py). Nothing is written to disk."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import mosaic_synth as S  # noqa: E402

from studio.engine.stack import mosaic as MO  # noqa: E402
from studio.engine.stack import mosaicbuild as MB  # noqa: E402

BUMPS = [(0.6, 0.2), (1.5, 0.3), (2.2, 0.1)]


@pytest.fixture(scope="module")
def relief_set():
    g = S.Ground(x1=4.5, y1=1.6, seed=1, relief=S.bumps(BUMPS, h=0.04, r=0.06))
    views, truth = S.strip(g, n_per_row=6, rows=2, length=2.6, seed=3, exposure=0.2, vig=0.3)
    sol = MO.solve(views)
    return views, truth, sol


def test_relief_measured_by_parallax_and_orthorectified(relief_set):
    views, truth, sol = relief_set
    sp = MB.plan(views, sol, MO.MosaicParams(resolution="full"))
    hm = sp["hm"]
    assert hm is not None and hm.alpha, hm.how if hm else "no heights"
    assert sp["ortho"]
    assert not hm.flipped                              # the synthetic depth is near = high
    assert hm.fit > 0.5
    # the relief's true size: the bumps stand 40 mm; the measured peak in mm within 25 %
    peak = float(np.max(sp["relief"])) * hm.alpha * sol.m_per_unit * 1000.0
    assert 30.0 < peak < 50.0, peak
    # orthorectified: the views no longer disagree in position at the bumps
    p99 = float(np.percentile(sp["misfit"][sp["valid"]], 99))
    assert p99 < 1.6, p99
    # vignetting measured from the overlaps (the set has 0.3 in log brightness at the corner)
    assert sp["vig"] is not None and -0.45 < sp["vig"][0] < -0.15


def test_flipped_depth_is_noticed(relief_set):
    """A camera whose Dual Pixel reading is the other way round: the parallax says so and the relief is turned."""
    views, truth, sol = relief_set
    flipped = []
    for v in views:
        flipped.append(type(v)(name=v.name, image=v.image, t=-np.asarray(v.t), conf=v.conf, mask=None, f=v.f,
                               colour=v.colour, path=v.path, meta=dict(v.meta), f_from=v.f_from))
    sp = MB.plan(flipped, sol, MO.MosaicParams(resolution="full"))
    hm = sp["hm"]
    assert hm.alpha and hm.flipped
    assert float(np.max(sp["relief"])) > -float(np.min(sp["relief"]))   # bumps stand up, not down


def test_nudge_without_height(relief_set):
    """With no height, a local nudge hides the leftover parallax at the bumps (and says so)."""
    views, truth, sol = relief_set
    off = MB.plan(views, sol, MO.MosaicParams(resolution="full", height=False, nudge=False))
    on = MB.plan(views, sol, MO.MosaicParams(resolution="full", height=False, nudge=True))
    assert off["hm"] is None and not off["ortho"]
    m_off = float(np.percentile(off["misfit"][off["valid"]], 99))
    m_on = float(np.percentile(on["misfit"][on["valid"]], 99))
    assert on["nudges"]
    assert m_on < m_off
    assert any("nudge" in n for n in on["notes"])


def test_moving_thing_and_shadow_left_out():
    g = S.Ground(x1=4.0, y1=1.6, seed=2)
    views, truth = S.strip(g, n_per_row=6, rows=2, length=2.4, seed=11, depth=False,
                           blob=(0.48, 0.02, 0.06), shadow_views=(3,))
    sol = MO.solve(views)
    pv = MB.preview(views, sol, MO.MosaicParams(resolution="full", height=False))
    rgb = pv["rgb"]
    red = (rgb[..., 0] > 0.45) & (rgb[..., 1] < 0.3)
    assert red.mean() < 2e-4, red.mean()
    assert float(pv["plan"]["ghosts"].max()) > 0.5
    assert any("left out by majority" in n for n in pv["plan"]["notes"])


def test_nearer_view_wins():
    """Where a high (coarser) view overlaps nearer ones, the nearer views are used."""
    g = S.Ground(x1=4.0, y1=1.6, seed=4)
    hs = [0.9, 0.95, 1.9, 1.0, 0.92, 1.05]
    views, truth = S.strip(g, n_per_row=6, rows=1, length=2.4, seed=7, heights=hs, tilt=5.0, depth=False)
    sol = MO.solve(views)
    sp = MB.plan(views, sol, MO.MosaicParams(resolution="full", height=False, nudge=False))
    lab = sp["labels"]
    L2 = [L for L in sp["layers"] if L.i == 2][0]
    y0, y1, x0, x1 = L2.box
    others = np.zeros_like(L2.valid)
    for L in sp["layers"]:
        if L.i == 2:
            continue
        a0, a1, b0, b1 = L.box
        yy0, yy1, xx0, xx1 = max(y0, a0), min(y1, a1), max(x0, b0), min(x1, b1)
        if yy1 > yy0 and xx1 > xx0:
            others[yy0 - y0:yy1 - y0, xx0 - x0:xx1 - x0] |= L.valid[yy0 - a0:yy1 - a0, xx0 - b0:xx1 - b0]
    shared = L2.valid & others
    share = float(np.mean(lab[y0:y1, x0:x1][shared] == 2))
    assert share < 0.25, share


def test_resolution_presets(relief_set):
    views, truth, sol = relief_set
    r, mm, how = MB.resolution(sol, views, MO.MosaicParams(resolution="game"))
    assert mm == 1.0 and abs(r * sol.m_per_unit - 0.001) < 1e-9
    r, mm, how = MB.resolution(sol, views, MO.MosaicParams(resolution="custom", mm_per_px=0.25))
    assert mm == 0.25
    keep = sol.m_per_unit
    sol.m_per_unit = None
    try:
        r, mm, how = MB.resolution(sol, views, MO.MosaicParams(resolution="game"))
        assert mm is None and "relative" in how
        with pytest.raises(MO.MosaicProblem):
            MB.resolution(sol, views, MO.MosaicParams(resolution="custom"))
    finally:
        sol.m_per_unit = keep


def test_canvas_round_trip():
    cv = MB.Canvas(0.002, 0.7, -1.0, 2.0, 500, 300)
    u, v = np.array([10.0, 250.5]), np.array([3.0, 120.0])
    x, y = cv.to_ground(u, v)
    u2, v2 = cv.from_ground(x, y)
    assert np.allclose(u, u2) and np.allclose(v, v2)
    du, dv = cv.vec(0.002 * np.cos(0.7), 0.002 * np.sin(0.7))
    assert abs(du - 1.0) < 1e-9 and abs(dv) < 1e-9
