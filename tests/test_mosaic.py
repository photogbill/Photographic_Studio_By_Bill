"""Flat Mosaic, the solve (BUILD_PLAN_MOSAIC.md M0; plan section 6 tests 1, 2, 7, 8): synthetic hand-held sets with
known poses, heights, tilt, distortion and exposure (tests/mosaic_synth.py). Nothing is written to disk."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import mosaic_synth as S  # noqa: E402

from studio.engine.stack import mosaic as MO  # noqa: E402


@pytest.fixture(scope="module")
def ground():
    return S.Ground(x1=4.5, y1=1.6, seed=1)


@pytest.fixture(scope="module")
def two_rows(ground):
    views, truth = S.strip(ground, n_per_row=6, rows=2, length=2.6, seed=3, k1=0.04, h_lo=1.0, h_hi=1.7,
                           tilt=10.0)
    sol = MO.solve(views)
    return views, truth, sol


def test_hand_held_poses_and_scale(two_rows):
    views, truth, sol = two_rows
    assert sol.method == "poses"
    assert all(sol.placed)
    assert sol.rms < 0.6
    err = S.align_to_truth(sol, truth)
    assert err["pos_err_m"] < 0.01                     # every camera within 1 cm
    assert err["ang_err_deg"] < 0.5
    # scale from the focus distances (exact in the synthetic set): within 2 %
    assert sol.scale_colour == "amber"
    assert abs(sol.m_per_unit / err["scale"] - 1) < 0.02
    # the shared distortion came back
    assert abs(sol.k1 - 0.04) < 0.015
    # heights differ between views and are reported
    hs = [sol.cam_height(i) * sol.m_per_unit for i in range(sol.n)]
    th = [c[2] for c in truth["C"]]
    assert max(abs(a - b) for a, b in zip(hs, th)) < 0.02


def test_drift_checked_with_two_rows(two_rows):
    _, _, sol = two_rows
    assert sol.drift["checked"]
    assert sol.drift["closures"] >= 3


def test_single_row_reports_unchecked_drift(ground):
    views, truth = S.strip(ground, n_per_row=6, rows=1, length=3.0, seed=5, tilt=6.0)
    sol = MO.solve(views)
    assert all(sol.placed)
    assert not sol.drift["checked"]
    assert "could not be checked" in sol.drift["text"]
    err = S.align_to_truth(sol, truth)
    assert err["pos_err_m"] < 0.02


def test_mixed_distances(ground):
    """One view at twice the height of its neighbours still joins (features matched at the right scale)."""
    hs = [0.9, 0.95, 1.9, 1.0, 0.92, 1.05]
    views, truth = S.strip(ground, n_per_row=6, rows=1, length=2.4, seed=7, heights=hs, tilt=5.0)
    sol = MO.solve(views)
    assert all(sol.placed)
    err = S.align_to_truth(sol, truth)
    assert err["pos_err_m"] < 0.02
    g = [sol.gsd(i) for i in range(sol.n)]
    assert g[2] > 1.6 * np.median(g)                  # the high view's pixels cover more ground


def test_free_warps_when_no_focal_length(ground):
    views, truth = S.strip(ground, n_per_row=5, rows=1, length=2.0, seed=9, tilt=4.0, focus=False)
    for v in views:
        v.f_from = "guessed (no focal length or pixel size): found by the solver"
    sol = MO.solve(views)
    assert sol.method == "free"
    assert all(sol.placed)
    assert sol.rms < 0.8
    assert sol.cam_height(0) is None
    assert sol.m_per_unit is None and sol.scale_colour == "violet"
    # the ground layout is right up to a similarity: distances between view centres keep their ratios
    c = np.array([sol.to_ground(i, np.array([[views[i].cx, views[i].cy]]))[0] for i in range(sol.n)])
    t = np.array([MO.hit(truth["C"][i], truth["R"][i][2:3, :])[0, :2] for i in range(sol.n)])
    dc = np.linalg.norm(np.diff(c, axis=0), axis=1)
    dt = np.linalg.norm(np.diff(t, axis=0), axis=1)
    r = dc / dt
    assert r.std() / r.mean() < 0.03


def test_ruler_scale(two_rows):
    views, truth, sol = two_rows
    # two ground points 0.5 m apart, seen in view 0: where they fall in its picture
    R, C = truth["R"][0], truth["C"][0]
    h, w = views[0].shape
    L = sol.lenses[0]
    c = MO.hit(C, R[2:3, :])[0]
    P = np.array([[c[0] - 0.2, c[1], 0.0], [c[0] + 0.3, c[1], 0.0]])
    px, _ = MO.project(P, R, C, L, truth["k1"])
    u = px[:, 0] / (w - 1)
    v = px[:, 1] / (h - 1)
    p = MO.MosaicParams(scale="ruler", ruler=f"0:{u[0]:.5f},{v[0]:.5f},{u[1]:.5f},{v[1]:.5f},500")
    MO.set_scale(sol, views, p)
    assert sol.scale_colour == "cyan"
    err = S.align_to_truth(sol, truth)
    assert abs(sol.m_per_unit / err["scale"] - 1) < 0.01


def test_parse_ruler():
    assert MO.parse_ruler("") is None
    v, a, b, mm = MO.parse_ruler("3:0.1,0.2,0.8,0.2,300")
    assert v == 3 and a == (0.1, 0.2) and mm == 300
    with pytest.raises(ValueError):
        MO.parse_ruler("0.1,0.2,0.8,0.2,300")
    with pytest.raises(ValueError):
        MO.parse_ruler("1:0.1,0.2,1.8,0.2,300")


def test_pose_from_homography_round_trip():
    R, C = S.pose(0.3, -0.2, 1.4, 12.0, 0.7, 0.4)
    L = MO.Lens(600.0, 255.5, 191.5, math.hypot(255.5, 191.5))
    H = MO.plane_homography(R, C, L)
    R2, C2 = MO.pose_from_homography(H * 3.7, L)
    assert np.allclose(R, R2, atol=1e-6)
    assert np.allclose(C, C2, atol=1e-6)


def test_too_few_views():
    with pytest.raises(MO.MosaicProblem):
        MO.solve([])


def test_gentle_curve_on_a_crowned_road():
    """A crowned road: the plane leaves misfit, the gentle curve takes it out and finds the bow (plan test 4)."""
    c = 0.06                                          # z = c (y - 0.27)^2: about 4 cm over the road's half width
    g = S.Ground(x1=4.0, y1=1.6, seed=6, crown=(0.0, 0.0, c))
    g.relief = lambda x, y: c * ((y - 0.27) ** 2 - y * y)    # shift the bow's centre to the middle of the rows
    views, truth = S.strip(g, n_per_row=5, rows=2, length=2.0, seed=13, tilt=6.0, depth=False)
    flat = MO.solve(views, MO.MosaicParams(surface="plane"))
    bow = MO.solve(views, MO.MosaicParams(surface="curve"))
    assert bow.surface == "curve" and bow.q is not None
    assert bow.rms < 0.95 * flat.rms, (bow.rms, flat.rms)
    s = S.align_to_truth(bow, truth)["scale"]         # metres per unit
    Q = np.array([[bow.q[0], bow.q[1] / 2], [bow.q[1] / 2, bow.q[2]]])
    ev = np.linalg.eigvalsh(Q)
    big = ev[np.argmax(np.abs(ev))]
    assert abs(big / s / c - 1) < 0.3, (big / s, c)
