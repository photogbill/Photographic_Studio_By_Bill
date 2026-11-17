"""Panorama layout (BUILD_PLAN_PANO_IR.md 2.4-2.5): rotations and focal length from frames rendered by a camera
turning on the synthetic sphere, with and without a focal length from the file; projections round-trip."""
from __future__ import annotations

import numpy as np
import pytest

from pano_synth import render, rot, world
from studio.engine.stack import bundle as BU
from studio.engine.stack import pano as PA
from studio.engine.stack import project as PJ

F, W, H = 600.0, 640, 480


def _frames(n=6, step=25.0, seed=2):
    rng = np.random.default_rng(seed)
    tex = world()
    Rs, ims = [], []
    for i in range(n):
        R = rot(i * step, rng.uniform(-2, 2), rng.uniform(-2, 2))
        Rs.append(R)
        ims.append(render(tex, R, F, W, H))
    return Rs, ims


def _check(sol, Rs):
    errs = []
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            if sol["R"][i] is None or sol["R"][j] is None:
                continue
            est = sol["R"][j] @ sol["R"][i].T
            tru = Rs[j] @ Rs[i].T
            errs.append(BU.angle_deg(est, tru))
    return max(errs)


@pytest.fixture(scope="module")
def frames():
    return _frames()


def test_rotations_and_focal_length_without_the_file(frames):
    Rs, ims = frames
    sol = PA.solve_layout(ims, 0.0, do_level=False)
    assert all(r is not None for r in sol["R"])
    assert sol["focal_source"] == "estimated"
    assert abs(sol["f"] - F) / F < 0.005, sol["f"]
    assert _check(sol, Rs) < 0.05
    assert sol["rms"] < 0.5


def test_rotations_with_the_focal_length_from_the_file(frames):
    Rs, ims = frames
    sol = PA.solve_layout(ims, F, do_level=False)
    assert sol["f"] == F and _check(sol, Rs) < 0.05


def test_level_makes_the_horizon_level(frames):
    Rs, ims = frames
    sol = PA.solve_layout(ims, F, do_level=True)
    # every camera's x axis lies in the horizontal plane after levelling (the frames were shot with small tilts)
    tilt = [abs(float(r[0][1])) for r in sol["R"]]
    assert max(tilt) < np.sin(np.deg2rad(2.5))


@pytest.mark.parametrize("proj", PJ.PROJECTIONS)
def test_projections_round_trip(proj):
    rng = np.random.default_rng(0)
    u = rng.uniform(-500, 500, 200)
    v = rng.uniform(-300, 300, 200)
    d = PJ.from_canvas(u, v, proj, 600.0)
    u2, v2, ok = PJ.to_canvas(d, proj, 600.0)
    assert ok.all()
    assert np.abs(u2 - u).max() < 0.01 and np.abs(v2 - v).max() < 0.01
    x, y, ok = PJ.to_camera(d, np.eye(3), 600.0, 320, 240)
    d2 = PJ.from_camera(x[ok], y[ok], np.eye(3), 600.0, 320, 240)
    assert np.allclose(d2, d[ok], atol=1e-9)
