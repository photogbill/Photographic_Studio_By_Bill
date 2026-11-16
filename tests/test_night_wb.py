"""Night white balance: SGP-LRD from RL-AWB (color/nightwb.py), with the setting picked by agreement."""
from __future__ import annotations

import numpy as np
import pytest


def _scene(name, light, level, seed=0):
    skd = pytest.importorskip("skimage.data")
    lin = (getattr(skd, name)().astype(float) / 255.0) ** 2.2
    lin = np.repeat(np.repeat(lin, 3, 0), 3, 1)
    rng = np.random.default_rng(seed)
    cam = lin * np.asarray(light) * level
    cam = cam + rng.normal(0, 0.004, cam.shape) * np.sqrt(cam + 1e-3) + rng.normal(0, 0.0015, cam.shape)
    return np.clip(cam, 0, 1)


def test_night_estimate_on_a_scene_with_nothing_grey():
    from studio.engine.color import autowb as aw
    from studio.engine.color import nightwb as nw
    errs, ge_errs = [], []
    for seed, light in enumerate(([1.0, 0.62, 0.30], [0.55, 0.75, 1.0], [1.0, 0.85, 0.45])):
        cam = _scene("coffee", light, 0.15, seed)
        est = nw.estimate(cam)
        assert est is not None and est["runs"] >= 30 and est["n"] in nw.GRID_N and est["p"] in nw.GRID_P
        assert abs(est["light"].sum() - 1.0) < 1e-9 and est["spread_deg"] >= 0
        errs.append(aw.angle_deg(est["light"], light))
        ge = aw.grey_edge(cam)
        ge_errs.append(aw.angle_deg(ge, light) if ge is not None else 90.0)
    assert np.median(errs) < 4.0 and np.median(errs) < np.median(ge_errs)   # coffee: no grey, grey edge is pulled


def test_night_estimate_still_answers_in_a_very_dark_frame():
    from studio.engine.color import nightwb as nw
    cam = _scene("coffee", [1.0, 0.62, 0.30], 0.01, 1)
    est = nw.estimate(cam)
    assert est is not None and np.all(est["light"] > 0)
    assert np.allclose(nw.sgp_lrd(np.full((64, 64, 3), 0.0)), 0.0)          # nothing to see: no answer
