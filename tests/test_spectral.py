"""Camera colour from Solomatov & Akkaynak's data (color/spectral.py): the two-light matrices that are the Studio's
default colour (camera_matrix version 2) and the spectral profiles, checked against known truth."""
from __future__ import annotations

import numpy as np
import pytest

from studio.engine.color import dcp
from studio.engine.color import spectral as sp


def test_the_data_is_there_with_its_attribution():
    import json
    import os
    for f in ("camera_sensitivities.json", "dual_matrices.json"):
        d = json.load(open(os.path.join(sp.data_dir(), f), encoding="utf-8"))
        assert "Solomatov" in d["source"] and "CC BY 4.0" in d["source"] and len(d["cameras"]) > 1000
    assert sp.camera_name("Canon", "Canon EOS R5") == "Canon EOS R5"
    assert sp.camera_name("SONY", "ILCE-7M4") == "Sony ILCE 7M4"
    assert sp.camera_name("Synthetic", "Test Camera") is None


def test_daylight_and_tungsten_light_are_what_the_cie_says():
    d = sp._load()
    for T, xy in ((6504, (0.3127, 0.3290)), (2856, (0.4476, 0.4074))):
        X = d["cmf"].T @ sp.illuminant(T)
        assert X[0] / X.sum() == pytest.approx(xy[0], abs=0.003) and X[1] / X.sum() == pytest.approx(xy[1], abs=0.003)


def test_a_camera_that_sees_like_the_eye_gets_a_perfect_profile():
    d = sp._load()
    cmf = d["cmf"]
    ideal = cmf @ np.array([[3.24, -0.97, 0.06], [-1.54, 1.88, -0.20], [-0.50, 0.04, 1.06]])   # XYZ -> some RGB
    for T in (1850.0, 3200.0, 6500.0):
        F = sp.forward_matrix(ideal, T)
        E = sp.illuminant(T)
        R = d["refl"] * E[None, :]
        C = (R @ ideal) / (ideal.T @ E)[None, :]
        X = R @ cmf
        X = X / (cmf.T @ E)[1]
        xw = cmf.T @ E
        from studio.engine.color.spaces import D50_XY, bradford, xy_to_XYZ
        B = bradford(xy_to_XYZ((xw[0] / xw.sum(), xw[1] / xw.sum())), xy_to_XYZ(D50_XY))
        err = np.abs(C @ F.T - X @ B.T).max()
        assert err < 1e-6                                    # a colorimetric camera is fitted exactly
        assert np.allclose(F @ np.ones(3), xy_to_XYZ(D50_XY) / xy_to_XYZ(D50_XY)[1], atol=1e-9)   # white held


def test_two_light_profile_uses_tungsten_in_warm_light_and_daylight_in_daylight():
    _, A, D = sp._duals()[dcp._norm("Canon EOS R5")]
    pr = sp.dual_profile("Canon", "Canon EOS R5", D)
    assert pr is not None and pr.illuminants == [17, 21]
    assert sp.dual_profile("Canon", "Canon EOS R5", D + 0.05) is None          # disagrees with the raw reader
    def mult_for(cm, T):
        from studio.engine.color.camera import multipliers_for_white, cct_to_xy
        return multipliers_for_white(cm, cct_to_xy(T))
    _, g_day = dcp.camera_to_working(pr, mult_for(D, 6500))
    _, g_tung = dcp.camera_to_working(pr, mult_for(A, 2856))
    assert g_day < 0.05 and g_tung > 0.95


def _raw_image(make, model, cam_xyz, wb):
    from conftest import synthetic_raw
    from studio.engine.pipeline.pipeline import Image
    y, x = np.mgrid[0:48, 0:64] / 64.0
    scene = np.stack([0.15 + 0.5 * x, 0.12 + 0.3 * y, 0.08 + 0.4 * (1 - x)], -1)
    raw = synthetic_raw(np.clip(scene, 0, 1), wb=wb)
    raw.meta.make, raw.meta.model = make, model
    raw.cam_xyz = np.asarray(cam_xyz, float)
    return Image(raw)


def test_new_edits_get_the_two_light_colour_and_old_sidecars_keep_theirs():
    from studio.engine.pipeline import pipeline as pl
    from studio.engine.pipeline.edit import Edit
    from studio.engine.color.camera import multipliers_for_white, cct_to_xy
    _, A, D = sp._duals()[dcp._norm("Canon EOS R5")]
    assert Edit().stage_versions["camera_matrix"] == 2
    old = Edit.from_dict({"stage_versions": {"camera_matrix": 1}})
    assert old.stage_versions["camera_matrix"] == 1
    warm = multipliers_for_white(D, cct_to_xy(2400))
    day = multipliers_for_white(D, cct_to_xy(6500))
    diffs = {}
    for name, m in (("warm", warm), ("day", day)):
        im = _raw_image("Canon", "Canon EOS R5", D, (float(m[0]), 1.0, float(m[2])))
        e2, e1 = Edit(), Edit.from_dict({"stage_versions": {"camera_matrix": 1}})
        e2.tone.curve_colour = e1.tone.curve_colour = "rich"        # the matrix alone is compared
        v2 = pl.render(im, e2, scale=0.5)
        v1 = pl.render(im, e1, scale=0.5)
        diffs[name] = float(np.abs(v2 - v1).mean())
    assert diffs["warm"] > 4 * diffs["day"]               # warm light changes, daylight barely does
    sp_e = Edit()
    sp_e.profile = sp.KEY
    im = _raw_image("Canon", "Canon EOS R5", D, (float(warm[0]), 1.0, float(warm[2])))
    out = pl.render(im, sp_e, scale=0.5)
    assert np.isfinite(out).all()
