"""The infrared hot spot (BUILD_PLAN_PANO_IR.md 1.5): synthetic bumps with known size."""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from studio.engine.lens import hotspot as HS


def _field(h=400, w=600, a=2 ** 0.4 - 1, s=0.35, texture=0.0, seed=1, falloff=-0.3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - (h / 2 - 0.5), xx - (w / 2 - 0.5)) / (math.hypot(h, w) / 2)
    base = 0.3 * np.exp2(falloff * r ** 2)
    if texture:
        base = base * np.exp(texture * gaussian_filter(rng.normal(0, 1, (h, w)), 4) * 6)
    return (base * (1 + a * np.exp(-(r / s) ** 2))).astype(np.float32)


def test_bump_measured_on_a_flat_field_within_five_percent():
    m = HS.fit_profile(_field())
    assert m is not None
    assert abs(m["stops"] - 0.4) / 0.4 < 0.05, m
    assert abs(m["s"] - 0.35) / 0.35 < 0.10, m


def test_bump_measured_on_a_textured_scene_within_fifteen_percent():
    m = HS.fit_profile(_field(texture=0.25), textured=True)
    assert m is not None
    assert abs(m["stops"] - 0.4) / 0.4 < 0.15, m


def test_nothing_found_without_a_bump():
    assert HS.fit_profile(_field(a=0.0)) is None
    assert HS.fit_profile(_field(a=0.0, texture=0.25), textured=True) is None


def test_gain_map_undoes_the_bump_and_regions_match():
    Y = _field(falloff=0.0)
    m = HS.fit_profile(Y)
    g = HS.gain_map(m, 1.0, Y.shape[0], Y.shape[1], (0, 0), (Y.shape[1], Y.shape[0]))
    flat = Y * g
    assert np.ptp(flat) / flat.mean() < 0.03
    part = HS.gain_map(m, 1.0, 50, 70, (100, 60), (Y.shape[1], Y.shape[0]))
    assert np.allclose(part, g[60:110, 100:170], atol=1e-6)
    assert np.all(HS.gain_map(m, 0.0, 10, 10, (0, 0), (10, 10)) == 1.0)


def test_learnt_from_several_photos_is_cyan_quality():
    frames = [_field(texture=0.25, seed=s) for s in range(6)]
    m = HS.learn(frames)
    assert m is not None and m["frames"] == 6 and "learnt" in m["source"]
    assert abs(m["stops"] - 0.4) / 0.4 < 0.10


def test_profiles_saved_on_this_pc(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_CALIB_DIR", str(tmp_path / "calib"))
    p = HS.save_profile("Samyang 20mm + 720", {"a": 0.3, "s": 0.4, "stops": 0.38}, lens="Samyang 20", filter_key="720")
    assert str(p).startswith(str(tmp_path))
    assert HS.load_profiles()[0]["name"] == "Samyang 20mm + 720"
