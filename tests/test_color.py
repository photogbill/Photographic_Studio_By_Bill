"""Colour maths, ICC profiles, white balance, tone curve and gamut compression."""
from __future__ import annotations

import io

import numpy as np
import pytest

from studio.engine.color import camera as C
from studio.engine.color import gamut, icc
from studio.engine.color import spaces as S
from studio.engine.color import tone as T
from conftest import CAM_XYZ_5D4


def test_srgb_matrix_matches_the_standard():
    M = S.SRGB.to_xyz
    ref = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    assert np.allclose(M, ref, atol=6e-4)


def test_prophoto_matrix_matches_romm():
    ref = np.array([[0.7977, 0.1352, 0.0313], [0.2880, 0.7119, 0.0001], [0.0, 0.0, 0.8249]])
    assert np.allclose(S.PROPHOTO.to_xyz_d50, ref, atol=6e-4)


def test_white_maps_to_white_between_all_spaces():
    for a in S.SPACES.values():
        for b in S.SPACES.values():
            assert np.allclose(S.conversion_matrix(a, b) @ np.ones(3), np.ones(3), atol=1e-9)


def test_transfer_curves_round_trip():
    x = np.linspace(0, 1, 1001)
    for sp in S.SPACES.values():
        assert np.allclose(sp.decode(sp.encode(x)), x, atol=1e-7)


def test_camera_matrix_keeps_neutral_neutral():
    M = C.camera_to_working(CAM_XYZ_5D4)
    assert np.allclose(M @ np.ones(3), np.ones(3), atol=1e-9)


def test_cct_and_tint_round_trip():
    x, y = C.cct_to_xy(6504)
    assert abs(x - 0.3127) < 0.003 and abs(y - 0.3290) < 0.006
    for T_, tint in [(3000, 0), (5500, 10), (7500, -15)]:
        m = C.multipliers_for_white(CAM_XYZ_5D4, C.temp_tint_to_xy(T_, tint))
        T2, t2 = C.temp_tint_from_multipliers(CAM_XYZ_5D4, m)
        assert abs(T2 - T_) / T_ < 0.02 and abs(t2 - tint) < 4


@pytest.mark.parametrize("key", ["adobergb", "srgb", "prophoto", "displayp3"])
def test_icc_profiles_agree_with_our_maths(key):
    from PIL import Image, ImageCms
    prof = icc.profile_bytes(key)
    assert prof[36:40] == b"acsp" and int.from_bytes(prof[:4], "big") == len(prof)
    src = ImageCms.ImageCmsProfile(io.BytesIO(prof))
    dst = ImageCms.ImageCmsProfile(io.BytesIO(icc.profile_bytes("srgb")))
    rng = np.random.default_rng(5)
    lin_srgb = rng.uniform(0.05, 0.95, (32, 32, 3)).astype(np.float32)    # inside sRGB, so inside all spaces
    sp = S.SPACES[key]
    src_lin = S.apply_matrix(lin_srgb, S.conversion_matrix(S.SRGB, sp))
    enc = np.clip(np.round(sp.encode(src_lin) * 255), 0, 255).astype(np.uint8)
    t = ImageCms.buildTransform(src, dst, "RGB", "RGB", renderingIntent=1)
    lcms = np.asarray(ImageCms.applyTransform(Image.fromarray(enc, "RGB"), t), np.float32)
    ours = S.SRGB.encode(S.apply_matrix(sp.decode(enc / 255.0).astype(np.float32), S.conversion_matrix(sp, S.SRGB)))
    assert np.abs(lcms - np.clip(np.round(ours * 255), 0, 255)).max() <= 2


def test_icc_profiles_are_deterministic():
    icc.profile_bytes.cache_clear()
    a = icc.profile_bytes("adobergb")
    icc.profile_bytes.cache_clear()
    assert a == icc.profile_bytes("adobergb")


def test_standard_curve_shape():
    x = np.geomspace(1e-5, 2.0, 5000).astype(np.float32)
    y = T.standard_curve(x)
    assert np.all(np.diff(y) >= -1e-7), "tone curve must be monotone"
    assert T.standard_curve(np.array([0.11]))[0] == pytest.approx(0.18, abs=1e-4)
    assert T.standard_curve(np.array([0.0]))[0] == 0.0
    assert T.standard_curve(np.array([0.11 * 8]))[0] == pytest.approx(1.0, abs=1e-6)   # white point: 3 stops
    # no jump at the shoulder join
    s = 0.11 * 2.0 ** 1.0
    a, b = T.standard_curve(np.array([s * 0.999, s * 1.001], np.float32))
    assert abs(b - a) < 2e-3


def test_gamut_compression_leaves_in_gamut_colours_alone():
    rng = np.random.default_rng(2)
    rgb = rng.uniform(0.2, 0.9, (1000, 3)).astype(np.float32)          # all distances well under thresholds
    assert np.array_equal(gamut.compress(rgb), rgb)


def test_gamut_compression_rescues_negative_channels_smoothly():
    v = np.stack([np.full(200, 0.3), np.linspace(0.2, -0.25, 200), np.full(200, 1.2)], 1).astype(np.float32)
    out = gamut.compress(v)
    assert out[:, 1].min() >= -1e-4                     # inside the limit -> no negative channel left
    assert np.all(np.diff(out[:, 1]) <= 1e-6)           # still monotone (no hue jump)
    assert np.allclose(out[:, 2], 1.2)                  # the largest channel is untouched


# ------------------------------------------------------------------------------------------ auto WB (plan 23.9)
def _lit_scene(light, h=240, w=360, seed=3, greys=True):
    """Camera RGB of coloured patches (and a few grey ones with texture) under a light of colour ``light``."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for j in range(0, h, 40):
        for i in range(0, w, 40):
            col = rng.uniform(0.05, 0.6, 3)
            if greys and (i // 40 + j // 40) % 4 == 0:
                col = np.full(3, rng.uniform(0.1, 0.5))             # a neutral surface
            shade = 1.0 + 0.15 * rng.standard_normal((40, 40, 1))   # shading: the same in every channel
            gloss = 0.08 * np.abs(rng.standard_normal((40, 40, 1)))  # surface reflection: the light's own colour
            img[j:j + 40, i:i + 40] = np.clip(col * shade + gloss, 0.01, 0.9)
    # (on a grey surface body and gloss have the same colour, so its chromaticity stays constant: the cue)
    return (img * np.asarray(light, np.float32)).astype(np.float32)


def test_grayness_index_finds_the_light():
    from studio.engine.color import autowb as AW
    light = np.array([0.9, 0.6, 0.3])                               # a warm light
    e, lights = AW.grayness_index(_lit_scene(light))
    assert e is not None and AW.angle_deg(e, light) < 3.0, f"off by {AW.angle_deg(e, light):.1f} degrees"
    assert len(lights) == 1
    assert AW.grayness_index(np.full((64, 64, 3), 0.99, np.float32))[0] is None, "a blown frame tells nothing"


def test_grayness_index_reports_two_lights():
    from studio.engine.color import autowb as AW
    a, b = np.array([0.9, 0.6, 0.3]), np.array([0.45, 0.6, 0.8])    # candle on the left, window on the right
    left, right = _lit_scene(a, seed=3), _lit_scene(b, seed=4)
    cam = np.concatenate([left, right], axis=1)
    e, lights = AW.grayness_index(cam, top=0.004)
    assert len(lights) == 2, "candle plus window: two light sources"
    got = sorted(lights, key=lambda t: -t[0][0])                    # warmest first
    assert AW.angle_deg(got[0][0], a) < 5.0 and AW.angle_deg(got[1][0], b) < 5.0
    assert abs(got[0][1] + got[1][1] - 1.0) < 1e-6


def test_keep_the_mood_and_defaults():
    from studio.engine.color import autowb as AW
    auto, shot = np.array([1.2, 1.0, 2.6]), np.array([2.0, 1.0, 1.4])
    assert np.allclose(AW.keep_mood(auto, shot, 1.0), auto / auto.min())
    assert np.allclose(AW.keep_mood(auto, shot, 0.0), shot / shot.min())
    half = AW.keep_mood(auto, shot, 0.5)
    half = half / half[1]
    assert np.allclose(half, np.sqrt(auto * shot)), "half-way is half the correction in every channel"
    assert AW.default_mood(2800, 6400) == 0.6, "a warm light in a low-light photo: the candle stays warm"
    assert AW.default_mood(2800, 100) == 0.9 and AW.default_mood(5500, 6400) == 0.9
    assert set(AW.METHODS) == {"grey_edge", "grayness"}


def test_auto_white_balance_in_the_pipeline():
    from conftest import scene, synthetic_raw
    from studio.engine.pipeline import pipeline as PL
    img = PL.Image(synthetic_raw(scene(240, 360), read_noise=2.0))
    for method in ("grey_edge", "grayness"):
        m, how, info = PL.auto_white_balance(img, method)
        assert m is not None and m.min() == pytest.approx(1.0) and how
        assert 0.0 < info["mood"] <= 1.0 and info["confidence"] in ("high", "medium", "low")
    full, _, _ = PL.auto_white_balance(img, "grey_edge", 1.0)
    none, _, _ = PL.auto_white_balance(img, "grey_edge", 0.0)
    shot = np.array([2.0, 1.0, 1.5])
    assert np.allclose(none / none[1], shot), "0% = as shot"
    half, _, _ = PL.auto_white_balance(img, "grey_edge", 0.5)
    assert np.all((half / half[1] - shot) * (full / full[1] - shot) >= -1e-9), "part-way lies between the two"
