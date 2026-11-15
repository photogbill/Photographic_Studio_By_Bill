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
