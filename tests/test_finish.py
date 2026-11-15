"""Phase 6 remainder: highlight rebuild, Dehaze, Texture, Grain, output sharpening, film toe, parametric curve,
Auto colour, B&W filters and camera profiles (.dcp)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import scene, synthetic_raw
from studio import engine as E
from studio.engine.color import dcp, grade as G, light as LT
from studio.engine.color.tone import standard_curve
from studio.engine.detail import finish as F
from studio.engine.pipeline.pipeline import Image, linear_rgb

LUM = np.array([0.288, 0.712, 0.0], np.float32)


def _hue(rgb):
    lab = G.to_oklab(rgb)
    return np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360


# ------------------------------------------------------------------------------------------ highlight rebuild
def _blob(h=160, w=240, peak=(3.0, 0.95, 0.5)):
    yy, xx = np.mgrid[0:h, 0:w]
    b = np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 30.0 ** 2)).astype(np.float32)
    sc = np.full((h, w, 3), 0.05, np.float32) + b[..., None] * np.asarray(peak, np.float32)
    return sc, b


def test_rebuild_gives_clipped_highlights_their_colour_back():
    sc, b = _blob()
    truth = linear_rgb(Image(synthetic_raw(sc)), E.Edit(highlight_mode="none"), cache=False)   # never clipped
    img = Image(synthetic_raw(sc, clip_white=True))
    clip = linear_rgb(img, E.Edit(highlight_mode="clip"), cache=False)
    reb = linear_rgb(img, E.Edit(highlight_mode="rebuild"), cache=False)
    core = b > 0.85                                                     # red clipped on the sensor here
    ht = _hue(truth[core])
    err_clip = np.abs((_hue(clip[core]) - ht + 180) % 360 - 180).mean()
    err_reb = np.abs((_hue(reb[core]) - ht + 180) % 360 - 180).mean()
    assert err_clip > 15 and err_reb < 0.25 * err_clip
    ring = (b > 0.2) & (b < 0.5)                                         # nothing clipped there: untouched
    assert np.allclose(reb[ring], linear_rgb(img, E.Edit(highlight_mode="none"), cache=False)[ring], atol=1e-5)


def test_rebuild_region_matches_full_frame():
    sc, _ = _blob()
    img = Image(synthetic_raw(sc, clip_white=True))
    e = E.Edit(highlight_mode="rebuild")
    full = linear_rgb(img, e, cache=False)
    part = linear_rgb(img, e, region=(100, 60, 60, 40), cache=False)
    assert np.abs(part - full[60:100, 100:160]).max() < 1e-4


def test_rebuild_with_nothing_clipped_changes_nothing():
    img = Image(synthetic_raw(scene(96, 128)))
    assert np.array_equal(linear_rgb(img, E.Edit(highlight_mode="rebuild"), cache=False),
                          linear_rgb(img, E.Edit(highlight_mode="none"), cache=False))


# ------------------------------------------------------------------------------------------ dehaze
def test_dehaze_restores_contrast_and_adds_haze():
    rng = np.random.default_rng(3)
    J = (0.05 + 0.4 * rng.random((120, 180, 1)) * np.array([1.0, 0.8, 0.6])).astype(np.float32)
    t = np.linspace(0.9, 0.35, 180, dtype=np.float32)[None, :, None]      # haze thicker to the right
    A = np.array([0.7, 0.75, 0.8], np.float32)
    I = J * t + A * (1 - t)
    out = F.dehaze(I, 100, A, 1.0, 600)
    far = slice(120, 180)
    assert out[:, far].std() > 1.8 * I[:, far].std()                       # contrast back in the distance
    assert np.abs(np.median(out[:, far] / np.maximum(J[:, far], 1e-3)) - 1) < 0.35
    hazy = F.dehaze(J, -60, A, 1.0, 600)
    assert hazy.std() < J.std() and F.dehaze(J, 0, A) is J


def test_dehaze_region_matches_full_frame():
    img = Image(synthetic_raw(scene(240, 360)))
    e = E.Edit()
    e.light.dehaze = 60
    full = E.render(img, e)
    part = E.render(img, e, region=(150, 90, 120, 80))
    assert np.abs(part - full[90:170, 150:270]).max() < 2e-3


# ------------------------------------------------------------------------------------------ texture
def test_texture_lifts_fine_detail_and_leaves_edges_alone():
    yy, xx = np.mgrid[0:120, 0:200].astype(np.float32)
    base = np.where(xx < 100, 0.05, 0.4).astype(np.float32)                # a big edge
    fine = 1 + 0.06 * np.sin(xx / 1.3) * np.cos(yy / 1.7)
    rgb = (base * fine)[..., None] * np.ones(3, np.float32)
    up, down = F.texture(rgb, 80), F.texture(rgb, -80)
    L = np.log2(rgb[..., 1])
    Lu, Ld = np.log2(up[..., 1]), np.log2(down[..., 1])
    area = (slice(10, 110), slice(10, 85))
    assert Lu[area].std() > 1.3 * L[area].std() and Ld[area].std() < 0.6 * L[area].std()
    ed = slice(95, 105)
    assert abs(np.median(Lu[:, ed][:, :4]) - np.median(L[:, ed][:, :4])) < 0.05      # no rim at the edge
    assert F.texture(rgb, 0) is rgb


def test_texture_region_matches_full_frame():
    img = Image(synthetic_raw(scene(200, 300)))
    e = E.Edit()
    e.finish.texture = 70
    full = E.render(img, e)
    part = E.render(img, e, region=(100, 60, 80, 60))
    assert np.abs(part - full[60:120, 100:180]).max() < 2e-3


# ------------------------------------------------------------------------------------------ grain
def test_grain_is_the_same_in_a_region_and_strongest_in_midtones():
    img = Image(synthetic_raw(scene(200, 300)))
    e = E.Edit()
    e.finish.grain, e.finish.grain_size, e.finish.grain_roughness = 60, 1.2, 40
    full = E.render(img, e)
    part = E.render(img, e, region=(120, 50, 90, 70))
    assert np.abs(part - full[50:120, 120:210]).max() < 1e-5
    assert np.array_equal(full, E.render(img, e))                          # deterministic
    flat = np.full((64, 64, 3), 0.18, np.float32)
    g = F.grain(flat, 60, 1.0, 50, 0, 0, 1, 1)
    black = F.grain(np.zeros_like(flat), 60, 1.0, 50, 0, 0, 1, 1)
    assert np.log2(g[..., 1] / 0.18).std() > 0.05 and np.abs(black).max() == 0
    assert F.grain(flat, 0, 1, 50, 0, 0, 1, 1) is flat


def test_grain_field_statistics_full_and_preview():
    f = F.grain_field(256, 256, 0, 0, 1, 1, 1.5, 0.0, 1)
    assert abs(f.std() - 1) < 0.08 and abs(f.mean()) < 0.05
    p = F.grain_field(128, 128, 0, 0, 0.25, 0.25, 1.5, 0.0, 1)             # a 1/4 preview averages grain away
    expect = 1.5 / math.hypot(1.5, 0.5 / 0.25)
    assert abs(p.std() - expect) < 0.12


# ------------------------------------------------------------------------------------------ output sharpening
def test_output_sharpening_by_medium():
    from studio.engine.io.export import ExportSpec, output_sharpen
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    img = (0.1 + 0.05 * np.sin(xx / 1.5))[..., None] * np.ones(3, np.float32)
    assert output_sharpen(img, ExportSpec()) is img
    soft = output_sharpen(img, ExportSpec(sharpen_for="screen", sharpen_amount="low"))
    hard = output_sharpen(img, ExportSpec(sharpen_for="matte", sharpen_amount="high"))
    assert img.std() < soft.std() < hard.std()
    assert np.log2(hard[..., 1] / img[..., 1]).max() <= 0.5 * 1.35 + 1e-6    # clamped: no strong rims
    with pytest.raises(ValueError):
        ExportSpec(sharpen_for="canvas").validate()


# ------------------------------------------------------------------------------------------ toe, parametric curve
def test_film_toe_bends_only_the_deep_shadows():
    x = 0.11 * np.exp2(np.linspace(-12, 3, 3000)).astype(np.float32)
    base = standard_curve(x)
    for toe in (-100, 100):
        y = standard_curve(x, toe=toe)
        assert np.all(np.diff(y) >= 0)
        mid = np.abs(np.log2(x / 0.11)) < 1.5
        assert np.allclose(y[mid], base[mid], rtol=1e-5)
    deep = np.log2(x / 0.11) < -5
    assert np.all(standard_curve(x, toe=80)[deep] > base[deep]) and np.all(standard_curve(x, toe=-80)[deep] < base[deep])
    assert np.array_equal(standard_curve(x, toe=0), base)


def test_parametric_curve_regions_and_monotone():
    y = np.linspace(0, 1, 2001)
    for p in ({"shadows": 100, "darks": -100, "lights": 100, "highlights": -100},
              {k: -100 for k in LT.PARAM_REGIONS}, {k: 100 for k in LT.PARAM_REGIONS}):
        f = LT.param_fn(p)
        assert np.all(np.diff(f(y)) > 0) and abs(f(np.array([0.0]))[0]) < 1e-9 and abs(f(np.array([1.0]))[0] - 1) < 1e-9
    f = LT.param_fn({"lights": 60})
    assert LT._enc(f(LT._dec(np.array([0.625]))))[0] > 0.625 + 0.03
    assert abs(LT._enc(f(LT._dec(np.array([0.1]))))[0] - 0.1) < 1e-6        # shadows untouched
    assert LT.param_fn({"lights": 0}) is None and not LT.is_identity(E.Light(param={"darks": 10}))


# ------------------------------------------------------------------------------------------ auto colour, B&W filters
def test_auto_colour_lifts_only_dull_photos():
    grey = np.full((32, 32, 3), 120, np.uint8)
    assert G.colourfulness(grey) == 0
    rng = np.random.default_rng(0)
    vivid = rng.integers(0, 256, (64, 64, 3)).astype(np.uint8)
    assert G.colourfulness(vivid) > 60 and G.auto_vibrance(G.colourfulness(vivid)) == 0
    assert 0 < G.auto_vibrance(30.0) <= 35 and G.auto_vibrance(5.0) == 35
    from studio.engine.pipeline.pipeline import auto_colour
    img = Image(synthetic_raw(scene(120, 180) * 0.0 + 0.2 + 0.02 * scene(120, 180)))        # nearly grey
    r = auto_colour(img, E.Edit())
    assert r["vibrance"] > 0 and r["colourfulness"] < 45


def test_bw_filter_presets_use_known_bands():
    for name, mix in G.BW_FILTERS.items():
        assert set(mix) == set(G.BANDS), name
    red = _px([0.8, 0.2, 0.15])
    c_red = E.Colour(bw=True, bw_mix=G.BW_FILTERS["Red filter"])
    c_blue = E.Colour(bw=True, bw_mix=G.BW_FILTERS["Blue filter"])
    assert G.to_oklab(G.apply(red, c_red))[0, 0, 0] > G.to_oklab(G.apply(red, c_blue))[0, 0, 0] + 0.03


def _px(srgb):
    from studio.engine.color.spaces import PROPHOTO, SRGB, apply_matrix, conversion_matrix
    return apply_matrix(np.asarray(srgb, np.float32).reshape(1, 1, 3), conversion_matrix(SRGB, PROPHOTO))


# ------------------------------------------------------------------------------------------ camera profiles
def _profile_like_the_camera(img, tmp_path, **kw):
    from studio.engine.color import camera as cam
    from studio.engine.color.spaces import PROPHOTO
    M = cam.camera_to_working(img.raw.cam_xyz)
    fm = PROPHOTO.to_xyz_d50 @ M                                             # WB'd camera -> XYZ D50
    pr = dcp.Profile(name="Test profile", camera="Test Camera", illuminants=[21],
                     color_matrices=[np.asarray(img.raw.cam_xyz, float)[:3, :3]], forward_matrices=[fm], **kw)
    return dcp.write(pr, str(tmp_path / "test.dcp"))


def test_dcp_round_trip_and_forward_matrix_reproduces_the_default(tmp_path):
    img = Image(synthetic_raw(scene(96, 128)))
    p = _profile_like_the_camera(img, tmp_path)
    pr = dcp.read(p)
    assert pr.name == "Test profile" and pr.camera == "Test Camera" and pr.illuminants == [21]
    base = linear_rgb(img, E.Edit(), cache=False)
    withp = linear_rgb(img, E.Edit(profile=p), cache=False)
    assert np.abs(withp - base).max() < 2e-4


def test_dcp_colour_matrix_path_and_two_illuminants(tmp_path):
    cm_a = np.array([[0.8, -0.1, -0.1], [-0.4, 1.3, 0.1], [-0.1, 0.2, 0.7]])
    cm_d = np.array([[0.7, -0.05, -0.1], [-0.45, 1.25, 0.2], [-0.1, 0.25, 0.65]])
    pr = dcp.Profile(name="two", camera="X", illuminants=[17, 21], color_matrices=[cm_a, cm_d])
    pr2 = dcp.read(dcp.write(pr, str(tmp_path / "two.dcp")))
    assert np.allclose(pr2.color_matrices[1], cm_d, atol=1e-6)
    # a white balance for D65 light weights illuminant 2 fully; for tungsten, illuminant 1
    from studio.engine.color.spaces import D65_XY, xy_to_XYZ
    n65 = cm_d @ xy_to_XYZ(D65_XY)
    _, g = dcp.white_and_weight(pr2, n65 / n65.max())
    assert g < 0.05
    M, _ = dcp.camera_to_working(pr2, 1.0 / (n65 / n65.max()))
    assert np.allclose(M @ np.ones(3), np.ones(3), atol=2e-3)                 # neutral stays neutral


def test_dcp_hue_table_turns_hues(tmp_path):
    img = Image(synthetic_raw(scene(96, 128)))
    nh, ns = 12, 4
    tab = np.zeros((1, nh, ns, 3), np.float32)
    tab[..., 0], tab[..., 1], tab[..., 2] = 30.0, 1.0, 1.0                  # +30 degrees everywhere
    p = _profile_like_the_camera(img, tmp_path, hsm_dims=(nh, ns, 1), hsm=[tab])
    base = linear_rgb(img, E.Edit(), cache=False)
    turned = linear_rgb(img, E.Edit(profile=p), cache=False)
    hb, ht = dcp._rgb_to_hsv(np.maximum(base, 0))[0], dcp._rgb_to_hsv(np.maximum(turned, 0))[0]
    sat = dcp._rgb_to_hsv(np.maximum(base, 0))[1] > 0.1
    d = ((ht - hb) * 60 + 180) % 360 - 180
    assert abs(np.median(d[sat]) - 30) < 1.0


def test_missing_profile_falls_back_to_the_camera_matrix():
    img = Image(synthetic_raw(scene(64, 96)))
    assert np.array_equal(linear_rgb(img, E.Edit(profile="nowhere.dcp"), cache=False),
                          linear_rgb(img, E.Edit(), cache=False))


def test_new_settings_default_to_no_change():
    img = Image(synthetic_raw(scene(96, 128)))
    d = E.Edit().to_dict()
    for k in ("finish", "profile"):
        d.pop(k)
    d["tone"].pop("toe")
    d["light"].pop("param")
    d["light"].pop("dehaze")
    assert np.array_equal(E.render(img, E.Edit()), E.render(img, E.Edit.from_dict(d)))
