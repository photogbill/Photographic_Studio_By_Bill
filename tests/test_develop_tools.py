"""Phase 6: the Light panel (stage 15), the Colour stage (17), looks, copy / paste, virtual copies, snapshots."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from conftest import scene, synthetic_raw
from studio import engine as E
from studio.engine.color import grade as G
from studio.engine.color import light as LT
from studio.engine.color.spaces import PROPHOTO, SRGB, apply_matrix, conversion_matrix
from studio.engine.pipeline.pipeline import Image, auto_light


def _img(h=200, w=300, sc=None):
    return Image(synthetic_raw(scene(h, w) if sc is None else sc))


def _srgb_to_pp(rgb):
    return apply_matrix(np.asarray(rgb, np.float32).reshape(1, -1, 3), conversion_matrix(SRGB, PROPHOTO))


# ------------------------------------------------------------------------------------------ OKLab
def test_oklab_matches_the_published_values():
    w = G.to_oklab(_srgb_to_pp([[1, 1, 1]]))[0, 0]
    assert abs(w[0] - 1) < 1e-4 and abs(w[1]) < 1e-4 and abs(w[2]) < 1e-4
    red = G.to_oklab(_srgb_to_pp([[1, 0, 0]]))[0, 0]                  # Ottosson: L 0.628, a 0.225, b 0.126
    assert abs(red[0] - 0.628) < 2e-3 and abs(red[1] - 0.2249) < 2e-3 and abs(red[2] - 0.1258) < 2e-3
    x = np.random.default_rng(0).random((20, 20, 3)).astype(np.float32)
    assert np.abs(G.from_oklab(G.to_oklab(x)) - x).max() < 1e-5


def test_hue_bands_are_a_partition_of_unity():
    h = np.linspace(0, 359.9, 3600)
    w = G.band_weights(h)
    assert np.allclose(w.sum(0), 1.0, atol=1e-5)
    for i, c in enumerate(G.BAND_HUES):
        assert abs(G.band_weights(np.array([c]))[i, 0] - 1.0) < 1e-6


# ------------------------------------------------------------------------------------------ colour stage
def _colour(**kw):
    c = E.Colour()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_neutrals_are_never_tinted_by_hue_controls():
    grey = np.full((4, 4, 3), 0.2, np.float32)
    c = _colour(hsl={n: [80, 80, 80] for n in G.BANDS}, ranges=[{"hue": 120, "width": 60, "falloff": 30,
                                                                  "shift": 40, "sat": 80, "lum": 80}],
                saturation=60, vibrance=60)
    out = G.apply(grey, c)
    assert np.abs(out - grey).max() < 2e-3


def test_saturation_and_black_and_white():
    px = _srgb_to_pp([[0.8, 0.3, 0.2]])
    c0 = np.hypot(*G.to_oklab(px)[0, 0, 1:])
    more = np.hypot(*G.to_oklab(G.apply(px, _colour(saturation=50)))[0, 0, 1:])
    assert abs(more / c0 - 1.5) < 0.02
    bw = G.apply(px, _colour(bw=True))
    assert np.allclose(bw[0, 0], bw[0, 0, 1], atol=1e-3)               # neutral
    assert abs(G.to_oklab(bw)[0, 0, 0] - G.to_oklab(px)[0, 0, 0]) < 1e-3   # same lightness


def test_hsl_band_only_touches_its_colour():
    green, blue = _srgb_to_pp([[0.2, 0.7, 0.2]]), _srgb_to_pp([[0.2, 0.3, 0.8]])
    c = _colour(hsl={"green": [0, -100, 0]})
    assert np.hypot(*G.to_oklab(G.apply(green, c))[0, 0, 1:]) < 0.02    # green desaturated
    assert np.abs(G.apply(blue, c) - blue).max() < 1e-3                # blue untouched


def test_colour_wheel_range_shifts_hue_and_keeps_lightness():
    px = _srgb_to_pp([[0.8, 0.25, 0.2]])                               # red-orange
    lab = G.to_oklab(px)[0, 0]
    h0 = math.degrees(math.atan2(lab[2], lab[1])) % 360
    c = _colour(ranges=[{"hue": h0, "width": 30, "falloff": 20, "shift": 30, "sat": 0, "lum": 0}])
    lab2 = G.to_oklab(G.apply(px, c))[0, 0]
    h2 = math.degrees(math.atan2(lab2[2], lab2[1])) % 360
    gate = float(G.chroma_gate(np.array([np.hypot(lab[1], lab[2])]))[0])
    assert abs(G.hue_dist(h2, h0) - 30 * gate) < 0.5
    assert abs(lab2[0] - lab[0]) < 1e-3


def test_grading_tints_shadows_not_highlights():
    dark, bright = np.full((1, 1, 3), 0.01, np.float32), np.full((1, 1, 3), 0.8, np.float32)
    c = _colour(grading={"shadows": [220, 100, 0], "balance": 0, "blending": 50})
    d = G.to_oklab(G.apply(dark, c))[0, 0]
    b = G.to_oklab(G.apply(bright, c))[0, 0]
    assert np.hypot(d[1], d[2]) > 0.03 and np.hypot(b[1], b[2]) < 0.01


def test_cube_lut_identity_and_parse(tmp_path):
    n = 9
    rows = [f"{r / (n - 1):.6f} {g / (n - 1):.6f} {b / (n - 1):.6f}" for b in range(n) for g in range(n) for r in range(n)]
    lut = G.parse_cube("TITLE \"id\"\nLUT_3D_SIZE 9\n" + "\n".join(rows))
    x = np.random.default_rng(1).random((10, 10, 3)).astype(np.float32) * 0.5
    assert np.abs(G.apply_lut(x, lut) - x).max() < 2e-3
    swap = G.parse_cube("LUT_3D_SIZE 2\n" + "\n".join(f"{b} {g} {r}" for b in (0, 1) for g in (0, 1) for r in (0, 1)))
    out = G.apply_lut(_srgb_to_pp([[1, 0, 0]]), swap)                  # red -> blue
    s = apply_matrix(out, conversion_matrix(PROPHOTO, SRGB))[0, 0]
    assert s[2] > 0.9 and s[0] < 0.1


# ------------------------------------------------------------------------------------------ light
def test_light_curve_stays_monotone_at_the_extremes():
    s = np.linspace(-16, 6, 20001)
    for sh in (-100, 100):
        for hi in (-100, 100):
            assert np.all(np.diff(LT.stop_shift(s, sh, hi, 3.0)) > 0)
    f = LT.curve_fn([[0.2, 0.05], [0.5, 0.6], [0.8, 0.85]])
    y = np.linspace(0, 1, 1000)
    assert np.all(np.diff(f(y)) >= -1e-9)


def _ramp(h=120, w=400, lo=-9.0, hi=2.5):
    """Grey ramp from lo to hi stops around mid grey, left to right: every tone, so each control has work."""
    s = np.linspace(lo, hi, w, dtype=np.float32)
    v = 0.18 * np.exp2(s)
    return np.repeat(np.repeat(v[None, :, None], h, 0), 3, 2), s


def test_light_controls_move_the_right_tones():
    sc, s = _ramp()
    img = _img(sc=sc)
    lum = np.array([0.288, 0.712, 0.0], np.float32)
    Y = (E.render(img, E.Edit()) @ lum).mean(0)
    dark, mid, bright = (s > -6) & (s < -4), (s > -0.5) & (s < 0.5), (s > 1.0) & (s < 2.0)

    def edit(**kw):
        e = E.Edit()
        for k, v in kw.items():
            setattr(e.light, k, v)
        return (E.render(img, e) @ lum).mean(0)
    up = edit(shadows=80)
    assert up[dark].mean() > Y[dark].mean() * 1.3 and abs(up[bright].mean() / Y[bright].mean() - 1) < 0.02
    assert abs(up[mid].mean() / Y[mid].mean() - 1) < 0.05            # mid grey barely moves
    down = edit(highlights=-80)
    assert down[bright].mean() < Y[bright].mean() * 0.95 and abs(down[dark].mean() / Y[dark].mean() - 1) < 0.02
    deeper = edit(blacks=-60)
    assert deeper[dark].mean() < Y[dark].mean() * 0.9
    brighter = edit(whites=60)
    assert brighter[bright].mean() > Y[bright].mean()
    for out in (up, down, deeper, brighter):                           # every control keeps tones in order
        assert np.all(np.diff(out) >= -1e-4)


def test_smart_lighting_lifts_shadows_and_keeps_detail_and_edges():
    rng = np.random.default_rng(2)
    sc = np.full((240, 360, 3), 0.004, np.float32)
    sc[:, 180:] = 0.5                                                   # dark room | lit area
    sc[:, :180] *= (1 + 0.3 * rng.standard_normal((240, 180, 1))).astype(np.float32).clip(0.3, 2)
    rgb = sc.copy()
    out = LT.smart_lighting(rgb, 100.0, 0.11, 1.0, 360)
    Lb, Lo = np.log2(rgb[..., 1]), np.log2(out[..., 1])
    assert (Lo - Lb)[:, :150].mean() > 0.8                              # shadows lifted ~1+ stop
    assert abs((Lo - Lb)[:, 200:].mean()) < 0.2                         # lit side kept
    assert abs(Lo[:, :150].std() / Lb[:, :150].std() - 1) < 0.1         # texture (detail) kept
    edge = (Lo - Lb)[:, 170:180].mean() - (Lo - Lb)[:, 140:150].mean()
    assert abs(edge) < 0.25                                             # no dark rim at the edge


def test_named_low_key_stays_low_key_under_auto():
    """Named test (section 17): a candle-lit, low-key frame keeps its brightness under Auto."""
    sc = np.full((200, 300, 3), 0.004, np.float32)                      # deep shadow
    yy, xx = np.mgrid[0:200, 0:300]
    face = np.exp(-((yy - 100) ** 2 + (xx - 150) ** 2) / (2 * 40.0 ** 2))
    sc += (0.06 * face)[..., None] * np.array([1.0, 0.8, 0.6], np.float32)
    sc[95:105, 200:205] = 3.0                                           # the candle flame, far over clip
    img = _img(sc=sc)
    a = auto_light(img, E.Edit())
    assert a["whites"] == 0.0                                           # a dark frame is not "dull"
    e = E.Edit()
    for k, v in a.items():
        setattr(e.light, k, v)
    y0 = E.render(img, E.Edit()) @ np.array([0.288, 0.712, 0.0], np.float32)
    y1 = E.render(img, e) @ np.array([0.288, 0.712, 0.0], np.float32)
    assert abs(math.log2(np.median(y1) / np.median(y0))) < 0.1


def test_light_and_colour_defaults_change_nothing():
    img = _img()
    d = E.Edit().to_dict()
    d.pop("light")
    d.pop("colour")
    assert np.array_equal(E.render(img, E.Edit()), E.render(img, E.Edit.from_dict(d)))


def test_smart_lighting_region_matches_full_frame():
    img = _img(320, 480)
    e = E.Edit()
    e.light.smart = 80
    full = E.render(img, e)
    part = E.render(img, e, region=(120, 100, 160, 120))
    assert np.abs(part - full[100:220, 120:280]).max() < 0.01


# ------------------------------------------------------------------------------------------ white balance
def test_auto_white_balance_finds_a_coloured_light():
    """A grey-edge estimate under a warm light gives warm-correcting multipliers."""
    from studio.engine.color.autowb import grey_edge
    sc = scene(256, 384)
    light = np.array([1.0, 0.75, 0.45], np.float32)                     # warm (candle-like) illuminant
    e = grey_edge(np.clip(sc * light / 1.2, 0, 1))
    assert e is not None
    m = (1 / e) / (1 / e).min()
    assert m[2] > m[1] > m[0] * 0.99


def test_eyedropper_makes_the_spot_neutral():
    from studio.engine.pipeline.pipeline import pick_white
    sc = np.full((120, 180, 3), 0.3, np.float32)
    img = _img(sc=sc)                                                   # synthetic raw undoes WB (2.0, 1.0, 1.5)
    m = pick_white(img, E.Edit(), 0.5, 0.5)
    assert m is not None and np.allclose(m / m[1], [2.0, 1.0, 1.5], rtol=0.03)


# ------------------------------------------------------------------------------------------ looks, copy, versions
def test_looks_apply_only_what_they_name():
    from studio.engine.presets import looks as LK
    names = [lk["name"] for lk in LK.list_looks()]
    assert "Bill's Candlelight (draft)" in names and "Natural" in names
    e = E.Edit(exposure_ev=0.7)
    e.white_balance.mode = "temp_tint"
    e.geometry.crop = [0.1, 0.1, 0.9, 0.9]
    e.lens.tag = "Laack 13.5cm Anastigmat"
    out = LK.apply_look(e, LK.find_look("Warm film"))
    assert out.exposure_ev == 0.7 and out.white_balance.mode == "temp_tint" and out.geometry.crop == [0.1, 0.1, 0.9, 0.9]
    assert out.lens.tag == "Laack 13.5cm Anastigmat" and out.colour.grading and out.tone.contrast == 1.3
    back = LK.apply_look(out, LK.find_look("Natural"))
    assert G.is_identity(back.colour) and LT.is_identity(back.light)
    for lk in LK.list_looks():                                          # every starter look renders
        E.Edit.from_dict(LK.apply_look(E.Edit(), LK.load_look(lk["path"])).to_dict())


def test_save_and_reload_a_look(tmp_path, monkeypatch):
    from studio.engine.presets import looks as LK
    monkeypatch.setenv("STUDIO_SETTINGS_DIR", str(tmp_path))
    e = E.Edit()
    e.colour.vibrance = 33
    p = LK.save_look(LK.look_from_edit(e, "My look", ["colour"]))
    assert json.load(open(p))["settings"]["colour"]["vibrance"] == 33
    assert LK.find_look("my look")["settings"]["colour"]["vibrance"] == 33


def test_copy_paste_groups():
    from studio.engine.presets.looks import copy_groups, describe_change
    a = E.Edit(exposure_ev=1.0)
    a.light.shadows = 40
    a.colour.saturation = 20
    a.geometry.rotate = 3
    a.lens.tag = "A"
    b = E.Edit()
    b.lens.tag = "B"
    c = copy_groups(a, b, ["light", "colour"])
    assert c.exposure_ev == 1.0 and c.light.shadows == 40 and c.colour.saturation == 20
    assert c.geometry.rotate == 0 and c.lens.tag == "B"
    assert describe_change(b, copy_groups(a, b, ["colour"])) == "Colour: saturation +20"


def test_virtual_copies_snapshots_and_history(tmp_path):
    raw = tmp_path / "IMG_0001.CR3"
    raw.write_bytes(b"x")
    p = str(raw)
    e = E.Edit(exposure_ev=0.5)
    E.save_edit(p, e)
    E.save_library_meta(p, {"rating": 4, "pick": 1, "label": ""})
    i = E.add_version(p, E.Edit(exposure_ev=-1.0), "Dark one")
    assert E.list_versions(p) == ["Master", "Dark one"] and i == 1
    assert E.load_version(p, 1).exposure_ev == -1.0 and E.load_version(p, 0).exposure_ev == 0.5
    v = E.load_version(p, 1)
    v.colour.bw = True
    E.save_version(p, 1, v)
    E.add_snapshot(p, "Before colour", e)
    E.save_history(p, 0, [("Exposure +0.5", e)])
    E.save_edit(p, E.Edit(exposure_ev=0.25))                            # saving the master keeps all of it
    assert E.load_version(p, 1).colour.bw and E.list_snapshots(p)[0]["name"] == "Before colour"
    assert E.load_library_meta(p)["rating"] == 4 and E.load_history(p, 0)[0][0] == "Exposure +0.5"
    E.delete_version(p, 1)
    assert E.list_versions(p) == ["Master"]
    with pytest.raises(ValueError):
        E.delete_version(p, 0)


def test_grading_lightness_works_without_a_tint():
    px = np.full((1, 1, 3), 0.01, np.float32)
    c = _colour(grading={"shadows": [0, 0, 50]})
    assert not G.is_identity(c)
    out = G.apply(px, c)
    assert G.to_oklab(out)[0, 0, 0] > G.to_oklab(px)[0, 0, 0] + 0.01
    assert np.allclose(out[0, 0], out[0, 0, 1], atol=1e-4)            # still neutral


def test_a_look_never_carries_brightness():
    from studio.engine.presets import looks as LK
    look = {"name": "x", "settings": {"exposure_ev": 1.5, "tone": {"contrast": 1.5, "mid_grey": 0.2,
                                                                    "match_camera_brightness": True}}}
    out = LK.apply_look(E.Edit(exposure_ev=-0.3), look)
    assert out.exposure_ev == -0.3 and out.tone.contrast == 1.5
    assert out.tone.mid_grey == E.Edit().tone.mid_grey and not out.tone.match_camera_brightness
    saved = LK.look_from_edit(E.Edit(exposure_ev=2.0), "y", ["light"])
    assert "exposure_ev" not in saved["settings"]
