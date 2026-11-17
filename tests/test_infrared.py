"""Infrared / full spectrum (BUILD_PLAN_PANO_IR.md 1.3-1.5d): the colour science, on synthetic truth and on Bill's real
850 nm frames when they are in test_raws.

The synthetic converted camera lives here (plan 1.4: "keep the curves in the test, not in the engine"): Gaussian
R, G, B dyes that all become transparent past 800 nm (as measured on Bill's 450D), silicon falling off in the near
infrared, long-pass filters at 550/590/665/720/850 nm, and foliage / sky / skin / grey reflectances.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from conftest import psnr, synthetic_raw

from studio.engine.color import infrared as irm
from studio.engine.pipeline.edit import Edit, WhiteBalance
from studio.engine.pipeline.pipeline import Image, linear_rgb, render

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------------------ the synthetic camera
L = np.arange(350, 1001, 5.0)


def _g(c, s):
    return np.exp(-0.5 * ((L - c) / s) ** 2)


def _sig(e, w=12.0):
    return 1 / (1 + np.exp(-(L - e) / w))


DYE_R = np.maximum(_g(600, 38), _sig(640, 15)) * 0.95
DYE_G = _g(535, 38) + _sig(790, 12) * 0.95 * (1 - _g(535, 38))
DYE_B = _g(460, 32) + _sig(828, 9) * 0.95 * (1 - _g(460, 32))
QE = np.where(L < 600, 0.55 + 0.45 * np.clip((L - 350) / 250, 0, 1), np.exp(-((L - 600) / 420) ** 2))
FILT = {"full": np.ones_like(L), "550": _sig(550, 8), "590": _sig(590, 8), "665": _sig(665, 8),
        "720": _sig(720, 8), "850": _sig(850, 8)}
REFL = {
    "foliage": 0.05 + 0.07 * _g(550, 25) + 0.45 * _sig(705, 12),
    "foliage2": 0.08 + 0.45 * np.clip((L - 400) / 600, 0, 1),        # soil / bark among the leaves
    "sky": np.clip(0.9 * (450 / L) ** 4, 0, 1) + 0.02,
    "skin": 0.2 + 0.4 * _sig(600, 40) - 0.05 * _sig(950, 30),
    "grey": 0.18 + 0 * L,
}


def camera_response(filt: str, material: str) -> np.ndarray:
    s = QE * FILT[filt] * REFL[material]
    return np.array([(s * DYE_R).sum(), (s * DYE_G).sum(), (s * DYE_B).sum()])


def scene_labels(h=192, w=288) -> np.ndarray:
    """0 foliage (lower 60% with blobs), 1 sky (top), 2 skin patch, 3 grey card."""
    lab = np.zeros((h, w), np.int32)
    lab[: int(0.3 * h)] = 1
    lab[int(0.55 * h):int(0.75 * h), int(0.08 * w):int(0.2 * w)] = 2
    lab[int(0.55 * h):int(0.75 * h), int(0.8 * w):int(0.92 * w)] = 3
    return lab


def resp_scale(filt: str) -> float:
    return float(np.array([camera_response(filt, n) for n in ("foliage", "sky", "skin", "grey")]).max())


def render_scene(filt: str, h=192, w=288, seed=3, peak=0.6, noise=0.002) -> tuple[np.ndarray, np.ndarray]:
    """Raw camera RGB (no white balance, 1 = clip) of the synthetic scene through a filter, and its labels."""
    rng = np.random.default_rng(seed)
    lab = scene_labels(h, w)
    names = ["foliage", "sky", "skin", "grey"]
    resp = np.array([camera_response(filt, n) for n in names])
    resp = resp / resp.max() * peak
    img = resp[lab].astype(np.float32)
    # real scenes mix materials: soil and bark (reflectance rising steadily into the infrared) among the leaves, at the same
    # brightness, so the colour varies across the foliage independently of brightness (spectrally: no colour at 850)
    f1, f2 = camera_response(filt, "foliage"), camera_response(filt, "foliage2")
    other = f2 * (f1[1] / f2[1]) / resp_scale(filt) * peak
    mix = np.clip(0.4 + 14.0 * gaussian_filter(rng.normal(0, 1, (h, w)), 20), 0, 1)
    fol = lab == 0
    img[fol] = (1 - mix[fol, None]) * img[fol] + mix[fol, None] * other
    tex = 1.0 + 0.25 * gaussian_filter(rng.normal(0, 1, (h, w)), 1.5)
    tex[lab != 0] = 1.0 + 0.03 * gaussian_filter(rng.normal(0, 1, (h, w)), 3)[lab != 0]
    img *= tex[..., None].astype(np.float32)
    img = np.stack([gaussian_filter(img[..., c], 1.0) for c in range(3)], -1)   # the lens and the AA filter
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 1e-4, 1.0), lab


def _image_of(raw_rgb: np.ndarray, **kw) -> Image:
    return Image(synthetic_raw(raw_rgb.astype(np.float32), wb=(1.0, 1.0, 1.0), **kw))


# ------------------------------------------------------------------------------------------ the family table
def test_family_table_matches_the_synthetic_model():
    for f in ("full", "550", "590", "665", "720"):
        st = irm.channel_stats(render_scene(f)[0])
        frg, fbg, fsr, fsb = irm.FAMILY[f]
        assert abs(st["rg_all"] - frg) < 0.01 and abs(st["bg_all"] - fbg) < 0.01, (f, st)
        assert abs(st["std_rg"] - fsr) < 0.01 and abs(st["std_bg"] - fsb) < 0.01, (f, st)
    # 850: the dyes pass the same light (Bill's measurement): the model agrees
    r = camera_response("850", "foliage")
    assert abs(r[0] / r[1] - 1) < 0.02 and abs(r[2] / r[1] - 1) < 0.02


def test_guess_says_850_for_a_monochrome_raw_and_finds_the_families():
    for f in ("850", "full", "550", "590", "665", "720"):
        cam, _ = render_scene(f, seed=9)
        g = irm.guess_from_stats(irm.channel_stats(cam))
        assert g["key"] == f, (f, g)
        assert g["sentence"].startswith("raw R/G")
    cam, _ = render_scene("850")
    assert irm.guess_from_stats(irm.channel_stats(cam))["confidence"] >= 0.7
    # a visible-light-like raw (R/G about 0.5) matches nothing
    vis = np.clip(render_scene("850")[0] * np.array([0.5, 1.0, 0.55], np.float32), 0, 1)
    assert irm.guess_from_stats(irm.channel_stats(vis))["key"] == "unknown"


def test_colour_meter_none_at_850_and_colour_at_590():
    def planes(f):
        cam, _ = render_scene(f, 384, 576, noise=0.004)
        im = _image_of(cam, read_noise=0.0)
        from studio.engine.raw import linearize as lin
        mos = lin.normalise(im.raw, im.levels(True))
        return irm._planes(mos, im.raw.cfa_rgb)
    m850 = irm.meter_from_planes(planes("850"))
    m720 = irm.meter_from_planes(planes("590"))       # (in the model, 720 nm keeps too little colour to count)
    assert not m850["colour"] and m850["ratio"] < irm.METER_LIMIT
    assert m720["colour"] and m720["ratio"] >= irm.METER_LIMIT
    # the meter guards the guess: a grey-looking raw with colour is caught
    g = irm.guess_from_stats({"rg": 1.0, "bg": 1.0, "spread_rg": 0.01, "spread_bg": 0.01, "corr": 1, "n": 99},
                             meter=m720["ratio"])
    assert g["key"] == "720" and g.get("caught")


def test_foliage_white_balance_neutralises_foliage_within_one_percent():
    cam, lab = render_scene("590", noise=0.001)
    m, rep = irm.foliage_multipliers_cam(cam)
    assert m is not None and "estimated" in rep["text"]
    fol = cam[lab == 0] * m
    ratio = np.median(fol, axis=0)
    ratio = ratio / ratio[1]
    assert abs(ratio[0] - 1) < 0.01 and abs(ratio[2] - 1) < 0.01


def test_presets_keep_white_white_and_the_matrix_step_is_right():
    P = irm._P()
    assert np.allclose(P @ np.ones(3), 1.0, atol=1e-6)
    for p in irm.PRESETS:
        assert p in irm.PRESET_INFO and irm.PRESET_INFO[p]["best"] and irm.PRESET_INFO[p]["weak"]
        M = irm.mixer(p)
        assert np.allclose(M.sum(axis=1), 1.0, atol=1e-6), p
    for k in irm.FILTER_ORDER:
        assert irm.FILTERS[k]["best"] and irm.FILTERS[k]["weak"]
    ir = irm.default_ir("590")
    M = irm.camera_to_working(ir, None)
    assert np.allclose(M @ np.ones(3), 1.0, atol=1e-6)
    swap = irm.mixer("swap_rb")
    assert np.allclose(swap @ np.array([1.0, 0.0, 0.0]), [0, 0, 1])
    assert irm.mixer("swap_rb", 0.0).tolist() == np.eye(3).tolist()
    assert irm.row_sums(irm.mixer("goldie")) == [1.0, 1.0, 1.0]


def test_edits_without_infrared_render_exactly_as_before():
    cam, _ = render_scene("590")
    im = _image_of(cam)
    e = Edit()
    assert "infrared" not in e.raw_key()
    off = Edit.from_dict(e.to_dict())
    off.infrared = dict(irm.default_ir("590"), on=False)
    assert off.raw_key() == e.raw_key()
    a = render(im, e, scale=0.5)
    b = render(im, off, scale=0.5)
    assert np.array_equal(a, b)
    # an old sidecar (no infrared field, no stage version) loads and keys the same
    d = e.to_dict()
    d.pop("infrared")
    d["stage_versions"].pop("infrared")
    old = Edit.from_dict(d)
    assert old.raw_key() == e.raw_key() and old.stage_versions["infrared"] == 1


def test_identity_profile_takes_camera_channels_as_they_are():
    cam, lab = render_scene("590")
    im = _image_of(cam)
    e = Edit(white_balance=WhiteBalance(mode="multipliers", multipliers=[1.0, 1.0, 1.0]))
    e.lens.auto_ca = e.lens.defringe = False
    e.infrared = irm.default_ir("unknown")
    rgb = linear_rgb(im, e, half_size=True, cache=False)
    from conftest import mosaic_of
    from studio.engine.raw.demosaic import half
    want = half(mosaic_of(cam), [[0, 1], [1, 2]]) @ irm._P().T
    inner = (slice(4, -4), slice(4, -4))
    assert np.abs(rgb[inner] - want[: rgb.shape[0], : rgb.shape[1]][inner]).max() < 0.003


# ------------------------------------------------------------------------------------------ mono raw (1.5b)
def _mono_scene(h=256, w=256, seed=5):
    """Fine detail near the sensor's resolution (branches, a zone plate), where demosaicing guesses."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = (xx - w / 2) ** 2 + (yy - h / 2) ** 2
    s = 0.3 + 0.18 * np.cos(np.pi * r2 / (1.2 * max(h, w)))              # a zone plate up to about 2.4 px period
    twigs = gaussian_filter((rng.random((h, w)) > 0.985).astype(float), 0.7) * 2.5
    s = s + 0.15 * np.clip(twigs, 0, 1)
    s[h // 3:h // 3 + 40, w // 4:w // 4 + 60] += 0.2
    s = gaussian_filter(s, 0.6)                                           # the lens and the AA filter
    return np.clip(s, 0.02, 0.9).astype(np.float32)


def test_mono_raw_beats_demosaic_and_leaves_no_grid():
    rng = np.random.default_rng(7)
    truth = _mono_scene()
    gains = np.array([0.97, 1.0, 0.99])
    raw_rgb = np.stack([truth * (gains[c] + 0.01 * truth) if c != 1 else truth for c in range(3)], -1)
    raw_rgb = raw_rgb + rng.normal(0, 0.002, raw_rgb.shape).astype(np.float32)
    raw_rgb = np.clip(raw_rgb, 0, 1)
    im = _image_of(raw_rgb)
    e = Edit(white_balance=WhiteBalance(mode="multipliers", multipliers=[1.0, 1.0, 1.0]))
    e.lens.auto_ca = e.lens.defringe = False
    e.infrared = irm.default_ir("850")
    e.infrared["bw"]["on"] = True
    mono = linear_rgb(im, e, cache=False)[..., 0]
    e2 = Edit.from_dict(e.to_dict())
    e2.infrared["mono_raw"] = False
    dem = linear_rgb(im, e2, cache=False)[..., 0]
    k = float(np.median(truth) / np.median(mono))
    k2 = float(np.median(truth) / np.median(dem))
    p_mono = psnr(mono * k, truth)
    p_dem = psnr(dem * k2, truth)
    assert p_mono > p_dem + 1.0, (p_mono, p_dem)
    # no 2x2 grid: the FFT at the Nyquist corner is below the noise floor
    F = np.abs(np.fft.fft2(mono - mono.mean()))
    h, w = F.shape
    corner = F[h // 2, w // 2] + F[h // 2, 0] + F[0, w // 2]
    floor = np.median(F[h // 4: 3 * h // 4, w // 4: 3 * w // 4])
    assert corner / 3 < 4 * floor


def test_mono_raw_removes_hot_pixels():
    truth = _mono_scene(128, 128)
    raw_rgb = np.repeat(truth[..., None], 3, axis=2).copy()
    im = _image_of(raw_rgb, read_noise=3.0)
    m = im.raw.mosaic
    spots = [(16 + 40, 32 + 50), (16 + 81, 32 + 21), (16 + 100, 32 + 100)]
    for y, x in spots:
        m[y, x] = 14000
    e = Edit(white_balance=WhiteBalance(mode="multipliers", multipliers=[1.0, 1.0, 1.0]))
    e.lens.auto_ca = e.lens.defringe = False
    e.infrared = irm.default_ir("850")
    out = linear_rgb(im, e, cache=False)[..., 0]
    for y, x in spots:
        assert out[y - 16, x - 32] < 0.95, (y, x, out[y - 16, x - 32])


# ------------------------------------------------------------------------------------------ black and white (1.5c)
def test_per_channel_exposure_on_identical_channels_is_one_global_exposure():
    rng = np.random.default_rng(2)
    c = rng.uniform(0.05, 0.6, (32, 32)).astype(np.float32)
    rgb = np.repeat(c[..., None], 3, axis=2)
    bw = irm.default_bw()
    bw.update(on=True, ev=[1.0, -0.5, 0.25])
    m = irm.mono_of(rgb, bw)
    gain = np.mean([2 ** 1.0, 2 ** -0.5, 2 ** 0.25])
    assert np.allclose(m, c * gain, rtol=1e-5)


def test_dark_sky_preset_darkens_sky_against_foliage_and_stays_neutral():
    cam, lab = render_scene("590", noise=0.0005)
    m, _ = irm.foliage_multipliers_cam(cam)
    wb = cam * m
    even = irm.default_bw()
    even["on"] = True
    dark = dict(even, mix=irm.bw_preset_weights("dark_sky").tolist(), preset="dark_sky")
    e_m = irm.mono_of(wb, even)
    d_m = irm.mono_of(wb, dark)
    sky_vs_fol_even = np.log2(np.median(e_m[lab == 1]) / np.median(e_m[lab == 0]))
    sky_vs_fol_dark = np.log2(np.median(np.maximum(d_m[lab == 1], 1e-4)) / np.median(d_m[lab == 0]))
    assert sky_vs_fol_even - sky_vs_fol_dark >= 1.0
    # through the pipeline the result is neutral
    im = _image_of(cam)
    e = Edit(white_balance=WhiteBalance(mode="multipliers", multipliers=m.tolist()))
    e.infrared = irm.default_ir("590")
    e.infrared["bw"] = dark
    e.infrared["bw"]["ev"] = [0.2, 0.0, 0.0]
    out = linear_rgb(im, e, half_size=True, cache=False)
    assert np.abs(out[..., 0] - out[..., 2]).max() < 1e-5 and np.abs(out[..., 0] - out[..., 1]).max() < 1e-5


def test_bw_settings_survive_off_on_and_a_sidecar_round_trip_and_recipes_reapply(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_SETTINGS_DIR", str(tmp_path / "settings"))
    ir = irm.default_ir("720")
    ir["bw"].update(on=True, ev=[0.5, 0, -0.3], contrast=[20, 0, -10], sep=[30, 0, 0],
                    curve=[[[0.25, 0.2], [0.75, 0.8]], None, None], mix=[1.2, 0.1, -0.3])
    colour_mix = list(ir["mixer"])
    ir["bw"]["on"] = False                                         # off: nothing lost
    ir["bw"]["on"] = True
    e = Edit()
    e.infrared = ir
    back = Edit.from_json(e.to_json())
    assert back.infrared == e.infrared and back.infrared["mixer"] == colour_mix
    irm.save_recipe("Dark trees", ir["bw"], camera="Canon EOS 450D", filter_key="720")
    rec = [r for r in irm.load_recipes("720") if r["name"] == "Dark trees"][0]
    fresh = irm.apply_recipe(irm.default_bw(), rec)
    for k in ("ev", "contrast", "sep", "curve", "mix", "keep"):
        assert fresh[k] == ir["bw"][k], k
    assert irm.last_recipe("Canon EOS 450D", "720")["name"] == "Dark trees"
    assert irm.last_recipe("Canon EOS 450D", "590") is None


def test_noise_meter_matches_the_measured_noise_of_a_mix():
    from studio.engine.stack.align import noise_sigma
    rng = np.random.default_rng(11)
    base = np.full((256, 256), 0.3, np.float32) + 0.05 * gaussian_filter(rng.normal(0, 1, (256, 256)), 8)
    common = rng.normal(0, 0.01, base.shape)                      # demosaicing makes channel noise correlated
    rgb = np.stack([base + 0.6 * common + 0.008 * rng.normal(0, 1, base.shape) for _ in range(3)], -1)
    rgb = rgb.astype(np.float32)
    bw = irm.default_bw()
    bw.update(on=True, mix=[-1.0, 3.0, -1.0])
    r = irm.noise_factor(rgb, bw)
    w = irm.mono_weights(bw)
    measured = noise_sigma(rgb @ w.astype(np.float32)) / noise_sigma(rgb.mean(axis=2))
    assert abs(r["factor"] - measured) / measured < 0.10
    assert r["factor"] < float(np.sqrt((w ** 2).sum()) * np.sqrt(3))   # far below the textbook bound
    assert r["colour"] == "amber" and "multiplies the noise" in r["text"]


def test_channel_tone_contrast_and_curve_keep_the_ends():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    rgb = np.stack([x, x, x], -1)[None]
    bw = irm.default_bw()
    bw.update(on=True, contrast=[60, -40, 0], curve=[None, None, [[0.5, 0.6]]])
    t = irm.channel_tone(rgb, bw)[0]
    for c in range(3):
        assert abs(t[0, c]) < 1e-6 and abs(t[-1, c] - 1) < 1e-4
        assert np.all(np.diff(t[:, c]) >= -1e-6)
    assert abs(t[18, 0] - 0.18) < 0.01                              # the pivot holds


def test_palette_is_added_colour_by_brightness():
    g = np.linspace(0, 1, 64, dtype=np.float32)
    out = np.stack([g, g, g], -1)[None]
    assert np.array_equal(irm.apply_palette(out.copy(), {"name": "ironbow", "amount": 0}), out)
    p = irm.apply_palette(out.copy(), {"name": "ironbow", "amount": 1.0})[0]
    Y = p @ np.array([0.2880, 0.7119, 0.0001])
    assert np.all(np.diff(Y) > -0.02)
    assert np.ptp(p[32]) > 0.05                                      # colour, not grey
    for k in irm.PALETTES:
        assert irm.PALETTE_INFO[k].endswith(".") and "added" in irm.PALETTE_INFO[k]


def test_measured_filter_white_balance_is_saved_on_this_pc(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_CALIB_DIR", str(tmp_path / "calib"))
    p = irm.save_filter_wb("Hoya R72", [2.0, 1.0, 4.0], camera="Canon EOS 450D", filter_key="720")
    assert str(p).startswith(str(tmp_path))
    got = irm.load_filter_wbs("Canon EOS 450D")
    assert got[0]["multipliers"] == [1.0, 0.5, 2.0] or got[0]["multipliers"][0] == pytest.approx(2.0)
    assert irm.load_filter_wbs("Canon EOS R5") == []


# ------------------------------------------------------------------------------------------ Bill's 850 frames
REAL = sorted((ROOT / "test_raws" / "infrared_850").glob("*.CR2"))


@pytest.mark.skipif(not REAL, reason="no 850 nm raws in test_raws/infrared_850")
def test_real_850_frames_are_guessed_850_with_no_usable_colour():
    from studio.engine.pipeline.pipeline import open_image
    for p in REAL[:2]:
        im = open_image(str(p))
        m = irm.colour_meter(im)
        g = irm.guess_filter(im, m["ratio"])
        assert g["key"] == "850", (p.name, g)
        assert not m["colour"], (p.name, m)
        assert 0.94 < g["rg"] < 1.0 and 0.97 < g["bg"] < 1.01


def test_channel_mask_from_red_over_blue_selects_the_foliage():
    from studio import engine as E
    from studio.engine.local import masks as mk
    cam, lab = render_scene("590", noise=0.001)
    im = _image_of(cam)
    m, _ = irm.foliage_multipliers_cam(cam)
    e = Edit(white_balance=WhiteBalance(mode="multipliers", multipliers=m.tolist()))
    sh = E.channel_shape(im, e, "r/b")
    V = mk.decode_bitmap(sh["data"], sh["w"], sh["h"]).astype(float) / 255
    lab_s = lab[::2, ::2][: V.shape[0], : V.shape[1]]
    fol, rest = np.median(V[lab_s == 0]), np.median(V[(lab_s == 1)])
    cut = 0.5 * (fol + rest)
    lo, hi = (0.0, cut) if fol < rest else (cut, 1.0)
    sh.update(lo=lo, hi=hi, soft=0.02)
    h, w = lab.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    from studio.engine.pipeline.pipeline import render
    rgb = render(im, e, region=(0, 0, w, h))
    ctx = mk.Ctx(rgb, xs, ys, w, h, 1.0, lambda y: y)
    a = mk.shape_alpha(sh, ctx)
    agree = np.mean((a > 0.5) == (lab == 0))
    assert agree >= 0.90, agree
