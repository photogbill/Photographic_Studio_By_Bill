""""Detail by scale" (plan 23.11): every microcontrast method, stackable, on synthetic data with known answers."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from conftest import scene, synthetic_raw
from studio import engine as E
from studio.engine.detail import microcontrast as M
from studio.engine.detail import scales as S
from studio.engine.pipeline import pipeline as PL
from studio.engine.pipeline.pipeline import Image

MID = 0.11
ALL = dict(tonal=1.0, local=1.0, crisp=1.0, separation=1.0, veil=1.0, broad=1.0)
FLAT = dict(shadows=1.0, midtones=1.0, highlights=1.0)


def _rgb(L):
    return np.repeat(np.exp2(np.asarray(L, np.float32))[..., None], 3, 2).astype(np.float32)


def _L(rgb):
    return np.log2(np.maximum(rgb[..., 1], 1e-12))


def _wave(amp, period, ev=0.0, h=384, w=640):
    x = np.arange(w)[None, :] + np.zeros((h, 1))
    return math.log2(MID) + ev + amp * np.sin(2 * np.pi * x / period)


def _gain(p, amp, period, ev=0.0, noise=None, long_edge=6000.0):
    L = _wave(amp, period, ev)
    out = _L(S.apply(_rgb(L), p, MID, noise, 1.0, long_edge))
    c = (slice(96, 288), slice(192, 448))
    return float(np.std(out[c]) / np.std(L[c]))


# ---------------------------------------------------------------------------------------------- basics
def test_defaults_change_nothing():
    rgb = _rgb(_wave(0.3, 40))
    assert S.apply(rgb, S.ScaleParams(), MID) is rgb
    e = E.Edit()
    assert not PL._mc_params(e).active


def test_old_edits_render_with_the_first_engine_bit_for_bit():
    """A sidecar from before this panel holds only microcontrast.amount: same code path, same pixels."""
    rgb = _rgb(_wave(0.3, 40) + 0.2 * np.random.default_rng(0).standard_normal((384, 640)).astype(np.float32) * 0.1)
    p = S.ScaleParams(local=0.8)
    assert p.only_local
    a = S.apply(rgb, p, MID, None, 1.0, 6000.0)
    b = M.apply(rgb, M.MCParams(amount=0.8), MID, None, 1.0, 6000.0)
    assert np.array_equal(a, b)


def test_edit_round_trip_and_switched_off_methods():
    e = E.Edit()
    e.detail.tonal, e.detail.crisp, e.detail.veil = 60.0, -20.0, 35.0
    e.microcontrast.amount = 0.4
    e.detail.off = ["crisp", "local"]
    e2 = E.Edit.from_json(e.to_json())
    assert e2.detail.tonal == 60.0 and e2.detail.off == ["crisp", "local"]
    p = PL._mc_params(e2)
    assert p.tonal == pytest.approx(0.6) and p.veil == pytest.approx(0.35)
    assert p.crisp == 0.0 and p.local == 0.0, "a switched-off method keeps its amount in the sidecar but does nothing"
    old = E.Edit.from_dict({"microcontrast": {"amount": 0.5}})            # a sidecar from before the panel
    assert old.detail.tonal == 0.0 and PL._mc_params(old).only_local


# ---------------------------------------------------------------------------------------------- the bands
def test_tonal_lifts_medium_scales_and_relaxes_the_finest():
    p = S.ScaleParams(tonal=1.0, **FLAT)
    assert _gain(p, 0.2, 64) > 2.2, "small tonal differences at medium scales must be plainly lifted"
    assert _gain(p, 0.2, 32) > 2.0
    assert 0.7 < _gain(p, 0.2, 3) < 0.95, "the finest scale is relaxed a little, never blurred"
    assert _gain(p, 1.0, 64) < _gain(p, 0.2, 64), "big differences get less than small ones"


def test_tonal_at_seventy_percent_is_plain_to_see():
    """23.11: the effect at 60-80% must be plainly visible. The first engine gave about +30% here."""
    assert _gain(S.ScaleParams(tonal=0.7), 0.2, 64) > 1.6          # midtones, default tone weights
    assert _gain(S.ScaleParams(local=0.7), 0.2, 64) < 1.4


def test_crisp_lifts_the_finest_scale_only_and_negative_softens_gently():
    p = S.ScaleParams(crisp=1.0, **FLAT)
    assert _gain(p, 0.2, 4) > 1.8
    assert _gain(p, 0.2, 128) < 1.1
    soft = S.ScaleParams(crisp=-0.5, **FLAT)
    g = _gain(soft, 0.2, 4)
    assert 0.55 < g < 0.9, "negative crisp softens, capped so it never looks blurred"
    assert abs(_gain(soft, 0.2, 128) - 1.0) < 0.05


def test_broad_lifts_large_shapes_only():
    L = _wave(0.3, 600, h=768, w=1536)
    p = S.ScaleParams(broad=1.0, **FLAT)
    out = _L(S.apply(_rgb(L), p, MID, None, 1.0, 6000.0))
    c = (slice(200, 560), slice(300, 1200))
    assert np.std(out[c]) / np.std(L[c]) > 1.25
    assert abs(_gain(p, 0.2, 16) - 1.0) < 0.05


def test_saturated_blue_gets_detail_too():
    """ProPhoto luminance weighs blue at 0.0001: a violet flower would count as black and get nothing."""
    L = _wave(0.2, 64)
    rgb = np.zeros(L.shape + (3,), np.float32)
    rgb[..., 2] = np.exp2(L) * 3.0
    rgb[..., 0] = np.exp2(L) * 0.2
    out = S.apply(rgb, S.ScaleParams(tonal=1.0), MID)
    c = (slice(96, 288), slice(192, 448))
    assert np.std(np.log2(out[..., 2][c])) / np.std(L[c]) > 1.5
    assert np.allclose(out[..., 0] / out[..., 2], rgb[..., 0] / rgb[..., 2], rtol=1e-4), "hue stays put"


# ---------------------------------------------------------------------------------- shared budget + meter
def test_stacked_methods_share_one_budget_per_scale():
    alone = S.plan(S.ScaleParams(tonal=1.0))
    assert not alone["bands"]["medium"]["limited"]
    both = S.plan(S.ScaleParams(tonal=1.0, local=1.0))
    assert both["bands"]["medium"]["limited"], "the meter turns amber when a stack reaches the guard"
    use = 0.8 * M.A_MAX * both["w_leg"] + both["a_tonal"] * S.SLOPE_COST
    assert use.max() <= 1.0 + 1e-9, "the no-tone-reversal limit applies to the total, not to each method"
    assert both["a_tonal"].sum() < alone["a_tonal"].sum(), "both are scaled back together where they overlap"
    fine = S.plan(S.ScaleParams(crisp=1.0, tonal=1.0))["bands"]["fine"]
    assert fine["total"] == pytest.approx(S.K_CRISP - S.FINE_RELAX) and not fine["limited"]


def test_meter_reports_each_methods_share():
    m = S.meter(S.ScaleParams(tonal=0.6, crisp=0.5, broad=0.2))
    assert set(m) == {"fine", "medium", "broad"}
    assert m["medium"]["parts"] == {"tonal": pytest.approx(0.6 * S.K_TONAL)}
    assert m["fine"]["parts"]["crisp"] > 0 > m["fine"]["parts"]["tonal"]
    assert m["broad"]["total"] > 0
    assert S.meter(S.ScaleParams())["medium"]["total"] == 0


def test_simple_level_is_a_saved_combination_of_the_method_sliders():
    assert S.from_simple(0.6, -1.0) == {"tonal": 0.6, "crisp": 0.0}
    assert S.from_simple(0.6, 0.0) == {"tonal": 0.6, "crisp": 0.6}
    assert S.from_simple(0.6, 1.0) == {"tonal": 0.0, "crisp": 0.6}
    for a, t in ((0.5, -0.5), (0.8, 0.3), (0.3, 0.0), (1.0, -1.0), (0.4, 1.0)):
        m = S.from_simple(a, t)
        a2, t2 = S.to_simple(m["tonal"], m["crisp"])
        assert a2 == pytest.approx(a, abs=1e-3) and t2 == pytest.approx(t, abs=1e-3)
    for t in np.linspace(-1, 0, 5):                    # on the Tonal side the medium band is never below the fine
        m = S.from_simple(0.7, float(t))
        assert m["tonal"] >= m["crisp"]
    for name, vals in S.PRESETS.items():
        assert set(vals) <= set(S.METHODS), name
    assert all(k in S.METHODS and S.METHODS[k]["best"] and S.METHODS[k]["weak"] for k in S.ORDER)


# ---------------------------------------------------------------------------------------- named tests
def _edge_scene():
    h, w = 384, 640
    L = np.full((h, w), math.log2(MID) - 1.5, np.float32)
    L[:, w // 2:] += 3.0                               # a 3-stop edge between two flat areas
    return L


PAIRS = [dict([a, b]) for a, b in itertools.combinations(ALL.items(), 2)]


@pytest.mark.parametrize("pair", PAIRS, ids=["+".join(p) for p in PAIRS])
def test_named_no_halos_for_every_pair_at_100(pair):
    """Named test (23.11): every pair of methods at 100% passes the no-halo test."""
    L = _edge_scene()
    out = _L(S.apply(_rgb(L), S.ScaleParams(**pair, **FLAT), MID))
    w = L.shape[1]
    for side in (slice(40, w // 2 - 6), slice(w // 2 + 6, w - 40)):
        flat = out[60:-60, side]
        assert float(flat.max() - flat.min()) < 0.06, "a flat area next to a big edge must stay flat (no halo)"


@pytest.mark.parametrize("pair", PAIRS, ids=["+".join(p) for p in PAIRS])
def test_named_candle_flame_survives_every_pair_at_100(pair):
    """Named test: a small flame 6 stops above a dark room keeps its brightness and gets no dark ring."""
    h, w = 384, 640
    yy, xx = np.mgrid[:h, :w]
    r2 = (xx - w / 2) ** 2 + ((yy - h / 2) / 2.2) ** 2
    L = (math.log2(MID) - 4.0 + 6.0 * np.exp(-r2 / (2 * 7.0 ** 2))).astype(np.float32)
    out = _L(S.apply(_rgb(L), S.ScaleParams(**pair, **FLAT), MID))
    veil = "veil" in pair                              # veil deepens the room by design; the flame must not move
    assert abs(float(out[h // 2, w // 2] - L[h // 2, w // 2])) < (0.25 if veil else 0.2)
    ring = out[h // 2 - 60:h // 2 + 60, w // 2 - 60:w // 2 + 60] - L[h // 2 - 60:h // 2 + 60, w // 2 - 60:w // 2 + 60]
    far = float(np.median(out[:40, :40] - L[:40, :40]))
    assert float(ring.min()) > far - 0.35, "no dark ring around the flame"


def test_named_stack_of_all_methods_never_reverses_tones():
    """Named test (23.11): a stack of all methods at 100% never reverses tones."""
    h, w = 256, 1024
    ramp = np.linspace(-6.0, 3.0, w)[None, :] + np.zeros((h, 1))
    L = (math.log2(MID) + ramp).astype(np.float32)
    out = _L(S.apply(_rgb(L), S.ScaleParams(**ALL, **FLAT), MID))
    row = out[h // 2, 8:-8]
    assert np.all(np.diff(row) > -1e-4), "a smooth ramp must stay in order"
    steps = math.log2(MID) - 5.0 + 0.25 * (np.arange(w) // 32)[None, :] + np.zeros((h, 1))     # a staircase
    out = _L(S.apply(_rgb(steps.astype(np.float32)), S.ScaleParams(**ALL, **FLAT), MID))
    means = [float(out[60:-60, i * 32 + 8:i * 32 + 24].mean()) for i in range(1, w // 32 - 1)]
    assert np.all(np.diff(means) > 0), "every step stays darker than the next"


def test_tonal_remap_is_monotone_up_to_the_budget():
    d = np.linspace(-4, 4, 40001).astype(np.float32)
    r = S._remap_tonal(d, 0.0, 0.98 / S.SLOPE_COST)
    assert np.all(np.diff(r) >= -1e-6), "tones must never swap order, even with the whole budget spent"
    assert np.all(np.diff(S._remap_tonal(d, 0.0, -S.SOFT_MAX)) > 0)
    assert abs(float(S._remap_tonal(np.float32(3.0), 0.0, 2.0)) - 3.0) < 1e-6, "an edge is left exactly alone"


def test_a_clean_gradient_is_left_alone():
    """No ripple on smooth gradients (skies, out-of-focus backgrounds), even with everything at 100%."""
    h, w = 256, 1024
    L = (math.log2(MID) + np.linspace(-6.0, 3.0, w)[None, :] + np.zeros((h, 1))).astype(np.float32)
    for kw, tol in ((dict(tonal=1.0), 0.01), (dict(tonal=1.0, local=1.0, crisp=1.0, broad=1.0), 0.05)):
        out = _L(S.apply(_rgb(L), S.ScaleParams(**kw, **FLAT), MID))
        # right up to the frame edge too: a gradient running off the frame is not a crease to "bring out"
        assert float(np.abs(out - L).max()) < tol


def test_named_high_iso_shadows_get_at_least_as_much_as_highlights():
    """Named test (23.11): R5 at ISO 4000 with the denoiser on: shadows gain >= highlights gain. The noise toe
    uses the noise the denoiser leaves, not the noise it started with."""
    a, b = 8.68e-4, 3.2e-6                             # the R5 ISO 4000 noise model measured from Bill's frame

    def sigma(y):                                       # as pipeline.detail_noise_fn builds it (green gain 1)
        y = np.maximum(np.asarray(y, np.float32), 0.0)
        return 0.8 * np.sqrt(a * y + b)
    p = S.ScaleParams(tonal=0.7)
    raw_sh = _gain(p, 0.15, 48, ev=-3.0, noise=sigma, long_edge=8191.0)
    sigma.resid_levels = [0.16, 0.28, 0.49, 0.74, 0.89]   # hybrid denoiser at 0.6 (measured)
    sh = _gain(p, 0.15, 48, ev=-3.0, noise=sigma, long_edge=8191.0)
    hi = _gain(p, 0.15, 48, ev=+1.5, noise=sigma, long_edge=8191.0)
    assert sh >= hi and sh > 1.5
    assert sh > raw_sh, "with the denoiser on, the shadows may be lifted more, not less"


def test_denoise_residual_table():
    e = E.Edit()
    assert PL.denoise_residual(e) == [1.0] * 5
    e.denoise.enabled, e.denoise.luma = True, 0.6
    r = PL.denoise_residual(e)
    assert r[0] < 0.25 and r[0] < r[2] < r[4] < 1.0
    e.denoise.grain = 0.5
    assert PL.denoise_residual(e)[0] > r[0], "keep-grain puts fine noise back"


def test_noise_is_not_lifted_as_if_it_were_detail():
    rng = np.random.default_rng(5)
    L = math.log2(MID) - 3.0 + 0.2 * rng.standard_normal((384, 640)).astype(np.float32)   # nothing but noise

    def sigma(y):
        return 0.2 * math.log(2.0) * np.asarray(y, np.float64)                            # 0.2 stops at every tone
    for p in (S.ScaleParams(tonal=1.0, **FLAT), S.ScaleParams(crisp=1.0, **FLAT)):
        out = _L(S.apply(_rgb(L), p, MID, sigma))
        assert np.std(out) < 1.2 * np.std(L)


# ----------------------------------------------------------------------- veil and tonal separation
def test_veil_takes_off_the_floor_and_holds_mid_grey():
    h, w = 256, 384
    truth = np.exp2(np.linspace(-9.0, 2.0, w))[None, :] * MID + np.zeros((h, 1))
    hazy = truth + 0.02 * MID * 4.0                      # a grey floor: dark tones merge
    rgb = np.repeat(hazy[..., None], 3, 2).astype(np.float32)
    out = S.apply(rgb, S.ScaleParams(veil=1.0), MID)[..., 1]
    dark = slice(4, 60)
    before = np.log2(hazy[0, dark][-1] / hazy[0, dark][0])
    after = np.log2(out[0, dark][-1] / out[0, dark][0])
    assert after > 1.5 * before, "dark tones separate once the floor is gone"
    i = int(np.argmin(np.abs(hazy[0] - MID)))
    assert abs(math.log2(out[0, i] / hazy[0, i])) < 0.05, "mid grey stays where it was"
    assert out.min() >= 0 and np.all(np.diff(out[0]) > 0)
    add = S.apply(rgb, S.ScaleParams(veil=-1.0), MID)[..., 1]
    assert np.log2(add[0, dark][-1] / add[0, dark][0]) < before, "a negative amount adds a haze"


def test_veil_leaves_a_dark_subject_alone():
    """A frame whose darkest tone is a mid-dark subject has no stray-light floor to remove (the floor is capped)."""
    rgb = np.full((128, 192, 3), MID * 0.8, np.float32)
    rgb[:, 96:] = MID * 1.6
    out = S.apply(rgb, S.ScaleParams(veil=1.0), MID)
    assert float(np.abs(np.log2(out / rgb)).max()) < 0.6


def test_tonal_separation_spreads_crowded_tones_without_shifting_the_range():
    rng = np.random.default_rng(2)
    h, w = 384, 512
    L = math.log2(MID) - 2.0 + gaussian_filter(rng.standard_normal((h, w)), 12) * 3.0     # crowded around -2 EV
    L[:40, :40] = math.log2(MID) - 6.0
    L[-40:, -40:] = math.log2(MID) + 2.0
    L = L.astype(np.float32)
    out = _L(S.apply(_rgb(L), S.ScaleParams(separation=1.0, **FLAT), MID))
    mid = (slice(60, -60), slice(60, -60))
    assert np.std(out[mid]) > 1.15 * np.std(L[mid]), "crowded tones are spread apart"
    assert abs(float(out[:40, :40].mean() - L[:40, :40].mean())) < 0.3, "the dark end stays put"
    assert abs(float(out[-40:, -40:].mean() - L[-40:, -40:].mean())) < 0.3, "the bright end stays put"
    # region by region: the same tone curve applied to tones in order keeps them in order
    f = S.measure_fields(_rgb(L), S.ScaleParams(separation=1.0, **FLAT), 1.0, MID)
    s = np.linspace(-10, 3, 400).astype(np.float32)
    for py, px in ((10.0, 10.0), (200.0, 300.0), (380.0, 500.0)):
        t = s + S.sample_separation(f, s, np.full_like(s, px), np.full_like(s, py))
        assert np.all(np.diff(t) > 0)


def test_balance_number_follows_the_methods():
    rng = np.random.default_rng(4)
    L = (math.log2(MID) + gaussian_filter(rng.standard_normal((512, 768)), 1.0) * 0.6
         + gaussian_filter(rng.standard_normal((512, 768)), 6.0) * 2.0).astype(np.float32)
    rgb = _rgb(L)
    base = S.balance_ratio(rgb[..., 1])
    tonal = S.balance_ratio(S.apply(rgb, S.ScaleParams(tonal=0.8, **FLAT), MID)[..., 1])
    crisp = S.balance_ratio(S.apply(rgb, S.ScaleParams(crisp=0.8, **FLAT), MID)[..., 1])
    assert tonal > base > crisp, "Tonal raises the medium-to-fine ratio (modelled), Crisp lowers it (etched)"


# ------------------------------------------------------------------------------------------ in the pipeline
def _image(h=320, w=480, **kw):
    return Image(synthetic_raw(scene(h, w), read_noise=2.0, **kw))


def test_pipeline_renders_every_method_and_the_region_matches_the_full_frame():
    """Veil, separation and broad are measured on the whole frame, so a region render (the 100% view) shows the
    same pixels as the export."""
    img = _image()
    e = E.Edit()
    e.demosaic = "mhc"
    base = E.render(img, e)
    e.detail.tonal, e.detail.crisp, e.detail.separation, e.detail.veil, e.detail.broad = 60, 40, 50, 40, 40
    e.microcontrast.amount = 0.3
    full = E.render(img, e)
    assert full.shape == base.shape and np.isfinite(full).all()
    assert float(np.abs(full - base).mean()) > 1e-3
    x, y, w, h = 160, 96, 160, 128
    part = E.render(img, e, region=(x, y, w, h))
    d = np.abs(part - full[y:y + h, x:x + w])
    assert float(d.max()) < 0.02 and float(d.mean()) < 0.002, f"region differs from the full frame: {d.max():.4f}"


def test_pipeline_off_switch_and_raw_cache():
    img = _image()
    e = E.Edit()
    e.demosaic = "mhc"
    base = E.render(img, e, scale=0.5)
    e.detail.tonal = 70
    e.detail.off = ["tonal"]
    assert np.array_equal(E.render(img, e, scale=0.5), base), "switched off = no change"
    key = e.raw_key()
    e.detail.off = None
    e.detail.veil, e.detail.broad = 30, 30
    assert not np.array_equal(E.render(img, e, scale=0.5), base)
    assert e.raw_key() == key, "the panel never forces the raw stages to run again"


def test_command_line_flags():
    """Develop Raws.bat / studio develop: --detail and --hdr reach the edit."""
    from studio import cli
    a = cli.build_parser().parse_args(["develop", "x.cr3", "--detail", "tonal=70,crisp=-20,veil=40,local=30",
                                       "--hdr", "lowkey:laplacian", "--hdr-amount", "80"])
    e = E.Edit()
    cli.apply_flags(e, a)
    assert (e.detail.tonal, e.detail.crisp, e.detail.veil) == (70, -20, 40) and e.microcontrast.amount == 0.3
    assert e.hdr.enabled and (e.hdr.strength, e.hdr.engine, e.hdr.amount) == ("lowkey", "laplacian", 80)
    a = cli.build_parser().parse_args(["develop", "x.cr3", "--detail", "Both"])
    e = E.Edit()
    cli.apply_flags(e, a)
    assert (e.detail.tonal, e.detail.crisp, e.detail.broad) == (55, 45, 20) and not e.hdr.enabled
    with pytest.raises(SystemExit):
        cli.apply_flags(E.Edit(), cli.build_parser().parse_args(["develop", "x.cr3", "--detail", "sparkle=10"]))
