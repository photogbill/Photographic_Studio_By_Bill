"""Measured highlights from the lens glow (raw/glow.py): DESAT for cameras."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from conftest import synthetic_raw
from studio import engine as E
from studio.engine.raw import glow as G
from studio.engine.pipeline.pipeline import Image

WG = 0.06                                    # share of the light in the glow


def kfun(r, r0=8.0, beta=2.6):
    """A lens glow (full-size pixels): light per pixel at distance r per unit of light."""
    rr = np.geomspace(3, 6000, 4000)
    kk = 1 / (1 + (rr / r0) ** 2) ** (beta / 2)
    trap = getattr(np, "trapezoid", None) or np.trapz
    e = trap(kk * 2 * np.pi * rr, rr)
    return (1 / (1 + (r / r0) ** 2) ** (beta / 2)) * WG / e


def through_lens(T):
    """What the sensor sees: (1 - glow share) of each light where it is, plus its glow everywhere."""
    h, w = T.shape[:2]
    yk = np.arange(-h + 1, h)[:, None]
    xk = np.arange(-w + 1, w)[None, :]
    K = kfun(np.maximum(np.hypot(yk, xk), 0.75))
    return np.stack([(1 - WG) * T[..., c] + fftconvolve(T[..., c], K, mode="same") for c in range(3)], -1)


def true_profile(name="TestLens", lens="Test lens", fnumber=None, w2=360):
    rs = np.geomspace(3, 1000, 40)                          # half-size pixels: 2x the radius, 4 photosites each
    return G.Profile(name=name, lens=lens, fnumber=fnumber, r=list(rs), k=[list(kfun(rs * 2) * 4)] * 3, plane_width=w2)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_SETTINGS_DIR", str(tmp_path))
    return tmp_path


def _scene(h=480, w=720, light=(180, 260, 9.0, (30, 18, 5)), face=True):
    yy, xx = np.mgrid[0:h, 0:w]
    T = np.full((h, w, 3), 0.01, np.float32)
    if face:
        T[200:420, 420:640] = [0.12, 0.09, 0.07]
    cy, cx, s, val = light
    g0 = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    for c, v in enumerate(val):
        T[..., c] += v * g0
    return T


# ------------------------------------------------------------------------------------------ the estimate (planes)
def _planes_case(extra_bg=False):
    rng = np.random.default_rng(2)
    h, w = 400, 600
    yy, xx = np.mgrid[0:h, 0:w]
    T = np.full((h, w, 3), 0.004, np.float32)
    T[150:300, 350:500] = [0.08, 0.06, 0.05]
    if extra_bg:                                            # a bright wall right beside the candle
        T[100:140, 165:230] = [0.6, 0.5, 0.4]

    def disc(cy, cx, r, val):
        rr = np.hypot(yy - cy, xx - cx)
        p = np.clip(1 - (rr / r) ** 2, 0, 1)
        for c in range(3):
            T[..., c] += val[c] * p

    disc(120, 150, 8, [60, 35, 10])                         # a candle: blown in every channel
    disc(300, 200, 5, [5, 2.5, 0.5])                        # a lamp: red and green blown, blue not
    rs = np.geomspace(1.5, 800, 40)

    def k2(r):                                              # the same glow, in half-size terms
        rr = np.geomspace(1.5, 3000, 4000)
        kk = 1.0 / (1 + (rr / 4.0) ** 2) ** 1.3
        trap = getattr(np, "trapezoid", None) or np.trapz
        return (1.0 / (1 + (r / 4.0) ** 2) ** 1.3) * WG / trap(kk * 2 * np.pi * rr, rr)
    prof = G.Profile(name="t", lens="T", r=list(rs), k=[list(k2(rs))] * 3, plane_width=w)
    yk = np.arange(-h + 1, h)[:, None]
    xk = np.arange(-w + 1, w)[None, :]
    K = k2(np.maximum(np.hypot(yk, xk), 0.75))
    O = np.stack([(1 - WG) * T[..., c] + fftconvolve(T[..., c], K, mode="same") for c in range(3)], -1)
    O += rng.normal(0, 0.0015, O.shape)
    mult = np.array([2.0, 1.0, 1.5])
    lim = G.CLIP * mult
    Kc = O >= lim
    P = np.minimum(O, lim).astype(np.float32)

    def flux(A, cy, cx, r):
        m = np.hypot(yy - cy, xx - cx) <= r + 3
        return A[m].sum(0)
    return T, P, Kc, mult, prof, flux


def test_blown_candle_is_measured_from_the_glow_with_a_lens_profile():
    T, P, Kc, mult, prof, flux = _planes_case()
    est, rep = G.estimate(P, Kc, mult, prof)
    tiers = sorted(r["tier"] for r in rep)
    assert tiers == ["lens", "lens"]
    true, got, clipped = flux(T, 120, 150, 8), flux(est, 120, 150, 8), flux(P, 120, 150, 8)
    assert (clipped / true).max() < 0.3                     # the clip kept under a third of the light
    ratio = got / true
    assert ratio.min() > 0.85 and ratio.max() < 1.1         # every channel within 15%
    assert abs((got[1] / got[0]) / (true[1] / true[0]) - 1) < 0.05    # and its colour within 5%
    assert abs((got[2] / got[0]) / (true[2] / true[0]) - 1) < 0.05


def test_partly_blown_light_is_measured_from_its_glow_colour_without_a_profile():
    T, P, Kc, mult, prof, flux = _planes_case()
    est, rep = G.estimate(P, Kc, mult, None)
    by = {(r["y"] // 2 // 10, r["x"] // 2 // 10): r["tier"] for r in rep}
    assert by[(12, 15)] == "rebuilt"                        # blown in every channel: nothing to measure from
    assert by[(30, 20)] == "photo"
    ratio = flux(est, 300, 200, 5) / flux(T, 300, 200, 5)
    assert ratio.min() > 0.9 and ratio.max() < 1.1


def test_a_bright_scene_beside_the_light_does_not_inflate_it():
    T, P, Kc, mult, prof, flux = _planes_case(extra_bg=True)
    est, rep = G.estimate(P, Kc, mult, prof)
    ratio = flux(est, 120, 150, 8) / flux(T, 120, 150, 8)
    assert ratio.max() < 1.15


# ------------------------------------------------------------------------------------------ measuring a lens
def _bracket(pedestal=1e-7):
    H, W = 600, 900
    yy, xx = np.mgrid[0:H, 0:W]
    T = np.zeros((H, W, 3), np.float32)
    g0 = np.exp(-((yy - 300) ** 2 + (xx - 450) ** 2) / (2 * 4.0 ** 2))
    for c, v in enumerate([1.0, 0.8, 0.5]):
        T[..., c] = v * 400 * g0
    O = through_lens(T) + pedestal
    raws = {}
    for i, t in enumerate([1 / 2000, 1 / 125, 1 / 8, 2.0]):
        s = t / (1 / 2000) / (O.max() * 1.3)
        r = synthetic_raw((O * s).astype(np.float32), clip_white=True, read_noise=2.0, seed=i)
        r.meta.exposure_s = t
        r.meta.lens = "Laack 135 test"
        raws[f"f{i}.cr2"] = r
    return raws


@pytest.mark.parametrize("pedestal", [1e-7, 3e-4])
def test_glow_profile_from_a_bracket_matches_the_lens(settings, monkeypatch, pedestal):
    import studio.engine.io.rawfile as RF
    raws = _bracket(pedestal)
    monkeypatch.setattr(RF, "open_raw", lambda p: raws[p])
    rep = E.measure_glow_profile(list(raws))
    assert rep["ok"] and rep["frames"] == 4 and rep["stops"] > 15
    p = G.load_profile(rep["name"])
    assert p.lens == "Laack 135 test" and str(settings) in rep["path"]
    rr = np.asarray(p.r)
    truth = kfun(rr * 2) * 4
    for c in range(3):
        ratio = np.asarray(p.k[c]) / truth
        assert 0.85 < ratio.min() and ratio.max() < 1.18, (c, ratio.round(2))
    assert rr[-1] / rr[0] > 10                              # over more than a decade of distance


def test_measuring_needs_a_frame_where_the_light_is_not_blown(settings, monkeypatch):
    import studio.engine.io.rawfile as RF
    raws = _bracket()
    raws.pop("f0.cr2")
    raws.pop("f1.cr2")
    monkeypatch.setattr(RF, "open_raw", lambda p: raws[p])
    rep = E.measure_glow_profile(list(raws))
    assert not rep["ok"] and "shortest" in rep["reason"]


def test_profile_is_picked_by_lens_and_nearest_aperture(settings):
    G.save_profile(true_profile("A_f4", "Laack 135", 4.0))
    G.save_profile(true_profile("A_f11", "Laack 135", 11.0))
    G.save_profile(true_profile("B", "Heliar", None))
    assert G.pick_profile("Laack 135", 5.6) == "A_f4"
    assert G.pick_profile("laack 135", 9.0) == "A_f11"
    assert G.pick_profile("Heliar", None) == "B"
    assert G.pick_profile("Tessar", 8.0) == ""
    assert [p["name"] for p in E.glow_profiles()] == ["A_f11", "A_f4", "B"]


# ------------------------------------------------------------------------------------------ in the renderer
def test_glow_mode_brings_back_a_blown_light(settings):
    G.save_profile(true_profile())
    O = through_lens(_scene())
    img = Image(synthetic_raw(O.astype(np.float32), clip_white=True, read_noise=3.0))
    core = (slice(176, 185), slice(256, 265))
    under = dict(exposure_ev=-5.0)
    rebuilt = E.render(img, E.Edit(highlight_mode="rebuild", **under))[core].mean(axis=(0, 1))
    glow = E.render(img, E.Edit(highlight_mode="glow", glow_profile="TestLens", **under))[core].mean(axis=(0, 1))
    truth = E.render(Image(synthetic_raw((O / 32).astype(np.float32), clip_white=True, read_noise=3.0 / 32)),
                     E.Edit())[core].mean(axis=(0, 1))
    assert np.abs(glow - truth).sum() < 0.6 * np.abs(rebuilt - truth).sum()      # much closer to what was there
    rep = E.glow_report(img, E.Edit(highlight_mode="glow", glow_profile="TestLens"))
    assert len(rep) == 1 and rep[0]["tier"] == "lens" and rep[0]["stops"] > 1
    assert abs(rep[0]["colour"][1] - 0.6) < 0.06 and abs(rep[0]["colour"][2] - 5 / 30) < 0.04
    # without a profile a light blown in every channel falls back to the colour rebuild, exactly
    nop = E.render(img, E.Edit(highlight_mode="glow", **under))[core].mean(axis=(0, 1))
    assert np.allclose(nop, rebuilt, atol=1e-6)
    e = E.Edit(highlight_mode="glow", glow_profile="TestLens")
    full = E.render(img, e)
    part = E.render(img, e, region=(200, 120, 160, 120))
    assert np.abs(part - full[120:240, 200:360]).max() < 1e-5


def test_glow_mode_changes_nothing_without_clipping(settings):
    G.save_profile(true_profile())
    T = _scene(light=(180, 260, 9.0, (0.3, 0.2, 0.1)))
    img = Image(synthetic_raw(through_lens(T).astype(np.float32), clip_white=True))
    a = E.render(img, E.Edit(highlight_mode="clip"))
    b = E.render(img, E.Edit(highlight_mode="glow", glow_profile="TestLens"))
    assert np.array_equal(a, b)
    assert E.glow_report(img, E.Edit(highlight_mode="glow")) == []


def test_glow_profile_belongs_to_the_light_group_but_never_to_a_look():
    from studio.engine.presets import looks
    src = E.Edit(highlight_mode="glow", glow_profile="TestLens")
    dst = looks.copy_groups(src, E.Edit(), ["light"])
    assert dst.highlight_mode == "glow" and dst.glow_profile == "TestLens"
    lk = looks.look_from_edit(src, "x", ["light"])
    assert "glow_profile" not in lk["settings"]
    assert E.Edit(highlight_mode="glow", glow_profile="a").raw_key() != E.Edit(highlight_mode="glow").raw_key()
    assert E.Edit(glow_profile="a").raw_key() == E.Edit().raw_key()      # ignored unless the mode is on
