"""Auto HDR on one raw (plan 23.6 / 23.12): measure the range, open shadows, hold highlights, keep local contrast."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from conftest import scene, synthetic_raw
from studio import engine as E
from studio.engine.color import hdr as H
from studio.engine.pipeline.pipeline import Image

MID = 0.11


def _rgb(S):
    return np.repeat((MID * np.exp2(np.asarray(S, np.float32)))[..., None], 3, 2).astype(np.float32)


def _S(rgb):
    return np.log2(np.maximum(rgb[..., 1], 1e-12) / MID)


def _contrasty(h=384, w=576, seed=1):
    """A backlit scene: a dark room (about -6 EV) with texture, a bright window (about +3.5 EV) with texture."""
    rng = np.random.default_rng(seed)
    tex = gaussian_filter(rng.standard_normal((h, w)), 3.0) * 1.2
    S = np.full((h, w), -6.0) + tex
    S[60:200, 330:520] = 3.5 + tex[60:200, 330:520]
    S[250:340, 60:260] = -1.0 + tex[250:340, 60:260]             # a mid-tone object
    return S.astype(np.float32)


def test_the_auto_part_measures_the_scene():
    S = _contrasty()
    r = H.scene_range(S)
    assert r["lo"] < -5.5 and r["hi"] > 3.0
    assert 1.5 < r["need_lift"] <= H.MAX_LIFT and 0.8 < r["need_pull"] <= H.MAX_PULL
    flat = H.plan(np.zeros((64, 64), np.float32), H.HDRParams())
    assert flat["fits"] and flat["lift"] == pytest.approx(0.6 * H.MIN_LIFT)


@pytest.mark.parametrize("engine", list(H.ENGINES))
def test_every_engine_opens_shadows_holds_highlights_and_keeps_local_contrast(engine):
    S = _contrasty()
    out, rep = H.apply_whole(_rgb(S), H.HDRParams("medium", 1.0, engine), MID)
    So = _S(out)
    room, window = (slice(220, 370), slice(300, 560)), (slice(80, 180), slice(350, 500))
    assert So[room].mean() > S[room].mean() + 0.6, "the dark room opens up"
    assert So[window].mean() < S[window].mean() - 0.3, "the window is held back"
    assert So[window].mean() > So[room].mean() + 3.0, "but a window is still far brighter than the room"
    for area in (room, window):                                    # local contrast (the texture) survives
        assert np.std(So[area]) > 0.7 * np.std(S[area])
    assert rep["lift"] > 0 and rep["pull"] > 0 and rep["engine"] == engine


@pytest.mark.parametrize("engine", list(H.ENGINES))
def test_named_no_halos_and_tones_stay_in_order(engine):
    """A 6-stop edge between two flat areas: each side stays flat (no bright or dark rim), and a smooth ramp
    stays in order."""
    h, w = 256, 512
    S = np.full((h, w), -5.0, np.float32)
    S[:, w // 2:] = 2.5
    out, _ = H.apply_whole(_rgb(S), H.HDRParams("heavy", 1.0, engine), MID)
    So = _S(out)
    for side in (slice(24, w // 2 - 4), slice(w // 2 + 4, w - 24)):
        flat = So[40:-40, side]
        assert float(flat.max() - flat.min()) < 0.25, f"rim of {flat.max() - flat.min():.2f} stops beside the edge"
    ramp = (np.linspace(-8.0, 4.0, w)[None, :] + np.zeros((h, 1))).astype(np.float32)
    out, _ = H.apply_whole(_rgb(ramp), H.HDRParams("heavy", 1.5, engine), MID)
    assert np.all(np.diff(_S(out)[h // 2, 8:-8]) > 0), "tones must stay in order"


def test_strengths_do_what_their_names_say():
    S = _contrasty()
    rgb = _rgb(S)
    room, window = (slice(220, 370), slice(300, 560)), (slice(80, 180), slice(350, 500))
    lift = {}
    for name in H.STRENGTHS:
        So = _S(H.apply_whole(rgb, H.HDRParams(name, 1.0), MID)[0])
        lift[name] = (float((So - S)[room].mean()), float((So - S)[window].mean()))
    assert lift["light"][0] < lift["medium"][0] < lift["heavy"][0], "Light < Medium < Heavy in the shadows"
    assert abs(lift["highlights"][0]) < 0.05 and lift["highlights"][1] < -0.3, "Highlights only: shadows as shot"
    assert abs(lift["shadows"][1]) < 0.05 and lift["shadows"][0] > 0.6, "Shadows only: highlights as shot"
    half = _S(H.apply_whole(rgb, H.HDRParams("medium", 0.5), MID)[0])
    assert 0.3 * lift["medium"][0] < float((half - S)[room].mean()) < 0.8 * lift["medium"][0], "Amount scales it"
    assert not H.HDRParams("medium", 0.0).active
    assert all(s["does"] for s in H.STRENGTHS.values())
    assert all(e["best"] and e["weak"] for e in H.ENGINES.values())


def test_named_low_key_stays_low_key():
    """Named test (23.6): a candlelit portrait stays a candlelit portrait: blacks stay black, the flame keeps
    its place, only the lower mid-tones rise."""
    h, w = 320, 480
    yy, xx = np.mgrid[:h, :w]
    S = np.full((h, w), -9.0, np.float32)                                         # a black room
    S += 6.5 * np.exp(-(((xx - 300) / 110.0) ** 2 + ((yy - 170) / 90.0) ** 2))    # a face lit to about -2.5 EV
    S = np.maximum(S, -9.0 + 0.0)
    flame = ((xx - 120) ** 2 + ((yy - 150) / 2.0) ** 2) < 8 ** 2
    S[flame] = 4.0
    out, rep = H.apply_whole(_rgb(S), H.HDRParams("lowkey", 1.0), MID)
    So = _S(out)
    black = (slice(0, 40), slice(380, 480))
    assert abs(float((So - S)[black].mean())) < 0.1, "blacks stay black"
    face = (slice(150, 190), slice(280, 320))
    assert 0.1 < float((So - S)[face].mean()) < 1.6, "the lower mid-tones rise"
    assert float(np.median(So)) < -6.0, "the photo as a whole stays dark"
    assert float(So[flame].mean()) > 2.0, "the flame is still the brightest thing by far"
    heavy = _S(H.apply_whole(_rgb(S), H.HDRParams("heavy", 1.0), MID)[0])
    assert float((heavy - S)[black].mean()) > float((So - S)[black].mean()) + 0.3, "Heavy would have lifted the room"


def test_noise_aware_lift():
    """23.12: the lift is limited where the signal is mostly noise. Lifting noise only shows noise."""
    S = _contrasty()
    rgb = _rgb(S)
    room = (slice(220, 370), slice(300, 560))

    def noisy(y):                                           # noise equal to the signal at about -6 EV
        return np.full_like(np.asarray(y, np.float64), MID * 2.0 ** -6)

    clean = _S(H.apply_whole(rgb, H.HDRParams("heavy", 1.0), MID, None)[0])
    out, rep = H.apply_whole(rgb, H.HDRParams("heavy", 1.0), MID, noisy)
    lim = _S(out)
    assert float((lim - S)[room].mean()) < 0.6 * float((clean - S)[room].mean())
    assert rep["noise_limited"] > 0.3
    noisy.resid_levels = [0.16, 0.3, 0.5, 0.75, 0.9]        # after the denoiser the shadows may be lifted further
    den = _S(H.apply_whole(rgb, H.HDRParams("heavy", 1.0), MID, noisy)[0])
    assert float((den - S)[room].mean()) > float((lim - S)[room].mean()) + 0.3


def test_colour_is_kept():
    S = _contrasty()
    rgb = _rgb(S) * np.array([1.3, 1.0, 0.6], np.float32)
    out, _ = H.apply_whole(rgb, H.HDRParams("heavy", 1.0), MID)
    assert np.allclose(out[..., 0] / out[..., 1], 1.3, rtol=1e-4) and np.allclose(out[..., 2] / out[..., 1], 0.6,
                                                                                    rtol=1e-4)


# ------------------------------------------------------------------------------------------ in the pipeline
def test_pipeline_off_by_default_region_matches_and_report():
    img = Image(synthetic_raw(scene(320, 480), read_noise=2.0))
    e = E.Edit()
    e.demosaic = "mhc"
    assert not e.hdr.enabled, "Auto HDR never runs by itself (Bill's rule: no automatic brightness changes)"
    base = E.render(img, e)
    assert E.hdr_report(img, e) is None
    e.hdr.enabled, e.hdr.strength = True, "heavy"
    full = E.render(img, e)
    assert float(np.abs(full - base).mean()) > 1e-3
    rep = E.hdr_report(img, e)
    assert rep["strength"] == "heavy" and "range" in rep and rep["lift"] >= 0
    x, y, w, h = 160, 96, 160, 128
    part = E.render(img, e, region=(x, y, w, h))
    d = np.abs(part - full[y:y + h, x:x + w])
    assert float(d.max()) < 0.02, f"the 100% view must match the export: {d.max():.4f}"
    e2 = E.Edit.from_json(e.to_json())
    assert e2.hdr.enabled and e2.hdr.strength == "heavy" and e2.raw_key() == e.raw_key()
    for eng in H.ENGINES:
        e.hdr.engine = eng
        assert np.isfinite(E.render(img, e, scale=0.5)).all()
