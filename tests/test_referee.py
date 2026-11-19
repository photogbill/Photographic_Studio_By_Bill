"""The Dual Pixel referee (studio/engine/raw/referee.py, BUILD_PLAN_NOISE H.1 and H.2) on synthetic halves with
known truth: a scene seen through two halves of a lens with opposite shading, each half with its own photon
noise; one part of the frame out of focus (the halves are shifted against each other there)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as nd_shift

from studio.engine.calib import synth as S
from studio.engine.calib.noise import NoiseModel
from studio.engine.raw import auto_strength as AS
from studio.engine.raw import denoise as DN
from studio.engine.raw import referee as REF

GAIN, READ = 2.0e-4, 1.0e-7                 # photon noise variance per unit signal, read noise variance (sum frame)


def _model(scale: float = 1.0) -> NoiseModel:
    return NoiseModel(a=np.full((2, 2), GAIN * scale, np.float32), b=np.full((2, 2), READ * scale, np.float32))


def _halves(n=384, level=0.25, noise=1.0, seed=0, defocus=True, texture=0.3):
    """(sum, half 1, half 2, the clean halves) as mosaics. Left third: out of focus, halves 3 px apart."""
    rng = np.random.default_rng(seed)
    leaves = np.asarray(S.dead_leaves(n, n, seed=seed + 1), np.float32)
    leaves = leaves.mean(-1) if leaves.ndim == 3 else leaves
    sc = level * (1.0 + texture * (leaves - leaves.mean()))
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :]
    s1, s2 = 1.3 - 0.08 * x, 0.7 + 0.08 * x                              # each half darker toward one side, and
    # one a different brightness (a real R5 frame changes about 0.05% per photosite; this is five times that)
    h1, h2 = 0.5 * sc * s1, 0.5 * sc * s2
    if defocus:
        blur = gaussian_filter(sc, 2.0)
        cut = n // 3
        h1[:, :cut] = (0.5 * nd_shift(blur, (0, 1.5), order=1, mode="nearest") * s1)[:, :cut]
        h2[:, :cut] = (0.5 * nd_shift(blur, (0, -1.5), order=1, mode="nearest") * s2)[:, :cut]
    sd = lambda v: np.sqrt(noise * (GAIN * v + READ / 2.0))              # noqa: E731
    a = h1 + rng.standard_normal(h1.shape).astype(np.float32) * sd(h1)
    b = h2 + rng.standard_normal(h2.shape).astype(np.float32) * sd(h2)
    return (a + b).astype(np.float32), a.astype(np.float32), b.astype(np.float32), h1, h2


def test_the_two_halves_measure_this_photos_noise_against_the_profile():
    for true_noise, want in ((1.0, 1.0), (1.44, 1.2)):                   # 1.44 x the variance = 1.2 x the noise
        ab, a, b, _, _ = _halves(noise=true_noise, seed=3)
        p = REF.prepare(ab, a, b, S.RGGB, _model())
        chk = REF.noise_check([p])
        assert chk["ratio"] == pytest.approx(want, abs=0.04), chk


def test_places_where_the_halves_do_not_show_the_same_scene_are_left_out():
    ab, a, b, _, _ = _halves(seed=4)
    p = REF.prepare(ab, a, b, S.RGGB, _model())
    w = p["w"]
    n = w.shape[1]
    assert float(w[:, : n // 3 - 12].mean()) < 0.35 < 0.85 < float(w[:, n // 3 + 12:].mean())


def test_the_referee_ranks_strengths_as_the_truth_does_and_knows_the_detail_kept():
    ab, a, b, h1, h2 = _halves(seed=5, defocus=False)
    model = _model()
    p = REF.prepare(ab, a, b, S.RGGB, model)
    clean = REF._planes(2.0 * h2, S.RGGB, model)[0]                      # the other half without its noise
    fn = DN.denoise_starlet
    c = REF.choose(fn, [p], 4, "luma", weight=0.0)
    true = {s: REF._wmean((fn(p["ya"], s, 4) * p["gain"] - clean) ** 2, p["w"]) for s in AS.CANDIDATES}
    best = min(true, key=true.get)
    assert true[c["strength"]] <= true[best] * 1.1 + 0.005, (c["table"], true)
    for s, r in c["table"]:
        assert r == pytest.approx(true[s], abs=0.06), (s, r, true[s])    # the error itself, not only the order
    from studio.lab.data import texture_kept
    for s, kept in c["kept_table"]:
        f = fn(p["ya"], s, 4) * p["gain"]
        assert kept == pytest.approx(texture_kept(f, None, clean) / texture_kept(p["ya"] * p["gain"], None, clean),
                                     abs=0.05), s
    careful = REF.choose(fn, [p], 4, "luma")
    assert careful["strength"] <= c["strength"] and careful["kept"] >= c["kept"]


def test_a_dark_or_unfocused_photo_gives_no_verdict_and_a_normal_raw_none(monkeypatch):
    from conftest import synthetic_raw
    from studio.engine import Edit
    from studio.engine.pipeline.pipeline import Image, clean_auto
    rng = np.random.default_rng(1)
    rgb = np.clip(0.2 + 0.05 * rng.standard_normal((480, 640, 3)), 0.01, 1.0).astype(np.float32)
    image = Image(synthetic_raw(rgb))
    edit = Edit()
    edit.denoise.enabled = True
    assert REF.judge_photo(image, edit, [(0, 0, 384, 384)], "hybrid") is None        # not a Dual Pixel file
    # a Dual Pixel photo, fed from the synthetic halves
    state = {}

    def fake_halves(image, frame2, box, use_ob=True):
        return state["ab"], state["a"], state["b"]
    monkeypatch.setattr(REF, "second_frame", lambda image: (None, None))
    monkeypatch.setattr(REF, "halves", fake_halves)
    monkeypatch.setattr(type(image), "noise_model", lambda self, use_ob=True: _model())
    state["ab"], state["a"], state["b"], _, _ = _halves(level=0.0004, seed=6)        # read noise rules: too dark
    r = REF.judge_photo(image, edit, [(0, 0, 384, 384)], "starlet")
    assert "luma" not in r and "too dark" in r["note"]
    state["ab"], state["a"], state["b"], _, _ = _halves(seed=7, defocus=False)
    r = REF.judge_photo(image, edit, [(0, 0, 384, 384)] * 2, "starlet")
    assert r["kind"] == "referee" and 0.3 <= r["luma"] <= 2.6
    assert r["noise_ratio"] is None and r["noise_ratio_estimated_model"] == pytest.approx(1.0, abs=0.05)
    assert r["link"] is None or r["link"] > -0.04, "independent halves: no shared noise found"
    assert r["used"] > 0.8 and r["kept"] is not None
    edit.denoise.method = "starlet"
    out = clean_auto(image, edit)                                         # the pipeline takes the referee's word
    assert out["kind"] == "referee" and image.auto_report["luma"] == out["luma"]


def test_halves_that_share_their_noise_are_not_trusted_to_judge(monkeypatch):
    """The file stores the sum and one half, so in a dark photo the halves share read noise in opposite
    directions. That is measured in the photo's flat places, and then the referee gives no verdict."""
    ab, a, b, _, _ = _halves(seed=9, defocus=False, texture=0.0)
    p = REF.prepare(ab, a, b, S.RGGB, _model())
    assert abs(REF.noise_link([p])) < 0.04
    shared = np.random.default_rng(2).standard_normal(a.shape).astype(np.float32) * np.sqrt(GAIN * 0.125) * 0.5
    a2, b2 = a + shared, b - shared                             # the same disturbance, opposite ways
    p2 = REF.prepare(a2 + b2, a2, b2, S.RGGB, _model())
    link = REF.noise_link([p2])
    assert link < -REF.MAX_LINK, link
