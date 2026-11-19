"""BUILD_PLAN_NOISE: real-noise truth (calib/synth.py, calib/deep.py) and the detail guard (raw/guard.py)."""
from __future__ import annotations

import numpy as np
import pytest

from studio.engine.calib import builder as B
from studio.engine.calib import deep as D
from studio.engine.calib import profile as P
from studio.engine.calib import synth as S
from studio.engine.raw import denoise as DN
from studio.engine.raw import guard as G
from test_calibration_drop import pile  # noqa: F401  (the fixture: files whose "raw" content is synthetic)


def _camera(add, isos=(800, 6400), rn=(4.0, 30.0), gain=(0.45, 0.057), band=1.5):
    files = []
    for iso, r, g in zip(isos, rn, gain):
        files += [add(f"d_{iso}_{s}.cr2", iso=iso, rn=r, row_band=band, seed=iso + s, hot=[(40, 44)]) for s in range(5)]
        files += [add(f"f_{iso}_{s}.cr2", iso=iso, rn=r, level_dn=5000, gain=g, seed=9000 + iso + s) for s in range(2)]
    return files


def test_deep_calibration_builds_a_noise_bank_and_the_fixed_pattern(pile):
    add, opened, _ = pile
    files = _camera(add)
    prof = B.calibrate_drop(files, workers=1).built[0][0]
    rep, = D.deep_calibrate(files, workers=1)
    assert rep["model"] == "Test Camera" and set(rep["groups"]) == {"800|short", "6400|short"}
    bank = D.find_bank("Synthetic", "Test Camera")
    patches, info = bank.patches(800)
    assert patches.dtype == np.uint16 and patches.shape[1:] == (256, 256) and patches.shape[0] == 20
    assert info["sigma_dn"] == pytest.approx(4.0, abs=0.4) and abs(info["neighbour_h"]) < 0.05
    assert bank.patches(1600) is None and bank.isos() == [800.0, 6400.0]
    rows, cols = bank.fixed_lines(800)
    assert rows.shape == (400,) and cols.shape == (600,)
    # the synthetic camera's row banding is random from frame to frame: almost none of it is fixed
    assert rep["fixed"]["800"]["one_frame_rows_dn"] == pytest.approx(1.5, abs=0.4)
    assert rep["fixed"]["800"]["fixed_rows_dn"] < 0.6
    src = S.NoiseSource(prof, bank)
    assert src.isos() == [800.0, 6400.0] and src.has_real_noise(6400)


def test_synthetic_photo_has_the_cameras_noise(pile):
    """The harness is honest: a flat grey made at ISO 6400 has the variance the camera's own flats have."""
    add, _, _ = pile
    files = _camera(add)
    prof = B.calibrate_drop(files, workers=1).built[0][0]
    D.deep_calibrate(files, workers=1)
    src = S.NoiseSource(prof, D.find_bank("Synthetic", "Test Camera"))
    rng = np.random.default_rng(0)
    for iso, rn, gain in ((800, 4.0, 0.45), (6400, 30.0, 0.057)):
        span, model = src.range_dn(iso), src.model(iso)
        for level in (0.05, 0.3):
            noisy = src.add(np.full((512, 512), level, np.float32), iso, rng)
            p = noisy[0::2, 0::2]
            p = p - np.median(p, axis=1, keepdims=True)              # the bank carries row banding: not pixel noise
            want = level * span / gain + rn * rn                     # DN^2: photons + read noise
            assert p.var() * span ** 2 == pytest.approx(want, rel=0.12)
            assert model.sigma(level) ** 2 * span ** 2 == pytest.approx(want, rel=0.12)
            assert noisy.mean() == pytest.approx(level, rel=0.03)
    formula = S.NoiseSource(prof, None)
    assert not formula.has_real_noise(800) and "formula" in formula.model(800).notes[0]


def test_reduce_planes_keeps_the_mosaic_and_halves_the_noise():
    rng = np.random.default_rng(1)
    rgb = S.dead_leaves(256, 256, seed=3)
    mos = S.mosaic_of(rgb) + rng.normal(0, 0.02, (256, 256)).astype(np.float32)
    small = S.reduce_planes(mos, 2)
    assert small.shape == (128, 128)
    flat = S.reduce_planes(np.full((256, 256), 0.4, np.float32) + rng.normal(0, 0.02, (256, 256)), 2)
    assert flat.std() == pytest.approx(0.01, rel=0.1)
    grey = S.reduce_planes(S.mosaic_of(np.dstack([np.full((64, 64), v, np.float32) for v in (0.2, 0.4, 0.6)])))
    assert np.allclose(grey[0::2, 0::2], 0.2) and np.allclose(grey[1::2, 1::2], 0.6)       # colours stay put


# ------------------------------------------------------------------------------------------------ the guard
def _texture(n=384, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    t = 2.2 * np.sin(x / 2.3 + 0.7 * np.sin(y / 9.0)) * np.cos(y / 2.9)             # fine, low-contrast texture
    t[:, : n // 2] = 0.0                                                            # left half: flat
    return (40.0 + t).astype(np.float32), rng


def test_guard_returns_almost_nothing_from_pure_noise():
    rng = np.random.default_rng(5)
    noise = rng.standard_normal((512, 512)).astype(np.float32)
    back, rep = G.structure(noise)
    assert float(np.mean(back * back)) < 0.01                          # under 1% of the noise energy
    assert all(abs(v - 1.0) < 0.1 for v in rep["band_energy"])         # white noise reads 1.0 in every band
    out, r = G.give_back(noise + 10.0, np.full_like(noise, 10.0))      # a denoiser that removed all of it
    assert r["returned"] < 0.01


def test_guard_gives_back_texture_an_oversmoothing_cleaner_took():
    clean, rng = _texture()
    noisy = clean + rng.standard_normal(clean.shape).astype(np.float32)
    crude = DN.denoise_starlet(noisy, 4.0, 4)                           # far too strong: texture goes with noise
    out, rep = G.give_back(noisy, crude)
    tex = np.s_[32:-32, 224:-32]
    flat = np.s_[32:-32, 32:160]
    err = lambda a, r: float(np.mean((a[r] - clean[r]) ** 2))           # noqa: E731
    assert err(out, tex) < 0.75 * err(crude, tex)                       # texture came back
    assert err(out, flat) < err(crude, flat) + 0.03                     # and the flat half did not get noisier
    assert rep["returned"] > 0.003 and rep["band_energy"][1] > rep["band_energy_after"][1]


def test_noise_envelope_stops_invention():
    rng = np.random.default_rng(2)
    noisy = 20.0 + rng.standard_normal((256, 256)).astype(np.float32)
    invented = np.full_like(noisy, 20.0)
    y, x = np.mgrid[0:256, 0:256]
    invented += (6.0 * np.sin(x / 3.0)).astype(np.float32) * (y > 128)  # a "texture" far above the noise
    out, held = G.envelope(noisy, invented)
    assert np.abs(out[160:, :] - 20.0).std() < 0.5 * np.abs(invented[160:, :] - 20.0).std()
    assert held > 0.01
    same, held0 = G.envelope(noisy, np.full_like(noisy, 20.0))          # an honest result passes untouched
    assert held0 < 1e-3 and np.abs(same - 20.0).max() < 0.5


def test_denoise_mosaic_with_the_guard_keeps_more_texture():
    from studio.engine.calib.noise import NoiseModel
    rng = np.random.default_rng(4)
    y, x = np.mgrid[0:256, 0:256].astype(np.float32)
    a, b = 2e-4, 1e-6
    tex = np.sin(x / 4.6 + 0.7 * np.sin(y / 18.0)) * np.cos(y / 5.8)     # fine texture at about the noise level
    clean = (0.25 + 0.012 * tex).astype(np.float32)                     # (noise sigma here is about 0.007)
    noisy = (rng.poisson(clean / a) * a + rng.normal(0, np.sqrt(b), clean.shape)).astype(np.float32)
    model = NoiseModel(a=np.full((2, 2), a), b=np.full((2, 2), b), source="calibrated", iso=6400)
    st = {}
    plain = DN.denoise_mosaic(noisy, S.RGGB, model, DN.DenoiseParams(method="starlet", luma=2.5, chroma=2.5))
    kept = DN.denoise_mosaic(noisy, S.RGGB, model, DN.DenoiseParams(method="starlet", luma=2.5, chroma=2.5,
                                                                    guard=True), stats=st)
    assert np.mean((kept - clean) ** 2) < np.mean((plain - clean) ** 2)
    assert 0.0 < st["guard"]["returned"] < 0.9


def test_the_harness_scores_methods_on_real_noise(pile, tmp_path):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "tools"))
    import noise_bench as NB
    add, _, _ = pile
    files = _camera(add)
    prof = B.calibrate_drop(files, workers=1).built[0][0]
    D.deep_calibrate(files, workers=1)
    res = NB.run(prof, [800.0, 6400.0], quick=True, say=lambda *_: None)
    rows = res["rows"]
    assert {r["method"] for r in rows} == {"hybrid", "starlet", "nlm", "starlet x2.5", "hybrid, auto strength",
                                           "fusion, auto strength", "collab (BM3D class), auto"} and len(rows) == 28
    assert all(r["real_noise"] and 0 < r["noise_left"] < 1.2 and "texture_kept" in r and "blotches_left" in r
               for r in rows)
    for iso in (800.0, 6400.0):                               # the guard never loses texture; usually it gains
        for m in ("hybrid", "starlet", "nlm"):
            off, on = [next(r for r in rows if r["iso"] == iso and r["method"] == m and r["guard"] is g)
                       for g in (False, True)]
            assert on["texture_kept"] >= off["texture_kept"] - 0.005
        off, on = [next(r for r in rows if r["iso"] == iso and r["method"] == "starlet x2.5" and r["guard"] is g)
                   for g in (False, True)]
        assert on["texture_kept"] >= off["texture_kept"] and on["residual"] <= off["residual"]
        if iso == 6400.0:                                     # where over-cleaning bites, the guard gives back
            assert on["texture_kept"] > off["texture_kept"] + 0.02 and on["noise_left"] < off["noise_left"]
    text = NB.page(res)
    assert "Texture kept" in text and "data:image/png" in text and "detail guard" in text


def test_the_harness_is_checked_against_real_flats_and_scores_photos_that_have_no_truth(pile):
    """A.4: a harness flat has the noise of the camera's real flat pair; a wrong gain is caught. A.5: a real
    photo, which has no truth, still gets an estimated error and strength for every method."""
    import copy
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "tools"))
    import noise_bench as NB
    add, _, _ = pile
    files = _camera(add)
    prof = B.calibrate_drop(files, workers=1).built[0][0]
    D.deep_calibrate(files, workers=1)
    src = S.NoiseSource(prof, D.find_bank("Synthetic", "Test Camera"))
    hon = NB.honesty(prof, src)
    assert {r["iso"] for r in hon} == {800.0, 6400.0} and all(r["ok"] for r in hon), hon
    wrong = copy.deepcopy(prof)
    for e in wrong.isos:
        e.gain_e_per_dn *= 2.0                                 # half the photon noise: the check must notice
    bad = NB.honesty(wrong, S.NoiseSource(wrong, D.find_bank("Synthetic", "Test Camera")))
    assert bad and not any(r["ok"] for r in bad)
    photo = add("photo.cr2", iso=6400, rn=30.0, level_dn=3000, gain=0.057, seed=77)
    import studio.engine.io.rawfile as R
    from studio.engine.pipeline.pipeline import Image
    said = []
    rows = NB.real_photos([photo], methods=("starlet",), say=said.append, opener=lambda f: Image(R.open_raw(f)))
    assert rows, said
    assert rows[0]["kind"] == "strength" and 0.0 <= rows[0]["noise_left"] < 0.6 and rows[0]["luma"] >= 1.0
    text = NB.page(dict(camera="Test Camera", made="now", rows=[], crops={}, honesty=hon, photos=rows))
    assert "Is this benchmark honest?" in text and "2 of 2 do" in text and "photo.cr2" in text
