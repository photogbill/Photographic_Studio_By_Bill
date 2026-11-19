"""Noise model, VST with exact unbiased inverse, and the starlet denoiser (section 8), on synthetic data."""
from __future__ import annotations

import dataclasses

import math

import numpy as np
import pytest

from conftest import mosaic_of, scene
from studio.engine.calib import noise as N
from studio.engine.raw import denoise as DN

PAT = np.array([[0, 1], [1, 2]])


def noisy_version(clean, a, b, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.poisson(np.maximum(clean, 0) / a) * a + rng.normal(0, np.sqrt(b), clean.shape)).astype(np.float32)


@pytest.mark.parametrize("mu", [0.0005, 0.003, 0.02, 0.2])
def test_vst_gives_unit_noise_and_exact_inverse_is_unbiased(mu):
    a, b = 1e-3, (3e-3) ** 2
    z = noisy_version(np.full(300_000, mu, np.float32), a, b)
    D = N.anscombe(z, a, b)
    assert 0.9 < D.std() < 1.12
    est = N.inverse_anscombe_exact(np.array([D.mean()]), a, b)[0]
    assert abs(est - mu) < 0.03 * mu + 2e-5
    if mu <= 0.003:   # the algebraic inverse is visibly biased in deep shadows; the exact one is not
        alg = N.inverse_anscombe_algebraic(np.array([D.mean()]), a, b)[0]
        assert abs(alg - mu) > 3 * abs(est - mu)


def test_noise_model_is_recovered_from_one_image():
    clean = mosaic_of(scene(1024, 1536)) * 0.6
    a_true, b_true = 2e-3, (4e-3) ** 2
    z = noisy_version(clean, a_true, b_true, seed=3)
    m = N.estimate_from_image(z, read_noise=np.full((2, 2), np.sqrt(b_true)))
    assert m.a[0, 0] == pytest.approx(a_true, rel=0.2)
    assert not m.notes                      # the fit worked (no fallback default used)


def test_noise_model_scaling_and_serialisation():
    m = N.NoiseModel(a=np.full((2, 2), 1e-3), b=np.full((2, 2), 1e-5), source="calibrated", iso=6400)
    m2 = m.scaled(2.0)
    assert m2.a[0, 0] == pytest.approx(2e-3) and m2.b[0, 0] == pytest.approx(4e-5)
    assert N.NoiseModel.from_dict(m.to_dict()).to_dict() == m.to_dict()


def test_yc_rotation_is_orthonormal_and_invertible():
    rng = np.random.default_rng(1)
    planes = [rng.standard_normal((64, 64)).astype(np.float32) for _ in range(4)]
    yc = DN.to_yc(*planes)
    back = DN.from_yc(*yc)
    for p, q in zip(planes, back):
        assert np.allclose(p, q, atol=1e-5)
    for comp in yc:                           # unit noise stays unit noise
        assert 0.93 < comp.std() < 1.07


def test_starlet_reconstructs_exactly():
    x = np.random.default_rng(2).standard_normal((96, 128)).astype(np.float32)
    assert np.allclose(sum(DN.starlet(x, 5)), x, atol=1e-5)


def test_denoiser_improves_psnr_and_keeps_brightness():
    clean = np.clip(mosaic_of(scene(256, 384)) * 0.4, 0, 1)
    a, b = 1.5e-3, (5e-3) ** 2
    z = noisy_version(clean, a, b, seed=4)
    model = N.NoiseModel(a=np.full((2, 2), a), b=np.full((2, 2), b))
    den = DN.denoise_mosaic(z, PAT, model)

    def psnr(x):
        return 10 * np.log10(1 / np.mean((np.sqrt(np.clip(x, 0, 1)) - np.sqrt(clean))[16:-16, 16:-16] ** 2))
    assert psnr(den) > psnr(z) + 4.0
    assert abs(float(den.mean() - clean.mean())) < 0.002 * max(float(clean.mean()), 1e-3) + 2e-4


def test_deep_shadows_stay_neutral_after_denoise():
    """Section 21 change 1 / named test: deep shadows stay neutral (no tint from VST bias)."""
    clean = np.full((256, 256), 0.004, np.float32)          # a deep, flat, neutral shadow (after WB = equal)
    a, b = 2e-3, (6e-3) ** 2
    z = noisy_version(clean, a, b, seed=5)
    den = DN.denoise_mosaic(z, PAT, N.NoiseModel(a=np.full((2, 2), a), b=np.full((2, 2), b)))
    means = [float(den[i::2, j::2].mean()) for i in (0, 1) for j in (0, 1)]
    assert max(means) - min(means) < 0.0006
    assert abs(np.mean(means) - 0.004) < 0.0006


def test_clipped_highlights_are_left_alone():
    clean = np.full((128, 128), 0.5, np.float32)
    clean[40:80, 40:80] = 1.0
    a, b = 1e-3, 1e-5
    z = noisy_version(clean, a, b)
    z[40:80, 40:80] = 1.0
    den = DN.denoise_mosaic(z, PAT, N.NoiseModel(a=np.full((2, 2), a), b=np.full((2, 2), b)))
    assert np.allclose(den[45:75, 45:75], 1.0, atol=1e-6)


def test_impulse_filter_keeps_small_bright_blobs():
    """Candle-flame style test: a 3x3 lens-blurred highlight survives; a single-pixel spike does not."""
    rng = np.random.default_rng(6)
    d = rng.standard_normal((64, 64)).astype(np.float32)
    d[10, 10] += 40.0                                        # hot pixel
    d[30:33, 30:33] += np.array([[10, 20, 10], [20, 40, 20], [10, 20, 10]], np.float32)   # blurred point
    out, n = DN.remove_impulses(d)
    assert out[10, 10] < 5.0
    assert out[31, 31] > 30.0


def _banded(row_sigma, seed=0):
    from conftest import synthetic_raw
    from studio.engine.pipeline.pipeline import Image
    rng = np.random.default_rng(seed)
    img = synthetic_raw(np.full((300, 400, 3), 0.2, np.float32), left=128, read_noise=0.0)
    full = img.mosaic.astype(np.float64)
    full += rng.normal(0, 4.0, full.shape) + rng.normal(0, row_sigma, (full.shape[0], 1))
    img.mosaic = np.clip(np.rint(full), 0, 65535).astype(np.uint16)
    return Image(img)


def _row_noise(arr):
    g = arr[:, :, 1]
    return float(np.std(g.mean(axis=1)))


def test_row_banding_is_removed_when_present():
    from studio import engine as E
    img = _banded(3.0)
    rb = img.row_banding()
    assert rb.weight > 0.8
    on, off = E.Edit(), E.Edit(banding=False)
    on.tone.mode = off.tone.mode = "linear"
    assert _row_noise(E.render(img, on)) < 0.5 * _row_noise(E.render(img, off))


def test_row_banding_never_added_to_a_clean_frame():
    from studio import engine as E
    img = _banded(0.0, seed=3)
    rb = img.row_banding()
    assert rb.weight < 0.3
    on, off = E.Edit(), E.Edit(banding=False)
    on.tone.mode = off.tone.mode = "linear"
    assert _row_noise(E.render(img, on)) <= 1.05 * _row_noise(E.render(img, off))


# ------------------------------------------------------------------------------------------ safeguard (23.12)
def _noisy_flat(level=0.05, n=256, seed=11, a=8.68e-4, b=3.2e-6):
    from studio.engine.calib.noise import NoiseModel
    rng = np.random.default_rng(seed)
    model = NoiseModel(a=np.full((2, 2), a), b=np.full((2, 2), b), source="test")
    clean = np.full((n, n), level, np.float32)
    sig = math.sqrt(a * level + b)
    return clean, clean + sig * rng.standard_normal((n, n)).astype(np.float32), model, sig


def test_safeguard_leaves_pure_noise_to_the_denoiser():
    """Pure noise moves by more than 4.5 sigma about 7 times in a million: nothing visible is let back in."""
    clean, noisy, model, sig = _noisy_flat(n=384)
    st = {}
    plain = DN.denoise_mosaic(noisy, PAT, model, DN.DenoiseParams(method="starlet", luma=1.0, chroma=1.5))
    safe = DN.denoise_mosaic(noisy, PAT, model, DN.DenoiseParams(method="starlet", luma=1.0, chroma=1.5,
                                                              safeguard=4.5), stats=st)
    assert st["sites"] == noisy.size and st["protected"] <= 30
    mse = lambda x: float(np.mean((x - clean) ** 2))
    assert mse(safe) <= 1.03 * mse(plain), "the safeguard must not cost noise reduction"


def test_named_safeguard_keeps_stars_and_sparks():
    """Named test (23.12): a spark far above the noise survives the denoiser when the safeguard is on."""
    clean, noisy, model, sig = _noisy_flat()
    sparks = [(40, 60), (101, 33), (200, 180), (150, 121)]
    for y, x in sparks:
        noisy[y, x] += 40.0 * sig                           # one photosite, 40 sigma above its surroundings
    st = {}
    p = dict(method="starlet", luma=1.5, chroma=2.0)
    plain = DN.denoise_mosaic(noisy, PAT, model, DN.DenoiseParams(**p))
    safe = DN.denoise_mosaic(noisy, PAT, model, DN.DenoiseParams(**p, safeguard=4.5), stats=st)
    kept_plain = np.mean([(plain[y, x] - 0.05) / (40.0 * sig) for y, x in sparks])
    kept_safe = np.mean([(safe[y, x] - 0.05) / (40.0 * sig) for y, x in sparks])
    # the guarantee: a photosite ends within 1.5 k sigma of where it started (here 40 sigma up: at least 80% kept)
    assert kept_safe > 0.8, f"sparks must survive, kept {kept_safe:.2f}"
    assert kept_safe > kept_plain + 0.1, f"without the safeguard: {kept_plain:.2f}"
    assert st["protected"] >= 1


def test_safeguard_blends_between_k_and_one_and_a_half_k():
    orig = np.array([0.0, 3.0, 4.5, 5.8, 6.75, 10.0], np.float32)
    out, n = DN.safeguard(np.zeros_like(orig), orig, 4.5)
    assert out[0] == 0 and out[1] == 0 and out[2] == 0, "moves up to k sigma are the denoiser's business"
    assert 0 < out[3] < orig[3]
    assert out[4] == pytest.approx(6.75) and out[5] == pytest.approx(10.0), "beyond 1.5 k the original is kept"
    assert n == 3


def test_safeguard_in_the_edit_and_old_sidecars():
    from studio import engine as E
    e = E.Edit()
    assert e.denoise.safeguard and e.denoise.guard and e.stage_versions["denoise"] == 4
    assert e.denoise.auto and e.denoise.lift
    assert E.Edit.from_json(e.to_json()).denoise.safeguard
    off = E.Edit()
    off.denoise.safeguard = False
    assert off.raw_key() != e.raw_key()
    old = E.Edit.from_dict({"denoise": {"enabled": True, "luma": 0.6}, "stage_versions": {"denoise": 1}})
    assert old.denoise.enabled and not old.denoise.safeguard, "an older sidecar renders exactly as it did"
    v2 = E.Edit.from_dict({"denoise": {"enabled": True, "safeguard": True}, "stage_versions": {"denoise": 2}})
    assert v2.denoise.safeguard and not v2.denoise.guard, "written before the detail guard: renders as it did"
    v3 = E.Edit.from_dict({"denoise": {"enabled": True, "guard": True}, "stage_versions": {"denoise": 3}})
    assert v3.denoise.guard and not v3.denoise.auto and not v3.denoise.lift, "before automatic strength: as it was"
    g_off = E.Edit()
    g_off.denoise.guard = False
    assert g_off.raw_key() != e.raw_key()
    assert set(DN.METHOD_INFO) == {"hybrid", "starlet", "nlm", "mppca", "learned", "fusion", "collab"}
    assert all(i["best"] and i["weak"] and i["speed"] for i in DN.METHOD_INFO.values())


# ------------------------------------------------------------------ BUILD_PLAN_NOISE D.2, D.4, D.5, G
def _plane(amp=40.0, n=256, seed=3, noise_seed=0):
    from conftest import scene
    rng = np.random.default_rng(noise_seed)
    clean = (amp * scene(n, n, seed=seed)[..., 1]).astype(np.float32) + 10.0
    return clean, clean + rng.standard_normal(clean.shape).astype(np.float32)


def test_collaborative_baseline_beats_the_other_classical_methods():
    """The BM3D-class yardstick (raw/collab.py): on a natural-looking plane it has the lowest error."""
    clean, noisy = _plane()
    err = lambda a: float(np.mean((a - clean) ** 2))                     # noqa: E731
    collab = err(DN.denoise_collab(noisy, 1.0))
    assert collab < err(DN.denoise_starlet(noisy, 1.0, 4)) and collab < err(DN.denoise_nlm(noisy, 1.0))
    assert collab < 0.2 < err(noisy)
    flat = np.full((96, 130), 20.0, np.float32) + np.random.default_rng(1).standard_normal((96, 130)).astype(np.float32)
    out = DN.denoise_collab(flat, 1.0)
    assert out.shape == flat.shape and abs(float(out.mean()) - 20.0) < 0.05 and float(out.std()) < 0.2


def test_sure_estimates_the_error_without_the_truth_and_picks_a_good_strength():
    from studio.engine.raw import auto_strength as AS
    for amp in (8.0, 200.0):
        clean, noisy = _plane(amp, 384)
        for fn, sc in ((DN.denoise_starlet, 4), (DN.denoise_nlm, 0)):
            c = AS.choose(fn, AS.pieces(noisy, 5, 192), sc, weight=0.0)             # plain least error
            true = {s: float(np.mean((fn(noisy, s, sc) - clean) ** 2)) for s in AS.CANDIDATES}
            assert c["risk"] == pytest.approx(true[c["strength"]], abs=0.04)        # the estimate is the error
            assert true[c["strength"]] <= min(true.values()) * 1.12 + 0.01          # and the pick is near the best
    clean, noisy = _plane(8.0)
    busy, noisy_busy = _plane(200.0)
    light = AS.choose(DN.denoise_starlet, [noisy_busy], 4)["strength"]
    heavy = AS.choose(DN.denoise_starlet, [noisy], 4)["strength"]
    assert light < heavy, "a detailed photo is cleaned more lightly than a smooth one"


def test_the_share_of_fine_detail_kept_is_known_without_the_truth_and_decides_the_strength():
    """Stein's lemma on the finest band: how much real detail a cleaning keeps, estimated from the noisy photo
    alone, matches what the truth says; and weighing lost detail makes the choice lighter where there is detail."""
    from studio.engine.calib import synth as S
    from studio.engine.raw import auto_strength as AS
    from studio.lab.data import texture_kept
    leaves = np.asarray(S.dead_leaves(288, 288, seed=1), np.float32)
    leaves = leaves.mean(-1) if leaves.ndim == 3 else leaves
    clean = (leaves - leaves.mean()) * 10.0                              # texture a few times the noise
    noisy = clean + np.random.default_rng(3).standard_normal(clean.shape).astype(np.float32)
    for fn, sc in ((DN.denoise_nlm, 0), (DN.denoise_starlet, 4)):
        c = AS.choose(fn, [noisy], sc)
        for s, kept in c["kept_table"]:
            assert kept == pytest.approx(texture_kept(fn(noisy, s, sc), noisy, clean), abs=0.03), (fn.__name__, s)
        plain = AS.choose(fn, [noisy], sc, weight=0.0)
        assert c["strength"] < plain["strength"] and c["kept"] > plain["kept"] + 0.04
    flat = np.random.default_rng(4).standard_normal((192, 192)).astype(np.float32) + 30.0
    c = AS.choose(DN.denoise_nlm, [flat], 0)
    assert c["kept"] is None and c["strength"] >= 1.6, "nothing fine to keep: cleaned hard"


def test_blend_choice_for_a_cleaner_without_a_strength():
    from scipy.ndimage import uniform_filter
    from studio.engine.raw import auto_strength as AS
    clean, noisy = _plane(8.0)
    four = [np.stack([noisy] * 4)]
    blur = lambda y: np.stack([uniform_filter(p, 5) for p in y])                    # noqa: E731
    good = AS.choose_blend(blur, four, weight=0.0)                                  # plain least error
    bad = AS.choose_blend(lambda y: np.stack([uniform_filter(p, 31) + 3.0 for p in y]), four, weight=0.0)
    assert good["strength"] > 0.9 > bad["strength"] and good["risk"] < bad["risk"]  # (bad shifts brightness)
    assert AS.choose_blend(blur, four)["strength"] == good["strength"]              # nothing fine here to lose
    _, detailed = _plane(20.0)                                  # detail first: where there is fine detail,
    four = [np.stack([detailed] * 4)]                           # a blur is not taken in full
    soft = lambda y: np.stack([uniform_filter(p, 3) for p in y])                    # noqa: E731
    plain, careful = AS.choose_blend(soft, four, weight=0.0), AS.choose_blend(soft, four)
    assert careful["strength"] < plain["strength"] and careful["kept"] > plain["kept"]


def test_fusion_takes_texture_from_one_method_and_flat_areas_from_the_other():
    clean, noisy = _plane(60.0)
    clean[:, :128] = 30.0                                                # left half flat, right half detailed
    noisy = clean + np.random.default_rng(4).standard_normal(clean.shape).astype(np.float32)
    w = DN.texture_weight(noisy)
    assert float(w[32:-32, 16:100].mean()) < 0.15 and float(w[32:-32, 150:240].mean()) > 0.6
    err = lambda a, r: float(np.mean((a[r] - clean[r]) ** 2))            # noqa: E731
    flat, tex = np.s_[16:-16, 8:112], np.s_[16:-16, 144:248]
    fus, nlm, star = DN.denoise_fusion(noisy, 1.0), DN.denoise_nlm(noisy, 1.0), DN.denoise_starlet(noisy, 1.0, 4)
    assert err(fus, flat) < err(nlm, flat)                               # smoother where it is flat
    assert err(fus, tex) < err(star, tex) * 1.05                         # and no worse than wavelets on texture


def test_lifted_shadows_are_cleaned_harder_and_only_they(tmp_path):
    from studio.engine.calib.noise import NoiseModel
    rng = np.random.default_rng(6)
    a, b = 2e-4, 1e-6
    clean = np.full((192, 256), 0.004, np.float32)                      # deep shadow on the left ...
    clean[:, 128:] = 0.3                                                # ... well above mid grey on the right
    noisy = (rng.poisson(clean / a) * a + rng.normal(0, np.sqrt(b), clean.shape)).astype(np.float32)
    model = NoiseModel(a=np.full((2, 2), a), b=np.full((2, 2), b), source="calibrated", iso=3200)
    pat = np.array([[0, 1], [1, 2]])
    base = DN.DenoiseParams(method="starlet", luma=0.6, chroma=0.6)
    st = {}
    plain = DN.denoise_mosaic(noisy, pat, model, base)
    lifted = DN.denoise_mosaic(noisy, pat, model, dataclasses.replace(base, lift=3.0, lift_mid=0.11), stats=st)
    dark, bright = np.s_[16:-16, 16:112], np.s_[16:-16, 144:240]
    assert lifted[dark].std() < 0.9 * plain[dark].std()                  # the shadows are cleaner
    assert np.allclose(lifted[bright], plain[bright], atol=1e-6)         # nothing else changed
    assert st["lift"]["stops"] == 3.0 and 0.3 < st["lift"]["share"] < 0.7
    from studio import engine as E
    e = E.Edit()
    assert e.shadow_lift() == 0.0
    e.exposure_ev, e.light.shadows = 1.2, 50
    assert e.shadow_lift() == 2.0
    e.denoise.enabled = True
    k1 = e.raw_key()
    e.light.shadows = 52                                                # within the same half stop: nothing re-runs
    assert e.raw_key() == k1
    e.light.shadows = 100
    assert e.raw_key() != k1


def test_auto_strength_allows_for_noise_that_neighbours_share():
    """Some cameras filter their raws a little, so neighbouring photosites of one colour share noise (Bill's R5 at
    ISO 400: 0.10). Told the camera's figure, the estimates stay right; not told, they are off."""
    from studio.engine.calib import synth as S
    from studio.engine.raw import auto_strength as AS
    from studio.lab.data import texture_kept
    rho = 0.15
    n = AS.shaped(np.random.default_rng(8), (320, 320), rho)
    assert float(n.std()) == pytest.approx(1.0, abs=0.02)
    assert float(np.mean(n[:, :-1] * n[:, 1:])) == pytest.approx(rho, abs=0.02)
    assert float(np.mean(n[:-1] * n[1:])) == pytest.approx(rho, abs=0.02)
    assert AS.fine_noise(0.0) == AS.FINE_NOISE and AS.fine_noise(rho) < AS.FINE_NOISE - 0.1
    leaves = np.asarray(S.dead_leaves(320, 320, seed=1), np.float32)
    leaves = leaves.mean(-1) if leaves.ndim == 3 else leaves
    clean = (leaves - leaves.mean()) * 10.0
    noisy = clean + n
    fn = DN.denoise_starlet
    told, untold = AS.choose(fn, [noisy], 4, corr=rho), AS.choose(fn, [noisy], 4)
    err_told = err_untold = 0.0
    for (s, r), (_, r0), (_, k) in zip(told["table"], untold["table"], told["kept_table"]):
        f = fn(noisy, s, 4)
        true = float(np.mean((f - clean) ** 2))
        err_told, err_untold = max(err_told, abs(r - true)), max(err_untold, abs(r0 - true))
        assert k == pytest.approx(texture_kept(f, noisy, clean), abs=0.03), s
    assert err_told < 0.04 < err_untold, (err_told, err_untold)
