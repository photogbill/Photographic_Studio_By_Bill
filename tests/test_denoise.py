"""Noise model, VST with exact unbiased inverse, and the starlet denoiser (section 8), on synthetic data."""
from __future__ import annotations

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
