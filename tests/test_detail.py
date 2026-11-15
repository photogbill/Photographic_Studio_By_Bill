"""Phase 6 microcontrast (stage 16) and stage 12 capture sharpening, on synthetic data with known answers."""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

from conftest import scene, synthetic_raw
from studio import engine as E
from studio.engine.detail import microcontrast as M
from studio.engine.detail import pyramid as P
from studio.engine.detail import sharpen as S
from studio.engine.pipeline.pipeline import Image

MID = 0.11


def _rgb(y):
    return np.repeat(np.asarray(y, np.float32)[..., None], 3, 2)


# ------------------------------------------------------------------------------------------ pyramid
def test_pyramid_reconstructs_exactly():
    x = np.random.default_rng(0).random((257, 391)).astype(np.float32)
    g = P.gaussian(x, 5)
    assert np.abs(P.collapse(P.laplacian_from_gaussian(g)) - x).max() < 1e-5


# ------------------------------------------------------------------------------------------ microcontrast
def test_remap_is_monotone_at_maximum_strength():
    d = np.linspace(-3, 3, 20001).astype(np.float32)
    for toe in (0.0, 0.05, 0.3):
        r = M.remap(d, 0.0, M.A_MAX, 1.0, toe)
        assert np.all(np.diff(r) > 0), "tones must never swap order"


def test_named_no_halos_at_maximum():
    """Named test (section 17): no halos from microcontrast at maximum. A 3-stop edge between flat areas."""
    p = M.MCParams(amount=1.0, shadows=1, midtones=1, highlights=1)
    for edge in (3.0, 6.0):
        y = np.full((256, 512), MID * 2 ** -2.5, np.float32)
        y[:, 256:] = MID * 2 ** (-2.5 + edge)
        out = M.apply(_rgb(y), p, MID, None, 1.0, 6000)
        change = np.abs(np.log2(out[..., 1] / y))
        assert change.max() < 0.02, f"{edge}-stop edge changed by {change.max():.3f} stops"


def test_shades_of_shadow_are_separated():
    """Small tonal variations at medium scales (the 'variations of shadow') get stronger; pixel noise does not."""
    rng = np.random.default_rng(0)
    t = gaussian_filter(rng.standard_normal((512, 512)), 8)
    t = t / t.std() * 0.15
    y = (MID * 2 ** (-2 + t)).astype(np.float32)
    out = M.apply(_rgb(y), M.MCParams(amount=1.0), MID, None, 1.0, 6000)
    assert np.log2(out[..., 1]).std() > 1.25 * np.log2(y).std()
    n = (MID * 2 ** (-2 + rng.standard_normal((512, 512)) * 0.05)).astype(np.float32)
    outn = M.apply(_rgb(n), M.MCParams(amount=1.0), MID, None, 1.0, 6000)
    assert abs(np.log2(outn[..., 1]).std() / np.log2(n).std() - 1) < 0.03


def test_noise_aware_in_deep_shadow():
    rng = np.random.default_rng(1)
    base = MID * 2 ** -4
    sig = lambda v: np.sqrt(2e-4 * np.asarray(v) + 4e-7)          # noqa: E731
    y = gaussian_filter((base + rng.standard_normal((768, 768)) * sig(base)).astype(np.float32), 0.6)

    def band(x):
        return (gaussian_filter(x, 4) - gaussian_filter(x, 16)).std()
    blind = M.apply(_rgb(y), M.MCParams(amount=1.0), MID, None, 1.0, 6000)[..., 1]
    aware = M.apply(_rgb(y), M.MCParams(amount=1.0), MID, sig, 1.0, 6000)[..., 1]
    gain_blind, gain_aware = band(blind) / band(y) - 1, band(aware) / band(y) - 1
    assert gain_aware < 0.5 * gain_blind


def test_shadow_emphasis_by_default():
    p = M.MCParams(amount=1.0)
    assert M.tone_strength(p, -3.0) > M.tone_strength(p, 0.0) > M.tone_strength(p, 2.5)
    sh, mid, hi = M.tone_bands(np.linspace(-9, 3, 50))
    assert np.all(sh + mid + hi <= 1 + 1e-9)


def test_scale_band_follows_resolution():
    """The same features are touched at fit-to-screen and at 100%: weights depend on full-resolution size."""
    p = M.MCParams(amount=1.0)
    full = M.level_weights(p, 1.0, 6000, 12)
    quarter = M.level_weights(p, 0.25, 6000, 12)
    assert np.allclose(full[2:10], quarter[0:8])
    assert full[0] == 0 and full[1] == 0                            # pixel detail and grain are left alone


# ------------------------------------------------------------------------------------------ capture sharpening
def _blocks(n=512, up=4, seed=3):
    rng = np.random.default_rng(seed)
    hi = np.full((n * up, n * up), 0.1, np.float32)
    for _ in range(40):
        y, x = rng.integers(0, n * up - 200, 2)
        hh, ww = rng.integers(60, 300, 2)
        hi[y:y + hh, x:x + ww] = rng.uniform(0.03, 0.8)
    return hi


def _shoot(hi, sigma, n=512, up=4):
    return gaussian_filter(hi, sigma * up).reshape(n, up, n, up).mean((1, 3))


def test_blur_is_measured_from_edges():
    hi = _blocks()
    for sigma in (0.6, 1.0, 1.6):
        truth = math.hypot(sigma, 0.289)                           # plus the pixel's own width
        b = S.summarise_blur(S.edge_blur_samples(_shoot(hi, sigma)))
        assert b.confidence != "default"
        assert abs(b.sigma - truth) < 0.12, (sigma, b)


def test_deconvolution_recovers_detail_without_ringing():
    hi = _blocks()
    sharp = hi.reshape(512, 4, 512, 4).mean((1, 3))
    img = _shoot(hi, 1.2)
    b = S.summarise_blur(S.edge_blur_samples(img))
    out = S.capture_sharpen(_rgb(img), S.SharpenParams(amount=1.0, sigma=b.sigma),
                            lambda v: np.sqrt(1e-5 * np.asarray(v) + 1e-8))[..., 1]
    assert np.abs(out - sharp).mean() < 0.7 * np.abs(img - sharp).mean()
    assert S.summarise_blur(S.edge_blur_samples(out)).sigma < b.sigma - 0.2
    assert (out / maximum_filter(img, 5)).max() < 1.05            # no bright halos
    assert (out / np.maximum(minimum_filter(img, 5), 1e-6)).min() > 0.85


def test_named_candle_flame_keeps_a_clean_edge():
    """Named test: a candle flame (far over clip) on a dark background gets no dark ring from sharpening."""
    img = np.full((200, 200), 0.004, np.float32)
    img[95:105, 97:103] = 8.0
    img = gaussian_filter(img, 1.0)
    out = S.capture_sharpen(_rgb(img), S.SharpenParams(amount=1.0, sigma=1.0),
                            lambda v: np.sqrt(1e-5 * np.asarray(v) + 1e-8))[..., 1]
    assert out.min() >= 0.9 * 0.004


def test_flat_noise_is_not_sharpened():
    rng = np.random.default_rng(4)
    sig = lambda v: np.sqrt(1e-4 * np.asarray(v) + 1e-6)          # noqa: E731
    y = (0.05 + rng.standard_normal((300, 300)) * float(sig(0.05))).astype(np.float32)
    y = gaussian_filter(y, 0.6)
    out = S.capture_sharpen(_rgb(y), S.SharpenParams(amount=1.0, sigma=1.0), sig)[..., 1]
    assert out.std() < 1.1 * y.std()


def test_strips_match_one_piece():
    hi = _blocks()
    img = _shoot(hi, 1.0)
    p = S.SharpenParams(amount=0.8, sigma=1.0)
    nf = lambda v: np.sqrt(1e-5 * np.asarray(v) + 1e-8)           # noqa: E731
    a = S.capture_sharpen(_rgb(img), p, nf, rows=64)
    sk = S.target_kernel(1.0, 0.8)
    b = S._sharpen_block(_rgb(img), p, nf, sk, int(np.clip(round(6 + 8 * sk), 6, 24)))
    assert np.abs(a - b).max() < 1e-4


# ------------------------------------------------------------------------------------------ in the pipeline
def _image(h=256, w=384, **kw):
    return Image(synthetic_raw(scene(h, w), read_noise=2.0, **kw))


def test_new_stages_are_off_by_default_and_change_nothing():
    img = _image()
    e = E.Edit()
    assert not e.sharpen.enabled and e.microcontrast.amount == 0 and e.geometry.crop is None
    assert not e.lens.vignette
    a = E.render(img, e)
    d = e.to_dict()
    for k in ("geometry", "sharpen", "microcontrast"):
        d.pop(k)
    b = E.render(img, E.Edit.from_dict(d))                          # an old sidecar without the new sections
    assert np.array_equal(a, b)


def test_microcontrast_region_matches_full_frame():
    img = _image(320, 480)
    e = E.Edit()
    e.microcontrast.amount = 1.0
    full = E.render(img, e)
    part = E.render(img, e, region=(100, 80, 160, 120))
    ref = full[80:200, 100:260]
    assert np.abs(part - ref).max() < 0.01


def test_sharpening_in_pipeline_region_matches_full_frame():
    img = _image(320, 480)
    e = E.Edit()
    e.sharpen.enabled = True
    e.sharpen.amount = 1.0
    full = E.render(img, e)
    part = E.render(img, e, region=(100, 80, 160, 120))
    assert np.abs(part - full[80:200, 100:260]).max() < 1e-3
    assert img.capture_blur().sigma > 0
