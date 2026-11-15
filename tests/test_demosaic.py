"""Stage 10 demosaic: correctness on synthetic scenes with known ground truth."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import mosaic_of, psnr, scene
from studio.engine.raw import demosaic as D

PAT = np.array([[0, 1], [1, 2]])


@pytest.fixture(scope="module")
def truth():
    return scene()


@pytest.mark.parametrize("algo", ["bilinear", "mhc", "menon"])
def test_flat_colour_is_reproduced_exactly(algo):
    img = np.zeros((64, 64, 3), np.float32) + np.array([0.3, 0.5, 0.2], np.float32)
    out = D.demosaic(mosaic_of(img), PAT, algo)
    assert np.allclose(out[4:-4, 4:-4], img[4:-4, 4:-4], atol=1e-5)


def test_quality_ordering(truth):
    m = mosaic_of(truth)
    scores = {a: psnr(D.demosaic(m, PAT, a), truth) for a in ("bilinear", "mhc", "menon")}
    assert scores["bilinear"] > 28
    assert scores["mhc"] > scores["bilinear"] + 1.0
    assert scores["menon"] > scores["mhc"]


@pytest.mark.parametrize("pattern", [((0, 1), (1, 2)), ((2, 1), (1, 0)), ((1, 0), (2, 1)), ((1, 2), (0, 1))])
def test_all_four_bayer_phases(truth, pattern):
    m = mosaic_of(truth, pattern)
    assert psnr(D.demosaic(m, np.array(pattern), "menon"), truth) > 30


def test_tiled_equals_whole_frame(truth):
    m = mosaic_of(truth)
    whole = D.menon(m, PAT)
    tiled = D.demosaic(m, PAT, "menon", strip_rows=64, overlap=16, workers=3)
    assert np.array_equal(whole[16:-16], tiled[16:-16])


def test_half_size_superpixel():
    img = np.random.default_rng(3).uniform(0, 1, (32, 48, 3)).astype(np.float32)
    out = D.demosaic(mosaic_of(img), PAT, "half")
    assert out.shape == (16, 24, 3)
    assert np.allclose(out[..., 0], img[0::2, 0::2, 0])
    assert np.allclose(out[..., 1], 0.5 * (img[0::2, 1::2, 1] + img[1::2, 0::2, 1]))
    assert np.allclose(out[..., 2], img[1::2, 1::2, 2])


def test_rejects_non_bayer():
    with pytest.raises(ValueError):
        D.demosaic(np.zeros((8, 8), np.float32), np.array([[0, 1], [2, 2]]), "menon")


# ------------------------------------------------------------------------------------------ plan 23.8 (session 11)
@pytest.mark.parametrize("algo", ["rcd", "lmmse", "dual"])
def test_new_methods_reproduce_flat_colour(algo):
    img = np.zeros((64, 64, 3), np.float32) + np.array([0.3, 0.5, 0.2], np.float32)
    out = D.demosaic(mosaic_of(img), PAT, algo)
    assert np.allclose(out[6:-6, 6:-6], img[6:-6, 6:-6], atol=1e-4)


def test_rcd_beats_menon_clean_and_noisy(truth):
    m = mosaic_of(truth)
    assert psnr(D.demosaic(m, PAT, "rcd"), truth) > psnr(D.demosaic(m, PAT, "menon"), truth) + 1.0
    noisy = m + np.random.default_rng(5).normal(0, 0.01, m.shape).astype(np.float32)
    assert psnr(D.demosaic(noisy, PAT, "rcd"), truth) > psnr(D.demosaic(noisy, PAT, "menon"), truth)
    assert psnr(D.demosaic(m, PAT, "lmmse"), truth) > psnr(D.demosaic(m, PAT, "mhc"), truth) + 5.0


@pytest.mark.parametrize("algo", ["rcd", "lmmse"])
@pytest.mark.parametrize("pattern", [((0, 1), (1, 2)), ((2, 1), (1, 0)), ((1, 0), (2, 1)), ((1, 2), (0, 1))])
def test_new_methods_all_bayer_phases(truth, pattern, algo):
    assert psnr(D.demosaic(mosaic_of(truth, pattern), np.array(pattern), algo), truth) > 48


@pytest.mark.parametrize("algo", ["rcd", "lmmse", "dual"])
def test_new_methods_tiled_equal_whole_frame(truth, algo):
    m = mosaic_of(truth)
    whole = D._FUNCS[algo](m, PAT)
    tiled = D.demosaic(m, PAT, algo, strip_rows=64, overlap=16, workers=3)
    assert np.allclose(whole[16:-16], tiled[16:-16], atol=1e-6)


def test_green_equilibration_fixes_a_mismatch_and_leaves_detail():
    rng = np.random.default_rng(0)
    img = np.full((96, 96, 3), 0.3, np.float32)
    img[:, 60:] = [0.05, 0.6, 0.1]                                  # a strong edge (real detail)
    m = mosaic_of(img)
    bad = m.copy()
    bad[1::2, 0::2] *= 1.03                                         # the second green reads 3 % high
    bad += rng.normal(0, 0.0005, bad.shape).astype(np.float32)
    fixed = D.green_equilibrate(bad, PAT)
    flat = slice(8, 50)
    before = abs(bad[0:90:2, 1:50:2][flat].mean() - bad[1:90:2, 0:50:2][flat].mean())
    after = abs(fixed[0:90:2, 1:50:2][flat].mean() - fixed[1:90:2, 0:50:2][flat].mean())
    assert before > 0.008 and after < 0.002
    assert np.array_equal(fixed[0::2, 0::2], bad[0::2, 0::2]) and np.array_equal(fixed[1::2, 1::2], bad[1::2, 1::2])
    e1 = abs(fixed[0:90:2, 61:95:2].mean() - bad[0:90:2, 61:95:2].mean())      # beside the edge: hardly moved
    assert e1 < 0.02


def test_rcd_is_the_default_and_old_edits_keep_menon():
    from studio.engine.pipeline.edit import Edit
    assert Edit().demosaic == "rcd" and not Edit().green_eq
    old = Edit().to_dict()
    old["demosaic"] = "menon"
    old.pop("green_eq")
    e = Edit.from_dict(old)
    assert e.demosaic == "menon" and not e.green_eq
    assert Edit(green_eq=True).raw_key() != Edit().raw_key()
