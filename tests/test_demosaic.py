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
