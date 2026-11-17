"""Content-aware Fill (engine/local/fill.py), the Retouch "fill" kind, and the ONNX ops LaMa needs.

Synthetic scenes with a known truth: a textured ground under a textured sky with a sloping horizon. The hole is cut
across the horizon, so a good fill has to continue both the texture and the line."""
from __future__ import annotations

import numpy as np
import pytest

from studio.engine.local import fill as fl
from studio.engine.local import onnxnp
from studio.engine.local import retouch as rt


def _scene(h=160, w=240, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    horizon = 0.55 * h + 0.08 * (xx - w / 2)
    sky = np.array([0.35, 0.45, 0.70]) * (1 + 0.04 * rng.standard_normal((h, w, 1)))
    stripes = 0.5 + 0.5 * np.sin(xx / 3.0 + yy / 5.0)                     # a ground texture with structure
    ground = np.array([0.10, 0.16, 0.06]) * (0.7 + 0.6 * stripes[..., None]) * \
        (1 + 0.06 * rng.standard_normal((h, w, 1)))
    img = np.where((yy > horizon)[..., None], ground, sky).astype(np.float32)
    return img, horizon


def _hole(h, w, cy, cx, r):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.hypot(yy - cy, xx - cx) < r


def _err(a, b, m):
    return float(np.abs(fl.encode(a[m]) - fl.encode(b[m])).mean())


def test_patchmatch_continues_the_horizon_and_the_texture():
    img, _ = _scene()
    h, w = img.shape[:2]
    hole = _hole(h, w, 0.55 * h, w / 2, 22)
    broken = img.copy()
    broken[hole] = 1.0                                                     # a bright flare blob
    out, rep = fl.fill(broken, hole, method="patchmatch", seed=1)
    flat = broken.copy()
    flat[hole] = img[~hole].mean(0)
    e_fill, e_flat = _err(out, img, hole), _err(flat, img, hole)
    assert rep["method"] == "patchmatch"
    assert e_fill < 0.45 * e_flat, (e_fill, e_flat)
    # the texture is carried in, not smoothed away: the filled ground varies about as much as the real ground
    yy = np.mgrid[0:h, 0:w][0]
    g = hole & (yy > 0.55 * h + 8)
    assert fl.encode(out[g]).std(0).mean() > 0.5 * fl.encode(img[g]).std(0).mean()
    # outside the hole nothing changes
    assert np.array_equal(out[~hole], broken[~hole])


def test_fill_is_the_same_every_time_for_one_seed():
    img, _ = _scene(96, 128)
    hole = _hole(96, 128, 50, 64, 12)
    a, _ = fl.fill(img, hole, method="patchmatch", seed=7)
    b, _ = fl.fill(img, hole, method="patchmatch", seed=7)
    assert np.array_equal(a, b)


def test_a_good_plan_guides_patchmatch():
    img, _ = _scene()
    h, w = img.shape[:2]
    hole = _hole(h, w, 0.55 * h, w / 2, 26)
    enc = fl.encode(img)
    start = enc.copy()
    start[hole] = 0.5
    plain = fl.patchmatch_fill(start, hole, seed=2)
    guided = fl.patchmatch_fill(start, hole, seed=2, guide=enc)
    e_plain = float(np.abs(plain[hole] - enc[hole]).mean())
    e_guided = float(np.abs(guided[hole] - enc[hole]).mean())
    assert e_guided < e_plain, (e_guided, e_plain)


def test_fill_says_when_lama_is_missing(monkeypatch):
    monkeypatch.setattr(fl, "lama_available", lambda: False)
    img, _ = _scene(64, 96)
    hole = _hole(64, 96, 32, 48, 8)
    _, rep = fl.fill(img, hole, method="auto", seed=0)
    assert "PatchMatch" in rep["text"] and "Convert Models.bat" in rep["text"]
    assert set(fl.METHODS) == {"auto", "lama", "patchmatch"}
    for m in fl.METHODS.values():
        assert m["best"] and m["weak"]


def test_retouch_fill_spot_removes_a_blob():
    img, _ = _scene(200, 300)
    h, w = img.shape[:2]
    broken = img.copy()
    blob = _hole(h, w, 0.55 * h, 150, 14)
    broken[blob] = 2.0
    spot = {"kind": "fill", "pts": [[150 / w, 0.55]], "size": 18 / 300, "feather": 0.15, "opacity": 100,
            "src": [0.0, 0.0], "enabled": True}
    assert rt.active([spot]) and "fill" in rt.NO_SOURCE

    def fetch(x0, y0, x1, y1):
        return broken[y0:y1, x0:x1]
    res = rt.patch(spot, fetch, w, h, 1)
    assert res is not None
    out = broken.copy()
    rt.paste(out, 0, 0, 1, res)
    assert out[blob].max() < 1.0                                          # the blob is gone
    assert _err(out, img, blob) < 0.5 * _err(broken, img, blob)
    far = ~_hole(h, w, 0.55 * h, 150, 19)
    assert np.allclose(out[far], broken[far])
    # the same spot fills the same way twice (the seed comes from where it is)
    res2 = rt.patch(spot, fetch, w, h, 1)
    assert np.array_equal(res[0], res2[0]) and rt.fill_seed(spot) == rt.fill_seed(dict(spot))


def test_onnx_dft_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 16, 5)).astype(np.float32)
    real = x[..., None]
    f = onnxnp._dft([real], {"axis": 1, "onesided": 1})
    ref = np.fft.rfft(x, axis=1)
    assert np.allclose(f[..., 0], ref.real, atol=1e-4) and np.allclose(f[..., 1], ref.imag, atol=1e-4)
    back = onnxnp._dft([f, np.array([16])], {"axis": 1, "onesided": 1, "inverse": 1})
    assert np.allclose(back[..., 0], x, atol=1e-4)
    full = onnxnp._dft([real], {"axis": 1})
    assert np.allclose(full[..., 0] + 1j * full[..., 1], np.fft.fft(x, axis=1), atol=1e-4)


def test_onnx_dilated_conv_matches_direct_sum():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 2, 12, 12)).astype(np.float32)
    k = rng.standard_normal((3, 2, 3, 3)).astype(np.float32)
    d = 2
    out = onnxnp._conv(x, k, None, {"dilations": [d, d], "pads": [d, d, d, d]})
    xp = np.pad(x, ((0, 0), (0, 0), (d, d), (d, d)))
    ref = np.zeros((1, 3, 12, 12), np.float32)
    for o in range(3):
        for i in range(3):
            for j in range(3):
                ref[0, o] += (k[o, :, i, j, None, None] * xp[0, :, i * d:i * d + 12, j * d:j * d + 12]).sum(0)
    assert np.allclose(out, ref, atol=1e-4)


@pytest.mark.skipif(not fl.lama_available(), reason="LaMa not converted on this PC (Convert Models.bat)")
def test_lama_fill_continues_structure_and_keeps_the_colour():
    """With the real model: a false-colour (infrared-like pink) scene keeps its colour inside the hole (an 8-bit
    conversion once painted cyan blobs there), and the fill beats a flat one."""
    img, _ = _scene(256, 384)
    img = img[..., [2, 0, 1]] * np.array([1.6, 0.7, 1.1], np.float32)        # pink sky, purple ground
    h, w = img.shape[:2]
    hole = _hole(h, w, 0.55 * h, w / 2, 30)
    broken = img.copy()
    broken[hole] = img.max()
    out, rep = fl.fill(broken, hole, method="lama", seed=0)
    assert rep["method"] == "lama"
    flat = broken.copy()
    flat[hole] = img[~hole].mean(0)
    assert _err(out, img, hole) < 0.6 * _err(flat, img, hole)
    ratio = fl.encode(out[hole]).mean(0) / fl.encode(img[hole]).mean(0)
    assert np.all(np.abs(ratio - 1) < 0.08), ratio
