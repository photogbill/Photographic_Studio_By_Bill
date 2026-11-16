"""Clean for scans and JPEGs (local/scanclean.py): noise measured in the picture itself, removed there."""
from __future__ import annotations

import io
import os

import numpy as np
import pytest


def _ref(name="coffee"):
    skd = pytest.importorskip("skimage.data")
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(getattr(skd, name)().astype(np.float32) / 255.0, (1.0, 1.0, 0))   # grain-free


def _jpeg(a, q=90):
    from PIL import Image as PI
    b = io.BytesIO()
    PI.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(b, "JPEG", quality=q)
    return np.asarray(PI.open(io.BytesIO(b.getvalue())).convert("RGB"), np.float32) / 255.0


def _psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a.astype(np.float64) - b) ** 2))


def test_measured_noise_and_cleaning():
    from studio.engine.local import scanclean as sc
    from studio.engine.raw.denoise import starlet_noise_sigma
    rng = np.random.default_rng(0)
    for name in ("coffee", "astronaut"):
        ref = _ref(name)
        noisy = np.clip(ref + rng.normal(0, 0.02, ref.shape).astype(np.float32), 0, 1)     # 5 levels of noise
        m = sc.measure(noisy)
        wn = starlet_noise_sigma()
        y0 = np.median(m["sigma"][0][0][2:6])
        assert 0.7 < y0 / (0.02 * wn[0]) < 1.5                      # the finest band's noise found
        for b in range(1, len(m["sigma"][0])):                       # coarse bands capped by the finest
            assert np.all(m["sigma"][0][b] <= m["sigma"][0][0] * wn[b] / wn[0] * sc.CORR + 1e-9)
        out = sc.clean(noisy, m)
        assert out.shape == noisy.shape and out.dtype == np.float32
        assert _psnr(out, ref) > _psnr(noisy, ref) + 3.0
        j = _jpeg(noisy)
        assert _psnr(sc.clean(j, sc.measure(j)), ref) > _psnr(j, ref) + 1.0                # a JPEG of it too
    clean = _ref("coffee")
    assert _psnr(sc.clean(clean, sc.measure(clean)), clean) > 38.0   # a clean picture is barely touched
    big = np.clip(np.tile(_ref("coffee"), (3, 2, 1))[:1200, :900] + rng.normal(0, 0.02, (1200, 900, 3)), 0, 1)
    big = big.astype(np.float32)
    mb = sc.measure(big)                                           # done in strips on every core: the same
    assert np.abs(sc.clean(big, mb) - sc._clean_block(big, mb, 1.0, 1.5, 0.0, True, {}, None)).max() < 1e-6
    grey = clean.mean(axis=2)
    g = sc.clean(grey + rng.normal(0, 0.02, grey.shape).astype(np.float32), sc.measure(grey))
    assert g.ndim == 2


def test_scan_clean_through_the_pipeline(tmp_path):
    from PIL import Image as PI
    from studio import engine
    rng = np.random.default_rng(3)
    ref = _ref("coffee")
    noisy = np.clip(ref + rng.normal(0, 0.03, ref.shape), 0, 1)
    p = str(tmp_path / "noisy_scan.png")
    PI.fromarray((noisy * 255 + 0.5).astype(np.uint8)).save(p)
    im = engine.open(p)
    e = engine.Edit()
    e.denoise.enabled = True
    assert e.raw_key() != engine.Edit().raw_key()
    plain, cleaned = engine.render(im, engine.Edit()), engine.render(im, e)
    assert im.scan_noise_report["levels"] > 1.0
    from scipy.ndimage import gaussian_filter
    hp = lambda a: a - gaussian_filter(a, (2, 2, 0))                 # noqa: E731
    assert hp(cleaned).std() < 0.75 * hp(plain).std()                # grain removed
    part = engine.render(im, e, region=(200, 120, 160, 120))
    assert np.abs(part - cleaned[120:240, 200:360]).mean() < 2e-3   # a 100% view matches the whole render
    e2 = engine.Edit.from_dict(e.to_dict())
    e2.denoise.luma, e2.denoise.chroma = 0.3, 0.3
    assert np.abs(engine.render(im, e2) - cleaned).mean() > 1e-4     # the strength sliders work on scans


def test_clean_panel_says_raw_is_best():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    note = page.clean_scan_note.text().lower()
    assert "best work on raw files" in note and "shoot raw" in note
    app.processEvents()
