"""JPEG repair inside the file's quantisation intervals (local/jpegfix.py)."""
from __future__ import annotations

import os

import numpy as np
import pytest


def _photo():
    skd = pytest.importorskip("skimage.data")
    return skd.astronaut()


def _save(a, path, quality):
    from PIL import Image as PI
    PI.fromarray(a).save(path, "JPEG", quality=quality)
    return path


def test_repair_improves_and_stays_inside_the_file(tmp_path):
    from PIL import Image as PI
    from studio.engine.local import jpegfix as jf
    orig = _photo()
    psnr = lambda a: 10 * np.log10(255 ** 2 / np.mean((a - orig.astype(float)) ** 2))   # noqa: E731
    Y = lambda a: np.asarray(a, float) @ [0.299, 0.587, 0.114]                         # noqa: E731
    for qual in (15, 40):
        p = _save(orig, str(tmp_path / f"q{qual}.jpg"), qual)
        inf = jf.info(p)
        assert inf and abs(inf["quality"] - qual) <= 2
        dec = np.asarray(PI.open(p).convert("RGB"), float)
        fixed, rep = jf.repair_file(p)
        fx = fixed.astype(float) / 257.0
        assert fixed.dtype == np.uint16 and fixed.shape == orig.shape and rep["quality"] == inf["quality"]
        assert psnr(fx) > psnr(dec) + 0.2                         # closer to the original than the decoder
        assert jf.blockiness(Y(fx)) < jf.blockiness(Y(dec)) - 0.15
        y = np.asarray(PI.open(p).convert("YCbCr"), float)[..., 0]
        assert jf.consistent(jf.repair_plane(y, inf["q"][0]), inf["q"][0], y) > 0.9995   # compresses to the file
    grey = str(tmp_path / "grey.jpg")
    PI.fromarray(orig.mean(axis=2).astype(np.uint8)).save(grey, "JPEG", quality=20)
    g, _ = jf.repair_file(grey)
    assert g.ndim == 2 and g.shape == orig.shape[:2]
    png = str(tmp_path / "x.png")
    PI.fromarray(orig).save(png)
    assert jf.info(png) is None


def test_jpeg_repair_through_the_pipeline(tmp_path):
    from PIL import Image as PI
    from studio import engine
    p = _save(_photo(), str(tmp_path / "scan.jpg"), 20)
    im = engine.open(p)
    assert engine.jpeg_repairable(im)
    e = engine.Edit()
    e.jpeg = {"on": True}
    assert e.raw_key() != engine.Edit().raw_key()
    plain, fixed = engine.render(im, engine.Edit()), engine.render(im, e)
    assert im.jpeg_report["quality"] > 0
    assert 1e-4 < np.abs(fixed - plain).mean() < 0.02            # a gentle change
    part = engine.render(im, e, region=(160, 128, 160, 120))
    assert np.abs(part - fixed[128:248, 160:320]).mean() < 2e-3
    png = str(tmp_path / "scan.png")
    PI.fromarray(_photo()).save(png)
    im2 = engine.open(png)
    assert not engine.jpeg_repairable(im2)
    assert np.array_equal(engine.render(im2, e), engine.render(im2, engine.Edit()))   # PNG: nothing to repair
    from studio.engine.presets.looks import describe_change
    assert describe_change(engine.Edit(), engine.Edit.from_dict({"jpeg": {"on": True}})).startswith("JPEG repair")


def test_jpeg_repair_panel():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    e = E.Edit()
    e.jpeg = {"on": True}
    page.controls_from_edit(e)
    page.edit = e
    assert page.edit_from_controls().jpeg == {"on": True}
    page.controls_from_edit(E.Edit())
    page.edit = E.Edit()
    assert page.edit_from_controls().jpeg is None and not page.jp_on.isVisibleTo(page)   # no JPEG open: hidden
    app.processEvents()
