"""The Panorama page's Flat mosaic (BUILD_PLAN_MOSAIC.md M5), offscreen: Auto chooses the mosaic for Focus Stack
results, Find layout shows the views from above, Preview shows every view, Build writes the game surfaces."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))

from conftest import ROOT  # noqa: E402
import mosaic_synth as S  # noqa: E402
from test_stack_pages import _app, _wait  # noqa: E402

OUT = ROOT / "outputs" / "pytest_mosaic_page"


def _results():
    """Synthetic Focus Stack results (a picture and its NAME.depth.npz) of a road walked in two rows."""
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    g = S.Ground(x1=3.0, y1=1.4, seed=12, relief=S.bumps([(0.5, 0.2), (1.0, 0.35)], h=0.03, r=0.05))
    views, _ = S.strip(g, n_per_row=3, rows=2, length=1.0, seed=31)
    paths = []
    for v in views:
        p = OUT / f"ROAD_{v.name}_Focus_Studio.png"
        Image.fromarray(v.colour).save(p)
        np.savez_compressed(os.path.splitext(p)[0] + ".depth.npz", depth=np.asarray(v.t, np.float16),
                            confidence=np.asarray(v.conf, np.float16), label=np.full(v.t.shape, 2, np.uint8))
        paths.append(str(p))
    return paths


def test_mosaic_page_end_to_end():
    app, M = _app()
    paths = _results()
    page = M.PanoramaPage(M.Settings())
    page.resize(1300, 800)
    page.show()
    page.out_dir = str(OUT / "out")
    assert page.kind.key() == "auto"
    page.set_frames(paths)
    page.find_layout()
    _wait(app, page, 600)
    assert page.kind_key() == "mosaic", page.jobbar.text.text()
    assert "Flat mosaic" in page.kind_chip.text()
    assert page.m is not None and all(page.m["sol"].placed), page.jobbar.text.text()
    assert page.view_pick.currentText() == "Layout" and page.result.result is not None
    assert page.m_method.isVisibleTo(page) and not page.proj_group.isVisibleTo(page)
    page.m_res.combo.setCurrentIndex(list(page.m_res.info).index("full"))
    page.preview()
    _wait(app, page, 600)
    assert page.mlast is not None, page.jobbar.text.text()
    for v in range(page.view_pick.count()):
        page.view_pick.setCurrentIndex(v)
        app.processEvents()
        assert page.result.result is not None, page.view_pick.currentText()
    page.build_full()
    _wait(app, page, 900)
    assert page.last_result and page.last_result.endswith("_colour_Studio.png"), page.jobbar.text.text()
    base = page.last_result[:-len("_colour_Studio.png")]
    for kind in ("height", "normal", "coverage", "misfit"):
        assert os.path.exists(f"{base}_{kind}_Studio.png"), kind
    assert os.path.exists(f"{base}_surface_Studio.json")
    page.close()


def test_kind_switch_shows_the_right_settings():
    app, M = _app()
    page = M.PanoramaPage(M.Settings())
    page.resize(1100, 700)
    page.show()
    page.kind.combo.setCurrentIndex(list(page.kind.info).index("panorama"))
    app.processEvents()
    assert not page.m_method.isVisibleTo(page) and page.proj_group.isVisibleTo(page)
    page.kind.combo.setCurrentIndex(list(page.kind.info).index("mosaic"))
    app.processEvents()
    assert page.m_method.isVisibleTo(page) and not page.proj_group.isVisibleTo(page)
    assert page.view_pick.findText("Height") >= 0
    page.close()
