"""Cognitive Adjustments: depth from one photo (Depth Anything V2 Small, local/depth.py) - the haze model, depth
masks, the depth view, and the panel."""
from __future__ import annotations

import os

import numpy as np
import pytest


def test_haze_model_adds_and_clears_by_distance():
    from studio.engine.local import depth as dp
    rng = np.random.default_rng(3)
    rgb = (rng.random((40, 60, 3)) * 0.5 + 0.05).astype(np.float32)
    D = np.tile(np.linspace(1.0, 0.0, 60, dtype=np.float32), (40, 1))     # near on the left, far on the right
    A = np.array([0.8, 0.85, 0.9], np.float32)
    h = dp.haze(rgb, D, 60.0, A)
    near, far = np.abs(h - rgb)[:, :6].mean(), np.abs(h - rgb)[:, -6:].mean()
    assert near < 0.01 and far > 0.1                                       # the distance hazed, nearby left alone
    assert np.abs(h[:, -6:] - A).mean() < np.abs(rgb[:, -6:] - A).mean()   # towards the haze colour
    back = dp.haze(h, D, -60.0, A)                                          # clearing inverts adding (where t >= .35)
    assert np.abs(back - rgb)[:, :40].mean() < 2e-3
    assert np.array_equal(dp.haze(rgb, D, 0.0, A), rgb)
    assert not dp.active(None) and not dp.active({"haze": 0}) and dp.active({"haze": -5})
    rgba = dp.colour_map(D)
    assert rgba.shape == (40, 60, 4) and rgba[0, 0, 0] > rgba[0, 0, 2] and rgba[0, -1, 2] > rgba[0, -1, 0]


@pytest.fixture(scope="module")
def astronaut_png(tmp_path_factory):
    skd = pytest.importorskip("skimage.data")
    from PIL import Image as PI
    p = str(tmp_path_factory.mktemp("depth") / "astronaut.png")
    PI.fromarray(skd.astronaut()).save(p)
    return p


def test_depth_through_the_pipeline(astronaut_png):
    from studio import engine
    from studio.engine.local import depth as dp
    if not dp.available():
        pytest.skip("depth model not installed")
    im = engine.open(astronaut_png)
    shp = engine.depth_shape(im, engine.Edit())
    assert shp["type"] == "depth" and im.depth_report["ms"] > 0
    from studio.engine.local import masks as mk
    D = mk.decode_bitmap(shp["data"], shp["w"], shp["h"]).astype(np.float32) / 255
    h, w = D.shape
    helmet, corner = D[int(.8 * h):, int(.7 * w):].mean(), D[:h // 8, int(.75 * w):].mean()
    assert helmet > corner + 0.4                         # the helmet at the front is nearer than the wall behind
    e = engine.Edit()
    e.depth = {"haze": 70.0}
    assert e.raw_key() != engine.Edit().raw_key()
    plain = engine.render(im, engine.Edit())
    whole = engine.render(im, e)
    d = np.abs(whole - plain).mean(axis=2)
    H, W = d.shape
    assert d[:H // 8, int(.75 * W):].mean() > 3 * d[int(.8 * H):, int(.7 * W):].mean()   # far hazed more than near
    part = engine.render(im, e, region=(300, 200, 160, 120))
    assert np.abs(part - whole[200:320, 300:460]).mean() < 3e-3     # a 100% view matches the whole render
    e2 = engine.Edit()                                               # a depth mask selects the near things
    near = dict(shp, lo=0.7, hi=1.0, soft=0.05)
    e2.masks = [{"name": "near", "enabled": True, "adjust": {}, "shapes": [near]}]
    a = engine.render_mask(im, e2, 0, 0.5)
    ah, aw = a.shape
    assert a[int(.8 * ah):, int(.7 * aw):].mean() > 0.6 and a[:ah // 8, int(.75 * aw):].mean() < 0.1
    v = engine.depth_view(im, engine.Edit(), 0.5)
    assert v.shape[2] == 4 and v.shape[:2] == a.shape


def test_depth_panel_round_trip():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    e = E.Edit()
    e.depth = {"haze": -40.0}
    page.controls_from_edit(e)
    page.edit = e
    assert page.edit_from_controls().depth == {"haze": -40.0}
    page.controls_from_edit(E.Edit())
    page.edit = E.Edit()
    assert page.edit_from_controls().depth is None
    assert "depth" in page.mask_new and page.KIND_NAMES["depth"] == "Depth"
    from studio.engine.presets.looks import describe_change
    assert describe_change(E.Edit(), E.Edit.from_dict({"depth": {"haze": 20}})).startswith("Depth haze")
    app.processEvents()
