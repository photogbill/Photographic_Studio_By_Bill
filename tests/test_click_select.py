"""Click to select with SAM 2.1 (local/sam.py): the runner matches the model, Object masks and Colour a part use it,
the size setting, and Convert Models."""
from __future__ import annotations

import os

import numpy as np
import pytest


def test_runner_takes_named_inputs_and_checks_shapes():
    from studio.engine.local.onnxnp import Graph, GraphError
    g = Graph({"nodes": [{"op": "Add", "in": ["a", "b"], "out": ["y"], "attr": {}, "free": []}],
               "inputs": ["a", "b"], "outputs": ["y"], "input_shape": [1, 3],
               "input_shapes": {"a": [1, 3], "b": [0, 3]}, "qaxis": {}}, {})
    y = g(a=np.ones((1, 3)), b=np.full((2, 3), 2.0))["y"]          # 0 = any size on that axis
    assert y.shape == (2, 3) and float(y.max()) == 3.0
    with pytest.raises(GraphError):
        g(a=np.ones((1, 4)), b=np.ones((1, 3)))
    with pytest.raises(GraphError):
        g(a=np.ones((1, 3)))


def test_size_choice():
    from studio.engine.local import sam
    have = sam.installed()
    assert sam.choose("classic") is None
    assert sam.choose(None) == (have[0] if have else None)
    assert sam.choose("large") == ("large" if "large" in have else (have[0] if have else None))


@pytest.fixture(scope="module")
def astronaut_png(tmp_path_factory):
    skd = pytest.importorskip("skimage.data")
    from PIL import Image as PI
    p = str(tmp_path_factory.mktemp("sam") / "astronaut.png")
    PI.fromarray(skd.astronaut()).save(p)
    return p


def test_sam_outlines_what_is_clicked(astronaut_png):
    from studio import engine
    from studio.engine.local import masks as mk
    from studio.engine.local import sam
    if not sam.available("tiny"):
        pytest.skip("SAM 2.1 Tiny not installed")
    im = engine.open(astronaut_png)
    face = engine.object_mask(im, engine.Edit(), [(0.45, 0.22)], [], model="tiny")     # one click on the face
    assert face["model"] == "sam2.1-tiny" and im.sam_report["size"] == "tiny"
    a = mk.decode_bitmap(face["data"], face["w"], face["h"]) > 127
    h, w = a.shape
    assert a[int(.20 * h), int(.45 * w)] and 0.015 < a.mean() < 0.08     # the face, not the suit or the wall
    assert not a[int(.7 * h), int(.3 * w)] and not a[int(.05 * h), int(.9 * w)]
    both = engine.object_mask(im, engine.Edit(), [(0.45, 0.22), (0.3, 0.7)], [], model="tiny")
    b = mk.decode_bitmap(both["data"], both["w"], both["h"]) > 127
    assert b.mean() > 3 * a.mean() and b[int(.7 * h), int(.3 * w)]       # a second click adds the suit
    e = engine.Edit()
    e.masks = [{"name": "face", "enabled": True, "adjust": {}, "shapes": [face]}]
    m = engine.render_mask(im, e, 0, 0.5)
    mh, mw = m.shape
    assert m[int(.2 * mh), int(.45 * mw)] > 0.9 and m[int(.05 * mh), int(.9 * mw)] < 0.05
    hint = engine.colour_hint_area(im, engine.Edit(), [(0.3, 0.7)], [], model="tiny")   # Colour a part: the suit
    hb = mk.decode_bitmap(hint["data"], hint["w"], hint["h"]) > 127
    assert hint["model"] == "sam2.1-tiny" and hb[int(.7 * h), int(.3 * w)] and not hb[int(.2 * h), int(.45 * w)]
    classic = engine.object_mask(im, engine.Edit(), [(0.45, 0.22)], [], model="classic")
    assert "model" not in classic                                           # the Studio's own method still works


def test_convert_models_finds_downloads(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("convert_models", os.path.join(
        os.path.dirname(__file__), "..", "tools", "convert_models.py"))
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    for f in ("model.onnx", "sam2.1_hiera_tiny_20260221.zip", "sam2.1_hiera_large_20260221.zip",
              "ddcolor-onnx-float.zip", "global_checkpoints.zip"):
        (tmp_path / f).write_bytes(b"x")
    monkeypatch.setattr(cm, "MODELS", str(tmp_path))
    monkeypatch.setattr(cm, "APP", str(tmp_path / "nowhere"))
    found = [(k, os.path.basename(p), s) for k, p, s in cm.find_downloads()]
    assert ("depth", "model.onnx", None) in found
    assert ("sam", "sam2.1_hiera_tiny_20260221.zip", "tiny") in found
    assert ("sam", "sam2.1_hiera_large_20260221.zip", "large") in found
    assert len(found) == 3                                                  # other downloads are left alone


def test_click_select_setting_in_the_panel():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    from studio.engine.local import sam
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    cb = page.sel_model
    datas = [cb.itemData(i) for i in range(cb.count())]
    assert datas == ["tiny", "large", "classic"]
    for i, d in enumerate(datas[:2]):                     # sizes that are not installed are shown but greyed out
        assert cb.model().item(i).isEnabled() == (d in sam.installed())
    assert page._sel_model() == sam.choose(cb.currentData())
    app.processEvents()
