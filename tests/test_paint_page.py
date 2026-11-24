"""The Paint page on synthetic pen strokes (BUILD_PLAN_PAINT P0/P1): a quick check that the window parts work."""
from __future__ import annotations

import os

import numpy as np

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def page(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setenv("STUDIO_PAINT_DIR", str(tmp_path / "paint"))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from studio.app.paint import PaintPage
    from studio.engine.paint.tiles import Document
    p = PaintPage()
    p.resize(1200, 800)
    p.canvas.set_document(Document(600, 400))
    p._set_doc(p.canvas.doc)
    yield p
    p.deleteLater()
    app.processEvents()


def _line(y=200):
    from studio.engine.paint.smoothing import Sample
    return [Sample(50 + i * 2.5, y, 0.3 + 0.7 * i / 199, i / 233) for i in range(200)]


def test_a_stroke_paints_shows_and_undoes(page):
    c = page.canvas
    page.set_colour((1, 0, 0))
    c.stroke_from(_line())
    assert c.doc.layer.tiles and c.doc.dirty
    assert c._img[200, 300, 0] > 200 and c._img[200, 300, 1] < 60       # the display copy shows red
    page.undo()
    assert not c.doc.layer.tiles and c._img[200, 300, 1] > 250           # back to white paper
    page.redo()
    assert c.doc.layer.tiles


def test_layers_eraser_and_view(page):
    c = page.canvas
    page.add_layer()
    assert len(c.doc.layers) == 2 and page.layer_list.topLevelItemCount() == 2
    c.stroke_from(_line())
    page.set_eraser(True)
    c.stroke_from(_line())
    assert c.doc.layer.read(300, 200, 1, 1)[0, 0, 3] < 0.05
    page.undo(), page.undo(), page.undo()
    assert len(c.doc.layers) == 1
    c.rotate_by(15), c.toggle_mirror(), c.set_zoom(2.0)
    p = c.to_doc(c.transform().map(c.to_doc(c.rect().center())))       # the view maps both ways
    q = c.to_doc(c.rect().center())
    assert abs(p.x() - q.x()) < 1e-6 and abs(p.y() - q.y()) < 1e-6


def test_tablet_check_reads_samples(page):
    for i in range(50):
        page._sample({"kind": "tablet", "pressure": i / 50, "tilt_x": 0, "tilt_y": 0, "rotation": 0,
                      "eraser": False, "t": i / 233, "device": "HUION Huion Tablet_HS64 Pen", "x": i, "y": 0})
    page.tablet._update_diag()
    assert "Pressure arrives" in page.tablet.diag.text()
    assert "HS64" in page.tablet.info.text()


def test_save_open_and_autosave(page, tmp_path):
    c = page.canvas
    c.stroke_from(_line())
    p = str(tmp_path / "mine.ora")
    assert page._write(p) and not c.doc.dirty
    from studio.engine.paint import ora
    assert ora.load(p).layers[0].tiles
    page.saver.changed()
    page.saver.last = 0
    page._autosave()
    from studio.engine.paint import autosave
    assert autosave.recoverable()


def test_fill_and_gradient_tools(page):
    from studio.engine.paint.smoothing import Sample
    c = page.canvas
    page.set_tool("fill")
    page.set_colour((0, 0, 1))
    c.stroke_from([Sample(300, 200, 1, 0)])
    assert c.doc.layer.read(10, 10, 1, 1)[0, 0, 2] > 0.99      # the empty canvas filled blue
    page.add_layer()
    page.set_tool("gradient")
    c.stroke_from([Sample(0, 200, 1, 0), Sample(600, 200, 1, 0.1)])
    assert c.doc.layer.read(5, 200, 1, 1)[0, 0, 3] > 0.99
    page.undo()
    assert not c.doc.layer.tiles


def test_layer_groups_in_the_panel(page):
    c = page.canvas
    page.add_layer()
    page.group_layer()                                     # the new layer goes into a group
    doc = c.doc
    assert doc.layer.is_group and page.layer_list.topLevelItemCount() == 2
    top = page.layer_list.topLevelItem(0)
    assert top.childCount() == 1 and top.isExpanded()
    assert page.l_blend.currentData() == "pass"
    page.move_layer(-1)                                    # the group (closed or not) moves below Layer 1
    assert doc.layers[-1].name == "Layer 1"
    page.undo()
    assert doc.layers[-1].is_group
    page.ungroup_layer()
    assert not any(l.is_group for l in doc.layers) and len(doc.layers) == 2


def test_pickers_and_swatches(page):
    page.picker_cb.setCurrentIndex(1)                       # triangle
    page.set_colour((0.8, 0.3, 0.2))
    assert np.allclose(page.pickers[1].rgb(), (0.8, 0.3, 0.2), atol=0.01)
    page.picker_cb.setCurrentIndex(2)                       # wide gamut
    assert np.allclose(page.pickers[2].rgb(), (0.8, 0.3, 0.2), atol=0.01)
    page.pickers[2].update()
    page.pickers[2].grab()                                  # paints the OKLCH plane
    n = len(page._swatch_sets)
    page._swatch_sets.append(__import__("studio.engine.paint.swatches", fromlist=["x"]).SwatchSet("Mine"))
    page._swatch_i = n
    page._swatch_add()
    page._swatch_save()
    assert page._swatch_sets[n].path and page._swatch_sets[n].path.endswith("Mine.gpl")
    page.sw_grid.picked.emit((0.1, 0.2, 0.3))
    assert page.canvas.colour == (0.1, 0.2, 0.3)


def test_tablet_key_mapping(page):
    page.set_key("undo", "F13")                              # as if a tablet key sent F13
    assert page.keys["undo"] == ["Ctrl+Z", "F13"] and page._key_btns["undo"].text() == "Ctrl+Z,  F13"
    page.canvas.doc.add_layer()
    page.canvas.undo.layers_changing("x")
    page.canvas.undo.layers_changed()
    n = page.canvas.undo.current
    sc = [sc for sc in page._shortcuts if sc.key().toString() == "F13"]
    assert len(sc) == 1
    sc[0].activated.emit()
    assert page.canvas.undo.current is not n                  # F13 undid the step
    page._reset_keys()
    assert page.keys["undo"] == ["Ctrl+Z"]


def test_export_webp_from_the_page(page, tmp_path):
    page.canvas.doc.path = str(tmp_path / "Lake.ora")
    t = page.export({"fmt": "webp", "space": "displayp3", "bits": 8, "quality": 90, "lossless": False})
    assert t and t.endswith("Lake_Studio.webp") and os.path.isfile(t)
    t2 = page.export({"fmt": "webp", "space": "srgb", "bits": 8, "quality": 90, "lossless": False})
    assert t2 == t                                          # our own earlier export is replaced, nothing else


def test_side_by_side_branches(page):
    from studio.engine.paint.brush import paint_stroke
    c = page.canvas
    u = c.undo
    paint_stroke(c.doc.layer, page.brushes[0], _line(100), colour=(1, 0, 0), undo=u)     # idea A: red
    u.undo()
    paint_stroke(c.doc.layer, page.brushes[0], _line(300), colour=(0, 0, 1), undo=u)     # idea B: blue
    here = u.current
    before = {k: t.copy() for k, t in c.doc.layer.tiles.items()}
    dlg = page.compare_steps(None, show=False)
    left, right = dlg._images
    assert u.current is here                                   # the painting was not changed
    assert set(before) == set(c.doc.layer.tiles)
    assert all(np.array_equal(before[k], c.doc.layer.tiles[k]) for k in before)
    sy, sx = left.shape[0] / 400, left.shape[1] // 2
    assert left[int(300 * sy), sx, 2] > 0.5 > left[int(300 * sy), sx, 0]        # where I am: blue
    assert right[int(100 * sy), sx, 0] > 0.5 > right[int(100 * sy), sx, 2]      # the other branch: red


def test_smudge_brush_in_the_window(page):
    c = page.canvas
    page.set_colour((1, 0, 0))
    c.stroke_from(_line(200))                                # a red stroke
    i = next(k for k, b in enumerate(page.brushes) if b.mode == "smudge" and b.smudge_kind == "smear")
    page.select_brush(i)
    assert not page.smudge_box.isHidden() and page.sm_kind.currentData() == "smear"
    page.sm_all.setChecked(True)
    assert page.brushes[i].smudge_all
    from studio.engine.paint.smoothing import Sample
    down = [Sample(300, 190 + k * 0.5, 1.0, k / 200) for k in range(80)]   # drag across the red line
    c.stroke_from(down)
    assert c.doc.layer.read(300, 225, 1, 1)[0, 0, 3] > 0.05                # red pulled below the line
    page.select_brush(0)
    assert page.smudge_box.isHidden()


def test_mypaint_brush_loads_and_paints(page, tmp_path):
    import math
    from studio.engine.paint.mypaint import MyPaintBrush
    b = MyPaintBrush("x")
    b.base.update(radius_logarithmic=math.log(6), opaque=1.0, hardness=0.8, dabs_per_actual_radius=3.0)
    b.comment = "a test charcoal"
    p = tmp_path / "Charcoal_test.myb"
    p.write_text(b.to_myb(), encoding="utf-8")
    bad = tmp_path / "Old.myb"
    bad.write_text("version 2\nradius_logarithmic 2.0", encoding="utf-8")
    assert page.load_mypaint([str(p), str(bad)]) == 1
    br = page.brushes[page.brush_i]
    assert br.mode == "mypaint" and br.name == "Charcoal test" and page.smudge_box.isHidden()
    page.set_colour((0, 0, 1))
    page.canvas.stroke_from(_line(200))
    c = page.canvas.doc.layer.read(300, 200, 1, 1)[0, 0]
    assert c[3] > 0.5 and c[2] > c[0]
    assert page._saved_mypaint()                               # kept beside the app for next time
