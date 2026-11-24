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


def test_brush_editor_live_preview_texture_and_saving(page):
    i = next(k for k, b in enumerate(page.brushes) if b.name == "Round brush")
    page.select_brush(i)
    ed = page.open_brush_editor(show=False)
    before = ed.render_preview().copy()
    ed.grain_cb.setCurrentIndex(ed.grain_cb.findData("rough"))
    ed._grain_chosen(ed.grain_cb.currentIndex())             # a grain with no strength is given some
    assert page.brushes[i].grain == "rough" and page.brushes[i].grain_strength > 0
    after = ed.render_preview()
    assert np.abs(after.astype(int) - before.astype(int)).max() > 30       # the preview shows the grain
    ed.sliders["size"][0].setValue(60)
    assert page.brushes[i].size == 60 and page.s_size.value() == 60        # the Brush tab follows
    ed.dyn_boxes[("opacity", "tilt")].setChecked(True)
    assert "tilt" in page.brushes[i].dynamics["opacity"]
    ed.save_as_new("Rough round")
    assert page.brushes[-1].name == "Rough round" and page.brushes[i].grain == ""   # the original is untouched
    from studio.engine.paint import brushfile
    assert "Rough round" in [b.name for b in brushfile.saved()]
    ed.sliders["size"][0].setValue(5)
    ed.revert()
    assert page.brushes[-1].size == 60
    page2 = type(page)()                                     # a new start brings the saved brush back
    assert any(b.name == "Rough round" and b.grain == "rough" for b in page2.brushes)
    page2.deleteLater()


def test_brush_list_filter_search_and_import(page, tmp_path):
    import struct
    assert len(page.brushes) >= 100
    page.brush_cat.setCurrentIndex(page.brush_cat.findData("Ink and pens"))
    shown = [page.brushes[i].name for i in range(len(page.brushes)) if not page.brush_list.item(i).isHidden()]
    assert "Dip pen (G nib)" in shown and "Mop" not in shown
    page.brush_cat.setCurrentIndex(0)
    page.brush_find.setText("charcoal")
    shown = [page.brushes[i].name for i in range(len(page.brushes)) if not page.brush_list.item(i).isHidden()]
    assert "Vine charcoal" in shown and "Highlighter" not in shown
    page.brush_find.setText("")
    tip = (np.ones((16, 16)) * 255).astype(np.uint8)
    nb = b"Square\0"
    gbr = struct.pack(">IIIII", 28 + len(nb), 2, 16, 16, 1) + b"GIMP" + struct.pack(">I", 15) + nb + tip.tobytes()
    (tmp_path / "sq.gbr").write_bytes(gbr)
    (tmp_path / "bad.abr").write_bytes(b"\0\x09")
    n = len(page.brushes)
    assert page.import_brushes([str(tmp_path / "sq.gbr"), str(tmp_path / "bad.abr")]) == 1
    assert page.brushes[-1].name == "Square" and len(page.brushes) == n + 1 and page.brush_i == n
    page.canvas.stroke_from(_line(200))
    assert page.canvas.doc.layer.tiles


def test_watercolour_dries_in_the_window_and_relief_light(page):
    c = page.canvas
    page.add_layer()
    i = next(k for k, b in enumerate(page.brushes) if b.name == "Wet round (flows and dries)")
    page.select_brush(i)
    page.set_colour((0.2, 0.3, 0.7))
    c.stroke_from(_line(200))
    lay = c.doc.layer
    assert lay.medium == "watercolour" and lay.wet is not None
    for _ in range(400):
        if lay.wet is None:
            break
        c.dry_tick()
    assert lay.wet is None and c._img[200, 300, 2] > c._img[200, 300, 0] + 30     # dried, blue on screen
    page.undo()
    assert not any(t[..., 3].any() for t in lay.tiles.values())
    page.redo()
    j = next(k for k, b in enumerate(page.brushes) if b.name == "Impasto round")
    page.select_brush(page.brushes.index(page.brushes[j]))
    page.canvas.doc.active = 0                                # back to the ordinary layer for thick paint
    c.stroke_from(_line(120))
    before = c._img[110, 300].copy()
    page.relief_cb.setChecked(True)
    assert c.light is not None and np.abs(c._img[110, 300].astype(int) - before.astype(int)).max() > 3
    ed = page.open_brush_editor(show=False)
    assert "impasto" in ed.sliders and "water" in ed.sliders


def test_select_edit_transform_mask_adjust_in_the_window(page):
    from studio.engine.paint.smoothing import Sample
    c = page.canvas
    page.set_colour((1, 0, 0))
    page.fill_sel()                                           # nothing selected: fills the whole layer
    assert c.doc.layer.read(10, 10, 1, 1)[0, 0, 3] > 0.99
    page.set_tool("rect")
    c.stroke_from([Sample(100, 100, 1, 0), Sample(200, 150, 1, 0.1), Sample(300, 200, 1, 0.2)])
    assert c.doc.selection is not None and c.doc.selection[150, 200] == 1 and c.doc.selection[50, 50] == 0
    assert "Selected" in page.sel_note.text()
    page.copy_sel()
    page.paste()
    assert len(c.doc.layers) == 2 and c.doc.layer.read(150, 150, 1, 1)[0, 0, 3] > 0.99
    page.select_none()
    page.tf["tx"].setValue(50)                               # move the pasted piece 50 px right
    page.transform_apply()
    assert c.doc.layer.read(320, 150, 1, 1)[0, 0, 3] > 0.99 and c.doc.layer.read(120, 150, 1, 1)[0, 0, 3] == 0
    page.undo()
    assert c.doc.layer.read(120, 150, 1, 1)[0, 0, 3] > 0.99
    page.add_mask()
    page.mask_cb.setChecked(True)
    page.set_tool("brush")
    page.set_colour((0, 0, 0))                               # black on the mask hides
    c.stroke_from([Sample(150 + k, 150, 1, k / 100) for k in range(60)])
    assert c.doc.layer.mask_values(170, 150, 1, 1)[0, 0] < 0.3
    page.mask_cb.setChecked(False)
    page.adj_cb.setCurrentIndex(page.adj_cb.findData("invert"))
    page.add_adjustment()
    assert c.doc.layer.is_adjust and c.doc.composite()[10, 10, 1] > 0.99   # red inverted to cyan
    page.set_tool("wand")
    c.doc.active = 0
    c.stroke_from([Sample(10, 10, 1, 0)])
    assert c.doc.selection is not None


def test_save_and_open_psd(page, tmp_path):
    page.set_colour((0, 0.5, 0))
    page.canvas.stroke_from(_line(200))
    page.add_layer()
    p = str(tmp_path / "work.psd")
    assert page._write(p)
    assert page.open_path(p)
    assert len(page.canvas.doc.layers) == 2 and page.canvas.doc.layers[0].tiles and page.canvas.doc.path == p


def test_transform_handles_special_layers_and_working_space(page, tmp_path):
    from studio.engine.paint.smoothing import Sample
    c = page.canvas
    page.set_colour((1, 0, 0))
    page.select_all()
    page.fill_sel()
    page.select_none()
    from studio.engine.paint import select
    c.set_selection(select.rect(600, 400, 100, 100, 200, 200))
    page.copy_sel()
    page.paste()
    page.select_none()
    page.start_transform("free")
    assert c.tool == "transform" and len(c._tf_handles()) == 4
    c.stroke_from([Sample(150, 150, 1, 0), Sample(200, 150, 1, 0.1), Sample(250, 150, 1, 0.2)])  # drag inside: move
    page.transform_apply()
    assert c.doc.layer.read(280, 150, 1, 1)[0, 0, 3] > 0.99 and c.doc.layer.read(120, 150, 1, 1)[0, 0, 3] == 0
    page.start_transform("perspective")
    x0, y0 = c._tf_handles()[0]
    c.stroke_from([Sample(x0, y0, 1, 0), Sample(x0 - 30, y0 - 30, 1, 0.1)])  # pull one corner out
    page.transform_apply()
    page.undo()
    page.start_transform("cage")
    for x, y in ((190, 90), (310, 90), (310, 210), (190, 210)):
        c.stroke_from([Sample(x, y, 1, 0)])
    assert len(c.tf_src) == 4
    page.transform_cancel()
    page.add_fill_layer()
    assert c.doc.layer.kind == "fill"
    page.add_filter_layer()
    assert c.doc.layer.kind == "filter"
    page.convert_space("displayp3")
    assert c.doc.space == "displayp3" and "Display P3" in page.space_lbl.text()
    page.set_colour((1, 0, 0))
    assert c.colour[0] < 0.95                                  # sRGB red in P3 numbers
    page.undo()
    assert c.doc.space == "srgb" and "sRGB" in page.space_lbl.text()   # undo brings the space back too
    c.doc.path = str(tmp_path / "w.ora")
    t = page.export({"fmt": "exr", "space": "srgb", "bits": 8, "quality": 90, "lossless": False})
    assert t and t.endswith("w_Studio.exr")
    assert page.open_path(t) and c.doc.space == "linear"
