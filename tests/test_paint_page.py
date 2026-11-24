"""The Paint page on synthetic pen strokes (BUILD_PLAN_PAINT P0/P1): a quick check that the window parts work."""
from __future__ import annotations

import os

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
    assert len(c.doc.layers) == 2 and page.layer_list.count() == 2
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
