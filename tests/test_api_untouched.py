"""Untouched sidecars follow today's rendering; edited ones keep theirs (api.load_edit, 2026-11-23)."""
from __future__ import annotations

import json
import os

from studio.engine import api
from studio.engine.pipeline.edit import STAGE_VERSIONS, Edit


def _write(tmp_path, edit_dict, library=None):
    raw = tmp_path / "x.cr2"
    raw.write_bytes(b"not a raw")
    doc = {"edit": edit_dict}
    if library:
        doc["library"] = library
    with open(str(raw) + api.SIDECAR_SUFFIX, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    return str(raw)


def test_an_untouched_sidecar_from_an_older_version_gets_todays_rendering(tmp_path):
    old = Edit().to_dict()
    old["stage_versions"] = {k: 1 for k in STAGE_VERSIONS}
    old.pop("calibrated", None)
    old["lens"]["tag"] = "Laack 13.5cm Anastigmat"
    raw = _write(tmp_path, old, {"rating": 3})
    e = api.load_edit(raw)
    assert e.stage_versions == dict(STAGE_VERSIONS)
    assert e.lens.tag == "Laack 13.5cm Anastigmat"            # a fact about the photo: kept
    assert not api.has_edit(raw)


def test_an_edited_sidecar_keeps_its_old_rendering(tmp_path):
    old = Edit().to_dict()
    old["stage_versions"] = {k: 1 for k in STAGE_VERSIONS}
    old["exposure_ev"] = 0.7
    raw = _write(tmp_path, old)
    e = api.load_edit(raw)
    assert e.stage_versions["camera_matrix"] == 1 and e.stage_versions["tone"] == 1 and e.exposure_ev == 0.7
    assert api.has_edit(raw)


def test_scrolling_the_panel_does_not_move_sliders():
    import pytest
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication, QScrollArea, QSlider, QVBoxLayout, QWidget
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    guard = M._WheelGuard(app)
    app.installEventFilter(guard)
    try:
        area = QScrollArea()
        inner = QWidget()
        lay = QVBoxLayout(inner)
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(0, 100)
        sl.setValue(50)
        lay.addWidget(sl)
        for _ in range(40):
            lay.addWidget(QSlider(Qt.Orientation.Horizontal))
        area.setWidget(inner)
        area.resize(200, 150)
        area.show()
        app.processEvents()
        area.setFocus()                                           # nothing clicked yet
        sl.clearFocus()
        ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120), Qt.MouseButton.NoButton,
                         Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(sl, ev)
        assert sl.value() == 50                                   # not clicked: the slider stays
        assert area.verticalScrollBar().value() > 0                # and the panel scrolled instead
        sl.setFocus()                                             # clicked: now the wheel may move it
        QApplication.sendEvent(sl, ev)
        assert sl.value() != 50
    finally:
        app.removeEventFilter(guard)
