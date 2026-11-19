"""Clean in Develop (BUILD_PLAN_NOISE I): Auto strength, lifted shadows, the camera's calibration, the new
methods and the "What Clean removed" view. Qt runs off-screen; the photo is synthetic with known noise."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import scene, synthetic_raw


def _noisy_image(strength=60.0, seed=3, level=0.25):
    from studio.engine.pipeline.pipeline import Image
    rgb = np.clip(scene(640, 800, seed=seed) * level + 0.02, 0.0, 1.0).astype(np.float32)
    raw = synthetic_raw(rgb, read_noise=strength, seed=seed)
    raw.meta.iso = 6400.0
    return Image(raw)


def _page():
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    M.Settings.set = lambda self, k, v: self.d.__setitem__(k, v)
    page = M.DevelopPage(M.Settings())
    page.schedule = lambda: None
    page.render_now = lambda: None
    return app, M, page


def test_clean_panel_auto_lift_calibration_and_methods_round_trip():
    from studio import engine as E
    from studio.engine.raw.denoise import METHOD_INFO
    app, M, page = _page()
    methods = [page.dn_m.itemText(i) for i in range(page.dn_m.count())]
    assert set(methods) == set(METHOD_INFO) and methods[0] == "hybrid"
    for m in methods:                                           # every method says what it is best for and weak at
        page.dn_m.setCurrentText(m)
        assert "Best for" in page.dn_info.text() and "Weak at" in page.dn_info.text()
    for w in (page.dn_auto, page.dn_lift, page.dn_cal):         # each choice explains itself the same way
        assert "Best for" in w.toolTip() and "Weak at" in w.toolTip()
    assert page.dn_auto.isChecked() and not page.dn_l.isEnabled() and not page.dn_c.isEnabled()
    e = E.Edit()
    e.denoise.enabled, e.denoise.method = True, "fusion"
    e.denoise.auto, e.denoise.lift, e.calibrated, e.denoise.luma, e.denoise.chroma = False, False, False, 0.8, 2.1
    page.controls_from_edit(e)
    page.edit = e
    assert not page.dn_auto.isChecked() and page.dn_l.isEnabled() and not page.dn_lift.isChecked()
    assert not page.dn_cal.isChecked() and page.edit_from_controls().to_dict() == e.to_dict()
    e2 = E.Edit()
    e2.denoise.enabled, e2.denoise.lean, e2.denoise.luma = True, 1.3, 0.6
    page.controls_from_edit(e2)
    page.edit = e2
    assert page.dn_auto.isChecked() and page.edit_from_controls().to_dict() == e2.to_dict()
    # once Auto has chosen for the open photo, Strength is the lean and the sliders only show the choice
    img = type("I", (), {})()
    img.raw = type("R", (), {"is_scan": False})()
    img.meta = type("M", (), {"lens": "", "iso": 100, "iso_label": "ISO 100", "manual_lens": True, "fnumber": None})()
    img.scan_noise_report, img.calibration_report = None, {}
    img.auto_report = dict(luma=0.85, chroma=1.6, kept=0.96, kind="strength", note="")
    img.denoise_report = dict(protected=0, sites=1, luma=round(0.85 * 1.3, 3), chroma=round(1.6 * 1.3, 3), auto=True)
    page.image = img
    page._dn_safe_refresh()
    assert page.dn_s.value() == pytest.approx(1.3) and not page.dn_l.isEnabled()
    assert "leaned to 130%" in page.dn_auto_chip.text()
    page.dn_s.set_value(0.8, emit=False)
    got = page.edit_from_controls()
    assert got.denoise.lean == pytest.approx(0.8) and got.denoise.luma == pytest.approx(0.6), "luma is not rewritten"
    # Auto ticked, but it could not choose for this photo (an uncalibrated camera): the sliders must work
    img.auto_report = dict(note="Auto needs this camera's noise to be measured.")
    img.denoise_report = dict(protected=0, sites=1, luma=0.6, chroma=1.5, auto=False)
    page.controls_from_edit(e2)
    page._dn_safe_refresh()
    assert page.dn_l.isEnabled() and page.dn_c.isEnabled() and "noise to be measured" in page.dn_auto_chip.text()
    page.dn_l.set_value(1.4, emit=False)
    page.dn_s.set_value(1.5, emit=False)
    got = page.edit_from_controls()
    assert got.denoise.luma == pytest.approx(2.1) and got.denoise.lean == pytest.approx(1.3), "the sliders decide"
    app.processEvents()


def test_auto_and_calibration_say_what_they_did_in_colour():
    from studio.app import theme as T
    app, M, page = _page()
    img = type("I", (), {})()
    img.raw = type("R", (), {"is_scan": False})()
    img.denoise_report = dict(protected=3, sites=1000, luma=0.85, chroma=1.6, auto=True)
    img.auto_report = dict(luma=0.85, chroma=1.6, kept=0.962, kind="strength", note="")
    img.calibration_report = dict(used=["black", "fixed pattern", "hot pixels"], black=[0.2, 0.2, -0.3, -0.4],
                                  rows_dn=0.81, cols_dn=0.12, hot=37, warm=112, mapped=4106)
    img.scan_noise_report = None
    page.image = img
    page.edit.denoise.enabled = True
    page._dn_safe_refresh()
    t = page.dn_auto_chip.text()
    assert "luminance 0.85" in t and "colour 1.60" in t and "96% of the finest detail" in t and "estimated" in t
    assert T.AMBER in page.dn_auto_chip.styleSheet()            # estimated: amber
    assert page.dn_l.value() == pytest.approx(0.85) and page.dn_c.value() == pytest.approx(1.6)
    c = page.dn_cal_chip.text()
    assert "37 hot and 112 warm" in c and "4,106" in c and "fixed banding" in c and "black level" in c
    assert T.CYAN in page.dn_cal_chip.styleSheet()              # measured on this body: cyan
    img.auto_report = dict(luma=1.0, chroma=1.6, kept=0.93, kind="referee", noise_ratio=1.02, note="")
    page._dn_auto_refresh()
    assert "two halves" in page.dn_auto_chip.text() and "within 2%" in page.dn_auto_chip.text()
    assert T.CYAN in page.dn_auto_chip.styleSheet()             # measured by the two halves: cyan
    img.auto_report = dict(note="Automatic strength is not used at ISO 100: neighbouring photosites share noise.")
    page._dn_auto_refresh()
    assert "not used at ISO 100" in page.dn_auto_chip.text() and page.dn_l.isEnabled()
    # Auto off: the sliders start from what the photo chose
    img.auto_report = dict(luma=0.85, chroma=1.6, kept=0.962, kind="strength", note="")
    page._dn_auto_refresh()
    page.dn_auto.setChecked(False)
    assert page.dn_l.isEnabled() and page.dn_l.value() == pytest.approx(0.85)
    e = page.edit_from_controls()
    assert not e.denoise.auto and e.denoise.luma == pytest.approx(0.85) and e.denoise.chroma == pytest.approx(1.6)
    img.calibration_report = {}
    page._dn_cal_refresh()
    assert "Cameras page" in page.dn_cal_chip.text()
    app.processEvents()


def test_what_clean_removed_shows_static_for_noise_and_marks_detail():
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import removed as RV
    from studio.app import theme as T
    from studio.engine.pipeline.pipeline import clean_removed, removed_boxes
    app = QApplication.instance() or QApplication([])
    image = _noisy_image()
    edit = E.Edit()
    edit.denoise.enabled = True
    boxes = removed_boxes(image)
    assert len(boxes) == 5 and all(b[2] == 512 and b[3] == 512 for _, b in boxes)
    res = clean_removed(image, edit, boxes[0][1])
    n = res["numbers"]
    assert res["removed"].shape == (256, 256) and 0.3 < n["strength"] < 1.1 and n["not_noise"] < 0.03
    before, after, rem = RV.panes(res)
    assert before.shape == (192, 192, 3) and rem.dtype == np.uint8 and abs(float(rem.mean()) - 128) < 12
    assert float(after.astype(float).std()) <= float(before.astype(float).std())
    # cleaned far too hard, with the guard off: picture goes out with the noise, and the view says so
    hard = E.Edit()
    hard.denoise.enabled, hard.denoise.auto, hard.denoise.method = True, False, "starlet"
    hard.denoise.luma, hard.denoise.guard, hard.denoise.safeguard = 4.0, False, False
    busy = _noisy_image(strength=25.0, level=0.6)
    worst = max((clean_removed(busy, hard, b)["numbers"] for _, b in removed_boxes(busy)),
                key=lambda v: v["not_noise"])
    assert worst["not_noise"] > 0.03
    col, text = RV.verdict(worst)
    assert col == T.RED and "Detail was removed" in text and "detail guard is off" in text
    col, text = RV.verdict(dict(n, not_noise=0.001))
    assert col == T.GREEN and "Only noise" in text
    dlg = RV.RemovedDialog(image, edit, sync=True)
    assert "television static" in RV.WHAT and all(v.pixmap() is not None and not v.pixmap().isNull() for v in dlg.views)
    assert "Clean ran with hybrid" in dlg.verdict.text()
    dlg.no_guard.setChecked(True)                               # the same piece without the guard
    assert (0, True) in dlg.results and dlg.results[(0, True)]["guard"] is False
    dlg.where.setCurrentIndex(2)
    assert (2, True) in dlg.results
    dlg.close()
    app.processEvents()
