"""The Infrared tab (BUILD_PLAN_PANO_IR.md 1.6, 1.8), offscreen: it builds and fits, opens Bill's 850 nm frame in
black and white with the guess and the meter, keeps every change in the photo's own edit, and stays in step with
Develop (Detail by scale, the IR white-balance chip). Real raws are copied into outputs/ first: tests never write
beside Bill's photos."""
from __future__ import annotations

import os
import shutil
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from conftest import ROOT  # noqa: E402

OUT = ROOT / "outputs" / "pytest_infrared_page"
REAL = ROOT / "test_raws" / "infrared_850" / "IMG_3156.CR2"


def _app():
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    from studio.app import theme as T
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(T.QSS)
    M.Settings.set = lambda self, k, v: self.d.__setitem__(k, v)
    return app, M


def _until(app, cond, timeout=180):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(10):
        app.processEvents()
    return cond()


def test_infrared_page_builds_empty_and_fits():
    app, M = _app()
    page = M.InfraredPage(M.Settings())
    page.resize(1040, 600)
    page.show()
    app.processEvents()
    assert page.centre.currentIndex() == 0
    sc = page.panel_scroll
    assert sc.widget().minimumSizeHint().width() <= sc.viewport().width()
    assert page.minimumSizeHint().height() <= 600
    assert not page.open_b.isEnabled()                     # nothing to open yet
    page.close()


def test_detail_by_scale_here_and_in_develop_change_the_same_edit():
    app, M = _app()
    from studio import engine as E
    from studio.app.infrared import DetailByScaleBox
    box = DetailByScaleBox()
    e = E.Edit()
    box.load(e)
    box.amount.set_value(70)                                # Simple level: Amount 70 on the Tonal side
    box.rows["veil"][1].set_value(25)
    box.store(e)
    dev = M.DevelopPage(M.Settings())
    dev.render_now = lambda: None
    dev.schedule = lambda: None
    dev.controls_from_edit(e)
    dev.edit = e
    assert abs(dev.ds_amount.value() - 70) < 0.6 and abs(dev.ds_rows["veil"][1].value() - 25) < 1e-6
    back = dev.edit_from_controls()
    assert back.detail == e.detail and back.microcontrast == e.microcontrast
    # and the other way: Develop's settings show here
    dev.ds_rows["crisp"][1].set_value(40)
    e2 = dev.edit_from_controls()
    box.load(e2)
    assert abs(box.rows["crisp"][1].value() - 40) < 1e-6


def test_develop_shows_the_infrared_white_balance_chip():
    app, M = _app()
    from studio import engine as E
    from studio.engine.color import infrared as irm
    dev = M.DevelopPage(M.Settings())
    dev.render_now = lambda: None
    e = E.Edit()
    e.white_balance = E.WhiteBalance(mode="multipliers", multipliers=[1.03, 1.0, 1.01])
    e.infrared = irm.default_ir("850")
    dev._wb_chip(e)
    assert dev.wb_chip.text().startswith("IR white balance: set in Infrared")
    assert not dev.temp.isEnabled() and not dev.tint.isEnabled()
    e.infrared["on"] = False
    dev._wb_chip(e)
    assert not dev.wb_chip.text().startswith("IR")


@pytest.mark.skipif(not REAL.is_file(), reason="Bill's 850 nm raw is not in test_raws/infrared_850")
def test_real_850_frame_opens_in_black_and_white_and_carries_to_develop():
    app, M = _app()
    from studio import engine as E
    OUT.mkdir(parents=True, exist_ok=True)
    a = OUT / "IR_A.CR2"
    b = OUT / "IR_B.CR2"
    for p in (a, b):
        if not p.exists():
            shutil.copyfile(REAL, p)
        from studio.engine.api import SIDECAR_SUFFIX
        for side in (str(p) + SIDECAR_SUFFIX,):
            if os.path.exists(side):
                os.remove(side)
    page = M.InfraredPage(M.Settings())
    page.resize(1300, 800)
    page.show()
    opened = []
    page.open_result.connect(opened.append)
    page.set_frames([str(a), str(b)])
    assert _until(app, lambda: page.image is not None and page.result.result is not None, 300)
    assert "850" in page.guess_chip.text()
    assert page.bw_on.isChecked() and page.mono_raw.isChecked() and page.mono_raw.isEnabled()
    assert "no usable colour" in page.meter_chip.toolTip()
    assert page.edit.infrared["on"] and page.edit.white_balance.mode == "multipliers"
    # touching a B&W control keeps black and white and records the setting
    page.ch_ev[0].set_value(0.5)
    assert page.edit.infrared["bw"]["ev"][0] == 0.5 and page.edit.infrared["bw"]["on"]
    # a colour look switches B&W off but keeps its settings
    page.preset_pick.setCurrentIndex(page.preset_pick.findData("swap_rb"))
    page._preset_picked()
    assert _until(app, lambda: not page.edit.infrared["bw"]["on"], 60)
    assert page.edit.infrared["bw"]["ev"][0] == 0.5
    # Apply to the shoot, then Open in Develop: both sidecars carry the infrared edit
    page.apply_to_shoot()
    eb = E.load_edit(str(b))
    assert eb.infrared and eb.infrared["on"] and eb.infrared["preset"] == "swap_rb"
    page.build_full()
    assert opened == [str(a)]
    ea = E.load_edit(str(a))
    assert ea.infrared == page.edit.infrared and ea.white_balance == page.edit.white_balance
    # Develop renders the same picture from that sidecar
    img = E.open(str(a))
    r1 = E.render(img, ea, scale=0.25)
    r2 = E.render(img, page.edit, scale=0.25)
    assert np.array_equal(r1, r2)
    page.close()
