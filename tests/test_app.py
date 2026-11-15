"""Phase 4: the desktop app starts, shows every workspace, and develops a photo (offscreen, no display needed)."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_app_starts_with_all_workspaces(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    from studio.app import theme as T
    monkeypatch.setattr(M.Settings, "set", lambda self, k, v: self.d.__setitem__(k, v))
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(T.QSS)
    w = M.MainWindow()
    assert w.stack.count() == len(M.NAV) == 9
    for i in range(w.stack.count()):
        w.nav.setCurrentRow(i)
        app.processEvents()
    w.library.load_folder(str(tmp_path))
    assert w.library.files == []
    w.close()


def test_develop_controls_round_trip():
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    e = E.Edit(exposure_ev=0.5)
    e.denoise.enabled = True
    e.lens.defringe = False
    e.white_balance.mode = "temp_tint"
    e.white_balance.temperature = 3200
    page.controls_from_edit(e)
    page.edit = e
    back = page.edit_from_controls()
    assert back.exposure_ev == 0.5 and back.denoise.enabled and not back.lens.defringe
    assert back.white_balance.mode == "temp_tint" and back.white_balance.temperature == 3200
    app.processEvents()


def test_new_develop_controls_round_trip():
    """Phase 5/6 controls: geometry, lens, microcontrast and sharpening survive edit -> controls -> edit."""
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    e = E.Edit()
    e.geometry.rotate = -2.5
    e.geometry.vertical = 30
    e.geometry.crop = [0.1, 0.1, 0.9, 0.9]
    e.geometry.distortion = {"model": "poly3", "terms": [-0.02], "norm": 1.0, "source": "manual"}
    e.lens.vignette_model = {"model": "pa", "terms": [-0.2, 0, 0], "norm": 1.0, "source": "estimated from this photo"}
    e.lens.vignette = True
    e.lens.tag = "Laack 13.5cm Anastigmat"
    e.microcontrast.amount = 0.6
    e.microcontrast.scale = 0.5
    e.sharpen.enabled = True
    e.sharpen.amount = 0.8
    page.controls_from_edit(e)
    page.edit = e
    back = page.edit_from_controls()
    assert back.to_dict() == e.to_dict()
    assert page.dist.isEnabled() and page.vig.isEnabled()
    app.processEvents()


def test_crop_frame_keeps_its_aspect():
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    v = M.Viewer()
    v.resize(900, 600)
    v.after = M.to_qimage(np.zeros((400, 600, 3), np.uint8))
    v.crop_mode, v.crop_ratio = True, 1.0
    got = []
    v.crop_changed.connect(got.append)
    r = v._fit(v.after)
    v._crop_press(QPointF(r.right(), r.bottom()))                   # grab the bottom-right corner
    v._crop_move(QPointF(r.left() + 0.5 * r.width(), r.top() + 0.9 * r.height()))
    v.mouseReleaseEvent(None)
    x0, y0, x1, y1 = got[-1]
    assert abs((x1 - x0) * 600 - (y1 - y0) * 400) < 2                 # square in pixels
    app.processEvents()


# ------------------------------------------------------------------------------------------ Phase 6 panels
class _FakeImage:
    """Stands in for an opened raw so the panel logic can be tested without rendering."""
    meta = type("M", (), {"lens": "", "iso": 100, "iso_label": "ISO 100", "manual_lens": True})()


def _page(monkeypatch=None):
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    M.Settings.set = lambda self, k, v: self.d.__setitem__(k, v)
    page = M.DevelopPage(M.Settings())
    page.schedule = lambda: None                     # no renders in these tests
    page.render_now = lambda: None
    page.image = _FakeImage()
    page.controls_from_edit(page.edit)              # what opening a photo does
    page._reset_history(page.edit)
    page._canonical()
    return app, M, page


def test_light_and_colour_controls_round_trip():
    from studio import engine as E
    app, M, page = _page()
    e = E.Edit()
    e.light.highlights, e.light.shadows, e.light.whites, e.light.blacks, e.light.smart = -30, 40, 12, -8, 55
    e.light.curve = {"master": [[0.0, 0.0], [0.3, 0.25], [1.0, 1.0]], "b": [[0.0, 0.05], [1.0, 0.95]]}
    c = e.colour
    c.saturation, c.vibrance, c.bw = 12, -20, True
    c.bw_mix = {"red": 30.0, "blue": -40.0}
    c.hsl = {"orange": [5.0, -12.0, 3.0]}
    c.ranges = [dict(M.RANGE_DEFAULT, hue=200.0, shift=-15.0, sat=20.0)]
    c.grading = {"shadows": [220.0, 30.0, 0.0], "highlights": [60.0, 10.0, 5.0], "balance": -20.0}
    e.white_balance.mode = "multipliers"
    e.white_balance.multipliers = [2.1, 1.0, 1.6]
    e.white_balance.temperature, e.white_balance.tint = 3100, 4
    page.controls_from_edit(e)
    page.edit = e
    assert page.edit_from_controls().to_dict() == e.to_dict()
    assert page.bw_box.isHidden() is (page.level.currentIndex() != 1)
    app.processEvents()


def test_history_merges_one_slider_drag_into_one_step():
    app, M, page = _page()
    for v in (0.1, 0.25, 0.4, 0.5):                  # one drag of the Exposure slider
        page.ev.set_value(v)
    assert [lb for lb, _ in page.steps] == ["Opened", "Exposure +0.5"]
    page.l_sh.set_value(30)
    assert page.steps[-1][0] == "Light: shadows +30" and len(page.steps) == 3
    page.do_undo()
    assert page.edit.light.shadows == 0 and page.l_sh.value() == 0 and page.redo_b.isEnabled()
    page.jump_history(0)
    assert page.edit.exposure_ev == 0
    page.c_sat.set_value(10)                          # a new change after going back drops the redo steps
    assert [lb for lb, _ in page.steps] == ["Opened", "Colour: saturation +10"]
    app.processEvents()


def test_look_paste_and_snapshot_from_the_panel(tmp_path):
    from studio import engine as E
    app, M, page = _page()
    raw = tmp_path / "IMG_0002.CR3"
    raw.write_bytes(b"x")
    page.path = str(raw)
    page.ev.set_value(0.7)
    page.look_combo.setCurrentIndex(page.look_combo.findText("Warm film"))
    page.apply_look()
    assert page.edit.exposure_ev == 0.7 and page.edit.colour.grading and page.steps[-1][0] == "Look: Warm film"
    n = len(page.steps)
    page.ev.set_value(0.8)                            # the look's settings are stored as the controls show them
    assert len(page.steps) == n + 1 and page.steps[-1][0] == "Exposure +0.8"
    M.CLIP.update(edit=E.Edit(exposure_ev=-1.0), groups=["light"], source="other.CR3")
    page.paste_settings()
    assert page.edit.exposure_ev == -1.0 and page.steps[-1][0] == "Paste settings"
    page.save_sidecar()
    E.add_snapshot(page.path, "warm", page.steps[-2][1])
    page._fill_snapshots()
    page.snap_list.setCurrentRow(0)
    page.restore_snapshot()
    assert page.edit.exposure_ev == 0.8 and page.steps[-1][0] == "Snapshot: warm"
    app.processEvents()


def test_virtual_copies_from_the_panel(tmp_path):
    from studio import engine as E
    app, M, page = _page()
    raw = tmp_path / "IMG_0003.CR3"
    raw.write_bytes(b"x")
    page.path = str(raw)
    page.ev.set_value(0.5)
    page.save_sidecar()
    page._fill_versions()
    page._version_picked(page.ver.findData("new"))
    assert page.version == 1 and E.list_versions(page.path) == ["Master", "Copy 1"]
    page.c_bw.setChecked(True)
    page.save_sidecar()
    page._version_picked(page.ver.findData(0))
    assert page.version == 0 and not page.edit.colour.bw and page.edit.exposure_ev == 0.5
    assert E.load_version(page.path, 1).colour.bw
    app.processEvents()


def test_colour_wheel_picker_and_widgets():
    import numpy as np
    from PySide6.QtCore import QPointF
    from studio.app import widgets as W
    app, M, page = _page()
    page.last_a8 = np.tile(np.array([230, 140, 40], np.uint8), (60, 90, 1))          # an orange photo
    page._set_pick("range")
    page.open_loupe(0.5, 0.5)
    assert page.pick is None and len(page._ranges) == 1 and 45 < page._ranges[0]["hue"] < 80
    page.r_shift.set_value(25)
    assert page.edit.colour.ranges[0]["shift"] == 25 and page.steps[-1][0] == "Colour: colour wheel changed"
    page.last_a8 = np.full((60, 90, 3), 128, np.uint8)                                # grey: refused
    page._set_pick("range")
    page.open_loupe(0.5, 0.5)
    assert len(page._ranges) == 1
    cv = W.CurveEditor()
    cv.resize(300, 230)
    assert cv.value() is None
    r = cv._plot()
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QMouseEvent
    def mouse(kind, x, y):
        return QMouseEvent(kind, QPointF(x, y), QPointF(x, y), Qt.MouseButton.LeftButton,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    cv.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, r.left() + 0.5 * r.width(), r.top() + 0.5 * r.height()))
    cv.mouseMoveEvent(mouse(QEvent.Type.MouseMove, r.left() + 0.5 * r.width(), r.top() + 0.3 * r.height()))
    v = cv.value()["master"]
    assert len(v) == 3 and abs(v[1][0] - 0.5) < 0.02 and abs(v[1][1] - 0.7) < 0.02
    gw = W.GradeWheel("Shadows")
    gw.resize(150, 150)
    c, R = gw._geom()
    gw._set_from(QPointF(c.x(), c.y() - R / 2))                                       # straight up: 90 deg, 50 %
    assert abs(gw.hue - 90) < 1 and abs(gw.strength - 50) < 1
    assert W.hue_dist(10, 350) == 20
    app.processEvents()


def test_panel_fits_its_width():
    from PySide6.QtWidgets import QScrollArea
    app, M, page = _page()
    page.level.setCurrentIndex(1)
    for g in page.findChildren(M.Group):
        g.btn.setChecked(True)
    page.resize(1400, 900)
    page.show()
    app.processEvents()
    scroll = page.findChild(QScrollArea)
    assert scroll.widget().minimumSizeHint().width() <= scroll.viewport().width()
    page.close()


def test_library_paste_and_version_export_names(tmp_path):
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    M.Settings.set = lambda self, k, v: self.d.__setitem__(k, v)
    raws = []
    for n in ("A.CR3", "B.CR3"):
        p = tmp_path / n
        p.write_bytes(b"x")
        raws.append(str(p))
    E.save_edit(raws[1], E.Edit(exposure_ev=0.3))
    E.save_library_meta(raws[1], {"rating": 5, "pick": 1, "label": ""})
    w = M.MainWindow()
    src = E.Edit(exposure_ev=2.0)
    src.colour.vibrance = 40
    src.geometry.rotate = 5
    M.CLIP.update(edit=src, groups=["colour"], source="src.CR3")
    w.paste_to_files(raws)
    for f in raws:
        assert E.load_edit(f).colour.vibrance == 40 and E.load_edit(f).geometry.rotate == 0
    assert E.load_edit(raws[1]).exposure_ev == 0.3 and E.load_library_meta(raws[1])["rating"] == 5
    E.add_version(raws[0], E.Edit(), "Dark / moody")
    p = M.version_export_path(raws[0], 1, E.ExportSpec(format="tiff"), str(tmp_path))
    assert os.path.basename(p) == "A_Dark_moody.tif"
    w.close()
    app.processEvents()


def test_export_presets(tmp_path):
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    st = M.Settings()
    st.d = {"export_dir": str(tmp_path)}
    st.set = lambda k, v: st.d.__setitem__(k, v)
    dlg = M.ExportDialog(["a.CR3"], st)
    dlg.preset.setCurrentIndex(dlg.preset.findText("Web: JPEG, sRGB, 2048 px"))
    dlg._preset_picked(dlg.preset.currentIndex())
    sp = dlg.spec()
    assert (sp.format, sp.space, sp.long_edge, sp.bits) == ("jpeg", "srgb", 2048, 8)
    assert st.d["export_preset"] == "Web: JPEG, sRGB, 2048 px"
    st.d["export_presets"] = {"Mine": [0, 2, 3000]}
    dlg2 = M.ExportDialog(["a.CR3"], st)
    dlg2.preset.setCurrentIndex(dlg2.preset.findText("Mine"))
    dlg2._preset_picked(dlg2.preset.currentIndex())
    sp = dlg2.spec()
    assert (sp.format, sp.space, sp.long_edge) == ("tiff", "prophoto", 3000)
    app.processEvents()


def test_phase6_extras_round_trip_and_bw_filter():
    from studio import engine as E
    app, M, page = _page()
    e = E.Edit(highlight_mode="rebuild")
    e.light.dehaze = 35
    e.light.param = {"lights": 20.0, "shadows": -15.0}
    e.tone.toe, e.tone.shoulder = 30.0, 0.8
    e.finish.texture, e.finish.grain, e.finish.grain_size, e.finish.grain_roughness = -20.0, 40.0, 1.6, 70.0
    e.finish.grain_colour = True
    page.controls_from_edit(e)
    page.edit = e
    assert page.edit_from_controls().to_dict() == e.to_dict()
    assert abs(page.filmic.params["toe"] - 30) < 1e-6 and abs(page.filmic.params["shoulder"] - 0.8) < 1e-6
    page.controls_from_edit(E.Edit())
    page.edit = E.Edit()
    page._reset_history(page.edit)
    page._canonical()
    page.bw_filter.setCurrentIndex(page.bw_filter.findText("Red filter"))
    page._bw_filter_picked(page.bw_filter.currentIndex())
    assert page.edit.colour.bw and page.edit.colour.bw_mix["blue"] == -60
    app.processEvents()


def test_export_dialog_output_sharpening(tmp_path):
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    st = M.Settings()
    st.d = {"export_dir": str(tmp_path)}
    st.set = lambda k, v: st.d.__setitem__(k, v)
    dlg = M.ExportDialog(["a.CR3"], st)
    dlg.preset.setCurrentIndex(dlg.preset.findText("Print: 16-bit TIFF, ProPhoto RGB, full size"))
    dlg._preset_picked(dlg.preset.currentIndex())
    sp = dlg.spec()
    assert sp.sharpen_for == "glossy" and sp.sharpen_amount == "standard" and sp.space == "prophoto"
    st.d["export_presets"] = {"Old": [3, 1, 800]}                    # saved before sharpening existed
    dlg2 = M.ExportDialog(["a.CR3"], st)
    dlg2.preset.setCurrentIndex(dlg2.preset.findText("Old"))
    dlg2._preset_picked(dlg2.preset.currentIndex())
    assert dlg2.spec().sharpen_for == "none"
    app.processEvents()


def test_masks_from_the_panel():
    from conftest import scene, synthetic_raw
    from studio import engine as E
    from studio.engine.pipeline.pipeline import Image
    app, M, page = _page()
    page.image = Image(synthetic_raw(scene(120, 180)))
    page.new_mask("radial")
    assert page.viewer.tool == "radial" and page.mask_list.count() == 1
    page._drag_done(0.5, 0.5, 0.7, 0.5)
    sh = page.edit.masks[0]["shapes"][0]
    assert page.viewer.tool is None and sh["type"] == "radial" and abs(sh["cx"] - 0.5) < 0.01 and 0.15 < sh["rx"] < 0.25
    page.m_sl["exposure"].set_value(0.8)
    assert page.edit.masks[0]["adjust"]["exposure"] == 0.8
    page.new_mask("brush")
    page._stroke_done(([(0.2, 0.2), (0.4, 0.3)], False))
    page._stroke_done(([(0.6, 0.6), (0.7, 0.7)], True))                  # Alt: erase
    strokes = page.edit.masks[1]["shapes"][0]["strokes"]
    assert len(strokes) == 2 and strokes[1]["erase"] and not strokes[0]["erase"]
    page.mask_add.setCurrentIndex(page.mask_add.findText("- Radial (subtract)"))
    page._mask_add_picked(page.mask_add.findText("- Radial (subtract)"))
    page._drag_done(0.3, 0.25, 0.35, 0.25)
    assert page.edit.masks[1]["shapes"][-1]["op"] == "subtract"
    page.new_mask("luminance")
    page.r_lo.set_value(30)
    assert page.edit.masks[2]["shapes"][0]["lo"] == 0.3
    page.mask_list.item(2).setCheckState(M.Qt.CheckState.Unchecked)
    assert page.edit.masks[2]["enabled"] is False
    page.delete_mask()
    assert len(page.edit.masks) == 2 and any(lb.startswith("Mask") for lb, _ in page.steps)
    out = E.render(page.image, page.edit)                                # the edit renders
    assert out.shape[2] == 3 and np.isfinite(out).all()
    page.controls_from_edit(page.edit)                                   # and survives a round trip
    assert page.edit_from_controls().to_dict() == page.edit.to_dict()
    app.processEvents()


def _wait(app, cond, timeout=20.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    return cond()


def test_retouch_and_skin_from_the_panel():
    from conftest import scene, synthetic_raw
    from studio import engine as E
    from studio.engine.pipeline.pipeline import Image
    app, M, page = _page()
    page.image = Image(synthetic_raw(scene(120, 180)))
    page.path = "synthetic.cr2"
    page.rt_btn["heal"].setChecked(True)
    assert page.viewer.tool == "spot" and page.rt_kind == "heal"
    page.rt_size.set_value(3.0)                                          # a small test frame: a 5 px spot
    page._spot_new([(0.5, 0.5)])                                         # one click: a round heal spot
    assert _wait(app, lambda: page.edit.retouch and len(page.edit.retouch) == 1)
    sp = page.edit.retouch[0]
    assert sp["kind"] == "heal" and len(sp["pts"]) == 1 and sp["src"] != [0.0, 0.0]
    assert page.steps[-1][0] == "Heal spot"
    page._spot_new([(0.2, 0.2), (0.25, 0.22), (0.3, 0.25), (0.35, 0.3)])   # a brushed stroke
    assert _wait(app, lambda: len(page.edit.retouch) == 2)
    assert len(page.edit.retouch[1]["pts"]) >= 2
    page._spot_picked(0)
    page.rt_size.set_value(2.0)
    assert abs(page.edit.retouch[0]["size"] - 0.02) < 1e-9
    page._spot_moved(0, "src", 0.05, 0.0)                                # drag the source circle
    assert page.steps[-1][0] == "Spot source moved"
    page._refresh_spot_view()
    assert len(page.viewer.spots) == 2 and page.viewer.spots[0]["sel"]
    page._spot_delete(1)
    assert len(page.edit.retouch) == 1
    page._set_spot_tool(None)
    assert page.viewer.tool is None and not page.rt_btn["heal"].isChecked()
    page.new_mask("skin")
    assert page.edit.masks[0]["shapes"][0]["type"] == "skin"
    page.m_sl["smooth_skin"].set_value(60)
    assert page.edit.masks[0]["adjust"]["smooth_skin"] == 60
    out = E.render(page.image, page.edit)
    assert np.isfinite(out).all()
    page.controls_from_edit(page.edit)
    assert page.edit_from_controls().to_dict() == page.edit.to_dict()
    page.clear_spots()
    assert page.edit.retouch is None
    app.processEvents()


def test_zoom_view_maps_clicks_to_the_whole_photo():
    from PySide6.QtCore import QPointF
    app, M, page = _page()
    v = page.viewer
    v.after = M.to_qimage(np.zeros((100, 150, 3), np.uint8))
    v.resize(600, 400)
    r = v._fit(v.after)
    centre = QPointF(r.center().x(), r.center().y())
    assert np.allclose(v._uv(centre), (0.5, 0.5), atol=0.01)
    v.win = (0.2, 0.4, 0.4, 0.6)                                         # a 100% window of the photo
    assert np.allclose(v._uv(centre), (0.3, 0.5), atol=0.01)
    p = v._px((0.3, 0.5), r)
    assert abs(p.x() - centre.x()) < 1 and abs(p.y() - centre.y()) < 1
    assert v._long_px(r) > 4 * max(r.width(), r.height())


def test_glow_mode_from_the_light_panel(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_SETTINGS_DIR", str(tmp_path))
    from conftest import scene, synthetic_raw
    from test_glow import true_profile
    from studio.engine.raw import glow as G
    from studio.engine.pipeline.pipeline import Image
    G.save_profile(true_profile("TestLens", "Test lens"))
    app, M, page = _page()
    page.image = Image(synthetic_raw(scene(120, 180) * 4.0, clip_white=True))
    page.path = "synthetic.cr2"
    page.controls_from_edit(page.edit)
    assert page.glow_combo.isHidden()
    page.hl_mode.setCurrentIndex(2)
    assert page.edit.highlight_mode == "glow" and not page.glow_combo.isHidden()
    k = page.glow_combo.findData("TestLens")
    assert k > 0
    page.glow_combo.setCurrentIndex(k)
    assert page.edit.glow_profile == "TestLens"
    page._glow_refresh()
    assert _wait(app, lambda: "blown light" in page.glow_chip.text() or "No blown" in page.glow_chip.text())
    page.controls_from_edit(page.edit)
    assert page.edit_from_controls().to_dict() == page.edit.to_dict()
    page.hl_mode.setCurrentIndex(1)
    assert page.edit.highlight_mode == "rebuild" and page.glow_combo.isHidden()


def test_portrait_and_sculpting_from_the_panel():
    from test_portrait import face_dict, sample_face, skin_image
    from conftest import synthetic_raw
    from studio import engine as E
    from studio.engine.pipeline.pipeline import Image
    app, M, page = _page()
    h, w = 300, 400
    page.image = Image(synthetic_raw(skin_image(h, w)))
    page.path = "synthetic.cr2"
    page.controls_from_edit(page.edit)
    assert not page.pt_blem.isEnabled()                                  # no faces yet
    page._portrait["faces"] = [face_dict(sample_face(200, 160, 150), w, h)]
    page._portrait_changed("Portrait: 1 face found")
    assert page.pt_blem.isEnabled() and "1 face" in page.pt_info.text()
    page.pt_sl["younger"].set_value(40)
    assert page.edit.portrait["younger"] == 40 and len(page.edit.portrait["faces"]) == 1
    assert page.steps[-1][0] == "Portrait: younger +40"
    page._refresh_portrait_view()
    assert len(page.viewer.faces) == 1 and len(page.viewer.faces[0]["lines"]) == 6
    page.pt_eyes.setChecked(True)
    assert page.viewer.tool == "eyes"
    page.sh_btn["butt"].setChecked(True)                                 # one tool at a time
    assert page.viewer.tool == "shape" and not page.pt_eyes.isChecked() and page.viewer.shape_tool == "butt"
    page._shape_drawn(0.6, 0.6, 0.8, 0.62)
    sc = page.edit.portrait["sculpt"]
    assert len(sc) == 1 and sc[0]["tool"] == "butt" and sc[0]["lift"] == 40 and page._shsel == 0
    assert np.allclose(sc[0]["up"], [0, -1], atol=1e-6)
    page.sh_size.set_value(-30)
    assert page.edit.portrait["sculpt"][0]["size"] == -30
    page.sc_btn["push"].setChecked(True)
    assert page.viewer.tool == "sculpt" and page.sculpt_kind == "push"
    page._sculpt_stroke([(0.3, 0.5), (0.32, 0.5), (0.34, 0.5), (0.36, 0.5)])
    assert page.edit.portrait["sculpt"][-1]["kind"] == "push" and page.steps[-1][0] == "Sculpt: push"
    out = E.render(page.image, page.edit, scale=0.5)
    assert np.isfinite(out).all()
    page.controls_from_edit(page.edit)
    assert page.edit_from_controls().to_dict() == page.edit.to_dict()
    page.sculpt_undo()
    assert len(page.edit.portrait["sculpt"]) == 1
    page.sculpt_clear()
    assert "sculpt" not in page.edit.portrait
    page.remove_blemishes()
    assert _wait(app, lambda: not page.viewer.busy)
    page.forget_faces()
    assert "faces" not in (page.edit.portrait or {}) and not page.pt_blem.isEnabled()
    page._set_portrait_tool(None)
    assert page.viewer.tool is None
    app.processEvents()


def test_person_mask_from_the_panel(monkeypatch):
    from conftest import synthetic_raw
    from test_people import _synthetic_person
    from studio import engine as E
    from studio.engine.local import masks as mk
    from studio.engine.pipeline import pipeline as pl
    app, M, page = _page()
    rgb, person, pts, _ = _synthetic_person(200, 150)
    page.image = pl.Image(synthetic_raw(np.clip(rgb, 0, 1)))
    page.path = "synthetic.cr2"
    page.controls_from_edit(page.edit)
    fake = {"type": "person", "op": "add", "part": "person", "w": 150, "h": 200,
            "data": mk.encode_bitmap((person * 255).astype(np.uint8)),
            "people": [[[x / 150, y / 200, v] for x, y, v in pts]], "faces": []}
    monkeypatch.setattr(pl, "find_people", lambda img, e, size=640: dict(fake))
    monkeypatch.setattr(E, "find_people", lambda img, e, size=640: dict(fake), raising=False)
    page.new_mask("person")
    assert _wait(app, lambda: page.edit.masks and page.edit.masks[0]["shapes"])
    m = page.edit.masks[0]
    assert m["name"] == "Person 1" and m["shapes"][0]["type"] == "person"
    assert not page.person_part_row.isHidden()
    page.person_part.setCurrentIndex(page.person_part.findData("hair"))
    page._person_part_picked(page.person_part.currentIndex())
    assert page.edit.masks[0]["shapes"][0]["part"] == "hair" and page.edit.masks[0]["name"] == "Person 1: Hair"
    assert page.steps[-1][0] == "Person mask: hair"
    page.m_sl["exposure"].set_value(0.5)
    out = E.render(page.image, page.edit, scale=0.5)
    assert np.isfinite(out).all()
    page.controls_from_edit(page.edit)
    assert page.edit_from_controls().to_dict() == page.edit.to_dict()
    app.processEvents()
