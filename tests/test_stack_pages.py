"""The three stacking pages (BUILD_PLAN_TABS 2.4), offscreen: they build, show their empty state, fit their panel,
take frames, and run Preview and Build end to end on synthetic frames (served through monkeypatched frames
functions: there is no raw writer)."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from conftest import ROOT  # noqa: E402

OUT = ROOT / "outputs" / "pytest_stack_pages"


def _app():
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    from studio.app import theme as T
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(T.QSS)
    M.Settings.set = lambda self, k, v: self.d.__setitem__(k, v)
    return app, M


def _wait(app, page, timeout=240):
    t0 = time.time()
    app.processEvents()
    while page.jobbar.busy and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(20):
        app.processEvents()
    assert not page.jobbar.busy, "the job did not finish"


@pytest.mark.parametrize("cls", ["FocusStackPage", "SuperResPage", "DualPixelPage", "PanoramaPage", "InfraredPage"])
def test_page_builds_empty_and_fits(cls):
    app, M = _app()
    page = getattr(M, cls)(M.Settings())
    page.resize(1040, 600)
    page.show()
    app.processEvents()
    assert page.centre.currentIndex() == 0                  # the empty state, never a blank panel
    sc = page.panel_scroll
    assert sc.widget().minimumSizeHint().width() <= sc.viewport().width(), cls
    assert page.minimumSizeHint().height() <= 600
    page.close()


def test_frames_in_tray_and_unreadable_named(tmp_path):
    app, M = _app()
    page = M.FocusStackPage(M.Settings())
    OUT.mkdir(parents=True, exist_ok=True)
    a, b = OUT / "A_0001.CR2", OUT / "notes.txt"
    a.write_bytes(b"")
    b.write_text("not a photo")
    page.add_files([str(a), str(b)])
    app.processEvents()
    assert page.tray.count() == 2 and page.centre.currentIndex() == 1
    bad = page.tray.item(1)
    assert "left out" in bad.text() and "not a raw" in bad.toolTip()
    assert page.frame_paths() == [str(a)]
    assert not page.build_b.isEnabled()                     # one frame is not a stack
    page.clear()
    assert page.tray.count() == 0 and page.centre.currentIndex() == 0
    page.close()


def test_focus_page_preview_build_and_cancel(monkeypatch):
    import test_focus_stack as TF
    from studio import engine as E
    app, M = _app()
    frames, _ = TF.make_bracket(n=5, h=384, w=576, seed=4)
    paths = TF.install(monkeypatch, frames, TF._names(5), "page")
    page = M.FocusStackPage(M.Settings())
    page.resize(1280, 720)
    page.show()
    page.set_frames(paths)
    app.processEvents()
    page.out_dir = str(OUT / "focus")
    page.preview()
    _wait(app, page)
    assert page.analysis is not None and page.preview_img is not None
    assert page.result.result is not None
    for i in range(page.view_pick.count()):
        page.view_pick.setCurrentIndex(i)
        app.processEvents()
        assert page.result.result is not None
    page.result.toggle_reference(True)
    assert page.result.view.chip.startswith("Best single")
    page.build_full()
    _wait(app, page)
    assert page.last_result.endswith("_Focus_Studio.tif") and os.path.exists(page.last_result)
    img = E.open(page.last_result)
    assert img.raw.scene_linear and page.open_b.isEnabled()
    # Cancel: nothing is left behind
    os.remove(page.last_result)
    page.last_result = ""
    page.build_full()
    page.jobbar.cancel()
    _wait(app, page)
    leftovers = [f for f in os.listdir(page.out_dir) if f.endswith(".partial")]
    assert not leftovers
    assert "Cancel" in page.jobbar.text.text() or os.path.exists(page.target(
        os.path.basename(paths[0]), paths[0]))
    page._drop_analysis()
    page.close()


def test_superres_page_preview_and_build(monkeypatch):
    import test_superres as TS
    from studio.engine.stack import align as A
    app, M = _app()
    truth4, meta = TS.make_truth(h=160, w=240, seed=1)
    rng = np.random.default_rng(3)
    mos = []
    for k in range(8):
        T = TS._sim(1.0, 0.0, rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), meta["h"], meta["w"])
        _, m = TS.frame_from(truth4, meta, T, rng=np.random.default_rng(10 + k))
        mos.append(m)
    paths = TS.install(monkeypatch, mos, "page")
    page = M.SuperResPage(M.Settings())
    page.resize(1280, 720)
    page.show()
    page.set_frames(paths)
    page.out_dir = str(OUT / "superres")
    for i in range(page.mode_group.buttons().__len__()):
        page.mode_group.button(i).setChecked(True)
        app.processEvents()
    page.mode_group.button(1).setChecked(True)              # Cleaner, same size: fast
    app.processEvents()
    page.preview()
    _wait(app, page)
    assert page.analysis is not None and page.last is not None
    assert page.coverage.dots and "px" in page.size_line.text()
    for i in range(page.view_pick.count()):
        page.view_pick.setCurrentIndex(i)
        app.processEvents()
    page.build_full()
    _wait(app, page)
    assert page.last_result.endswith("_Stack_Studio.tif") and os.path.exists(page.last_result)
    page._drop_analysis()
    page.close()


def test_dual_pixel_page(monkeypatch):
    import test_dualpixel as TD
    from studio.engine.dualpixel import decode
    from studio.engine.io.intermediate import read_info
    app, M = _app()
    dp, _ = TD.make_dual(h=256, w=384, seed=5, bright=False)
    OUT.mkdir(parents=True, exist_ok=True)
    src = OUT / "IMG_DUAL.CR2"
    src.write_bytes(b"")
    dp.path = str(src)
    monkeypatch.setattr(decode, "open_dual", lambda path: dp)
    page = M.DualPixelPage(M.Settings())
    page.resize(1280, 720)
    page.show()
    page.add_files([str(src)])
    _wait(app, page)
    assert page.dp is dp and page.result.result is not None
    for i in range(page.view_pick.count()):
        page.view_pick.setCurrentIndex(i)
        _wait(app, page)
        assert page.view_pick.currentText() in page.cache
    page.out_dir = str(OUT / "dual")
    page.build_full()
    _wait(app, page)
    assert page.last_result.endswith("_DP-HDR_Studio.tif") and read_info(page.last_result)["method"] == "hdr"
    page.export_halves((1, 2))
    _wait(app, page)
    assert os.path.exists(os.path.join(page.out_dir, "IMG_DUAL_DP-B_Studio.tif"))
    page.export_depth()
    _wait(app, page)
    page.export_3d("wiggle")
    _wait(app, page)
    assert os.path.exists(os.path.join(page.out_dir, "IMG_DUAL_DP-3D_Studio.gif"))
    # a normal raw: said in plain words
    monkeypatch.setattr(decode, "open_dual", lambda path: (_ for _ in ()).throw(
        decode.UnsupportedRaw(f"x.cr2: {decode.NOT_DUAL} {decode.HOW_TO}")))
    page.add_files([str(src)])
    _wait(app, page)
    assert page.dp is None and decode.NOT_DUAL in page.result.view.note
    page.close()


def test_library_sends_to_the_tabs(monkeypatch):
    app, M = _app()
    w = M.MainWindow()
    got = []
    monkeypatch.setattr(w.focus, "set_frames", lambda files: got.append(("focus", files)))
    monkeypatch.setattr(w.dualpixel, "set_frames", lambda files: got.append(("dp", files)))
    w.library.send_to.emit("focus", ["a.cr3", "b.cr3"])
    assert w.nav.currentRow() == 3 and got[-1] == ("focus", ["a.cr3", "b.cr3"])
    w.library.send_to.emit("dualpixel", ["c.cr3"])
    assert w.nav.currentRow() == 2 and got[-1] == ("dp", ["c.cr3"])
    monkeypatch.setattr(w.panorama, "set_frames", lambda files: got.append(("pano", files)))
    monkeypatch.setattr(w.infrared, "set_frames", lambda files: got.append(("ir", files)))
    w.library.send_to.emit("panorama", ["d.cr2", "e.cr2"])
    assert w.nav.currentRow() == M.ROW["Panorama"] and got[-1] == ("pano", ["d.cr2", "e.cr2"])
    w.library.send_to.emit("infrared", ["f.cr2", "g.cr2"])
    assert w.nav.currentRow() == M.ROW["Infrared"] and got[-1] == ("ir", ["f.cr2", "g.cr2"])
    monkeypatch.setattr(w.cardstacks, "set_sources", lambda items: got.append(("card", items)))
    w.library.send_to.emit("cardstacks", ["h.cr3", "i.cr3", "j.cr3"])
    assert w.nav.currentRow() == M.ROW["Card to Stacks"] and got[-1] == ("card", ["h.cr3", "i.cr3", "j.cr3"])
    assert [n for _, n in M.NAV][:8] == ["Library", "Develop", "Dual Pixel Lab", "Focus Stack", "Card to Stacks",
                                         "Super-Res Stack", "Infrared", "Panorama"]
    w.close()


def test_panorama_page_finds_layout_previews_and_builds():
    """End to end on the synthetic sphere frames that tests/test_pano.py writes (linear TIFFs)."""
    import test_pano as TP
    app, M = _app()
    paths = sorted(str(p) for p in TP.OUT.glob("SYN_000?.tif"))
    if len(paths) < 4:
        pytest.skip("run tests/test_pano.py first (it writes the synthetic frames)")
    page = M.PanoramaPage(M.Settings())
    page.resize(1300, 800)
    page.show()
    opened = []
    page.open_result.connect(opened.append)
    page.set_frames(paths)
    assert page.find_b.isEnabled() and page.preview_b.isEnabled()
    page.find_layout()
    _wait(app, page)
    assert page.analysis is not None and all(page.analysis.placed), page.jobbar.text.text()
    assert page.view_pick.currentText() == "Layout" and page.result.result is not None
    page.size_pick.setCurrentIndex(2)                       # 25%: quick
    page.preview()
    _wait(app, page)
    assert page.last is not None and page.result.result is not None
    for v in range(4):
        page.view_pick.setCurrentIndex(v)
        app.processEvents()
        assert page.result.result is not None
    page.build_full()
    _wait(app, page)
    assert page.last_result.endswith("_Pano_Studio.tif") and os.path.exists(page.last_result)
    page.open_b.click()
    assert opened == [page.last_result]
    page.close()


def test_card_to_stacks_page_lists_puts_back_and_rejudges():
    """The Card to Stacks queue on the synthetic card of tests/test_sorter.py: one row per stack, burst and single
    shot; the planted bad frames unticked with their reasons in red; ticking one puts it back (amber, "put back by
    you"); a stricter setting judges again; Stack all is ready for both stacks."""
    import test_sorter as TS
    from PySide6.QtCore import Qt
    from studio.app import theme as T
    app, M = _app()
    page = M.CardStacksPage(M.Settings())
    page.resize(1100, 700)
    page.show()
    app.processEvents()
    assert page.centre.currentIndex() == 0                  # empty state first
    card, bad = TS.make_card()
    page.sources = ["(synthetic card)"]
    page.groups = TS.group(card)
    page._fill()
    page._say_report()
    assert page.tree.topLevelItemCount() == 4 and page.centre.currentIndex() == 1
    top = page.tree.topLevelItem(0)
    assert top.childCount() == 13 and "Focus stack" in top.text(0)
    soft = [top.child(i) for i in range(top.childCount()) if bad.get(top.child(i).text(0)) == "soft"][0]
    assert soft.checkState(0) == Qt.CheckState.Unchecked and "soft everywhere" in soft.text(1)
    assert soft.foreground(1).color().name().lower() == T.RED.lower()
    assert page.all_b.isEnabled()
    n_out = len(page.groups[0].left_out)
    soft.setCheckState(0, Qt.CheckState.Checked)            # one click puts it back
    app.processEvents()
    f = page.groups[0].frames[top.indexOfChild(soft)]
    assert f.status == "warn" and f.reason.startswith("put back by you")
    assert len(page.groups[0].left_out) == n_out - 1
    assert page.peak.isChecked()                            # Show what's in focus (Bill 2026-11-17)
    page.tree.setCurrentItem(soft)                          # the frame, cyan where it is in focus
    t0 = time.time()
    while "in focus in" not in page.view_note.text() and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.02)
    assert "in focus in" in page.view_note.text()
    assert page.view.pixmap() is not None and not page.view.pixmap().isNull()
    page.tree.setCurrentItem(top)                           # the stack: cyan in focus, amber missed
    t0 = time.time()
    while "no frame got sharp" not in page.view_note.text() and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.02)
    assert "no frame got sharp" in page.view_note.text()
    page.peak.setChecked(False)                             # off: the plain preview
    page.tree.setCurrentItem(soft)
    app.processEvents()
    assert "sharpest in" in page.view_note.text()
    page.strict.set_key("strict")
    page._rejudge()
    assert len(page.groups[0].left_out) >= n_out            # judged again from what was read
    assert "2 focus stacks" in page.report.text()
    page.clear()
    assert page.tree.topLevelItemCount() == 0 and page.centre.currentIndex() == 0
    page.close()


def test_focus_page_depth_view_and_save_surface(monkeypatch):
    """The Depth view (coloured by how each place was measured) and "Save surface for games" on a synthetic
    hand-held stack of Dual Pixel RAWs (BUILD_PLAN_DEPTH_STACK F)."""
    import depth_synth as DS
    app, M = _app()
    frames, halves, _ = DS.make_stack(n=6, h=384, w=576)
    paths = DS.install(monkeypatch, frames, halves, "page_depth")
    page = M.FocusStackPage(M.Settings())
    page.resize(1280, 720)
    page.show()
    page.set_frames(paths)
    app.processEvents()
    page.out_dir = str(OUT / "depth")
    assert not page.surface_b.isEnabled()                   # nothing measured yet
    page.preview()
    _wait(app, page)
    assert page.analysis is not None and page.analysis.dual_pixel
    page.view_pick.setCurrentText("Depth")
    app.processEvents()
    assert page.result.view.chip.startswith("Depth · measured two ways")
    assert "Measured two ways" in page.report.text() and "Focus and Dual Pixel" in page.depth_method.text()
    page.depth_flip.setChecked(True)
    app.processEvents()
    assert page.surface_params().near in ("first", "last")
    page.depth_ratio.setCurrentText("1:2")
    assert page.surface_b.isEnabled()
    page.save_surface()
    _wait(app, page)
    files = page.last_surface
    assert set(files) == {"colour", "height", "normal", "confidence", "agreement", "json"}
    assert all(os.path.exists(p) for p in files.values())
    import json
    doc = json.load(open(files["json"], encoding="utf-8"))
    assert doc["height"]["near_end"].startswith("set by you") and doc["lateral"]["mm_per_pixel"]
    page._drop_analysis()
    page.close()
