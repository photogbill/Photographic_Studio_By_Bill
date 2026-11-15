"""Phase 4b-2..4: the Library (plan 24.7): speed on big folders (background listing, disk thumbnail cache, only
what is on screen, the index), Review, filters and sort, Process picks, read-only folders. Offscreen, no display.

Real raws are needed for thumbnails and previews: the sample raws in the app folder, linked many times over (hard
links cost no disk space). Tests that need them are skipped where they are not present.
"""
from __future__ import annotations

import os
import shutil
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = [os.path.join(ROOT, n) for n in ("IMG_3001.CR2", "3C9A6198.CR3") if os.path.isfile(os.path.join(ROOT, n))]
needs_raws = pytest.mark.skipif(not SAMPLES, reason="needs the sample raws in the app folder")


def _link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def shoot(folder, n=6):
    """A folder of n raws (hard links of the samples, so each has its own path and sidecar)."""
    os.makedirs(folder, exist_ok=True)
    out = []
    for i in range(n):
        src = SAMPLES[i % len(SAMPLES)]
        ext = os.path.splitext(src)[1]
        p = os.path.join(folder, f"IMG_{i:04d}{ext}")
        if not os.path.exists(p):
            _link(src, p)
        out.append(p)
    return out


@pytest.fixture()
def app(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from studio.app import common, theme as T
    monkeypatch.setenv("STUDIO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("STUDIO_SETTINGS_DIR", str(tmp_path / "settings"))
    from studio.engine.io import thumbs
    monkeypatch.setattr(thumbs, "_DEFAULT", None)
    from studio.engine import api
    api.reload_central_sidecars()
    monkeypatch.setattr(common.Settings, "set", lambda self, k, v: self.d.__setitem__(k, v))
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(T.QSS)
    yield a
    api.reload_central_sidecars()


def pump(app, cond, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def library(app):
    from studio.app import common
    from studio.app.library import LibraryPage
    st = common.Settings()
    st.d = {}
    page = LibraryPage(st)
    page.resize(1200, 800)
    page.show()
    app.processEvents()
    return page


def loaded(page):
    return not page.loading and not page.meta_pending


# ------------------------------------------------------------------------------------------ 4b-2 speed
@needs_raws
def test_folder_opens_in_the_background_and_only_visible_thumbs_are_made(app, tmp_path):
    from studio.engine.io.thumbs import default_cache
    files = shoot(str(tmp_path / "big"), 300)
    page = library(app)
    t0 = time.perf_counter()
    page.load_folder(str(tmp_path / "big"))
    assert time.perf_counter() - t0 < 0.5                       # the window answers at once
    assert pump(app, lambda: loaded(page), 60)
    assert len(page.files) == 300
    assert pump(app, lambda: len(page.pix) >= len(page.visible_rows(0)), 30)
    app.processEvents()
    made = sum(1 for f in files if default_cache().has(f, os.path.getsize(f), os.path.getmtime(f)))
    assert 0 < made < 120                                       # on screen plus one screen ahead, not all 300
    # scrolling on asks for the rest as they come into view
    page.grid.scrollToBottom()
    assert pump(app, lambda: files[-1] in page.pix or page.model.rows[-1]["path"] in page.pix, 30)
    page.close()


@needs_raws
def test_second_visit_comes_from_the_cache_and_the_index(app, tmp_path):
    from studio.engine.io import libindex
    shoot(str(tmp_path / "f"), 40)
    page = library(app)
    page.load_folder(str(tmp_path / "f"))
    assert pump(app, lambda: loaded(page) and len(page.pix) >= 10, 60)
    entries = libindex.list_photos(str(tmp_path / "f"))
    assert libindex.load_entries(entries) == 0                  # everything now comes from the index
    t0 = time.perf_counter()
    page.pix.clear()
    page.load_folder(str(tmp_path / "f"))
    assert pump(app, lambda: loaded(page) and len(page.pix) >= 10, 30)
    assert time.perf_counter() - t0 < 5
    assert page.timings["meta"] < 2.0
    page.close()


def test_thumb_cache_cap_and_clear(tmp_path):
    import numpy as np
    from PIL import Image
    from studio.engine.io.thumbs import ThumbCache
    c = ThumbCache(str(tmp_path / "thumbs"), cap_bytes=60_000)
    for i in range(40):                                         # fake cached thumbnails, oldest first
        f = c.file_for(f"/x/{i}.CR3", 1, float(i))
        os.makedirs(os.path.dirname(f), exist_ok=True)
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (60, 90, 3), np.uint8)).save(f, quality=90)
        os.utime(f, (1000 + i, 1000 + i))
    used = c.usage()
    assert used > 60_000
    c.prune()
    assert c.usage() <= 60_000 * 0.9
    assert not c.has("/x/0.CR3", 1, 0.0) and c.has("/x/39.CR3", 1, 39.0)    # least recently used went first
    assert c.clear() > 0 and c.usage() == 0


def test_index_is_only_a_cache(tmp_path, monkeypatch):
    from studio import engine as E
    from studio.engine.io import libindex
    monkeypatch.setenv("STUDIO_CACHE_DIR", str(tmp_path / "cache"))
    d = tmp_path / "p"
    d.mkdir()
    for n in ("A.CR3", "B.CR3"):
        (d / n).write_bytes(b"x" * 10)
    E.save_library_meta(str(d / "B.CR3"), {"rating": 4, "pick": 1, "label": ""})
    entries = libindex.list_photos(str(d))
    assert [e["side_mtime"] > 0 for e in entries] == [False, True]
    got = {}
    libindex.load_entries(entries, on_batch=lambda b: got.update(b))
    assert got[str(d / "B.CR3")]["rating"] == 4 and got[str(d / "A.CR3")]["rating"] == 0
    time.sleep(0.02)
    E.save_library_meta(str(d / "B.CR3"), {"rating": 2, "pick": 0, "label": ""})   # the sidecar is the truth
    got.clear()
    entries = libindex.list_photos(str(d))
    libindex.load_entries(entries, on_batch=lambda b: got.update(b))
    assert got[str(d / "B.CR3")]["rating"] == 2
    os.remove(libindex.LibraryIndex().path)                     # deleting the index loses nothing
    got.clear()
    libindex.load_entries(libindex.list_photos(str(d)), on_batch=lambda b: got.update(b))
    assert got[str(d / "B.CR3")]["rating"] == 2


def test_subfolders_skip_only_studio_and_hidden_folders(tmp_path):
    from studio.engine.io import libindex
    for rel in ("a/IMG_1.CR3", "outputs/IMG_2.CR3", "_studio/IMG_3.CR3", ".hidden/IMG_4.CR3", "IMG_0.CR3"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    names = sorted(e["name"] for e in libindex.list_photos(str(tmp_path), recursive=True))
    assert names == ["IMG_0.CR3", "IMG_1.CR3", "IMG_2.CR3"]
    assert [e["name"] for e in libindex.list_photos(str(tmp_path))] == ["IMG_0.CR3"]


# ------------------------------------------------------------------------------------------ ratings, filters, sort
@needs_raws
def test_rating_keys_save_in_the_background_and_filters_and_sort(app, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent
    from studio import engine as E
    files = shoot(str(tmp_path / "s"), 6)
    page = library(app)
    page.load_folder(str(tmp_path / "s"))
    assert pump(app, lambda: loaded(page), 30)
    cr2 = [f for f in page.files if f.endswith(".CR2")]
    assert page.files[:3] == sorted(cr2)                        # capture time: the 2016 CR2s come first
    page.select_path(page.files[0])
    for key, text in ((Qt.Key.Key_5, "5"), (Qt.Key.Key_P, "p")):
        page.grid.keyPressEvent  # (the grid's own handler is bypassed by the page's event filter)
        app.sendEvent(page.grid, QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier, text))
    first = page.files[0]
    page.set_flag([page.files[1]], "x")
    page.set_flag([page.files[2]], "3")
    assert page.writer.wait(5000)
    assert E.load_library_meta(first) == {"rating": 5, "pick": 1, "label": ""}
    assert "1 pick" in page.count.text() and "1 reject" in page.count.text()
    page.filter.setCurrentIndex(6)                               # picks only
    assert page.files == [first]
    page.filter.setCurrentIndex(3)                               # three stars and up
    assert set(page.files) == {first, sorted(cr2)[2]}
    page.filter.setCurrentIndex(0)
    page.unrated.setChecked(True)
    assert len(page.files) == 3
    page.unrated.setChecked(False)
    page.cam_f.setCurrentIndex(page.cam_f.findData("Canon EOS R5"))
    assert page.files and all(f.endswith(".CR3") for f in page.files)
    page.cam_f.setCurrentIndex(0)
    page.iso_f.setCurrentIndex(4)                                # ISO 6400 and up: none
    assert page.files == []
    page.iso_f.setCurrentIndex(0)
    page.search.setText("0004")
    page.apply_filter()
    assert [os.path.basename(f) for f in page.files] == ["IMG_0004.CR2"]
    page.search.clear()
    page.sort.setCurrentIndex(page.sort.findData("Rating"))
    assert page.files[0] == first
    page.sort.setCurrentIndex(page.sort.findData("File name"))
    assert [os.path.basename(f) for f in page.files] == sorted(os.path.basename(f) for f in files)
    page.close()


def test_a_rating_that_cannot_be_saved_is_seen_and_can_go_to_this_pc(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from studio import engine as E
    from studio.engine import api
    d = tmp_path / "ro"
    d.mkdir()
    (d / "A.CR3").write_bytes(b"x")
    page = library(app)
    page.model.set_entries([{"path": str(d / "A.CR3"), "name": "A.CR3", "size": 1, "mtime": 0.0, "side_mtime": 0.0,
                             "rating": 0, "pick": 0, "label": ""}])
    shown = []

    def fake_exec(box):
        shown.append(box.text())
        for b in box.buttons():
            if box.buttonRole(b) == QMessageBox.ButtonRole.AcceptRole:
                b.click()
        return 0
    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    real = api.save_library_meta
    calls = {"n": 0}

    def readonly_once(path, meta, central_dir=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(13, "Permission denied")
        return real(path, meta, central_dir)
    monkeypatch.setattr(E, "save_library_meta", readonly_once)
    page.set_flag([str(d / "A.CR3")], "4")
    assert pump(app, lambda: shown and calls["n"] >= 2 and page.writer.wait(10), 10)
    assert "cannot be saved" in shown[0]
    assert E.central_folder_for(str(d))
    assert E.load_library_meta(str(d / "A.CR3"))["rating"] == 4
    assert not os.path.exists(str(d / "A.CR3.psb.json"))        # nothing written beside the photo
    page.close()


def test_central_sidecars_redirect_every_read_and_write(tmp_path, monkeypatch):
    from studio import engine as E
    from studio.engine import api
    monkeypatch.setenv("STUDIO_SETTINGS_DIR", str(tmp_path / "settings"))
    api.reload_central_sidecars()
    d = tmp_path / "drive"
    d.mkdir()
    raw = str(d / "A.CR3")
    open(raw, "wb").close()
    E.save_edit(raw, E.Edit(exposure_ev=0.7))                    # an edit made before the switch
    c = E.use_central_sidecars(str(d))
    assert E.sidecar_path(raw) == os.path.join(c, "A.CR3.psb.json")
    assert E.load_edit(raw).exposure_ev == 0.7                   # copied in once
    E.save_library_meta(raw, {"rating": 3, "pick": 0, "label": ""})
    assert E.load_library_meta(raw)["rating"] == 3
    api.reload_central_sidecars()                                # survives a restart
    assert E.sidecar_path(raw).startswith(c)
    E.use_central_sidecars(str(d), on=False)
    assert E.sidecar_path(raw) == raw + ".psb.json"
    api.reload_central_sidecars()


# ------------------------------------------------------------------------------------------ 4b-3 Review
@needs_raws
def test_review_rates_moves_zooms_and_compares(app, tmp_path):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from studio import engine as E
    shoot(str(tmp_path / "r"), 4)
    page = library(app)
    page.load_folder(str(tmp_path / "r"))
    assert pump(app, lambda: loaded(page), 30)
    page.select_path(page.files[0])
    page.open_review()
    rv = page.review
    assert page.body.currentIndex() == 1 and rv.path == page.files[0]
    assert pump(app, lambda: (rv.cache.get(rv.path) or {}).get("fit") is not None, 20)
    assert rv.view_a.chip == "Camera preview"

    def key(k, text=""):
        app.sendEvent(rv, QKeyEvent(QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier, text))
    key(Qt.Key.Key_4, "4")                                       # rate, then on to the next (on in Review)
    assert rv.path == page.files[1]
    key(Qt.Key.Key_X, "x")
    assert rv.path == page.files[2]
    key(Qt.Key.Key_Left)
    assert rv.path == page.files[1]
    assert page.writer.wait(5000)
    assert E.load_library_meta(page.files[0])["rating"] == 4
    assert E.load_library_meta(page.files[1])["pick"] == -1
    # 100% at the pointer; the zoom and position stay when moving on
    rv.view_a.set_zoom(True, 0.3, 0.6)
    assert pump(app, lambda: (rv.cache.get(rv.path) or {}).get("full") is not None, 20)
    full = rv.cache[rv.path]["full"]
    assert max(full.width(), full.height()) >= 3000
    key(Qt.Key.Key_Right)
    assert rv.view_a.zoom and abs(rv.view_a.cu - 0.3) < 1e-6
    # compare: two views, zoom locked together
    key(Qt.Key.Key_C, "c")
    assert rv.compare and rv.view_b.isVisible()
    rv.view_a.set_zoom(True, 0.7, 0.2)
    assert rv.view_b.zoom and abs(rv.view_b.cu - 0.7) < 1e-6 and abs(rv.view_b.cv - 0.2) < 1e-6
    key(Qt.Key.Key_C, "c")
    assert not rv.view_b.isVisible()
    key(Qt.Key.Key_Escape)
    assert page.body.currentIndex() == 0 and page.current_path() == rv.path
    page.close()


@needs_raws
def test_review_from_the_raw_and_small_previews(app, tmp_path, monkeypatch):
    from studio.app import review as R
    shoot(str(tmp_path / "r2"), 2)
    page = library(app)
    page.load_folder(str(tmp_path / "r2"))
    assert pump(app, lambda: loaded(page), 30)
    page.select_path(page.files[0])
    page.open_review()
    rv = page.review
    rv.mode_c.setCurrentIndex(rv.mode_c.findData("raw"))
    assert pump(app, lambda: (rv.cache.get(rv.path) or {}).get("raw") is not None, 90)
    app.processEvents()
    assert rv.view_a.chip == "From the raw" and rv.view_a.chip_colour == R.MODES["raw"][1]
    rv.mode_c.setCurrentIndex(rv.mode_c.findData("preview"))
    # a camera whose preview is small: 100% switches to the raw by itself, and says so
    monkeypatch.setattr(R, "SMALL_PREVIEW", 100000)
    rv.view_a.set_zoom(True, 0.5, 0.5)
    assert pump(app, lambda: rv.view_a.tile is not None, 90)
    assert "100% is shown from the raw" in rv.view_a.note and rv.view_a.chip == "From the raw"
    page.close()


# ------------------------------------------------------------------------------------------ 4b-4 Process picks
@needs_raws
def test_process_picks_exports_with_make_it_good_only_where_there_is_no_edit(app, tmp_path):
    from studio import engine as E
    from studio.app import process as P
    files = shoot(str(tmp_path / "pp"), 2)
    E.save_edit(files[1], E.Edit(exposure_ev=0.5))               # this one has its own edit
    spec = E.ExportSpec(format="jpeg", bits=8, space="srgb", long_edge=300)
    rep = P.process_files(files, spec, str(tmp_path / "out"), "auto")
    assert len(rep["out"]) == 2 and not rep["failed"] and rep["auto"] == 1
    assert all(os.path.isfile(p) for p in rep["out"])
    e0, e1 = E.load_edit(files[0]), E.load_edit(files[1])
    assert e0.sharpen.enabled and e0.gamut_compress              # Make it good was saved into the first
    assert e0.exposure_ev == 0.0                                 # and never touched brightness
    assert e1.exposure_ev == 0.5 and not e1.sharpen.enabled      # the edited one kept its own edit
    import threading
    ev = threading.Event()
    ev.set()
    assert P.process_files(files, spec, str(tmp_path / "out2"), "own", cancel=ev)["out"] == []


@needs_raws
def test_process_dialog_choices(app, tmp_path):
    from studio.app import common
    from studio.app.process import ProcessDialog
    st = common.Settings()
    st.d = {"export_dir": str(tmp_path)}
    dlg = ProcessDialog(st, ["a", "b"], ["c"], ["a", "b", "c", "d"])
    assert dlg.which() == "picks" and dlg.result_files() == ["a", "b"] and dlg.go.text() == "Process 2 photos"
    for b in dlg.which_g.buttons():
        if b.property("key") == "shown":
            b.setChecked(True)
    assert dlg.result_files() == ["a", "b", "c", "d"] and dlg.go.text() == "Process 4 photos"
    for b in dlg.mode_g.buttons():
        if b.property("key") == "auto":
            b.setChecked(True)
    assert dlg.mode() == "auto" and st.d["process_mode"] == "auto"
    dlg2 = ProcessDialog(st, [], ["c"], ["c"])
    assert dlg2.which() == "selected"


def test_make_it_good_lives_in_the_engine_and_never_touches_brightness():
    from studio import engine as E
    from studio.engine.presets.auto import make_it_good_settings
    meta = type("M", (), {"iso": 6400, "lens": "", "manual_lens": False, "iso_label": "ISO 6400"})()
    e, character = make_it_good_settings(meta, E.Edit(exposure_ev=0.3))
    assert e.denoise.enabled and e.banding and e.gamut_compress and e.sharpen.enabled and not character
    assert e.exposure_ev == 0.3
    meta.iso = 100
    e2, _ = make_it_good_settings(meta, E.Edit())
    assert not e2.denoise.enabled


# ------------------------------------------------------------------------------------------ 4b-1 dialogs
def test_import_dialog_end_to_end(app, tmp_path, monkeypatch):
    import datetime as dt
    from studio.app import importer as IM
    card = tmp_path / "EOS_DIGITAL"
    d = card / "DCIM" / "100CANON"
    d.mkdir(parents=True)
    when = dt.datetime(2026, 11, 14, 21, 0).timestamp()
    for n in ("IMG_0001.CR3", "IMG_0002.CR3", "IMG_0002.JPG"):
        (d / n).write_bytes(os.urandom(4000))
        os.utime(d / n, (when, when))
    monkeypatch.setenv("STUDIO_CARD_ROOTS", str(card))
    monkeypatch.setenv("STUDIO_PHOTOS_DIR", str(tmp_path / "Photos"))
    page = library(app)
    page.settings.d["photos_dir"] = str(tmp_path / "Photos")
    dlg = IM.open_import(page)
    assert pump(app, lambda: dlg.scan is not None and dlg.scan.confirmed and dlg.plan is not None, 10)
    app.processEvents()
    assert dlg.go.isEnabled() and dlg.go.text() == "Import 2 photos"
    assert dlg.left_lbl.text() == "1 JPEG stays on the card."
    dlg.day_rows["2026-11-14"][3].setText("2026-11-14 Kelly: candlelight")
    dlg.replan()
    assert "2026-11-14 Kelly candlelight" in dlg.saved_as.text()
    dlg.start()
    assert pump(app, lambda: not dlg.running, 20)
    assert dlg.report.all_checked and len(dlg.report.copied) == 2
    assert "You can take the card out" in dlg.s_safe.text()
    folder = tmp_path / "Photos" / "2026" / "2026-11-14 Kelly candlelight"
    assert sorted(os.listdir(folder)) == ["IMG_0001.CR3", "IMG_0002.CR3"]
    dlg._open_in_library()
    assert pump(app, lambda: page.folder == str(folder) and not page.loading, 10)
    assert len(page.files) == 2
    # again: nothing new
    dlg2 = IM.open_import(page)
    assert pump(app, lambda: dlg2.scan is not None and dlg2.scan.confirmed, 10)
    app.processEvents()
    assert not dlg2.go.isEnabled() and dlg2.reason.text() == "Nothing new to import."
    dlg2.close()
    page.close()


def test_photos_folder_dialog_creates_nothing_until_confirmed(app, tmp_path, monkeypatch):
    from studio.app import common
    from studio.app.importer import PhotosFolderDialog
    monkeypatch.setenv("STUDIO_PHOTOS_DIR", str(tmp_path / "Photos"))
    monkeypatch.setenv("STUDIO_DRIVE_ROOTS", str(tmp_path))
    st = common.Settings()
    st.d = {}
    dlg = PhotosFolderDialog(st)
    assert dlg.path.text() == str(tmp_path / "Photos") and not (tmp_path / "Photos").exists()
    assert "photos_dir" not in st.d
    dlg.path.setText(str(tmp_path / "Elsewhere" / "Photos"))
    dlg._ok()
    assert st.d["photos_dir"] == str(tmp_path / "Elsewhere" / "Photos") and (tmp_path / "Elsewhere" / "Photos").is_dir()
    once = PhotosFolderDialog(st, for_this_import=True)
    once.path.setText(str(tmp_path / "Once"))
    once._ok()
    assert once.chosen == str(tmp_path / "Once") and st.d["photos_dir"] == str(tmp_path / "Elsewhere" / "Photos")


def test_card_found_notice(app, tmp_path, monkeypatch):
    from studio.engine.io.ingest import CardWatcher
    a = tmp_path / "A"
    (a / "DCIM").mkdir(parents=True)
    monkeypatch.setenv("STUDIO_CARD_ROOTS", str(a))
    w = CardWatcher()
    assert w.poll() == []                                        # cards already in at start are not announced
    b = tmp_path / "B"
    (b / "DCIM" / "100CANON").mkdir(parents=True)
    (b / "DCIM" / "100CANON" / "IMG_0001.CR3").write_bytes(b"x" * 100)
    monkeypatch.setenv("STUDIO_CARD_ROOTS", os.pathsep.join([str(a), str(b)]))
    assert [c.root for c in w.poll()] == [str(b)]
    assert w.poll() == []                                        # not again while it stays in
    page = library(app)
    page.settings.d["photos_dir"] = str(tmp_path / "Photos")
    page._cards = CardWatcher()
    page._cards.poll()
    c = tmp_path / "C"
    (c / "DCIM").mkdir(parents=True)
    (c / "DCIM" / "IMG_0002.CR3").write_bytes(b"y" * 100)
    monkeypatch.setenv("STUDIO_CARD_ROOTS", os.pathsep.join([str(a), str(b), str(c)]))
    page._polling = False
    page._poll_cards()
    assert pump(app, lambda: page.notice.isVisible(), 10)
    assert "1 photo, 1 new" in page.notice_text.text()
    page.close()


def test_library_top_bar_fits_1366():
    """24.7 C: the Library still fits at 1366x768 (the sidebar takes 240 px)."""
    from PySide6.QtWidgets import QApplication
    from studio.app import common
    from studio.app.library import LibraryPage
    QApplication.instance() or QApplication([])
    st = common.Settings()
    st.d = {}
    page = LibraryPage(st)
    assert page.minimumSizeHint().width() <= 1366 - 240
    page.close()
