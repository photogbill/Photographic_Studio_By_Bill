"""Polish pass before the wider test round (2026-11-16). Each test is a bug a tester would have met:

* every background job stayed in memory for the whole session, and with it every photo opened in Develop
  (hundreds of MB each): the Studio grew until it stopped;
* an export replaced whatever had its name: the camera's own JPEG beside a raw, the scan being exported, the first
  of two photos with the same name (now every export is named NAME_Studio, and never replaces a stranger's file);
* one unreadable photo, or a folder that could not be made, ended a whole export;
* a photo that was only looked at got a sidecar (and "could not save" on a read-only card);
* a raw the decoder cannot read left "Opening ..." on screen for good, with the decoder's own cryptic text;
* a first start showed a blank Library.
"""
from __future__ import annotations

import gc
import os
import time
import weakref

import numpy as np
import pytest

from conftest import scene, synthetic_raw

from studio import engine as E
from studio.engine.io import export as X
from studio.engine.pipeline.pipeline import Image


def make_image(path="synthetic.cr2"):
    img = Image(synthetic_raw(scene(64, 96)))
    img.raw.path = path
    return img


# ------------------------------------------------------------------------------------------ exports destroy nothing
def _camera_jpeg(path):
    from PIL import Image as PI
    PI.fromarray(np.full((8, 12, 3), 90, np.uint8)).save(path, "JPEG")
    return open(path, "rb").read()


def test_every_export_is_named_after_its_photo_with_studio_added(tmp_path):
    """Bill's rule (2026-11-16): each export has its own name, the original's with _Studio added. So a raw exported
    as JPEG into its own folder cannot have the name of the camera's JPEG beside it."""
    theirs = tmp_path / "IMG_0001.jpg"
    before = _camera_jpeg(str(theirs))
    img = make_image(str(tmp_path / "IMG_0001.CR2"))
    out = E.export(img, E.Edit(), E.ExportSpec(format="jpeg"), out_dir=str(tmp_path))
    assert os.path.basename(out) == "IMG_0001_Studio.jpg"
    assert theirs.read_bytes() == before
    assert X.is_own_export(out) and not X.is_own_export(str(theirs))
    assert X.export_name("D:/shoot/old print.JPG", E.ExportSpec(format="tiff")) == "old print_Studio.tif"
    assert X.export_name("x/IMG_7_Studio.tif", E.ExportSpec(format="png")) == "IMG_7_Studio.png"      # not twice
    assert X.export_name("x/IMG_7.CR3", E.ExportSpec(format="tiff"), extra="Dark_one") == "IMG_7_Dark_one_Studio.tif"


def test_a_strangers_file_with_the_exports_name_is_kept(tmp_path):
    """The safety net behind the naming rule: a file the Studio did not write is never replaced, whatever it is
    called. The export goes beside it as NAME-2."""
    theirs = tmp_path / "IMG_0003_Studio.jpg"
    before = _camera_jpeg(str(theirs))
    out = E.export(make_image(str(tmp_path / "IMG_0003.CR2")), E.Edit(), E.ExportSpec(format="jpeg"),
                   out_dir=str(tmp_path))
    assert os.path.basename(out) == "IMG_0003_Studio-2.jpg"
    assert theirs.read_bytes() == before


@pytest.mark.parametrize("fmt,bits,ext", [("tiff", 16, ".tif"), ("png", 16, ".png"), ("png", 8, ".png"),
                                          ("jpeg", 8, ".jpg")])
def test_exporting_again_replaces_the_studios_own_file(tmp_path, fmt, bits, ext):
    """Export, change a slider, export again: one file, the new one (no pile of numbered copies)."""
    img = make_image(str(tmp_path / "raws" / "IMG_0002.CR2"))
    spec = E.ExportSpec(format=fmt, bits=bits)
    a = E.export(img, E.Edit(), spec, out_dir=str(tmp_path))
    first = open(a, "rb").read()
    b = E.export(img, E.Edit(exposure_ev=1.0), spec, out_dir=str(tmp_path))
    assert a == b == str(tmp_path / ("IMG_0002_Studio" + ext))
    assert open(b, "rb").read() != first
    assert sorted(os.listdir(tmp_path)) == ["IMG_0002_Studio" + ext]


def test_two_photos_with_one_name_in_a_batch_both_survive(tmp_path):
    """IMG_0001 from two cameras (or a raw and a scan) exported together: two files, also the second time round."""
    a_img, b_img = make_image(str(tmp_path / "a" / "IMG_0001.CR2")), make_image(str(tmp_path / "b" / "IMG_0001.CR2"))
    for _ in range(2):
        taken: set = set()
        a = E.export(a_img, E.Edit(), E.ExportSpec(), out_dir=str(tmp_path), taken=taken)
        b = E.export(b_img, E.Edit(exposure_ev=1.0), E.ExportSpec(), out_dir=str(tmp_path), taken=taken)
        assert [os.path.basename(a), os.path.basename(b)] == ["IMG_0001_Studio.tif", "IMG_0001_Studio-2.tif"]
    assert sorted(os.listdir(tmp_path)) == ["IMG_0001_Studio-2.tif", "IMG_0001_Studio.tif"]


def test_an_export_opened_again_and_exported_beside_itself_is_not_replaced(tmp_path):
    """Old-photo work on an earlier export: plate_Studio.tif opened and exported as TIFF into its own folder. It
    keeps its name once (no _Studio_Studio) and the file being worked from stays as it is."""
    src = E.export(make_image(str(tmp_path / "x" / "plate.CR2")), E.Edit(), E.ExportSpec(), out_dir=str(tmp_path))
    assert os.path.basename(src) == "plate_Studio.tif" and X.is_own_export(src)
    before = open(src, "rb").read()
    scan = E.open(src)
    out = E.export(scan, E.Edit(exposure_ev=0.5), E.ExportSpec(), out_dir=str(tmp_path))
    assert os.path.basename(out) == "plate_Studio-2.tif"
    assert open(src, "rb").read() == before


def test_one_bad_photo_does_not_end_an_export(tmp_path, monkeypatch):
    """The window's Export: a photo that cannot be opened is reported, the others are still exported."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
    from PySide6.QtGui import QDesktopServices
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(M.Settings, "set", lambda self, k, v: self.d.__setitem__(k, v))
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda *a: True))
    good = make_image(str(tmp_path / "good.CR2"))

    def fake_open(path):
        if path.endswith("bad.CR2"):
            raise E.UnsupportedRaw("bad.CR2: damaged")
        return good
    monkeypatch.setattr(E, "open", fake_open)
    dest = tmp_path / "out"
    monkeypatch.setattr(M.ExportDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(M.ExportDialog, "spec", lambda self: E.ExportSpec(format="jpeg"))
    w = M.MainWindow()
    try:
        orig_init = M.ExportDialog.__init__

        def init(self, files, settings, parent=None):
            orig_init(self, files, settings, parent)
            self.dest.setText(str(dest))
        monkeypatch.setattr(M.ExportDialog, "__init__", init)
        w.export([str(tmp_path / "bad.CR2"), str(tmp_path / "good.CR2")])
        t = time.time()
        while not (dest / "good_Studio.jpg").exists() and time.time() - t < 60:
            app.processEvents()
            time.sleep(0.02)
        for _ in range(20):
            app.processEvents()
            time.sleep(0.01)
        assert (dest / "good_Studio.jpg").exists()
        assert "1 could not be exported: bad.CR2" in w.statusBar().currentMessage()
        # a folder that cannot be made: said in a box, not an error from the depths
        said = []
        monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: said.append(a[2])))
        blocker = tmp_path / "file.txt"
        blocker.write_text("x")
        dest = blocker / "sub"
        w.export([str(tmp_path / "good.CR2")])
        assert said and "cannot be used" in said[0]
    finally:
        w.close()


# ------------------------------------------------------------------------------------------ memory
def test_background_jobs_are_forgotten_once_their_result_is_delivered():
    """Every job (and the photo its work held) used to stay in memory until the Studio closed."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio.app import common as C
    app = QApplication.instance() or QApplication([])
    C.POOL.waitForDone(20000)
    app.processEvents()

    class Photo:
        pass
    photo = Photo()
    held = weakref.ref(photo)
    got = []
    before = len(C._LIVE)
    C.run_bg(lambda p=photo: id(p), "ok", lambda tok, r: got.append(tok))
    C.run_bg(lambda p=photo: 1 / 0, "bad", None, lambda tok, err: got.append(tok))
    C.run_bg(lambda p=photo: None, "nobody listens")
    del photo
    t = time.time()
    while (len(got) < 2 or len(C._LIVE) > before) and time.time() - t < 20:
        app.processEvents()
        time.sleep(0.01)
    gc.collect()
    assert sorted(got) == ["bad", "ok"]
    assert len(C._LIVE) <= before
    assert held() is None


# ------------------------------------------------------------------------------------------ sidecars and settings
class _FakeImage:
    meta = type("M", (), {"lens": "", "iso": 100, "iso_label": "ISO 100", "manual_lens": True})()


def test_a_photo_that_was_only_looked_at_gets_no_sidecar(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(M.Settings, "set", lambda self, k, v: self.d.__setitem__(k, v))
    page = M.DevelopPage(M.Settings())
    page.schedule = lambda: None
    page.render_now = lambda: None
    raw = tmp_path / "IMG_0009.CR3"
    raw.write_bytes(b"x")
    side = str(raw) + ".psb.json"
    page.path, page.image = str(raw), _FakeImage()
    page.controls_from_edit(page.edit)
    page._reset_history(page.edit)
    page._canonical()
    page._mark_saved()                                   # what opening a photo does
    page.save_sidecar()
    assert not os.path.exists(side)                      # nothing changed: nothing written
    page.ev.set_value(0.6)
    page.save_sidecar()
    assert os.path.exists(side) and E.load_edit(str(raw)).exposure_ev == 0.6
    stamp = os.stat(side).st_mtime_ns
    time.sleep(0.02)
    page.save_sidecar()                                  # saved already: not written again
    assert os.stat(side).st_mtime_ns == stamp
    page.do_undo()
    page.save_sidecar()                                  # undo is a change too
    assert E.load_edit(str(raw)).exposure_ev == 0
    app.processEvents()


def test_settings_are_swapped_in_whole(tmp_path, monkeypatch):
    """app.json is written under another name and swapped in; a value that cannot be saved does not empty it."""
    pytest.importorskip("PySide6")
    from studio.app import common as C
    monkeypatch.setenv("STUDIO_SETTINGS_DIR", str(tmp_path))
    st = C.Settings()
    _REAL_SET(st, "photos_dir", "E:/Photos")
    _REAL_SET(st, "broken", object())                    # not JSON: refused, the file keeps what it had
    assert not (tmp_path / "app.json.partial").exists() or (tmp_path / "app.json").exists()
    again = C.Settings()
    assert again.get("photos_dir") == "E:/Photos"


try:                                                     # the real Settings.set (other test files replace it)
    from studio.app.common import Settings as _S
    _REAL_SET = _S.set
except Exception:                                        # noqa: BLE001  (no PySide6: those tests skip)
    _REAL_SET = None


# ------------------------------------------------------------------------------------------ files that will not open
def test_an_unreadable_raw_says_why_in_plain_words(tmp_path):
    pytest.importorskip("rawpy")
    junk = tmp_path / "DSC_0001.NEF"
    junk.write_bytes(b"not a raw file at all" * 400)
    with pytest.raises(E.UnsupportedRaw) as err:
        E.open(str(junk))
    text = str(err.value)
    assert "DSC_0001.NEF" in text and "b'" not in text
    empty = tmp_path / "IMG_0001.CR2"
    empty.write_bytes(b"")
    with pytest.raises(E.UnsupportedRaw) as err:
        E.open(str(empty))
    assert "IMG_0001.CR2" in str(err.value) and "b'" not in str(err.value)


def test_a_photo_that_will_not_open_says_so_where_the_photo_would_be(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    pytest.importorskip("rawpy")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(M.Settings, "set", lambda self, k, v: self.d.__setitem__(k, v))
    page = M.DevelopPage(M.Settings())
    junk = tmp_path / "DSC_0002.NEF"
    junk.write_bytes(b"junk" * 2000)
    said = []
    page.status.connect(said.append)
    page.open(str(junk))
    t = time.time()
    while not said and time.time() - t < 20:
        app.processEvents()
        time.sleep(0.01)
    assert said and "DSC_0002.NEF" in said[0]
    assert "could not be opened" in page.viewer.message and "Opening" not in page.viewer.message
    assert page.path is None                             # the same photo can be tried again
    assert not os.path.exists(str(junk) + ".psb.json")


def test_lumix_rw2_header_is_read(tmp_path):
    """Panasonic RW2 is a TIFF with its own magic number (0x55): camera and exposure are read, not left blank."""
    from conftest import tiff_bytes
    from studio.engine.io.metadata import read_metadata
    data = bytearray(tiff_bytes({0x010F: (2, "Panasonic"), 0x0110: (2, "DC-G9")},
                                sub_ifds={0x8769: {0x829A: (5, [(1, 250)]), 0x8827: (3, [400])}}))
    data[2:4] = b"\x55\x00"
    p = tmp_path / "P1000001.RW2"
    p.write_bytes(bytes(data))
    md = read_metadata(str(p))
    assert md.make == "Panasonic" and md.model == "DC-G9" and md.iso == 400 and abs(md.exposure_s - 1 / 250) < 1e-9


# ------------------------------------------------------------------------------------------ first start
def test_an_empty_library_says_what_to_do(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(M.Settings, "set", lambda self, k, v: self.d.__setitem__(k, v))
    lib = M.LibraryPage(M.Settings())
    lib.resize(1200, 800)
    lib.show()
    lib.load_folder(str(tmp_path))
    t = time.time()
    while lib.loading and time.time() - t < 20:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert lib.files == [] and lib.empty.isVisible()
    assert "Open folder" in lib.empty.text() and "Import from card" in lib.empty.text()
    lib.model.set_entries([{"path": str(tmp_path / "a.CR2"), "name": "a.CR2", "size": 1, "mtime": 0.0,
                            "rating": 0, "pick": 0, "label": "", "lens_tag": "", "has_edit": 0, "taken": "",
                            "camera": "", "lens": "", "iso": 100, "exposure": None, "fnumber": None, "focal": None}])
    lib.apply_filter()
    assert not lib.empty.isVisible()
    lib.filter.setCurrentIndex(lib.filter.findText("Picks only"))
    app.processEvents()
    assert lib.empty.isVisible() and "match the filters" in lib.empty.text()
    lib.meta.stop()
    lib.hide()


# ------------------------------------------------------------------------------------------ speed, layout, installer
def test_measurements_asked_for_together_are_made_once(monkeypatch):
    """Opening a photo starts the first render and the lateral-CA, noise and blur measurements at the same time.
    Each thread used to make its own copy of a measurement it needed (the CA estimate ran twice per photo)."""
    import threading
    from studio.engine.lens import auto
    img = Image(synthetic_raw(scene(128, 192)))
    calls = []
    real = auto.estimate_lateral_ca

    def slow(*a, **k):
        calls.append(1)
        time.sleep(0.3)                                  # long enough for the other threads to arrive
        return real(*a, **k)
    monkeypatch.setattr(auto, "estimate_lateral_ca", slow)
    cas, got = [], []
    threads = [threading.Thread(target=lambda: cas.append(img.lateral_ca())) for _ in range(4)]
    threads.append(threading.Thread(target=lambda: got.append(E.render(img, E.Edit(), scale=0.5).shape)))
    threads.append(threading.Thread(target=lambda: got.append(img.capture_blur(E.Edit()))))
    threads.append(threading.Thread(target=lambda: got.append(img.noise_model())))
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert len(cas) == 4 and len(got) == 3               # nobody was left waiting for somebody else
    assert len(calls) == 1
    assert all(c is cas[0] for c in cas)


def test_develop_panel_fits_before_a_photo_is_open_and_with_a_composite_subject():
    """The panel was wider than its space (text and boxes cut off at the right) before the first photo was opened,
    and whenever a composite subject was selected: its two buttons did not fit side by side."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QScrollArea
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    M.Settings.set = lambda self, k, v: self.d.__setitem__(k, v)
    page = M.DevelopPage(M.Settings())
    page.resize(1400, 900)
    page.show()
    app.processEvents()
    scroll = page.findChild(QScrollArea)
    assert not page.cp_btns.isVisibleTo(page) and not page.st_amt.isVisibleTo(page)     # tools that are off
    for level in (0, 1):
        page.level.setCurrentIndex(level)
        page.apply_level()
        for shown in (False, True):
            for w_ in (page.cp_list, page.cp_x, page.cp_y, page.cp_size, page.cp_rot, page.cp_soft, page.cp_row,
                       page.cp_btns, page.cp_mamt):
                w_.setVisible(shown)
            app.processEvents()
            assert scroll.widget().minimumSizeHint().width() <= scroll.viewport().width(), (level, shown)
    page.close()


def test_windows_installer_checks_drive_d_and_free_space():
    """D: is often a memory card, a USB stick or a small recovery partition: it is the default only when it is an
    internal drive with room. Windows scripts keep CRLF line endings."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "Install Photographic Studio.bat"), "rb") as fh:
        data = fh.read()
    assert b"\n" not in data.replace(b"\r\n", b""), "the installer needs CRLF line endings"
    text = data.decode("ascii")
    assert "DriveType -eq 'Fixed'" in text and "AvailableFreeSpace -gt 5GB" in text
    assert 'if exist "D:\\" set "DEST=' not in text
    assert "AvailableFreeSpace -lt 2GB" in text and ":room_ok" in text
    # a new install opens the folder window first (Bill, 2026-11-16: the user picks where it installs); an update
    # run from inside a Studio folder does not
    fresh = text[text.index('set "SUGGEST=%HERE%"'):text.index("\r\n:confirm\r\n")]
    assert "goto :browse" in fresh and "$o.TopMost = $true" in text and "$env:SUGGESTSEL" in text
    update = text[text.index('if exist "%HERE%\\studio\\__init__.py" ('):text.index('set "SUGGEST=%HERE%"')]
    assert "goto :confirm" in update and "goto :browse" not in update
    # every label that is jumped to exists
    import re
    labels = re.findall(r"(?m)^:(\w+)", text)
    assert len(labels) == len(set(labels)), "a label appears twice"
    for target in re.findall(r"goto :(\w+)", text):
        assert target in labels, target


# ------------------------------------------------------------------------------------------ error reports
def _fresh_reports(monkeypatch, tmp_path):
    from studio import log as slog
    monkeypatch.setenv("STUDIO_ERROR_DIR", str(tmp_path / "error_reports"))
    monkeypatch.setattr(slog, "_reported", {})
    return slog


def test_an_unexpected_error_saves_a_dated_report_by_itself(tmp_path, monkeypatch):
    """Bill (2026-11-16): error reports generate automatically into a folder, date and time in the file name."""
    import re
    slog = _fresh_reports(monkeypatch, tmp_path)

    def fault(n):
        try:
            return [1, 2, 3][n]
        except IndexError:
            return slog.save_error_report("Testing the report")
    first = fault(7)
    assert first is not None and first.parent == tmp_path / "error_reports"
    assert re.fullmatch(r"error_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(_\d+)?\.txt", first.name)
    text = first.read_text(encoding="utf-8")
    assert "IndexError" in text and "What:  Testing the report" in text and "Photographic Studio By Bill" in text
    assert "numpy" in text                                # the versions and the end of the log come with it
    assert fault(9) == first                              # the same fault again: no second file
    try:
        {}["missing"]
    except KeyError:
        other = slog.save_error_report("Another fault")   # a different fault, the same second: its own file
    assert other is not None and other != first
    assert sorted(f.name for f in (tmp_path / "error_reports").iterdir()) == sorted([first.name, other.name])
    assert slog.latest_error_report() in (first, other)
    assert not list((tmp_path / "error_reports").glob("*.partial"))


def test_a_failed_job_reports_itself_and_expected_problems_do_not(tmp_path, monkeypatch):
    """A fault in background work writes the report and tells the window; a raw that cannot be read, a read-only
    card or no internet is not a fault of the Studio and writes none."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio.app import common as C
    app = QApplication.instance() or QApplication([])
    _fresh_reports(monkeypatch, tmp_path)
    told, failed = [], []
    C.BUS.report_saved.connect(told.append)

    def unreadable():
        raise E.UnsupportedRaw("x.RAF: a Fujifilm X-Trans sensor")

    def read_only():
        raise PermissionError("read-only card")

    def fault():
        raise ValueError("region outside the image")
    try:
        for fn in (unreadable, read_only, fault):
            C.run_bg(fn, fn.__name__, None, lambda tok, err: failed.append(tok))
        t = time.time()
        while (len(failed) < 3 or not told) and time.time() - t < 20:
            app.processEvents()
            time.sleep(0.01)
        for _ in range(10):
            app.processEvents()
    finally:
        C.BUS.report_saved.disconnect(told.append)
    assert sorted(failed) == ["fault", "read_only", "unreadable"]
    files = list((tmp_path / "error_reports").glob("error_*.txt"))
    assert len(files) == 1 and told == [str(files[0])]
    text = files[0].read_text(encoding="utf-8")
    assert "ValueError: region outside the image" in text and "Background work failed (fault)" in text


def test_a_hard_crash_becomes_a_report_at_the_next_start(tmp_path, monkeypatch):
    """A crash inside Qt or the raw decoder leaves no Python error; what faulthandler wrote as the Studio died is
    turned into an error report the next time it starts."""
    import faulthandler
    slog = _fresh_reports(monkeypatch, tmp_path)
    monkeypatch.setenv("STUDIO_LOG_DIR", str(tmp_path / "logs"))
    (tmp_path / "logs").mkdir()
    was_on = faulthandler.is_enabled()
    try:
        assert slog.watch_for_crashes() is None           # a normal last run: nothing to report
        slog._crash_file.close()
        (tmp_path / "logs" / "crash_trace.log").write_text(
            'Fatal Python error: Segmentation fault\n\nThread 0x1 (most recent call first):\n'
            '  File "studio/engine/raw/demosaic.py", line 250 in rcd\n', encoding="utf-8")
        rep = slog.watch_for_crashes()
        assert rep is not None and "Segmentation fault" in rep.read_text(encoding="utf-8")
        assert "stopped unexpectedly" in rep.read_text(encoding="utf-8")
        assert (tmp_path / "logs" / "crash_trace.log").stat().st_size == 0      # reported once
    finally:
        faulthandler.disable()
        if slog._crash_file is not None:
            slog._crash_file.close()
            slog._crash_file = None
        if was_on:
            faulthandler.enable()


# ------------------------------------------------------------------------------------------ the push comment
def test_the_push_comment_writes_itself(tmp_path):
    """Bill (2026-11-16): push stays just push; the comment about the changes is filled in from the lines written
    as each part was built (docs/NEXT_UPDATE.md), and each update's lines are filed in CHANGELOG.md."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("whats_new", os.path.join(root, "tools", "whats_new.py"))
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    pend, log = tmp_path / "NEXT_UPDATE.md", tmp_path / "CHANGELOG.md"
    pend.write_text(W.PENDING_HEADER, encoding="utf-8")
    assert W.pending_lines(pend) == []                    # the examples in the header are not changes
    W.add_line("New: Focus Stack merges a focus bracket.", pend)
    W.add_line("Fixed:   a  thing.", pend)
    assert W.pending_lines(pend) == ["New: Focus Stack merges a focus bracket.", "Fixed: a thing."]
    msg = W.commit_message(W.pending_lines(pend))
    assert msg.splitlines()[0] == "Update: New: Focus Stack merges a focus bracket. (and 1 more)"
    assert "- Fixed: a thing." in msg
    assert W.release("v0.2.0", pend, log, today="2026-11-16") == 2
    assert W.pending_lines(pend) == []                    # the waiting list is empty again
    W.add_line("New: Super-Res Stack.", pend)
    assert W.release("0.2.1", pend, log, today="2026-11-17") == 1
    text = log.read_text(encoding="utf-8")
    assert text.index("## 0.2.1 (2026-11-17)") < text.index("## 0.2.0 (2026-11-16)")     # newest first
    assert W.notes("0.2.0", log) == "- New: Focus Stack merges a focus bracket.\n- Fixed: a thing.\n"
    assert W.notes("9.9.9", log) == "" and W.release("0.2.2", pend, log) == 0
    # the scripts that use it: CRLF line endings, every label that is jumped to exists
    import re
    for name, must in (("push.bat", ("whats_new.py\" --message", ":ask_msg", ":msg_done")),
                       ("Release.bat", ("whats_new.py\" --release %NEWV%", "git add CHANGELOG.md docs/NEXT_UPDATE.md"))):
        with open(os.path.join(root, name), "rb") as fh:
            data = fh.read()
        assert b"\n" not in data.replace(b"\r\n", b""), f"{name} needs CRLF line endings"
        bat = data.decode("ascii")
        for m in must:
            assert m in bat, (name, m)
        labels = re.findall(r"(?m)^:(\w+)", bat)
        assert len(labels) == len(set(labels))
        for target in re.findall(r"goto :(\w+)", bat):
            assert target in labels, (name, target)
    for name in ("docs/NEXT_UPDATE.md", "CHANGELOG.md", "CLAUDE.md"):
        assert os.path.isfile(os.path.join(root, name)), name
