"""Phase 4b-1: Import from card (BUILD_PLAN.md section 24). The copy rules (24.6) and every named test in 24.9.

Cards are faked with folders (STUDIO_CARD_ROOTS): a DCIM tree of small files whose capture times come from
their modified time, or from a real EXIF date in a tiny TIFF (what a CR2 is) or JPEG.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys

import pytest

from studio.engine.io import ingest as I


# ------------------------------------------------------------------------------------------ fake cards
def put(path, data: bytes, when: dt.datetime | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    if when is not None:
        t = when.timestamp()
        os.utime(path, (t, t))
    return path


def tiff_raw(path, when: dt.datetime, extra: dict | None = None, pad: int = 2000):
    """A tiny TIFF with an EXIF-style date in IFD0, saved under a raw name (a CR2 is a TIFF)."""
    from PIL import Image
    info = {306: when.strftime("%Y:%m:%d %H:%M:%S")}
    info.update(extra or {})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = str(path) + ".tif"
    Image.new("RGB", (8 + pad % 7, 8)).save(tmp, tiffinfo=info)
    with open(tmp, "ab") as fh:                    # make files of different sizes and contents
        fh.write(os.urandom(pad))
    os.replace(tmp, path)
    os.utime(path, (when.timestamp(), when.timestamp()))
    return path


def card(tmp_path, files: dict, name="card") -> str:
    """files: {relative path under DCIM: (bytes or None for random, datetime)}"""
    root = tmp_path / name
    for rel, (data, when) in files.items():
        put(str(root / "DCIM" / rel), data if data is not None else os.urandom(3000 + len(rel) * 13), when)
    return str(root)


def tree_state(root) -> dict:
    """Every file under root: relative path -> (size, mtime, sha256). Used to prove the card never changes."""
    out = {}
    for d, _, files in os.walk(root):
        for n in files:
            p = os.path.join(d, n)
            st = os.stat(p)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns, hashlib.sha256(fh.read()).hexdigest())
    return out


def do_import(src, dest, **kw):
    scan = I.scan_source(src, videos=kw.pop("scan_videos", False))
    I.confirm_capture_times(scan)
    plan = I.plan_import(scan, I.ImportOptions(dest=str(dest), **kw))
    return plan, I.run_import(plan)


T1 = dt.datetime(2026, 11, 14, 21, 3, 55)


# ------------------------------------------------------------------------------------------ the named tests
def test_the_source_is_never_written_to(tmp_path):
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1), "100CANON/IMG_0002.CR3": (None, T1),
                          "100CANON/IMG_0002.JPG": (None, T1), "101CANON/MVI_0003.MP4": (None, T1)})
    tiff_raw(os.path.join(src, "DCIM", "100CANON", "IMG_0004.CR2"), T1)
    before = tree_state(src)
    plan, rep = do_import(src, tmp_path / "Photos", jpegs=True, videos=True)
    assert rep.all_checked and len(rep.copied) == 5
    assert tree_state(src) == before
    # and again with the source made read-only (a write attempt would raise)
    for d, dirs, files in os.walk(src):
        for n in files:
            os.chmod(os.path.join(d, n), stat.S_IREAD)
    try:
        plan, rep = do_import(src, tmp_path / "Photos2", jpegs=True, videos=True)
        assert rep.all_checked and len(rep.copied) == 5
        assert tree_state(src) == before
    finally:
        for d, dirs, files in os.walk(src):
            for n in files:
                os.chmod(os.path.join(d, n), stat.S_IREAD | stat.S_IWRITE)


def test_a_second_import_copies_nothing(tmp_path):
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1), "100CANON/IMG_0002.CR3": (None, T1)})
    _, rep = do_import(src, tmp_path / "Photos")
    assert len(rep.copied) == 2
    plan, rep2 = do_import(src, tmp_path / "Photos")
    assert plan.to_copy == [] and plan.imported_before == 2
    assert rep2.copied == [] and len(rep2.skipped) == 2
    # without the record, the destination itself stops a second copy (rule 4)
    os.remove(I.record_path(str(tmp_path / "Photos")))
    plan, rep3 = do_import(src, tmp_path / "Photos")
    assert [it.action for it in plan.items] == [I.EXISTS, I.EXISTS]
    assert rep3.copied == [] and [w for _, w in rep3.skipped] == ["already in the folder"] * 2
    folder = os.path.join(tmp_path, "Photos", "2026", "2026-11-14")
    assert sorted(os.listdir(folder)) == ["IMG_0001.CR3", "IMG_0002.CR3"]


def test_same_name_different_content_keeps_both(tmp_path):
    dest = tmp_path / "Photos"
    folder = dest / "2026" / "2026-11-14"
    put(str(folder / "IMG_0001.CR3"), b"an older photo, different size")
    put(str(folder / "IMG_0002.CR3"), b"B" * 3000)                    # same size as the card's, other content
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1), "100CANON/IMG_0002.CR3": (b"C" * 3000, T1),
                          "101CANON/IMG_0001.CR3": (None, T1)})       # counter wrapped: two different IMG_0001
    plan, rep = do_import(src, dest)
    names = sorted(os.listdir(folder))
    assert names == ["IMG_0001.CR3", "IMG_0001_2.CR3", "IMG_0001_3.CR3", "IMG_0002.CR3", "IMG_0002_2.CR3"]
    assert (folder / "IMG_0001.CR3").read_bytes() == b"an older photo, different size"   # never overwritten
    assert (folder / "IMG_0002.CR3").read_bytes() == b"B" * 3000
    assert (folder / "IMG_0002_2.CR3").read_bytes() == b"C" * 3000
    assert len(rep.renamed) == 3 and rep.all_checked
    # the dry run said so beforehand (one is decided when copying: same size)
    acts = {it.src.rel: (it.action, it.name, it.alt) for it in plan.items}
    assert acts["DCIM/100CANON/IMG_0002.CR3"] == (I.EXISTS, "IMG_0002.CR3", "IMG_0002_2.CR3")


def test_a_partial_left_by_a_crash_is_cleaned_and_copied_again(tmp_path):
    dest = tmp_path / "Photos"
    folder = dest / "2026" / "2026-11-14"
    put(str(folder / "IMG_0001.CR3.partial"), b"half a file")
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1)})
    _, rep = do_import(src, dest)
    assert sorted(os.listdir(folder)) == ["IMG_0001.CR3"]
    assert any(c.endswith("IMG_0001.CR3.partial") for c in rep.cleaned)
    with open(os.path.join(src, "DCIM", "100CANON", "IMG_0001.CR3"), "rb") as fh:
        assert (folder / "IMG_0001.CR3").read_bytes() == fh.read()


def test_a_read_error_on_one_file_does_not_stop_the_others(tmp_path, monkeypatch):
    src = card(tmp_path, {f"100CANON/IMG_000{i}.CR3": (None, T1) for i in range(1, 5)})
    real_open = open
    fails = {"IMG_0002.CR3": 99, "IMG_0003.CR3": 2}                 # 2 always fails; 3 fails twice, then reads

    def flaky_open(p, mode="r", *a, **k):
        n = os.path.basename(str(p))
        if "r" in mode and "b" in mode and os.path.dirname(str(p)).endswith("100CANON") and fails.get(n, 0) > 0:
            fails[n] -= 1
            raise OSError(5, "Input/output error", str(p))
        return real_open(p, mode, *a, **k)
    monkeypatch.setattr(I, "open", flaky_open, raising=False)
    _, rep = do_import(src, tmp_path / "Photos")
    assert [os.path.basename(r) for r, _ in rep.copied] == ["IMG_0001.CR3", "IMG_0003.CR3", "IMG_0004.CR3"]
    assert [os.path.basename(r) for r, _ in rep.failed] == ["IMG_0002.CR3"]
    assert "could not be read" in rep.failed[0][1]
    assert not rep.all_checked
    log = open(rep.log_path, encoding="utf-8").read()
    assert log.count("RETRY    DCIM/100CANON/IMG_0003.CR3") == 2


def test_a_card_pulled_mid_copy_leaves_no_half_written_file(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "BLOCK", 1024)
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (os.urandom(5000), T1),
                          "100CANON/IMG_0002.CR3": (os.urandom(9000), T1),
                          "100CANON/IMG_0003.CR3": (os.urandom(5000), T1)})
    real_open = open

    class Pulled:
        def __init__(self, fh):
            self.fh, self.n = fh, 0

        def readinto(self, b):
            self.n += 1
            if self.n == 3:                                         # the card disappears during the 2nd file
                shutil.move(src, str(tmp_path / "pocket"))
                raise OSError(19, "No such device")
            return self.fh.readinto(b)

        def __enter__(self):
            return self

        def __exit__(self, *e):
            self.fh.close()

    def pulling_open(p, mode="r", *a, **k):
        fh = real_open(p, mode, *a, **k)
        return Pulled(fh) if str(p).endswith("IMG_0002.CR3") and "r" in mode else fh
    monkeypatch.setattr(I, "open", pulling_open, raising=False)
    _, rep = do_import(src, tmp_path / "Photos")
    folder = tmp_path / "Photos" / "2026" / "2026-11-14"
    assert sorted(os.listdir(folder)) == ["IMG_0001.CR3"]            # no IMG_0002 and no .partial
    assert "taken out" in rep.stopped and not rep.all_checked
    # put it back: the import picks up where it left off
    shutil.move(str(tmp_path / "pocket"), src)
    monkeypatch.setattr(I, "open", real_open, raising=False)
    _, rep2 = do_import(src, tmp_path / "Photos")
    assert len(rep2.copied) == 2 and rep2.all_checked
    assert sorted(os.listdir(folder)) == ["IMG_0001.CR3", "IMG_0002.CR3", "IMG_0003.CR3"]


def test_too_little_space_blocks_the_start(tmp_path, monkeypatch):
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1)})
    monkeypatch.setattr(I, "free_space", lambda p: (3 * 1024 ** 3, 100 * 1024 ** 3))   # 3 GB free, 5 GB reserve
    plan, rep = do_import(src, tmp_path / "Photos")
    assert not plan.fits and not plan.can_start and plan.shortfall > 0
    assert rep.copied == [] and "not enough free space" in rep.stopped
    assert not (tmp_path / "Photos" / "2026").exists()
    monkeypatch.setattr(I, "free_space", lambda p: (50 * 1024 ** 3, 100 * 1024 ** 3))
    plan = I.plan_import(I.scan_source(src), I.ImportOptions(dest=str(tmp_path / "Photos"), second=str(tmp_path / "B")))
    assert plan.fits and plan.can_start


def test_raws_only_leaves_the_jpeg_of_a_pair_behind_and_says_so(tmp_path):
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1), "100CANON/IMG_0001.JPG": (None, T1),
                          "100CANON/MVI_0002.MP4": (None, T1)})
    plan, rep = do_import(src, tmp_path / "Photos")
    assert [os.path.basename(d) for _, d in rep.copied] == ["IMG_0001.CR3"]
    assert plan.left_text() == "1 JPEG and 1 video stay on the card."
    assert rep.left_text == "1 JPEG and 1 video stay on the card."
    assert any("stay on the card" in line for line in rep.summary_lines())


def test_a_card_with_no_raws_says_so(tmp_path):
    src = card(tmp_path, {"100CANON/IMG_0001.JPG": (None, T1), "100CANON/IMG_0002.JPG": (None, T1)})
    plan = I.plan_import(I.scan_source(src), I.ImportOptions(dest=str(tmp_path / "Photos")))
    assert plan.no_raws and plan.to_copy == [] and not plan.can_start
    assert plan.left_text() == "2 JPEGs stay on the card."
    plan = I.plan_import(I.scan_source(src), I.ImportOptions(dest=str(tmp_path / "Photos"), jpegs=True))
    assert len(plan.to_copy) == 2


def test_several_days_make_several_folders_and_2am_belongs_to_the_day_before(tmp_path):
    src = tmp_path / "card"
    d = src / "DCIM" / "100CANON"
    tiff_raw(d / "IMG_0001.CR2", dt.datetime(2026, 11, 14, 21, 0, 0))
    tiff_raw(d / "IMG_0002.CR2", dt.datetime(2026, 11, 15, 2, 0, 0))     # still the night shoot of the 14th
    tiff_raw(d / "IMG_0003.CR2", dt.datetime(2026, 11, 15, 10, 30, 0))
    for p in d.iterdir():                                                # file times are wrong; the photo is right
        os.utime(p, (dt.datetime(2026, 12, 1).timestamp(),) * 2)
    scan = I.scan_source(str(src))
    I.confirm_capture_times(scan)
    assert [f.taken_from for f in scan.files] == ["photo"] * 3
    plan = I.plan_import(scan, I.ImportOptions(dest=str(tmp_path / "Photos"),
                                               names={"2026-11-14": "2026-11-14 Kelly candlelight"}))
    assert [(g.key, g.total) for g in plan.groups] == [("2026-11-14", 2), ("2026-11-15", 1)]
    I.run_import(plan)
    p = tmp_path / "Photos" / "2026"
    assert sorted(os.listdir(p)) == ["2026-11-14 Kelly candlelight", "2026-11-15"]
    assert sorted(os.listdir(p / "2026-11-14 Kelly candlelight")) == ["IMG_0001.CR2", "IMG_0002.CR2"]
    # midnight rule moved, no year level, everything in one folder
    assert I.day_key(dt.datetime(2026, 11, 15, 2, 0), "00:00") == "2026-11-15"
    plan = I.plan_import(scan, I.ImportOptions(dest=str(tmp_path / "P2"), by_year=False, one_folder=True,
                                               names={"*": "Weekend"}))
    assert {it.folder for it in plan.to_copy} == {os.path.join(str(tmp_path / "P2"), "Weekend")}
    # a day whose tick is off stays on the card, and the summary says so
    plan = I.plan_import(scan, I.ImportOptions(dest=str(tmp_path / "P3"), skip_days=["2026-11-15"]))
    assert len(plan.to_copy) == 2 and plan.days_off == 1
    assert "1 file from unticked days stays on the card." == plan.left_text()


def test_unsafe_characters_in_a_folder_name_are_made_safe(tmp_path):
    assert I.safe_folder_name('Kelly: candle*light? <test> | "x". ', "2026-11-14") == "Kelly candlelight test x"
    assert I.safe_folder_name("a/b\\c", "d") == "abc"
    assert I.safe_folder_name("CON", "d") == "_CON"
    assert I.safe_folder_name("com1.txt", "d") == "_com1.txt"
    assert I.safe_folder_name(" ... ", "2026-11-14") == "2026-11-14"
    assert I.safe_folder_name("x" * 200, "d") == "x" * 80
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1)})
    plan = I.plan_import(I.scan_source(src), I.ImportOptions(dest=str(tmp_path / "Photos"),
                                                             names={"2026-11-14": "2026-11-14 Kelly: candle?"}))
    assert plan.groups[0].name == "2026-11-14 Kelly candle"
    assert plan.to_copy[0].folder.endswith(os.path.join("2026", "2026-11-14 Kelly candle"))


def test_a_deleted_photo_is_not_imported_again_unless_asked(tmp_path):
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1), "100CANON/IMG_0002.CR3": (None, T1)})
    _, rep = do_import(src, tmp_path / "Photos")
    os.remove(rep.copied[0][1])                                         # a reject deleted after import
    plan, rep2 = do_import(src, tmp_path / "Photos")
    assert rep2.copied == [] and plan.imported_before == 2
    plan, rep3 = do_import(src, tmp_path / "Photos", show_imported_before=True)
    assert [os.path.basename(d) for _, d in rep3.copied] == ["IMG_0001.CR3"]
    assert [w for _, w in rep3.skipped] == ["already in the folder"]


def test_the_second_copy_is_identical_to_the_first(tmp_path):
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1), "100CANON/IMG_0002.CR3": (None, T1),
                          "100CANON/IMG_0002.JPG": (None, T1)})
    _, rep = do_import(src, tmp_path / "Photos", second=str(tmp_path / "Backup"), jpegs=True)
    assert rep.all_checked and not rep.second_failed
    a, b = tmp_path / "Photos", tmp_path / "Backup"
    files_a = {os.path.relpath(os.path.join(d, n), a) for d, _, fs in os.walk(a) for n in fs if "_studio" not in d}
    files_b = {os.path.relpath(os.path.join(d, n), b) for d, _, fs in os.walk(b) for n in fs}
    assert files_a == files_b and len(files_a) == 3
    for rel in files_a:
        assert (a / rel).read_bytes() == (b / rel).read_bytes()
        assert os.path.getmtime(a / rel) == pytest.approx(os.path.getmtime(b / rel), abs=2)


def test_no_serial_number_in_the_log_or_the_import_record(tmp_path):
    serial = "SN-083021001234"
    src = tmp_path / "card"
    tiff_raw(src / "DCIM" / "100CANON" / "IMG_0001.CR2", T1, extra={42033: serial, 42032: "Bill"})
    _, rep = do_import(str(src), tmp_path / "Photos")
    assert rep.all_checked
    log = open(rep.log_path, encoding="utf-8").read()
    record = open(I.record_path(str(tmp_path / "Photos")), encoding="utf-8").read()
    assert serial not in log and serial not in record
    for line in record.splitlines():
        assert set(json.loads(line)) == {"name", "size", "taken", "hash", "to", "when"}


# ------------------------------------------------------------------------------------------ more of 24.3-24.8
def test_cards_are_found_by_their_dcim_folder(tmp_path, monkeypatch):
    a = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1)}, name="EOS_DIGITAL")
    plain = tmp_path / "usb_stick"
    plain.mkdir()
    monkeypatch.setenv("STUDIO_CARD_ROOTS", os.pathsep.join([a, str(plain), str(tmp_path / "missing")]))
    cards = I.find_cards()
    assert [c.root for c in cards] == [a] and cards[0].title == "EOS_DIGITAL"
    monkeypatch.setenv("STUDIO_DRIVE_ROOTS", os.pathsep.join([a, str(plain)]))
    drives = I.list_drives()
    assert [d.card for d in drives] == [True, False] and all(d.free > 0 for d in drives)


def test_plan_is_a_dry_run_and_matches_what_happens(tmp_path):
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1), "100CANON/IMG_0002.CR3": (None, T1)})
    dest = tmp_path / "Photos"
    plan = I.plan_import(I.scan_source(src), I.ImportOptions(dest=str(dest)))
    assert not dest.exists()                                            # nothing written by the plan
    rep = I.run_import(plan)
    assert [d for _, d in rep.copied] == [it.dest for it in plan.to_copy]
    assert plan.groups[0].new == 2 and plan.bytes_needed == sum(it.src.size for it in plan.to_copy)


def test_copies_keep_the_card_time_and_the_record_travels_with_the_folder(tmp_path):
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1)})
    _, rep = do_import(src, tmp_path / "Photos")
    final = rep.copied[0][1]
    assert os.path.getmtime(final) == pytest.approx(T1.timestamp(), abs=1)
    rec = [json.loads(x) for x in open(I.record_path(str(tmp_path / "Photos")), encoding="utf-8")]
    assert rec[0]["to"] == "2026/2026-11-14/IMG_0001.CR3" and rec[0]["hash"].startswith("blake2b:")
    # the Photos folder moved to another drive: the record (inside it) still knows the card
    shutil.move(str(tmp_path / "Photos"), str(tmp_path / "Moved"))
    plan = I.plan_import(I.scan_source(src), I.ImportOptions(dest=str(tmp_path / "Moved")))
    assert plan.imported_before == 1 and plan.to_copy == []


def test_source_inside_destination_and_missing_drive_are_refused(tmp_path):
    dest = tmp_path / "Photos"
    src = card(dest, {"100CANON/IMG_0001.CR3": (None, T1)})
    plan = I.plan_import(I.scan_source(src), I.ImportOptions(dest=str(dest)))
    assert plan.problem and not plan.can_start
    rep = I.run_import(plan)
    assert rep.copied == [] and rep.stopped


def test_cancel_stops_cleanly(tmp_path):
    import threading
    src = card(tmp_path, {f"100CANON/IMG_{i:04d}.CR3": (None, T1) for i in range(1, 6)})
    ev = threading.Event()
    seen = []

    def prog(p):
        seen.append(p["done"])
        if p["done"] >= 2:
            ev.set()
    scan = I.scan_source(src)
    plan = I.plan_import(scan, I.ImportOptions(dest=str(tmp_path / "Photos")))
    rep = I.run_import(plan, on_progress=prog, cancel=ev)
    folder = tmp_path / "Photos" / "2026" / "2026-11-14"
    assert rep.cancelled and len(rep.copied) == 2 and not rep.all_checked
    assert sorted(os.listdir(folder)) == ["IMG_0001.CR3", "IMG_0002.CR3"]   # nothing half-written
    _, rep2 = do_import(src, tmp_path / "Photos")
    assert len(rep2.copied) == 3


def test_a_copy_that_does_not_match_is_tried_again_then_failed(tmp_path, monkeypatch):
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1), "100CANON/IMG_0002.CR3": (None, T1)})
    real = I._hash_back
    bad = {"n": 0}

    def flaky(path, check):
        h, n = real(path, check)
        if "IMG_0001" in path and bad["n"] < 1:                 # one bad read-back: copied again, then fine
            bad["n"] += 1
            return "0" * 64, n
        if "IMG_0002" in path:                                  # always bad: failed
            return "0" * 64, n
        return h, n
    monkeypatch.setattr(I, "_hash_back", flaky)
    _, rep = do_import(src, tmp_path / "Photos")
    assert [os.path.basename(d) for _, d in rep.copied] == ["IMG_0001.CR3"]
    assert [os.path.basename(r) for r, _ in rep.failed] == ["IMG_0002.CR3"]
    folder = tmp_path / "Photos" / "2026" / "2026-11-14"
    assert sorted(os.listdir(folder)) == ["IMG_0001.CR3"]


def test_videos_outside_dcim_only_when_asked(tmp_path):
    src = card(tmp_path, {"100MSDCF/DSC00001.ARW": (None, T1)})
    put(os.path.join(src, "PRIVATE", "M4ROOT", "CLIP", "C0001.MP4"), b"v" * 500, T1)
    assert I.scan_source(src).count(I.VIDEO) == 0
    sc = I.scan_source(src, videos=True)
    assert sc.count(I.VIDEO) == 1
    plan = I.plan_import(sc, I.ImportOptions(dest=str(tmp_path / "Photos"), videos=True))
    assert len(plan.to_copy) == 2


def test_capture_time_from_jpeg_exif(tmp_path):
    from PIL import Image
    p = tmp_path / "card" / "DCIM" / "100CANON" / "IMG_0001.JPG"
    p.parent.mkdir(parents=True)
    ex = Image.Exif()
    ex[0x0132] = "2026:11:14 23:59:00"
    Image.new("RGB", (8, 8)).save(p, exif=ex)
    sc = I.scan_source(str(tmp_path / "card"))
    I.confirm_capture_times(sc)
    assert sc.files[0].taken == "2026-11-14 23:59:00" and sc.files[0].taken_from == "photo"


def test_cli_import_dry_run_then_import(tmp_path):
    from studio import cli
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1), "100CANON/IMG_0001.JPG": (None, T1)})
    dest = tmp_path / "Photos"
    assert cli.main(["import", src, "--to", str(dest), "--dry-run"]) == 0
    assert not dest.exists()
    assert cli.main(["import", src, "--to", str(dest), "--name", "Test shoot"]) == 0
    assert os.listdir(dest / "2026" / "Test shoot") == ["IMG_0001.CR3"]
    assert cli.main(["import", src, "--to", str(dest)]) == 0           # nothing new: still fine


def test_photos_folder_and_sidecars_are_never_committed():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(os.path.join(root, ".git")) or shutil.which("git") is None:
        pytest.skip("not a git copy")
    for rel in ("Photos/2026/2026-11-14/IMG_0001.CR3", "IMG_0001.CR3.psb.json",
                "studio/IMG_0001.CR3.psb.json", "Photos/_studio/imports.jsonl"):
        r = subprocess.run(["git", "-C", root, "check-ignore", "-q", "--no-index", rel], capture_output=True)
        assert r.returncode == 0, f"{rel} would be committed"


def test_photos_dir_is_not_created_until_confirmed(monkeypatch, tmp_path):
    from studio import paths
    monkeypatch.setenv("STUDIO_PHOTOS_DIR", str(tmp_path / "MyPhotos"))
    assert paths.photos_dir() == tmp_path / "MyPhotos"
    assert I.default_photos_dir() == str(tmp_path / "MyPhotos")
    assert not (tmp_path / "MyPhotos").exists()
    monkeypatch.delenv("STUDIO_PHOTOS_DIR")
    assert paths.photos_dir() == paths.DATA_ROOT / "Photos"


def test_onedrive_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("OneDrive", str(tmp_path / "OneDrive"))
    assert I.inside_onedrive(str(tmp_path / "OneDrive" / "Pictures" / "Photos"))
    assert not I.inside_onedrive(str(tmp_path / "Photos"))
    src = card(tmp_path, {"100CANON/IMG_0001.CR3": (None, T1)})
    plan = I.plan_import(I.scan_source(src), I.ImportOptions(dest=str(tmp_path / "OneDrive" / "Photos")))
    assert any("OneDrive" in w for w in plan.warnings) and plan.can_start


if sys.platform == "win32":
    def test_uncached_read_back_on_windows(tmp_path):
        """Windows only: the read-back past the file cache agrees with a normal read (it is used only if so)."""
        p = tmp_path / "x.bin"
        data = os.urandom(5 * 1024 * 1024 + 777)
        p.write_bytes(data)
        assert I._hash_uncached(str(p)) == I._hash_cached(str(p))
        assert I._choose_check(str(tmp_path / "_studio")) == I.CHECK_UNCACHED


def test_frames_left_out_of_a_stack_go_into_the_days_left_out_folder(tmp_path):
    """Sort focus stacks at import (Bill 2026-11-17: copy everything, the left-out frames in a separate folder):
    they are copied and checked like the rest, into <day>/Left out, on the second copy too; Open in Library shows
    the day itself; a second import copies nothing; the card is never written to."""
    src = card(tmp_path, {f"100CANON/IMG_{i:04d}.CR3": (None, T1) for i in range(1, 6)})
    before = tree_state(src)
    out = {"dcim/100canon/img_0002.cr3": "soft everywhere", "dcim/100canon/img_0004.cr3": "adds nothing"}
    scan = I.scan_source(src)
    I.confirm_capture_times(scan)
    rels = {f.rel.lower() for f in scan.files}
    out = {k: v for k, v in out.items() if k in rels} or {f.rel.lower(): "soft everywhere" for f in scan.files[1:4:2]}
    plan, rep = do_import(src, tmp_path / "Photos", left_out=out, second=str(tmp_path / "Backup"))
    assert rep.all_checked and len(rep.copied) == 5 and rep.left_out == 2
    day = os.path.join(tmp_path, "Photos", "2026", "2026-11-14")
    assert sorted(os.listdir(os.path.join(day, I.LEFT_OUT_DIR))) == sorted(os.path.basename(k).upper()
                                                                          for k in out)
    assert len([n for n in os.listdir(day) if n.endswith(".CR3")]) == 3
    assert os.path.isdir(os.path.join(tmp_path, "Backup", "2026", "2026-11-14", I.LEFT_OUT_DIR))
    assert rep.folders == [day]
    assert any("Left out of focus stacks" in ln for ln in rep.summary_lines())
    assert all("left out of a stack" in it.note for it in plan.items if os.path.basename(it.folder) == I.LEFT_OUT_DIR)
    plan2, rep2 = do_import(src, tmp_path / "Photos", left_out=out)
    assert plan2.to_copy == [] and rep2.copied == []
    assert tree_state(src) == before
