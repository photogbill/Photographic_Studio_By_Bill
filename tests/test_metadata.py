"""True ISO (section 4): 65535 in EXIF must never reach the UI or a profile lookup."""
from __future__ import annotations

import struct

import pytest

from conftest import canon_shotinfo, find_sample_raws, tiff_bytes
from studio.engine.io import metadata as M
from studio.engine.io.tiffmeta import CANON_UUID, read_tiff_meta


def test_invalid_iso_values_are_rejected():
    for bad in (65535, 32767, 0, -100, None, "abc", float("nan")):
        assert M.valid_iso(bad) is None
    assert M.valid_iso(102400) == 102400
    assert M.valid_iso((6400, 6400)) == 6400


def test_snap_iso_to_third_stops():
    assert M.snap_iso(31948.7) == 32000
    assert M.snap_iso(102400.0) == 102400
    assert M.snap_iso(1250.4) == 1250


@pytest.mark.parametrize("iso", [100, 125, 1600, 6400, 32000, 51200, 102400, 204800])
def test_canon_shotinfo_decodes_known_isos(iso):
    import math
    base_raw = round(32 * math.log2(iso * 32 / 100.0))
    assert M.canon_shotinfo_iso(tuple(canon_shotinfo(base_raw))) == iso


def test_resolve_prefers_valid_exif_then_32bit_then_makernote():
    assert M.resolve_iso({"exif_iso": 6400, "canon_shotinfo": 6400.0}) == (6400, "exif_iso")
    assert M.resolve_iso({"exif_iso": 65535, "exif_rei": 102400}) == (102400, "exif_rei")
    assert M.resolve_iso({"exif_iso": 65535, "canon_shotinfo": 102400.0}) == (102400, "canon_shotinfo")
    assert M.resolve_iso({"exif_iso": 65535, "libraw": 102400.0}) == (102400, "libraw")
    assert M.resolve_iso({"exif_iso": 65535}) == (None, "")


def _cr2_like(path, iso_field, rei=None):
    exif = {0x8827: (3, [iso_field]), 0x829A: (5, [(1, 250)]), 0x829D: (5, [(56, 10)])}
    if rei is not None:
        exif[0x8830] = (3, [2])
        exif[0x8832] = (4, [rei])
    data = tiff_bytes({0x010F: (2, "Canon"), 0x0110: (2, "Canon EOS 5D Mark IV"), 0x0112: (3, [1])},
                      sub_ifds={0x8769: exif})
    path.write_bytes(data)


def test_cr2_65535_with_recommended_exposure_index(tmp_path):
    p = tmp_path / "high.CR2"
    _cr2_like(p, 65535, rei=102400)
    md = M.read_metadata(str(p))
    assert md.iso == 102400 and md.iso_source == "exif_rei"
    assert md.iso_label == "ISO 102400"
    assert md.model == "Canon EOS 5D Mark IV"
    assert md.exposure_s == pytest.approx(1 / 250)
    assert md.fnumber == pytest.approx(5.6)


def test_cr2_65535_falls_back_to_libraw(tmp_path):
    p = tmp_path / "high2.CR2"
    _cr2_like(p, 65535)
    md = M.read_metadata(str(p), libraw_iso=51200.0)
    assert md.iso == 51200 and md.iso_source == "libraw"


def _box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + typ + payload


def test_cr3_like_container_reads_makernote_iso(tmp_path):
    """A CR3 keeps IFD0, EXIF and the Canon maker note in CMT1/CMT2/CMT3 boxes (each a little TIFF)."""
    import math
    cmt1 = tiff_bytes({0x010F: (2, "Canon"), 0x0110: (2, "Canon EOS R5"), 0x0112: (3, [8])})
    cmt2 = tiff_bytes({0x8827: (3, [65535]), 0x829D: (5, [(28, 10)])})
    base_raw = round(32 * math.log2(102400 * 32 / 100.0))
    cmt3 = tiff_bytes({0x0004: (3, canon_shotinfo(base_raw, temp_raw=170))})
    uuid = _box(b"uuid", CANON_UUID + _box(b"CMT1", cmt1) + _box(b"CMT2", cmt2) + _box(b"CMT3", cmt3))
    data = _box(b"ftyp", b"crx \x00\x00\x00\x01crx isom") + _box(b"moov", uuid) + _box(b"mdat", b"\0" * 64)
    p = tmp_path / "high.CR3"
    p.write_bytes(data)
    tm = read_tiff_meta(str(p))
    assert tm.container == "isobmff"
    md = M.read_metadata(str(p))
    assert md.model == "Canon EOS R5"
    assert md.iso == 102400 and md.iso_source == "canon_shotinfo"
    assert md.orientation == 8
    assert md.camera_temperature_c == 42.0


def test_unknown_file_does_not_crash(tmp_path):
    p = tmp_path / "junk.raw"
    p.write_bytes(b"not a raw file at all" * 10)
    md = M.read_metadata(str(p), libraw_iso=800.0)
    assert md.iso == 800 and md.iso_source == "libraw"


@pytest.mark.parametrize("path", find_sample_raws(), ids=lambda p: p.name)
def test_real_files_makernote_agrees_with_exif(path):
    """On Bill's own files: the maker-note ISO must equal the EXIF ISO wherever EXIF is valid."""
    md = M.read_metadata(str(path))
    c = md.iso_candidates
    assert md.iso is not None
    if M.valid_iso(c.get("exif_iso")) and c.get("canon_shotinfo"):
        assert c["canon_shotinfo"] == pytest.approx(M.valid_iso(c["exif_iso"]))
