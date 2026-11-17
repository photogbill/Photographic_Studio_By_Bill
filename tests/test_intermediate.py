"""The linear intermediate (io/intermediate.py): a stack result goes back into Develop as scene light."""
from __future__ import annotations

import os

import numpy as np

from conftest import ROOT, scene, synthetic_raw

OUT = ROOT / "outputs" / "pytest_intermediate"


def _out(name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    for q in (p, OUT / (name + ".partial")):
        if q.exists():
            q.unlink()
    return str(p)


def test_round_trip_renders_like_the_raw_and_keeps_values_above_one():
    from studio import engine as E
    from studio.engine.io.intermediate import read_info, write_intermediate
    from studio.engine.pipeline.pipeline import Image, linear_rgb
    sc = scene(192, 288, seed=3)
    raw_img = Image(synthetic_raw(sc))
    e = E.Edit()
    lin = linear_rgb(raw_img, e, None, cache=False)
    lin = lin.copy()
    lin[10:20, 10:20] = 1.6                         # brighter than the sensor clip: must survive
    path = _out("round_trip_Focus_Studio.tif")
    write_intermediate(path, lin, {"make": "Synthetic", "model": "Test Camera"},
                       {"kind": "focus", "sources": ["a.cr2", "b.cr2"], "method": "auto"})
    info = read_info(path)
    assert info and info["kind"] == "focus" and info["sources"] == ["a.cr2", "b.cr2"]
    img = E.open(path)
    assert img.raw.scene_linear
    back = linear_rgb(img, E.Edit(), None, cache=False)
    assert np.allclose(back[10:20, 10:20], 1.6, atol=1e-5)
    mask = np.ones(lin.shape[:2], bool)
    mask[8:22, 8:22] = False
    assert np.max(np.abs(back - lin)[mask]) < 1e-5
    # the default render of the intermediate matches the default render of the raw it came from
    a = E.render(raw_img, E.Edit())
    b = E.render(img, E.Edit())
    rel = np.abs(a - b)[mask] / np.maximum(a[mask], 0.02)
    assert float(np.mean(rel)) < 0.005, float(np.mean(rel))


def test_own_export_mark_preview_and_other_files():
    from studio.engine.io.export import is_own_export
    from studio.engine.io.intermediate import label, read_info, read_preview, write_intermediate
    from studio.engine.io.scanfile import scan_preview
    rgb = np.random.default_rng(0).random((120, 160, 3)).astype(np.float32) * 0.3
    path = _out("x_to_y_SuperRes2x_Studio.tif")
    write_intermediate(path, rgb, {}, {"kind": "superres", "scale": 2.0})
    assert is_own_export(path)                       # a later build may replace it
    assert not os.path.exists(path + ".partial")
    assert label(read_info(path)) == "Super-res 2x"
    pv = read_preview(path)
    assert pv is not None and pv.dtype == np.uint8 and pv.shape == (120, 160, 3)
    thumb = scan_preview(path, 64)
    assert thumb.dtype == np.uint8 and max(thumb.shape[:2]) <= 64
    # the thumbnail has the default look on: brighter than the flat linear values shown as they are
    assert thumb.mean() > 255 * float(np.mean(rgb)) * 1.2
    import tifffile
    plain = _out("plain_float.tif")
    tifffile.imwrite(plain, rgb, photometric="rgb")
    assert read_info(plain) is None
    from studio import engine as E
    assert not E.open(plain).raw.scene_linear
    assert label(None) == ""


def test_interrupted_write_leaves_no_partial(monkeypatch):
    import tifffile
    from studio.engine.io.intermediate import write_intermediate
    path = _out("broken_Focus_Studio.tif")

    class Boom(RuntimeError):
        pass

    def bad(*a, **k):
        raise Boom("disk full")
    monkeypatch.setattr(tifffile.TiffWriter, "write", bad)
    rgb = np.zeros((16, 16, 3), np.float32)
    try:
        write_intermediate(path, rgb, {}, {"kind": "focus"})
    except Boom:
        pass
    assert not os.path.exists(path) and not os.path.exists(path + ".partial")


def test_library_entry_names_a_stack_result():
    from studio.engine.io.intermediate import write_intermediate
    from studio.engine.io.libindex import read_entry
    path = _out("a_to_b_Focus_Studio.tif")
    write_intermediate(path, np.full((40, 60, 3), 0.2, np.float32), {}, {"kind": "focus"})
    e = read_entry({"path": path, "side_mtime": 0})
    assert e["stack"] == "Focus stack"
    import tifffile
    plain = _out("plain16.tif")
    tifffile.imwrite(plain, np.zeros((8, 8, 3), np.uint16), photometric="rgb")
    assert read_entry({"path": plain, "side_mtime": 0})["stack"] is None
