"""Old photos and film: scans open and render as they are, negatives turn positive, dust and scratches are taken
out, black and white is colourized (with the model stood in for when it is not installed)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import scene


def _write_tiff16(path, disp_pp, space="srgb", icc=True):
    import tifffile
    from studio.engine.color import spaces as sp
    from studio.engine.color.icc import profile_bytes
    S = sp.SPACES[space]
    a = np.clip(sp.apply_matrix(np.ascontiguousarray(disp_pp, np.float32), sp.conversion_matrix(sp.PROPHOTO, S)),
                0, 1)
    enc = (S.encode(a) * 65535 + 0.5).astype(np.uint16)
    tags = [(34675, "B", len(profile_bytes(space)), profile_bytes(space), True)] if icc else []
    tifffile.imwrite(path, enc, photometric="rgb", extratags=tags)
    return enc


def _display(scene_lin):
    """A scene rendered with the default Studio Standard curve (what a scan file holds)."""
    from studio.engine import Edit
    from studio.engine.color import tone
    t = Edit().tone
    xs, ys = tone.curve_lut(lambda v: tone.standard_curve(v, mid_in=t.mid_grey, contrast=t.contrast,
                                                          shoulder=t.shoulder, white=t.white))
    return tone.apply_curve_rgb(np.ascontiguousarray(scene_lin, np.float32), xs, ys)


def test_scans_render_exactly_as_the_file(tmp_path):
    """A default edit shows a scan as it is: 8-bit sRGB JPEG and PNG, 16-bit Adobe RGB TIFF, a grey JPEG."""
    from PIL import Image as PI
    from studio import engine
    from studio.engine.io.export import ExportSpec, to_output
    disp = _display(scene(192, 256) * 1.4)
    tif = str(tmp_path / "a.tif")
    enc = _write_tiff16(tif, disp, "adobergb")
    s8 = to_output(disp, ExportSpec(format="jpeg", space="srgb", bits=8))
    PI.fromarray(s8).save(tmp_path / "b.png")
    PI.fromarray(s8).save(tmp_path / "c.jpg", quality=95)
    g = (s8.astype(np.float32) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    PI.fromarray(g).save(tmp_path / "d.jpg", quality=95)
    im = engine.open(tif)
    assert im.raw.is_scan and im.raw.bits == 16 and "Adobe" in im.raw.profile.name and im.raw.profile.source == "icc"
    o = to_output(engine.render(im, engine.Edit()), ExportSpec(format="tiff", space="adobergb", bits=16))
    d = np.abs(o.astype(float) - enc.astype(float)) / 65535.0
    assert d.mean() < 2e-4 and np.percentile(d, 99.9) < 3e-3
    for name, src in (("b.png", s8), ("c.jpg", np.asarray(PI.open(tmp_path / "c.jpg")))):
        im = engine.open(str(tmp_path / name))
        o = to_output(engine.render(im, engine.Edit()), ExportSpec(format="jpeg", space="srgb", bits=8))
        assert np.abs(o.astype(float) - src.astype(float)).mean() < 0.6        # dither: well under a code value
    im = engine.open(str(tmp_path / "d.jpg"))
    assert im.raw.grey and im.raw.mono
    r = engine.render(im, engine.Edit())
    assert np.abs(r[..., 0] - r[..., 1]).max() < 1e-4                          # grey stays grey


def test_icc_profiles_and_orientation(tmp_path):
    from PIL import Image as PI
    from studio.engine.color import spaces as sp
    from studio.engine.color.icc import profile_bytes
    from studio.engine.io import scanfile as sf
    for key in ("srgb", "adobergb", "prophoto"):
        p = sf.parse_icc(profile_bytes(key))
        S = sp.SPACES[key]
        assert np.allclose(p.to_xyz_d50, S.to_xyz_d50, atol=2e-4)
        v = np.linspace(0, 1, 11)
        assert np.allclose(p.curves[0](v), S.decode(v), atol=3e-3)
    with pytest.raises(ValueError):
        sf.parse_icc(b"not a profile" * 20)
    a = (np.arange(40 * 60 * 3) % 251).reshape(40, 60, 3).astype(np.uint8)
    im = PI.fromarray(a)
    ex = im.getexif()
    ex[274] = 6                                         # rotate 90 degrees clockwise to view
    im.save(tmp_path / "o.jpg", exif=ex, quality=95)
    s = sf.open_scan(str(tmp_path / "o.jpg"))
    assert (s.width, s.height) == (40, 60) and s.meta.orientation == 1
    prev = sf.scan_preview(str(tmp_path / "o.jpg"), 32)
    assert prev.shape[0] > prev.shape[1]


def test_library_lists_scans_but_not_the_camera_jpeg_beside_a_raw(tmp_path):
    from PIL import Image as PI
    from studio.engine.io.libindex import list_photos
    PI.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "old_print.jpg")
    PI.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "IMG_0001.JPG")
    (tmp_path / "IMG_0001.CR3").write_bytes(b"x")
    names = [e["name"] for e in list_photos(str(tmp_path))]
    assert names == ["IMG_0001.CR3", "old_print.jpg"]


def test_negatives_turn_positive(tmp_path):
    """A synthetic colour negative (orange base, per-layer contrast) and a B&W one come back as the scene."""
    from studio import engine
    from studio.engine.local import film as fm
    from studio.engine.pipeline import pipeline as pl
    E = scene(192, 256) * 1.2
    ref = _display(E)
    for kind in ("colour", "bw"):
        T = fm.negative_from(E, kind)
        T = np.pad(T, ((10, 10), (10, 10), (0, 0)), constant_values=1.0)     # a clear holder edge round it
        path = str(tmp_path / f"neg_{kind}.tif")
        _write_tiff16(path, T, "srgb")
        im = engine.open(path)
        e = engine.Edit()
        e.film = {"kind": kind}
        p = pl.film_params(im, e)
        assert 4.0 <= p["stops"] <= 12.0 and all(h > l for l, h in zip(p["lo"], p["hi"]))
        out = engine.render(im, e)[10:-10, 10:-10]
        L = np.array([0.288, 0.7119, 0.0001])
        lo = np.log2(np.maximum(out.reshape(-1, 3) @ L, 1e-4))
        lr = np.log2(np.maximum(ref.reshape(-1, 3) @ L, 1e-4))
        assert np.corrcoef(lo, lr)[0, 1] > 0.99
        reg = (40, 30, 64, 48)                                     # a zoomed region renders like the whole
        whole = engine.render(im, e)
        part = engine.render(im, e, region=reg)
        assert np.allclose(part, whole[30:78, 40:104], atol=1e-5)
        if kind == "bw":
            assert np.abs(out[..., 0] - out[..., 2]).max() < 1e-3
    e2 = engine.Edit()
    e2.film = {"kind": "colour", "range": 1.3}
    assert pl.film_params(im, e2)["range"] == 1.3


def test_dust_and_scratches_are_found_and_filled(tmp_path):
    from studio.engine.local import scratches as sc
    base = scene(256, 384, seed=3) * 0.8
    dmg, truth = sc.damage(base, n_specks=60, n_lines=5, seed=5)
    Y = lambda a: (a.reshape(-1, 3) @ sc.LUMA).reshape(a.shape[:2])         # noqa: E731
    d = {"on": True, "amount": 50}
    m, st = sc.detect(Y(dmg), d)
    assert (m & truth).sum() / truth.sum() > 0.9
    m0, _ = sc.detect(Y(base), d)
    assert m0.mean() < 0.02                                        # a clean photo is (almost) left alone
    out = sc.clean(dmg, d, 0, 0)
    err = lambda a: np.abs(np.log2(np.maximum(Y(a), 1e-4)) - np.log2(np.maximum(Y(base), 1e-4)))[truth].mean()  # noqa
    assert err(dmg) > 1.0 and err(out) < 0.35
    assert np.abs(out - dmg)[~np.asarray(m)].max() == 0            # nothing outside the marks changes
    # through the pipeline: a scan of the damaged photo, whole and zoomed region agree
    from studio import engine
    path = str(tmp_path / "damaged.tif")
    _write_tiff16(path, _display(dmg), "srgb")
    im = engine.open(path)
    e = engine.Edit()
    e.scratches = {"on": True, "amount": 50}
    whole = engine.render(im, e)
    assert im.scratch_report and im.scratch_report["marks"] > 10
    part = engine.render(im, e, region=(100, 60, 120, 90))
    assert np.abs(part - whole[60:150, 100:220]).max() < 2e-3
    assert e.raw_key() != engine.Edit().raw_key()


def test_scratch_detector_network_layers():
    """The NumPy U-Net runner: blur pool, bilinear x2 (PyTorch align_corners=False), folded batch norm."""
    from studio.engine.local import scratchnet as sn
    x = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    up = sn._up2(x)
    assert up.shape == (8, 8, 1)
    assert np.isclose(up[0, 0, 0], 0.0) and np.isclose(up[1, 1, 0], 1.25)      # source (0.25, 0.25): 4 * .25 + .25
    assert np.allclose(sn._up2(np.ones((3, 5, 2), np.float32)), 1.0)
    bp = sn._blurpool(np.ones((6, 6, 3), np.float32))
    assert bp.shape == (3, 3, 3) and np.allclose(bp, 1.0)
    rng = np.random.default_rng(0)
    sd = {"w.weight": rng.normal(size=(4, 3, 3, 3)), "w.bias": rng.normal(size=4),
          "bn.weight": rng.uniform(0.5, 2, 4), "bn.bias": rng.normal(size=4), "bn.running_mean": rng.normal(size=4),
          "bn.running_var": rng.uniform(0.5, 2, 4)}
    w = np.transpose(sd["w.weight"], (2, 3, 1, 0)).astype(np.float32)
    inp = rng.normal(size=(8, 8, 3)).astype(np.float32)
    raw = sn._conv(inp, w.astype(np.float16), sd["w.bias"].astype(np.float32), act=False)
    k = sd["bn.weight"] / np.sqrt(sd["bn.running_var"] + 1e-5)
    ref = (raw - sd["bn.running_mean"]) * k + sd["bn.bias"]
    wf = (sd["w.weight"] * k[:, None, None, None])
    bf = (sd["w.bias"] - sd["bn.running_mean"]) * k + sd["bn.bias"]
    got = sn._conv(inp, np.transpose(wf, (2, 3, 1, 0)).astype(np.float16), bf.astype(np.float32), act=False)
    assert np.abs(got - ref).max() < 5e-2


@pytest.mark.skipif(not os.path.isfile(os.path.join(os.path.dirname(__file__), "..", "models", "scratches",
                                                    "scratch_detector.npz")), reason="scratch detector not installed")
def test_scratch_detector_sees_a_scratch_not_a_plain_photo():
    from studio.engine.local import scratchnet as sn
    g = np.full((256, 256), 0.45, np.float32)
    g += 0.05 * np.sin(np.arange(256) / 9.0)[None, :]
    clean = sn.net().probability(g)
    s = g.copy()
    for i in range(30, 220):
        s[i, i // 2 + 40:i // 2 + 42] = 0.97                   # a thin white scratch
    p = sn.net().probability(s)
    line = np.zeros_like(s, bool)
    for i in range(30, 220):
        line[i, i // 2 + 40:i // 2 + 42] = True
    assert p[line].mean() > 0.3 and clean.max() < 0.3 and p[~line].mean() < 0.05


def test_colourize_keeps_brightness_and_adds_colour(tmp_path, monkeypatch):
    """With the model stood in for (a fixed warm colour on the left, cool on the right), a black and white scan gets
    colour while every pixel keeps its L*."""
    from studio import engine
    from studio.engine.local import colourize as cz
    from studio.engine.pipeline import pipeline as pl
    L = np.linspace(5, 95, 11)
    lin = cz.lab_to_linear_srgb(L, 0 * L, 0 * L)
    assert np.allclose(cz.L_of_Y(lin[:, 1]), L, atol=1e-3)
    E = scene(128, 192)
    Y = E @ np.array([0.288, 0.7119, 0.0001], np.float32)
    grey = np.repeat(Y[..., None], 3, axis=2)
    path = str(tmp_path / "bw.tif")
    _write_tiff16(path, _display(grey), "srgb")
    im = engine.open(path)
    assert im.raw.mono

    def fake_predict(Lmap):
        h, w = 64, 64
        ab = np.zeros((h, w, 2), np.float32)
        ab[:, : w // 2] = (25.0, 35.0)
        ab[:, w // 2:] = (-10.0, -30.0)
        return ab, 1.0
    monkeypatch.setattr(cz, "available", lambda: True)
    monkeypatch.setattr(cz, "predict", fake_predict)
    e = engine.Edit()
    e.colourize = {"on": True, "amount": 1.0}
    out = engine.render(im, e)
    before = engine.render(im, engine.Edit())
    lum = lambda a: cz.L_of_Y(a.reshape(-1, 3) @ np.array([0.288, 0.7119, 0.0001], np.float32))  # noqa: E731
    assert np.abs(lum(out) - lum(before)).mean() < 0.6                    # brightness kept (L*, 0..100)
    left, right = out[:, :60].reshape(-1, 3).mean(0), out[:, -60:].reshape(-1, 3).mean(0)
    assert left[0] > left[2] * 1.15 and right[2] > right[0] * 1.1         # warm left, cool right
    part = engine.render(im, e, region=(50, 40, 80, 60))
    assert np.abs(part - out[40:100, 50:130]).max() < 2e-3
    assert pl.colour_map(im, e)[0].shape == (64, 64, 2)


def test_old_photo_panel_round_trip_and_scans_in_develop(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    e = E.Edit()
    e.film = {"kind": "colour", "range": 1.2}
    e.scratches = {"on": True, "amount": 70.0, "size": 12.0}
    e.colourize = {"on": True, "amount": 1.3}
    page.controls_from_edit(e)
    page.edit = e
    back = page.edit_from_controls()
    assert back.film == {"kind": "colour", "range": 1.2}
    assert back.scratches == {"on": True, "amount": 70.0, "size": 12.0}
    assert back.colourize == {"on": True, "amount": 1.3}
    page.controls_from_edit(E.Edit())
    page.edit = E.Edit()
    d = page.edit_from_controls()
    assert d.film is None and d.scratches is None and d.colourize is None
    from studio.engine.presets.looks import describe_change
    assert describe_change(E.Edit(), back).startswith(("Film", "Film, "))
    app.processEvents()


def test_onnx_runner_layers():
    """The NumPy ONNX runner's convolution (plain, grouped, depthwise, strided), depth-to-space and layer norm
    against direct formulas."""
    from studio.engine.local import onnxnp as on
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 4, 9, 11)).astype(np.float32)

    def naive(x, w, b, s, p, g):
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        n, c, h, wd = x.shape
        co, ci, kh, kw = w.shape
        oh, ow = (h + 2 * p - kh) // s + 1, (wd + 2 * p - kw) // s + 1
        out = np.zeros((1, co, oh, ow), np.float32)
        og = co // g
        for o in range(co):
            gi = o // og
            for yy in range(oh):
                for xx in range(ow):
                    patch = xp[0, gi * ci:(gi + 1) * ci, yy * s:yy * s + kh, xx * s:xx * s + kw]
                    out[0, o, yy, xx] = (patch * w[o]).sum() + b[o]
        return out
    for co, g, s, p, k in ((6, 1, 1, 1, 3), (4, 2, 2, 1, 3), (4, 4, 1, 3, 7), (8, 1, 4, 0, 4)):
        w = rng.normal(size=(co, 4 // g, k, k)).astype(np.float32)
        b = rng.normal(size=co).astype(np.float32)
        got = on._conv(x, w, b, {"strides": [s, s], "pads": [p] * 4, "group": g})
        assert np.allclose(got, naive(x, w, b, s, p, g), atol=1e-4)
    y = on._d2s(np.arange(16, dtype=np.float32).reshape(1, 4, 2, 2), {"blocksize": 2, "mode": "CRD"})
    assert y.shape == (1, 1, 4, 4) and y[0, 0, 0, 1] == 4 and y[0, 0, 1, 0] == 8
    t = rng.normal(size=(2, 5, 8)).astype(np.float32)
    ln = on._layernorm(t, np.ones(8, np.float32), np.zeros(8, np.float32), {"axis": -1, "epsilon": 1e-6})
    assert np.allclose(ln.mean(-1), 0, atol=1e-5) and np.allclose(ln.std(-1), 1, atol=1e-3)


@pytest.mark.skipif(not os.path.isfile(os.path.join(os.path.dirname(__file__), "..", "models", "colour",
                                                    "ddcolor.npz")), reason="colour model not installed")
def test_ddcolor_model_runs_and_gives_colour():
    from studio.engine.local import colourize as cz
    E = scene(192, 256, seed=2)
    L = cz.L_of_Y(E @ np.array([0.288, 0.7119, 0.0001], np.float32))
    ab, ms = cz.predict(L)
    assert ab.shape == (256, 256, 2) and np.isfinite(ab).all()
    assert np.hypot(ab[..., 0], ab[..., 1]).max() > 5.0 and np.abs(ab).max() < 128


def test_colour_hint_spreads_over_shading_and_stops_at_edges(tmp_path):
    """A click spreads over a part's light and shade (a gradual ramp) and stops at a sharp edge; Alt-clicks stop
    it too. Laid in, the part takes the colour and keeps every pixel's brightness."""
    from studio import engine
    from studio.engine.local import colourize as cz
    from studio.engine.local import masks as mk
    from studio.engine.pipeline import pipeline as pl
    h, w = 120, 200
    L = np.full((h, w), 0.3)
    L[:, :100] = np.linspace(0.35, 0.85, 100)[None, :]         # part A: a fold from dark to light
    m = cz.geodesic_area(L, [[0.1, 0.5]])
    assert m[:, :96].mean() > 250 and m[:, 104:].max() == 0
    flat = np.full((h, w), 0.5)
    m2 = cz.geodesic_area(flat, [[0.1, 0.5]], [[0.9, 0.5]])     # no edge at all: the Alt-click decides
    assert m2[:, :80].min() == 255 and m2[:, 120:].max() == 0
    # through the pipeline, on a grey scan with the same layout
    grey = np.full((h, w), 0.05, np.float32)
    grey[:, :100] = np.linspace(0.08, 0.6, 100)[None, :]
    path = str(tmp_path / "grey.tif")
    _write_tiff16(path, _display(np.repeat(grey[..., None], 3, axis=2)), "srgb")
    im = engine.open(path)
    area = pl.colour_hint_area(im, engine.Edit(), [[0.2, 0.5]])
    bm = mk.decode_bitmap(area["data"], area["w"], area["h"])
    assert bm[:, : int(area["w"] * 0.45)].mean() > 250 and bm[:, int(area["w"] * 0.55):].max() == 0
    e = engine.Edit()
    e.colourize = {"on": False, "amount": 1.0, "hints": [{**area, "rgb": [0.9, 0.3, 0.1]}]}
    out = engine.render(im, e)
    before = engine.render(im, engine.Edit())
    lum = lambda a: cz.L_of_Y(a.reshape(-1, 3) @ np.array([0.288, 0.7119, 0.0001], np.float32))  # noqa: E731
    assert np.abs(lum(out) - lum(before)).max() < 0.8                       # brightness kept everywhere
    a = out[:, 20:80].reshape(-1, 3)
    assert (a[:, 0] > a[:, 2] * 1.15).mean() > 0.95     # the part is warm (paler near white: the gamut)
    b = out[:, 130:].reshape(-1, 3)
    assert np.abs(b[:, 0] - b[:, 2]).max() < 2e-3                           # the rest stays grey


def test_colour_hints_round_trip_in_the_panel():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    e = E.Edit()
    hint = {"fg": [[0.2, 0.5]], "bg": [], "rgb": [0.9, 0.3, 0.1], "spread": 0.12, "w": 4, "h": 4,
            "data": "eNpjYEAFAAAQAAE="}
    e.colourize = {"on": False, "amount": 1.0, "hints": [hint]}
    page.controls_from_edit(e)
    page.edit = e
    back = page.edit_from_controls()
    assert back.colourize["hints"] == [hint] and not back.colourize["on"]
    assert page.hint_list.count() == 1
    app.processEvents()
