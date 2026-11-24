"""Paint's engine on synthetic strokes with known truth (BUILD_PLAN_PAINT P0/P1). Quick tests only."""
from __future__ import annotations

import math

import numpy as np
import pytest

from studio.engine.paint import autosave, ora
from studio.engine.paint.brush import BrushSettings, LayerLocked, Stroke, paint_stroke, presets
from studio.engine.paint.colour import pick
from studio.engine.paint.curves import Curve
from studio.engine.paint.smoothing import MODES, Sample, Smoother
from studio.engine.paint.tablet import PenStats, diagnose
from studio.engine.paint.tiles import Document
from studio.engine.paint.undo import UndoStack


def _line(x0, x1, y, p0, p1, n=200):
    return [Sample(x0 + (x1 - x0) * i / (n - 1), y, p0 + (p1 - p0) * i / (n - 1), i / 200) for i in range(n)]


def _hard(size=40, curve=None):
    return BrushSettings("t", size=size, hardness=1.0, spacing=0.05,
                         dynamics={"size": {"pressure": curve or Curve()}, "opacity": {}, "flow": {}})


def test_pressure_ramp_gives_the_widths_the_curve_says():
    for curve in (Curve(), Curve.preset("Soft")):
        doc = Document(800, 200)
        paint_stroke(doc.layer, _hard(40, curve), _line(50, 750, 100, 0.0, 1.0))
        a = doc.layer.read(0, 0, 800, 200)[..., 3]
        for x in (200, 400, 600):
            p = (x - 50) / 700
            width = (a[:, x] > 0.5).sum()
            assert abs(width - 40 * curve(p)) <= 2.5, (x, width, 40 * curve(p))


def test_wash_never_goes_past_the_stroke_opacity_and_buildup_does():
    doc = Document(400, 200)
    b = _hard(30)
    b.opacity, b.flow = 0.5, 0.3
    back_and_forth = _line(50, 350, 100, 1, 1) + _line(350, 50, 100, 1, 1)
    paint_stroke(doc.layer, b, back_and_forth)
    assert abs(doc.layer.read(200, 100, 1, 1)[0, 0, 3] - 0.5) < 0.01
    lay2 = doc.add_layer()
    b.mode = "buildup"
    paint_stroke(lay2, b, back_and_forth)
    assert lay2.read(200, 100, 1, 1)[0, 0, 3] > 0.9


def test_undo_restores_tiles_bit_exact_and_redo_comes_back():
    doc = Document(1000, 600)
    u = UndoStack(doc)
    paint_stroke(doc.layer, _hard(), _line(10, 900, 300, 1, 1), (1, 0, 0), u)
    first = {k: t.copy() for k, t in doc.layer.tiles.items()}
    paint_stroke(doc.layer, _hard(60), _line(100, 500, 280, 0.5, 1), (0, 0, 1), u)
    after = {k: t.copy() for k, t in doc.layer.tiles.items()}
    assert u.undo() == "Stroke"
    assert set(doc.layer.tiles) == set(first) and all(np.array_equal(doc.layer.tiles[k], first[k]) for k in first)
    u.redo()
    assert all(np.array_equal(doc.layer.tiles[k], after[k]) for k in after)
    u.undo(), u.undo()
    assert not doc.layer.tiles


def test_layer_changes_undo():
    doc = Document(300, 300)
    u = UndoStack(doc)
    u.layers_changing("Add layer")
    doc.add_layer("Ink")
    u.layers_changed()
    assert len(doc.layers) == 2
    u.undo()
    assert [l.name for l in doc.layers] == ["Layer 1"]
    u.redo()
    assert doc.layer.name == "Ink"


def test_eraser_alpha_lock_and_locked_layer():
    doc = Document(400, 200)
    paint_stroke(doc.layer, _hard(60), _line(50, 350, 100, 1, 1), (1, 0, 0))
    paint_stroke(doc.layer, _hard(20), _line(200, 200.5, 100, 1, 1), eraser=True)
    assert doc.layer.read(200, 100, 1, 1)[0, 0, 3] < 0.01
    doc.layer.alpha_lock = True
    paint_stroke(doc.layer, _hard(200), _line(20, 380, 100, 1, 1), (0, 0, 1))
    px = doc.layer.read(100, 100, 1, 1)[0, 0]
    assert px[2] > 0.99 and px[0] < 0.01                        # recoloured blue
    assert doc.layer.read(100, 20, 1, 1)[0, 0, 3] == 0          # outside the old paint stays see-through
    doc.layer.locked = True
    with pytest.raises(LayerLocked):
        Stroke(doc.layer, _hard(), (0, 0, 0))


def test_colour_picker_reads_the_picture_and_the_layer():
    doc = Document(200, 200)
    paint_stroke(doc.layer, _hard(80), _line(20, 180, 100, 1, 1), (0.2, 0.6, 0.4))
    assert np.allclose(pick(doc, 100, 100, 2), (0.2, 0.6, 0.4), atol=0.01)
    assert np.allclose(pick(doc, 100, 5), (1, 1, 1), atol=0.01)            # the paper
    assert pick(doc, 100, 5, layer_only=True) is None


def test_ora_round_trip(tmp_path):
    doc = Document(600, 400, background=(0.9, 0.85, 0.8))
    paint_stroke(doc.layer, _hard(50), _line(20, 580, 200, 1, 1), (0.8, 0.1, 0.1))
    ink = doc.add_layer("Ink")
    ink.opacity, ink.blend, ink.alpha_lock = 0.6, "multiply", True
    paint_stroke(ink, _hard(20), _line(300, 300.5, 50, 1, 1), (0, 0, 0.5))
    p = str(tmp_path / "painting.ora")
    ora.save(doc, p)
    back = ora.load(p)
    assert (back.width, back.height) == (600, 400) and [l.name for l in back.layers] == ["Layer 1", "Ink"]
    assert back.layers[1].blend == "multiply" and back.layers[1].alpha_lock and abs(back.layers[1].opacity - 0.6) < 1e-3
    assert np.allclose(back.background, (0.9, 0.85, 0.8), atol=1e-5)
    assert np.abs(back.composite() - doc.composite()).max() < 2.5 / 255


def test_autosave_and_recovery_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_PAINT_DIR", str(tmp_path / "paint"))
    doc = Document(300, 300)
    paint_stroke(doc.layer, _hard(), _line(10, 290, 150, 1, 1), (0, 0.5, 0))
    s = autosave.AutoSaver(doc, "Sketch")
    s.changed()
    assert s.due(0)
    s.save_now()
    found = autosave.recoverable()
    assert len(found) == 1 and found[0]["title"] == "Sketch"
    back = autosave.recover(found[0])
    assert np.abs(back.composite() - doc.composite()).max() < 2.5 / 255
    s.close()
    assert autosave.recoverable() == []


def test_smoothing_modes_all_run_and_adaptive_calms_a_shaky_slow_line():
    rng = np.random.default_rng(1)
    raw = [Sample(i * 0.5, 100 + rng.normal(0, 1.5), 1, i / 200) for i in range(400)]
    for mode in MODES:
        sm = Smoother(mode, 0.6)
        out = [o for s in raw for o in sm.feed(Sample(**vars(s)))] + sm.finish()
        assert out and abs(out[-1].x - raw[-1].x) < 2, mode              # every mode ends where the pen stopped
    sm = Smoother("adaptive", 0.8)
    out = [o for s in raw for o in sm.feed(Sample(**vars(s)))]
    assert np.std([o.y for o in out[50:]]) < 0.5 * np.std([s.y for s in raw[50:]])


def test_tablet_diagnosis_says_when_pressure_is_missing():
    st = PenStats()
    for i in range(100):
        st.add_tablet(1.0, 0, 0, i / 233)
    text = " ".join(t for _, t in diagnose(st, "linux"))
    assert "pressure is not arriving" in text and "No tilt" in text
    st2 = PenStats()
    for i in range(50):
        st2.add_mouse()
    assert diagnose(st2, "linux")[0][0] == "bad"
    st3 = PenStats()
    for i in range(300):
        st3.add_tablet(i / 300, 0, 0, i / 233)
    d = diagnose(st3, "linux")
    assert d[0][0] == "ok" and "233" in " ".join(t for _, t in d)


def test_every_preset_says_what_it_is_for_and_paints():
    for b in presets():
        assert b.best_for and b.weak_at
        doc = Document(300, 120)
        if b.eraser:
            paint_stroke(doc.layer, _hard(100), _line(10, 290, 60, 1, 1))
        paint_stroke(doc.layer, b, _line(30, 270, 60, 0.8, 0.8))


def _ring(doc, gap_deg=0.0):
    """A black circle outline (radius 100, 6 px line) with a gap of ``gap_deg`` degrees, as line art."""
    import math
    from studio.engine.paint.brush import BrushSettings
    b = BrushSettings("ink", size=6, hardness=1.0, spacing=0.05, dynamics={"size": {}, "opacity": {}, "flow": {}})
    n = 720
    pts = [Sample(200 + 100 * math.cos(math.radians(a)), 200 + 100 * math.sin(math.radians(a)), 1, i / 200)
           for i, a in enumerate(i * 360 / n for i in range(n + 1)) if a >= gap_deg]
    paint_stroke(doc.layer, b, pts, (0, 0, 0))


def test_fill_stays_inside_line_art_with_a_gap_only_when_gap_closing_is_on():
    from studio.engine.paint.fill import fill
    doc = Document(400, 400)
    _ring(doc, gap_deg=8)                                     # a ~8 px break in the line (after the round ends)
    flats = doc.add_layer("Flats")
    doc_leak = fill(doc, 200, 200, (1, 0, 0), tolerance=0.2, gap=0, source="reference", reference=0)
    assert flats.read(5, 5, 1, 1)[0, 0, 3] > 0.9              # without gap closing the fill leaks out
    flats.tiles.clear()
    fill(doc, 200, 200, (1, 0, 0), tolerance=0.2, gap=6, source="reference", reference=0)
    assert flats.read(200, 200, 1, 1)[0, 0, 3] > 0.9 and flats.read(5, 5, 1, 1)[0, 0, 3] == 0
    assert flats.read(200, 107, 1, 1)[0, 0, 3] > 0.9          # it reaches up to the line
    assert doc_leak is not None


def test_gradient_ends_and_blend_spaces():
    from studio.engine.paint.colour import mix, over
    from studio.engine.paint.fill import gradient_pixels
    g = gradient_pixels(101, 3, (0, 1), (101, 1), (1, 0, 0, 1), (0, 0, 1, 1), "linear", "oklab")
    assert np.allclose(g[1, 0, :3], (1, 0, 0), atol=0.05) and np.allclose(g[1, -1, :3], (0, 0, 1), atol=0.05)
    t = gradient_pixels(11, 1, (0, 0), (11, 0), (0, 1, 0, 1), (1, 1, 1, 0), "linear", "classic")
    assert t[0, -1, 3] < 0.06 and t[0, 0, 3] > 0.94
    blue, yellow = np.array([0, 0, 1.0]), np.array([1, 1, 0.0])
    lum = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    assert lum(mix(blue, yellow, 0.5, "oklab")) > lum(mix(blue, yellow, 0.5, "classic"))   # no grey dip
    base = np.array([[[0, 1, 0, 1.0]]], np.float32)
    a = np.array([[[0.5]]], np.float32)
    light = over(base, (1, 0, 0), a, "light")[0, 0]
    classic = over(base, (1, 0, 0), a, "classic")[0, 0]
    assert light[0] > classic[0] and abs(light[3] - 1) < 1e-6   # red over green mixes brighter in light


def test_image_tip_from_a_picture(tmp_path):
    from PIL import Image
    from studio.engine.paint.brush import load_tip
    im = np.full((60, 80), 255, np.uint8)
    im[20:40, 20:60] = 0                                       # a dark bar on white paper (wider than tall)
    p = tmp_path / "mark.png"
    Image.fromarray(im).save(p)
    tip = load_tip(str(p))
    assert tip.shape == (40, 40) and tip.max() > 0.99 and tip[0, 0] < 0.01 and tip[20, 20] > 0.99
    doc = Document(300, 120)
    b = _hard(40)
    b.tip = tip
    paint_stroke(doc.layer, b, _line(40, 260, 60, 1, 1))
    assert doc.layer.read(150, 60, 1, 1)[0, 0, 3] > 0.9


def test_branching_undo_keeps_both_ideas_and_jumps_between_them():
    doc = Document(400, 200)
    u = UndoStack(doc)
    paint_stroke(doc.layer, _hard(30), _line(20, 380, 50, 1, 1), (0, 0, 0), u)        # A
    paint_stroke(doc.layer, _hard(30), _line(20, 380, 100, 1, 1), (1, 0, 0), u)       # B (idea 1: red)
    red = {k: t.copy() for k, t in doc.layer.tiles.items()}
    red_node = u.current
    u.undo()
    paint_stroke(doc.layer, _hard(30), _line(20, 380, 100, 1, 1), (0, 0, 1), u)       # C (idea 2: blue), a branch
    blue = {k: t.copy() for k, t in doc.layer.tiles.items()}
    assert len(u.path()[1].children) == 2                                             # both ideas are kept
    u.jump(red_node)
    assert all(np.array_equal(doc.layer.tiles[k], red[k]) for k in red)
    u.jump(u.path()[1].children[1])
    assert all(np.array_equal(doc.layer.tiles[k], blue[k]) for k in blue)
    tree = u.tree()
    assert [n.name for n, _, _ in tree] == ["Start", "Stroke", "Stroke", "Stroke"] and tree[-1][2]
    u.budget = 1                                                                      # memory runs out:
    u._trim()                                                                         # the branch away from here goes
    assert len(u.path()) >= 2 and all(len(n.children) <= 1 for n, _, _ in u.tree())


def _solid(doc, lay, rgb, a=1.0, rect=(0, 0, 64, 64)):
    x0, y0, x1, y1 = rect
    px = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
    px[..., :3] = np.asarray(rgb, np.float32) * a
    px[..., 3] = a
    lay.write(x0, y0, px)


def _names(doc):
    return [(l.name, None if l.parent is None else doc.by_uid(l.parent).name) for l in doc.layers]


def test_layer_groups_composite_move_save_and_undo(tmp_path):
    doc = Document(64, 64, background=(1.0, 1.0, 1.0))
    base = doc.layer
    _solid(doc, base, (0.5, 0.5, 0.5))                     # grey
    shade = doc.add_layer("Shade")
    _solid(doc, shade, (0.5, 0.5, 0.5))                    # a Multiply layer of the same grey
    shade.blend = "multiply"
    g = doc.add_group("G")                                  # wraps Shade
    assert _names(doc) == [("Layer 1", None), ("Shade", "G"), ("G", None)]
    # pass-through: Shade multiplies with the grey below -> 0.25
    assert abs(doc.composite()[0, 0, 0] - 0.25) < 1e-5
    # isolated: Shade only sees the group's own empty backdrop -> grey over grey = 0.5
    g.blend = "normal"
    assert abs(doc.composite()[0, 0, 0] - 0.5) < 1e-5
    # pass-through at half opacity: halfway between without (0.5) and with (0.25)
    g.blend, g.opacity = "pass", 0.5
    assert abs(doc.composite()[0, 0, 0] - 0.375) < 1e-5
    g.opacity, g.visible = 1.0, False                      # a hidden group hides what is in it
    assert abs(doc.composite()[0, 0, 0] - 0.5) < 1e-5 and not doc.shown(shade)
    g.visible = True

    # strokes find their layer: a selected group passes the stroke to its top layer; a locked group stops it
    doc.active = doc.index_of(g)
    assert doc.paint_layer() is shade and doc.layer is shade
    g.locked = True
    with pytest.raises(LayerLocked, match="locked group"):
        doc.paint_layer()
    g.locked = False

    # moving in and out: Shade up comes out of G above it; down again goes back in at the top
    u = UndoStack(doc)
    u.layers_changing("Move")
    assert doc.move(1)
    u.layers_changed()
    assert _names(doc) == [("Layer 1", None), ("G", None), ("Shade", None)]
    assert doc.move(-1) and _names(doc) == [("Layer 1", None), ("Shade", "G"), ("G", None)]
    assert doc.move(-1) and _names(doc) == [("Layer 1", None), ("Shade", None), ("G", None)]  # out below G
    assert doc.move(-1) and _names(doc) == [("Shade", None), ("Layer 1", None), ("G", None)]  # past Layer 1
    assert doc.move(1) and doc.move(1) and _names(doc) == [("Layer 1", None), ("Shade", "G"), ("G", None)]
    g.collapsed = True                                      # a closed group is stepped over as one
    assert doc.move(1) and _names(doc) == [("Layer 1", None), ("G", None), ("Shade", None)]
    assert doc.move(-1) and _names(doc) == [("Layer 1", None), ("Shade", None), ("G", None)]
    g.collapsed = False
    assert doc.move(1) and _names(doc) == [("Layer 1", None), ("Shade", "G"), ("G", None)]
    u.undo()                                                # undo the first move: back where it began
    assert _names(doc) == [("Layer 1", None), ("Shade", "G"), ("G", None)]
    shade, g = doc.layers[1], doc.layers[2]                 # undo put copies in place: look them up again

    # nested group, then .ora both ways keeps the tree, modes, opacity and the picture
    doc.active = doc.index_of(shade)
    inner = doc.add_group("Inner")
    inner.blend, inner.opacity = "normal", 0.8
    g.opacity = 0.6                                          # faded pass-through: kept by our own attribute
    want = doc.composite()
    p = str(tmp_path / "groups.ora")
    ora.save(doc, p)
    back = ora.load(p)
    assert _names(back) == [("Layer 1", None), ("Shade", "Inner"), ("Inner", "G"), ("G", None)]
    bg = back.layers[3]
    assert bg.is_group and bg.blend == "pass" and abs(bg.opacity - 0.6) < 1e-3
    assert back.layers[2].blend == "normal" and back.layers[1].blend == "multiply"
    assert np.abs(back.composite() - want).max() < 2 / 255

    # flattening a group gives one layer that looks the same; deleting a group takes its contents
    k = back.index_of(back.layers[3])
    back.flatten_group(k)
    assert len(back.layers) == 2 and np.abs(back.composite() - want).max() < 2 / 255
    d2 = Document(64, 64)
    d2.add_group("Empty", wrap=False)
    assert not d2.delete(0) or any(not l.is_group for l in d2.layers)


def test_ora_from_elsewhere_with_groups_reads_as_groups(tmp_path):
    import io
    import zipfile
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(buf, "PNG")
    xml = ('<?xml version="1.0"?><image w="8" h="8"><stack>'
           '<stack name="Figure" isolation="isolate" opacity="0.5">'
           '<layer name="Red" src="data/a.png" x="0" y="0"/></stack>'
           '<layer name="Paper" src="data/a.png" x="0" y="0" visibility="hidden"/></stack></image>')
    p = tmp_path / "krita.ora"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "image/openraster")
        z.writestr("stack.xml", xml)
        z.writestr("data/a.png", buf.getvalue())
    d = ora.load(str(p))
    assert _names(d) == [("Paper", None), ("Red", "Figure"), ("Figure", None)]
    assert d.layers[2].blend == "normal" and abs(d.layers[2].opacity - 0.5) < 1e-6


def test_swatches_round_trip_palette_from_painting_and_pickers(tmp_path):
    from studio.engine.paint import swatches
    from studio.engine.paint.colour import (in_gamut, max_chroma, oklch_to_srgb, srgb_to_oklch, sv_to_triangle,
                                            triangle_corners, triangle_to_sv)
    sw = swatches.SwatchSet("Test", note="for a test")
    sw.add((1.0, 0.0, 0.0), "Red")
    sw.add((0.2, 0.4, 0.6))
    p = swatches.save_gpl(sw, str(tmp_path / "t.gpl"))
    back = swatches.load(p)
    assert back.name == "Test" and back.names[0] == "Red" and back.note == "for a test"
    assert np.allclose(back.colours, [(1, 0, 0), (51 / 255, 102 / 255, 153 / 255)])
    # four known colour masses come back as the palette, dark to light
    doc = Document(64, 64)
    truth = [(0.1, 0.1, 0.4), (0.8, 0.2, 0.1), (0.2, 0.7, 0.3), (0.95, 0.9, 0.6)]
    for k, c in enumerate(truth):
        _solid(doc, doc.layer, c, rect=(16 * k, 0, 16 * k + 16, 64))
    got = swatches.from_painting(doc, 4).colours
    for c in truth:
        assert min(np.abs(np.array(g) - c).max() for g in got) < 0.02
    L = [srgb_to_oklch(np.array(g, np.float32))[0] for g in got]
    assert L == sorted(L)
    # triangle: corners map to (s, v) and back
    tri = triangle_corners(100, 100, 80, 0.3)
    for s_, v_ in ((1, 1), (0, 1), (0.5, 0.5), (0.2, 0.9)):
        x, y = sv_to_triangle(s_, v_, tri)
        assert np.allclose(triangle_to_sv(x, y, tri), (s_, v_), atol=1e-6)
    # gamut: sRGB red sits on the sRGB edge; P3 holds more chroma; out-of-gamut colours come back inside
    Lr, Cr, Hr = (float(v) for v in srgb_to_oklch(np.array([1, 0, 0], np.float32)))
    assert abs(max_chroma(Lr, Hr) - Cr) < 2e-3 and max_chroma(Lr, Hr, "p3") > Cr + 0.01
    assert not in_gamut(0.6, 0.3, 150) and in_gamut(0.6, 0.05, 150)
    r = oklch_to_srgb(0.6, 0.3, 150)
    assert all(0 <= v <= 1 for v in r) and abs(float(srgb_to_oklch(np.array(r, np.float32))[0]) - 0.6) < 0.01


def test_key_map_assign_take_over_and_save(tmp_path):
    from studio.engine.paint import keymap
    m = keymap.defaults()
    assert not keymap.conflicts(m)
    taken = keymap.assign(m, "undo", "F13", slot=1)          # a tablet key as undo's second key
    assert taken is None and m["undo"] == ["Ctrl+Z", "F13"]
    taken = keymap.assign(m, "eraser", "F13")                 # the same key for the eraser takes it from undo
    assert taken == "undo" and m["undo"] == ["Ctrl+Z"] and m["eraser"][0] == "F13"
    assert keymap.action_for(m, "F13") == "eraser" and not keymap.conflicts(m)
    p = keymap.save(m, tmp_path / "keys.json")
    back = keymap.load(p)
    assert back["eraser"] == m["eraser"] and back["popup"] == []


def test_exports_webp_and_profiles(tmp_path):
    from PIL import Image
    from studio.engine.io.export import free_target, is_own_export
    from studio.engine.paint import export as pexport
    doc = Document(32, 16, background=None)
    _solid(doc, doc.layer, (1.0, 0.0, 0.0), rect=(0, 0, 16, 16))           # red half, see-through half
    for fmt in [f for f in pexport.FORMATS if f != "exr"]:      # EXR has its own test (Pillow cannot read it)
        p = str(tmp_path / f"a_Studio{pexport.FORMATS[fmt][1]}")
        pexport.export(doc, p, fmt, "srgb", bits=16 if fmt in ("png", "tiff") else 8, lossless=True)
        assert is_own_export(p), fmt                        # marked: a second export replaces only our own file
        with Image.open(p) as im:
            assert im.info.get("icc_profile"), fmt
            if fmt != "jpeg" and fmt != "tiff":
                rgba = np.asarray(im.convert("RGBA"))
                assert rgba[8, 4, 3] == 255 and rgba[8, 24, 3] == 0, fmt   # see-through kept
    assert free_target(str(tmp_path / "a_Studio.webp")) == str(tmp_path / "a_Studio.webp")
    # converting to Display P3: sRGB red is inside P3, so it is not P3's full red (known value 0.9175, 0.2003, 0.1386)
    f = pexport.flat_pixels(doc, "displayp3")
    assert np.allclose(f[8, 4, :3], (0.9175, 0.2003, 0.1386), atol=0.003)


def test_ora_16bit_layers_keep_soft_glazes(tmp_path):
    doc = Document(64, 64, background=None)
    ramp = np.zeros((64, 64, 4), np.float32)
    a = np.linspace(0.0, 0.02, 64, dtype=np.float32)[None, :] * np.ones((64, 1), np.float32)  # a very faint glaze
    ramp[..., 0] = 0.7 * a
    ramp[..., 3] = a
    doc.layer.write(0, 0, ramp)
    for bits, tol in ((16, 1e-4), (8, None)):
        doc.bits = bits
        p = str(tmp_path / f"g{bits}.ora")
        ora.save(doc, p)
        back = ora.load(p)
        got = back.layer.read(0, 0, 64, 64)[..., 3]
        assert back.bits == bits
        if tol:
            assert np.abs(got - a).max() < tol                     # 16-bit: every shade of the glaze kept
        else:
            assert len(np.unique(got)) <= 7                        # 8-bit: the glaze falls into a few steps
    # our decoder against Pillow on an ordinary 16-bit picture written by our encoder (the Sub filter path too)
    from studio.engine.paint.ora import _decode_png16, _to_png16
    px = np.random.default_rng(1).random((5, 7, 4)).astype(np.float32)
    px[..., :3] *= px[..., 3:4]
    back = _decode_png16(_to_png16(px))
    assert np.abs(back - px).max() < 2e-5
    # a PNG whose rows use the Sub filter (as other programs write): decoded to the same values
    import struct
    import zlib
    from studio.engine.paint.ora import _chunk
    v = np.random.default_rng(2).integers(0, 65536, (3, 4, 4)).astype(">u2")
    rows = v.reshape(3, -1).view(np.uint8)
    sub = rows.copy()
    sub[:, 8:] = rows[:, 8:] - rows[:, :-8]                      # 8 bytes per RGBA16 pixel
    raw = np.concatenate([np.ones((3, 1), np.uint8), sub], 1)
    png = (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 3, 16, 6, 0, 0, 0))
           + _chunk(b"IDAT", zlib.compress(raw.tobytes())) + _chunk(b"IEND", b""))
    d = _decode_png16(png)
    want = v.astype(np.float32) / 65535
    want[..., :3] *= want[..., 3:4]
    assert np.abs(d - want).max() < 1e-6


def test_lpm_blend_style_keeps_ends_and_self_mixes_and_brush_runs():
    from studio.engine.paint.colour import SPACES, lpm_filter, mix
    a, b = np.array([0.1, 0.2, 0.8], np.float32), np.array([0.95, 0.85, 0.1], np.float32)
    assert np.allclose(mix(a, b, 0.0, "lpm"), a, atol=1e-6) and np.allclose(mix(a, b, 1.0, "lpm"), b, atol=1e-6)
    assert np.allclose(mix(a, a, 0.5, "lpm"), a, atol=1e-6)        # a colour mixed with itself stays itself
    mid = mix(a, b, 0.5, "lpm")
    assert np.abs(mid - mix(a, b, 0.5, "classic")).max() > 0.005 and (0 <= mid).all() and (mid <= 1).all()
    # the published filter itself, on a known value: grey 0.5 at the curve's middle -> 0.6601 (worked by hand)
    assert abs(float(lpm_filter(np.array([0.5, 0.5, 0.5], np.float32), 0.5)[0]) - 0.6601) < 1e-3
    assert "lpm" in SPACES
    doc = Document(200, 100)
    doc.layer.write(0, 0, np.concatenate([np.ones((100, 200, 3), np.float32) * a, np.ones((100, 200, 1),
                                                                                      np.float32)], -1))
    br = _hard(30)
    br.space, br.opacity = "lpm", 0.5
    paint_stroke(doc.layer, br, _line(20, 180, 50, 1.0, 1.0), colour=tuple(b))
    assert doc.layer.read(100, 50, 1, 1)[0, 0, 3] > 0.99


def _red_blue(doc):
    _solid(doc, doc.layer, (1.0, 0.0, 0.0), rect=(0, 0, 100, 100))
    _solid(doc, doc.layer, (0.0, 0.0, 1.0), rect=(100, 0, 200, 100))


def _reach(doc, y=50):
    """How far red was carried into the blue half along row y (pixels with red > 0.25)."""
    row = doc.layer.read(100, y, 100, 1)[0]
    red = np.nonzero(row[:, 0] > 0.25)[0]
    return 0 if not len(red) else int(red.max()) + 1


def test_smudge_carries_paint_by_its_length_and_dull_mixes():
    from studio.engine.paint.brush import SMUDGE_KINDS
    reach = {}
    for length in (0.3, 0.95):
        doc = Document(200, 100)
        _red_blue(doc)
        b = BrushSettings("s", size=20, hardness=0.5, spacing=0.05, mode="smudge", smudge_kind="smear",
                          smudge_length=length, dynamics={"size": {}, "opacity": {}, "flow": {}})
        paint_stroke(doc.layer, b, _line(60, 180, 50, 1, 1))
        reach[length] = _reach(doc)
        assert doc.layer.read(150, 90, 1, 1)[0, 0, 2] > 0.99          # off the stroke: untouched blue
    assert reach[0.95] > reach[0.3] + 10, reach                       # a longer smudge carries red further
    # dull: one averaged colour; a short stroke across the edge leaves a purple mix on it
    doc = Document(200, 100)
    _red_blue(doc)
    b = BrushSettings("d", size=20, hardness=0.5, spacing=0.05, mode="smudge", smudge_kind="dull",
                      smudge_length=0.7, dynamics={"size": {}, "opacity": {}, "flow": {}})
    paint_stroke(doc.layer, b, _line(85, 115, 50, 1, 1))
    c = doc.layer.read(108, 50, 1, 1)[0, 0]
    assert c[0] > 0.15 and c[2] > 0.15 and abs(c[3] - 1) < 1e-4       # a mix of both, still opaque
    assert set(SMUDGE_KINDS) == {"smear", "dull"}


def test_smudge_colour_rate_and_reading_every_layer():
    # oil: colour rate mixes the brush colour into the paint it picks up (white paper layer, blue brush)
    doc = Document(200, 100)
    _solid(doc, doc.layer, (1.0, 1.0, 1.0), rect=(0, 0, 200, 100))
    oil = next(p for p in presets() if p.mode == "smudge" and p.colour_rate > 0)
    paint_stroke(doc.layer, oil, _line(20, 180, 50, 1, 1), colour=(0.0, 0.0, 1.0))
    c = doc.layer.read(60, 50, 1, 1)[0, 0]
    assert c[2] > c[0] + 0.1                                         # bluish, mixed with the white it met
    # every layer: smudging on an empty layer above pulls the red from below onto it
    doc = Document(200, 100, background=None)
    _red_blue(doc)
    top = doc.add_layer()
    b = BrushSettings("s", size=20, hardness=0.5, spacing=0.05, mode="smudge", smudge_kind="smear",
                      smudge_length=0.9, smudge_all=True, dynamics={"size": {}, "opacity": {}, "flow": {}})
    paint_stroke(top, b, _line(60, 160, 50, 1, 1),
                 sample=lambda x0, y0, w, h: doc.composite(x0, y0, w, h, background=False))
    assert top.read(110, 50, 1, 1)[0, 0, 0] > 0.3 and top.read(110, 90, 1, 1)[0, 0, 3] == 0
    # this layer only (the default): the empty layer has nothing to pull
    top2 = doc.add_layer()
    b.smudge_all = False
    paint_stroke(top2, b, _line(60, 160, 50, 1, 1),
                 sample=lambda x0, y0, w, h: doc.composite(x0, y0, w, h, background=False))
    assert not top2.tiles


def test_pigment_mixing_blue_and_yellow_make_green():
    from studio.engine.paint.colour import from_hex, mix, srgb_to_reflectance, reflectance_to_srgb
    blue, yellow = np.array(from_hex("#002185"), np.float32), np.array(from_hex("#fcd200"), np.float32)
    for c in (blue, yellow, np.array([1, 1, 1], np.float32), np.array([0.2, 0.4, 0.8], np.float32)):
        assert np.abs(reflectance_to_srgb(srgb_to_reflectance(c)) - c).max() < 1 / 255   # spectra round-trip
    g = mix(blue, yellow, 0.5, "pigment")
    assert g[1] > g[0] + 0.2 and g[1] > g[2] + 0.2                  # green, as paint does (spectral.js: #3d933e)
    assert np.abs(g - np.array(from_hex("#3d933e"))).max() < 2 / 255
    assert np.abs(mix(blue, yellow, 0.0, "pigment") - blue).max() < 1 / 255
    assert np.abs(mix(blue, blue, 0.5, "pigment") - blue).max() < 1 / 255
    # the fast path (one brush colour, rounded paint amounts) matches the exact per-pixel path, light touches too
    rng = np.random.default_rng(0)
    A = rng.random((2000, 3)).astype(np.float32)
    t = (rng.random(2000) ** 3).astype(np.float32)[:, None]          # many light touches
    B = np.broadcast_to(blue, (2000, 3)).copy()
    B[0] += 1e-6                                                       # forces the exact branch
    assert np.abs(mix(A, blue, t, "pigment") - mix(A, B, t, "pigment")).max() < 1.5 / 255
    # the brush paints with it: yellow over blue goes green where they overlap
    doc = Document(200, 100)
    _solid(doc, doc.layer, tuple(blue), rect=(0, 0, 200, 100))
    br = _hard(30)
    br.space, br.opacity = "pigment", 0.5
    paint_stroke(doc.layer, br, _line(20, 180, 50, 1.0, 1.0), colour=tuple(yellow))
    c = doc.layer.read(100, 50, 1, 1)[0, 0]
    assert c[1] > c[0] and c[1] > c[2]


def _mp_brush(**base):
    from studio.engine.paint.mypaint import MyPaintBrush
    b = MyPaintBrush("t")
    b.base.update(base)
    return b


def _mp_stroke(layer, b, samples, colour=(0, 0, 0)):
    from studio.engine.paint.mypaint import MyPaintStroke
    st = MyPaintStroke(layer, b, colour, seed=1, sample=layer.read)
    for s in samples:
        st.add(s)
    st.finish()
    return st


def test_mypaint_brush_reads_myb_and_widths_follow_its_curves(tmp_path):
    import math
    from studio.engine.paint.mypaint import MyPaintBrush, load_myb
    b = _mp_brush(radius_logarithmic=math.log(10), hardness=0.95, opaque=1.0, dabs_per_actual_radius=4.0,
                  anti_aliasing=0.0, opaque_multiply=1.0)
    assert b.curves.pop("opaque_multiply")                     # libmypaint's default: opacity follows pressure
    b.curves["radius_logarithmic"] = {"pressure": [(0.0, -1.0), (1.0, 0.0)]}   # radius 10 * e^(p - 1)
    assert abs(b.value("radius_logarithmic", {"pressure": 0.5}) - (math.log(10) - 0.5)) < 1e-9
    p = tmp_path / "Test_brush.myb"
    p.write_text(b.to_myb().replace('"settings": {', '"settings": {"from_the_future": {"base_value": 1, '
                                                     '"inputs": {}}, '), encoding="utf-8")
    back = load_myb(str(p))                                     # an unknown setting is ignored
    assert back.name == "Test brush" and back.curves["radius_logarithmic"]["pressure"][0] == (0.0, -1.0)
    doc = Document(800, 200)
    samples = [Sample(50 + i * 3.5, 100, 0.2 + 0.8 * i / 199, i / 200) for i in range(200)]
    st = _mp_stroke(doc.layer, back, samples)
    a = doc.layer.read(0, 0, 800, 200)[..., 3]
    for x, pr in ((120, 0.2 + 0.8 * 20 / 199), (680, 0.2 + 0.8 * 180 / 199)):
        want = 2 * 10 * math.exp(pr - 1)
        got = float((a[:, x] > 0.5).sum())
        assert abs(got - want) < 2.5, (x, got, want)
    assert st.dabs > 200
    with pytest.raises(ValueError):
        MyPaintBrush.from_myb('{"version": 2}')


def test_mypaint_eraser_and_smudge():
    import math
    doc = Document(200, 100)
    _red_blue(doc)
    er = _mp_brush(radius_logarithmic=math.log(8), hardness=0.9, eraser=1.0, dabs_per_actual_radius=4.0)
    _mp_stroke(doc.layer, er, [Sample(20 + i, 20, 0.8, i / 100) for i in range(60)])
    assert abs(doc.layer.read(50, 20, 1, 1)[0, 0, 3] - 0.2) < 0.05   # opaque linearize: about 1 - 0.8 is left
    _mp_stroke(doc.layer, er, [Sample(20 + i, 20, 1.0, i / 100) for i in range(60)])
    assert doc.layer.read(50, 20, 1, 1)[0, 0, 3] < 0.05 and doc.layer.read(50, 80, 1, 1)[0, 0, 3] > 0.99
    sm = _mp_brush(radius_logarithmic=math.log(10), hardness=0.5, smudge=1.0, smudge_length=0.9,
                   dabs_per_actual_radius=4.0, paint_mode=0.0)
    _mp_stroke(doc.layer, sm, [Sample(60 + i, 60, 1.0, i / 100) for i in range(100)])
    c = doc.layer.read(115, 60, 1, 1)[0, 0]
    assert c[0] > 0.2                                           # red carried into the blue half


def _cover(doc, rect=(0, 0, 800, 200)):
    x0, y0, x1, y1 = rect
    return float((doc.layer.read(x0, y0, x1 - x0, y1 - y0)[..., 3] > 0.5).mean())


def test_paper_grain_follows_pressure_and_stays_on_the_paper():
    from studio.engine.paint import texture
    from studio.engine.paint.curves import Curve
    for k in texture.GRAINS:
        g = texture.grain(k)
        assert g.shape == (texture.SIDE, texture.SIDE) and abs(float(g.min())) < 1e-6 and abs(float(g.max()) - 1) < 1e-6
        assert np.abs(g[0] - g[-1]).mean() < 0.25                     # tileable: the edges meet
    cov = {}
    for strength, p in ((0.0, 0.3), (1.0, 0.3), (1.0, 1.0)):
        doc = Document(800, 200)
        b = BrushSettings("g", size=40, hardness=1.0, spacing=0.05, grain="rough", grain_strength=strength,
                          dynamics={"size": {}, "opacity": {}, "flow": {}})
        paint_stroke(doc.layer, b, _line(50, 750, 100, p, p))
        cov[(strength, p)] = _cover(doc)
        if strength and p == 0.3:
            a1 = doc.layer.read(0, 0, 800, 200)[..., 3].copy()
            paint_stroke(Document(800, 200).layer, b, _line(50, 750, 100, p, p))
            doc2 = Document(800, 200)
            paint_stroke(doc2.layer, b, _line(50, 750, 104, p, p))   # moved 4 px: the holes stay where the paper is
            a2 = doc2.layer.read(0, 0, 800, 200)[..., 3]
            both = (a1 > 0) & (a2 > 0)
            assert np.corrcoef(a1[both], a2[both])[0, 1] > 0.8
    assert cov[(1.0, 0.3)] < cov[(0.0, 0.3)] * 0.7                   # a light touch skips the valleys
    assert cov[(1.0, 1.0)] > cov[(1.0, 0.3)] * 1.3                   # more pressure fills them


def test_dual_tip_scatter_count_and_colour_jitter():
    base = dict(size=40, hardness=1.0, spacing=0.1, dynamics={"size": {}, "opacity": {}, "flow": {}})
    plain, dual = Document(800, 200), Document(800, 200)
    paint_stroke(plain.layer, BrushSettings("p", **base), _line(50, 750, 100, 1, 1))
    st = paint_stroke(dual.layer, BrushSettings("d", dual=True, dual_size=0.2, dual_count=3, dual_scatter=0.6,
                                                **base), _line(50, 750, 100, 1, 1))
    assert _cover(dual) < _cover(plain) * 0.9
    spray = Document(800, 200)
    st2 = paint_stroke(spray.layer, BrushSettings("s", count=5, scatter=1.0, **base), _line(50, 750, 100, 1, 1))
    assert st2.dabs == 5 * st.dabs
    for per_stroke in (False, True):
        doc = Document(800, 200)
        b = BrushSettings("j", jitter_h=0.3, jitter_v=0.3, jitter_per_stroke=per_stroke, **base)
        b.spacing = 0.6                                              # separate dabs, so each keeps its colour
        paint_stroke(doc.layer, b, _line(50, 750, 100, 1, 1), colour=(0.8, 0.3, 0.2))
        row = doc.layer.read(60, 100, 680, 1)[0]
        cols = row[row[:, 3] > 0.99, :3]
        spread = float(cols.std(axis=0).max())
        assert (spread < 1e-4) if per_stroke else (spread > 0.03), (per_stroke, spread)


def test_brush_files_round_trip_and_preview(tmp_path):
    from studio.engine.paint import brushfile
    from studio.engine.paint.curves import Curve
    b = BrushSettings("My chalk", size=33, grain="cold_press", grain_strength=0.8, dual=True, jitter_v=0.1,
                      dynamics={"size": {"pressure": Curve.preset("Soft")}, "opacity": {"tilt": Curve()}, "flow": {}},
                      best_for="testing", weak_at="nothing")
    b.tip = np.linspace(0, 1, 64, dtype=np.float32)[None, :] * np.ones((64, 1), np.float32)
    p = brushfile.save(b, tmp_path)
    back = brushfile.load(p)
    assert back.name == "My chalk" and back.grain == "cold_press" and back.dual and back.size == 33
    assert back.dynamics["size"]["pressure"].points == b.dynamics["size"]["pressure"].points
    assert back.tip is not None and abs(float(back.tip[0, -1]) - 1) < 0.01
    assert [x.name for x in brushfile.saved(tmp_path)] == ["My chalk"]
    for br in (b, BrushSettings("e", eraser=True, size=30), BrushSettings("s", mode="smudge", size=30)):
        img = brushfile.preview(br, 200, 70)
        assert img.shape == (70, 200, 3) and img.dtype == np.uint8
        assert np.abs(img.astype(int) - img[2, 2].astype(int)).max() > 40 or br.mode == "smudge" or br.eraser


def test_new_engines_spray_chalk_sketch_hatching_shape_clone_normal():
    flat = {"size": {}, "opacity": {}, "flow": {}}
    # spray: particles spread over the brush's width, more particles than steps
    doc = Document(800, 200)
    st = paint_stroke(doc.layer, BrushSettings("s", size=80, mode="spray", count=4, spacing=0.3, dynamics=flat),
                      _line(100, 700, 100, 1, 1))
    a = doc.layer.read(0, 0, 800, 200)[..., 3]
    rows = np.nonzero(a.max(1) > 0.3)[0]
    assert rows.max() - rows.min() > 40 and 0.02 < _cover(doc, (100, 60, 700, 140)) < 0.9
    # chalk: a light touch leaves more gaps than a heavy one
    cov = []
    for p in (0.2, 1.0):
        d = Document(800, 200)
        paint_stroke(d.layer, BrushSettings("c", size=30, mode="chalk", spacing=0.3, dynamics=flat),
                     _line(100, 700, 100, p, p))
        cov.append(_cover(d, (100, 85, 700, 115)))
    assert cov[0] < cov[1] * 0.8
    # hatching: lines at the set spacing and angle (0 deg = horizontal lines)
    d = Document(400, 200)
    paint_stroke(d.layer, BrushSettings("h", size=120, hardness=1.0, mode="hatching", hatch_angle=0,
                                        hatch_spacing=10, crosshatch=1.0, dynamics=flat),
                 _line(100, 300, 100, 0.4, 0.4))
    col = d.layer.read(200, 60, 1, 80)[:, 0, 3]
    peaks = np.nonzero((col[1:-1] > 0.5) & (col[1:-1] >= col[:-2]) & (col[1:-1] >= col[2:]))[0]
    gaps = np.diff([k for k in peaks if k == peaks[0] or k - 1 not in peaks])
    assert len(gaps) >= 5 and np.all(np.abs(gaps[1:] - 10) <= 1), gaps   # the first gap touches the window edge
    row = d.layer.read(150, 100, 100, 1)[0, :, 3]
    assert np.ptp(row) < 0.05                                  # along a horizontal line: no crossing lines
    # sketch: lines join back to earlier points, so ink appears off the pen's own path
    d = Document(400, 400)
    pts = [Sample(200 + 120 * np.cos(t), 200 + 120 * np.sin(t), 0.8, k / 100)
           for k, t in enumerate(np.linspace(0, 1.6 * np.pi, 120))]
    paint_stroke(d.layer, BrushSettings("k", size=6, mode="sketch", sketch_reach=40, sketch_density=0.6,
                                        dynamics=flat), pts)
    a = d.layer.read(0, 0, 400, 400)[..., 3]
    yy, xx = np.mgrid[0:400, 0:400]
    off_path = np.abs(np.hypot(xx - 200, yy - 200) - 120) > 10
    assert int((off_path & (a > 0.02)).sum()) > 200           # ink where the pen never went: the web lines
    # shape: the outline is filled on lifting the pen
    d = Document(300, 300)
    sq = [Sample(x, y, 1, k / 50) for k, (x, y) in enumerate([(50, 50), (250, 50), (250, 250), (50, 250)])]
    paint_stroke(d.layer, BrushSettings("f", mode="shape", dynamics=flat), sq, colour=(0, 1, 0))
    assert d.layer.read(150, 150, 1, 1)[0, 0, 3] > 0.99 and d.layer.read(20, 20, 1, 1)[0, 0, 3] == 0
    # clone: copies red from the left half to the right half
    d = Document(200, 100)
    _red_blue(d)
    from studio.engine.paint.brush import make_stroke
    st = make_stroke(d.layer, BrushSettings("cl", size=20, hardness=1.0, mode="clone", dynamics=flat),
                     clone=(-100, 0))
    for s in _line(120, 180, 50, 1, 1):
        st.add(s)
    st.finish()
    c = d.layer.read(150, 50, 1, 1)[0, 0]
    assert c[0] > 0.9 and c[2] < 0.1
    with pytest.raises(LayerLocked):
        paint_stroke(d.layer, BrushSettings("cl", mode="clone", dynamics=flat), _line(120, 180, 50, 1, 1))
    # tangent normal: a mouse stroke to the right leans the normal right (red above mid), green near mid
    d = Document(400, 100)
    paint_stroke(d.layer, BrushSettings("n", size=20, hardness=1.0, mode="normal", dynamics=flat),
                 _line(50, 350, 50, 1, 1))
    n = d.layer.read(200, 50, 1, 1)[0, 0, :3]
    assert n[0] > 0.75 and abs(n[1] - 0.5) < 0.05 and n[2] > 0.75


def _tip_disc(n=24):
    yy, xx = np.mgrid[0:n, 0:n]
    return (np.hypot(xx - n / 2 + 0.5, yy - n / 2 + 0.5) < n / 2 - 1).astype(np.uint8) * 255


def _gbr(name, tip, spacing=20):
    import struct
    nb = name.encode() + b"\0"
    h, w = tip.shape
    return struct.pack(">IIIII", 28 + len(nb), 2, w, h, 1) + b"GIMP" + struct.pack(">I", spacing) + nb + tip.tobytes()


def _packbits(row: bytes) -> bytes:
    out, i = bytearray(), 0
    while i < len(row):
        j = i
        while j < len(row) and j - i < 128 and row[j] == row[i]:
            j += 1
        if j - i >= 2:
            out += bytes([257 - (j - i)]) + row[i:i + 1]
            i = j
        else:
            out += bytes([0]) + row[i:i + 1]
            i += 1
    return bytes(out)


def test_imports_gimp_photoshop_krita_clip_studio(tmp_path):
    import io
    import sqlite3
    import struct
    from PIL import Image, PngImagePlugin
    from studio.engine.paint.imports import ImportProblem, import_brushes
    tip = _tip_disc()
    (tmp_path / "a.gbr").write_bytes(_gbr("Disc", tip))
    b = import_brushes(tmp_path / "a.gbr")[0]
    assert b.name == "Disc" and abs(b.spacing - 0.2) < 1e-6 and np.array_equal(b.tip > 0.5, tip > 0)
    (tmp_path / "h.gih").write_bytes(b"Hose\n2 ncells:2\n" + _gbr("c1", tip) + _gbr("c2", 255 - tip))
    h = import_brushes(tmp_path / "h.gih")[0]
    assert h.name == "Hose" and len(h.tips) == 2
    # Photoshop v6.2: an 8BIM samp section with one raw and one RLE-packed tip
    def sample(t, compress):
        hgt, w = t.shape
        body = b"\0" * 301 + struct.pack(">iiiiHB", 0, 0, hgt, w, 8, compress)
        if compress:
            rows = [_packbits(t[y].tobytes()) for y in range(hgt)]
            body += b"".join(struct.pack(">H", len(r)) for r in rows) + b"".join(rows)
        else:
            body += t.tobytes()
        pad = (4 - len(body) % 4) % 4
        return struct.pack(">I", len(body)) + body + b"\0" * pad
    samp = sample(tip, 0) + sample(255 - tip, 1)
    abr = struct.pack(">HH", 6, 2) + b"8BIMdesc" + struct.pack(">I", 4) + b"xxxx" + b"8BIMsamp" + \
        struct.pack(">I", len(samp)) + samp
    (tmp_path / "ps.abr").write_bytes(abr)
    ps = import_brushes(tmp_path / "ps.abr")
    assert len(ps) == 2 and np.array_equal(ps[0].tip > 0.5, tip > 0) and np.array_equal(ps[1].tip > 0.5, tip == 0)
    # Photoshop v2: one sampled brush with a name
    name = "Leaf".encode("utf-16-be") + b"\0\0"
    v2 = struct.pack(">IH", 0, 30) + struct.pack(">I", 5) + name + b"\1" + b"\0" * 8 + \
        struct.pack(">iiiiHB", 0, 0, 24, 24, 8, 0) + tip.tobytes()
    (tmp_path / "old.abr").write_bytes(struct.pack(">HH", 2, 1) + struct.pack(">HI", 2, len(v2)) + v2)
    o = import_brushes(tmp_path / "old.abr")[0]
    assert o.name == "Leaf" and abs(o.spacing - 0.3) < 1e-6
    # Krita: the preset XML in a PNG text chunk
    bd = '<Brush type="auto_brush" spacing="0.07" angle="0"><MaskGenerator diameter="42" ratio="0.5" hfade="0.3" ' \
         'vfade="0.3" type="circle"/></Brush>'
    xml = ('<Preset paintopid="colorsmudge" name="Wet blender"><param name="brush_definition" type="string">'
           f'<![CDATA[{bd}]]></param><param name="OpacityValue" type="string">0.6</param>'
           '<param name="PressureSize" type="string">true</param></Preset>')
    meta = PngImagePlugin.PngInfo()
    meta.add_text("preset", xml)
    Image.new("RGB", (8, 8)).save(tmp_path / "k.kpp", "PNG", pnginfo=meta)
    k = import_brushes(tmp_path / "k.kpp")[0]
    assert (k.name, k.mode, k.size, k.roundness, k.opacity) == ("Wet blender", "smudge", 42, 0.5, 0.6)
    assert abs(k.hardness - 0.3) < 1e-6 and "pressure" in k.dynamics["size"]
    # Clip Studio: a SQLite file with the tip PNG in a blob, a thumbnail column ignored
    png = io.BytesIO()
    Image.fromarray(np.dstack([np.zeros_like(tip)] * 3 + [tip]), "RGBA").save(png, "PNG")
    db = tmp_path / "c.sut"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE Node (NodeName TEXT, NodeThumbnail BLOB)")
    con.execute("INSERT INTO Node VALUES ('Real pencil', ?)", (b"junk" + png.getvalue(),))
    con.execute("CREATE TABLE MaterialFile (FileData BLOB)")
    con.execute("INSERT INTO MaterialFile VALUES (?)", (b"header" + png.getvalue() + b"tail",))
    con.commit()
    con.close()
    c = import_brushes(db)[0]
    assert c.name == "Real pencil" and not c.tips and np.array_equal(c.tip > 0.5, tip > 0)
    (tmp_path / "x.abr").write_bytes(b"\0\x09")
    with pytest.raises(ImportProblem):
        import_brushes(tmp_path / "x.abr")


def test_brush_pack_has_100_good_brushes_that_all_paint():
    from studio.engine.paint.brush import MODES, make_stroke
    from studio.engine.paint.pack import CATEGORIES, pack
    brushes = pack()
    assert len(brushes) >= 100
    names = [b.name for b in brushes]
    assert len(names) == len(set(names))
    for b in brushes:
        assert b.best_for and b.weak_at and b.tags in CATEGORIES and b.mode in MODES, b.name
        doc = Document(160, 80)
        wet = b.mode == "watercolour"
        if not wet:                                              # watercolour needs an empty layer of its own
            _solid(doc, doc.layer, (0.5, 0.6, 0.7), rect=(0, 0, 80, 80))
        if wet and b.pigment == 0:
            continue                                             # clean water on blank paper leaves nothing
        st = make_stroke(doc.layer, b, (0.2, 0.1, 0.05), seed=1, clone=(-40, 0) if b.mode == "clone" else None)
        pts = _line(30, 130, 40, 0.3, 1.0, n=40)
        if b.mode == "shape":
            pts = [Sample(x, y, 1, k / 10) for k, (x, y) in enumerate([(30, 10), (130, 10), (130, 70), (30, 70)])]
        for s in pts:
            st.add(s)
        st.finish()
        if wet:
            _dry(doc, doc.layer)
        ref = np.zeros((80, 160, 4), np.float32) if wet else _ref_solid()
        changed = np.abs(doc.layer.read(0, 0, 160, 80) - ref).max()
        assert changed > 0.01, b.name                            # every brush leaves a mark (or takes one away)


def _ref_solid():
    doc = Document(160, 80)
    _solid(doc, doc.layer, (0.5, 0.6, 0.7), rect=(0, 0, 80, 80))
    return doc.layer.read(0, 0, 160, 80)


def _wc(**kw):
    kw.setdefault("dynamics", {"size": {}, "opacity": {}, "flow": {}})
    return BrushSettings("w", size=60, hardness=0.5, mode="watercolour", spacing=0.1, **kw)


def _dry(doc, lay, n=2000):
    k = 0
    while lay.wet is not None and lay.wet.active and k < n:
        lay.wet.step(4)
        k += 4
    if lay.wet is not None:
        lay.wet.force_dry()
    return k


def test_watercolour_dries_to_its_colour_with_dark_edges_and_conserves_pigment():
    doc = Document(300, 200)
    lay = doc.layer
    u = UndoStack(doc)
    paint_stroke(lay, _wc(water=0.8, pigment=1.0, granulation=0.0, edge_darkening=0.8),
                 [Sample(80 + i, 100, 1.0, i / 100) for i in range(140)], colour=(0.2, 0.4, 0.8), undo=u)
    assert lay.medium == "watercolour" and lay.blend == "multiply" and lay.wet.active
    total_wet = float(lay.wet.G.sum() * lay.wet.f ** 2 + lay.wet.D.sum())
    steps = _dry(doc, lay)
    assert 10 < steps < 2000 and lay.wet is None                     # it dries by itself
    img = doc.composite()
    # what it dried to: blue in the middle, close to the brush colour's strength
    c = img[100, 150, :3]
    assert c[2] > c[0] + 0.2
    # dark edges: near the top edge of the wash it is darker than in the middle
    col = img[:, 150, 0]
    inside = np.nonzero(col < 0.98)[0]
    edge = float(min(col[inside.min():inside.min() + 6].min(), col[inside.max() - 5:inside.max() + 1].min()))
    mid = float(col[100])
    assert edge < mid - 0.01, (edge, mid)                           # the darkest rim is darker than the middle
    # pigment is neither made nor lost by the flow (density sums match)
    from studio.engine.paint.tiles import read_tiles
    T = read_tiles(lay.tiles, 0, 0, 300, 200, 4)
    dens = -np.log(np.clip(1 - T[..., 3:4] + T[..., :3], 1e-4, 1)).sum()
    assert abs(float(dens) - total_wet) / total_wet < 0.03
    # undo takes the stroke and its drying back
    u.undo()
    assert not any(t[..., 3].any() for t in lay.tiles.values())


def test_watercolour_granulates_lifts_and_needs_its_own_layer():
    doc = Document(300, 200)
    paint_stroke(doc.layer, _wc(water=0.6, pigment=1.0, granulation=1.0, edge_darkening=0.0, grain="rough"),
                 [Sample(60 + i * 1.5, 100, 1.0, i / 100) for i in range(120)], colour=(0.6, 0.3, 0.2))
    _dry(doc, doc.layer)
    from studio.engine.paint import texture
    img = doc.composite()[85:115, 100:200, 0]
    paper = texture.patch(texture.grain("rough"), 100, 85, 100, 30)
    assert np.corrcoef(img.ravel(), paper.ravel())[0, 1] > 0.15     # lighter on ridges, darker in valleys
    before = float(doc.composite()[100, 150, 0])
    paint_stroke(doc.layer, _wc(water=0.9, pigment=0.0, lift=0.6),
                 [Sample(130 + i * 0.5, 100, 1.0, i / 100) for i in range(80)])
    _dry(doc, doc.layer)
    assert float(doc.composite()[100, 150, 0]) > before + 0.03       # lifted: lighter
    other = Document(100, 100)
    paint_stroke(other.layer, BrushSettings("r", size=10), _line(10, 90, 50, 1, 1))
    with pytest.raises(LayerLocked, match="layer of its own"):
        paint_stroke(other.layer, _wc(), _line(10, 90, 50, 1, 1))
    with pytest.raises(LayerLocked, match="watercolour layer"):
        paint_stroke(doc.layer, BrushSettings("r", size=10), _line(10, 90, 50, 1, 1))


def test_impasto_relief_light_knife_and_bristles():
    doc = Document(300, 120)
    b = BrushSettings("thick", size=30, hardness=0.8, spacing=0.05, impasto=0.5,
                      dynamics={"size": {}, "opacity": {}, "flow": {}})
    u = UndoStack(doc)
    paint_stroke(doc.layer, b, _line(40, 260, 60, 1, 1), colour=(0.8, 0.5, 0.3), undo=u)
    assert doc.layer.height and float(doc.relief(150, 60, 1, 1)[0, 0]) > 0.3
    flat = doc.composite()
    lit = doc.composite(light=(90.0, 35.0, 0.8))                     # light from above
    top, bottom = lit[60 - 13, 150, 0] - flat[60 - 13, 150, 0], lit[60 + 13, 150, 0] - flat[60 + 13, 150, 0]
    assert top > 0.02 and bottom < -0.02                              # the upper slope lit, the lower in shade
    assert np.allclose(lit[5, 5], flat[5, 5])                         # flat paper unchanged
    u.undo()
    assert not any(t.any() for t in doc.layer.height.values())       # thickness is undone with the stroke
    u.redo()
    knife = BrushSettings("k", size=30, hardness=1.0, mode="smudge", smudge_kind="smear", smudge_length=0.9,
                          spacing=0.05, dynamics={"size": {}, "opacity": {}, "flow": {}})
    paint_stroke(doc.layer, knife, [Sample(150, 60 + k * 0.5, 1, k / 100) for k in range(80)])
    assert float(doc.relief(150, 95, 1, 1)[0, 0]) > 0.05                   # the ridge pushed down the canvas
    # bristles: streaky strokes that run dry
    d = Document(800, 200)
    br = BrushSettings("b", size=40, mode="bristle", bristles=20, paint_load=0.3, pickup=0.0, spacing=0.05,
                       dynamics={"size": {}, "opacity": {}, "flow": {}})
    st = Stroke(d.layer, br, (0.1, 0.2, 0.6), seed=1)
    for smp in _line(50, 750, 100, 1, 1):
        st.add(smp)
    st.finish()
    a = d.layer.read(0, 0, 800, 200)[..., 3]
    start, end = a[70:130, 80:160].mean(), a[70:130, 600:700].mean()
    assert start > end + 0.05                                         # it runs dry along the stroke
    gaps = 0
    for x in range(100, 500, 10):                     # streaks: gaps open as bristles run dry unevenly
        col = a[:, x]
        rows = np.nonzero(col > 0.1)[0]
        if len(rows):
            gaps += int((col[rows.min():rows.max()] < 0.1).sum() > 0)
    assert gaps >= 3


def test_ora_keeps_relief_and_watercolour_layers(tmp_path):
    doc = Document(300, 120)
    paint_stroke(doc.layer, BrushSettings("t", size=30, impasto=0.6, spacing=0.05,
                                          dynamics={"size": {}, "opacity": {}, "flow": {}}), _line(40, 260, 60, 1, 1))
    wl = doc.add_layer("Wash")
    paint_stroke(wl, _wc(pigment=0.8), _line(40, 260, 60, 1, 1), colour=(0.3, 0.5, 0.7))
    _dry(doc, wl)
    p = str(tmp_path / "thick.ora")
    ora.save(doc, p)
    back = ora.load(p)
    assert back.layers[1].medium == "watercolour" and back.layers[1].blend == "multiply"
    assert abs(float(back.relief(150, 60, 1, 1)[0, 0]) - float(doc.relief(150, 60, 1, 1)[0, 0])) < 1e-3
    lit = (90.0, 35.0, 0.8)
    assert np.abs(back.composite(light=lit) - doc.composite(light=lit)).max() < 3 / 255


def test_selections_clip_painting_fill_and_edit():
    from studio.engine.paint import edit, select
    from studio.engine.paint.fill import fill
    doc = Document(200, 100)
    r = select.rect(200, 100, 20, 20, 80, 80)
    e = select.ellipse(200, 100, 100, 10, 180, 90)
    assert r[50, 50] == 1 and r[10, 10] == 0 and abs(e[50, 140] - 1) < 1e-6 and e[12, 102] == 0
    both = select.combine(r, e, "add")
    assert both[50, 50] == 1 and both[50, 140] == 1
    assert select.combine(both, r, "subtract")[50, 50] == 0 and select.combine(r, e, "intersect").max() == 0
    g, s = select.grow(r, 5), select.shrink(r, 5)
    assert g[50, 17] > 0.9 and s[50, 22] < 0.1 and s[50, 50] == 1
    f = select.feather(r, 8)
    assert 0.2 < f[50, 20] < 0.8
    doc.selection = r
    paint_stroke(doc.layer, BrushSettings("b", size=30, dynamics={"size": {}, "opacity": {}, "flow": {}}),
                 _line(0, 200, 50, 1, 1), clip=doc.selection)
    a = doc.layer.read(0, 0, 200, 100)[..., 3]
    assert a[50, 50] > 0.9 and a[50, 120] == 0                     # painted only inside the selection
    d2 = Document(200, 100)
    d2.selection = r
    fill(d2, 150, 50, (1, 0, 0))
    a2 = d2.layer.read(0, 0, 200, 100)[..., 3]
    assert a2[50, 50] > 0.99 and a2[50, 150] == 0
    w = select.wand(d2, 50, 50, 0.05)
    assert w[50, 50] == 1 and w[50, 150] == 0
    cr = select.colour_range(d2, (1, 0, 0), 0.05)
    assert cr[50, 50] == 1 and cr[50, 150] < 0.01
    clip = edit.copy(d2, d2.layer, select.rect(200, 100, 30, 30, 40, 40))
    lay = edit.paste(d2, clip)
    assert lay.read(35, 35, 1, 1)[0, 0, 3] > 0.99 and lay.read(50, 50, 1, 1)[0, 0, 3] == 0
    edit.clear(d2, d2.layers[0], select.rect(200, 100, 20, 20, 50, 80))
    assert d2.layers[0].read(30, 50, 1, 1)[0, 0, 3] == 0 and d2.layers[0].read(70, 50, 1, 1)[0, 0, 3] > 0.99
    assert select.bounds(r) == (20, 20, 80, 80)


def test_transforms_resample_once_from_the_original_and_undo():
    from studio.engine.paint import edit, transform
    doc = Document(200, 100)
    _solid(doc, doc.layer, (0, 0, 1), rect=(20, 20, 60, 60))
    u = UndoStack(doc)
    t = edit.TransformSession(doc, doc.layer)
    t.free(tx=30, ty=10)
    assert doc.layer.read(70, 50, 1, 1)[0, 0, 3] > 0.99 and doc.layer.read(30, 30, 1, 1)[0, 0, 3] == 0
    for _ in range(9):
        t.free(tx=30.3, ty=10.3)                                    # ten moves: still worked from the original
    t.free(tx=30, ty=10)
    assert doc.layer.read(51, 31, 38, 38)[..., 3].min() > 0.99     # edges are as sharp as after one move
    t.commit(u)
    u.undo()
    assert doc.layer.read(30, 30, 1, 1)[0, 0, 3] > 0.99
    u.redo()
    assert doc.layer.read(70, 50, 1, 1)[0, 0, 3] > 0.99
    t2 = edit.TransformSession(doc, doc.layer)
    t2.perspective(t2.corners())                                    # corners where they were: unchanged
    assert doc.layer.read(70, 50, 1, 1)[0, 0, 3] > 0.99
    t2.free(angle=90)
    a = doc.layer.read(0, 0, 200, 100)[..., 3]
    ys, xs = np.nonzero(a > 0.5)
    assert abs((xs.max() - xs.min()) - (ys.max() - ys.min())) <= 2  # a square stays square when turned
    t2.cancel()
    t3 = edit.TransformSession(doc, doc.layer)
    t3.liquify_push(50, 50, 15, 0, 20)                              # push right at the square's left edge
    assert doc.layer.read(54, 50, 1, 1)[0, 0, 3] < 0.5             # the edge moved right
    t3.cancel()
    H = transform.homography([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 0), (2, 0), (2, 2), (0, 2)])
    assert np.allclose(H, np.diag([2, 2, 1]), atol=1e-9)


def test_masks_clipping_adjustment_layers_and_filters(tmp_path):
    from studio.engine.paint import edit, select
    doc = Document(100, 100, background=(1, 1, 1))
    base = doc.layer
    _solid(doc, base, (1, 0, 0), rect=(0, 0, 50, 100))             # red left half
    top = doc.add_layer("Top")
    _solid(doc, top, (0, 0, 1), rect=(0, 0, 100, 100))              # blue everywhere
    top.clip = True
    img = doc.composite()
    assert img[50, 25, 2] > 0.99 and img[50, 25, 0] < 0.01 and img[50, 75, 1] > 0.99   # blue only over the red
    top.clip = False
    edit.add_mask(top, select.rect(100, 100, 0, 0, 100, 50), doc)  # show the top half only
    img = doc.composite()
    assert img[25, 75, 1] < 0.01 and img[75, 75, 1] > 0.99           # blue at the top, paper below
    u = UndoStack(doc)
    u.begin("mask")
    u.touch(top, ("m", 0, 0))
    top.mask[(0, 0)][:] = 1.0
    u.end()
    assert doc.composite()[75, 75, 1] < 0.01
    u.undo()
    assert doc.composite()[75, 75, 1] > 0.99                         # mask edits are undone
    doc.active = 1
    adj = doc.add_adjustment("invert")
    img = doc.composite()
    assert img[25, 75, 0] > 0.99 and img[25, 75, 2] < 0.01          # blue turned to yellow (inverted)
    adj.opacity = 0.0
    assert doc.composite()[25, 75, 1] < 0.01
    with pytest.raises(LayerLocked, match="adjustment layer"):
        doc.active = doc.index_of(adj)
        doc.paint_layer()
    d = Document(100, 100)
    _solid(d, d.layer, (1, 1, 1), rect=(40, 40, 60, 60))
    edit.filter_layer(d, d.layer, "blur", 1.0)
    a = d.layer.read(0, 0, 100, 100)[..., 3]
    assert 0.05 < a[50, 37] < 0.6 and a[50, 50] > 0.9               # blurred edges


def test_ora_keeps_masks_clipping_and_adjustments(tmp_path):
    from studio.engine.paint import edit, select
    doc = Document(100, 100)
    _solid(doc, doc.layer, (1, 0, 0), rect=(0, 0, 50, 100))
    top = doc.add_layer("Top")
    _solid(doc, top, (0, 0, 1), rect=(0, 0, 100, 100))
    top.clip = True
    edit.add_mask(top, select.rect(100, 100, 0, 0, 100, 50), doc)
    doc.add_adjustment("posterize", {"levels": 3})
    want = doc.composite()
    p = str(tmp_path / "m.ora")
    ora.save(doc, p)
    back = ora.load(p)
    assert back.layers[1].clip and back.layers[1].mask is not None and back.layers[2].is_adjust
    assert np.abs(back.composite() - want).max() < 2 / 255


def test_psd_round_trip_layers_groups_masks_clipping(tmp_path):
    from studio.engine.paint import edit, psd, select
    doc = Document(120, 80, background=(1, 1, 1))
    doc.layer.name = "Base"
    _solid(doc, doc.layer, (1, 0, 0), rect=(0, 0, 60, 80))
    top = doc.add_layer("Glaze")
    _solid(doc, top, (0, 0, 1), rect=(30, 0, 120, 80), a=0.5)
    top.blend, top.opacity, top.clip = "multiply", 0.8, True
    g = doc.add_group("Group")
    edit.add_mask(top, select.rect(120, 80, 0, 0, 120, 40), doc)
    want = doc.composite()
    p = str(tmp_path / "t.psd")
    psd.save(doc, p)
    back = psd.load(p)
    names = [(l.name, l.kind, None if l.parent is None else back.by_uid(l.parent).name) for l in back.layers]
    assert names == [("Base", "paint", None), ("Glaze", "paint", "Group"), ("Group", "group", None)], names
    gl = back.layers[1]
    assert gl.blend == "multiply" and abs(gl.opacity - 0.8) < 0.01 and gl.clip and gl.mask is not None
    back.background = (1, 1, 1)
    assert np.abs(back.composite() - want).max() < 3 / 255
    with pytest.raises(psd.PSDProblem):
        (tmp_path / "x.psd").write_bytes(b"nope")
        psd.load(str(tmp_path / "x.psd"))


def _lzf_literal(data: bytes) -> bytes:
    out = bytearray()
    for i in range(0, len(data), 32):
        chunk = data[i:i + 32]
        out += bytes([len(chunk) - 1]) + chunk
    return bytes(out)


def test_kra_layers_groups_and_lzf(tmp_path):
    import zipfile
    from studio.engine.paint import kra
    # LZF: a literal run then a back reference (copy 5 bytes from 3 back, overlapping)
    assert kra.lzf_decompress(bytes([2]) + b"abc" + bytes([(3 << 5) | 0, 2]), 8) == b"abcabcab"
    tile = np.zeros((4, 64, 64), np.uint8)                       # planar B, G, R, A: a red opaque tile
    tile[2], tile[3] = 255, 255
    body = b"\x01" + _lzf_literal(tile.tobytes())
    data = (b"VERSION 2\nTILEWIDTH 64\nTILEHEIGHT 64\nPIXELSIZE 4\nDATA 1\n" +
            f"0,0,LZF,{len(body)}\n".encode() + body)
    xml = ('<?xml version="1.0"?><DOC xmlns="http://www.calligra.org/DTD/krita"><IMAGE name="pic" width="100" '
           'height="80" colorspacename="RGBA"><layers>'
           '<layer nodetype="grouplayer" name="Figure" opacity="255" visible="1" passthrough="1"><layers>'
           '<layer nodetype="paintlayer" name="Red" filename="layer2" x="10" y="5" opacity="128" visible="1" '
           'compositeop="multiply" colorspacename="RGBA"/></layers></layer>'
           '<layer nodetype="vectorlayer" name="Shapes"/>'
           '</layers></IMAGE></DOC>')
    p = tmp_path / "old.kra"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "application/x-krita")
        z.writestr("maindoc.xml", xml)
        z.writestr("pic/layers/layer2", data)
    doc, notes = kra.load(str(p))
    assert [(l.name, l.kind) for l in doc.layers] == [("Red", "paint"), ("Figure", "group")]
    red = doc.layers[0]
    assert red.blend == "multiply" and abs(red.opacity - 128 / 255) < 1e-6 and doc.layers[1].blend == "pass"
    assert red.read(12, 7, 1, 1)[0, 0, 0] > 0.99 and red.read(5, 5, 1, 1)[0, 0, 3] == 0   # offset kept
    assert any("Shapes" in n for n in notes)


def test_filter_fill_file_layers_studio_curve_and_clean(tmp_path):
    from PIL import Image
    from studio.engine.paint import adjust
    doc = Document(80, 60, background=(1, 1, 1))
    _solid(doc, doc.layer, (0, 0, 0), rect=(30, 0, 50, 60))      # a black bar
    doc.add_special("filter", {"kind": "blur", "amount": 1.0}, "Blur")
    img = doc.composite()
    assert 0.05 < img[30, 28, 0] < 0.95                            # the bar's edge is blurred below the filter
    tile_edge = doc.composite(0, 0, 40, 60)                         # a part on its own matches the whole
    assert np.abs(tile_edge - img[:, :40]).max() < 1e-4
    doc.layers[-1].visible = False
    f = doc.add_special("fill", {"colour": (0.0, 0.5, 0.0)}, "Green")
    f.opacity = 0.5
    assert abs(doc.composite()[5, 5, 1] - 0.75) < 1e-4              # half green over white paper
    Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(tmp_path / "ref.png")
    fl = doc.add_special("file", {"path": str(tmp_path / "ref.png"), "x": 5, "y": 5}, "Reference")
    assert doc.composite()[7, 7, 0] > 0.99 and doc.composite()[30, 70, 0] < 0.9
    with pytest.raises(LayerLocked, match="file layer"):
        doc.active = doc.index_of(fl)
        doc.paint_layer()
    p = str(tmp_path / "s.ora")
    ora.save(doc, p)
    back = ora.load(p)
    assert [l.kind for l in back.layers] == ["paint", "filter", "fill", "file"]
    assert np.abs(back.composite() - doc.composite()).max() < 2 / 255
    # the Studio's Natural curve: mid-grey darkened by an S-curve keeps grey grey
    x = np.array([[0.4, 0.4, 0.4], [0.8, 0.3, 0.2]], np.float32)
    y = adjust.apply("natural_curve", {"points": [(0, 0), (0.25, 0.15), (0.75, 0.85), (1, 1)]}, x)
    assert abs(y[0, 0] - y[0, 1]) < 1e-3 and y[0, 0] < 0.4
    r0, r1 = x[1, 0] / x[1, 1], y[1, 0] / max(y[1, 1], 1e-6)
    assert abs(r0 - r1) / r0 < 0.25                                 # the colour mix is kept (roughly)
    # Clean (the Studio's scan cleaner) takes noise out of a flat area
    rng = np.random.default_rng(0)
    noisy = np.clip(0.5 + rng.normal(0, 0.05, (96, 96, 3)), 0, 1).astype(np.float32)
    px = np.concatenate([noisy, np.ones((96, 96, 1), np.float32)], -1)
    out = adjust.filter_pixels("clean", px, 1.0)
    assert out[..., :3].std() < noisy.std() * 0.6


def test_cage_mesh_and_subject_select():
    from studio.engine.paint import edit, select, transform
    w = transform.mean_value_coords(np.array([5.0]), np.array([5.0]), [(0, 0), (10, 0), (10, 10), (0, 10)])
    assert abs(w.sum() - 1) < 1e-9 and np.allclose(w, 0.25, atol=1e-6)   # the centre of a square: equal weights
    doc = Document(120, 80)
    _solid(doc, doc.layer, (1, 0, 0), rect=(20, 20, 50, 50))
    t = edit.TransformSession(doc, doc.layer)
    cage = [(15, 15), (55, 15), (55, 55), (15, 55)]
    t.cage(cage, [(x + 40, y) for x, y in cage])                    # move the whole cage 40 px right
    a = doc.layer.read(0, 0, 120, 80)[..., 3]
    assert a[35, 75] > 0.9 and a[35, 30] < 0.1
    t.cancel()
    t.mesh(np.zeros((6, 6, 2)))                                     # an unmoved mesh changes nothing
    assert doc.layer.read(35, 35, 1, 1)[0, 0, 3] > 0.99
    t.cancel()
    try:
        m = select.subject(doc, 35, 35)
        assert m.shape == (80, 120)
    except select.SelectProblem:
        pass                                                        # no SAM model in this test copy: said plainly


def test_working_spaces_hdr_exr_and_psd_profiles(tmp_path):
    import struct
    import zlib
    from studio.engine.paint import exr, psd, space
    from studio.engine.paint.export import export
    from studio.engine.paint.colour import in_gamut, oklch_to_space, srgb_to_oklch
    c = np.array([0.8, 0.3, 0.1], np.float32)
    for sp in space.SPACES:
        back = space.convert(space.convert(c, "srgb", sp), sp, "srgb")
        assert np.abs(back - c).max() < 1e-4, sp                    # every space round-trips
    L, C, H = (float(v) for v in srgb_to_oklch(np.array([0.0, 1.0, 0.0], np.float32)))
    assert in_gamut(L, C * 1.05, H, "displayp3") and not in_gamut(L, C * 1.05, H, "srgb")
    p3 = oklch_to_space(L, C * 1.05, H, "displayp3")
    assert max(p3) <= 1 and min(p3) >= 0
    doc = Document(40, 30, background=(1, 1, 1))
    _solid(doc, doc.layer, (0.8, 0.3, 0.1), rect=(0, 0, 20, 30))
    want = doc.composite()
    space.convert_document(doc, "displayp3")
    assert doc.space == "displayp3" and doc.layer.read(5, 5, 1, 1)[0, 0, 0] < 0.8   # same colour, P3 numbers
    from studio.engine.paint.export import flat_pixels
    assert np.abs(flat_pixels(doc, "srgb")[..., :3] - want[..., :3]).max() < 2e-3  # it still shows the same
    p = str(tmp_path / "p3.psd")
    psd.save(doc, p)
    assert psd.load(p).space == "displayp3"                          # the profile travels with the file
    # a linear HDR painting: values above 1 survive an EXR round trip
    hdr = Document(16, 8, background=None)
    hdr.space = "linear"
    px = np.zeros((8, 16, 4), np.float32)
    px[..., 0], px[..., 3] = 4.0, 1.0
    hdr.layer.write(0, 0, px)
    e = str(tmp_path / "a_Studio.exr")
    export(hdr, e, "exr")
    got = exr.load(e)
    assert abs(got[3, 3, 0] - 4.0) < 1e-5 and abs(got[3, 3, 3] - 1) < 1e-6
    from studio.engine.io.export import is_own_export
    assert is_own_export(e)
    # EXR's ZIP compression undone: predictor and interleave reversed
    raw = np.arange(64, dtype=np.uint8).tobytes()
    a = np.frombuffer(raw, np.uint8)
    inter = np.concatenate([a[0::2], a[1::2]]).astype(np.int32)
    pred = np.concatenate([inter[:1], (np.diff(inter) + 128) & 0xFF]).astype(np.uint8)
    assert exr._undo_zip(zlib.compress(pred.tobytes())) == raw
    # PSD ZIP-with-prediction channel data
    from studio.engine.paint.psd import _R, _channel
    rows = np.array([[10, 12, 15], [200, 201, 199]], np.uint8)
    delta = np.concatenate([rows[:, :1], (np.diff(rows.astype(np.int16), axis=1) & 0xFF).astype(np.uint8)], 1)
    blob = struct.pack(">H", 3) + zlib.compress(delta.tobytes())
    ch = _channel(_R(blob), 2, 3, 8, len(blob))
    assert np.allclose(ch * 255, rows)


# ---------------------------------------------------------------- P6: drawing aids
def test_aids_assistants_snap():
    from studio.engine.paint import aids
    g = aids.Guide([aids.Assistant("ruler", [(0, 50), (100, 50)])])
    pts = [g(x, 50 + (3 if x % 2 else -3)) for x in range(0, 60, 2)]
    assert all(abs(y - 50) < 1e-6 for _, y in pts[5:])                 # held on the line once decided
    # perspective with two points: a stroke heading to the right VP stays on the line to it
    g = aids.Guide([aids.Assistant("perspective", [(-1000, 100), (1000, 100)])])
    for i in range(40):
        x, y = g(100 + i * 2, 300 - i * 0.4 + (1 if i % 2 else -1))
    vx, vy = 1000, 100
    cross = (x - 100) * (vy - 299) - (y - 299) * (vx - 100)          # the stroke starts at (100, 299)
    assert abs(cross) / math.hypot(vx - 100, vy - 299) < 1e-6
    # uprights stay upright
    g = aids.Guide([aids.Assistant("perspective", [(-1000, 100), (1000, 100)])])
    for i in range(30):
        x, y = g(200 + (1 if i % 2 else 0), 300 + i * 2)
    assert abs(x - 200) < 1e-6
    # concentric: keeps the radius it starts at
    g = aids.Guide([aids.Assistant("concentric", [(0, 0), (100, 0), (0, 50)])])
    for i in range(60):
        t = i * 0.05
        x, y = g(50 * math.cos(t) * 1.03, 25 * math.sin(t) * 0.97)
    assert abs(math.sqrt((x / 50) ** 2 + (y / 25) ** 2) - 1.03) < 1e-6     # the ring it started on
    # half strength moves halfway
    g = aids.Guide([aids.Assistant("ruler", [(0, 0), (100, 0)])], strength=0.5)
    for x in range(0, 40, 2):
        last = g(x, 4.0)
    assert abs(last[1] - 2.0) < 1e-6


def test_aids_symmetry_and_colour():
    from studio.engine.paint import aids
    assert aids.symmetry_maps("x", 100, 80)[0](10, 5) == (90, 5)
    r = aids.symmetry_maps("radial", 100, 100, spokes=4)
    assert len(r) == 3 and np.allclose(r[1](60, 50), (40, 50))
    assert len(aids.symmetry_maps("mandala", 100, 100, spokes=4)) == 7
    assert (130, 20) in [f(30, 20) for f in aids.symmetry_maps("wrap", 100, 80)]
    from studio.engine.paint.colour import srgb_to_oklch
    red = (0.8, 0.2, 0.2)
    h0 = float(srgb_to_oklch(red)[2])
    c = aids.harmony(red, "complementary")
    assert abs(((float(srgb_to_oklch(c[1])[2]) - h0 - 180) + 180) % 360 - 180) < 3
    green = (0.2, 0.7, 0.2)
    m = aids.apply_mask(green, "complementary", rotation=h0)       # sectors on red and its complement
    hm = float(srgb_to_oklch(m)[2])
    assert any(abs(((hm - (h0 + c0)) + 180) % 360 - 180) <= w / 2 + 1 for c0, w in aids.GAMUT_MASKS["complementary"][1])
    rg = aids.simulate_vision(np.array([[0.8, 0.2, 0.2], [0.2, 0.6, 0.2]], np.float32), "deutan")
    orig = np.abs(np.array([0.6, -0.4, 0]))
    assert np.abs(rg[0] - rg[1]).sum() < orig.sum()
    assert np.allclose(aids.simulate_vision(np.full((2, 3), 0.5, np.float32), "protan"), 0.5, atol=0.02)


def test_palette_from_image(tmp_path):
    from PIL import Image
    from studio.engine.paint import swatches
    a = np.zeros((40, 40, 3), np.uint8)
    a[:, :20] = (200, 20, 20)
    a[:, 20:] = (20, 20, 200)
    p = tmp_path / "ref.png"
    Image.fromarray(a).save(p)
    sw = swatches.from_image(str(p), count=2)
    assert len(sw.colours) == 2
    assert sw.colours[0][2] > 0.6 and sw.colours[1][0] > 0.6 or sw.colours[1][2] > 0.6 and sw.colours[0][0] > 0.6
