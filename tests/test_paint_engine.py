"""Paint's engine on synthetic strokes with known truth (BUILD_PLAN_PAINT P0/P1). Quick tests only."""
from __future__ import annotations

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
    for fmt in pexport.FORMATS:
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
