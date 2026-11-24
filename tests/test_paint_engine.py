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
