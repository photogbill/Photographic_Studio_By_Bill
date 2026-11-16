"""Cognitive Adjustments, batch 1: printing dots (halftone) found and taken out, the colourizer's input evened for
faded photos and its colour balance, and the "Show what changed" map."""
from __future__ import annotations

import os

import numpy as np
import pytest

from test_old_photo import _display, _write_tiff16
from test_plate_lab import _old_photo


def _logY(a):
    return np.log(np.maximum(a.astype(np.float64).mean(axis=2), 1e-3))


def test_halftone_screen_is_found_exactly_and_taken_out():
    from studio.engine.local import halftone as ht
    base, _ = _old_photo(5, 600, 800, reds=False)
    for period, angle in ((6.0, 45.0), (4.5, 15.0), (3.2, 30.0)):
        dot = ht.screen(base, period, angle)
        m = ht.measure(dot)
        assert m["peaks"] and abs(m["period"] - period) < 0.1
        assert min(abs(m["angle"] - angle % 90), 90 - abs(m["angle"] - angle % 90)) < 2.0
        out = ht.clean(dot, m["peaks"], {"on": True})
        err = lambda a: np.abs(_logY(a) - _logY(base)).mean()      # noqa: E731
        assert err(out) < 0.35 * err(dot)
    for seed in (1, 2, 3):                                          # a photo with no screen: nothing found
        clean, _ = _old_photo(seed, 600, 800, reds=False)
        assert ht.measure(clean)["peaks"] == []


def test_halftone_through_the_pipeline(tmp_path):
    from studio import engine
    from studio.engine.local import halftone as ht
    base, _ = _old_photo(8, 512, 640, reds=False)
    path = str(tmp_path / "printed.tif")
    _write_tiff16(path, _display(ht.screen(base, 5.0, 45.0)), "srgb")
    im = engine.open(path)
    e = engine.Edit()
    e.halftone = {"on": True}
    whole = engine.render(im, e)
    assert im.halftone_report["peaks"] and abs(im.halftone_report["period"] - 5.0) < 0.15
    plain = engine.render(im, engine.Edit())
    hp = lambda a: _logY(a) - _logY(a).mean()                       # noqa: E731
    from scipy import ndimage as ndi
    fine = lambda a: np.abs(hp(a) - ndi.gaussian_filter(hp(a), 2.0)).mean()   # noqa: E731
    assert fine(whole) < 0.5 * fine(plain)                          # the dot texture is mostly gone
    part = engine.render(im, e, region=(200, 150, 160, 120))
    assert np.abs(part - whole[150:270, 200:360]).mean() < 4e-3     # a 100% view matches the whole render
    assert e.raw_key() != engine.Edit().raw_key()
    cm = engine.change_map(im, e, 0.25)
    assert cm.max() > 0.05 and cm.shape[0] > 50                     # the change map sees the work


def test_colourizer_input_is_evened_and_balance_moves_the_colour():
    from studio.engine.local import colourize as cz
    L = np.linspace(30, 60, 1000, dtype=np.float32).reshape(20, 50)  # a faded photo: L* 30-60 only
    Le = cz.even_tones(L)
    assert Le.min() < 8 and Le.max() > 92
    full = np.linspace(2, 98, 1000, dtype=np.float32).reshape(20, 50)
    assert np.array_equal(cz.even_tones(full), full)                # a full-range photo is shown as it is
    rgb = np.full((16, 16, 3), 0.18, np.float32)
    ab = np.zeros((16, 16, 2), np.float32)
    lin = lambda Y: Y                                                # noqa: E731  identity display for the test
    t0, _, _ = cz.display_target(rgb, ab, 1.0, 1, 1.0, lin)
    tw, _, _ = cz.display_target(rgb, ab, 1.0, 1, 1.0, lin, warm=100.0)
    tc, _, _ = cz.display_target(rgb, ab, 1.0, 1, 1.0, lin, warm=-100.0)
    tm, _, _ = cz.display_target(rgb, ab, 1.0, 1, 1.0, lin, tint=100.0)
    assert tw[..., 0].mean() > t0[..., 0].mean() and tw[..., 2].mean() < t0[..., 2].mean()   # warmer
    assert tc[..., 2].mean() > t0[..., 2].mean()                                             # cooler
    assert tm[..., 1].mean() < t0[..., 1].mean()                                             # magenta
    from studio.engine import Edit
    assert Edit().stage_versions["colourize"] == 2
    assert Edit.from_dict({"stage_versions": {"colourize": 1}}).stage_versions["colourize"] == 1   # old: as was


def test_cognitive_panel_round_trip():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    e = E.Edit()
    e.halftone = {"on": True, "amount": 80.0}
    e.colourize = {"on": True, "amount": 1.2, "warm": 30.0, "tint": -20.0}
    page.controls_from_edit(e)
    page.edit = e
    back = page.edit_from_controls()
    assert back.halftone == {"on": True, "amount": 80.0}
    assert back.colourize == {"on": True, "amount": 1.2, "warm": 30.0, "tint": -20.0}
    page.controls_from_edit(E.Edit())
    page.edit = E.Edit()
    d = page.edit_from_controls()
    assert d.halftone is None and d.colourize is None
    from studio.engine.presets.looks import describe_change
    assert describe_change(E.Edit(), E.Edit.from_dict({"halftone": {"on": True}})).startswith("Printing dots")
    app.processEvents()
