"""Colour match from a reference photo, solved as the Studio's own slider settings (local/colourmatch.py)."""
from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def astronaut(tmp_path_factory):
    skd = pytest.importorskip("skimage.data")
    from PIL import Image as PI
    from studio import engine
    p = str(tmp_path_factory.mktemp("cm") / "astronaut.png")
    PI.fromarray(skd.astronaut()).save(p)
    return engine.open(p)


def _dlab(a, b):
    from studio.engine.color.grade import to_oklab
    return float(np.abs(to_oklab(np.clip(a, 0, None)) - to_oklab(np.clip(b, 0, None))).mean())


def test_colour_match_recovers_known_settings(astronaut):
    """The reference is the same photo developed with known settings; the match must find settings that look the
    same, and here they should be close to the ones used."""
    from studio import engine
    im = astronaut
    ref = engine.Edit()
    ref.exposure_ev = -0.5
    ref.white_balance.mode = "multipliers"
    ref.white_balance.multipliers = [0.9, 1.0, 1.15]
    ref.colour.saturation = -30
    p, mult, rep = engine.colour_match_solve(im, engine.Edit(), im, ref)
    out = engine.colour_match_apply(engine.Edit(), p, mult, image=im)
    R, A, B = (engine.render(im, e, scale=0.5) for e in (ref, out, engine.Edit()))
    assert _dlab(A, R) < 0.15 * _dlab(B, R)
    assert abs(out.exposure_ev + 0.5) < 0.1 and abs(out.colour.saturation + 30) < 8
    m = np.array(out.white_balance.multipliers)
    assert abs(m[0] - 0.9) < 0.05 and abs(m[2] - 1.15) < 0.06
    assert out.white_balance.temperature > 0 and "cooler" in rep["text"]
    split = engine.Edit()                                            # split toning only: found by the wheels
    split.colour.grading = {"shadows": [180, 40, 0], "highlights": [40, 40, 0]}
    p, mult, rep = engine.colour_match_solve(im, engine.Edit(), im, split)
    out = engine.colour_match_apply(engine.Edit(), p, mult)
    g = out.colour.grading
    assert abs(((g["shadows"][0] - 180) + 180) % 360 - 180) < 25 and g["shadows"][1] > 20
    assert abs(((g["highlights"][0] - 40) + 180) % 360 - 180) < 30 and g["highlights"][1] > 20
    R, A = engine.render(im, split, scale=0.5), engine.render(im, out, scale=0.5)
    assert _dlab(A, R) < 0.4 * _dlab(engine.render(im, engine.Edit(), scale=0.5), R)


def test_colour_match_amount_is_additive():
    from studio import engine
    from studio.engine.local import colourmatch as cm
    p = np.array([0.4, 0.1, 0.05, -0.08, 10.0, 0.1, -0.05, 0.0, 0.08])
    mult = np.array([2.0, 1.0, 1.5])
    e0 = engine.Edit()
    z = engine.colour_match_apply(e0, p, mult, 0.0)
    assert z.exposure_ev == e0.exposure_ev and z.colour.saturation == e0.colour.saturation
    full = engine.colour_match_apply(e0, p, mult, 1.0)
    half = engine.colour_match_apply(e0, p, mult, 0.5)
    m_half = np.array(half.white_balance.multipliers)
    twice = cm.apply(half, p * 0.5, m_half)
    assert abs(twice.exposure_ev - full.exposure_ev) < 1e-6
    assert abs(twice.tone.contrast - full.tone.contrast) < 1e-3
    assert np.allclose(twice.white_balance.multipliers, full.white_balance.multipliers, atol=1e-4)
    assert abs(twice.colour.saturation - full.colour.saturation) < 0.11
    for k in ("shadows", "highlights"):
        assert np.allclose(twice.colour.grading[k][:2], full.colour.grading[k][:2], atol=0.2)
    assert "exposure +0.40" in cm.describe(p)


def test_colour_match_panel():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    assert page.cm_btn.text().startswith("Match colours") and not page.cm_amt.isVisibleTo(page)
    page._colour_match_amount()                                      # nothing matched yet: nothing happens
    app.processEvents()
