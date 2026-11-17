"""Composite: a subject from another photo laid on this one (pipeline/composite.py)."""
from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def photos(tmp_path_factory):
    skd = pytest.importorskip("skimage.data")
    from PIL import Image as PI
    d = tmp_path_factory.mktemp("cp")
    src, dst = str(d / "astronaut.png"), str(d / "coffee.png")
    PI.fromarray(skd.astronaut()).save(src)
    PI.fromarray(skd.coffee()).save(dst)
    return src, dst


def _layer(src, model):
    from studio import engine
    im = engine.open(src)
    shp = engine.object_mask(im, engine.Edit(), [(0.45, 0.22), (0.3, 0.7)], [], model=model)
    return {"src": src, "mask": {"w": shp["w"], "h": shp["h"], "data": shp["data"]}, "x": 0.7, "y": 0.6,
            "size": 0.6, "rotate": 0.0, "flip": False, "grain": False, "feather": 1.0, "enabled": True}


def test_subject_is_placed_sized_and_only_there(photos):
    from studio import engine
    from studio.engine.local import sam
    from studio.engine.pipeline import pipeline as P
    src, dst = photos
    L = _layer(src, "tiny" if sam.available("tiny") else "classic")
    im = engine.open(dst)
    plain = engine.render(im, engine.Edit())
    e = engine.Edit()
    e.composite = [L]
    out = engine.render(im, e)
    H, W = out.shape[:2]
    diff = np.abs(out - plain).max(axis=2) > 0.02
    ys, xs = np.nonzero(diff)
    assert diff.mean() > 0.03                                        # something was laid on
    assert abs(0.5 * (ys.min() + ys.max()) / H - 0.6) < 0.06         # centred where asked
    assert abs(0.5 * (xs.min() + xs.max()) / W - 0.7) < 0.08
    assert abs((ys.max() - ys.min() + 1) / H - 0.6) < 0.08           # as tall as asked
    assert np.array_equal(out[:40, :40], plain[:40, :40])            # the rest untouched
    x, y, w, h = P.view_window(im, e, 0.7, 0.6, 200, 160)             # a 100% view matches the whole render
    v = engine.render_view(im, e, 0.7, 0.6, 200, 160)
    assert np.abs(v - out[y:y + h, x:x + w]).mean() < 0.02
    e2 = engine.Edit()
    e2.composite = [dict(L, flip=True, rotate=20.0)]
    assert np.abs(engine.render(im, e2) - out).mean() > 1e-3
    e3 = engine.Edit()
    e3.composite = [dict(L, enabled=False)]
    assert np.array_equal(engine.render(im, e3), plain)
    e.masks = [{"name": "m", "enabled": True, "adjust": {}, "shapes": [{"type": "luminance", "op": "add", "lo": 0.3,
                                                                         "hi": 1.0, "soft": 0.1}]}]
    m1 = engine.render_mask(im, e, 0, 0.5)                           # masks never see the subjects
    e.composite = None
    assert np.array_equal(m1, engine.render_mask(im, e, 0, 0.5))


def test_grain_is_matched_and_colours_follow_the_scene(photos):
    from studio import engine
    from studio.engine.local import sam
    from studio.engine.pipeline import composite as cp
    src, dst = photos
    L = _layer(src, "tiny" if sam.available("tiny") else "classic")
    im = engine.open(dst)
    rng = np.random.default_rng(0)
    plain = engine.render(im, engine.Edit())
    # Grain clearly above the subject's own: the astronaut photo measures about 0.046 by itself (its JPEG
    # texture), so a 5% grain left almost nothing to add and the old 1.5x check could not pass.
    noisy = (plain * np.exp(rng.standard_normal(plain.shape[:2]) * 0.10)[..., None]).astype(np.float32)
    eg, en = engine.Edit(), engine.Edit()
    eg.composite, en.composite = [dict(L, grain=True)], [dict(L, grain=False)]
    out, out0 = cp.apply(im, eg, noisy), cp.apply(im, en, noisy)
    al = np.abs(out0 - noisy).max(axis=2) > 0.02
    from scipy.ndimage import binary_erosion
    inside = binary_erosion(al, iterations=4)
    g, n0 = cp.noise_sigma(out, inside.astype(np.float32)), cp.noise_sigma(out0, inside.astype(np.float32))
    assert abs(g - cp.noise_sigma(noisy)) < 0.15 * cp.noise_sigma(noisy)     # the subject's grain matches the scene
    assert g > 1.5 * n0
    p, mult, rep = engine.colour_match_solve(cp._source(src), engine.Edit(), im, engine.Edit())
    Lm = dict(L, match={"p": list(map(float, p)), "mult": list(map(float, mult)), "amount": 1.0})
    e0, e1 = engine.Edit(), engine.Edit()
    e0.composite, e1.composite = [L], [Lm]
    a0, a1 = engine.render(im, e0), engine.render(im, e1)
    assert np.abs(a1 - a0).mean() > 1e-3                           # the subject is developed differently
    from studio.engine.local import colourmatch as cmod
    srci = cp._source(src)
    target = cmod.stats(plain)                                      # the subject's photo, developed with the match,
    before = cmod.stats(engine.render(srci, cp.source_edit(L), scale=0.5))   # has this photo's colour statistics
    after = cmod.stats(engine.render(srci, cp.source_edit(Lm), scale=0.5))
    assert np.linalg.norm((after - target) * cmod.WEIGHT) < 0.3 * np.linalg.norm((before - target) * cmod.WEIGHT)


def test_composite_panel_round_trip(photos):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    L = {"src": photos[0], "name": "astronaut 1", "mask": {"w": 4, "h": 4, "data": ""}, "x": 0.3, "y": 0.4,
         "size": 0.5, "rotate": 10.0, "flip": True, "grain": True, "feather": 2.0, "enabled": True, "match": None}
    e = E.Edit()
    e.composite = [L]
    page.controls_from_edit(e)
    page.edit = e
    assert page.edit_from_controls().composite == [L] and page.cp_list.count() == 1
    page.cp_size.set_value(80)
    page._composite_params()
    assert page.edit_from_controls().composite[0]["size"] == 0.8
    page.controls_from_edit(E.Edit())
    page.edit = E.Edit()
    assert page.edit_from_controls().composite is None
    app.processEvents()
