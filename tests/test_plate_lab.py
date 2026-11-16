"""The Plate Lab: stains and corrosion are unmixed, not painted over, and the places where the picture may be gone
are marked, not invented. Tested on synthetic stains over a clean picture, where the right answer is known."""
from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import scene
from test_old_photo import _display, _write_tiff16


def _logY(a):
    return np.log(np.maximum(a.astype(np.float64).mean(axis=2), 1e-4))


def _old_photo(seed, h=384, w=512, reds=True):
    """A sepia-toned black and white picture, with (optionally) a few small magenta-reds of its own, as a
    Hillotype's lips and cheeks have: real colour that must not be taken for a stain."""
    g = scene(h, w, seed=seed).mean(axis=2, keepdims=True) * 0.7
    base = (g * np.array([1.06, 1.0, 0.90], np.float32)).astype(np.float32)
    red = np.zeros((h, w), bool)
    if reds:
        yy, xx = np.mgrid[0:h, 0:w]
        for cy, cx in ((0.3, 0.3), (0.6, 0.7), (0.75, 0.25)):
            red |= np.hypot(yy - cy * h, xx - cx * w) < 0.03 * h
        base[red] *= np.array([1.25, 0.80, 1.0], np.float32)
    return base, red


def test_stains_are_unmixed_and_the_picture_under_them_comes_back():
    from studio.engine.local import stains as st
    base, red = _old_photo(11)
    stained, wt, core = st.add_stains(base, n=45, seed=4)
    m = st.measure(stained)
    out = st.clean(stained, m["med_w"], m["med_g"], m["coef"], {"on": True}, angle=m["angle"])
    thin = (wt > 0.3) & ~core & ~red
    err = lambda a: np.abs(_logY(a) - _logY(base))[thin].mean()          # noqa: E731
    e0, e1 = err(stained), err(out)
    assert e0 > 0.08
    assert e1 < 0.45 * e0                                                 # most of the stain's brightness is gone
    _, w0, _ = st.axes(stained)
    _, w1, _ = st.axes(out)
    _, wb, _ = st.axes(base)
    assert np.abs(w1 - wb)[thin].mean() < 0.4 * np.abs(w0 - wb)[thin].mean()   # and its tint
    # the picture's own detail under a thin stain survives (unmixing, not painting): correlation of fine detail
    from scipy import ndimage as ndi
    hp = lambda a: _logY(a) - ndi.gaussian_filter(_logY(a), 3)            # noqa: E731
    c = np.corrcoef(hp(out)[thin], hp(base)[thin])[0, 1]
    assert c > 0.8
    # bloom cores are found for the overlay (the picture is gone there) and a clean photo is left nearly alone
    lost = m["lost"] > 0
    assert lost.any() and (lost & core).sum() / max(1, core.sum()) > 0.5
    mc = st.measure(base)
    oc = st.clean(base, mc["med_w"], mc["med_g"], mc["coef"], {"on": True}, angle=mc["angle"])
    assert np.abs(_logY(oc) - _logY(base)).mean() < 0.01
    assert mc["lost_frac"] < 0.01
    # the picture's own magenta-reds (away from the stains) keep their colour: they point another way than stains
    keep = red & (wt < 0.05)
    _, wr, _ = st.axes(out)
    assert keep.sum() > 50 and np.abs(wr - wb)[keep].mean() < 0.03


def test_stains_through_the_pipeline_whole_and_zoomed_agree(tmp_path):
    from studio import engine
    from studio.engine.local import stains as st
    base, _ = _old_photo(7, 256, 384, reds=False)
    stained, wt, core = st.add_stains(base, n=30, seed=9)
    path = str(tmp_path / "stained.tif")
    _write_tiff16(path, _display(stained), "srgb")
    im = engine.open(path)
    e = engine.Edit()
    e.stains = {"on": True}
    whole = engine.render(im, e)
    plain = engine.render(im, engine.Edit())
    assert im.stain_report and 0.005 < im.stain_report["stained"] < 0.6 and im.stain_report["coef"][0] > 0.2
    assert np.abs(whole - plain).max() > 0.02                            # it did something (on the stains)
    part = engine.render(im, e, region=(100, 60, 120, 90))
    assert np.abs(part - whole[60:150, 100:220]).max() < 3e-3            # a 100% view matches the whole render
    assert e.raw_key() != engine.Edit().raw_key()
    shp = engine.stain_lost_shape(im, e)
    assert shp and shp["type"] == "object" and shp["w"] > 0
    assert engine.stain_lost_shape(im, engine.Edit()) is None
    weaker = engine.Edit()
    weaker.stains = {"on": True, "amount": 50.0}
    half = engine.render(im, weaker)
    assert np.abs(half - plain).mean() < np.abs(whole - plain).mean()   # Strength scales the correction


def test_plate_lab_panel_round_trip():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    e = E.Edit()
    e.stains = {"on": True, "amount": 120.0, "sensitivity": 70.0}
    page.controls_from_edit(e)
    page.edit = e
    back = page.edit_from_controls()
    assert back.stains == {"on": True, "amount": 120.0, "sensitivity": 70.0}
    page.controls_from_edit(E.Edit())
    page.edit = E.Edit()
    assert page.edit_from_controls().stains is None
    from studio.engine.presets.looks import describe_change
    assert describe_change(E.Edit(), back).startswith("Stains")
    app.processEvents()
