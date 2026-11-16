"""Relight (local/relight.py): the light moved using the depth map's surfaces and the light measured on the photo."""
from __future__ import annotations

import os

import numpy as np
import pytest


def _sphere(n=200):
    yy, xx = (np.mgrid[0:n, 0:n] - n / 2 + 0.5) / (0.4 * n)
    r2 = xx ** 2 + yy ** 2
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    return xx, yy, z, r2 < 1


def test_normals_and_measured_light_on_a_known_shape():
    from studio.engine.local import relight as rl
    xx, yy, z, inside = _sphere()
    D = 0.3 + 0.5 * z                                      # nearer in the middle (near = larger)
    N = rl.normals(D)
    core = inside & (np.hypot(xx, yy) < 0.8)
    assert np.mean(N[..., 0][core & (xx > 0.3)]) > 0.2     # the right side faces right
    assert np.mean(N[..., 1][core & (yy < -0.3)]) < -0.2   # the top faces up (image y points down)
    lam = rl.light_vector(135.0, 35.0)                     # top left
    Y = np.clip(N @ lam, 0, None) * 0.8 + 0.1               # the photo, lit from the top left
    coef = rl.fit_light(Y, N)
    c = coef[1:]
    az = np.degrees(np.arctan2(-c[1], c[0]))
    assert abs(((az - 135) + 180) % 360 - 180) < 20         # measured from the top left
    g = rl.gain(N, coef, {"amount": 80, "azimuth": -45.0, "elevation": 35.0, "soft": 20})   # move it bottom right
    assert np.mean(g[core & (xx > 0.3) & (yy > 0.3)]) > 1.15 * np.mean(g[core & (xx < -0.3) & (yy < -0.3)])
    assert abs(float(np.median(g)) - 1.0) < 0.05            # overall brightness kept
    assert not rl.active(None) and rl.active({"amount": 10}) and rl.direction_name(90, 30) == "from the top"


@pytest.fixture(scope="module")
def astronaut_png(tmp_path_factory):
    skd = pytest.importorskip("skimage.data")
    from PIL import Image as PI
    p = str(tmp_path_factory.mktemp("rl") / "astronaut.png")
    PI.fromarray(skd.astronaut()).save(p)
    return p


def test_relight_through_the_pipeline(astronaut_png):
    from studio import engine
    from studio.engine.local import depth as dp
    if not dp.available():
        pytest.skip("depth model not installed")
    im = engine.open(astronaut_png)
    plain = engine.render(im, engine.Edit())
    res = {}
    for name, az in (("left", 180.0), ("right", 0.0)):
        e = engine.Edit()
        e.relight = {"amount": 70, "azimuth": az, "elevation": 30, "soft": 30}
        res[name] = engine.render(im, e)
    ratio = lambda a: a.mean(axis=2) / np.maximum(plain.mean(axis=2), 1e-4)      # noqa: E731
    fl, fr = (slice(70, 170), slice(175, 230)), (slice(70, 170), slice(230, 285))   # the face's two halves
    gl, gr = ratio(res["left"]), ratio(res["right"])
    assert gl[fl].mean() > gl[fr].mean() + 0.05 and gr[fr].mean() > gr[fl].mean() + 0.1
    assert "azimuth" in im.relight_report
    e = engine.Edit()
    e.relight = {"amount": 70, "azimuth": 180.0, "elevation": 30, "soft": 30}
    assert e.raw_key() != engine.Edit().raw_key()
    part = engine.render(im, e, region=(200, 100, 160, 120))
    assert np.abs(part - res["left"][100:220, 200:360]).mean() < 2e-3
    from scipy.ndimage import gaussian_filter
    tex = lambda a: a.mean(axis=2) - gaussian_filter(a.mean(axis=2), 2)                # noqa: E731
    assert np.corrcoef(tex(res["left"]).ravel(), tex(plain).ravel())[0, 1] > 0.97     # texture untouched


def test_relight_panel_round_trip():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio import engine as E
    from studio.app import main as M
    app = QApplication.instance() or QApplication([])
    page = M.DevelopPage(M.Settings())
    e = E.Edit()
    e.relight = {"amount": 40.0, "azimuth": -60.0, "elevation": 20.0, "soft": 55.0}
    page.controls_from_edit(e)
    page.edit = e
    assert page.edit_from_controls().relight == e.relight
    page.controls_from_edit(E.Edit())
    page.edit = E.Edit()
    assert page.edit_from_controls().relight is None
    from studio.engine.presets.looks import describe_change
    assert describe_change(E.Edit(), E.Edit.from_dict({"relight": {"amount": 30}})).startswith("Relight")
    app.processEvents()
