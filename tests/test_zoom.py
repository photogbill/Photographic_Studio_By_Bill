"""Develop's zoom (app/zoom.py and the Viewer): continuous zoom around the pointer, live dragging with no snap-back,
the navigator, click to zoom, and the sharp tile rendered for just the part on screen. A friend's first test
(2026-11-15) found zooming and dragging "very clunky": these pin down the fix."""
from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest

from studio.app import zoom as Z

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def state(ow=6000, oh=4000, vw=1200, vh=800, dpr=1.0):
    zs = Z.ZoomState()
    zs.set_photo(ow, oh)
    zs.set_view(vw, vh, dpr)
    return zs


# ------------------------------------------------------------------------------------------ geometry
def test_fit_and_photo_rect():
    zs = state()
    assert math.isclose(zs.fit_z(), min(1164 / 6000, 764 / 4000))
    l, t, w, h = zs.photo_rect()
    assert math.isclose(l + w / 2, 600) and math.isclose(t + h / 2, 400)          # centred when fitted
    zs2 = state(dpr=2.0)                                     # a 200% display scale: fit zoom doubles in device pixels
    assert math.isclose(zs2.fit_z(), 2 * zs.fit_z())
    assert zs2.photo_rect() == pytest.approx(zs.photo_rect())


def test_zoom_keeps_the_point_under_the_pointer():
    zs = state()
    anchor = (900.0, 300.0)
    u, v = zs.uv_at(*anchor)
    for z in (0.3, 0.5, 1.0, 2.0, 4.0):
        zs.zoom_to(z, anchor)
        assert zs.uv_at(*anchor) == pytest.approx((u, v), abs=1e-9)
    zs.zoom_to(None)
    assert zs.z is None and (zs.cu, zs.cv) == (0.5, 0.5)
    zs.zoom_to(zs.fit_z() * 0.5, anchor)                     # below fit = fit
    assert zs.z is None
    zs.zoom_to(50.0, anchor)
    assert zs.z == Z.MAX_ZOOM


def test_edges_stay_filled():
    zs = state()
    zs.zoom_to(1.0, (2.0, 2.0))                              # zoom at the very corner
    l, t, w, h = zs.photo_rect()
    assert l <= 1e-9 and t <= 1e-9                           # no empty band past the photo's edge
    zs.pan_by(5000, 5000)                                    # drag far past the corner
    l, t, w, h = zs.photo_rect()
    assert math.isclose(l, 0, abs_tol=1e-9) and math.isclose(t, 0, abs_tol=1e-9)
    zs.pan_by(-1e6, -1e6)
    l, t, w, h = zs.photo_rect()
    assert math.isclose(l + w, 1200) and math.isclose(t + h, 800)
    tall = state(ow=2000, oh=8000)                          # a portrait photo: narrower than the view at 25%
    tall.zoom_to(0.25)
    l, t, w, h = tall.photo_rect()
    assert w < 1200 and math.isclose(l + w / 2, 600)         # centred across, movable up and down


def test_wheel_is_smooth_and_stops_at_100():
    zs = state()
    seen = []
    for _ in range(40):
        zs.wheel(120, (600, 400))
        seen.append(zs.z)
    assert seen[-1] == Z.MAX_ZOOM
    assert 1.0 in seen                                       # crossing 100% lands on it exactly once
    ratios = [b / a for a, b in zip(seen, seen[1:]) if a and b and a != 1.0 and b != 1.0 and b < Z.MAX_ZOOM]
    assert all(r <= Z.NOTCH_FACTOR * 1.0001 for r in ratios)  # no jumps: at most one notch's worth each time
    zs.zoom_to(None)
    for _ in range(12):
        zs.wheel(15, (600, 400))                             # a touchpad: many small turns
    assert zs.z is not None and zs.z < 1.0
    for _ in range(80):
        zs.wheel(-120, (600, 400))
    assert zs.z is None


def test_steps():
    zs = state()
    got = []
    for _ in range(10):
        zs.step(+1)
        got.append(zs.z)
    assert got[:6] == pytest.approx([0.25, 1 / 3, 0.5, 2 / 3, 1.0, 2.0])
    for _ in range(10):
        zs.step(-1)
    assert zs.z is None


def test_tile_request_and_covers():
    zs = state()
    assert zs.tile_request() is None                        # fitted: the preview is enough
    zs.zoom_to(1.0)
    x, y, w, h, sc = zs.tile_request()
    vx, vy, vw, vh = zs.visible_window()
    assert sc == 1.0 and vw == pytest.approx(1200) and vh == pytest.approx(800)
    assert x <= vx and y <= vy and x + w >= vx + vw and y + h >= vy + vh       # what is on screen, and a margin
    assert w <= vw * 1.2 and h <= vh * 1.2
    uv = (x / 6000, y / 4000, (x + w) / 6000, (y + h) / 4000)
    assert Z.covers(uv, 1.0, zs.visible_window(), 6000, 4000, 1.0)
    zs.pan_by(-30, 20)                                       # a nudge stays inside the margin: no new render
    assert Z.covers(uv, 1.0, zs.visible_window(), 6000, 4000, 1.0)
    zs.pan_by(-600, 0)                                       # half a screen: a new render
    assert not Z.covers(uv, 1.0, zs.visible_window(), 6000, 4000, 1.0)
    assert not Z.covers(uv, 0.5, zs.visible_window(), 6000, 4000, 1.0)     # another zoom: another render
    zs.zoom_to(3.0)
    assert zs.tile_request()[4] == 1.0                       # above 100%: rendered at 100%, enlarged on screen
    big = state(vw=3800, vh=2100)
    big.zoom_to(1.0)
    x, y, w, h, sc = big.tile_request(margin=0.5, budget=9e6)
    assert w * h <= 9e6 * 1.01                               # the margin gives way to the pixel budget
    assert Z.label(None) == "Fit" and Z.label(1.0) == "100%" and Z.label(1 / 3) == "33%"


# ------------------------------------------------------------------------------------------ the viewer
def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _viewer(w=150 * 4, h=100 * 4):
    from studio.app import main as M
    app = _app()
    v = M.Viewer()
    v.resize(900, 600)
    v.after = M.to_qimage(np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8))
    v.zs.set_photo(6000, 4000)
    v._sync_view()
    return app, M, v


def _mouse(v, kind, pos, button=None, buttons=None):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    b = button or Qt.MouseButton.LeftButton
    t = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove,
         "release": QEvent.Type.MouseButtonRelease, "double": QEvent.Type.MouseButtonDblClick}[kind]
    held = buttons if buttons is not None else (b if kind in ("press", "move", "double") else Qt.MouseButton.NoButton)
    ev = QMouseEvent(t, QPointF(*pos), QPointF(*pos), b if kind != "move" else Qt.MouseButton.NoButton, held,
                     Qt.KeyboardModifier.NoModifier)
    {"press": v.mousePressEvent, "move": v.mouseMoveEvent, "release": v.mouseReleaseEvent,
     "double": v.mouseDoubleClickEvent}[kind](ev)


def _wheel(v, pos, delta):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    ev = QWheelEvent(QPointF(*pos), QPointF(*pos), QPoint(0, 0), QPoint(0, delta), Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    v.wheelEvent(ev)


def test_click_zooms_to_100_where_clicked_and_back():
    app, M, v = _viewer()
    r = v.photo_rect()
    pos = (r.left() + 0.7 * r.width(), r.top() + 0.3 * r.height())
    _mouse(v, "press", pos)
    _mouse(v, "release", pos)
    assert v.zoomed and v.zs.z == 1.0
    assert v._uv(v._px((0.7, 0.3), v.photo_rect())) == pytest.approx((0.7, 0.3))
    from PySide6.QtCore import QPointF
    assert v._uv(QPointF(*pos)) == pytest.approx((0.7, 0.3), abs=1e-6)     # the clicked point stayed put
    _mouse(v, "double", pos)                                 # the 2nd click of a double-click does not zoom out
    assert v.zoomed
    _mouse(v, "press", pos)
    _mouse(v, "release", pos)
    assert not v.zoomed


def test_drag_moves_the_photo_live_and_it_stays():
    from PySide6.QtCore import Qt
    app, M, v = _viewer()
    v.set_zoom(1.0)
    c0 = (v.zs.cu, v.zs.cv)
    l0 = v.photo_rect().left()
    _mouse(v, "press", (450, 300))
    _mouse(v, "move", (350, 260), buttons=Qt.MouseButton.LeftButton)
    assert v.photo_rect().left() == pytest.approx(l0 - 100)              # follows the mouse during the drag
    _mouse(v, "release", (350, 260))
    assert v.zoomed                                          # a drag is not a click: still zoomed
    assert v.photo_rect().left() == pytest.approx(l0 - 100)              # and no snap back on release
    assert (v.zs.cu, v.zs.cv) != c0
    # middle and right buttons move it too, even with a tool in hand
    v.tool = "brush"
    l1 = v.photo_rect().left()
    _mouse(v, "press", (450, 300), button=Qt.MouseButton.MiddleButton)
    _mouse(v, "move", (500, 300), buttons=Qt.MouseButton.MiddleButton)
    _mouse(v, "release", (500, 300), button=Qt.MouseButton.MiddleButton)
    assert v.photo_rect().left() == pytest.approx(l1 + 50)


def test_wheel_zooms_around_the_pointer_and_the_navigator_jumps():
    from PySide6.QtCore import QPointF
    app, M, v = _viewer()
    pos = (200.0, 150.0)
    uv = v._uv(QPointF(*pos))
    for _ in range(6):
        _wheel(v, pos, 120)
    assert v.zoomed and v.zs.z > v.zs.fit_z() * 4
    assert v._uv(QPointF(*pos)) == pytest.approx(uv, abs=1e-6)
    v.grab()                                                 # paint once: the navigator box is laid out
    box = v._nav
    assert box is not None and box.right() <= v.width() and box.bottom() <= v.height()
    target = QPointF(box.left() + 0.8 * box.width(), box.top() + 0.8 * box.height())
    _mouse(v, "press", (target.x(), target.y()))
    _mouse(v, "release", (target.x(), target.y()))
    assert v.zoomed                                          # a click in the navigator moves, it does not zoom out
    x, y, w, h = v.zs.visible_window()
    assert (x + w / 2) / 6000 == pytest.approx(0.8, abs=0.02) and (y + h / 2) / 4000 == pytest.approx(0.8, abs=0.02)
    for _ in range(30):
        _wheel(v, pos, -120)
    assert not v.zoomed


def test_tools_map_through_any_zoom():
    from PySide6.QtCore import QPointF
    app, M, v = _viewer()
    v.set_zoom(2.0, QPointF(300, 200))
    r = v.photo_rect()
    for uv in ((0.1, 0.2), (0.5, 0.5), (0.93, 0.71)):
        assert v._uv(v._px(uv, r)) == pytest.approx(uv, abs=1e-9)
    assert v._long_px(r) == pytest.approx(6000 * 2.0)        # brush sizes scale with the zoom
    v.grab()


# ------------------------------------------------------------------------------------------ the page
def _wait(app, cond, t=60.0):
    end = time.time() + t
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_page_renders_a_sharp_tile_for_the_view_and_reuses_it(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_SETTINGS_DIR", str(tmp_path))
    from conftest import scene, synthetic_raw
    from studio import engine as E
    from studio.app import main as M
    from studio.engine.pipeline.pipeline import Image
    app = _app()
    page = M.DevelopPage(M.Settings())
    page.resize(1300, 800)
    page.show()
    page.image = Image(synthetic_raw(scene(1200, 1800)))
    page.path = "synthetic.cr2"
    page.controls_from_edit(page.edit)
    page.render_now()
    assert _wait(app, lambda: page.viewer.after is not None and not page._render_live)
    v = page.viewer
    calls = []
    real = E.render_window
    monkeypatch.setattr(E, "render_window", lambda *a, **k: calls.append(a[2:]) or real(*a, **k))
    v.set_zoom(1.0)
    page._view_changed()
    assert _wait(app, lambda: v.tile is not None and not page._render_live)
    assert len(calls) == 1 and page.zoom_b.text() == "100%"
    (x, y, w, h), sc = calls[0]
    assert sc == 1.0
    ref = M.display8(real(page.image, page.edit, (x, y, w, h), 1.0))
    t = v.tile.convertToFormat(v.tile.Format.Format_RGB888)
    got = np.frombuffer(t.constBits(), np.uint8).reshape(t.height(), t.bytesPerLine())[:, :t.width() * 3]
    assert got.reshape(t.height(), t.width(), 3).shape == ref.shape
    assert np.abs(got.reshape(ref.shape).astype(int) - ref.astype(int)).max() <= 1   # the tile is that window
    v.zs.pan_by(-10, 5)                                      # a nudge inside the margin: no new render
    page._view_changed()
    app.processEvents()
    assert len(calls) == 1
    v.zs.pan_by(-v.width(), 0)                               # a screen away: a new one
    page._view_changed()
    assert _wait(app, lambda: len(calls) == 2 and not page._render_live)
    page.edit.exposure_ev = 0.7                              # a slider while zoomed: the tile first, then the preview
    page.render_now()
    assert _wait(app, lambda: len(calls) == 3 and not page._render_live and not page._dirty_base)
    page._escape()                                           # Esc with nothing in hand: the whole photo
    assert not v.zoomed
    assert _wait(app, lambda: v.tile is None and page.zoom_b.text() == "Fit", 5)
    page.close()
