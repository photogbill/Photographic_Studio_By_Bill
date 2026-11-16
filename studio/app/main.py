"""Photographic Studio By Bill - the desktop app (Phase 4 shell, Phase 5 lens and shape, Phase 6 develop panels).

One window, a sidebar of workspaces (section 1, rule 0.3.12): Library, Develop, Dual Pixel Lab, Focus Stack,
Super-Res Stack, Lenses, Cameras, Export, Help. The interface talks to the engine only through its API
(studio.engine) and never blocks: every render, thumbnail and export runs in the background.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import traceback

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QFont, QImage, QKeySequence,
                           QLinearGradient, QPainter, QPainterPath, QPainterPathStroker, QPen,
                           QShortcut)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
                               QFileDialog, QFrame, QInputDialog,
                               QGridLayout, QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem, QMainWindow,
                               QMenu, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
                               QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from .. import __version__, log as slog, paths
from . import theme as T
from . import zoom as Z
from .widgets import (CurveEditor, DetailMeter, FilmicPlot, GradeWheel, GroupDialog, HueRing, SwatchRow, oklch_colour,
                      srgb8_hue)

from .common import (GLOW_HOWTO, POOL, STARS, _LIVE, Job, LensDialog, Settings, _Sig, card, chip,  # noqa: F401
                     dot_icon, hbox, label, load_thumb, qalpha, run_bg, tag_lens, to_qimage)
from .common import display8  # noqa: F401
from .exporting import ExportDialog, QLineEdit_, version_export_path  # noqa: F401
from .library import LibraryPage  # noqa: F401  (section 24: the Library lives in its own file)

log = slog.get("app")


def bw8(a8: np.ndarray) -> np.ndarray:
    """A display image as black and white (view only: "Judge in black and white" in Detail by scale)."""
    y = (a8[..., 0].astype(np.float32) * 0.2126 + a8[..., 1] * 0.7152 + a8[..., 2] * 0.0722 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(y[..., None], 3, axis=2))


def changed_keys(a, b) -> frozenset:
    """Which settings differ between two edits ('light.shadows', 'exposure_ev', ...)."""
    da, db = a.to_dict(), b.to_dict()
    out = set()
    for k in set(da) | set(db):
        va, vb = da.get(k), db.get(k)
        if va == vb:
            continue
        if isinstance(va, dict) and isinstance(vb, dict):
            out |= {f"{k}.{x}" for x in set(va) | set(vb) if va.get(x) != vb.get(x)}
        else:
            out.add(k)
    return frozenset(out)


# Copied settings, shared by Develop and the Library (Copy in one, Paste in either)
CLIP: dict = {"edit": None, "groups": [], "source": ""}
EYE_COLOURS = [("none", "Unchanged"), ("blue", "Blue"), ("green", "Green"), ("hazel", "Hazel"), ("grey", "Grey"),
               ("amber", "Amber"), ("brown", "Brown"), ("violet", "Violet"), ("custom", "Pick a colour...")]
# Demosaic methods (plan 23.8, rule 23.1: each with what it is best for and weak at; measured in RESULTS.md)
DEMOSAICS = [
    ("rcd", "RCD (default)", "Best for: fine detail and round edges, and it held up best with noise added in our "
                             "tests. Weak at: nothing measured so far; on very noisy skies try Dual."),
    ("menon", "Menon", "The Studio's first default (edits made before 2026-11-15 use it). Best for: clean "
                       "low-ISO detail. Weak at: a little behind RCD everywhere we measured."),
    ("lmmse", "LMMSE", "Best for: smooth, quiet results in noise; fewest maze patterns. Weak at: slightly softer "
                       "fine detail."),
    ("dual", "Dual (RCD + LMMSE)", "Best for: high ISO with both detail and big flat areas: RCD where there is "
                                   "detail, LMMSE in skies and skin. Weak at: about twice as slow."),
    ("mhc", "MHC (fast)", "Best for: speed. Weak at: zipper edges and colour fringes on fine detail."),
]
RANGE_DEFAULT = {"hue": 25.0, "width": 40.0, "falloff": 25.0, "shift": 0.0, "sat": 0.0, "lum": 0.0}


class ValueSlider(QWidget):
    """Label, slider, value. Double-click resets. Scroll nudges. Tooltip explains in plain English."""
    changed = Signal(float)

    def __init__(self, name, lo, hi, default, step=0.01, fmt="{:+.2f}", tip=""):
        super().__init__()
        self.lo, self.hi, self.default, self.step, self.fmt = lo, hi, default, step, fmt
        self.n = int(round((hi - lo) / step))
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setVerticalSpacing(2)
        self.name = label(name)
        self.val = label("", "dim")
        self.val.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.s = QSlider(Qt.Orientation.Horizontal)
        self.s.setRange(0, self.n)
        lay.addWidget(self.name, 0, 0)
        lay.addWidget(self.val, 0, 1)
        lay.addWidget(self.s, 1, 0, 1, 2)
        self.setToolTip(tip + "\nDouble-click to reset." if tip else "Double-click to reset.")
        self._exact = float(default)
        self.s.valueChanged.connect(self._moved)
        self.set_value(default, emit=False)

    def value(self) -> float:
        return self._exact

    def set_value(self, v, emit=True):
        """Show ``v`` exactly (a value set by Auto or a look is not snapped to the slider's step)."""
        self._exact = float(min(max(v, self.lo), self.hi))
        self.s.blockSignals(True)
        self.s.setValue(int(round((self._exact - self.lo) / self.step)))
        self.s.blockSignals(False)
        self.val.setText(self.fmt.format(self._exact))
        if emit:
            self.changed.emit(self._exact)

    def _moved(self, pos):
        self._exact = self.lo + pos * self.step
        self.val.setText(self.fmt.format(self._exact))
        self.changed.emit(self._exact)

    def mouseDoubleClickEvent(self, e):
        self.set_value(self.default)


class ElideLabel(QLabel):
    """A label that shortens its text with "..." when there is no room (the photo's name in Develop's top bar: a
    long file name used to push the window wider than a 1366-pixel laptop screen). The full text is the tooltip."""

    def __init__(self, text="", name=None, min_width=60):
        super().__init__()
        if name:
            self.setObjectName(name)
        self._full = ""
        self.setMinimumWidth(min_width)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)   # full name, or less
        self.setText(text)

    def setText(self, text):                        # noqa: N802 (Qt name)
        self._full = text or ""
        self.setToolTip(self._full)
        self._elide()

    def text(self):
        return self._full

    def sizeHint(self):
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 8, super().sizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def _elide(self):
        QLabel.setText(self, self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle,
                                                           max(10, self.width())))


class FlexBar(QFrame):
    """A top bar whose less-used controls fold into a "..." menu when the window is narrow, so the bar never sets
    the window's minimum width. On Linux a maximized or full-screen window whose minimum width grows past the
    screen is dropped back to a normal window: in 0.1.3 every opened photo did that (its file name widened the
    bar, already 1,521 pixels with the panel and sidebar)."""

    TITLE_ROOM = 170                                # the title's share counted when deciding to fold

    def __init__(self):
        super().__init__()
        self.setObjectName("topbar")
        self.lay = QHBoxLayout(self)
        self.more = QToolButton()
        self.more.setText("⋯")
        self.more.setToolTip("More: the controls there is no room for")
        self.more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_menu = QMenu(self.more)
        self.more_menu.aboutToShow.connect(self._fill_more)
        self.more.setMenu(self.more_menu)
        self.more.hide()
        self.title = None
        self.fixed: list = []                       # always shown
        self.folds: list = []                       # (widget, fill(menu)) folded first
        self._busy = False
        self.folded = False

    def setup(self, title, folds, fixed):
        self.title = title
        self.folds, self.fixed = list(folds), list(fixed)
        self.lay.addWidget(title)
        self.lay.addStretch(1)
        for w, _ in self.folds:
            self.lay.addWidget(w)
        self.lay.addWidget(self.more)
        for w in self.fixed:
            self.lay.addWidget(w)
        self._set_minimum()

    def _widths(self):
        m = self.lay.contentsMargins()
        sp = self.lay.spacing()
        fixed = m.left() + m.right() + sum(w.sizeHint().width() + sp for w in self.fixed)
        folds = sum(w.sizeHint().width() + sp for w, _ in self.folds)
        return fixed, folds, self.more.sizeHint().width() + sp

    def _set_minimum(self):
        fixed, _, more = self._widths()
        self.setMinimumWidth(fixed + more + self.title.minimumWidth() + self.lay.spacing())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _fit(self):
        if self._busy or self.title is None:
            return
        self._busy = True
        try:
            fixed, folds, more = self._widths()
            title = min(self.title.sizeHint().width(), self.TITLE_ROOM)
            fold = self.width() < fixed + folds + title
            if fold != self.folded or self.more.isVisible() != fold:
                self.folded = fold
                for w, _ in self.folds:
                    w.setVisible(not fold)
                self.more.setVisible(fold)
                log.debug("top bar %s at %d px", "folded" if fold else "unfolded", self.width())
        finally:
            self._busy = False

    def _fill_more(self):
        self.more_menu.clear()
        for _, fill in self.folds:
            fill(self.more_menu)


class Group(QWidget):
    """A collapsible panel group (Clean, Light, Colour, ...). Click the header to fold it away."""

    def __init__(self, title, subtitle=""):
        super().__init__()
        self.title = title
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 6)
        self.btn = QPushButton(f"▾  {title}")
        self.btn.setObjectName("group")
        self.btn.setCheckable(True)
        self.btn.setChecked(True)
        self.btn.toggled.connect(lambda on: self.btn.setText(f"{'▾' if on else '▸'}  {title}"))
        self.body = QWidget()
        self.lay = QVBoxLayout(self.body)
        self.lay.setContentsMargins(6, 6, 6, 6)
        self.lay.setSpacing(6)
        if subtitle:
            self.lay.addWidget(label(subtitle, "dim", wrap=True))
        v.addWidget(self.btn)
        v.addWidget(self.body)
        self.btn.toggled.connect(self.body.setVisible)

    def add(self, w):
        self.lay.addWidget(w)
        return w


class Histogram(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(96)
        self.h = None
        self.clip = (0.0, 0.0)

    def set_image(self, a8: np.ndarray | None):
        if a8 is None:
            self.h = None
        else:
            s = a8[::2, ::2].reshape(-1, 3)
            self.h = [np.bincount(s[:, c], minlength=256).astype(np.float32) for c in range(3)]
            n = s.shape[0]
            self.clip = (float((s.max(1) >= 254).sum()) / n, float((s.min(1) <= 1).sum()) / n)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setPen(QPen(QColor(T.LINE)))
        p.setBrush(QColor(T.BG2))
        p.drawRoundedRect(r, 10, 10)
        if self.h is not None:
            m = max(float(np.percentile(np.concatenate(self.h), 99.5)), 1.0)
            for c, col in enumerate(("#ef4444", "#22c55e", "#3b82f6")):
                path = QPainterPath(QPointF(r.left(), r.bottom()))
                for i in range(256):
                    x = r.left() + r.width() * i / 255.0
                    y = r.bottom() - min(1.0, self.h[c][i] / m) * (r.height() - 8)
                    path.lineTo(x, y)
                path.lineTo(r.right(), r.bottom())
                q = QColor(col)
                q.setAlpha(70)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(q)
                p.drawPath(path)
            p.setPen(QColor(T.DIM))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(r.adjusted(8, 4, -8, -4), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
                       f"highlights clipped {100 * self.clip[0]:.1f}%   shadows {100 * self.clip[1]:.1f}%")


class Viewer(QWidget):
    """The big live image: fit to window or any zoom up to 400% (wheel around the pointer, click for 100%, drag to
    move, a navigator in the corner), before/after wipe, the crop frame, and the mask tools (brush, gradient,
    radial, click-an-object, colour pick) with a red overlay of the selected mask.

    Zoomed, the whole-photo preview (``after``) is drawn enlarged at once and a sharp render of the part on screen
    (``tile``, at ``tile_uv`` on the photo) is laid over it when it arrives, so the photo always follows the mouse."""
    clicked = Signal(float, float)
    crop_changed = Signal(object)
    stroke_done = Signal(object)                 # brush: [(u, v), ...]
    drag_done = Signal(float, float, float, float)   # gradient / radial: start and end (u, v)
    tool_click = Signal(float, float, bool)      # object / colour: (u, v, alt held)
    spot_new = Signal(object)                    # retouch: [(u, v), ...] (one point = a round spot)
    spot_moved = Signal(int, str, float, float)  # retouch: spot index, "dst" | "src", moved by (du, dv)
    spot_picked = Signal(int)                    # retouch: spot selected (-1 = none)
    spot_deleted = Signal(int)                   # retouch: Alt-click on a spot
    view_changed = Signal()                      # zoom or position changed (sent once the movement settles)
    eyes_done = Signal(object)                   # portrait: the two eye clicks [(u, v), (u, v)]
    sculpt_done = Signal(object)                 # portrait: a sculpting stroke [(u, v), ...]
    shape_drawn = Signal(float, float, float, float)   # portrait: a new shape from one end to the other
    shape_picked = Signal(int)                   # portrait: shape selected (-1 = none)
    shape_moved = Signal(int, float, float)      # portrait: shape index, moved by (du, dv)
    busy_changed = Signal(bool)                  # work started (True) or finished (False): the panel's chip follows

    BUSY_SHOW_S = 0.15                           # quick renders finish before the indicator appears (no flicker)

    def __init__(self):
        super().__init__()
        self.after = self.before = None
        self.split = None
        self._busy = False
        self.busy_since = 0.0                      # perf_counter() when the current work started
        self.busy_took = 0.0                       # how long the last piece of work took, seconds
        self._spin = QTimer(self, interval=40, timeout=self.update)   # animates the indicator only while busy
        self.message = "Open a photo from the Library (double-click it)."
        self._drag = False
        self.crop_mode = False
        self.comp = "thirds"                       # composition overlay (local/composition.py) or "off"
        self.comp_orient = 0                       # Shift+O: mirrored left-right / top-bottom
        self.comp_outside = False                  # also on the whole photo, outside the crop tool (after O)
        self.comp_flash = 0.0                      # when the overlay last changed (its name shows for a moment)
        self.crop = (0.0, 0.0, 1.0, 1.0)          # fractions of the displayed (uncropped) frame
        self.crop_ratio = None                     # width / height in pixels, None = free
        self._cdrag = None
        self.tool = None                           # None | brush | linear | radial | object | colour
        self.overlay = None                        # QImage (RGBA) of the selected mask, drawn over the photo
        self.brush_frac = 0.03                     # brush radius as a fraction of the photo's long edge
        self.brush_scale = 1.0                     # S-frame long edge / displayed long edge (crops)
        self._pts = []
        self._mouse = None
        self._cursor_before = None                 # the tool's cursor while a drag moves the photo
        self.zs = Z.ZoomState()                    # the zoom and where the photo sits (zoom.py)
        self.tile = None                           # sharp render of part of the photo (QImage), when zoomed
        self.tile_uv = None                        # where it sits on the photo: (u0, v0, u1, v1)
        self.tile_scale = 0.0                      # the scale it was rendered at
        self._press = None                         # (position, centre u, v) of a left press that may become a click
        self._space = False                        # Space held: dragging moves the photo, whatever the tool
        self._nav = None                           # the navigator's box (QRectF) while zoomed
        self._nav_drag = False
        self.zoom_flash = 0.0                      # when the zoom last changed (its hint shows for a moment)
        self._settle = QTimer(self, singleShot=True, interval=140, timeout=self._settled)
        self._hint_timer = QTimer(self, singleShot=True, interval=3200, timeout=self.update)   # hides the hint
        self.spots = []                            # retouch spots to draw: {"pts", "src", "r", "kind", "sel", "auto"}
        self.show_spots = False
        self._sdrag = None                         # (index, "dst" | "src", start uv)
        self._pan = None                           # (start position, start centre) while the photo is dragged
        self.faces = []                            # portrait: [{"lines": [[(u, v), ...], ...], "how"}]
        self.show_faces = False
        self.shapes = []                           # sculpt shapes: [{"outline": [(u, v), ...], "sel", "tool"}]
        self.show_shapes = False
        self.shape_tool = None                     # the shape being drawn ("breast", "thigh", ...)
        self.shape_ratio = 0.5                     # its width / length
        self._shdrag = None                        # (index, start uv) while a shape is moved
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    # A friend's first test (2026-11-15): "the filters aren't working on the right-hand side" - they were, just
    # slowly, and the only sign was a small label in the far corner. Work now shows as a moving bar along the top of
    # the photo and a "Working... 1.4 s" pill, and the Controls chip on the right says the same.
    @property
    def busy(self) -> bool:
        return self._busy

    @busy.setter
    def busy(self, on: bool):
        on = bool(on)
        if on == self._busy:
            return
        self._busy = on
        if on:
            self.busy_since = time.perf_counter()
            self._spin.start()
        else:
            self.busy_took = time.perf_counter() - self.busy_since
            self._spin.stop()
        self.update()
        self.busy_changed.emit(on)

    def busy_for(self) -> float:
        """Seconds the current work has been running (0 when idle)."""
        return time.perf_counter() - self.busy_since if self._busy else 0.0

    def _paint_busy(self, p, r):
        """Indeterminate progress: a cyan segment sweeping along the top edge of the photo, and a pill with a
        turning arc and the seconds so far. Nothing for the first 0.15 s, so quick renders don't flicker."""
        t = self.busy_for()
        if t < self.BUSY_SHOW_S:
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar = QRectF(r.left(), max(2.0, r.top() - 7), r.width(), 4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 28))
        p.drawRoundedRect(bar, 2, 2)
        seg = bar.width() * 0.28
        x = bar.left() - seg + ((t * 0.75) % 1.0) * (bar.width() + seg)
        g = QLinearGradient(x, 0, x + seg, 0)
        g.setColorAt(0.0, qalpha(T.CYAN, 0.0))
        g.setColorAt(0.5, QColor(T.CYAN))
        g.setColorAt(1.0, qalpha(T.CYAN, 0.0))
        p.setBrush(g)
        p.setClipRect(bar)
        p.drawRoundedRect(QRectF(x, bar.top(), seg, bar.height()), 2, 2)
        p.setClipping(False)
        text = f"Working...  {t:.1f} s"
        p.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        fm = p.fontMetrics()
        h = fm.height() + 12
        w = fm.horizontalAdvance(text) + h + 18
        box = QRectF(self.width() / 2 - w / 2, r.top() + 10, w, h)
        p.setBrush(QColor(0, 0, 0, 190))
        p.setPen(QPen(qalpha(T.AMBER, 0.55), 1))
        p.drawRoundedRect(box, h / 2, h / 2)
        d = h - 12
        arc = QRectF(box.left() + 9, box.top() + 6, d, d)
        p.setPen(QPen(QColor(255, 255, 255, 40), 2.5))
        p.drawEllipse(arc)
        p.setPen(QPen(QColor(T.AMBER), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(arc, int(-t * 360 * 16) % (360 * 16), 100 * 16)
        p.setPen(QColor(T.AMBER))
        p.drawText(QRectF(arc.right() + 8, box.top(), w - d - 22, h),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
        p.restore()

    def _fit(self, img: QImage) -> QRectF:
        r = QRectF(self.rect()).adjusted(Z.MARGIN, Z.MARGIN, -Z.MARGIN, -Z.MARGIN)
        if img is None or img.width() == 0:
            return r
        k = min(r.width() / img.width(), r.height() / img.height())
        w, h = img.width() * k, img.height() * k
        return QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h)

    # -------------------------------------------------------------------------- zoom and position
    @property
    def zoomed(self) -> bool:
        return self.zs.z is not None and self.zs.ready

    def photo_rect(self) -> QRectF:
        """The whole photo on screen (logical pixels); zoomed, it is bigger than the view and mostly outside it."""
        if self.zoomed:
            return QRectF(*self.zs.photo_rect())
        return self._fit(self.after)

    def _sync_view(self):
        self.zs.set_view(self.width(), self.height(), self.devicePixelRatioF())

    def _view_moved(self, entered: bool = False):
        """Repaint at once; tell the page (for a sharp render) once the movement settles."""
        if entered:
            self.zoom_flash = time.perf_counter()
            self._hint_timer.start()
        if not self.tool and self._pan is None:
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.zoomed else Qt.CursorShape.ArrowCursor)
        self.update()
        self._settle.start()

    def _settled(self):
        self.view_changed.emit()

    def set_zoom(self, z, anchor=None):
        """Zoom to ``z`` (device pixels per photo pixel, 1.0 = 100%; None = fit), keeping the photo point under
        ``anchor`` (a view position, QPointF) in place; without one, the middle of the view."""
        if self.after is None or self.crop_mode:
            return
        self._sync_view()
        was = self.zoomed
        self.zs.zoom_to(z, None if anchor is None else (anchor.x(), anchor.y()))
        self._view_moved(entered=not was and self.zoomed)

    def zoom_step(self, direction: int, anchor=None):
        if self.after is None or self.crop_mode:
            return
        self._sync_view()
        was = self.zoomed
        self.zs.step(direction, None if anchor is None else (anchor.x(), anchor.y()))
        self._view_moved(entered=not was and self.zoomed)

    def toggle_zoom(self, anchor=None):
        """A click or Z: the whole photo <-> 100% at that point (200% for photos smaller than the view)."""
        if self.zoomed:
            self.set_zoom(None)
        else:
            self._sync_view()
            fz = self.zs.fit_z()
            self.set_zoom(1.0 if fz < 0.95 else min(self.zs.max_z(), fz * 2.0), anchor)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._sync_view()
        if self.zoomed:
            self._settle.start()

    def _draw_part(self, p, target: QRectF, img: QImage):
        """Draw ``img`` stretched over ``target`` (which may be far bigger than the view): only the visible part."""
        vis = target.intersected(QRectF(self.rect()))
        if vis.isEmpty() or img is None or img.width() == 0 or target.width() <= 0 or target.height() <= 0:
            return
        sx, sy = img.width() / target.width(), img.height() / target.height()
        src = QRectF((vis.left() - target.left()) * sx, (vis.top() - target.top()) * sy, vis.width() * sx,
                     vis.height() * sy)
        p.drawImage(vis, img, src)

    NAV_SIZE = 168                                  # the navigator's long edge, logical pixels

    def _nav_box(self) -> QRectF | None:
        if self.after is None or not self.zoomed:
            return None
        w0, h0 = self.after.width(), self.after.height()
        k = self.NAV_SIZE / float(max(w0, h0, 1))
        nw, nh = w0 * k, h0 * k
        return QRectF(self.width() - nw - 18, self.height() - nh - 18, nw, nh)

    def _paint_navigator(self, p):
        """Where you are: the whole photo small in the corner, the part on screen outlined (click or drag in it to
        go somewhere else)."""
        box = self._nav_box()
        self._nav = box
        if box is None:
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.setBrush(QColor(0, 0, 0, 190))
        p.drawRoundedRect(box.adjusted(-5, -5, 5, 5), 7, 7)
        p.drawImage(box, self.after)
        x, y, w, h = self.zs.visible_window()
        ow, oh = max(self.zs.ow, 1), max(self.zs.oh, 1)
        vr = QRectF(box.left() + x / ow * box.width(), box.top() + y / oh * box.height(), w / ow * box.width(),
                    h / oh * box.height())
        shade = QPainterPath()
        shade.addRect(box)
        hole = QPainterPath()
        hole.addRect(vr)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 120))
        p.drawPath(shade.subtracted(hole))
        p.setPen(QPen(QColor(T.CYAN), 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(vr)
        p.restore()

    def _nav_jump(self, pos):
        box = self._nav or self._nav_box()
        if box is None:
            return
        u = min(max((pos.x() - box.left()) / box.width(), 0.0), 1.0)
        v = min(max((pos.y() - box.top()) / box.height(), 0.0), 1.0)
        self.zs.centre_on(u, v)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        g = QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0, QColor("#0a0f1a"))
        g.setColorAt(1, QColor("#05070c"))
        p.fillRect(self.rect(), g)
        if self.after is None:
            p.setPen(QColor(T.DIM))
            p.setFont(QFont("Segoe UI", 12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            if self.busy:                                    # opening a photo: the slowest render of all
                self._paint_busy(p, self._fit(None))
            return
        r = self.photo_rect()
        if self.zoomed:
            # the whole-photo preview, enlarged: the photo follows the mouse at once; the sharp tile goes on top
            self._draw_part(p, r, self.after)
            if self.tile is not None and self.tile_uv is not None:
                u0, v0, u1, v1 = self.tile_uv
                tr = QRectF(r.left() + u0 * r.width(), r.top() + v0 * r.height(), (u1 - u0) * r.width(),
                            (v1 - v0) * r.height())
                if self.zs.z >= 2.0:                         # 200% and up: photo pixels as crisp squares
                    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
                self._draw_part(p, tr, self.tile)
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        else:
            p.drawImage(r, self.after)
        if self.split is not None and self.before is not None and not self.zoomed:
            x = r.left() + r.width() * self.split
            p.save()
            p.setClipRect(QRectF(r.left(), r.top(), x - r.left(), r.height()))
            p.drawImage(r, self.before)
            p.restore()
            p.setPen(QPen(QColor(T.CYAN), 2))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setBrush(QColor(T.CYAN))
            p.drawEllipse(QPointF(x, r.center().y()), 7, 7)
            self._tag(p, QPointF(r.left() + 12, r.top() + 12), "Before (as shot)")
            self._tag(p, QPointF(r.right() - 12, r.top() + 12), "After", right=True)
        if self.overlay is not None and (self.tool is not None or self.show_overlay):
            self._draw_part(p, r, self.overlay)
        if self.spots and (self.tool == "spot" or self.show_spots):
            self._paint_spots(p, r)
        if self.faces and (self.show_faces or self.tool == "eyes"):
            self._paint_faces(p, r)
        if self.shapes and (self.show_shapes or self.tool in ("shape", "sculpt")):
            self._paint_shapes(p, r)
        if self.tool:
            self._paint_tool(p, r)
        if self.zoomed:
            self._paint_navigator(p)
            hint = ("   drag to move  ·  wheel to zoom  ·  click, Z or Esc for the whole photo"
                    if time.perf_counter() - self.zoom_flash < 3.0 else "")
            self._tag(p, QPointF(self.width() - 14, 12), Z.label(self.zs.z) + hint, right=True)
        else:
            self._nav = None
        if self.crop_mode:
            self._paint_crop(p, r)
        elif self.comp_outside and not self.zoomed:
            self._paint_comp(p, r)
        if self.busy:
            self._paint_busy(p, r)

    show_overlay = False

    # -------------------------------------------------------------------------- mask tools
    def _uv(self, pos):
        """Widget position -> position on the whole photo (0..1), at any zoom."""
        r = self.photo_rect()
        return ((pos.x() - r.left()) / max(r.width(), 1e-6), (pos.y() - r.top()) / max(r.height(), 1e-6))

    def _px(self, uv, r):
        return QPointF(r.left() + uv[0] * r.width(), r.top() + uv[1] * r.height())

    def _long_px(self, r):
        """The whole photo's long edge in widget pixels (bigger than the widget when zoomed)."""
        return max(r.width(), r.height())

    def _paint_spots(self, p, r):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        L = self._long_px(r)
        for k, sp in enumerate(self.spots):
            rad = max(3.0, sp["r"] * L)
            col = QColor(T.AMBER) if sp.get("auto") else QColor(T.CYAN)
            if sp.get("kind") in ("redeye", "pupil"):
                col = QColor(239, 68, 68)
            if sp.get("sel"):
                col = QColor(T.TEXT)
            dst = [self._px(uv, r) for uv in sp["pts"]]
            src = [self._px(uv, r) for uv in sp.get("src") or []]
            for pts, dash in ((src, True), (dst, False)):
                if not pts:
                    continue
                pen = QPen(col, 2.0 if sp.get("sel") else 1.5, Qt.PenStyle.DashLine if dash else Qt.PenStyle.SolidLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                if len(pts) == 1:
                    p.drawEllipse(pts[0], rad, rad)
                else:
                    path = QPainterPath(pts[0])
                    for q in pts[1:]:
                        path.lineTo(q)
                    stroker = QPainterPathStroker()
                    stroker.setWidth(2 * rad)
                    stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
                    stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                    p.drawPath(stroker.createStroke(path))
            if src and (sp.get("sel") or self.tool == "spot"):
                p.setPen(QPen(col, 1, Qt.PenStyle.DotLine))
                p.drawLine(dst[0], src[0])

    def _spot_at(self, uv):
        """(index, "dst" | "src") of the spot under a photo position, or None. Sources are checked first only
        for the selected spot, so a source never hides another spot."""
        if not self.spots:
            return None
        r = self.photo_rect()
        L = self._long_px(r)
        here = self._px(uv, r)
        best = None
        for k, sp in enumerate(self.spots):
            rad = max(4.0, sp["r"] * L)
            for which in (("src", "dst") if sp.get("sel") else ("dst",)):
                pts = sp["pts"] if which == "dst" else (sp.get("src") or [])
                for q in pts:
                    d = math.hypot(self._px(q, r).x() - here.x(), self._px(q, r).y() - here.y())
                    if d <= rad and (best is None or d < best[0]):
                        best = (d, k, which)
        return None if best is None else (best[1], best[2])

    def _paint_faces(self, p, r):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for f in self.faces:
            col = QColor(T.CYAN) if f.get("how") == "eyes" else QColor(T.AMBER)
            if not f.get("enabled", True):
                col = QColor(T.DIM)
            p.setPen(QPen(col, 1.4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for line in f["lines"]:
                pts = [self._px(uv, r) for uv in line]
                if len(pts) > 1:
                    path = QPainterPath(pts[0])
                    for q in pts[1:]:
                        path.lineTo(q)
                    p.drawPath(path)

    def _shape_path(self, outline, r):
        pts = [self._px(uv, r) for uv in outline]
        path = QPainterPath(pts[0])
        for q in pts[1:]:
            path.lineTo(q)
        path.closeSubpath()
        return path

    def _paint_shapes(self, p, r):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for sh in self.shapes:
            col = QColor(T.TEXT) if sh.get("sel") else QColor(T.VIOLET)
            p.setPen(QPen(col, 2.0 if sh.get("sel") else 1.4, Qt.PenStyle.SolidLine if sh.get("sel")
                          else Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(self._shape_path(sh["outline"], r))

    def _shape_at(self, uv):
        r = self.photo_rect()
        here = self._px(uv, r)
        for k in range(len(self.shapes) - 1, -1, -1):
            if self._shape_path(self.shapes[k]["outline"], r).contains(here):
                return k
        return None

    @staticmethod
    def oval(a, b, ratio, n=48):
        """An oval from one end (a) to the other (b), width = ratio x length, as (u, v) points (display space)."""
        cx, cy = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        dx, dy = 0.5 * (b[0] - a[0]), 0.5 * (b[1] - a[1])
        nx, ny = -dy * ratio, dx * ratio
        return [(cx + dx * math.cos(t) + nx * math.sin(t), cy + dy * math.cos(t) + ny * math.sin(t))
                for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]

    def _paint_tool(self, p, r):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.tool in ("eyes", "sculpt", "shape"):
            tips = {"eyes": "Click the centre of each eye (either order).  Esc: cancel",
                    "sculpt": "Drag to push (Shrink / Grow: click or drag over the area).  Esc: done",
                    "shape": "Drag from one end of the area to the other; drag a shape to move it.  Esc: done"}
            self._tag(p, QPointF(r.left() + 12, r.bottom() - 34), tips[self.tool], colour=T.CYAN)
            if self.tool == "eyes":
                p.setPen(QPen(QColor(T.CYAN), 2))
                for uv in self._pts:
                    c = self._px(uv, r)
                    p.drawLine(QPointF(c.x() - 8, c.y()), QPointF(c.x() + 8, c.y()))
                    p.drawLine(QPointF(c.x(), c.y() - 8), QPointF(c.x(), c.y() + 8))
            elif self.tool == "sculpt":
                rad = self.brush_frac * self._long_px(r) * self.brush_scale
                if self._pts:
                    path = QPainterPath(self._px(self._pts[0], r))
                    for uv in self._pts[1:]:
                        path.lineTo(self._px(uv, r))
                    p.setPen(QPen(QColor(T.CYAN), 1.5, Qt.PenStyle.DashLine))
                    p.drawPath(path)
                if self._mouse is not None:
                    p.setPen(QPen(QColor(T.TEXT), 1.5))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    c = self._px(self._mouse, r)
                    p.drawEllipse(c, rad, rad)
                    p.setPen(QPen(QColor(T.TEXT), 1, Qt.PenStyle.DotLine))
                    p.drawEllipse(c, rad * 0.5, rad * 0.5)
            elif self.tool == "shape" and len(self._pts) == 2 and self._shdrag is None:
                p.setPen(QPen(QColor(T.CYAN), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(self._shape_path(self.oval(self._pts[0], self._pts[1], self.shape_ratio), r))
            return
        tips = {"brush": "Paint the area (Alt: erase).  Esc: done", "linear": "Drag from full effect to none",
                "radial": "Drag from the centre outward", "object": "Click the object (Alt-click: not this)",
                "colour": "Click the colour to select",
                "spot": "Click a spot, or brush along a hair.  Drag a circle to move it; Alt-click removes.  Esc: done"}
        self._tag(p, QPointF(r.left() + 12, r.bottom() - 34), tips.get(self.tool, ""), colour=T.CYAN)
        rad = self.brush_frac * self._long_px(r) * self.brush_scale
        if self.tool == "spot" and self._pts and self._sdrag is None:
            path = QPainterPath(self._px(self._pts[0], r))
            for uv in self._pts[1:]:
                path.lineTo(self._px(uv, r))
            p.setPen(QPen(QColor(34, 211, 238, 110), 2 * rad, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                          Qt.PenJoinStyle.RoundJoin))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        if self.tool == "spot" and self._mouse is not None and self._sdrag is None:
            p.setPen(QPen(QColor(T.TEXT), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(self._px(self._mouse, r), rad, rad)
        if self.tool == "brush" and self._pts:
            path = QPainterPath(self._px(self._pts[0], r))
            for uv in self._pts[1:]:
                path.lineTo(self._px(uv, r))
            pen = QPen(QColor(239, 68, 68, 110), 2 * rad, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                       Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        if self.tool == "brush" and self._mouse is not None:
            p.setPen(QPen(QColor(T.TEXT), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(self._px(self._mouse, r), rad, rad)
        if self.tool in ("linear", "radial") and len(self._pts) == 2:
            a, b = self._px(self._pts[0], r), self._px(self._pts[1], r)
            p.setPen(QPen(QColor(T.CYAN), 2))
            if self.tool == "linear":
                p.drawLine(a, b)
                d = QPointF(-(b.y() - a.y()), b.x() - a.x())
                n = math.hypot(d.x(), d.y()) or 1.0
                d = QPointF(d.x() / n * 2000, d.y() / n * 2000)
                for c in (a, b):
                    p.drawLine(c - d, c + d)
            else:
                rr = math.hypot(b.x() - a.x(), b.y() - a.y())
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(a, rr, rr)
                p.setPen(QPen(QColor(T.CYAN), 1, Qt.PenStyle.DashLine))
                p.drawEllipse(a, rr * 0.5, rr * 0.5)

    def _tool_press(self, e):
        uv = self._uv(e.position())
        if self.tool == "eyes":
            self._pts.append(uv)
            if len(self._pts) >= 2:
                pts, self._pts = list(self._pts[:2]), []
                self.eyes_done.emit(pts)
            self.update()
            return
        if self.tool == "sculpt":
            self._pts = [uv]
            self.update()
            return
        if self.tool == "shape":
            k = self._shape_at(uv)
            if k is not None and self.shape_tool is None:
                self._shdrag = (k, uv)
                self.shape_picked.emit(k)
            elif k is not None and self.shape_tool is not None and not (e.modifiers() &
                                                                       Qt.KeyboardModifier.ShiftModifier):
                self._shdrag = (k, uv)
                self.shape_picked.emit(k)
            else:
                self._shdrag = None
                self._pts = [uv, uv]
            self.update()
            return
        if self.tool == "spot":
            hit = self._spot_at(uv)
            alt = bool(e.modifiers() & Qt.KeyboardModifier.AltModifier)
            if hit is not None and alt:
                self.spot_deleted.emit(hit[0])
            elif hit is not None:
                self._sdrag = (hit[0], hit[1], uv)
                self.spot_picked.emit(hit[0])
            else:
                self._sdrag = None
                self._pts = [uv]
            self.update()
            return
        if self.tool == "brush":
            self._pts = [uv]
            self._erase = bool(e.modifiers() & Qt.KeyboardModifier.AltModifier)
        elif self.tool in ("linear", "radial"):
            self._pts = [uv, uv]
        elif self.tool in ("object", "colour"):
            self.tool_click.emit(uv[0], uv[1], bool(e.modifiers() & Qt.KeyboardModifier.AltModifier))
        self.update()

    def _tool_move(self, e):
        uv = self._uv(e.position())
        self._mouse = uv
        if self.tool in ("sculpt", "shape", "eyes"):
            if e.buttons() & Qt.MouseButton.LeftButton:
                if self.tool == "sculpt" and self._pts:
                    last = self._pts[-1]
                    if math.hypot(uv[0] - last[0], uv[1] - last[1]) > 0.002:
                        self._pts.append(uv)
                elif self.tool == "shape" and self._shdrag is not None:
                    k, start = self._shdrag
                    du, dv = uv[0] - start[0], uv[1] - start[1]
                    sh = self.shapes[k]
                    base = sh.get("_base") or list(sh["outline"])
                    sh["_base"] = base
                    sh["outline"] = [(q[0] + du, q[1] + dv) for q in base]
                elif self.tool == "shape" and len(self._pts) == 2:
                    self._pts[1] = uv
            self.update()
            return
        if self.tool == "spot" and e.buttons() & Qt.MouseButton.LeftButton:
            if self._sdrag is not None:
                k, which, start = self._sdrag
                du, dv = uv[0] - start[0], uv[1] - start[1]
                sp = self.spots[k]
                key = "pts" if which == "dst" else "src"
                sp["_drag"] = (key, du, dv)
                base = sp.get("_base_" + key) or [tuple(q) for q in sp[key]]
                sp["_base_" + key] = base
                sp[key] = [(q[0] + du, q[1] + dv) for q in base]
            elif self._pts:
                last = self._pts[-1]
                if math.hypot(uv[0] - last[0], uv[1] - last[1]) > 0.001:
                    self._pts.append(uv)
            self.update()
            return
        if self.tool == "brush" and self._pts and e.buttons() & Qt.MouseButton.LeftButton:
            last = self._pts[-1]
            if math.hypot(uv[0] - last[0], uv[1] - last[1]) > 0.002:
                self._pts.append(uv)
        elif self.tool in ("linear", "radial") and self._pts and e.buttons() & Qt.MouseButton.LeftButton:
            self._pts[1] = uv
        self.update()

    def _tool_release(self, e):
        if self.tool == "sculpt":
            if self._pts:
                self.sculpt_done.emit(list(self._pts))
            self._pts = []
            self.update()
            return
        if self.tool == "shape":
            uv = self._uv(e.position())
            if self._shdrag is not None:
                k, start = self._shdrag
                if 0 <= k < len(self.shapes):
                    self.shapes[k].pop("_base", None)
                du, dv = uv[0] - start[0], uv[1] - start[1]
                if math.hypot(du, dv) > 0.0005:
                    self.shape_moved.emit(k, du, dv)
            elif len(self._pts) == 2:
                (u0, v0), (u1, v1) = self._pts
                if math.hypot(u1 - u0, v1 - v0) > 0.01:
                    self.shape_drawn.emit(u0, v0, u1, v1)
            self._shdrag = None
            self._pts = []
            self.update()
            return
        if self.tool == "eyes":
            return
        if self.tool == "spot":
            if self._sdrag is not None:
                k, which, start = self._sdrag
                uv = self._uv(e.position())
                du, dv = uv[0] - start[0], uv[1] - start[1]
                if 0 <= k < len(self.spots):
                    for key in ("pts", "src"):
                        self.spots[k].pop("_base_" + key, None)
                    self.spots[k].pop("_drag", None)
                if math.hypot(du, dv) > 0.0005:
                    self.spot_moved.emit(k, which, du, dv)
            elif self._pts:
                pts = list(self._pts)
                span = max(math.hypot(q[0] - pts[0][0], q[1] - pts[0][1]) for q in pts)
                self.spot_new.emit([pts[0]] if span < 0.004 else pts)
            self._sdrag = None
            self._pts = []
            self.update()
            return
        if self.tool == "brush" and self._pts:
            self.stroke_done.emit((list(self._pts), getattr(self, "_erase", False)))
        elif self.tool in ("linear", "radial") and len(self._pts) == 2:
            (u0, v0), (u1, v1) = self._pts
            if math.hypot(u1 - u0, v1 - v0) > 0.005:
                self.drag_done.emit(u0, v0, u1, v1)
        self._pts = []
        self.update()

    # -------------------------------------------------------------------------- crop frame
    def _crop_rect(self, r: QRectF) -> QRectF:
        x0, y0, x1, y1 = self.crop
        return QRectF(r.left() + x0 * r.width(), r.top() + y0 * r.height(), (x1 - x0) * r.width(),
                      (y1 - y0) * r.height())

    def _paint_crop(self, p, r):
        c = self._crop_rect(r)
        shade = QColor(0, 0, 0, 150)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(shade)
        for q in (QRectF(r.left(), r.top(), r.width(), c.top() - r.top()),
                  QRectF(r.left(), c.bottom(), r.width(), r.bottom() - c.bottom()),
                  QRectF(r.left(), c.top(), c.left() - r.left(), c.height()),
                  QRectF(c.right(), c.top(), r.right() - c.right(), c.height())):
            if q.width() > 0 and q.height() > 0:
                p.drawRect(q)
        p.setBrush(Qt.BrushStyle.NoBrush)
        self._paint_comp(p, c)
        p.setPen(QPen(QColor(T.CYAN), 2))
        p.drawRect(c)
        p.setBrush(QColor(T.CYAN))
        for pt in (c.topLeft(), c.topRight(), c.bottomLeft(), c.bottomRight()):
            p.drawRect(QRectF(pt.x() - 5, pt.y() - 5, 10, 10))
        self._tag(p, QPointF(r.left() + 12, r.top() + 12), "Crop: drag the frame or its corners; press Enter or "
                  "the Crop button when done")

    def _paint_comp(self, p, c):
        """The composition overlay in the rectangle c: thin white lines with a dark edge (seen on any photo) and the
        method's strong points as small rings; its name for a moment after it changes."""
        from ..engine.local import composition as cp
        if self.comp not in cp.NAMES or c.width() < 8 or c.height() < 8:
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(Qt.BrushStyle.NoBrush)
        aspect = c.width() / c.height()
        paths = []
        for pl in cp.lines(self.comp, aspect, self.comp_orient):
            path = QPainterPath(QPointF(c.left() + pl[0][0] * c.width(), c.top() + pl[0][1] * c.height()))
            for x, y in pl[1:]:
                path.lineTo(QPointF(c.left() + x * c.width(), c.top() + y * c.height()))
            paths.append(path)
        for pen in (QPen(QColor(0, 0, 0, 90), 3), QPen(QColor(255, 255, 255, 150), 1)):
            p.setPen(pen)
            for path in paths:
                p.drawPath(path)
        p.setPen(QPen(QColor(T.CYAN), 1.5))
        for x, y in cp.points(self.comp, aspect, self.comp_orient):
            p.drawEllipse(QPointF(c.left() + x * c.width(), c.top() + y * c.height()), 5, 5)
        p.restore()
        if time.perf_counter() - self.comp_flash < 2.5:
            turn = "  ·  Shift+O turns it" if cp.ORIENTATIONS.get(self.comp, 1) > 1 else ""
            self._tag(p, QPointF(c.center().x() - 150, c.bottom() - 34), f"{cp.NAMES[self.comp]}  ·  O: next{turn}",
                      colour=T.CYAN)

    def _crop_press(self, pos):
        r = self.photo_rect()
        c = self._crop_rect(r)
        corners = {"tl": c.topLeft(), "tr": c.topRight(), "bl": c.bottomLeft(), "br": c.bottomRight()}
        for k, pt in corners.items():
            if abs(pos.x() - pt.x()) < 14 and abs(pos.y() - pt.y()) < 14:
                self._cdrag = (k, pos, self.crop)
                return
        if c.contains(pos):
            self._cdrag = ("move", pos, self.crop)

    def _crop_move(self, pos):
        kind, start, (x0, y0, x1, y1) = self._cdrag
        r = self.photo_rect()
        dx, dy = (pos.x() - start.x()) / r.width(), (pos.y() - start.y()) / r.height()
        if kind == "move":
            w, h = x1 - x0, y1 - y0
            nx0, ny0 = min(max(x0 + dx, 0.0), 1.0 - w), min(max(y0 + dy, 0.0), 1.0 - h)
            self.crop = (nx0, ny0, nx0 + w, ny0 + h)
        else:
            ax = x1 if "l" in kind else x0                  # the corner that stays put
            ay = y1 if "t" in kind else y0
            mx = min(max((x0 if "l" in kind else x1) + dx, 0.0), 1.0)
            my = min(max((y0 if "t" in kind else y1) + dy, 0.0), 1.0)
            w, h = max(abs(mx - ax), 0.04), max(abs(my - ay), 0.04)
            if self.crop_ratio:                              # keep the aspect: width/height in pixels
                k = self.crop_ratio * r.height() / r.width()
                if w / h > k:
                    w = h * k
                else:
                    h = w / k
                lim_w = ax if "l" in kind else 1.0 - ax
                lim_h = ay if "t" in kind else 1.0 - ay
                s = min(1.0, lim_w / w if w > 0 else 1.0, lim_h / h if h > 0 else 1.0)
                w, h = w * s, h * s
            nx0 = ax - w if "l" in kind else ax
            ny0 = ay - h if "t" in kind else ay
            self.crop = (max(nx0, 0.0), max(ny0, 0.0), min(nx0 + w, 1.0), min(ny0 + h, 1.0))
        self.update()

    def _tag(self, p, pt, text, right=False, colour=T.TEXT):
        p.setFont(QFont("Segoe UI", 9))
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(text) + 16, fm.height() + 8
        x = pt.x() - w if right else pt.x()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRoundedRect(QRectF(x, pt.y(), w, h), 6, 6)
        p.setPen(QColor(colour))
        p.drawText(QRectF(x, pt.y(), w, h), Qt.AlignmentFlag.AlignCenter, text)

    def mousePressEvent(self, e):
        if self.after is None:
            return
        pos, btn = e.position(), e.button()
        left = btn == Qt.MouseButton.LeftButton
        if self.zoomed and left and self._nav is not None and self._nav.contains(pos):
            self._nav_drag = True                          # the navigator: go where you click, follow a drag
            self._nav_jump(pos)
            return
        pan_button = btn in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton)
        free = not self.tool and not self.picking and not self.crop_mode
        if self.zoomed and (pan_button or (left and (self._space or free))):
            # move the photo: it follows the mouse at once (no waiting for a render)
            self._pan = (pos, (self.zs.cu, self.zs.cv))
            self._press = pos if (left and free and not self._space) else None   # no drag = a click
            self._cursor_before = self.cursor()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if pan_button:
            return
        if self.tool:
            self._tool_press(e)
            return
        if self.crop_mode:
            self._crop_press(pos)
            return
        r = self.photo_rect()
        if self.split is not None and abs(pos.x() - (r.left() + r.width() * self.split)) < 12:
            self._drag = True
            return
        if r.contains(pos):
            if self.picking:
                self.clicked.emit(*self._uv(pos))          # an eyedropper is waiting: the click picks
            else:
                self._press = pos                          # released without a drag: zoom to 100% here

    picking = False                                # an eyedropper is waiting: clicks pick, they do not zoom or pan

    def mouseDoubleClickEvent(self, e):
        # the first click of a double-click already zoomed in; a second would zoom straight back out
        if self.tool or self.crop_mode or self.picking:
            super().mouseDoubleClickEvent(e)

    def wheelEvent(self, e):
        if self.after is None or self.crop_mode:
            return
        d = e.angleDelta().y() or e.angleDelta().x()
        if not d:
            return
        self._sync_view()
        was = self.zoomed
        self.zs.wheel(d, (e.position().x(), e.position().y()))   # around the pointer, any amount (touchpads too)
        self._view_moved(entered=not was and self.zoomed)

    def event(self, e):
        # touchpad pinch (macOS, Linux Wayland): zoom around the fingers
        if e.type() == e.Type.NativeGesture and self.after is not None and not self.crop_mode:
            try:
                if e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                    self._sync_view()
                    was = self.zoomed
                    z = self.zs.current() * (1.0 + float(e.value()))
                    self.zs.zoom_to(z, (e.position().x(), e.position().y()))
                    self._view_moved(entered=not was and self.zoomed)
                    return True
            except (AttributeError, TypeError):
                pass
        return super().event(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = True                             # Space + drag moves the photo, even with a tool
            if self.zoomed and self._pan is None:
                self._cursor_before = self.cursor()
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = False
            if self._pan is None and getattr(self, "_cursor_before", None) is not None:
                self.setCursor(self._cursor_before)
                self._cursor_before = None
            return
        super().keyReleaseEvent(e)

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self.after is not None and not self.tool:
            self._mouse = self._uv(pos)                    # where Z zooms to
        if self._nav_drag:
            self._nav_jump(pos)
            return
        if self._pan is not None:
            start, (cu0, cv0) = self._pan
            self.zs.cu, self.zs.cv = cu0, cv0
            self.zs.pan_by(pos.x() - start.x(), pos.y() - start.y())
            if self._press is not None and abs(pos.x() - self._press.x()) + abs(pos.y() - self._press.y()) > 4:
                self._press = None                         # a drag, not a click
            self.update()
            return
        if self._press is not None and abs(pos.x() - self._press.x()) + abs(pos.y() - self._press.y()) > 4:
            self._press = None
        if self.tool and self.after is not None:
            self._tool_move(e)
            return
        if self.crop_mode and self._cdrag is not None and self.after is not None:
            self._crop_move(pos)
            return
        if self._drag and self.after is not None:
            r = self.photo_rect()
            self.split = min(max((pos.x() - r.left()) / r.width(), 0.0), 1.0)
            self.update()

    def mouseReleaseEvent(self, e):
        pos = e.position() if e is not None else QPointF()
        left = e is not None and e.button() == Qt.MouseButton.LeftButton
        if self._nav_drag:
            self._nav_drag = False
            self._view_moved()
            return
        if self._pan is not None:
            click = self._press is not None and left
            self._pan, self._press = None, None
            before = getattr(self, "_cursor_before", None)
            if before is not None and (self.tool or self._space):
                self.setCursor(before)
            self._cursor_before = None
            if click:
                self.toggle_zoom(pos)                      # zoomed and clicked without a drag: the whole photo
            else:
                self._view_moved()
            return
        if self._press is not None and left:
            self._press = None
            if not self.tool and not self.crop_mode and not self.picking:
                self.toggle_zoom(pos)
                return
        self._press = None
        if self.tool and self.after is not None:
            self._tool_release(e)
            return
        self._drag = False
        if self._cdrag is not None:
            self._cdrag = None
            self.crop_changed.emit(tuple(round(v, 5) for v in self.crop))


# ============================================================================================ Develop
class DevelopPage(QWidget):
    status = Signal(str)
    export_requested = Signal(list)

    def __init__(self, settings: Settings):
        super().__init__()
        from ..engine import Edit
        self.settings = settings
        self.image = None
        self.path = None
        self.edit = Edit()
        self.steps: list = [("Opened", self.edit)]   # labelled history: (label, Edit); hist[hpos] is what is shown
        self.hpos = 0
        self._last_push = (0.0, None)
        self.version = 0                            # 0 = master, 1.. = virtual copies (all in the one sidecar)
        self.pick = None                            # None | "wb" | "range": what the next click on the photo picks
        self.last_a8 = None
        self._hsl: dict = {}
        self._ranges: list = []
        self._rsel = -1
        self.token = 0
        self.judge_bw = False                               # view only: "Judge in black and white"
        self._render_live = False              # a render is running (only one at a time)
        self._render_again = False             # the controls changed while it ran: render once more when it ends
        self._render_img = None                # the photo that render belongs to
        self._dirty_base = False               # the whole-photo preview needs rendering for the current settings
        self._dirty_tile = False               # the zoomed view's sharp tile does
        self.before_img = None
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        centre = QVBoxLayout()
        centre.setContentsMargins(0, 0, 0, 0)
        centre.setSpacing(0)
        bar = FlexBar()
        bl = bar.lay
        bl.setContentsMargins(12, 8, 12, 8)
        bl.setSpacing(5)                            # fits a 1366-pixel laptop screen with the panel open
        self.title = ElideLabel("Develop", "h2", min_width=40)
        self.ba = QPushButton("Before / After")
        self.ba.setCheckable(True)
        self.ba.setToolTip("Wipe between the photo as shot and your edit (B)")
        self.ba.toggled.connect(self.toggle_before)
        self.undo_b = QPushButton("Undo")
        self.redo_b = QPushButton("Redo")
        self.undo_b.clicked.connect(self.do_undo)
        self.redo_b.clicked.connect(self.do_redo)
        self.mig = QPushButton("✦ Make it good")
        self.mig.setObjectName("primary")
        self.mig.setToolTip("Automatic clean-up for this ISO, lens-safe defaults. Never changes brightness.")
        self.mig.clicked.connect(self.make_it_good)
        exp = QPushButton("Export...")
        exp.setToolTip("Export the version you are looking at (Ctrl+E)")
        exp.clicked.connect(lambda: self.path and self.export_requested.emit([(self.path, self.version)]))
        self.ver = QComboBox()
        self.ver.setToolTip("Virtual copies: several different edits of the same raw, all kept in its one sidecar")
        self.ver.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.ver.addItem("Master", 0)
        self.ver.activated.connect(self._version_picked)
        self.copy_b = QPushButton("Copy")
        self.copy_b.setToolTip("Copy this photo's settings - you choose which groups (Ctrl+Shift+C)")
        self.copy_b.clicked.connect(self.copy_settings)
        self.paste_b = QPushButton("Paste")
        self.paste_b.setToolTip("Paste the copied settings onto this photo (Ctrl+Shift+V). To paste onto many "
                                "photos at once, select them in the Library and press Paste settings.")
        self.paste_b.clicked.connect(self.paste_settings)
        self.zoom_b = QToolButton()
        self.zoom_b.setText("Fit")
        self.zoom_b.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.zoom_b.setToolTip("Zoom. Click: 100% / the whole photo (or click the photo, or Z). The mouse wheel "
                               "zooms around the pointer; drag the photo to move it (or the middle or right "
                               "button, or Space + drag with a tool). Ctrl+= in, Ctrl+- out, Ctrl+0 fit, Ctrl+1 "
                               "100%. The arrow lists the steps.")
        self.zoom_b.clicked.connect(lambda: self.viewer.toggle_zoom(self._zoom_anchor()))
        zm = QMenu(self.zoom_b)
        for text, z in (("Fit: the whole photo   Ctrl+0", None), ("25%", 0.25), ("33%", 1 / 3), ("50%", 0.5),
                        ("67%", 2 / 3), ("100%   Ctrl+1", 1.0), ("200%", 2.0), ("400%", 4.0)):
            zm.addAction(text, lambda z=z: self.viewer.set_zoom(z, self._zoom_anchor()))
        self.zoom_b.setMenu(zm)
        def fill_versions(menu):
            sub = menu.addMenu("Virtual copies")
            for i in range(self.ver.count()):
                a = sub.addAction(self.ver.itemText(i))
                a.setCheckable(isinstance(self.ver.itemData(i), int))
                a.setChecked(i == self.ver.currentIndex())
                a.triggered.connect(lambda _=False, i=i: (self.ver.setCurrentIndex(i), self._version_picked(i)))

        def fill_button(btn, text):
            def fill(menu):
                a = menu.addAction(text)
                a.setEnabled(btn.isEnabled())
                if btn.isCheckable():
                    a.setCheckable(True)
                    a.setChecked(btn.isChecked())
                    a.triggered.connect(lambda on: btn.setChecked(on))
                else:
                    a.triggered.connect(btn.click)
            return fill
        bar.setup(self.title, [(self.ver, fill_versions),
                               (self.copy_b, fill_button(self.copy_b, "Copy settings...   Ctrl+Shift+C")),
                               (self.paste_b, fill_button(self.paste_b, "Paste settings   Ctrl+Shift+V")),
                               (self.ba, fill_button(self.ba, "Before / After   B"))],
                  [self.undo_b, self.redo_b, self.zoom_b, exp, self.mig])
        self.bar = bar
        centre.addWidget(bar)
        self.viewer = Viewer()
        self.viewer.clicked.connect(self._photo_clicked)
        self.viewer.view_changed.connect(self._view_changed)
        self.viewer.crop_changed.connect(self._crop_drawn)
        self.viewer.stroke_done.connect(self._stroke_done)
        self.viewer.drag_done.connect(self._drag_done)
        self.viewer.tool_click.connect(self._tool_click)
        self.viewer.spot_new.connect(self._spot_new)
        self.viewer.spot_moved.connect(self._spot_moved)
        self.viewer.spot_picked.connect(self._spot_picked)
        self.viewer.spot_deleted.connect(self._spot_delete)
        self.viewer.eyes_done.connect(self._eyes_clicked)
        self.viewer.sculpt_done.connect(self._sculpt_stroke)
        self.viewer.shape_drawn.connect(self._shape_drawn)
        self.viewer.shape_picked.connect(self._shape_picked)
        self.viewer.shape_moved.connect(self._shape_moved)
        self._portrait: dict = {}                   # the edit's "portrait": faces, sliders, sculpting
        self._shsel = -1                            # selected sculpt shape (index into the sculpt list)
        self._spots: list = []                      # retouch spots (the edit's "retouch" list)
        self._ssel = -1
        self.rt_kind = "heal"
        self._masks: list = []
        self._msel = -1
        self._mask_pending = None                   # (kind, op) waiting for the next drawing on the photo
        self.before_key = None
        centre.addWidget(self.viewer, 1)
        self.strip = QListWidget()
        self.strip.setViewMode(QListView.ViewMode.IconMode)
        self.strip.setFlow(QListView.Flow.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setIconSize(QSize(110, 74))
        self.strip.setFixedHeight(104)
        self.strip.setMovement(QListView.Movement.Static)
        self.strip.itemClicked.connect(lambda it: self.open(it.data(Qt.ItemDataRole.UserRole)))
        centre.addWidget(self.strip)
        cw = QWidget()
        cw.setLayout(centre)
        root.addWidget(cw, 1)
        root.addWidget(self._build_panel())
        self.timer = QTimer(self, singleShot=True, interval=140, timeout=self.render_now)
        self.save_timer = QTimer(self, singleShot=True, interval=900, timeout=self.save_sidecar)
        self.ds_ratio_timer = QTimer(self, singleShot=True, interval=1200, timeout=self._ds_measure_balance)
        QShortcut(QKeySequence("B"), self, activated=self.ba.toggle)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.do_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.do_redo)
        QShortcut(QKeySequence("Esc"), self, activated=self._escape)
        QShortcut(QKeySequence("C"), self, activated=self.crop_b.toggle)
        QShortcut(QKeySequence("O"), self, activated=self.cycle_overlay)
        QShortcut(QKeySequence("Shift+O"), self, activated=self.turn_overlay)
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=lambda: self.crop_b.setChecked(False))
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self, activated=lambda: self.crop_b.setChecked(False))
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, activated=self.copy_settings)
        QShortcut(QKeySequence("Ctrl+Shift+V"), self, activated=self.paste_settings)
        QShortcut(QKeySequence("Z"), self, activated=lambda: self.viewer.toggle_zoom(self._zoom_anchor()))
        for keys, how in (("Ctrl+=", 1), ("Ctrl++", 1), ("Ctrl+-", -1)):
            QShortcut(QKeySequence(keys), self, activated=lambda d=how: self.viewer.zoom_step(d, self._zoom_anchor()))
        QShortcut(QKeySequence("Ctrl+0"), self, activated=lambda: self.viewer.set_zoom(None))
        QShortcut(QKeySequence("Ctrl+1"), self, activated=lambda: self.viewer.set_zoom(1.0, self._zoom_anchor()))
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=lambda: self._spot_delete(self._ssel))
        self._refresh_history()

    # ------------------------------------------------------------------------------------ panel
    def _build_panel(self):
        from ..engine.color import grade as G
        from ..engine.pipeline.edit import Denoise, Tone
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(360)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(14, 12, 14, 12)
        lv = QHBoxLayout()
        lv.addWidget(label("Controls", "h2"))
        lv.addStretch(1)
        self.work_chip = chip("Ready", T.DIM)          # right beside the sliders: is it still working?
        self.work_chip.setToolTip("Amber while the photo is being worked on (it can take a few seconds on big raws "
                                  "or slower PCs); afterwards it shows how long the last update took.")
        lv.addWidget(self.work_chip)
        self._work_tick = QTimer(self, interval=200, timeout=self._show_work)
        self.viewer.busy_changed.connect(self._busy_changed)
        self.level = QComboBox()
        self.level.addItems(["Simple", "Full"])
        self.level.setCurrentIndex(1 if self.settings.get("level") == "Full" else 0)
        self.level.currentIndexChanged.connect(self.apply_level)
        lv.addWidget(self.level)
        outer.addLayout(lv)
        self.histo = Histogram()
        outer.addWidget(self.histo)
        self.info = label("", "dim", wrap=True)
        outer.addWidget(self.info)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 4, 0)
        # Looks
        g = Group("Looks", "A look changes only what it names - never exposure, white balance, crop or the lens tag.")
        self.look_combo = g.add(QComboBox())
        self.look_combo.setToolTip("Violet = shared starter looks, cyan = your own (kept in settings\\looks)")
        ab = QPushButton("Apply look")
        ab.setObjectName("primary")
        ab.clicked.connect(self.apply_look)
        sb = QPushButton("Save look...")
        sb.setToolTip("Save the current settings (the groups you choose) as your own look")
        sb.clicked.connect(self.save_look)
        g.add(hbox(ab, sb))
        self._fill_looks()
        bl.addWidget(g)
        # Clean
        g = Group("Clean", "Noise is removed from the raw data itself, before colour is made.")
        self.dn_on = g.add(QCheckBox("Clean up noise"))
        self.dn_s = g.add(ValueSlider("Strength", 0.0, 2.0, 1.0, 0.05, "{:.0%}", "1 = matched to this camera's "
                                      "measured noise at this ISO."))
        self.dn_l = g.add(ValueSlider("Luminance (grain)", 0.0, 2.5, Denoise().luma, 0.05, "{:.2f}",
                                      "Fine, grain-like noise. Lower keeps more texture."))
        self.dn_c = g.add(ValueSlider("Colour (blotches)", 0.0, 4.0, Denoise().chroma, 0.05, "{:.2f}",
                                      "Coloured mottling, the ugly part of high-ISO noise."))
        self.dn_g = g.add(ValueSlider("Keep grain", 0.0, 1.0, 0.0, 0.05, "{:.0%}",
                                      "Puts back a fine, film-like fraction of the luminance noise."))
        self.dn_m = g.add(QComboBox())
        self.dn_m.addItems(["hybrid", "starlet", "nlm", "mppca"])
        self.dn_m.setToolTip("Method: hybrid = NLM luminance + starlet colour (recommended)")
        self.dn_info = g.add(label("", "dim", wrap=True))
        self.dn_m.currentIndexChanged.connect(lambda *_: self._dn_method_text())
        self._dn_method_text()
        self.dn_safe = g.add(QCheckBox("Protect stars, catchlights and sparks"))
        self.dn_safe.setChecked(True)
        self.dn_safe.setToolTip("After cleaning, anything that moved by more than 4.5 times its measured noise was "
                                "not noise: the original is kept there. Pure noise almost never does that, so "
                                "nothing visible is let back in.")
        self.dn_safe_chip = g.add(chip("Protected: -"))
        self.band = g.add(QCheckBox("Remove row banding (measured, never added)"))
        self.band.setChecked(True)
        self.noise_chip = g.add(chip("Noise model: -"))
        self.full_only = [self.dn_l, self.dn_c, self.dn_g, self.dn_m, self.dn_info]
        bl.addWidget(g)
        # Lens
        g = Group("Lens", "Works without a lens profile, including contactless vintage glass.")
        row = QHBoxLayout()
        self.fix_b = QPushButton("◎ Fix lens")
        self.fix_b.setObjectName("primary")
        self.fix_b.setToolTip("Every automatic lens fix in one go. Shows what it fixed and where each number came "
                              "from. Vignetting is measured but left off unless you tick it.")
        self.fix_b.clicked.connect(self.fix_lens)
        self.fix_b.setMinimumWidth(150)
        row.addWidget(self.fix_b)
        row.addStretch(1)
        rw = QWidget()
        rw.setLayout(row)
        g.add(rw)
        self.fix_report = QVBoxLayout()
        self.fix_report.setSpacing(3)
        frw = QWidget()
        frw.setLayout(self.fix_report)
        g.add(frw)
        self.lens_combo = g.add(QComboBox())
        self.lens_combo.setToolTip("Which lens took this photo (for adapted lenses the camera records nothing)")
        self._fill_lenses()
        self.lens_combo.activated.connect(self._lens_picked)
        self.ca = g.add(QCheckBox("Fix colour fringing (measured)"))
        self.dfr = g.add(QCheckBox("Remove purple / green fringes"))
        self.dfr_s = g.add(ValueSlider("Fringe removal", 0.0, 2.0, 1.0, 0.05, "{:.0%}",
                                       "Only acts next to bright, high-contrast edges; purple subjects keep "
                                       "their colour."))
        self.dist = g.add(QCheckBox("Correct distortion"))
        self.dist.setToolTip("Available after Fix lens finds a lens profile or measures straight lines. Never "
                             "guessed on organic subjects.")
        self.dist_chip = g.add(chip("Distortion: not measured yet (press Fix lens)"))
        self.dist_m = g.add(ValueSlider("Distortion by hand", -100, 100, 0, 1, "{:+.0f}",
                                        "+ pulls in barrel distortion, - pushes out pincushion."))
        self.vig = g.add(QCheckBox("Correct vignetting (off = character)"))
        self.vig.setToolTip("Off by default. Becomes available once Fix lens has measured the falloff "
                            "(lens profile or estimated from the photo).")
        self.vig_s = g.add(ValueSlider("Falloff removed", 0.0, 1.5, 1.0, 0.05, "{:.0%}",
                                       "How much of the measured corner darkening to remove."))
        self.ca.setToolTip("Lateral chromatic aberration, measured from this photo: no lens profile needed.")
        self.ca_chip = g.add(chip("Lateral CA: -"))
        bl.addWidget(g)
        # Shape
        g = Group("Shape", "Straighten, perspective and crop. One resample, no quality loss beyond it.")
        self.rot = g.add(ValueSlider("Straighten", -15.0, 15.0, 0.0, 0.05, "{:+.2f}°",
                                     "+ turns the photo counter-clockwise."))
        row = QHBoxLayout()
        self.level_b = QPushButton("Auto level")
        self.level_b.setToolTip("Levels the photo from its straight lines (horizon, walls). Does nothing when "
                                "there are none to trust.")
        self.level_b.clicked.connect(self.auto_level)
        self.crop_b = QPushButton("Crop")
        self.crop_b.setCheckable(True)
        self.crop_b.setToolTip("Draw the crop on the photo (C). Enter or click again when done.")
        self.crop_b.toggled.connect(self.toggle_crop)
        self.crop_reset = QPushButton("Reset crop")
        self.crop_reset.clicked.connect(self.reset_crop)
        for b in (self.level_b, self.crop_b, self.crop_reset):
            row.addWidget(b)
        rw = QWidget()
        rw.setLayout(row)
        g.add(rw)
        self.aspect = g.add(QComboBox())
        self.aspect.addItems(["Original ratio", "Free", "3:2", "4:3", "5:4", "4:5", "1:1", "16:9", "2:1"])
        self.aspect.setToolTip("Shape of the crop frame")
        self.aspect.activated.connect(self._aspect_picked)
        # composition: an overlay for each method, and crops suggested by method
        from ..engine.local import composition as cp
        self.comp_combo = QComboBox()
        self.comp_combo.addItem("Overlay: off")
        for _, name in cp.OVERLAYS:
            self.comp_combo.addItem(name)
        self.comp_combo.setToolTip("Composition overlay on the crop frame (O: next one; also shows on the whole "
                                   "photo once you press O outside the crop tool)")
        self.comp_combo.activated.connect(self._comp_picked)
        self.comp_turn = QPushButton("Turn")
        self.comp_turn.setToolTip("Mirror the overlay (golden spiral and triangles): Shift+O")
        self.comp_turn.clicked.connect(self.turn_overlay)
        g.add(hbox(self.comp_combo, self.comp_turn))
        self.suggest_b = QPushButton("Suggest crops")
        self.suggest_b.setObjectName("primary")
        self.suggest_b.setToolTip("Crops that put the eyes (or the subject) on the chosen overlay's strong points, "
                                  "keep heads whole with room above, leave space in front of a turned face and "
                                  "never cut at a joint. With the overlay off, any method. Uses the crop shape above.")
        self.suggest_b.clicked.connect(self.suggest_crops)
        g.add(hbox(self.suggest_b))
        self.suggest_list = QListWidget()
        self.suggest_list.setWordWrap(True)
        self.suggest_list.setMaximumHeight(118)
        self.suggest_list.setToolTip("Click one to try it (then adjust the frame, Enter when done)")
        self.suggest_list.itemClicked.connect(self._suggestion_picked)
        self.suggest_list.hide()
        g.add(self.suggest_list)
        self._suggestions = []
        last = self.settings.get("comp_overlay", "thirds")
        self.viewer.comp = last if (last in cp.NAMES or last == "off") else "thirds"
        self.viewer.comp_orient = int(self.settings.get("comp_orient", 0) or 0)
        self._sync_comp_combo()
        self.vert = g.add(ValueSlider("Vertical perspective", -100, 100, 0, 1, "{:+.0f}",
                                      "+ widens the top: straightens buildings that lean back."))
        self.horiz = g.add(ValueSlider("Horizontal perspective", -100, 100, 0, 1, "{:+.0f}",
                                       "+ makes the right side taller."))
        self.constrain = g.add(QCheckBox("Constrain to image (no empty corners)"))
        self.constrain.setChecked(True)
        bl.addWidget(g)
        # Light
        g = Group("Light", "Nothing here changes exposure by itself. Auto sets highlights, whites and blacks only.")
        self.auto_light_b = QPushButton("Auto")
        self.auto_light_b.setToolTip("Highlights, whites and blacks for this photo. Never exposure or shadows: a "
                                     "low-key photo stays low-key.")
        self.auto_light_b.clicked.connect(self.auto_light)
        lr = QPushButton("Reset light")
        lr.setToolTip("Resets highlights, shadows, whites, blacks, Smart lighting and the curve (not exposure)")
        lr.clicked.connect(self.reset_light)
        g.add(hbox(self.auto_light_b, lr))
        self.ev = g.add(ValueSlider("Exposure", -4.0, 4.0, 0.0, 0.05, "{:+.2f} EV"))
        self.con = g.add(ValueSlider("Contrast", 0.8, 1.9, Tone().contrast, 0.01, "{:.2f}"))
        self.l_hl = g.add(ValueSlider("Highlights", -100, 100, 0, 1, "{:+.0f}",
                                      "- brings back bright tones (sky, candle-lit skin), + brightens them. Works "
                                      "in stops on the whole photo: no halos."))
        self.l_sh = g.add(ValueSlider("Shadows", -100, 100, 0, 1, "{:+.0f}",
                                      "+ opens up the dark tones, - deepens them. Mid grey stays where it is."))
        self.l_wh = g.add(ValueSlider("Whites", -100, 100, 0, 1, "{:+.0f}",
                                      "+ reaches white sooner (a brighter top end), - holds more of the "
                                      "highlights."))
        self.l_bk = g.add(ValueSlider("Blacks", -100, 100, 0, 1, "{:+.0f}",
                                      "- deepens the black point, + lifts it to a matte black."))
        self.l_smart = g.add(ValueSlider("Smart lighting", 0, 100, 0, 1, "{:.0f}",
                                         "Opens up deep shadow areas region by region (like DxO Smart Lighting), "
                                         "keeping their texture and leaving lit areas alone. 0 = off."))
        from ..engine.color import hdr as HD
        self.hdr_on = g.add(QCheckBox("Auto HDR  (opens shadows, holds highlights)"))
        self.hdr_on.setToolTip("Measures how much range this photo has and makes room for it without flattening "
                               "local contrast or making halos. It changes brightness, so it is never on unless "
                               "you turn it on.")
        self.hdr_strength = QComboBox()
        for key, st in HD.STRENGTHS.items():
            self.hdr_strength.addItem(st["name"], key)
        self.hdr_strength.setCurrentIndex(list(HD.STRENGTHS).index("medium"))
        g.add(self.hdr_strength)
        self.hdr_text = g.add(label("", "dim", wrap=True))
        self.hdr_amt = g.add(ValueSlider("  amount", 0, 150, 100, 1, "{:.0f}%",
                                         "How much of that strength: 100% is what the photo was measured to need."))
        self.hdr_engine = QComboBox()
        for key, en in HD.ENGINES.items():
            self.hdr_engine.addItem(en["name"], key)
        g.add(self.hdr_engine)
        self.hdr_engine_text = g.add(label("", "dim", wrap=True))
        self.hdr_chip = g.add(chip("Auto HDR: off"))
        self.hdr_strength.currentIndexChanged.connect(lambda *_: self._hdr_texts())
        self.hdr_engine.currentIndexChanged.connect(lambda *_: self._hdr_texts())
        self._hdr_texts()
        self.l_haze = g.add(ValueSlider("Dehaze", -100, 100, 0, 1, "{:+.0f}",
                                        "+ cuts through haze and mist (more contrast and colour far away), - adds a "
                                        "soft haze. Makes hazy photos darker overall: adjust exposure after."))
        self.tone = g.add(QComboBox())
        self.tone.addItems(["Studio Standard", "Camera match", "Linear (no curve)"])
        self.mcb = g.add(QCheckBox("Match camera brightness (optional)"))
        self.mcb.setToolTip("Off by default. Shifts brightness per camera body to match the camera's JPEG.")
        self.hl_mode = g.add(QComboBox())
        self.hl_mode.addItems(["Clipped highlights: plain clip", "Clipped highlights: rebuild colour",
                               "Clipped highlights: measure from lens glow"])
        self.hl_mode.setToolTip("Where a light or a colour channel clipped (a candle flame, a sunlit petal): plain clip "
                                "makes it white or yellow; rebuild colour borrows the colour from the edge of the "
                                "clipped area; measure from lens glow reads how bright and what colour the blown light "
                                "really was from the glow the lens spread around it (as NASA does for the Sun). Shows "
                                "when you pull the highlights or exposure down.")
        self.glow_combo = g.add(QComboBox())
        self.glow_combo.setToolTip("Your lens's measured glow (Library > Measure lens glow). Without one, only lights "
                                   "that blew out in some colours but not all can be measured.")
        self.glow_combo.currentIndexChanged.connect(self._glow_combo_changed)
        self.glow_chip = g.add(chip("", T.DIM))
        self.glow_chip.setWordWrap(True)
        bl.addWidget(g)
        # Tone curve
        g = Group("Tone curve", "Drawn in screen terms: the middle of the box is the middle of the screen's range.")
        self.curve_btns = QButtonGroup(self)
        btns = []
        for i, t in enumerate(("RGB", "Red", "Green", "Blue")):
            b = QPushButton(t)
            b.setCheckable(True)
            b.setChecked(i == 0)
            b.setStyleSheet("padding: 5px 8px;")
            self.curve_btns.addButton(b, i)
            btns.append(b)
        cr = QPushButton("Reset")
        cr.setStyleSheet("padding: 5px 8px;")
        cr.setToolTip("Reset this channel's curve")
        g.add(hbox(*btns, cr))
        self.curve = g.add(CurveEditor())
        self.curve_btns.idClicked.connect(self._curve_channel)
        cr.clicked.connect(self.curve.reset_channel)
        self.curve.changed.connect(self.controls_changed)
        g.add(label("Curve regions (parametric):", "dim"))
        self.p_hi = g.add(ValueSlider("Highlights region", -100, 100, 0, 1, "{:+.0f}"))
        self.p_li = g.add(ValueSlider("Lights region", -100, 100, 0, 1, "{:+.0f}"))
        self.p_da = g.add(ValueSlider("Darks region", -100, 100, 0, 1, "{:+.0f}"))
        self.p_sh = g.add(ValueSlider("Shadows region", -100, 100, 0, 1, "{:+.0f}"))
        g.add(label("Film curve (Studio Standard):", "dim"))
        self.filmic = g.add(FilmicPlot())
        self.t_toe = g.add(ValueSlider("Toe", -100, 100, 0, 1, "{:+.0f}",
                                       "How the deepest shadows bend, like film: + opens them softly, - crushes "
                                       "them. Mid grey and highlights do not move."))
        self.t_shoulder = g.add(ValueSlider("Highlight roll-off starts", 0.3, 2.2, 1.0, 0.05, "{:+.2f} EV",
                                            "Stops above mid grey where highlights start to compress toward white. "
                                            "Lower = softer, earlier roll-off."))
        self.curve_group = g
        bl.addWidget(g)
        # Colour
        g = Group("Colour", "White balance, saturation, vibrance and black & white.")
        self.wb = g.add(QComboBox())
        self.wb.addItems(["As shot", "Daylight", "Custom", "Picked / Auto"])
        self.wb.activated.connect(self._wb_mode_picked)
        self.pick_wb = QPushButton("◉ Pick white")
        self.pick_wb.setCheckable(True)
        self.pick_wb.setToolTip("Then click something in the photo that should be neutral grey or white")
        self.pick_wb.toggled.connect(lambda on: self._set_pick("wb" if on else None))
        self.auto_wb_b = QPushButton("Auto")
        self.auto_wb_b.setToolTip("Estimates the colour of the light from the photo's edges (grey-edge). It "
                                  "neutralises candle warmth: for candle-lit portraits As shot usually looks truer.")
        self.auto_wb_b.clicked.connect(lambda: self.auto_wb())
        g.add(hbox(self.pick_wb, self.auto_wb_b))
        from ..engine.color import autowb as AW
        self.wb_method = QComboBox()
        for key, info in AW.METHODS.items():
            self.wb_method.addItem(f"Auto uses: {info['name']}", key)
        self.wb_method.setToolTip("\n".join(f"{i['name']}. Best for: {i['best']} Weak at: {i['weak']}"
                                             for i in AW.METHODS.values()))
        g.add(self.wb_method)
        self.wb_mood = g.add(ValueSlider("Auto: keep the mood", 0, 100, 100, 5, "{:.0f}% neutral",
                                         "How far Auto corrects the light. 100% = fully neutral; less keeps part "
                                         "of the light's own colour. Auto picks 60% for warm low-light photos (a "
                                         "candle should stay warm) and 90% otherwise; move it and Auto runs again."))
        self.wb_mood.changed.connect(lambda v: self.auto_wb(mood=v / 100.0) if self._wb_auto_used else None)
        self.wb_chip = g.add(chip("Auto white balance: not used"))
        self._wb_auto_used = False
        self.temp = g.add(ValueSlider("Temperature", 2000, 12000, 5500, 50, "{:.0f} K"))
        self.tint = g.add(ValueSlider("Tint", -100, 100, 0, 1, "{:+.0f}"))
        self.temp.changed.connect(self._wb_slider_moved)          # runs before controls_changed (connected later)
        self.tint.changed.connect(self._wb_slider_moved)
        self.wb_chip = g.add(chip("White balance: as shot"))
        self.c_sat = g.add(ValueSlider("Saturation", -100, 100, 0, 1, "{:+.0f}",
                                       "All colours alike. -100 = no colour at all."))
        self.c_vib = g.add(ValueSlider("Vibrance", -100, 100, 0, 1, "{:+.0f}",
                                       "Muted colours more than strong ones; skin tones are protected."))
        self.auto_col_b = QPushButton("Auto colour")
        self.auto_col_b.setToolTip("Measures how colourful the photo is and adds vibrance if it is dull. Never "
                                   "lowers colour, never touches white balance or brightness.")
        self.auto_col_b.clicked.connect(self.auto_colour)
        g.add(hbox(self.auto_col_b))
        self.c_bw = g.add(QCheckBox("Black && white"))
        self.bw_box = QWidget()
        bwl = QVBoxLayout(self.bw_box)
        bwl.setContentsMargins(0, 0, 0, 0)
        bwl.setSpacing(2)
        self.bw_filter = QComboBox()
        self.bw_filter.addItems(["Filter: none (as mixed below)", *G.BW_FILTERS.keys()])
        self.bw_filter.setToolTip("Like a coloured filter on the lens with black & white film: sets the mixer "
                                  "below (red darkens blue skies, green lightens foliage...)")
        self.bw_filter.activated.connect(self._bw_filter_picked)
        bwl.addWidget(self.bw_filter)
        bwl.addWidget(label("How bright each colour becomes in black & white:", "dim", wrap=True))
        self.bw_sl = []
        for n, h in zip(G.BANDS, G.BAND_HUES):
            sl = ValueSlider(n.capitalize(), -100, 100, 0, 1, "{:+.0f}")
            sl.name.setStyleSheet(f"color:{oklch_colour(0.78, 0.12, h).name()};")
            bwl.addWidget(sl)
            self.bw_sl.append(sl)
        g.add(self.bw_box)
        self.prof_combo = g.add(QComboBox())
        self.prof_combo.setToolTip("Camera profile: how camera colours become real colours. Studio uses the camera "
                                   "matrix; add .dcp profiles (DNG camera profiles) you own or made. Your profiles "
                                   "stay in settings\\camera_profiles and are never shared.")
        self.prof_combo.activated.connect(self._profile_picked)
        self._fill_profiles("")
        self.lut_combo = g.add(QComboBox())
        self.lut_combo.setToolTip("A 3D look-up table (.cube) laid over your colour settings. Your own .cube files "
                                  "live in settings\\luts.")
        self.lut_amt = g.add(ValueSlider("  LUT amount", 0.0, 1.0, 1.0, 0.01, "{:.0%}"))
        self._fill_luts("")
        self.lut_combo.activated.connect(self._lut_picked)
        bl.addWidget(g)
        # Colour wheel
        g = Group("Colour wheel", "Choose a colour range on the wheel and move it to another hue, or change its "
                                  "saturation and lightness. Lightness is kept when the hue moves.")
        self.rng_combo = QComboBox()
        self.rng_combo.activated.connect(self._range_selected)
        nb = QPushButton("+")
        nb.setToolTip("New colour range")
        nb.clicked.connect(lambda: self.new_range())
        db = QPushButton("−")
        db.setToolTip("Remove this colour range")
        db.clicked.connect(self.remove_range)
        self.pick_rng = QPushButton("◉")
        self.pick_rng.setCheckable(True)
        self.pick_rng.setToolTip("Choose the range by clicking a colour in the photo")
        self.pick_rng.toggled.connect(lambda on: self._set_pick("range" if on else None))
        for b in (nb, db, self.pick_rng):
            b.setStyleSheet("padding: 5px 10px;")
        g.add(hbox((self.rng_combo, 1), nb, db, self.pick_rng, stretch=False))
        self.ring = g.add(HueRing())
        self.ring.changed.connect(self._ring_dragged)
        self.ring.picked.connect(self._range_selected)
        self.r_shift = g.add(ValueSlider("Hue shift", -90, 90, 0, 1, "{:+.0f}°",
                                         "Sends the chosen colours toward another hue."))
        self.r_sat = g.add(ValueSlider("Saturation", -100, 100, 0, 1, "{:+.0f}"))
        self.r_lum = g.add(ValueSlider("Lightness", -100, 100, 0, 1, "{:+.0f}"))
        self.r_width = g.add(ValueSlider("Range width", 5, 120, 40, 1, "{:.0f}°",
                                         "How wide a slice of the wheel gets the full effect."))
        self.r_fall = g.add(ValueSlider("Softness", 5, 90, 25, 1, "{:.0f}°",
                                        "How gently the effect fades beyond the range."))
        for w in (self.r_shift, self.r_sat, self.r_lum, self.r_width, self.r_fall):
            w.changed.connect(self._range_sliders)
        self.wheel_group = g
        bl.addWidget(g)
        # HSL
        g = Group("HSL", "Eight colour bands: pick one, then move its hue, saturation and lightness. A dot marks "
                         "the bands you have changed.")
        self.sw = g.add(SwatchRow(G.BANDS, G.BAND_HUES))
        self.sw.selected.connect(lambda _: self._refresh_hsl_ui())
        self.hsl_h = g.add(ValueSlider("Hue", -100, 100, 0, 1, "{:+.0f}", "100 = 30 degrees round the colour wheel."))
        self.hsl_s = g.add(ValueSlider("Saturation", -100, 100, 0, 1, "{:+.0f}"))
        self.hsl_l = g.add(ValueSlider("Lightness", -100, 100, 0, 1, "{:+.0f}"))
        for w in (self.hsl_h, self.hsl_s, self.hsl_l):
            w.changed.connect(self._hsl_sliders)
        self.hsl_group = g
        bl.addWidget(g)
        # Colour grading
        g = Group("Colour grading", "Tint the shadows, midtones and highlights separately (split toning). Works on "
                                    "black & white too.")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self.wheels, self.wheel_l = {}, {}
        for i, (k, t) in enumerate((("shadows", "Shadows"), ("midtones", "Midtones"), ("highlights", "Highlights"),
                                    ("global", "Global"))):
            cell = QWidget()
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(0)
            w = GradeWheel(t)
            sl = ValueSlider("lightness", -100, 100, 0, 1, "{:+.0f}")
            cl.addWidget(w)
            cl.addWidget(sl)
            w.changed.connect(self.controls_changed)
            sl.changed.connect(self.controls_changed)
            self.wheels[k], self.wheel_l[k] = w, sl
            grid.addWidget(cell, i // 2, i % 2)
        gw = QWidget()
        gw.setLayout(grid)
        g.add(gw)
        self.g_bal = g.add(ValueSlider("Balance", -100, 100, 0, 1, "{:+.0f}",
                                       "- = more of the photo counts as shadows, + = more as highlights."))
        self.g_blend = g.add(ValueSlider("Blending", 0, 100, 50, 1, "{:.0f}",
                                         "How softly shadows, midtones and highlights overlap."))
        self.grade_group = g
        bl.addWidget(g)
        # Detail
        g = Group("Detail", "Detail by scale: microcontrast in every sense of the word. Pick an Amount and a "
                            "Balance, or (Full) dial in each method and combine them.")
        from ..engine.detail import scales as DS
        self.ds_preset = QComboBox()
        self.ds_preset.addItems(["Preset..."] + list(DS.PRESETS))
        self.ds_preset.setToolTip("Tonal: shades and modelling (Bill's microcontrast). Crisp: fine bite. Both: sharp "
                                  "and tonal together. Soft glow: a gentle haze, softened fine detail.")
        self.ds_preset.activated.connect(self._ds_preset_picked)
        self.ds_bw = QCheckBox("Judge in black and white")
        self.ds_bw.setToolTip("Shows the photo in black and white while you adjust (it does not change the edit).\n"
                              "Microcontrast shows best once the colour is out of the way.")
        self.ds_bw.toggled.connect(self._ds_bw_toggled)
        g.add(hbox(self.ds_preset, self.ds_bw))
        self.ds_amount = g.add(ValueSlider("Amount", 0, 100, 0, 1, "{:.0f}",
                                           "How much microcontrast. 60-80 is plain to see at screen size; big edges "
                                           "are left alone (no halos) and noise is not lifted as if it were detail."))
        self.ds_balance = g.add(ValueSlider("Balance  (tonal - crisp)", -100, 100, -50, 1, "{:+.0f}",
                                            "Left = Tonal: medium scales up, the finest scale relaxed (modelled, "
                                            "more shades). Middle = both. Right = Crisp: the finest scale up "
                                            "(etched). Starts on the Tonal side."))
        self.ds_meter = g.add(DetailMeter())
        self.ds_ratio = g.add(chip("Balance (medium : fine): -"))
        self.ds_ratio.setToolTip("The balance number: medium-scale tonal variation over pixel-level contrast, before "
                                 "and after, measured at 100% in the middle of the photo. Higher reads as modelled, "
                                 "lower as etched. (On the same leaf: Heliar 1.35, Canon 180 macro 1.21, Sigma Art "
                                 "1.06.)")
        self.ds_note = g.add(chip(""))
        self.ds_note.setVisible(False)
        self.ds_diff = QPushButton("Recover diffraction loss")
        self.ds_diff.setToolTip("Turns on Recover lens sharpness, which undoes the blur measured in this photo.")
        self.ds_diff.clicked.connect(self._ds_recover_diffraction)
        self.ds_diff.setVisible(False)
        g.add(self.ds_diff)
        # every method: its own switch, its own amount, and what it is good and bad at (rule 23.1)
        self.ds_rows = {}
        self.ds_methods_box = QWidget()
        mb = QVBoxLayout(self.ds_methods_box)
        mb.setContentsMargins(0, 4, 0, 0)
        mb.setSpacing(4)
        mb.addWidget(label("Methods (each has its own amount; any number can be on):", "dim", wrap=True))
        for key in ("tonal", "local", "crisp", "separation", "veil", "broad"):
            info = DS.METHODS[key]
            on = QCheckBox(info["name"])
            on.setChecked(True)
            on.setStyleSheet(f"QCheckBox {{ color: {DetailMeter.COLOURS[key]}; font-weight: 600; }}")
            sl = ValueSlider("  amount", info["lo"], info["hi"], 0, 1, "{:+.0f}" if info["lo"] < 0 else "{:.0f}",
                             f"{info['does']}\nBest for: {info['best']}\nWeak at: {info['weak']}")
            txt = label(f"{info['does']}  Best for: {info['best'].rstrip('.').lower()}.  Weak at: "
                        f"{info['weak'].rstrip('.').lower()}.", "dim", wrap=True)
            mb.addWidget(on)
            mb.addWidget(sl)
            mb.addWidget(txt)
            self.ds_rows[key] = (on, sl)
            on.toggled.connect(self.controls_changed)
            sl.changed.connect(lambda _v, k=key: self._ds_method_moved(k))
        g.add(self.ds_methods_box)
        self.mc = self.ds_rows["local"][1]                  # "Local contrast" is the first engine (microcontrast)
        g.add(label("Where it works (all methods):", "dim"))
        self.mc_sh = g.add(ValueSlider("  in shadows", 0.0, 1.0, 1.0, 0.05, "{:.0%}"))
        self.mc_mid = g.add(ValueSlider("  in midtones", 0.0, 1.0, 0.7, 0.05, "{:.0%}"))
        self.mc_hi = g.add(ValueSlider("  in highlights", 0.0, 1.0, 0.3, 0.05, "{:.0%}"))
        self.mc_sc = g.add(ValueSlider("  local contrast scale (fine - broad)", -1.0, 1.0, 0.0, 0.05, "{:+.2f}",
                                       "Local contrast only: which size of tonal detail is lifted, leaf veins "
                                       "(fine) to whole leaves (broad)."))
        self.f_tex = g.add(ValueSlider("Texture (earlier tool)", -100, 100, 0, 1, "{:+.0f}",
                                       "The earlier Texture slider, kept so older edits can still be changed. Crisp "
                                       "replaces it: set this to 0 and use Crisp instead."))
        self.f_tex.setVisible(False)
        self.ds_amount.changed.connect(self._ds_simple_moved)
        self.ds_balance.changed.connect(self._ds_simple_moved)
        self.sh_on = g.add(QCheckBox("Recover lens sharpness"))
        self.sh_on.setToolTip("Undoes part of the blur measured in this photo by deconvolution. Off for "
                              "character lenses unless you turn it on. Judge it at 100% (click the photo).")
        self.sh_s = g.add(ValueSlider("  amount", 0.0, 1.5, 0.5, 0.05, "{:.0%}",
                                      "100% = deconvolve to a 0.5 px target. Never to infinity: no ringing."))
        self.blur_chip = g.add(chip("Measured blur: -"))
        g.add(label("Film grain:", "dim"))
        self.f_grain = g.add(ValueSlider("Grain", 0, 100, 0, 1, "{:.0f}",
                                         "Film-like grain, strongest in the midtones. The preview shows what a "
                                         "downsized export would; judge the grain itself at 100%."))
        self.f_gsize = g.add(ValueSlider("  grain size", 0.5, 3.0, 1.0, 0.05, "{:.2f} px"))
        self.f_grough = g.add(ValueSlider("  roughness", 0, 100, 50, 1, "{:.0f}"))
        self.f_gcol = g.add(QCheckBox("  colour grain (like colour film)"))
        g.add(label("Demosaic (how each pixel's colour is worked out):", "dim", wrap=True))
        self.dem = g.add(QComboBox())
        for key, name, text in DEMOSAICS:
            self.dem.addItem(name, key)
            self.dem.setItemData(self.dem.count() - 1, text, Qt.ItemDataRole.ToolTipRole)
        self.dem_text = g.add(label(DEMOSAICS[0][2], "dim", wrap=True))
        self.dem.currentIndexChanged.connect(lambda i: self.dem_text.setText(DEMOSAICS[i][2]))
        self.geq = g.add(QCheckBox("Even out the two greens"))
        self.geq.setToolTip("Some sensors read the two green pixels of each 2x2 group slightly differently, which "
                            "shows as a fine maze in skies and walls. This evens them out only where they nearly "
                            "agree, so real detail is left alone. Off by default.")
        self.detail_group = g
        bl.addWidget(g)
        # Masks (local adjustments)
        g = Group("Masks", "Change one part of the photo: paint it, draw a gradient or a circle, click an object, "
                           "select a person (or just their hair, skin, clothes...), or pick a colour or brightness "
                           "range. Masks stay on their subject if you crop later.")
        self.mask_new = {}
        btns = []
        for kind, text, tip in (("brush", "Brush", "Paint where the change goes"),
                                ("linear", "Gradient", "A straight fade, e.g. a darker sky"),
                                ("radial", "Radial", "A circle or ellipse, e.g. light on a face"),
                                ("object", "Object", "Click an object to select it (a leaf, a face)"),
                                ("colour", "Colour", "Every area of one colour"),
                                ("luminance", "Brightness", "Every area in a range of brightness"),
                                ("skin", "Skin", "Every area of skin tone (then Smooth skin below). Narrow it to "
                                                 "one face with  x Brush"),
                                ("person", "Person", "Select the person (found on this PC), then pick a part: skin, "
                                                     "face, hair, clothes, arms, legs, torso or the background")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setStyleSheet("padding: 5px 6px;")
            b.clicked.connect(lambda _=False, k=kind: self.new_mask(k))
            self.mask_new[kind] = b
            btns.append(b)
        g.add(hbox(*btns[:4]))
        g.add(hbox(*btns[4:]))
        self.person_part = QComboBox()
        from ..engine.local.people import PART_NAMES, PARTS
        for k in PARTS:
            self.person_part.addItem(PART_NAMES[k], k)
        self.person_part.setToolTip("Which part of the person this mask selects (worked out from the person, their "
                                    "body points, the face map and their own skin colour, then snapped to the edges)")
        self.person_part.activated.connect(self._person_part_picked)
        self.person_part_row = hbox(label("Part"), self.person_part)
        g.add(self.person_part_row)
        self.mask_list = g.add(QListWidget())
        self.mask_list.setFixedHeight(96)
        self.mask_list.currentRowChanged.connect(self._mask_selected)
        self.mask_list.itemChanged.connect(self._mask_item_changed)
        self.mask_del = QPushButton("Delete")
        self.mask_del.clicked.connect(self.delete_mask)
        self.mask_inv = QPushButton("Invert")
        self.mask_inv.setCheckable(True)
        self.mask_inv.toggled.connect(self._mask_invert)
        self.mask_show = QPushButton("Show mask")
        self.mask_show.setCheckable(True)
        self.mask_show.setToolTip("Red overlay of the selected mask (always on while a mask tool is active)")
        self.mask_show.toggled.connect(self._mask_show)
        for b in (self.mask_del, self.mask_inv, self.mask_show):
            b.setStyleSheet("padding: 5px 8px;")
        g.add(hbox(self.mask_del, self.mask_inv, self.mask_show))
        self.mask_add = QComboBox()
        self.mask_add.addItems(["Refine: add to mask...", "+ Brush", "+ Gradient", "+ Radial", "+ Object",
                                "+ Colour", "+ Brightness", "- Brush (subtract)", "- Gradient (subtract)",
                                "- Radial (subtract)", "- Object (subtract)", "x Brightness (only where)",
                                "x Colour (only where)", "x Brush (only where)", "x Skin (only where)",
                                "+ Skin", "+ Person", "- Person (subtract)", "x Person (only where)"])
        self.mask_add.setToolTip("Add another shape to the selected mask: + adds, - takes away, x keeps only "
                                 "where both are")
        self.mask_add.activated.connect(self._mask_add_picked)
        g.add(self.mask_add)
        self.mask_tool_box = QWidget()
        mtl = QVBoxLayout(self.mask_tool_box)
        mtl.setContentsMargins(0, 0, 0, 0)
        mtl.setSpacing(2)
        self.b_size = ValueSlider("Brush size", 0.5, 25, 3, 0.5, "{:.1f}%", "Radius as a share of the photo's long edge")
        self.b_feather = ValueSlider("Brush feather", 0, 100, 50, 1, "{:.0f}")
        self.b_flow = ValueSlider("Brush flow", 5, 100, 100, 1, "{:.0f}")
        self.b_auto = QCheckBox("Auto mask (stay on the colour you start on)")
        self.b_erase = QCheckBox("Erase (or hold Alt while painting)")
        self.b_done = QPushButton("Done painting")
        self.b_done.clicked.connect(lambda: self._set_mask_tool(None))
        for w in (self.b_size, self.b_feather, self.b_flow, self.b_auto, self.b_erase, self.b_done):
            mtl.addWidget(w)
        self.b_size.changed.connect(lambda v: setattr(self.viewer, "brush_frac", v / 100.0))
        g.add(self.mask_tool_box)
        self.range_box = QWidget()
        rbl = QVBoxLayout(self.range_box)
        rbl.setContentsMargins(0, 0, 0, 0)
        rbl.setSpacing(2)
        self.r_lo = ValueSlider("From brightness", 0, 100, 60, 1, "{:.0f}%")
        self.r_hi = ValueSlider("To brightness", 0, 100, 100, 1, "{:.0f}%")
        self.r_soft = ValueSlider("Softness", 1, 50, 10, 1, "{:.0f}%")
        self.c_width = ValueSlider("Colour range width", 5, 120, 30, 1, "{:.0f}°")
        self.c_soft = ValueSlider("Colour softness", 2, 90, 20, 1, "{:.0f}°")
        self.c_chroma = ValueSlider("Ignore greys below", 0, 20, 3, 0.5, "{:.1f}")
        self.r_feather = ValueSlider("Radial feather", 1, 100, 50, 1, "{:.0f}")
        for w in (self.r_lo, self.r_hi, self.r_soft, self.c_width, self.c_soft, self.c_chroma, self.r_feather):
            rbl.addWidget(w)
            w.changed.connect(self._shape_params_changed)
        g.add(self.range_box)
        g.add(label("Adjust inside the mask:", "dim"))
        self.m_sl = {}
        for key, name, lo, hi, step, fmt in (
                ("opacity", "Amount (opacity)", 0, 100, 1, "{:.0f}%"), ("exposure", "Exposure", -4, 4, 0.05, "{:+.2f} EV"),
                ("contrast", "Contrast", -100, 100, 1, "{:+.0f}"), ("highlights", "Highlights", -100, 100, 1, "{:+.0f}"),
                ("shadows", "Shadows", -100, 100, 1, "{:+.0f}"), ("whites", "Whites", -100, 100, 1, "{:+.0f}"),
                ("blacks", "Blacks", -100, 100, 1, "{:+.0f}"), ("temperature", "Temperature", -100, 100, 1, "{:+.0f}"),
                ("tint", "Tint", -100, 100, 1, "{:+.0f}"), ("saturation", "Saturation", -100, 100, 1, "{:+.0f}"),
                ("vibrance", "Vibrance", -100, 100, 1, "{:+.0f}"), ("texture", "Texture", -100, 100, 1, "{:+.0f}"),
                ("clarity", "Clarity", -100, 100, 1, "{:+.0f}"), ("dehaze", "Dehaze", -100, 100, 1, "{:+.0f}"),
                ("sharpness", "Sharpness", -100, 100, 1, "{:+.0f}"), ("noise", "Noise reduction", 0, 100, 1, "{:.0f}"),
                ("smooth_skin", "Smooth skin", 0, 100, 1, "{:.0f}"),
                ("skin_size", "  blemish size", -100, 100, 1, "{:+.0f}")):
            sl = g.add(ValueSlider(name, lo, hi, 100 if key == "opacity" else 0, step, fmt))
            sl.changed.connect(lambda v, k=key: self._mask_adjust(k, v))
            self.m_sl[key] = sl
        self.mask_group = g
        bl.addWidget(g)
        # Retouch
        g = Group("Retouch", "Remove dust, spots and stray hairs. Heal blends the repair in; Clone copies exactly; "
                             "Red eye fixes flash eyes. Click a spot or brush along a hair; the source is chosen "
                             "for you (drag its dashed circle to change it). Alt-click removes a spot. Zoom to 100% "
                             "with Z or the mouse wheel.")
        self.rt_btn = {}
        grp = QButtonGroup(self)
        grp.setExclusive(False)
        for kind, text, tip in (("heal", "Heal", "Repair a spot with texture from nearby, matched to its light"),
                                ("clone", "Clone", "Copy another place exactly (for hard edges and patterns)"),
                                ("redeye", "Red eye", "Click a red pupil"),
                                ("pupil", "Pupil", "Click a pupil the flash blew out (white or golden), or an "
                                 "animal's glowing eye (green, yellow, white): it becomes a dark pupil and keeps "
                                 "its catchlight")):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setStyleSheet("padding: 5px 6px;")
            b.toggled.connect(lambda on, k=kind: self._set_spot_tool(k if on else None))
            grp.addButton(b)
            self.rt_btn[kind] = b
        g.add(hbox(*self.rt_btn.values()))
        self.rt_size = g.add(ValueSlider("Spot size", 0.05, 8, 0.35, 0.05, "{:.2f}%",
                                         "Radius as a share of the photo's long edge (the selected spot, or new ones)"))
        self.rt_feather = g.add(ValueSlider("Feather", 0, 100, 25, 1, "{:.0f}"))
        self.rt_opacity = g.add(ValueSlider("Opacity", 0, 100, 100, 1, "{:.0f}%"))
        for w in (self.rt_size, self.rt_feather, self.rt_opacity):
            w.changed.connect(self._spot_params_changed)
        self.rt_find = QPushButton("Find dust")
        self.rt_find.setToolTip("Look for dust spots in this photo (dark, round, soft blobs in smooth areas). Check "
                                "each one: a bird in the sky looks the same. Alt-click removes a wrong one.")
        self.rt_find.clicked.connect(self.find_dust)
        self.rt_map = QPushButton("Dust map")
        self.rt_map.setToolTip("Heal the dust this camera always has, from its dust map (only spots this photo "
                               "shows). Make the map in the Library: select photos, press Learn dust map.")
        self.rt_map.clicked.connect(self.dust_from_map)
        for b in (self.rt_find, self.rt_map):
            b.setStyleSheet("padding: 5px 8px;")
        g.add(hbox(self.rt_find, self.rt_map))
        self.rt_del = QPushButton("Delete spot")
        self.rt_del.clicked.connect(lambda: self._spot_delete(self._ssel))
        self.rt_clear = QPushButton("Clear all")
        self.rt_clear.clicked.connect(self.clear_spots)
        self.rt_show = QPushButton("Show spots")
        self.rt_show.setCheckable(True)
        self.rt_show.toggled.connect(self._show_spots)
        for b in (self.rt_del, self.rt_clear, self.rt_show):
            b.setStyleSheet("padding: 5px 8px;")
        g.add(hbox(self.rt_del, self.rt_clear, self.rt_show))
        self.rt_info = g.add(chip("No spots", T.DIM))
        self.rt_map_chip = g.add(chip("Dust map: none for this camera yet", T.DIM))
        self.retouch_group = g
        bl.addWidget(g)
        self._build_portrait(bl)
        # History
        g = Group("History", "Click a step to go back to it. Snapshots are named moments to return to. Both are "
                             "kept in the sidecar.")
        self.hist_list = g.add(QListWidget())
        self.hist_list.setFixedHeight(150)
        self.hist_list.itemClicked.connect(lambda it: self.jump_history(self.hist_list.row(it)))
        snap = QPushButton("Snapshot...")
        snap.setToolTip("Keep the current settings under a name")
        snap.clicked.connect(self.take_snapshot)
        rest = QPushButton("Restore")
        rest.clicked.connect(self.restore_snapshot)
        dels = QPushButton("Delete")
        dels.clicked.connect(self.delete_snapshot)
        g.add(hbox(snap, rest, dels))
        self.snap_list = g.add(QListWidget())
        self.snap_list.setFixedHeight(90)
        self.snap_list.itemDoubleClicked.connect(lambda _: self.restore_snapshot())
        bl.addWidget(g)
        self.full_only += getattr(self, "full_only_portrait", [])
        self.full_only += [self.dist_m, self.vig_s, self.vert, self.horiz, self.constrain, self.mc_sh, self.mc_mid,
                           self.mc_hi, self.mc_sc, self.ds_methods_box, self.hdr_engine, self.hdr_engine_text,
                           self.dem, self.dem_text, self.geq, self.dfr_s, self.tone, self.mcb, self.lut_combo,
                           self.lut_amt, self.curve_group, self.wheel_group, self.hsl_group, self.grade_group,
                           self.hl_mode, self.prof_combo, self.f_gsize, self.f_grough, self.f_gcol, self.glow_combo]
        bl.addStretch(1)
        for cb in body.findChildren(QComboBox):           # long lens names must not widen the panel
            cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(12)
        for lb in body.findChildren(QLabel):
            if lb.objectName() == "chip":
                lb.setWordWrap(True)
        for w in body.findChildren(QCheckBox):
            sp = w.sizePolicy()
            sp.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
            w.setSizePolicy(sp)
            w.setMinimumWidth(40)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        reset = QPushButton("Reset all")
        reset.clicked.connect(self.reset)
        outer.addWidget(reset)
        for w in (self.dn_s, self.dn_l, self.dn_c, self.dn_g, self.ev, self.con, self.temp, self.tint, self.dfr_s,
                  self.dist_m, self.vig_s, self.rot, self.vert, self.horiz, self.mc_sh, self.mc_mid,
                  self.mc_hi, self.mc_sc, self.sh_s, self.l_hl, self.l_sh, self.l_wh, self.l_bk, self.l_smart,
                  self.c_sat, self.c_vib, self.lut_amt, self.g_bal, self.g_blend, *self.bw_sl, self.l_haze,
                  self.p_hi, self.p_li, self.p_da, self.p_sh, self.t_toe, self.t_shoulder, self.f_tex, self.f_grain,
                  self.f_gsize, self.f_grough):
            w.changed.connect(self.controls_changed)
        for w in (self.dn_on, self.band, self.mcb, self.ca, self.dfr, self.dist, self.vig, self.constrain, self.sh_on,
                  self.c_bw, self.f_gcol, self.dn_safe, self.hdr_on):
            w.toggled.connect(self.controls_changed)
        self.hdr_amt.changed.connect(self.controls_changed)
        for w in (self.hdr_strength, self.hdr_engine):
            w.currentIndexChanged.connect(self.controls_changed)
        for w in (self.t_toe, self.t_shoulder, self.con, self.l_wh):
            w.changed.connect(lambda *_: self._update_filmic())
        self.c_bw.toggled.connect(lambda _: self._bw_visibility())
        self.geq.toggled.connect(self.controls_changed)
        for w in (self.dn_m, self.tone, self.wb, self.dem, self.hl_mode):
            w.currentIndexChanged.connect(self.controls_changed)
        self.hl_mode.currentIndexChanged.connect(self._hl_mode_changed)
        self._remember_groups(body)
        self._refresh_range_ui()
        self._refresh_hsl_ui()
        self.apply_level()
        return panel

    FOLDED = ("Tone curve", "Colour wheel", "HSL", "Colour grading")      # folded until you open them

    def _remember_groups(self, body):
        state = self.settings.get("groups_open", {})
        for g in body.findChildren(Group):
            g.btn.setChecked(bool(state.get(g.title, g.title not in self.FOLDED)))
            g.body.setVisible(g.btn.isChecked())
            g.btn.toggled.connect(lambda on, t=g.title: self._group_toggled(t, on))

    def _group_toggled(self, title, on):
        state = dict(self.settings.get("groups_open", {}))
        state[title] = on
        self.settings.set("groups_open", state)

    def _bw_visibility(self):
        self.bw_box.setVisible(self.level.currentIndex() == 1 and self.c_bw.isChecked())

    def apply_level(self):
        full = self.level.currentIndex() == 1
        self.settings.set("level", "Full" if full else "Simple")
        for w in self.full_only:
            w.setVisible(full)
        self._bw_visibility()

    # ------------------------------------------------------------------------------------ edit <-> controls
    def edit_from_controls(self):
        from ..engine import Edit
        e = Edit.from_dict(self.edit.to_dict())
        s = self.dn_s.value()
        e.denoise.enabled = self.dn_on.isChecked()
        e.denoise.method = self.dn_m.currentText()
        e.denoise.luma = round(self.dn_l.value() * s, 3)
        e.denoise.chroma = round(self.dn_c.value() * s, 3)
        e.denoise.grain = round(self.dn_g.value(), 3)
        e.denoise.safeguard = self.dn_safe.isChecked()
        e.banding = self.band.isChecked()
        e.exposure_ev = round(self.ev.value(), 3)
        e.tone.contrast = round(self.con.value(), 3)
        e.tone.mode = ["standard", "camera_match", "linear"][self.tone.currentIndex()]
        e.tone.match_camera_brightness = self.mcb.isChecked()
        e.white_balance.mode = ["as_shot", "daylight", "temp_tint", "multipliers"][self.wb.currentIndex()]
        if e.white_balance.mode == "multipliers" and not e.white_balance.multipliers:
            e.white_balance.mode = "as_shot"
        e.white_balance.temperature = round(self.temp.value())
        e.white_balance.tint = round(self.tint.value())
        e.demosaic = self.dem.currentData() or "rcd"
        e.green_eq = self.geq.isChecked()
        e.lens.auto_ca = self.ca.isChecked()
        e.lens.defringe = self.dfr.isChecked()
        e.lens.defringe_strength = round(self.dfr_s.value(), 3)
        e.lens.vignette = self.vig.isChecked() and e.lens.vignette_model is not None
        e.lens.vignette_amount = round(self.vig_s.value(), 3)
        g = e.geometry
        if g.distortion is not None:                      # with nothing measured the switch means nothing
            g.distortion_amount = 1.0 if self.dist.isChecked() else 0.0
        g.distortion_manual = round(self.dist_m.value(), 1)
        g.rotate = round(self.rot.value(), 2)
        g.vertical = round(self.vert.value(), 1)
        g.horizontal = round(self.horiz.value(), 1)
        g.constrain = self.constrain.isChecked()
        m = e.microcontrast
        m.amount = round(self.ds_rows["local"][1].value() / 100.0, 3)
        m.shadows, m.midtones, m.highlights = (round(w.value(), 3) for w in (self.mc_sh, self.mc_mid, self.mc_hi))
        m.scale = round(self.mc_sc.value(), 3)
        for k in ("tonal", "crisp", "separation", "veil", "broad"):
            setattr(e.detail, k, round(self.ds_rows[k][1].value(), 1))
        off = [k for k, (on, _) in self.ds_rows.items() if not on.isChecked()]
        e.detail.off = off or None
        e.sharpen.enabled = self.sh_on.isChecked()
        e.sharpen.amount = round(self.sh_s.value(), 3)
        from ..engine.color import grade as G
        L = e.light
        L.highlights, L.shadows, L.whites, L.blacks, L.smart = (round(w.value(), 1) for w in (
            self.l_hl, self.l_sh, self.l_wh, self.l_bk, self.l_smart))
        L.curve = self.curve.value()
        L.dehaze = round(self.l_haze.value(), 1)
        e.hdr.enabled = self.hdr_on.isChecked()
        e.hdr.strength = self.hdr_strength.currentData() or "medium"
        e.hdr.amount = round(self.hdr_amt.value(), 1)
        e.hdr.engine = self.hdr_engine.currentData() or "fusion"
        prm = {k: round(w.value(), 1) for k, w in (("highlights", self.p_hi), ("lights", self.p_li),
                                                   ("darks", self.p_da), ("shadows", self.p_sh)) if w.value()}
        L.param = prm or None
        e.tone.toe = round(self.t_toe.value(), 1)
        e.tone.shoulder = round(self.t_shoulder.value(), 3)
        e.highlight_mode = ["clip", "rebuild", "glow"][self.hl_mode.currentIndex()] if e.highlight_mode in (
            "clip", "rebuild", "glow") else e.highlight_mode
        gp = self.glow_combo.currentData()
        e.glow_profile = gp if (gp and gp != "__measure__") else ""
        e.profile = self.prof_combo.currentData() or ""
        if e.profile == "__add__":
            e.profile = self.edit.profile
        fi = e.finish
        fi.texture = round(self.f_tex.value(), 1)
        fi.grain = round(self.f_grain.value(), 1)
        fi.grain_size = round(self.f_gsize.value(), 3)
        fi.grain_roughness = round(self.f_grough.value(), 1)
        fi.grain_colour = self.f_gcol.isChecked()
        C = e.colour
        C.saturation = round(self.c_sat.value(), 1)
        C.vibrance = round(self.c_vib.value(), 1)
        C.bw = self.c_bw.isChecked()
        C.bw_mix = {n: round(sl.value(), 1) for n, sl in zip(G.BANDS, self.bw_sl) if abs(sl.value()) > 1e-6} or None
        C.hsl = {k: [round(float(x), 1) for x in v] for k, v in self._hsl.items()
                 if any(abs(float(x)) > 1e-6 for x in v)} or None
        C.ranges = [dict(r) for r in self._ranges] or None
        C.grading = self._grading_value()
        C.lut = self.lut_combo.currentData() or ""
        if C.lut == "__add__":
            C.lut = self.edit.colour.lut
        C.lut_amount = round(self.lut_amt.value(), 3)
        import copy as _copy
        e.masks = _copy.deepcopy(self._masks) or None
        e.retouch = _copy.deepcopy(self._spots) or None
        e.portrait = self._portrait_value()
        self.temp.setEnabled(self.wb.currentIndex() in (2, 3))
        self.tint.setEnabled(self.wb.currentIndex() in (2, 3))
        return e

    def _grading_value(self):
        g = {}
        for k, w in self.wheels.items():
            lum = round(self.wheel_l[k].value(), 1)
            if w.strength >= 0.05 or abs(lum) > 1e-6:
                g[k] = [round(w.hue, 1), round(w.strength, 1), lum]
        bal, blend = round(self.g_bal.value(), 1), round(self.g_blend.value(), 1)
        if bal:
            g["balance"] = bal
        if blend != 50:
            g["blending"] = blend
        return g or None

    def controls_from_edit(self, e):
        ws = [self.dn_s, self.dn_l, self.dn_c, self.dn_g, self.ev, self.con, self.temp, self.tint, self.dn_on,
              self.band, self.mcb, self.dn_m, self.tone, self.wb, self.dem, self.ca, self.dfr, self.dfr_s,
              self.dist, self.dist_m, self.vig, self.vig_s, self.rot, self.vert, self.horiz, self.constrain,
              self.mc, self.mc_sh, self.mc_mid, self.mc_hi, self.mc_sc, self.sh_on, self.sh_s, self.lens_combo,
              self.aspect, self.c_bw, self.lut_combo, self.hl_mode, self.prof_combo, self.f_gcol, self.geq]
        for w in ws:
            w.blockSignals(True)
        self.dn_on.setChecked(e.denoise.enabled)
        self.dn_s.set_value(1.0, emit=False)
        self.dn_l.set_value(e.denoise.luma, emit=False)
        self.dn_c.set_value(e.denoise.chroma, emit=False)
        self.dn_g.set_value(e.denoise.grain, emit=False)
        self.dn_m.setCurrentText(e.denoise.method)
        self.dn_safe.blockSignals(True)
        self.dn_safe.setChecked(bool(e.denoise.safeguard))
        self.dn_safe.blockSignals(False)
        self._dn_method_text()
        self.band.setChecked(e.banding)
        self.ev.set_value(e.exposure_ev, emit=False)
        self.con.set_value(e.tone.contrast, emit=False)
        self.tone.setCurrentIndex({"standard": 0, "camera_match": 1, "linear": 2}.get(e.tone.mode, 0))
        self.mcb.setChecked(e.tone.match_camera_brightness)
        self.wb.setCurrentIndex({"as_shot": 0, "daylight": 1, "temp_tint": 2, "multipliers": 3}.get(
            e.white_balance.mode, 0))
        self.temp.set_value(e.white_balance.temperature, emit=False)
        self.tint.set_value(e.white_balance.tint, emit=False)
        i = self.dem.findData(e.demosaic)
        self.dem.setCurrentIndex(i if i >= 0 else self.dem.findData("rcd"))
        self.geq.setChecked(bool(getattr(e, "green_eq", False)))
        self.ca.setChecked(e.lens.auto_ca)
        self.dfr.setChecked(e.lens.defringe)
        self.dfr_s.set_value(e.lens.defringe_strength, emit=False)
        g = e.geometry
        self.dist.setEnabled(g.distortion is not None)
        self.dist.setChecked(g.distortion is not None and g.distortion_amount > 0)
        self.dist_m.set_value(g.distortion_manual, emit=False)
        self.vig.setEnabled(e.lens.vignette_model is not None)
        self.vig.setChecked(e.lens.vignette and e.lens.vignette_model is not None)
        self.vig_s.set_value(e.lens.vignette_amount, emit=False)
        self.rot.set_value(g.rotate, emit=False)
        self.vert.set_value(g.vertical, emit=False)
        self.horiz.set_value(g.horizontal, emit=False)
        self.constrain.setChecked(g.constrain)
        self.aspect.setCurrentIndex(max(0, self.aspect.findText(
            {"original": "Original ratio", "free": "Free"}.get(g.aspect, g.aspect))))
        m = e.microcontrast
        off = set(e.detail.off or ())
        for k, (on, sl) in self.ds_rows.items():
            on.blockSignals(True)
            on.setChecked(k not in off)
            on.blockSignals(False)
            sl.set_value(m.amount * 100.0 if k == "local" else getattr(e.detail, k), emit=False)
        self._ds_sync_simple()
        self._ds_refresh(e)
        self.mc_sh.set_value(m.shadows, emit=False)
        self.mc_mid.set_value(m.midtones, emit=False)
        self.mc_hi.set_value(m.highlights, emit=False)
        self.mc_sc.set_value(m.scale, emit=False)
        self.sh_on.setChecked(e.sharpen.enabled)
        self.sh_s.set_value(e.sharpen.amount, emit=False)
        self._fill_lenses(e.lens.tag)
        from ..engine.color import grade as G
        L = e.light
        for w, v in ((self.l_hl, L.highlights), (self.l_sh, L.shadows), (self.l_wh, L.whites), (self.l_bk, L.blacks),
                     (self.l_smart, L.smart)):
            w.set_value(v, emit=False)
        self.curve.set_curves(L.curve)
        self.l_haze.set_value(L.dehaze, emit=False)
        for w in (self.hdr_on, self.hdr_strength, self.hdr_engine):
            w.blockSignals(True)
        self.hdr_on.setChecked(bool(e.hdr.enabled))
        self.hdr_strength.setCurrentIndex(max(0, self.hdr_strength.findData(e.hdr.strength)))
        self.hdr_engine.setCurrentIndex(max(0, self.hdr_engine.findData(e.hdr.engine)))
        for w in (self.hdr_on, self.hdr_strength, self.hdr_engine):
            w.blockSignals(False)
        self.hdr_amt.set_value(e.hdr.amount, emit=False)
        self._hdr_texts()
        prm = L.param or {}
        for w, k in ((self.p_hi, "highlights"), (self.p_li, "lights"), (self.p_da, "darks"), (self.p_sh, "shadows")):
            w.set_value(float(prm.get(k, 0)), emit=False)
        self.t_toe.set_value(e.tone.toe, emit=False)
        self.t_shoulder.set_value(e.tone.shoulder, emit=False)
        self.hl_mode.setCurrentIndex({"rebuild": 1, "glow": 2}.get(e.highlight_mode, 0))
        self._fill_glow(e.glow_profile)
        self._glow_visibility()
        self._fill_profiles(e.profile)
        fi = e.finish
        self.f_tex.set_value(fi.texture, emit=False)
        self.f_tex.setVisible(bool(fi.texture))             # only for edits made with the earlier Texture slider
        self.f_grain.set_value(fi.grain, emit=False)
        self.f_gsize.set_value(fi.grain_size, emit=False)
        self.f_grough.set_value(fi.grain_roughness, emit=False)
        self.f_gcol.setChecked(bool(fi.grain_colour))
        C = e.colour
        self.c_sat.set_value(C.saturation, emit=False)
        self.c_vib.set_value(C.vibrance, emit=False)
        self.c_bw.setChecked(bool(C.bw))
        for n, sl in zip(G.BANDS, self.bw_sl):
            sl.set_value(float((C.bw_mix or {}).get(n, 0)), emit=False)
        self._hsl = {k: [float(x) for x in (list(v) + [0, 0, 0])[:3]] for k, v in (C.hsl or {}).items()}
        self._ranges = [{**RANGE_DEFAULT, **{k: float(v) for k, v in r.items()}} for r in (C.ranges or [])]
        self._rsel = min(max(self._rsel, 0), len(self._ranges) - 1) if self._ranges else -1
        g = C.grading or {}
        for k, w in self.wheels.items():
            v = (list(g.get(k) or []) + [0, 0, 0])[:3]
            w.set_value(float(v[0]), float(v[1]))
            self.wheel_l[k].set_value(float(v[2]), emit=False)
        self.g_bal.set_value(float(g.get("balance", 0)), emit=False)
        self.g_blend.set_value(float(g.get("blending", 50)), emit=False)
        self._fill_luts(C.lut)
        self.lut_amt.set_value(C.lut_amount, emit=False)
        import copy as _copy
        self._masks = _copy.deepcopy(e.masks or [])
        self._msel = min(max(self._msel, 0), len(self._masks) - 1) if self._masks else -1
        self._spots = _copy.deepcopy(e.retouch or [])
        self._ssel = self._ssel if 0 <= self._ssel < len(self._spots) else -1
        self._portrait_from_edit(e)
        for w in ws:
            w.blockSignals(False)
        self._refresh_range_ui()
        self._refresh_hsl_ui()
        self._refresh_masks()
        self._refresh_spots()
        self._bw_visibility()
        self._update_filmic()
        self.temp.setEnabled(self.wb.currentIndex() in (2, 3))
        self.tint.setEnabled(self.wb.currentIndex() in (2, 3))
        self._wb_chip(e)
        self._dist_chip(e)

    def controls_changed(self, *_):
        if self.image is None:
            return
        if self._push(self.edit_from_controls()):
            self._ds_refresh(self.edit)
            self.schedule()

    def _hdr_texts(self):
        from ..engine.color import hdr as HD
        st = HD.STRENGTHS.get(self.hdr_strength.currentData() or "medium")
        en = HD.ENGINES.get(self.hdr_engine.currentData() or "fusion")
        self.hdr_text.setText(st["does"] if st else "")
        self.hdr_engine_text.setText(f"Best for: {en['best']} Weak at: {en['weak']}" if en else "")

    def _hdr_refresh(self):
        """The Light panel's Auto HDR report: what the photo was measured to need and what was done."""
        if self.image is None or not self.edit.hdr.enabled:
            self.hdr_chip.setText("Auto HDR: off")
            return
        from .. import engine as E
        try:
            r = E.hdr_report(self.image, self.shown_edit())
        except Exception:                                   # a report must never stop the editor
            slog.log_exception("auto hdr report failed", log)
            r = None
        if not r:
            self.hdr_chip.setText("Auto HDR: -")
            return
        txt = (f"Scene range {r['range']:.1f} stops. Shadows opened up to {r['lifted']:.1f}, highlights held "
               f"up to {r['pulled']:.1f}.")
        if r.get("fits"):
            txt = f"Scene range {r['range']:.1f} stops: it already fits, so only a little was done."
        if r.get("noise_limited", 0) > 0.15:
            txt += f" Lift held back {r['noise_limited'] * 100:.0f}% where the shadows are mostly noise" + (
                "." if self.edit.denoise.enabled else " (Clean up noise first to open them further).")
        self.hdr_chip.setText(txt)

    def _dn_method_text(self):
        from ..engine.raw.denoise import METHOD_INFO
        i = METHOD_INFO.get(self.dn_m.currentText())
        self.dn_info.setText(f"{i['name']}. Best for: {i['best']} Weak at: {i['weak']} Speed: {i['speed']}" if i
                             else "")

    def _dn_safe_refresh(self):
        r = getattr(self.image, "denoise_report", None) if self.image is not None else None
        on = self.image is not None and self.edit.denoise.enabled and self.edit.denoise.safeguard
        if not on or not r:
            self.dn_safe_chip.setText("Protected: -")
            return
        self.dn_safe_chip.setText(f"Protected: {r['protected']:,} photosites kept as shot "
                                  f"({r['protected'] / max(r['sites'], 1) * 1e6:.0f} per million)")

    # ------------------------------------------------------------------------------------ detail by scale
    def _ds_sync_simple(self):
        """Show the Amount and Balance that the Tonal and Crisp method sliders add up to."""
        from ..engine.detail import scales as DS
        a, t = DS.to_simple(self.ds_rows["tonal"][1].value() / 100.0, self.ds_rows["crisp"][1].value() / 100.0)
        self.ds_amount.set_value(a * 100.0, emit=False)
        if a > 1e-6:
            self.ds_balance.set_value(t * 100.0, emit=False)

    def _ds_simple_moved(self, *_):
        """Amount / Balance are a saved combination of the Tonal and Crisp sliders (plan 23.11)."""
        from ..engine.detail import scales as DS
        m = DS.from_simple(self.ds_amount.value() / 100.0, self.ds_balance.value() / 100.0)
        self.ds_rows["tonal"][1].set_value(m["tonal"] * 100.0, emit=False)
        self.ds_rows["crisp"][1].set_value(m["crisp"] * 100.0, emit=False)
        self.ds_preset.setCurrentIndex(0)
        self.controls_changed()

    def _ds_method_moved(self, key):
        if key in ("tonal", "crisp"):
            self._ds_sync_simple()
        self.ds_preset.setCurrentIndex(0)
        self.controls_changed()

    def _ds_preset_picked(self, index):
        from ..engine.detail import scales as DS
        name = self.ds_preset.itemText(index)
        if name not in DS.PRESETS:
            return
        vals = DS.PRESETS[name]
        for k, (on, sl) in self.ds_rows.items():
            on.blockSignals(True)
            on.setChecked(True)
            on.blockSignals(False)
            sl.set_value(float(vals.get(k, 0.0)), emit=False)
        self._ds_sync_simple()
        self.controls_changed()
        self.ds_preset.setCurrentIndex(index)

    def _ds_bw_toggled(self, on):
        self.judge_bw = bool(on)
        if self.image is not None:
            self.render_now()

    def _ds_recover_diffraction(self):
        self.sh_on.setChecked(True)
        self.status.emit("Recover lens sharpness is on: it undoes the blur measured in this photo, diffraction "
                         "included. Judge it at 100%.")

    def _ds_refresh(self, e):
        """The meter, the diffraction note and (in the background) the balance number for edit ``e``."""
        from ..engine.detail import scales as DS
        from ..engine.pipeline import pipeline as PL
        sp = PL._mc_params(e)
        self.ds_meter.set_data(DS.meter(sp))
        limited = [n for n, b in DS.meter(sp).items() if b["limited"]]
        note = ""
        if limited:
            note = f"Limited in the {' and '.join(limited)} band to protect edges and noise."
        fnum = getattr(self.image.meta, "fnumber", None) if self.image is not None else None
        show_diff = False
        if fnum and fnum >= 8 and (sp.crisp > 0 or sp.tonal > 0):
            if fnum >= 16:
                note += (" " if note else "") + (f"At f/{fnum:g} diffraction removed the finest detail before the "
                                                  "sensor saw it: Crisp cannot bring back what was never recorded.")
            else:
                note += (" " if note else "") + (f"At f/{fnum:g} diffraction has softened every scale a little. "
                                                  "Recover it first, then add detail.")
            show_diff = not e.sharpen.enabled
        self.ds_note.setText(note)
        self.ds_note.setVisible(bool(note))
        self.ds_diff.setVisible(show_diff)
        if not sp.active or self.image is None:
            self.ds_ratio.setText("Balance (medium : fine): -")
            return
        self.ds_ratio_timer.start()

    def _ds_measure_balance(self):
        if self.image is None or not self.isVisible():      # nobody is looking: no background render
            return
        from .. import engine as E
        img, edit = self.image, self.shown_edit()
        self._ds_tok = getattr(self, "_ds_tok", 0) + 1
        run_bg(lambda: E.detail_balance(img, edit), self._ds_tok, self._ds_balance_ready, lambda *_: None)

    def _ds_balance_ready(self, tok, res):
        if tok != getattr(self, "_ds_tok", 0) or res is None:
            return
        before, after = res
        if before <= 0:
            self.ds_ratio.setText("Balance (medium : fine): -")
            return
        word = "more modelled" if after > before * 1.02 else ("more etched" if after < before * 0.98 else "unchanged")
        self.ds_ratio.setText(f"Balance (medium : fine): {before:.2f} -> {after:.2f}  ({word})")

    def schedule(self):
        self.timer.start()
        self.save_timer.start()

    def commit(self, new, message=None, label=None):
        """Make ``new`` the current edit (a history step called ``label``), refresh the controls and re-render."""
        if message:
            self.status.emit(message)
        if not self._push(new, label):
            return
        self.controls_from_edit(self.edit)
        self._canonical()
        self.schedule()

    def _canonical(self):
        """Store the edit as the controls express it (same render), so the next slider move is not mistaken
        for a change of some setting written differently by a look or an older sidecar."""
        canon = self.edit_from_controls()
        if canon.to_dict() != self.edit.to_dict():
            self.edit = canon
            self.steps[self.hpos] = (self.steps[self.hpos][0], canon)

    # ------------------------------------------------------------------------------------ history
    def _reset_history(self, edit, steps=None):
        """Start the history for a photo (or version): the saved steps if they end where the sidecar does."""
        if steps and steps[-1][1].to_dict() == edit.to_dict():
            self.steps = list(steps)
        else:
            self.steps = [("Opened", edit)]
        self.hpos = len(self.steps) - 1
        self.edit = self.steps[self.hpos][1]
        self._last_push = (0.0, None)
        self._refresh_history()

    def _push(self, new, label=None) -> bool:
        """Make ``new`` the current edit as a history step. Moving the same slider again within two seconds
        updates that step instead of adding one, so the list reads 'Exposure +0.5', not fifty nudges."""
        from ..engine.presets.looks import describe_change
        if new.to_dict() == self.edit.to_dict():
            return False
        del self.steps[self.hpos + 1:]
        keys = changed_keys(self.edit, new)
        now = time.monotonic()
        t0, k0 = self._last_push
        if label is None and self.hpos > 0 and k0 == keys and now - t0 < 2.0:
            prev = self.steps[self.hpos - 1][1]
            if new.to_dict() == prev.to_dict():           # dragged back to where it started: no step at all
                self.steps.pop()
                self.hpos -= 1
            else:
                self.steps[self.hpos] = (describe_change(prev, new), new)
        else:
            self.steps.append((label or describe_change(self.edit, new), new))
            self.hpos += 1
        if len(self.steps) > 300:
            del self.steps[:len(self.steps) - 300]
            self.hpos = len(self.steps) - 1
        self._last_push = (now, keys if label is None else None)
        self.edit = self.steps[self.hpos][1]
        self._refresh_history()
        return True

    def _refresh_history(self):
        lw = self.hist_list
        if lw.count() != len(self.steps):
            lw.clear()
            for lb, _ in self.steps:
                lw.addItem(lb)
        for i, (lb, _) in enumerate(self.steps):
            it = lw.item(i)
            it.setText(("▸ " if i == self.hpos else "   ") + lb)
            it.setForeground(QColor(T.TEXT if i <= self.hpos else T.DIM))
        lw.scrollToItem(lw.item(self.hpos))
        self.undo_b.setEnabled(self.hpos > 0)
        self.redo_b.setEnabled(self.hpos < len(self.steps) - 1)

    def jump_history(self, i):
        if not 0 <= i < len(self.steps) or i == self.hpos:
            return
        self.hpos = i
        self.edit = self.steps[i][1]
        self._last_push = (0.0, None)
        self.controls_from_edit(self.edit)
        self._refresh_history()
        self.schedule()

    # ------------------------------------------------------------------------------------ light
    def auto_light(self):
        if self.image is None:
            return
        self.auto_light_b.setEnabled(False)
        img, e0, path = self.image, self.edit_from_controls(), self.path

        def work():
            from ..engine.pipeline.pipeline import auto_light
            return auto_light(img, e0)
        run_bg(work, path, self._auto_lit, lambda _, err: (self.auto_light_b.setEnabled(True),
                                                             self.status.emit(f"Auto light problem: {err}")))

    def _auto_lit(self, path, res):
        self.auto_light_b.setEnabled(True)
        if path != self.path:
            return
        e = self.edit_from_controls()
        for k, v in res.items():
            setattr(e.light, k, round(float(v), 1))
        self.commit(e, f"Auto light: highlights {e.light.highlights:+.0f}, whites {e.light.whites:+.0f}, blacks "
                       f"{e.light.blacks:+.0f}. Exposure and shadows untouched.", label="Auto light")

    def reset_light(self):
        if self.image is None:
            return
        e = self.edit_from_controls()
        e.light = type(e.light)()
        self.commit(e, "Highlights, shadows, whites, blacks, Smart lighting and the curve reset.", label="Reset light")

    def _curve_channel(self, i):
        self.curve.set_channel(CurveEditor.CHANNELS[i])
        self.curve.set_histogram(self.last_a8)

    # ------------------------------------------------------------------------------------ white balance
    def _wb_mode_picked(self, idx):
        if idx == 3 and not self.edit.white_balance.multipliers:
            self.wb.blockSignals(True)
            self.wb.setCurrentIndex({"as_shot": 0, "daylight": 1, "temp_tint": 2}.get(self.edit.white_balance.mode, 0))
            self.wb.blockSignals(False)
            self.pick_wb.setChecked(True)

    def _wb_slider_moved(self, *_):
        """Moving Temperature or Tint after a pick (or Auto) carries on from there as a Custom white balance."""
        if self.wb.currentIndex() == 3:
            self.wb.blockSignals(True)
            self.wb.setCurrentIndex(2)
            self.wb.blockSignals(False)

    def _wb_chip(self, e):
        wb = e.white_balance
        if wb.mode == "multipliers" and wb.multipliers:
            txt, col = f"White balance: picked / auto, about {wb.temperature:.0f} K, tint {wb.tint:+.0f}", T.AMBER
        elif wb.mode == "temp_tint":
            txt, col = f"White balance: custom, {wb.temperature:.0f} K, tint {wb.tint:+.0f}", T.DIM
        elif wb.mode == "daylight":
            txt, col = "White balance: the camera's daylight setting", T.DIM
        else:
            txt, col = "White balance: as shot (the camera's choice)", T.DIM
        self.wb_chip.setText(txt)
        self.wb_chip.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")

    def _set_pick(self, kind):
        self.pick = kind
        for b, k in ((self.pick_wb, "wb"), (self.pick_rng, "range")):
            b.blockSignals(True)
            b.setChecked(kind == k)
            b.blockSignals(False)
        self.viewer.setCursor(Qt.CursorShape.CrossCursor if kind else Qt.CursorShape.ArrowCursor)
        self.viewer.picking = bool(kind)
        if kind == "wb":
            self.status.emit("Click something in the photo that should be neutral grey or white (Esc cancels).")
        elif kind == "range":
            self.status.emit("Click a colour in the photo to choose it on the colour wheel (Esc cancels).")

    def _pick_white_at(self, u, v):
        self._set_pick(None)
        img, e0, path = self.image, self.edit_from_controls(), self.path

        def work():
            from ..engine.pipeline.pipeline import pick_white, wb_temp_tint
            m = pick_white(img, e0, u, v)
            return m, (wb_temp_tint(img, m) if m is not None else None)
        run_bg(work, path, lambda p, r: self._wb_found(p, r, "Pick white", "the spot you clicked"),
               lambda _, err: self.status.emit(f"Pick white problem: {err}"))

    def auto_wb(self, mood=None):
        """Auto white balance. mood None = the default for this photo; else 0..1 (1 = fully neutral)."""
        if self.image is None:
            return
        img, path = self.image, self.path

        method = self.wb_method.currentData() or "grey_edge"

        def work():
            from ..engine.pipeline.pipeline import auto_white_balance, wb_temp_tint
            m, how, info = auto_white_balance(img, method, mood)
            return m, (wb_temp_tint(img, m) if m is not None else None), how, info
        run_bg(work, path, lambda p, r: self._auto_wb_found(p, r),
               lambda _, err: self.status.emit(f"Auto white balance problem: {err}"))

    def _auto_wb_found(self, path, r):
        if path != self.path:
            return
        m, tt, how, info = r
        if m is not None and info:
            self._wb_auto_used = True
            self.wb_mood.set_value(round(info["mood"] * 100.0), emit=False)
            self.wb_chip.setText(self.wb_report_text(info))
        self._wb_found(path, (m, tt), "Auto white balance", how)

    @staticmethod
    def wb_report_text(info: dict) -> str:
        """Both estimates, how well they agree, and how far the light was corrected (plan 23.9)."""
        parts = []
        if "grey_edge_k" in info:
            parts.append(f"grey edge {info['grey_edge_k']} K")
        if "grayness_k" in info:
            parts.append(f"grayness index {info['grayness_k']} K")
        txt = "Light: " + ", ".join(parts) if parts else "Light: one estimate only"
        if info.get("agreement_deg") is not None:
            word = {"high": "they agree: confident", "medium": "they roughly agree",
                    "low": "they disagree: check by eye"}[info["confidence"]]
            txt += f" ({word})."
        if info.get("lights", 1) > 1:
            txt += " Two light sources found (about " + " and ".join(f"{k} K" for k in info["lights_k"]) + ")."
        if not info.get("in_band", True):
            txt += " The estimate was far from any real light and was pulled back."
        txt += f" Corrected {info['mood'] * 100:.0f}% of the way" + (" to keep the mood." if info["mood"] < 0.99
                                                                    else ".")
        return txt

    def _wb_found(self, path, res, what, how):
        if path != self.path:
            return
        m, tt = res
        if m is None:
            self.status.emit(f"{what}: nothing reliable to measure there ({how}); white balance left as it was.")
            return
        e = self.edit_from_controls()
        e.white_balance.mode = "multipliers"
        e.white_balance.multipliers = [round(float(x), 5) for x in m]
        e.white_balance.temperature, e.white_balance.tint = round(tt[0]), round(tt[1])
        self.commit(e, f"{what} ({how}): about {tt[0]:.0f} K, tint {tt[1]:+.0f}. Move Temperature or Tint to "
                       f"fine-tune.", label=what)

    # ------------------------------------------------------------------------------------ colour wheel
    def new_range(self, hue=None):
        r = dict(RANGE_DEFAULT)
        if hue is not None:
            r["hue"] = round(float(hue), 1)
        self._ranges.append(r)
        self._rsel = len(self._ranges) - 1
        self._refresh_range_ui()
        self.controls_changed()

    def remove_range(self):
        if 0 <= self._rsel < len(self._ranges):
            self._ranges.pop(self._rsel)
            self._rsel = min(self._rsel, len(self._ranges) - 1)
            self._refresh_range_ui()
            self.controls_changed()

    def _range_selected(self, i):
        if 0 <= i < len(self._ranges):
            self._rsel = i
            self._refresh_range_ui()

    def _ring_dragged(self):
        self._refresh_range_sliders()
        self.controls_changed()

    def _range_sliders(self, *_):
        if not 0 <= self._rsel < len(self._ranges):
            return
        r = self._ranges[self._rsel]
        r["shift"], r["sat"], r["lum"], r["width"], r["falloff"] = (round(w.value(), 1) for w in (
            self.r_shift, self.r_sat, self.r_lum, self.r_width, self.r_fall))
        self.ring.set_ranges(self._ranges, self._rsel)
        self.controls_changed()

    @staticmethod
    def _hue_name(h):
        from ..engine.color import grade as G
        d = [abs(float((h - c + 180) % 360 - 180)) for c in G.BAND_HUES]
        return G.BANDS[int(np.argmin(d))]

    def _refresh_range_ui(self):
        self.rng_combo.blockSignals(True)
        self.rng_combo.clear()
        for i, r in enumerate(self._ranges):
            self.rng_combo.addItem(dot_icon(oklch_colour(0.72, 0.12, r["hue"]).name()),
                                   f"Range {i + 1}: {self._hue_name(r['hue'])}")
        if not self._ranges:
            self.rng_combo.addItem("No colour range yet: press + or ◉")
        self.rng_combo.setCurrentIndex(max(0, self._rsel))
        self.rng_combo.blockSignals(False)
        self.ring.set_ranges(self._ranges, self._rsel)
        self._refresh_range_sliders()

    def _refresh_range_sliders(self):
        has = 0 <= self._rsel < len(self._ranges)
        r = self._ranges[self._rsel] if has else RANGE_DEFAULT
        for w, k in ((self.r_shift, "shift"), (self.r_sat, "sat"), (self.r_lum, "lum"), (self.r_width, "width"),
                     (self.r_fall, "falloff")):
            w.setEnabled(has)
            w.set_value(r[k], emit=False)
        if has:
            self.rng_combo.setItemIcon(self._rsel, dot_icon(oklch_colour(0.72, 0.12, r["hue"]).name()))
            self.rng_combo.setItemText(self._rsel, f"Range {self._rsel + 1}: {self._hue_name(r['hue'])}")

    def _pick_range_at(self, u, v):
        self._set_pick(None)
        a8 = self.last_a8
        if a8 is None:
            return
        h, w = a8.shape[:2]
        x, y = min(w - 1, int(u * w)), min(h - 1, int(v * h))
        hue, chroma = srgb8_hue(a8[max(0, y - 3):y + 4, max(0, x - 3):x + 4])
        if chroma < 0.02:
            self.status.emit("That spot is almost grey: pick something more colourful for the colour wheel.")
            return
        if 0 <= self._rsel < len(self._ranges):
            self._ranges[self._rsel]["hue"] = round(hue, 1)
            self._refresh_range_ui()
            self.controls_changed()
        else:
            self.new_range(hue)
        self.status.emit(f"Colour wheel: range set to {self._hue_name(hue)} ({hue:.0f}°). Now move Hue shift, "
                         f"Saturation or Lightness.")

    # ------------------------------------------------------------------------------------ HSL
    def _hsl_sliders(self, *_):
        from ..engine.color import grade as G
        band = G.BANDS[self.sw.current()]
        self._hsl[band] = [round(w.value(), 1) for w in (self.hsl_h, self.hsl_s, self.hsl_l)]
        self._mark_bands()
        self.controls_changed()

    def _refresh_hsl_ui(self):
        from ..engine.color import grade as G
        v = self._hsl.get(G.BANDS[self.sw.current()], [0.0, 0.0, 0.0])
        for w, x in zip((self.hsl_h, self.hsl_s, self.hsl_l), v):
            w.set_value(float(x), emit=False)
        self._mark_bands()

    def _mark_bands(self):
        from ..engine.color import grade as G
        self.sw.set_marked([any(abs(float(x)) > 1e-6 for x in self._hsl.get(n, [0, 0, 0])) for n in G.BANDS])

    # ------------------------------------------------------------------------------------ LUTs
    def _fill_luts(self, current=None):
        from ..engine.color import grade as G
        if current is None:
            current = self.lut_combo.currentData() or ""
        self.lut_combo.blockSignals(True)
        self.lut_combo.clear()
        self.lut_combo.addItem("LUT: none", "")
        for n in G.list_luts():
            self.lut_combo.addItem("LUT: " + os.path.splitext(n)[0], n)
        if current and self.lut_combo.findData(current) < 0:
            self.lut_combo.addItem(f"LUT: {current} (file missing)", current)
        self.lut_combo.addItem("Add a .cube file...", "__add__")
        self.lut_combo.setCurrentIndex(max(0, self.lut_combo.findData(current)))
        self.lut_combo.blockSignals(False)

    def _lut_picked(self, idx):
        if self.lut_combo.itemData(idx) == "__add__":
            import shutil
            from ..engine.color import grade as G
            f, _ = QFileDialog.getOpenFileName(self, "Choose a 3D LUT", "", "3D LUT (*.cube)")
            if not f:
                self._fill_luts(self.edit.colour.lut)
                return
            try:
                with open(f, encoding="utf-8", errors="replace") as fh:
                    G.parse_cube(fh.read())
                d = paths.get("SETTINGS") / "luts"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, d / os.path.basename(f))
            except (OSError, ValueError) as err:
                QMessageBox.warning(self, "LUT", f"That file can't be used: {err}")
                self._fill_luts(self.edit.colour.lut)
                return
            self._fill_luts(os.path.basename(f))
        self.controls_changed()

    # ------------------------------------------------------------------------------------ looks, copy / paste
    def _fill_looks(self, select=None):
        from ..engine.presets import looks as LK
        self.look_combo.clear()
        for lk in LK.list_looks():
            self.look_combo.addItem(dot_icon(T.VIOLET if lk["source"] == "shared" else T.CYAN), lk["name"], lk["path"])
            self.look_combo.setItemData(self.look_combo.count() - 1, lk["about"] or lk["name"],
                                        Qt.ItemDataRole.ToolTipRole)
        if select:
            self.look_combo.setCurrentIndex(max(0, self.look_combo.findText(select)))

    def apply_look(self):
        from ..engine.presets import looks as LK
        path = self.look_combo.currentData()
        if self.image is None or not path:
            return
        lk = LK.load_look(path)
        name = lk.get("name") or os.path.basename(path)
        self.commit(LK.apply_look(self.edit_from_controls(), lk),
                    f"Look applied: {name}. Exposure, white balance, crop and the lens tag are untouched.",
                    label=f"Look: {name}")

    def save_look(self):
        from ..engine.presets import looks as LK
        groups = [("light", "Light: contrast, highlights, shadows, whites, blacks, curve (never exposure)"),
                  ("white_balance", "White balance"),
                  ("colour", "Colour: saturation, vibrance, HSL, colour wheel, grading, B&W, LUT"),
                  ("detail", "Detail: microcontrast, sharpening")]
        dlg = GroupDialog(self, "Save look", "Your look is saved in settings\\looks on this PC and never uploaded. "
                          "Share the .psblook file if you want to.", groups, {"light", "colour", "detail"},
                          ask_name=True)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name = dlg.name.text().strip()
        p = LK.save_look(LK.look_from_edit(self.edit_from_controls(), name, dlg.chosen(), dlg.about.text().strip()))
        self._fill_looks(name)
        self.status.emit(f"Look saved: {p}")

    def copy_settings(self):
        if self.image is None:
            return
        from ..engine import Edit
        from ..engine.presets.looks import DEFAULT_COPY, GROUPS
        dlg = GroupDialog(self, "Copy settings", "Choose what to copy. Paste it onto another photo here, or onto "
                          "many photos at once in the Library (Paste settings).", [(k, v[0]) for k, v in GROUPS.items()],
                          set(self.settings.get("copy_groups", DEFAULT_COPY)))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        CLIP.update(edit=Edit.from_dict(self.edit_from_controls().to_dict()), groups=dlg.chosen(),
                    source=os.path.basename(self.path))
        self.settings.set("copy_groups", dlg.chosen())
        self.status.emit(f"Copied {len(dlg.chosen())} group(s) of settings from {CLIP['source']}.")

    def paste_settings(self):
        if self.image is None:
            return
        if CLIP["edit"] is None:
            self.status.emit("Nothing copied yet: press Copy settings on the photo you want to copy from.")
            return
        from ..engine.presets.looks import copy_groups
        self.commit(copy_groups(CLIP["edit"], self.edit_from_controls(), CLIP["groups"]),
                    f"Pasted settings from {CLIP['source']}.", label="Paste settings")

    # ------------------------------------------------------------------------------------ versions, snapshots
    def _fill_versions(self):
        from .. import engine as E
        self.ver.blockSignals(True)
        self.ver.clear()
        try:
            names = E.list_versions(self.path) if self.path else ["Master"]
        except (OSError, ValueError):
            names = ["Master"]
        for i, n in enumerate(names):
            self.ver.addItem(n, i)
        if self.path:
            self.ver.addItem("New virtual copy", "new")
            if self.version > 0:
                self.ver.addItem("Delete this copy", "del")
        self.ver.setCurrentIndex(min(self.version, len(names) - 1))
        self.ver.blockSignals(False)

    def _version_picked(self, idx):
        from .. import engine as E
        if self.image is None:
            self._fill_versions()
            return
        data = self.ver.itemData(idx)
        self.save_sidecar()
        if data == "new":
            name = E.list_versions(self.path)[self.version]
            i = E.add_version(self.path, self.edit)
            self._switch_version(i, f"Virtual copy made from {name}. Edit it freely: {name} is untouched.")
        elif data == "del":
            name = E.list_versions(self.path)[self.version]
            if QMessageBox.question(self, "Delete copy", f"Delete '{name}'? The master and other copies stay.") \
                    != QMessageBox.StandardButton.Yes:
                self._fill_versions()
                return
            E.delete_version(self.path, self.version)
            self._switch_version(0, f"'{name}' deleted.")
        elif data is not None and int(data) != self.version:
            self._switch_version(int(data))

    def _switch_version(self, i, msg=None):
        from .. import engine as E
        self.version = i
        self._reset_history(E.load_version(self.path, i), E.load_history(self.path, i))
        self.controls_from_edit(self.edit)
        self._canonical()
        self._fill_versions()
        self.render_now()
        self.status.emit(msg or f"Showing {E.list_versions(self.path)[i]}.")

    def _fill_snapshots(self):
        from .. import engine as E
        self.snap_list.clear()
        if not self.path:
            return
        for sn in E.list_snapshots(self.path):
            self.snap_list.addItem(f"{sn.get('name')}    {sn.get('time', '')}")

    def take_snapshot(self):
        if self.image is None:
            return
        from .. import engine as E
        n = len(E.list_snapshots(self.path))
        name, ok = QInputDialog.getText(self, "Snapshot", "A name for this moment:", text=f"Snapshot {n + 1}")
        if not ok:
            return
        self.save_sidecar()
        E.add_snapshot(self.path, name.strip() or f"Snapshot {n + 1}", self.edit)
        self._fill_snapshots()
        self.status.emit("Snapshot kept in the sidecar.")

    def restore_snapshot(self):
        from .. import engine as E
        i = self.snap_list.currentRow()
        if self.image is None or i < 0:
            return
        sn = E.list_snapshots(self.path)[i]
        e = E.Edit.from_dict(sn.get("edit", {}))
        e.lens.tag = self.edit.lens.tag                     # which lens took it is a fact, not an edit
        self.commit(e, f"Snapshot '{sn.get('name')}' restored (Undo brings back what you had).",
                    label=f"Snapshot: {sn.get('name')}")

    def delete_snapshot(self):
        from .. import engine as E
        i = self.snap_list.currentRow()
        if self.image is None or i < 0:
            return
        E.delete_snapshot(self.path, i)
        self._fill_snapshots()

    # ------------------------------------------------------------------------------------ Phase 6 extras
    def _update_filmic(self):
        from ..engine.color.light import white_shift
        sh = self.t_shoulder.value()
        self.filmic.set_params(contrast=self.con.value(), shoulder=sh, toe=self.t_toe.value(),
                               white=max(self.edit.tone.white + white_shift(self.l_wh.value()), sh + 0.3))

    def auto_colour(self):
        if self.image is None:
            return
        self.auto_col_b.setEnabled(False)
        img, e0, path = self.image, self.edit_from_controls(), self.path

        def work():
            from ..engine.pipeline.pipeline import auto_colour
            return auto_colour(img, e0)
        run_bg(work, path, self._auto_coloured, lambda _, err: (self.auto_col_b.setEnabled(True),
                                                                  self.status.emit(f"Auto colour problem: {err}")))

    def _auto_coloured(self, path, res):
        self.auto_col_b.setEnabled(True)
        if path != self.path:
            return
        if res["vibrance"] <= 0:
            self.status.emit(f"Auto colour: the photo is already colourful enough (colourfulness {res['colourfulness']:.0f}"
                             f"); nothing changed.")
            return
        e = self.edit_from_controls()
        e.colour.vibrance = res["vibrance"]
        self.commit(e, f"Auto colour: colourfulness {res['colourfulness']:.0f}, vibrance set to "
                       f"{res['vibrance']:+.0f}. White balance and brightness untouched.", label="Auto colour")

    def _bw_filter_picked(self, idx):
        from ..engine.color import grade as G
        if idx <= 0:
            return
        mix = G.BW_FILTERS[self.bw_filter.itemText(idx)]
        for n, sl in zip(G.BANDS, self.bw_sl):
            sl.set_value(float(mix.get(n, 0)), emit=False)
        if not self.c_bw.isChecked():
            self.c_bw.setChecked(True)
        self.controls_changed()
        self.status.emit(f"Black & white: {self.bw_filter.itemText(idx).lower()} (the mixer below shows what it did).")

    def _fill_profiles(self, current=None):
        from ..engine.color import dcp
        if current is None:
            current = self.prof_combo.currentData() or ""
        cam = getattr(self.image.meta, "model", None) if self.image is not None else None
        self.prof_combo.blockSignals(True)
        self.prof_combo.clear()
        self.prof_combo.addItem("Profile: Studio (camera matrix)", "")
        try:
            profs = dcp.list_profiles(cam) if cam else []
        except OSError:
            profs = []
        for pr in profs:
            self.prof_combo.addItem(dot_icon(T.VIOLET if pr["source"] == "shared" else T.CYAN),
                                    f"Profile: {pr['name']}", pr["file"])
        if current and self.prof_combo.findData(current) < 0:
            self.prof_combo.addItem(f"Profile: {current} (not for this camera, or missing)", current)
        self.prof_combo.addItem("Add a camera profile (.dcp)...", "__add__")
        self.prof_combo.setCurrentIndex(max(0, self.prof_combo.findData(current)))
        self.prof_combo.blockSignals(False)

    def _profile_picked(self, idx):
        if self.prof_combo.itemData(idx) == "__add__":
            from ..engine.color import dcp
            f, _ = QFileDialog.getOpenFileName(self, "Choose a DNG camera profile", "", "Camera profile (*.dcp)")
            if not f:
                self._fill_profiles(self.edit.profile)
                return
            try:
                name = dcp.add_profile(f)
                pr = dcp.load(name)
            except (OSError, ValueError) as err:
                QMessageBox.warning(self, "Camera profile", f"That file can't be used: {err}")
                self._fill_profiles(self.edit.profile)
                return
            cam = getattr(self.image.meta, "model", "") if self.image is not None else ""
            if cam and pr.camera.strip().lower() != cam.strip().lower():
                QMessageBox.information(self, "Camera profile", f"Added '{pr.name}'. It is made for {pr.camera}, "
                                        f"so it is offered only for photos from that camera.")
                self._fill_profiles(self.edit.profile)
                return
            self._fill_profiles(name)
        self.controls_changed()

    # ------------------------------------------------------------------------------------ masks
    KIND_NAMES = {"brush": "Brush", "linear": "Gradient", "radial": "Radial", "object": "Object", "colour": "Colour",
                  "skin": "Skin", "person": "Person",
                  "luminance": "Brightness"}

    def _cur_mask(self):
        return self._masks[self._msel] if 0 <= self._msel < len(self._masks) else None

    def _refresh_masks(self):
        lw = self.mask_list
        lw.blockSignals(True)
        lw.clear()
        for m in self._masks:
            it = QListWidgetItem(m.get("name", "Mask"))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable)
            it.setCheckState(Qt.CheckState.Checked if m.get("enabled", True) else Qt.CheckState.Unchecked)
            lw.addItem(it)
        if 0 <= self._msel < len(self._masks):
            lw.setCurrentRow(self._msel)
        lw.blockSignals(False)
        self._refresh_mask_controls()

    def _refresh_mask_controls(self):
        m = self._cur_mask()
        for w in (*self.m_sl.values(), self.mask_del, self.mask_inv, self.mask_add):
            w.setEnabled(m is not None)
        adj = (m or {}).get("adjust", {})
        for k, sl in self.m_sl.items():
            sl.set_value(float((m or {}).get("opacity", 100)) if k == "opacity" else float(adj.get(k, 0)), emit=False)
        self.mask_inv.blockSignals(True)
        self.mask_inv.setChecked(bool((m or {}).get("invert")))
        self.mask_inv.blockSignals(False)
        tool = self.viewer.tool
        self.mask_tool_box.setVisible(tool in ("brush", "object"))
        for w in (self.b_size, self.b_feather, self.b_flow, self.b_auto, self.b_erase):
            w.setVisible(tool == "brush")
        shapes = (m or {}).get("shapes", [])
        t = shapes[-1]["type"] if shapes else None
        groups = {"luminance": (self.r_lo, self.r_hi, self.r_soft), "colour": (self.c_width, self.c_soft, self.c_chroma),
                  "radial": (self.r_feather,)}
        shown = groups.get(t, ())
        for w in (self.r_lo, self.r_hi, self.r_soft, self.c_width, self.c_soft, self.c_chroma, self.r_feather):
            w.setVisible(w in shown)
        self.range_box.setVisible(bool(shown))
        persons = [x for x in shapes if x.get("type") == "person"]
        self.person_part_row.setVisible(bool(persons))
        if persons:
            k = self.person_part.findData(persons[-1].get("part", "person"))
            self.person_part.blockSignals(True)
            self.person_part.setCurrentIndex(max(k, 0))
            self.person_part.blockSignals(False)
        if shapes:
            sh = shapes[-1]
            vals = {self.r_lo: 100 * sh.get("lo", 0.6), self.r_hi: 100 * sh.get("hi", 1.0),
                    self.r_soft: 100 * sh.get("soft", 0.1), self.c_width: sh.get("width", 30),
                    self.c_soft: sh.get("soft", 20) if t == "colour" else 20, self.c_chroma: 100 * sh.get("chroma", 0.03),
                    self.r_feather: 100 * sh.get("feather", 0.5)}
            for w, v in vals.items():
                w.set_value(float(v), emit=False)

    def new_mask(self, kind):
        if self.image is None:
            return
        base = self.KIND_NAMES[kind]
        n = 1 + sum(1 for m in self._masks if str(m.get("name", "")).startswith(base))
        self._masks.append({"name": f"{base} {n}", "enabled": True, "invert": False, "opacity": 100, "shapes": [],
                            "adjust": {}})
        self._msel = len(self._masks) - 1
        self._refresh_masks()
        self._start_shape(kind, "add")

    def _start_shape(self, kind, op):
        if self._cur_mask() is None:
            return
        if kind == "person":
            self._find_person(op)
            return
        if kind == "skin":
            self._add_shape({"type": "skin", "op": op})
            self.mask_show.setChecked(True)
            self.status.emit("Skin selected (red). Now raise Smooth skin under 'Adjust inside the mask'. To keep it "
                             "to one face, pick 'x Brush (only where)' and paint over that face.")
            return
        if kind == "luminance":
            self._add_shape({"type": "luminance", "op": op, "lo": 0.6, "hi": 1.0, "soft": 0.1})
            self.mask_show.setChecked(True)
            self.status.emit("Brightness range: set From and To below (the red overlay shows what is selected), then "
                             "use the sliders under 'Adjust inside the mask'.")
            return
        self._mask_pending = (kind, op)
        self._set_mask_tool(kind)

    def _find_person(self, op):
        from .. import engine as E
        from ..engine.local import people as pp
        if self.image is None:
            return
        if not pp.available():
            self.status.emit("The people models are missing from models\\people. Run setup again (or copy that "
                             "folder back from the download).")
            return
        img, e, path, idx = self.image, self.edit_from_controls(), self.path, self._msel
        self.viewer.busy = True
        self.viewer.update()
        self.status.emit("Finding the person...")

        def done(p, shape):
            if p != self.path:
                return
            self.viewer.busy = False
            m = self._masks[idx] if 0 <= idx < len(self._masks) else None
            if m is None:
                return
            if shape is None:
                if not m["shapes"]:
                    self._masks.pop(idx)
                    self._msel = min(self._msel, len(self._masks) - 1)
                    self._refresh_masks()
                self.status.emit("No person found in this photo.")
                self.viewer.update()
                return
            shape["op"] = op
            self._msel = idx
            self._add_shape(shape)
            self.mask_show.setChecked(True)
            n = len(shape.get("people") or [])
            self.status.emit(f"Person selected ({n} found, red overlay). Pick a part in 'Part' (skin, face, hair, "
                             f"clothes, arms, legs, torso, background), then use the sliders below.")
        run_bg(lambda: E.find_people(img, e), path, done,
               lambda _, err: (setattr(self.viewer, "busy", False), self.status.emit(f"Person problem: {err}")))

    def _person_part_picked(self, idx):
        m = self._cur_mask()
        if m is None:
            return
        persons = [x for x in m.get("shapes", []) if x.get("type") == "person"]
        if not persons:
            return
        part = self.person_part.itemData(idx)
        persons[-1]["part"] = part
        from ..engine.local.people import PART_NAMES
        if str(m.get("name", "")).startswith("Person"):
            base = m["name"].split(":")[0]
            m["name"] = base if part == "person" else f"{base}: {PART_NAMES[part]}"
            self._refresh_masks()
        self.commit(self.edit_from_controls(), label=f"Person mask: {PART_NAMES[part].lower()}")
        self._update_overlay()

    def _set_mask_tool(self, kind):
        if self.image is None:
            kind = None
        if kind and self.viewer.tool == "spot":
            self._set_spot_tool(None)
        if kind and self.viewer.tool in ("eyes", "sculpt", "shape"):
            self._set_portrait_tool(None)
        if kind and self.viewer.crop_mode:
            self.crop_b.setChecked(False)
        self.viewer.tool = kind
        if kind:
            self.viewer.brush_frac = self.b_size.value() / 100.0
            self.viewer.brush_scale = self._brush_scale()
        else:
            self._mask_pending = None
        cur = {None: Qt.CursorShape.ArrowCursor, "brush": Qt.CursorShape.BlankCursor}.get(kind, Qt.CursorShape.CrossCursor)
        self.viewer.setCursor(cur)
        self._refresh_mask_controls()
        self._update_overlay()
        self.viewer.update()

    def _brush_scale(self):
        from .. import engine as E
        _, _, sw, sh = self.image.raw.crop
        ow, oh = E.output_size(self.image, self.edit)
        return max(sw, sh) / float(max(ow, oh))

    def _add_shape(self, shape):
        m = self._cur_mask()
        if m is None:
            return
        m["shapes"].append(shape)
        self.controls_changed()
        self._refresh_mask_controls()
        self._update_overlay()
        if not any(abs(float(v)) > 1e-6 for v in m["adjust"].values()):
            self.status.emit("Mask drawn. Now move the sliders under 'Adjust inside the mask' (red = selected).")

    def _s_points(self, uvs):
        from .. import engine as E
        u = np.array([p[0] for p in uvs], float)
        v = np.array([p[1] for p in uvs], float)
        xs, ys = E.view_to_s(self.image, self.edit, u, v)
        return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(np.atleast_1d(xs), np.atleast_1d(ys))]

    def _stroke_done(self, payload):
        from .. import engine as E
        uvs, alt = payload
        m = self._cur_mask()
        if m is None or self.image is None or not uvs:
            return
        op = self._mask_pending[1] if self._mask_pending else "add"
        auto = self.b_auto.isChecked()
        stroke = {"pts": self._s_points(uvs), "size": round(self.b_size.value() / 100.0, 4),
                  "feather": round(self.b_feather.value() / 100.0, 3), "flow": round(self.b_flow.value() / 100.0, 3),
                  "erase": bool(alt or self.b_erase.isChecked()), "ref": None}

        def put(ref):
            stroke["ref"] = ref
            sh = next((x for x in reversed(m["shapes"]) if x["type"] == "brush" and x.get("op", "add") == op
                       and bool(x.get("auto")) == auto), None)
            if sh is None:
                sh = {"type": "brush", "op": op, "auto": auto, "strokes": []}
                m["shapes"].append(sh)
            sh["strokes"].append(stroke)
            self.controls_changed()
            self._refresh_mask_controls()
            self._update_overlay()
        if auto and not stroke["erase"]:
            img, e0, (u0, v0), path = self.image, self.edit_from_controls(), uvs[0], self.path
            e0.masks = None
            run_bg(lambda: E.mask_features(img, e0, u0, v0), path, lambda p, ref: put(ref) if p == self.path else None,
                   lambda _, err: self.status.emit(f"Auto mask problem: {err}"))
        else:
            put(None)

    def _drag_done(self, u0, v0, u1, v1):
        from .. import engine as E
        m = self._cur_mask()
        kind = self.viewer.tool
        if m is None or self.image is None:
            return
        op = self._mask_pending[1] if self._mask_pending else "add"
        (xa, ya), (xb, yb) = self._s_points([(u0, v0), (u1, v1)])
        if kind == "linear":
            shape = {"type": "linear", "op": op, "x0": xa, "y0": ya, "x1": xb, "y1": yb}
        else:
            _, _, sw, sh = self.image.raw.crop
            r = math.hypot((xb - xa) * sw, (yb - ya) * sh) / float(max(sw, sh))
            shape = {"type": "radial", "op": op, "cx": xa, "cy": ya, "rx": round(r, 5), "ry": round(r, 5), "angle": 0.0,
                     "feather": round(self.r_feather.value() / 100.0, 3)}
        self._set_mask_tool(None)
        self._add_shape(shape)
        _ = E

    def _tool_click(self, u, v, alt):
        from .. import engine as E
        m = self._cur_mask()
        kind = self.viewer.tool
        if m is None or self.image is None:
            return
        op = self._mask_pending[1] if self._mask_pending else "add"
        if kind == "colour":
            a8 = self.last_a8
            if a8 is None:
                return
            h, w = a8.shape[:2]
            x, y = min(w - 1, int(u * w)), min(h - 1, int(v * h))
            hue, chroma = srgb8_hue(a8[max(0, y - 3):y + 4, max(0, x - 3):x + 4])
            if chroma < 0.02:
                self.status.emit("That spot is almost grey: click something more colourful (or use Brightness).")
                return
            self._set_mask_tool(None)
            self._add_shape({"type": "colour", "op": op, "hue": round(hue, 1), "width": round(self.c_width.value(), 1),
                             "soft": round(self.c_soft.value(), 1), "chroma": round(self.c_chroma.value() / 100.0, 4)})
            return
        if kind == "object":
            pt = self._s_points([(u, v)])[0]
            idx = next((i for i in range(len(m["shapes"]) - 1, -1, -1) if m["shapes"][i]["type"] == "object"
                        and m["shapes"][i].get("op", "add") == op), None)
            fg = list(m["shapes"][idx].get("fg", [])) if idx is not None else []
            bg = list(m["shapes"][idx].get("bg", [])) if idx is not None else []
            (bg if alt else fg).append(pt)
            if not fg:
                self.status.emit("Click the object first; Alt-click then marks what does not belong to it.")
                return
            img, e0, path = self.image, self.edit_from_controls(), self.path
            self.status.emit("Finding the object...")
            self.viewer.busy = True
            self.viewer.update()

            def done(p, shape):
                self.viewer.busy = False
                if p != self.path or self._cur_mask() is not m:
                    return
                shape["op"] = op
                if idx is not None and idx < len(m["shapes"]):
                    m["shapes"][idx] = shape
                else:
                    m["shapes"].append(shape)
                self.controls_changed()
                self._refresh_mask_controls()
                self._update_overlay()
                self.status.emit("Object selected. Click again to add a part, Alt-click to take a part away, Esc "
                                 "when done.")
            run_bg(lambda: E.object_mask(img, e0, fg, bg), path, done,
                   lambda _, err: (setattr(self.viewer, "busy", False), self.status.emit(f"Object problem: {err}")))

    def _mask_selected(self, row):
        self._msel = row
        self._refresh_mask_controls()
        self._update_overlay()

    def _mask_item_changed(self, it):
        row = self.mask_list.row(it)
        if 0 <= row < len(self._masks):
            self._masks[row]["enabled"] = it.checkState() == Qt.CheckState.Checked
            self._masks[row]["name"] = it.text().strip() or self._masks[row].get("name", "Mask")
            self.controls_changed()

    def delete_mask(self):
        if self._cur_mask() is None:
            return
        self._masks.pop(self._msel)
        self._msel = min(self._msel, len(self._masks) - 1)
        self._set_mask_tool(None)
        self._refresh_masks()
        self.controls_changed()
        self._update_overlay()

    def _mask_invert(self, on):
        m = self._cur_mask()
        if m is not None:
            m["invert"] = bool(on)
            self.controls_changed()
            self._update_overlay()

    def _mask_show(self, on):
        self.viewer.show_overlay = bool(on)
        self._update_overlay()

    def _mask_adjust(self, key, v):
        m = self._cur_mask()
        if m is None:
            return
        if key == "opacity":
            m["opacity"] = round(float(v), 1)
        else:
            m["adjust"][key] = round(float(v), 3)
        self.controls_changed()

    def _mask_add_picked(self, idx):
        text = self.mask_add.itemText(idx)
        self.mask_add.setCurrentIndex(0)
        if idx <= 0:
            return
        kinds = {"Brush": "brush", "Gradient": "linear", "Radial": "radial", "Object": "object", "Colour": "colour",
                 "Brightness": "luminance", "Skin": "skin", "Person": "person"}
        op = {"+": "add", "-": "subtract", "x": "intersect"}[text[0]]
        kind = next(v for k, v in kinds.items() if k in text)
        self._start_shape(kind, op)

    def _shape_params_changed(self, *_):
        m = self._cur_mask()
        if m is None or not m["shapes"]:
            return
        sh = m["shapes"][-1]
        if sh["type"] == "luminance":
            sh["lo"], sh["hi"] = round(self.r_lo.value() / 100.0, 3), round(self.r_hi.value() / 100.0, 3)
            sh["soft"] = round(self.r_soft.value() / 100.0, 3)
        elif sh["type"] == "colour":
            sh["width"], sh["soft"] = round(self.c_width.value(), 1), round(self.c_soft.value(), 1)
            sh["chroma"] = round(self.c_chroma.value() / 100.0, 4)
        elif sh["type"] == "radial":
            sh["feather"] = round(self.r_feather.value() / 100.0, 3)
        else:
            return
        self.controls_changed()
        self._update_overlay()

    def _update_overlay(self):
        if not hasattr(self, "ov_timer"):
            self.ov_timer = QTimer(self, singleShot=True, interval=180, timeout=self._render_overlay)
            self._ov_token = 0
        self.ov_timer.start()

    def _render_overlay(self):
        from .. import engine as E
        m = self._cur_mask()
        show = self.viewer.tool is not None or self.mask_show.isChecked()
        if self.image is None or m is None or not show or not m.get("shapes"):
            self.viewer.overlay = None
            self.viewer.update()
            return
        self._ov_token += 1
        tok, img, idx = self._ov_token, self.image, self._msel
        e = self.edit_from_controls()
        sc = max(0.05, self.preview_scale(e) * 0.5)
        run_bg(lambda: E.render_mask(img, e, idx, sc), tok, self._overlay_ready,
               lambda _, err: self.status.emit(f"Mask overlay problem: {err}"))

    def _overlay_ready(self, tok, a):
        if tok != getattr(self, "_ov_token", -1):
            return
        h, w = a.shape
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 239, 68, 68
        rgba[..., 3] = np.clip(a * 0.5 * 255, 0, 255).astype(np.uint8)
        self.viewer.overlay = QImage(np.ascontiguousarray(rgba).data, w, h, 4 * w,
                                     QImage.Format.Format_RGBA8888).copy()
        self.viewer.update()

    # ------------------------------------------------------------------------------------ lens
    def _fill_lenses(self, current: str | None = None):
        from ..engine.lens import library as lib
        if current is None:
            current = self.lens_combo.currentData() or ""
        self.lens_combo.blockSignals(True)
        self.lens_combo.clear()
        cam = (self.image.meta.lens if self.image is not None else "") or "manual lens, not tagged"
        self.lens_combo.addItem(f"Lens: {cam} (camera record)", "")
        for e in lib.list_lenses():
            self.lens_combo.addItem("Lens: " + e.label(), e.name)
        if current and self.lens_combo.findData(current) < 0:
            self.lens_combo.addItem(f"Lens: {current}", current)
        self.lens_combo.addItem("Add a lens...", "__add__")
        self.lens_combo.setCurrentIndex(max(0, self.lens_combo.findData(current)))
        self.lens_combo.blockSignals(False)

    def _lens_picked(self, idx):
        if self.image is None:
            return
        data = self.lens_combo.itemData(idx)
        if data == "__add__":
            dlg = LensDialog(self, self.edit.lens.tag, 1)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self._fill_lenses(self.edit.lens.tag)
                return
            data = dlg.choice
        e = self.edit_from_controls()
        e.lens.tag = data or ""
        self.settings.set("last_lens", e.lens.tag)
        self.commit(e, f"Lens set to {e.lens.tag or 'the camera record'} (kept in the sidecar, written on export).")

    def _dist_chip(self, e):
        d = e.geometry.distortion
        if d is None:
            self.dist_chip.setText("Distortion: not measured yet (press Fix lens)")
            self.dist_chip.setStyleSheet(f"color:{T.DIM};")
            return
        src = d.get("source", "")
        col = T.VIOLET if src.startswith("lensfun") else (T.CYAN if "wizard" in src else T.AMBER)
        self.dist_chip.setText(f"Distortion: {src}")
        self.dist_chip.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")

    def fix_lens(self):
        if self.image is None:
            return
        self.fix_b.setEnabled(False)
        self.status.emit("Fix lens: measuring...")
        img, e0, path = self.image, self.edit_from_controls(), self.path

        def work():
            from ..engine.lens.fix import fix_lens
            return fix_lens(img, e0)
        run_bg(work, path, self._fixed, self._fix_failed)

    def _fix_failed(self, _, err):
        self.fix_b.setEnabled(True)
        self.status.emit(f"Fix lens problem: {err}")

    def _fixed(self, path, res):
        self.fix_b.setEnabled(True)
        if path != self.path:
            return
        e, rep = res
        self.show_fix_report(rep)
        self.commit(e, "Fix lens: done (see the Lens panel for what was fixed and where the numbers came from).",
                    label="Fix lens")

    def show_fix_report(self, rep):
        while self.fix_report.count():
            w = self.fix_report.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        cols = {"cyan": T.CYAN, "violet": T.VIOLET, "amber": T.AMBER, "dim": T.DIM}
        for r in rep:
            c = cols.get(r["tier"], T.DIM)
            lb = label(f"<span style='color:{c}'>●</span> <b>{r['what']}</b>: {r['text']}", "dim", wrap=True)
            lb.setTextFormat(Qt.TextFormat.RichText)
            self.fix_report.addWidget(lb)

    # ------------------------------------------------------------------------------------ shape
    def auto_level(self):
        if self.image is None:
            return
        self.level_b.setEnabled(False)
        img, e0, path = self.image, self.edit_from_controls(), self.path

        def work():
            from ..engine.lens.fix import auto_straighten
            return auto_straighten(img, e0)
        run_bg(work, path, self._levelled, lambda _, err: (self.level_b.setEnabled(True),
                                                             self.status.emit(f"Auto level problem: {err}")))

    def _levelled(self, path, res):
        self.level_b.setEnabled(True)
        if path != self.path:
            return
        if not res:
            self.status.emit("Auto level: no straight lines to trust in this photo; left as it is.")
            return
        e = self.edit_from_controls()
        e.geometry.rotate = float(res["rotate"])
        self.commit(e, f"Auto level: turned {res['rotate']:+.2f}° from {res['lines']} straight lines "
                       f"({res['confidence']}).", label="Auto level")

    def _aspect_ratio(self):
        """The crop ratio (width / height of the displayed frame) for the chosen aspect, or None for free."""
        t = self.aspect.currentText()
        if t == "Free" or self.image is None:
            return None
        from .. import engine as E
        e = self.edit_from_controls()
        e.geometry.crop = None
        w, h = E.output_size(self.image, e)
        if t == "Original ratio":
            return w / h
        a, b = (float(v) for v in t.split(":"))
        r = max(a, b) / min(a, b)
        return r if w >= h else 1.0 / r

    def _aspect_picked(self, _):
        if self.image is None:
            return
        from .. import engine as E
        from ..engine.lens.geometry import crop_for_aspect
        e = self.edit_from_controls()
        t = self.aspect.currentText()
        e.geometry.aspect = {"Original ratio": "original", "Free": "free"}.get(t, t)
        if t not in ("Free",):
            g0 = E.Edit.from_dict(e.to_dict())
            g0.geometry.crop = None
            w, h = E.output_size(self.image, g0)
            c = crop_for_aspect(w, h, e.geometry.aspect)
            e.geometry.crop = list(c) if c else None
        self.viewer.crop_ratio = self._aspect_ratio()
        if self.viewer.crop_mode:
            self.viewer.crop = tuple(e.geometry.crop or (0.0, 0.0, 1.0, 1.0))
        self.commit(e)

    def toggle_crop(self, on):
        if self.image is None:
            self.crop_b.setChecked(False)
            return
        if on and self.viewer.zoomed:
            self.viewer.set_zoom(None)
        self.viewer.crop_mode = on
        self.viewer.crop = tuple(self.edit.geometry.crop or (0.0, 0.0, 1.0, 1.0))
        self.viewer.crop_ratio = self._aspect_ratio()
        self._escape(keep_zoom=True)
        self.render_now()

    def _crop_drawn(self, crop):
        e = self.edit_from_controls()
        full = all(abs(a - b) < 1e-3 for a, b in zip(crop, (0.0, 0.0, 1.0, 1.0)))
        e.geometry.crop = None if full else [float(v) for v in crop]
        if self._push(e, "Crop"):
            self.save_timer.start()                       # no re-render while drawing: the frame shows it

    def reset_crop(self):
        if self.image is None:
            return
        e = self.edit_from_controls()
        e.geometry.crop = None
        self.viewer.crop = (0.0, 0.0, 1.0, 1.0)
        self.commit(e)

    # ------------------------------------------------------------------------------------ composition
    def _sync_comp_combo(self):
        from ..engine.local import composition as cp
        keys = ["off"] + [k for k, _ in cp.OVERLAYS]
        self.comp_combo.setCurrentIndex(keys.index(self.viewer.comp) if self.viewer.comp in keys else 0)
        self.comp_turn.setEnabled(cp.ORIENTATIONS.get(self.viewer.comp, 1) > 1)

    def _set_overlay(self, name, orient=None):
        v = self.viewer
        v.comp = name
        if orient is not None:
            v.comp_orient = orient
        v.comp_flash = time.perf_counter()
        self.settings.set("comp_overlay", name)
        self.settings.set("comp_orient", v.comp_orient)
        self._sync_comp_combo()
        v.update()
        QTimer.singleShot(2600, v.update)                 # the name tag fades after a moment

    def _comp_picked(self, idx):
        from ..engine.local import composition as cp
        keys = ["off"] + [k for k, _ in cp.OVERLAYS]
        self._set_overlay(keys[idx])

    def cycle_overlay(self):
        """O: the next composition overlay (off is one of them). Outside the crop tool the first press shows the
        current one on the whole photo; cycling to off hides it again."""
        from ..engine.local import composition as cp
        v = self.viewer
        if not v.crop_mode and not v.comp_outside:
            v.comp_outside = True
            self._set_overlay(v.comp if v.comp != "off" else "thirds")
            return
        keys = [k for k, _ in cp.OVERLAYS] + ["off"]
        nxt = keys[(keys.index(v.comp) + 1) % len(keys)] if v.comp in keys else keys[0]
        if nxt == "off" and not v.crop_mode:
            v.comp_outside = False
        self._set_overlay(nxt)

    def turn_overlay(self):
        """Shift+O: mirror the overlay (the golden spiral has four ways round, the golden triangles two)."""
        from ..engine.local import composition as cp
        n = cp.ORIENTATIONS.get(self.viewer.comp, 1)
        if n > 1:
            self._set_overlay(self.viewer.comp, (self.viewer.comp_orient + 1) % n)

    def suggest_crops(self):
        if self.image is None:
            return
        from .. import engine as E
        from ..engine.local import composition as cp
        e = self.edit_from_controls()
        ratio = self._aspect_ratio()
        method = self.viewer.comp if self.viewer.comp in cp.METHODS else None
        img = self.image
        self.suggest_b.setEnabled(False)
        self.viewer.busy = True
        self.status.emit("Looking at the photo for crops (faces, people, what stands out)...")

        def done(_, res):
            self.suggest_b.setEnabled(True)
            if img is not self.image:
                return
            self._suggestions = res
            self.suggest_list.clear()
            for i, sgn in enumerate(res, 1):
                self.suggest_list.addItem(f"{i}. {sgn['text']}")
            self.suggest_list.setVisible(bool(res))
            if not self._render_live:
                self.viewer.busy = False
            self.status.emit(f"{len(res)} suggested crop{'s' if len(res) != 1 else ''}: click one to try it."
                             if res else "No crop suggestions for this photo.")
            if res:
                self._apply_suggestion(res[0])

        def fail(_, err):
            self.suggest_b.setEnabled(True)
            if not self._render_live:
                self.viewer.busy = False
            self.status.emit(f"Crop suggestions problem: {err}")
        run_bg(lambda: E.suggest_crops(img, e, ratio, method), "suggest", done, fail)

    def _suggestion_picked(self, item):
        i = self.suggest_list.row(item)
        if 0 <= i < len(self._suggestions):
            self._apply_suggestion(self._suggestions[i])

    def _apply_suggestion(self, sgn):
        """Try a suggested crop: the crop tool opens on it, with the overlay that explains it."""
        self._set_overlay(sgn["method"], int(sgn.get("orient", 0)))
        e = self.edit_from_controls()
        e.geometry.crop = [float(v) for v in sgn["crop"]]
        if not self.crop_b.isChecked():
            self.crop_b.setChecked(True)
        self.viewer.crop = tuple(e.geometry.crop)
        self.viewer.update()
        self._push(e, "Suggested crop")
        self.save_timer.start()

    # ------------------------------------------------------------------------------------ actions
    def set_strip(self, files: list, thumbs: dict):
        self.strip.clear()
        for f in files:
            it = QListWidgetItem("")
            it.setData(Qt.ItemDataRole.UserRole, f)
            ic = thumbs.get(f)
            if ic is not None:
                it.setIcon(ic)
            it.setToolTip(os.path.basename(f))
            self.strip.addItem(it)

    def open(self, path: str):
        from .. import engine as E
        if path == self.path:
            return
        self.save_sidecar()
        self.path, self.image, self.before_img = path, None, None
        self._wb_auto_used = False                          # Auto white balance belongs to the photo it ran on
        self.wb_chip.setText("Auto white balance: not used")
        self.before_key = None
        if self.crop_b.isChecked():
            self.crop_b.setChecked(False)
        self.show_fix_report([])
        self.viewer.after = None
        self.viewer.message = f"Opening {os.path.basename(path)} ..."
        self.viewer.zs.z = None                             # a new photo opens whole
        self.viewer.tile = self.viewer.tile_uv = None
        self.viewer.update()
        self.title.setText(os.path.basename(path))

        def work():
            img = E.open(path)
            return img, E.load_edit(path), E.load_history(path, 0)
        run_bg(work, path, self._opened, lambda _, err: self.status.emit(f"Could not open the file: {err}"))

    def _opened(self, path, res):
        if path != self.path:
            return
        self.image, edit, steps = res
        self.version = 0
        self._ssel = -1
        self.viewer.zs.z = None
        self.viewer.tile = self.viewer.tile_uv = None
        self.zoom_b.setText("Fit")
        self._reset_history(edit, steps)
        self.controls_from_edit(self.edit)
        self._canonical()
        self._fill_versions()
        self._fill_snapshots()
        m = self.image.meta
        exp = (f"1/{round(1 / m.exposure_s)} s" if m.exposure_s and m.exposure_s < 0.5 else
               (f"{m.exposure_s:g} s" if m.exposure_s else ""))
        bits = [m.model or "Unknown camera", m.lens or "manual lens", m.iso_label, exp,
                f"f/{m.fnumber:g}" if m.fnumber else "",
                f"camera {m.camera_temperature_c:.0f} °C" if m.camera_temperature_c is not None else ""]
        self.info.setText("  ·  ".join(b for b in bits if b))
        self._map_chip()
        self._fill_lenses(self.edit.lens.tag)
        if not self.edit.lens.tag and (m.manual_lens or not m.lens) and self.settings.get("last_lens"):
            self.status.emit(f"Manual lens. Last lens used: {self.settings.get('last_lens')} - pick it in the Lens "
                             f"panel (or tag the whole shoot in the Library).")
        self.render_now()
        img = self.image
        run_bg(lambda: img.noise_model(), path, self._noise_ready)
        run_bg(lambda: img.lateral_ca(), path, self._ca_ready)
        run_bg(lambda: img.capture_blur(self.edit), path, self._blur_ready)

    def _ca_ready(self, path, ca):
        if path != self.path:
            return
        if ca.confidence == "none":
            self.ca_chip.setText("Lateral CA: none measurable in this photo")
            self.ca_chip.setStyleSheet(f"color:{T.DIM};")
        else:
            self.ca_chip.setText(f"Lateral CA: {ca.max_shift_px:.1f} px at the corners, corrected ({ca.confidence})")
            self.ca_chip.setStyleSheet(f"color:{T.AMBER}; border-color:{T.alpha(T.AMBER, 0.4)};")

    def _blur_ready(self, path, b):
        if path != self.path:
            return
        col = T.AMBER if b.confidence != "default" else T.DIM
        self.blur_chip.setText(f"Measured blur: {b.sigma:.2f} px ({b.source}, {b.confidence})")
        self.blur_chip.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")

    def _noise_ready(self, path, model):
        if path != self.path:
            return
        src = model.source
        col = T.CYAN if src.startswith("calibrated") else (T.VIOLET if src == "shared" else T.AMBER)
        txt = {"estimated": "estimated from this photo"}.get(src, src)
        self.noise_chip.setText(f"Noise model: {txt}  ({model.confidence})")
        self.noise_chip.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")

    def shown_edit(self):
        """The edit as the viewer shows it: while cropping, the whole frame (the crop is drawn on top)."""
        from .. import engine as E
        e = self.edit
        if self.viewer.crop_mode:
            e = E.Edit.from_dict(e.to_dict())
            e.geometry.crop = None
        return e

    def preview_scale(self, edit=None):
        from .. import engine as E
        w, h = E.output_size(self.image, edit or self.shown_edit())
        target = max(self.viewer.width(), self.viewer.height()) * self.devicePixelRatioF() * 1.1
        return min(1.0, max(0.08, target / max(w, h)))

    def _busy_changed(self, on: bool):
        if on:
            self._work_tick.start()
        else:
            self._work_tick.stop()
        self._show_work()

    def _show_work(self):
        v = self.viewer
        if v.busy:
            t = v.busy_for()
            text, colour = (f"● Working... {t:.0f} s" if t >= 1.0 else "● Working..."), T.AMBER
        else:
            text, colour = (f"Ready  ({v.busy_took:.1f} s)" if v.busy_took else "Ready"), T.DIM
        self.work_chip.setText(text)
        self.work_chip.setStyleSheet(f"color:{colour}; border-color:{T.alpha(colour, 0.45 if v.busy else 0.33)};")

    def render_now(self):
        """The settings changed (or something they depend on did): render again. The zoomed view's sharp tile goes
        first when zoomed (it is what you are looking at), then the whole-photo preview (the navigator, the
        histogram, and what shows round the tile while you move)."""
        if self.image is None:
            return
        self._dirty_base = True
        self._dirty_tile = True
        self._pump()

    def _view_changed(self):
        """The viewer zoomed or moved (once the movement settles): a sharp tile if the part on screen needs one."""
        v = self.viewer
        self.zoom_b.setText(Z.label(v.zs.z) if v.zoomed else "Fit")
        if v.zoomed and self.ba.isChecked():
            self.ba.setChecked(False)               # the before/after wipe is for the whole photo
        if not v.zoomed:
            v.tile = v.tile_uv = None               # (a new zoom renders a new tile; this one may be stale)
            v.update()
        self._pump()

    def _tile_needed(self, edit):
        """(x, y, w, h, scale) of the sharp render the zoomed view needs, or None (fit, the preview is sharp
        enough at this zoom, or the tile already on screen holds everything visible)."""
        from .. import engine as E
        v = self.viewer
        if not v.zoomed or v.crop_mode or v.after is None:
            return None
        ow, oh = E.output_size(self.image, edit)
        if (ow, oh) != (v.zs.ow, v.zs.oh):
            v.zs.set_photo(ow, oh)
        if v.zs.z <= v.after.width() / float(ow) * 1.05:
            return None                              # the whole-photo preview already has these pixels
        req = v.zs.tile_request()
        if req is None:
            return None
        if not self._dirty_tile and Z.covers(v.tile_uv, v.tile_scale, v.zs.visible_window(), ow, oh, req[4]):
            return None
        return req

    def _pump(self):
        """Start the next render, if one is due and none is running: the tile, then the preview."""
        if self.image is None:
            return
        if self._render_live:
            self._render_again = True
            self.viewer.busy = True
            return
        self._render_again = False
        edit = self.shown_edit()
        req = self._tile_needed(edit)
        if req is not None:
            self._start_tile(edit, req)
        elif self._dirty_base or self.viewer.after is None:
            self._start_base(edit)
        else:
            self.viewer.busy = False
            self.viewer.update()

    def _start_base(self, edit):
        from .. import engine as E
        self._dirty_base = False
        self.token += 1
        tok, img, sc = self.token, self.image, self.preview_scale(edit)
        self._render_live, self._render_img = True, img
        ow, oh = E.output_size(img, edit)
        self.viewer.zs.set_photo(ow, oh)
        # "before" = as shot, but with the same straighten / perspective / crop, so the wipe lines up
        bkey = (json.dumps(edit.geometry.__dict__, sort_keys=True, default=str), round(sc, 4), self.judge_bw)
        need_before = self.before_img is None or bkey != self.before_key
        self.viewer.busy = True
        self.viewer.update()
        t0 = time.perf_counter()
        bw = self.judge_bw

        def work():
            a8 = display8(E.render(img, edit, scale=sc))
            if bw:
                a8 = bw8(a8)
            b8 = None
            if need_before:
                b = E.Edit()
                b.geometry = edit.geometry
                b8 = display8(E.render(img, b, scale=sc))
                if bw:
                    b8 = bw8(b8)
            return a8, b8, bkey, time.perf_counter() - t0
        run_bg(work, tok, self._rendered, self._render_failed)

    def _start_tile(self, edit, req):
        from .. import engine as E
        self._dirty_tile = False
        self.token += 1
        tok, img = self.token, self.image
        self._render_live, self._render_img = True, img
        x, y, w, h, scale = req
        ow, oh = E.output_size(img, edit)
        uv = (x / ow, y / oh, (x + w) / ow, (y + h) / oh)
        self.viewer.busy = True
        t0 = time.perf_counter()
        bw = self.judge_bw
        log.debug("zoom tile: %dx%d+%d+%d at %.3f", w, h, x, y, scale)

        def work():
            a8 = display8(E.render_window(img, edit, (x, y, w, h), scale))
            return (bw8(a8) if bw else a8), uv, scale, time.perf_counter() - t0
        run_bg(work, tok, self._tile_rendered, self._render_failed)

    def _tile_rendered(self, tok, res):
        self._render_live = False
        if tok == self.token and self._render_img is self.image and self.viewer.zoomed:
            a8, uv, scale, dt = res
            self.viewer.tile, self.viewer.tile_uv, self.viewer.tile_scale = to_qimage(a8), uv, scale
            self.viewer.update()
            self.status.emit(f"{Z.label(self.viewer.zs.z)} view rendered in {dt:.1f} s")
        self._pump()

    def _render_finished(self) -> bool:
        """Bookkeeping when a render ends; starts the next one if any is due. True when one was started."""
        self._render_live = False
        self._pump()
        return self._render_live

    def _render_failed(self, tok, err):
        self.status.emit(f"Render problem: {err}")
        self._render_live = False
        self._pump()

    def _zoom_anchor(self):
        """Where Z and the zoom keys zoom to: the pointer when it is over the photo, else the middle."""
        v = self.viewer
        m = v._mouse
        if m is None or not (0.0 <= m[0] <= 1.0 and 0.0 <= m[1] <= 1.0) or not v.underMouse():
            return None
        return v._px(m, v.photo_rect())

    # ------------------------------------------------------------------------------------ lens glow (raw/glow.py)
    def _fill_glow(self, current: str):
        from .. import engine as E
        self.glow_combo.blockSignals(True)
        self.glow_combo.clear()
        self.glow_combo.addItem("No lens glow profile (from the photo only)", "")
        for p in E.glow_profiles():
            f = f" f/{p['fnumber']:g}" if p.get("fnumber") else ""
            self.glow_combo.addItem(f"Lens glow: {p['lens'] or p['name']}{f}  ({p['made']})", p["name"])
        self.glow_combo.addItem("How to measure a lens's glow...", "__measure__")
        i = self.glow_combo.findData(current) if current else 0
        self.glow_combo.setCurrentIndex(max(i, 0))
        self.glow_combo.blockSignals(False)

    def _glow_visibility(self):
        on = self.hl_mode.currentIndex() == 2
        self.glow_combo.setVisible(on)
        self.glow_chip.setVisible(on)

    def _hl_mode_changed(self, i):
        from .. import engine as E
        if i == 2 and self.image is not None and not self.glow_combo.currentData():
            name = E.pick_glow_profile(self.image, self.edit_from_controls())
            if name:                                    # your profile for this lens, picked for you
                k = self.glow_combo.findData(name)
                if k >= 0:
                    self.glow_combo.blockSignals(True)
                    self.glow_combo.setCurrentIndex(k)
                    self.glow_combo.blockSignals(False)
                    self.controls_changed()
        self._glow_visibility()

    def _glow_combo_changed(self, _):
        if self.glow_combo.currentData() == "__measure__":
            self.glow_combo.blockSignals(True)
            self.glow_combo.setCurrentIndex(0)
            self.glow_combo.blockSignals(False)
            QMessageBox.information(self, "Measure a lens's glow", GLOW_HOWTO)
            return
        self.controls_changed()

    def _glow_refresh(self):
        """Tell Bill what the glow measured (after each render, in the background)."""
        from .. import engine as E
        if self.image is None or self.edit.highlight_mode != "glow":
            return
        img, e, path = self.image, self.edit, self.path

        def done(p, rep):
            if p != self.path:
                return
            if not rep:
                self.glow_chip.setText("No blown lights in this photo")
                self.glow_chip.setStyleSheet("")
                return
            lens = [r for r in rep if r["tier"] == "lens"]
            photo = [r for r in rep if r["tier"] == "photo"]
            rest = len(rep) - len(lens) - len(photo)
            bits = []
            if lens:
                bits.append(f"{len(lens)} measured from your lens profile (up to +{max(r['stops'] for r in lens):.1f} "
                            "stops over the clip)")
            if photo:
                bits.append(f"{len(photo)} measured from this photo (estimate)")
            if rest:
                bits.append(f"{rest} too faint or fully blown without a profile (colour rebuilt)")
            self.glow_chip.setText(f"{len(rep)} blown light{'s' if len(rep) != 1 else ''}: " + "; ".join(bits))
            col = T.CYAN if lens else (T.AMBER if photo else None)
            self.glow_chip.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};" if col else "")
        run_bg(lambda: E.glow_report(img, e), path, done, lambda _, err: log.warning("glow report: %s", err))

    # ------------------------------------------------------------------------------------ portrait
    PT_SLIDERS = (("younger", "Younger", "One slider for all of the face work below (each slider can still go "
                   "further on its own). Lines, under-eye shadows, smile lines, even skin, a firmer jaw, brighter "
                   "eyes and teeth, all at once and in proportion."),
                  ("lines", "Lines", "Softens wrinkles and fine lines: forehead, between the brows, crow's feet, "
                   "under the eyes. Pores, freckles and moles stay."),
                  ("under_eyes", "Under eyes", "Lifts dark circles and the hollows under the eyes, and evens "
                   "their bluish cast"),
                  ("smile_lines", "Smile lines", "Lifts the shadow of the folds from nose to mouth and mouth to chin "
                   "part way, so the face keeps its shape"),
                  ("even_skin", "Even skin", "Evens blotches and uneven colour on the face (pores stay)"),
                  ("jawline", "Jawline", "Firms the jaw below the cheeks a little (a small warp of the jowls)"),
                  ("double_chin", "Double chin", "Lifts the soft area under the jaw towards it, softens the crease "
                   "and deepens the jaw's own shadow a little. Skin only: a collar or a beard under the chin stays "
                   "put. Not part of Younger."),
                  ("eyes", "Eyes", "Brighter whites, fewer red veins, a crisper iris"),
                  ("iris_light", "  iris brightness", "Brightens the coloured part of the eye (not the pupil or the "
                   "catchlights). Keep it gentle in candle-lit or low-key photos: a bright iris in a dark face "
                   "looks false. Not part of Younger."),
                  ("iris_clarity", "  iris clarity", "Brings out the fibres of the iris. Not part of Younger."),
                  ("iris_colour", "  iris colour strength", "Richer iris colour. Not part of Younger."),
                  ("ring", "  outer ring", "Darkens the outer edge of the iris (the limbal ring): a classic "
                   "retouch that makes eyes look brighter and younger. Not part of Younger."),
                  ("catch", "  catchlights", "Brightens the catchlights already in the eyes (nothing is painted "
                   "in). Not part of Younger."),
                  ("teeth", "Teeth", "Whiter teeth, towards the colour of the eye whites (so a warm candle-lit "
                                     "photo stays warm), and a touch brighter"))
    SHAPES = (("breast", "Breast lift & size", 0.85, "Drag across one breast (do each side on its own). Lift raises "
               "it, the lower part most; Size makes it a little fuller or smaller."),
              ("butt", "Butt improvement tool", 0.85, "Drag across one side of the bottom. Lift raises it, Size makes "
               "it a little rounder or smaller."),
              ("waist", "Slim waist", 0.5, "Drag along the waist (top to bottom). Slim narrows it across."),
              ("tummy", "Slim tummy", 0.6, "Drag along the tummy (top to bottom). Slim draws it in."),
              ("arm", "Slim arm", 0.35, "Drag along the arm, shoulder to elbow or elbow to wrist. Slim narrows "
               "it across."),
              ("thigh", "Slim thigh", 0.4, "Drag along the thigh, hip to knee. Slim narrows it across."))

    def _build_portrait(self, bl):
        self.full_only_portrait = []
        g = Group("Portrait", "Faces are found on this PC (Google's MediaPipe face models, run by the Studio "
                              "itself). Every slider starts at 0 and stays subtle at 100: the aim is the person on "
                              "a good day, not a different person. Blemishes become Heal spots in Retouch (amber), "
                              "so you can remove any you want to keep.")
        self.pt_find = QPushButton("Find faces")
        self.pt_find.setToolTip("Find the faces in this photo and map their features (eyes, brows, lips, outline)")
        self.pt_find.clicked.connect(self.find_faces)
        self.pt_eyes = QPushButton("Click the eyes")
        self.pt_eyes.setCheckable(True)
        self.pt_eyes.setToolTip("When a face is not found (deep shadow, half-lit by a candle): click the centre of "
                                "each eye and the face is mapped from there")
        self.pt_eyes.toggled.connect(lambda on: self._set_portrait_tool("eyes" if on else None))
        self.pt_show = QPushButton("Show faces")
        self.pt_show.setCheckable(True)
        self.pt_show.toggled.connect(self._show_faces)
        for b in (self.pt_find, self.pt_eyes, self.pt_show):
            b.setStyleSheet("padding: 5px 6px;")
        g.add(hbox(self.pt_find, self.pt_eyes, self.pt_show))
        self.pt_info = g.add(chip("No faces yet: press Find faces", T.DIM))
        self.pt_blem = QPushButton("Remove blemishes")
        self.pt_blem.setObjectName("primary")
        self.pt_blem.setToolTip("Find spots and red blemishes on the faces and heal them (Retouch spots, amber). "
                                "Very dark, well-defined marks are left alone: usually moles or beauty marks.")
        self.pt_blem.clicked.connect(self.remove_blemishes)
        self.pt_chin = QPushButton("Remove double chin")
        self.pt_chin.setObjectName("primary")
        self.pt_chin.setToolTip("One press: sets Double chin to 60 (the slider below fine-tunes it, 0 takes it off)")
        self.pt_chin.clicked.connect(self.remove_double_chin)
        self.pt_forget = QPushButton("Forget faces")
        self.pt_forget.setToolTip("Remove the mapped faces (the face sliders then do nothing until faces are found "
                                  "again)")
        self.pt_forget.clicked.connect(self.forget_faces)
        for b in (self.pt_blem, self.pt_chin, self.pt_forget):
            b.setStyleSheet("padding: 5px 8px;")
        g.add(hbox(self.pt_blem, self.pt_chin))
        g.add(hbox(self.pt_forget))
        self.pt_sl = {}
        for key, name, tip in self.PT_SLIDERS:
            sl = g.add(ValueSlider(name, 0, 100, 0, 1, "{:.0f}", tip))
            sl.changed.connect(self.controls_changed)
            self.pt_sl[key] = sl
            if key in ("iris_light", "iris_clarity", "iris_colour", "ring", "catch"):
                self.full_only_portrait.append(sl)
                sl.changed.connect(lambda *_: self._ensure_irises())
        # eye colour and red eye (need the irises, found with the faces)
        g.add(label("Eye colour", "dim"))
        self.ec_preset = QComboBox()
        for key, name in EYE_COLOURS:
            self.ec_preset.addItem(name, key)
        self.ec_preset.setToolTip("A new colour for the iris: the fibres, the pupil and the catchlights stay. Big "
                                  "changes (dark brown to pale blue) can look painted; smaller ones look real.")
        self.ec_which = QComboBox()
        for key, name in (("both", "Both eyes"), ("right", "Person's right"), ("left", "Person's left")):
            self.ec_which.addItem(name, key)
        g.add(self.ec_preset)
        g.add(self.ec_which)
        self.ec_amount = g.add(ValueSlider("  colour amount", 0, 100, 0, 1, "{:.0f}", "How far the iris moves to "
                                           "the new colour (0 = unchanged)"))
        self.ec_preset.activated.connect(self._eye_colour_picked)
        self.ec_which.activated.connect(lambda _: self._portrait_changed("Portrait: eye colour"))
        self.ec_amount.changed.connect(lambda *_: self._portrait_changed("Portrait: eye colour"))
        self.ec_custom = None
        self.pt_red = QPushButton("Fix red eye")
        self.pt_red.setObjectName("primary")
        self.pt_red.setToolTip("Finds red pupils (and pupils the flash blew out white) in the mapped eyes and fixes "
                               "them: Red eye and Pupil spots in Retouch (amber), removable one by one. For an "
                               "animal's glowing eyes use Retouch > Pupil.")
        self.pt_red.clicked.connect(self.fix_red_eye)
        self.pt_red.setStyleSheet("padding: 5px 8px;")
        g.add(hbox(self.pt_red))
        g.add(label("Body sculpting", "dim"))
        self.sc_btn = {}
        for kind, text, tip in (("push", "Push", "Drag to move a contour gently (an arm, a waistline, a stray "
                                 "bulge)"), ("shrink", "Shrink", "Click or drag over an area to make it a little "
                                 "smaller"), ("grow", "Grow", "Click or drag over an area to make it a little "
                                 "fuller")):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setStyleSheet("padding: 5px 6px;")
            b.toggled.connect(lambda on, k=kind: self._set_portrait_tool(("sculpt", k) if on else None))
            self.sc_btn[kind] = b
        g.add(hbox(*self.sc_btn.values()))
        self.sc_size = g.add(ValueSlider("Brush size", 1, 30, 8, 0.5, "{:.1f}%", "Radius as a share of the photo's "
                                         "long edge. Bigger and gentler looks more natural."))
        self.sc_strength = g.add(ValueSlider("Strength", 10, 100, 50, 1, "{:.0f}", "How far one stroke moves "
                                             "things (always capped, so nothing tears)"))
        self.sc_size.changed.connect(lambda v: (setattr(self.viewer, "brush_frac", v / 100.0), self.viewer.update()))
        self.sh_btn = {}
        rows = []
        for tool, text, ratio, tip in self.SHAPES:
            b = QPushButton(text.replace("&", "&&"))
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setStyleSheet("padding: 5px 6px;")
            b.toggled.connect(lambda on, t=tool: self._set_portrait_tool(("shape", t) if on else None))
            self.sh_btn[tool] = b
            rows.append(b)
        for i in range(0, len(rows), 2):
            g.add(hbox(*rows[i:i + 2]))
        self.sh_lift = g.add(ValueSlider("Lift", -100, 100, 0, 1, "{:+.0f}", "Raises (+) or lowers (-) the selected "
                                         "shape, the lower part most"))
        self.sh_size = g.add(ValueSlider("Size", -100, 100, 0, 1, "{:+.0f}", "Fuller (+) or smaller (-)"))
        self.sh_slim = g.add(ValueSlider("Slim", -100, 100, 0, 1, "{:+.0f}", "Narrower across (+) or wider (-)"))
        self.sh_width = g.add(ValueSlider("Shape width", 20, 120, 50, 1, "{:.0f}%", "How wide the selected oval is, "
                                          "as a share of its length. Keep it snug to the body."))
        for w in (self.sh_lift, self.sh_size, self.sh_slim, self.sh_width):
            w.changed.connect(self._sculpt_shape_changed)
        self.sc_undo = QPushButton("Undo last")
        self.sc_undo.clicked.connect(self.sculpt_undo)
        self.sh_del = QPushButton("Delete shape")
        self.sh_del.clicked.connect(self.shape_delete)
        self.sc_clear = QPushButton("Clear sculpting")
        self.sc_clear.clicked.connect(self.sculpt_clear)
        for b in (self.sc_undo, self.sh_del, self.sc_clear):
            b.setStyleSheet("padding: 5px 8px;")
        g.add(hbox(self.sc_undo, self.sh_del, self.sc_clear))
        self.sc_show = QPushButton("Show shapes")
        self.sc_show.setCheckable(True)
        self.sc_show.toggled.connect(self._show_shapes)
        self.sc_show.setStyleSheet("padding: 5px 8px;")
        g.add(hbox(self.sc_show))
        self.sc_info = g.add(chip("No sculpting", T.DIM))
        self.portrait_group = g
        bl.addWidget(g)

    def _portrait_value(self):
        import copy as _copy
        p = _copy.deepcopy(self._portrait or {})
        for k, sl in getattr(self, "pt_sl", {}).items():
            v = round(sl.value(), 1)
            if v:
                p[k] = v
            else:
                p.pop(k, None)
        ec = self._eye_colour_value() if hasattr(self, "ec_preset") else None
        if ec:
            p["eye_colour"] = ec
        else:
            p.pop("eye_colour", None)
        if not p.get("faces"):
            p.pop("faces", None)
        if not p.get("sculpt"):
            p.pop("sculpt", None)
        return p or None

    def _eye_colour_value(self):
        key = self.ec_preset.currentData() or "none"
        amt = round(self.ec_amount.value(), 1)
        if key == "none" or amt <= 0:
            return None
        d = {"preset": key, "amount": amt, "which": self.ec_which.currentData() or "both"}
        if key == "custom" and self.ec_custom:
            d.update(hue=round(self.ec_custom[0], 2), sat=round(self.ec_custom[1], 4), light=round(self.ec_custom[2], 4))
        return d

    def _eye_colour_picked(self, idx):
        if self.ec_preset.itemData(idx) == "custom":
            from PySide6.QtWidgets import QColorDialog
            c = QColorDialog.getColor(QColor(90, 130, 180), self, "Eye colour")
            if not c.isValid():
                self.ec_preset.setCurrentIndex(0)
                return
            from ..engine.local.eyes import custom_from_srgb
            self.ec_custom = custom_from_srgb((c.red(), c.green(), c.blue()))
        if self.ec_amount.value() <= 0 and self.ec_preset.itemData(idx) != "none":
            self.ec_amount.set_value(70, emit=False)
        self._portrait_changed("Portrait: eye colour")

    def _portrait_from_edit(self, e):
        import copy as _copy
        self._portrait = _copy.deepcopy(e.portrait or {})
        for k, sl in self.pt_sl.items():
            sl.set_value(float(self._portrait.get(k, 0) or 0), emit=False)
        ec = self._portrait.get("eye_colour") or {}
        for w in (self.ec_preset, self.ec_which):
            w.blockSignals(True)
        self.ec_preset.setCurrentIndex(max(0, self.ec_preset.findData(ec.get("preset", "none"))))
        self.ec_which.setCurrentIndex(max(0, self.ec_which.findData(ec.get("which", "both"))))
        for w in (self.ec_preset, self.ec_which):
            w.blockSignals(False)
        self.ec_amount.set_value(float(ec.get("amount", 0) or 0), emit=False)
        self.ec_custom = (ec["hue"], ec["sat"], ec["light"]) if ec.get("preset") == "custom" and "hue" in ec else None
        n = len(self._portrait.get("sculpt") or [])
        if not (0 <= self._shsel < n) or (self._portrait.get("sculpt") or [{}])[self._shsel].get("kind") != "shape":
            self._shsel = -1
        self._refresh_portrait_ui()

    def _faces(self):
        return [f for f in (self._portrait.get("faces") or []) if f.get("enabled", True)]

    def _refresh_portrait_ui(self):
        from ..engine.local import faces as fz
        faces = self._faces()
        if not fz.available():
            self.pt_info.setText("Face models missing (models\\faces). Run setup again.")
            self.pt_info.setStyleSheet(f"color:{T.AMBER};")
        elif not faces:
            self.pt_info.setText("No faces yet: press Find faces (or Click the eyes)")
            self.pt_info.setStyleSheet("")
        else:
            found = sum(1 for f in faces if f.get("how") != "eyes")
            txt = f"{len(faces)} face{'s' if len(faces) != 1 else ''} mapped"
            if found:
                txt += f" ({found} found automatically, amber outline)"
            if len(faces) > found:
                txt += f" ({len(faces) - found} from your eye clicks, cyan)"
            self.pt_info.setText(txt)
            col = T.AMBER if found else T.CYAN
            self.pt_info.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")
        has = bool(faces)
        for w in [self.pt_blem, self.pt_chin, self.pt_forget] + list(self.pt_sl.values()):
            w.setEnabled(has)
        self.pt_sl["jawline"].setEnabled(has)
        sc = self._portrait.get("sculpt") or []
        shapes = sum(1 for x in sc if x.get("kind") == "shape")
        strokes = len(sc) - shapes
        self.sc_info.setText("No sculpting" if not sc else
                             ", ".join(t for t in (f"{strokes} stroke{'s' if strokes != 1 else ''}" if strokes else "",
                                                    f"{shapes} shape{'s' if shapes != 1 else ''}" if shapes else "")
                                       if t))
        sel = sc[self._shsel] if 0 <= self._shsel < len(sc) else None
        for w in (self.sh_lift, self.sh_size, self.sh_slim, self.sh_width, self.sh_del):
            w.setEnabled(sel is not None)
        if sel is not None:
            lift_tool = sel.get("tool") in ("breast", "butt")
            self.sh_lift.set_value(float(sel.get("lift", 0)), emit=False)
            self.sh_size.set_value(float(sel.get("size", 0)), emit=False)
            self.sh_slim.set_value(float(sel.get("slim", 0)), emit=False)
            self.sh_width.set_value(float(sel.get("b", 0.04)) / max(float(sel.get("a", 0.05)), 1e-6) * 100.0,
                                    emit=False)
            self.sh_lift.setVisible(True)
            self.sh_size.setVisible(lift_tool)
            self.sh_slim.setVisible(not lift_tool)
        self.sc_undo.setEnabled(bool(sc))
        self.sc_clear.setEnabled(bool(sc))

    def _refresh_portrait_view(self):
        """Faces and sculpt shapes on the displayed photo."""
        from .. import engine as E
        if self.image is None:
            self.viewer.faces, self.viewer.shapes = [], []
            return
        e = self.shown_edit()
        try:
            self.viewer.faces = E.face_outlines(self.image, e) if self._portrait.get("faces") else []
        except Exception:                                     # noqa: BLE001  (an overlay must never stop the app)
            self.viewer.faces = []
        _, _, sw, sh = self.image.raw.crop
        long = float(max(sw, sh))
        shapes = []
        for k, x in enumerate(self._portrait.get("sculpt") or []):
            if x.get("kind") != "shape":
                continue
            c = np.asarray(x.get("c", [0.5, 0.5]), float) * [sw, sh]
            e1 = np.asarray(x.get("e1", [1, 0]), float)
            e1 = e1 / (np.hypot(*e1) + 1e-12)
            e2 = np.array([-e1[1], e1[0]])
            A, B = float(x.get("a", 0.05)) * long, float(x.get("b", 0.04)) * long
            t = np.linspace(0, 2 * math.pi, 48, endpoint=False)
            P = c[None, :] + A * np.cos(t)[:, None] * e1 + B * np.sin(t)[:, None] * e2
            u, v = E.s_to_view(self.image, e, P[:, 0] / sw, P[:, 1] / sh)
            shapes.append({"outline": list(zip(np.atleast_1d(u), np.atleast_1d(v))), "sel": k == self._shsel,
                           "tool": x.get("tool"), "index": k})
        self.viewer.shapes = shapes
        self.viewer.update()

    def _show_faces(self, on):
        self.viewer.show_faces = bool(on)
        self.viewer.update()

    def _show_shapes(self, on):
        self.viewer.show_shapes = bool(on)
        self.viewer.update()

    def _set_portrait_tool(self, which):
        """which: None | "eyes" | ("sculpt", kind) | ("shape", tool)."""
        if self.image is None:
            which = None
        kind = which if isinstance(which, str) else (which[0] if which else None)
        sub = which[1] if isinstance(which, tuple) else None
        for b, on in [(self.pt_eyes, kind == "eyes")] + [(b, kind == "sculpt" and sub == k)
                                                          for k, b in self.sc_btn.items()] + \
                [(b, kind == "shape" and sub == t) for t, b in self.sh_btn.items()]:
            b.blockSignals(True)
            b.setChecked(on)
            b.blockSignals(False)
        if kind:
            if self.viewer.tool == "spot":
                self._set_spot_tool(None)
            elif self.viewer.tool and self.viewer.tool not in ("eyes", "sculpt", "shape"):
                self._set_mask_tool(None)
            if self.viewer.crop_mode:
                self.crop_b.setChecked(False)
            self.viewer.tool = kind
            self.viewer._pts = []
            self.sculpt_kind = sub if kind == "sculpt" else None
            self.viewer.shape_tool = sub if kind == "shape" else None
            if kind == "shape":
                self.viewer.shape_ratio = dict((t, r) for t, _, r, _ in self.SHAPES).get(sub, 0.5)
                self._refresh_portrait_view()
            self.viewer.brush_scale = self._brush_scale()
            self.viewer.brush_frac = self.sc_size.value() / 100.0
            self.viewer.setCursor(Qt.CursorShape.BlankCursor if kind == "sculpt" else Qt.CursorShape.CrossCursor)
            if kind == "eyes":
                self.viewer.show_faces = True
                self.status.emit("Click the centre of each eye of the face to map.")
        elif self.viewer.tool in ("eyes", "sculpt", "shape"):
            self.viewer.tool = None
            self.viewer.shape_tool = None
            self.viewer._pts = []
            self.viewer.show_faces = self.pt_show.isChecked()
            self.viewer.setCursor(Qt.CursorShape.ArrowCursor)
        self.viewer.update()

    def _portrait_changed(self, label):
        eye_on = any(self.pt_sl[k].value() > 0 for k in ("iris_light", "iris_clarity", "iris_colour", "ring", "catch")) \
            or (self.ec_amount.value() > 0 and (self.ec_preset.currentData() or "none") != "none")
        if eye_on:
            self._ensure_irises()
        self.commit(self.edit_from_controls(), label=label)
        self._refresh_portrait_ui()
        self._refresh_portrait_view()

    def find_faces(self):
        from .. import engine as E
        from ..engine.local import faces as fz
        if self.image is None:
            return
        if not fz.available():
            self.status.emit("The face models are missing from models\\faces. Run setup again (or copy that folder "
                             "back from the download).")
            return
        img, e, path = self.image, self.edit_from_controls(), self.path
        self.viewer.busy = True
        self.viewer.update()
        self.status.emit("Finding faces...")

        def done(p, faces):
            if p != self.path:
                return
            self.viewer.busy = False
            if not faces:
                self.status.emit("No face found. If there is one (a face half in shadow, say), press Click the eyes "
                                 "and click the centre of each eye.")
                self.viewer.update()
                return
            self._portrait["faces"] = faces
            self.pt_show.setChecked(True)
            self._portrait_changed(f"Portrait: {len(faces)} face{'s' if len(faces) != 1 else ''} found")
            self.status.emit(f"{len(faces)} face{'s' if len(faces) != 1 else ''} found (amber outline). Now try "
                             f"Remove blemishes, or the sliders.")
        run_bg(lambda: E.find_faces(img, e), path, done,
               lambda _, err: (setattr(self.viewer, "busy", False), self.status.emit(f"Find faces problem: {err}")))

    def _eyes_clicked(self, pts):
        from .. import engine as E
        if self.image is None or len(pts) < 2:
            return
        (u1, v1), (u2, v2) = pts[:2]
        img, e, path = self.image, self.edit_from_controls(), self.path
        self.viewer.busy = True
        self.viewer.update()

        def done(p, face):
            if p != self.path:
                return
            self.viewer.busy = False
            self._set_portrait_tool(None)
            if face is None:
                self.status.emit("Could not map a face from those two clicks. Click the centre of each eye, a "
                                 "little more carefully, or try at 100%.")
                self.viewer.update()
                return
            faces = list(self._portrait.get("faces") or [])
            faces.append(face)
            self._portrait["faces"] = faces
            self.pt_show.setChecked(True)
            self._portrait_changed("Portrait: face from eyes")
            self.status.emit("Face mapped from your clicks (cyan outline). Check that the outline sits on the face.")
        run_bg(lambda: E.face_from_eyes(img, e, u1, v1, u2, v2), path, done,
               lambda _, err: (setattr(self.viewer, "busy", False), self.status.emit(f"Face problem: {err}")))

    def forget_faces(self):
        if not self._portrait.get("faces"):
            return
        self._portrait.pop("faces", None)
        self._portrait_changed("Portrait: faces forgotten")

    DOUBLE_CHIN_PRESS = 60

    def remove_double_chin(self):
        """One press for the Double chin slider (a friend's request): 60 is clearly there but still subtle. Pressed
        again it leaves a value you have set yourself alone."""
        sl = self.pt_sl["double_chin"]
        if sl.value() < self.DOUBLE_CHIN_PRESS:
            sl.set_value(self.DOUBLE_CHIN_PRESS)                  # emits: the edit and the history follow
        self.status.emit("Double chin: lifted and softened (fine-tune with the Double chin slider; 0 takes it off).")

    def remove_blemishes(self):
        from .. import engine as E
        if self.image is None or not self._faces():
            return
        img, e, path = self.image, self.edit_from_controls(), self.path
        self.viewer.busy = True
        self.viewer.update()
        self.status.emit("Looking for blemishes...")

        def done(p, res):
            if p != self.path:
                return
            self.viewer.busy = False
            spots = res.get("spots") or []
            moles = res.get("moles", 0)
            keep = f" {moles} mole{'s' if moles != 1 else ''} or beauty mark{'s' if moles != 1 else ''} left " \
                   f"alone." if moles else ""
            if not spots:
                self.status.emit("Remove blemishes: nothing found." + keep)
                self.viewer.update()
                return
            self._spots += spots
            self._ssel = -1
            self.rt_show.setChecked(True)
            self._spots_changed(f"Retouch: {len(spots)} blemishes")
            self.status.emit(f"Remove blemishes: {len(spots)} healed (amber circles, in Retouch).{keep} Alt-click "
                             f"a circle with the Heal tool to bring one back.")
        run_bg(lambda: E.find_blemishes(img, e), path, done,
               lambda _, err: (setattr(self.viewer, "busy", False), self.status.emit(f"Blemish problem: {err}")))

    def fix_red_eye(self):
        """Red pupils and blown-out pupils in the mapped eyes become Red eye / Pupil spots (amber)."""
        from .. import engine as E
        if self.image is None or not self._faces():
            self.status.emit("Fix red eye: find the faces first (Find faces, or Click the eyes).")
            return
        img, path = self.image, self.path
        self.viewer.busy = True
        self.viewer.update()
        self.status.emit("Looking for red eyes...")

        def work():
            e = self.edit_from_controls()
            faces = E.add_irises(img, e)
            e.portrait = dict(e.portrait or {}, faces=faces)
            return faces, E.find_red_eyes(img, e)

        def done(p, res):
            if p != self.path:
                return
            self.viewer.busy = False
            faces, found = res
            self._portrait["faces"] = faces
            spots = found.get("spots") or []
            if not spots:
                self.status.emit("Fix red eye: no red or blown-out pupils found in the mapped eyes. For one that was "
                                 "missed, or an animal's eye, use Retouch > Red eye or Pupil and click it.")
                self._portrait_changed("Portrait: irises found")
                return
            self._spots += spots
            self._ssel = -1
            self.rt_show.setChecked(True)
            self._spots_changed(f"Retouch: {len(spots)} red eye fix{'es' if len(spots) != 1 else ''}")
            self.status.emit(f"Fix red eye: {found['red']} red, {found['blown']} blown-out pupil(s) fixed (red "
                             f"circles in Retouch; Alt-click one with a Retouch tool to undo it).")
        run_bg(work, path, done,
               lambda _, err: (setattr(self.viewer, "busy", False), self.status.emit(f"Red eye problem: {err}")))

    def _ensure_irises(self):
        """Faces mapped before the eye tools have no irises: find them once, in the background, when an eye tool
        is first used."""
        from .. import engine as E
        if self.image is None or getattr(self, "_iris_job", False):
            return
        faces = self._faces()
        if not faces or all(f.get("irises") for f in faces):
            return
        self._iris_job = True
        img, path, e = self.image, self.path, self.edit_from_controls()

        def done(p, res):
            self._iris_job = False
            if p != self.path:
                return
            self._portrait["faces"] = res
            self._portrait_changed("Portrait: irises found")
        run_bg(lambda: E.add_irises(img, e), path, done, lambda *_: setattr(self, "_iris_job", False))

    # sculpting
    def _sculpt_stroke(self, uvs):
        if self.image is None or not uvs or not getattr(self, "sculpt_kind", None):
            return
        _, _, sw, sh = self.image.raw.crop
        long = float(max(sw, sh))
        r = round(self.sc_size.value() / 100.0 * self._brush_scale(), 5)
        R = r * long
        pts = self._s_points(uvs)
        st = self.sc_strength.value() / 100.0
        kind = self.sculpt_kind
        if kind == "push":
            if len(pts) < 2:
                return
            keep, vec = [], []
            a = pts[0]
            for q in pts[1:]:                                 # one push per quarter radius of the drag
                d = math.hypot((q[0] - a[0]) * sw, (q[1] - a[1]) * sh)
                if d >= 0.25 * R or q is pts[-1]:
                    keep.append([round(a[0], 5), round(a[1], 5)])
                    vec.append([round((q[0] - a[0]) * st, 6), round((q[1] - a[1]) * st, 6)])
                    a = q
            item = {"kind": "push", "pts": keep, "vec": vec, "r": r, "enabled": True}
        else:
            keep = [pts[0]]
            for q in pts[1:]:
                if math.hypot((q[0] - keep[-1][0]) * sw, (q[1] - keep[-1][1]) * sh) > 0.5 * R:
                    keep.append(q)
            item = {"kind": kind, "pts": keep, "r": r, "amount": round(0.5 * st, 3), "enabled": True}
        sc = list(self._portrait.get("sculpt") or [])
        sc.append(item)
        self._portrait["sculpt"] = sc
        self._portrait_changed(f"Sculpt: {kind}")

    def _shape_drawn(self, u0, v0, u1, v1):
        from .. import engine as E
        tool = self.viewer.shape_tool
        if self.image is None or not tool:
            return
        _, _, sw, sh = self.image.raw.crop
        long = float(max(sw, sh))
        (x0, y0), (x1, y1) = self._s_points([(u0, v0), (u1, v1)])
        a = np.array([x0 * sw, y0 * sh])
        b = np.array([x1 * sw, y1 * sh])
        L = float(np.hypot(*(b - a))) / 2.0
        if L < 2:
            return
        e1 = (b - a) / (2 * L)
        ratio = dict((t, r) for t, _, r, _ in self.SHAPES).get(tool, 0.5)
        up = E.shape_up(self.image, self.edit, 0.5 * (u0 + u1), 0.5 * (v0 + v1))
        defaults = {"breast": {"lift": 40, "size": 0}, "butt": {"lift": 40, "size": 20}}.get(tool, {"slim": 40})
        item = {"kind": "shape", "tool": tool, "c": [round((x0 + x1) / 2, 5), round((y0 + y1) / 2, 5)],
                "e1": [round(float(e1[0]), 5), round(float(e1[1]), 5)], "a": round(L / long, 5),
                "b": round(L * ratio / long, 5), "up": up, "lift": 0, "size": 0, "slim": 0, "enabled": True}
        item.update(defaults)
        sc = list(self._portrait.get("sculpt") or [])
        sc.append(item)
        self._portrait["sculpt"] = sc
        self._shsel = len(sc) - 1
        self.sc_show.setChecked(True)
        from ..engine.presets.looks import SHAPE_NAMES
        self._portrait_changed(f"Sculpt: {SHAPE_NAMES.get(tool, 'shape')} added")
        self.status.emit("Shape added with a gentle setting. Use the sliders below it; drag the oval to move it.")

    def _shape_picked(self, k):
        sh = self.viewer.shapes[k] if 0 <= k < len(self.viewer.shapes) else None
        self._shsel = sh["index"] if sh else -1
        self._refresh_portrait_ui()
        self._refresh_portrait_view()

    def _shape_moved(self, k, du, dv):
        sh = self.viewer.shapes[k] if 0 <= k < len(self.viewer.shapes) else None
        sc = self._portrait.get("sculpt") or []
        if sh is None or not (0 <= sh["index"] < len(sc)) or self.image is None:
            return
        x = sc[sh["index"]]
        from .. import engine as E
        cu, cv = E.s_to_view(self.image, self.edit, [x["c"][0]], [x["c"][1]])
        (nx, ny), = self._s_points([(float(cu[0]) + du, float(cv[0]) + dv)])
        x["c"] = [nx, ny]
        self._shsel = sh["index"]
        self._portrait_changed("Sculpt: shape moved")

    def _sculpt_shape_changed(self, *_):
        sc = self._portrait.get("sculpt") or []
        if not (0 <= self._shsel < len(sc)):
            return
        x = sc[self._shsel]
        x["lift"] = round(self.sh_lift.value(), 1)
        x["size"] = round(self.sh_size.value(), 1)
        x["slim"] = round(self.sh_slim.value(), 1)
        x["b"] = round(float(x.get("a", 0.05)) * self.sh_width.value() / 100.0, 5)
        self.controls_changed()
        self._refresh_portrait_view()

    def sculpt_undo(self):
        sc = list(self._portrait.get("sculpt") or [])
        if not sc:
            return
        sc.pop()
        self._portrait["sculpt"] = sc
        self._shsel = -1
        self._portrait_changed("Sculpt: removed")

    def shape_delete(self):
        sc = list(self._portrait.get("sculpt") or [])
        if not (0 <= self._shsel < len(sc)):
            return
        sc.pop(self._shsel)
        self._portrait["sculpt"] = sc
        self._shsel = -1
        self._portrait_changed("Sculpt: removed")

    def sculpt_clear(self):
        if not self._portrait.get("sculpt"):
            return
        self._portrait.pop("sculpt", None)
        self._shsel = -1
        self._portrait_changed("Sculpting cleared")

    # ------------------------------------------------------------------------------------ retouch
    def _set_spot_tool(self, kind):
        if self.image is None:
            kind = None
        for k, b in self.rt_btn.items():
            b.blockSignals(True)
            b.setChecked(k == kind)
            b.blockSignals(False)
        if kind:
            if self.viewer.tool in ("eyes", "sculpt", "shape"):
                self._set_portrait_tool(None)
            elif self.viewer.tool and self.viewer.tool != "spot":
                self._set_mask_tool(None)
            if self.viewer.crop_mode:
                self.crop_b.setChecked(False)
            self.rt_kind = kind
            self.viewer.tool = "spot"
            self.viewer.brush_scale = self._brush_scale()
            self.viewer.brush_frac = self.rt_size.value() / 100.0
            self.viewer.setCursor(Qt.CursorShape.BlankCursor)
            if self._ssel < 0:
                self.rt_feather.set_value({"heal": 25, "clone": 50, "pupil": 25}.get(kind, 30), emit=False)
            if not self.viewer.zoomed:
                self.status.emit("Retouch: zoom to 100% (Z or the mouse wheel) to see small dust spots.")
        elif self.viewer.tool == "spot":
            self.viewer.tool = None
            self.viewer.setCursor(Qt.CursorShape.ArrowCursor)
        self._refresh_spot_view()

    def _show_spots(self, on):
        self.viewer.show_spots = bool(on)
        self.viewer.update()

    def _map_chip(self):
        from ..engine.local import dust as du
        m = self.image.meta if self.image is not None else None
        dm = du.load_map(getattr(m, "make", "") or "", getattr(m, "model", "") or "") if m else None
        if dm:
            self.rt_map_chip.setText(f"Dust map: {len(dm['spots'])} spots, learned on this PC from {dm['photos']} "
                                     f"photos ({dm.get('made', '')})")
            self.rt_map_chip.setStyleSheet(f"color:{T.CYAN}; border-color:{T.alpha(T.CYAN, 0.4)};")
        else:
            self.rt_map_chip.setText("Dust map: none for this camera yet (Library: Learn dust map)")
            self.rt_map_chip.setStyleSheet("")
        self.rt_map.setEnabled(bool(dm))

    def _refresh_spots(self):
        sel = self._spots[self._ssel] if 0 <= self._ssel < len(self._spots) else None
        self.rt_del.setEnabled(sel is not None)
        self.rt_clear.setEnabled(bool(self._spots))
        if sel is not None:
            self.rt_size.set_value(float(sel.get("size", 0.008)) * 100.0, emit=False)
            self.rt_feather.set_value(float(sel.get("feather", 0.25)) * 100.0, emit=False)
            self.rt_opacity.set_value(float(sel.get("opacity", 100)), emit=False)
        n = len(self._spots)
        auto = sum(1 for sp in self._spots if sp.get("auto"))
        self.rt_info.setText("No spots" if not n else f"{n} spot{'s' if n != 1 else ''}"
                             + (f" ({auto} found automatically, amber)" if auto else ""))
        self._refresh_spot_view()

    def _refresh_spot_view(self):
        """Where the spots sit on the displayed photo (they are stored on the photo itself)."""
        from .. import engine as E
        if self.image is None or not self._spots:
            self.viewer.spots = []
            self.viewer.update()
            return
        e = self.shown_edit()
        _, _, sw, sh = self.image.raw.crop
        bs = self._brush_scale()
        out = []
        for k, sp in enumerate(self._spots):
            pts = np.asarray(sp["pts"], float).reshape(-1, 2)
            dx, dy = sp.get("src") or (0.0, 0.0)
            u, v = E.s_to_view(self.image, e, pts[:, 0], pts[:, 1])
            d = {"pts": list(zip(np.atleast_1d(u), np.atleast_1d(v))), "r": float(sp.get("size", 0.008)) / bs,
                 "kind": sp.get("kind"), "sel": k == self._ssel, "auto": sp.get("auto")}
            if sp.get("kind") not in ("redeye", "pupil"):
                su, sv = E.s_to_view(self.image, e, pts[:, 0] + dx, pts[:, 1] + dy)
                d["src"] = list(zip(np.atleast_1d(su), np.atleast_1d(sv)))
            out.append(d)
        self.viewer.spots = out
        self.viewer.update()

    def _spots_changed(self, label=None):
        self.commit(self.edit_from_controls(), label=label)
        self._refresh_spots()

    def _spot_new(self, uvs):
        from .. import engine as E
        if self.image is None or not uvs:
            return
        pts = self._s_points(uvs)
        if self.edit.portrait:                               # on a sculpted photo: where the clicked pixels come from
            xs, ys = E.sculpt_source(self.image, self.edit, [q[0] for q in pts], [q[1] for q in pts])
            pts = [[round(float(a), 5), round(float(b), 5)] for a, b in zip(np.atleast_1d(xs), np.atleast_1d(ys))]
        _, _, sw, sh = self.image.raw.crop
        size = round(self.rt_size.value() / 100.0, 5)
        R = size * max(sw, sh)
        keep = [pts[0]]                                       # thin a brushed stroke to steps of half a radius
        for q in pts[1:]:
            if math.hypot((q[0] - keep[-1][0]) * sw, (q[1] - keep[-1][1]) * sh) > 0.5 * R:
                keep.append(q)
        if len(pts) > 1 and keep[-1] != pts[-1]:
            keep.append(pts[-1])
        spot = {"kind": self.rt_kind, "pts": keep, "size": size, "feather": round(self.rt_feather.value() / 100.0, 3),
                "opacity": round(self.rt_opacity.value(), 1), "src": [0.0, 0.0], "enabled": True, "auto": "", "v": 2}

        path = self.path

        def add(_, off):
            if self.path != path:
                return
            spot["src"] = off
            self._spots.append(spot)
            self._ssel = len(self._spots) - 1
            self.viewer.busy = False
            self._spots_changed({"heal": "Heal spot", "clone": "Clone spot", "redeye": "Red eye",
                                 "pupil": "Pupil"}[spot["kind"]])
        if spot["kind"] in ("redeye", "pupil"):
            add(None, [0.0, 0.0])
            return
        img, e, others = self.image, self.edit_from_controls(), [dict(x) for x in self._spots]
        self.viewer.busy = True
        self.viewer.update()
        run_bg(lambda: E.auto_source(img, e, spot, others), self.path, add,
               lambda _, err: (setattr(self.viewer, "busy", False), self.status.emit(f"Retouch problem: {err}")))

    def _spot_picked(self, k):
        self._ssel = k
        self._refresh_spots()

    def _spot_moved(self, k, which, du, dv):
        from .. import engine as E
        from ..engine.local import retouch as rt
        if not (0 <= k < len(self._spots)) or self.image is None:
            return
        sp = self._spots[k]
        e = self.shown_edit()
        pts = np.asarray(sp["pts"], float).reshape(-1, 2)
        dx, dy = sp.get("src") or (0.0, 0.0)
        ax, ay = (pts[0, 0], pts[0, 1]) if which == "dst" else (pts[0, 0] + dx, pts[0, 1] + dy)
        u, v = E.s_to_view(self.image, e, ax, ay)
        nx, ny = E.view_to_s(self.image, e, float(u) + du, float(v) + dv)
        mx, my = float(nx) - ax, float(ny) - ay
        _, _, sw, sh = self.image.raw.crop
        if which == "dst":                                    # the spot moves; its source stays where it was
            sp["pts"] = [[round(p[0] + mx, 6), round(p[1] + my, 6)] for p in pts]
            sp["src"] = [dx - mx, dy - my]
        else:
            sp["src"] = [dx + mx, dy + my]
        if sp.get("kind") not in ("redeye", "pupil"):
            sp["src"] = rt.clamp_source(sp, sw, sh)
        sp["auto"] = sp.get("auto", "")
        self._ssel = k
        self._spots_changed("Spot moved" if which == "dst" else "Spot source moved")

    def _spot_delete(self, k):
        if not (0 <= k < len(self._spots)):
            return
        self._spots.pop(k)
        self._ssel = -1
        self._spots_changed("Spot removed")

    def clear_spots(self):
        if not self._spots:
            return
        self._spots = []
        self._ssel = -1
        self._spots_changed("Retouch cleared")

    def _spot_params_changed(self, *_):
        if self.viewer.tool == "spot":
            self.viewer.brush_frac = self.rt_size.value() / 100.0
            self.viewer.update()
        if not (0 <= self._ssel < len(self._spots)):
            return
        sp = self._spots[self._ssel]
        sp["size"] = round(self.rt_size.value() / 100.0, 5)
        sp["feather"] = round(self.rt_feather.value() / 100.0, 3)
        sp["opacity"] = round(self.rt_opacity.value(), 1)
        self.controls_changed()
        self._refresh_spot_view()

    def _add_found(self, spots, what):
        self.viewer.busy = False
        if spots is None:
            self.status.emit("No dust map for this camera yet. In the Library, select 5-20 photos of different "
                             "scenes (sky or a plain wall at f/11 or smaller is ideal) and press Learn dust map.")
            return
        if not spots:
            self.status.emit(f"{what}: nothing found.")
            return
        self._spots += spots
        self._ssel = -1
        self.rt_show.setChecked(True)
        self._spots_changed(f"Retouch: {len(spots)} dust spots")
        self.status.emit(f"{what}: {len(spots)} spot{'s' if len(spots) != 1 else ''} healed (amber circles). Check "
                         f"them at 100%: Alt-click removes a wrong one.")

    def find_dust(self):
        from .. import engine as E
        if self.image is None:
            return
        img, e, path = self.image, self.edit_from_controls(), self.path
        self.viewer.busy = True
        self.viewer.update()
        self.status.emit("Looking for dust...")
        run_bg(lambda: E.find_dust(img, e), path,
               lambda p, sp: self._add_found(sp, "Find dust") if p == self.path else None,
               lambda _, err: (setattr(self.viewer, "busy", False), self.status.emit(f"Find dust problem: {err}")))

    def dust_from_map(self):
        from .. import engine as E
        if self.image is None:
            return
        img, e, path = self.image, self.edit_from_controls(), self.path
        self.viewer.busy = True
        self.viewer.update()
        self.status.emit("Finding the dust map's spots in this photo...")
        run_bg(lambda: E.dust_from_map(img, e), path,
               lambda p, sp: self._add_found(sp, "Dust map") if p == self.path else None,
               lambda _, err: (setattr(self.viewer, "busy", False), self.status.emit(f"Dust map problem: {err}")))

    def _rendered(self, tok, res):
        current = tok == self.token and self._render_img is self.image
        more = self._render_finished()           # newer settings start now; this result is still shown meanwhile
        if not current:
            if not more and not self._render_live:
                self.viewer.busy = False
            return
        a8, b8, bkey, dt = res
        self.viewer.after = to_qimage(a8)
        self._refresh_spot_view()
        self._refresh_portrait_view()
        self._glow_refresh()
        self._dn_safe_refresh()
        self._hdr_refresh()
        if b8 is not None:
            self.before_img = to_qimage(b8)
            self.before_key = bkey
        self.viewer.before = self.before_img
        if not more:
            self.viewer.busy = False
        self.viewer.update()
        self.histo.set_image(a8)
        self.last_a8 = a8
        self.curve.set_histogram(a8)
        self.status.emit(f"Rendered in {dt:.1f} s")

    def _photo_clicked(self, u, v):
        """A click on the photo while an eyedropper waits (other clicks zoom, in the viewer itself)."""
        if self.image is None or self.viewer.crop_mode:
            return
        if self.pick == "wb":
            self._pick_white_at(u, v)
        elif self.pick == "range":
            self._pick_range_at(u, v)

    def _escape(self, keep_zoom: bool = False):
        """Esc: put down the eyedropper or tool in hand; with nothing in hand, back to the whole photo."""
        busy_tool = bool(self.pick) or bool(self.viewer.tool)
        if self.pick:
            self._set_pick(None)
        if self.viewer.tool in ("eyes", "sculpt", "shape"):
            self._set_portrait_tool(None)
        elif self.viewer.tool == "spot":
            self._set_spot_tool(None)
        elif self.viewer.tool:
            self._set_mask_tool(None)
        if not busy_tool and not keep_zoom and self.viewer.zoomed:
            self.viewer.set_zoom(None)
        self.viewer.update()

    def toggle_before(self, on):
        if on and self.viewer.zoomed:
            self.viewer.set_zoom(None)              # the wipe compares the whole photo
        self.viewer.split = 0.5 if on else None
        self.viewer.update()

    def make_it_good(self):
        """Automatic clean-up: noise for the ISO, lens fixes, conservative capture sharpening (not for character
        lenses). Never touches brightness (Bill's rule). The settings come from the engine (presets/auto.py), which
        the Library's Process picks uses too."""
        if self.image is None:
            return
        from ..engine.presets.auto import make_it_good_settings
        e, character = make_it_good_settings(self.image.meta, self.edit_from_controls())
        self.commit(e, label="Make it good")
        note = "character lens: sharpening left off" if character else "gentle capture sharpening"
        self.status.emit(f"Make it good: noise set for {self.image.meta.iso_label}, {note}, fixing the lens...; "
                         f"brightness untouched.")
        self.fix_lens()

    def reset(self):
        from ..engine import Edit
        if self.image is None:
            return
        e = Edit()
        e.lens.tag = self.edit.lens.tag                    # which lens took it is a fact, not an edit
        self.show_fix_report([])
        self.commit(e, "All edits reset (the lens tag is kept).", label="Reset all")

    def do_undo(self):
        self.jump_history(self.hpos - 1)

    def do_redo(self):
        self.jump_history(self.hpos + 1)

    def save_sidecar(self):
        if self.image is None or self.path is None:
            return
        from .. import engine as E
        try:
            E.save_version(self.path, self.version, self.edit)
            E.save_history(self.path, self.version, self.steps[:self.hpos + 1])
        except (OSError, IndexError, KeyError) as err:
            self.status.emit(f"Could not save the sidecar (read-only drive?): {err}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.image is not None:
            self.timer.start(300)


# ============================================================================================ Export
# ============================================================================================ other workspaces
class InfoPage(QWidget):
    """A page of text cards (Help, Cameras, Lenses...). It scrolls: without that, a window shorter than all the
    cards squeezed every card to a sliver of clipped text (a friend's first look at Help, on Linux), and the
    page's height became the whole window's minimum (1,042 pixels, taller than a laptop screen)."""

    def __init__(self, title, subtitle, phase=None, bullets=(), colour=T.CYAN):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("infopage")
        scroll.setWidget(content)
        outer.addWidget(scroll)
        self.scroll = scroll
        v = QVBoxLayout(content)
        v.setContentsMargins(40, 34, 40, 34)
        head = QHBoxLayout()
        head.addWidget(label(title, "h1"))
        if phase:
            head.addSpacing(12)
            head.addWidget(chip(phase, colour))
        head.addStretch(1)
        v.addLayout(head)
        v.addWidget(label(subtitle, "dim", wrap=True))
        v.addSpacing(16)
        self.cards = QGridLayout()
        self.cards.setSpacing(14)
        for i, (name, text) in enumerate(bullets):
            c = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(18, 16, 18, 16)
            cl.addWidget(label(name, "h2"))
            cl.addWidget(label(text, "dim", wrap=True))
            cl.addStretch(1)
            self.cards.addWidget(c, i // 3, i % 3)
        for col in range(3):
            self.cards.setColumnStretch(col, 1)        # equal columns (a narrow one made tall, half-empty rows)
        v.addLayout(self.cards)
        self.body = QVBoxLayout()
        v.addLayout(self.body)
        v.addStretch(1)


class CamerasPage(InfoPage):
    status = Signal(str)

    def __init__(self):
        super().__init__("Cameras", "Teach the Studio your camera. A profile is numbers only: noise, banding, "
                         "black level and gain per ISO. It stays on this PC unless you choose to share it.",
                         "Calibration engine ready", [
                             ("Quick", "Dark frames only (body cap on), every ISO up and back down. About 15 "
                                       "minutes. Gives read noise, banding and hot pixels."),
                             ("Full", "Adds pairs of evenly lit flats at each ISO: measures gain, so the denoiser "
                                      "knows the noise exactly (cyan)."),
                             ("Complete", "Adds the exposure-latitude series (ETTR / ETTL). Coming with the "
                                          "wizard in Phase 7.")])
        row = QHBoxLayout()
        b = QPushButton("Calibrate a camera from a folder of frames...")
        b.setObjectName("primary")
        b.clicked.connect(self.calibrate)
        row.addWidget(b)
        row.addStretch(1)
        self.body.addSpacing(10)
        self.body.addLayout(row)
        self.out = label("", "dim", wrap=True)
        self.out.setTextFormat(Qt.TextFormat.RichText)
        self.body.addWidget(self.out)
        self.refresh()

    def refresh(self):
        import glob
        rows = []
        for d, tier, col in ((paths.get("CALIBRATION_LOCAL") / "profiles", "calibrated on this PC", T.CYAN),
                             (paths.PROFILES / "cameras", "shared", T.VIOLET)):
            for f in sorted(glob.glob(str(d / "*.json"))):
                try:
                    p = json.load(open(f, encoding="utf-8"))
                    rows.append(f"<span style='color:{col}'>●</span> <b>{p.get('model')}</b> - {p.get('method')}"
                                f" profile, {len(p.get('isos', []))} ISOs ({tier})")
                except Exception:
                    pass
        self.out.setText("<br>".join(rows) or "No camera profiles yet. Until there is one, the noise is estimated "
                                               "from each photo (<span style='color:#f59e0b'>amber</span>).")

    def calibrate(self):
        d = QFileDialog.getExistingDirectory(self, "Folder of calibration frames")
        if not d:
            return
        self.out.setText("Reading the frames... (this can take a few minutes)")

        def work():
            from ..cli import collect
            from ..engine.calib.builder import build_profile
            from ..engine.io.rawfile import open_raw
            prof, rep = build_profile([open_raw(f) for f in collect([d], recursive=True)])
            return prof.save(), prof, rep
        run_bg(work, d, self._done, lambda _, e: self.out.setText(f"Calibration problem: {e}"))

    def _done(self, _, res):
        path, prof, rep = res
        lines = [f"<b>{prof.model}</b>: {prof.method} profile, quality {prof.quality.get('score')}. "
                 f"Frames: {rep.counts()}"]
        for e in prof.isos:
            g = f"{e.gain_e_per_dn:.4f} e-/DN" if e.gain_e_per_dn else "gain: quick profile"
            lines.append(f"ISO {e.iso:g}: read noise {np.mean(e.read_noise_dn):.2f} DN, row banding "
                         f"{e.row_banding_dn or 0:.2f} DN, {g}, {e.hot_pixels} hot pixels")
        lines += [f"<span style='color:#f59e0b'>note:</span> {w}" for w in rep.warnings]
        self.out.setText("<br>".join(lines))
        self.refresh()


class LensesPage(InfoPage):
    """Lens tools: what Fix lens does, the lensfun status, and the lens list for tagging contactless glass."""

    def __init__(self, settings):
        super().__init__("Lenses", "Automatic fixes that work even on contactless vintage glass, the lenses you "
                         "tag photos with, and (Phase 12) lens rendering signatures.", "Phase 5", [
                             ("Fix lens", "One button in Develop: lateral colour measured from the photo, purple "
                                          "fringes, distortion from a lens profile or from straight lines (never "
                                          "guessed on organic subjects), vignetting measured."),
                             ("Vignetting", "Measured, but correction OFF by default: your lens's character "
                                            "stays."),
                             ("Tag your lenses", "Adapted lenses record nothing. Select a shoot in the Library and "
                                                 "press Tag lens: it goes in each sidecar and into exports."),
                             ("Shape", "Straighten (with Auto level), perspective and crop in Develop: one "
                                       "resample for all of them."),
                             ("Recover lens sharpness", "Capture sharpening by deconvolution, off for lenses you "
                                                        "mark as character lenses."),
                             ("Render like my lens", "Laack, Heliar, Apo-Lanthar, Otus... an open signature "
                                                     "library (Phase 12).")], T.VIOLET)
        self.settings = settings
        from ..engine.lens import lensfun_bridge
        ok = lensfun_bridge.available()
        self.body.addWidget(chip(lensfun_bridge.status(), T.VIOLET if ok else T.DIM))
        row = QHBoxLayout()
        b = QPushButton("Add a lens...")
        b.clicked.connect(self.add_lens)
        row.addWidget(b)
        row.addStretch(1)
        self.body.addSpacing(8)
        self.body.addLayout(row)
        self.out = label("", "dim", wrap=True)
        self.out.setTextFormat(Qt.TextFormat.RichText)
        self.body.addWidget(self.out)
        self.refresh()

    def refresh(self):
        from ..engine.lens import library as lib
        mine = {e.name.lower() for e in lib._read(lib.mine_path())}
        rows = []
        for e in lib.list_lenses():
            col = T.CYAN if e.name.lower() in mine else T.VIOLET
            bits = [f"{e.focal_mm:g} mm" if e.focal_mm else "", f"f/{e.max_aperture:g}" if e.max_aperture else "",
                    "character lens" if e.character else ""]
            rows.append(f"<span style='color:{col}'>●</span> <b>{e.name}</b> "
                        f"{' · '.join(b for b in bits if b)}")
        self.out.setText(f"<b>Lens list</b> (<span style='color:{T.VIOLET}'>shared starter list</span>, "
                         f"<span style='color:{T.CYAN}'>yours</span>):<br>" + "<br>".join(rows))

    def add_lens(self):
        dlg = LensDialog(self, "", 0)
        dlg.setWindowTitle("Add a lens")
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.refresh()


class HelpPage(InfoPage):
    update_mode = Signal(str)                     # "off" | "notify" | "auto"
    update_check = Signal()

    UPDATE_MODES = (("off", "Off: never check"), ("notify", "Tell me when there is a new version"),
                    ("auto", "Update automatically"))

    def __init__(self):
        super().__init__("Help", f"Photographic Studio By Bill {__version__}. Free forever, GPL-3.0. Your photos "
                         "never leave this PC.", None, [
                             ("Library", "Point it at a folder (the tree, Open folder, or drop a folder on the "
                                         "window). Double-click or Enter develops. 0-5 rate, P pick, X reject, U "
                                         "clear. Filter by rating, camera, lens, ISO, date or name; sort by time, "
                                         "name, rating or ISO."),
                             ("Import from card", "Copies the photos from a card into your Photos folder, one "
                                                  "folder per shooting day (named by the date, or your own name). "
                                                  "Raws only unless you tick JPEGs or Videos. Every file is "
                                                  "checked; the card is never changed; a second import copies "
                                                  "only what is new."),
                             ("Review", "Space opens the photo large. Left / Right move, 0-5 / P / X / U rate (and "
                                        "go on to the next), Z or a click is 100%, C compares two side by side, "
                                        "Enter develops, Space or Esc goes back. Show: Camera preview (instant) or "
                                        "From the raw (true noise and detail)."),
                             ("Process picks", "Exports your picks (or the selected photos) in the background, each "
                                               "with its own edit, or Make it good on photos not edited yet."),
                             ("Develop", "B before/after, Ctrl+Z / Ctrl+Y undo and redo, double-click a slider "
                                         "to reset it, Esc puts a tool down, F11 full screen. On a narrow window "
                                         "Copy, Paste, Before / After and the virtual copies are under the ⋯ "
                                         "button."),
                             ("Colours mean something", "Cyan = measured on your camera. Violet = shared profile. "
                                                        "Amber = estimated from the photo."),
                             ("Make it good", "Noise clean-up for the photo's ISO, Fix lens, and gentle "
                                              "capture sharpening (not for character lenses). It never changes "
                                              "brightness."),
                             ("Shape and detail", "C draws a crop, Enter finishes it. Microcontrast lifts the "
                                                  "shades of shadow; judge sharpening at 100% (click the photo)."),
                             ("Composition", "O cycles the overlays (thirds, golden ratio, golden spiral, triangles, "
                                             "diagonals, dynamic symmetry, armature, rabatment, centre, radial, "
                                             "grid); Shift+O turns the spiral or triangles. Suggest crops (in the "
                                             "Crop group) puts the eyes or subject on the overlay's strong points."),
                             ("Light and colour", "Auto in Light never touches exposure. Pick white: press it, "
                                                  "then click something neutral. The colour wheel moves one range "
                                                  "of colours; HSL and grading are in Full."),
                             ("Looks, copies, history", "Looks change only what they name. Ctrl+Shift+C / V copy "
                                                        "and paste settings (Library: Paste settings for many). "
                                                        "Virtual copies and snapshots live in the sidecar."),
                             ("Masks", "Change one part of a photo: Brush, Gradient, Radial, Object (click it, "
                                       "Alt-click what is not), Colour, Brightness or Skin (then Smooth skin). Esc "
                                       "ends a tool; Show mask turns the red overlay on."),
                             ("Lens glow", "Light > Clipped highlights: measure from lens glow reads how bright a "
                                           "blown light really was from the glow around it. Measure each lens once: "
                                           "Library > Measure lens glow (a bracket of a small light in the dark)."),
                             ("Eyes", "After Find faces, each iris is found too (Show faces draws it). Portrait: "
                                      "Eye colour (a preset or Pick a colour, then its amount; both eyes or one), and "
                                      "in Full the iris brightness, clarity, colour strength, outer ring and "
                                      "catchlights. Fix red eye fixes red pupils and pupils the flash blew out white; "
                                      "for an animal's glowing eyes use Retouch > Pupil and click each."),
                             ("Retouch", "Heal, Clone, Red eye: click a spot or brush along a hair; drag a circle "
                                         "or its dashed source to move it; Alt-click removes. Find dust looks in one "
                                         "photo; Learn dust map (Library) learns your sensor's dust from several."),
                             ("Portrait", "Find faces maps each face on this PC (or Click the eyes for a face in "
                                          "deep shadow). Remove blemishes heals spots as Retouch spots (moles stay). "
                                          "Younger moves every face slider at once; each slider is subtle at 100. Remove double "
                                          "chin lifts the soft area under the jaw (skin only)."),
                             ("Person mask", "Masks > Person selects the people in the photo (found on this PC); "
                                             "then pick a part: skin, face, hair, clothes, arms, legs, torso or the "
                                             "background. Edges snap to the photo."),
                             ("Body sculpting", "Push, Shrink, Grow brushes; shape tools (breast lift & size, butt "
                                                "improvement tool, slim waist, tummy, arm, thigh): drag along the "
                                                "area, then set Lift, Size or Slim. Every warp is capped."),
                             ("Zoom and move", "Click the photo for 100% there, click again for the whole photo. "
                                               "The mouse wheel zooms around the pointer, from the whole photo to "
                                               "400%. Drag to move (with a tool on: middle- or right-drag, or "
                                               "Space + drag). The map in the corner jumps where you click. Z, "
                                               "Ctrl+= / Ctrl+-, Ctrl+0 whole photo, Ctrl+1 100%, Esc whole photo. "
                                               "The photo moves at once; the sharp detail follows a moment later."),
                             ("Sidecars", "Edits live in NAME.CR3.psb.json beside each raw. The raw is never "
                                          "touched."),
                             ("Working...", "While the photo is being worked on, a bar moves along its top edge and "
                                            "the chip beside Controls turns amber; big raws on slower PCs can take a "
                                            "few seconds per change."),
                             ("Problems?", "Copy the diagnostic report below and attach it. It holds no photos, "
                                           "serial numbers or locations.")])
        b = QPushButton("Copy diagnostic report")
        b.clicked.connect(self.copy_diag)
        row = QHBoxLayout()
        row.addWidget(b)
        row.addStretch(1)
        self.body.addSpacing(10)
        self.body.addLayout(row)
        # Updates (off until turned on: the only thing it sends is a question to GitHub for the newest version)
        self.body.addSpacing(16)
        self.body.addWidget(label("Check for updates", "h2"))
        self.body.addWidget(label("Asks GitHub for the newest version number, at most once a day. Nothing else is "
                                  "sent: never a photo, a file name or anything about you. Updates keep your "
                                  "settings, calibrations and photos.", "dim", wrap=True))
        self.upd_combo = QComboBox()
        for _, text in self.UPDATE_MODES:
            self.upd_combo.addItem(text)
        self.upd_combo.activated.connect(lambda i: self.update_mode.emit(self.UPDATE_MODES[i][0]))
        self.upd_now = QPushButton("Check now")
        self.upd_now.clicked.connect(self.update_check.emit)
        row = QHBoxLayout()
        row.addWidget(self.upd_combo)
        row.addWidget(self.upd_now)
        row.addStretch(1)
        self.body.addLayout(row)
        self.upd_info = label("", "dim", wrap=True)
        self.body.addWidget(self.upd_info)

    def set_update_state(self, mode: str, kind: str):
        keys = [k for k, _ in self.UPDATE_MODES]
        self.upd_combo.setCurrentIndex(keys.index(mode) if mode in keys else 0)
        how = {"git": "This is a git copy (a developer's own): you will be told about new versions, and update it "
                      "with git yourself.",
               "msi": "Installed with the MSI: an update downloads the new MSI and Windows Installer puts it in.",
               "bat": "Installed with Install Photographic Studio.bat: updates go in when the Studio restarts.",
               "sh": "Installed with install-linux.sh: updates go in when the Studio restarts.",
               "folder": "Updates go in when the Studio restarts."}.get(kind, "")
        self.upd_info.setText(how)

    def set_update_info(self, text: str):
        self.upd_info.setText(text)

    def copy_diag(self):
        QApplication.clipboard().setText(slog.diagnostic_report())
        QMessageBox.information(self, "Diagnostic report", "Copied to the clipboard.")


# ============================================================================================ main window
NAV = [("▦", "Library"), ("◐", "Develop"), ("◫", "Dual Pixel Lab"), ("≋", "Focus Stack"), ("✦", "Super-Res Stack"),
       ("◎", "Lenses"), ("▣", "Cameras"), ("⇪", "Export"), ("?", "Help")]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.setWindowTitle(f"Photographic Studio By Bill  {__version__}")
        self.resize(1600, 980)
        root = QWidget()
        root.setObjectName("root")
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(240)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 12)
        sv.addWidget(label("Photographic Studio", "brand"))
        sv.addWidget(label("By Bill  ·  free raw studio", "brandsub"))
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        for icon, name in NAV:
            QListWidgetItem(f"{icon}    {name}", self.nav)
        sv.addWidget(self.nav, 1)
        self.upd_card = QFrame()                       # updates: asked once, then shown only when there is news
        self.upd_card.setObjectName("card")
        self.upd_card.setStyleSheet(f"QFrame#card {{ border: 1px solid {T.alpha(T.AMBER, 0.5)}; }}")
        uv = QVBoxLayout(self.upd_card)
        uv.setContentsMargins(12, 10, 12, 10)
        uv.setSpacing(6)
        self.upd_text = label("", None, wrap=True)
        self.upd_text.setStyleSheet(f"color:{T.AMBER};")
        uv.addWidget(self.upd_text)
        self.upd_btns = QVBoxLayout()
        self.upd_btns.setSpacing(4)
        uv.addLayout(self.upd_btns)
        wrap = QWidget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(12, 0, 12, 8)
        wl.addWidget(self.upd_card)
        sv.addWidget(wrap)
        self.upd_card.hide()
        self.upd_kind, self.upd_rel, self.upd_msi = "folder", None, None
        self.jobs = label("", "dim")
        self.jobs.setContentsMargins(18, 0, 18, 0)
        sv.addWidget(self.jobs)
        h.addWidget(side)
        self.stack = QStackedWidget()
        self.library = LibraryPage(self.settings)
        self.develop = DevelopPage(self.settings)
        self.cameras = CamerasPage()
        self.lenses = LensesPage(self.settings)
        self.help = HelpPage()
        self.help.update_mode.connect(self._set_update_mode)
        self.help.update_check.connect(lambda: self.check_updates(manual=True))
        pages = [
            self.library, self.develop,
            InfoPage("Dual Pixel Lab", "Everything Canon Dual Pixel RAW can do: split A/B, +1 stop highlight "
                     "recovery, depth maps, focus check, micro-refocus, 3D output.", "Phase 8", [
                         ("Split and view", "A, B, A+B and the difference, built in (what DPRSplit does)."),
                         ("Highlight recovery", "About one extra stop in candle flames: A clips later."),
                         ("Noise vs detail map", "Halves agree on detail, disagree on noise (0.96 vs 0.02 "
                                                 "measured on your eagle file)."),
                         ("Depth masks", "Subject / background masks from the A-B offset."),
                         ("Air steadiness", "Lab: turbulence read-out at extreme range (your observation)."),
                         ("3D output", "Wigglegram, anaglyph, depth map export.")], T.VIOLET),
            InfoPage("Focus Stack", "Merge a focus bracket into one sharp image, straight from the raws.", "Phase 10",
                     [("Align", "Scale + shift + rotation for focus breathing, sub-pixel."),
                      ("Merge", "Depth-map, pyramid, or auto; noise-aware weights."),
                      ("Fix-ups", "Halo suppression, retouch from any frame, gap detection.")], T.VIOLET),
            InfoPage("Super-Res Stack", "Handheld bursts to higher resolution and lower noise. Moon and night sky "
                     "modes.", "Phase 11", [
                         ("Drizzle", "Hubble's method for undersampled, dithered frames."),
                         ("Lucky imaging", "Moon: keep the sharpest regions of hundreds of frames."),
                         ("Reach", "Burst + 2x merge + diffraction recovery: a longer lens for static "
                                   "subjects.")], T.VIOLET),
            self.lenses,
            self.cameras,
            InfoPage("Export", "Select photos in the Library and press Export, or use Export in Develop. 16-bit "
                     "TIFF in Adobe RGB by default; PNG and JPEG too.", None, [
                         ("Send to Photoshop", "Phase 13: TIFF hand-off and a Photoshop panel."),
                         ("Send to PhotoCraft", "Phase 13."),
                         ("Privacy", "Serial numbers and GPS are never written.")]),
            self.help,
        ]
        for p in pages:
            self.stack.addWidget(p)
        h.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        QShortcut(QKeySequence("F11"), self, activated=self.toggle_full_screen)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.currentRowChanged.connect(self._nav_changed)
        self.nav.setCurrentRow(0)
        self.library.open_photo.connect(self.open_in_develop)
        self.library.lens_tagged.connect(self._lens_tagged)
        self.library.paste_requested.connect(self.paste_to_files)
        self.library.export_requested.connect(self.export)
        self.library.save_open_edit = self.develop.save_sidecar
        self.develop.status.connect(lambda t: self.statusBar().showMessage(t, 6000))
        self.develop.export_requested.connect(self.export)
        self.cameras.status.connect(lambda t: self.statusBar().showMessage(t, 6000))
        exp = QAction("Export", self)
        exp.setShortcut(QKeySequence("Ctrl+E"))
        exp.triggered.connect(lambda: self.export(
            self.library.selected_files() if self.stack.currentWidget() is self.library and
            self.library.selected_files() else ([(self.develop.path, self.develop.version)] if self.develop.path
                                                else self.library.selected_files())))
        self.addAction(exp)
        self.statusBar().showMessage("Ready. Photos never leave this PC.")
        self.tick = QTimer(self, interval=500, timeout=self._jobs)
        self.tick.start()
        # the last folder; if none, the Photos folder (section 24.3); else the app folder (its sample raws)
        folder = self.settings.get("folder") or self.settings.get("photos_dir") or str(paths.APP_ROOT)
        if os.path.isdir(folder):
            QTimer.singleShot(50, lambda: self.library.load_folder(folder))
        QTimer.singleShot(1500, self._updates_start)
        QTimer.singleShot(2500, self.library.start_card_watch)

    # ------------------------------------------------------------------------------------ updates
    def _upd_show(self, text, buttons=()):
        """The sidebar update card: a message and its buttons [(text, callback, primary?)]; empty text hides it."""
        while self.upd_btns.count():
            w = self.upd_btns.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        if not text:
            self.upd_card.hide()
            return
        self.upd_text.setText(text)
        for t, fn, primary in buttons:
            b = QPushButton(t)
            if primary:
                b.setObjectName("primary")
            b.clicked.connect(fn)
            self.upd_btns.addWidget(b)
        self.upd_card.show()

    def _updates_start(self):
        from .. import updates as U
        self.upd_kind = U.install_kind()
        mode = self.settings.get("updates")
        self.help.set_update_state(mode or "off", self.upd_kind)
        st = U.staged()
        if st and self.upd_kind in ("bat", "sh", "folder"):
            self._upd_show(f"Update {st} is downloaded. It goes in when the Studio restarts.",
                           [("Restart now", self._upd_restart, True), ("Later", lambda: self._upd_show(""), False)])
            return
        if mode is None:
            if self.upd_kind != "git":                 # asked once; a developer's git copy is never asked
                self._upd_show("Keep Photographic Studio up to date? It only asks GitHub for the newest version "
                               "number: no photos, nothing about you.",
                               [("Update automatically", lambda: self._set_update_mode("auto"), True),
                                ("Just tell me", lambda: self._set_update_mode("notify"), False),
                                ("No thanks", lambda: self._set_update_mode("off"), False)])
            return
        last = float(self.settings.get("updates_last", 0) or 0)
        if mode in ("notify", "auto") and time.time() - last > U.CHECK_EVERY_S:
            self.check_updates(manual=False)

    def _set_update_mode(self, mode):
        self.settings.set("updates", mode)
        self.help.set_update_state(mode, self.upd_kind)
        log.info("updates: %s", mode)
        self._upd_show("")
        if mode != "off":
            self.check_updates(manual=False)

    def check_updates(self, manual=False):
        from .. import updates as U
        self.help.set_update_info("Checking GitHub for the newest version...")

        def done(_, rel):
            self.settings.set("updates_last", time.time())
            self.upd_rel = rel
            if not rel["newer"]:
                self.help.set_update_info(f"You have the newest version ({__version__}).")
                if manual:
                    self.statusBar().showMessage(f"You have the newest version ({__version__}).", 6000)
                return
            tag = rel["tag"]
            self.help.set_update_info(f"New version {tag} is out (you have {__version__}).")
            notes = ("What's new", lambda: QDesktopServices.openUrl(QUrl(rel["page"])), False)
            if self.upd_kind == "git":
                self._upd_show(f"New version {tag} is out. This is a git copy: update it with git.",
                               [notes, ("OK", lambda: self._upd_show(""), False)])
            elif self.settings.get("updates") == "auto" and not manual:
                self._upd_download(rel, apply_now=False)
            else:
                self._upd_show(f"New version {tag} is out (you have {__version__}).",
                               [("Update now", lambda: self._upd_download(rel, apply_now=True), True), notes,
                                ("Later", lambda: self._upd_show(""), False)])

        def fail(_, err):
            self.help.set_update_info(f"Could not reach GitHub ({err}). The Studio tries again later.")
            if manual:
                self.statusBar().showMessage("Could not reach GitHub to check for updates.", 6000)
        run_bg(lambda: U.check(), "updates", done, fail)

    def _upd_download(self, rel, apply_now=False):
        from .. import updates as U
        tag, msi = rel["tag"], self.upd_kind == "msi"
        self._upd_show(f"Downloading {tag}...")

        def done(_, path):
            if msi:
                self.upd_msi = path
                self._upd_show(f"Update {tag} is downloaded. It installs when you close the Studio.",
                               [("Install now", self._upd_install_msi, True)])
                if apply_now:
                    self._upd_install_msi()
            else:
                self._upd_show(f"Update {tag} is ready. It goes in the next time the Studio starts.",
                               [("Restart now", self._upd_restart, True), ("Later", lambda: self._upd_show(""), False)])
                if apply_now:
                    self._upd_restart()

        def fail(_, err):
            self._upd_show(f"The update could not be downloaded ({err}).",
                           [("Try again", lambda: self._upd_download(rel, apply_now), True),
                            ("Later", lambda: self._upd_show(""), False)])
        run_bg(lambda: (U.stage_msi(rel) if msi else U.stage_folder_update(rel)), "update-download", done, fail)

    def _upd_restart(self):
        from .. import updates as U
        if U.restart_into_update():
            self.close()
        else:
            self._upd_show(f"Close the Studio and start it again ({U.launcher_name()}) to finish the update.")

    def _upd_install_msi(self):
        from .. import updates as U
        if self.upd_msi and U.run_msi(self.upd_msi):
            self.upd_msi = None
            self.close()

    def _jobs(self):
        n = POOL.activeThreadCount()
        self.jobs.setText(f"● working ({n})" if n else "")
        self.jobs.setStyleSheet(f"color:{T.AMBER};" if n else "")

    def _lens_tagged(self, files, lens):
        """The Library tagged photos: keep the open photo's in-memory edit in step, or its autosave would undo it."""
        if self.develop.path in files and self.develop.image is not None:
            self.develop.edit.lens.tag = lens
            self.develop._fill_lenses(lens)
        self.lenses.refresh()

    def paste_to_files(self, files):
        """Library: paste the copied settings onto the master edit of every selected photo."""
        from .. import engine as E
        from ..engine.presets.looks import copy_groups
        if CLIP["edit"] is None:
            QMessageBox.information(self, "Paste settings", "Nothing copied yet. Open a photo in Develop and press "
                                    "Copy settings first.")
            return
        if not files:
            QMessageBox.information(self, "Paste settings", "Select the photos to paste onto first (click, "
                                    "Shift-click or Ctrl+A).")
            return
        self.develop.save_sidecar()
        n, bad = 0, []
        for f in files:
            try:
                E.save_edit(f, copy_groups(CLIP["edit"], E.load_edit(f), CLIP["groups"]))
                n += 1
            except (OSError, ValueError) as err:
                log.warning("paste settings: %s: %s", f, err)
                bad.append(os.path.basename(f))
        if self.develop.path in files and self.develop.version == 0 and self.develop.image is not None:
            self.develop.commit(E.load_edit(self.develop.path), label="Paste settings (Library)")
        self.library.refresh_paths(files)
        msg = f"Settings from {CLIP['source']} pasted onto {n} photo{'s' if n != 1 else ''}."
        if bad:
            msg += f" Could not write: {', '.join(bad[:5])}{'...' if len(bad) > 5 else ''} (read-only drive?)."
        self.statusBar().showMessage(msg, 10000)

    def _nav_changed(self, row):
        """Back in the Library: the photo just edited shows its new badges (edited, lens tag)."""
        if row == 0 and getattr(self, "develop", None) is not None and self.develop.path:
            self.develop.save_sidecar()
            self.library.refresh_paths([self.develop.path])

    def open_in_develop(self, path):
        self.develop.set_strip(self.library.files, self.library.icons())
        self.nav.setCurrentRow(1)
        self.develop.open(path)

    def export(self, files):
        """Export photos: items are paths (their master edit) or (path, version) for a virtual copy."""
        items = [(f, 0) if isinstance(f, str) else (f[0], int(f[1])) for f in files if f]
        items = [(f, v) for f, v in items if f]
        if not items:
            self.statusBar().showMessage("Select photos in the Library first.", 5000)
            return
        self.develop.save_sidecar()
        dlg = ExportDialog([f for f, _ in items], self.settings, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        spec, dest = dlg.spec(), dlg.dest.text()
        os.makedirs(dest, exist_ok=True)

        def work():
            from .. import engine as E
            out = []
            for f, v in items:
                img = E.open(f)
                if v:
                    out.append(E.export(img, E.load_version(f, v), spec, out_path=version_export_path(f, v, spec, dest)))
                else:
                    out.append(E.export(img, E.load_edit(f), spec, out_dir=dest))
                img.drop_cache()
            return out
        self.statusBar().showMessage(f"Exporting {len(items)} photo(s) in the background...")
        run_bg(work, dest, self._exported, lambda _, e: self.statusBar().showMessage(f"Export problem: {e}"),
               keep=True)

    def _exported(self, dest, out):
        self.statusBar().showMessage(f"Exported {len(out)} file(s) to {dest}", 10000)
        QDesktopServices.openUrl(QUrl.fromLocalFile(dest))

    def toggle_full_screen(self):
        """F11: full screen and back (to maximized if it was)."""
        if self.isFullScreen():
            self.showMaximized() if self._was_max else self.showNormal()
        else:
            self._was_max = self.isMaximized()
            self.showFullScreen()

    _was_max = False

    def show_as_last_time(self):
        """Open as it was closed: the same place and size, maximized or full screen again."""
        import base64
        geo = self.settings.get("window_geometry")
        if geo:
            try:
                self.restoreGeometry(base64.b64decode(geo))
            except Exception:                          # noqa: BLE001  (an odd saved value: default size)
                log.debug("window geometry not restored")
        state = self.settings.get("window_state", "")
        if state == "full":
            self.showFullScreen()
        elif state == "max":
            self.showMaximized()
        else:
            self.show()

    def _remember_window(self):
        import base64
        state = "full" if self.isFullScreen() else ("max" if self.isMaximized() else "normal")
        self.settings.set("window_state", state)
        self.settings.set("window_geometry", base64.b64encode(bytes(self.saveGeometry())).decode("ascii"))

    def closeEvent(self, e):
        from . import common as C
        busy = C.kept_jobs_running()
        if busy:
            ask = QMessageBox(QMessageBox.Icon.Question, "Photographic Studio",
                              f"{busy} export or import job(s) are still writing files. Closing now leaves them "
                              "unfinished.", parent=self)
            wait = ask.addButton("Let them finish", QMessageBox.ButtonRole.RejectRole)
            ask.addButton("Close anyway", QMessageBox.ButtonRole.DestructiveRole)
            ask.exec()
            if ask.clickedButton() is wait:
                self.statusBar().showMessage("Still working: the Studio can close when the export or import is "
                                             "done.", 10000)
                e.ignore()
                return
            log.warning("closed while %d export/import job(s) were still running", busy)
        self.develop.save_sidecar()
        self._remember_window()
        self.library.meta.stop()
        self.library.loader.reset()
        self.library.writer.wait(5000)                 # ratings typed just before closing are saved
        if self.upd_msi is not None:                  # an MSI update downloaded earlier goes in as the Studio closes
            from .. import updates as U
            U.run_msi(self.upd_msi)
            self.upd_msi = None
        super().closeEvent(e)


def _excepthook(t, v, tb):
    from . import common as C
    if C._SHUTTING_DOWN:
        log.debug("exception while closing (ignored): %s", v)
        return
    log.error("unhandled exception in the app\n%s", "".join(traceback.format_exception(t, v, tb)))
    import threading
    if QApplication.instance() is None or threading.current_thread() is not threading.main_thread():
        return                                     # a message box only from the window's own thread
    try:
        QMessageBox.critical(None, "Photographic Studio", f"Something went wrong: {v}\n\nDetails are in "
                             f"{os.path.join('logs', 'studio.log')} (Help > Copy diagnostic report).")
    except Exception:
        pass


def run(argv=None) -> int:
    slog.setup(console_level=None)
    sys.excepthook = _excepthook
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("Photographic Studio By Bill")
    app.setStyleSheet(T.QSS)
    lock = _single_instance(app)
    if lock is None:
        return 0
    w = MainWindow()
    w.show_as_last_time()
    log.info("app started")
    code = app.exec()
    clean = finish_jobs()
    log.info("app closed (exit %s%s)", code, "" if clean else "; previews still rendering were dropped")
    lock.unlock()
    # A preview render still running when Python tears down crashed the exit (seen on Linux: "The Studio stopped
    # with an error" after closing it mid-render). Everything worth keeping is saved by now (sidecars, settings,
    # ratings; export and import jobs were waited for), so leave without tearing down.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code))


def _single_instance(app):
    """One Studio at a time (a slow start tempted a second click, and two windows on the same photos fight over
    sidecars). Returns the held lock, or None when the person chose not to open a second one."""
    from PySide6.QtCore import QLockFile
    try:
        d = paths.get("SETTINGS")
        d.mkdir(parents=True, exist_ok=True)
        lock = QLockFile(str(d / "studio.lock"))
        lock.setStaleLockTime(0)                   # never stale by age; a dead owner is still detected
        # after an update the old window may still be closing: give it time to let go
        if lock.tryLock(15000 if os.environ.get("STUDIO_AFTER_UPDATE") else 1500):
            return lock
    except Exception:                              # noqa: BLE001  (a lock problem must never stop the Studio)
        slog.log_exception("single-instance lock", log)
        return _NoLock()
    box = QMessageBox(QMessageBox.Icon.Information, "Photographic Studio",
                      "Photographic Studio is already open (look for its window, or wait a moment if you just "
                      "started it).")
    box.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
    another = box.addButton("Open another window anyway", QMessageBox.ButtonRole.ActionRole)
    box.exec()
    if box.clickedButton() is another:
        log.info("second window opened on request")
        return _NoLock()
    log.info("already open: second start declined")
    return None


class _NoLock:
    def unlock(self):
        pass


def finish_jobs(timeout_ms: int = 1500) -> bool:
    """At exit: no new background work; jobs not started are dropped; export / import jobs (keep=True) are waited
    for; previews are given ``timeout_ms`` and then abandoned. True when nothing is left running."""
    from . import common as C
    from .library import EXTRA_POOLS
    C.begin_shutdown()
    POOL.clear()                                   # jobs not started yet are dropped
    for p in EXTRA_POOLS:
        p.clear()
    t_end = time.time() + 3600
    while any(j.keep and j.running for j in list(C._LIVE)) and time.time() < t_end:
        POOL.waitForDone(250)                      # files the person asked for are finished
    done = POOL.waitForDone(timeout_ms)
    for p in EXTRA_POOLS:
        done = p.waitForDone(500) and done
    return done
