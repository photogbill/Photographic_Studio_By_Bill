"""Photographic Studio By Bill - the desktop app (Phase 4 shell, Phase 5 lens and shape, Phase 6 develop panels).

One window, a sidebar of workspaces (section 1, rule 0.3.12): Library, Develop, Dual Pixel Lab, Focus Stack,
Card to Stacks, 3D Scan, Super-Res Stack, Infrared, Panorama, Paint, Lenses, Cameras, Export, Help. The interface talks to the engine only through its API
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
from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QFont, QImage, QKeySequence,
                           QLinearGradient, QPainter, QPainterPath, QPainterPathStroker, QPen,
                           QShortcut)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
                               QFileDialog, QFrame, QInputDialog,
                               QGridLayout, QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem, QMainWindow,
                               QMenu, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from .. import __version__, log as slog, paths
from . import theme as T
from . import zoom as Z
from .widgets import (CurveEditor, DetailMeter, FilmicPlot, GradeWheel, GroupDialog, HueRing, SwatchRow, oklch_colour,
                      srgb8_hue)

from .common import (GLOW_HOWTO, POOL, STARS, _LIVE, Job, LensDialog, Settings, _Sig, card, chip,  # noqa: F401
                     dot_icon, hbox, label, load_thumb, qalpha, run_bg, tag_lens, to_qimage)
from .common import BUS, display8, report_problem  # noqa: F401
from .exporting import ExportDialog, QLineEdit_, version_export_path  # noqa: F401
from .library import LibraryPage  # noqa: F401  (section 24: the Library lives in its own file)
from .dualpixel import DualPixelPage  # noqa: E402  (the three stacking tabs: BUILD_PLAN_TABS)
from .focusstack import FocusStackPage  # noqa: E402
from .cardstacks import CardStacksPage  # noqa: E402
from .scan import ScanPage  # noqa: E402  (3D Scan: BUILD_PLAN_DP3D H)
from .superres import SuperResPage  # noqa: E402
from .infrared import InfraredPage  # noqa: E402  (BUILD_PLAN_PANO_IR: the Infrared tab)
from .panorama import PanoramaPage  # noqa: E402  (BUILD_PLAN_PANO_IR: the Panorama tab)
from .paint import PaintPage  # noqa: E402  (BUILD_PLAN_PAINT: digital painting)
from .cameras import CamerasPage  # noqa: E402  (the drop-everything calibration, 2026-11-18)

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
    """Label, slider, value. Double-click resets. Scroll nudges once clicked (_WheelGuard). Tooltip explains in plain English."""
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
    body_clicked = Signal(int)                   # portrait: a found body part clicked (index, -1 = none there)
    body_drawn = Signal(float, float, float, float)   # portrait: a part defined by a drag along a limb
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
        self.ghosts = []                           # flare ghosts to draw: {"uv", "r", "on", "sel", "n"}
        self.show_ghosts = False
        self._sdrag = None                         # (index, "dst" | "src", start uv)
        self._pan = None                           # (start position, start centre) while the photo is dragged
        self.faces = []                            # portrait: [{"lines": [[(u, v), ...], ...], "how"}]
        self.show_faces = False
        self.shapes = []                           # sculpt shapes: [{"outline": [(u, v), ...], "sel", "tool"}]
        self.show_shapes = False
        self.shape_tool = None                     # the shape being drawn ("breast", "thigh", ...)
        self.shape_ratio = 0.5                     # its width / length
        self._shdrag = None                        # (index, start uv) while a shape is moved
        self.body_parts = []                       # found body parts: [{"outline": [(u, v)], "name", "how", "note"}]
        self.body_hover = -1                       # the part under the pointer
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
            p.drawText(self.rect().adjusted(40, 20, -40, -20),
                       int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), self.message)
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
        if self.ghosts and self.show_ghosts:
            self._paint_ghosts(p, r)
        if self.faces and (self.show_faces or self.tool == "eyes"):
            self._paint_faces(p, r)
        if self.shapes and (self.show_shapes or self.tool in ("shape", "sculpt", "body")):
            self._paint_shapes(p, r)
        if self.body_parts and self.tool == "body":
            self._paint_body(p, r)
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

    def _paint_ghosts(self, p, r):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        L = self._long_px(r)
        for g in self.ghosts:
            c = self._px(g["uv"], r)
            rad = max(4.0, g["r"] * L)
            col = QColor(T.AMBER) if g.get("on", True) else QColor(T.DIM)
            p.setPen(QPen(col, 2.2 if g.get("sel") else 1.4, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, rad, rad)
            self._tag(p, QPointF(c.x() + rad * 0.72, c.y() - rad * 0.72), str(g["n"]), colour=col.name())

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

    def _paint_body(self, p, r):
        """Every found part as a faint outline; the one under the pointer filled, with its name. Cyan: both edges
        measured from the person's outline; amber: an edge estimated."""
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for k, bp in enumerate(self.body_parts):
            col = QColor(T.CYAN) if bp.get("how") == "measured" else QColor(T.AMBER)
            path = self._shape_path(bp["outline"], r)
            if k == self.body_hover:
                fill = QColor(col)
                fill.setAlpha(70)
                p.setBrush(fill)
                p.setPen(QPen(col, 2.2))
            else:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(col, 1.0, Qt.PenStyle.DotLine))
            p.drawPath(path)
        if 0 <= self.body_hover < len(self.body_parts) and self._mouse is not None:
            bp = self.body_parts[self.body_hover]
            col = T.CYAN if bp.get("how") == "measured" else T.AMBER
            txt = bp["name"] + ("" if bp.get("how") == "measured" else "  (edge estimated)")
            if bp.get("note"):
                txt += "  " + bp["note"]
            c = self._px(self._mouse, r)
            self._tag(p, QPointF(c.x() + 16, c.y() - 26), txt, colour=col)

    def _body_at(self, uv):
        """The smallest found part under ``uv``, or -1."""
        r = self.photo_rect()
        here = self._px(uv, r)
        best, area = -1, None
        for k, bp in enumerate(self.body_parts):
            path = self._shape_path(bp["outline"], r)
            if path.contains(here):
                b = path.boundingRect()
                a = b.width() * b.height()
                if area is None or a < area:
                    best, area = k, a
        return best

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
        if self.tool in ("eyes", "sculpt", "shape", "body"):
            tips = {"body": "Point at a part to see its name; click to select it.  Nothing found there? Drag along "
                            "the limb.  Esc: done",
                    "eyes": "Click the centre of each eye (either order).  Esc: cancel",
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
            elif self.tool == "body" and len(self._pts) == 2:
                p.setPen(QPen(QColor(T.CYAN), 2, Qt.PenStyle.DashLine))
                p.drawLine(self._px(self._pts[0], r), self._px(self._pts[1], r))
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
        if self.tool == "body":
            self._pts = [uv, uv]
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
        elif self.tool in ("object", "colour", "hint"):
            self.tool_click.emit(uv[0], uv[1], bool(e.modifiers() & Qt.KeyboardModifier.AltModifier))
        self.update()

    def _tool_move(self, e):
        uv = self._uv(e.position())
        self._mouse = uv
        if self.tool == "body":
            if e.buttons() & Qt.MouseButton.LeftButton and len(self._pts) == 2:
                self._pts[1] = uv
            else:
                self.body_hover = self._body_at(uv)
            self.update()
            return
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
        if self.tool == "body":
            if len(self._pts) == 2:
                (u0, v0), (u1, v1) = self._pts
                if math.hypot(u1 - u0, v1 - v0) > 0.015:
                    self.body_drawn.emit(u0, v0, u1, v1)
                else:
                    self.body_clicked.emit(self._body_at((u0, v0)))
            self._pts = []
            self.update()
            return
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
    web_requested = Signal(list)                 # Save for web: [(path, version)]

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
        self._saved_state = None               # what the sidecar holds (see save_sidecar): None = not known
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
        self.web_b = QPushButton("Save for web")
        self.web_b.setToolTip("A small sRGB JPEG for the internet (2048 px long edge, sharpened for screens): the "
                              "smallest file that still looks the same as the best-quality one. Goes to your export "
                              "folder as NAME_web_Studio.jpg.")
        self.web_b.clicked.connect(lambda: self.path and self.web_requested.emit([(self.path, self.version)]))
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
                  [self.undo_b, self.redo_b, self.zoom_b, exp, self.web_b, self.mig])
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
        self.viewer.body_clicked.connect(self._body_clicked)
        self.viewer.body_drawn.connect(self._body_drawn)
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
        # Old photo and film
        g = Group("Old photo and film", "For scans and camera copies of prints, slides and negatives: turn a negative "
                  "into a positive, take out dust and scratches, give a black and white photo its colour.")
        self.film_kind = g.add(QComboBox())
        self.film_kind.addItem("Positive (a print, a slide or a digital photo)", "")
        self.film_kind.addItem("Black and white negative", "bw")
        self.film_kind.addItem("Colour negative", "colour")
        self.film_kind.setToolTip("A negative is turned positive from its own film base and density range, "
                                  "measured on the whole frame. The orange mask of colour film goes with it.")
        self.pick_base = QPushButton("Pick the film base")
        self.pick_base.setCheckable(True)
        self.pick_base.setToolTip("Click clear film between frames or at the edge (no picture on it). Without a "
                                  "pick, the clearest part of the film is used.")
        self.pick_base.toggled.connect(lambda on: self._set_pick("film" if on else None))
        self.base_auto = QPushButton("Find it again")
        self.base_auto.setToolTip("Forget the picked film base and use the clearest part of the film")
        self.base_auto.clicked.connect(self._film_base_auto)
        self.film_btns = g.add(hbox(self.pick_base, self.base_auto))
        self.film_range = g.add(ValueSlider("Negative contrast", 0.6, 1.6, 1.0, 0.01, "{:.0%}",
                                            "How much of the negative's range becomes the print's range. Lower "
                                            "is softer, higher is punchier (on top of Contrast)."))
        self.film_chip = g.add(chip("Film: a positive"))
        self.st_on = g.add(QCheckBox("Remove stains and corrosion"))
        self.st_on.setToolTip("Foxing, water stains, the rust-coloured corrosion on old plates: each is a coloured "
                              "layer over the picture. The Studio measures how the stain's colour changes the "
                              "brightness and takes both out, so the picture under a thin stain comes back instead "
                              "of being painted over. Where a stain is so strong that the picture under it may be "
                              "gone, nothing is invented: 'Show what's lost' marks those places.")
        self.st_amt = g.add(ValueSlider("Stain strength", 0, 150, 100, 1, "{:.0f}%",
                                        "How much of the measured stain is taken out. 100% is the measurement; go "
                                        "higher if a ghost of the stain is left, lower if it leaves a pale spot."))
        self.st_sens = g.add(ValueSlider("Sensitivity", 0, 100, 50, 1, "{:.0f}",
                                         "Higher catches fainter stains. If real warm colours (a cheek, wood, a "
                                         "sepia tone) start to fade, lower it."))
        self.st_show = g.add(QCheckBox("Show what's lost (amber)"))
        self.st_show.setToolTip("Marks the cores of strong stains, where the picture itself may be gone. These "
                                "places are left as they are, never filled with made-up detail.")
        self.st_show.toggled.connect(self._stain_show)
        self.st_chip = g.add(chip("Stains: off"))
        self.scr_on = g.add(QCheckBox("Remove dust and scratches"))
        self.scr_on.setToolTip("Finds specks, hairs, scratches and cracks and fills them from the light and grain "
                               "around them. Microsoft's scratch detector (trained on real old photos) decides "
                               "which long marks are damage when it is installed.")
        self.scr_amt = g.add(ValueSlider("Strength", 0, 100, 50, 1, "{:.0f}",
                                         "Higher finds fainter marks. If real detail starts to go (a catchlight, a "
                                         "star, a button), lower it."))
        self.scr_size = g.add(ValueSlider("Largest mark", 2, 30, 8, 1, "{:.0f} px",
                                          "The widest speck or scratch to take out, in pixels of the full-size "
                                          "photo. Larger damage: use Heal in Retouch."))
        self.scr_chip = g.add(chip("Dust and scratches: off"))
        self.col_on = g.add(QCheckBox("Colourize a black and white photo (automatic)"))
        self.col_on.setToolTip("DDColor (Kang et al. 2023, Apache 2.0) suggests colours for a black and white "
                               "photo, on this PC. The brightness of every pixel stays exactly as it was; only "
                               "colour is added. The colours are a plausible guess, not a record.")
        self.col_amt = g.add(ValueSlider("Colour strength", 0.0, 1.5, 1.0, 0.05, "{:.0%}",
                                         "How strong the suggested colours are. Automatic colourizers tend to be "
                                         "shy; above 100% gives them more life."))
        self.col_warm = g.add(ValueSlider("Colour: cool - warm", -100, 100, 0, 1, "{:+.0f}",
                                          "Shifts the suggested colours towards blue (-) or towards yellow (+). Your "
                                          "own coloured parts are not changed."))
        self.col_tint = g.add(ValueSlider("Colour: green - magenta", -100, 100, 0, 1, "{:+.0f}",
                                          "Shifts the suggested colours towards green (-) or towards magenta (+)."))
        self.col_chip = g.add(chip("Colourize: off"))
        self.hint_b = QPushButton("🎨 Colour a part")
        self.hint_b.setCheckable(True)
        self.hint_b.setToolTip("Click a part of the photo (a coat, a car, the sky) and choose its colour: it spreads "
                               "over that part and keeps the photo's own light and shade. Shift-click adds more of "
                               "the part, Alt-click takes a piece away. Works with or without the automatic colour.")
        self.hint_b.toggled.connect(lambda on: self._set_hint_tool(on))
        g.add(self.hint_b)
        self.hint_list = g.add(QListWidget())
        self.hint_list.setMaximumHeight(96)
        self.hint_list.currentRowChanged.connect(self._hint_selected)
        self.hint_col_b = QPushButton("Change colour")
        self.hint_col_b.clicked.connect(self._hint_recolour)
        self.hint_del_b = QPushButton("Remove")
        self.hint_del_b.clicked.connect(self._hint_remove)
        self.hint_btns = g.add(hbox(self.hint_col_b, self.hint_del_b))
        self.hint_spread = g.add(ValueSlider("Spread", 3, 30, 8, 1, "{:.0f}",
                                             "How far the selected part's colour spreads: higher crosses stronger "
                                             "edges in the photo. Alt-click is the sure way to stop it somewhere."))
        self.hint_spread.changed.connect(lambda *_: self._hint_spread_timer.start())
        self._hint_spread_timer = QTimer(self, singleShot=True, interval=350, timeout=self._hint_respread)
        self._hints, self._hsel = [], -1
        self.old_only = [self.film_btns, self.film_range, self.st_amt, self.st_sens, self.st_show, self.scr_amt,
                         self.scr_size, self.col_amt]
        bl.addWidget(g)
        self.old_group = g
        # Cognitive Adjustments: our own local tools, built from research (docs/PROGRESS.md decisions 89-)
        g = Group("Cognitive Adjustments", "Tools built from research papers and run on this PC. Each one "
                                           "measures what is in the photo; nothing is invented.")
        self.ht_on = g.add(QCheckBox("Remove printing dots (halftone)"))
        self.ht_on.setToolTip("For scans of printed photos (a newspaper cutting, a yearbook, a magazine or book "
                              "page). The printing screen is a regular grid of dots; the Studio finds its exact "
                              "spacing and angle and takes out only those frequencies, so the picture's own detail "
                              "stays. If no screen is found, nothing is changed.")
        self.ht_amt = g.add(ValueSlider("Dot removal", 0, 100, 100, 1, "{:.0f}%",
                                        "How much of the printing screen is taken out."))
        self.ht_chip = g.add(chip("Printing dots: off"))
        self.jp_on = g.add(QCheckBox("Repair JPEG blocks and ringing"))
        self.jp_on.setToolTip("For JPEG files (shown only for them). A JPEG keeps each 8 x 8 block only to within "
                              "a rounding step; the usual decoder shows the middle of every step, which makes the "
                              "blocks and the halos round edges. This picks the smoothest picture that still "
                              "matches the file exactly, so nothing is invented. Brightness and colour are each "
                              "repaired in the file's own form; the result keeps 16 bits.")
        self.jp_chip = g.add(chip("JPEG repair: off"))
        self.dp_haze = g.add(ValueSlider("Depth haze", -100, 100, 0, 1, "{:+.0f}",
                                         "Air between the camera and the scene, by distance: right adds haze to "
                                         "the distance (pushes it back), left clears distant haze. Nearby things "
                                         "are left nearly alone. The depth is worked out from the photo on this PC "
                                         "(Depth Anything V2): an order from near to far, not metres; glass, "
                                         "mirrors and water can fool it."))
        self.dp_show = g.add(QCheckBox("Show depth map"))
        self.dp_show.setToolTip("Colour the photo by distance: red and yellow are near, blue is far. The same map is "
                                "used by Depth haze and by Masks > Depth.")
        self.dp_show.toggled.connect(self._depth_show)
        self.dp_chip = g.add(chip("Depth: not worked out yet"))
        self.rl_amt = g.add(ValueSlider("Relight", 0, 100, 0, 1, "{:.0f}",
                                        "Move the light. The shapes of things come from the depth map, the light "
                                        "already on the photo is measured from how brightness follows them, and "
                                        "each place is brightened or darkened by how the new light would fall "
                                        "there. Only the light changes: no detail or feature is drawn, texture is "
                                        "untouched. The shapes are a guess from one photo, so start gently."))
        self.rl_az = g.add(ValueSlider("Light from", -180, 180, 135, 1, "{:+.0f}°",
                                       "Where the light comes from: 0 right, 90 top, 180 left, -90 below "
                                       "(135 = top left)."))
        self.rl_el = g.add(ValueSlider("Light height", 0, 90, 35, 1, "{:.0f}°",
                                       "0 = grazing from the side (deep shadows), 90 = from the camera (flat)."))
        self.rl_soft = g.add(ValueSlider("Light softness", 0, 100, 40, 1, "{:.0f}",
                                         "How much soft fill reaches the shadow side."))
        self.rl_chip = g.add(chip("Relight: off"))
        self.cm_btn = g.add(QPushButton("Match colours to another photo..."))
        self.cm_btn.setToolTip("Pick a photo whose colours you like. The Studio works out exposure, contrast, white "
                               "balance, saturation and the shadow and highlight colour wheels that give this photo "
                               "the same colour mood, and moves those sliders. Nothing is painted: every change is "
                               "an ordinary slider you can move afterwards (or Undo).")
        self.cm_btn.clicked.connect(self._colour_match)
        self.cm_amt = g.add(ValueSlider("Match amount", 0, 150, 100, 1, "{:.0f}%",
                                        "How much of the colour match is applied (it moves the same sliders)."))
        self.cm_amt.changed.connect(lambda *_: self._cm_timer.start())
        self._cm_timer = QTimer(self, singleShot=True, interval=300, timeout=self._colour_match_amount)
        self.cm_amt.setVisible(False)
        self.cm_chip = g.add(chip("Colour match: none"))
        self._cm = None
        # Composite: subjects from other photos (pipeline/composite.py)
        self.cp_add = g.add(QPushButton("Add a subject from another photo..."))
        self.cp_add.setToolTip("Pick a photo, click the person or thing you want (click-to-select outlines it), "
                               "and it is placed into this photo. Move, size and turn it with the sliders below; "
                               "Match its colours re-develops it under this photo's light. Every pixel of the "
                               "subject comes from its own photo; nothing is generated.")
        self.cp_add.clicked.connect(self._composite_add)
        self.cp_list = g.add(QListWidget())
        self.cp_list.setMaximumHeight(80)
        self.cp_list.currentRowChanged.connect(self._composite_selected)
        self.cp_list.itemChanged.connect(self._composite_item_changed)
        self.cp_x = g.add(ValueSlider("Subject across", 0, 100, 50, 0.5, "{:.0f}%", "Left (0) to right (100)."))
        self.cp_y = g.add(ValueSlider("Subject up / down", 0, 100, 55, 0.5, "{:.0f}%", "Top (0) to bottom (100)."))
        self.cp_size = g.add(ValueSlider("Subject size", 5, 150, 50, 1, "{:.0f}%",
                                         "Its height as a share of this photo's height."))
        self.cp_rot = g.add(ValueSlider("Subject turn", -180, 180, 0, 0.5, "{:+.0f}°", "+ = counter-clockwise."))
        self.cp_soft = g.add(ValueSlider("Subject edge softness", 0, 10, 1, 0.1, "{:.1f} px"))
        self.cp_flip = QCheckBox("Mirror")
        self.cp_grain = QCheckBox("Match grain")
        self.cp_grain.setToolTip("Adds this photo's measured noise to the subject when it is cleaner, so it does "
                                 "not look pasted on.")
        self.cp_row = g.add(hbox(self.cp_flip, self.cp_grain))
        self.cp_match_b = QPushButton("Match its colours")      # (a longer name made the panel wider than its space)
        self.cp_match_b.setToolTip("Match the subject's colours to this photo. "
                                   "Works out exposure, contrast, white balance, saturation and colour wheels that "
                                   "give the subject's photo this photo's colour mood, and develops the subject "
                                   "with them.")
        self.cp_match_b.clicked.connect(self._composite_match)
        self.cp_del_b = QPushButton("Remove subject")
        self.cp_del_b.clicked.connect(self._composite_delete)
        self.cp_btns = g.add(hbox(self.cp_match_b, self.cp_del_b))
        self.cp_mamt = g.add(ValueSlider("Subject colour match", 0, 150, 100, 1, "{:.0f}%"))
        self._layers, self._lsel = [], -1
        for w_ in (self.cp_x, self.cp_y, self.cp_size, self.cp_rot, self.cp_soft, self.cp_mamt):
            w_.changed.connect(self._composite_params)
        for w_ in (self.cp_flip, self.cp_grain):
            w_.toggled.connect(self._composite_params)
        self.chg_show = g.add(QCheckBox("Show what the automatic tools changed"))
        self.chg_show.setToolTip("A heat map over the photo: how much printing-dot removal, stain removal, dust and "
                                 "scratch removal and colourizing changed each place. Untouched places stay clear, "
                                 "so you can see that a face was left alone.")
        self.chg_show.toggled.connect(self._change_show)
        bl.addWidget(g)
        self.cog_group = g
        # Clean
        g = Group("Clean", "Noise is removed from the raw data itself, before colour is made: the best place to "
                           "do it. Shoot raw for the cleanest results.")
        self.dn_on = g.add(QCheckBox("Clean up noise"))
        self.dn_auto = g.add(QCheckBox("Auto: the photo chooses the strength"))
        self.dn_auto.setChecked(True)
        self.dn_auto.setToolTip(
            "The camera's noise is known exactly, so the Studio can work out, from the photo alone, how much error "
            "each strength would leave and how much of the finest detail it would keep. It tries a handful of "
            "strengths on five pieces of the photo and takes the lightest one that is close to the best, counting "
            "lost detail five times as heavily as noise left in. A photo of smooth walls is cleaned hard, a photo "
            "full of fine texture gently.\n\nA Dual Pixel photo is judged by its own two halves instead "
            "(measured, not estimated).\n\nBest for: every raw photo from a camera you have calibrated on the "
            "Cameras page. Weak at: it needs the camera's noise to be measured, so for an uncalibrated camera, "
            "for ISO settings where neighbouring photosites share noise, and for scans or JPEGs it says so and "
            "the sliders decide. Untick it to set the strengths yourself.")
        self.dn_auto.toggled.connect(self._dn_auto_toggled)
        self.dn_auto_chip = g.add(chip("Auto: -"))
        self.dn_auto_chip.setWordWrap(True)
        self.dn_s = g.add(ValueSlider("Strength", 0.0, 2.0, 1.0, 0.05, "{:.0%}", "With Auto on: 100% is the "
                                      "photo's own choice; move it to lean lighter or heavier. With Auto off: "
                                      "1 = matched to the measured noise (the camera's own at this ISO for a raw, "
                                      "measured in the picture for a scan or JPEG)."))
        self.dn_l = g.add(ValueSlider("Luminance (grain)", 0.0, 2.5, Denoise().luma, 0.05, "{:.2f}",
                                      "Fine, grain-like noise. Lower keeps more texture."))
        self.dn_c = g.add(ValueSlider("Colour (blotches)", 0.0, 4.0, Denoise().chroma, 0.05, "{:.2f}",
                                      "Coloured mottling, the ugly part of high-ISO noise."))
        self.dn_g = g.add(ValueSlider("Keep grain", 0.0, 1.0, 0.0, 0.05, "{:.0%}",
                                      "Puts back a fine, film-like fraction of the luminance noise."))
        self.dn_m = g.add(QComboBox())
        self.dn_m.addItems(["hybrid", "fusion", "starlet", "nlm", "mppca", "collab", "learned"])
        self.dn_m.setToolTip("How the noise is removed. Each method says below what it is best for and weak at. "
                             "Hybrid is the recommended one; Fusion uses two methods, each where it is best; "
                             "Collaborative is the slow yardstick; Learned is this camera's own trained network.")
        self.dn_info = g.add(label("", "dim", wrap=True))
        self.dn_m.currentIndexChanged.connect(lambda *_: self._dn_method_text())
        self._dn_method_text()
        self.dn_safe = g.add(QCheckBox("Protect stars, catchlights and sparks"))
        self.dn_safe.setChecked(True)
        self.dn_safe.setToolTip("After cleaning, anything that moved by more than 4.5 times its measured noise was "
                                "not noise: the original is kept there. Pure noise almost never does that, so "
                                "nothing visible is let back in.")
        self.dn_safe_chip = g.add(chip("Protected: -"))
        self.dn_guard = g.add(QCheckBox("Detail guard: give back whatever was not noise"))
        self.dn_guard.setChecked(True)
        self.dn_guard.setToolTip("What Clean removed is tested against the camera's known noise, band by band. "
                                 "Structure that stands above the noise is returned; where more was removed than "
                                 "noise can account for, the surplus is returned too (the area keeps a little grain "
                                 "and all of its texture); and nothing may be added that noise could not have "
                                 "hidden. It applies to every method, the learned one included.")
        self.dn_guard_chip = g.add(chip("Detail returned: -"))
        self.dn_removed = g.add(QPushButton("Show what Clean removed..."))
        self.dn_removed.setToolTip("The proof for this photo: a piece of it before Clean, after Clean, and what "
                                   "was taken out. If what was taken out looks like television static, only noise "
                                   "was removed.")
        self.dn_removed.clicked.connect(self._dn_show_removed)
        self.dn_lift = g.add(QCheckBox("Clean lifted shadows harder"))
        self.dn_lift.setChecked(True)
        self.dn_lift.setToolTip("Clean works on the raw data, before brightness is changed. When you brighten the "
                                "photo or open the shadows (Exposure, Shadows, Smart lighting, Blacks, Auto HDR), "
                                "noise that was invisible in the dark is lifted into view. With this on, the parts "
                                "that end up lifted are cleaned harder, in proportion to how far they are lifted; "
                                "parts at normal brightness are not touched.\n\nBest for: underexposed photos "
                                "and strong shadow recovery. Weak at: it follows the sliders in half-stop steps, "
                                "and does not know about brightening done with masks or the tone curve.")
        self.dn_cal = g.add(QCheckBox("Use this camera's calibration"))
        self.dn_cal.setChecked(True)
        self.dn_cal.setToolTip("If this camera was calibrated on the Cameras page, what its dark frames showed is "
                               "taken out before Clean looks at the photo:\n"
                               "- hot and warm photosites from the camera's own map (a warm one is only replaced "
                               "where it actually shows in this photo);\n"
                               "- the part of the row and column banding that is the same in every photo;\n"
                               "- the small difference between the black of the picture area and of the masked "
                               "border.\nThese are not random, so no noise reduction should have to guess at "
                               "them.\n\nBest for: long exposures, high ISO, deep shadows. Weak at: needs a "
                               "profile for this camera (and Deep calibration for the fixed pattern); a camera "
                               "without one is simply left alone.")
        self.dn_cal_chip = g.add(chip("Calibration: -"))
        self.dn_cal_chip.setWordWrap(True)
        self.band = g.add(QCheckBox("Remove row banding (measured, never added)"))
        self.band.setChecked(True)
        self.noise_chip = g.add(chip("Noise model: -"))
        self.full_only = [self.dn_l, self.dn_c, self.dn_g, self.dn_m, self.dn_info, self.dn_lift, self.dn_cal,
                          self.dn_cal_chip]
        self.clean_scan_note = g.add(label(
            "Scan or JPEG: Clean measures the noise in the picture itself and removes it there. It does its best "
            "work on raw files, where it knows every photosite's noise from the sensor, before the camera has "
            "demosaiced, sharpened and compressed it; on a JPEG some noise is already smeared into the picture and "
            "cannot be told apart. For the cleanest results, shoot raw. (Film grain is noise to it too: lower "
            "Luminance, or use Keep grain, to keep it.)", None, wrap=True))
        self.clean_scan_note.setStyleSheet(f"color:{T.AMBER};")
        self.clean_scan_note.hide()
        self.clean_group = g
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
        # flare ghosts (local/flare.py): the round reflections a bright light leaves, worst in infrared
        g.add(label("Flare ghosts", "dim"))
        self.fl_find = QPushButton("Find flare ghosts")
        self.fl_find.setToolTip("Look for the round ghost reflections a bright light (the sun, a lamp) leaves "
                                "along a line through the centre of the photo. Infrared shows them most: lens "
                                "coatings are made for visible light.")
        self.fl_find.clicked.connect(self.find_flare)
        self.fl_show = QPushButton("Show")
        self.fl_show.setCheckable(True)
        self.fl_show.setToolTip("Draw the ghosts on the photo (amber: found by estimate)")
        self.fl_show.toggled.connect(self._show_ghosts)
        for b in (self.fl_find, self.fl_show):
            b.setStyleSheet("padding: 5px 8px;")
        g.add(hbox(self.fl_find, self.fl_show))
        self.fl_list = QListWidget()
        self.fl_list.setMaximumHeight(92)
        self.fl_list.setToolTip("Untick a ghost to leave it alone (a real round thing, not a reflection)")
        self.fl_list.itemChanged.connect(self._ghost_ticked)
        self.fl_list.currentRowChanged.connect(lambda *_: self._refresh_ghost_view())
        self.fl_list.hide()
        g.add(self.fl_list)
        self.fl_on = g.add(QCheckBox("Subtract the glow"))
        self.fl_on.setToolTip("Best for: faint ghosts over detail you want to keep (leaves, rock, water): the "
                              "added light is measured and taken away, the detail under it stays.\nWeak at: "
                              "strong or blown-out ghosts: a faint rim can stay, and nothing is left to recover "
                              "where the ghost clipped.")
        self.fl_on.toggled.connect(self.controls_changed)
        self.fl_amt = g.add(ValueSlider("Glow removed", 0.0, 1.5, 1.0, 0.05, "{:.0%}",
                                        "How much of the measured ghost light to take away."))
        self.fl_amt.changed.connect(self.controls_changed)
        self.fl_rebuild = QPushButton("Rebuild ghosts with Fill")
        self.fl_rebuild.setToolTip("Best for: strong or blown-out ghosts over sky, water, grass or foliage: each "
                                   "ticked ghost becomes a Fill spot in Retouch, rebuilt from what is around it "
                                   "(LaMa when installed).\nWeak at: a ghost over something only it covers (a "
                                   "face, a sign): what is made there is a plausible guess, not what was there.")
        self.fl_rebuild.setStyleSheet("padding: 5px 8px;")
        self.fl_rebuild.clicked.connect(self.rebuild_ghosts)
        g.add(hbox(self.fl_rebuild))
        self.fl_chip = g.add(chip("Flare: not looked for yet", T.DIM))
        self.fl_chip.setWordWrap(True)
        self._flare = None
        self._flare_controls()
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
        self.tone.addItems(["Studio Standard", "Camera match", "Linear (no curve)",
                            "Lightroom-like brightness (estimated)"])
        self.tone.setToolTip("Studio Standard: the Studio's own curve. Camera match: fitted to the camera's JPEG. "
                             "Linear: no curve.\nLightroom-like: Adobe's published default curve (or the Lightroom "
                             "profile's own curve) and Lightroom's hidden per-camera brightness lift, read from "
                             "Lightroom's profiles on this PC. Estimated: Lightroom's newest process may differ a "
                             "little; best with a Lightroom profile chosen below.")
        self.tone.activated.connect(self._tone_mode_note)
        self.curve_col = g.add(QComboBox())
        self.curve_col.addItems(["Curve colour: Natural", "Curve colour: Rich"])
        self.curve_col.setToolTip("How contrast treats colour.\nNatural (default for new photos): the curve sets "
                                  "brightness and every colour keeps its mix, so skin stays true; matches what "
                                  "Lightroom shows and prints. Best for: portraits, printing, accuracy. Weak at: "
                                  "nothing for accuracy; a little less punch.\nRich: the curve bends red, green and "
                                  "blue separately, like slide film: warm colours get redder and more saturated. Best "
                                  "for: bold landscapes. Weak at: skin, which goes red.")
        self.curve_col.activated.connect(lambda _: self.controls_changed())
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
                                   "stay in settings\\camera_profiles and are never shared.\nAmber: Lightroom's own "
                                   "profiles for this camera, found on this PC (read where Adobe installed them, never "
                                   "copied). Adobe Standard matches Lightroom's colour; 'Camera ...' ones match the "
                                   "camera's picture styles. On a PC without Lightroom the photo falls back to the "
                                   "camera matrix.\n\nStudio (camera matrix): for 1,013 cameras it now blends a "
                                   "daylight and a tungsten matrix by the photo's white balance (Solomatov & "
                                   "Akkaynak 2023), so warm and candle light stay true. Best for: everything. Weak "
                                   "at: nothing new; daylight looks as before.\nSpectral (Solomatov & Akkaynak "
                                   "2023): a profile calculated for the exact light from the camera's predicted "
                                   "colour sensitivities. Best for: very warm or unusual light. Weak at: a little "
                                   "more saturated than Lightroom.")
        self.prof_combo.activated.connect(self._profile_picked)
        self._fill_profiles("")
        self.sp_on = QCheckBox("Soft proof (show the print)")
        self.sp_on.setToolTip("Pick the ICC profile of your printer and paper (from the paper maker or your lab). The "
                              "preview then shows the print: duller blacks, paper white, colours the paper cannot "
                              "reach. Your edit and exports are not changed.")
        self.sp_pick = QPushButton("Printer / paper profile...")
        self.sp_pick.clicked.connect(self._pick_proof_profile)
        self.sp_intent = QComboBox()
        self.sp_intent.addItems(["Perceptual", "Relative colorimetric"])
        self.sp_intent.setToolTip("Perceptual: squeezes every colour onto the paper, keeping their relations (best "
                                  "for vivid photos). Relative colorimetric: prints exactly what fits and clips the "
                                  "rest (best for portraits and photos already within the paper's range).")
        self.sp_gamut = QCheckBox("Mark unprintable colours (magenta)")
        for w in (self.sp_on, self.sp_intent, self.sp_gamut):
            (w.toggled if isinstance(w, QCheckBox) else w.currentIndexChanged).connect(lambda *_: self._proof_changed())
        g.add(self.sp_on)
        g.add(self.sp_pick)
        g.add(self.sp_intent)
        g.add(self.sp_gamut)
        self.sp_info = g.add(chip("Soft proof: no printer profile chosen", T.DIM))
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
                                                     "face, hair, clothes, arms, legs, torso or the background"),
                                ("depth", "Depth", "Select by distance: near things, far things or a band between "
                                                   "(the depth is worked out from the photo on this PC)"),
                                ("channel", "Channel", "Select by one raw channel, or the difference or ratio of two "
                                                       "(red / blue separates infrared-bright foliage from sky and "
                                                       "skin in a 590-720 nm infrared photo)")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setStyleSheet("padding: 5px 6px;")
            b.clicked.connect(lambda _=False, k=kind: self.new_mask(k))
            self.mask_new[kind] = b
            btns.append(b)
        g.add(hbox(*btns[:3]))
        g.add(hbox(*btns[3:6]))
        g.add(hbox(*btns[6:]))
        self.sel_model = QComboBox()
        self.sel_model.setToolTip("What finds the outline when you click an object (Masks > Object) or a part to "
                                  "colour (Colour a part). SAM 2.1 (Meta, run on this PC) outlines almost anything "
                                  "from one click; it looks at the photo once (Tiny about 10 s, Large longer), then "
                                  "each click is instant. Classic is the Studio's own colour-and-edge method: no "
                                  "model, instant, best on things with a clear colour.")
        self._fill_sel_models()
        self.sel_model.currentIndexChanged.connect(self._sel_model_changed)
        g.add(hbox(label("Click-to-select:", "dim"), self.sel_model))
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
                                "+ Skin", "+ Person", "- Person (subtract)", "x Person (only where)", "+ Depth",
                                "- Depth (subtract)", "x Depth (only where)", "+ Channel", "- Channel (subtract)",
                                "x Channel (only where)"])
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
        self.d_lo = ValueSlider("From (0 far, 100 near)", 0, 100, 60, 1, "{:.0f}",
                                "How far away the selection starts: 0 is the most distant part of the photo, 100 "
                                "the nearest")
        self.d_hi = ValueSlider("To (0 far, 100 near)", 0, 100, 100, 1, "{:.0f}")
        self.d_soft = ValueSlider("Depth softness", 1, 50, 10, 1, "{:.0f}")
        self.ch_src = QComboBox()
        from ..engine.pipeline.pipeline import CHANNEL_SOURCES
        for k, n in CHANNEL_SOURCES.items():
            self.ch_src.addItem(n, k)
        self.ch_src.setToolTip("What the mask reads: one white-balanced raw channel, or the difference or ratio of "
                               "two (before any infrared channel mix). Red / blue separates infrared-bright foliage "
                               "from sky and skin in 590-720 nm infrared photos. Measured from the raw.")
        self.ch_src.activated.connect(self._channel_source_picked)
        self.ch_row = hbox(label("Reads"), self.ch_src)
        rbl.addWidget(self.ch_row)
        self.ch_row.setVisible(False)
        self.ch_lo = ValueSlider("From (0 low, 100 high)", 0, 100, 50, 1, "{:.0f}",
                                 "Where the selection starts along the channel's range in this photo")
        self.ch_hi = ValueSlider("To (0 low, 100 high)", 0, 100, 100, 1, "{:.0f}")
        self.ch_soft = ValueSlider("Channel softness", 1, 50, 10, 1, "{:.0f}")
        self._range_ws = (self.r_lo, self.r_hi, self.r_soft, self.c_width, self.c_soft, self.c_chroma, self.r_feather,
                          self.d_lo, self.d_hi, self.d_soft, self.ch_lo, self.ch_hi, self.ch_soft)
        for w in self._range_ws:
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
                                 "its catchlight"),
                                ("fill", "Fill", "Rebuild an area from what is around it: click or brush over a "
                                 "flare, a sign, a wire or a person. LaMa (generative) plans it when installed, "
                                 "PatchMatch brings in real texture from your photo")):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setStyleSheet("padding: 5px 6px;")
            b.toggled.connect(lambda on, k=kind: self._set_spot_tool(k if on else None))
            grp.addButton(b)
            self.rt_btn[kind] = b
        g.add(hbox(*self.rt_btn.values()))
        from ..engine.local import fill as _fl
        self.rt_fill_method = QComboBox()
        self.rt_fill_method.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        for key, m in _fl.METHODS.items():
            self.rt_fill_method.addItem(m["name"], key)
            self.rt_fill_method.setItemData(self.rt_fill_method.count() - 1,
                                            f"Best for: {m['best']}\nWeak at: {m['weak']}",
                                            Qt.ItemDataRole.ToolTipRole)
        self.rt_fill_method.currentIndexChanged.connect(self._fill_method_changed)
        self.rt_fill_info = chip("", T.DIM)
        self.rt_fill_row = QWidget()
        fr = QVBoxLayout(self.rt_fill_row)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.setSpacing(2)
        fr.addWidget(hbox(label("Fill with"), (self.rt_fill_method, 1), stretch=False))
        fr.addWidget(self.rt_fill_info)
        g.add(self.rt_fill_row)
        self.rt_fill_row.setVisible(False)
        self._fill_method_changed()
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
                  self.c_bw, self.f_gcol, self.dn_safe, self.dn_guard, self.hdr_on, self.dn_auto, self.dn_lift,
                  self.dn_cal):
            w.toggled.connect(self.controls_changed)
        self.hdr_amt.changed.connect(self.controls_changed)
        for w in (self.film_range, self.st_amt, self.st_sens, self.scr_amt, self.scr_size, self.col_amt,
                  self.col_warm, self.col_tint, self.ht_amt, self.dp_haze, self.rl_amt, self.rl_az, self.rl_el,
                  self.rl_soft):
            w.changed.connect(self.controls_changed)
        for w in (self.ht_on, self.st_on, self.scr_on, self.col_on, self.jp_on):
            w.toggled.connect(self.controls_changed)
            w.toggled.connect(lambda *_: self._old_photo_visibility())
        self.film_kind.currentIndexChanged.connect(self.controls_changed)
        self.film_kind.currentIndexChanged.connect(lambda *_: self._old_photo_visibility())
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
        self._dn_auto_ui()
        self._old_photo_visibility()        # before a photo is open too: sliders of tools that are off stay hidden
        self._glow_visibility()
        self._composite_controls()
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
        e.denoise.auto = self.dn_auto.isChecked()
        if self._dn_auto_in_effect():              # the photo chose: the two sliders only show its choice, and
            e.denoise.lean = round(s, 3)           # Strength leans lighter or heavier from it
        else:                                      # no Auto, or Auto could not choose here: the sliders decide
            e.denoise.luma = round(self.dn_l.value() * s, 3)
            e.denoise.chroma = round(self.dn_c.value() * s, 3)
        e.denoise.grain = round(self.dn_g.value(), 3)
        e.denoise.safeguard = self.dn_safe.isChecked()
        e.denoise.guard = self.dn_guard.isChecked()
        e.denoise.lift = self.dn_lift.isChecked()
        e.calibrated = self.dn_cal.isChecked()
        e.banding = self.band.isChecked()
        e.exposure_ev = round(self.ev.value(), 3)
        e.tone.contrast = round(self.con.value(), 3)
        e.tone.mode = ["standard", "camera_match", "linear", "lightroom"][self.tone.currentIndex()]
        e.tone.match_camera_brightness = self.mcb.isChecked()
        from ..engine.pipeline.pipeline import natural_curve
        want = "natural" if self.curve_col.currentIndex() == 0 else "rich"
        e.tone.curve_colour = ""
        implied = "natural" if natural_curve(e, self.image) else "rich"
        e.tone.curve_colour = "" if want == implied else want      # untouched photos stay untouched
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
        kind = self.film_kind.currentData() or ""
        if kind:
            f = dict(self.edit.film or {}) if (self.edit.film or {}).get("kind") == kind else {}
            f["kind"] = kind
            rng = round(self.film_range.value(), 3)
            if abs(rng - 1.0) > 1e-6:
                f["range"] = rng
            else:
                f.pop("range", None)
            e.film = f
        else:
            e.film = None
        sa, ss = round(self.st_amt.value()), round(self.st_sens.value())
        if self.st_on.isChecked() or (sa, ss) != (100, 50):
            e.stains = {"on": self.st_on.isChecked(), "amount": float(sa), "sensitivity": float(ss)}
        else:
            e.stains = None
        amt, size = round(self.scr_amt.value()), round(self.scr_size.value())
        if self.scr_on.isChecked() or (amt, size) != (50, 8):
            e.scratches = {"on": self.scr_on.isChecked(), "amount": float(amt), "size": float(size)}
        else:
            e.scratches = None
        ha = round(self.ht_amt.value())
        if self.ht_on.isChecked() or ha != 100:
            e.halftone = {"on": self.ht_on.isChecked(), "amount": float(ha)}
        else:
            e.halftone = None
        dh = round(self.dp_haze.value())
        e.depth = {"haze": float(dh)} if dh else None
        e.jpeg = {"on": True} if self.jp_on.isChecked() else None
        import copy as _cpy
        e.composite = _cpy.deepcopy(self._layers) or None
        ra = round(self.rl_amt.value())
        e.relight = ({"amount": float(ra), "azimuth": float(round(self.rl_az.value())),
                      "elevation": float(round(self.rl_el.value())), "soft": float(round(self.rl_soft.value()))}
                     if ra else None)
        ca = round(self.col_amt.value(), 3)
        cwm, ctn = round(self.col_warm.value()), round(self.col_tint.value())
        import copy as _cp
        if self.col_on.isChecked() or ca != 1.0 or self._hints or cwm or ctn:
            e.colourize = {"on": self.col_on.isChecked(), "amount": ca}
            if cwm or ctn:
                e.colourize["warm"], e.colourize["tint"] = float(cwm), float(ctn)
            if self._hints:
                e.colourize["hints"] = _cp.deepcopy(self._hints)
        else:
            e.colourize = None
        import copy as _copy
        e.masks = _copy.deepcopy(self._masks) or None
        e.retouch = _copy.deepcopy(self._spots) or None
        e.flare = self._flare_value()
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
        for box, on in ((self.dn_auto, e.denoise.auto), (self.dn_lift, e.denoise.lift),
                        (self.dn_cal, getattr(e, "calibrated", True))):
            box.blockSignals(True)
            box.setChecked(bool(on))
            box.blockSignals(False)
        self.dn_s.set_value(float(e.denoise.lean) if self._dn_auto_in_effect() else 1.0, emit=False)
        self.dn_l.set_value(e.denoise.luma, emit=False)
        self.dn_c.set_value(e.denoise.chroma, emit=False)
        self.dn_g.set_value(e.denoise.grain, emit=False)
        self.dn_m.setCurrentText(e.denoise.method)
        self.dn_safe.blockSignals(True)
        self.dn_safe.setChecked(bool(e.denoise.safeguard))
        self.dn_safe.blockSignals(False)
        self.dn_guard.blockSignals(True)
        self.dn_guard.setChecked(bool(e.denoise.guard))
        self.dn_guard.blockSignals(False)
        self._dn_method_text()
        self._dn_auto_ui()
        self.band.setChecked(e.banding)
        self.ev.set_value(e.exposure_ev, emit=False)
        self.con.set_value(e.tone.contrast, emit=False)
        self.tone.setCurrentIndex({"standard": 0, "camera_match": 1, "linear": 2, "lightroom": 3}.get(e.tone.mode, 0))
        self.mcb.setChecked(e.tone.match_camera_brightness)
        from ..engine.pipeline.pipeline import natural_curve
        self.curve_col.setCurrentIndex(0 if natural_curve(e, self.image) else 1)
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
        for w in (self.film_kind, self.ht_on, self.st_on, self.scr_on, self.col_on):
            w.blockSignals(True)
        hf = e.halftone or {}
        self.ht_on.setChecked(bool(hf.get("on")))
        self.ht_amt.set_value(float(hf.get("amount", 100)), emit=False)
        self.dp_haze.set_value(float((e.depth or {}).get("haze", 0.0)), emit=False)
        import copy as _cpy
        self._layers = _cpy.deepcopy(e.composite or [])
        self._lsel = min(max(self._lsel, 0), len(self._layers) - 1) if self._layers else -1
        self._fill_layers()
        from ..engine.local import relight as _rl
        rs = _rl.settings(e.relight)
        for w_, k_ in ((self.rl_amt, "amount"), (self.rl_az, "azimuth"), (self.rl_el, "elevation"),
                       (self.rl_soft, "soft")):
            w_.set_value(float(rs[k_]), emit=False)
        self.jp_on.blockSignals(True)
        self.jp_on.setChecked(bool((e.jpeg or {}).get("on")))
        self.jp_on.blockSignals(False)
        self.film_kind.setCurrentIndex(max(0, self.film_kind.findData((e.film or {}).get("kind", ""))))
        self.film_range.set_value(float((e.film or {}).get("range", 1.0)), emit=False)
        sn = e.stains or {}
        self.st_on.setChecked(bool(sn.get("on")))
        self.st_amt.set_value(float(sn.get("amount", 100)), emit=False)
        self.st_sens.set_value(float(sn.get("sensitivity", 50)), emit=False)
        sc = e.scratches or {}
        self.scr_on.setChecked(bool(sc.get("on")))
        self.scr_amt.set_value(float(sc.get("amount", 50)), emit=False)
        self.scr_size.set_value(float(sc.get("size", 8)), emit=False)
        co = e.colourize or {}
        self.col_on.setChecked(bool(co.get("on")))
        self.col_amt.set_value(float(co.get("amount", 1.0)), emit=False)
        self.col_warm.set_value(float(co.get("warm", 0.0)), emit=False)
        self.col_tint.set_value(float(co.get("tint", 0.0)), emit=False)
        import copy as _cp
        self._hints = _cp.deepcopy(co.get("hints") or [])
        self._hsel = min(self._hsel, len(self._hints) - 1)
        self._fill_hints()
        for w in (self.film_kind, self.ht_on, self.st_on, self.scr_on, self.col_on):
            w.blockSignals(False)
        self._old_photo_visibility()
        import copy as _copy
        self._masks = _copy.deepcopy(e.masks or [])
        self._msel = min(max(self._msel, 0), len(self._masks) - 1) if self._masks else -1
        self._spots = _copy.deepcopy(e.retouch or [])
        self._ssel = self._ssel if 0 <= self._ssel < len(self._spots) else -1
        self._flare = _copy.deepcopy(e.flare) if e.flare else None
        self._flare_controls()
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
        sr = getattr(self.image, "scan_noise_report", None) if self.image is not None else None
        if sr and self.image is not None and getattr(self.image.raw, "is_scan", False):
            if self.edit.denoise.enabled:
                self.noise_chip.setText(f"Noise: about {sr['levels']:.1f} levels (8-bit), measured in this picture")
                self.dn_safe_chip.setText(f"Protected: {sr['protected']:,} pixels kept as scanned "
                                          f"({sr['protected'] / max(sr['sites'], 1) * 1e6:.0f} per million)"
                                          if self.edit.denoise.safeguard else "Protected: -")
            else:
                self.dn_safe_chip.setText("Protected: -")
            return
        r = getattr(self.image, "denoise_report", None) if self.image is not None else None
        self._dn_guard_refresh(r)
        self._dn_auto_refresh()
        self._dn_cal_refresh()
        on = self.image is not None and self.edit.denoise.enabled and self.edit.denoise.safeguard
        if not on or not r:
            self.dn_safe_chip.setText("Protected: -")
            return
        self.dn_safe_chip.setText(f"Protected: {r['protected']:,} photosites kept as shot "
                                  f"({r['protected'] / max(r['sites'], 1) * 1e6:.0f} per million)")

    # ---- Clean: Auto, calibration, "What Clean removed" (BUILD_PLAN_NOISE C, D.4, E.4, H)
    def _is_scan(self) -> bool:
        image = getattr(self, "image", None)
        return image is not None and bool(getattr(getattr(image, "raw", None), "is_scan", False))

    def _dn_auto_active(self) -> bool:
        """Auto decides the strengths: ticked, and the picture is a raw (a scan or JPEG uses the sliders)."""
        return self.dn_auto.isChecked() and not self._is_scan()

    def _dn_auto_in_effect(self) -> bool:
        """Auto has chosen for the open photo (it is ticked, and this photo's report holds a choice). Only then
        does Strength lean from the choice; in every other case the sliders are what Clean uses."""
        image = getattr(self, "image", None)
        rep = getattr(image, "auto_report", None) if image is not None else None
        return self._dn_auto_active() and bool(rep) and "luma" in rep

    def _dn_auto_ui(self):
        """With Auto on, the two strength sliders only show what the photo chose: greyed, not editable. Where
        Auto could not choose (the report says why), they work as always."""
        scan = self._is_scan()
        image = getattr(self, "image", None)
        rep = getattr(image, "auto_report", None) if image is not None else None
        told_no = bool(rep) and "luma" not in rep            # Auto looked at this photo and could not choose
        auto = self._dn_auto_active() and not told_no
        self.dn_auto.setEnabled(not scan)
        for w in (self.dn_l, self.dn_c):
            w.setEnabled(not auto)
        for w in (self.dn_lift, self.dn_cal, self.dn_removed):
            w.setEnabled(not scan)

    def _dn_auto_toggled(self, on: bool):
        """Auto off: the sliders start from what the photo chose, so nothing jumps. Auto on: lean back to 100%."""
        r = getattr(self.image, "denoise_report", None) if self.image is not None else None
        if not on and r and r.get("auto"):
            self.dn_l.set_value(float(r["luma"]), emit=False)
            self.dn_c.set_value(float(r["chroma"]), emit=False)
        self.dn_s.set_value(1.0, emit=False)
        self._dn_auto_ui()

    def _dn_auto_refresh(self):
        """What Auto chose for this photo, how it knows, and how much fine detail that keeps."""
        if self.image is None or not self.edit.denoise.enabled or not self._dn_auto_active():
            self.dn_auto_chip.setText("Auto: off, the sliders decide" if not self.dn_auto.isChecked() else
                                      "Auto: for raw photos (this picture uses the sliders)" if self._is_scan() else
                                      "Auto: -")
            self.dn_auto_chip.setStyleSheet("")
            return
        rep = getattr(self.image, "auto_report", None) or {}
        r = getattr(self.image, "denoise_report", None) or {}
        note = rep.get("note") or ""
        if "luma" not in rep:
            self.dn_auto_chip.setText(note or "Auto: working it out...")
            self.dn_auto_chip.setStyleSheet(f"color:{T.AMBER}; border-color:{T.alpha(T.AMBER, 0.4)};" if note else "")
            for w in (self.dn_l, self.dn_c):       # nothing was chosen: the sliders decide, so they must work
                w.setEnabled(True)
            return
        lean = float(getattr(self.edit.denoise, "lean", 1.0) or 1.0)
        luma, chroma = (r["luma"], r["chroma"]) if r.get("auto") else (rep["luma"] * lean, rep["chroma"] * lean)
        for w in (self.dn_l, self.dn_c):
            w.setEnabled(False)
        self.dn_l.set_value(float(luma), emit=False)
        self.dn_c.set_value(float(chroma), emit=False)
        if abs(self.dn_s.value() - lean) > 1e-6:   # Strength shows the lean kept with this photo's edit
            self.dn_s.set_value(lean, emit=False)
        kind = rep.get("kind")
        measured = kind == "referee"
        col = T.CYAN if measured else T.AMBER
        if kind == "blend":
            text = f"Auto: {100 * rep['luma']:.0f}% of the learned network's cleaning"
        else:
            text = f"Auto: luminance {luma:.2f}, colour {chroma:.2f}"
        if rep.get("kept") is not None and abs(lean - 1.0) < 1e-3:
            text += f" · keeps about {100 * rep['kept']:.0f}% of the finest detail"
        if measured:
            text += " · measured by the two halves of this Dual Pixel photo"
            if rep.get("noise_ratio"):
                d = 100.0 * (float(rep["noise_ratio"]) - 1.0)
                text += (f" (its noise matches the camera's profile within {abs(d):.0f}%)" if abs(d) < 8 else
                         f" (its noise is {abs(d):.0f}% {'above' if d > 0 else 'below'} the camera's profile)")
        else:
            text += " · estimated from this photo"
        if abs(lean - 1.0) >= 1e-3:
            text += f" · leaned to {100 * lean:.0f}% by the Strength slider"
        self.dn_auto_chip.setText(text + (". " + note if note else ""))
        self.dn_auto_chip.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")

    def _dn_cal_refresh(self):
        """What this camera's calibration took out of the photo before Clean (cyan: measured on this body)."""
        if self.image is None or self._is_scan():
            self.dn_cal_chip.setText("Calibration: -")
            self.dn_cal_chip.setStyleSheet("")
            return
        if not self.dn_cal.isChecked():
            self.dn_cal_chip.setText("Calibration: not used")
            self.dn_cal_chip.setStyleSheet("")
            return
        rep = getattr(self.image, "calibration_report", None)
        if not rep:
            self.dn_cal_chip.setText("Calibration: nothing measured for this camera at this ISO yet (Cameras page: "
                                     "drop its lens-cap frames, then Deep calibration)")
            self.dn_cal_chip.setStyleSheet("")
            return
        bits = []
        if "hot pixels" in rep.get("used", ()):
            bits.append(f"{rep.get('hot', 0):,} hot and {rep.get('warm', 0):,} warm photosites healed "
                        f"(of {rep.get('mapped', 0):,} on this camera's map)")
        if "fixed pattern" in rep.get("used", ()):
            bits.append(f"fixed banding taken out (rows {rep.get('rows_dn', 0):.2f}, columns "
                        f"{rep.get('cols_dn', 0):.2f} raw levels)")
        if "black" in rep.get("used", ()):
            bits.append(f"black level corrected by up to {max(abs(v) for v in rep['black']):.2f} raw levels")
        self.dn_cal_chip.setText("Calibration: " + "; ".join(bits))
        self.dn_cal_chip.setStyleSheet(f"color:{T.CYAN}; border-color:{T.alpha(T.CYAN, 0.4)};")

    def _dn_show_removed(self):
        if self.image is None or self._is_scan():
            self.status.emit("Open a raw photo first: this view is for raw files.")
            return
        from .removed import RemovedDialog
        edit = self.edit_from_controls()
        if not edit.denoise.enabled:
            edit.denoise.enabled = True            # the question is what Clean WOULD remove: answer it either way
        self._removed_dialog = RemovedDialog(self.image, edit, self)
        self._removed_dialog.show()

    def _dn_guard_refresh(self, r):
        """The detail guard's number, and which network cleaned the photo when the learned method is chosen."""
        on = self.image is not None and self.edit.denoise.enabled
        g = (r or {}).get("guard") if on and self.edit.denoise.guard else None
        if g:
            self.dn_guard_chip.setText(f"Detail returned: {100 * g['returned']:.2f}% of what was removed")
            self.dn_guard_chip.setStyleSheet(f"color:{T.CYAN}; border-color:{T.alpha(T.CYAN, 0.4)};")
        else:
            self.dn_guard_chip.setText("Detail returned: -")
            self.dn_guard_chip.setStyleSheet("")
        learned = (r or {}).get("learned") if on else None
        if learned and self.dn_m.currentText() == "learned":
            self.dn_info.setText(f"Learned: {learned}." if "Hybrid used" not in learned else
                                 "Learned: no model for this camera yet, so Hybrid was used. Train one on the "
                                 "Cameras page.")

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
        ir = e.infrared or {}
        if ir.get("on"):
            m = wb.multipliers if (wb.mode == "multipliers" and wb.multipliers) else None
            nums = f" (R {m[0]:.2f} · G {m[1]:.2f} · B {m[2]:.2f})" if m else ""
            col = T.CYAN if str(ir.get("wb_source", "")).startswith("measured") else T.AMBER
            self.wb_chip.setText(f"IR white balance: set in Infrared{nums}")
            self.wb_chip.setToolTip("This photo is edited as an infrared / full-spectrum raw: its white balance and "
                                    "channel mix are set in the Infrared tab. Temperature and Tint mean nothing so "
                                    "far from daylight, so they are not used.")
            self.wb_chip.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")
            for w in (self.temp, self.tint):
                w.setEnabled(False)
            return
        self.wb_chip.setToolTip("")
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
        for b, k in ((self.pick_wb, "wb"), (self.pick_rng, "range"), (self.pick_base, "film")):
            b.blockSignals(True)
            b.setChecked(kind == k)
            b.blockSignals(False)
        self.viewer.setCursor(Qt.CursorShape.CrossCursor if kind else Qt.CursorShape.ArrowCursor)
        self.viewer.picking = bool(kind)
        if kind == "wb":
            self.status.emit("Click something in the photo that should be neutral grey or white (Esc cancels).")
        elif kind == "range":
            self.status.emit("Click a colour in the photo to choose it on the colour wheel (Esc cancels).")
        elif kind == "film":
            self.status.emit("Click clear film with no picture on it: between frames or at the edge (Esc cancels).")

    def _pick_film_base_at(self, u, v):
        self._set_pick(None)
        if not (self.edit.film or {}).get("kind"):
            self.status.emit("Choose Black and white negative or Colour negative first.")
            return
        img, e0, path = self.image, self.edit_from_controls(), self.path

        def work():
            from ..engine.pipeline.pipeline import film_base_at
            return film_base_at(img, e0, u, v)

        def done(p, base):
            if p != self.path or not (self.edit.film or {}).get("kind"):
                return
            from .. import engine as E
            e = E.Edit.from_dict(self.edit.to_dict())
            e.film = {k: v for k, v in (e.film or {}).items() if k not in ("lo", "hi", "stops")}
            e.film["base"] = base
            self.commit(e, "Film base picked: the negative is measured again from it.", "Pick the film base")
        run_bg(work, path, done, lambda _, err: self.status.emit(f"Film base problem: {err}"))

    def _film_base_auto(self):
        if self.image is None or not (self.edit.film or {}).get("kind"):
            return
        from .. import engine as E
        e = E.Edit.from_dict(self.edit.to_dict())
        e.film = {k: v for k, v in (e.film or {}).items() if k not in ("base", "lo", "hi", "stops")}
        self.commit(e, "Film base: the clearest part of the film is used.", "Find the film base")

    # ------------------------------------------------------------------------------------ colour hints
    def _set_hint_tool(self, on):
        if on and self.image is None:
            on = False
        if on:
            if self.viewer.tool and self.viewer.tool != "hint":
                self._set_mask_tool(None)
            self.viewer.tool = "hint"
            self.viewer.setCursor(Qt.CursorShape.CrossCursor)
            self.status.emit("Click a part of the photo to give it a colour. Shift-click adds more of it, Alt-click "
                             "takes a piece away. Esc when done.")
            self._render_hint_overlay()
        elif self.viewer.tool == "hint":
            self.viewer.tool = None
            self.viewer.setCursor(Qt.CursorShape.ArrowCursor)
            self.viewer.overlay = None
            self.viewer.update()
        self.hint_b.blockSignals(True)
        self.hint_b.setChecked(bool(on))
        self.hint_b.blockSignals(False)

    def _fill_hints(self):
        self.hint_list.blockSignals(True)
        self.hint_list.clear()
        for i, h in enumerate(self._hints):
            r, g, b = (int(round(255 * c)) for c in h.get("rgb", [0.5, 0.5, 0.5]))
            it = QListWidgetItem(f"Part {i + 1}   #{r:02X}{g:02X}{b:02X}" + ("" if h.get("data") else "  (finding...)"))
            pm = QImage(14, 14, QImage.Format.Format_RGB32)
            pm.fill(QColor(r, g, b))
            from PySide6.QtGui import QIcon, QPixmap
            it.setIcon(QIcon(QPixmap.fromImage(pm)))
            self.hint_list.addItem(it)
        if 0 <= self._hsel < len(self._hints):
            self.hint_list.setCurrentRow(self._hsel)
        self.hint_list.blockSignals(False)
        self._old_photo_visibility()

    def _pick_colour(self, start, title):
        from PySide6.QtWidgets import QColorDialog
        r, g, b = (int(round(255 * c)) for c in start)
        c = QColorDialog.getColor(QColor(r, g, b), self, title)
        if not c.isValid():
            return None
        return [round(c.redF(), 4), round(c.greenF(), 4), round(c.blueF(), 4)]

    def _hint_click(self, u, v, alt, shift):
        if self.image is None:
            return
        pt = self._s_points([(u, v)])[0]
        sel = self._hsel if 0 <= self._hsel < len(self._hints) else (len(self._hints) - 1 if self._hints else -1)
        if (alt or shift) and sel >= 0:
            idx = sel
            h = dict(self._hints[idx])
            h["fg"], h["bg"] = list(h.get("fg", [])), list(h.get("bg", []))
            (h["bg"] if alt else h["fg"]).append(pt)
        elif alt:
            self.status.emit("Click a part first; Alt-click then takes a piece of it away.")
            return
        else:
            start = [0.5, 0.5, 0.5]
            if self.last_a8 is not None:                  # start from the colour shown there (the model's guess)
                a8 = self.last_a8
                hh, ww = a8.shape[:2]
                x, y = min(ww - 1, int(u * ww)), min(hh - 1, int(v * hh))
                start = [float(c) / 255.0 for c in np.median(a8[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
                                                               .reshape(-1, 3), axis=0)]
            rgb = self._pick_colour(start, "Colour for this part")
            if rgb is None:
                return
            idx = len(self._hints)
            h = {"fg": [pt], "bg": [], "rgb": rgb, "spread": 0.08}
        self._hint_area(idx, h, "Part coloured. Shift-click adds more of it, Alt-click takes a piece away; click "
                                "elsewhere for another part. Esc when done.")

    def _hint_area(self, idx, h, message):
        """Work out a hint's area in the background, then put it in the list and render."""
        from .. import engine as E
        img, e0, path = self.image, self.edit_from_controls(), self.path
        fg, bg = list(h.get("fg", [])), list(h.get("bg", []))
        spread = float(h.get("spread", 0.08))
        model = self._sel_model()
        self.status.emit(self._sel_status(model, "part"))
        self.viewer.busy = True
        self.viewer.update()

        def done(p, area):
            self.viewer.busy = False
            if p != self.path:
                return
            h.pop("model", None)
            h.update(area)
            if idx < len(self._hints):
                self._hints[idx] = h
            else:
                self._hints.append(h)
            self._hsel = idx
            self._fill_hints()
            self.controls_changed()
            self._render_hint_overlay()
            self.status.emit(message)
        run_bg(lambda: E.colour_hint_area(img, e0, fg, bg, spread, model=model), path, done,
               lambda _, err: (setattr(self.viewer, "busy", False), self.status.emit(f"Colour a part: {err}")))

    def _hint_selected(self, row):
        self._hsel = row
        if 0 <= row < len(self._hints):
            self.hint_spread.set_value(round(100 * float(self._hints[row].get("spread", 0.08))), emit=False)
            self.hint_spread.setVisible(not self._hints[row].get("model"))
        self._render_hint_overlay()

    def _hint_respread(self):
        i = self._hsel
        if not (0 <= i < len(self._hints)) or self.image is None:
            return
        h = dict(self._hints[i])
        sp_ = round(self.hint_spread.value() / 100.0, 4)
        if abs(float(h.get("spread", 0.08)) - sp_) < 1e-6:
            return
        h["spread"] = sp_
        self._hint_area(i, h, f"Part {i + 1}: spread {self.hint_spread.value():.0f}.")

    def _change_show(self, on):
        """Show what the automatic tools changed: a heat map (clear = untouched, yellow to red = more change)."""
        self.viewer.show_overlay = (bool(on) or self.st_show.isChecked() or self.mask_show.isChecked()
                                    or self.dp_show.isChecked())
        if not on:
            self.viewer.overlay = None
            self.viewer.update()
        self._update_overlay()

    def _render_change_overlay(self):
        from .. import engine as E
        if self.image is None:
            return
        e = self.edit_from_controls()
        img = self.image
        sc = max(0.05, self.preview_scale(e) * 0.5)
        self._ov_token = getattr(self, "_ov_token", 0) + 1
        tok = self._ov_token

        def ready(t, a):
            if t != getattr(self, "_ov_token", -1) or not self.chg_show.isChecked():
                return
            hh, ww = a.shape
            rgba = np.zeros((hh, ww, 4), np.uint8)
            rgba[..., 0] = 255                                    # yellow (a little) to red (a lot)
            rgba[..., 1] = (255 * (1.0 - 0.85 * a)).astype(np.uint8)
            rgba[..., 3] = np.clip(np.sqrt(a) * 0.75 * 255, 0, 255).astype(np.uint8)
            self.viewer.overlay = QImage(np.ascontiguousarray(rgba).data, ww, hh, 4 * ww,
                                         QImage.Format.Format_RGBA8888).copy()
            self.viewer.update()
            changed = float((a > 0.05).mean())
            self.status.emit(f"The automatic tools changed {100 * changed:.1f}% of the photo (red: most change).")
        self.status.emit("Comparing the photo with and without the automatic tools...")
        run_bg(lambda: E.change_map(img, e, sc), tok, ready,
               lambda _, err: self.status.emit(f"Change overlay problem: {err}"))

    def _depth_show(self, on):
        """Show depth map: the photo coloured by distance (red and yellow near, blue far)."""
        self.viewer.show_overlay = (bool(on) or self.chg_show.isChecked() or self.st_show.isChecked()
                                    or self.mask_show.isChecked())
        if not on:
            self.viewer.overlay = None
            self.viewer.update()
        self._update_overlay()

    def _render_depth_overlay(self):
        from .. import engine as E
        from ..engine.local import depth as dp
        if self.image is None:
            return
        if not dp.available():
            self.status.emit("The depth model is missing from models\\depth. Run Convert Models.bat.")
            return
        e = self.edit_from_controls()
        img = self.image
        sc = max(0.05, self.preview_scale(e) * 0.5)
        self._ov_token = getattr(self, "_ov_token", 0) + 1
        tok = self._ov_token

        def ready(t, rgba):
            if t != getattr(self, "_ov_token", -1) or not self.dp_show.isChecked():
                return
            hh, ww = rgba.shape[:2]
            self.viewer.overlay = QImage(np.ascontiguousarray(rgba).data, ww, hh, 4 * ww,
                                         QImage.Format.Format_RGBA8888).copy()
            self.viewer.update()
            self._depth_chip()
            self.status.emit("Depth map: red and yellow are near, blue is far (an order, not metres).")
        self.status.emit("Working out how far away each part of the photo is (a few seconds, once per photo)...")
        run_bg(lambda: E.depth_view(img, e, sc), tok, ready,
               lambda _, err: self.status.emit(f"Depth map problem: {err}"))

    def _depth_chip(self):
        from ..engine.local import depth as dp
        if not dp.available():
            self.dp_chip.setText("Depth: the model is not installed (models\\depth)")
            self.dp_chip.setStyleSheet(f"color:{T.DIM};")
            return
        r = getattr(self.image, "depth_report", None) if self.image is not None else None
        if r and r.get("source") == "measured":
            self.dp_chip.setText("Depth: measured by the focus stack (Dual Pixel and focus)")
            self.dp_chip.setStyleSheet(f"color:{T.CYAN}; border-color:{T.alpha(T.CYAN, 0.4)};")
        elif r:
            self.dp_chip.setText(f"Depth: estimated on this PC in {r['ms'] / 1000:.1f} s (Depth Anything)")
            self.dp_chip.setStyleSheet(f"color:{T.AMBER}; border-color:{T.alpha(T.AMBER, 0.4)};")
        else:
            self.dp_chip.setText("Depth: worked out when first used (a few seconds)")
            self.dp_chip.setStyleSheet(f"color:{T.DIM};")

    def _stain_show(self, on):
        """Show what's lost: the cores of strong stains (amber), where the picture may be gone."""
        self.viewer.show_overlay = (bool(on) or self.chg_show.isChecked() or self.mask_show.isChecked()
                                    or self.dp_show.isChecked())
        if not on:
            self.viewer.overlay = None
            self.viewer.update()
        self._update_overlay()

    def _render_stain_overlay(self):
        from .. import engine as E
        if self.image is None:
            return
        e = self.edit_from_controls()
        img = self.image
        sc = max(0.05, self.preview_scale(e) * 0.5)
        self._ov_token = getattr(self, "_ov_token", 0) + 1
        tok = self._ov_token

        def work():
            shp = E.stain_lost_shape(img, e)
            if shp is None:
                return None
            e2 = type(e).from_dict(e.to_dict())
            e2.masks = [{"name": "lost", "enabled": True, "adjust": {}, "shapes": [shp]}]
            return E.render_mask(img, e2, 0, sc)

        def ready(t, a):
            if t != getattr(self, "_ov_token", -1) or not self.st_show.isChecked():
                return
            if a is None:
                self.viewer.overlay = None
                self.viewer.update()
                return
            hh, ww = a.shape
            rgba = np.zeros((hh, ww, 4), np.uint8)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = 255, 176, 0          # amber: the picture may be gone here
            rgba[..., 3] = np.clip(a * 0.55 * 255, 0, 255).astype(np.uint8)
            self.viewer.overlay = QImage(np.ascontiguousarray(rgba).data, ww, hh, 4 * ww,
                                         QImage.Format.Format_RGBA8888).copy()
            self.viewer.update()
        self.status.emit("Marking where the picture may be lost under stains...")
        run_bg(work, tok, ready, lambda _, err: self.status.emit(f"Stain overlay problem: {err}"))

    def _render_hint_overlay(self):
        """While colouring parts, the selected part is shown tinted in its own colour, so you see exactly where
        the colour will go."""
        from .. import engine as E
        i = self._hsel
        if self.image is None or self.viewer.tool != "hint" or not (0 <= i < len(self._hints)) \
                or not self._hints[i].get("data"):
            if self.viewer.tool == "hint":
                self.viewer.overlay = None
                self.viewer.update()
            return
        h = self._hints[i]
        e = self.edit_from_controls()
        e.masks = [{"name": "part", "enabled": True, "adjust": {},
                    "shapes": [{"type": "object", "op": "add", "w": h["w"], "h": h["h"], "data": h["data"]}]}]
        img = self.image
        sc = max(0.05, self.preview_scale(e) * 0.5)
        rgb = tuple(int(round(255 * c)) for c in h.get("rgb", [1, 0.3, 0.3]))
        self._ov_token = getattr(self, "_ov_token", 0) + 1
        tok = self._ov_token

        def ready(t, a):
            if t != getattr(self, "_ov_token", -1) or self.viewer.tool != "hint":
                return
            hh, ww = a.shape
            rgba = np.zeros((hh, ww, 4), np.uint8)
            edge = np.minimum((np.abs(np.gradient(a)[0]) + np.abs(np.gradient(a)[1])) * 4, 1.0)
            for c in range(3):                         # a light tint of the part's colour, a white outline
                rgba[..., c] = (rgb[c] * (1 - edge) + 255 * edge).astype(np.uint8)
            rgba[..., 3] = np.clip((a * 0.25 * (1 - edge) + edge * 0.95) * 255, 0, 255).astype(np.uint8)
            self.viewer.overlay = QImage(np.ascontiguousarray(rgba).data, ww, hh, 4 * ww,
                                         QImage.Format.Format_RGBA8888).copy()
            self.viewer.update()
        run_bg(lambda: E.render_mask(img, e, 0, sc), tok, ready,
               lambda _, err: self.status.emit(f"Part overlay problem: {err}"))

    def _hint_recolour(self):
        i = self.hint_list.currentRow()
        if not (0 <= i < len(self._hints)):
            return
        rgb = self._pick_colour(self._hints[i].get("rgb", [0.5, 0.5, 0.5]), f"Colour for part {i + 1}")
        if rgb is None:
            return
        self._hints[i]["rgb"] = rgb
        self._fill_hints()
        self.controls_changed()

    def _hint_remove(self):
        i = self.hint_list.currentRow()
        if not (0 <= i < len(self._hints)):
            return
        del self._hints[i]
        self._hsel = min(i, len(self._hints) - 1)
        self._fill_hints()
        self.controls_changed()

    def _old_photo_visibility(self):
        neg = bool(self.film_kind.currentData())
        self.film_btns.setVisible(neg)
        self.film_range.setVisible(neg)
        for w in (self.st_amt, self.st_sens, self.st_show):
            w.setVisible(self.st_on.isChecked())
        if not self.st_on.isChecked() and self.st_show.isChecked():
            self.st_show.setChecked(False)
        self.scr_amt.setVisible(self.scr_on.isChecked())
        self.scr_size.setVisible(self.scr_on.isChecked())
        self.col_amt.setVisible(self.col_on.isChecked())
        self.col_warm.setVisible(self.col_on.isChecked())
        self.col_tint.setVisible(self.col_on.isChecked())
        self.ht_amt.setVisible(self.ht_on.isChecked())
        for w_ in (self.rl_az, self.rl_el, self.rl_soft):
            w_.setVisible(self.rl_amt.value() > 0)
        from .. import engine as E
        jp = self.image is not None and E.jpeg_repairable(self.image)
        self.jp_on.setVisible(jp or self.jp_on.isChecked())
        self.jp_chip.setVisible(jp)
        self.hint_list.setVisible(bool(self._hints))
        self.hint_btns.setVisible(bool(self._hints))
        sel = self._hints[self._hsel] if 0 <= self._hsel < len(self._hints) else {}
        self.hint_spread.setVisible(bool(self._hints) and not sel.get("model"))   # SAM outlines have no spread

    def _old_photo_refresh(self):
        """The Old photo and film chips: what the film measure, the dust and scratch pass and the colourizer did."""
        from .. import engine as E
        e = self.edit
        if (e.film or {}).get("kind") and self.image is not None:
            try:
                p = E.film_params(self.image, e)
                src = "picked" if "base" in (e.film or {}) else "found"
                self.film_chip.setText(f"Film: {'colour' if p['kind'] == 'colour' else 'black and white'} "
                                       f"negative, base {src}, {p['stops'] * float(p.get('range', 1.0)):.1f} stops "
                                       "of scene range")
                self.film_chip.setStyleSheet(f"color:{T.AMBER}; border-color:{T.alpha(T.AMBER, 0.4)};")
            except Exception:
                slog.log_exception("film report failed", log)
        else:
            self.film_chip.setText("Film: a positive")
            self.film_chip.setStyleSheet(f"color:{T.DIM};")
        if (e.halftone or {}).get("on"):
            hr = getattr(self.image, "halftone_report", None) if self.image is not None else None
            if hr and hr.get("peaks"):
                txt = (f"Printing dots: a screen every {hr['period']:.1f} px at {hr['angle']:.0f}° found and "
                       "taken out")
                col = T.VIOLET
            elif hr is not None:
                txt = "Printing dots: no printing screen found, nothing changed"
                col = T.DIM
            else:
                txt, col = "Printing dots: on", T.VIOLET
            self.ht_chip.setText(txt)
            self.ht_chip.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")
        else:
            self.ht_chip.setText("Printing dots: off")
            self.ht_chip.setStyleSheet(f"color:{T.DIM};")
        if self.chg_show.isChecked():
            self._update_overlay()
        self._depth_chip()
        from ..engine.local import relight as _rl
        if _rl.active(e.relight):
            rs = _rl.settings(e.relight)
            txt = f"Relight: light {_rl.direction_name(rs['azimuth'], rs['elevation'])}"
            rr = getattr(self.image, "relight_report", None) if self.image is not None else None
            if rr:
                txt += f"; the photo's own light measured {_rl.direction_name(rr['azimuth'], rr['elevation'])}"
            self.rl_chip.setText(txt)
            self.rl_chip.setStyleSheet(f"color:{T.VIOLET}; border-color:{T.alpha(T.VIOLET, 0.4)};")
        else:
            self.rl_chip.setText("Relight: off")
            self.rl_chip.setStyleSheet(f"color:{T.DIM};")
        if (e.jpeg or {}).get("on"):
            jr = getattr(self.image, "jpeg_report", None) if self.image is not None else None
            if jr:
                self.jp_chip.setText(f"JPEG repair: saved at about quality {jr['quality']}; repaired within what "
                                     f"the file allows ({jr['ms'] / 1000:.1f} s)")
            else:
                self.jp_chip.setText("JPEG repair: on")
            self.jp_chip.setStyleSheet(f"color:{T.VIOLET}; border-color:{T.alpha(T.VIOLET, 0.4)};")
        else:
            self.jp_chip.setText("JPEG repair: off")
            self.jp_chip.setStyleSheet(f"color:{T.DIM};")
        if (e.stains or {}).get("on"):
            sr = getattr(self.image, "stain_report", None) if self.image is not None else None
            if sr:
                txt = (f"Stains: {100 * sr['stained']:.0f}% of the photo unmixed; {100 * sr['lost']:.1f}% may have "
                       "lost the picture (left as it is)")
            else:
                txt = "Stains: on"
            self.st_chip.setText(txt)
            self.st_chip.setStyleSheet(f"color:{T.VIOLET}; border-color:{T.alpha(T.VIOLET, 0.4)};")
        else:
            self.st_chip.setText("Stains: off")
            self.st_chip.setStyleSheet(f"color:{T.DIM};")
        if self.st_show.isChecked():
            self._update_overlay()
        r = getattr(self.image, "scratch_report", None) if self.image is not None else None
        if (e.scratches or {}).get("on"):
            from ..engine.local import scratchnet
            how = "learned detector + filter" if scratchnet.available() else "filter only (detector not installed)"
            n = (r or {}).get("marks")
            txt = f"Dust and scratches: {n} marks taken out ({how})" if n is not None else \
                f"Dust and scratches: on ({how})"
            if r and r.get("half"):
                txt += "; the full-size photo is searched more finely"
            col = T.VIOLET if scratchnet.available() else T.AMBER
            self.scr_chip.setText(txt)
            self.scr_chip.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")
        else:
            self.scr_chip.setText("Dust and scratches: off")
            self.scr_chip.setStyleSheet(f"color:{T.DIM};")
        from ..engine.local import colourize as cz
        nh = len(cz.hints_of(e.colourize))
        if (e.colourize or {}).get("on") or nh:
            if cz.available() or not (e.colourize or {}).get("on"):     # your own colours need no model
                rep = getattr(self.image, "colour_report", None) or {}
                txt = ("Colourize: DDColor" + (f", {rep['ms']:.0f} ms" if rep.get("ms") else "")) \
                    if (e.colourize or {}).get("on") else "Colourize: your colours only"
                if nh:
                    txt += f"; {nh} part{'s' if nh != 1 else ''} coloured by you"
                col = T.VIOLET
            else:
                txt = "Colourize: the colour model is missing (models/colour/ddcolor.npz); reinstall or update." + \
                    (f" Your {nh} coloured part{'s' if nh != 1 else ''} still show." if nh else "")
                col = T.AMBER
            self.col_chip.setText(txt)
            self.col_chip.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")
        else:
            self.col_chip.setText("Colourize: off" if cz.available() else "Colourize: off (DDColor model not "
                                  "installed yet)")
            self.col_chip.setStyleSheet(f"color:{T.DIM};")

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
        if "night_k" in info:
            parts.append(f"night {info['night_k']} K")
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
            try:
                i = E.add_version(self.path, self.edit)
            except OSError as err:
                self._sidecar_problem("The virtual copy could not be made", err)
                self._fill_versions()
                return
            self._switch_version(i, f"Virtual copy made from {name}. Edit it freely: {name} is untouched.")
        elif data == "del":
            name = E.list_versions(self.path)[self.version]
            if QMessageBox.question(self, "Delete copy", f"Delete '{name}'? The master and other copies stay.") \
                    != QMessageBox.StandardButton.Yes:
                self._fill_versions()
                return
            try:
                E.delete_version(self.path, self.version)
            except (OSError, IndexError) as err:
                self._sidecar_problem("The copy could not be deleted", err)
                self._fill_versions()
                return
            self._switch_version(0, f"'{name}' deleted.")
        elif data is not None and int(data) != self.version:
            self._switch_version(int(data))

    def _switch_version(self, i, msg=None):
        from .. import engine as E
        self.version = i
        self._reset_history(E.load_version(self.path, i), E.load_history(self.path, i))
        self.controls_from_edit(self.edit)
        self._canonical()
        self._mark_saved()
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
        try:
            E.add_snapshot(self.path, name.strip() or f"Snapshot {n + 1}", self.edit)
        except OSError as err:
            self._sidecar_problem("The snapshot could not be kept", err)
            return
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
        try:
            E.delete_snapshot(self.path, i)
        except (OSError, IndexError) as err:
            self._sidecar_problem("The snapshot could not be deleted", err)
        self._fill_snapshots()

    def _sidecar_problem(self, what: str, err) -> None:
        """A sidecar could not be written (photos on a read-only card or drive): said plainly, with the way out."""
        log.warning("%s: %s", what, err)
        self.status.emit(f"{what}: the photo's folder cannot be written to ({err}). In the Library, Tools > Keep "
                         f"this folder's ratings and edits on this PC stores them here instead.")

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

    def _tone_mode_note(self, idx):
        """Say what Lightroom-like uses for this photo (the lift is read from Lightroom's profiles on this PC)."""
        if idx != 3 or self.image is None:
            return
        from .. import engine as E
        try:
            info = E.lightroom_info(self.image, self.edit_from_controls())
        except Exception as exc:                              # noqa: BLE001
            report_problem("reading Lightroom's brightness for this camera", exc)
            return
        lift = (f"Lightroom's brightness lift for this camera: {info['lift']:+.2f} stops (read from Lightroom's "
                f"profiles on this PC)" if info["lift_known"] else
                "Lightroom's brightness lift for this camera is not known on this PC (no Lightroom camera profiles "
                "for it), so none is added")
        curve = ("the profile's own curve" if info["curve"] == "profile" else "Adobe's default curve")
        self.status.emit(f"Lightroom-like: {lift}; {curve}. Estimated, until checked against a Lightroom export.")

    def _pick_proof_profile(self):
        start = self.settings.get("proof_profile", "") or (r"C:\Windows\System32\spool\drivers\color"
                                                           if os.name == "nt" else "")
        f, _ = QFileDialog.getOpenFileName(self, "Printer / paper profile", start, "ICC profile (*.icc *.icm)")
        if not f:
            return
        self.settings.set("proof_profile", f)
        self.sp_on.setChecked(True)
        self._proof_changed()

    def _proof_changed(self):
        from ..engine.color import display
        prof = self.settings.get("proof_profile", "") or ""
        on = self.sp_on.isChecked() and bool(prof) and os.path.isfile(prof)
        if self.sp_on.isChecked() and not on:
            self.sp_info.setText("Soft proof: choose the printer / paper profile first")
            self.sp_info.setStyleSheet(f"color:{T.AMBER};")
        elif on:
            self.sp_info.setText(f"Soft proof: {os.path.basename(prof)}")
            self.sp_info.setStyleSheet(f"color:{T.CYAN}; border-color:{T.alpha(T.CYAN, 0.4)};")
        else:
            self.sp_info.setText("Soft proof: off" if prof else "Soft proof: no printer profile chosen")
            self.sp_info.setStyleSheet("")
        display.set_proof(prof if on else None, ["perceptual", "relative"][self.sp_intent.currentIndex()],
                          self.sp_gamut.isChecked())
        self.before_img = None
        if self.image is not None:
            self.render_now()

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
        if self.image is not None:
            from ..engine.color import spectral as sp
            try:
                has_ssf = sp.camera_name(getattr(self.image.meta, "make", ""), cam or "") is not None
            except (OSError, ValueError) as exc:
                report_problem("reading the spectral camera data", exc)
                has_ssf = False
            if has_ssf:
                self.prof_combo.addItem(dot_icon(T.VIOLET), f"Profile: {sp.NAME}", sp.KEY)
        for pr in profs:
            self.prof_combo.addItem(dot_icon(T.VIOLET if pr["source"] == "shared" else T.CYAN),
                                    f"Profile: {pr['name']}", pr["file"])
        if self.image is not None:
            try:
                lr = dcp.lightroom_profiles(getattr(self.image.meta, "make", ""), cam or "")
            except Exception as exc:                          # noqa: BLE001  (a list must never stop the app)
                report_problem("listing Lightroom's camera profiles", exc)
                lr = []
            for pr in lr:
                self.prof_combo.addItem(dot_icon(T.AMBER), f"Profile: {pr['name']} (Lightroom)", pr["file"])
        if current and self.prof_combo.findData(current) < 0:
            what = (f"{current[3:]} (Lightroom: not on this PC, so the camera matrix is used)"
                    if current.startswith(dcp.LR_PREFIX) else f"{current} (not for this camera, or missing)")
            self.prof_combo.addItem(f"Profile: {what}", current)
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
                  "skin": "Skin", "person": "Person", "depth": "Depth", "channel": "Channel",
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
                  "radial": (self.r_feather,), "depth": (self.d_lo, self.d_hi, self.d_soft),
                  "channel": (self.ch_lo, self.ch_hi, self.ch_soft)}
        shown = groups.get(t, ())
        for w in self._range_ws:
            w.setVisible(w in shown)
        self.ch_row.setVisible(t == "channel")
        if t == "channel":
            self.ch_src.setCurrentIndex(max(0, self.ch_src.findData(shapes[-1].get("source", "r/b"))))
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
                    self.r_feather: 100 * sh.get("feather", 0.5), self.d_lo: 100 * sh.get("lo", 0.6),
                    self.d_hi: 100 * sh.get("hi", 1.0), self.d_soft: 100 * sh.get("soft", 0.1),
                    self.ch_lo: 100 * sh.get("lo", 0.5), self.ch_hi: 100 * sh.get("hi", 1.0),
                    self.ch_soft: 100 * sh.get("soft", 0.1)}
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
        if kind == "depth":
            self._depth_mask(op)
            return
        if kind == "channel":
            self._channel_mask(op)
            return
        if kind == "luminance":
            self._add_shape({"type": "luminance", "op": op, "lo": 0.6, "hi": 1.0, "soft": 0.1})
            self.mask_show.setChecked(True)
            self.status.emit("Brightness range: set From and To below (the red overlay shows what is selected), then "
                             "use the sliders under 'Adjust inside the mask'.")
            return
        self._mask_pending = (kind, op)
        self._set_mask_tool(kind)

    def _fill_sel_models(self):
        """Click-to-select choices: each SAM 2.1 size (greyed out with how to get it when not installed) and
        Classic; the last choice is remembered."""
        from ..engine.local import sam
        have = sam.installed()
        cb = self.sel_model
        cb.blockSignals(True)
        cb.clear()
        for size, name in sam.SIZES.items():
            if size in have:
                cb.addItem(f"SAM 2.1 {name}", size)
            else:
                cb.addItem(f"SAM 2.1 {name.split(' (')[0]}: not installed (Convert Models.bat)", size)
                cb.model().item(cb.count() - 1).setEnabled(False)
        cb.addItem("Classic (no model, instant)", "classic")
        want = self.settings.get("select_model") or (have[0] if have else "classic")
        if want != "classic" and want not in have:
            want = have[0] if have else "classic"
        cb.setCurrentIndex(max(0, cb.findData(want)))
        cb.blockSignals(False)

    def _sel_model_changed(self, *_):
        self.settings.set("select_model", self.sel_model.currentData())

    def _sel_model(self):
        """The SAM 2.1 size to use for the next click, or None for the Studio's own method."""
        from ..engine.local import sam
        return sam.choose(self.sel_model.currentData() if hasattr(self, "sel_model") else None)

    def _sel_status(self, model, what):
        """What to say while a click is worked out (the first SAM click on a photo takes a while)."""
        from ..engine.local import sam
        if model is None:
            return f"Finding the {what}..."
        r = getattr(self.image, "sam_report", None)
        if r and r.get("size") == model:
            return f"Finding the {what}..."
        secs = "about 10 seconds" if model == "tiny" else "up to a minute or two"
        return (f"SAM 2.1 {sam.SIZES[model].split(' (')[0]} is looking at the photo ({secs}, once per photo; "
                "after that each click is instant)...")

    # ------------------------------------------------------------------------------------ composite
    def _fill_layers(self):
        lw = self.cp_list
        lw.blockSignals(True)
        lw.clear()
        for L in self._layers:
            it = QListWidgetItem(L.get("name") or os.path.basename(L.get("src", "subject")))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if L.get("enabled", True) else Qt.CheckState.Unchecked)
            lw.addItem(it)
        if 0 <= self._lsel < len(self._layers):
            lw.setCurrentRow(self._lsel)
        lw.blockSignals(False)
        self._composite_controls()

    def _composite_controls(self):
        L = self._layers[self._lsel] if 0 <= self._lsel < len(self._layers) else None
        show = L is not None
        self.cp_list.setVisible(bool(self._layers))
        for w_ in (self.cp_x, self.cp_y, self.cp_size, self.cp_rot, self.cp_soft, self.cp_row, self.cp_btns):
            w_.setVisible(show)
        self.cp_mamt.setVisible(show and bool((L or {}).get("match")))
        if not show:
            return
        for w_, v in ((self.cp_x, 100 * L.get("x", 0.5)), (self.cp_y, 100 * L.get("y", 0.55)),
                      (self.cp_size, 100 * L.get("size", 0.5)), (self.cp_rot, L.get("rotate", 0.0)),
                      (self.cp_soft, L.get("feather", 1.0)),
                      (self.cp_mamt, 100 * float((L.get("match") or {}).get("amount", 1.0)))):
            w_.set_value(float(v), emit=False)
        for w_, v in ((self.cp_flip, L.get("flip", False)), (self.cp_grain, L.get("grain", True))):
            w_.blockSignals(True)
            w_.setChecked(bool(v))
            w_.blockSignals(False)

    def _composite_selected(self, row):
        self._lsel = row
        self._composite_controls()

    def _composite_item_changed(self, it):
        row = self.cp_list.row(it)
        if 0 <= row < len(self._layers):
            self._layers[row]["enabled"] = it.checkState() == Qt.CheckState.Checked
            self.controls_changed()

    def _composite_params(self, *_):
        if not (0 <= self._lsel < len(self._layers)):
            return
        L = self._layers[self._lsel]
        L["x"], L["y"] = round(self.cp_x.value() / 100.0, 4), round(self.cp_y.value() / 100.0, 4)
        L["size"] = round(self.cp_size.value() / 100.0, 4)
        L["rotate"] = round(self.cp_rot.value(), 1)
        L["feather"] = round(self.cp_soft.value(), 2)
        L["flip"], L["grain"] = self.cp_flip.isChecked(), self.cp_grain.isChecked()
        if L.get("match"):
            L["match"]["amount"] = round(self.cp_mamt.value() / 100.0, 3)
        self.controls_changed()

    def _composite_add(self):
        from ..engine.io.rawfile import RAW_EXTENSIONS
        from ..engine.io.scanfile import SCAN_EXTENSIONS
        from .subject_picker import SubjectPicker
        if self.image is None:
            return
        pat = " ".join(f"*{x}" for x in RAW_EXTENSIONS + SCAN_EXTENSIONS)
        start = os.path.dirname(self.path) if self.path else ""
        f, _ = QFileDialog.getOpenFileName(self, "The photo the subject is in", start, f"Photos ({pat})")
        if not f:
            return
        try:
            dlg = SubjectPicker(self, f, self._sel_model())
        except Exception as err:
            slog.log_exception("composite: could not open the source photo", log)
            self.status.emit(f"Could not open {os.path.basename(f)}: {err}")
            return
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.shape:
            return
        sh = dlg.shape
        n = 1 + sum(1 for L in self._layers if L.get("src") == f)
        L = {"src": f, "name": f"{os.path.splitext(os.path.basename(f))[0]} {n}",
             "mask": {"w": sh["w"], "h": sh["h"], "data": sh["data"]}, "fg": sh.get("fg", []),
             "bg": sh.get("bg", []), "model": sh.get("model", "classic"), "x": 0.5, "y": 0.55, "size": 0.5,
             "rotate": 0.0, "flip": False, "grain": True, "feather": 1.0, "enabled": True, "match": None}
        self._layers.append(L)
        self._lsel = len(self._layers) - 1
        self._fill_layers()
        self.controls_changed()
        self.status.emit("Subject placed in the middle. Move and size it with the sliders; Match its colours "
                         "re-develops it under this photo's light.")

    def _composite_delete(self):
        if 0 <= self._lsel < len(self._layers):
            self._layers.pop(self._lsel)
            self._lsel = min(self._lsel, len(self._layers) - 1)
            self._fill_layers()
            self.controls_changed()

    def _composite_match(self):
        """Colour match the subject's photo to this one: solved as that photo's settings (colourmatch.py)."""
        from .. import engine as E
        from ..engine.pipeline import composite as cp
        if not (0 <= self._lsel < len(self._layers)) or self.image is None:
            return
        L = self._layers[self._lsel]
        img, path = self.image, self.path
        here = self.edit_from_controls()
        here.composite = None
        self.viewer.busy = True
        self.viewer.update()
        self.status.emit("Matching the subject's colours to this photo (a few seconds)...")

        def work():
            src = cp._source(L["src"])
            se = cp.source_edit(dict(L, match=None))
            return E.colour_match_solve(src, se, img, here)

        def done(p_, res):
            self.viewer.busy = False
            self.viewer.update()
            if p_ != self.path or L not in self._layers:
                return
            p, mult, rep_ = res
            L["match"] = {"p": list(map(float, p)), "mult": list(map(float, mult)), "amount": 1.0}
            self._composite_controls()
            self.controls_changed()
            self.status.emit(f"Subject developed under this photo's light: {rep_['text']}.")
        run_bg(work, path, done, lambda _, err: (setattr(self.viewer, "busy", False), self.viewer.update(),
                                                 self.status.emit(f"Subject colour match problem: {err}")))

    def _colour_match(self):
        """Match colours to another photo: solved as slider settings (local/colourmatch.py), in the background."""
        from .. import engine as E
        from ..engine.io.rawfile import RAW_EXTENSIONS
        from ..engine.io.scanfile import SCAN_EXTENSIONS
        if self.image is None:
            return
        pat = " ".join(f"*{x}" for x in RAW_EXTENSIONS + SCAN_EXTENSIONS)
        start = os.path.dirname(self.path) if self.path else ""
        f, _ = QFileDialog.getOpenFileName(self, "Match colours to this photo", start, f"Photos ({pat})")
        if not f:
            return
        img, e0, path = self.image, self.edit_from_controls(), self.path
        name = os.path.basename(f)
        self.viewer.busy = True
        self.viewer.update()
        self.status.emit(f"Matching the colours of {name}: trying settings on small copies (a few seconds)...")

        def work():
            ref = E.open(f)
            try:
                ref_edit = E.load_edit(f)
            except Exception:
                slog.log_exception("colour match: reference sidecar unreadable, using defaults", log)
                ref_edit = E.Edit()
            return E.colour_match_solve(img, e0, ref, ref_edit)

        def done(p_, res):
            self.viewer.busy = False
            self.viewer.update()
            if p_ != self.path:
                return
            p, mult, rep_ = res
            amt = self.cm_amt.value() / 100.0
            self._cm = {"p": list(map(float, p)), "mult": list(map(float, mult)), "applied": amt, "name": name,
                        "text": rep_["text"]}
            new = E.colour_match_apply(self.edit_from_controls(), p, mult, amt, image=img)
            self.cm_amt.setVisible(True)
            self.cm_chip.setText(f"Colour match: {name} ({rep_['text']})")
            self.cm_chip.setStyleSheet(f"color:{T.VIOLET}; border-color:{T.alpha(T.VIOLET, 0.4)};")
            self.commit(new, f"Colours matched to {name}: {rep_['text']}. Every change is in the normal sliders; "
                             "Match amount scales it.", label="Colour match")
        run_bg(work, path, done, lambda _, err: (setattr(self.viewer, "busy", False), self.viewer.update(),
                                                 self.status.emit(f"Colour match problem: {err}")))

    def _colour_match_amount(self):
        """Match amount moved: apply the difference to the same sliders (the rest of the edit stays as it is)."""
        from .. import engine as E
        cm = self._cm
        if not cm or self.image is None:
            return
        amt = self.cm_amt.value() / 100.0
        d = amt - float(cm["applied"])
        if abs(d) < 1e-6:
            return
        cm["applied"] = amt
        from ..engine.pipeline.pipeline import wb_multipliers
        cur = self.edit_from_controls()
        new = E.colour_match_apply(cur, cm["p"], wb_multipliers(self.image, cur), d, image=self.image)
        self.commit(new, f"Colour match at {amt * 100:.0f}%.", label="Colour match amount")

    def _depth_mask(self, op):
        """A depth mask: the photo's depth map (local/depth.py, worked out once per photo on this PC), then From and
        To below choose the distances."""
        from .. import engine as E
        from ..engine.local import depth as dp
        if self.image is None:
            return
        if not dp.available():
            self.status.emit("The depth model is missing from models\\depth. Run Convert Models.bat (or copy that "
                             "folder back from the download).")
            return
        img, e, path, idx = self.image, self.edit_from_controls(), self.path, self._msel
        self.viewer.busy = True
        self.viewer.update()
        self.status.emit("Working out how far away each part of the photo is (a few seconds, once per photo)...")

        def done(p, shape):
            self.viewer.busy = False
            self.viewer.update()
            if p != self.path or not (0 <= idx < len(self._masks)):
                return
            shape["op"] = op
            shape["lo"], shape["hi"], shape["soft"] = 0.6, 1.0, 0.1
            self._msel = idx
            self._add_shape(shape)
            self.mask_show.setChecked(True)
            how = ("measured by the focus stack" if shape.get("source") == "measured" else
                   "estimated from the picture")
            self.status.emit(f"Depth ({how}): the nearest part of the photo is selected (red). Move From and To to "
                             f"choose the distances (0 = farthest, 100 = nearest).")
        run_bg(lambda: E.depth_shape(img, e), path, done,
               lambda _, err: (setattr(self.viewer, "busy", False), self.status.emit(f"Depth mask: {err}")))

    def _channel_mask(self, op, source: str | None = None, replace: bool = False):
        """A Channel mask (Infrared 1.5c): one raw channel, or a difference or ratio of two, measured from the raw;
        From and To choose the range. ``replace``: a new source for the selected mask's channel shape."""
        from .. import engine as E
        if self.image is None:
            return
        if getattr(self.image.raw, "is_scan", False):
            self.status.emit("Channel masks read the raw camera channels: this photo is not a raw.")
            return
        src = source or self.ch_src.currentData() or "r/b"
        img, e, path, idx = self.image, self.edit_from_controls(), self.path, self._msel
        self.viewer.busy = True
        self.viewer.update()

        def done(p, shape):
            self.viewer.busy = False
            self.viewer.update()
            if p != self.path or not (0 <= idx < len(self._masks)):
                return
            self._msel = idx
            m = self._masks[idx]
            if replace and m["shapes"] and m["shapes"][-1].get("type") == "channel":
                old = m["shapes"][-1]
                for k in ("op", "lo", "hi", "soft"):
                    shape[k] = old.get(k, shape[k])
                m["shapes"][-1] = shape
                self.controls_changed()
                self._refresh_mask_controls()
                self._update_overlay()
                return
            shape["op"] = op
            self._add_shape(shape)
            self.mask_show.setChecked(True)
            self.status.emit(f"Channel ({E.CHANNEL_SOURCES.get(src, src)}): the upper half of its range is selected "
                             "(red). Move From and To to choose; measured from the raw.")

        def fail(_, err):
            self.viewer.busy = False
            self.viewer.update()
            self.status.emit(f"Channel mask: {err}")           # (the background job already wrote any report)
        run_bg(lambda: E.channel_shape(img, e, src), path, done, fail)

    def _channel_source_picked(self, *_):
        m = self._cur_mask()
        if m is not None and m["shapes"] and m["shapes"][-1].get("type") == "channel":
            self._channel_mask(m["shapes"][-1].get("op", "add"), self.ch_src.currentData(), replace=True)

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
        if kind and self.viewer.tool == "hint":
            self._set_hint_tool(False)
        if kind and self.viewer.tool == "spot":
            self._set_spot_tool(None)
        if kind and self.viewer.tool in ("eyes", "sculpt", "shape", "body"):
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
        if self.viewer.tool == "hint":
            shift = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
            self._hint_click(u, v, alt, shift)
            return
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
            model = self._sel_model()
            self.status.emit(self._sel_status(model, "object"))
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
                by = " (SAM 2.1)" if shape.get("model") else ""
                self.status.emit(f"Object selected{by}. Click again to add a part, Alt-click to take a part away, "
                                 "Esc when done.")
            run_bg(lambda: E.object_mask(img, e0, fg, bg, model=model), path, done,
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
        self.viewer.show_overlay = (bool(on) or self.chg_show.isChecked() or self.st_show.isChecked()
                                    or self.dp_show.isChecked())
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
                 "Brightness": "luminance", "Skin": "skin", "Person": "person", "Depth": "depth",
                 "Channel": "channel"}
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
        elif sh["type"] == "depth":
            sh["lo"], sh["hi"] = round(self.d_lo.value() / 100.0, 3), round(self.d_hi.value() / 100.0, 3)
            sh["soft"] = round(self.d_soft.value() / 100.0, 3)
        elif sh["type"] == "channel":
            sh["lo"], sh["hi"] = round(self.ch_lo.value() / 100.0, 3), round(self.ch_hi.value() / 100.0, 3)
            sh["soft"] = round(self.ch_soft.value() / 100.0, 3)
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
        if self.viewer.tool == "hint":
            self._render_hint_overlay()
            return
        if self.viewer.tool is None and getattr(self, "chg_show", None) is not None and self.chg_show.isChecked():
            self._render_change_overlay()
            return
        if self.viewer.tool is None and getattr(self, "dp_show", None) is not None and self.dp_show.isChecked():
            self._render_depth_overlay()
            return
        if self.viewer.tool is None and getattr(self, "st_show", None) is not None and self.st_show.isChecked():
            self._render_stain_overlay()
            return
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
        self._cm = None                                    # a colour match belongs to the photo it was made on
        if hasattr(self, "cm_amt"):
            self.cm_amt.setVisible(False)
            self.cm_amt.set_value(100, emit=False)
            self.cm_chip.setText("Colour match: none")
            self.cm_chip.setStyleSheet(f"color:{T.DIM};")
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
        run_bg(work, path, self._opened, self._open_failed)

    def _open_failed(self, path, err):
        """The photo could not be opened: say so where the photo would be (not only in the status bar, which
        left 'Opening ...' on screen for good), and let the same photo be tried again."""
        if path != self.path:
            return
        name = os.path.basename(path)
        why = str(err)
        if why.startswith(name + ": "):                     # the engine names the file too: once is enough
            why = why[len(name) + 2:]
        why = why[:1].upper() + why[1:]
        self.path = None
        self.viewer.busy = False
        self.viewer.message = (f"{name} could not be opened.\n\n{why}\n\nThe file itself is untouched. Details are "
                               f"in logs/studio.log (Help > Copy diagnostic report).")
        self.viewer.update()
        self.status.emit(f"Could not open {name}: {why}")

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
        self._mark_saved()                                  # nothing to write until something changes
        self._fill_versions()
        self._fill_snapshots()
        m = self.image.meta
        scan = bool(getattr(self.image.raw, "is_scan", False))
        if scan:
            r = self.image.raw
            bits = [f"Scan: {r.width} x {r.height}", f"{r.bits}-bit {'grey' if r.grey else 'RGB'}",
                    r.profile.name + (" (assumed)" if r.profile.source == "assumed" else ""),
                    "black and white" if r.mono else "", m.datetime[:10] if m.datetime else ""]
        else:
            exp = (f"1/{round(1 / m.exposure_s)} s" if m.exposure_s and m.exposure_s < 0.5 else
                   (f"{m.exposure_s:g} s" if m.exposure_s else ""))
            bits = [m.model or "Unknown camera", m.lens or "manual lens", m.iso_label, exp,
                    f"f/{m.fnumber:g}" if m.fnumber else "",
                    f"camera {m.camera_temperature_c:.0f} °C" if m.camera_temperature_c is not None else ""]
        self.info.setText("  ·  ".join(b for b in bits if b))
        raw_only = (self.band, self.dn_m, self.dn_info)        # row banding and the method picker need raw data
        for w in self.clean_group.findChildren(QWidget):
            if w is not self.clean_scan_note and w.parent() is self.clean_group.body:
                w.setEnabled(not (scan and w in raw_only))
        self._dn_auto_ui()
        self._dn_auto_refresh()
        self._dn_cal_refresh()
        self.clean_scan_note.setVisible(scan)
        if scan:
            self.noise_chip.setText("Noise: measured in this picture when Clean is on")
            self.noise_chip.setStyleSheet(f"color:{T.AMBER}; border-color:{T.alpha(T.AMBER, 0.4)};")
        self._old_photo_refresh()
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
        bkey = (json.dumps(edit.geometry.__dict__, sort_keys=True, default=str), round(sc, 4), self.judge_bw,
                json.dumps(edit.stage_versions, sort_keys=True))
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
                b.stage_versions = dict(edit.stage_versions)    # the same rendering, so the wipe shows only the edit
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
        self.bd_find = QPushButton("Find bodies")
        self.bd_find.setObjectName("primary")
        self.bd_find.setToolTip("Finds everyone's body on this PC (the same Google MediaPipe body model as Masks > "
                                "Person) and works out their parts: arms, legs, neck, shoulders, waist, hips, and "
                                "tummy, chest and seat when side-on. Then point and click.")
        self.bd_find.clicked.connect(self.find_bodies)
        self.bd_pick = QPushButton("Pick a part")
        self.bd_pick.setCheckable(True)
        self.bd_pick.setToolTip("Point at the photo: the part under the pointer lights up with its name; click to "
                                "select it. Cyan: its outline was measured; amber: an edge was estimated (an arm "
                                "against the side, loose clothes). Where nothing was found, drag along the limb.")
        self.bd_pick.toggled.connect(lambda on: self._set_portrait_tool("body" if on else None))
        self.bd_legs = QPushButton("Longer legs")
        self.bd_legs.setToolTip("Stretches the legs from the hips down (at most 6% at 100). Needs the hips and an "
                                "ankle in view. Faces and the top half never move.")
        self.bd_legs.clicked.connect(self.body_legs)
        for b in (self.bd_find, self.bd_pick, self.bd_legs):
            b.setStyleSheet("padding: 5px 8px;")
        g.add(hbox(self.bd_find, self.bd_pick, self.bd_legs))
        self.bd_mirror = QCheckBox("Both sides (an arm or leg: do the other one too)")
        self.bd_mirror.setChecked(True)
        g.add(self.bd_mirror)
        self.bd_amount = g.add(ValueSlider("Slim the selected part", -100, 100, 30, 1, "{:+.0f}",
                                           "Draws the part's outline toward its own centre line (+) or out (-), "
                                           "fading to nothing at the joints, so a limb narrows without moving. "
                                           "Faces and hands never move; capped at 12% of the width. For Longer "
                                           "legs: how much longer (0 to 100)."))
        self.bd_amount.changed.connect(self._body_amount_changed)
        self.bd_info = g.add(chip("No bodies yet: press Find bodies", T.DIM))
        g.add(label("By hand", "dim"))
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
        if not (0 <= self._shsel < n) or (self._portrait.get("sculpt") or [{}])[self._shsel].get("kind") not in \
                ("shape", "part", "legs"):
            self._shsel = -1
        self._bparts_key = None
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
        parts = sum(1 for x in sc if x.get("kind") in ("part", "legs"))
        strokes = len(sc) - shapes - parts
        self.sc_info.setText("No sculpting" if not sc else
                             ", ".join(t for t in (f"{strokes} stroke{'s' if strokes != 1 else ''}" if strokes else "",
                                                    f"{shapes} shape{'s' if shapes != 1 else ''}" if shapes else "",
                                                    f"{parts} body part{'s' if parts != 1 else ''}" if parts else "")
                                       if t))
        sel_any = sc[self._shsel] if 0 <= self._shsel < len(sc) else None
        sel = sel_any if sel_any is not None and sel_any.get("kind") == "shape" else None
        body = sel_any if sel_any is not None and sel_any.get("kind") in ("part", "legs") else None
        for w in (self.sh_lift, self.sh_size, self.sh_slim, self.sh_width):
            w.setEnabled(sel is not None)
        self.sh_del.setEnabled(sel_any is not None)
        self._refresh_body_ui(body)
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
        from ..engine.local import bodies as bd
        for k, x in enumerate(self._portrait.get("sculpt") or []):
            polys = []
            if x.get("kind") == "part":
                polys = [bd.outline(gq, sw, sh) for gq in x.get("geo") or []]
            elif x.get("kind") == "legs":
                h, f = np.asarray(x["h"], float) * [sw, sh], np.asarray(x["f"], float) * [sw, sh]
                d = (f - h) / (np.hypot(*(f - h)) + 1e-9)
                nv = np.array([-d[1], d[0]]) * float(x.get("w", 0.05)) * long
                polys = [np.array([h + nv, f + nv, f - nv, h - nv])]
            for P in polys:
                u, v = E.s_to_view(self.image, e, P[:, 0] / sw, P[:, 1] / sh)
                shapes.append({"outline": list(zip(np.atleast_1d(u), np.atleast_1d(v))), "sel": k == self._shsel,
                               "tool": x.get("kind"), "index": k})
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
        self._refresh_body_view(e)
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
        if which and self.viewer.tool == "hint":
            self._set_hint_tool(False)
        kind = which if isinstance(which, str) else (which[0] if which else None)
        sub = which[1] if isinstance(which, tuple) else None
        for b, on in [(self.pt_eyes, kind == "eyes"), (self.bd_pick, kind == "body")] + \
                [(b, kind == "sculpt" and sub == k)
                                                          for k, b in self.sc_btn.items()] + \
                [(b, kind == "shape" and sub == t) for t, b in self.sh_btn.items()]:
            b.blockSignals(True)
            b.setChecked(on)
            b.blockSignals(False)
        if kind:
            if self.viewer.tool == "spot":
                self._set_spot_tool(None)
            elif self.viewer.tool and self.viewer.tool not in ("eyes", "sculpt", "shape", "body"):
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
            if kind == "body":
                self.viewer.setCursor(Qt.CursorShape.PointingHandCursor)
                self._refresh_portrait_view()
                if not (self._portrait.get("bodies") or {}).get("people"):
                    self.status.emit("No bodies found yet: press Find bodies first, or drag along a limb to "
                                     "define it by hand.")
        elif self.viewer.tool in ("eyes", "sculpt", "shape", "body"):
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
        if x.get("kind") != "shape":                         # found body parts stay on the body
            self._shsel = sh["index"]
            self._refresh_portrait_ui()
            self._refresh_portrait_view()
            return
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
        if x.get("kind") != "shape":
            return
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

    # body parts (local/bodies.py, BUILD_PLAN_PORTRAIT H.3)
    def find_bodies(self):
        from .. import engine as E
        from ..engine.local import people as pp
        if self.image is None:
            return
        if not pp.available():
            self.status.emit("The people models are missing from models\\people. Run setup again (or copy that "
                             "folder back from the download).")
            return
        img, e, path = self.image, self.edit_from_controls(), self.path
        self.viewer.busy = True
        self.viewer.update()
        self.status.emit("Finding bodies...")

        def done(p, store):
            if p != self.path:
                return
            self.viewer.busy = False
            if store is None:
                self.status.emit("No body found in this photo. You can still drag along a limb with Pick a part.")
                self.viewer.update()
                return
            self._portrait["bodies"] = store
            self._bparts_key = None
            self._portrait_changed("Bodies found")
            parts = self._body_parts()
            n = len(store["people"])
            if not self.bd_pick.isChecked():
                self.bd_pick.setChecked(True)
            else:
                self._refresh_portrait_view()
            self.status.emit(f"{n} {'person' if n == 1 else 'people'}, {len(parts)} parts found. Point at a part to "
                             f"see its name, click to select it, then use 'Slim the selected part'.")

        def fail(_, err):
            self.viewer.busy = False
            self.viewer.update()
            self.status.emit(f"Find bodies problem: {err}")
        run_bg(lambda: E.find_bodies(img, e), path, done, fail)

    def _body_parts(self) -> list:
        """The found parts of the stored bodies (worked out once per photo and kept)."""
        from ..engine.local import bodies as bd
        store = self._portrait.get("bodies") or {}
        if self.image is None or not store.get("people"):
            return []
        _, _, sw, sh = self.image.raw.crop
        key = (hash(store.get("data")), json.dumps(store["people"]), sw, sh)
        if getattr(self, "_bparts_key", None) != key:
            try:
                self._bparts = bd.find_parts(store["people"], bd.decode_mask(store), sw, sh)
            except Exception as exc:                          # noqa: BLE001
                report_problem("measuring the body parts", exc)
                self._bparts = []
            self._bparts_key = key
        return self._bparts

    def _refresh_body_view(self, e=None):
        from .. import engine as E
        from ..engine.local import bodies as bd
        parts = self._body_parts() if self.viewer.tool == "body" else []
        if not parts or self.image is None:
            self.viewer.body_parts, self.viewer.body_hover = [], -1
            return
        e = e or self.shown_edit()
        _, _, sw, sh = self.image.raw.crop
        out = []
        for q in parts:
            P = bd.outline(q, sw, sh)
            u, v = E.s_to_view(self.image, e, P[:, 0] / sw, P[:, 1] / sh)
            out.append({"outline": list(zip(np.atleast_1d(u), np.atleast_1d(v))), "name": q["name"],
                        "how": bd.how(q), "note": q.get("note", "")})
        self.viewer.body_parts = out

    def _refresh_body_ui(self, sel):
        store = self._portrait.get("bodies") or {}
        has = self.image is not None
        self.bd_find.setEnabled(has)
        self.bd_pick.setEnabled(has)
        self.bd_legs.setEnabled(bool(store.get("people")))
        self.bd_amount.setEnabled(sel is not None)
        if sel is not None:
            v = float(sel.get("slim", 0)) if sel.get("kind") == "part" else float(sel.get("amount", 0))
            self.bd_amount.set_value(v, emit=False)
            how = sel.get("how", "measured")
            col = T.CYAN if how == "measured" and sel.get("kind") == "part" else T.AMBER
            txt = f"Selected: {sel.get('name', 'part')}"
            if sel.get("kind") == "part":
                txt += " (outline measured)" if how == "measured" else " (an edge estimated: check it)"
            self.bd_info.setText(txt)
            self.bd_info.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")
        elif store.get("people"):
            parts = self._body_parts()
            meas = sum(1 for q in parts if q.get("how_l") == "measured" and q.get("how_r") == "measured")
            n = len(store["people"])
            self.bd_info.setText(f"{n} {'person' if n == 1 else 'people'}: {len(parts)} parts ({meas} measured, "
                                 f"cyan; {len(parts) - meas} estimated, amber)")
            self.bd_info.setStyleSheet(f"color:{T.CYAN}; border-color:{T.alpha(T.CYAN, 0.4)};")
        else:
            self.bd_info.setText("No bodies yet: press Find bodies")
            self.bd_info.setStyleSheet("")

    def _add_sculpt(self, items, label_, select=0):
        sc = list(self._portrait.get("sculpt") or [])
        first = len(sc)
        sc.extend(items)
        self._portrait["sculpt"] = sc
        self._shsel = first + select
        self.sc_show.setChecked(True)
        self._portrait_changed(label_)
        self._refresh_portrait_view()

    def _body_clicked(self, k):
        from ..engine.local import bodies as bd
        parts = self._body_parts()
        if self.image is None:
            return
        if not (0 <= k < len(parts)):
            self.status.emit("No part found there. Drag along the limb (one end to the other) to define it by hand.")
            return
        q = parts[k]
        _, _, sw, sh = self.image.raw.crop
        people = (self._portrait.get("bodies") or {}).get("people") or []
        sc = self._portrait.get("sculpt") or []
        if q["part"] in bd.LIFT_PARTS:
            items = bd.lift_shapes(q, people, sw, sh)
            self._add_sculpt(items, f"Sculpt: {q['name'].lower()} lift")
            self.status.emit(f"{q['name']}: placed {len(items)} lift shape{'s' if len(items) != 1 else ''} (amber: "
                             f"placed from the body points, not measured). Use Lift and Size; drag to adjust.")
            return
        mirror = self.bd_mirror.isChecked()
        item = bd.item_for(q, parts, people, sw, sh, mirror=mirror, slim=30.0)
        for i, x in enumerate(sc):                            # already sculpted: select it instead of adding
            if x.get("kind") == "part" and (x.get("id") == item["id"] or x.get("id") == q["id"]):
                self._shsel = i
                self._refresh_portrait_ui()
                self._refresh_portrait_view()
                self.status.emit(f"{x.get('name')} selected.")
                return
        self._add_sculpt([item], f"Sculpt: {item['name'].lower()}")
        msg = f"{item['name']}: slimmed 30 to start. Move 'Slim the selected part' to taste."
        if bd.how(q) != "measured":
            msg += " One edge was estimated (amber), so check the result closely."
        if q.get("note"):
            msg += " " + q["note"]
        self.status.emit(msg)

    def _body_drawn(self, u0, v0, u1, v1):
        from ..engine.local import bodies as bd
        if self.image is None:
            return
        _, _, sw, sh = self.image.raw.crop
        a, b = self._s_points([(u0, v0), (u1, v1)])
        store = self._portrait.get("bodies") or {}
        mask = bd.decode_mask(store) if store.get("data") else None
        q = bd.manual_part(a, b, mask, sw, sh)
        item = bd.item_for(q, [], store.get("people") or [], sw, sh, slim=30.0)
        item["id"] = f"drawn.{len(self._portrait.get('sculpt') or [])}"
        self._add_sculpt([item], "Sculpt: drawn part")
        self.status.emit("Part defined along your drag" + (" (its outline measured from the person, cyan)."
                                                             if bd.how(q) == "measured" else
                                                             " (its width estimated, amber: Find bodies first to "
                                                             "measure it).") + " Use 'Slim the selected part'.")

    def body_legs(self):
        from ..engine.local import bodies as bd
        if self.image is None:
            return
        people = (self._portrait.get("bodies") or {}).get("people") or []
        sc = self._portrait.get("sculpt") or []
        sel = sc[self._shsel] if 0 <= self._shsel < len(sc) else {}
        person = 0
        for gq in (sel.get("geo") or [])[:1]:                 # the person of the selected part, when there is one
            for i, q in enumerate(self._body_parts()):
                if q["a"] == gq["a"] and q["b"] == gq["b"]:
                    person = q["person"]
        _, _, sw, sh = self.image.raw.crop
        item = bd.legs_item(people, person, sw, sh, amount=30.0)
        if item is None:
            self.status.emit("Longer legs needs the hips and at least one ankle in view.")
            return
        self._add_sculpt([item], "Sculpt: longer legs")
        self.status.emit("Longer legs at 30: move the slider for more or less (100 = 6% longer).")

    def _body_amount_changed(self, v):
        sc = self._portrait.get("sculpt") or []
        if not (0 <= self._shsel < len(sc)):
            return
        x = sc[self._shsel]
        if x.get("kind") == "part":
            x["slim"] = round(v, 1)
        elif x.get("kind") == "legs":
            x["amount"] = round(max(v, 0.0), 1)
            if v < 0:
                self.bd_amount.set_value(0, emit=False)
        else:
            return
        self.controls_changed()
        self._refresh_portrait_view()

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
        if kind and self.viewer.tool == "hint":
            self._set_hint_tool(False)
        for k, b in self.rt_btn.items():
            b.blockSignals(True)
            b.setChecked(k == kind)
            b.blockSignals(False)
        if kind:
            if self.viewer.tool in ("eyes", "sculpt", "shape", "body"):
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
                self.rt_feather.set_value({"heal": 25, "clone": 50, "pupil": 25, "fill": 15}.get(kind, 30), emit=False)
            if not self.viewer.zoomed:
                self.status.emit("Retouch: zoom to 100% (Z or the mouse wheel) to see small dust spots.")
        elif self.viewer.tool == "spot":
            self.viewer.tool = None
            self.viewer.setCursor(Qt.CursorShape.ArrowCursor)
        self._show_fill_row()
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
            if sp.get("kind") not in ("redeye", "pupil", "fill"):
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
        if self.rt_kind == "fill":
            spot["method"] = self.rt_fill_method.currentData() or "auto"

        path = self.path

        def add(_, off):
            if self.path != path:
                return
            spot["src"] = off
            self._spots.append(spot)
            self._ssel = len(self._spots) - 1
            self.viewer.busy = False
            self._spots_changed({"heal": "Heal spot", "clone": "Clone spot", "redeye": "Red eye",
                                 "pupil": "Pupil", "fill": "Fill"}[spot["kind"]])
        if spot["kind"] in ("redeye", "pupil", "fill"):
            add(None, [0.0, 0.0])
            return
        img, e, others = self.image, self.edit_from_controls(), [dict(x) for x in self._spots]
        self.viewer.busy = True
        self.viewer.update()
        run_bg(lambda: E.auto_source(img, e, spot, others), self.path, add,
               lambda _, err: (setattr(self.viewer, "busy", False), self.status.emit(f"Retouch problem: {err}")))

    def _spot_picked(self, k):
        self._ssel = k
        if 0 <= k < len(self._spots) and self._spots[k].get("kind") == "fill":
            i = self.rt_fill_method.findData(self._spots[k].get("method", "auto"))
            self.rt_fill_method.blockSignals(True)
            self.rt_fill_method.setCurrentIndex(max(0, i))
            self.rt_fill_method.blockSignals(False)
            self._fill_method_changed(apply=False)
        self._show_fill_row()
        self._refresh_spots()

    def _show_fill_row(self):
        sel = self._spots[self._ssel] if 0 <= self._ssel < len(self._spots) else None
        self.rt_fill_row.setVisible(self.rt_kind == "fill" and self.viewer.tool == "spot"
                                    or bool(sel and sel.get("kind") == "fill"))

    def _fill_method_changed(self, *_, apply=True):
        from ..engine.local import fill as fl
        key = self.rt_fill_method.currentData() or "auto"
        m = fl.METHODS.get(key, fl.METHODS["auto"])
        have = fl.lama_available()
        txt = f"Best for: {m['best']}"
        if key in ("auto", "lama") and not have:
            txt += " LaMa is not installed yet: PatchMatch is used. Convert Models.bat adds LaMa."
        self.rt_fill_info.setText(txt)
        self.rt_fill_info.setToolTip(f"Best for: {m['best']}\nWeak at: {m['weak']}")
        self.rt_fill_info.setWordWrap(True)
        if not apply or not (0 <= self._ssel < len(self._spots)):
            return
        sp = self._spots[self._ssel]
        if sp.get("kind") == "fill" and sp.get("method", "auto") != key:
            sp["method"] = key
            self._spots_changed("Fill method")

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
        if sp.get("kind") not in ("redeye", "pupil", "fill"):
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

    # ------------------------------------------------------------------ flare ghosts (local/flare.py)
    def _flare_value(self):
        fl = self._flare
        if not fl or not fl.get("ghosts"):
            return None
        out = dict(fl)
        out["on"] = self.fl_on.isChecked()
        out["amount"] = round(self.fl_amt.value(), 3)
        out["ghosts"] = [dict(g) for g in fl["ghosts"]]
        return out

    def _flare_controls(self):
        """Show self._flare in the Lens group's flare controls (no signals)."""
        fl = self._flare or {}
        ghosts = fl.get("ghosts") or []
        for w in (self.fl_on, self.fl_amt, self.fl_list):
            w.blockSignals(True)
        self.fl_on.setChecked(bool(fl.get("on")) and bool(ghosts))
        self.fl_amt.set_value(float(fl.get("amount", 1.0)), emit=False)
        self.fl_list.clear()
        for n, g in enumerate(ghosts, 1):
            it = QListWidgetItem(f"Ghost {n}  ·  {100 * g.get('r', 0):.1f}% wide  ·  "
                                 f"strength {g.get('strength', 0):.2f}")
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if g.get("on", True) else Qt.CheckState.Unchecked)
            self.fl_list.addItem(it)
        for w in (self.fl_on, self.fl_amt, self.fl_list):
            w.blockSignals(False)
        self.fl_list.setVisible(bool(ghosts))
        for w in (self.fl_on, self.fl_amt, self.fl_rebuild, self.fl_show):
            w.setEnabled(bool(ghosts))
        if not fl:
            self.fl_chip.setText("Flare: not looked for yet")
            self.fl_chip.setStyleSheet(f"color:{T.DIM};")
        else:
            txt = fl.get("text") or (f"{len(ghosts)} flare ghost{'s' if len(ghosts) != 1 else ''} found"
                                     if ghosts else "No flare ghosts found.")
            rep_ = getattr(self.image, "flare_report", "") if self.image is not None else ""
            self.fl_chip.setText(txt + (f" {rep_}" if rep_ else ""))
            col = T.AMBER if ghosts else T.DIM
            self.fl_chip.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")
        self._refresh_ghost_view()

    def _refresh_ghost_view(self):
        from .. import engine as E
        fl = self._flare or {}
        ghosts = fl.get("ghosts") or []
        if self.image is None or not ghosts:
            self.viewer.ghosts = []
            self.viewer.update()
            return
        e = self.shown_edit()
        bs = self._brush_scale()
        sel = self.fl_list.currentRow()
        xs = np.array([g["x"] for g in ghosts], float)
        ys = np.array([g["y"] for g in ghosts], float)
        u, v = E.s_to_view(self.image, e, xs, ys)
        self.viewer.ghosts = [{"uv": (float(a), float(b)), "r": float(g.get("r", 0.01)) / bs,
                               "on": g.get("on", True), "sel": k == sel, "n": k + 1}
                              for k, (g, a, b) in enumerate(zip(ghosts, np.atleast_1d(u), np.atleast_1d(v)))]
        self.viewer.update()

    def _show_ghosts(self, on):
        self.viewer.show_ghosts = bool(on)
        self._refresh_ghost_view()

    def _ghost_ticked(self, item):
        if not self._flare or not self._flare.get("ghosts"):
            return
        k = self.fl_list.row(item)
        if 0 <= k < len(self._flare["ghosts"]):
            self._flare["ghosts"][k]["on"] = item.checkState() == Qt.CheckState.Checked
            self._refresh_ghost_view()
            self.controls_changed()

    def find_flare(self):
        from .. import engine as E
        if self.image is None:
            return
        img, e, path = self.image, self.edit_from_controls(), self.path
        self.viewer.busy = True
        self.viewer.update()
        self.status.emit("Looking for flare ghosts...")
        run_bg(lambda: E.find_flare(img, e), path,
               lambda p, res: self._flare_found(res) if p == self.path else None,
               lambda _, err: (setattr(self.viewer, "busy", False),
                               self.status.emit(f"Find flare ghosts problem: {err}")))

    def _flare_found(self, res):
        self.viewer.busy = False
        ghosts = list(res.get("ghosts") or [])
        self._flare = {"on": bool(ghosts), "amount": 1.0, "ghosts": ghosts, "axis": res.get("axis"),
                       "source": res.get("source"), "text": res.get("text", "")}
        self._flare_controls()
        if ghosts:
            self.fl_show.setChecked(True)
            self.commit(self.edit_from_controls(), label=f"Flare: {len(ghosts)} ghosts")
            self.status.emit(f"{len(ghosts)} flare ghost{'s' if len(ghosts) != 1 else ''} found (amber circles) "
                             f"and their glow subtracted. Strong or blown-out ones: press Rebuild ghosts with Fill.")
        else:
            self.status.emit(res.get("text") or "No flare ghosts found.")

    def rebuild_ghosts(self):
        """Each ticked ghost becomes a Retouch Fill spot; its glow is no longer subtracted (the fill replaces it)."""
        fl = self._flare or {}
        todo = [g for g in fl.get("ghosts") or [] if g.get("on", True)]
        if self.image is None or not todo:
            self.status.emit("Tick the ghosts to rebuild first.")
            return
        method = self.rt_fill_method.currentData() or "auto"
        for g in todo:
            self._spots.append({"kind": "fill", "pts": [[float(g["x"]), float(g["y"])]],
                                "size": round(1.3 * float(g.get("r", 0.01)), 5), "feather": 0.15, "opacity": 100.0,
                                "src": [0.0, 0.0], "enabled": True, "auto": "flare", "v": 2, "method": method})
            g["on"] = False
        self._ssel = -1
        self._flare_controls()
        self.rt_show.setChecked(True)
        self._spots_changed(f"Flare: {len(todo)} ghosts rebuilt")
        self.status.emit(f"{len(todo)} ghost{'s are' if len(todo) != 1 else ' is'} now Fill spots in Retouch (each "
                         f"takes a few seconds to build). Alt-click one there to undo it.")

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
        self._old_photo_refresh()
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
        elif self.pick == "film":
            self._pick_film_base_at(u, v)

    def _escape(self, keep_zoom: bool = False):
        """Esc: put down the eyedropper or tool in hand; with nothing in hand, back to the whole photo."""
        busy_tool = bool(self.pick) or bool(self.viewer.tool)
        if self.pick:
            self._set_pick(None)
        if self.viewer.tool == "hint":
            self._set_hint_tool(False)
        elif self.viewer.tool in ("eyes", "sculpt", "shape", "body"):
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
        before = self.edit_from_controls()
        e, character = make_it_good_settings(self.image.meta, before)
        if e.to_dict() != before.to_dict():
            self.commit(e, label="Make it good")
        noise = (f"noise reduction set for {self.image.meta.iso_label}" if e.denoise.enabled and not
                 before.denoise.enabled else f"{self.image.meta.iso_label}: no noise reduction needed"
                 if not e.denoise.enabled else "noise reduction already on")
        sharp = ("character lens: sharpening left off" if character else
                 "gentle capture sharpening added" if e.sharpen.enabled and not before.sharpen.enabled else
                 "sharpening already on")
        self.status.emit(f"Make it good: {noise}; {sharp}; checking the lens now (the Lens panel will say what it "
                         f"found). It never changes brightness or colour, so on a clean, sharp photo the change is "
                         f"small; that means the photo needed little.")
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

    def _save_state(self) -> str:
        """What save_sidecar would write, as one value to compare: the photo, the version, its edit and the history
        labels up to here."""
        return json.dumps([self.path, self.version, self.edit.to_dict(),
                           [lb for lb, _ in self.steps[:self.hpos + 1]]], sort_keys=True, default=str)

    def _mark_saved(self):
        """The edit in hand is what the sidecar holds (just loaded, or just written)."""
        self._saved_state = self._save_state()

    def save_sidecar(self):
        """Write the edit and its history to the photo's sidecar, if anything changed since it was loaded or last
        saved. A photo that was only looked at gets no sidecar (and no 'could not save' on a read-only card)."""
        if self.image is None or self.path is None:
            return
        from .. import engine as E
        state = self._save_state()
        if state == self._saved_state:
            return
        try:
            E.save_version(self.path, self.version, self.edit)
            E.save_history(self.path, self.version, self.steps[:self.hpos + 1])
            self._saved_state = state
        except (OSError, IndexError, KeyError) as err:
            log.warning("sidecar for %s not saved: %s", self.path, err)
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


class LensesPage(InfoPage):
    """Lens tools: what Fix lens does, the lensfun status, and the lens list for tagging contactless glass."""

    def __init__(self, settings):
        super().__init__("Lenses", "Automatic fixes that work even on contactless vintage glass, and the lenses you "
                         "tag photos with. Lens rendering signatures are planned.", "Lens fixes ready", [
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
                             ("Render like my lens", "Planned: Laack, Heliar, Apo-Lanthar, Otus... an open "
                                                     "signature library. Not built yet.")], T.VIOLET)
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
                             ("Focus Stack", "Merges photos focused at different distances into one sharp photo "
                                             "(right-click photos in the Library: Send to Focus Stack). Shoot "
                                             "manual exposure, overlap the focus steps, and keep the subject still; "
                                             "hand-held works with Line up locally on."),
                             ("Card to Stacks", "Point it at a card or a folder (or right-click photos in the "
                                                "Library: Find focus stacks; the import summary offers it too). It "
                                                "finds every focus stack, leaves out the frames that would hurt "
                                                "each one with the reason (soft everywhere, adds nothing, shaken, "
                                                "subject moved), and stacks them all. Bursts, including AI Servo "
                                                "action, are listed and left alone. Nothing is deleted; tick a "
                                                "frame to put it back."),
                             ("Super-Res Stack", "Merges a burst into one cleaner photo, or a larger one (1.5x, 2x) "
                                                 "when the frames are shifted a little between shots. Hand-held "
                                                 "bursts, the moon and night sky each have their own mode; the "
                                                 "coverage chart says whether more resolution was really there."),
                             ("Dual Pixel Lab", "Opens Canon Dual Pixel RAW files (Dual Pixel RAW on in the camera; "
                                                "checked on the R5, the 5D Mark IV not yet): the two half-views, a depth map, 3D "
                                                "(wigglegram, anaglyph, side by side) and up to a stop of "
                                                "highlights the main image clipped. A normal raw says so."),
                             ("Infrared", "For infrared and full-spectrum raws (Library: right-click, Edit in "
                                          "Infrared tab). It guesses the filter, sets white balance far past the "
                                          "camera's range, and offers looks and black and white from the channels; "
                                          "Open in Develop carries on with every tool."),
                             ("Panorama", "Give it overlapping frames (Library: right-click, Send to Panorama): it "
                                          "finds where each one points, evens out exposure and vignetting, keeps "
                                          "the frame without flare where two overlap, and blends the seams. Turn "
                                          "the camera about the lens, not your body, and overlap by a third."),
                             ("Flare and Fill", "Develop > Lens > Find flare ghosts takes away the round "
                                                "reflections a bright light leaves; Retouch > Fill rebuilds "
                                                "anything you brush over from what is around it (LaMa when "
                                                "Convert Models.bat has added it)."),
                             ("Results go back to Develop", "Every stack is written beside its photos as "
                                                            "FIRST_to_LAST_..._Studio.tif, a linear file that opens "
                                                            "in Develop with every tool working, like a raw."),
                             ("Develop", "B before/after, Ctrl+Z / Ctrl+Y undo and redo, double-click a slider "
                                         "to reset it, Esc puts a tool down, F11 full screen. On a narrow window "
                                         "Copy, Paste, Before / After and the virtual copies are under the ⋯ "
                                         "button."),
                             ("Colours mean something", "Cyan = measured on your camera. Violet = shared profile. "
                                                        "Amber = estimated from the photo."),
                             ("Make it good", "Noise clean-up for the photo's ISO, Fix lens, and gentle "
                                              "capture sharpening (not for character lenses). It never changes "
                                              "brightness."),
                             ("Clean (noise)", "Develop > Clean removes noise from the raw data, before colour is "
                                               "made. Auto lets each photo choose its own strength, detail first: "
                                               "it works once the camera is calibrated on the Cameras page "
                                               "(lens-cap frames, then flat pairs), and a Dual Pixel photo is "
                                               "judged by its own two halves. The chips say what happened: what "
                                               "Auto chose and how much fine detail that keeps, what the camera's "
                                               "calibration took out (hot pixels, fixed banding), what the detail "
                                               "guard gave back. Show what Clean removed is the proof: if what was "
                                               "taken out looks like television static, only noise was removed. "
                                               "The full guide is docs\\CLEAN.md in the Studio's folder."),
                             ("Teaching Clean your camera", "Cameras page, in this order: drop lens-cap frames "
                                                            "(any ISO, any number), drop flat pairs, press Deep "
                                                            "calibration on the camera's card. That alone turns on "
                                                            "Auto and the calibration in Clean. Optional: choose "
                                                            "about 200 varied clean photos and train the camera's "
                                                            "own network overnight (method 'learned'). The card "
                                                            "lists what Clean takes from the calibration and what "
                                                            "it leaves alone, with the reason."),
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
                             ("Body sculpting", "Find bodies, then Pick a part: point at an arm, leg, waist, hips, "
                                                "neck or shoulders to see its name, click it, set Slim. Both sides "
                                                "does the other arm or leg too; Longer legs stretches from the hips "
                                                "down. By hand: Push, Shrink, Grow brushes and the shape tools. Faces "
                                                "and hands never move; every warp is capped."),
                             ("Old photos & scans", "The Library shows TIFF, JPEG and PNG scans beside raws. "
                                                    "Develop > Old photo and film: choose Colour or Black and white "
                                                    "negative to turn a negative positive (Pick the film base on "
                                                    "clear film if the colour is off); Remove stains and corrosion "
                                                    "(unmixed, not painted over; Show what's lost marks where the "
                                                    "picture may be gone); Remove dust and scratches; "
                                                    "Colourize a black and white photo (only colour is added; the "
                                                    "brightness stays). Colour a part: click a coat or the sky and "
                                                    "choose its colour; Shift-click adds more, Alt-click stops it, "
                                                    "the white outline shows where it goes."),
                             ("Cognitive Adjustments", "Develop > Cognitive Adjustments: our own tools, built from "
                                                       "research and run on this PC; each measures what is in the "
                                                       "photo and invents nothing. Remove printing dots: for scans "
                                                       "of newspaper, yearbook or book pictures (the screen is found "
                                                       "exactly; nothing changes if there is none). Repair JPEG "
                                                       "blocks and ringing (JPEG files): the smoothest picture that "
                                                       "still matches the file exactly. Add a subject from another "
                                                       "photo: click it out (click-to-select), place, size and turn "
                                                       "it, match its colours and grain. Relight: moves the light "
                                                       "(Light from, height, softness) using the shapes in the "
                                                       "depth map; only the light changes. Depth haze: "
                                                       "adds or clears haze by distance, using a depth map worked "
                                                       "out from the photo (Show depth map: red near, blue far; "
                                                       "Masks > Depth selects by distance). Match colours to "
                                                       "another photo: moves exposure, contrast, white balance, "
                                                       "saturation and the colour wheels to that photo's colour "
                                                       "mood (all ordinary sliders; Match amount scales it). "
                                                       "Show what the "
                                                       "automatic tools changed: a heat map, clear where nothing was "
                                                       "touched."),
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
                             ("Problems?", "When something goes wrong the Studio saves an error report by "
                                           "itself, one dated file per problem, in the error_reports folder: "
                                           "send Bill the newest file with a line about what you were doing. For "
                                           "anything else, copy the diagnostic report below. Neither holds photos, "
                                           "serial numbers or locations.")])
        b = QPushButton("Copy diagnostic report")
        b.clicked.connect(self.copy_diag)
        ob = QPushButton("Open error reports folder")
        ob.setToolTip("Where the Studio saves a dated report file by itself whenever something goes wrong")
        ob.clicked.connect(self.open_reports)
        row = QHBoxLayout()
        row.addWidget(b)
        row.addWidget(ob)
        row.addStretch(1)
        self.body.addSpacing(10)
        self.body.addLayout(row)
        self.report_lbl = label("", "dim", wrap=True)
        self.body.addWidget(self.report_lbl)
        self.set_last_report(None)
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
        # The analysis cache (BUILD_PLAN_DP3D A.4): stacks analysed once, kept beside the Studio
        self.body.addSpacing(16)
        self.body.addWidget(label("Analysis cache", "h2"))
        self.body.addWidget(label("A focus stack's analysis (the slow part) is kept, so making surfaces, leaves or a "
                                  "different merge of the same stack later starts at once. It lives in the "
                                  "analysis_cache folder beside the Studio, never on C:, and the oldest analyses "
                                  "go first when it reaches its size.", "dim", wrap=True))
        self.cache_cap = QComboBox()
        self.CACHE_CAPS = (10, 30, 60, 120, 250)
        for g_ in self.CACHE_CAPS:
            self.cache_cap.addItem(f"Keep up to {g_} GB", g_)
        try:
            from ..engine.stack import anacache as AC
            cur = AC.cap_gb()
        except Exception:                              # noqa: BLE001  (a display only)
            cur = 60
        self.cache_cap.setCurrentIndex(min(range(len(self.CACHE_CAPS)), key=lambda i: abs(self.CACHE_CAPS[i] - cur)))
        self.cache_cap.activated.connect(self._cache_cap)
        cb = QPushButton("Clear the analysis cache")
        cb.clicked.connect(self._cache_clear)
        row = QHBoxLayout()
        row.addWidget(self.cache_cap)
        row.addWidget(cb)
        row.addStretch(1)
        self.body.addLayout(row)
        self.cache_info = label("", "dim", wrap=True)
        self.body.addWidget(self.cache_info)
        # Blender (BUILD_PLAN_DP3D F.3): where it is, for "Open in Blender"
        self.body.addSpacing(16)
        self.body.addWidget(label("Blender", "h2"))
        self.blender_info = label("", "dim", wrap=True)
        self.body.addWidget(self.blender_info)
        bb = QPushButton("Choose blender.exe...")
        bb.clicked.connect(self._choose_blender)
        row = QHBoxLayout()
        row.addWidget(bb)
        row.addStretch(1)
        self.body.addLayout(row)
        self._blender_state()

    def _blender_state(self):
        try:
            from ..engine.io import blender as BL
            exe, how = BL.program()
        except Exception as e:                         # noqa: BLE001  (a display only)
            exe, how = "", str(e)
        self.blender_info.setText(f"Open in Blender uses {exe} ({how})." if exe else how + ".")

    def _choose_blender(self):
        f, _ = QFileDialog.getOpenFileName(self, "Where is blender.exe?", "", "Blender (blender.exe blender)")
        if f:
            Settings().set("blender_program", f)
            self._blender_state()

    def _cache_cap(self, i):
        Settings().set("analysis_cache_gb", int(self.CACHE_CAPS[i]))
        self.cache_info.setText(f"The analysis cache keeps up to {self.CACHE_CAPS[i]} GB.")

    def _cache_clear(self):
        try:
            from ..engine.stack import anacache as AC
            n = AC.clear()
            self.cache_info.setText(f"Cleared: {n} analysis(es) removed.")
        except OSError as e:
            self.cache_info.setText(f"Could not clear it all: {e}")

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

    def open_reports(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(slog.error_reports_dir())))

    def set_last_report(self, path=None):
        """The newest error report: the one just written, else the newest already in the folder."""
        path = path or slog.latest_error_report()
        self.report_lbl.setText(f"Newest error report: {path}" if path else
                                "Error reports: none so far. They are saved in the error_reports folder beside "
                                "the Studio.")


# ============================================================================================ main window
NAV = [("▦", "Library"), ("◐", "Develop"), ("◫", "Dual Pixel Lab"), ("≋", "Focus Stack"), ("☷", "Card to Stacks"),
       ("⬡", "3D Scan"), ("✦", "Super-Res Stack"),
       ("◑", "Infrared"), ("⌒", "Panorama"), ("✎", "Paint"), ("◎", "Lenses"), ("▣", "Cameras"), ("⇪", "Export"), ("?", "Help")]
ROW = {name: i for i, (_, name) in enumerate(NAV)}


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
        self.dualpixel = DualPixelPage(self.settings)
        self.focus = FocusStackPage(self.settings)
        self.cardstacks = CardStacksPage(self.settings)
        self.scan = ScanPage(self.settings)
        self.superres = SuperResPage(self.settings)
        self.infrared = InfraredPage(self.settings)
        self.infrared.sidecar_written.connect(self._ir_sidecar)
        self.panorama = PanoramaPage(self.settings)
        self.paint = PaintPage(self.settings)
        self.paint.status.connect(lambda t: self.statusBar().showMessage(t, 6000))
        for pg in (self.dualpixel, self.focus, self.cardstacks, self.scan, self.superres, self.infrared,
                   self.panorama):
            pg.open_result.connect(self.open_result)
            pg.result_written.connect(self._stack_written)
            pg.status.connect(lambda t: self.statusBar().showMessage(t, 6000))
        self.help.update_mode.connect(self._set_update_mode)
        self.help.update_check.connect(lambda: self.check_updates(manual=True))
        BUS.report_saved.connect(self._report_saved)
        pages = [
            self.library, self.develop,
            self.dualpixel, self.focus, self.cardstacks, self.scan, self.superres, self.infrared, self.panorama,
            self.paint,
            self.lenses,
            self.cameras,
            InfoPage("Export", "Select photos in the Library and press Export, or use Export in Develop. 16-bit "
                     "TIFF in Adobe RGB by default; PNG and JPEG too.", None, [
                         ("Every export has its own name", "The photo's name with _Studio added: IMG_0001.CR2 "
                          "becomes IMG_0001_Studio.tif, so an export never has the name of your original or of "
                          "the camera's JPEG. Exporting again replaces the Studio's own earlier export; a file "
                          "the Studio did not write is never replaced."),
                         ("Send to Photoshop", "Planned: a TIFF hand-off and a Photoshop panel. Not built yet; "
                          "for now export a TIFF and open it in Photoshop."),
                         ("Send to PhotoCraft", "Planned. Not built yet."),
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
        self.library.send_to.connect(self.send_to_tab)
        self.library.lens_tagged.connect(self._lens_tagged)
        self.library.paste_requested.connect(self.paste_to_files)
        self.library.export_requested.connect(self.export)
        self.library.save_open_edit = self.develop.save_sidecar
        self.develop.status.connect(lambda t: self.statusBar().showMessage(t, 6000))
        self.develop.export_requested.connect(self.export)
        self.develop.web_requested.connect(self.export_web)
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

    def _report_saved(self, path: str, lead: str = "Something went wrong. "):
        """An error report file was written (from any thread): say where it is; Help shows it too."""
        self.statusBar().showMessage(f"{lead}An error report was saved: {path}  (Help > Open error reports folder)",
                                     30000)
        self.help.set_last_report(path)

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
        if row == ROW["Infrared"] and getattr(self, "develop", None) is not None and self.develop.path:
            self.develop.save_sidecar()               # the Infrared tab reads the photo's edit from its sidecar
        if row == 0 and getattr(self, "develop", None) is not None and self.develop.path:
            self.develop.save_sidecar()
            self.library.refresh_paths([self.develop.path])

    def send_to_tab(self, where: str, files: list):
        """From the Library: Send to Focus Stack / Super-Res Stack, Open in Dual Pixel Lab, Edit in Infrared tab."""
        page, row = {"focus": (self.focus, ROW["Focus Stack"]), "superres": (self.superres, ROW["Super-Res Stack"]),
                     "cardstacks": (self.cardstacks, ROW["Card to Stacks"]),
                     "dualpixel": (self.dualpixel, ROW["Dual Pixel Lab"]),
                     "infrared": (self.infrared, ROW["Infrared"]),
                     "panorama": (self.panorama, ROW["Panorama"])}[where]
        if self.develop.path and self.develop.path in files:
            self.develop.save_sidecar()
        self.nav.setCurrentRow(row)
        if where == "cardstacks":
            page.set_sources(files)
        else:
            page.set_frames(files)

    def _ir_sidecar(self, path: str):
        """The Infrared tab wrote a photo's edit: Develop (if it has that photo open) and the Library catch up."""
        from .. import engine as E
        if self.develop.path and os.path.normcase(self.develop.path) == os.path.normcase(path):
            try:
                self.develop.commit(E.load_edit(path), label="Infrared tab")
            except (OSError, ValueError):
                slog.log_exception("reloading the edit the Infrared tab wrote", log)
        try:
            self.library.refresh_paths([path])
        except RuntimeError:
            pass

    def open_result(self, path: str):
        """A stacking tool's result: show it in the Library (its folder) and open it in Develop."""
        folder = os.path.dirname(path)
        if os.path.normcase(os.path.abspath(folder)) == os.path.normcase(os.path.abspath(self.library.folder or "")):
            self.library.reload(select=path)
        else:
            self.library.load_folder(folder, select=path)
        self.develop.set_strip(self.library.files, self.library.icons())
        self.nav.setCurrentRow(1)
        self.develop.open(path)

    def _stack_written(self, path: str):
        folder = os.path.dirname(path)
        if self.library.folder and os.path.normcase(os.path.abspath(folder)) == \
                os.path.normcase(os.path.abspath(self.library.folder)):
            self.library.reload(select=path)
        self.statusBar().showMessage(f"Saved {os.path.basename(path)}", 6000)

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
        spec, dest = dlg.spec(), dlg.dest.text().strip()
        try:
            if not dest:
                raise OSError("no folder was chosen")
            os.makedirs(dest, exist_ok=True)
        except OSError as err:
            log.warning("export: folder %r unusable: %s", dest, err)
            QMessageBox.warning(self, "Export", f"The export folder cannot be used:\n{dest or '(empty)'}\n\n{err}\n\n"
                                "Pick another folder and export again.")
            return

        def work():
            from .. import engine as E
            out, failed, taken = [], [], set()
            for f, v in items:
                img = None
                try:                                    # one bad photo does not stop the rest
                    img = E.open(f)
                    if v:
                        out.append(E.export(img, E.load_version(f, v), spec,
                                            out_path=version_export_path(f, v, spec, dest), taken=taken))
                    else:
                        out.append(E.export(img, E.load_edit(f), spec, out_dir=dest, taken=taken))
                except Exception as err:               # noqa: BLE001
                    slog.log_exception(f"export failed: {f}", log)
                    report_problem(f"Export could not export {os.path.basename(f)}", err)
                    failed.append((os.path.basename(f), str(err)))
                finally:
                    if img is not None:                 # one photo in memory at a time
                        img.drop_cache()
                    img = None
            return out, failed
        self.statusBar().showMessage(f"Exporting {len(items)} photo(s) in the background...")
        run_bg(work, dest, self._exported, lambda _, e: self.statusBar().showMessage(f"Export problem: {e}"),
               keep=True)

    def export_web(self, files):
        """Save for web: no dialog; the photo as a small sRGB JPEG in the export folder (io/export.web_jpeg_bytes)."""
        from .. import engine as E
        items = [(f, 0) if isinstance(f, str) else (f[0], int(f[1])) for f in files if f]
        if not items:
            return
        self.develop.save_sidecar()
        dest = self.settings.get("export_dir", str(paths.get("OUTPUTS") / "export"))
        try:
            os.makedirs(dest, exist_ok=True)
        except OSError as err:
            QMessageBox.warning(self, "Save for web", f"The export folder cannot be used:\n{dest}\n\n{err}\n\n"
                                "Choose another folder once in Export..., then try again.")
            return

        def work():
            out, failed, taken = [], [], set()
            for f, v in items:
                img = None
                try:
                    img = E.open(f)
                    spec = E.ExportSpec(web=True)
                    if v:
                        p = version_export_path(f, v, spec, dest)
                        out.append(E.export(img, E.load_version(f, v), spec, out_path=p, taken=taken))
                    else:
                        out.append(E.export(img, E.load_edit(f), spec, out_dir=dest, taken=taken))
                except Exception as err:               # noqa: BLE001
                    slog.log_exception(f"save for web failed: {f}", log)
                    report_problem(f"Save for web could not save {os.path.basename(f)}", err)
                    failed.append((os.path.basename(f), str(err)))
                finally:
                    if img is not None:
                        img.drop_cache()
            return out, failed

        def done(_, res):
            out, failed = res
            if out:
                kb = sum(os.path.getsize(p) for p in out if os.path.exists(p)) / 1024
                self.statusBar().showMessage(f"Saved for web: {os.path.basename(out[0])} ({kb:.0f} KB, sRGB, 2048 px) "
                                             f"in {dest}", 12000)
            if failed:
                self.statusBar().showMessage(f"Save for web problem: {failed[0][1]}", 12000)
        self.statusBar().showMessage("Saving for web...")
        run_bg(work, dest, done, lambda _, e: self.statusBar().showMessage(f"Save for web problem: {e}"), keep=True)

    def _exported(self, dest, res):
        out, failed = res
        msg = f"Exported {len(out)} file(s) to {dest}"
        if failed:
            msg += (f". {len(failed)} could not be exported: " + ", ".join(n for n, _ in failed[:4]) +
                    ("..." if len(failed) > 4 else "") + " (details in logs/studio.log)")
        self.statusBar().showMessage(msg, 15000 if failed else 10000)
        if out:
            QDesktopServices.openUrl(QUrl.fromLocalFile(dest))
        elif failed:
            QMessageBox.warning(self, "Export", f"Nothing could be exported.\n\n{failed[0][0]}: {failed[0][1]}\n\n"
                                "Details are in logs/studio.log (Help > Copy diagnostic report).")

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
    trace = "".join(traceback.format_exception(t, v, tb))
    log.error("unhandled exception in the app\n%s", trace)
    report = slog.save_error_report("Something went wrong in the Studio's window.", trace)
    import threading
    if QApplication.instance() is None or threading.current_thread() is not threading.main_thread():
        return                                     # a message box only from the window's own thread
    try:
        show_error_box(f"Something went wrong: {v}", report)
    except Exception:
        pass


def show_error_box(text: str, report=None, parent=None) -> None:
    """The box for a fault in the Studio: what happened, and where the error report was saved by itself (with a
    button that opens its folder), so a tester only has to send the file."""
    if report:
        text += (f"\n\nAn error report was saved:\n{report}\n\nPlease send that file to Bill with a line about what "
                 f"you were doing. It holds no photos, serial numbers or locations.")
    else:
        text += f"\n\nDetails are in {os.path.join('logs', 'studio.log')} (Help > Copy diagnostic report)."
    box = QMessageBox(QMessageBox.Icon.Critical, "Photographic Studio", text, parent=parent)
    open_b = box.addButton("Open the folder", QMessageBox.ButtonRole.ActionRole) if report else None
    box.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
    box.exec()
    if open_b is not None and box.clickedButton() is open_b:
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(str(report))))


class _WheelGuard(QObject):
    """The mouse wheel scrolls the panel; it moves a slider, list or number box only after that control was clicked
    (has focus). Without this, scrolling down the controls nudged every slider passing under the pointer, so the
    photo kept changing while someone was only reading the options (a tester, 2026-11-23)."""

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QAbstractScrollArea, QAbstractSlider, QAbstractSpinBox
        if ev.type() == QEvent.Type.Wheel and isinstance(obj, (QAbstractSlider, QComboBox, QAbstractSpinBox)) \
                and not obj.hasFocus():
            w = obj.parentWidget()
            while w is not None and not isinstance(w, QAbstractScrollArea):
                w = w.parentWidget()
            if w is not None and w.verticalScrollBar() is not obj:
                sb = w.verticalScrollBar()
                steps = ev.angleDelta().y() / 120.0
                sb.setValue(sb.value() - int(round(steps * QApplication.wheelScrollLines() * sb.singleStep())))
            return True
        return False


def run(argv=None) -> int:
    slog.setup(console_level=None)
    sys.excepthook = _excepthook
    crashed = slog.watch_for_crashes()             # a hard crash last time becomes an error report now
    argv = list(argv or sys.argv)
    # Paint needs every pen event as it happens (BUILD_PLAN_PAINT P0.2): no merging of tablet events.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_CompressTabletEvents, False)
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("Photographic Studio By Bill")
    app.setStyleSheet(T.QSS)
    app._wheel_guard = _WheelGuard(app)
    app.installEventFilter(app._wheel_guard)
    lock = _single_instance(app)
    if lock is None:
        return 0
    w = MainWindow()
    try:
        from ..engine.color import dcp as _dcp
        _dcp.warm_lightroom_index()                # Lightroom's profiles indexed in the background
    except Exception:                              # noqa: BLE001
        log.debug("Lightroom profile index not started", exc_info=True)
    w.show_as_last_time()
    if "--paint" in argv:                          # Start Paint.bat / start-paint.sh open straight into Paint
        w.nav.setCurrentRow(ROW["Paint"])

    def _screen_note():
        try:
            from ..engine.color import display as _disp
            d = _disp.current()
            msg = (f"Screen: {d.name} - the preview is colour-managed (ProPhoto straight to your monitor's profile)."
                   if d.managed else "Screen: this monitor has no colour profile, so Windows treats it as sRGB. "
                   "Calibrate it (a colorimeter) so what you see is what you print.")
            w.statusBar().showMessage(msg, 12000)
        except Exception:                          # noqa: BLE001  (a note must never stop the Studio)
            log.debug("screen note failed", exc_info=True)
    QTimer.singleShot(2500, _screen_note)
    log.info("app started")
    if crashed:
        QTimer.singleShot(1200, lambda: w._report_saved(str(crashed), "The Studio stopped unexpectedly last time. "))
    code = app.exec()
    clean = finish_jobs()
    log.info("app closed (exit %s%s)", code, "" if clean else "; previews still rendering were dropped")
    lock.unlock()
    # A preview render still running when Python tears down crashed the exit (seen on Linux: "The Studio stopped
    # with an error" after closing it mid-render). Everything worth keeping is saved by now (sidecars, settings,
    # ratings; export and import jobs were waited for), so leave without tearing down.
    logging.shutdown()
    _flush_streams()
    os._exit(int(code))


def _flush_streams() -> None:
    """Flush the console streams before leaving. The installed Studio (the MSI's Photographic Studio.exe) runs with
    no console, so Python gives it None for stdout and stderr: flushing those crashed every close (0.1.14)."""
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            stream.flush()
        except (OSError, ValueError, AttributeError):
            pass


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
