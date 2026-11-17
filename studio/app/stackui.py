"""Parts shared by the Focus Stack, Super-Res Stack and Dual Pixel Lab pages (BUILD_PLAN_TABS 2.1).

Every page has the same skeleton, so a person who learns one knows the others:

    Title  [chips]                                      [Add frames...] [Clear] [Folder...] [?]
    +----------------------------------------------------+---------------------------+
    |  ResultView (fit / 100%, drag; B shows the best     |  panel: method with Best  |
    |  single frame in the same place; Z 100%)            |  for / Weak at, options,  |
    |  or, with nothing loaded, three lines on what the   |  honest limits, Preview,  |
    |  tab does, how to give it frames, what to shoot      |  Build full size, Open in |
    +----------------------------------------------------+  Develop                  |
    |  FrameTray: thumbnails in order, ticks, chips        |                           |
    +----------------------------------------------------+---------------------------+
    |  progress: text . bar . Cancel                                                     |

Results are colour-coded by how they were made (rule 0.3.11): cyan measured and used, amber used with a warning,
red left out. Long work runs in the background with a Cancel (threading.Event), like Process picks.
"""
from __future__ import annotations

import os
import threading

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QListView, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

from .. import log as slog
from . import theme as T
from .common import Settings, card, chip, hbox, label, load_thumb, report_problem, run_bg, to_qimage

log = slog.get("app.stack")

PANEL_W = 360
STATUS_COLOURS = {"ok": T.CYAN, "warn": T.AMBER, "dropped": T.RED, "": T.DIM}


def _widgets():
    """ValueSlider and Group live in main.py (imported late: main imports these pages)."""
    from . import main as M
    return M.ValueSlider, M.Group


# ============================================================================================ background with progress
class _Progress(QObject):
    moved = Signal(float, str)


class JobBar(QFrame):
    """Text, bar, Cancel. ``start(fn(progress, cancel), on_done, on_fail, keep)``: fn runs in the background;
    ``progress(fraction, text)`` may be called from it; Cancel sets the Event the work checks."""

    def __init__(self):
        super().__init__()
        self.setObjectName("bottombar")
        h = QHBoxLayout(self)
        h.setContentsMargins(14, 6, 14, 6)
        self.text = label("", "dim")
        self.text.setMinimumWidth(60)
        self.text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedWidth(240)
        self.cancel_b = QPushButton("Cancel")
        self.cancel_b.clicked.connect(self.cancel)
        h.addWidget(self.text, 1)
        h.addWidget(self.bar)
        h.addWidget(self.cancel_b)
        self._p = _Progress()
        self._p.moved.connect(self._show)
        self.event: threading.Event | None = None
        self.busy = False
        self.bar.hide()
        self.cancel_b.hide()

    def _show(self, f, text):
        self.bar.setValue(int(1000 * f))
        if text:
            self.text.setText(text)

    def say(self, text: str, colour: str = T.DIM):
        self.text.setText(text)
        self.text.setStyleSheet(f"color:{colour};")

    def start(self, fn, on_done, on_fail=None, keep=False, text="Working..."):
        if self.busy:
            return False
        self.busy = True
        self.event = threading.Event()
        ev = self.event
        self.say(text)
        self.bar.setValue(0)
        self.bar.show()
        self.cancel_b.show()
        self.cancel_b.setEnabled(True)

        def progress(f, msg=""):
            try:
                self._p.moved.emit(float(f), str(msg))
            except RuntimeError:                     # the page is gone
                pass

        def done(_, r):
            self._end()
            on_done(r)

        def fail(_, err):
            self._end()
            cancelled = ev.is_set()
            if on_fail:
                on_fail(err, cancelled)
        run_bg(lambda: fn(progress, ev), "stack job", done, fail, keep=keep)
        return True

    def cancel(self):
        if self.event is not None:
            self.event.set()
            self.cancel_b.setEnabled(False)
            self.say("Cancelling...")

    def _end(self):
        self.busy = False
        self.bar.hide()
        self.cancel_b.hide()


# ============================================================================================ frames
class FrameTray(QListWidget):
    """Thumbnails in order, a tick on each (untick to leave a frame out), a status line under each. Drag to
    reorder."""
    changed = Signal()
    selected = Signal(int)

    def __init__(self, thumb: int = 104):
        super().__init__()
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(False)
        self.setIconSize(QSize(thumb, int(thumb * 0.67)))
        self.setGridSize(QSize(thumb + 26, int(thumb * 0.67) + 52))
        self.setMovement(QListView.Movement.Snap)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(int(thumb * 0.67) + 74)
        self.setWordWrap(True)
        self.setSpacing(4)
        self.thumb = thumb
        self.itemChanged.connect(lambda *_: self.changed.emit())
        self.model().rowsMoved.connect(lambda *_: self.changed.emit())
        self.currentRowChanged.connect(self.selected.emit)
        self._gen = 0

    def set_frames(self, items: list):
        """items: [(path, ok, problem)]"""
        self._gen += 1
        gen = self._gen
        self.blockSignals(True)
        self.clear()
        for path, ok, problem in items:
            it = QListWidgetItem(os.path.basename(path))
            it.setData(Qt.ItemDataRole.UserRole, path)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if ok else Qt.CheckState.Unchecked)
            if not ok:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsUserCheckable)
                it.setText(f"{os.path.basename(path)}\nleft out")
                it.setForeground(QColor(T.RED))
                it.setToolTip(problem)
            self.addItem(it)
        self.blockSignals(False)
        for i, (path, ok, _) in enumerate(items):
            if ok:
                run_bg(lambda p=path: load_thumb(p, self.thumb * 2), (gen, i), self._thumb_done)
        self.changed.emit()

    def _thumb_done(self, token, arr):
        gen, i = token
        if gen != self._gen or i >= self.count():
            return
        try:
            self.item(i).setIcon(QIcon(QPixmap.fromImage(to_qimage(arr))))
        except RuntimeError:
            pass

    def paths(self, checked_only: bool = True) -> list:
        out = []
        for i in range(self.count()):
            it = self.item(i)
            if checked_only and it.checkState() != Qt.CheckState.Checked:
                continue
            out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def set_status(self, path: str, text: str, status: str = "ok", tip: str = ""):
        for i in range(self.count()):
            it = self.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == path:
                it.setText(f"{os.path.basename(path)}\n{text}")
                it.setForeground(QColor(STATUS_COLOURS.get(status, T.DIM)))
                it.setToolTip(tip or text)
                return

    def reorder(self, paths: list):
        """Put the items in this order (others after)."""
        self.blockSignals(True)
        items = [self.takeItem(0) for _ in range(self.count())]
        by = {it.data(Qt.ItemDataRole.UserRole): it for it in items}
        for p in paths:
            if p in by:
                self.addItem(by.pop(p))
        for it in items:
            if it.data(Qt.ItemDataRole.UserRole) in by:
                self.addItem(it)
        self.blockSignals(False)

    def reverse(self):
        self.reorder(self.paths(False)[::-1])
        self.changed.emit()


# ============================================================================================ result view
class ResultView(QWidget):
    """The picture: PhotoView (fit / 100%, drag) with B to show the best single frame in the same place."""
    clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        from .review import PhotoView
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.view = PhotoView()
        v.addWidget(self.view)
        self.result: QImage | None = None
        self.reference: QImage | None = None
        self.showing_ref = False
        self.result_chip = ("Result", T.CYAN)
        self.ref_chip = ("Best single frame", T.AMBER)

    def set_images(self, result: QImage | None, reference: QImage | None = None, full_w: int = 0, full_h: int = 0,
                   chip_text: str = "Result", chip_colour: str = T.CYAN):
        self.result, self.reference = result, reference
        self.result_chip = (chip_text, chip_colour)
        self._fw, self._fh = full_w, full_h
        self.showing_ref = False
        self._apply()

    def _apply(self):
        img = self.reference if (self.showing_ref and self.reference is not None) else self.result
        c = self.ref_chip if (self.showing_ref and self.reference is not None) else self.result_chip
        if img is not None:
            self.view.set_image(img, self._fw or img.width(), self._fh or img.height())
        else:
            self.view.set_image(None)
        self.view.chip, self.view.chip_colour = c
        self.view.update()

    def toggle_reference(self, on: bool | None = None):
        self.showing_ref = (not self.showing_ref) if on is None else bool(on)
        self._apply()

    def zoom_100(self):
        self.view.set_zoom(not self.view.zoom)

    def fit(self):
        self.view.set_zoom(False)

    def note(self, text: str):
        self.view.note = text
        self.view.update()


# ============================================================================================ method picker
class MethodPicker(QWidget):
    """A combo of methods with what each does, is best for, is weak at, and its speed (plan 23.1)."""
    changed = Signal(str)

    def __init__(self, info: dict, keys=None, title: str = ""):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        if title:
            v.addWidget(label(title))
        self.combo = QComboBox()
        self.combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.text = label("", "dim", wrap=True)
        v.addWidget(self.combo)
        v.addWidget(self.text)
        self.info = info
        self.set_keys(keys or list(info))
        self.combo.currentIndexChanged.connect(self._changed)

    def set_keys(self, keys, current: str | None = None):
        self.combo.blockSignals(True)
        self.combo.clear()
        for k in keys:
            self.combo.addItem(self.info[k]["name"], k)
        if current is not None and current in keys:
            self.combo.setCurrentIndex(list(keys).index(current))
        self.combo.blockSignals(False)
        self._show()

    def key(self) -> str:
        return self.combo.currentData() or ""

    def set_key(self, k: str):
        i = self.combo.findData(k)
        if i >= 0:
            self.combo.setCurrentIndex(i)
        self._show()

    def _show(self):
        d = self.info.get(self.key(), {})
        parts = [d.get("does", "")]
        if d.get("best"):
            parts.append(f"<b>Best for:</b> {d['best']}")
        if d.get("weak"):
            parts.append(f"<b>Weak at:</b> {d['weak']}")
        if d.get("speed"):
            parts.append(f"<b>Speed:</b> {d['speed']}")
        self.text.setText("<br>".join(p for p in parts if p))

    def _changed(self, *_):
        self._show()
        self.changed.emit(self.key())


def honest_card(text: str, title: str = "Honest limits", colour: str = T.AMBER) -> QFrame:
    c = card()
    c.setStyleSheet(f"QFrame#card {{ border: 1px solid {T.alpha(colour, 0.45)}; }}")
    v = QVBoxLayout(c)
    v.setContentsMargins(12, 10, 12, 10)
    t = label(title, "h2")
    t.setStyleSheet(f"color:{colour}; font-size:10.5pt;")
    v.addWidget(t)
    v.addWidget(label(text, "dim", wrap=True))
    return c


# ============================================================================================ the page skeleton
class StackPage(QWidget):
    """The shared skeleton (see the module text). Subclasses fill the panel (``build_panel``) and react to
    ``frames_changed``."""
    open_result = Signal(str)                        # a written result: open it in Develop
    status = Signal(str)
    result_written = Signal(str)                     # the Library refreshes when it shows that folder
    MIN_FRAMES = 2
    SETTINGS_KEY = "stack"

    def __init__(self, title: str, what: str, how: str, shoot: str, settings: Settings | None = None,
                 add_text: str = "Add frames..."):
        super().__init__()
        self.settings = settings or Settings()
        self.setAcceptDrops(True)
        self.items: list = []                        # [(path, ok, problem)]
        self.last_result = ""
        self.out_dir = ""
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        top = QFrame()
        top.setObjectName("topbar")
        th = QHBoxLayout(top)
        th.setContentsMargins(16, 10, 16, 10)
        self.title = label(title, "h2")
        th.addWidget(self.title)
        self.chips = QHBoxLayout()
        self.chips.setSpacing(6)
        th.addLayout(self.chips)
        th.addStretch(1)
        self.add_b = QPushButton(add_text)
        self.add_b.clicked.connect(self.choose_files)
        self.clear_b = QPushButton("Clear")
        self.clear_b.clicked.connect(self.clear)
        self.help_b = QPushButton("?")
        self.help_b.setFixedWidth(34)
        self.help_b.setToolTip(f"{what}\n\n{how}\n\n{shoot}")
        self.help_b.clicked.connect(lambda: QMessageBox.information(self, title, f"{what}\n\n{how}\n\n{shoot}"))
        for b in (self.add_b, self.clear_b, self.help_b):
            th.addWidget(b)
        lv.addWidget(top)
        self.centre = QStackedWidget()
        self.empty = self._empty_state(title, what, how, shoot)
        self.result = ResultView()
        self.centre.addWidget(self.empty)
        self.centre.addWidget(self.result)
        lv.addWidget(self.centre, 1)
        self.tray = FrameTray()
        self.tray.changed.connect(self._tray_changed)
        lv.addWidget(self.tray)
        self.jobbar = JobBar()
        lv.addWidget(self.jobbar)
        outer.addWidget(left, 1)
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(PANEL_W)
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(14, 12, 14, 12)
        self.panel_scroll = QScrollArea()
        self.panel_scroll.setWidgetResizable(True)
        self.panel_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        self.panel = QVBoxLayout(body)
        self.panel.setContentsMargins(0, 0, 6, 0)
        self.panel.setSpacing(8)
        self.panel_scroll.setWidget(body)
        pv.addWidget(self.panel_scroll, 1)
        self.actions = QVBoxLayout()
        self.actions.setSpacing(6)
        pv.addLayout(self.actions)
        outer.addWidget(panel)
        self.build_panel()
        QShortcut(QKeySequence("B"), self, activated=lambda: self.result.toggle_reference())
        QShortcut(QKeySequence("Z"), self, activated=self.result.zoom_100)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.result.fit)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.build_full)
        self._refresh()

    # ------------------------------------------------------------------------------------ building blocks
    def _empty_state(self, title, what, how, shoot) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addStretch(1)
        c = card()
        c.setMaximumWidth(620)
        cv = QVBoxLayout(c)
        cv.setContentsMargins(26, 22, 26, 22)
        cv.setSpacing(10)
        h = label(title, "h1")
        cv.addWidget(h)
        for t, col in ((what, T.TEXT), (how, T.DIM), (shoot, T.DIM)):
            lb = label(t, None, wrap=True)
            lb.setStyleSheet(f"color:{col};")
            cv.addWidget(lb)
        self.empty_extra = QVBoxLayout()
        cv.addLayout(self.empty_extra)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(c, 4)
        row.addStretch(1)
        v.addLayout(row)
        v.addStretch(2)
        return w

    def group(self, title: str, subtitle: str = ""):
        _, Group = _widgets()
        g = Group(title, subtitle)
        self.panel.addWidget(g)
        return g

    def slider(self, name, lo, hi, default, step=0.05, fmt="{:.2f}", tip=""):
        ValueSlider, _ = _widgets()
        return ValueSlider(name, lo, hi, default, step, fmt, tip)

    def tick(self, text: str, on: bool, tip: str = "") -> QCheckBox:
        c = QCheckBox(text)
        c.setChecked(on)
        if tip:
            c.setToolTip(tip)
        return c

    def build_panel(self):
        """Subclasses fill self.panel (groups) and self.actions (the buttons at the bottom)."""

    def add_actions(self, preview_text="Preview", build_text="Build full size"):
        self.preview_b = QPushButton(preview_text)
        self.preview_b.clicked.connect(self.preview)
        self.build_b = QPushButton(build_text)
        self.build_b.setObjectName("primary")
        self.build_b.setToolTip("Ctrl+Enter")
        self.build_b.clicked.connect(self.build_full)
        self.open_b = QPushButton("Open in Develop")
        self.open_b.clicked.connect(lambda: self.last_result and self.open_result.emit(self.last_result))
        self.folder_b = QPushButton("Folder...")
        self.folder_b.setToolTip("Where results are saved (beside the first frame unless you pick a folder)")
        self.folder_b.clicked.connect(self.choose_folder)
        self.actions.addWidget(hbox(self.preview_b, (self.build_b, 1), stretch=False))
        self.actions.addWidget(hbox((self.open_b, 1), self.folder_b, stretch=False))
        self.where = label("", "dim", wrap=True)
        self.actions.addWidget(self.where)

    # ------------------------------------------------------------------------------------ frames in
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        files = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if files:
            self.add_files(files)
            e.acceptProposedAction()

    def choose_files(self):
        from ..engine.io.rawfile import RAW_EXTENSIONS
        start = self.settings.get("folder") or ""
        pats = " ".join(f"*{x}" for x in RAW_EXTENSIONS) + " *.tif *.tiff *.jpg *.jpeg *.png"
        files, _ = QFileDialog.getOpenFileNames(self, "Add frames", start, f"Photos ({pats});;All files (*)")
        if files:
            self.add_files(files)

    def expand(self, files: list) -> list:
        from ..engine.stack.frames import supported
        out = []
        for f in files:
            if os.path.isdir(f):
                out += sorted(os.path.join(f, n) for n in os.listdir(f) if supported(os.path.join(f, n)))
            else:
                out.append(f)
        return out

    def add_files(self, files: list, replace: bool = False):
        from ..engine.stack.frames import probe
        files = self.expand(files)
        have = [p for p, _, _ in self.items] if not replace else []
        new = [f for f in files if f not in have]
        infos = probe(have + new)
        self.items = [(f.path, f.ok, f.problem) for f in infos]
        self.tray.set_frames(self.items)
        if self.items and not self.out_dir:
            self._default_folder()
        self.frames_changed()
        self._refresh()

    def set_frames(self, files: list):
        """From the Library: replace whatever was here."""
        self.items = []
        self.out_dir = ""
        self.add_files(files, replace=True)

    def clear(self):
        if self.jobbar.busy:
            return
        self.items = []
        self.tray.set_frames([])
        self.out_dir = ""
        self.result.set_images(None)
        self.frames_changed()
        self._refresh()

    def frame_paths(self) -> list:
        return self.tray.paths(True)

    def _tray_changed(self):
        self.frames_changed()
        self._refresh()

    def frames_changed(self):
        """Subclasses: drop an analysis that no longer matches the frames."""

    # ------------------------------------------------------------------------------------ output folder
    def _default_folder(self):
        from ..engine.stack.frames import default_out_dir
        saved = self.settings.get(f"{self.SETTINGS_KEY}_folder")
        if saved and os.path.isdir(saved):
            self.out_dir = saved
            note = ""
        else:
            ok = [p for p, o, _ in self.items if o]
            if not ok:
                return
            self.out_dir, note = default_out_dir(ok[0])
        self.where.setText(f"Saves to: {self.out_dir}" + (f"\n{note}" if note else ""))

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Where results go", self.out_dir or "")
        if d:
            self.out_dir = d
            self.settings.set(f"{self.SETTINGS_KEY}_folder", d)
            self.where.setText(f"Saves to: {self.out_dir}")

    def target(self, name: str, source: str) -> str:
        from ..engine.io.export import free_target
        if not self.out_dir:
            self._default_folder()
        os.makedirs(self.out_dir, exist_ok=True)
        return free_target(os.path.join(self.out_dir, name), source)

    # ------------------------------------------------------------------------------------ state
    def _refresh(self):
        has = bool(self.items)
        self.centre.setCurrentIndex(1 if has else 0)
        n = len(self.frame_paths())
        enough = n >= self.MIN_FRAMES
        for b in ("preview_b", "build_b"):
            if hasattr(self, b):
                getattr(self, b).setEnabled(enough and not self.jobbar.busy)
        if hasattr(self, "open_b"):
            self.open_b.setEnabled(bool(self.last_result) and os.path.exists(self.last_result))
        self.set_chips()

    def set_chips(self, extra: list | None = None):
        while self.chips.count():
            w = self.chips.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        n = len(self.frame_paths())
        if not self.items:
            return
        self.chips.addWidget(chip(f"{n} frame{'s' if n != 1 else ''}", T.CYAN if n >= self.MIN_FRAMES else T.AMBER))
        bad = sum(1 for _, ok, _ in self.items if not ok)
        if bad:
            self.chips.addWidget(chip(f"{bad} left out", T.RED))
        for text, colour in (extra or []):
            self.chips.addWidget(chip(text, colour))

    def failed(self, what: str, err: str, cancelled: bool):
        if cancelled:
            self.jobbar.say("Cancelled. Nothing was written.", T.AMBER)
        else:
            self.jobbar.say(f"{what}: {err}", T.RED)
        self._refresh()

    def preview(self):
        """Subclasses."""

    def build_full(self):
        """Subclasses."""


def qimage_rgb8(a8: np.ndarray) -> QImage:
    return to_qimage(np.ascontiguousarray(a8[..., :3]))


def scene_to_qimage(rgb_scene: np.ndarray, long_edge: int | None = None) -> QImage:
    """A scene-linear ProPhoto result shown with the default look (as Develop shows it)."""
    from ..engine.io.intermediate import preview_srgb
    return to_qimage(preview_srgb(rgb_scene, long_edge or max(rgb_scene.shape[:2])))


class CoverageDiagram(QWidget):
    """The frames' sub-pixel positions as dots in a unit square: spread out = detail to gain, one clump = none."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.dots: list = []
        self.moved = True

    def set_dots(self, dots, moved: bool):
        self.dots, self.moved = list(dots), moved
        self.update()

    def sizeHint(self):
        return QSize(200, 160)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = min(self.width(), self.height()) - 16
        r = QRectF((self.width() - s) / 2, 8, s, s)
        p.setPen(QPen(QColor(T.LINE), 1))
        p.setBrush(QColor(T.BG2))
        p.drawRoundedRect(r, 8, 8)
        p.setPen(QPen(QColor(T.LINE), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(r.center().x(), r.top()), QPointF(r.center().x(), r.bottom()))
        p.drawLine(QPointF(r.left(), r.center().y()), QPointF(r.right(), r.center().y()))
        col = QColor(T.CYAN if self.moved else T.AMBER)
        p.setPen(Qt.PenStyle.NoPen)
        for fy, fx in self.dots:
            c = QPointF(r.left() + fx * r.width(), r.top() + fy * r.height())
            glow = QColor(col)
            glow.setAlpha(60)
            p.setBrush(glow)
            p.drawEllipse(c, 9, 9)
            p.setBrush(col)
            p.drawEllipse(c, 4, 4)
        p.setPen(QColor(T.DIM))
        p.setFont(QFont("Segoe UI", 8))
        if not self.dots:
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Sub-pixel positions\nappear after Preview")
        p.end()
