"""The Card to Stacks page (BUILD_PLAN_PLATE_GLOW_BURST_MACRO.md, block F): a whole card or folder in, every focus
stack found, the frames that would hurt each one left out with a plain reason, and every stack merged.

    Title  [chips]                                         [Choose folder...] [Clear] [?]
    +-------------------------------------------------------+---------------------------+
    |  the queue: one row per stack / burst / single shot,  |  Find the stacks:         |
    |  its frames under it, each ticked (used) or not, with |  strictness (best / weak) |
    |  the reason in its colour (cyan ok, amber used with   |  pause that ends a stack  |
    |  care, red left out)                                  |  [Find stacks]            |
    +-------------------------------------------------------+  Stack them:              |
    |  the selected frame, or the stack's result             |  [Stack all] [This one]   |
    +-------------------------------------------------------+  honest limits            |
    |  progress: text . bar . Cancel                                                     |

Nothing is deleted or moved. A frame left out is only left out of its stack; ticking it puts it back. Results go
beside the frames as FIRST_to_LAST_Focus_Studio.tif (the same as the Focus Stack page writes), or into the folder
chosen with Folder....
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
                               QSplitter, QStackedWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import log as slog
from . import theme as T
from .common import Settings, card, chip, hbox, label
from .stackui import PANEL_W, JobBar, MethodPicker, _widgets, honest_card

log = slog.get("app.cardstacks")

STATUS_COLOUR = {"ok": T.CYAN, "warn": T.AMBER, "out": T.RED}
KIND_COLOUR = {"focus": T.TEXT, "burst": T.DIM, "single": T.DIM}
ROLE_GROUP = Qt.ItemDataRole.UserRole + 1
ROLE_FRAME = Qt.ItemDataRole.UserRole + 2


class CardStacksPage(QWidget):
    open_result = Signal(str)
    status = Signal(str)
    result_written = Signal(str)
    SETTINGS_KEY = "card_stacks"

    def __init__(self, settings: Settings | None = None):
        super().__init__()
        from ..engine.stack import sorter as SO
        self.SO = SO
        self.settings = settings or Settings()
        self.setAcceptDrops(True)
        self.sources: list = []              # folders / files chosen
        self.groups: list = []               # sorter.Group
        self.results: dict = {}              # id(group) -> path written
        self.out_dir = ""                    # "" = beside each stack's frames
        self.queue: list = []                # groups still to stack in this run
        self._filling = False
        self._peaks: dict = {}               # (id(group), frame index or -1) -> (rgb, in focus, missed)
        self._peak_token = 0
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
        th.addWidget(label("Card to Stacks", "h2"))
        self.chips = QHBoxLayout()
        self.chips.setSpacing(6)
        th.addLayout(self.chips)
        th.addStretch(1)
        self.add_b = QPushButton("Choose folder...")
        self.add_b.clicked.connect(self.choose_folder)
        self.clear_b = QPushButton("Clear")
        self.clear_b.clicked.connect(self.clear)
        help_b = QPushButton("?")
        help_b.setFixedWidth(34)
        help_b.setToolTip(f"{SO.WHAT}\n\n{SO.HOW}")
        help_b.clicked.connect(lambda: QMessageBox.information(self, "Card to Stacks", f"{SO.WHAT}\n\n{SO.HOW}"))
        for b in (self.add_b, self.clear_b, help_b):
            th.addWidget(b)
        lv.addWidget(top)
        self.centre = QStackedWidget()
        self.centre.addWidget(self._empty_state())
        split = QSplitter(Qt.Orientation.Vertical)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Stack / frame", "Why"])
        self.tree.setColumnWidth(0, 380)
        self.tree.setUniformRowHeights(True)
        self.tree.itemChanged.connect(self._ticked)
        self.tree.currentItemChanged.connect(self._selected)
        self.tree.itemDoubleClicked.connect(self._double)
        split.addWidget(self.tree)
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumHeight(180)
        self.view.setStyleSheet(f"background:{T.alpha('#000000', 0.25)};")
        self.view_note = label("", "dim")
        vw = QWidget()
        vv = QVBoxLayout(vw)
        vv.setContentsMargins(8, 4, 8, 4)
        vv.addWidget(self.view, 1)
        vv.addWidget(self.view_note)
        split.addWidget(vw)
        split.setSizes([520, 300])
        self.centre.addWidget(split)
        lv.addWidget(self.centre, 1)
        self.jobbar = JobBar()
        lv.addWidget(self.jobbar)
        outer.addWidget(left, 1)
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(PANEL_W)
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(14, 12, 14, 12)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        self.panel = QVBoxLayout(body)
        self.panel.setContentsMargins(0, 0, 6, 0)
        self.panel.setSpacing(8)
        sc.setWidget(body)
        pv.addWidget(sc, 1)
        self.actions = QVBoxLayout()
        self.actions.setSpacing(6)
        pv.addLayout(self.actions)
        outer.addWidget(panel)
        self._build_panel()
        self._refresh()

    # ------------------------------------------------------------------------------------ layout
    def _empty_state(self) -> QWidget:
        SO = self.SO
        w = QWidget()
        v = QVBoxLayout(w)
        v.addStretch(1)
        c = card()
        c.setMaximumWidth(620)
        cv = QVBoxLayout(c)
        cv.setContentsMargins(26, 22, 26, 22)
        cv.setSpacing(10)
        cv.addWidget(label("Card to Stacks", "h1"))
        for t, col in ((SO.WHAT, T.TEXT), (SO.HOW, T.DIM)):
            lb = label(t, None, wrap=True)
            lb.setStyleSheet(f"color:{col};")
            cv.addWidget(lb)
        b = QPushButton("Choose folder...")
        b.setObjectName("primary")
        b.clicked.connect(self.choose_folder)
        cv.addWidget(hbox(b))
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(c, 4)
        row.addStretch(1)
        v.addLayout(row)
        v.addStretch(2)
        return w

    def _build_panel(self):
        SO = self.SO
        ValueSlider, Group = _widgets()
        g = Group("Find the stacks", "Frames close in time, of one scene, with the sharp area moving through depth.")
        self.panel.addWidget(g)
        self.strict = g.add(MethodPicker(SO.STRICTNESS, title="How strict"))
        self.strict.set_key(self.settings.get(f"{self.SETTINGS_KEY}_strictness", "normal"))
        self.strict.changed.connect(self._rejudge)
        self.gap = g.add(ValueSlider("Pause that ends a stack", 0.5, 10.0,
                                     float(self.settings.get(f"{self.SETTINGS_KEY}_gap", SO.GAP_S)), 0.5, "{:.1f} s",
                                     "A longer pause between two frames starts a new stack. Longer if you stop to "
                                     "check the screen in the middle of a stack."))
        self.find_b = QPushButton("Find stacks")
        self.find_b.clicked.connect(self.find)
        g.add(self.find_b)
        g = Group("Stack them", "Each stack is lined up (locally too, for hand-held), checked again for movement, "
                                "and merged with Auto.")
        self.panel.addWidget(g)
        self.local = QCheckBox("Line up locally too (hand-held)")
        self.local.setChecked(True)
        self.local.setToolTip("Follows the small local shifts a hand-held sweep makes between frames. Harmless on a "
                              "tripod.")
        g.add(self.local)
        self.report = label("", "dim", wrap=True)
        g.add(self.report)
        g = Group("View", "Click a stack or a frame in the list.")
        self.panel.addWidget(g)
        self.peak = QCheckBox("Show what's in focus")
        self.peak.setChecked(bool(self.settings.get(f"{self.SETTINGS_KEY}_peak", True)))
        self.peak.setToolTip("A stack: cyan where some frame got it sharp, amber where there is detail but no frame "
                             "got it sharp (missed focus, or background blur you wanted).\nA frame: cyan where that "
                             "frame is in focus.\nMeasured on the camera's preview of each raw, lined up.")
        self.peak.toggled.connect(self._peak_toggled)
        g.add(self.peak)
        self.panel.addWidget(honest_card(SO.HONEST))
        self.panel.addStretch(1)
        self.all_b = QPushButton("Stack all")
        self.all_b.setObjectName("primary")
        self.all_b.clicked.connect(self.stack_all)
        self.one_b = QPushButton("Stack this one")
        self.one_b.clicked.connect(self.stack_selected)
        self.open_b = QPushButton("Open in Develop")
        self.open_b.clicked.connect(self._open_selected)
        self.folder_b = QPushButton("Folder...")
        self.folder_b.setToolTip("Where results are saved (beside each stack's frames unless you pick a folder)")
        self.folder_b.clicked.connect(self.choose_out)
        self.actions.addWidget(hbox(self.one_b, (self.all_b, 1), stretch=False))
        self.actions.addWidget(hbox((self.open_b, 1), self.folder_b, stretch=False))
        self.where = label("Saves beside each stack's frames.", "dim", wrap=True)
        self.actions.addWidget(self.where)

    # ------------------------------------------------------------------------------------ sources
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        items = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if items:
            self.set_sources(items)
            e.acceptProposedAction()

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "A card or a folder of photos", self.settings.get("folder") or "")
        if d:
            self.set_sources([d])

    def set_sources(self, items: list, find: bool = True):
        """Folders or files (from the Library, the import summary, a drop). Starts finding the stacks."""
        if self.jobbar.busy:
            return
        self.sources = list(items)
        self.groups, self.results = [], {}
        self._fill()
        self.centre.setCurrentIndex(1)
        if find:
            self.find()
        self._refresh()

    def clear(self):
        if self.jobbar.busy:
            return
        self.sources, self.groups, self.results = [], [], {}
        self._fill()
        self.view.clear()
        self.view_note.setText("")
        self.report.setText("")
        self.jobbar.say("")
        self._refresh()

    def choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Where results go", self.out_dir or "")
        if d:
            self.out_dir = d
            self.where.setText(f"Saves to: {d}")

    # ------------------------------------------------------------------------------------ find
    def strictness(self) -> str:
        return self.strict.key() or "normal"

    def find(self):
        SO = self.SO
        if not self.sources or self.jobbar.busy:
            return
        strict, gap = self.strictness(), self.gap.value()
        self.settings.set(f"{self.SETTINGS_KEY}_strictness", strict)
        self.settings.set(f"{self.SETTINGS_KEY}_gap", gap)
        sources = list(self.sources)

        def work(progress, cancel):
            paths = SO.find_paths(sources)
            if not paths:
                raise ValueError("no photos the Studio can stack were found there")
            progress(0.0, f"Reading {len(paths)} photos")
            return SO.sort_frames(paths, strict, gap, progress, cancel)

        def done(groups):
            self.groups, self.results = groups, {}
            self._fill()
            self._say_report()
            self._refresh()

        def fail(err, cancelled):
            self.jobbar.say("Cancelled." if cancelled else f"Finding stacks: {err}", T.AMBER if cancelled else T.RED)
            self._refresh()

        if self.jobbar.start(work, done, fail, text="Reading the photos..."):
            self._refresh()

    def _rejudge(self, *_):
        """A new strictness: judge the stacks again from what was read (seconds). Frames put back or left out by
        hand are judged again too."""
        SO = self.SO
        if not self.groups or self.jobbar.busy:
            return
        self.settings.set(f"{self.SETTINGS_KEY}_strictness", self.strictness())
        for g in self.groups:
            if g.kind == "focus" and all(f.energy is not None for f in g.frames):
                SO.judge(g.frames, self.strictness())
        self._fill()
        self._say_report()

    def _say_report(self):
        SO = self.SO
        text = SO.report(self.groups)
        self.report.setText(text)
        self.jobbar.say(text, T.CYAN)

    # ------------------------------------------------------------------------------------ the queue
    def stacks(self) -> list:
        return [g for g in self.groups if g.kind == "focus"]

    def _fill(self):
        self._filling = True
        self.tree.clear()
        for gi, g in enumerate(self.groups):
            it = QTreeWidgetItem(self.tree)
            it.setData(0, ROLE_GROUP, gi)
            self._group_text(it, g)
            if g.kind == "focus":
                for fi, f in enumerate(g.frames):
                    ch = QTreeWidgetItem(it)
                    ch.setData(0, ROLE_GROUP, gi)
                    ch.setData(0, ROLE_FRAME, fi)
                    ch.setFlags(ch.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    self._frame_text(ch, f)
                it.setExpanded(len(self.stacks()) <= 6)
            else:
                for f in g.frames:
                    ch = QTreeWidgetItem(it)
                    ch.setText(0, f.name)
                    ch.setForeground(0, QBrush(QColor(T.DIM)))
                    ch.setData(0, ROLE_GROUP, gi)
                    ch.setData(0, ROLE_FRAME, g.frames.index(f))
        self._filling = False
        self._refresh()

    def _group_text(self, it, g):
        SO = self.SO
        n, k = len(g.frames), len(g.left_out)
        head = f"{SO.KIND_TEXT[g.kind]}: {g.frames[0].name} to {g.frames[-1].name}" if n > 1 else \
            f"{SO.KIND_TEXT[g.kind]}: {g.frames[0].name}"
        it.setText(0, head)
        why = f"{n} frames" + (f", {len(g.used)} used, {k} left out" if g.kind == "focus" else "")
        done = self.results.get(id(g))
        if done:
            why += f"  ·  saved {os.path.basename(done)}"
        if g.note:
            why += f"  ·  {g.note}"
        it.setText(1, why)
        colour = T.GREEN if done else KIND_COLOUR.get(g.kind, T.TEXT)
        if g.kind == "focus" and len(g.used) < SO.MIN_STACK:
            colour = T.AMBER
        for c in (0, 1):
            it.setForeground(c, QBrush(QColor(colour)))
        f = it.font(0)
        f.setBold(g.kind == "focus")
        it.setFont(0, f)

    def _frame_text(self, ch, f):
        ch.setText(0, f.name)
        ch.setCheckState(0, Qt.CheckState.Unchecked if f.status == "out" else Qt.CheckState.Checked)
        ch.setText(1, f.reason or ("used" if f.status == "ok" else ""))
        col = QColor(STATUS_COLOUR.get(f.status, T.DIM))
        for c in (0, 1):
            ch.setForeground(c, QBrush(col))
        ch.setToolTip(1, f.reason or "Used: it is the sharpest frame somewhere in the stack.")

    def _item_frame(self, it):
        if it is None:
            return None, None
        gi, fi = it.data(0, ROLE_GROUP), it.data(0, ROLE_FRAME)
        g = self.groups[gi] if gi is not None and 0 <= gi < len(self.groups) else None
        f = g.frames[fi] if g is not None and fi is not None else None
        return g, f

    def _ticked(self, it, col):
        """One click puts a frame back or leaves it out (by hand: the reason says so)."""
        if self._filling or col != 0:
            return
        g, f = self._item_frame(it)
        if g is None or f is None or g.kind != "focus":
            return
        on = it.checkState(0) == Qt.CheckState.Checked
        if on and f.status == "out":
            f.status, f.reason = "warn", "put back by you" + (f" (was: {f.reason})" if f.reason else "")
        elif not on and f.status != "out":
            f.status, f.reason = "out", "left out by you"
        else:
            return
        self.results.pop(id(g), None)
        self._filling = True
        self._frame_text(it, f)
        self._group_text(it.parent(), g)
        self._filling = False
        self._say_report()
        self._refresh()

    def _peak_toggled(self, on):
        self.settings.set(f"{self.SETTINGS_KEY}_peak", bool(on))
        self._selected(self.tree.currentItem())

    def _show_peak(self, g, f):
        """The focus overlay (sorter.focus_overlay) for a stack or one of its frames, worked out in the background
        the first time and kept."""
        from .common import run_bg
        SO = self.SO
        k = -1 if f is None else g.frames.index(f)
        key = (id(g), k, tuple(x.status == "out" for x in g.frames))
        if key in self._peaks:
            self._paint_peak(g, f, *self._peaks[key])
            return
        if any(x.grey is None for x in g.frames):
            return
        self._peak_token += 1
        token = self._peak_token
        self.view_note.setText("Finding what is in focus...")

        def done(_, r):
            self._peaks[key] = r
            if token == self._peak_token:
                self._paint_peak(g, f, *r)

        def failed(_, err):
            if token == self._peak_token:
                self.view_note.setText(f"Could not show what is in focus: {err}")
        run_bg(lambda: SO.focus_overlay(g.frames, None if k < 0 else k, T.CYAN, T.AMBER), "focus peak", done, failed)

    def _paint_peak(self, g, f, rgb, inf, missed):
        from .common import to_qimage
        self._set_pixmap(QPixmap.fromImage(to_qimage(np.ascontiguousarray(rgb))))
        if f is None:
            self.view_note.setText(f"{g.frames[0].name} to {g.frames[-1].name}  ·  in focus in some frame: "
                                   f"{100 * inf:.0f}% of the frame (cyan)  ·  detail no frame got sharp: "
                                   f"{100 * missed:.1f}% (amber)  ·  dimmed: not in every frame")
        else:
            self.view_note.setText(f"{f.name}  ·  {f.reason or f.status}  ·  in focus in {100 * inf:.1f}% of the "
                                   f"frame (cyan)")

    def _selected(self, it, _prev=None):
        g, f = self._item_frame(it)
        if g is None:
            return
        if self.peak.isChecked() and g.kind == "focus" and not (f is None and self.results.get(id(g))):
            self._show_peak(g, f)
            self._refresh()
            return
        if f is None:
            path = self.results.get(id(g))
            if path and os.path.exists(path):
                self._show_result(path)
            elif g.frames and g.frames[len(g.frames) // 2].grey is not None:
                self._show_grey(g.frames[len(g.frames) // 2], "middle frame")
        else:
            self._show_grey(f, f.reason or f.status)
        self._refresh()

    def _double(self, it, _col):
        g, f = self._item_frame(it)
        if g is not None and f is None and self.results.get(id(g)):
            self.open_result.emit(self.results[id(g)])

    def _show_grey(self, f, note: str):
        if f.grey is None:
            self.view.clear()
            self.view_note.setText(f"{f.name}: {f.problem or 'no preview'}")
            return
        a = np.ascontiguousarray((np.clip(f.grey, 0, 1) * 255).astype(np.uint8))
        h, w = a.shape
        qi = QImage(a.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        self._set_pixmap(QPixmap.fromImage(qi))
        self.view_note.setText(f"{f.name}  ·  {note}  ·  sharpest in {100 * f.share:.1f}% of the stack's detail")

    def _show_result(self, path: str):
        from ..engine.io.intermediate import read_preview
        from .common import to_qimage
        try:
            a = read_preview(path)
        except (OSError, ValueError):
            a = None
        if a is None:
            self.view.clear()
            self.view_note.setText(os.path.basename(path))
            return
        self._set_pixmap(QPixmap.fromImage(to_qimage(np.ascontiguousarray(a[..., :3]))))
        self.view_note.setText(f"{os.path.basename(path)}  ·  double-click the stack to open it in Develop")

    def _set_pixmap(self, pm: QPixmap):
        h = max(120, self.view.height() - 4)
        w = max(160, self.view.width() - 4)
        self.view.setPixmap(pm.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))

    def _open_selected(self):
        g, _ = self._item_frame(self.tree.currentItem())
        path = self.results.get(id(g)) if g is not None else None
        if path:
            self.open_result.emit(path)

    # ------------------------------------------------------------------------------------ stacking
    def _ready(self, g) -> bool:
        return g.kind == "focus" and len(g.used) >= self.SO.MIN_STACK

    def stack_all(self):
        self.queue = [g for g in self.stacks() if self._ready(g) and id(g) not in self.results]
        self._next()

    def stack_selected(self):
        g, _ = self._item_frame(self.tree.currentItem())
        if g is not None and self._ready(g):
            self.queue = [g]
            self._next()

    def _next(self):
        """Stack the queue one group at a time (each is its own background job, so every finished stack shows up
        in the list at once and Cancel stops after the current frame)."""
        SO = self.SO
        if not self.queue or self.jobbar.busy:
            return
        g = self.queue.pop(0)
        total = len([x for x in self.stacks() if self._ready(x)])
        k = total - len(self.queue)
        from ..engine.stack.focus import FocusParams
        params = FocusParams(local=self.local.isChecked())
        out_dir = self.out_dir or None

        def work(progress, cancel):
            return SO.stack_group(g, out_dir, params,
                                  lambda f, m: progress(f, f"Stack {k} of {total}: {m}"), cancel)

        def done(r):
            self.results[id(g)] = r["path"]
            self.result_written.emit(r["path"])
            self._fill()
            self._select_group(g)
            self._say_report()
            self.jobbar.say(f"Saved {os.path.basename(r['path'])} ({r['used']} frames)."
                            + (f" {len(self.queue)} to go." if self.queue else " All done."), T.GREEN)
            self._next()

        def fail(err, cancelled):
            self.queue = []
            if cancelled:
                self.jobbar.say("Cancelled. Stacks already saved are kept; nothing else was written.", T.AMBER)
            else:
                self.jobbar.say(f"Stacking {g.frames[0].name}: {err}", T.RED)
            self._fill()
            self._refresh()

        self.jobbar.start(work, done, fail, keep=True, text=f"Stacking {k} of {total}...")
        self._refresh()

    def _select_group(self, g):
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            gi = it.data(0, ROLE_GROUP)
            if gi is not None and self.groups[gi] is g:
                self.tree.setCurrentItem(it)
                return

    # ------------------------------------------------------------------------------------ state
    def _refresh(self):
        busy = self.jobbar.busy
        self.centre.setCurrentIndex(1 if (self.sources or self.groups) else 0)
        self.find_b.setEnabled(bool(self.sources) and not busy)
        ready = [g for g in self.stacks() if self._ready(g)]
        self.all_b.setEnabled(bool(ready) and not busy)
        g, _ = self._item_frame(self.tree.currentItem())
        self.one_b.setEnabled(g is not None and self._ready(g) and not busy)
        self.open_b.setEnabled(g is not None and bool(self.results.get(id(g))))
        self.add_b.setEnabled(not busy)
        self.clear_b.setEnabled(not busy)
        while self.chips.count():
            w = self.chips.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        if self.groups:
            st = self.stacks()
            self.chips.addWidget(chip(f"{len(st)} stack{'s' * (len(st) != 1)}", T.CYAN if st else T.AMBER))
            out = sum(len(x.left_out) for x in st)
            if out:
                self.chips.addWidget(chip(f"{out} left out", T.RED))
            done = len(self.results)
            if done:
                self.chips.addWidget(chip(f"{done} saved", T.GREEN))
