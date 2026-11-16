"""The Library workspace (plan sections 6.2 and 24): point it at a folder, look through the raws, decide what to
process, without Windows Explorer. Import from card brings photos in; Review judges them large; Process picks sends
the chosen ones on.

Speed on big folders (24.7 B): the folder is listed in the background; thumbnails come from the disk cache
(cache/thumbs) and only for what is on screen plus one screen ahead; ratings and camera details are read in the
background and remembered in cache/library_index.sqlite. Nothing slow runs on the window's own thread.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict

from PySide6.QtCore import (QAbstractListModel, QModelIndex, QObject, QRect, QRectF, QRunnable, QSize, Qt,
                            QThreadPool, QTimer, QUrl, Signal)
from PySide6.QtGui import QColor, QDesktopServices, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
                               QFileSystemModel, QFrame, QHBoxLayout, QInputDialog, QLineEdit, QListView, QListWidget,
                               QListWidgetItem, QMenu, QMessageBox, QProgressBar, QPushButton, QSlider, QSplitter,
                               QStackedWidget, QStyle, QStyledItemDelegate, QToolButton, QTreeView, QVBoxLayout,
                               QWidget)

from . import theme as T
from .common import (GLOW_HOWTO, STARS, LensDialog, Settings, chip, label, log, run_bg, tag_lens,  # noqa: F401
                     to_qimage)

ROLE_PATH = Qt.ItemDataRole.UserRole
ROLE_ENTRY = Qt.ItemDataRole.UserRole + 1

RATING_FILTERS = ["All photos", "★ and up", "★★ and up", "★★★ and up", "★★★★ and up", "★★★★★", "Picks only",
                  "Hide rejects", "Rejects only"]
SORTS = ["Capture time", "File name", "Rating", "ISO"]
SORT_LABELS = {"Capture time": "By time", "File name": "By name", "Rating": "By rating", "ISO": "By ISO"}
ISO_FILTERS = [("Any ISO", None, None), ("ISO up to 400", None, 400), ("ISO 400-1600", 400, 1600),
               ("ISO 1600-6400", 1600, 6400), ("ISO 6400 and up", 6400, None)]
PIX_CACHE = 900                                   # thumbnails kept in memory (the rest come back from the disk cache)


# ============================================================================================ background helpers
class _Runner(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.fn()
        except Exception:                             # noqa: BLE001  (logged with its traceback, rule 0.3.5)
            from .. import log as slog
            slog.log_exception("library background work failed", log)


EXTRA_POOLS: list = []                            # pools the app waits for at exit (see main.finish_jobs)


def _pool(n: int) -> QThreadPool:
    p = QThreadPool()
    p.setMaxThreadCount(max(1, n))
    EXTRA_POOLS.append(p)
    return p


class ThumbLoader(QObject):
    """Thumbnails for what is on screen. ``want`` replaces the waiting list (scrolling on drops what is no longer
    needed); a few readers work through it (2 on a spinning drive, more on an SSD)."""
    ready = Signal(int, str, QImage)

    def __init__(self, cache=None):
        super().__init__()
        from ..engine.io.thumbs import default_cache
        self.cache = cache or default_cache()
        self.pool = _pool(2)
        self.lock = threading.Lock()
        self.queue: list = []
        self.inflight: set = set()
        self.failed: set = set()
        self.active = 0
        self.gen = 0

    def set_readers(self, n: int) -> None:
        self.pool.setMaxThreadCount(max(1, n))

    def reset(self) -> int:
        with self.lock:
            self.gen += 1
            self.queue = []
            return self.gen

    def want(self, entries: list) -> None:
        with self.lock:
            self.queue = [(e["path"], e["size"], e["mtime"]) for e in entries
                          if e["path"] not in self.inflight and e["path"] not in self.failed]
            need = min(self.pool.maxThreadCount(), len(self.queue)) - self.active
            self.active += max(0, need)
        for _ in range(max(0, need)):
            self.pool.start(_Runner(self._work))

    def _work(self):
        try:
            while True:
                with self.lock:
                    if not self.queue:
                        return
                    p, size, mtime = self.queue.pop(0)
                    self.inflight.add(p)
                    gen = self.gen
                img = None
                try:
                    img = to_qimage(self.cache.get(p, size, mtime))
                except Exception as e:                # noqa: BLE001  (an unreadable raw: placeholder stays)
                    log.warning("thumbnail failed for %s: %s", p, e)
                    with self.lock:
                        self.failed.add(p)
                finally:
                    with self.lock:
                        self.inflight.discard(p)
                if img is not None:
                    self.ready.emit(gen, p, img)
        finally:
            with self.lock:
                self.active -= 1


class MetaLoader(QObject):
    """Ratings, picks, lens tags and camera details, from the index or the files, in batches."""
    batch = Signal(int, list)
    done = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.pool = _pool(1)
        self.cancel = threading.Event()
        self.gen = 0

    def start(self, gen: int, entries: list) -> None:
        self.cancel.set()
        self.cancel = threading.Event()
        cancel = self.cancel
        self.gen = gen

        def work():
            from ..engine.io.libindex import load_entries
            n = load_entries([dict(e) for e in entries], on_batch=lambda b: self.batch.emit(gen, b), cancel=cancel)
            self.done.emit(gen, n)
        self.pool.start(_Runner(work))

    def stop(self):
        self.cancel.set()


class SidecarWriter(QObject):
    """Saves ratings and picks in the background, one after another, in order. A rating that cannot be saved is
    reported (24.7 C: it must be seen)."""
    saved = Signal(str, dict)
    failed = Signal(str, dict, str)

    def __init__(self):
        super().__init__()
        self.pool = _pool(1)

    def save(self, path: str, meta: dict, entry: dict | None = None) -> None:
        meta = dict(meta)
        entry = dict(entry or {})

        def work():
            from .. import engine as E
            try:
                side = E.save_library_meta(path, meta)
            except OSError as err:
                self.failed.emit(path, meta, str(err))
                return
            try:                                     # keep the index in step, so the next visit is instant
                from ..engine.io.libindex import LibraryIndex
                st = os.stat(path)
                data = {k: entry.get(k) for k in ("rating", "pick", "label", "lens_tag", "has_edit", "taken", "camera",
                                                  "lens", "iso", "exposure", "fnumber", "focal")}
                data.update(rating=meta.get("rating", 0), pick=meta.get("pick", 0), label=meta.get("label", ""))
                idx = LibraryIndex()
                idx.update(path, st.st_size, st.st_mtime, os.path.getmtime(side), data)
                idx.close()
            except Exception:                         # noqa: BLE001  (the index is only a cache)
                pass
            self.saved.emit(path, meta)
        self.pool.start(_Runner(work))

    def wait(self, ms: int = 5000) -> bool:
        return self.pool.waitForDone(ms)


# ============================================================================================ model and delegate
class PhotoModel(QAbstractListModel):
    """The photos of the open folder: ``all`` as listed, ``rows`` as filtered and sorted for the grid."""

    def __init__(self):
        super().__init__()
        self.all: list = []
        self.rows: list = []
        self.by_path: dict = {}
        self.row_of: dict = {}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, idx, role=Qt.ItemDataRole.DisplayRole):
        if not idx.isValid() or idx.row() >= len(self.rows):
            return None
        e = self.rows[idx.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return e["name"]
        if role == ROLE_PATH:
            return e["path"]
        if role == ROLE_ENTRY:
            return e
        if role == Qt.ItemDataRole.ToolTipRole:
            return entry_details(e, multiline=True)
        return None

    def set_entries(self, entries: list) -> None:
        self.beginResetModel()
        self.all = entries
        self.by_path = {e["path"]: e for e in entries}
        self.rows = list(entries)
        self.row_of = {e["path"]: i for i, e in enumerate(self.rows)}
        self.endResetModel()

    def set_rows(self, rows: list) -> None:
        self.beginResetModel()
        self.rows = rows
        self.row_of = {e["path"]: i for i, e in enumerate(rows)}
        self.endResetModel()

    def changed(self, path: str) -> None:
        r = self.row_of.get(path)
        if r is not None:
            ix = self.index(r)
            self.dataChanged.emit(ix, ix)

    def index_of(self, path: str) -> QModelIndex:
        r = self.row_of.get(path)
        return self.index(r) if r is not None else QModelIndex()


def entry_details(e: dict, multiline: bool = False) -> str:
    """'IMG_0001.CR3 · 2026-11-14 21:03 · Canon EOS R5 · Laack 13.5cm · ISO 3200 · 1/60 s · f/4.5 · 135 mm'"""
    parts = [e["name"]]
    if e.get("taken"):
        parts.append(e["taken"][:16])
    if e.get("camera"):
        parts.append(e["camera"])
    lens = e.get("lens_tag") or e.get("lens")
    if lens:
        parts.append(lens + (" (tagged)" if e.get("lens_tag") else ""))
    if e.get("iso"):
        parts.append(f"ISO {e['iso']:.0f}")
    x = e.get("exposure")
    if x:
        parts.append(f"1/{1 / x:.0f} s" if x < 0.5 else f"{x:.1f} s".replace(".0 s", " s"))
    if e.get("fnumber"):
        parts.append(f"f/{e['fnumber']:g}")
    if e.get("focal"):
        parts.append(f"{e['focal']:.0f} mm")
    return ("\n" if multiline else "  ·  ").join(parts)


class PhotoDelegate(QStyledItemDelegate):
    """Paints one photo: thumbnail, name, stars, pick / reject, and small badges (edited, lens tagged)."""

    def __init__(self, page, strip: bool = False):
        super().__init__(page)
        self.page = page
        self.strip = strip

    def sizeHint(self, opt, idx):
        s = self.page.strip_size if self.strip else self.page.thumb_size
        return QSize(s + 16, int(s * 0.72) + (16 if self.strip else 56))

    def paint(self, p: QPainter, opt, idx):
        e = idx.data(ROLE_ENTRY)
        if e is None:
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        sel = bool(opt.state & QStyle.StateFlag.State_Selected)
        r = QRectF(opt.rect).adjusted(4, 4, -4, -4)
        p.setPen(QPen(QColor(T.CYAN if sel else T.LINE), 2 if sel else 1))
        p.setBrush(QColor("#12324a" if sel else T.PANEL))
        p.drawRoundedRect(r, 9, 9)
        text_h = 0 if self.strip else 40
        ir = r.adjusted(6, 6, -6, -6 - text_h)
        pm = self.page.pix.get(e["path"])
        if pm is not None and not pm.isNull():
            s = min(ir.width() / pm.width(), ir.height() / pm.height())
            w, h = pm.width() * s, pm.height() * s
            tr = QRectF(ir.center().x() - w / 2, ir.center().y() - h / 2, w, h)
            p.drawPixmap(tr, pm, QRectF(pm.rect()))
            if e.get("pick") == -1:                  # a reject looks dimmed
                p.fillRect(tr, QColor(0, 0, 0, 150))
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(T.BG2))
            p.drawRoundedRect(ir, 6, 6)
        # badges
        bx = ir.left() + 4
        for show, txt, col in ((e.get("has_edit"), "✎", T.VIOLET), (e.get("lens_tag"), "◎", T.CYAN)):
            if show:
                br = QRectF(bx, ir.top() + 4, 20, 18)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 0, 0, 170))
                p.drawRoundedRect(br, 5, 5)
                p.setPen(QColor(col))
                p.drawText(br, Qt.AlignmentFlag.AlignCenter, txt)
                bx += 24
        if e.get("pick") in (1, -1):
            br = QRectF(ir.right() - 24, ir.top() + 4, 20, 18)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 170))
            p.drawRoundedRect(br, 5, 5)
            p.setPen(QColor(T.GREEN if e["pick"] == 1 else T.RED))
            p.drawText(br, Qt.AlignmentFlag.AlignCenter, "✔" if e["pick"] == 1 else "✖")
        if self.strip:
            if e.get("rating"):
                p.setPen(QColor(T.AMBER))
                p.setFont(QFont("Segoe UI", 7))
                p.drawText(ir.adjusted(0, 0, -2, -1), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
                           STARS[e["rating"]])
            p.restore()
            return
        tr = QRectF(r.left() + 8, r.bottom() - text_h, r.width() - 16, 20)
        p.setPen(QColor(T.TEXT))
        fm = p.fontMetrics()
        p.drawText(tr, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   fm.elidedText(e["name"], Qt.TextElideMode.ElideMiddle, int(tr.width())))
        sr = QRectF(r.left() + 8, r.bottom() - text_h + 18, r.width() - 16, 18)
        p.setPen(QColor(T.AMBER if e.get("rating") else T.DIM))
        p.drawText(sr, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   STARS[e.get("rating") or 0] or ("·" if e.get("rating") is not None else "…"))
        if e.get("iso"):
            p.setPen(QColor(T.DIM))
            p.drawText(sr, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"ISO {e['iso']:.0f}")
        p.restore()


# ============================================================================================ the Library page
class LibraryPage(QWidget):
    open_photo = Signal(str)
    lens_tagged = Signal(list, str)
    paste_requested = Signal(list)
    export_requested = Signal(list)
    process_requested = Signal(list)
    folder_loaded = Signal(str)

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.folder = ""
        self.gen = 0
        self.loading = False
        self.meta_pending = False
        self.pix: OrderedDict = OrderedDict()
        self.thumb_size = int(settings.get("library_thumb", 220) or 220)
        self.strip_size = 96
        self.warned_folders: set = set()
        self.timings: dict = {}
        self.save_open_edit = lambda: None          # set by the main window (Develop's unsaved edit)
        self.loader = ThumbLoader()
        self.loader.ready.connect(self._thumb_ready)
        self.meta = MetaLoader()
        self.meta.batch.connect(self._meta_batch)
        self.meta.done.connect(self._meta_done)
        self.writer = SidecarWriter()
        self.writer.failed.connect(self._save_failed)
        self.model = PhotoModel()
        self.setAcceptDrops(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_topbar())
        root.addWidget(self._build_filterbar())
        self.notice = self._build_notice()
        root.addWidget(self.notice)
        self.body = QStackedWidget()
        split = QSplitter()
        split.addWidget(self._build_left())
        self.grid = QListView()
        self.grid.setObjectName("grid")
        self.grid.setModel(self.model)
        self.grid.setItemDelegate(PhotoDelegate(self))
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setMovement(QListView.Movement.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.grid.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.grid.verticalScrollBar().setSingleStep(40)
        self.grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self._context_menu)
        self.grid.doubleClicked.connect(lambda ix: self.open_photo.emit(ix.data(ROLE_PATH)))
        self.grid.verticalScrollBar().valueChanged.connect(self._schedule_thumbs)
        self.grid.installEventFilter(self)
        self._apply_thumb_size()
        split.addWidget(self.grid)
        split.setSizes([250, 1200])
        split.setStretchFactor(1, 1)
        self.body.addWidget(split)
        from .review import ReviewView
        self.review = ReviewView(self)
        self.body.addWidget(self.review)
        root.addWidget(self.body, 1)
        self.progress = self._build_progress()
        root.addWidget(self.progress)
        self.hint = label("Double-click or Enter to develop.  Space: Review.  Keys: 0-5 rate, P pick, X reject, "
                          "U clear flag.  Ratings live in the sidecar beside each raw, never in a catalogue.", "dim",
                          wrap=True)                # wraps on a narrow window instead of widening it
        self.hint.setContentsMargins(18, 6, 18, 8)
        root.addWidget(self.hint)
        self.placeholder = QPixmap()
        self._thumb_timer = QTimer(self, singleShot=True, interval=40, timeout=self._request_thumbs)
        self._filter_timer = QTimer(self, singleShot=True, interval=250, timeout=self.apply_filter)
        self._cards = None
        QShortcut(QKeySequence("Ctrl+Shift+V"), self, activated=lambda: self.paste_requested.emit(
            self.selected_files()))
        self._refresh_recent()

    # ------------------------------------------------------------------------------------ building
    def _build_topbar(self):
        top = QFrame()
        top.setObjectName("topbar")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(18, 10, 18, 6)
        self.title = label("Library", "h1")
        self.path_lbl = label("", "dim")
        self.path_lbl.setMinimumWidth(80)
        from PySide6.QtWidgets import QSizePolicy
        self.path_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.count = chip("0 photos", T.CYAN)
        self.import_b = QPushButton("Import from card...")
        self.import_b.setObjectName("primary")
        self.import_b.setToolTip("Copy the photos from a memory card into your Photos folder, checked, into a folder "
                                 "named by the date (or your own name). The card is never changed.")
        self.import_b.clicked.connect(self.import_from_card)
        self.review_b = QPushButton("Review")
        self.review_b.setToolTip("See the selected photo large, check focus at 100%, rate with the keyboard (Space)")
        self.review_b.clicked.connect(self.open_review)
        self.process_b = QPushButton("Process picks...")
        self.process_b.setToolTip("Export your picks (or the selected photos) in one go, in the background")
        self.process_b.clicked.connect(lambda: self.process_picks())
        open_b = QPushButton("Open folder...")
        open_b.clicked.connect(self.choose_folder)
        self.tools_b = QToolButton()
        self.tools_b.setText("Tools ▾")
        self.tools_b.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        m = QMenu(self.tools_b)
        m.addAction("◎ Tag lens...", self.tag_selected).setToolTip(
            "Tell the Studio which (contactless) lens the selected photos were taken with")
        m.addAction("Paste settings  (Ctrl+Shift+V)", lambda: self.paste_requested.emit(self.selected_files()))
        m.addAction("Learn dust map", self.learn_dust)
        m.addAction("Measure lens glow", self.measure_glow)
        m.addSeparator()
        m.addAction("Photos folder...", self.choose_photos_folder)
        self.central_act = m.addAction("Keep this folder's ratings and edits on this PC", self.toggle_central)
        self.central_act.setCheckable(True)
        m.addAction("Clear thumbnail cache", self.clear_cache)
        m.setToolTipsVisible(True)
        self.tools_b.setMenu(m)
        tl.addWidget(self.title)
        tl.addSpacing(12)
        tl.addWidget(self.path_lbl, 1)
        tl.addWidget(self.count)
        tl.addSpacing(8)
        for w in (self.import_b, self.review_b, self.process_b, open_b, self.tools_b):
            tl.addWidget(w)
        return top

    def _build_filterbar(self):
        bar = QFrame()
        bar.setObjectName("topbar")
        fl = QHBoxLayout(bar)
        fl.setContentsMargins(18, 4, 18, 8)
        fl.setSpacing(6)
        self.filter = QComboBox()
        self.filter.addItems(RATING_FILTERS)
        self.filter.setToolTip("Show photos by rating or flag")
        self.cam_f = QComboBox()
        self.cam_f.addItem("Any camera", "")
        self.lens_f = QComboBox()
        self.lens_f.addItem("Any lens", "")
        self.iso_f = QComboBox()
        for name, lo, hi in ISO_FILTERS:
            self.iso_f.addItem(name, (lo, hi))
        self.date_f = QComboBox()
        self.date_f.addItem("Any date", "")
        for c in (self.filter, self.cam_f, self.lens_f, self.iso_f, self.date_f):
            c.setMaximumWidth(170)
            c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(5)           # (the window fits a 1280-pixel-wide screen)
        self.unrated = QCheckBox("Not rated yet")
        self.unrated.setToolTip("Only photos with no stars and no pick / reject flag")
        self.search = QLineEdit()
        self.search.setPlaceholderText("File name...")
        self.search.setClearButtonEnabled(True)
        self.search.setMaximumWidth(130)
        self.search.setMinimumWidth(80)
        self.sort = QComboBox()
        self.sort.setToolTip("Sort by")
        self.sort.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.sort.setMinimumContentsLength(8)
        for s in SORTS:
            self.sort.addItem(SORT_LABELS[s], s)
        self.sort.setCurrentIndex(max(0, SORTS.index(self.settings.get("library_sort", "Capture time"))
                                      if self.settings.get("library_sort") in SORTS else 0))
        self.size_s = QSlider(Qt.Orientation.Horizontal)
        self.size_s.setRange(120, 400)
        self.size_s.setValue(self.thumb_size)
        self.size_s.setMaximumWidth(90)
        self.size_s.setMinimumWidth(60)
        self.size_s.setToolTip("Thumbnail size")
        self.size_s.valueChanged.connect(self._size_changed)
        self.subfolders = QCheckBox("Include subfolders")
        self.subfolders.setChecked(bool(self.settings.get("library_subfolders", False)))
        self.subfolders.toggled.connect(self._subfolders_toggled)
        for w in (self.filter, self.cam_f, self.lens_f, self.iso_f, self.date_f):
            w.currentIndexChanged.connect(self.apply_filter)
        self.unrated.toggled.connect(self.apply_filter)
        self.search.textChanged.connect(lambda _: self._filter_timer.start())
        self.sort.currentIndexChanged.connect(self._sort_changed)
        for w in (self.filter, self.cam_f, self.lens_f, self.iso_f, self.date_f, self.unrated, self.search):
            fl.addWidget(w)
        fl.addStretch(1)
        fl.addWidget(self.sort)
        fl.addWidget(label("Size", "dim"))
        fl.addWidget(self.size_s)
        return bar

    def _build_notice(self):
        f = QFrame()
        f.setObjectName("card")
        f.setStyleSheet(f"QFrame#card {{ border: 1px solid {T.alpha(T.CYAN, 0.5)}; margin: 8px 18px 0 18px; }}")
        h = QHBoxLayout(f)
        h.setContentsMargins(14, 8, 10, 8)
        self.notice_text = label("", None)
        self.notice_text.setStyleSheet(f"color:{T.CYAN};")
        go = QPushButton("Import...")
        go.setObjectName("primary")
        go.clicked.connect(lambda: self.import_from_card(self._notice_card))
        x = QPushButton("✕")
        x.setFixedSize(38, 34)
        x.setToolTip("Hide this notice (it comes back when another card is put in)")
        x.clicked.connect(f.hide)
        h.addWidget(self.notice_text, 1)
        h.addWidget(go)
        h.addWidget(x)
        f.hide()
        self._notice_card = None
        return f

    def _build_left(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 0, 8)
        v.setSpacing(6)
        self.photos_b = QPushButton("⌂  Photos")
        self.photos_b.setToolTip("Your Photos folder, where Import from card puts new photos")
        self.photos_b.clicked.connect(self.open_photos_folder)
        v.addWidget(self.photos_b)
        v.addWidget(label("Recent and pinned", "dim"))
        self.recent = QListWidget()
        self.recent.setMaximumHeight(150)
        self.recent.itemClicked.connect(lambda it: self.load_folder(it.data(Qt.ItemDataRole.UserRole)))
        self.recent.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.recent.customContextMenuRequested.connect(self._recent_menu)
        v.addWidget(self.recent)
        v.addWidget(self.subfolders)
        self.fsmodel = QFileSystemModel()
        self.fsmodel.setRootPath("")
        from PySide6.QtCore import QDir
        self.fsmodel.setFilter(QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.Drives)
        self.tree = QTreeView()
        self.tree.setModel(self.fsmodel)
        for c in (1, 2, 3):
            self.tree.hideColumn(c)
        self.tree.setHeaderHidden(True)
        self.tree.clicked.connect(lambda idx: self.load_folder(self.fsmodel.filePath(idx)))
        v.addWidget(self.tree, 1)
        return w

    def _build_progress(self):
        f = QFrame()
        f.setObjectName("bottombar")
        h = QHBoxLayout(f)
        h.setContentsMargins(18, 8, 18, 8)
        self.prog_text = label("", None)
        self.prog_bar = QProgressBar()
        self.prog_bar.setMaximumWidth(360)
        self.prog_cancel = QPushButton("Cancel")
        self.prog_open = QPushButton("Open folder")
        self.prog_close = QPushButton("✕")
        self.prog_close.setFixedWidth(34)
        self.prog_close.clicked.connect(f.hide)
        h.addWidget(self.prog_text, 1)
        h.addWidget(self.prog_bar)
        h.addWidget(self.prog_cancel)
        h.addWidget(self.prog_open)
        h.addWidget(self.prog_close)
        f.hide()
        return f

    # ------------------------------------------------------------------------------------ folders
    @property
    def files(self) -> list:
        """The photos shown, in the order shown."""
        return [e["path"] for e in self.model.rows]

    @property
    def all_files(self) -> list:
        return [e["path"] for e in self.model.all]

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose a folder of raw files", self.settings.get("folder", ""))
        if d:
            self.load_folder(d)

    def photos_dir(self) -> str:
        return str(self.settings.get("photos_dir") or "")

    def open_photos_folder(self):
        d = self.photos_dir()
        if not d:
            d = self.choose_photos_folder()
        if d:
            from ..engine.io.ingest import drive_connected
            if not os.path.isdir(d):
                if not drive_connected(d):
                    QMessageBox.information(self, "Photos folder", f"Your Photos folder is on {os.path.splitdrive(d)[0]}"
                                            f", which is not connected. Connect the drive and try again.")
                    return
                os.makedirs(d, exist_ok=True)
            self.load_folder(d)

    def choose_photos_folder(self) -> str:
        from .importer import PhotosFolderDialog
        dlg = PhotosFolderDialog(self.settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.chosen
        return ""

    def load_folder(self, folder: str, select: str | None = None):
        """Open a folder: listed in the background, so the window answers at once whatever its size."""
        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            return
        self.meta.stop()
        self.gen = self.loader.reset()
        gen = self.gen
        self.folder = folder
        self.loading = True
        self.settings.set("folder", folder)
        self._remember_recent(folder)
        self.path_lbl.setText(folder)
        self.path_lbl.setToolTip(folder)
        self.count.setText("Opening...")
        self.model.set_entries([])
        self.pix.clear()
        self.warned_unsaved = False
        recursive = self.subfolders.isChecked()
        self.timings = {"start": time.perf_counter()}
        try:
            from ..engine.io.thumbs import reader_count
            self.loader.set_readers(reader_count(folder))
        except Exception:                             # noqa: BLE001
            self.loader.set_readers(2)
        from .. import engine as E
        self.central_act.setChecked(bool(E.central_folder_for(folder)))

        def listed(token, entries):
            if token != self.gen:
                return
            for e in entries:
                e.update(rating=None, pick=0, label="", lens_tag="", has_edit=0, taken="", camera="", lens="",
                         iso=None, exposure=None, fnumber=None, focal=None)
            self.timings["listed"] = time.perf_counter() - self.timings["start"]
            self.model.set_entries(entries)
            self.loading = False
            self._reset_filter_choices()
            self.apply_filter()
            self._schedule_thumbs()
            self.meta_pending = True
            self.meta.start(gen, entries)
            if select:
                self.select_path(select)
            self.folder_loaded.emit(folder)

        def failed(token, err):
            if token == self.gen:
                self.loading = False
                self.count.setText("0 photos")
                QMessageBox.warning(self, "Library", f"Could not open {folder}: {err}")
        from ..engine.io.libindex import list_photos
        run_bg(lambda: list_photos(folder, recursive), gen, listed, failed)
        idx = self.fsmodel.index(folder)
        if idx.isValid():
            self.tree.setCurrentIndex(idx)
            self.tree.scrollTo(idx)

    def reload(self, select: str | None = None):
        if self.folder:
            self.load_folder(self.folder, select)

    def _subfolders_toggled(self, on):
        self.settings.set("library_subfolders", bool(on))
        self.reload()

    # ------------------------------------------------------------------------------------ recent and pinned
    def _remember_recent(self, folder):
        rec = [f for f in (self.settings.get("recent_folders") or []) if os.path.normcase(f) != os.path.normcase(folder)]
        self.settings.set("recent_folders", ([folder] + rec)[:8])
        self._refresh_recent()

    def _refresh_recent(self):
        self.recent.clear()
        pinned = self.settings.get("pinned_folders") or []
        recent = [f for f in (self.settings.get("recent_folders") or []) if f not in pinned]
        for f, pin in [(f, True) for f in pinned] + [(f, False) for f in recent[:6]]:
            it = QListWidgetItem(("📌 " if pin else "") + (os.path.basename(f.rstrip("\\/")) or f))
            it.setToolTip(f + ("  (pinned; right-click to unpin)" if pin else "  (right-click to pin)"))
            it.setData(Qt.ItemDataRole.UserRole, f)
            it.setData(Qt.ItemDataRole.UserRole + 1, pin)
            self.recent.addItem(it)

    def _recent_menu(self, pos):
        it = self.recent.itemAt(pos)
        if it is None:
            return
        f, pin = it.data(Qt.ItemDataRole.UserRole), it.data(Qt.ItemDataRole.UserRole + 1)
        m = QMenu(self)
        pinned = list(self.settings.get("pinned_folders") or [])
        if pin:
            m.addAction("Unpin", lambda: (self.settings.set("pinned_folders", [x for x in pinned if x != f]),
                                          self._refresh_recent()))
        else:
            m.addAction("Pin", lambda: (self.settings.set("pinned_folders", pinned + [f]), self._refresh_recent()))
            m.addAction("Remove from the list", lambda: (self.settings.set(
                "recent_folders", [x for x in (self.settings.get("recent_folders") or []) if x != f]),
                self._refresh_recent()))
        m.exec(self.recent.mapToGlobal(pos))

    # ------------------------------------------------------------------------------------ thumbnails
    def _apply_thumb_size(self):
        s = self.thumb_size
        self.grid.setGridSize(QSize(s + 16, int(s * 0.72) + 56))
        self.grid.setIconSize(QSize(s, int(s * 0.72)))

    def _size_changed(self, v):
        self.thumb_size = int(v)
        self.settings.set("library_thumb", self.thumb_size)
        self._apply_thumb_size()
        self.grid.doItemsLayout()
        self._schedule_thumbs()

    def _schedule_thumbs(self, *_):
        self._thumb_timer.start()

    def visible_rows(self, ahead: float = 1.0) -> list:
        """Rows on screen, then one screen ahead (24.7 B: only what is on screen is asked for)."""
        n = self.model.rowCount()
        if not n:
            return []
        vp = self.grid.viewport().rect()
        gs = self.grid.gridSize()
        per_row = max(1, vp.width() // max(1, gs.width()))
        top = self.grid.verticalScrollBar().value()
        first = max(0, (top // max(1, gs.height()) - 1) * per_row)
        ext = QRect(vp.left(), vp.top(), vp.width(), int(vp.height() * (1 + ahead)))
        on, later = [], []
        for r in range(first, n):
            rect = self.grid.visualRect(self.model.index(r))
            if rect.top() > ext.bottom():
                break
            if rect.intersects(vp):
                on.append(r)
            elif rect.intersects(ext):
                later.append(r)
        if not on and not later:                   # not laid out yet (hidden): the first screenful
            return list(range(min(n, per_row * 6)))
        return on + later

    def _request_thumbs(self):
        rows = self.visible_rows()
        if self.body.currentIndex() == 1:
            rows = self.review.wanted_rows() + rows
        want = [self.model.rows[r] for r in rows if self.model.rows[r]["path"] not in self.pix]
        self.loader.want(want)

    def _thumb_ready(self, gen, path, img):
        if gen != self.gen:
            return
        self.pix[path] = QPixmap.fromImage(img)
        self.pix.move_to_end(path)
        while len(self.pix) > PIX_CACHE:
            self.pix.popitem(last=False)
        self.model.changed(path)
        self.review.thumb_ready(path)
        if "first_thumb" not in self.timings:
            self.timings["first_thumb"] = time.perf_counter() - self.timings.get("start", time.perf_counter())

    def icons(self) -> dict:
        from PySide6.QtGui import QIcon
        return {p: QIcon(pm) for p, pm in self.pix.items()}

    def clear_cache(self):
        from ..engine.io.thumbs import default_cache
        freed = default_cache().clear()
        self.pix.clear()
        QMessageBox.information(self, "Thumbnail cache", f"Cleared {freed / 1e6:.0f} MB of thumbnails. They are made "
                                "again as folders are opened.")
        self._schedule_thumbs()

    # ------------------------------------------------------------------------------------ ratings and details
    def _meta_batch(self, gen, batch):
        if gen != self.gen:
            return
        for path, data in batch:
            e = self.model.by_path.get(path)
            if e is not None:
                e.update(data)
                e["loaded"] = True
                self.model.changed(path)
        self._update_filter_choices()
        if self._filters_active() or self.sort.currentData() != "File name":
            self._filter_timer.start()
        self._update_count()
        self.review.meta_ready()

    def _meta_done(self, gen, n_read):
        if gen != self.gen:
            return
        self.meta_pending = False
        self.timings["meta"] = time.perf_counter() - self.timings.get("start", time.perf_counter())
        log.info("library %s: %d photos, listed in %.2f s, details in %.2f s (%d read from files)", self.folder,
                 len(self.model.all), self.timings.get("listed", 0), self.timings["meta"], n_read)
        self._update_filter_choices()
        self.apply_filter()

    def refresh_paths(self, paths_: list) -> None:
        """Read a few photos' sidecars again (after Develop or Paste settings changed them), in the background."""
        es = [dict(self.model.by_path[p]) for p in paths_ if p in self.model.by_path]
        if not es:
            return
        gen = self.gen

        def work():
            from ..engine.io.libindex import read_entry
            out = []
            for e in es:
                try:
                    e["side_mtime"] = os.path.getmtime(e["path"] + ".psb.json") if os.path.exists(
                        e["path"] + ".psb.json") else e.get("side_mtime", 0.0)
                except OSError:
                    pass
                e["side_mtime"] = e.get("side_mtime") or 1.0
                out.append((e["path"], read_entry(e)))
            return out
        run_bg(work, gen, lambda g, batch: self._meta_batch(g, batch))

    def set_flag(self, paths_: list, key: str) -> None:
        """Rate or flag photos ('0'-'5', 'p', 'x', 'u'): the grid updates at once, the sidecars are written in the
        background."""
        for p in paths_:
            e = self.model.by_path.get(p)
            if e is None:
                continue
            m = {"rating": int(e.get("rating") or 0), "pick": int(e.get("pick") or 0), "label": e.get("label", "")}
            if key in "012345":
                m["rating"] = int(key)
            elif key == "p":
                m["pick"] = 1
            elif key == "x":
                m["pick"] = -1
            elif key == "u":
                m["pick"] = 0
            e.update(m)
            self.model.changed(p)
            self.writer.save(p, m, e)
        self._update_count()
        if self._filters_active():
            self._filter_timer.start()

    def _save_failed(self, path, meta, err):
        log.warning("could not save rating for %s (read-only drive?): %s", path, err)
        folder = os.path.dirname(path)
        if folder in self.warned_folders:
            return
        self.warned_folders.add(folder)
        box = QMessageBox(self)
        box.setWindowTitle("Ratings not saved")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(f"Ratings for photos in this folder cannot be saved beside them:\n{folder}\n\n({err})\n\nThe drive "
                    f"may be read-only. Keep this folder's ratings and edits on this PC instead? (They are kept in the "
                    f"Studio's settings folder, never shared.)")
        keep = box.addButton("Keep them on this PC", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Not now", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is keep:
            from .. import engine as E
            E.use_central_sidecars(folder)
            self.central_act.setChecked(True)
            for p in self.files:
                if os.path.dirname(p) == folder:
                    e = self.model.by_path[p]
                    if e.get("rating") or e.get("pick"):
                        self.writer.save(p, {"rating": int(e.get("rating") or 0), "pick": int(e.get("pick") or 0),
                                             "label": e.get("label", "")}, e)

    def toggle_central(self):
        from .. import engine as E
        if not self.folder:
            self.central_act.setChecked(False)
            return
        on = not bool(E.central_folder_for(self.folder))
        if on:
            r = QMessageBox.question(self, "Keep ratings on this PC", "Keep the ratings and edits of photos in this "
                                     f"folder on this PC (in the Studio's settings folder) instead of beside the "
                                     f"photos?\n\n{self.folder}\n\nUseful for read-only drives. Sidecars already "
                                     f"there are copied in once.")
            if r != QMessageBox.StandardButton.Yes:
                self.central_act.setChecked(False)
                return
        E.use_central_sidecars(self.folder, on)
        self.central_act.setChecked(on)
        self.reload()

    # ------------------------------------------------------------------------------------ filter and sort
    def _filters_active(self) -> bool:
        return bool(self.filter.currentIndex() or self.cam_f.currentIndex() or self.lens_f.currentIndex() or
                    self.iso_f.currentIndex() or self.date_f.currentIndex() or self.unrated.isChecked() or
                    self.search.text().strip())

    def _reset_filter_choices(self):
        for c, first in ((self.cam_f, "Any camera"), (self.lens_f, "Any lens"), (self.date_f, "Any date")):
            c.blockSignals(True)
            cur = c.currentData()
            c.clear()
            c.addItem(first, "")
            if cur:
                c.addItem(cur, cur)
                c.setCurrentIndex(1)
            c.blockSignals(False)

    def _update_filter_choices(self):
        es = self.model.all
        for c, values in ((self.cam_f, sorted({e["camera"] for e in es if e.get("camera")})),
                          (self.lens_f, sorted({(e.get("lens_tag") or e.get("lens")) for e in es
                                                if e.get("lens_tag") or e.get("lens")})),
                          (self.date_f, sorted({e["taken"][:10] for e in es if e.get("taken")}, reverse=True))):
            have = {c.itemData(i) for i in range(c.count())}
            new = [v for v in values if v not in have]
            if new:
                c.blockSignals(True)
                cur = c.currentData()
                items = sorted(set(values) | (have - {""}), reverse=(c is self.date_f))
                c.clear()
                c.addItem(c.property("first") or {self.cam_f: "Any camera", self.lens_f: "Any lens",
                                                  self.date_f: "Any date"}[c], "")
                for v in items:
                    c.addItem(v, v)
                c.setCurrentIndex(max(0, c.findData(cur)))
                c.blockSignals(False)

    def passes(self, e: dict) -> bool:
        k = self.filter.currentIndex()
        rating, pick = int(e.get("rating") or 0), int(e.get("pick") or 0)
        if 1 <= k <= 5 and rating < k:
            return False
        if k == 6 and pick != 1:
            return False
        if k == 7 and pick == -1:
            return False
        if k == 8 and pick != -1:
            return False
        cam = self.cam_f.currentData()
        if cam and e.get("camera") != cam:
            return False
        lens = self.lens_f.currentData()
        if lens and (e.get("lens_tag") or e.get("lens")) != lens:
            return False
        lo, hi = self.iso_f.currentData() or (None, None)
        iso = e.get("iso")
        if (lo or hi) and (iso is None or (lo and iso < lo) or (hi and iso > hi)):
            return False
        d = self.date_f.currentData()
        if d and (e.get("taken") or "")[:10] != d:
            return False
        if self.unrated.isChecked() and (rating or pick):
            return False
        t = self.search.text().strip().lower()
        if t and t not in e["name"].lower():
            return False
        return True

    def _sort_key(self):
        s = self.sort.currentData()
        if s == "File name":
            return lambda e: (e["name"].lower(), e["path"].lower()), False
        if s == "Rating":
            return lambda e: (-(e.get("rating") or 0), -(e.get("pick") or 0), e.get("taken") or "", e["name"]), False
        if s == "ISO":
            return lambda e: (e.get("iso") or 0, e.get("taken") or "", e["name"]), False
        return (lambda e: (e.get("taken") or _ftime(e["mtime"]), e["name"].lower())), False

    def _sort_changed(self, _):
        self.settings.set("library_sort", self.sort.currentData())
        self.apply_filter()

    def apply_filter(self, *_):
        sel = set(self.selected_files())
        cur = self.current_path()
        key, rev = self._sort_key()
        rows = sorted((e for e in self.model.all if self.passes(e)), key=key, reverse=rev)
        if [e["path"] for e in rows] != self.files:
            self.model.set_rows(rows)
            self._restore_selection(sel, cur)
        self._update_count()
        self._schedule_thumbs()

    def _update_count(self):
        es = self.model.all
        total, shown = len(es), len(self.model.rows)
        picks = sum(1 for e in es if e.get("pick") == 1)
        rejects = sum(1 for e in es if e.get("pick") == -1)
        head = f"{shown} of {total} photos" if shown != total else f"{total} photo{'s' if total != 1 else ''}"
        self.count.setText(f"{head} · {picks} pick{'s' if picks != 1 else ''} · {rejects} reject"
                           f"{'s' if rejects != 1 else ''}" + (" · reading..." if self.meta_pending else ""))

    # ------------------------------------------------------------------------------------ selection
    def selected_files(self) -> list:
        rows = sorted(ix.row() for ix in self.grid.selectionModel().selectedIndexes())
        return [self.model.rows[r]["path"] for r in rows if r < len(self.model.rows)]

    def current_path(self) -> str:
        ix = self.grid.currentIndex()
        return ix.data(ROLE_PATH) if ix.isValid() else ""

    def select_path(self, path: str, clear: bool = True) -> bool:
        from PySide6.QtCore import QItemSelectionModel
        ix = self.model.index_of(path)
        if not ix.isValid():
            return False
        sm = self.grid.selectionModel()
        flags = QItemSelectionModel.SelectionFlag.Select
        if clear:
            flags |= QItemSelectionModel.SelectionFlag.Clear
        sm.setCurrentIndex(ix, flags)
        self.grid.scrollTo(ix)
        return True

    def _restore_selection(self, sel: set, cur: str):
        from PySide6.QtCore import QItemSelection, QItemSelectionModel
        sm = self.grid.selectionModel()
        s = QItemSelection()
        for p in sel:
            ix = self.model.index_of(p)
            if ix.isValid():
                s.select(ix, ix)
        sm.select(s, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        ix = self.model.index_of(cur)
        if ix.isValid():
            sm.setCurrentIndex(ix, QItemSelectionModel.SelectionFlag.NoUpdate)

    # ------------------------------------------------------------------------------------ keys, menus, drops
    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if obj is self.grid and ev.type() == QEvent.Type.KeyPress:
            if self._key(ev):
                return True
        return super().eventFilter(obj, ev)

    def _key(self, e) -> bool:
        k = e.key()
        if k == Qt.Key.Key_Space:
            self.open_review()
            return True
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            p = self.current_path()
            if p:
                self.open_photo.emit(p)
            return True
        t = e.text().lower()
        if t and t in "012345pxu" and not (e.modifiers() & (Qt.KeyboardModifier.ControlModifier |
                                                          Qt.KeyboardModifier.AltModifier)):
            files = self.selected_files()
            if files:
                self.set_flag(files, t)
            return True
        return False

    def keyPressEvent(self, e):
        if not self._key(e):
            super().keyPressEvent(e)

    def _context_menu(self, pos):
        ix = self.grid.indexAt(pos)
        if ix.isValid() and ix.data(ROLE_PATH) not in self.selected_files():
            self.select_path(ix.data(ROLE_PATH))
        files = self.selected_files()
        if not files:
            return
        m = QMenu(self)
        m.addAction("Open in Develop", lambda: self.open_photo.emit(files[0]))
        m.addAction("Review", self.open_review)
        m.addAction(f"Process {len(files)} selected...", lambda: self.process_picks("selected"))
        m.addAction("Export...", lambda: self.export_requested.emit(files))
        m.addSeparator()
        m.addAction("◎ Tag lens...", self.tag_selected)
        m.addAction("Paste settings", lambda: self.paste_requested.emit(files))
        m.addSeparator()
        m.addAction("Show in Explorer", lambda: show_in_explorer(files[0]))
        m.addAction("Copy path", lambda: QApplication.clipboard().setText("\n".join(files)))
        m.exec(self.grid.viewport().mapToGlobal(pos))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.isLocalFile() for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if os.path.isdir(p):
                self.load_folder(p)
                return
            if os.path.isfile(p):
                self.load_folder(os.path.dirname(p), select=p)
                return

    # ------------------------------------------------------------------------------------ Review
    def open_review(self):
        if not self.model.rows:
            return
        p = self.current_path() or (self.selected_files() or [self.model.rows[0]["path"]])[0]
        self.body.setCurrentIndex(1)
        self.review.start(p)
        self.hint.setText("Review: Left / Right move.  0-5 rate, P pick, X reject, U clear.  Z or click: 100%.  "
                          "C: compare two.  Enter: develop.  Space or Esc: back to the grid.")

    def close_review(self, path: str | None = None):
        self.body.setCurrentIndex(0)
        self.hint.setText("Double-click or Enter to develop.  Space: Review.  Keys: 0-5 rate, P pick, X reject, "
                          "U clear flag.  Ratings live in the sidecar beside each raw, never in a catalogue.")
        if path:
            self.select_path(path)
        self.grid.setFocus()
        self._schedule_thumbs()

    # ------------------------------------------------------------------------------------ tools
    def tag_selected(self):
        files = self.selected_files()
        if not files:
            QMessageBox.information(self, "Tag lens", "Select the photos taken with that lens first "
                                    "(click, Shift-click or Ctrl+A).")
            return
        from .. import engine as E
        current = E.load_edit(files[0]).lens.tag
        dlg = LensDialog(self, current, len(files))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            n = tag_lens(files, dlg.choice)
            self.settings.set("last_lens", dlg.choice)
            for f in files:
                e = self.model.by_path.get(f)
                if e is not None:
                    e["lens_tag"] = dlg.choice
                    self.model.changed(f)
            self._update_filter_choices()
            self.lens_tagged.emit(files, dlg.choice)
            QMessageBox.information(self, "Tag lens", f"{n} photo{'s' if n != 1 else ''} tagged: "
                                    f"{dlg.choice or 'camera record'}.")

    def measure_glow(self):
        files = self.selected_files()
        if len(files) < 3:
            QMessageBox.information(self, "Measure lens glow", GLOW_HOWTO)
            return
        from .. import engine as E
        lens = ""
        try:
            from ..engine.io.metadata import read_metadata
            lens = read_metadata(files[0]).lens or ""
        except Exception:                                    # noqa: BLE001  (a name is only a suggestion)
            lens = ""
        lens, ok = QInputDialog.getText(self, "Measure lens glow", "Which lens is this? (the name your photos "
                                        "will be matched by)", text=lens)
        if not ok:
            return

        def done(_, rep):
            if rep.get("ok"):
                msg = (f"Glow profile '{rep['name']}' saved: {rep['frames']} frames, {rep['stops']:.0f} stops "
                       f"measured, glow holds {100 * max(rep['wing']):.1f}% of the light.\n\nIn Develop pick "
                       f"Light > Clipped highlights: measure from lens glow.")
            else:
                msg = f"No profile made: {rep.get('reason', 'unknown problem')}."
            if rep.get("skipped"):
                msg += "\n\nSkipped: " + "; ".join(f"{os.path.basename(p)} ({why})" for p, why in rep["skipped"][:8])
            QMessageBox.information(self, "Measure lens glow", msg)
        run_bg(lambda: E.measure_glow_profile(files, lens.strip()), "glow", done,
               lambda _, err: QMessageBox.warning(self, "Measure lens glow", f"Problem: {err}"), keep=True)

    def learn_dust(self):
        files = self.selected_files()
        if len(files) < 3:
            QMessageBox.information(self, "Learn dust map", "Select at least 3 photos (5-20 is better) from one "
                                    "camera, of different scenes. Dust shows best in plain sky or a plain wall shot "
                                    "at f/11 or smaller; frames of the same scene (a focus stack) count once.")
            return
        from .. import engine as E
        self.dust_files = files

        def done(_, rep):
            if rep.get("ok"):
                msg = (f"Dust map for {rep['camera']}: {rep['spots']} spot{'s' if rep['spots'] != 1 else ''} found "
                       f"in the same place in at least {rep['min_seen']} of {rep['photos']} photos.\n\nSaved on this "
                       f"PC only (calibration_local). In Develop, Retouch > Dust map heals them in any photo.")
            else:
                msg = f"No dust map made: {rep.get('reason', 'unknown problem')}."
            if rep.get("skipped"):
                msg += "\n\nSkipped: " + "; ".join(f"{os.path.basename(p)} ({why})" for p, why in rep["skipped"][:8])
            QMessageBox.information(self, "Learn dust map", msg)
        QMessageBox.information(self, "Learn dust map", f"Learning from {len(files)} photos in the background "
                                "(about 2-4 seconds each). A message appears when it is done.")
        run_bg(lambda: E.learn_dust_map(files), "dust", done,
               lambda _, err: QMessageBox.warning(self, "Learn dust map", f"Problem: {err}"), keep=True)

    # ------------------------------------------------------------------------------------ import from card
    def import_from_card(self, card=None):
        from .importer import open_import
        self.notice.hide()
        open_import(self, card)

    def imported(self, folder: str):
        """The import dialog's Open in Library: show the new folder with all photos visible."""
        self.filter.setCurrentIndex(0)
        for c in (self.cam_f, self.lens_f, self.iso_f, self.date_f):
            c.setCurrentIndex(0)
        self.unrated.setChecked(False)
        self.search.clear()
        self.load_folder(folder)
        w = self.window()
        if hasattr(w, "nav"):
            w.nav.setCurrentRow(0)

    def start_card_watch(self):
        from ..engine.io.ingest import CardWatcher
        self._cards = CardWatcher()
        self._card_timer = QTimer(self, interval=2000, timeout=self._poll_cards)
        self._card_timer.start()
        self._polling = False

    def _poll_cards(self):
        if self._polling or self._cards is None:
            return
        self._polling = True
        watcher = self._cards
        opts = self._import_options()

        def work():
            new = watcher.poll()
            out = []
            for c in new:
                from ..engine.io import ingest as I
                sc = I.scan_source(c.root)
                if opts is not None:
                    counts = I.summary_counts(sc, opts)
                else:
                    counts = {"photos": sc.photos, "new": None}
                out.append((c, counts))
            return out

        def done(_, found):
            self._polling = False
            if found:
                c, counts = found[-1]
                self._notice_card = c
                new = f", {counts['new']} new" if counts.get("new") is not None else ""
                n = counts["photos"]
                self.notice_text.setText(f"Card found: {c.title} · {n} photo{'s' if n != 1 else ''}{new}.")
                self.notice.show()
        run_bg(work, "cards", done, lambda *_: setattr(self, "_polling", False))

    def _import_options(self):
        d = self.photos_dir()
        if not d:
            return None
        from ..engine.io.ingest import ImportOptions
        s = self.settings
        return ImportOptions(dest=d, raws=bool(s.get("import_raws", True)), jpegs=bool(s.get("import_jpegs", False)),
                             videos=bool(s.get("import_videos", False)), by_year=bool(s.get("import_by_year", True)),
                             day_start=str(s.get("import_day_start", "04:00")))

    # ------------------------------------------------------------------------------------ Process picks
    def process_picks(self, which: str | None = None):
        from .process import ProcessDialog, run_process
        if not self.model.all:
            QMessageBox.information(self, "Process picks", "Open a folder of photos first.")
            return
        self.save_open_edit()
        sel = self.selected_files()
        picks = [e["path"] for e in self.model.rows if e.get("pick") == 1]
        dlg = ProcessDialog(self.settings, picks, sel, self.files, which, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        files, spec, dest, mode = dlg.result_files(), dlg.spec(), dlg.dest.text(), dlg.mode()
        if not files:
            return
        run_process(self, files, spec, dest, mode)


def _ftime(mtime: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
    except (OverflowError, OSError, ValueError):
        return ""


def show_in_explorer(path: str) -> None:
    """Open the folder with the file selected (Windows), or just the folder elsewhere."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))
    except OSError as e:
        log.warning("show in explorer failed: %s", e)
