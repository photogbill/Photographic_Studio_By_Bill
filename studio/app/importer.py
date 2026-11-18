"""Import from card (plan 24.3-24.5): the Photos folder dialog and the one-screen import dialog. The copying itself
is the engine's (studio/engine/io/ingest.py); this file only shows what the dry run decided and runs the import in
the background, so the Studio stays usable while it copies."""
from __future__ import annotations

import os
import threading

from PySide6.QtCore import QObject, Qt, QTime, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLineEdit, QProgressBar, QPushButton, QScrollArea, QStackedWidget, QTimeEdit,
                               QVBoxLayout, QWidget)

from . import theme as T
from .common import label, log, run_bg
from ..engine.io import ingest as I

RAW_SIZE_TEXT = "An external drive is fine. Raw files are large: about 20 to 60 MB each."


def _coloured(lb, colour):
    lb.setStyleSheet(f"color:{colour};")
    return lb


# ============================================================================================ the Photos folder
class PhotosFolderDialog(QDialog):
    """'Where should your photos be kept?' (24.3). Nothing is created or saved until the photographer confirms.
    ``for_this_import`` picks a folder for one import only (the Photos folder setting is left alone)."""

    def __init__(self, settings, parent=None, for_this_import: bool = False):
        super().__init__(parent)
        self.settings = settings
        self.for_this_import = for_this_import
        self.chosen = ""
        self.setWindowTitle("Photos folder" if not for_this_import else "Folder for this import")
        self.setMinimumWidth(560)
        v = QVBoxLayout(self)
        v.addWidget(label("Where should your photos be kept?" if not for_this_import
                          else "Where should this import go?", "h2"))
        v.addWidget(label("Import from card copies new photos here, into a folder for each shooting day.", "dim",
                          wrap=True))
        current = str(settings.get("photos_dir") or "")
        self.path = QLineEdit(current or I.default_photos_dir())
        self.path.textChanged.connect(self._refresh)
        row = QHBoxLayout()
        row.addWidget(self.path, 1)
        choose = QPushButton("Choose a folder...")
        choose.clicked.connect(self._choose)
        row.addWidget(choose)
        v.addLayout(row)
        drives = [d for d in I.list_drives() if not d.card]
        if drives:
            v.addWidget(label("Or put it on a drive:", "dim"))
            grid = QGridLayout()
            for i, d in enumerate(drives[:8]):
                b = QPushButton(f"{d.title}  ·  {I.human_bytes(d.free)} free")
                b.setToolTip(os.path.join(d.root, "Photos"))
                b.clicked.connect(lambda _=False, r=d.root: self.path.setText(os.path.join(r, "Photos")))
                grid.addWidget(b, i // 2, i % 2)
            v.addLayout(grid)
        v.addWidget(label(RAW_SIZE_TEXT, "dim", wrap=True))
        self.free = label("", None)
        v.addWidget(self.free)
        self.warn = _coloured(label("", None, wrap=True), T.AMBER)
        v.addWidget(self.warn)
        if current and not for_this_import:
            v.addWidget(label("Changing the Photos folder never moves or copies the photos already in the old one.",
                              "dim", wrap=True))
        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.ok = QPushButton("Use this folder")
        self.ok.setObjectName("primary")
        self.ok.clicked.connect(self._ok)
        btns.addWidget(cancel)
        btns.addWidget(self.ok)
        v.addLayout(btns)
        self._refresh()

    def _choose(self):
        d = QFileDialog.getExistingDirectory(self, "Where should your photos be kept?", self.path.text())
        if d:
            self.path.setText(d)

    def _refresh(self, *_):
        p = self.path.text().strip()
        if not p:
            self.free.setText("")
            self.ok.setEnabled(False)
            return
        if not I.drive_connected(p):
            self.free.setText(f"The drive {I.drive_of(p)} is not connected.")
            self.free.setStyleSheet(f"color:{T.RED};")
            self.ok.setEnabled(False)
        else:
            free, total = I.free_space(p)
            self.free.setText(f"{I.human_bytes(free)} free on {I.drive_of(p)}" if free is not None else "")
            self.free.setStyleSheet(f"color:{T.CYAN};")
            self.ok.setEnabled(True)
        self.warn.setText("This folder is inside OneDrive: raw files will fill your cloud storage quickly. It works, "
                          "but an external drive or a folder outside OneDrive is usually better."
                          if I.inside_onedrive(p) else "")

    def _ok(self):
        p = os.path.abspath(self.path.text().strip())
        try:
            os.makedirs(p, exist_ok=True)
        except OSError as e:
            self.free.setText(f"Could not use that folder: {e.strerror or e}")
            self.free.setStyleSheet(f"color:{T.RED};")
            return
        self.chosen = p
        if not self.for_this_import:
            self.settings.set("photos_dir", p)
            log.info("photos folder set: %s", p)
        self.accept()


# ============================================================================================ the import dialog
class _Sig(QObject):
    progress = Signal(dict)
    times = Signal(int, int)


class ImportDialog(QDialog):
    """Import from card, one screen (24.5). Non-modal: the Studio stays usable while it copies."""

    def __init__(self, page, card=None):
        super().__init__(page)
        self.page = page
        self.settings = page.settings
        self.setWindowTitle("Import from card")
        self.setMinimumWidth(720)
        self.setModal(False)
        self.scan: I.Scan | None = None
        self.plan: I.ImportPlan | None = None
        self.source = ""
        self.cards: list = []
        self.records: dict = {}
        self.day_rows: dict = {}
        self.names: dict = {}
        self.dest_override = ""
        self.cancel = threading.Event()
        self.running = False
        self.report = None
        self.scan_token = 0
        self.left_out: dict = {}                       # source rel (lower) -> reason, from Sort focus stacks
        self.sorting = False
        self.sort_token = 0
        self.sig = _Sig(self)
        self.sig.progress.connect(self._progress)
        v = QVBoxLayout(self)
        self.pages = QStackedWidget()
        v.addWidget(self.pages)
        self.pages.addWidget(self._build_setup())
        self.pages.addWidget(self._build_progress())
        self.pages.addWidget(self._build_summary())
        self._want_card = card
        self.timer = QTimer(self, interval=2000, timeout=self._refresh_cards)
        self.timer.start()
        self._refresh_cards()

    # ------------------------------------------------------------------------------------ setup page
    def _build_setup(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.addWidget(label("Import from card", "h2"))
        g = QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(8)
        g.setColumnStretch(1, 1)
        r = 0
        g.addWidget(label("Card", "dim"), r, 0, Qt.AlignmentFlag.AlignTop)
        crow = QHBoxLayout()
        self.card_c = QComboBox()
        self.card_c.currentIndexChanged.connect(self._card_picked)
        folder_b = QPushButton("Choose a folder instead...")
        folder_b.setToolTip("Import from any folder (a card copied earlier, a friend's drive): same rules")
        folder_b.clicked.connect(self._choose_source)
        crow.addWidget(self.card_c, 1)
        crow.addWidget(folder_b)
        g.addLayout(crow, r, 1)
        r += 1
        self.card_note = label("", "dim", wrap=True)
        g.addWidget(self.card_note, r, 1)
        r += 1
        g.addWidget(label("Copy", "dim"), r, 0, Qt.AlignmentFlag.AlignTop)
        tick = QHBoxLayout()
        tick.setSpacing(18)
        self.raws = QCheckBox("Raw files")
        self.jpegs = QCheckBox("JPEGs")
        self.videos = QCheckBox("Videos")
        self.raws.setChecked(bool(self.settings.get("import_raws", True)))
        self.jpegs.setChecked(bool(self.settings.get("import_jpegs", False)))
        self.videos.setChecked(bool(self.settings.get("import_videos", False)))
        for b in (self.raws, self.jpegs, self.videos):
            tick.addWidget(b)
            b.toggled.connect(self._ticks_changed)
        tick.addStretch(1)
        g.addLayout(tick, r, 1)
        r += 1
        self.left_lbl = _coloured(label("", None, wrap=True), T.AMBER)
        g.addWidget(self.left_lbl, r, 1)
        r += 1
        self.noraw = QFrame()
        nh = QHBoxLayout(self.noraw)
        nh.setContentsMargins(0, 0, 0, 0)
        nh.addWidget(_coloured(label("This card has no raw files, only JPEGs.", None), T.AMBER))
        nb = QPushButton("Copy the JPEGs")
        nb.clicked.connect(lambda: self.jpegs.setChecked(True))
        nh.addWidget(nb)
        nh.addStretch(1)
        self.noraw.hide()
        g.addWidget(self.noraw, r, 1)
        r += 1
        g.addWidget(label("Into", "dim"), r, 0)
        irow = QHBoxLayout()
        self.dest_lbl = label("", None)
        change = QPushButton("Change...")
        change.setToolTip("Change your Photos folder (the photos already there are never moved)")
        change.clicked.connect(self._change_dest)
        self.space_lbl = label("", "dim")
        irow.addWidget(self.dest_lbl, 1)
        irow.addWidget(change)
        irow.addWidget(self.space_lbl)
        g.addLayout(irow, r, 1)
        r += 1
        self.dest_problem = QFrame()
        dh = QHBoxLayout(self.dest_problem)
        dh.setContentsMargins(0, 0, 0, 0)
        self.dest_problem_lbl = _coloured(label("", None, wrap=True), T.RED)
        retry = QPushButton("Try again")
        retry.clicked.connect(self.replan)
        other = QPushButton("Use another folder for this import only")
        other.clicked.connect(self._other_dest)
        dh.addWidget(self.dest_problem_lbl, 1)
        dh.addWidget(retry)
        dh.addWidget(other)
        self.dest_problem.hide()
        g.addWidget(self.dest_problem, r, 1)
        r += 1
        g.addWidget(label("Folders", "dim"), r, 0, Qt.AlignmentFlag.AlignTop)
        fbox = QVBoxLayout()
        self.days_w = QWidget()
        self.days_l = QGridLayout(self.days_w)
        self.days_l.setContentsMargins(0, 0, 0, 0)
        self.days_l.setColumnStretch(4, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.days_w)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.days_scroll = scroll
        fbox.addWidget(scroll)
        orow = QHBoxLayout()
        self.one = QCheckBox("Put everything in one folder")
        self.one.toggled.connect(self._one_toggled)
        self.one_name = QLineEdit()
        self.one_name.textEdited.connect(lambda t: (self.names.__setitem__("*", t), self.replan()))
        self.one_name.hide()
        orow.addWidget(self.one)
        orow.addWidget(self.one_name, 1)
        fbox.addLayout(orow)
        self.saved_as = label("", None, wrap=True)
        self.saved_as.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        fbox.addWidget(self.saved_as)
        self.warn_lbl = _coloured(label("", None, wrap=True), T.AMBER)
        fbox.addWidget(self.warn_lbl)
        opts = QHBoxLayout()
        self.before = QCheckBox("Show photos imported before")
        self.before.setToolTip("Copy photos that were imported once already (for example, ones you deleted since)")
        self.before.toggled.connect(self.replan)
        self.by_year = QCheckBox("Group folders by year")
        self.by_year.setChecked(bool(self.settings.get("import_by_year", True)))
        self.by_year.toggled.connect(self._folder_opts)
        self.day_start = QTimeEdit()
        self.day_start.setDisplayFormat("HH:mm")
        hh, mm = divmod(I.day_start_minutes(self.settings.get("import_day_start", "04:00")), 60)
        self.day_start.setTime(QTime(hh, mm))
        self.day_start.setToolTip("Photos taken before this time count as the day before, so a night shoot that "
                                  "runs past midnight stays in one folder")
        self.day_start.timeChanged.connect(self._folder_opts)
        opts.addWidget(self.before)
        opts.addStretch(1)
        opts.addWidget(self.by_year)
        opts.addSpacing(16)
        opts.addWidget(label("A new day starts at", "dim"))
        opts.addWidget(self.day_start)
        fbox.addLayout(opts)
        srow2 = QHBoxLayout()
        self.sort_c = QCheckBox("Sort focus stacks")
        self.sort_c.setChecked(bool(self.settings.get("import_sort_stacks", False)))
        self.sort_c.setToolTip("Finds the focus stacks on the card and copies the frames that would hurt a stack "
                               "(soft everywhere, adds nothing, shaken...) into a \"Left out\" folder inside that "
                               "day's folder. Everything is still copied; bursts and single shots are not touched.\n"
                               "Best for: cards of hand-held focus stacks. Weak at: it reads every raw's preview "
                               "first (about a minute per 250 photos).")
        self.sort_c.toggled.connect(self._sort_toggled)
        self.sort_lbl = label("", "dim", wrap=True)
        srow2.addWidget(self.sort_c)
        srow2.addWidget(self.sort_lbl, 1)
        fbox.addLayout(srow2)
        g.addLayout(fbox, r, 1)
        r += 1
        g.addWidget(label("Second copy", "dim"), r, 0)
        srow = QHBoxLayout()
        self.second_on = QCheckBox("Also copy to a second drive")
        self.second_on.setChecked(bool(self.settings.get("import_second_on", False)))
        self.second_on.toggled.connect(self._second_changed)
        self.second = QLineEdit(str(self.settings.get("import_second_dir", "") or ""))
        self.second.setPlaceholderText("e.g. F:\\Photos backup")
        self.second.editingFinished.connect(self._second_changed)
        sb = QPushButton("Choose...")
        sb.clicked.connect(self._choose_second)
        srow.addWidget(self.second_on)
        srow.addWidget(self.second, 1)
        srow.addWidget(sb)
        g.addLayout(srow, r, 1)
        r += 1
        self.second_space = label("", "dim")
        g.addWidget(self.second_space, r, 1)
        v.addLayout(g)
        v.addStretch(1)
        btns = QHBoxLayout()
        self.reason = label("", "dim", wrap=True)
        btns.addWidget(self.reason, 1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.close)
        self.go = QPushButton("Import")
        self.go.setObjectName("primary")
        self.go.setDefault(True)
        self.go.clicked.connect(self.start)
        btns.addWidget(cancel)
        btns.addWidget(self.go)
        v.addLayout(btns)
        return w

    # ------------------------------------------------------------------------------------ progress and summary
    def _build_progress(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(label("Importing", "h2"))
        self.p_count = label("", None)
        self.p_bar = QProgressBar()
        self.p_bar.setRange(0, 1000)
        self.p_bytes = label("", "dim")
        self.p_file = label("", "dim")
        v.addWidget(self.p_count)
        v.addWidget(self.p_bar)
        v.addWidget(self.p_bytes)
        v.addWidget(self.p_file)
        v.addWidget(label("You can keep working in the Studio while this runs. Every file is checked after it is "
                          "copied; the card is only ever read.", "dim", wrap=True))
        v.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.p_cancel = QPushButton("Cancel")
        self.p_cancel.clicked.connect(self._cancel)
        row.addWidget(self.p_cancel)
        v.addLayout(row)
        return w

    def _build_summary(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(label("Import finished", "h2"))
        self.s_lines = label("", None, wrap=True)
        self.s_lines.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.s_lines.setTextFormat(Qt.TextFormat.RichText)
        self.s_lines.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.s_lines)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        v.addWidget(sc, 1)
        self.s_safe = _coloured(label("", None, wrap=True), T.GREEN)
        v.addWidget(self.s_safe)
        row = QHBoxLayout()
        row.addStretch(1)
        logb = QPushButton("View log")
        logb.clicked.connect(lambda: self.report and QDesktopServices.openUrl(QUrl.fromLocalFile(self.report.log_path)))
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        self.s_stacks = QPushButton("Find focus stacks")
        self.s_stacks.setToolTip("Opens Card to Stacks on what was just copied: every focus stack found, the frames "
                                 "that would hurt it left out with a reason, ready to stack.")
        self.s_stacks.clicked.connect(self._find_stacks)
        row.addWidget(self.s_stacks)
        self.s_open = QPushButton("Open in Library")
        self.s_open.setObjectName("primary")
        self.s_open.setDefault(True)
        self.s_open.clicked.connect(self._open_in_library)
        row.addWidget(logb)
        row.addWidget(close)
        row.addWidget(self.s_open)
        v.addLayout(row)
        return w

    # ------------------------------------------------------------------------------------ cards
    def _refresh_cards(self):
        if self.running or self.pages.currentIndex() != 0:
            return

        def done(_, cards):
            roots = [c.root for c in cards]
            old = [c.root for c in self.cards]
            if roots == old and self.card_c.count():
                return
            self.cards = cards
            cur = self.source
            self.card_c.blockSignals(True)
            self.card_c.clear()
            for c in cards:
                self.card_c.addItem(c.title, c.root)
            if self.source and self.source not in roots and not self._source_is_card():
                self.card_c.addItem(f"Folder: {self.source}", self.source)
            want = self._want_card.root if self._want_card is not None else cur
            self._want_card = None
            i = self.card_c.findData(want) if want else (0 if cards else -1)
            self.card_c.setCurrentIndex(max(0, i) if self.card_c.count() else -1)
            self.card_c.blockSignals(False)
            if not self.card_c.count():
                self.source, self.scan, self.plan = "", None, None
                self.card_note.setText("No card found. Put the card in a card reader (a camera plugged in by its "
                                       "cable is not supported yet), or choose a folder instead.")
                self._update_ui()
            elif self.card_c.currentData() != self.source:
                self._card_picked()
        run_bg(I.find_cards, "find-cards", done)

    def _source_is_card(self) -> bool:
        return os.path.isdir(os.path.join(self.source, "DCIM")) if self.source else False

    def _card_picked(self, *_):
        root = self.card_c.currentData()
        if not root or root == self.source and self.scan is not None:
            return
        self.set_source(root)

    def _choose_source(self):
        d = QFileDialog.getExistingDirectory(self, "Import from which folder?", self.source or "")
        if d:
            if self.card_c.findData(d) < 0:
                self.card_c.blockSignals(True)
                self.card_c.addItem(f"Folder: {d}", d)
                self.card_c.blockSignals(False)
            self.card_c.setCurrentIndex(self.card_c.findData(d))
            self.set_source(d)

    def set_source(self, root: str):
        """Look at a card or folder: the listing first (fast), then the capture times from the photos."""
        self.source = root
        self.scan, self.plan = None, None
        self.left_out, self.sorting = {}, False
        self.sort_token += 1
        self.sort_lbl.setText("")
        self.scan_token += 1
        token = self.scan_token
        self.card_note.setText("Looking at the card...")
        self._update_ui()
        videos = self.videos.isChecked()

        def listed(_, sc):
            if token != self.scan_token:
                return
            self.scan = sc
            n = sc.photos
            self.card_note.setText(f"{n} photo{'s' if n != 1 else ''} ({sc.count(I.RAW)} raw, {sc.count(I.JPEG)} "
                                   f"JPEG), {sc.count(I.VIDEO)} video{'s' if sc.count(I.VIDEO) != 1 else ''}. Reading "
                                   f"capture times...")
            self.replan()

            def confirm():
                I.confirm_capture_times(sc, cancel=lambda: token != self.scan_token)
                return sc

            def confirmed(_, sc2):
                if token != self.scan_token:
                    return
                self.card_note.setText(f"{n} photo{'s' if n != 1 else ''} ({sc2.count(I.RAW)} raw, "
                                       f"{sc2.count(I.JPEG)} JPEG), {sc2.count(I.VIDEO)} "
                                       f"video{'s' if sc2.count(I.VIDEO) != 1 else ''}."
                                       + (f" {len(sc2.errors)} could not be read." if sc2.errors else ""))
                self.replan()
                if self.sort_c.isChecked():
                    self._sort_stacks()
            run_bg(confirm, "confirm-times", confirmed)

        def failed(_, err):
            if token == self.scan_token:
                self.card_note.setText(f"Could not read the card: {err}")
        run_bg(lambda: I.scan_source(root, videos=videos), "scan", listed, failed)

    # ------------------------------------------------------------------------------------ options
    def dest(self) -> str:
        return self.dest_override or str(self.settings.get("photos_dir") or "")

    def options(self) -> I.ImportOptions:
        names = {k: row[3].text() for k, row in self.day_rows.items()}
        names.update({k: v for k, v in self.names.items() if k == "*"})
        skip = [k for k, row in self.day_rows.items() if not row[0].isChecked()]
        t = self.day_start.time()
        return I.ImportOptions(dest=self.dest(), raws=self.raws.isChecked(), jpegs=self.jpegs.isChecked(),
                               videos=self.videos.isChecked(), by_year=self.by_year.isChecked(),
                               day_start=f"{t.hour():02d}:{t.minute():02d}", one_folder=self.one.isChecked(),
                               names=names, skip_days=skip,
                               second=self.second.text().strip() if self.second_on.isChecked() else "",
                               show_imported_before=self.before.isChecked(),
                               left_out=dict(self.left_out) if self.sort_c.isChecked() else {})

    # ------------------------------------------------------------------------------------ Sort focus stacks
    def _sort_toggled(self, on):
        self.settings.set("import_sort_stacks", bool(on))
        if on and self.scan is not None and not self.left_out and not self.sorting:
            self._sort_stacks()
        else:
            self.sort_lbl.setText(self._sort_text() if on else "")
            self.replan()

    def _sort_text(self) -> str:
        n = len(self.left_out)
        return (f"{n} frame{'s' if n != 1 else ''} left out of focus stacks will go into \"{I.LEFT_OUT_DIR}\" "
                f"folders." if n else "No frames to leave out.")

    def _sort_stacks(self):
        """Read every raw's preview on the card and find the frames a focus stack would leave out (background)."""
        if self.scan is None:
            return
        from ..engine.stack import sorter as SO
        raws = [f for f in self.scan.files if f.kind == I.RAW]
        if not raws:
            self.sort_lbl.setText("No raw files to sort.")
            return
        self.sort_token += 1
        token = self.sort_token
        self.sorting = True
        by_path = {os.path.normcase(f.path): f.rel.lower() for f in raws}
        self.sort_lbl.setText(f"Sorting focus stacks: reading {len(raws)} photos...")
        self._update_ui()
        stop = threading.Event()
        self._sort_stop = stop

        def work():
            def prog(fr, msg=""):
                if token != self.sort_token:
                    stop.set()
            return SO.left_out_map([f.path for f in raws], progress=prog, cancel=stop)

        def done(_, mp):
            if token != self.sort_token:
                return
            self.sorting = False
            self.left_out = {by_path[os.path.normcase(p)]: why for p, why in mp.items()
                             if os.path.normcase(p) in by_path}
            self.sort_lbl.setText(self._sort_text())
            self.replan()

        def failed(_, err):
            if token != self.sort_token:
                return
            self.sorting = False
            self.left_out = {}
            self.sort_lbl.setText(f"Sorting focus stacks did not work ({err}); everything is copied as usual.")
            self.replan()
        run_bg(work, "sort-stacks", done, failed)

    def _record(self, dest: str):
        if not dest:
            return None
        k = os.path.normcase(os.path.abspath(dest))
        if k not in self.records:
            self.records[k] = I.ImportRecord(I.record_path(os.path.abspath(dest)))
        return self.records[k]

    def _ticks_changed(self, *_):
        self.settings.set("import_raws", self.raws.isChecked())
        self.settings.set("import_jpegs", self.jpegs.isChecked())
        self.settings.set("import_videos", self.videos.isChecked())
        if self.sender() is self.videos and self.source and self.scan is not None and self.videos.isChecked() \
                and not self.scan.videos_outside:
            self.set_source(self.source)                # look in the brand video folders outside DCIM too
            return
        self.replan()

    def _folder_opts(self, *_):
        t = self.day_start.time()
        self.settings.set("import_by_year", self.by_year.isChecked())
        self.settings.set("import_day_start", f"{t.hour():02d}:{t.minute():02d}")
        self.replan()

    def _one_toggled(self, on):
        self.one_name.setVisible(on)
        if on and not self.one_name.text():
            first = next((k for k, row in self.day_rows.items() if row[0].isChecked()), "")
            self.one_name.setText(self.day_rows[first][3].text() if first else "")
            self.names["*"] = self.one_name.text()
        for row in self.day_rows.values():
            row[3].setVisible(not on)
            row[2].setVisible(not on)
        self.replan()

    def _second_changed(self, *_):
        self.settings.set("import_second_on", self.second_on.isChecked())
        self.settings.set("import_second_dir", self.second.text().strip())
        self.replan()

    def _choose_second(self):
        d = QFileDialog.getExistingDirectory(self, "Second copy: which folder?", self.second.text())
        if d:
            self.second.setText(d)
            self.second_on.setChecked(True)
            self._second_changed()

    def _change_dest(self):
        dlg = PhotosFolderDialog(self.settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.dest_override = ""
            self.replan()

    def _other_dest(self):
        dlg = PhotosFolderDialog(self.settings, self, for_this_import=True)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.dest_override = dlg.chosen
            self.replan()

    def ensure_photos_folder(self) -> bool:
        if self.dest():
            return True
        dlg = PhotosFolderDialog(self.settings, self)
        return dlg.exec() == QDialog.DialogCode.Accepted

    # ------------------------------------------------------------------------------------ the plan, shown
    def replan(self, *_):
        if self.scan is None:
            self.plan = None
            self._update_ui()
            return
        o = self.options()
        self.plan = I.plan_import(self.scan, o, self._record(o.dest))
        self._update_ui()

    def _sync_day_rows(self):
        keys = [g.key for g in self.plan.groups] if self.plan else []
        if keys == list(self.day_rows):
            return
        while self.days_l.count():
            it = self.days_l.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        old = self.day_rows
        self.day_rows = {}
        for i, k in enumerate(keys):
            tick = QCheckBox(k if k != I.UNDATED else "No date")
            prev = old.get(k)
            tick.setChecked(prev[0].isChecked() if prev else True)
            tick.toggled.connect(self.replan)
            new = label("", "dim")
            arrow = label("→", "dim")
            name = QLineEdit(prev[3].text() if prev else I.default_name(k))
            name.setToolTip("The folder name: keep the date in front (folders then sort by time), type after it, or "
                            "replace it")
            name.textEdited.connect(self.replan)
            name.setVisible(not self.one.isChecked())
            arrow.setVisible(not self.one.isChecked())
            self.days_l.addWidget(tick, i, 0)
            self.days_l.addWidget(new, i, 1)
            self.days_l.addWidget(arrow, i, 2)
            self.days_l.addWidget(name, i, 3, 1, 2)
            self.day_rows[k] = (tick, new, arrow, name)
        self.days_scroll.setFixedHeight(min(5, max(1, len(keys))) * 40 + 6)

    def _update_ui(self):
        p = self.plan
        dest = self.dest()
        self.dest_lbl.setText(dest or "(no Photos folder yet)")
        self.dest_lbl.setToolTip(dest)
        self._sync_day_rows()
        problem = ""
        if p is None:
            self.left_lbl.setText("")
            self.noraw.hide()
            self.space_lbl.setText("")
            self.saved_as.setText("")
            self.warn_lbl.setText("")
            self.go.setText("Import")
            self.go.setEnabled(False)
            self.reason.setText("")
            self.dest_problem.hide()
            return
        for g in p.groups:
            row = self.day_rows.get(g.key)
            if row:
                row[1].setText(f"{g.new} new" + (f" of {g.total}" if g.total != g.new else ""))
        self.left_lbl.setText(p.left_text("the card" if self._source_is_card() else "the source folder"))
        self.noraw.setVisible(p.no_raws and not self.jpegs.isChecked())
        need = p.bytes_needed
        if p.free is not None:
            txt = f"{I.human_bytes(p.free)} free  ·  this import needs {I.human_bytes(need)}"
            if not p.fits and need:
                txt += f"  ·  {I.human_bytes(p.shortfall)} short (2 GB or 5% is kept free)"
                self.space_lbl.setStyleSheet(f"color:{T.RED};")
            else:
                self.space_lbl.setStyleSheet(f"color:{T.DIM};")
            self.space_lbl.setText(txt)
        else:
            self.space_lbl.setText("")
        if dest and not I.drive_connected(dest):
            self.dest_problem_lbl.setText(f"Your Photos folder is on {I.drive_of(dest)}, which is not connected.")
            self.dest_problem.show()
        else:
            self.dest_problem.hide()
        folders = []
        for g in p.groups:
            if g.chosen and g.folder and g.folder not in folders and (g.new or not p.to_copy):
                folders.append(g.folder)
        if folders:
            more = f"  (and {len(folders) - 1} more folder{'s' if len(folders) > 2 else ''})" if len(folders) > 1 else ""
            self.saved_as.setText(f"Saved as:  {folders[0]}{os.sep}{more}")
        else:
            self.saved_as.setText("")
        if self.scan is not None and self.card_c.currentIndex() >= 0:
            i = self.card_c.currentIndex()
            base = next((c.title for c in self.cards if c.root == self.source), f"Folder: {self.source}")
            self.card_c.setItemText(i, f"{base}  ·  {self.scan.photos} photo{'s' if self.scan.photos != 1 else ''}"
                                       f"  ·  {len(p.to_copy)} new")
        warns = list(p.warnings)
        if p.imported_before and not self.before.isChecked():
            warns.insert(0, f"{p.imported_before} photo{'s were' if p.imported_before != 1 else ' was'} imported "
                            f"before and will not be copied again.")
        renamed = sum(1 for it in p.to_copy if it.renamed)
        if renamed:
            warns.append(f"{renamed} photo{'s have' if renamed != 1 else ' has'} the same name as a different photo "
                         f"already there: kept under a new name (_2), never overwritten.")
        self.warn_lbl.setText("\n".join(warns))
        if p.options.second:
            if p.second_free is None:
                self.second_space.setText(f"The drive {I.drive_of(p.options.second)} is not connected.")
                self.second_space.setStyleSheet(f"color:{T.RED};")
            else:
                ok = p.second_fits
                self.second_space.setText(f"{I.human_bytes(p.second_free)} free on the second drive"
                                          + ("" if ok else f"  ·  {I.human_bytes(p.second_shortfall)} short"))
                self.second_space.setStyleSheet(f"color:{T.DIM if ok else T.RED};")
        else:
            self.second_space.setText("")
        n = len(p.to_copy)
        self.go.setText(f"Import {n} photo{'s' if n != 1 else ''}" if n else "Import")
        if p.problem:
            problem = p.problem
        elif not n:
            problem = "Nothing new to import." if p.groups or p.imported_before else "Nothing to import with these ticks."
        elif not p.fits:
            problem = "Not enough free space on the destination."
        elif not p.second_fits:
            problem = "Not enough free space for the second copy."
        if self.sorting and self.sort_c.isChecked() and not problem:
            problem = "Sorting focus stacks first (untick Sort focus stacks to import without it)."
            self.go.setEnabled(False)
        else:
            self.go.setEnabled(p.can_start)
        self.reason.setText(problem)
        self.reason.setStyleSheet(f"color:{T.AMBER};" if self.sorting and self.sort_c.isChecked() else
                                  (f"color:{T.RED};" if problem and n else ""))

    # ------------------------------------------------------------------------------------ running
    def start(self):
        if self.plan is None or not self.plan.can_start or self.running:
            return
        self.replan()                                  # the newest look at the destination
        if not self.plan.can_start:
            return
        plan = self.plan
        self.running = True
        self.cancel = threading.Event()
        self.pages.setCurrentIndex(1)
        self.p_count.setText(f"Copying {len(plan.to_copy)} photos...")
        self.p_cancel.setEnabled(True)
        cancel = self.cancel

        def work():
            return I.run_import(plan, on_progress=self.sig.progress.emit, cancel=cancel)
        run_bg(work, "import", self._finished, self._failed, keep=True)

    def _progress(self, info: dict):
        tot = max(1, info["bytes_total"])
        self.p_bar.setValue(int(1000 * info["bytes_done"] / tot))
        self.p_count.setText(f"Copying {min(info['done'] + 1, info['total'])} of {info['total']}")
        speed = info.get("speed") or 0
        eta = info.get("eta")
        left = f"  ·  about {_dur(eta)} left" if eta else ""
        self.p_bytes.setText(f"{I.human_bytes(info['bytes_done'])} of {I.human_bytes(info['bytes_total'])}"
                             + (f"  ·  {I.human_bytes(speed)}/s" if speed else "") + left)
        self.p_file.setText(info.get("current") or "")

    def _cancel(self):
        self.cancel.set()
        self.p_cancel.setEnabled(False)
        self.p_file.setText("Stopping after this file (it is finished or removed, never left half-written)...")

    def _failed(self, _, err):
        self.running = False
        self.pages.setCurrentIndex(2)
        self.s_lines.setText(f"<span style='color:{T.RED}'>The import stopped: {err}</span><br>Details are in the log.")
        self.s_safe.setText("")

    def _finished(self, _, rep):
        self.running = False
        self.report = rep
        self.pages.setCurrentIndex(2)
        lines = []
        for ln in rep.summary_lines():
            colour = T.RED if ln.startswith(("Failed", "    ", "Stopped")) else (
                T.AMBER if ln.startswith(("Second copy", "Left on", "Cancelled", "Same name")) else T.TEXT)
            lines.append(f"<span style='color:{colour}'>{_html(ln)}</span>")
        if rep.folders:
            lines.append("")
            lines.append(f"<span style='color:{T.DIM}'>Into:</span>")
            lines += [f"<span style='color:{T.CYAN}'>{_html(f)}</span>" for f in rep.folders[:6]]
        self.s_lines.setText("<br>".join(lines))
        where = "card" if self._source_is_card() else "source folder"
        if rep.all_checked:
            self.s_safe.setText(f"All photos are copied and checked. You can take the {where} out. The Studio wrote "
                                f"nothing to the {where}." if where == "card" else
                                "All photos are copied and checked. The Studio wrote nothing to the source folder.")
        else:
            self.s_safe.setText("")
        self.s_open.setEnabled(bool(rep.folders))
        self.s_stacks.setEnabled(bool(rep.folders))
        for k in list(self.records):                   # the record grew: read it again next time
            self.records.pop(k, None)

    def _open_in_library(self):
        rep = self.report
        if rep and rep.folders:
            self.page.imported(rep.folders[0])
        self.close()

    def _find_stacks(self):
        rep = self.report
        if rep and rep.folders:
            self.page.imported(rep.folders[0])
            self.page.send_to.emit("cardstacks", list(rep.folders))
        self.close()

    def closeEvent(self, e):
        if self.running:
            self._cancel()
            e.ignore()                                 # closes itself once the current file is done
            QTimer.singleShot(300, self.close)
            return
        self.timer.stop()
        self.scan_token += 1
        self.sort_token += 1                           # a running sort stops at its next photo
        super().closeEvent(e)


def _dur(s: float) -> str:
    s = int(max(0, s))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


def _html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("  ", "&nbsp;&nbsp;")


def open_import(page, card=None) -> ImportDialog | None:
    """The Library's Import from card button. One dialog at a time; asks for the Photos folder first if needed."""
    dlg = getattr(page, "_import_dlg", None)
    if dlg is not None and dlg.isVisible():
        dlg.raise_()
        dlg.activateWindow()
        return dlg
    dlg = ImportDialog(page, card)
    page._import_dlg = dlg
    if not dlg.ensure_photos_folder():
        dlg.timer.stop()
        dlg.deleteLater()
        page._import_dlg = None
        return None
    dlg.replan()
    dlg.resize(780, 560)
    dlg.show()
    return dlg
