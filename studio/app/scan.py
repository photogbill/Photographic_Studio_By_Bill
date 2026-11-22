"""The 3D Scan section (BUILD_PLAN_DP3D H.1, H.2): its Sets page.

    Title  [chips]                                         [Choose folder...] [?]
    +-------------------------------------------------------+---------------------------+
    |  every stack found in the folder: name, frames,       |  What to make:            |
    |  Dual Pixel or not, frames left out and why, and what |   Surfaces for games      |
    |  was made (cyan done, amber skipped, red failed)      |   Leaves (foliage)        |
    |                                                       |  Options (maps, tiling,   |
    |                                                       |   game size, real size)   |
    |                                                       |  [Make them all]          |
    +-------------------------------------------------------+  honest limits            |
    |  progress: text . bar . Cancel                                                     |

Point it at a folder (or a card): Card to Stacks' sorter finds every focus stack and every single Dual Pixel photo,
and one button runs them all through the jobs ticked, with a queue that carries on after a Cancel or a crash
(stacks_job.json in the output folder). Results go into a "Studio surfaces" folder beside the photos. Each stack is
analysed once (the analysis cache), so surfaces, then leaves, then a model cost one analysis. "Make a 3D model" runs the
camera solver on the ticked stacks of one object and shows the model below the list (scanmodel.ModelView), with the
coach's advice.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QMessageBox, QPushButton,
                               QScrollArea, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import log as slog
from . import theme as T
from .common import Settings, card, chip, hbox, label
from .stackui import PANEL_W, JobBar, _widgets, honest_card

log = slog.get("app.scan")

WHAT = ("Point it at a folder of your Dual Pixel stacks (or a card): it finds every focus stack and every single "
        "Dual Pixel photo, and makes game surfaces (colour, measured height, normal, ambient occlusion, de-lit "
        "colour ...) and leaf cards for all of them in one go.")
HOW = ("Choose folder..., then Make them all. Stopped half way? Press it again: it carries on where it stopped. "
       "Results go into a 'Studio surfaces' folder beside your photos.")
HONEST = ("Height is measured where the picture has texture and the frames were in focus; plain areas are filled "
          "in and marked (violet). Sizes are in millimetres when the lens records its focus distance (the Makro-"
          "Planar does not); measuring the distance from the stack itself is the next build. Albedo and roughness "
          "are estimates (amber). A 3D model needs neighbouring stacks to share about half of what they see; the "
          "back and underside that no stack saw are filled from the outline and shown violet.")
STATUS_COLOUR = {"done": T.CYAN, "skipped": T.AMBER, "failed": T.RED, "running": T.TEXT, "pending": T.DIM}
SIZES = (("As shot", 0), ("2048 square", 2048), ("4096 square", 4096), ("1024 square", 1024))
TILES = (("No", ""), ("Yes: seam cut (brick, planks)", "cut"), ("Yes: blend (even textures)", "blend"))


class ScanPage(QWidget):
    open_result = Signal(str)
    status = Signal(str)
    result_written = Signal(str)
    SETTINGS_KEY = "scan_sets"

    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings()
        self.setAcceptDrops(True)
        self.folder = ""
        self.stacks: list = []
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
        th.addWidget(label("3D Scan · Sets", "h2"))
        self.chips = QHBoxLayout()
        self.chips.setSpacing(6)
        th.addLayout(self.chips)
        th.addStretch(1)
        self.add_b = QPushButton("Choose folder...")
        self.add_b.clicked.connect(self.choose_folder)
        help_b = QPushButton("?")
        help_b.setFixedWidth(34)
        help_b.setToolTip(f"{WHAT}\n\n{HOW}")
        help_b.clicked.connect(lambda: QMessageBox.information(self, "3D Scan", f"{WHAT}\n\n{HOW}\n\n{HONEST}"))
        self.none_b = QPushButton("Untick all")
        self.none_b.setToolTip("Untick every stack, then tick the stacks of one object for a 3D model.")
        self.none_b.clicked.connect(self.untick_all)
        th.addWidget(self.none_b)
        th.addWidget(self.add_b)
        th.addWidget(help_b)
        lv.addWidget(top)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Stack", "Frames", "Dual Pixel", "Made"])
        self.tree.setColumnWidth(0, 330)
        self.tree.setColumnWidth(1, 70)
        self.tree.setColumnWidth(2, 90)
        self.tree.setUniformRowHeights(True)
        lv.addWidget(self.tree, 1)
        from .scanmodel import ModelView
        self.model_view = ModelView()
        self.model_view.hide()
        lv.addWidget(self.model_view, 2)
        self.empty = card()
        ev = QVBoxLayout(self.empty)
        ev.setContentsMargins(26, 22, 26, 22)
        ev.addWidget(label("3D Scan", "h1"))
        for t, col in ((WHAT, T.TEXT), (HOW, T.DIM)):
            lb = label(t, None, wrap=True)
            lb.setStyleSheet(f"color:{col};")
            ev.addWidget(lb)
        b = QPushButton("Choose folder...")
        b.setObjectName("primary")
        b.clicked.connect(self.choose_folder)
        ev.addWidget(hbox(b))
        lv.addWidget(self.empty)
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
        pv.addLayout(self.actions)
        outer.addWidget(panel)
        self._build_panel()
        self._refresh()

    # ------------------------------------------------------------------------------------ panel
    def _build_panel(self):
        ValueSlider, Group = _widgets()
        g = Group("What to make", "For every stack ticked in the list.")
        self.panel.addWidget(g)
        k = self.SETTINGS_KEY
        self.do_surface = QCheckBox("Surfaces for games")
        self.do_surface.setChecked(bool(self.settings.get(f"{k}_surface", True)))
        self.do_surface.setToolTip("Colour, measured height, normal, confidence and agreement maps and a JSON file "
                                   "saying how each was made (and the extra game maps below).")
        self.do_leaves = QCheckBox("Leaves (cards, atlas, glb)")
        self.do_leaves.setChecked(bool(self.settings.get(f"{k}_leaves", False)))
        self.do_leaves.setToolTip("Each leaf cut out by its measured depth, with alpha, a normal map, a pivot at the "
                                  "stem and a bent mesh; packed into an atlas and glb files for Blender and Godot.")
        self.do_relief = QCheckBox("Front reliefs (glb)")
        self.do_relief.setChecked(bool(self.settings.get(f"{k}_relief", False)))
        self.do_relief.setToolTip("Each stack's measured depth as a cut-out relief mesh, textured, for Blender and "
                                  "Godot: the front of the subject as that stack saw it (not a full model). Good when "
                                  "the stacks round an object are too few or too far apart for a 3D model.")
        g.add(self.do_surface)
        g.add(self.do_relief)
        g.add(self.do_leaves)
        g = Group("Options")
        self.panel.addWidget(g)
        self.all_maps = QCheckBox("Every game map (AO, albedo, ORM...)")
        self.all_maps.setChecked(bool(self.settings.get(f"{k}_all_maps", True)))
        self.all_maps.setToolTip("Measured: height, normal, AO, cavity, curvature. Estimated (amber): albedo and the "
                                 "fine relief. Off: the five standard files only.")
        g.add(self.all_maps)
        self.tile = QComboBox()
        for t, v in TILES:
            self.tile.addItem(t, v)
        g.add(hbox(label("Tileable", "dim"), (self.tile, 1), stretch=False))
        self.size = QComboBox()
        for t, v in SIZES:
            self.size.addItem(t, v)
        g.add(hbox(label("Game size", "dim"), (self.size, 1), stretch=False))
        self.coin = QCheckBox("Real size from a coin in the frame")
        self.coin.setChecked(bool(self.settings.get(f"{k}_coin", False)))
        self.coin.setToolTip("Only if a frame happens to show a US quarter. Without it, the size comes from the "
                             "focus distance when the lens records one (and its view edge to edge); otherwise "
                             "sizes are relative and say so.")
        g.add(self.coin)
        self.report = label("", "dim", wrap=True)
        g.add(self.report)
        g = Group("3D model of one object", "Tick the stacks of one object (hand-held, any order), then Make a "
                  "3D model. Each stack is a view; the Studio places them itself.")
        self.panel.addWidget(g)
        self.lods = QCheckBox("Game LODs (two smaller versions)")
        self.lods.setChecked(bool(self.settings.get(f"{k}_lods", False)))
        self.ground = QCheckBox("Cut at the ground")
        self.ground.setChecked(bool(self.settings.get(f"{k}_ground", False)))
        self.ground.setToolTip("When the object stands on the ground and some ground came with it.")
        g.add(self.lods)
        g.add(self.ground)
        from PySide6.QtWidgets import QDoubleSpinBox
        self.length = QDoubleSpinBox()
        self.length.setRange(0.0, 100.0)
        self.length.setDecimals(3)
        self.length.setSingleStep(0.05)
        self.length.setSuffix(" m")
        self.length.setSpecialValueText("not known")
        self.length.setValue(float(self.settings.get(f"{k}_length", 0.0) or 0.0))
        self.length.setToolTip("The object's longest side, if you know it. Not needed: without it the model has the "
                               "same shape and texture, only its size in metres is unknown.")
        g.add(hbox(label("Length (optional)", "dim"), self.length, stretch=True))
        self.model_b = QPushButton("Make a 3D model")
        self.model_b.setToolTip("Cameras, points, a textured glb for Blender and Godot, and a Blender dataset when the "
                                "scan is good enough (it says why when not).")
        self.model_b.clicked.connect(self.make_model)
        g.add(self.model_b)
        self.panel.addWidget(honest_card(HONEST))
        self.panel.addStretch(1)
        self.go_b = QPushButton("Make them all")
        self.go_b.setObjectName("primary")
        self.go_b.clicked.connect(self.make_all)
        self.open_b = QPushButton("Open the results folder")
        self.open_b.clicked.connect(self.open_folder)
        self.blend_b = QPushButton("Open in Blender")
        self.blend_b.setToolTip("The selected stack's surface as a displaced, textured plane in Blender (made first "
                                "with Make them all).")
        self.blend_b.clicked.connect(self.open_in_blender)
        self.actions.addWidget(self.go_b)
        self.actions.addWidget(hbox((self.open_b, 1), self.blend_b, stretch=False))

    # ------------------------------------------------------------------------------------ folder
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        p = [u.toLocalFile() for u in e.mimeData().urls()]
        d = next((x for x in p if os.path.isdir(x)), None) or (os.path.dirname(p[0]) if p else "")
        if d:
            self.set_folder(d)

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "A folder of stacks", self.folder or
                                             self.settings.get(f"{self.SETTINGS_KEY}_folder", ""))
        if d:
            self.set_folder(d)

    def set_folder(self, folder: str):
        if self.jobbar.busy:
            return
        self.folder = folder
        self.settings.set(f"{self.SETTINGS_KEY}_folder", folder)
        from ..engine.stack import batch as B
        self.out_dir = os.path.join(folder, B.DEFAULT_SUBFOLDER)

        def work(progress, cancel):
            return B.find_stacks([folder], "normal", progress, cancel)

        def done(stacks):
            self.stacks = stacks
            self._fill()
            n = sum(1 for s in stacks if s.kind == "focus" and len(s.paths) >= 2)
            d = sum(1 for s in stacks if len(s.paths) == 1 and s.dual_pixel)
            self.jobbar.say(f"{n} focus stack(s) and {d} single Dual Pixel photo(s) found.", T.CYAN)
            self._refresh()

        def fail(err, cancelled):
            self.jobbar.say("Stopped." if cancelled else f"Could not read the folder: {err}", T.AMBER)
            self._refresh()
        self.jobbar.start(work, done, fail, text="Finding the stacks...")
        self._refresh()

    def _fill(self, items: dict | None = None):
        self.tree.clear()
        for s in self.stacks:
            usable = (s.kind == "focus" and len(s.paths) >= 2) or (len(s.paths) == 1 and s.dual_pixel)
            it = QTreeWidgetItem([s.name, str(len(s.paths)), (f"{s.dual_pixel} of {len(s.paths)}" if s.paths else ""),
                                  "" if usable else ("burst: not a stack" if s.kind == "burst" else
                                                     "one ordinary photo")])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(0, Qt.CheckState.Checked if usable else Qt.CheckState.Unchecked)
            col = T.CYAN if s.dual_pixel == len(s.paths) and s.paths else (T.AMBER if s.dual_pixel else T.DIM)
            it.setForeground(2, QBrush(QColor(col)))
            if not usable:
                it.setForeground(0, QBrush(QColor(T.DIM)))
            for lo in s.left_out:
                c = QTreeWidgetItem([lo["name"], "", "", lo["reason"]])
                c.setForeground(3, QBrush(QColor(T.RED)))
                it.addChild(c)
            self.tree.addTopLevelItem(it)
        if items:
            self._show_items(items)

    def _show_items(self, items: dict):
        by = {}
        for it in items.values():
            by.setdefault(it.get("stack"), []).append(it)
        for i in range(self.tree.topLevelItemCount()):
            row = self.tree.topLevelItem(i)
            got = by.get(row.text(0)) or []
            if not got:
                continue
            parts = [f"{x['job']}: {x['status']}" for x in got]
            worst = ("failed" if any(x["status"] == "failed" for x in got) else
                     "done" if all(x["status"] == "done" for x in got) else got[0]["status"])
            row.setText(3, ", ".join(parts))
            row.setForeground(3, QBrush(QColor(STATUS_COLOUR.get(worst, T.DIM))))
            errs = [x.get("error", "") for x in got if x.get("error")]
            row.setToolTip(3, "\n".join(errs))

    def untick_all(self):
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Unchecked)

    def selected(self) -> list:
        out = []
        for i in range(self.tree.topLevelItemCount()):
            if self.tree.topLevelItem(i).checkState(0) == Qt.CheckState.Checked:
                out.append(self.stacks[i])
        return out

    # ------------------------------------------------------------------------------------ making
    def params(self):
        from ..engine.stack import surface as SURF
        tile = self.tile.currentData() or ""
        return SURF.SurfaceParams(tileable=bool(tile), tile_method=tile or "cut", size=int(self.size.currentData()),
                                  scale="coin" if self.coin.isChecked() else "",
                                  maps="all" if self.all_maps.isChecked() else "")

    def make_all(self):
        if self.jobbar.busy or not self.stacks:
            return
        jobs = (["surface"] if self.do_surface.isChecked() else []) + \
            (["relief"] if self.do_relief.isChecked() else []) + (["foliage"] if self.do_leaves.isChecked() else [])
        if not jobs:
            self.jobbar.say("Tick Surfaces, Front reliefs or Leaves first.", T.AMBER)
            return
        k = self.SETTINGS_KEY
        for name, w in (("surface", self.do_surface), ("relief", self.do_relief), ("leaves", self.do_leaves),
                        ("all_maps", self.all_maps), ("coin", self.coin)):
            self.settings.set(f"{k}_{name}", w.isChecked())
        stacks = self.selected()
        sp = self.params()
        out_dir = self.out_dir

        def work(progress, cancel):
            from ..engine.stack import batch as B
            return B.run(stacks, jobs, out_dir, sp, None, progress, cancel, resume=True)

        def done(doc):
            from ..engine.stack import batch as B
            self._show_items(doc.get("items", {}))
            st = B.summary(doc)
            colour = T.GREEN if not st.get("failed") else T.AMBER
            self.jobbar.say(("Stopped: press Make them all to carry on. " if doc.get("state") == "stopped" else
                             "Done. ") + ", ".join(f"{v} {k_}" for k_, v in st.items()) + f". In {out_dir}", colour)
            self.report.setText(f"Job file: {os.path.join(out_dir, B.JOB_FILE)}")
            for it in doc.get("items", {}).values():
                for q in it.get("outputs", []):
                    if q.endswith(".json"):
                        self.result_written.emit(q)
                        break
            self._refresh()

        def fail(err, cancelled):
            self.jobbar.say("Stopped: press Make them all to carry on." if cancelled else f"Problem: {err}",
                            T.AMBER)
            self._refresh()
        self.jobbar.start(work, done, fail, keep=True, text="Making surfaces...")
        self._refresh()

    def make_model(self):
        if self.jobbar.busy:
            return
        stacks = [st for st in self.selected() if (st.kind == "focus" and len(st.paths) >= 2) or
                  (len(st.paths) == 1 and st.dual_pixel)]
        if len(stacks) < 2:
            self.jobbar.say("Tick at least two stacks of the same object.", T.AMBER)
            return
        k = self.SETTINGS_KEY
        self.settings.set(f"{k}_lods", self.lods.isChecked())
        self.settings.set(f"{k}_ground", self.ground.isChecked())
        self.settings.set(f"{k}_length", float(self.length.value()))
        out_dir = os.path.join(self.folder, "Studio models")
        length = float(self.length.value())
        base = os.path.basename(os.path.normpath(self.folder)) or "scan"
        lods, ground = self.lods.isChecked(), self.ground.isChecked()

        def work(progress, cancel):
            from ..engine.scan import model as MD
            from ..engine.scan import views as VW
            views = []
            for i, st in enumerate(stacks):
                progress(0.4 * i / len(stacks), f"Reading view {i + 1} of {len(stacks)}: {st.name}")
                if len(st.paths) == 1:
                    views.append(VW.from_dual(st.paths[0]))
                else:
                    views.append(VW.from_stack(st.paths, st.base, cancel=cancel))
            return MD.build(views, out_dir, base, MD.ModelParams(lods=lods, ground_cut=ground, known_length_m=length),
                            MD.MAKES,
                            lambda f, m: progress(0.4 + 0.6 * f, m), cancel)

        def done(r):
            self.model_view.show_model(r)
            self.model_view.show()
            self.out_dir = out_dir
            self.last_model = r["paths"].get("json")
            g = r["json"]["gate"]
            self.jobbar.say(f"Model written in {out_dir}." + ("" if g["passed"] else " No Blender dataset (see why "
                                                                                    "below the model)."),
                            T.GREEN if g["passed"] else T.AMBER)
            if self.last_model:
                self.result_written.emit(self.last_model)
            self._refresh()

        def fail(err, cancelled):
            self.jobbar.say("Stopped." if cancelled else f"No model: {err}", T.AMBER)
            self._refresh()
        self.jobbar.start(work, done, fail, keep=True, text="Making the 3D model...")
        self._refresh()

    def open_in_blender(self):
        from ..engine.io import blender as BL
        from ..engine.stack import batch as B
        row = self.tree.currentIndex().row() if self.tree.currentItem() is not None else -1
        if not (0 <= row < len(self.stacks)):
            self.jobbar.say("Click a stack in the list first.", T.AMBER)
            return
        si = self.stacks[row]
        js = os.path.join(self.out_dir, f"{si.base}_surface_Studio.json")
        if not os.path.isfile(js):
            self.jobbar.say("Make its surface first (Make them all).", T.AMBER)
            return
        try:
            r = BL.run(js, None, background=False)
            self.jobbar.say(f"Blender is opening the surface ({r['from']}).", T.CYAN)
        except FileNotFoundError as e:
            self.jobbar.say(str(e), T.AMBER)
        except OSError as e:
            self.jobbar.say(f"Blender could not be started: {e}", T.AMBER)

    def open_folder(self):
        d = self.out_dir if self.out_dir and os.path.isdir(self.out_dir) else self.folder
        if d and os.path.isdir(d):
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def _refresh(self):
        has = bool(self.stacks)
        self.tree.setVisible(has)
        self.empty.setVisible(not has)
        self.go_b.setEnabled(has and not self.jobbar.busy)
        self.model_b.setEnabled(has and not self.jobbar.busy)
        self.add_b.setEnabled(not self.jobbar.busy)
        self.none_b.setVisible(has)
        self.open_b.setEnabled(bool(self.folder))
        while self.chips.count():
            w = self.chips.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        if has:
            n = sum(1 for s in self.stacks if s.kind == "focus" and len(s.paths) >= 2)
            self.chips.addWidget(chip(f"{n} stacks", T.CYAN))
            dp = sum(1 for s in self.stacks if s.dual_pixel)
            if dp:
                self.chips.addWidget(chip(f"{dp} with Dual Pixel", T.CYAN))
