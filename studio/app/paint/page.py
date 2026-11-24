"""The Paint page in the Studio's sidebar (BUILD_PLAN_PAINT P0-P1).

Canvas in the middle; on the right Brush, Colour, Layers and Tablet check. Files: New, Open / Save OpenRaster (.ora,
the artist's own file, where she chooses), Export a flat PNG / TIFF / JPEG as NAME_Studio.ext (never over anything
the Studio did not write). Autosave every minute beside the app, and an offer to recover after a crash.

Keys: B brush, F fill, G gradient, E eraser, X swap colours, [ ] brush size, Ctrl+Z undo, Ctrl+Shift+Z / Ctrl+Y redo, Tab canvas only,
Alt + click pick colour; the canvas has its own view keys (see canvas.py).
"""
from __future__ import annotations

import logging
import os

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox,
                               QPushButton, QScrollArea, QSlider, QTabWidget, QToolButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from ...engine.paint import autosave, keymap, ora, swatches as swatch_files
from ...engine.paint.brush import MODES as PAINT_MODES, SMUDGE_KINDS, presets
from ...engine.paint.colour import SPACES, from_hex, to_hex
from ...engine.paint.fill import FILL_SOURCES, GAP_NOTE, GRADIENT_KINDS
from ...engine.paint.curves import PRESET_NOTES, PRESETS, Curve
from ...engine.paint.smoothing import MODES as SMOOTH_MODES
from ...engine.paint.tiles import BLEND_MODES, GROUP_BLEND, GROUP_NOTE, Document
from .. import theme as T
from ..common import Settings, report_problem
from .canvas import PaintCanvas
from .tabletcheck import TabletCheck
from .widgets import ColourWheel, CurveBox, GamutPicker, SwatchGrid, TrianglePicker

log = logging.getLogger("studio.paint")


def zipfile_error():
    import zipfile
    return zipfile.BadZipFile

SIZES = [("2000 x 1500 (sketch)", 2000, 1500), ("A4 at 300 dpi (2480 x 3508)", 2480, 3508),
         ("A3 at 300 dpi (3508 x 4961)", 3508, 4961), ("1920 x 1080 (screen)", 1920, 1080),
         ("3840 x 2160 (4K screen)", 3840, 2160), ("3000 x 3000 (square)", 3000, 3000)]


def _slider(lo, hi, val):
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    return s


class PaintPage(QWidget):
    status = Signal(str)

    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings()
        from ...engine.paint.pack import pack
        self.brushes = pack()
        self._add_saved_brushes()
        self._add_saved_mypaint()
        self.brush_i = 2
        self.bg_colour = (1.0, 1.0, 1.0)
        self.recent: list = []
        self.saver = None
        self._asked_recovery = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.bar = self._toolbar()
        root.addWidget(self.bar)
        mid = QHBoxLayout()
        mid.setSpacing(0)
        self.canvas = PaintCanvas()
        mid.addWidget(self.canvas, 1)
        self.side = QTabWidget()
        self.side.setFixedWidth(330)
        self.side.addTab(self._scroll(self._brush_tab()), "Brush")
        self.side.addTab(self._scroll(self._colour_tab()), "Colour")
        self.side.addTab(self._scroll(self._layers_tab()), "Layers")
        self.side.addTab(self._scroll(self._fill_tab()), "Fill")
        self.side.addTab(self._scroll(self._select_tab()), "Select")
        self.side.addTab(self._history_tab(), "History")
        self.side.addTab(self._scroll(self._keys_tab()), "Keys")
        self.tablet = TabletCheck()
        self.side.addTab(self._scroll(self.tablet), "Tablet check")
        mid.addWidget(self.side)
        root.addLayout(mid, 1)
        c = self.canvas
        c.sample_seen.connect(self._sample)
        c.colour_picked.connect(self.set_colour)
        c.changed.connect(self._changed)
        c.view_changed.connect(lambda t: self.view_lbl.setText(t))
        c.message.connect(self._say)
        c.popup_requested.connect(self.popup)
        c.selection_changed.connect(self._selection_changed)
        self._load_prefs()
        self.picker_cb.setCurrentIndex(int(self.settings.get("paint_picker", 0) or 0))
        names = [sw.name for sw in self._swatch_sets]
        if self.settings.get("paint_swatch_set") in names:
            self._swatch_i = names.index(self.settings.get("paint_swatch_set"))
            self._refresh_swatches()
        self.select_brush(self.brush_i)
        self.set_tool("brush")
        self.set_colour((0.1, 0.12, 0.2))
        self._new_saver("Untitled")
        self._refresh_layers()
        self._shortcuts: list = []
        self._apply_keys()
        self._autosave_timer = QTimer(self, interval=15000, timeout=self._autosave)
        self._autosave_timer.start()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.on_quit)

    # ------------------------------------------------------------------ layout
    def _scroll(self, w):
        s = QScrollArea()
        s.setWidgetResizable(True)
        s.setWidget(w)
        return s

    def _toolbar(self):
        w = QWidget()
        w.setObjectName("toolbar")
        h = QHBoxLayout(w)
        h.setContentsMargins(8, 6, 8, 6)

        def btn(text, fn, tip=""):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            h.addWidget(b)
            return b
        btn("New", self.new, "A new painting")
        btn("Open", self.open, "Open an OpenRaster (.ora) painting (Krita, GIMP, MyPaint read and write it too)")
        btn("Save", self.save, "Save the painting as .ora, with its layers (Ctrl+S)")
        btn("Save as", self.save_as, "Save under a new name (Ctrl+Shift+S)")
        btn("Export", self.export, "A flat PNG, TIFF or JPEG, named NAME_Studio")
        h.addSpacing(12)
        btn("Undo", self.undo, "Ctrl+Z")
        btn("Redo", self.redo, "Ctrl+Shift+Z or Ctrl+Y")
        h.addSpacing(12)
        self.tool_btns = {}
        for key, text, tip in (("brush", "Brush", "B: paint with the chosen brush"),
                               ("fill", "Fill", "F: fill an area (settings in the Fill tab)"),
                               ("gradient", "Gradient", "G: drag to lay a gradient (settings in the Fill tab)")):
            b = btn(text, lambda _=False, k=key: self.set_tool(k), tip)
            b.setCheckable(True)
            self.tool_btns[key] = b
        self.erase_btn = btn("Eraser", lambda: self.set_eraser(not self.canvas.eraser_mode),
                             "E: paint takes paint away (the pen's eraser end always erases)")
        self.erase_btn.setCheckable(True)
        h.addSpacing(12)
        btn("Fit", lambda: self.canvas.fit(), "0: the whole painting in view")
        btn("100 %", lambda: self.canvas.set_zoom(1.0), "1: actual pixels")
        btn("⟲", lambda: self.canvas.rotate_by(-15), "4: rotate the view left")
        btn("⟳", lambda: self.canvas.rotate_by(15), "6: rotate the view right (5 resets)")
        btn("Mirror", lambda: self.canvas.toggle_mirror(), "M: flip the view to check proportions (the painting is "
                                                           "not changed)")
        self.value_cb = QCheckBox("Values")
        self.value_cb.setToolTip("Show the painting in greys to check light and dark (the painting is not changed)")
        self.value_cb.toggled.connect(self._value_view)
        h.addWidget(self.value_cb)
        h.addStretch(1)
        self.view_lbl = QLabel("")
        self.view_lbl.setStyleSheet(f"color:{T.DIM}")
        h.addWidget(self.view_lbl)
        self.title_lbl = QLabel("")
        h.addWidget(self.title_lbl)
        return w

    def _note(self, text=""):
        lb = QLabel(text)
        lb.setWordWrap(True)
        lb.setStyleSheet(f"color:{T.DIM}")
        return lb

    def _brush_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        from ...engine.paint.pack import CATEGORIES
        frow = QHBoxLayout()
        self.brush_cat = QComboBox()
        self.brush_cat.addItem("All brushes", "")
        for c in CATEGORIES:
            self.brush_cat.addItem(c, c)
        self.brush_cat.currentIndexChanged.connect(self._filter_brushes)
        self.brush_find = QLineEdit()
        self.brush_find.setPlaceholderText("Find a brush…")
        self.brush_find.setToolTip("Searches names and what each brush is best for (try: charcoal, foliage, ink)")
        self.brush_find.textChanged.connect(self._filter_brushes)
        frow.addWidget(self.brush_cat)
        frow.addWidget(self.brush_find)
        v.addLayout(frow)
        self.brush_list = QListWidget()
        self.brush_list.setMaximumHeight(230)
        for b in self.brushes:
            it = QListWidgetItem(b.name)
            it.setToolTip(f"{b.tags}\nBest for: {b.best_for}\nWeak at: {b.weak_at}")
            self.brush_list.addItem(it)
        self.brush_list.currentRowChanged.connect(self.select_brush)
        v.addWidget(self.brush_list)
        self.brush_note = self._note()
        v.addWidget(self.brush_note)
        f = QFormLayout()
        self.s_size = _slider(1, 1000, 30)
        self.s_hard = _slider(0, 100, 80)
        self.s_opac = _slider(1, 100, 100)
        self.s_flow = _slider(1, 100, 100)
        self.s_space = _slider(2, 100, 10)
        for name, s in (("Size", self.s_size), ("Hardness", self.s_hard), ("Opacity", self.s_opac),
                        ("Flow", self.s_flow), ("Spacing", self.s_space)):
            s.valueChanged.connect(self._brush_from_ui)
            f.addRow(name, s)
        self.mode_cb = QComboBox()
        for k, (name, _, _) in PAINT_MODES.items():
            self.mode_cb.addItem(name, k)
        self.mode_cb.currentIndexChanged.connect(self._brush_from_ui)
        f.addRow("Paint", self.mode_cb)
        self.space_cb = QComboBox()
        for k, (name, _, _) in SPACES.items():
            self.space_cb.addItem(name, k)
        self.space_cb.currentIndexChanged.connect(self._brush_from_ui)
        f.addRow("Mix colours", self.space_cb)
        v.addLayout(f)
        self.mode_note = self._note()
        v.addWidget(self.mode_note)
        self.smudge_box = QWidget()
        sf = QFormLayout(self.smudge_box)
        sf.setContentsMargins(0, 0, 0, 0)
        self.sm_kind = QComboBox()
        for k, (name, _, _) in SMUDGE_KINDS.items():
            self.sm_kind.addItem(name, k)
        self.sm_len = _slider(0, 100, 80)
        self.sm_len.setToolTip("How far the brush carries the paint it picks up")
        self.sm_rate = _slider(0, 100, 0)
        self.sm_rate.setToolTip("How much of the painting colour is mixed into the carried paint (an oily, "
                                "loaded brush). 0 = smudge only")
        self.sm_all = QCheckBox("Read every layer (pull paint from below up onto this layer)")
        sf.addRow("Smudge", self.sm_kind)
        sf.addRow("Smudge length", self.sm_len)
        sf.addRow("Colour rate", self.sm_rate)
        sf.addRow(self.sm_all)
        for w_ in (self.sm_kind,):
            w_.currentIndexChanged.connect(self._brush_from_ui)
        for w_ in (self.sm_len, self.sm_rate):
            w_.valueChanged.connect(self._brush_from_ui)
        self.sm_all.toggled.connect(self._brush_from_ui)
        v.addWidget(self.smudge_box)
        eb = QPushButton("Edit brush…")
        eb.setToolTip("Every setting of this brush, with a live preview stroke: tip, dynamics, paper grain, a second "
                      "tip, scatter and colour jitter")
        eb.clicked.connect(lambda: self.open_brush_editor())
        v.addWidget(eb)
        ib = QPushButton("Import brushes (.abr, .sut, .kpp, .gbr, .gih)…")
        ib.setToolTip("Brushes from Photoshop, Clip Studio, Krita and GIMP. The tips come across; most of the other "
                      "programs' settings do not, so set the dynamics in Edit brush afterwards.")
        ib.clicked.connect(lambda: self.import_brushes())
        v.addWidget(ib)
        mb = QPushButton("Load MyPaint brushes (.myb)…")
        mb.setToolTip("MyPaint's brushes, with all their curves, painted by our port of MyPaint's brush engine")
        mb.clicked.connect(lambda: self.load_mypaint())
        v.addWidget(mb)
        tiprow = QHBoxLayout()
        tb = QPushButton("Tip from a picture...")
        tb.setToolTip("Any picture becomes the brush's tip: a scan of an ink mark, a leaf, a sponge. Dark marks on a "
                      "light ground paint where they are dark.")
        tb.clicked.connect(self.load_tip)
        rb = QPushButton("Round tip")
        rb.clicked.connect(self.round_tip)
        tiprow.addWidget(tb)
        tiprow.addWidget(rb)
        v.addLayout(tiprow)
        self.tip_lbl = self._note("Tip: round")
        v.addWidget(self.tip_lbl)
        v.addWidget(QLabel("Smoothing"))
        self.smooth_cb = QComboBox()
        for k, (name, _, _) in SMOOTH_MODES.items():
            self.smooth_cb.addItem(name, k)
        self.smooth_cb.currentIndexChanged.connect(self._smoothing_from_ui)
        v.addWidget(self.smooth_cb)
        self.smooth_s = _slider(0, 100, 30)
        self.smooth_s.valueChanged.connect(self._smoothing_from_ui)
        v.addWidget(self.smooth_s)
        self.smooth_note = self._note()
        v.addWidget(self.smooth_note)
        v.addWidget(QLabel("Your pen's pressure curve (for every brush)"))
        self.curve_cb = QComboBox()
        self.curve_cb.addItems(list(PRESETS))
        self.curve_cb.addItem("Custom")
        self.curve_cb.activated.connect(self._curve_preset)
        v.addWidget(self.curve_cb)
        self.curve_box = CurveBox()
        self.curve_box.curve_changed.connect(self._curve_edited)
        v.addWidget(self.curve_box)
        self.curve_note = self._note("Drag the points; double-click adds one, right-click removes one.")
        v.addWidget(self.curve_note)
        v.addStretch(1)
        return w

    def _colour_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        from PySide6.QtWidgets import QStackedWidget
        self.picker_cb = QComboBox()
        self.pickers = [ColourWheel(), TrianglePicker(), GamutPicker()]
        notes = ["Quick hue, then saturation across and value down. Weak at: even lightness steps.",
                 "Tints, shades and tones of one hue are straight lines to the white and black corners (as Krita "
                 "and MyPaint). Weak at: an exact saturation.",
                 "OKLCH: lightness up, chroma across, hue below, in even steps to the eye. Cyan line: the edge of "
                 "sRGB, what the canvas holds; violet: Display P3. Weak at: past the cyan line the colour is brought "
                 "in to sRGB at the same lightness and hue."]
        for name in ("Wheel and square", "Triangle", "Wide gamut (OKLCH)"):
            self.picker_cb.addItem(name)
        self.picker_stack = QStackedWidget()
        for pk in self.pickers:
            pk.colour_changed.connect(lambda c, pk=pk: self._picked(pk, c))
            self.picker_stack.addWidget(pk)
        self.wheel = self.pickers[0]
        self.picker_note = self._note(notes[0])

        def pick_picker(i):
            self.picker_stack.setCurrentIndex(i)
            self.pickers[i].set_rgb(self.canvas.colour)
            self.picker_note.setText(notes[i])
            self.settings.set("paint_picker", i)
        self.picker_cb.currentIndexChanged.connect(pick_picker)
        v.addWidget(self.picker_cb)
        v.addWidget(self.picker_stack)
        v.addWidget(self.picker_note)
        row = QHBoxLayout()
        self.fg_sw = QPushButton()
        self.fg_sw.setFixedSize(46, 46)
        self.fg_sw.setToolTip("Painting colour")
        self.bg_sw = QPushButton()
        self.bg_sw.setFixedSize(30, 30)
        self.bg_sw.setToolTip("Second colour (X swaps them)")
        self.bg_sw.clicked.connect(self.swap_colours)
        row.addWidget(self.fg_sw)
        row.addWidget(self.bg_sw)
        self.hex_ed = QLineEdit()
        self.hex_ed.setMaxLength(7)
        self.hex_ed.editingFinished.connect(self._hex)
        row.addWidget(self.hex_ed)
        v.addLayout(row)
        self.gamut_lbl = self._note("")
        v.addWidget(self.gamut_lbl)
        v.addWidget(QLabel("Recent colours"))
        self.recent_row = QHBoxLayout()
        v.addLayout(self.recent_row)
        v.addWidget(QLabel("Swatches"))
        self.sw_cb = QComboBox()
        self.sw_cb.activated.connect(self._swatch_set_chosen)
        v.addWidget(self.sw_cb)
        self.sw_grid = SwatchGrid()
        self.sw_grid.picked.connect(self.set_colour)
        self.sw_grid.remove_requested.connect(self._swatch_remove)
        v.addWidget(self.sw_grid)
        self.sw_note = self._note("")
        v.addWidget(self.sw_note)
        r1, r2 = QHBoxLayout(), QHBoxLayout()
        for r, text, fn, tip in (
                (r1, "+ Colour", self._swatch_add, "Add the painting colour to this set"),
                (r1, "From painting", self._swatch_from_painting,
                 "A new set of the painting's main colour masses (dark to light)"),
                (r2, "New set", self._swatch_new, "Start an empty set"),
                (r2, "Save set", self._swatch_save, "Save this set (as .gpl: Krita, GIMP, Inkscape and MyPaint "
                                                     "read it)"),
                (r2, "Load set…", self._swatch_load, "Bring in a .gpl or Krita .kpl swatch set")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            r.addWidget(b)
        v.addLayout(r1)
        v.addLayout(r2)
        self._swatch_sets = swatch_files.saved_sets()
        self._swatch_i = 0
        self._refresh_swatches()
        self._space_box(v)
        v.addWidget(self._note("Alt + click on the painting picks its colour. The Values button on the bar shows "
                               "the painting in greys."))
        v.addStretch(1)
        return w

    def _layers_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.layer_list = QTreeWidget()
        self.layer_list.setHeaderHidden(True)
        self.layer_list.setIndentation(16)
        self.layer_list.currentItemChanged.connect(self._layer_selected)
        self.layer_list.itemDoubleClicked.connect(self._rename_layer)
        self.layer_list.itemExpanded.connect(lambda it: self._fold(it, False))
        self.layer_list.itemCollapsed.connect(lambda it: self._fold(it, True))
        v.addWidget(self.layer_list)
        row = QHBoxLayout()
        row2 = QHBoxLayout()
        for r, text, fn, tip in (
                (row, "+", self.add_layer, "New layer (inside the selected group when a group is selected)"),
                (row, "−", self.delete_layer, "Delete the layer, or the group with everything in it"),
                (row, "▲", lambda: self.move_layer(1), "Move up. Past an open group it goes in; at the top of a "
                                                       "group it comes out"),
                (row, "▼", lambda: self.move_layer(-1), "Move down. Past an open group it goes in; at the bottom "
                                                        "of a group it comes out"),
                (row2, "Group", self.group_layer, "Put the selected layer into a new group"),
                (row2, "Ungroup", self.ungroup_layer, "Remove the selected group, keeping what is in it"),
                (row2, "Merge", self.merge_down, "A layer: merge into the layer below it. A group: merge it "
                                                  "into one layer that looks the same")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            r.addWidget(b)
        v.addLayout(row)
        v.addLayout(row2)
        f = QFormLayout()
        self.l_opac = _slider(0, 100, 100)
        self.l_opac.sliderReleased.connect(self._layer_props)
        self.l_opac.valueChanged.connect(self._layer_opacity_live)
        f.addRow("Opacity", self.l_opac)
        self.l_blend = QComboBox()
        for k, (name, _, _) in BLEND_MODES.items():
            self.l_blend.addItem(name, k)
        self.l_blend.activated.connect(self._layer_props)
        f.addRow("Blend", self.l_blend)
        v.addLayout(f)
        self.blend_note = self._note()
        v.addWidget(self.blend_note)
        self.l_vis = QCheckBox("Visible")
        self.l_lock = QCheckBox("Locked (no painting)")
        self.l_alpha = QCheckBox("Alpha lock (paint only where there is paint)")
        for cb in (self.l_vis, self.l_lock, self.l_alpha):
            cb.clicked.connect(self._layer_props)
            v.addWidget(cb)
        bgrow = QHBoxLayout()
        bgrow.addWidget(QLabel("Paper"))
        self.paper_cb = QComboBox()
        self.paper_cb.addItems(["White", "Warm paper", "Grey", "Black", "See-through"])
        self.paper_cb.activated.connect(self._paper)
        bgrow.addWidget(self.paper_cb)
        v.addLayout(bgrow)
        from ...engine.paint.adjust import ADJUSTMENTS
        mrow = QHBoxLayout()
        for text, fn, tip in (("Add mask", self.add_mask, "A mask that shows everything (or the selection)"),
                              ("Apply mask", self.apply_mask, "Bake the mask into the paint"),
                              ("Delete mask", self.delete_mask, "Remove the mask")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            mrow.addWidget(b)
        v.addLayout(mrow)
        self.mask_cb = QCheckBox("Paint on the mask (black hides, white shows)")
        self.mask_cb.toggled.connect(lambda on: setattr(self.canvas, "mask_mode", on))
        v.addWidget(self.mask_cb)
        self.clip_cb = QCheckBox("Clip to the layer below (shows only on its paint)")
        self.clip_cb.toggled.connect(self._clip_toggle)
        v.addWidget(self.clip_cb)
        arow = QHBoxLayout()
        self.adj_cb = QComboBox()
        for k, (n, best, weak) in ADJUSTMENTS.items():
            self.adj_cb.addItem(n, k)
            self.adj_cb.setItemData(self.adj_cb.count() - 1, f"Best for: {best}. Weak at: {weak}.",
                                    Qt.ItemDataRole.ToolTipRole)
        ab = QPushButton("Add adjustment layer")
        ab.clicked.connect(self.add_adjustment)
        eb2 = QPushButton("Edit…")
        eb2.clicked.connect(self.edit_adjustment)
        arow.addWidget(self.adj_cb)
        arow.addWidget(ab)
        arow.addWidget(eb2)
        v.addLayout(arow)
        krow = QHBoxLayout()
        for text, fn, tip in (("Filter layer", self.add_filter_layer, "A filter (the one chosen in Select > "
                                "Filters) on everything below, changeable later; nothing is painted over"),
                              ("Fill layer", self.add_fill_layer, "A layer of the painting colour that fills "
                               "the canvas: change its colour any time, mask it, fade it"),
                              ("File layer…", self.add_file_layer, "A picture file linked as a layer: when the "
                               "file changes, Reload shows the new version"),
                              ("Reload", self.reload_file_layers, "Read every linked picture again")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            krow.addWidget(b)
        v.addLayout(krow)
        v.addWidget(QLabel("Paint relief (impasto)"))
        self.relief_cb = QCheckBox("Light the paint's thickness")
        self.relief_cb.setToolTip("Thick paint (brushes with Thickness, knives, bristles) shows its ridges under a "
                                  "light you can move. Only the view changes, not the paint.")
        self.relief_cb.toggled.connect(self._relief)
        v.addWidget(self.relief_cb)
        rf = QFormLayout()
        self.relief_az = _slider(0, 359, 120)
        self.relief_el = _slider(5, 85, 35)
        self.relief_st = _slider(0, 200, 80)
        for name, sl in (("Light from", self.relief_az), ("Light height", self.relief_el),
                         ("Strength", self.relief_st)):
            sl.valueChanged.connect(self._relief)
            rf.addRow(name, sl)
        v.addLayout(rf)
        self.bits_cb = QCheckBox("Save layers in 16-bit")
        self.bits_cb.setToolTip("Best for: soft glazes and slow fades keep every shade. Weak at: larger files; "
                                "MyPaint and Krita read it (Krita turns it to 8-bit), other programs may not.")
        self.bits_cb.toggled.connect(self._bits)
        v.addWidget(self.bits_cb)
        v.addWidget(self._note("Double-click a layer to rename it. The top of the list is the top layer. Groups "
                               "keep layers together; the arrow beside a group opens and closes it. A selected "
                               "group passes strokes to the top layer inside it."))
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------ brush
    def select_brush(self, i):
        if i < 0:
            return
        self.brush_i = i
        b = self.brushes[i]
        if self.brush_list.currentRow() != i:
            self.brush_list.setCurrentRow(i)
        for s, val in ((self.s_size, b.size), (self.s_hard, b.hardness * 100), (self.s_opac, b.opacity * 100),
                       (self.s_flow, b.flow * 100), (self.s_space, b.spacing * 100)):
            s.blockSignals(True)
            s.setValue(int(round(val)))
            s.blockSignals(False)
        for cb, val in ((self.mode_cb, list(PAINT_MODES).index(b.mode)), (self.space_cb, list(SPACES).index(b.space))):
            cb.blockSignals(True)
            cb.setCurrentIndex(val)
            cb.blockSignals(False)
        for w_, val in ((self.sm_kind, list(SMUDGE_KINDS).index(b.smudge_kind)), (self.sm_len, b.smudge_length * 100),
                        (self.sm_rate, b.colour_rate * 100), (self.sm_all, b.smudge_all)):
            w_.blockSignals(True)
            if w_ is self.sm_kind:
                w_.setCurrentIndex(val)
            elif w_ is self.sm_all:
                w_.setChecked(bool(val))
            else:
                w_.setValue(int(round(val)))
            w_.blockSignals(False)
        self.tip_lbl.setText("Tip: picture" if b.tip is not None else "Tip: round")
        self.brush_note.setText(f"<b>Best for:</b> {b.best_for}<br><b>Weak at:</b> {b.weak_at}")
        self._mode_note()
        self.canvas.brush = b
        self.set_eraser(False) if not b.eraser else None

    def _brush_from_ui(self):
        b = self.brushes[self.brush_i]
        b.size = float(self.s_size.value())
        b.hardness = self.s_hard.value() / 100
        b.opacity = self.s_opac.value() / 100
        b.flow = self.s_flow.value() / 100
        b.spacing = self.s_space.value() / 100
        b.mode = self.mode_cb.currentData()
        b.space = self.space_cb.currentData()
        b.smudge_kind = self.sm_kind.currentData()
        b.smudge_length = self.sm_len.value() / 100
        b.colour_rate = self.sm_rate.value() / 100
        b.smudge_all = self.sm_all.isChecked()
        self._mode_note()
        self.canvas.update()

    def _mode_note(self):
        b = self.brushes[self.brush_i]
        _, best, weak = PAINT_MODES[b.mode]
        sn, sbest, sweak = SPACES[b.space]
        text = (f"{PAINT_MODES[b.mode][0]}: best for {best}. Weak at: {weak}.<br>"
                f"{sn}: best for {sbest}. Weak at: {sweak}.")
        smudging = b.mode == "smudge"
        self.smudge_box.setVisible(smudging)
        if smudging:
            kn, kb, kw = SMUDGE_KINDS[b.smudge_kind]
            text += f"<br>{kn}: best for {kb}. Weak at: {kw}. (Mix colours does not apply to smudging.)"
        self.mode_note.setText(text)

    def load_mypaint(self, paths=None):
        """Add MyPaint brushes to the list (the .myb files are copied into Paint's brushes folder, so they stay)."""
        from ...engine.paint.brush import from_myb
        from ...engine.paint.mypaint import load_myb
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, "MyPaint brushes", "", "MyPaint brushes (*.myb)")
        added, bad = 0, []
        for p in paths or []:
            try:
                b = from_myb(load_myb(p))
            except (OSError, ValueError, KeyError) as err:
                bad.append(f"{os.path.basename(p)}: {err}")
                continue
            except Exception as err:                   # noqa: BLE001
                report_problem(f"Paint: reading MyPaint brush {p}", err)
                bad.append(os.path.basename(p))
                continue
            self._keep_myb(p)
            self.brushes.append(b)
            it = QListWidgetItem(b.name)
            it.setToolTip(f"Best for: {b.best_for}\nWeak at: {b.weak_at}")
            self.brush_list.addItem(it)
            added += 1
        if added:
            self.select_brush(len(self.brushes) - 1)
        msg = f"{added} MyPaint brush{'es' if added != 1 else ''} added."
        if bad:
            msg += " Could not read: " + "; ".join(bad[:3]) + (" …" if len(bad) > 3 else "")
        self._say(msg)
        return added

    def _keep_myb(self, path):
        import shutil
        from ... import paths as P
        d = P.get("PAINT") / "brushes"
        try:
            d.mkdir(parents=True, exist_ok=True)
            dst = d / os.path.basename(path)
            if os.path.abspath(path) != os.path.abspath(dst):
                shutil.copyfile(path, dst)
        except OSError as err:
            log.info("MyPaint brush not kept: %s", err)

    def _add_saved_brushes(self):
        """Brushes saved from the brush editor come back at start; one saved under a preset's name replaces it."""
        from ...engine.paint import brushfile
        try:
            mine = brushfile.saved()
        except OSError as err:
            log.info("saved brushes not read: %s", err)
            return
        names = [b.name for b in self.brushes]
        for b in mine:
            if b.name in names:
                self.brushes[names.index(b.name)] = b
            else:
                self.brushes.append(b)
                names.append(b.name)

    def _filter_brushes(self, *_):
        cat = self.brush_cat.currentData()
        words = self.brush_find.text().lower().split()
        for i, b in enumerate(self.brushes):
            text = f"{b.name} {b.best_for} {b.tags}".lower()
            show = (not cat or b.tags == cat) and all(w in text for w in words)
            self.brush_list.item(i).setHidden(not show)

    def import_brushes(self, paths=None):
        """Brushes from other programs: each becomes a Paint brush, saved beside the app so it stays."""
        from ...engine.paint import brushfile
        from ...engine.paint.imports import EXTS, ImportProblem, import_brushes
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, "Import brushes", "",
                                                    "Brushes (" + " ".join("*" + e for e in EXTS) + ")")
        added, bad = 0, []
        for p in paths or []:
            try:
                found = import_brushes(p)
            except (OSError, ImportProblem) as err:
                bad.append(f"{os.path.basename(p)}: {err}")
                continue
            except Exception as err:                   # noqa: BLE001
                report_problem(f"Paint: importing brushes from {p}", err)
                bad.append(f"{os.path.basename(p)} (an error report was saved)")
                continue
            for b in found:
                try:
                    brushfile.save(b)
                except OSError as err:
                    log.info("imported brush not kept: %s", err)
                self.add_brush(b)
                added += 1
        msg = f"{added} brush{'es' if added != 1 else ''} imported."
        if bad:
            msg += " Could not read: " + "; ".join(bad[:3]) + (" …" if len(bad) > 3 else "")
        self._say(msg)
        return added

    def add_brush(self, b):
        self.brushes.append(b)
        it = QListWidgetItem(b.name)
        it.setToolTip(f"Best for: {b.best_for}\nWeak at: {b.weak_at}")
        self.brush_list.addItem(it)
        self.select_brush(len(self.brushes) - 1)

    def brush_changed(self, i):
        """The brush editor changed brush ``i``: keep the Brush tab and the list in step."""
        b = self.brushes[i]
        it = self.brush_list.item(i)
        if it is not None:
            it.setText(b.name)
            it.setToolTip(f"Best for: {b.best_for}\nWeak at: {b.weak_at}")
        if i == self.brush_i:
            self.select_brush(i)
        self.canvas.update()

    def open_brush_editor(self, show: bool = True):
        from .brusheditor import BrushEditor
        ed = getattr(self, "_brush_editor", None)
        if ed is not None:
            ed.close()
            ed.deleteLater()
        self._brush_editor = BrushEditor(self, self.brush_i)
        if show:
            self._brush_editor.show()
        return self._brush_editor

    def _add_saved_mypaint(self):
        """MyPaint brushes loaded before (kept in paint/brushes beside the app) come back at start."""
        from ...engine.paint.brush import from_myb
        from ...engine.paint.mypaint import load_myb
        for p in self._saved_mypaint():
            try:
                self.brushes.append(from_myb(load_myb(p)))
            except (OSError, ValueError, KeyError) as err:
                log.info("MyPaint brush %s skipped: %s", p, err)

    def _saved_mypaint(self):
        from ... import paths as P
        d = P.get("PAINT") / "brushes"
        return sorted(str(p) for p in d.glob("*.myb")) if d.is_dir() else []

    def load_tip(self):
        from ...engine.paint.brush import load_tip
        path, _ = QFileDialog.getOpenFileName(self, "A picture for the brush tip", self.settings.get("paint_folder", ""),
                                              "Pictures (*.png *.jpg *.jpeg *.tif *.tiff *.webp *.bmp)")
        if not path:
            return
        try:
            tip = load_tip(path)
        except OSError as err:
            self._say(f"That picture could not be read: {err}")
            return
        except Exception as err:                       # noqa: BLE001
            report_problem(f"Paint: brush tip from {path}", err)
            self._say("The tip could not be made; an error report was saved.")
            return
        b = self.brushes[self.brush_i]
        b.tip = tip
        self.tip_lbl.setText(f"Tip: {os.path.basename(path)}")
        self._say("The brush now paints with that picture as its tip.")

    def round_tip(self):
        self.brushes[self.brush_i].tip = None
        self.tip_lbl.setText("Tip: round")

    def set_tool(self, tool):
        from ...engine.paint import select
        self.canvas.tool = tool
        self.canvas._sel = None
        for k, b in self.tool_btns.items():
            b.setChecked(k == tool)
        for k, b in getattr(self, "sel_btns", {}).items():
            b.setChecked(k == tool)
        if tool in ("fill", "gradient"):
            self.side.setCurrentIndex(3)                # the Fill tab
        if tool in select.TOOLS:
            n, best, weak = select.TOOLS[tool]
            self._say(f"{n}: best for {best}. Weak at: {weak}.")
        elif tool == "liquify":
            self._say("Liquify: drag to push paint (the brush size sets how much). Best for: nudging shapes. Weak "
                      "at: big straight moves (use Transform).")
        else:
            self._say({"brush": "Brush", "fill": "Fill: click an area",
                       "gradient": "Gradient: drag across the painting"}.get(tool, tool))

    # ------------------------------------------------------------------ select, edit, transform, filters (P4)
    def _select_tab(self):
        from PySide6.QtWidgets import QGridLayout, QDoubleSpinBox
        from ...engine.paint import adjust, select
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGridLayout()
        self.sel_btns = {}
        for i, (k, (name, best, weak)) in enumerate(list(select.TOOLS.items()) + [("liquify", ("Liquify", "", ""))]):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setToolTip(f"Best for: {best}. Weak at: {weak}." if best else "Push paint around with a brush")
            b.clicked.connect(lambda _=False, k=k: self.set_tool(k))
            g.addWidget(b, i // 3, i % 3)
            self.sel_btns[k] = b
        v.addLayout(g)
        f = QFormLayout()
        self.sel_mode_cb = QComboBox()
        for k, n in select.MODES.items():
            self.sel_mode_cb.addItem(n, k)
        self.sel_mode_cb.currentIndexChanged.connect(lambda: setattr(self.canvas, "sel_mode",
                                                                     self.sel_mode_cb.currentData()))
        f.addRow("New selection", self.sel_mode_cb)
        self.sel_amount = QDoubleSpinBox()
        self.sel_amount.setRange(0.5, 200)
        self.sel_amount.setValue(8)
        f.addRow("Pixels (grow, shrink, feather)", self.sel_amount)
        v.addLayout(f)
        row = None
        for i, (text, fn) in enumerate((("All", self.select_all), ("None", self.select_none),
                                        ("Invert", lambda: self._sel_op("invert")),
                                        ("Grow", lambda: self._sel_op("grow")),
                                        ("Shrink", lambda: self._sel_op("shrink")),
                                        ("Feather", lambda: self._sel_op("feather")),
                                        ("From layer", self.select_from_layer))):
            if i % 4 == 0:
                row = QHBoxLayout()
                v.addLayout(row)
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        self.sel_note = self._note("Nothing selected: everything can be painted.")
        v.addWidget(self.sel_note)
        v.addWidget(QLabel("Edit"))
        row = QHBoxLayout()
        for text, fn, tip in (("Cut", self.cut, "Ctrl+X"), ("Copy", self.copy_sel, "Ctrl+C"),
                              ("Paste", self.paste, "Ctrl+V: a new layer"), ("Delete", self.delete_sel, "Del"),
                              ("Fill", self.fill_sel, "Fill the selection with the painting colour")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        v.addWidget(QLabel("Transform (the selection, or the whole layer)"))
        from ...engine.paint.transform import KINDS as TF_KINDS
        trow = QHBoxLayout()
        for k in ("free", "perspective", "warp", "mesh", "cage"):
            n, best, weak = TF_KINDS[k]
            b = QPushButton(n)
            b.setToolTip(f"On the canvas. Best for: {best}. Weak at: {weak}.")
            b.clicked.connect(lambda _=False, k=k: self.start_transform(k))
            trow.addWidget(b)
        v.addLayout(trow)
        tf = QFormLayout()
        self.tf = {}
        for key, label, lo, hi, val, step in (("tx", "Move across", -5000, 5000, 0, 1), ("ty", "Move down", -5000,
                                              5000, 0, 1), ("scale", "Scale %", 1, 1000, 100, 1),
                                             ("angle", "Rotate", -360, 360, 0, 1), ("shx", "Skew", -2, 2, 0, 0.05)):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setValue(val)
            sp.valueChanged.connect(self._transform_preview)
            tf.addRow(label, sp)
            self.tf[key] = sp
        self.tf_flip = QCheckBox("Flip across")
        self.tf_flip.toggled.connect(self._transform_preview)
        tf.addRow(self.tf_flip)
        v.addLayout(tf)
        row = QHBoxLayout()
        for text, fn in (("Apply", self.transform_apply), ("Cancel", self.transform_cancel)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        v.addWidget(self._note("Transforms work from the pixels kept at the start, so the result is resampled only "
                               "once and stays sharp. Perspective, warp and liquify: Liquify above pushes paint; "
                               "the others from the engine (scripting) for now."))
        v.addWidget(QLabel("Filters"))
        frow = QHBoxLayout()
        self.filter_cb = QComboBox()
        for k, (n, best, weak) in adjust.FILTERS.items():
            self.filter_cb.addItem(n, k)
            self.filter_cb.setItemData(self.filter_cb.count() - 1, f"Best for: {best}. Weak at: {weak}.",
                                       Qt.ItemDataRole.ToolTipRole)
        self.filter_amt = _slider(5, 300, 100)
        fb = QPushButton("Apply")
        fb.clicked.connect(self.apply_filter)
        frow.addWidget(self.filter_cb)
        frow.addWidget(self.filter_amt)
        frow.addWidget(fb)
        v.addLayout(frow)
        v.addStretch(1)
        self._tsession = None
        for seq, fn in (("Return", self.transform_apply), ("Escape", self.transform_cancel),
                        ("Ctrl+A", self.select_all), ("Ctrl+Shift+A", self.select_none), ("Ctrl+X", self.cut),
                        ("Ctrl+C", self.copy_sel), ("Ctrl+V", self.paste), ("Delete", self.delete_sel)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        return w

    def _selection_changed(self):
        from ...engine.paint import select
        sel = self.canvas.doc.selection
        b = None if sel is None else select.bounds(sel)
        self.sel_note.setText("Nothing selected: everything can be painted." if b is None else
                              f"Selected: {b[2] - b[0]} x {b[3] - b[1]} px. Painting, fill, filters and transforms "
                              "stay inside it.")

    def select_all(self):
        d = self.canvas.doc
        self.canvas.doc.selection = None
        self.canvas.set_selection(np.ones((d.height, d.width), np.float32))

    def select_none(self):
        self.canvas.doc.selection = None
        self.canvas.selection_changed.emit()
        self.canvas.update()

    def select_from_layer(self):
        from ...engine.paint import select
        self.canvas.set_selection(select.from_layer(self.canvas.doc, self.canvas.doc.layer))

    def _sel_op(self, op):
        from ...engine.paint import select
        d = self.canvas.doc
        if d.selection is None:
            self._say("Make a selection first.")
            return
        n = self.sel_amount.value()
        d.selection = {"invert": select.invert, "grow": lambda m: select.grow(m, n),
                       "shrink": lambda m: select.shrink(m, n), "feather": lambda m: select.feather(m, n)}[op](
            d.selection)
        self.canvas.selection_changed.emit()
        self.canvas.update()

    def _edit_layer(self):
        try:
            return self.canvas.doc.paint_layer()
        except Exception as err:                        # LayerLocked: say why
            self._say(str(err))
            return None

    def copy_sel(self):
        from ...engine.paint import edit
        lay = self._edit_layer()
        if lay is None:
            return None
        self._clip = edit.copy(self.canvas.doc, lay, self.canvas.doc.selection)
        self._say("Copied." if self._clip else "Nothing to copy here.")
        return self._clip

    def cut(self):
        if self.copy_sel():
            self.delete_sel()

    def paste(self):
        from ...engine.paint import edit
        if not getattr(self, "_clip", None):
            self._say("Nothing copied yet.")
            return
        edit.paste(self.canvas.doc, self._clip, self.canvas.undo)
        self._after_edit()

    def delete_sel(self):
        from ...engine.paint import edit
        lay = self._edit_layer()
        if lay is not None and edit.clear(self.canvas.doc, lay, self.canvas.doc.selection, self.canvas.undo):
            self._after_edit()

    def fill_sel(self):
        from ...engine.paint import edit
        lay = self._edit_layer()
        if lay is not None and edit.fill_selection(self.canvas.doc, lay, self.canvas.doc.selection,
                                                   self.canvas.colour, self.canvas.undo):
            self._after_edit()

    def apply_filter(self):
        from ...engine.paint import edit
        lay = self._edit_layer()
        if lay is None:
            return
        try:
            edit.filter_layer(self.canvas.doc, lay, self.filter_cb.currentData(), self.filter_amt.value() / 100,
                              self.canvas.doc.selection, self.canvas.undo)
        except Exception as err:                        # noqa: BLE001
            report_problem("Paint: filter", err)
            self._say("The filter failed; an error report was saved.")
            return
        self._after_edit()

    def start_transform(self, kind):
        from ...engine.paint.transform import KINDS as TF_KINDS
        if self.canvas.start_transform(kind):
            self._reset_transform_fields()
            n, best, weak = TF_KINDS[kind]
            hint = {"free": "drag inside to move, a corner to scale, outside to rotate",
                    "perspective": "drag the four corners", "warp": "drag the grid's points",
                    "mesh": "drag the grid's points", "cage": "click to place the cage's points around the "
                    "shape, then drag them"}[kind]
            self._say(f"{n}: {hint}. Enter or Apply keeps it, Esc or Cancel undoes it.")

    def _transform_preview(self, *_):
        c = self.canvas
        if c.transform_session is None or getattr(c, "tf_kind", "") != "free":
            if not c.start_transform("free"):
                return
        sc = self.tf["scale"].value() / 100
        c.tf.update(tx=self.tf["tx"].value(), ty=self.tf["ty"].value(), sx=-sc if self.tf_flip.isChecked() else sc,
                    sy=sc, angle=self.tf["angle"].value(), shx=self.tf["shx"].value())
        c.apply_transform()

    def _reset_transform_fields(self):
        for k, sp in self.tf.items():
            sp.blockSignals(True)
            sp.setValue(100 if k == "scale" else 0)
            sp.blockSignals(False)
        self.tf_flip.blockSignals(True)
        self.tf_flip.setChecked(False)
        self.tf_flip.blockSignals(False)

    def transform_apply(self):
        if self.canvas.transform_session is not None:
            self.canvas.finish_transform(True)
            self._reset_transform_fields()
            self.set_tool("brush")
            self._after_edit()

    def transform_cancel(self):
        if self.canvas.transform_session is not None:
            self.canvas.finish_transform(False)
            self._reset_transform_fields()
            self.set_tool("brush")

    def _after_edit(self):
        self.canvas.doc.dirty = True
        self._refresh_layers()
        self.canvas.refresh()
        self._changed()

    def popup(self, where):
        """The pop-up palette (right click, or the pen's barrel button): brushes, recent colours, eraser, tools."""
        m = QMenu(self)
        for i, b in enumerate(self.brushes):
            a = m.addAction(b.name)
            a.setCheckable(True)
            a.setChecked(i == self.brush_i)
            a.setToolTip(f"Best for: {b.best_for}")
            a.triggered.connect(lambda _=False, j=i: (self.select_brush(j), self.set_tool("brush")))
        m.addSeparator()
        for col in self.recent[:10]:
            pm = QPixmap(16, 16)
            pm.fill(QColor(to_hex(col)))
            a = m.addAction(QIcon(pm), to_hex(col))
            a.triggered.connect(lambda _=False, cc=col: self.set_colour(cc))
        m.addSeparator()
        m.addAction("Swap colours (X)", self.swap_colours)
        m.addAction("Eraser on / off (E)", lambda: self.set_eraser(not self.canvas.eraser_mode))
        m.addAction("Fill (F)", lambda: self.set_tool("fill"))
        m.addAction("Gradient (G)", lambda: self.set_tool("gradient"))
        m.addAction("Undo", self.undo)
        m.exec(where)

    def _fill_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("<b>Fill</b>"))
        f = QFormLayout()
        self.f_tol = _slider(0, 100, 10)
        self.f_gap = _slider(0, 20, 0)
        self.f_grow = _slider(0, 6, 1)
        for name, sl in (("Tolerance", self.f_tol), ("Close gaps (px)", self.f_gap), ("Grow under lines", self.f_grow)):
            sl.valueChanged.connect(self._fill_from_ui)
            f.addRow(name, sl)
        self.f_src = QComboBox()
        for k, (name, _, _) in FILL_SOURCES.items():
            self.f_src.addItem(name, k)
        self.f_src.currentIndexChanged.connect(self._fill_from_ui)
        f.addRow("Look at", self.f_src)
        self.f_ref = QComboBox()
        self.f_ref.currentIndexChanged.connect(self._fill_from_ui)
        f.addRow("Reference layer", self.f_ref)
        v.addLayout(f)
        self.fill_note = self._note()
        v.addWidget(self.fill_note)
        v.addWidget(QLabel("<b>Gradient</b>"))
        g = QFormLayout()
        self.g_kind = QComboBox()
        for k, (name, _, _) in GRADIENT_KINDS.items():
            self.g_kind.addItem(name, k)
        self.g_space = QComboBox()
        for k, (name, _, _) in SPACES.items():
            self.g_space.addItem(name, k)
        self.g_space.setCurrentIndex(list(SPACES).index("oklab"))
        self.g_clear = QCheckBox("Painting colour to see-through (instead of to the second colour)")
        for wdg in (self.g_kind, self.g_space):
            wdg.currentIndexChanged.connect(self._fill_from_ui)
        self.g_clear.toggled.connect(self._fill_from_ui)
        g.addRow("Shape", self.g_kind)
        g.addRow("Mix colours", self.g_space)
        v.addLayout(g)
        v.addWidget(self.g_clear)
        self.grad_note = self._note()
        v.addWidget(self.grad_note)
        v.addStretch(1)
        return w

    def _history_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(self._note("Every step of the painting. Click one to go back to it. Painting after going back "
                               "starts a new branch (↳) instead of throwing the later steps away, so two ideas can "
                               "both be kept: click between them to compare."))
        self.hist_list = QListWidget()
        self.hist_list.itemClicked.connect(self._history_clicked)
        self.hist_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.hist_list.customContextMenuRequested.connect(self._history_menu)
        v.addWidget(self.hist_list, 1)
        b = QPushButton("Side by side with the other branch")
        b.setToolTip("Show where you are next to the newest step of another branch. Right-click any step to "
                     "compare with that one instead.")
        b.clicked.connect(lambda: self.compare_steps(None))
        v.addWidget(b)
        self._hist_nodes = []
        return w

    def _history_menu(self, pos):
        item = self.hist_list.itemAt(pos)
        if item is None:
            return
        node = self._hist_nodes[self.hist_list.row(item)]
        m = QMenu(self)
        m.addAction("Side by side with where I am", lambda: self.compare_steps(node))
        m.addAction("Go to this step", lambda: self._history_clicked(item))
        m.exec(self.hist_list.mapToGlobal(pos))

    def compare_steps(self, node=None, show: bool = True):
        """Where you are and another step side by side (the painting is not changed). ``show=False`` builds the
        window without opening it (tests). Returns the window, or None."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        u = self.canvas.undo
        node = node or u.other_tip()
        if node is None or node is u.current:
            self._say("There is no other branch yet. Go back a few steps in History and paint something different, "
                      "then compare the two.")
            return None
        try:
            imgs = [u.picture_at(u.current, 900), u.picture_at(node, 900)]
        except Exception as err:                       # noqa: BLE001
            report_problem("Paint: side by side", err)
            self._say("Side by side failed; an error report was saved.")
            return None
        dlg = QDialog(self)
        dlg.setWindowTitle("Side by side")
        lay = QVBoxLayout(dlg)
        row = QHBoxLayout()
        for img, title in zip(imgs, (f"Where you are: {u.current.name}", f"Other: {node.name}")):
            col = QVBoxLayout()
            lbl = QLabel()
            a = img[..., 3:4]
            rgb = np.where(a > 0, img[..., :3] / np.maximum(a, 1e-8), 1.0) * a + (1 - a)
            arr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
            from PySide6.QtGui import QImage
            qi = QImage(arr.data, arr.shape[1], arr.shape[0], 3 * arr.shape[1], QImage.Format.Format_RGB888).copy()
            pm = QPixmap.fromImage(qi)
            lbl.setPixmap(pm.scaled(560, 560, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation))
            col.addWidget(QLabel(title))
            col.addWidget(lbl)
            row.addLayout(col)
        lay.addLayout(row)
        bb = QDialogButtonBox()
        stay = bb.addButton("Stay where I am", QDialogButtonBox.ButtonRole.RejectRole)
        go = bb.addButton("Go to the other", QDialogButtonBox.ButtonRole.AcceptRole)
        stay.clicked.connect(dlg.reject)
        go.clicked.connect(dlg.accept)
        lay.addWidget(bb)
        dlg._images = imgs
        if not show:
            return dlg
        if dlg.exec() == QDialog.DialogCode.Accepted:
            u.jump(node)
            self._after_history(f"Now at: {node.name}")
        return dlg

    def _refresh_history(self):
        if not hasattr(self, "hist_list"):
            return
        u = self.canvas.undo
        on_path = set(id(n) for n in u.path())
        self.hist_list.clear()
        self._hist_nodes = []
        cur_row = 0
        for node, depth, branch in u.tree():
            mark = "↳ " if branch else ""
            it = QListWidgetItem("    " * depth + mark + node.name)
            if node is u.current:
                it.setForeground(QColor(T.CYAN))
                f = it.font()
                f.setBold(True)
                it.setFont(f)
                cur_row = len(self._hist_nodes)
            elif id(node) not in on_path:
                it.setForeground(QColor(T.DIM))
            self.hist_list.addItem(it)
            self._hist_nodes.append(node)
        self.hist_list.scrollToItem(self.hist_list.item(cur_row))

    def _history_clicked(self, item):
        row = self.hist_list.row(item)
        if 0 <= row < len(self._hist_nodes):
            node = self._hist_nodes[row]
            self.canvas.undo.jump(node)
            self._after_history(f"Back at: {node.name}")

    def _fill_from_ui(self, *_):
        c = self.canvas
        src = self.f_src.currentData()
        c.fill_opts = {"tolerance": self.f_tol.value() / 100, "gap": self.f_gap.value(), "grow": self.f_grow.value(),
                       "source": src, "reference": self.f_ref.currentData()}
        _, best, weak = FILL_SOURCES[src]
        gap = f" {GAP_NOTE[0]}: best for {GAP_NOTE[1]}; weak at: {GAP_NOTE[2]}." if self.f_gap.value() else ""
        self.fill_note.setText(f"Best for: {best}. Weak at: {weak}.{gap}")
        c.grad_opts = {"kind": self.g_kind.currentData(), "space": self.g_space.currentData(),
                       "to_clear": self.g_clear.isChecked()}
        _, gb, gw = GRADIENT_KINDS[c.grad_opts["kind"]]
        _, sb, sw = SPACES[c.grad_opts["space"]]
        self.grad_note.setText(f"Best for: {gb}; mixing best for {sb}. Weak at: {gw}; {sw}.")

    def nudge_size(self, f):
        self.s_size.setValue(max(1, min(1000, int(round(self.s_size.value() * f + (0.5 if f > 1 else -0.5))))))

    def set_eraser(self, on: bool):
        self.canvas.eraser_mode = bool(on)
        self.erase_btn.setChecked(bool(on))
        self._say("Eraser" if on else "Brush")

    def _smoothing_from_ui(self):
        k = self.smooth_cb.currentData()
        self.canvas.smoothing = k
        self.canvas.smooth_strength = self.smooth_s.value() / 100
        _, best, weak = SMOOTH_MODES[k]
        self.smooth_note.setText(f"Best for: {best}. Weak at: {weak}.")
        self.settings.set("paint_smoothing", [k, self.smooth_s.value()])

    def _curve_preset(self, i):
        name = self.curve_cb.itemText(i)
        if name in PRESETS:
            self.curve_box.set_curve(Curve.preset(name))
            self._curve_edited(self.curve_box.curve, preset=name)

    def _curve_edited(self, curve, preset=None):
        self.canvas.pen_curve = curve
        if preset is None:
            self.curve_cb.setCurrentText("Custom")
            self.curve_note.setText("Your own curve. Drag the points; double-click adds one, right-click removes one.")
        else:
            best, weak = PRESET_NOTES[preset]
            self.curve_note.setText(f"Best for: {best}. Weak at: {weak}.")
        self.settings.set("paint_pen_curve", curve.to_list())

    def _load_prefs(self):
        pts = self.settings.get("paint_pen_curve")
        if pts:
            c = Curve(pts)
            self.curve_box.set_curve(c)
            self.canvas.pen_curve = c
            name = next((n for n, p in PRESETS.items() if [list(q) for q in p] == c.to_list()), "Custom")
            self.curve_cb.setCurrentText(name)
        sm = self.settings.get("paint_smoothing", ["basic", 30])
        self.smooth_cb.setCurrentIndex(max(0, list(SMOOTH_MODES).index(sm[0]) if sm[0] in SMOOTH_MODES else 1))
        self.smooth_s.setValue(int(sm[1]))
        self._smoothing_from_ui()
        self.recent = [tuple(c) for c in self.settings.get("paint_recent", [])][:12]

    # ------------------------------------------------------------------ colour
    def _picked(self, pk, c):
        if isinstance(pk, GamutPicker) and self.canvas.doc.space not in ("srgb",):
            from ...engine.paint.colour import oklch_to_space
            self.set_colour(c, from_wheel=pk, working=oklch_to_space(pk.L, pk.C, pk.H, self.canvas.doc.space))
        else:
            self.set_colour(c, from_wheel=pk)

    # ---- working colour space, soft proof, HDR view (P5)
    def _space_box(self, v):
        from ...engine.paint.space import SPACES
        v.addWidget(QLabel("Working colour space"))
        self.space_lbl = self._note("")
        v.addWidget(self.space_lbl)
        row = QHBoxLayout()
        self.wspace_cb = QComboBox()
        for k, (n, best, weak) in SPACES.items():
            self.wspace_cb.addItem(n, k)
            self.wspace_cb.setItemData(self.wspace_cb.count() - 1, f"Best for: {best}. Weak at: {weak}.",
                                      Qt.ItemDataRole.ToolTipRole)
        b = QPushButton("Convert to it")
        b.setToolTip("Convert the whole painting to this working space (the colours stay as they look)")
        b.clicked.connect(lambda: self.convert_space(self.wspace_cb.currentData()))
        row.addWidget(self.wspace_cb)
        row.addWidget(b)
        v.addLayout(row)
        self.exp_s = _slider(-60, 60, 0)
        self.exp_s.setToolTip("Linear (HDR) paintings: viewing exposure, in tenths of a stop. Only the view changes.")
        self.exp_s.valueChanged.connect(self._exposure)
        v.addWidget(QLabel("HDR view exposure"))
        v.addWidget(self.exp_s)
        v.addWidget(QLabel("Soft proof (show it as printed)"))
        prow = QHBoxLayout()
        self.proof_cb = QCheckBox("Soft proof")
        self.proof_cb.toggled.connect(self._proof)
        self.proof_gamut = QCheckBox("Mark colours the paper cannot hold")
        self.proof_gamut.toggled.connect(self._proof)
        pb = QPushButton("Printer profile…")
        pb.clicked.connect(self._pick_proof)
        prow.addWidget(self.proof_cb)
        prow.addWidget(pb)
        v.addLayout(prow)
        v.addWidget(self.proof_gamut)
        self.proof_lbl = self._note("No printer profile chosen. Pick the .icc your printer or print lab gives for "
                                    "the paper (CMYK profiles work too).")
        v.addWidget(self.proof_lbl)
        self._refresh_space()

    def _refresh_space(self):
        from ...engine.paint.space import SPACES
        if not hasattr(self, "space_lbl"):
            return
        sp = self.canvas.doc.space
        n, best, weak = SPACES[sp]
        self.space_lbl.setText(f"This painting: {n}. Best for: {best}. Weak at: {weak}.")
        self.wspace_cb.setCurrentIndex(max(0, self.wspace_cb.findData(sp)))
        self.exp_s.setEnabled(sp == "linear")

    def convert_space(self, dst):
        from ...engine.paint import space
        doc = self.canvas.doc
        if dst == doc.space:
            return
        u = self.canvas.undo
        src = doc.space
        u.layers_changing("Convert colour space")
        space.convert_document(doc, dst)
        u.layers_changed()
        self._refresh_space()
        self.canvas.refresh()
        self._changed()
        self._say(f"Converted from {space.SPACES[src][0]} to {space.SPACES[dst][0]}. The colours look the same; "
                  "Ctrl+Z goes back.")

    def _exposure(self, v):
        self.canvas.exposure = v / 10.0
        self.canvas.refresh()

    def _pick_proof(self):
        path, _ = QFileDialog.getOpenFileName(self, "Printer or paper profile", "", "Colour profiles (*.icc *.icm)")
        if path:
            self._proof_path = path
            self.proof_lbl.setText(f"Printer profile: {os.path.basename(path)}")
            self.proof_cb.setChecked(True)
            self._proof()

    def _proof(self, *_):
        from ...engine.color import display
        path = getattr(self, "_proof_path", None)
        try:
            display.set_proof(path if (self.proof_cb.isChecked() and path) else None,
                              gamut=self.proof_gamut.isChecked())
        except Exception as err:                         # noqa: BLE001
            report_problem("Paint: soft proof", err)
            self._say("Soft proof could not be turned on; an error report was saved.")
            return
        if self.proof_cb.isChecked() and not path:
            self._say("Pick the printer's profile first (Printer profile…).")
        self.canvas.refresh()

    # ------------------------------------------------------------------ swatches
    def _refresh_swatches(self):
        self.sw_cb.blockSignals(True)
        self.sw_cb.clear()
        for sw in self._swatch_sets:
            self.sw_cb.addItem(sw.name + ("" if sw.path or sw in self._swatch_sets[:len(swatch_files.BUILT_IN)]
                                          else "  (not saved)"))
        self._swatch_i = min(self._swatch_i, len(self._swatch_sets) - 1)
        self.sw_cb.setCurrentIndex(self._swatch_i)
        self.sw_cb.blockSignals(False)
        sw = self._swatch_sets[self._swatch_i]
        self.sw_grid.set_colours(sw.colours, sw.names)
        self.sw_note.setText((sw.note + " " if sw.note else "") + "Click a swatch to paint with it; right-click "
                             "removes it from the set.")

    def _swatch_set_chosen(self, i):
        self._swatch_i = i
        self.settings.set("paint_swatch_set", self._swatch_sets[i].name)
        self._refresh_swatches()

    def _swatch_add(self):
        self._swatch_sets[self._swatch_i].add(self.canvas.colour, to_hex(self.canvas.colour))
        self._refresh_swatches()

    def _swatch_remove(self, i):
        self._swatch_sets[self._swatch_i].remove(i)
        self._refresh_swatches()

    def _swatch_new(self):
        name, ok = QInputDialog.getText(self, "New swatch set", "Name:", text="My swatches")
        if ok and name.strip():
            self._swatch_sets.append(swatch_files.SwatchSet(name.strip()))
            self._swatch_i = len(self._swatch_sets) - 1
            self._refresh_swatches()

    def _swatch_from_painting(self):
        n, ok = QInputDialog.getInt(self, "Palette from the painting", "How many colours?", 8, 2, 32)
        if not ok:
            return
        try:
            sw = swatch_files.from_painting(self.canvas.doc, n)
        except Exception as err:                       # noqa: BLE001
            report_problem("Paint: palette from the painting", err)
            self._say("The palette could not be made; an error report was saved.")
            return
        self._swatch_sets.append(sw)
        self._swatch_i = len(self._swatch_sets) - 1
        self._refresh_swatches()
        self._say(f"{len(sw.colours)} colours taken from the painting. Save set keeps them.")

    def _swatch_save(self):
        sw = self._swatch_sets[self._swatch_i]
        if sw.path is None and sw.name in [b.name for b in swatch_files.BUILT_IN]:
            sw.name = sw.name + " (mine)"
        try:
            path = sw.path or str(swatch_files.swatch_dir() / (swatch_files.safe_name(sw.name) + ".gpl"))
            if sw.path and sw.path.lower().endswith(".kpl"):
                path = sw.path[:-4] + ".gpl"
            swatch_files.save_gpl(sw, path)
        except OSError as err:
            self._say(f"The swatch set could not be saved: {err}")
            return
        self._refresh_swatches()
        self._say(f"Saved {os.path.basename(path)} in Paint's swatches folder.")

    def _swatch_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load a swatch set", "", "Swatch sets (*.gpl *.kpl)")
        if not path:
            return
        try:
            sw = swatch_files.load(path)
            sw.path = None                              # a copy is kept in Paint's own folder when saved
            if not sw.colours:
                self._say("That file has no RGB colours in it.")
                return
            swatch_files.save_gpl(sw, str(swatch_files.swatch_dir() / (swatch_files.safe_name(sw.name) + ".gpl")))
        except (OSError, ValueError, KeyError) as err:
            self._say(f"That swatch set could not be read: {err}")
            return
        except Exception as err:                       # noqa: BLE001
            report_problem(f"Paint: loading swatches {path}", err)
            self._say("That swatch set could not be read; an error report was saved.")
            return
        self._swatch_sets.append(sw)
        self._swatch_i = len(self._swatch_sets) - 1
        self._refresh_swatches()
        self._say(f"Loaded {sw.name} ({len(sw.colours)} colours); a copy is kept in Paint's swatches folder.")

    def set_colour(self, rgb, from_wheel=False, working=None):
        """``rgb`` is sRGB (what the pickers show); the brush gets it in the painting's working space. A wide-gamut
        colour from the OKLCH picker comes as ``working`` already, so colours past sRGB are kept."""
        from ...engine.paint.space import from_srgb
        rgb = tuple(float(c) for c in rgb[:3])
        self.canvas.colour = tuple(working) if working is not None else from_srgb(rgb, self.canvas.doc.space)
        for pk in self.pickers:
            if pk is not from_wheel:
                pk.set_rgb(rgb)
        gp = self.pickers[2]
        self.gamut_lbl.setText("Past the sRGB edge: the canvas holds the nearest sRGB colour (same lightness and hue)."
                               if self.picker_stack.currentIndex() == 2 and gp.out_of_srgb() else "")
        self.hex_ed.setText(to_hex(rgb))
        self.fg_sw.setStyleSheet(f"background:{to_hex(rgb)}; border:1px solid {T.LINE};")
        self.bg_sw.setStyleSheet(f"background:{to_hex(self.bg_colour)}; border:1px solid {T.LINE};")
        self.canvas.second_colour = tuple(self.bg_colour)

    def _remember_colour(self):
        c = tuple(round(v, 4) for v in self.canvas.colour)
        if c in self.recent:
            self.recent.remove(c)
        self.recent = [c] + self.recent[:11]
        self.settings.set("paint_recent", [list(x) for x in self.recent])
        while self.recent_row.count():
            it = self.recent_row.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for col in self.recent:
            b = QPushButton()
            b.setFixedSize(22, 22)
            b.setStyleSheet(f"background:{to_hex(col)}; border:1px solid {T.LINE};")
            b.clicked.connect(lambda _=False, cc=col: self.set_colour(cc))
            self.recent_row.addWidget(b)
        self.recent_row.addStretch(1)

    def swap_colours(self):
        fg = self.canvas.colour
        self.bg_colour, new = fg, self.bg_colour
        self.set_colour(new)

    def _hex(self):
        try:
            self.set_colour(from_hex(self.hex_ed.text()))
        except ValueError:
            self.hex_ed.setText(to_hex(self.canvas.colour))

    def _value_view(self, on):
        self.canvas.value_view = on
        self.canvas.refresh()

    # ------------------------------------------------------------------ layers
    def _refresh_layers(self):
        doc = self.canvas.doc
        self._shown_active = doc.active
        tree = self.layer_list
        tree.blockSignals(True)
        tree.clear()
        kids = doc._kids()
        current = None

        def add(parent_item, i):
            nonlocal current
            lay = doc.layers[i]
            flags = ("" if lay.visible else "  (hidden)") + ("  🔒" if lay.locked else "") + \
                    ("  α" if lay.alpha_lock else "")
            if lay.is_group:
                mode = "pass-through" if lay.blend == "pass" else BLEND_MODES.get(lay.blend, ("",))[0]
                text = f"▣ {lay.name}  ({mode}{'' if lay.opacity >= 1 else f', {lay.opacity * 100:.0f} %'}){flags}"
            else:
                extra = ("  ↧clip" if lay.clip else "") + ("  ▭mask" if lay.mask is not None else "")
                text = f"{'◐ ' if lay.is_adjust else ''}{lay.name}{flags}{extra}"
            it = QTreeWidgetItem([text])
            it.setData(0, Qt.UserRole, i)
            if not doc.shown(lay):
                it.setForeground(0, QColor(T.DIM))
            (parent_item.addChild(it) if parent_item is not None else tree.addTopLevelItem(it))
            if i == doc.active:
                current = it
            for j in reversed(kids.get(lay.uid, [])):
                add(it, j)
            if lay.is_group:
                it.setExpanded(not lay.collapsed)
        for i in reversed(kids.get(None, [])):
            add(None, i)
        if current is not None:
            p = current.parent()
            while p is not None:                       # never hide the selected layer inside a closed group
                p.setExpanded(True)
                doc.layers[p.data(0, Qt.UserRole)].collapsed = False
                p = p.parent()
            tree.setCurrentItem(current)
        tree.blockSignals(False)
        lay = doc.layer
        self.l_opac.blockSignals(True)
        self.l_opac.setValue(int(round(lay.opacity * 100)))
        self.l_opac.blockSignals(False)
        self.l_blend.blockSignals(True)
        self.l_blend.clear()
        modes = {**GROUP_BLEND, **BLEND_MODES} if lay.is_group else BLEND_MODES
        for k, (name, _, _) in modes.items():
            self.l_blend.addItem(name, k)
        self.l_blend.setCurrentIndex(max(0, list(modes).index(lay.blend) if lay.blend in modes else 0))
        self.l_blend.blockSignals(False)
        _, best, weak = modes.get(lay.blend, BLEND_MODES["normal"])
        note = f"Best for: {best}. Weak at: {weak}."
        if lay.is_group and lay.blend != "pass":
            note = GROUP_NOTE["isolated"] + " " + note
        self.blend_note.setText(note)
        if hasattr(self, "f_ref"):
            keep = self.f_ref.currentData()
            self.f_ref.blockSignals(True)
            self.f_ref.clear()
            for i in reversed(range(len(doc.layers))):
                if not doc.layers[i].is_group:
                    self.f_ref.addItem("    " * doc.depth(doc.layers[i]) + doc.layers[i].name, i)
            j = self.f_ref.findData(keep)
            self.f_ref.setCurrentIndex(j if j >= 0 else 0)
            self.f_ref.blockSignals(False)
            self._fill_from_ui()
        self.l_vis.setChecked(lay.visible)
        self.l_lock.setChecked(lay.locked)
        self.l_alpha.setChecked(lay.alpha_lock)
        self.l_alpha.setEnabled(not lay.is_group)
        if hasattr(self, "clip_cb"):
            self.clip_cb.blockSignals(True)
            self.clip_cb.setChecked(lay.clip)
            self.clip_cb.blockSignals(False)
        self.bits_cb.blockSignals(True)
        self.bits_cb.setChecked(doc.bits == 16)
        self.bits_cb.blockSignals(False)

    def _layer_selected(self, item, _prev=None):
        if item is None:
            return
        doc = self.canvas.doc
        doc.active = int(item.data(0, Qt.UserRole))
        self._refresh_layers()

    def _fold(self, item, collapsed):
        """Opening and closing a group is a view setting: saved with the painting, not an undo step."""
        lay = self.canvas.doc.layers[int(item.data(0, Qt.UserRole))]
        lay.collapsed = collapsed

    def _layer_change(self, name, fn):
        u = self.canvas.undo
        u.layers_changing(name)
        ok = fn()
        if ok is False:
            u._pending = None
            return False
        u.layers_changed()
        self.canvas.doc.dirty = True
        self._refresh_layers()
        self.canvas.refresh()
        self._changed()
        return True

    def add_layer(self):
        self._layer_change("Add layer", lambda: self.canvas.doc.add_layer())

    def group_layer(self):
        self._layer_change("Group", lambda: self.canvas.doc.add_group())
        self._say("Grouped. Pass-through: the layers inside blend with the picture as before. Pick Isolated "
                  "modes in Blend to treat the group as one picture.")

    def ungroup_layer(self):
        doc = self.canvas.doc
        if not doc.layer.is_group:
            self._say("Select a group to ungroup.")
            return
        g = doc.layer
        changes = g.blend != "pass" and (g.opacity < 1 or g.blend != "normal")
        if self._layer_change("Ungroup", lambda: doc.ungroup()) and changes:
            self._say("Ungrouped. The group's opacity and blend mode are gone, so the picture may look different; "
                      "Ctrl+Z puts the group back.")

    def delete_layer(self):
        doc = self.canvas.doc
        if not self._layer_change("Delete group" if doc.layer.is_group else "Delete layer", lambda: doc.delete()):
            self._say("A painting keeps at least one layer to paint on.")

    def move_layer(self, d):
        self._layer_change("Move layer", lambda: self.canvas.doc.move(d))

    def merge_down(self):
        doc = self.canvas.doc
        if doc.layer.is_group:
            self._layer_change("Merge group", lambda: doc.flatten_group() is not None)
            return
        if not self._layer_change("Merge down", lambda: doc.merge_down()):
            self._say("There is no layer directly below this one in the same group to merge into.")

    def _rename_layer(self, item, _col=0):
        doc = self.canvas.doc
        name, ok = QInputDialog.getText(self, "Rename", "Name:", text=doc.layer.name)
        if ok and name.strip():
            self._layer_change("Rename", lambda: setattr(doc.layer, "name", name.strip()))

    def _layer_opacity_live(self, v):
        self.canvas.doc.layer.opacity = v / 100
        self.canvas.refresh()

    def _layer_props(self):
        doc = self.canvas.doc
        lay = doc.layer
        vals = (self.l_opac.value() / 100, self.l_blend.currentData(), self.l_vis.isChecked(),
                self.l_lock.isChecked(), self.l_alpha.isChecked())

        def go():
            lay.opacity, lay.blend, lay.visible, lay.locked, lay.alpha_lock = vals
        self._layer_change("Layer settings", go)

    def add_filter_layer(self):
        from ...engine.paint.adjust import FILTERS
        k = self.filter_cb.currentData() if hasattr(self, "filter_cb") else "blur"
        amt = self.filter_amt.value() / 100 if hasattr(self, "filter_amt") else 1.0
        self._layer_change("Add filter layer", lambda: self.canvas.doc.add_special(
            "filter", {"kind": k, "amount": amt}, FILTERS[k][0]) is not None)

    def add_fill_layer(self):
        col = list(self.canvas.colour)
        self._layer_change("Add fill layer", lambda: self.canvas.doc.add_special("fill", {"colour": col}, "Fill")
                           is not None)

    def add_file_layer(self, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Link a picture as a layer", "",
                                                  "Pictures (*.png *.jpg *.jpeg *.tif *.tiff *.webp *.ora)")
        if not path:
            return
        from ...engine.paint.tiles import load_picture
        if load_picture(path) is None:
            self._say("That picture could not be read.")
            return
        self._layer_change("Add file layer", lambda: self.canvas.doc.add_special(
            "file", {"path": path, "x": 0, "y": 0}, os.path.basename(path)) is not None)

    def reload_file_layers(self):
        for lay in self.canvas.doc.layers:
            if lay.kind == "file":
                lay._file_px = None
        self.canvas.refresh()
        self._say("Linked pictures reloaded.")

    def add_mask(self):
        from ...engine.paint import edit
        d = self.canvas.doc
        self._layer_change("Add mask", lambda: edit.add_mask(d.layer, d.selection, d) is not None)

    def apply_mask(self):
        from ...engine.paint import edit
        self._layer_change("Apply mask", lambda: edit.apply_mask(self.canvas.doc.layer))

    def delete_mask(self):
        self._layer_change("Delete mask", lambda: setattr(self.canvas.doc.layer, "mask", None))

    def _clip_toggle(self, on):
        lay = self.canvas.doc.layer
        if lay.clip != on:
            self._layer_change("Clipping mask", lambda: setattr(lay, "clip", on))

    def add_adjustment(self):
        k = self.adj_cb.currentData()
        self._layer_change("Add adjustment", lambda: self.canvas.doc.add_adjustment(k) is not None)
        self.edit_adjustment()

    def edit_adjustment(self, show=True):
        """The adjustment layer's settings, live on the canvas."""
        from PySide6.QtWidgets import QDialog, QDoubleSpinBox
        from ...engine.paint.adjust import ADJUSTMENTS
        lay = self.canvas.doc.layer
        if not lay.is_adjust:
            self._say("Select an adjustment layer to change it.")
            return None
        params = lay.adj["params"]
        dlg = QDialog(self)
        dlg.setWindowTitle(ADJUSTMENTS[lay.adj["kind"]][0])
        f = QFormLayout(dlg)
        _, best, weak = ADJUSTMENTS[lay.adj["kind"]]
        f.addRow(self._note(f"Best for: {best}. Weak at: {weak}."))
        ranges = {"black": (0, 1), "white": (0, 1), "gamma": (0.1, 5), "hue": (-180, 180), "sat": (0, 3),
                  "light": (-1, 1), "brightness": (-1, 1), "contrast": (-1, 3), "warmth": (-1, 1),
                  "tint": (-1, 1), "levels": (2, 32)}
        for key, val in params.items():
            if key == "points":
                box = CurveBox(Curve(val))
                box.curve_changed.connect(lambda c, *_: (params.__setitem__("points", c.points),
                                                         self.canvas.refresh()))
                f.addRow("Curve", box)
                continue
            sp = QDoubleSpinBox()
            lo, hi = ranges.get(key, (-10, 10))
            sp.setRange(lo, hi)
            sp.setSingleStep(0.05 if hi - lo <= 6 else 1)
            sp.setValue(float(val))
            sp.valueChanged.connect(lambda x, key=key: (params.__setitem__(key, x), self.canvas.refresh()))
            f.addRow(key.capitalize(), sp)
        if show:
            dlg.show()
        return dlg

    def _relief(self, *_):
        on = self.relief_cb.isChecked()
        self.canvas.light = (float(self.relief_az.value()), float(self.relief_el.value()),
                             self.relief_st.value() / 100.0) if on else None
        self.canvas.refresh()

    def _bits(self, on):
        doc = self.canvas.doc
        if doc.bits != (16 if on else 8):
            doc.bits = 16 if on else 8
            doc.dirty = True
            self._changed()

    def _paper(self, i):
        doc = self.canvas.doc
        doc.background = [(1, 1, 1), (0.96, 0.93, 0.86), (0.5, 0.5, 0.5), (0, 0, 0), None][i]
        doc.dirty = True
        self.canvas.refresh()
        self._changed()

    # ------------------------------------------------------------------ undo
    def undo(self):
        name = self.canvas.undo.undo()
        self._after_history(f"Undone: {name}" if name else "Nothing to undo.")

    def redo(self):
        name = self.canvas.undo.redo()
        self._after_history(f"Redone: {name}" if name else "Nothing to redo.")

    def _after_history(self, msg):
        self._refresh_history()
        self._refresh_layers()
        self._set_doc_space_hook()
        self.canvas.refresh()
        self._changed()
        self._say(msg)

    # ------------------------------------------------------------------ files
    def _title(self):
        doc = self.canvas.doc
        name = os.path.basename(doc.path) if doc.path else "Untitled"
        self.title_lbl.setText(("● " if doc.dirty else "") + name)
        return name

    def _changed(self):
        if getattr(self, "_shown_active", None) != self.canvas.doc.active:
            self._refresh_layers()                    # a stroke on a group went to a layer inside it
        self._refresh_history()
        if self.saver:
            self.saver.changed()
        self._remember_colour() if self.canvas.colour not in self.recent[:1] else None
        self._title()

    def _new_saver(self, title):
        if self.saver:
            self.saver.close()
        self.saver = autosave.AutoSaver(self.canvas.doc, title)

    def _ask_keep(self) -> bool:
        """True when it is fine to replace the open painting."""
        if not self.canvas.doc.dirty:
            return True
        r = QMessageBox.question(self, "Paint", "The painting has changes that are not saved. Save it first?",
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                 QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.save()
        return r == QMessageBox.StandardButton.Discard

    def new(self):
        if not self._ask_keep():
            return
        names = [s[0] for s in SIZES]
        choice, ok = QInputDialog.getItem(self, "New painting", "Size:", names, 0, False)
        if not ok:
            return
        _, w, h = SIZES[names.index(choice)]
        from ...engine.paint.space import SPACES
        snames = [f"{n}: best for {best}" for n, best, _ in SPACES.values()]
        sp, ok = QInputDialog.getItem(self, "New painting", "Working colour space:", snames, 0, False)
        if not ok:
            return
        doc = Document(w, h)
        doc.space = list(SPACES)[snames.index(sp)]
        self._set_doc(doc)

    def _set_doc_space_hook(self):
        if hasattr(self, "space_lbl"):
            self._refresh_space()

    def _set_doc(self, doc):
        self.canvas.set_document(doc)
        self._new_saver(os.path.basename(doc.path) if doc.path else "Untitled")
        self._refresh_layers()
        self._refresh_history()
        self._title()
        self._set_doc_space_hook()

    def open(self):
        if not self._ask_keep():
            return
        start = self.settings.get("paint_folder", "")
        path, _ = QFileDialog.getOpenFileName(self, "Open a painting", start,
                                              "Paintings (*.ora *.psd *.psb *.kra *.exr);;OpenRaster (*.ora);;"
                                              "Photoshop (*.psd *.psb);;Krita (*.kra);;OpenEXR (*.exr)")
        if not path:
            return
        self.open_path(path)

    def open_path(self, path) -> bool:
        """Open .ora, .psd or .kra. A .kra is opened as a new painting (saved as .ora); anything Paint could not
        bring across is said."""
        from ...engine.paint import kra, psd
        notes = []
        low = path.lower()
        try:
            if low.endswith((".psd", ".psb")):
                doc = psd.load(path)
                doc.path = path if low.endswith(".psd") else None    # .psb is opened, not written
            elif low.endswith(".exr"):
                from ...engine.paint import exr
                from ...engine.paint.tiles import Layer
                px = exr.load(path)
                doc = Document(px.shape[1], px.shape[0], None, [Layer(os.path.basename(path))], 0, None)
                doc.space = "linear"
                doc.layer.write(0, 0, px)
                notes = ["Opened as a linear (HDR) painting; save it as .ora, and export EXR to keep values above "
                         "white."]
            elif low.endswith(".kra"):
                doc, notes = kra.load(path)
                doc.path = None                           # Paint does not write .kra: save as .ora or .psd
            else:
                doc = ora.load(path)
        except (OSError, KeyError, ValueError, zipfile_error()) as err:
            self._say(f"That file could not be opened ({err}).")
            return False
        except Exception as err:                       # noqa: BLE001
            report_problem(f"Paint: opening {path}", err)
            self._say("Opening failed; an error report was saved.")
            return False
        self.settings.set("paint_folder", os.path.dirname(path))
        self._set_doc(doc)
        msg = f"Opened {os.path.basename(path)}."
        if notes:
            msg += " " + " ".join(notes[:3]) + (" …" if len(notes) > 3 else "")
        if low.endswith(".kra"):
            msg += " Save it as .ora or .psd to keep your changes."
        self._say(msg)
        return True

    def save(self) -> bool:
        doc = self.canvas.doc
        if not doc.path:
            return self.save_as()
        return self._write(doc.path)

    def save_as(self) -> bool:
        start = os.path.join(self.settings.get("paint_folder", ""), self._title() if self.canvas.doc.path else
                             "Painting.ora")
        path, flt = QFileDialog.getSaveFileName(self, "Save the painting", start,
                                                "OpenRaster: Paint, Krita, GIMP, MyPaint (*.ora);;"
                                                "Photoshop, layers kept (*.psd)")
        if not path:
            return False
        if not path.lower().endswith((".ora", ".psd")):
            path += ".psd" if "psd" in (flt or "") else ".ora"
        return self._write(path)

    def _write(self, path) -> bool:
        doc = self.canvas.doc
        try:
            if path.lower().endswith(".psd"):
                from ...engine.paint import psd
                psd.save(doc, path)
            else:
                ora.save(doc, path)
        except PermissionError:
            self._say("That folder cannot be written to (read-only). Choose another place.")
            return False
        except OSError as err:
            self._say(f"The painting could not be saved there: {err}")
            return False
        except Exception as err:                       # noqa: BLE001
            report_problem(f"Paint: saving {path}", err)
            self._say("Saving failed; an error report was saved. Your autosave copy is still safe.")
            return False
        doc.path, doc.dirty = path, False
        self.settings.set("paint_folder", os.path.dirname(path))
        if self.saver:
            self.saver.title = os.path.basename(path)
        self._title()
        self._say(f"Saved {os.path.basename(path)}")
        return True

    def export_options(self):
        """The export choices: format, colour profile, bits, quality. Returns a dict, or None when cancelled."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QSpinBox
        from ...engine.paint.export import FORMATS, PROFILES
        dlg = QDialog(self)
        dlg.setWindowTitle("Export")
        f = QFormLayout(dlg)
        fmt, prof, bits = QComboBox(), QComboBox(), QComboBox()
        for k, (name, ext, _, _) in FORMATS.items():
            fmt.addItem(f"{name} ({ext})", k)
        for k, (name, _, _) in PROFILES.items():
            prof.addItem(name, k)
        bits.addItem("8-bit", 8)
        bits.addItem("16-bit (smooth gradients, larger)", 16)
        q = QSpinBox()
        q.setRange(50, 100)
        q.setValue(int(self.settings.get("paint_export_quality", 92) or 92))
        lossless = QCheckBox("Lossless (WebP: exact, larger)")
        relief = QCheckBox("With the paint's relief, lit as on the canvas")
        relief.setChecked(self.canvas.light is not None)
        relief.setEnabled(any(l.height for l in self.canvas.doc.layers))
        note = self._note("")
        for cb, key in ((fmt, "paint_export_format"), (prof, "paint_export_profile"), (bits, "paint_export_bits")):
            j = cb.findData(self.settings.get(key))
            if j >= 0:
                cb.setCurrentIndex(j)

        def upd(*_):
            k = fmt.currentData()
            _, _, best, weak = FORMATS[k]
            pn, pb, pw = PROFILES[prof.currentData()]
            bits.setEnabled(k in ("png", "tiff"))
            q.setEnabled(k in ("jpeg", "webp") and not (k == "webp" and lossless.isChecked()))
            lossless.setEnabled(k == "webp")
            note.setText(f"{FORMATS[k][0]}. Best for: {best}. Weak at: {weak}.\n{pn}. Best for: {pb}. Weak at: "
                         f"{pw}. The canvas is sRGB: other profiles hold the same colours, described for that "
                         "space.")
        for w_ in (fmt, prof, bits):
            w_.currentIndexChanged.connect(upd)
        lossless.toggled.connect(upd)
        f.addRow("Format", fmt)
        f.addRow("Colour profile", prof)
        f.addRow("Bits", bits)
        f.addRow("Quality", q)
        f.addRow("", lossless)
        f.addRow("", relief)
        f.addRow(note)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        f.addRow(bb)
        upd()
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        opts = {"fmt": fmt.currentData(), "space": prof.currentData(), "bits": bits.currentData(),
                "quality": q.value(), "lossless": lossless.isChecked(),
                "light": (self.canvas.light or (120.0, 35.0, 0.8)) if relief.isChecked() else None}
        for key, val in (("paint_export_format", opts["fmt"]), ("paint_export_profile", opts["space"]),
                         ("paint_export_bits", opts["bits"]), ("paint_export_quality", opts["quality"])):
            self.settings.set(key, val)
        return opts

    def export(self, opts: dict | None = None):
        from ...engine.io.export import free_target
        from ...engine.paint import export as pexport
        doc = self.canvas.doc
        stem = os.path.splitext(os.path.basename(doc.path))[0] if doc.path else "Painting"
        folder = os.path.dirname(doc.path) if doc.path else self.settings.get("paint_folder", "")
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Export into which folder?")
            if not folder:
                return None
        opts = opts or self.export_options()
        if not opts:
            return None
        ext = pexport.FORMATS[opts["fmt"]][1]
        target = free_target(os.path.join(folder, f"{stem}_Studio{ext}"), source=doc.path)
        try:
            pexport.export(doc, target, **opts)
        except OSError as err:
            self._say(f"The export could not be written: {err}")
            return None
        except Exception as err:                       # noqa: BLE001
            report_problem(f"Paint: exporting {target}", err)
            self._say("Export failed; an error report was saved.")
            return None
        self._say(f"Exported {os.path.basename(target)} ({pexport.PROFILES[opts['space']][0]}).")
        return target

    # ------------------------------------------------------------------ autosave and recovery
    def _autosave(self):
        if self.saver and self.saver.due(60):
            try:
                self.saver.save_now()
            except OSError as err:
                self._say(f"Autosave could not write its copy: {err}")
            except Exception as err:                   # noqa: BLE001
                report_problem("Paint: autosave", err)

    def showEvent(self, e):
        super().showEvent(e)
        if not self._asked_recovery:
            self._asked_recovery = True
            QTimer.singleShot(300, self.offer_recovery)

    def offer_recovery(self):
        found = [m for m in autosave.recoverable() if not (self.saver and m.get("session") == self.saver.session)]
        if not found:
            return
        m = found[0]
        box = QMessageBox(QMessageBox.Icon.Question, "Paint",
                          f"Paint found a painting that was not closed properly (\"{m.get('title')}\", autosaved "
                          f"{m.get('saved')}). Open it?" + (f"\n\n{len(found) - 1} more are waiting." if len(found) > 1
                                                             else ""))
        open_b = box.addButton("Open it", QMessageBox.ButtonRole.AcceptRole)
        drop_b = box.addButton("Throw it away", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is open_b:
            try:
                doc = autosave.recover(m)
            except Exception as err:                   # noqa: BLE001
                report_problem("Paint: recovering an autosave", err)
                self._say("The recovery copy could not be opened; an error report was saved.")
                return
            self._set_doc(doc)
            doc.dirty = True
            autosave.discard(m)
            self.saver.changed()
            self._title()
            self._say("Recovered. Save it to keep it.")
        elif box.clickedButton() is drop_b:
            autosave.discard(m)

    def on_quit(self):
        """At exit: a saved painting's recovery copy is removed; an unsaved one is written and kept for next time."""
        if not self.saver:
            return
        try:
            if self.canvas.doc.dirty:
                self.saver.changed()
                self.saver.save_now()
            else:
                self.saver.close()
        except Exception as err:                       # noqa: BLE001
            report_problem("Paint: saving on exit", err)

    # ------------------------------------------------------------------ misc
    def toggle_canvas_only(self):
        on = self.side.isVisible()
        self.side.setVisible(not on)
        self.bar.setVisible(not on)

    def _sample(self, info):
        self.tablet.seen(info)
        if info.get("kind") == "tablet":
            self.curve_box.live = info["pressure"] if info["pressure"] > 0 else None
            self.curve_box.update()

    # ------------------------------------------------------------------ keys (pen and tablet keys too)
    def _actions(self):
        c = self.canvas
        return {
            "undo": self.undo, "redo": self.redo, "brush": lambda: self.set_tool("brush"),
            "fill": lambda: self.set_tool("fill"), "gradient": lambda: self.set_tool("gradient"),
            "eraser": lambda: self.set_eraser(not c.eraser_mode), "swap": self.swap_colours,
            "smaller": lambda: self.nudge_size(1 / 1.15), "bigger": lambda: self.nudge_size(1.15),
            "popup": lambda: self.popup(QCursor.pos()), "pick": self._pick_here,
            "canvas_only": self.toggle_canvas_only, "rotate_left": lambda: c.rotate_by(-15),
            "rotate_right": lambda: c.rotate_by(15), "rotate_reset": lambda: c.rotate_by(None),
            "mirror": c.toggle_mirror, "fit": c.fit, "zoom_100": lambda: c.set_zoom(1.0),
            "zoom_in": lambda: c.set_zoom(c.zoom * 1.25), "zoom_out": lambda: c.set_zoom(c.zoom / 1.25),
            "new_layer": self.add_layer, "values": self._toggle_values,
            "save": self.save, "save_as": self.save_as, "open": self.open,
        }

    def _apply_keys(self):
        for sc in self._shortcuts:
            sc.setEnabled(False)
            sc.deleteLater()
        self._shortcuts = []
        acts = self._actions()
        for a, keys in self.keys.items():
            for k in keys:
                if a not in acts or not k:
                    continue
                sc = QShortcut(QKeySequence(k), self)
                sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                sc.activated.connect(acts[a])
                self._shortcuts.append(sc)

    def _pick_here(self):
        from PySide6.QtCore import QPointF
        d = self.canvas.to_doc(QPointF(self.canvas.mapFromGlobal(QCursor.pos())))
        x, y = d.x(), d.y()
        from ...engine.paint.colour import pick
        col = pick(self.canvas.doc, x, y, 1)
        if col is not None:
            self.set_colour(col)

    def _toggle_values(self):
        self.value_cb.setChecked(not self.canvas.value_view)     # its toggled signal does the rest

    def _keys_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(self._note(
            "Tablet keys and pen buttons: set them to send a key in the tablet's own settings (Huion's driver, or "
            "OpenTabletDriver on Linux), then click an action here and press that tablet key. Each action can have "
            "two keys, one on the keyboard and one on the tablet. Holding Space pans; the pen's barrel button opens "
            "the pop-up palette."))
        self.keys = keymap.load()
        self.keys_form = QFormLayout()
        v.addLayout(self.keys_form)
        self._key_btns = {}
        for a, (name, what, _) in keymap.ACTIONS.items():
            b = QPushButton()
            b.setToolTip(f"{name}: {what}. Click, then press a key (or a tablet key). Esc: cancel. Backspace: "
                         "no key." if what else f"{name}. Click, then press a key.")
            b.clicked.connect(lambda _=False, a=a: self._capture_key(a))
            self._key_btns[a] = b
            self.keys_form.addRow(name, b)
        self.keys_note = self._note("")
        v.addWidget(self.keys_note)
        reset = QPushButton("All keys back to how they came")
        reset.clicked.connect(self._reset_keys)
        v.addWidget(reset)
        v.addStretch(1)
        self._refresh_keys()
        return w

    def _refresh_keys(self):
        for a, b in self._key_btns.items():
            b.setText(",  ".join(self.keys.get(a, [])) or "(no key)")

    def _capture_key(self, action):
        from PySide6.QtWidgets import QDialog
        name = keymap.ACTIONS[action][0]
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Key for {name}")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"Press the key, or the tablet key, for \u201c{name}\u201d.\n"
                             "Esc cancels. Backspace leaves it with no key."))
        got = {}

        def kp(e):
            k = e.key()
            if k in (Qt.Key.Key_Shift, Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_Meta, Qt.Key.Key_AltGr):
                return
            if k == Qt.Key.Key_Escape and not e.modifiers():
                dlg.reject()
                return
            if k == Qt.Key.Key_Backspace and not e.modifiers():
                got["clear"] = True
            else:
                got["key"] = QKeySequence(e.keyCombination().toCombined()).toString()
            dlg.accept()
        dlg.keyPressEvent = kp
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.set_key(action, got.get("key"))

    def set_key(self, action, key):
        """Give ``key`` to ``action`` (None: no key). A keyboard key goes in the first place, an extra key (a tablet
        key, or a second keyboard key) in the second."""
        if key is None:
            keymap.clear(self.keys, action)
            msg = f"{keymap.ACTIONS[action][0]} has no key now."
        else:
            have = self.keys.get(action, [])
            slot = 0 if not have else (len(have) if len(have) < keymap.MAX_KEYS else 1)
            taken = keymap.assign(self.keys, action, key, slot)
            msg = f"{key} now does {keymap.ACTIONS[action][0]}."
            if taken:
                msg += f" It no longer does {keymap.ACTIONS[taken][0]}."
        self._save_keys()
        self.keys_note.setText(msg)
        self._say(msg)

    def _reset_keys(self):
        self.keys = keymap.defaults()
        self._save_keys()
        self.keys_note.setText("Every key is back to how it came.")

    def _save_keys(self):
        try:
            keymap.save(self.keys)
        except OSError as err:
            self._say(f"The keys could not be saved: {err}")
        self._apply_keys()
        self._refresh_keys()

    def keyPressEvent(self, e):
        if e.text():
            self.tablet.key(QKeySequence(e.keyCombination().toCombined()).toString() or e.text())
        super().keyPressEvent(e)

    def _say(self, text):
        self.status.emit(text)
