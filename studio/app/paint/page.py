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
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox,
                               QPushButton, QScrollArea, QSlider, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from ...engine.paint import autosave, ora
from ...engine.paint.brush import MODES as PAINT_MODES, presets
from ...engine.paint.colour import SPACES, from_hex, to_hex
from ...engine.paint.fill import FILL_SOURCES, GAP_NOTE, GRADIENT_KINDS
from ...engine.paint.curves import PRESET_NOTES, PRESETS, Curve
from ...engine.paint.smoothing import MODES as SMOOTH_MODES
from ...engine.paint.tiles import BLEND_MODES, Document
from .. import theme as T
from ..common import Settings, report_problem
from .canvas import PaintCanvas
from .tabletcheck import TabletCheck
from .widgets import ColourWheel, CurveBox

log = logging.getLogger("studio.paint")

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
        self.brushes = presets()
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
        self._load_prefs()
        self.select_brush(self.brush_i)
        self.set_tool("brush")
        self.set_colour((0.1, 0.12, 0.2))
        self._new_saver("Untitled")
        self._refresh_layers()
        for seq, fn in (("Ctrl+Z", self.undo), ("Ctrl+Shift+Z", self.redo), ("Ctrl+Y", self.redo),
                        ("B", lambda: self.set_tool("brush")), ("F", lambda: self.set_tool("fill")),
                        ("G", lambda: self.set_tool("gradient")), ("E", lambda: self.set_eraser(not c.eraser_mode)),
                        ("X", self.swap_colours), ("[", lambda: self.nudge_size(1 / 1.15)),
                        ("]", lambda: self.nudge_size(1.15)), ("Tab", self.toggle_canvas_only),
                        ("Ctrl+S", self.save), ("Ctrl+Shift+S", self.save_as), ("Ctrl+O", self.open)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
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
        self.brush_list = QListWidget()
        self.brush_list.setMaximumHeight(190)
        for b in self.brushes:
            it = QListWidgetItem(b.name)
            it.setToolTip(f"Best for: {b.best_for}\nWeak at: {b.weak_at}")
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
        self.wheel = ColourWheel()
        self.wheel.colour_changed.connect(lambda c: self.set_colour(c, from_wheel=True))
        v.addWidget(self.wheel)
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
        v.addWidget(QLabel("Recent colours"))
        self.recent_row = QHBoxLayout()
        v.addLayout(self.recent_row)
        v.addWidget(self._note("Alt + click on the painting picks its colour. The Values button on the bar shows "
                               "the painting in greys."))
        v.addStretch(1)
        return w

    def _layers_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.layer_list = QListWidget()
        self.layer_list.currentRowChanged.connect(self._layer_selected)
        self.layer_list.itemDoubleClicked.connect(self._rename_layer)
        v.addWidget(self.layer_list)
        row = QHBoxLayout()
        for text, fn, tip in (("+", self.add_layer, "New layer"), ("−", self.delete_layer, "Delete layer"),
                              ("▲", lambda: self.move_layer(1), "Move up"),
                              ("▼", lambda: self.move_layer(-1), "Move down"),
                              ("Merge", self.merge_down, "Merge into the layer below")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
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
        v.addWidget(self._note("Double-click a layer to rename it. The top of the list is the top layer."))
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
        self._mode_note()
        self.canvas.update()

    def _mode_note(self):
        b = self.brushes[self.brush_i]
        _, best, weak = PAINT_MODES[b.mode]
        sn, sbest, sweak = SPACES[b.space]
        self.mode_note.setText(f"{PAINT_MODES[b.mode][0]}: best for {best}. Weak at: {weak}.<br>"
                               f"{sn}: best for {sbest}. Weak at: {sweak}.")

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
        self.canvas.tool = tool
        for k, b in self.tool_btns.items():
            b.setChecked(k == tool)
        if tool in ("fill", "gradient"):
            self.side.setCurrentIndex(3)                # the Fill tab
        self._say({"brush": "Brush", "fill": "Fill: click an area", "gradient": "Gradient: drag across the "
                   "painting"}[tool])

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
    def set_colour(self, rgb, from_wheel=False):
        rgb = tuple(float(c) for c in rgb[:3])
        self.canvas.colour = rgb
        if not from_wheel:
            self.wheel.set_rgb(rgb)
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
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for lay in reversed(doc.layers):
            flags = ("" if lay.visible else "  (hidden)") + ("  🔒" if lay.locked else "") + \
                    ("  α" if lay.alpha_lock else "")
            self.layer_list.addItem(f"{lay.name}{flags}")
        self.layer_list.setCurrentRow(len(doc.layers) - 1 - doc.active)
        self.layer_list.blockSignals(False)
        lay = doc.layer
        self.l_opac.blockSignals(True)
        self.l_opac.setValue(int(round(lay.opacity * 100)))
        self.l_opac.blockSignals(False)
        self.l_blend.setCurrentIndex(list(BLEND_MODES).index(lay.blend))
        _, best, weak = BLEND_MODES[lay.blend]
        self.blend_note.setText(f"Best for: {best}. Weak at: {weak}.")
        if hasattr(self, "f_ref"):
            keep = self.f_ref.currentData()
            self.f_ref.blockSignals(True)
            self.f_ref.clear()
            for i in reversed(range(len(doc.layers))):
                self.f_ref.addItem(doc.layers[i].name, i)
            j = self.f_ref.findData(keep)
            self.f_ref.setCurrentIndex(j if j >= 0 else 0)
            self.f_ref.blockSignals(False)
            self._fill_from_ui()
        self.l_vis.setChecked(lay.visible)
        self.l_lock.setChecked(lay.locked)
        self.l_alpha.setChecked(lay.alpha_lock)

    def _layer_selected(self, row):
        if row < 0:
            return
        doc = self.canvas.doc
        doc.active = len(doc.layers) - 1 - row
        self._refresh_layers()

    def _layer_change(self, name, fn):
        u = self.canvas.undo
        u.layers_changing(name)
        fn()
        u.layers_changed()
        self.canvas.doc.dirty = True
        self._refresh_layers()
        self.canvas.refresh()
        self._changed()

    def add_layer(self):
        self._layer_change("Add layer", lambda: self.canvas.doc.add_layer())

    def delete_layer(self):
        doc = self.canvas.doc
        if len(doc.layers) == 1:
            self._say("A painting keeps at least one layer.")
            return

        def go():
            del doc.layers[doc.active]
            doc.active = max(0, doc.active - 1)
        self._layer_change("Delete layer", go)

    def move_layer(self, d):
        doc = self.canvas.doc
        j = doc.active + d
        if not 0 <= j < len(doc.layers):
            return

        def go():
            doc.layers[doc.active], doc.layers[j] = doc.layers[j], doc.layers[doc.active]
            doc.active = j
        self._layer_change("Move layer", go)

    def merge_down(self):
        doc = self.canvas.doc
        if doc.active == 0:
            self._say("The bottom layer has nothing below it to merge into.")
            return

        def go():
            from ...engine.paint.tiles import TILE, blend
            top, low = doc.layers[doc.active], doc.layers[doc.active - 1]
            for key in set(top.tiles) | set(low.tiles):
                a = low.tiles.get(key)
                b = top.tiles.get(key)
                a = np.zeros((TILE, TILE, 4), np.float32) if a is None else a
                if b is not None and top.visible:
                    a = blend(a * low.opacity, b * top.opacity, top.blend)
                    low.tiles[key] = a.astype(np.float32)
            low.opacity = 1.0 if top.visible else low.opacity
            del doc.layers[doc.active]
            doc.active -= 1
        self._layer_change("Merge down", go)

    def _rename_layer(self, item):
        doc = self.canvas.doc
        name, ok = QInputDialog.getText(self, "Rename layer", "Name:", text=doc.layer.name)
        if ok and name.strip():
            self._layer_change("Rename layer", lambda: setattr(doc.layer, "name", name.strip()))

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
        self._refresh_layers()
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
        self._set_doc(Document(w, h))

    def _set_doc(self, doc):
        self.canvas.set_document(doc)
        self._new_saver(os.path.basename(doc.path) if doc.path else "Untitled")
        self._refresh_layers()
        self._title()

    def open(self):
        if not self._ask_keep():
            return
        start = self.settings.get("paint_folder", "")
        path, _ = QFileDialog.getOpenFileName(self, "Open a painting", start, "OpenRaster (*.ora)")
        if not path:
            return
        try:
            doc = ora.load(path)
        except (OSError, KeyError, ValueError) as err:
            self._say(f"That file could not be opened as an OpenRaster painting ({err}).")
            return
        except Exception as err:                       # noqa: BLE001
            report_problem(f"Paint: opening {path}", err)
            self._say("Opening failed; an error report was saved.")
            return
        self.settings.set("paint_folder", os.path.dirname(path))
        self._set_doc(doc)
        self._say(f"Opened {os.path.basename(path)}")

    def save(self) -> bool:
        doc = self.canvas.doc
        if not doc.path:
            return self.save_as()
        return self._write(doc.path)

    def save_as(self) -> bool:
        start = os.path.join(self.settings.get("paint_folder", ""), self._title() if self.canvas.doc.path else
                             "Painting.ora")
        path, _ = QFileDialog.getSaveFileName(self, "Save the painting", start, "OpenRaster (*.ora)")
        if not path:
            return False
        if not path.lower().endswith(".ora"):
            path += ".ora"
        return self._write(path)

    def _write(self, path) -> bool:
        doc = self.canvas.doc
        try:
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

    def export(self):
        from PIL import Image
        from ...engine.io.export import free_target
        doc = self.canvas.doc
        stem = os.path.splitext(os.path.basename(doc.path))[0] if doc.path else "Painting"
        folder = os.path.dirname(doc.path) if doc.path else self.settings.get("paint_folder", "")
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Export into which folder?")
            if not folder:
                return
        kinds = ["PNG (keeps see-through areas)", "TIFF", "JPEG (for the web)"]
        kind, ok = QInputDialog.getItem(self, "Export", "Format:", kinds, 0, False)
        if not ok:
            return
        ext = {0: ".png", 1: ".tif", 2: ".jpg"}[kinds.index(kind)]
        target = free_target(os.path.join(folder, f"{stem}_Studio{ext}"), source=doc.path)
        try:
            px = doc.composite()
            a = px[..., 3:4]
            rgb = np.where(a > 0, px[..., :3] / np.maximum(a, 1e-8), 0)
            arr = (np.clip(np.concatenate([rgb, a], -1), 0, 1) * 255 + 0.5).astype(np.uint8)
            im = Image.fromarray(arr, "RGBA")
            if ext == ".jpg":
                im = im.convert("RGB")
                im.save(target, quality=92)
            else:
                im.save(target)
        except OSError as err:
            self._say(f"The export could not be written: {err}")
            return
        except Exception as err:                       # noqa: BLE001
            report_problem(f"Paint: exporting {target}", err)
            self._say("Export failed; an error report was saved.")
            return
        self._say(f"Exported {os.path.basename(target)}")

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

    def keyPressEvent(self, e):
        if e.text():
            self.tablet.key(QKeySequence(e.keyCombination().toCombined()).toString() or e.text())
        super().keyPressEvent(e)

    def _say(self, text):
        self.status.emit(text)
