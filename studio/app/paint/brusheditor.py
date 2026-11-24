"""The brush editor with a live preview stroke (BUILD_PLAN_PAINT P2; Moko's best idea).

Every change repaints a real stroke with the real engine at the top of the window (light touch to full pressure and
back), and the brush on the canvas changes at the same time. Sections: Tip, Paint, Dynamics (which pen inputs drive
size, opacity and flow, each through its own curve), Texture (paper grain, a second tip, scatter and count) and
Colour (jitter). "Save as new brush" keeps it in paint/brushes beside the app; "Back to how it came" undoes every
change made since the window opened.
"""
from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout, QGridLayout, QHBoxLayout, QInputDialog,
                               QLabel, QLineEdit, QPushButton, QSlider, QTabWidget, QVBoxLayout, QWidget)

from ...engine.paint import brushfile, texture
from ...engine.paint.brush import INPUT_NOTES, INPUTS, MODES, SMUDGE_KINDS
from ...engine.paint.colour import SPACES
from ...engine.paint.curves import Curve
from ..common import report_problem
from .widgets import CurveBox

TARGETS = {"size": "Size", "opacity": "Opacity", "flow": "Flow"}

# attribute -> (label, low, high, scale (slider units per 1), tooltip)
TIP = [("size", "Size", 1, 1000, 1, "diameter in canvas pixels at full input"),
       ("min_size", "Smallest size", 0, 100, 100, "size at the lightest touch, as a share of Size"),
       ("hardness", "Hardness", 0, 100, 100, "1 = a hard edge, 0 = very soft"),
       ("roundness", "Roundness", 5, 100, 100, "1 = a circle; less flattens the tip into an ellipse"),
       ("angle", "Angle", -90, 90, 1, "the tip's angle in degrees (ellipses and picture tips)"),
       ("spacing", "Spacing", 2, 200, 100, "the gap between dabs, as a share of the dab's size")]
PAINT = [("opacity", "Opacity", 1, 100, 100, "the most paint one stroke can lay down"),
         ("flow", "Flow", 1, 100, 100, "how much each dab adds"),
         ("smudge_length", "Smudge length", 0, 100, 100, "smudge: how far carried paint goes"),
         ("colour_rate", "Colour rate", 0, 100, 100, "smudge: how much painting colour is mixed in")]
TEXTURE = [("grain_strength", "Grain strength", 0, 100, 100, "0 = no grain, 1 = the paper's full tooth"),
           ("grain_scale", "Grain size", 20, 400, 100, "how big the paper's texture is on the canvas"),
           ("dual_size", "Second tip size", 2, 100, 100, "the second tip as a share of the dab"),
           ("dual_count", "Second tip marks", 1, 16, 1, "marks of the second tip in each dab"),
           ("dual_scatter", "Second tip spread", 0, 100, 100, "how far the marks spread inside the dab"),
           ("scatter", "Scatter", 0, 200, 100, "dabs thrown sideways at random, as a share of their size"),
           ("count", "Count", 1, 16, 1, "dabs at each spot (with scatter: a spray)")]
MEDIA = [("impasto", "Thickness (impasto)", 0, 100, 100, "how thick the paint is: its relief shows under the light "
                                                           "(Layers > Paint relief)"),
         ("water", "Water", 5, 100, 100, "watercolour: how wet each stroke is (wetter flows further and dries "
                                         "slower)"),
         ("pigment", "Pigment", 0, 100, 100, "watercolour: colour strength; 0 = clean water"),
         ("lift", "Lift", 0, 100, 100, "watercolour: clean water takes settled paint off the paper"),
         ("granulation", "Granulation", 0, 100, 100, "watercolour: pigment settling in the paper's valleys"),
         ("edge_darkening", "Dark edges", 0, 100, 100, "watercolour: pigment carried to the edges as it dries"),
         ("bristles", "Bristles", 3, 80, 1, "bristle brushes: how many bristles"),
         ("paint_load", "Paint load", 5, 400, 100, "bristle brushes: how long the paint lasts along a stroke"),
         ("pickup", "Pick-up", 0, 100, 100, "bristle brushes: how much colour the bristles take from the canvas")]
COLOUR = [("jitter_h", "Hue jitter", 0, 100, 100, "each dab's hue moves at random by up to this much"),
          ("jitter_s", "Saturation jitter", 0, 100, 100, ""),
          ("jitter_v", "Value jitter", 0, 100, 100, "")]


class BrushEditor(QDialog):
    def __init__(self, page, index: int):
        super().__init__(page)
        self.page, self.index = page, index
        self.b = page.brushes[index]
        self.original = self.b.copy()
        self.original.dynamics = {t: {i: Curve(c.points) for i, c in ins.items()}
                                  for t, ins in self.b.dynamics.items()}
        self.setWindowTitle(f"Brush editor: {self.b.name}")
        self.setModal(False)
        self.resize(460, 720)
        v = QVBoxLayout(self)
        self.preview_lbl = QLabel()
        self.preview_lbl.setMinimumHeight(120)
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setToolTip("A real stroke with this brush: a light touch rising to full pressure and "
                                    "easing off again")
        v.addWidget(self.preview_lbl)
        f = QFormLayout()
        self.name_ed = QLineEdit(self.b.name)
        self.name_ed.editingFinished.connect(lambda: self._set("name", self.name_ed.text().strip() or self.b.name))
        f.addRow("Name", self.name_ed)
        self.best_ed = QLineEdit(self.b.best_for)
        self.best_ed.editingFinished.connect(lambda: self._set("best_for", self.best_ed.text().strip()))
        self.weak_ed = QLineEdit(self.b.weak_at)
        self.weak_ed.editingFinished.connect(lambda: self._set("weak_at", self.weak_ed.text().strip()))
        f.addRow("Best for", self.best_ed)
        f.addRow("Weak at", self.weak_ed)
        v.addLayout(f)
        self.tabs = QTabWidget()
        self.sliders: dict = {}
        self.tabs.addTab(self._sliders_tab(TIP), "Tip")
        self.tabs.addTab(self._paint_tab(), "Paint")
        self.tabs.addTab(self._dynamics_tab(), "Dynamics")
        self.tabs.addTab(self._texture_tab(), "Texture")
        self.tabs.addTab(self._colour_tab(), "Colour")
        self.tabs.addTab(self._sliders_tab(MEDIA), "Media")
        v.addWidget(self.tabs, 1)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        v.addWidget(self.note)
        row = QHBoxLayout()
        for text, fn, tip in (("Save as new brush…", self.save_as_new, "Keep a copy under a new name (kept "
                                                                        "beside the app)"),
                              ("Save", self.save, "Keep these settings for next time"),
                              ("Back to how it came", self.revert, "Undo every change made in this window"),
                              ("Close", self.close, "")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        self._timer = QTimer(self, singleShot=True, interval=80, timeout=self.render_preview)
        self.render_preview()

    # ---- building
    def _slider_row(self, form, attr, label, lo, hi, scale, tip):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        val = QLabel()
        val.setMinimumWidth(42)
        s.setToolTip(tip)

        def changed(n, attr=attr, scale=scale, val=val):
            x = n / scale
            if attr in ("count", "dual_count", "bristles"):
                x = int(n)
            val.setText(f"{x:g}" if isinstance(x, int) or scale == 1 else f"{x:.2f}")
            self._set(attr, x)
        s.valueChanged.connect(changed)
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(s, 1)
        h.addWidget(val)
        form.addRow(label, w)
        self.sliders[attr] = (s, scale, val)
        s.blockSignals(True)
        s.setValue(int(round(float(getattr(self.b, attr)) * scale)))
        s.blockSignals(False)
        val.setText(f"{getattr(self.b, attr):g}" if scale == 1 else f"{getattr(self.b, attr):.2f}")

    def _sliders_tab(self, spec, extra=None):
        w = QWidget()
        f = QFormLayout(w)
        for row in spec:
            self._slider_row(f, *row)
        if extra:
            extra(f)
        return w

    def _paint_tab(self):
        def extra(f):
            self.mode_cb = QComboBox()
            for k, (n, _, _) in MODES.items():
                if k != "mypaint" or self.b.mode == "mypaint":
                    self.mode_cb.addItem(n, k)
            self.mode_cb.setCurrentIndex(max(0, self.mode_cb.findData(self.b.mode)))
            self.mode_cb.currentIndexChanged.connect(lambda: self._set("mode", self.mode_cb.currentData()))
            f.addRow("Paint", self.mode_cb)
            self.space_cb = QComboBox()
            for k, (n, _, _) in SPACES.items():
                self.space_cb.addItem(n, k)
            self.space_cb.setCurrentIndex(max(0, self.space_cb.findData(self.b.space)))
            self.space_cb.currentIndexChanged.connect(lambda: self._set("space", self.space_cb.currentData()))
            f.addRow("Mix colours", self.space_cb)
            self.kind_cb = QComboBox()
            for k, (n, _, _) in SMUDGE_KINDS.items():
                self.kind_cb.addItem(n, k)
            self.kind_cb.setCurrentIndex(max(0, self.kind_cb.findData(self.b.smudge_kind)))
            self.kind_cb.currentIndexChanged.connect(lambda: self._set("smudge_kind", self.kind_cb.currentData()))
            f.addRow("Smudge", self.kind_cb)
            self.eraser_cb = QCheckBox("Eraser (takes paint away)")
            self.eraser_cb.setChecked(self.b.eraser)
            self.eraser_cb.toggled.connect(lambda on: self._set("eraser", on))
            f.addRow(self.eraser_cb)
        return self._sliders_tab(PAINT, extra)

    def _dynamics_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("Which pen inputs change what. Tick a box, then shape its curve below."))
        g = QGridLayout()
        self.dyn_boxes = {}
        for c, i in enumerate(INPUTS):
            lbl = QLabel(i.capitalize())
            lbl.setToolTip(INPUT_NOTES[i])
            g.addWidget(lbl, 0, c + 1)
        for r, (t, tn) in enumerate(TARGETS.items()):
            g.addWidget(QLabel(tn), r + 1, 0)
            for c, i in enumerate(INPUTS):
                cb = QCheckBox()
                cb.setToolTip(f"{tn} follows {i}: {INPUT_NOTES[i]}")
                cb.setChecked(i in self.b.dynamics.get(t, {}))
                cb.toggled.connect(lambda on, t=t, i=i: self._dyn_toggle(t, i, on))
                g.addWidget(cb, r + 1, c + 1)
                self.dyn_boxes[(t, i)] = cb
        v.addLayout(g)
        self.curve_for = QComboBox()
        self.curve_for.currentIndexChanged.connect(self._curve_pick)
        v.addWidget(self.curve_for)
        self.dyn_curve = CurveBox()
        self.dyn_curve.curve_changed.connect(self._curve_edit)
        v.addWidget(self.dyn_curve, 1)
        self._refresh_curve_list()
        return w

    def _texture_tab(self):
        def extra(f):
            self.grain_cb = QComboBox()
            self.grain_cb.addItem("No grain", "")
            for k, (n, _, _) in texture.GRAINS.items():
                self.grain_cb.addItem(n, k)
            self.grain_cb.addItem("From a picture…", "picture")
            self.grain_cb.setCurrentIndex(max(0, self.grain_cb.findData(self.b.grain)))
            self.grain_cb.activated.connect(self._grain_chosen)
            f.insertRow(0, "Paper grain", self.grain_cb)
            self.grain_p = QCheckBox("Pressing harder fills the paper's valleys")
            self.grain_p.setChecked(self.b.grain_pressure)
            self.grain_p.toggled.connect(lambda on: self._set("grain_pressure", on))
            f.insertRow(3, self.grain_p)
            self.dual_cb = QCheckBox("Second tip (breaks up the dab, as Photoshop's Dual Brush)")
            self.dual_cb.setChecked(self.b.dual)
            self.dual_cb.toggled.connect(lambda on: self._set("dual", on))
            f.insertRow(4, self.dual_cb)
        return self._sliders_tab(TEXTURE, extra)

    def _colour_tab(self):
        def extra(f):
            self.jit_stroke = QCheckBox("One colour per stroke (instead of every dab)")
            self.jit_stroke.setChecked(self.b.jitter_per_stroke)
            self.jit_stroke.toggled.connect(lambda on: self._set("jitter_per_stroke", on))
            f.addRow(self.jit_stroke)
        return self._sliders_tab(COLOUR, extra)

    # ---- changes
    def _set(self, attr, value):
        setattr(self.b, attr, value)
        if attr == "name":
            self.setWindowTitle(f"Brush editor: {value}")
        self.page.brush_changed(self.index)
        self._explain(attr)
        self._timer.start()

    def _explain(self, attr):
        b = self.b
        if attr == "grain" or attr.startswith("grain"):
            if b.grain in texture.GRAINS:
                n, best, weak = texture.GRAINS[b.grain]
                self.note.setText(f"{n}: best for {best}. Weak at: {weak}.")
        elif attr == "mode":
            n, best, weak = MODES[b.mode]
            self.note.setText(f"{n}: best for {best}. Weak at: {weak}.")
        elif attr == "space":
            n, best, weak = SPACES[b.space]
            self.note.setText(f"{n}: best for {best}. Weak at: {weak}.")
        elif attr == "dual":
            self.note.setText("Second tip: best for chalk, pastel, sponges and broken dry-brush edges. Weak at: "
                              "solid even colour.")
        elif attr in ("water", "pigment", "lift", "granulation", "edge_darkening"):
            self.note.setText("Watercolour (Paint: Watercolour) flows and dries on its own layer: dark edges, blooms, "
                              "granulation and lifting. Best for: transparent washes. Weak at: opaque cover.")
        elif attr in ("bristles", "paint_load", "pickup"):
            self.note.setText("Bristles (Paint: Bristles): each bristle carries its own paint and runs dry. Best for: "
                              "oil and acrylic strokes. Weak at: flat even colour.")
        elif attr in ("jitter_h", "jitter_s", "jitter_v"):
            self.note.setText("Colour jitter: best for foliage, flowers and crowds of marks. Weak at: flat colour.")

    def _grain_chosen(self, i):
        key = self.grain_cb.itemData(i)
        if key == "picture":
            from PySide6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(self, "A picture as paper grain", "",
                                                  "Pictures (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
            if not path:
                self.grain_cb.setCurrentIndex(max(0, self.grain_cb.findData(self.b.grain)))
                return
            try:
                from PIL import Image
                g = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
                self.b.grain_tile = texture.from_picture(g)
            except OSError as err:
                self.note.setText(f"That picture could not be read: {err}")
                return
            except Exception as err:                    # noqa: BLE001
                report_problem(f"Paint: grain from {path}", err)
                self.note.setText("That picture could not be used; an error report was saved.")
                return
        if key and self.b.grain_strength <= 0:
            self.sliders["grain_strength"][0].setValue(70)     # a grain with no strength would do nothing
        self._set("grain", key)

    def _dyn_toggle(self, t, i, on):
        d = self.b.dynamics.setdefault(t, {})
        if on and i not in d:
            d[i] = Curve() if i != "random" else Curve([(0, 0.5), (1, 1)])
        elif not on:
            d.pop(i, None)
        self._refresh_curve_list(select=(t, i) if on else None)
        self._set("dynamics", self.b.dynamics)

    def _refresh_curve_list(self, select=None):
        self.curve_for.blockSignals(True)
        self.curve_for.clear()
        for t, ins in self.b.dynamics.items():
            for i in ins:
                self.curve_for.addItem(f"Curve: {TARGETS.get(t, t)} by {i}", (t, i))
        if select is not None:
            j = self.curve_for.findData(select)
            self.curve_for.setCurrentIndex(max(0, j))
        self.curve_for.blockSignals(False)
        self._curve_pick()

    def _curve_pick(self, *_):
        key = self.curve_for.currentData()
        self.dyn_curve.setEnabled(key is not None)
        if key:
            t, i = key
            self.dyn_curve.set_curve(self.b.dynamics[t][i])

    def _curve_edit(self, curve, *_):
        key = self.curve_for.currentData()
        if key:
            t, i = key
            self.b.dynamics[t][i] = curve
            self._set("dynamics", self.b.dynamics)

    # ---- preview, saving
    def render_preview(self):
        w = max(200, self.preview_lbl.width() or 400)
        try:
            img = brushfile.preview(self.b, w, 110, tuple(self.page.canvas.colour))
        except Exception as err:                        # noqa: BLE001
            report_problem("Paint: brush preview", err)
            self.preview_lbl.setText("The preview could not be drawn; an error report was saved.")
            return None
        img = np.ascontiguousarray(img)
        q = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format.Format_RGB888).copy()
        self.preview_lbl.setPixmap(QPixmap.fromImage(q))
        self._last_preview = img
        return img

    def save(self):
        try:
            p = brushfile.save(self.b)
        except OSError as err:
            self.note.setText(f"The brush could not be saved: {err}")
            return None
        self.b.saved_as = str(p)
        self.note.setText(f"Saved. It will be in the brush list next time ({p.name}).")
        return p

    def save_as_new(self, name: str | None = None):
        if name is None:
            name, ok = QInputDialog.getText(self, "Save as new brush", "Name:", text=self.b.name + " (mine)")
            if not ok or not name.strip():
                return None
        new = self.b.copy()
        new.dynamics = {t: {i: Curve(c.points) for i, c in ins.items()} for t, ins in self.b.dynamics.items()}
        new.name = name.strip()
        new.tags = "Mine"
        self.page.add_brush(new)
        self.b.__dict__.update(copy.copy(self.original.__dict__))   # the brush it came from stays as it was
        self.index, self.b = len(self.page.brushes) - 1, new
        self.original = new.copy()
        self.name_ed.setText(new.name)
        self.setWindowTitle(f"Brush editor: {new.name}")
        return self.save()

    def revert(self):
        self.b.__dict__.update(copy.copy(self.original.__dict__))
        self.b.dynamics = {t: {i: Curve(c.points) for i, c in ins.items()}
                           for t, ins in self.original.dynamics.items()}
        for attr, (s, scale, val) in self.sliders.items():
            s.blockSignals(True)
            s.setValue(int(round(float(getattr(self.b, attr)) * scale)))
            s.blockSignals(False)
        for (t, i), cb in self.dyn_boxes.items():
            cb.blockSignals(True)
            cb.setChecked(i in self.b.dynamics.get(t, {}))
            cb.blockSignals(False)
        self._refresh_curve_list()
        self.name_ed.setText(self.b.name)
        for cb, val in ((self.mode_cb, self.b.mode), (self.space_cb, self.b.space), (self.kind_cb, self.b.smudge_kind),
                        (self.grain_cb, self.b.grain)):
            cb.blockSignals(True)
            cb.setCurrentIndex(max(0, cb.findData(val)))
            cb.blockSignals(False)
        for cb, val in ((self.eraser_cb, self.b.eraser), (self.grain_p, self.b.grain_pressure),
                        (self.dual_cb, self.b.dual), (self.jit_stroke, self.b.jitter_per_stroke)):
            cb.blockSignals(True)
            cb.setChecked(bool(val))
            cb.blockSignals(False)
        self.page.brush_changed(self.index)
        self.note.setText("Back to how it was when this window opened.")
        self.render_preview()
