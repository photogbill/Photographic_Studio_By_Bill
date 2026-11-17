"""The Infrared tab (BUILD_PLAN_PANO_IR.md 1.6): infrared and full-spectrum raws, before ordinary editing.

One photo (or a shoot) from the Library: pick the filter (guessed from the raw, amber, with its numbers), set white
balance far outside the camera's range (foliage, a neutral, or a filter measured once on a grey card: cyan), choose a
look (channel swap, false colour, black and white from the channels), and shape tones in black and white (per channel
exposure, contrast and separation, one mix with a live noise meter; Detail by scale, the same settings Develop has).
Everything it sets lives in the photo's own edit (``edit.infrared`` and the white balance), so Open in Develop carries
on on the raw with every tool, and Apply to the shoot pastes the infrared part to the rest of the strip.

Keys: B shows the camera's own rendering in the same place, Z 100%, Esc fit, Ctrl+Enter Open in Develop.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QInputDialog, QLabel, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from .. import log as slog
from . import theme as T
from .common import chip, display8, hbox, label, run_bg, to_qimage
from .stackui import StackPage, honest_card

log = slog.get("app.infrared")

WHAT = ("Infrared and full-spectrum raws break normal colour: the camera's white balance cannot go far enough and its "
        "colour matrix was made for visible light. Pick the filter, click foliage, choose a look, then edit in "
        "Develop as usual.")
HOW = ("Give it photos: in the Library right-click them, Edit in Infrared tab (the others become the strip below); "
       "or drop them here.")
SHOOT = ("What to shoot: a frame with a grey card or white paper in sun through each filter lets the Studio measure "
         "that filter once (cyan). Foliage and sky in the frame help the presets.")
LIMITS = ("An 850 nm raw records no colour at all: any colour you see (a warm candle, a palette) is added afterwards. "
          "Lens coatings are made for visible light: with the sun at the wrong angle infrared shows a strong flare "
          "that no tool can take out of one photo (in a panorama the flare guard keeps only one). "
          "White balance far from daylight costs noise: one channel may be multiplied 4-15 times, so use Clean. "
          "The filter guess and the colour looks are tuned on a model of a converted camera until real 590-720 nm "
          "frames have been measured.")

DS_KEYS = ("tonal", "local", "crisp", "separation", "veil", "broad")


def _widgets():
    from . import main as M
    return M.ValueSlider, M.Group


class _Times:
    """A ValueSlider format for a log-scale gain: shows the multiplier itself."""

    def format(self, v):
        return f"{float(np.exp(v)):.2f}x"


def gray_qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a, np.uint8)
    h, w = a.shape
    return QImage(a.data, w, h, w, QImage.Format.Format_Grayscale8).copy()


# ============================================================================================ Detail by scale
class DetailByScaleBox(QWidget):
    """Detail by scale (plan 23.11 / 1.5d) bound to an edit's ``detail`` and ``microcontrast`` fields: the same
    fields, ranges and Amount / Balance conversion (``scales.from_simple`` / ``to_simple``) as Develop's Detail group,
    so what is set here shows there and the reverse."""
    changed = Signal()

    def __init__(self, judge_bw: bool = True):
        super().__init__()
        from ..engine.detail import scales as DS
        ValueSlider, _ = _widgets()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        self.preset = QComboBox()
        self.preset.addItems(["Preset..."] + list(DS.PRESETS))
        self.preset.activated.connect(self._preset_picked)
        self.judge = QCheckBox("Judge in black and white")
        self.judge.setChecked(judge_bw)
        self.judge.setToolTip("Shows the photo in black and white while you adjust (it does not change the edit).")
        v.addWidget(hbox(self.preset, self.judge))
        self.amount = ValueSlider("Amount", 0, 100, 0, 1, "{:.0f}", "How much: the shades of shadow. 60-80 is plain "
                                  "to see; big edges are left alone and noise is not lifted.")
        self.balance = ValueSlider("Balance  (tonal - crisp)", -100, 100, -50, 1, "{:+.0f}",
                                   "Left = Tonal (modelled, more shades), right = Crisp (etched).")
        v.addWidget(self.amount)
        v.addWidget(self.balance)
        self.rows = {}
        self.methods = QWidget()
        mv = QVBoxLayout(self.methods)
        mv.setContentsMargins(0, 2, 0, 0)
        mv.setSpacing(2)
        for key in DS_KEYS:
            info = DS.METHODS[key]
            on = QCheckBox(info["name"])
            on.setChecked(True)
            sl = ValueSlider("  amount", info["lo"], info["hi"], 0, 1, "{:+.0f}" if info["lo"] < 0 else "{:.0f}",
                             f"{info['does']}\nBest for: {info['best']}\nWeak at: {info['weak']}")
            mv.addWidget(on)
            mv.addWidget(sl)
            self.rows[key] = (on, sl)
            on.toggled.connect(lambda *_: self.changed.emit())
            sl.changed.connect(lambda _v, k=key: self._method_moved(k))
        self.sh = ValueSlider("  in shadows", 0.0, 1.0, 1.0, 0.05, "{:.0%}")
        self.mid = ValueSlider("  in midtones", 0.0, 1.0, 0.7, 0.05, "{:.0%}")
        self.hi = ValueSlider("  in highlights", 0.0, 1.0, 0.3, 0.05, "{:.0%}")
        mv.addWidget(label("Where it works (all methods):", "dim"))
        for w in (self.sh, self.mid, self.hi):
            mv.addWidget(w)
            w.changed.connect(lambda *_: self.changed.emit())
        v.addWidget(self.methods)
        self.amount.changed.connect(self._simple_moved)
        self.balance.changed.connect(self._simple_moved)

    def load(self, e):
        m = e.microcontrast
        off = set(e.detail.off or ())
        for k, (on, sl) in self.rows.items():
            on.blockSignals(True)
            on.setChecked(k not in off)
            on.blockSignals(False)
            sl.set_value(m.amount * 100.0 if k == "local" else getattr(e.detail, k), emit=False)
        for w, val in ((self.sh, m.shadows), (self.mid, m.midtones), (self.hi, m.highlights)):
            w.set_value(val, emit=False)
        self._sync_simple()

    def store(self, e):
        m = e.microcontrast
        m.amount = round(self.rows["local"][1].value() / 100.0, 3)
        m.shadows, m.midtones, m.highlights = (round(w.value(), 3) for w in (self.sh, self.mid, self.hi))
        for k in ("tonal", "crisp", "separation", "veil", "broad"):
            setattr(e.detail, k, round(self.rows[k][1].value(), 1))
        off = [k for k, (on, _) in self.rows.items() if not on.isChecked()]
        e.detail.off = off or None
        return e

    def _sync_simple(self):
        from ..engine.detail import scales as DS
        a, t = DS.to_simple(self.rows["tonal"][1].value() / 100.0, self.rows["crisp"][1].value() / 100.0)
        self.amount.set_value(a * 100.0, emit=False)
        if a > 1e-6:
            self.balance.set_value(t * 100.0, emit=False)

    def _simple_moved(self, *_):
        from ..engine.detail import scales as DS
        m = DS.from_simple(self.amount.value() / 100.0, self.balance.value() / 100.0)
        self.rows["tonal"][1].set_value(m["tonal"] * 100.0, emit=False)
        self.rows["crisp"][1].set_value(m["crisp"] * 100.0, emit=False)
        self.preset.setCurrentIndex(0)
        self.changed.emit()

    def _method_moved(self, key):
        if key in ("tonal", "crisp"):
            self._sync_simple()
        self.preset.setCurrentIndex(0)
        self.changed.emit()

    def _preset_picked(self, index):
        from ..engine.detail import scales as DS
        name = self.preset.itemText(index)
        if name not in DS.PRESETS:
            return
        vals = DS.PRESETS[name]
        for k, (on, sl) in self.rows.items():
            on.blockSignals(True)
            on.setChecked(True)
            on.blockSignals(False)
            sl.set_value(float(vals.get(k, 0.0)), emit=False)
        self._sync_simple()
        self.preset.setCurrentIndex(index)
        self.changed.emit()


# ============================================================================================ the page
class InfraredPage(StackPage):
    SETTINGS_KEY = "infrared"
    MIN_FRAMES = 1
    sidecar_written = Signal(str)                     # the window refreshes Develop and the Library

    def __init__(self, settings=None):
        from ..engine.color import infrared as irm
        self.irm = irm
        self.image = None
        self.path = ""
        self.edit = None
        self.report: dict = {}
        self.pick = None
        self._tok = 0
        self._rtok = 0
        self._before = None
        self._loading = False
        super().__init__("Infrared", WHAT, HOW, SHOOT, settings, add_text="Open photos...")
        self.tray.currentRowChanged.connect(self._strip_row)
        self.result.view.clicked.connect(self._view_clicked)
        self.render_timer = QTimer(self, singleShot=True, interval=140, timeout=self.render_now)
        self.meter_timer = QTimer(self, singleShot=True, interval=450, timeout=self._measure_bw)

    # ------------------------------------------------------------------------------------ panel
    def build_panel(self):
        irm = self._irm()
        ValueSlider, _ = _widgets()
        g = self.group("Filter", "Which filter the photo was taken through. Guessed from the raw itself.")
        self.filter_pick = g.add(QComboBox())
        for k in irm.FILTER_ORDER:
            self.filter_pick.addItem(irm.FILTERS[k]["label"], k)
        self.filter_pick.activated.connect(self._filter_picked)
        self.filter_text = g.add(label("", "dim", wrap=True))
        self.guess_chip = g.add(chip("Filter guess: open a photo", T.DIM))
        self.meter_chip = g.add(chip("Colour meter: -", T.DIM))
        self.meter_chip.setToolTip("Is there any colour in this raw? The one dimension of colour a black and white "
                                   "raw cannot have, against the two green photosites of each quad, which see "
                                   "identical light. Measured from the raw (cyan).")
        self.measure_b = QPushButton("Measure this filter...")
        self.measure_b.setToolTip("Click a grey card or white paper in sun in the photo: its white balance is saved "
                                  "for this camera and filter on this PC and offered first next time (cyan).")
        self.measure_b.setCheckable(True)
        self.measure_b.toggled.connect(lambda on: self._set_pick("card" if on else None))
        g.add(self.measure_b)

        g = self.group("White balance", "Far outside the camera's range. Infrared photographers make foliage "
                       "neutral; a grey card is best.")
        self.auto_fol_b = QPushButton("Auto foliage")
        self.auto_fol_b.setToolTip("Finds the bright, moderately textured areas (foliage, not sky, not glints) and "
                                   "makes them neutral (estimated: amber).")
        self.auto_fol_b.clicked.connect(self.auto_foliage)
        self.click_fol_b = QPushButton("Click foliage")
        self.click_fol_b.setCheckable(True)
        self.click_fol_b.toggled.connect(lambda on: self._set_pick("foliage" if on else None))
        self.click_neu_b = QPushButton("Click a neutral")
        self.click_neu_b.setCheckable(True)
        self.click_neu_b.toggled.connect(lambda on: self._set_pick("neutral" if on else None))
        g.add(hbox(self.auto_fol_b, self.click_fol_b))
        g.add(hbox(self.click_neu_b))
        self.measured_pick = g.add(QComboBox())
        self.measured_pick.setToolTip("White balances measured for a filter on this PC (Measure this filter...)")
        self.measured_pick.activated.connect(self._measured_picked)
        self.mult_sl = []
        for n in ("Red gain", "Green gain", "Blue gain"):
            sl = ValueSlider(n, -3.0, 3.0, 0.0, 0.01, _Times(), "The channel's multiplier on a log scale "
                             "(0.05x to 20x); the smallest of the three is always 1.")
            g.add(sl)
            sl.changed.connect(self._mult_moved)
            self.mult_sl.append(sl)
        self.mult_text = g.add(label("", "dim", wrap=True))
        self.wb_chip = g.add(chip("White balance: -", T.DIM))
        self.noise_chip = g.add(chip("", T.AMBER))
        self.noise_chip.hide()

        g = self.group("Look", "Colour from the channels: swaps and false colour. Each says what it is good for.")
        self.preset_pick = g.add(QComboBox())
        for k in irm.PRESETS:
            self.preset_pick.addItem(irm.PRESET_INFO[k]["label"], k)
        self.preset_pick.activated.connect(self._preset_picked)
        self.preset_text = g.add(label("", "dim", wrap=True))
        self.strength = g.add(ValueSlider("Strength", 0.0, 1.5, 1.0, 0.05, "{:.0%}",
                                          "How far the look moves from the camera channels (colour looks only)."))
        self.strength.changed.connect(self._strength_moved)
        self.palette_pick = g.add(QComboBox())
        self.palette_pick.addItem("Palette by brightness: none", "")
        for k in irm.PALETTES:
            self.palette_pick.addItem(f"Palette: {k.capitalize()} (added colour)", k)
        self.palette_pick.setToolTip("\n".join(f"{k.capitalize()}: {v}" for k, v in irm.PALETTE_INFO.items()))
        self.palette_pick.activated.connect(self._palette_changed)
        self.palette_amt = g.add(ValueSlider("Palette amount", 0, 100, 100, 1, "{:.0f}%"))
        self.palette_amt.changed.connect(self._palette_changed)
        self.added_chip = g.add(chip("Added colour: not in the raw", T.AMBER))
        self.added_chip.hide()

        g = self.group("Black and white", "The three channels are three handles on the picture: change exposure, "
                       "contrast and separation per channel, then one mix. You always see the black and white result "
                       "while adjusting; B shows the plain average.")
        self.bw_on = g.add(QCheckBox("Black and white from the channels"))
        self.bw_on.toggled.connect(self._bw_toggled)
        self.mono_raw = g.add(QCheckBox("Black and white raw (no demosaic)"))
        self.mono_raw.setToolTip(
            "850 nm and deeper: every photosite is used as a brightness sample once the four of each quad are brought "
            "to the same gain. Best for: sharp 850 nm daylight subjects (foliage, branches, architecture): full "
            "resolution, no colour noise, no demosaic patterns. Weak at: nothing is gained when the lens or the light "
            "limits sharpness; wrong for any filter that still records colour (offered only when there is none).")
        self.mono_raw.toggled.connect(self._mono_toggled)
        self.thumbs = []
        row = QWidget()
        rl = QGridLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        for c, (n, col) in enumerate((("Red", T.RED), ("Green", T.CYAN), ("Blue", T.VIOLET))):
            lab = QLabel()
            lab.setFixedSize(86, 60)
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setStyleSheet(f"border:1px solid {T.LINE}; background:{T.BG2};")
            rl.addWidget(lab, 0, c)
            t = label(n, "dim")
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rl.addWidget(t, 1, c)
            self.thumbs.append(lab)
        g.add(row)
        self.bw_preset = g.add(QComboBox())
        self.bw_preset.addItem("Mix preset...", "")
        for k in irm.BW_PRESETS:
            self.bw_preset.addItem(irm.BW_PRESET_INFO[k]["label"], k)
        self.bw_preset.setToolTip("\n".join(f"{v['label']}. Best for: {v['best']} Weak at: {v['weak']}"
                                            for v in irm.BW_PRESET_INFO.values()))
        self.bw_preset.activated.connect(self._bw_preset_picked)
        self.ch_ev, self.ch_con, self.ch_sep, self.ch_mix = [], [], [], []
        for c, n in enumerate(("Red", "Green", "Blue")):
            hd = g.add(label(f"{n} channel"))
            hd.setStyleSheet("font-weight:600;")
            ev = g.add(ValueSlider("  exposure", -3, 3, 0, 0.05, "{:+.2f} EV"))
            co = g.add(ValueSlider("  contrast", -100, 100, 0, 1, "{:+.0f}"))
            se = g.add(ValueSlider("  separation", -100, 100, 0, 1, "{:+.0f}",
                                   "How much of what this channel sees differently from the other two is kept: "
                                   "-100 none, +100 twice."))
            mx = g.add(ValueSlider("  in the mix", -300, 300, 33.3, 1, "{:+.0f}%"))
            for w in (ev, co, se, mx):
                w.changed.connect(self._bw_moved)
            self.ch_ev.append(ev)
            self.ch_con.append(co)
            self.ch_sep.append(se)
            self.ch_mix.append(mx)
        self.keep = g.add(QCheckBox("Keep brightness (mix adds up to 100%)"))
        self.keep.setChecked(True)
        self.keep.toggled.connect(self._bw_moved)
        self.mix_sum = g.add(label("", "dim"))
        self.noise_meter = g.add(chip("Noise: -", T.DIM))
        self.noise_meter.setToolTip("How much this mix multiplies the noise against an even mix, measured on the "
                                    "demosaiced photo at the same brightness. At 850 nm every 1% of tonal change from "
                                    "an extreme mix costs about a doubling of noise: the Light panel, Detail by scale "
                                    "and masks move tones there without that cost.")
        self.recipe_pick = g.add(QComboBox())
        self.recipe_pick.activated.connect(self._recipe_picked)
        save_r = QPushButton("Save as B&W recipe...")
        save_r.clicked.connect(self.save_recipe)
        g.add(save_r)

        g = self.group("Detail by scale", "The shades of shadow, while the photo is in black and white. The same "
                       "settings as Develop's Detail group. On an 850 nm photo this is the main tool for tonal "
                       "separation: it costs no extra noise, unlike an extreme channel mix.")
        self.ds = g.add(DetailByScaleBox(judge_bw=True))
        self.ds.changed.connect(self._ds_changed)
        self.ds.judge.toggled.connect(lambda *_: self.render_now())

        g = self.group("Channel mixer", "Rows: the output red, green and blue made from the camera's red, green and "
                       "blue (linear light). The looks set it; you can move any number.")
        grid = QWidget()
        gl = QGridLayout(grid)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(3)
        self.mix_spin = []
        for r, rn in enumerate(("R out", "G out", "B out")):
            gl.addWidget(label(rn, "dim"), r + 1, 0)
            for c in range(3):
                sp = QDoubleSpinBox()
                sp.setRange(-3.0, 3.0)
                sp.setSingleStep(0.05)
                sp.setDecimals(2)
                sp.setMaximumWidth(78)
                sp.valueChanged.connect(self._mixer_edited)
                gl.addWidget(sp, r + 1, c + 1)
                self.mix_spin.append(sp)
        for c, cn in enumerate(("from R", "from G", "from B")):
            gl.addWidget(label(cn, "dim"), 0, c + 1)
        g.add(grid)
        self.row_sum = g.add(label("", "dim"))
        self.profile_pick = g.add(QComboBox())
        self.profile_pick.addItem("Camera channels as they are", "identity")
        self.profile_pick.addItem("Camera colour matrix (IR-cut filter on)", "visible")
        self.profile_pick.setToolTip("Camera channels as they are: for infrared filters (no visible-light colour "
                                     "science pretended). Camera colour matrix: a full-spectrum camera with an IR-cut "
                                     "filter on the lens, which behaves like a normal camera.")
        from PySide6.QtWidgets import QSizePolicy
        self.profile_pick.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.profile_pick.activated.connect(self._profile_picked)

        g = self.group("Hot spot", "Many lenses show a bright centre in infrared (reflections inside the lens). "
                       "Correction is off by default, like vignetting.")
        self.hs_measure = QPushButton("Measure from this photo")
        self.hs_measure.setToolTip("Best on a flat frame: clear sky or a plain wall")
        self.hs_measure.clicked.connect(self.measure_hotspot)
        g.add(self.hs_measure)
        self.hs_amount = g.add(ValueSlider("Correction", 0, 100, 0, 1, "{:.0f}%"))
        self.hs_amount.changed.connect(self._hotspot_moved)
        self.hs_chip = g.add(chip("Hot spot: not measured", T.DIM))
        self.hs_saved = g.add(QComboBox())
        self.hs_saved.setToolTip("Hot spots learnt for a lens and filter on this PC (Library: Tools, Learn hot spot)")
        self.hs_saved.activated.connect(self._hotspot_saved_picked)

        g = self.group("Fusion", "A visible frame and an infrared frame of the same scene (tripod): infrared detail "
                       "with visible colour, colour-infrared, or a vegetation map.")
        self.fu_text = g.add(label("No visible frame added.", "dim", wrap=True))
        self.fu_add = QPushButton("Add the visible frame...")
        self.fu_add.clicked.connect(self.add_visible)
        g.add(self.fu_add)
        from ..engine.stack import fusion as FU
        self.fu_mode = g.add(QComboBox())
        for k, d in FU.METHOD_INFO.items():
            self.fu_mode.addItem(d["name"], k)
        self.fu_mode.setToolTip("\n".join(f"{d['name']}. Best for: {d['best']} Weak at: {d['weak']}"
                                          for d in FU.METHOD_INFO.values()))
        self.fu_build = QPushButton("Build fusion")
        self.fu_build.clicked.connect(self.build_fusion)
        g.add(self.fu_build)
        g.add(label(FU.HOW_TO, "dim", wrap=True))
        self.visible_path = ""
        g = self.group("Honest limits")
        g.add(honest_card(LIMITS))
        self.panel.addStretch(1)
        from PySide6.QtWidgets import QSizePolicy
        for cb in (self.filter_pick, self.measured_pick, self.preset_pick, self.palette_pick, self.bw_preset,
                   self.recipe_pick, self.hs_saved, self.fu_mode):
            cb.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        self.apply_b = QPushButton("Apply to the shoot")
        self.apply_b.setToolTip("Paste this photo's infrared settings and white balance onto the other photos in the "
                                "strip below")
        self.apply_b.clicked.connect(self.apply_to_shoot)
        self.open_b = QPushButton("Open in Develop")
        self.open_b.setObjectName("primary")
        self.open_b.setToolTip("Ctrl+Enter")
        self.open_b.clicked.connect(self.build_full)
        self.reset_b = QPushButton("Start again")
        self.reset_b.setToolTip("The filter guess, white balance and look from scratch")
        self.reset_b.clicked.connect(self.reset)
        self.off_b = QPushButton("Normal colour")
        self.off_b.setToolTip("Turn the infrared treatment off for this photo (its settings are kept)")
        self.off_b.clicked.connect(self.turn_off)
        self.actions.addWidget(hbox((self.apply_b, 1), (self.open_b, 1), stretch=False))
        self.actions.addWidget(hbox((self.reset_b, 1), (self.off_b, 1), stretch=False))
        self.where = label("", "dim", wrap=True)
        self.actions.addWidget(self.where)
        self._enable(False)

    def _irm(self):
        if not hasattr(self, "irm"):
            from ..engine.color import infrared as irm
            self.irm = irm
        return self.irm

    def _enable(self, on: bool):
        for w in (self.filter_pick, self.measure_b, self.auto_fol_b, self.click_fol_b, self.click_neu_b,
                  self.preset_pick, self.strength, self.bw_on, self.apply_b, self.open_b, self.reset_b, self.off_b,
                  self.hs_measure, self.fu_add):
            w.setEnabled(on)

    # ------------------------------------------------------------------------------------ photos in
    def add_files(self, files: list, replace: bool = False):
        from ..engine.stack.frames import probe
        files = self.expand(files)
        if not files:
            return
        have = [p for p, _, _ in self.items] if not replace else []
        new = [f for f in files if f not in have]
        infos = probe(have + new)
        self.items = [(f.path, f.ok, f.problem) for f in infos]
        self.tray.set_frames(self.items)
        first = next((p for p, ok, _ in self.items if ok and (p in new or replace)), None)
        self._refresh()
        if first:
            self.open_photo(first)

    def _strip_row(self, row):
        if self._loading or row < 0 or row >= len(self.items):
            return
        p, ok, _ = self.items[row]
        if ok and p != self.path:
            self.open_photo(p)

    def open_photo(self, path: str):
        from .. import engine as E
        self.save_sidecar()
        self._tok += 1
        tok = self._tok
        self.jobbar.say(f"Opening {os.path.basename(path)}: measuring its channels...")
        self._enable(False)

        def work():
            img = E.open(path)
            e = E.load_edit(path)
            rep = {}
            if (e.infrared or {}).get("on"):
                rep["meter"] = self.irm.colour_meter(img)
                rep["guess"] = (e.infrared or {}).get("guess") or self.irm.guess_filter(img, rep["meter"]["ratio"])
                rep["kept"] = True
            else:
                prev = e.infrared
                e, rep = self.irm.start(img, e)
                if prev and prev.get("v"):           # switched off earlier: its own settings come back
                    e.infrared = dict(self.irm.normalise(prev), on=True, guess=rep["guess"])
                    rep["kept"] = True
            return img, e, rep

        def done(t, res):
            if t != self._tok:
                return
            self.image, self.edit, self.report = res
            self.path = path
            self._before = None
            self.settings.set("folder", os.path.dirname(path))
            self._loading = True
            for i, (p, _, _) in enumerate(self.items):
                if p == path:
                    self.tray.setCurrentRow(i)
            self._loading = False
            self._enable(True)
            self.controls_from_edit()
            self.jobbar.say(self._open_text(), T.DIM)
            self.render_now()
            self._measure_bw()
            self._fill_saved()

        def fail(t, err):
            if t == self._tok:
                self.jobbar.say(f"Could not open {os.path.basename(path)}: {err}", T.RED)
        run_bg(work, tok, done, fail)

    def _open_text(self) -> str:
        g = (self.report or {}).get("guess") or {}
        txt = g.get("sentence", "")
        if self.report.get("recipe"):
            txt += f"  Your last B&W recipe for this filter was applied: {self.report['recipe']}."
        return txt

    def set_frames(self, files: list):
        self.items = []
        self.add_files(files, replace=True)

    def clear(self):
        self.save_sidecar()
        self.image, self.edit, self.path = None, None, ""
        super().clear()
        self._enable(False)

    # ------------------------------------------------------------------------------------ edit <-> controls
    def controls_from_edit(self):
        irm = self.irm
        e = self.edit
        ir = irm.normalise(e.infrared) or irm.default_ir()
        e.infrared = ir
        self.filter_pick.setCurrentIndex(max(0, self.filter_pick.findData(ir["filter"])))
        f = irm.FILTERS.get(ir["filter"], irm.FILTERS["unknown"])
        self.filter_text.setText(f"<b>Best for:</b> {f['best']}<br><b>Weak at:</b> {f['weak']}")
        g = (self.report or {}).get("guess") or ir.get("guess") or {}
        if g:
            self.guess_chip.setText(f"Guess: {irm.FILTERS.get(g.get('key'), irm.FILTERS['unknown'])['label']} "
                                    f"({100 * g.get('confidence', 0):.0f}%)")
            self.guess_chip.setToolTip(g.get("sentence", ""))
            self._colour(self.guess_chip, T.AMBER)
        m = (self.report or {}).get("meter") or {}
        if m:
            self.meter_chip.setText(m.get("text", "")[:70])
            self.meter_chip.setToolTip(m.get("text", ""))
            self._colour(self.meter_chip, T.CYAN)
        mono_ok = ir["filter"] == "850" or (m and not m.get("colour", True))
        self.mono_raw.setEnabled(bool(mono_ok))
        self.mono_raw.blockSignals(True)
        self.mono_raw.setChecked(bool(ir.get("mono_raw")) and bool(mono_ok))
        self.mono_raw.blockSignals(False)
        mult = irm.wb_from(e)
        for sl, v in zip(self.mult_sl, mult):
            sl.set_value(float(np.log(v)), emit=False)
        self._show_wb()
        self.preset_pick.setCurrentIndex(max(0, self.preset_pick.findData(ir.get("preset", "none"))))
        self._show_preset()
        self.strength.set_value(float(ir.get("strength", 1.0)), emit=False)
        pal = ir.get("palette") or {}
        self.palette_pick.setCurrentIndex(max(0, self.palette_pick.findData(pal.get("name", ""))))
        self.palette_amt.set_value(100.0 * float(pal.get("amount", 1.0)), emit=False)
        bw = ir["bw"]
        self.bw_on.blockSignals(True)
        self.bw_on.setChecked(bool(bw.get("on")))
        self.bw_on.blockSignals(False)
        for c in range(3):
            self.ch_ev[c].set_value(float(bw["ev"][c]), emit=False)
            self.ch_con[c].set_value(float(bw["contrast"][c]), emit=False)
            self.ch_sep[c].set_value(float(bw["sep"][c]), emit=False)
            self.ch_mix[c].set_value(100.0 * float(bw["mix"][c]), emit=False)
        self.keep.blockSignals(True)
        self.keep.setChecked(bool(bw.get("keep", True)))
        self.keep.blockSignals(False)
        self._show_mix_sum()
        self.ds.load(e)
        mix = np.asarray(ir["mixer"], float).reshape(3, 3)
        for i, sp in enumerate(self.mix_spin):
            sp.blockSignals(True)
            sp.setValue(float(mix.ravel()[i]))
            sp.blockSignals(False)
        self.row_sum.setText("Row sums: " + "  ".join(f"{v:.2f}" for v in irm.row_sums(mix)))
        self.profile_pick.setCurrentIndex(max(0, self.profile_pick.findData(ir.get("profile", "identity"))))
        hs = ir.get("hotspot") or {}
        self.hs_amount.set_value(100.0 * float(hs.get("amount", 0.0)), emit=False)
        self._show_hotspot()
        self._fill_recipes()
        self._bw_visibility()
        self.set_chips([(f"Infrared: {f['label']}", T.CYAN if str(ir.get('wb_source', '')).startswith('measured')
                         else T.AMBER)])

    def _colour(self, w, col):
        w.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")

    def _show_wb(self):
        irm = self.irm
        ir = self.edit.infrared
        m = irm.wb_from(self.edit)
        self.mult_text.setText(f"Multipliers: R {m[0]:.2f} · G {m[1]:.2f} · B {m[2]:.2f}")
        src = str(ir.get("wb_source", ""))
        names = {"foliage": ("Foliage white balance (estimated)", T.AMBER),
                 "click_foliage": ("Clicked foliage", T.AMBER), "neutral": ("Clicked a neutral", T.CYAN),
                 "as_shot": ("The camera's own white balance", T.DIM), "hand": ("Set by hand", T.DIM)}
        if src.startswith("measured:"):
            txt, col = f"Measured for this filter: {src.split(':', 1)[1]}", T.CYAN
        else:
            txt, col = names.get(src, ("White balance", T.DIM))
        self.wb_chip.setText(txt)
        self._colour(self.wb_chip, col)
        gw = irm.gain_warning(m)
        self.noise_chip.setVisible(bool(gw))
        self.noise_chip.setText(gw)

    def _show_preset(self):
        d = self.irm.PRESET_INFO.get(self.preset_pick.currentData(), {})
        self.preset_text.setText(f"<b>Best for:</b> {d.get('best', '')}<br><b>Weak at:</b> {d.get('weak', '')}")
        ir = self.edit.infrared if self.edit is not None else {}
        added = (ir or {}).get("preset") == "candle_glow" or bool(((ir or {}).get("palette") or {}).get("name"))
        self.added_chip.setVisible(added)

    def _bw_visibility(self):
        on = self.bw_on.isChecked()
        for ws in (self.ch_ev, self.ch_con, self.ch_sep, self.ch_mix):
            for w in ws:
                w.setEnabled(on)
        for w in (self.keep, self.bw_preset, self.recipe_pick):
            w.setEnabled(on)
        self.strength.setEnabled(not on)

    def _show_mix_sum(self):
        s = sum(w.value() for w in self.ch_mix)
        note = " (normalised to 100%)" if self.keep.isChecked() and abs(s - 100) > 0.5 else ""
        self.mix_sum.setText(f"The mix adds up to {s:.0f}%{note}.")

    def changed(self, label_text: str = ""):
        """Something in the edit changed: render again (debounced); the sidecar is saved when leaving the photo."""
        if self.edit is None:
            return
        self._dirty = True
        self.render_timer.start()
        if label_text:
            self.jobbar.say(label_text, T.DIM)

    # ------------------------------------------------------------------------------------ reactions
    def _filter_picked(self, *_):
        if self.image is None:
            return
        key = self.filter_pick.currentData()
        ir = self.irm.normalise(self.edit.infrared)
        fresh = self.irm.default_ir(key, ir.get("guess"))
        fresh.update(hotspot=ir.get("hotspot"), wb_source=ir.get("wb_source"))
        if not ir["bw"].get("on") or key == "850":
            self.edit.infrared = fresh
            self.edit = self.irm.apply_preset(self.image, self.edit, fresh["preset"], 1.0)
        else:
            fresh["bw"] = ir["bw"]
            self.edit.infrared = fresh
        self.controls_from_edit()
        self.changed(f"Filter: {self.irm.FILTERS[key]['label']}")

    def _set_pick(self, kind):
        self.pick = kind
        for b, k in ((self.click_fol_b, "foliage"), (self.click_neu_b, "neutral"), (self.measure_b, "card")):
            b.blockSignals(True)
            b.setChecked(kind == k)
            b.blockSignals(False)
        self.result.view.click_zooms = kind is None
        if kind:
            self.result.note({"foliage": "Click foliage in the photo", "neutral": "Click something neutral grey",
                              "card": "Click the grey card or white paper"}[kind])
        else:
            self.result.note("")

    def _view_clicked(self):
        if self.pick is None or self.image is None:
            return
        from PySide6.QtGui import QCursor
        from .. import engine as E
        v = self.result.view
        u, w_ = v.uv_at(v.mapFromGlobal(QCursor.pos()))
        if not (0 <= u <= 1 and 0 <= w_ <= 1):
            return
        kind = self.pick
        self._set_pick(None)
        img, e = self.image, self.edit
        fx, fy = E.view_to_s(img, e, u, w_)
        _, _, sw, sh = img.raw.crop
        xs, ys = float(fx) * sw, float(fy) * sh

        def work():
            from ..engine.color.autowb import patch_multipliers
            if kind == "foliage":
                return self.irm.foliage_at(img, xs, ys)
            return patch_multipliers(img, xs, ys, radius=4)

        def done(_, m):
            if m is None:
                self.jobbar.say("That spot is clipped or too dark to measure: try another.", T.AMBER)
                return
            if kind == "card":
                name, ok = QInputDialog.getText(self, "Measure this filter", "Name this filter (for example 'Hoya "
                                                "R72' or '590 nm'):", text=self.irm.FILTERS[e.infrared['filter']]
                                                ['label'])
                if not ok or not name.strip():
                    return
                cam = getattr(img.meta, "model", "") or ""
                self.irm.save_filter_wb(name.strip(), m, camera=cam, filter_key=e.infrared["filter"])
                self._fill_saved()
                self._set_wb(m, f"measured:{name.strip()}", f"Filter measured and saved: {name.strip()} (cyan).")
                return
            self._set_wb(m, "click_foliage" if kind == "foliage" else "neutral",
                         "Foliage clicked: made neutral." if kind == "foliage" else "Neutral clicked.")
        run_bg(work, "irpick", done, lambda _, err: self.jobbar.say(f"White balance: {err}", T.RED))

    def _set_wb(self, m, source: str, text: str):
        from ..engine.pipeline.edit import WhiteBalance
        m = np.asarray(m, float)
        m = m / m.min()
        self.edit.white_balance = WhiteBalance(mode="multipliers", multipliers=[round(float(v), 5) for v in m])
        self.edit.infrared["wb_source"] = source
        for sl, v in zip(self.mult_sl, m):
            sl.set_value(float(np.log(v)), emit=False)
        self._show_wb()
        self.set_chips([(f"Infrared: {self.irm.FILTERS[self.edit.infrared['filter']]['label']}",
                         T.CYAN if source.startswith("measured") else T.AMBER)])
        self._thumbs_and_meter_soon()
        self.changed(text)

    def auto_foliage(self):
        if self.image is None:
            return
        img = self.image
        run_bg(lambda: self.irm.foliage_multipliers(img), "irfol",
               lambda _, r: (self._set_wb(r[0], "foliage", r[1]["text"]) if r[0] is not None
                             else self.jobbar.say(r[1]["text"], T.AMBER)),
               lambda _, err: self.jobbar.say(f"Auto foliage: {err}", T.RED))

    def _mult_moved(self, *_):
        m = np.exp([sl.value() for sl in self.mult_sl])
        self._set_wb(m, "hand", "White balance set by hand.")

    def _fill_saved(self):
        cam = getattr(getattr(self.image, "meta", None), "model", "") or ""
        self.measured_pick.blockSignals(True)
        self.measured_pick.clear()
        saved = self.irm.load_filter_wbs(cam or None)
        self.measured_pick.addItem("Measured filters: none on this PC" if not saved else "Use a measured filter...",
                                   None)
        for d in saved:
            self.measured_pick.addItem(f"{d['name']}  ({d.get('camera') or 'any camera'})", d)
        self.measured_pick.blockSignals(False)
        from ..lens import hotspot as HS
        self.hs_saved.blockSignals(True)
        self.hs_saved.clear()
        hs = HS.load_profiles()
        self.hs_saved.addItem("Learnt hot spots: none on this PC" if not hs else "Use a learnt hot spot...", None)
        for d in hs:
            self.hs_saved.addItem(d.get("name", "hot spot"), d)
        self.hs_saved.blockSignals(False)

    def _measured_picked(self, idx):
        d = self.measured_pick.itemData(idx)
        if d:
            self._set_wb(d["multipliers"], f"measured:{d['name']}", f"White balance measured for {d['name']}.")

    def _preset_picked(self, *_):
        if self.image is None:
            return
        k = self.preset_pick.currentData()
        img, e, s = self.image, self.edit, self.strength.value()

        def done(_, ne):
            self.edit = ne
            self.controls_from_edit()
            self._thumbs_and_meter_soon()
            self.changed(f"Look: {self.irm.PRESET_INFO[k]['label']}")
        run_bg(lambda: self.irm.apply_preset(img, e, k, s), "irpreset", done,
               lambda _, err: self.jobbar.say(f"Look: {err}", T.RED))

    def _strength_moved(self, v):
        ir = self.edit.infrared
        if ir["bw"].get("on"):
            return
        ir["strength"] = round(float(v), 3)
        ir["mixer"] = [round(float(x), 6) for x in self.irm.mixer(ir.get("preset", "none"), v).ravel()]
        mix = np.asarray(ir["mixer"]).reshape(3, 3)
        for i, sp in enumerate(self.mix_spin):
            sp.blockSignals(True)
            sp.setValue(float(mix.ravel()[i]))
            sp.blockSignals(False)
        self.row_sum.setText("Row sums: " + "  ".join(f"{x:.2f}" for x in self.irm.row_sums(mix)))
        self.changed()

    def _palette_changed(self, *_):
        name = self.palette_pick.currentData()
        self.edit.infrared["palette"] = {"name": name, "amount": round(self.palette_amt.value() / 100.0, 3)} \
            if name else None
        self._show_preset()
        self.changed("Palette: added colour (not in the raw)." if name else "")

    def _bw_toggled(self, on):
        self.edit.infrared["bw"]["on"] = bool(on)
        self._bw_visibility()
        self._thumbs_and_meter_soon()
        self.changed("Black and white from the channels." if on else "Colour (the black and white settings are "
                     "kept).")

    def _mono_toggled(self, on):
        self.edit.infrared["mono_raw"] = bool(on)
        self.changed("Developed without demosaicing: every photosite a brightness sample." if on else
                     "Demosaiced as usual.")

    def _bw_moved(self, *_):
        bw = self.edit.infrared["bw"]
        bw["ev"] = [round(w.value(), 3) for w in self.ch_ev]
        bw["contrast"] = [round(w.value(), 1) for w in self.ch_con]
        bw["sep"] = [round(w.value(), 1) for w in self.ch_sep]
        bw["mix"] = [round(w.value() / 100.0, 4) for w in self.ch_mix]
        bw["keep"] = self.keep.isChecked()
        bw["preset"] = ""
        bw["recipe"] = ""
        if not bw.get("on"):                          # touching a B&W control shows the B&W result (plan 1.5c)
            bw["on"] = True
            self.bw_on.blockSignals(True)
            self.bw_on.setChecked(True)
            self.bw_on.blockSignals(False)
            self._bw_visibility()
        self._show_mix_sum()
        self.meter_timer.start()
        self.changed()

    def _bw_preset_picked(self, idx):
        k = self.bw_preset.itemData(idx)
        if not k or self.image is None:
            return
        img, e = self.image, self.edit

        def work():
            cam = self.irm.camera_half_wb(img, e) if k in ("soft_skin", "separation") else None
            return self.irm.bw_preset_weights(k, cam)

        def done(_, w):
            for c in range(3):
                self.ch_mix[c].set_value(100.0 * float(w[c]), emit=False)
            self._bw_moved()
            self.edit.infrared["bw"]["preset"] = k
            self.jobbar.say(f"Mix: {self.irm.BW_PRESET_INFO[k]['label']}"
                            + (" (estimated on this photo)" if k == "separation" else ""), T.DIM)
        run_bg(work, "irbwpreset", done, lambda _, err: self.jobbar.say(f"Mix preset: {err}", T.RED))

    def _fill_recipes(self):
        f = self.edit.infrared["filter"] if self.edit is not None else None
        self.recipe_pick.blockSignals(True)
        self.recipe_pick.clear()
        rs = self.irm.load_recipes(f)
        self.recipe_pick.addItem("B&W recipes: none saved for this filter" if not rs else "Apply a B&W recipe...",
                                 None)
        for r in rs:
            self.recipe_pick.addItem(r.get("name", "recipe"), r)
        self.recipe_pick.blockSignals(False)

    def _recipe_picked(self, idx):
        r = self.recipe_pick.itemData(idx)
        if not r:
            return
        self.edit.infrared["bw"] = self.irm.apply_recipe(self.edit.infrared["bw"], r)
        cam = getattr(self.image.meta, "model", "") or ""
        self.irm.remember_last(r.get("name", ""), cam, self.edit.infrared["filter"])
        self.controls_from_edit()
        self._thumbs_and_meter_soon()
        self.changed(f"B&W recipe: {r.get('name', '')}")

    def save_recipe(self):
        if self.edit is None:
            return
        name, ok = QInputDialog.getText(self, "Save as B&W recipe", "Name this black and white recipe:")
        if not ok or not name.strip():
            return
        cam = getattr(self.image.meta, "model", "") or ""
        self.irm.save_recipe(name.strip(), self.edit.infrared["bw"], camera=cam,
                             filter_key=self.edit.infrared["filter"])
        self._fill_recipes()
        self.jobbar.say(f"Saved B&W recipe '{name.strip()}' (on this PC; offered next time for this camera and "
                        "filter).", T.CYAN)

    def _ds_changed(self):
        self.ds.store(self.edit)
        self.changed()

    def _mixer_edited(self, *_):
        vals = [round(sp.value(), 4) for sp in self.mix_spin]
        ir = self.edit.infrared
        ir["mixer"] = vals
        ir["preset"] = ""
        self.preset_pick.setCurrentIndex(-1)
        self.row_sum.setText("Row sums: " + "  ".join(f"{v:.2f}" for v in self.irm.row_sums(vals)))
        self.changed()

    def _profile_picked(self, *_):
        self.edit.infrared["profile"] = self.profile_pick.currentData()
        self.changed()

    # ------------------------------------------------------------------------------------ hot spot
    def _show_hotspot(self):
        hs = (self.edit.infrared or {}).get("hotspot") if self.edit is not None else None
        if not hs or not hs.get("model"):
            self.hs_chip.setText("Hot spot: not measured")
            self._colour(self.hs_chip, T.DIM)
            self.hs_amount.setEnabled(False)
            return
        mdl = hs["model"]
        col = T.CYAN if mdl.get("frames", 1) >= 5 else T.AMBER
        self.hs_chip.setText(f"Hot spot: {mdl.get('stops', 0):.2f} stop, {mdl.get('source', 'measured')}")
        self._colour(self.hs_chip, col)
        self.hs_amount.setEnabled(True)

    def measure_hotspot(self):
        if self.image is None:
            return
        from ..lens import hotspot as HS
        img, e = self.image, self.edit

        def done(_, mdl):
            if mdl is None:
                self.jobbar.say("No hot spot found in this photo (or it is not flat enough to tell).", T.AMBER)
                return
            self.edit.infrared["hotspot"] = {"model": mdl, "amount": 1.0}
            self.hs_amount.set_value(100.0, emit=False)
            self._show_hotspot()
            self.changed(f"Hot spot measured: {mdl['stops']:.2f} stop (from one photo: estimated).")
        run_bg(lambda: HS.measure_image(img, e), "irhot", done,
               lambda _, err: self.jobbar.say(f"Hot spot: {err}", T.RED))

    def _hotspot_moved(self, v):
        hs = self.edit.infrared.get("hotspot")
        if hs:
            hs["amount"] = round(float(v) / 100.0, 3)
            self.changed()

    def _hotspot_saved_picked(self, idx):
        d = self.hs_saved.itemData(idx)
        if d and self.edit is not None:
            self.edit.infrared["hotspot"] = {"model": d["model"], "amount": 1.0}
            self.hs_amount.set_value(100.0, emit=False)
            self._show_hotspot()
            self.changed(f"Hot spot: {d.get('name', '')}")

    # ------------------------------------------------------------------------------------ fusion
    def add_visible(self):
        from PySide6.QtWidgets import QFileDialog
        from ..engine.io.rawfile import RAW_EXTENSIONS
        pats = " ".join(f"*{x}" for x in RAW_EXTENSIONS) + " *.tif *.tiff *.jpg *.jpeg"
        f, _ = QFileDialog.getOpenFileName(self, "The visible-light frame of the same scene",
                                           os.path.dirname(self.path) if self.path else "", f"Photos ({pats})")
        if f:
            self.visible_path = f
            self.fu_text.setText(f"Visible frame: {os.path.basename(f)}. Infrared frame: "
                                 f"{os.path.basename(self.path)}.")

    def build_fusion(self):
        if not (self.visible_path and self.path):
            self.jobbar.say("Add the visible frame first (same scene, tripod).", T.AMBER)
            return
        from ..engine.stack import fusion as FU
        from ..engine.stack.frames import result_name
        mode = self.fu_mode.currentData()
        self.save_sidecar()
        vis, ir_path = self.visible_path, self.path
        out = self.target(result_name([vis, ir_path], "IRfusion", ".tif"), ir_path)

        def work(progress, cancel):
            return FU.build(vis, ir_path, mode, out, progress, cancel)

        def done(rep):
            self.last_result = rep["path"]
            self.result_written.emit(rep["path"])
            self.jobbar.say(rep["text"], T.CYAN if rep.get("colour") == "cyan" else T.AMBER)
            self._refresh()
        self.jobbar.start(work, done, lambda err, c: self.failed("Fusion", err, c), keep=True,
                          text="Lining up the visible and infrared frames...")

    # ------------------------------------------------------------------------------------ rendering
    def render_now(self):
        if self.image is None or self.edit is None:
            return
        from .. import engine as E
        self._rtok += 1
        tok, img, e = self._rtok, self.image, self.edit
        ow, oh = E.output_size(img, e)
        sc = min(0.5, 1800.0 / max(ow, oh))
        need_before = self._before is None
        bw_view = self.ds.judge.isChecked() and (e.infrared or {}).get("bw", {}).get("on") is False and \
            any(abs(v) > 1e-6 for v in (e.detail.tonal, e.detail.crisp, e.detail.separation, e.detail.veil,
                                         e.detail.broad, e.microcontrast.amount))
        self.result.view.busy = True
        self.result.view.update()

        def work():
            from .main import bw8
            a8 = display8(E.render(img, e, scale=sc))
            if bw_view:
                a8 = bw8(a8)
            b8 = None
            if need_before:
                b8 = display8(E.render(img, E.Edit(), scale=sc))
            return a8, b8

        def done(t, res):
            if t != self._rtok:
                return
            a8, b8 = res
            if b8 is not None:
                self._before = to_qimage(b8)
            ir = self.edit.infrared or {}
            col = T.CYAN if str(ir.get("wb_source", "")).startswith("measured") else T.AMBER
            self.result.ref_chip = ("Camera as shot (B)", T.DIM)
            self.result.set_images(to_qimage(a8), self._before, ow, oh,
                                   chip_text="Infrared" + (": black and white" if ir.get("bw", {}).get("on") else ""),
                                   chip_colour=col)
            self.result.view.busy = False

        def fail(t, err):
            self.result.view.busy = False
            self.jobbar.say(f"Rendering: {err}", T.RED)
        run_bg(work, tok, done, fail)

    def _thumbs_and_meter_soon(self):
        self.meter_timer.start()

    def _measure_bw(self):
        """Channel thumbnails and, with black and white on, the noise meter (both in the background)."""
        if self.image is None:
            return
        img, e = self.image, self.edit
        bw_on = (e.infrared or {}).get("bw", {}).get("on")

        def work():
            cam = self.irm.camera_half_wb(img, e)
            th = self.irm.channel_thumbs(cam)
            nm = self.irm.noise_cost(img, e) if bw_on else None
            return th, nm

        def done(_, res):
            th, nm = res
            for lab, a in zip(self.thumbs, th):
                pm = QPixmap.fromImage(gray_qimage(a)).scaled(lab.width() - 4, lab.height() - 4,
                                                             Qt.AspectRatioMode.KeepAspectRatio,
                                                             Qt.TransformationMode.SmoothTransformation)
                lab.setPixmap(pm)
            if nm:
                self.noise_meter.setText(nm["text"])
                self._colour(self.noise_meter, T.CYAN if nm["colour"] == "cyan" else T.AMBER)
            else:
                self.noise_meter.setText("Noise: shown with black and white on")
                self._colour(self.noise_meter, T.DIM)
        run_bg(work, "irmeter", done, lambda *_: None)

    # ------------------------------------------------------------------------------------ actions
    def save_sidecar(self):
        if self.edit is None or not self.path or not getattr(self, "_dirty", False):
            return
        from .. import engine as E
        try:
            E.save_edit(self.path, self.edit)
            self._dirty = False
            self.sidecar_written.emit(self.path)
        except OSError as err:
            self.jobbar.say(f"Could not save the edit beside {os.path.basename(self.path)} ({err}): read-only "
                            "drive?", T.AMBER)

    def build_full(self):
        """Open in Develop (Ctrl+Enter): save the edit, then continue there on the same raw."""
        if not self.path:
            return
        self._dirty = True
        self.save_sidecar()
        self.open_result.emit(self.path)

    def apply_to_shoot(self):
        if self.edit is None:
            return
        from .. import engine as E
        part = self.irm.ir_part(self.edit)
        others = [p for p, ok, _ in self.items if ok and p != self.path]
        if not others:
            self.jobbar.say("Only one photo in the strip: give the tab the whole shoot from the Library.", T.AMBER)
            return
        self._dirty = True
        self.save_sidecar()
        bad = []
        for p in others:
            try:
                E.save_edit(p, self.irm.paste_ir(E.load_edit(p), part))
                self.sidecar_written.emit(p)
            except (OSError, ValueError):
                bad.append(os.path.basename(p))
        msg = f"Infrared settings applied to {len(others) - len(bad)} photo{'s' if len(others) - len(bad) != 1 else ''}."
        if bad:
            msg += f" Could not write: {', '.join(bad[:5])} (read-only drive?)."
        self.jobbar.say(msg, T.CYAN if not bad else T.AMBER)

    def reset(self):
        if self.image is None:
            return
        img = self.image
        from .. import engine as E
        base = E.load_edit(self.path)
        base.infrared = None

        def done(_, res):
            self.edit, self.report = res
            self.controls_from_edit()
            self._thumbs_and_meter_soon()
            self.changed(self._open_text())
        run_bg(lambda: self.irm.start(img, base), "irreset", done,
               lambda _, err: self.jobbar.say(f"Start again: {err}", T.RED))

    def turn_off(self):
        if self.edit is None:
            return
        self.edit = self.irm.turn_off(self.edit)
        self._dirty = True
        self.save_sidecar()
        self.jobbar.say("Infrared treatment off for this photo (its settings are kept: opening it here again "
                        "turns them back on).", T.DIM)
        self.render_now()

    def hideEvent(self, e):
        self.save_sidecar()
        super().hideEvent(e)
