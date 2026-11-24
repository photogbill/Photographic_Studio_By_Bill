"""Export dialog and file naming, shared by Develop, the Library's Export and Process picks (section 15, 24.7 C)."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QInputDialog, QPushButton,
                               QSpinBox, QVBoxLayout)

from .. import paths
from . import theme as T
from .common import Settings, dot_icon, label


class ExportDialog(QDialog):
    def __init__(self, files: list, settings: Settings, parent=None):
        super().__init__(parent)
        self.files, self.settings = files, settings
        self.setWindowTitle("Export")
        self.setMinimumWidth(460)
        v = QVBoxLayout(self)
        v.addWidget(label(f"Export {len(files)} photo{'s' if len(files) != 1 else ''}", "h2"))
        prow = QHBoxLayout()
        self.preset = QComboBox()
        self.preset.setToolTip("Export presets: the built-in ones, then yours (kept in settings\\app.json)")
        self.preset.activated.connect(self._preset_picked)
        save_p = QPushButton("Save preset...")
        save_p.clicked.connect(self._save_preset)
        prow.addWidget(self.preset, 1)
        prow.addWidget(save_p)
        v.addLayout(prow)
        grid = QGridLayout()
        self.fmt = QComboBox()
        self.fmt.addItems(["16-bit TIFF", "16-bit PNG", "8-bit PNG", "JPEG", "32-bit FITS (astronomy)"])
        self.fmt.setItemData(4, "For Siril, PixInsight and other astronomy programs: 32-bit float, linear light "
                                "(no gamma), nothing rounded. Weak at: photo programs and websites cannot open it.",
                             Qt.ToolTipRole)
        from ..engine.io import fits as _fx
        if not _fx.available():                              # astropy missing: not offered rather than failing
            self.fmt.model().item(4).setEnabled(False)
            self.fmt.setItemData(4, _fx.MISSING, Qt.ToolTipRole)
        self.space = QComboBox()
        self.space.addItems(["Adobe RGB", "sRGB", "ProPhoto RGB", "Display P3"])
        self.space.setCurrentIndex(1)                        # sRGB unless a preset says otherwise (Bill, 2026-11-23)
        self.space.setToolTip("sRGB: what every screen, phone and website expects (the default). Adobe RGB and "
                              "ProPhoto RGB keep more colour for printing and further editing; use them for prints "
                              "and archives. Display P3: Apple screens.")
        self.web = QCheckBox("Save for web: the smallest file at the best quality (sRGB JPEG, 2048 px)")
        self.web.setToolTip("Tries JPEG qualities and picks the smallest file that still looks the same as the "
                            "best-quality one (checked on detail and on colour), progressive, sharpened for screens. "
                            "Named NAME_web_Studio.jpg so it never replaces a full-size export.")
        self.size = QSpinBox()
        self.size.setRange(0, 20000)
        self.size.setSpecialValueText("Full size")
        self.size.setSuffix(" px long edge")
        self.dest = QLineEdit_(settings.get("export_dir", str(paths.get("OUTPUTS") / "export")))
        browse = QPushButton("...")
        browse.clicked.connect(self.browse)
        self.osh = QComboBox()
        self.osh.addItems(["None", "Screen", "Glossy paper", "Matte paper"])
        self.osh.setToolTip("Output sharpening, done after resizing, sized for where the picture will be seen")
        self.osh_amt = QComboBox()
        self.osh_amt.addItems(["Low", "Standard", "High"])
        self.osh_amt.setCurrentIndex(1)
        for r, (name, w) in enumerate([("Format", self.fmt), ("Colour space", self.space), ("Size", self.size)]):
            grid.addWidget(label(name, "dim"), r, 0)
            grid.addWidget(w, r, 1, 1, 2)
        grid.addWidget(label("Sharpen for", "dim"), 3, 0)
        grid.addWidget(self.osh, 3, 1)
        grid.addWidget(self.osh_amt, 3, 2)
        grid.addWidget(self.web, 5, 0, 1, 3)
        self.web.toggled.connect(self._web_toggled)
        grid.addWidget(label("Folder", "dim"), 4, 0)
        grid.addWidget(self.dest, 4, 1)
        grid.addWidget(browse, 4, 2)
        v.addLayout(grid)
        v.addWidget(label("Each file is named after its photo with _Studio added (IMG_0001_Studio.tif), so it never "
                          "has your original's name. Exporting again replaces the Studio's own earlier export. Edits "
                          "come from each photo's sidecar. Serial numbers and GPS are never written.", "dim",
                          wrap=True))
        self._fill_presets(settings.get("export_preset", ""))
        self._preset_picked(self.preset.currentIndex())
        for w in (self.fmt, self.space, self.osh, self.osh_amt):
            w.activated.connect(lambda _: self.preset.setCurrentIndex(0))
        self.size.editingFinished.connect(lambda: self.preset.setCurrentIndex(0)
                                          if self.size.value() != self._preset_size else None)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        go = QPushButton("Export")
        go.setObjectName("primary")
        go.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(go)
        v.addLayout(row)

    # (format index, colour space index, long edge 0 = full size, sharpen-for index, sharpen amount index)
    PRESETS = {"Archive: 16-bit TIFF, Adobe RGB, full size": (0, 0, 0, 0, 1),
               "Print: 16-bit TIFF, ProPhoto RGB, full size": (0, 2, 0, 2, 1),
               "Web: JPEG, sRGB, 2048 px": (3, 1, 2048, 1, 1),
               "Save for web: smallest file, best quality": (3, 1, 2048, 1, 1, 1),
               "Social: JPEG, sRGB, 1080 px": (3, 1, 1080, 1, 1)}

    def _fill_presets(self, current=""):
        self.preset.clear()
        self.preset.addItem("Custom (as set below)", None)
        for name, val in self.PRESETS.items():
            self.preset.addItem(dot_icon(T.VIOLET), name, list(val))
        for name, val in (self.settings.get("export_presets") or {}).items():
            self.preset.addItem(dot_icon(T.CYAN), name, list(val))
        self.preset.setCurrentIndex(max(0, self.preset.findText(current)))

    def _preset_picked(self, idx):
        val = self.preset.itemData(idx)
        self._preset_size = self.size.value()
        if val:
            val = list(val) + [0, 1][len(val) - 3:] if len(val) < 5 else list(val)
            self.fmt.setCurrentIndex(int(val[0]))
            self.space.setCurrentIndex(int(val[1]))
            self.size.setValue(int(val[2]))
            self.osh.setCurrentIndex(int(val[3]))
            self.osh_amt.setCurrentIndex(int(val[4]))
            self._preset_size = int(val[2])
            self.web.setChecked(len(val) > 5 and bool(val[5]))

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save export preset", "Name:",
                                        text=f"{self.fmt.currentText()}, {self.space.currentText()}")
        if not ok or not name.strip():
            return
        presets = dict(self.settings.get("export_presets") or {})
        presets[name.strip()] = [self.fmt.currentIndex(), self.space.currentIndex(), self.size.value(),
                                 self.osh.currentIndex(), self.osh_amt.currentIndex()]
        self.settings.set("export_presets", presets)
        self._fill_presets(name.strip())
        self._preset_picked(self.preset.currentIndex())

    def _web_toggled(self, on):
        for w in (self.fmt, self.space):
            w.setEnabled(not on)
        if on:
            self.fmt.setCurrentIndex(3)
            self.space.setCurrentIndex(1)
            if not self.size.value():
                self.size.setValue(2048)

    def browse(self):
        d = QFileDialog.getExistingDirectory(self, "Export folder", self.dest.text())
        if d:
            self.dest.setText(d)

    def spec(self):
        from ..engine import ExportSpec
        fmt, bits = [("tiff", 16), ("png", 16), ("png", 8), ("jpeg", 8), ("fits", 16)][self.fmt.currentIndex()]
        space = ["adobergb", "srgb", "prophoto", "displayp3"][self.space.currentIndex()]
        self.settings.set("export_dir", self.dest.text())
        self.settings.set("export_preset", self.preset.currentText() if self.preset.currentIndex() > 0 else "")
        return ExportSpec(format=fmt, bits=bits, space=space, long_edge=self.size.value() or None,
                          sharpen_for=["none", "screen", "glossy", "matte"][self.osh.currentIndex()],
                          sharpen_amount=["low", "standard", "high"][self.osh_amt.currentIndex()],
                          web=self.web.isChecked())


def QLineEdit_(text):
    from PySide6.QtWidgets import QLineEdit
    w = QLineEdit(text)
    return w


def version_export_path(raw_path: str, version: int, spec, dest: str) -> str:
    """File name for exporting a virtual copy: IMG_0001_Dark_one_Studio.tif beside IMG_0001_Studio.tif (the
    master)."""
    import re
    from .. import engine as E
    from ..engine.io.export import export_name
    tag = re.sub(r"[^\w\-]+", "_", E.list_versions(raw_path)[version]).strip("_") or f"copy{version}"
    return os.path.join(dest, export_name(raw_path, spec, extra=tag))
