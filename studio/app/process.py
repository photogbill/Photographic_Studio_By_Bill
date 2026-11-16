"""Process picks (plan 24.7 C): the step from "decided" to "processed". One dialog (which photos, what to apply,
the export preset and folder), then the work runs in the background with a progress bar and Cancel (6.1)."""
from __future__ import annotations

import os
import threading

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QButtonGroup, QFrame, QPushButton, QRadioButton, QVBoxLayout

from . import theme as T
from .common import label, log, run_bg
from .exporting import ExportDialog

WHICH = (("picks", "Picks"), ("selected", "Selected"), ("shown", "Everything shown"))
MODES = (("own", "Each photo's own edit",
          "Photos you have not edited are exported with the default settings."),
         ("auto", "Make it good on photos that have no edit yet",
          "Noise for the ISO, lens fixes, gentle sharpening; never brightness. Saved into each photo's edit, so "
          "Develop shows what it did. Photos you have edited keep their own edit."))


class ProcessDialog(ExportDialog):
    def __init__(self, settings, picks: list, selected: list, shown: list, which: str | None = None, parent=None):
        super().__init__(shown, settings, parent)
        self.lists = {"picks": picks, "selected": selected, "shown": shown}
        self.setWindowTitle("Process picks")
        lay = self.layout()
        head = lay.itemAt(0).widget()
        head.setText("Process photos")
        box = QFrame()
        bv = QVBoxLayout(box)
        bv.setContentsMargins(0, 4, 0, 8)
        bv.addWidget(label("Which photos", "dim"))
        self.which_g = QButtonGroup(self)
        default = which or ("picks" if picks else ("selected" if selected else "shown"))
        for k, name in WHICH:
            n = len(self.lists[k])
            b = QRadioButton(f"{name} ({n})")
            b.setProperty("key", k)
            b.setEnabled(n > 0)
            b.setChecked(k == default and n > 0)
            self.which_g.addButton(b)
            bv.addWidget(b)
        if self.which_g.checkedButton() is None:
            for b in self.which_g.buttons():
                if b.isEnabled():
                    b.setChecked(True)
                    break
        bv.addSpacing(6)
        bv.addWidget(label("What to apply", "dim"))
        self.mode_g = QButtonGroup(self)
        mode = settings.get("process_mode", "own")
        for k, name, text in MODES:
            b = QRadioButton(name)
            b.setProperty("key", k)
            b.setToolTip(text)
            b.setChecked(k == mode)
            self.mode_g.addButton(b)
            bv.addWidget(b)
            bv.addWidget(label("     " + text, "dim", wrap=True))
        if self.mode_g.checkedButton() is None:
            self.mode_g.buttons()[0].setChecked(True)
        lay.insertWidget(1, box)
        for b in self.findChildren(QPushButton):
            if b.text() == "Export":
                b.setText("Process")
                self.go = b
        self.which_g.buttonToggled.connect(self._which_changed)
        self._which_changed()

    def _which_changed(self, *_):
        n = len(self.result_files())
        if hasattr(self, "go"):
            self.go.setText(f"Process {n} photo{'s' if n != 1 else ''}")
            self.go.setEnabled(n > 0)

    def which(self) -> str:
        b = self.which_g.checkedButton()
        return b.property("key") if b is not None else "picks"

    def mode(self) -> str:
        b = self.mode_g.checkedButton()
        k = b.property("key") if b is not None else "own"
        self.settings.set("process_mode", k)
        return k

    def result_files(self) -> list:
        return list(self.lists.get(self.which(), []))


class _Progress(QObject):
    step = Signal(int, str)


def process_files(files: list, spec, dest: str, mode: str, cancel=None, on_step=None) -> dict:
    """The work itself (engine calls only, so it can be tested without a window). Returns a report."""
    from .. import engine as E
    out, failed, auto = [], [], 0
    os.makedirs(dest, exist_ok=True)
    for i, f in enumerate(files):
        if cancel is not None and cancel.is_set():
            break
        if on_step:
            on_step(i, os.path.basename(f))
        try:
            img = E.open(f)
            edit = E.load_edit(f)
            if mode == "auto" and not E.has_edit(f):
                edit, _notes = E.make_it_good(img, edit)
                auto += 1
                try:
                    E.save_edit(f, edit)
                except OSError as err:                 # a read-only drive: the export still gets the edit
                    log.warning("process: could not save Make it good into %s: %s", f, err)
            out.append(E.export(img, edit, spec, out_dir=dest))
            img.drop_cache()
        except Exception as err:                       # noqa: BLE001  (one bad photo does not stop the rest)
            log.warning("process: %s failed: %s", f, err)
            failed.append((os.path.basename(f), str(err)))
    return {"out": out, "failed": failed, "auto": auto, "cancelled": bool(cancel is not None and cancel.is_set()),
            "dest": dest, "total": len(files)}


def run_process(page, files: list, spec, dest: str, mode: str) -> None:
    """Run Process picks in the background, with the Library's progress bar and Cancel."""
    cancel = threading.Event()
    sig = _Progress(page)
    n = len(files)
    page.progress.show()
    page.prog_bar.setRange(0, n)
    page.prog_bar.setValue(0)
    page.prog_text.setText(f"Processing {n} photo{'s' if n != 1 else ''}...")
    page.prog_text.setStyleSheet("")
    page.prog_cancel.show()
    page.prog_open.hide()
    try:
        page.prog_cancel.clicked.disconnect()
    except (RuntimeError, TypeError):
        pass
    page.prog_cancel.clicked.connect(lambda: (cancel.set(), page.prog_text.setText("Stopping after this photo...")))
    sig.step.connect(lambda i, name: (page.prog_bar.setValue(i),
                                      page.prog_text.setText(f"Processing {i + 1} of {n}: {name}")))

    def done(_, rep):
        page.prog_bar.setValue(len(rep["out"]) + len(rep["failed"]))
        page.prog_cancel.hide()
        msg = f"Processed {len(rep['out'])} of {rep['total']} photo{'s' if rep['total'] != 1 else ''} into {dest}"
        if rep["auto"]:
            msg += f" (Make it good on {rep['auto']})"
        if rep["failed"]:
            msg += f". {len(rep['failed'])} failed: " + ", ".join(nm for nm, _ in rep["failed"][:4])
            page.prog_text.setStyleSheet(f"color:{T.AMBER};")
        if rep["cancelled"]:
            msg += ". Cancelled."
        page.prog_text.setText(msg + ".")
        page.prog_open.show()
        try:
            page.prog_open.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        page.prog_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(dest)))
        page.last_process = rep

    def fail(_, err):
        page.prog_cancel.hide()
        page.prog_text.setText(f"Process picks stopped: {err}")
        page.prog_text.setStyleSheet(f"color:{T.RED};")
    run_bg(lambda: process_files(files, spec, dest, mode, cancel, lambda i, nm: sig.step.emit(i, nm)), "process",
           done, fail, keep=True)
