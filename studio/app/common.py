"""Small pieces shared by the Studio's workspaces: background jobs, settings, little widgets, the lens picker.

Split out of main.py (2026-11-15, plan section 24.8) so the Library, Review and Import code can live in their own
files without importing the whole Develop workspace.
"""
from __future__ import annotations

import io
import json

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from .. import log as slog, paths
from . import theme as T

log = slog.get("app")
POOL = QThreadPool.globalInstance()
_LIVE: set = set()
_SHUTTING_DOWN = False                               # the Studio is closing: no new background work


# ============================================================================================ background jobs
class _Sig(QObject):
    done = Signal(object, object)
    failed = Signal(object, str)


class Job(QRunnable):
    def __init__(self, fn, token=None, on_done=None, on_fail=None, keep=False):
        super().__init__()
        self.fn, self.token, self.sig = fn, token, _Sig()
        self.keep = keep                             # writes files the person asked for: finished before exit
        self.running = False
        self.finished = False
        if on_done:
            self.sig.done.connect(on_done)
        if on_fail:
            self.sig.failed.connect(on_fail)
        _LIVE.add(self)

    def run(self):
        self.running = True
        try:
            r = self.fn()
            self._emit(self.sig.done, self.token, r)
        except Exception as e:                       # every exception is logged with its traceback (rule 0.3.5)
            if not _SHUTTING_DOWN:
                slog.log_exception("background job failed", log)
            self._emit(self.sig.failed, self.token, str(e))
        finally:
            self.running = False
            self.finished = True
            if _SHUTTING_DOWN:
                _LIVE.discard(self)
            else:
                QTimer.singleShot(0, lambda: _LIVE.discard(self))

    @staticmethod
    def _emit(signal, *args):
        # the window the result was for may be gone (the Studio closing): drop the result, do not crash the exit
        try:
            signal.emit(*args)
        except RuntimeError:
            pass


def run_bg(fn, token=None, on_done=None, on_fail=None, keep=False):
    """Run ``fn`` in the background; ``on_done(token, result)`` / ``on_fail(token, message)`` come back on the
    window's thread. ``keep=True`` for work that writes files the person asked for (export, import, process picks,
    measured profiles): closing the Studio waits for it (after asking). Everything else (previews, measurements of
    the open photo) is simply dropped when the Studio closes."""
    if _SHUTTING_DOWN:
        return
    POOL.start(Job(fn, token, on_done, on_fail, keep))


def kept_jobs_running() -> int:
    """How many background jobs that write the person's files are still at work."""
    return sum(1 for j in list(_LIVE) if j.keep and not j.finished)


def begin_shutdown() -> None:
    global _SHUTTING_DOWN
    _SHUTTING_DOWN = True


def to_qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a)
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def display8(arr_pp: np.ndarray) -> np.ndarray:
    from ..engine.io.export import ExportSpec, to_output
    return to_output(arr_pp, ExportSpec(format="jpeg", space="srgb", bits=8))


def load_thumb(path: str, size: int = 260) -> np.ndarray:
    """Embedded camera preview, oriented, reduced (Library and filmstrip: opens folders instantly)."""
    import rawpy
    from PIL import Image
    from ..engine.pipeline.pipeline import apply_orientation
    with rawpy.imread(path) as r:
        flip = r.sizes.flip
        th = r.extract_thumb()
    if th.format == rawpy.ThumbFormat.JPEG:
        im = Image.open(io.BytesIO(th.data))
        im.draft("RGB", (size * 2, size * 2))
        im = im.convert("RGB")
    else:
        im = Image.fromarray(th.data)
    im.thumbnail((size, size))
    return np.ascontiguousarray(apply_orientation(np.asarray(im), flip))


# ============================================================================================ settings
class Settings:
    def __init__(self):
        self.path = paths.get("SETTINGS") / "app.json"
        try:
            self.d = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.d = {}

    def get(self, k, default=None):
        return self.d.get(k, default)

    def set(self, k, v):
        self.d[k] = v
        try:
            self.path.write_text(json.dumps(self.d, indent=2), encoding="utf-8")
        except OSError:
            pass


# ============================================================================================ small widgets
def label(text, name=None, wrap=False):
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


def chip(text, colour=T.DIM):
    lb = QLabel(text)
    lb.setObjectName("chip")
    lb.setStyleSheet(f"color:{colour}; border-color:{T.alpha(colour, 0.33)};")
    return lb


def qalpha(colour: str, a: float) -> QColor:
    """'#rrggbb' with opacity as a QColor (QColor does not read the style-sheet 'rgba(...)' form)."""
    c = QColor(colour)
    c.setAlphaF(a)
    return c


def card():
    f = QFrame()
    f.setObjectName("card")
    return f


def dot_icon(colour: str, size: int = 12) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(colour))
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pm)


def hbox(*widgets, stretch=True) -> QWidget:
    """A row of widgets as one widget (for Group.add)."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        if isinstance(x, tuple):
            lay.addWidget(*x)
        else:
            lay.addWidget(x)
    if stretch:
        lay.addStretch(1)
    return w


GLOW_HOWTO = ("Measure your lens's glow once (per lens, and per aperture if you change it):\n\n"
              "1. In a dark room, put one small light in the middle of the frame: a small LED torch pointed at the "
              "camera 3-5 metres away (a candle works if it is far enough to look small). Nothing else bright in "
              "view.\n"
              "2. Camera on a tripod, manual exposure, the aperture you use with this lens.\n"
              "3. Shoot a bracket without moving anything: start where the light is NOT blown at all (it looks "
              "like a dim dot), then 2 stops longer each time, 6-8 frames, the last one many stops over.\n"
              "4. In the Library select those frames and press Measure lens glow.\n\n"
              "From then on, 'measure from lens glow' reads how bright and what colour each blown light was from the "
              "glow around it. The profile is kept on this PC only (settings\\glow).")


# ============================================================================================ lens tagging
class LensDialog(QDialog):
    """Pick (or add) the lens for photos taken with contactless glass. Writes nothing itself: the caller stores
    the choice in each photo's sidecar."""

    def __init__(self, parent=None, current: str = "", count: int = 1):
        super().__init__(parent)
        from PySide6.QtWidgets import QDoubleSpinBox, QLineEdit
        from ..engine.lens import library as lib
        self.setWindowTitle("Which lens?")
        self.setMinimumWidth(480)
        v = QVBoxLayout(self)
        v.addWidget(label(f"Lens for {count} photo{'s' if count != 1 else ''}", "h2"))
        v.addWidget(label("Adapted vintage lenses tell the camera nothing. Pick the lens once for the shoot: it is "
                          "kept in each photo's sidecar and written into exported files.", "dim", wrap=True))
        self.combo = QComboBox()
        self.combo.addItem("(none: use what the camera recorded)", "")
        for e in lib.list_lenses():
            self.combo.addItem(e.label(), e.name)
        i = self.combo.findData(current)
        self.combo.setCurrentIndex(max(i, 0))
        v.addWidget(self.combo)
        v.addSpacing(8)
        v.addWidget(label("Or add a lens of your own (kept private on this PC):", "dim"))
        grid = QGridLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Voigtlander Heliar 15cm f/4.5")
        self.focal = QDoubleSpinBox()
        self.focal.setRange(0, 2000)
        self.focal.setSuffix(" mm")
        self.focal.setSpecialValueText("focal length unknown")
        self.aperture = QDoubleSpinBox()
        self.aperture.setRange(0, 64)
        self.aperture.setPrefix("f/")
        self.aperture.setSpecialValueText("widest aperture unknown")
        self.character = QCheckBox("Character lens (its softness or glow is the point: no automatic sharpening)")
        for r, (n, w) in enumerate((("Name", self.name), ("Focal length", self.focal), ("Widest", self.aperture))):
            grid.addWidget(label(n, "dim"), r, 0)
            grid.addWidget(w, r, 1)
        v.addLayout(grid)
        v.addWidget(self.character)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Use this lens")
        ok.setObjectName("primary")
        ok.clicked.connect(self._ok)
        row.addWidget(cancel)
        row.addWidget(ok)
        v.addLayout(row)
        self.choice = current

    def _ok(self):
        from ..engine.lens import library as lib
        name = self.name.text().strip()
        if name:
            lib.add_lens(lib.LensEntry(name=name, focal_mm=self.focal.value() or None,
                                       max_aperture=self.aperture.value() or None,
                                       character=self.character.isChecked()))
            self.choice = name
        else:
            self.choice = self.combo.currentData() or ""
        self.accept()


def tag_lens(files: list, lens: str) -> int:
    """Store a lens tag in each photo's sidecar. Returns how many were written."""
    from .. import engine as E
    n = 0
    for f in files:
        try:
            e = E.load_edit(f)
            e.lens.tag = lens
            E.save_edit(f, e)
            n += 1
        except OSError as err:
            log.warning("could not tag %s (read-only drive?): %s", f, err)
    return n


STARS = ["", "★", "★★", "★★★", "★★★★", "★★★★★"]


