"""'What Clean removed' (BUILD_PLAN_NOISE E.4): the proof, for the open photo, that Clean took noise and nothing else.

Three views of the same piece of the photo, photosite for photosite: before Clean, after Clean, and the difference
between them made stronger so it can be seen. The difference is shown in units where this camera's noise is
exactly 1 at every brightness, so pure noise looks like even television static; anything of the picture in it
(an edge, hair, a weave) means detail went out with the noise. What the residual test finds that is not noise is
marked in magenta, and the verdict is said in words and colour.

The pictures are luminance of the raw data (before colour is made), so the three line up exactly.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from .. import log as slog
from . import theme as T
from .common import chip, label, report_problem, run_bg, to_qimage

log = slog.get("app.removed")

GAIN = 24.0                 # grey levels per unit of noise in the "removed" view (noise of strength 1 = +-24 of 255)
RIM = 32                    # plane pixels left off each side: a piece's rim is cleaned with less to go on
MAGENTA = np.array([255.0, 79.0, 216.0], np.float32)
WHAT = ("Clean should take out noise and nothing else. This is the proof for this photo.<br>"
        "<b>Left:</b> a piece of the photo before Clean. <b>Middle:</b> after. <b>Right:</b> what was taken out, "
        "made stronger so you can see it.<br>"
        "If the right-hand picture looks like <b>television static</b>, with nothing of the photo in it, only noise "
        "was removed. If you can see edges, hair or texture in it, detail went out with the noise: lower the "
        "strength, or leave the detail guard on. Anything the test finds that is not noise is marked in "
        "<span style='color:#ff4fd8'><b>magenta</b></span>.")


def verdict(n: dict) -> tuple:
    """(colour, plain words) for the numbers of one piece (pipeline.clean_removed)."""
    share = float(n.get("not_noise") or 0.0)
    strength = float(n.get("strength") or 0.0)
    if share < 0.005:
        col, head = T.GREEN, "Only noise was removed here."
    elif share < 0.03:
        col, head = T.AMBER, f"Almost only noise: {100 * share:.1f}% of what was removed is picture."
    else:
        col, head = T.RED, f"Detail was removed here: {100 * share:.1f}% of what was removed is picture, not noise."
    how = (f"Clean ran with {n.get('method', '')} at luminance {n.get('luma', 0):.2f}, colour "
           f"{n.get('chroma', 0):.2f}" + (" (chosen by the photo)" if n.get("auto") else " (your sliders)"))
    if strength > 1.08:
        took = (f"What was removed is {strength:.2f} times as strong as this camera's noise: more than the noise "
                f"went out.")
    elif strength > 0.93:
        took = "What was removed is as strong as this camera's noise: all of the noise, and no more."
    elif strength > 0.5:
        took = (f"What was removed has {strength:.2f} of the strength of this camera's noise: most of it; the "
                f"rest stays as fine grain.")
    else:
        took = (f"Little was taken out of this piece ({strength:.2f} of the strength of this camera's noise): it "
                f"holds fine detail that Clean would not risk, so most of its grain stays.")
    guard = ""
    if n.get("guard") and n.get("returned") is not None:
        guard = f" The detail guard gave back {100 * float(n['returned']):.2f}% of what Clean first removed."
    elif not n.get("guard"):
        guard = " The detail guard is off in this view."
    return col, f"{head} {took}{guard} {how}."


def panes(res: dict) -> tuple:
    """(before, after, removed) as uint8 RGB pictures of the inner part of a piece."""
    cut = (slice(RIM, -RIM), slice(RIM, -RIM))
    b, a, r, nn = (np.asarray(res[k], np.float32)[cut] for k in ("before", "after", "removed", "not_noise"))
    lo, hi = np.percentile(b, 0.5), np.percentile(b, 99.8)
    grey = lambda v: np.clip((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0      # noqa: E731
    rgb = lambda g: np.repeat(g[..., None], 3, 2)                                 # noqa: E731
    rem = rgb(np.clip(128.0 + GAIN * r, 0.0, 255.0))
    from scipy.ndimage import uniform_filter
    mark = np.clip((uniform_filter(np.abs(nn), 3) - 0.25) / 0.75, 0.0, 1.0)[..., None]
    rem = rem * (1.0 - 0.75 * mark) + MAGENTA * (0.75 * mark)
    return tuple(np.ascontiguousarray(np.rint(x).astype(np.uint8)) for x in (rgb(grey(b)), rgb(grey(a)), rem))


class RemovedDialog(QDialog):
    def __init__(self, image, edit, parent=None, sync: bool = False):
        super().__init__(parent)
        from ..engine.pipeline.pipeline import removed_boxes
        self.setWindowTitle("What Clean removed")
        self.image, self.edit, self.sync = image, edit, sync
        self.boxes = removed_boxes(image)
        self.cache, self.results = {}, {}
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(10)
        v.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)   # the window is as big as its pictures
        what = label(WHAT, wrap=True)
        what.setFixedWidth(3 * 386 + 24)
        v.addWidget(what)
        row = QHBoxLayout()
        row.addWidget(label("Piece of the photo:", "dim"))
        self.where = QComboBox()
        self.where.addItems([n for n, _ in self.boxes])
        self.where.setToolTip("Five pieces of the frame at full size. Look at the one with the finest detail.")
        row.addWidget(self.where)
        self.no_guard = QCheckBox("Show it without the detail guard")
        self.no_guard.setToolTip("See what Clean would have removed if the detail guard were off: the difference "
                                 "between the two is what the guard saves.")
        row.addWidget(self.no_guard)
        row.addStretch(1)
        v.addLayout(row)
        pics = QHBoxLayout()
        self.views = []
        pics.setSpacing(12)
        for cap in ("Before Clean", "After Clean", "What was removed (made stronger)"):
            col = QVBoxLayout()
            col.setSpacing(4)
            col.addWidget(label(cap, "dim"))
            lb = QLabel()
            lb.setFixedSize(386, 386)
            lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lb.setStyleSheet(f"border:1px solid {T.LINE}; background:{T.PANEL};")
            col.addWidget(lb)
            pics.addLayout(col)
            self.views.append(lb)
        pics.addStretch(1)
        v.addLayout(pics)
        self.verdict = chip("Working...")
        self.verdict.setWordWrap(True)
        self.verdict.setFixedWidth(3 * 386 + 24)
        self.verdict.setMinimumHeight(54)
        v.addWidget(self.verdict)
        shown = label("Shown: the brightness of the raw data before colour is made, each photosite of the sensor "
                      "twice its size. Noise of normal strength is the grey static you see when only noise was "
                      "removed.", "dim", wrap=True)
        shown.setFixedWidth(3 * 386 + 24)
        v.addWidget(shown)
        foot = QHBoxLayout()
        foot.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        foot.addWidget(close)
        v.addLayout(foot)
        self.where.currentIndexChanged.connect(lambda *_: self.show_piece())
        self.no_guard.toggled.connect(lambda *_: self.show_piece())
        self.show_piece()

    def _key(self) -> tuple:
        return (self.where.currentIndex(), self.no_guard.isChecked())

    def show_piece(self):
        key = self._key()
        if key in self.cache:
            return self._show(key, self.cache[key])
        self.verdict.setText("Working: cleaning this piece of the photo...")
        self.verdict.setStyleSheet("")
        box = self.boxes[key[0]][1]
        guard = False if key[1] else None

        def work():
            from ..engine.pipeline.pipeline import clean_removed
            return clean_removed(self.image, self.edit, box, guard)
        if self.sync:
            try:
                self._done(key, work())
            except Exception as e:                            # noqa: BLE001
                self._failed(key, str(e), e)
        else:
            run_bg(work, key, self._done, self._failed)

    def _done(self, key, res):
        self.cache[key] = res
        if key == self._key():
            self._show(key, res)

    def _failed(self, key, message, exc=None):
        report_problem("Showing what Clean removed", exc)
        self.verdict.setText(f"This piece could not be shown: {message}")
        self.verdict.setStyleSheet(f"color:{T.RED}; border-color:{T.alpha(T.RED, 0.4)};")

    def _show(self, key, res):
        self.results[key] = res["numbers"]
        for lb, arr in zip(self.views, panes(res)):
            pm = QPixmap.fromImage(to_qimage(arr))
            lb.setPixmap(pm.scaled(384, 384, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.FastTransformation))
        col, text = verdict(res["numbers"])
        self.verdict.setText(text)
        self.verdict.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.4)};")
        log.info("what Clean removed (%s, guard %s): %s", self.boxes[key[0]][0], "off" if key[1] else "as set",
                 res["numbers"])
