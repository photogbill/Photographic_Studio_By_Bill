"""Composite: pick the subject in another photo by clicking it (SAM 2.1 or the classic method), for
Cognitive Adjustments > Add a subject from another photo."""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from .. import log as slog
from . import theme as T
from .common import run_bg

log = slog.get("app.subject_picker")


class _Pic(QLabel):
    clicked = Signal(float, float, bool)            # u, v (0..1 of the picture), alt

    def mousePressEvent(self, ev):                   # noqa: N802 (Qt name)
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return
        ox = (self.width() - pm.width()) / 2.0
        oy = (self.height() - pm.height()) / 2.0
        u = (ev.position().x() - ox) / pm.width()
        v = (ev.position().y() - oy) / pm.height()
        if 0 <= u <= 1 and 0 <= v <= 1:
            alt = bool(ev.modifiers() & Qt.KeyboardModifier.AltModifier) or ev.button() == Qt.MouseButton.RightButton
            self.clicked.emit(u, v, alt)


class SubjectPicker(QDialog):
    """Shows the source photo; each click outlines the subject (Alt- or right-click: not this). ``shape`` is the
    result (an object-mask shape in the source's sensor frame) once accepted."""

    def __init__(self, parent, path: str, model: str | None):
        super().__init__(parent)
        from .. import engine as E
        self.setWindowTitle(f"Pick the subject in {os.path.basename(path)}")
        self.path, self.model = path, model
        self.fg, self.bg, self.shape = [], [], None
        self.src = E.open(path)
        try:
            self.src_edit = E.load_edit(path)
        except Exception:
            self.src_edit = E.Edit()
        self.src_edit.composite = None
        lay = QVBoxLayout(self)
        how = "SAM 2.1 " + ("Tiny" if model == "tiny" else "Large") if model else "the classic method"
        self.info = QLabel(f"Click the subject; click again to add a part; Alt-click (or right-click) a part that "
                           f"does not belong. Outlined with {how}. The first click can take a few seconds.")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.pic = _Pic()
        self.pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pic.setMinimumSize(640, 420)
        self.pic.clicked.connect(self._click)
        lay.addWidget(self.pic, 1)
        row = QHBoxLayout()
        self.undo_b = QPushButton("Start again")
        self.undo_b.clicked.connect(self._clear)
        self.ok_b = QPushButton("Use this subject")
        self.ok_b.setEnabled(False)
        self.ok_b.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(self.undo_b)
        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(self.ok_b)
        lay.addLayout(row)
        sc = min(1.0, 900.0 / max(E.output_size(self.src, self.src_edit)))
        self.base = self._display8(E.render(self.src, self.src_edit, scale=sc))
        self.scale = sc
        self._show(None)
        self.resize(960, 720)

    @staticmethod
    def _display8(lin):
        from ..engine.color import spaces as sp
        a = sp.srgb_encode(np.clip(sp.apply_matrix(np.ascontiguousarray(lin, np.float32),
                                                   sp.conversion_matrix(sp.PROPHOTO, sp.SRGB)), 0, 1))
        return np.ascontiguousarray((a * 255 + 0.5).astype(np.uint8))

    def _show(self, alpha):
        img = self.base.astype(np.float32)
        if alpha is not None:
            a = np.clip(alpha, 0, 1)[..., None]
            img = img * (1 - 0.45 * a) + np.array([255, 60, 60], np.float32) * 0.45 * a
        img = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
        h, w = img.shape[:2]
        q = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.pic.setPixmap(QPixmap.fromImage(q).scaled(self.pic.width(), self.pic.height(),
                                                       Qt.AspectRatioMode.KeepAspectRatio,
                                                       Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, ev):                       # noqa: N802 (Qt name)
        super().resizeEvent(ev)
        self._show(getattr(self, "_alpha", None))

    def _clear(self):
        self.fg, self.bg, self.shape, self._alpha = [], [], None, None
        self.ok_b.setEnabled(False)
        self._show(None)

    def _click(self, u, v, alt):
        from .. import engine as E
        xs, ys = E.view_to_s(self.src, self.src_edit, np.array([u]), np.array([v]))
        pt = [float(xs[0]), float(ys[0])]
        (self.bg if alt else self.fg).append(pt)
        if not self.fg:
            self.info.setText("Click the subject first; Alt-click then marks what does not belong.")
            return
        self.info.setText("Finding the subject...")
        self.info.setStyleSheet(f"color:{T.AMBER};")
        src, se, fg, bg, model, sc = self.src, self.src_edit, list(self.fg), list(self.bg), self.model, self.scale

        def work():
            shape = E.object_mask(src, se, fg, bg, model=model)
            e = type(se).from_dict(se.to_dict())
            e.masks = [{"name": "subject", "enabled": True, "adjust": {}, "shapes": [dict(shape, op="add")]}]
            return shape, E.render_mask(src, e, 0, sc)

        def done(_, res):
            self.shape, a = res
            if a.shape[:2] != self.base.shape[:2]:
                import cv2
                a = cv2.resize(a, (self.base.shape[1], self.base.shape[0]))
            self._alpha = a
            self._show(a)
            self.ok_b.setEnabled(True)
            self.info.setStyleSheet("")
            self.info.setText(f"Subject outlined ({100 * float((a > 0.5).mean()):.0f}% of the photo). Click to add a "
                              "part, Alt-click to take one away, then Use this subject.")

        def fail(_, err):
            self.info.setStyleSheet(f"color:{T.RED};")
            self.info.setText(f"Could not outline it: {err}")
            log.error("subject picker: %s", err)
        run_bg(work, None, done, fail)
