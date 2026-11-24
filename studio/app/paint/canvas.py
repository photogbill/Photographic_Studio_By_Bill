"""The painting canvas (BUILD_PLAN_PAINT P0.2).

Zoom (wheel, around the pointer), pan (middle button, or Space + drag), rotate (4 / 6, 5 resets), mirror view (M),
fit (0), 100 % (1). Pen and mouse both paint; the pen's eraser end erases; Alt + click picks a colour. Every pen event
is used as it arrives (tablet-event compression is turned off in studio/app/main.py), with its own timestamp, and
nothing is predicted ahead of the pen.

The picture is kept as an 8-bit display copy; only the rectangles a stroke changes are composited again.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPointingDevice, QTransform
from PySide6.QtWidgets import QWidget

from ...engine.paint.brush import LayerLocked, Stroke, make_stroke
from ...engine.paint.colour import pick
from ...engine.paint.curves import Curve
from ...engine.paint.smoothing import Sample, Smoother
from ...engine.paint.tiles import Document
from ...engine.paint.undo import UndoStack

log = logging.getLogger("studio.paint")


class PaintCanvas(QWidget):
    sample_seen = Signal(dict)       # every pen / mouse sample (the Tablet check listens)
    colour_picked = Signal(tuple)
    changed = Signal()               # the painting changed (autosave, title)
    view_changed = Signal(str)       # zoom / rotation text
    message = Signal(str)
    popup_requested = Signal(object)   # global QPoint: right click / pen barrel button opens the pop-up palette

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TabletTracking, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(300, 300)
        self.zoom, self.angle, self.mirror = 1.0, 0.0, False
        self.pan = QPointF(0, 0)
        self.brush = None                 # BrushSettings, set by the page
        self.colour = (0.0, 0.0, 0.0)
        self.eraser_mode = False
        self.smoothing, self.smooth_strength = "basic", 0.3
        self.pen_curve = Curve()          # the person's own pressure curve, before any brush
        self.value_view = False           # greyscale view to check values
        self.tool = "brush"               # brush, fill, gradient
        self.second_colour = (1.0, 1.0, 1.0)
        self.fill_opts = {"tolerance": 0.1, "gap": 0, "grow": 1, "source": "picture", "reference": None}
        self.grad_opts = {"kind": "linear", "space": "oklab", "to_clear": False}
        self._grad = None                 # (start, end) in doc space while dragging a gradient
        self.managed = True               # through the monitor's colour profile (color/display.py)
        self._stroke = None
        self._smoother = None
        self._pan_from = None
        self._space = False
        self._cursor = None
        self._t0 = time.perf_counter()
        self.set_document(Document())

    # ---- document
    def set_document(self, doc: Document):
        self.doc = doc
        self.undo = UndoStack(doc)
        self._img = np.zeros((doc.height, doc.width, 4), np.uint8)
        self._qimg = QImage(self._img.data, doc.width, doc.height, doc.width * 4,
                            QImage.Format.Format_RGBA8888_Premultiplied)
        self.refresh()
        self.fit()

    def refresh(self, rect=None):
        """Composite again (the whole picture, or one rectangle) into the display copy."""
        d = self.doc
        if rect is None:
            x0, y0, x1, y1 = 0, 0, d.width, d.height
        else:
            x0, y0 = max(0, rect[0]), max(0, rect[1])
            x1, y1 = min(d.width, rect[2]), min(d.height, rect[3])
            if x1 <= x0 or y1 <= y0:
                return
        px = d.composite(x0, y0, x1 - x0, y1 - y0)
        if self.value_view:
            v = px[..., 0] * 0.2126 + px[..., 1] * 0.7152 + px[..., 2] * 0.0722
            px = np.dstack([v, v, v, px[..., 3]])
        self._img[y0:y1, x0:x1] = self._to_screen(px)
        self.update()

    def _to_screen(self, px):
        """Premultiplied sRGB -> 8-bit for this monitor. With a monitor profile the picture goes the way Develop's
        does (sRGB -> linear ProPhoto -> the monitor's profile, color/display.py), so a painting looks the same
        in Paint, in Develop and in print; without one, sRGB is shown as it is."""
        out = np.empty(px.shape, np.uint8)
        out[..., 3] = (np.clip(px[..., 3], 0, 1) * 255 + 0.5).astype(np.uint8)
        d = None
        if self.managed:
            try:
                from ...engine.color import display
                d = display.current()
                d = d if d.managed else None
            except Exception:                                       # noqa: BLE001  (show unmanaged, never fail)
                log.debug("screen profile not available", exc_info=True)
                d = None
        if d is None:
            out[..., :3] = (np.clip(px[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
            return out
        from ...engine.color import spaces
        from ...engine.paint.colour import srgb_to_linear
        a = np.clip(px[..., 3:4], 0, 1)
        straight = np.clip(px[..., :3] / np.maximum(a, 1e-8), 0, 1)
        pp = spaces.apply_matrix(np.ascontiguousarray(srgb_to_linear(straight)),
                                 spaces.conversion_matrix(spaces.SRGB, spaces.PROPHOTO).astype(np.float32))
        rgb8 = d.to8(pp).astype(np.float32)
        out[..., :3] = (rgb8 * a + 0.5).astype(np.uint8)           # back to premultiplied for the QImage
        return out

    # ---- view
    def transform(self) -> QTransform:
        t = QTransform()
        t.translate(self.width() / 2 + self.pan.x(), self.height() / 2 + self.pan.y())
        t.rotate(self.angle)
        t.scale(-self.zoom if self.mirror else self.zoom, self.zoom)
        t.translate(-self.doc.width / 2, -self.doc.height / 2)
        return t

    def to_doc(self, p: QPointF) -> QPointF:
        inv, _ = self.transform().inverted()
        return inv.map(p)

    def fit(self):
        if self.width() < 10:
            self.zoom = 0.5
        else:
            self.zoom = min(self.width() / self.doc.width, self.height() / self.doc.height) * 0.92
        self.pan = QPointF(0, 0)
        self._view_note()

    def set_zoom(self, z: float, around: QPointF | None = None):
        z = max(0.02, min(32.0, z))
        if around is not None:
            before = self.to_doc(around)
            self.zoom = z
            after = self.transform().map(before)
            self.pan += around - after
        else:
            self.zoom = z
        self._view_note()

    def rotate_by(self, deg: float):
        self.angle = 0.0 if deg is None else (self.angle + deg) % 360
        self._view_note()

    def toggle_mirror(self):
        self.mirror = not self.mirror
        self._view_note()

    def _view_note(self):
        self.update()
        rot = f", rotated {self.angle:.0f}°" if self.angle else ""
        self.view_changed.emit(f"{self.zoom * 100:.0f} %{rot}{', mirrored' if self.mirror else ''}")

    # ---- drawing the widget
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1a1f2b"))
        p.setTransform(self.transform())
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 2)
        r = QRectF(0, 0, self.doc.width, self.doc.height)
        if self.doc.background is None:                        # checkerboard under a see-through canvas
            p.fillRect(r, QColor("#cfcfcf"))
        p.drawImage(r, self._qimg)
        p.setPen(QPen(QColor(0, 0, 0, 90), 0))
        p.drawRect(r)
        if self._grad is not None:                             # the gradient being dragged
            p.setPen(QPen(QColor(255, 255, 255, 220), 0))
            p.drawLine(self._grad[0], self._grad[1])
        if self._cursor is not None and self.brush is not None and self.tool == "brush":  # brush outline
            p.resetTransform()
            d = self.brush.size * self.zoom
            p.setPen(QPen(QColor(255, 255, 255, 160), 1))
            p.drawEllipse(self._cursor, d / 2, d / 2)
            p.setPen(QPen(QColor(0, 0, 0, 160), 1))
            p.drawEllipse(self._cursor, d / 2 + 1, d / 2 + 1)
        p.end()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if e.oldSize().width() < 10:
            self.fit()

    # ---- input
    def _sample(self, pos: QPointF, pressure: float, ts_ms: float | None, tilt_x=0.0, tilt_y=0.0,
                rotation=0.0) -> Sample:
        d = self.to_doc(pos)
        t = (ts_ms / 1000.0) if ts_ms else (time.perf_counter() - self._t0)
        return Sample(d.x(), d.y(), float(self.pen_curve(pressure)), t, tilt_x, tilt_y, rotation)

    def _begin(self, s: Sample, eraser: bool):
        if self.tool == "fill":
            self._fill(s)
            return
        if self.tool == "gradient":
            self._grad = (QPointF(s.x, s.y), QPointF(s.x, s.y))
            return
        if self.brush is None:
            return
        try:
            self.undo.begin("Erase" if (eraser or self.brush.eraser) else "Stroke")
            self._stroke = make_stroke(self.doc.paint_layer(), self.brush, self.colour, self.undo,
                                       eraser=True if eraser else None,
                                  sample=lambda x0, y0, w, h: self.doc.composite(x0, y0, w, h, background=False))
        except LayerLocked as err:
            self.undo.end()
            self.message.emit(str(err))
            self._stroke = None
            return
        self._smoother = Smoother(self.smoothing, self.smooth_strength)
        self._feed(s)

    def _feed(self, s: Sample):
        if self._grad is not None:
            self._grad = (self._grad[0], QPointF(s.x, s.y))
            self.update()
            return
        if self._stroke is None:
            return
        rect = None
        for o in self._smoother.feed(s):
            r = self._stroke.add(o)
            rect = _union(rect, r)
        if rect:
            self.refresh(rect)

    def _end(self):
        if self._grad is not None:
            self._gradient(*self._grad)
            self._grad = None
            return
        if self._stroke is None:
            return
        rect = None
        for o in self._smoother.finish():
            rect = _union(rect, self._stroke.add(o))
        rect = _union(rect, self._stroke.finish())
        self.undo.end()
        self._stroke = None
        if rect:
            self.refresh(rect)
        self.doc.dirty = True
        self.changed.emit()

    def tabletEvent(self, e):
        e.accept()
        pos = e.position()
        eraser = e.pointerType() == QPointingDevice.PointerType.Eraser
        info = {"kind": "tablet", "pressure": e.pressure(), "tilt_x": e.xTilt(), "tilt_y": e.yTilt(),
                "rotation": e.rotation(), "eraser": eraser, "t": e.timestamp() / 1000.0,
                "device": e.device().name() if e.device() else "", "x": pos.x(), "y": pos.y(),
                "buttons": int(e.buttons().value) if hasattr(e.buttons(), "value") else 0}
        self.sample_seen.emit(info)
        self._cursor = pos
        s = self._sample(pos, e.pressure(), e.timestamp(), e.xTilt(), e.yTilt(), e.rotation())
        tp = e.type()
        if tp == e.Type.TabletPress:
            if e.button() == Qt.MouseButton.RightButton:
                self.popup_requested.emit(self.mapToGlobal(pos.toPoint()))
            elif e.modifiers() & Qt.KeyboardModifier.AltModifier:
                self._pick(s)
            elif self._space or e.button() == Qt.MouseButton.MiddleButton:
                self._pan_from = pos
            else:
                self._begin(s, eraser or self.eraser_mode)
        elif tp == e.Type.TabletMove:
            if self._pan_from is not None:
                self.pan += pos - self._pan_from
                self._pan_from = pos
                self.update()
            elif self._stroke is not None:
                self._feed(s)
            else:
                self.update()
        elif tp == e.Type.TabletRelease:
            self._pan_from = None
            self._end()

    def mousePressEvent(self, e):
        self.sample_seen.emit({"kind": "mouse"})
        pos = e.position()
        if e.button() == Qt.MouseButton.RightButton:
            self.popup_requested.emit(self.mapToGlobal(pos.toPoint()))
        elif e.button() == Qt.MouseButton.MiddleButton or (self._space and e.button() == Qt.MouseButton.LeftButton):
            self._pan_from = pos
        elif e.button() == Qt.MouseButton.LeftButton:
            s = self._sample(pos, 1.0, None)
            if e.modifiers() & Qt.KeyboardModifier.AltModifier:
                self._pick(s)
            else:
                self._begin(s, self.eraser_mode)

    def mouseMoveEvent(self, e):
        pos = e.position()
        self._cursor = pos
        if self._pan_from is not None:
            self.pan += pos - self._pan_from
            self._pan_from = pos
            self.update()
        elif self._stroke is not None:
            self._feed(self._sample(pos, 1.0, None))
        else:
            self.update()

    def mouseReleaseEvent(self, e):
        self._pan_from = None
        self._end()

    def leaveEvent(self, e):
        self._cursor = None
        self.update()

    def wheelEvent(self, e):
        f = 1.15 ** (e.angleDelta().y() / 120)
        self.set_zoom(self.zoom * f, e.position())

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = True
        else:
            super().keyPressEvent(e)
            return
        e.accept()

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = False
        super().keyReleaseEvent(e)

    def _fill(self, s: Sample):
        from ...engine.paint import fill
        o = self.fill_opts
        try:
            rect = fill.fill(self.doc, s.x, s.y, self.colour, o["tolerance"], o["gap"], o["grow"], o["source"],
                             o["reference"], undo=self.undo)
        except LayerLocked as err:
            self.message.emit(str(err))
            return
        if rect is None:
            self.message.emit("Click inside the painting to fill.")
            return
        self.refresh(rect)
        self.doc.dirty = True
        self.changed.emit()

    def _gradient(self, a: QPointF, b: QPointF):
        from ...engine.paint import fill
        if math.hypot(b.x() - a.x(), b.y() - a.y()) < 2:
            self.message.emit("Drag across the painting to lay a gradient: from where it starts to where it ends.")
            return
        o = self.grad_opts
        c1 = (*self.second_colour, 0.0 if o["to_clear"] else 1.0)
        try:
            rect = fill.gradient(self.doc, (a.x(), a.y()), (b.x(), b.y()), (*self.colour, 1.0), c1, o["kind"],
                                 o["space"], undo=self.undo)
        except LayerLocked as err:
            self.message.emit(str(err))
            return
        self.refresh(rect)
        self.doc.dirty = True
        self.changed.emit()

    def _pick(self, s: Sample):
        c = pick(self.doc, s.x, s.y, 1)
        if c is not None:
            self.colour_picked.emit(c)

    # ---- for scripts and tests
    def stroke_from(self, samples, eraser=None):
        """Paint a stroke from doc-space Samples as if they came from the pen (tests, scripts)."""
        self._begin(samples[0], self.eraser_mode if eraser is None else eraser)
        for s in samples[1:]:
            self._feed(s)
        self._end()


def _union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def distance(a: QPointF, b: QPointF) -> float:
    return math.hypot(a.x() - b.x(), a.y() - b.y())
