"""Paint's small widgets: the HSV colour wheel (hue ring + saturation / value square) and the pressure-curve box."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QConicalGradient, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ...engine.paint.colour import hsv_to_rgb, rgb_to_hsv
from ...engine.paint.curves import Curve
from .. import theme as T


class ColourWheel(QWidget):
    """Hue on the ring, saturation across and value down the square. Emits colour_changed((r, g, b))."""
    colour_changed = Signal(tuple)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(220, 220)
        self.h, self.s, self.v = 0.6, 0.7, 0.4
        self._drag = None

    def set_rgb(self, rgb):
        h, s, v = rgb_to_hsv(*rgb[:3])
        if s > 0.001 and v > 0.001:
            self.h = h
        if v > 0.001:
            self.s = s
        self.v = v
        self.update()

    def rgb(self):
        return hsv_to_rgb(self.h, self.s, self.v)

    def _geo(self):
        side = min(self.width(), self.height())
        c = QPointF(self.width() / 2, self.height() / 2)
        ro = side / 2 - 2
        ri = ro * 0.82
        sq = ri * 1.3
        return c, ro, ri, QRectF(c.x() - sq / 2, c.y() - sq / 2, sq, sq)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, ro, ri, sq = self._geo()
        g = QConicalGradient(c, 0)
        for i in range(13):
            r, gg, b = hsv_to_rgb(i / 12, 1, 1)
            g.setColorAt(i / 12, QColor.fromRgbF(r, gg, b))
        ring = QPainterPath()
        ring.addEllipse(c, ro, ro)
        inner = QPainterPath()
        inner.addEllipse(c, ri, ri)
        p.fillPath(ring.subtracted(inner), g)
        r, gg, b = hsv_to_rgb(self.h, 1, 1)
        lg = QLinearGradient(sq.topLeft(), sq.topRight())
        lg.setColorAt(0, QColor(255, 255, 255))
        lg.setColorAt(1, QColor.fromRgbF(r, gg, b))
        p.fillRect(sq, lg)
        dg = QLinearGradient(sq.topLeft(), sq.bottomLeft())
        dg.setColorAt(0, QColor(0, 0, 0, 0))
        dg.setColorAt(1, QColor(0, 0, 0, 255))
        p.fillRect(sq, dg)
        a = math.radians(self.h * 360)
        rm = (ro + ri) / 2
        hp = QPointF(c.x() + rm * math.cos(a), c.y() - rm * math.sin(a))
        p.setPen(QPen(QColor("white"), 2))
        p.drawEllipse(hp, 5, 5)
        sp = QPointF(sq.left() + self.s * sq.width(), sq.top() + (1 - self.v) * sq.height())
        p.setPen(QPen(QColor("white" if self.v < 0.6 else "black"), 2))
        p.drawEllipse(sp, 5, 5)
        p.end()

    def mousePressEvent(self, e):
        c, ro, ri, sq = self._geo()
        pt = e.position()
        d = math.hypot(pt.x() - c.x(), pt.y() - c.y())
        self._drag = "ring" if ri - 4 <= d <= ro + 4 else ("square" if sq.adjusted(-6, -6, 6, 6).contains(pt) else None)
        self._move(pt)

    def mouseMoveEvent(self, e):
        self._move(e.position())

    def mouseReleaseEvent(self, e):
        self._drag = None

    def tabletEvent(self, e):
        e.ignore()                     # let the pen act as a mouse here

    def _move(self, pt):
        c, ro, ri, sq = self._geo()
        if self._drag == "ring":
            self.h = (math.degrees(math.atan2(c.y() - pt.y(), pt.x() - c.x())) % 360) / 360
        elif self._drag == "square":
            self.s = min(1, max(0, (pt.x() - sq.left()) / sq.width()))
            self.v = min(1, max(0, 1 - (pt.y() - sq.top()) / sq.height()))
        else:
            return
        self.update()
        self.colour_changed.emit(self.rgb())


class CurveBox(QWidget):
    """The pressure curve: drag the points, double-click to add one, right-click a point to remove it."""
    curve_changed = Signal(object)

    def __init__(self, curve: Curve | None = None):
        super().__init__()
        self.setMinimumSize(180, 150)
        self.curve = curve or Curve()
        self._drag = None
        self.live = None              # the current pen pressure, shown as a marker

    def set_curve(self, c: Curve):
        self.curve = c
        self.update()

    def _r(self):
        return QRectF(8, 8, self.width() - 16, self.height() - 16)

    def _to(self, x, y):
        r = self._r()
        return QPointF(r.left() + x * r.width(), r.bottom() - y * r.height())

    def _from(self, pt):
        r = self._r()
        return min(1, max(0, (pt.x() - r.left()) / r.width())), min(1, max(0, (r.bottom() - pt.y()) / r.height()))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._r()
        p.fillRect(self.rect(), QColor(T.PANEL))
        p.setPen(QPen(QColor(T.LINE), 1))
        for i in range(1, 4):
            p.drawLine(self._to(i / 4, 0), self._to(i / 4, 1))
            p.drawLine(self._to(0, i / 4), self._to(1, i / 4))
        p.drawRect(r)
        path = QPainterPath()
        for i in range(101):
            x = i / 100
            pt = self._to(x, self.curve(x))
            (path.moveTo if i == 0 else path.lineTo)(pt)
        p.setPen(QPen(QColor(T.CYAN), 2))
        p.drawPath(path)
        p.setBrush(QColor(T.TEXT))
        for x, y in self.curve.points:
            p.drawEllipse(self._to(x, y), 4, 4)
        if self.live is not None:
            p.setPen(QPen(QColor(T.AMBER), 1))
            p.drawLine(self._to(self.live, 0), self._to(self.live, 1))
        p.setPen(QColor(T.DIM))
        p.drawText(r.adjusted(4, 2, 0, 0), Qt.AlignmentFlag.AlignTop, "output")
        p.drawText(r.adjusted(0, 0, -4, -2), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight, "pen pressure")
        p.end()

    def _nearest(self, pt):
        best, bd = None, 12.0
        for i, (x, y) in enumerate(self.curve.points):
            q = self._to(x, y)
            d = math.hypot(q.x() - pt.x(), q.y() - pt.y())
            if d < bd:
                best, bd = i, d
        return best

    def mousePressEvent(self, e):
        i = self._nearest(e.position())
        if e.button() == Qt.MouseButton.RightButton:
            if i is not None and 0 < i < len(self.curve.points) - 1:
                pts = list(self.curve.points)
                del pts[i]
                self._set(pts)
            return
        self._drag = i

    def mouseDoubleClickEvent(self, e):
        x, y = self._from(e.position())
        self._set(list(self.curve.points) + [(x, y)])

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        x, y = self._from(e.position())
        pts = list(self.curve.points)
        i = self._drag
        if i == 0:
            x = 0.0
        elif i == len(pts) - 1:
            x = 1.0
        else:
            x = min(max(x, pts[i - 1][0] + 0.01), pts[i + 1][0] - 0.01)
        pts[i] = (x, y)
        self._set(pts, keep_order=True)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def _set(self, pts, keep_order=False):
        self.curve = Curve(pts)
        self.update()
        self.curve_changed.emit(self.curve)
