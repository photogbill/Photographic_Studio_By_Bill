"""Paint's small widgets: the colour pickers (HSV wheel with a square, HSV triangle, wide-gamut OKLCH), the swatch
grid and the pressure-curve box. Every picker has ``set_rgb(rgb)``, ``rgb()`` and emits ``colour_changed((r, g, b))``.

Pickers, each with what it is best for and weak at (plan 23.1):
- Wheel and square: quick hue, then saturation across and value down. Weak at: even lightness steps.
- Triangle (as Krita and MyPaint): tints, shades and tones of one hue are straight lines to the white and black
  corners. Weak at: picking an exact saturation.
- Wide gamut (OKLCH): lightness, chroma and hue in even steps to the eye, with the edges of sRGB (what the canvas
  holds) and Display P3 drawn. Weak at: the canvas is sRGB, so colours past the sRGB edge are brought in at the same
  lightness and hue."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QConicalGradient, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

import numpy as np

from ...engine.paint.colour import (hsv_to_rgb, in_gamut, linear_to_srgb, max_chroma, oklab_to_linear_srgb,
                                    oklch_to_oklab, oklch_to_srgb, rgb_to_hsv, srgb_to_oklch, sv_to_triangle,
                                    to_hex, triangle_corners, triangle_to_sv)
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


class TrianglePicker(ColourWheel):
    """Hue on the ring, an HSV triangle inside: the pure hue, white and black at its corners."""

    def _tri(self):
        c, ro, ri, _ = self._geo()
        return triangle_corners(c.x(), c.y(), ri - 3, self.h)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, ro, ri, _ = self._geo()
        g = QConicalGradient(c, 0)
        for i in range(13):
            r, gg, b = hsv_to_rgb(i / 12, 1, 1)
            g.setColorAt(i / 12, QColor.fromRgbF(r, gg, b))
        ring = QPainterPath()
        ring.addEllipse(c, ro, ro)
        inner = QPainterPath()
        inner.addEllipse(c, ri, ri)
        p.fillPath(ring.subtracted(inner), g)
        (hx, hy), (wx, wy), (bx, by) = tri = self._tri()
        poly = QPolygonF([QPointF(hx, hy), QPointF(wx, wy), QPointF(bx, by)])
        path = QPainterPath()
        path.addPolygon(poly)
        path.closeSubpath()
        r, gg, b = hsv_to_rgb(self.h, 1, 1)
        g1 = QLinearGradient(QPointF(wx, wy), QPointF(hx, hy))     # white -> hue along the top edge
        g1.setColorAt(0, QColor(255, 255, 255))
        g1.setColorAt(1, QColor.fromRgbF(r, gg, b))
        p.fillPath(path, g1)
        mx, my = (hx + wx) / 2, (hy + wy) / 2                      # darken towards the black corner
        g2 = QLinearGradient(QPointF(mx, my), QPointF(bx, by))
        g2.setColorAt(0, QColor(0, 0, 0, 0))
        g2.setColorAt(1, QColor(0, 0, 0, 255))
        p.fillPath(path, g2)
        a = math.radians(self.h * 360)
        rm = (ro + ri) / 2
        p.setPen(QPen(QColor("white"), 2))
        p.drawEllipse(QPointF(c.x() + rm * math.cos(a), c.y() - rm * math.sin(a)), 5, 5)
        x, y = sv_to_triangle(self.s, self.v, tri)
        p.setPen(QPen(QColor("white" if self.v < 0.6 else "black"), 2))
        p.drawEllipse(QPointF(x, y), 5, 5)
        p.end()

    def mousePressEvent(self, e):
        c, ro, ri, _ = self._geo()
        pt = e.position()
        d = math.hypot(pt.x() - c.x(), pt.y() - c.y())
        self._drag = "ring" if ri - 4 <= d <= ro + 4 else ("tri" if d < ri - 2 else None)
        self._move(pt)

    def _move(self, pt):
        c, ro, ri, _ = self._geo()
        if self._drag == "ring":
            self.h = (math.degrees(math.atan2(c.y() - pt.y(), pt.x() - c.x())) % 360) / 360
        elif self._drag == "tri":
            self.s, self.v = triangle_to_sv(pt.x(), pt.y(), self._tri())
        else:
            return
        self.update()
        self.colour_changed.emit(self.rgb())


class GamutPicker(QWidget):
    """OKLCH: a hue strip at the bottom, lightness up and chroma across above it. The cyan line is the edge of sRGB
    (what the canvas holds), the violet one the edge of Display P3; past the cyan line the colour shown is brought
    in to sRGB at the same lightness and hue."""
    colour_changed = Signal(tuple)
    C_MAX = 0.37

    def __init__(self):
        super().__init__()
        self.setMinimumSize(220, 240)
        self.L, self.C, self.H = 0.6, 0.1, 250.0
        self._img = None
        self._img_key = None
        self._drag = None

    def set_rgb(self, rgb):
        L, C, H = srgb_to_oklch(np.asarray(rgb[:3], np.float32))
        self.L, self.C = float(L), float(C)
        if self.C > 0.005:
            self.H = float(H)
        self.update()

    def rgb(self):
        return oklch_to_srgb(self.L, self.C, self.H)

    def out_of_srgb(self) -> bool:
        return not bool(in_gamut(self.L, self.C, self.H, "srgb"))

    def _geo(self):
        w, h = self.width(), self.height()
        strip = 18
        return QRectF(4, 4, w - 8, h - strip - 14), QRectF(4, h - strip - 4, w - 8, strip)

    def _plane(self, plane):
        w, h = max(8, int(plane.width())), max(8, int(plane.height()))
        key = (round(self.H, 1), w, h)
        if self._img_key != key:
            L = np.linspace(1, 0, h)[:, None] * np.ones((1, w))
            C = np.linspace(0, self.C_MAX, w)[None, :] * np.ones((h, 1))
            lin = oklab_to_linear_srgb(oklch_to_oklab(L, C, self.H))
            ok = np.all((lin >= -1e-4) & (lin <= 1 + 1e-4), -1)
            Cm = np.minimum(C, max_chroma(L, self.H, "srgb", 12))
            rgb = np.clip(linear_to_srgb(oklab_to_linear_srgb(oklch_to_oklab(L, Cm, self.H))), 0, 1)
            rgb = np.where(ok[..., None], rgb, rgb * 0.55 + 0.12)      # outside sRGB: shown dimmed
            arr = np.ascontiguousarray((rgb * 255 + 0.5).astype(np.uint8))
            self._img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            Ls = np.linspace(1, 0, h)
            self._edges = (max_chroma(Ls, self.H, "srgb", 14), max_chroma(Ls, self.H, "p3", 14))
            self._img_key = key
        return self._img

    def paintEvent(self, _):
        from .. import theme as T
        p = QPainter(self)
        plane, strip = self._geo()
        p.drawImage(plane, self._plane(plane))
        h = int(plane.height())
        Ls = np.linspace(1, 0, max(8, h))
        for edge, col in zip(self._edges, (T.CYAN, T.VIOLET)):
            pts = [QPointF(plane.left() + min(c, self.C_MAX) / self.C_MAX * plane.width(),
                           plane.top() + (1 - L) * plane.height()) for c, L in zip(edge, Ls)]
            p.setPen(QPen(QColor(col), 1.5))
            p.drawPolyline(QPolygonF(pts))
        g = QLinearGradient(strip.topLeft(), strip.topRight())
        for i in range(13):
            g.setColorAt(i / 12, QColor(to_hex(oklch_to_srgb(0.75, 0.12, i * 30))))
        p.fillRect(strip, g)
        p.setPen(QPen(QColor("white"), 2))
        x = strip.left() + self.H / 360 * strip.width()
        p.drawLine(QPointF(x, strip.top() - 2), QPointF(x, strip.bottom() + 2))
        cx = plane.left() + min(self.C, self.C_MAX) / self.C_MAX * plane.width()
        cy = plane.top() + (1 - self.L) * plane.height()
        p.setPen(QPen(QColor("white" if self.L < 0.6 else "black"), 2))
        p.drawEllipse(QPointF(cx, cy), 5, 5)
        p.end()

    def mousePressEvent(self, e):
        plane, strip = self._geo()
        pt = e.position()
        self._drag = "strip" if strip.adjusted(0, -6, 0, 6).contains(pt) else (
            "plane" if plane.adjusted(-6, -6, 6, 6).contains(pt) else None)
        self._move(pt)

    def mouseMoveEvent(self, e):
        self._move(e.position())

    def mouseReleaseEvent(self, e):
        self._drag = None

    def tabletEvent(self, e):
        e.ignore()

    def _move(self, pt):
        plane, strip = self._geo()
        if self._drag == "strip":
            self.H = min(360.0, max(0.0, (pt.x() - strip.left()) / strip.width() * 360)) % 360
        elif self._drag == "plane":
            self.C = min(self.C_MAX, max(0.0, (pt.x() - plane.left()) / plane.width() * self.C_MAX))
            self.L = min(1.0, max(0.0, 1 - (pt.y() - plane.top()) / plane.height()))
        else:
            return
        self.update()
        self.colour_changed.emit(self.rgb())


class SwatchGrid(QWidget):
    """A swatch set as a grid of squares. Click: use the colour. Right-click: remove it from the set."""
    picked = Signal(tuple)
    remove_requested = Signal(int)
    CELL = 22

    def __init__(self):
        super().__init__()
        self.colours: list = []
        self.names: list = []
        self.setMinimumHeight(self.CELL + 4)

    def set_colours(self, colours, names=None):
        self.colours = list(colours)
        self.names = list(names or [])
        self.updateGeometry()
        self._fit()
        self.update()

    def _cols(self):
        return max(1, (self.width() - 4) // self.CELL)

    def _fit(self):
        rows = max(1, (len(self.colours) + self._cols() - 1) // self._cols())
        self.setMinimumHeight(rows * self.CELL + 4)

    def resizeEvent(self, e):
        self._fit()
        super().resizeEvent(e)

    def _at(self, pt):
        c, r = int((pt.x() - 2) // self.CELL), int((pt.y() - 2) // self.CELL)
        i = r * self._cols() + c
        return i if 0 <= c < self._cols() and 0 <= i < len(self.colours) else None

    def paintEvent(self, _):
        p = QPainter(self)
        n = self._cols()
        for i, rgb in enumerate(self.colours):
            x, y = 2 + (i % n) * self.CELL, 2 + (i // n) * self.CELL
            p.fillRect(QRectF(x + 1, y + 1, self.CELL - 2, self.CELL - 2), QColor(to_hex(rgb)))
        p.end()

    def event(self, e):
        if e.type() == e.Type.ToolTip:
            i = self._at(QPointF(e.pos()))
            if i is not None:
                from PySide6.QtWidgets import QToolTip
                name = self.names[i] if i < len(self.names) and self.names[i] else ""
                QToolTip.showText(e.globalPos(), f"{name}  {to_hex(self.colours[i])}".strip(), self)
            return True
        return super().event(e)

    def mousePressEvent(self, e):
        i = self._at(e.position())
        if i is None:
            return
        if e.button() == Qt.MouseButton.RightButton:
            self.remove_requested.emit(i)
        else:
            self.picked.emit(tuple(self.colours[i]))

    def tabletEvent(self, e):
        e.ignore()


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
