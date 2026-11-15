"""Custom Develop widgets (Phase 6): the tone-curve editor, the DxO-style colour wheel, the colour-grading wheels,
the row of colour swatches for HSL, and the settings-group checklist used by copy / paste and Save look.

Pure Qt + NumPy. Every colour on these widgets is drawn in OKLCh with the engine's own conversion, so a hue on a
wheel is exactly the hue the engine shifts or tints toward.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from . import theme as T


# ============================================================================================ colour helpers
def oklab_to_srgb8(L, a, b) -> np.ndarray:
    """OKLab (arrays of any shape) -> sRGB uint8 (..., 3), clipped to the sRGB gamut for display."""
    from ..engine.color.grade import from_oklab
    from ..engine.color.spaces import PROPHOTO, SRGB, conversion_matrix
    lab = np.stack(np.broadcast_arrays(np.asarray(L, np.float32), np.asarray(a, np.float32),
                                       np.asarray(b, np.float32)), -1)
    shp = lab.shape
    pp = from_oklab(lab.reshape(1, -1, 3))
    s = np.clip(pp.reshape(-1, 3) @ conversion_matrix(PROPHOTO, SRGB).T.astype(np.float32), 0.0, 1.0)
    e = np.asarray(SRGB.encode(s), np.float32)
    return (np.clip(e, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).reshape(shp)


def oklch_colour(L: float, C: float, h_deg: float) -> QColor:
    h = math.radians(h_deg)
    r, g, b = (int(v) for v in oklab_to_srgb8(np.float32(L), np.float32(C * math.cos(h)),
                                               np.float32(C * math.sin(h))).reshape(3))
    return QColor(r, g, b)


def srgb8_hue(rgb8: np.ndarray) -> tuple[float, float]:
    """Mean OKLCh hue (degrees) and chroma of a patch of displayed sRGB pixels (the colour-wheel picker)."""
    from ..engine.color.grade import to_oklab
    from ..engine.color.spaces import PROPHOTO, SRGB, conversion_matrix
    s = np.asarray(SRGB.decode(np.asarray(rgb8, np.float32).reshape(-1, 3) / 255.0), np.float32)
    pp = s @ conversion_matrix(SRGB, PROPHOTO).T.astype(np.float32)
    lab = to_oklab(pp.reshape(1, -1, 3)).reshape(-1, 3)
    a, b = float(lab[:, 1].mean()), float(lab[:, 2].mean())
    return math.degrees(math.atan2(b, a)) % 360.0, math.hypot(a, b)


def _rgba_image(rgb8: np.ndarray, alpha: np.ndarray) -> QImage:
    a = np.ascontiguousarray(np.concatenate([rgb8, (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)[..., None]],
                                            -1))
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


def hue_dist(a, b):
    """Signed shortest angle from b to a, in degrees (-180..180)."""
    return (np.asarray(a, float) - b + 180.0) % 360.0 - 180.0


def _range_weight(h, r) -> np.ndarray:
    from ..engine.color.grade import range_weight
    return range_weight(np.asarray(h, np.float32), float(r.get("hue", 0)), float(r.get("width", 40)),
                        float(r.get("falloff", 25)))


# ============================================================================================ tone curve
class CurveEditor(QWidget):
    """Point curve in display (sRGB-encoded) terms, per channel. Click to add a point, drag to move it,
    double-click or right-click a point to remove it. The two end points move up and down only."""
    changed = Signal()
    CHANNELS = ("master", "r", "g", "b")
    COLOURS = {"master": T.TEXT, "r": "#ef4444", "g": "#22c55e", "b": "#3b82f6"}

    def __init__(self):
        super().__init__()
        self.setFixedHeight(230)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.curves: dict = {}
        self.channel = "master"
        self.hist = None
        self._drag = None
        self.setToolTip("Click to add a point, drag to move it, double-click (or right-click) a point to remove it.")

    # ---------------------------------------------------------------- state
    @staticmethod
    def _identity(pts) -> bool:
        return not pts or all(abs(float(x) - float(y)) < 1e-4 for x, y in pts)

    def pts(self, ch=None) -> list:
        return [list(p) for p in (self.curves.get(ch or self.channel) or [[0.0, 0.0], [1.0, 1.0]])]

    def set_curves(self, d):
        self.curves = {k: sorted([[float(x), float(y)] for x, y in v]) for k, v in (d or {}).items()
                       if k in self.CHANNELS and v}
        for k, v in self.curves.items():                      # keep explicit end points
            if v[0][0] > 0:
                v.insert(0, [0.0, 0.0])
            if v[-1][0] < 1:
                v.append([1.0, 1.0])
        self.update()

    def value(self):
        out = {k: [[round(x, 4), round(y, 4)] for x, y in v] for k, v in self.curves.items() if not self._identity(v)}
        return out or None

    def set_channel(self, ch: str):
        self.channel = ch
        self.update()

    def reset_channel(self):
        self.curves.pop(self.channel, None)
        self.update()
        self.changed.emit()

    def set_histogram(self, a8):
        if a8 is None:
            self.hist = None
        else:
            s = a8[::3, ::3].reshape(-1, 3).astype(np.float32)
            v = s[:, {"r": 0, "g": 1, "b": 2}[self.channel]] if self.channel != "master" else s @ np.float32(
                [0.2126, 0.7152, 0.0722])
            h = np.bincount(np.clip(v / 4.0, 0, 63).astype(int), minlength=64).astype(np.float32)
            self.hist = h / max(float(np.percentile(h, 98)), 1.0)
        self.update()

    # ---------------------------------------------------------------- geometry
    def _plot(self) -> QRectF:
        return QRectF(self.rect()).adjusted(10, 10, -10, -10)

    @staticmethod
    def _to_px(x, y, r) -> QPointF:
        return QPointF(r.left() + x * r.width(), r.bottom() - y * r.height())

    @staticmethod
    def _from_px(pt, r):
        return (min(max((pt.x() - r.left()) / r.width(), 0.0), 1.0),
                min(max((r.bottom() - pt.y()) / r.height(), 0.0), 1.0))

    @staticmethod
    def _fn(pts):
        from ..engine.color.light import pchip
        return pchip([p[0] for p in pts], [p[1] for p in pts])

    def _nearest(self, pos, r):
        best, bi = 1e9, None
        for i, (x, y) in enumerate(self.pts()):
            p = self._to_px(x, y, r)
            d = math.hypot(p.x() - pos.x(), p.y() - pos.y())
            if d < best:
                best, bi = d, i
        return bi if best <= 10 else None

    # ---------------------------------------------------------------- painting
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._plot()
        p.setPen(QPen(QColor(T.LINE)))
        p.setBrush(QColor(T.BG2))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 10, 10)
        if self.hist is not None:
            path = QPainterPath(QPointF(r.left(), r.bottom()))
            for i, v in enumerate(self.hist):
                path.lineTo(r.left() + r.width() * (i + 0.5) / 64.0, r.bottom() - min(1.0, float(v)) * r.height() * 0.9)
            path.lineTo(r.right(), r.bottom())
            q = QColor(self.COLOURS[self.channel])
            q.setAlpha(38)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(q)
            p.drawPath(path)
        grid = QColor(T.LINE)
        p.setPen(QPen(grid, 1))
        for k in (1, 2, 3):
            p.drawLine(QPointF(r.left() + r.width() * k / 4, r.top()), QPointF(r.left() + r.width() * k / 4, r.bottom()))
            p.drawLine(QPointF(r.left(), r.top() + r.height() * k / 4), QPointF(r.right(), r.top() + r.height() * k / 4))
        p.setPen(QPen(QColor(T.DIM), 1, Qt.PenStyle.DashLine))
        p.drawLine(self._to_px(0, 0, r), self._to_px(1, 1, r))
        xs = np.linspace(0, 1, 97)
        for ch in self.CHANNELS:
            if ch == self.channel or self._identity(self.curves.get(ch)):
                continue
            q = QColor(self.COLOURS[ch])
            q.setAlpha(90)
            self._draw_curve(p, r, self.pts(ch), xs, QPen(q, 1.2))
        self._draw_curve(p, r, self.pts(), xs, QPen(QColor(self.COLOURS[self.channel]), 2.0))
        for i, (x, y) in enumerate(self.pts()):
            c = self._to_px(x, y, r)
            p.setPen(QPen(QColor(T.BG), 1.5))
            p.setBrush(QColor(T.CYAN) if i == self._drag else QColor(self.COLOURS[self.channel]))
            p.drawEllipse(c, 5, 5)
        p.setPen(QColor(T.DIM))
        p.setFont(QFont("Segoe UI", 8))
        names = {"master": "RGB", "r": "Red", "g": "Green", "b": "Blue"}
        p.drawText(r.adjusted(4, 2, -4, -2), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                   names[self.channel])

    def _draw_curve(self, p, r, pts, xs, pen):
        ys = np.clip(self._fn(pts)(xs), 0, 1)
        path = QPainterPath(self._to_px(xs[0], ys[0], r))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(self._to_px(float(x), float(y), r))
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    # ---------------------------------------------------------------- mouse
    def _remove(self, i):
        pts = self.pts()
        if 0 < i < len(pts) - 1:
            pts.pop(i)
            self.curves[self.channel] = pts
            self.update()
            self.changed.emit()

    def mousePressEvent(self, e):
        r = self._plot()
        pos = e.position()
        i = self._nearest(pos, r)
        if e.button() == Qt.MouseButton.RightButton:
            if i is not None:
                self._remove(i)
            return
        if i is None:
            x, _ = self._from_px(pos, r)
            if not 0.01 < x < 0.99:
                return
            pts = self.pts()
            y = float(np.clip(self._fn(pts)(np.array([x]))[0], 0, 1))
            pts.append([x, y])
            pts.sort()
            i = next(k for k, pt in enumerate(pts) if pt[0] == x)
            self.curves[self.channel] = pts
            self.changed.emit()
        self._drag = i
        self.update()

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        r = self._plot()
        x, y = self._from_px(e.position(), r)
        pts = self.pts()
        i = self._drag
        if i == 0:
            x = 0.0
        elif i == len(pts) - 1:
            x = 1.0
        else:
            x = min(max(x, pts[i - 1][0] + 0.01), pts[i + 1][0] - 0.01)
        pts[i] = [round(x, 4), round(y, 4)]
        self.curves[self.channel] = pts
        self.update()
        self.changed.emit()

    def mouseReleaseEvent(self, e):
        self._drag = None
        self.update()

    def mouseDoubleClickEvent(self, e):
        i = self._nearest(e.position(), self._plot())
        if i is not None:
            self._remove(i)


# ============================================================================================ colour wheel (ranges)
RING_L, RING_C = 0.72, 0.11


class HueRing(QWidget):
    """DxO-style colour wheel. The outer ring is the photo's colours (the chosen range bright, the rest dimmed);
    the inner ring shows what every range does to them. Drag the white handle round the ring to choose the colour
    range, drag the cyan handle to send that range toward another hue. Click a small dot to select that range."""
    changed = Signal()
    picked = Signal(int)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(232)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.ranges: list = []
        self.sel = -1
        self._drag = None
        self._img = None
        self._key = None
        self.setToolTip("Outer ring: the photo's colours. Inner ring: after your changes.\n"
                        "Drag the white handle to pick a colour range; drag the cyan handle to shift it.")

    def set_ranges(self, ranges: list, sel: int):
        self.ranges, self.sel = ranges, sel
        self._img = None
        self.update()

    def _geom(self):
        R = min(self.width(), self.height()) / 2.0 - 12.0
        return QPointF(self.width() / 2.0, self.height() / 2.0), R

    def _render(self, R):
        n = int(2 * R) + 2
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        dx, dy = xx - n / 2.0 + 0.5, n / 2.0 - yy - 0.5
        rad = np.hypot(dx, dy)
        ang = np.degrees(np.arctan2(dy, dx)) % 360.0
        outer = np.clip(rad - 0.72 * R + 0.5, 0, 1) * np.clip(R - rad + 0.5, 0, 1)
        inner = np.clip(rad - 0.42 * R + 0.5, 0, 1) * np.clip(0.66 * R - rad + 0.5, 0, 1)
        w_sel = _range_weight(ang, self.ranges[self.sel]) if 0 <= self.sel < len(self.ranges) else np.ones_like(ang)
        h = np.radians(ang)
        rgb_o = oklab_to_srgb8(RING_L, RING_C * np.cos(h), RING_C * np.sin(h))
        # inner ring: the engine's own arithmetic for a pure colour at this hue (chroma gate included)
        gate = RING_C / (RING_C + 0.02)
        dh, kC, dL = np.zeros_like(ang), np.ones_like(ang), np.zeros_like(ang)
        for r in self.ranges:
            w = _range_weight(ang, r) * gate
            dh += w * float(r.get("shift", 0))
            kC *= 1.0 + w * float(r.get("sat", 0)) / 100.0
            dL += w * 0.15 * float(r.get("lum", 0)) / 100.0
        h2 = np.radians(ang + dh)
        C2 = RING_C * np.maximum(kC, 0)
        rgb_i = oklab_to_srgb8(RING_L + dL, C2 * np.cos(h2), C2 * np.sin(h2))
        rgb = np.where((outer > 0)[..., None], rgb_o, rgb_i)
        alpha = outer * (0.28 + 0.72 * w_sel) + inner
        return _rgba_image(rgb, alpha)

    @staticmethod
    def _at(c, rr, deg) -> QPointF:
        a = math.radians(deg)
        return QPointF(c.x() + rr * math.cos(a), c.y() - rr * math.sin(a))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, R = self._geom()
        key = (round(R), repr(self.ranges), self.sel)
        if self._img is None or key != self._key:
            self._img, self._key = self._render(R), key
        p.drawImage(QPointF(c.x() - self._img.width() / 2.0, c.y() - self._img.height() / 2.0), self._img)
        p.setPen(QColor(T.DIM))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRectF(c.x() - 0.4 * R, c.y() - 12, 0.8 * R, 24), Qt.AlignmentFlag.AlignCenter,
                   "outer: photo\ninner: result" if self.ranges else "no range yet")
        for i, r in enumerate(self.ranges):
            if i == self.sel:
                continue
            p.setPen(QPen(QColor(T.BG), 1))
            p.setBrush(QColor(T.TEXT))
            p.drawEllipse(self._at(c, 0.86 * R, float(r.get("hue", 0))), 3.5, 3.5)
        if 0 <= self.sel < len(self.ranges):
            r = self.ranges[self.sel]
            hue, half, fall = float(r.get("hue", 0)), float(r.get("width", 40)) / 2, float(r.get("falloff", 25))
            p.setPen(QPen(QColor(T.TEXT), 1))
            for d in (-half, half):
                p.drawLine(self._at(c, 0.70 * R, hue + d), self._at(c, R + 4, hue + d))
            p.setPen(QPen(QColor(T.DIM), 1, Qt.PenStyle.DotLine))
            for d in (-half - fall, half + fall):
                p.drawLine(self._at(c, 0.70 * R, hue + d), self._at(c, R + 4, hue + d))
            a, b = self._at(c, 0.86 * R, hue), self._at(c, 0.54 * R, hue + float(r.get("shift", 0)))
            p.setPen(QPen(QColor(T.CYAN), 1.5))
            p.drawLine(a, b)
            p.setPen(QPen(QColor(T.BG), 2))
            p.setBrush(QColor(T.TEXT))
            p.drawEllipse(a, 7, 7)
            p.setPen(QPen(QColor(T.CYAN), 2.5))
            p.setBrush(oklch_colour(RING_L, RING_C, hue + float(r.get("shift", 0))))
            p.drawEllipse(b, 6.5, 6.5)

    def _angle(self, pos) -> float:
        c, _ = self._geom()
        return math.degrees(math.atan2(c.y() - pos.y(), pos.x() - c.x())) % 360.0

    def mousePressEvent(self, e):
        c, R = self._geom()
        pos = e.position()

        def near(pt):
            return math.hypot(pt.x() - pos.x(), pt.y() - pos.y()) < 13
        if 0 <= self.sel < len(self.ranges):
            r = self.ranges[self.sel]
            if near(self._at(c, 0.54 * R, float(r.get("hue", 0)) + float(r.get("shift", 0)))):
                self._drag = "shift"
                return
            if near(self._at(c, 0.86 * R, float(r.get("hue", 0)))):
                self._drag = "hue"
                return
        for i, r in enumerate(self.ranges):
            if i != self.sel and near(self._at(c, 0.86 * R, float(r.get("hue", 0)))):
                self.picked.emit(i)
                return
        rad = math.hypot(pos.x() - c.x(), pos.y() - c.y())
        if 0 <= self.sel < len(self.ranges) and 0.70 * R <= rad <= R + 6:
            self._drag = "hue"
            self.mouseMoveEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag is None or not 0 <= self.sel < len(self.ranges):
            return
        r = self.ranges[self.sel]
        a = self._angle(e.position())
        if self._drag == "hue":
            r["hue"] = round(a, 1)
        else:
            r["shift"] = round(float(np.clip(hue_dist(a, float(r.get("hue", 0))), -90, 90)), 1)
        self._img = None
        self.update()
        self.changed.emit()

    def mouseReleaseEvent(self, e):
        self._drag = None


# ============================================================================================ grading wheel
class GradeWheel(QWidget):
    """One colour-grading wheel: drag the dot away from the centre to tint (direction = hue, distance =
    strength). Double-click to reset."""
    changed = Signal()

    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.hue, self.strength = 0.0, 0.0
        self.setFixedHeight(150)
        self.setMinimumWidth(130)
        self._img, self._n = None, 0
        self._drag = False
        self.setToolTip(f"{title}: drag the dot toward a colour to tint the {title.lower()}. Double-click to reset.")

    def set_value(self, hue: float, strength: float):
        self.hue, self.strength = float(hue), float(strength)
        self.update()

    def _geom(self):
        R = min(self.width(), self.height() - 30) / 2.0 - 4.0
        return QPointF(self.width() / 2.0, 16 + (self.height() - 30) / 2.0), R

    def _disc(self, R):
        n = int(2 * R) + 2
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        dx, dy = xx - n / 2.0 + 0.5, n / 2.0 - yy - 0.5
        rad = np.hypot(dx, dy) / R
        ang = np.arctan2(dy, dx)
        C = 0.10 * np.clip(rad, 0, 1)
        rgb = oklab_to_srgb8(0.62, C * np.cos(ang), C * np.sin(ang))
        return _rgba_image(rgb, np.clip((1.0 - rad) * R + 0.5, 0, 1))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, R = self._geom()
        if self._img is None or self._n != round(R):
            self._img, self._n = self._disc(R), round(R)
        p.setPen(QColor(T.TEXT))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(QRectF(0, 0, self.width(), 14), Qt.AlignmentFlag.AlignCenter, self.title)
        p.drawImage(QPointF(c.x() - self._img.width() / 2.0, c.y() - self._img.height() / 2.0), self._img)
        p.setPen(QPen(QColor(0, 0, 0, 90), 1))
        p.drawLine(QPointF(c.x() - R, c.y()), QPointF(c.x() + R, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - R), QPointF(c.x(), c.y() + R))
        a = math.radians(self.hue)
        pt = QPointF(c.x() + R * self.strength / 100.0 * math.cos(a), c.y() - R * self.strength / 100.0 * math.sin(a))
        p.setPen(QPen(QColor(T.BG), 2))
        p.setBrush(QColor(T.TEXT))
        p.drawEllipse(pt, 6, 6)
        p.setPen(QColor(T.DIM))
        p.setFont(QFont("Segoe UI", 8))
        txt = f"{self.hue:.0f}°  {self.strength:.0f}%" if self.strength >= 0.5 else "no tint"
        p.drawText(QRectF(0, self.height() - 14, self.width(), 14), Qt.AlignmentFlag.AlignCenter, txt)

    def _set_from(self, pos):
        c, R = self._geom()
        dx, dy = pos.x() - c.x(), c.y() - pos.y()
        self.strength = round(min(1.0, math.hypot(dx, dy) / R) * 100.0, 1)
        self.hue = round(math.degrees(math.atan2(dy, dx)) % 360.0, 1)
        self.update()
        self.changed.emit()

    def mousePressEvent(self, e):
        self._drag = True
        self._set_from(e.position())

    def mouseMoveEvent(self, e):
        if self._drag:
            self._set_from(e.position())

    def mouseReleaseEvent(self, e):
        self._drag = False

    def mouseDoubleClickEvent(self, e):
        self.hue, self.strength = 0.0, 0.0
        self.update()
        self.changed.emit()


# ============================================================================================ swatches
class SwatchRow(QWidget):
    """Eight colour buttons (red .. magenta) for choosing which HSL band the sliders below edit.
    A dot marks the bands you have changed."""
    selected = Signal(int)

    def __init__(self, names, hues):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = []
        for i, (n, h) in enumerate(zip(names, hues)):
            b = QPushButton("")
            b.setCheckable(True)
            b.setFixedHeight(24)
            b.setMinimumWidth(24)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.setToolTip(n.capitalize())
            col = oklch_colour(0.70, 0.13, h).name()
            b.setStyleSheet(f"QPushButton{{background:{col}; border:2px solid {T.BG2}; border-radius:6px; padding:0;"
                            f" color:#0b0f19; font-weight:700;}} QPushButton:checked{{border:2px solid {T.TEXT};}}")
            self.group.addButton(b, i)
            lay.addWidget(b)
            self.buttons.append(b)
        self.buttons[0].setChecked(True)
        self.group.idClicked.connect(self.selected.emit)

    def current(self) -> int:
        return max(0, self.group.checkedId())

    def set_marked(self, flags):
        for b, f in zip(self.buttons, flags):
            b.setText("•" if f else "")


# ============================================================================================ group checklist
class GroupDialog(QDialog):
    """Choose which groups of settings to copy, paste or save into a look (and, for a look, its name)."""

    def __init__(self, parent, title: str, intro: str, groups, checked, ask_name=False, name="", about=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(420)
        v = QVBoxLayout(self)
        h = QLabel(title)
        h.setObjectName("h2")
        v.addWidget(h)
        lb = QLabel(intro)
        lb.setObjectName("dim")
        lb.setWordWrap(True)
        v.addWidget(lb)
        self.name = self.about = None
        if ask_name:
            self.name = QLineEdit(name)
            self.name.setPlaceholderText("Name of the look")
            self.about = QLineEdit(about)
            self.about.setPlaceholderText("What it does (optional, shown as a tooltip)")
            v.addWidget(self.name)
            v.addWidget(self.about)
        self.boxes = {}
        for key, text in groups:
            cb = QCheckBox(text)
            cb.setChecked(key in checked)
            v.addWidget(cb)
            self.boxes[key] = cb
        row = QHBoxLayout()
        allb, noneb = QPushButton("All"), QPushButton("None")
        allb.clicked.connect(lambda: [c.setChecked(True) for c in self.boxes.values()])
        noneb.clicked.connect(lambda: [c.setChecked(False) for c in self.boxes.values()])
        row.addWidget(allb)
        row.addWidget(noneb)
        row.addStretch(1)
        cancel, ok = QPushButton("Cancel"), QPushButton("OK")
        ok.setObjectName("primary")
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self._ok)
        row.addWidget(cancel)
        row.addWidget(ok)
        v.addLayout(row)

    def _ok(self):
        if self.name is not None and not self.name.text().strip():
            self.name.setFocus()
            return
        if not self.chosen():
            return
        self.accept()

    def chosen(self) -> list:
        return [k for k, cb in self.boxes.items() if cb.isChecked()]


# ============================================================================================ filmic curve plot
class FilmicPlot(QWidget):
    """The base tone curve drawn the way film curves are: stops from mid grey across, screen brightness up.
    Marks mid grey, where the toe starts, where the highlight roll-off starts and where white is reached."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.params = dict(contrast=1.35, shoulder=1.0, white=3.0, toe=0.0)
        self.setToolTip("The base tone curve: stops below and above mid grey across, screen brightness up.\n"
                        "Toe = how the deepest shadows bend; roll-off = where highlights start to compress.")

    def set_params(self, **kw):
        self.params.update(kw)
        self.update()

    def paintEvent(self, e):
        from ..engine.color.tone import MID_OUT, TOE_START, standard_curve
        from ..engine.color.light import _enc
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(T.LINE)))
        p.setBrush(QColor(T.BG2))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 10, 10)
        r = QRectF(self.rect()).adjusted(10, 10, -10, -18)
        lo, hi = -8.0, 4.0
        s = np.linspace(lo, hi, 300)
        y = _enc(standard_curve(0.11 * np.exp2(s).astype(np.float32), mid_in=0.11, **self.params).astype(np.float64))

        def pt(sv, yv):
            return QPointF(r.left() + (sv - lo) / (hi - lo) * r.width(), r.bottom() - yv * r.height())
        p.setPen(QPen(QColor(T.LINE), 1))
        for sv in range(int(lo), int(hi) + 1, 2):
            p.drawLine(pt(sv, 0), pt(sv, 1))
        path = QPainterPath(pt(s[0], y[0]))
        for sv, yv in zip(s[1:], y[1:]):
            path.lineTo(pt(float(sv), float(yv)))
        p.setPen(QPen(QColor(T.TEXT), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.setFont(QFont("Segoe UI", 7))
        marks = [(0.0, "mid", T.CYAN, False), (TOE_START, "toe", T.VIOLET, False),
                 (self.params["shoulder"], "roll-off", T.AMBER, True), (self.params["white"], "white", T.TEXT, False)]
        for sv, txt, col, above in marks:
            yv = float(_enc(standard_curve(np.float32([0.11 * 2 ** sv]), mid_in=0.11, **self.params)
                            .astype(np.float64))[0]) if sv < self.params["white"] else 1.0
            c = pt(sv, yv)
            p.setPen(QPen(QColor(col), 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(c.x(), r.bottom()), c)
            p.setPen(QPen(QColor(T.BG), 1))
            p.setBrush(QColor(col))
            p.drawEllipse(c, 3.5, 3.5)
            p.setPen(QColor(col))
            if above:                                       # beside its point, so it never sits on "mid"
                p.drawText(QRectF(c.x() - 62, c.y() - 7, 56, 14),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, txt)
            else:
                p.drawText(QRectF(c.x() - 30, r.bottom() + 2, 60, 14), Qt.AlignmentFlag.AlignCenter, txt)
        _ = MID_OUT
