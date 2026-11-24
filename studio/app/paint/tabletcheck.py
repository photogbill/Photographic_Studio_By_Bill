"""The Tablet check panel (BUILD_PLAN_PAINT 0.2, P0.1): live pressure, tilt, speed, eraser end and keys, and a
plain-words diagnosis when pressure is not arriving. Draw in its test box or on the canvas; both feed it."""
from __future__ import annotations

import math
import sys
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPointingDevice
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ...engine.paint.tablet import PenStats, diagnose, tilt_angle
from .. import theme as T

COL = {"ok": T.GREEN, "warn": T.AMBER, "bad": T.RED}


class _Pad(QWidget):
    """A small test area that draws what the pen does: line width follows pressure."""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.setAttribute(Qt.WidgetAttribute.WA_TabletTracking, True)
        self.setMinimumHeight(170)
        self.trail: list = []

    def tabletEvent(self, e):
        e.accept()
        eraser = e.pointerType() == QPointingDevice.PointerType.Eraser
        info = {"kind": "tablet", "pressure": e.pressure(), "tilt_x": e.xTilt(), "tilt_y": e.yTilt(),
                "rotation": e.rotation(), "eraser": eraser, "t": e.timestamp() / 1000.0,
                "device": e.device().name() if e.device() else "", "x": e.position().x(), "y": e.position().y()}
        self.owner.seen(info)
        if e.type() == e.Type.TabletPress:
            self.trail.append([])
        if e.type() != e.Type.TabletRelease and e.pressure() > 0 and self.trail:
            self.trail[-1].append((e.position(), e.pressure()))
            self.trail = self.trail[-20:]
            self.update()

    def mousePressEvent(self, e):
        self.owner.seen({"kind": "mouse"})

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#f4f1ea"))
        p.setPen(QColor("#9a9a9a"))
        if not self.trail:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Draw here with the pen: press lightly, then hard.\nTurn the pen over to test an eraser end.")
        for stroke in self.trail:
            for (a, pa), (b, pb) in zip(stroke, stroke[1:]):
                p.setPen(QPen(QColor(30, 40, 70), 1 + 14 * (pa + pb) / 2, Qt.PenStyle.SolidLine,
                              Qt.PenCapStyle.RoundCap))
                p.drawLine(a, b)
        p.end()


class _Graph(QWidget):
    """Pressure over the last few seconds."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(70)
        self.vals: list = []

    def add(self, t, p):
        self.vals = (self.vals + [(t, p)])[-600:]
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        p.fillRect(r, QColor(T.PANEL))
        p.setPen(QColor(T.LINE))
        p.drawRect(r.adjusted(0, 0, -1, -1))
        if len(self.vals) > 1:
            t1 = self.vals[-1][0]
            path = QPainterPath()
            for i, (t, v) in enumerate(self.vals):
                x = r.width() - (t1 - t) / 5.0 * r.width()
                y = r.height() - 4 - v * (r.height() - 8)
                (path.moveTo if i == 0 else path.lineTo)(QPointF(x, y))
            p.setPen(QPen(QColor(T.CYAN), 1.5))
            p.drawPath(path)
        p.setPen(QColor(T.DIM))
        p.drawText(r.adjusted(6, 4, 0, 0), Qt.AlignmentFlag.AlignTop, "pressure, last 5 s")
        p.end()


class _Bar(QWidget):
    def __init__(self, colour):
        super().__init__()
        self.v, self.colour = 0.0, colour
        self.setFixedHeight(14)

    def set(self, v):
        self.v = max(0.0, min(1.0, v))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        r = self.rect()
        p.fillRect(r, QColor(T.PANEL))
        p.fillRect(QRectF(0, 0, r.width() * self.v, r.height()), QColor(self.colour))
        p.end()


class TabletCheck(QWidget):
    def __init__(self):
        super().__init__()
        self.stats = PenStats()
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        intro = QLabel("What your pen sends to Paint, live. Draw in the box (or on the canvas).")
        intro.setWordWrap(True)
        v.addWidget(intro)
        self.pad = _Pad(self)
        v.addWidget(self.pad)
        self.p_lbl = QLabel("Pressure: -")
        v.addWidget(self.p_lbl)
        self.p_bar = _Bar(T.CYAN)
        v.addWidget(self.p_bar)
        self.graph = _Graph()
        v.addWidget(self.graph)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self.info)
        v.addWidget(QLabel("Diagnosis"))
        self.diag = QLabel("")
        self.diag.setWordWrap(True)
        self.diag.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self.diag)
        row = QHBoxLayout()
        reset = QPushButton("Start again")
        reset.clicked.connect(self.reset)
        row.addWidget(reset)
        row.addStretch(1)
        v.addLayout(row)
        v.addStretch(1)
        self._last = None
        self._timer = QTimer(self, interval=400, timeout=self._update_diag)
        self._timer.start()
        self._update_diag()

    def reset(self):
        self.stats = PenStats()
        self.pad.trail.clear()
        self.pad.update()
        self.graph.vals.clear()
        self._update_diag()

    def seen(self, info: dict):
        if info.get("kind") == "mouse":
            if not self.stats.tablet_events:
                self.stats.add_mouse()
            return
        st = self.stats
        st.add_tablet(info["pressure"], info["tilt_x"], info["tilt_y"], info["t"], info["eraser"],
                      info.get("device", ""), info.get("rotation", 0.0))
        speed = 0.0
        if self._last and info["t"] > self._last["t"]:
            speed = math.hypot(info["x"] - self._last["x"], info["y"] - self._last["y"]) / (info["t"] - self._last["t"])
        self._last = info
        self.p_lbl.setText(f"Pressure: {info['pressure']:.3f}")
        self.p_bar.set(info["pressure"])
        self.graph.add(info["t"], info["pressure"])
        lean, direction = tilt_angle(info["tilt_x"], info["tilt_y"])
        self.info.setText(
            f"Tilt: x {info['tilt_x']:.0f}°, y {info['tilt_y']:.0f}° (lean {lean:.0f}°)<br>"
            f"Speed: {speed:.0f} px/s &nbsp; Rotation: {info.get('rotation', 0):.0f}°<br>"
            f"End: <b>{'eraser' if info['eraser'] else 'pen tip'}</b><br>"
            f"Device: {st.device or 'not named by the system'}")

    def key(self, name: str):
        self.stats.add_key(name)

    def _update_diag(self):
        plat = "linux" if sys.platform.startswith("linux") else ("windows" if sys.platform == "win32" else sys.platform)
        lines = diagnose(self.stats, plat)
        self.diag.setText("<br>".join(f"<span style='color:{COL[lv]}'>●</span> {txt}" for lv, txt in lines))


def now() -> float:
    return time.perf_counter()
