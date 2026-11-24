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
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
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
    selection_changed = Signal()
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
        self.light = None                 # impasto: (azimuth, elevation, strength) lighting the paint's relief
        self.sel_mode = "replace"         # how a new selection combines with the old one
        self.exposure = 0.0               # linear (HDR) paintings: viewing exposure in stops
        self._sel = None                  # points of the selection being drawn
        self._sel_img = None              # the shade outside the selection, made when the selection changes
        self._sel_for = None
        self.mask_mode = False            # paint on the selected layer's mask instead of its paint
        self.transform_session = None     # liquify (and other transforms) in progress
        self._wet_timer = QTimer(self, interval=40, timeout=self.dry_tick)
        self._wet_timer.start()
        self.tool = "brush"               # brush, fill, gradient
        self.second_colour = (1.0, 1.0, 1.0)
        self.fill_opts = {"tolerance": 0.1, "gap": 0, "grow": 1, "source": "picture", "reference": None}
        self.grad_opts = {"kind": "linear", "space": "oklab", "to_clear": False}
        self._grad = None                 # (start, end) in doc space while dragging a gradient
        self.managed = True               # through the monitor's colour profile (color/display.py)
        # drawing aids (P6, engine/paint/aids.py)
        self.assistants = []              # aids.Assistant list
        self.assist_on, self.assist_strength = True, 1.0
        self.assist_kind, self.persp_points = "ruler", 2   # what the Assistant tool places next
        self._assist_pts = []
        self.symmetry, self.spokes, self.sym_centre = "none", 6, None
        self.grid, self.show_grid, self.snap_grid = 64, False, False
        self.guides = []                  # ("h", y) or ("v", x) in canvas pixels
        self.vision = "normal"            # colour-blindness preview (view only)
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

    def dry_tick(self):
        """Watercolour: a few steps of flow and drying for every wet layer, then show what changed."""
        doc = getattr(self, "doc", None)
        if doc is None:
            return
        import time as _t
        t0 = _t.perf_counter()
        changed = False
        for lay in doc.layers:
            sim = lay.wet
            if sim is None:
                continue
            if not sim.active:
                lay.wet = None
                continue
            try:
                rect = sim.step(1)
            except Exception as err:                         # noqa: BLE001
                from ..common import report_problem
                report_problem("Paint: watercolour drying", err)
                sim.force_dry()
                lay.wet = None
                continue
            if rect is not None:
                self.refresh(rect)
                changed = True
        if changed:
            doc.dirty = True
        # keep the window responsive: a big wash that takes long to step is stepped less often
        cost = (_t.perf_counter() - t0) * 1000
        self._wet_timer.setInterval(int(min(400, max(40, 2.0 * cost))))

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
        px = d.composite(x0, y0, x1 - x0, y1 - y0, light=self.light)
        if self.value_view:
            v = px[..., 0] * 0.2126 + px[..., 1] * 0.7152 + px[..., 2] * 0.0722
            px = np.dstack([v, v, v, px[..., 3]])
        if self.vision != "normal":                            # colour-blindness preview (on the sRGB values)
            from ...engine.paint.aids import simulate_vision
            a = np.clip(px[..., 3:4], 0, 1)
            sim = simulate_vision(px[..., :3] / np.maximum(a, 1e-8), self.vision)
            px = np.concatenate([sim * a, px[..., 3:4]], -1).astype(np.float32)
        self._img[y0:y1, x0:x1] = self._to_screen(px)
        self.update()

    def _to_screen(self, px):
        """Premultiplied sRGB -> 8-bit for this monitor. With a monitor profile the picture goes the way Develop's
        does (sRGB -> linear ProPhoto -> the monitor's profile, color/display.py), so a painting looks the same
        in Paint, in Develop and in print; without one, sRGB is shown as it is."""
        out = np.empty(px.shape, np.uint8)
        out[..., 3] = (np.clip(px[..., 3], 0, 1) * 255 + 0.5).astype(np.uint8)
        space = getattr(self.doc, "space", "srgb")
        d = None
        try:
            from ...engine.color import display
            proof = bool(display.proof_state().get("path"))
            if self.managed or proof:
                d = display.current()
                d = d if (d.managed or proof) else None
        except Exception:                                       # noqa: BLE001  (show unmanaged, never fail)
            log.debug("screen profile not available", exc_info=True)
            d = None
        a = np.clip(px[..., 3:4], 0, 1)
        straight = px[..., :3] / np.maximum(a, 1e-8)
        if space == "linear" and self.exposure:
            straight = straight * (2.0 ** self.exposure)          # viewing exposure for HDR paintings
        if d is None and space == "srgb":
            out[..., :3] = (np.clip(px[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
            return out
        from ...engine.paint import space as WS
        if d is None:                                           # unmanaged screen: show as sRGB
            rgb = np.clip(WS.convert(straight, space, "srgb"), 0, 1)
            out[..., :3] = (rgb * a * 255 + 0.5).astype(np.uint8)
            return out
        pp = WS.to_prophoto_linear(np.clip(straight, 0, None), space)
        rgb8 = d.to8(np.ascontiguousarray(pp, np.float32)).astype(np.float32)
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
        sel = self.doc.selection
        if sel is not None:                                    # shade outside the selection
            if self._sel_for is not sel:
                a = np.ascontiguousarray(((1 - sel) * 110).astype(np.uint8))
                img = np.zeros(sel.shape + (4,), np.uint8)
                img[..., 3] = a
                img[..., 2] = (a > 0) * 40
                self._sel_img = QImage(img.data, sel.shape[1], sel.shape[0], 4 * sel.shape[1],
                                       QImage.Format.Format_RGBA8888).copy()
                self._sel_for = sel
            p.drawImage(r, self._sel_img)
        if self._sel:                                          # the outline being drawn
            p.setPen(QPen(QColor(34, 211, 238, 230), 0))
            pts = self._sel_outline()
            for a_, b_ in zip(pts, pts[1:]):
                p.drawLine(QPointF(*a_), QPointF(*b_))
        self._draw_aids(p)
        if self.tool == "transform" and self.transform_session is not None:
            self._draw_handles(p)
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

    def _is_clone(self):
        return self.brush is not None and self.brush.mode == "clone"

    def set_clone_source(self, x, y):
        """Clone: Ctrl+click sets where to copy from; the next stroke fixes the offset, and later strokes keep it
        (aligned cloning) until a new Ctrl+click."""
        self.clone_source = (x, y)
        self.clone_offset = None
        self.message.emit("Copying from here. Now paint where the copy should go.")

    SEL_TOOLS = ("rect", "ellipse", "lasso", "polygon", "wand", "colour")

    def _sel_outline(self):
        pts = self._sel or []
        if self.tool in ("rect", "ellipse") and len(pts) >= 2:
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            if self.tool == "rect":
                return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            t = np.linspace(0, 2 * np.pi, 64)
            cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2
            return list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))
        return list(pts)

    # ---- transforms with handles on the canvas
    def start_transform(self, kind: str):
        """Free, perspective, warp (4 x 4), mesh (8 x 8) or cage, on the selection or the whole layer."""
        from ...engine.paint import edit
        if self.transform_session is not None:
            self.transform_session.cancel()
        try:
            lay = self.doc.paint_layer()
        except LayerLocked as err:
            self.message.emit(str(err))
            return False
        ts = self.transform_session = edit.TransformSession(self.doc, lay, self.doc.selection)
        self.tf_kind, self.tool = kind, "transform"
        self.tf = dict(tx=0.0, ty=0.0, sx=1.0, sy=1.0, angle=0.0, shx=0.0)
        self.tf_pts = list(ts.corners()) if kind == "perspective" else []
        self.tf_src = []
        if kind in ("warp", "mesh"):
            n = 4 if kind == "warp" else 8
            gx, gy = np.linspace(0, self.doc.width, n), np.linspace(0, self.doc.height, n)
            self.tf_grid0 = np.stack(np.meshgrid(gx, gy), -1)
            self.tf_grid = self.tf_grid0.copy()
        self._tf_drag = None
        self.update()
        return True

    def _tf_handles(self):
        ts = self.transform_session
        k = self.tf_kind
        if k == "free":
            M = getattr(self, "tf_M", None)
            pts = ts.corners()
            if M is None:
                return pts
            out = []
            for x, y in pts:
                v = M @ np.array([x, y, 1.0])
                out.append((v[0] / v[2], v[1] / v[2]))
            return out
        if k == "perspective":
            return self.tf_pts
        if k in ("warp", "mesh"):
            return [tuple(p) for p in self.tf_grid.reshape(-1, 2)]
        return self.tf_pts

    def _draw_handles(self, p):
        pts = self._tf_handles()
        p.setPen(QPen(QColor(34, 211, 238, 230), 0))
        if self.tf_kind in ("free", "perspective", "cage") and len(pts) >= 2:
            ring = list(pts) + [pts[0]]
            for a_, b_ in zip(ring, ring[1:]):
                p.drawLine(QPointF(*a_), QPointF(*b_))
        if self.tf_kind in ("warp", "mesh"):
            g = self.tf_grid
            for i in range(g.shape[0]):
                for j in range(g.shape[1] - 1):
                    p.drawLine(QPointF(*g[i, j]), QPointF(*g[i, j + 1]))
                    p.drawLine(QPointF(*g[j, i]), QPointF(*g[j + 1, i]))
        r = 5 / max(self.zoom, 1e-3)
        for x, y in pts:
            p.drawRect(QRectF(x - r, y - r, 2 * r, 2 * r))

    def _tf_hit(self, x, y):
        best, bi = 12 / max(self.zoom, 1e-3), None
        for i, (hx, hy) in enumerate(self._tf_handles()):
            d = math.hypot(hx - x, hy - y)
            if d < best:
                best, bi = d, i
        return bi

    def _tf_press(self, s):
        k = self.tf_kind
        i = self._tf_hit(s.x, s.y)
        if k == "cage" and i is None:
            self.tf_src.append((s.x, s.y))                 # a new cage point
            self.tf_pts.append((s.x, s.y))
            self.update()
            return
        if k == "free" and i is None:
            pts = self._tf_handles()
            from ...engine.paint.select import polygon
            inside = polygon(self.doc.width, self.doc.height, pts)[int(min(max(s.y, 0), self.doc.height - 1)),
                                                                   int(min(max(s.x, 0), self.doc.width - 1))] > 0.5
            self._tf_drag = ("move" if inside else "rotate", (s.x, s.y), dict(self.tf))
            return
        self._tf_drag = ("handle", i, (s.x, s.y), dict(self.tf))

    def _tf_move(self, s):
        d = self._tf_drag
        if d is None:
            return
        ts, k = self.transform_session, self.tf_kind
        x0, y0, x1, y1 = ts.box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if d[0] == "move":
            sx_, sy_ = d[1]
            self.tf.update(tx=d[2]["tx"] + s.x - sx_, ty=d[2]["ty"] + s.y - sy_)
        elif d[0] == "rotate":
            c = (cx + self.tf["tx"], cy + self.tf["ty"])
            a0 = math.degrees(math.atan2(d[1][1] - c[1], d[1][0] - c[0]))
            a1 = math.degrees(math.atan2(s.y - c[1], s.x - c[0]))
            self.tf["angle"] = d[2]["angle"] + a1 - a0
        else:
            i = d[1]
            if k == "free":                                # a corner scales about the centre
                c = (cx + self.tf["tx"], cy + self.tf["ty"])
                hx, hy = ts.corners()[i]
                d0 = max(math.hypot(hx - cx, hy - cy), 1e-6)
                f = math.hypot(s.x - c[0], s.y - c[1]) / d0
                self.tf.update(sx=f * (1 if d[3]["sx"] >= 0 else -1), sy=f)
            elif k in ("perspective", "cage"):
                self.tf_pts[i] = (s.x, s.y)
            else:
                n = self.tf_grid.shape[0]
                self.tf_grid[i // n, i % n] = (s.x, s.y)
        self.apply_transform()

    def apply_transform(self):
        ts, k = self.transform_session, self.tf_kind
        if ts is None:
            return
        if k == "free":
            self.tf_M = ts.free(**self.tf)
        elif k == "perspective":
            ts.perspective(self.tf_pts)
        elif k in ("warp", "mesh"):
            ts.warp(self.tf_grid - self.tf_grid0, n=self.tf_grid.shape[0])
        elif k == "cage" and len(self.tf_src) >= 3:
            ts.cage(self.tf_src, self.tf_pts)
        self.refresh()

    def finish_transform(self, keep: bool = True):
        ts = self.transform_session
        if ts is None:
            return
        if keep:
            ts.commit(self.undo, {"free": "Transform", "perspective": "Perspective", "warp": "Warp",
                                  "mesh": "Mesh", "cage": "Cage"}.get(getattr(self, "tf_kind", ""), "Transform"))
            self.doc.dirty = True
        else:
            ts.cancel()
        self.transform_session = None
        self.tf_M = None
        self.tool = "brush"
        self.refresh()
        self.changed.emit()

    def _mask_proxy(self):
        """Painting on a mask: the mask shown as grey paint (white shows, black hides), painted, then read back."""
        from ...engine.paint.tiles import Layer
        lay = self.doc.layer
        if lay.mask is None:
            raise LayerLocked(f"\"{lay.name}\" has no mask. Add one in the Layers tab first.")
        proxy = Layer("mask")
        for k, t in lay.mask.items():
            proxy.tiles[k] = np.concatenate([np.repeat(t[..., None], 3, -1), np.ones_like(t[..., None])], -1)
        self._mask_layer = lay
        self._proxy = proxy
        return proxy

    def _mask_back(self):
        lay, proxy = self._mask_layer, self._proxy
        for k in self._stroke.orig:
            t = proxy.tiles.get(k)
            if t is None:
                continue
            self.undo.touch(lay, ("m",) + tuple(k))
            v = t[..., :3].mean(-1) + (1 - t[..., 3])         # over white: untouched areas stay shown
            lay.mask[k] = np.clip(v, 0, 1).astype(np.float32)
        self._mask_layer = None

    def set_selection(self, m):
        from ...engine.paint import select
        d = self.doc
        d.selection = None if m is None else select.combine(d.selection, m, self.sel_mode)
        if d.selection is not None and float(d.selection.max()) <= 0:
            d.selection = None
        self.selection_changed.emit()
        self.update()

    def _finish_selection(self):
        from ...engine.paint import select
        d, pts, tool = self.doc, self._sel, self.tool
        self._sel = None
        if tool == "rect" and len(pts) >= 2:
            m = select.rect(d.width, d.height, *pts[0], *pts[-1])
        elif tool == "ellipse" and len(pts) >= 2:
            m = select.ellipse(d.width, d.height, *pts[0], *pts[-1])
        elif tool in ("lasso", "polygon") and len(pts) >= 3:
            m = select.polygon(d.width, d.height, pts)
        else:
            self.update()
            return
        self.set_selection(m)

    def _begin(self, s: Sample, eraser: bool):
        if self.tool == "transform" and self.transform_session is not None:
            self._tf_press(s)
            return
        if self.tool == "assistant":
            self._place_assistant_point(*self.snap(s.x, s.y))
            return
        if self.tool == "symmetry_centre":
            self.sym_centre = self.snap(s.x, s.y)
            self.update()
            return
        if self.tool == "subject":
            from ...engine.paint import select
            try:
                self.set_selection(select.subject(self.doc, s.x, s.y, key=id(self.undo.current)))
            except select.SelectProblem as err:
                self.message.emit(str(err))
            except Exception as err:                         # noqa: BLE001
                from ..common import report_problem
                report_problem("Paint: subject select", err)
                self.message.emit("Subject select failed; an error report was saved.")
            return
        if self.tool in self.SEL_TOOLS:
            from ...engine.paint import select
            if self.tool == "wand":
                self.set_selection(select.wand(self.doc, s.x, s.y, getattr(self, "wand_tolerance", 0.1)))
            elif self.tool == "colour":
                c = pick(self.doc, s.x, s.y, 1)
                if c is not None:
                    self.set_selection(select.colour_range(self.doc, c, getattr(self, "colour_fuzz", 0.08)))
            elif self.tool == "polygon":
                s = Sample(*self.snap(s.x, s.y))
                pts = self._sel or []
                if len(pts) >= 3 and math.hypot(s.x - pts[0][0], s.y - pts[0][1]) * self.zoom < 10:
                    self._finish_selection()                  # clicked back on the first point: closed
                else:
                    self._sel = pts + [(s.x, s.y)]
                    self.update()
            else:
                self._sel = [self.snap(s.x, s.y)]
            return
        if self.tool == "liquify":
            from ...engine.paint import edit
            try:
                lay = self.doc.paint_layer()
            except LayerLocked as err:
                self.message.emit(str(err))
                return
            self.transform_session = edit.TransformSession(self.doc, lay, self.doc.selection)
            self._liq_last = (s.x, s.y)
            return
        if self.tool == "fill":
            self._fill(s)
            return
        if self.tool == "gradient":
            self._grad = (QPointF(s.x, s.y), QPointF(s.x, s.y))
            return
        if self.brush is None:
            return
        if self._is_clone() and getattr(self, "clone_source", None) is None:
            self.message.emit("Clone: Ctrl+click where to copy from first, then paint where the copy should go.")
            return
        try:
            self.undo.begin("Erase" if (eraser or self.brush.eraser) else "Stroke")
            clone = None
            if self._is_clone() and getattr(self, "clone_source", None) is not None:
                if getattr(self, "clone_offset", None) is None:
                    self.clone_offset = (self.clone_source[0] - s.x, self.clone_source[1] - s.y)
                clone = self.clone_offset
            target = self._mask_proxy() if self.mask_mode else self.doc.paint_layer()

            def factory():
                return make_stroke(target, self.brush, self.colour, None if self.mask_mode else self.undo,
                                   eraser=True if eraser else None, clone=clone, clip=self.doc.selection,
                                   sample=lambda x0, y0, w, h: self.doc.composite(x0, y0, w, h, background=False))
            self._stroke = self._guided(factory)
        except LayerLocked as err:
            self.undo.end()
            self.message.emit(str(err))
            self._stroke = None
            return
        self._smoother = Smoother(self.smoothing, self.smooth_strength)
        self._feed(s)

    def _feed(self, s: Sample):
        if self.tool == "transform" and self.transform_session is not None:
            self._tf_move(s)
            return
        if self._sel is not None and self.tool in ("rect", "ellipse", "lasso"):
            self._sel.append(self.snap(s.x, s.y) if self.tool != "lasso" else (s.x, s.y))
            self.update()
            return
        if self.tool == "liquify" and self.transform_session is not None:
            lx, ly = self._liq_last
            r = self.transform_session.liquify_push(s.x, s.y, s.x - lx, s.y - ly,
                                                    self.brush.size / 2 if self.brush else 40)
            self._liq_last = (s.x, s.y)
            if r:
                self.refresh(r)
            return
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
        if self.tool == "transform":
            self._tf_drag = None
            return
        if self._sel is not None and self.tool in ("rect", "ellipse", "lasso"):
            self._finish_selection()
            return
        if self.tool == "liquify" and self.transform_session is not None:
            self.transform_session.commit(self.undo, "Liquify")
            self.transform_session = None
            self.doc.dirty = True
            self.refresh()
            self.changed.emit()
            return
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
        if self.mask_mode and getattr(self, "_mask_layer", None) is not None:
            self._mask_back()
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
            elif (e.modifiers() & Qt.KeyboardModifier.ControlModifier) and self._is_clone():
                self.set_clone_source(s.x, s.y)
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
            elif (e.modifiers() & Qt.KeyboardModifier.ControlModifier) and self._is_clone():
                self.set_clone_source(s.x, s.y)
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
            from ...engine.paint.space import to_srgb
            self.colour_picked.emit(to_srgb(c, self.doc.space))      # the pickers work in sRGB

    # ---- for scripts and tests
    # ---- drawing aids (P6)
    POINTS = {"ruler": 2, "parallel": 2, "vanishing": 1, "ellipse": 3, "concentric": 3, "fisheye": 2, "spline": 4}

    def _guided(self, factory):
        """The brush stroke, through the assistants and the symmetry (one stroke when neither is on)."""
        from ...engine.paint import aids
        guide = aids.Guide(self.assistants, self.assist_strength, 8.0 / max(self.zoom, 0.05)) \
            if (self.assist_on and self.assistants) else None
        cx, cy = self.sym_centre or (self.doc.width / 2, self.doc.height / 2)
        maps = aids.symmetry_maps(self.symmetry, self.doc.width, self.doc.height, cx, cy, self.spokes)
        if guide is None and not maps:
            return factory()
        wrap = (self.doc.width, self.doc.height) if self.symmetry == "wrap" else None
        return aids.GuidedStroke(factory, maps, guide, wrap, margin=self.brush.size)

    def snap(self, x, y):
        """A point snapped to a guide line (within 8 screen pixels) and, when on, to the grid."""
        tol = 8.0 / max(self.zoom, 0.05)
        for kind, v in self.guides:
            if kind == "v" and abs(x - v) < tol:
                x = v
            elif kind == "h" and abs(y - v) < tol:
                y = v
        if self.snap_grid and self.grid > 0:
            g = self.grid
            x, y = round(x / g) * g, round(y / g) * g
        return float(x), float(y)

    def _place_assistant_point(self, x, y):
        from ...engine.paint import aids
        self._assist_pts.append((x, y))
        k = self.assist_kind
        need = self.persp_points if k == "perspective" else self.POINTS.get(k, 2)
        if len(self._assist_pts) >= need:
            self.assistants.append(aids.Assistant(k, self._assist_pts))
            self._assist_pts = []
            self.message.emit(f"{aids.ASSISTANTS[k][0]} placed. Paint and your strokes follow it.")
        else:
            self.message.emit(f"{aids.ASSISTANTS[k][0]}: point {len(self._assist_pts)} of {need} placed.")
        self.update()

    def _draw_aids(self, p):
        from PySide6.QtGui import QPainterPath
        w, h = self.doc.width, self.doc.height
        if self.show_grid and self.grid > 0 and self.grid * self.zoom >= 4:
            p.setPen(QPen(QColor(255, 255, 255, 45), 0))
            for gx in range(self.grid, w, self.grid):
                p.drawLine(QPointF(gx, 0), QPointF(gx, h))
            for gy in range(self.grid, h, self.grid):
                p.drawLine(QPointF(0, gy), QPointF(w, gy))
        p.setPen(QPen(QColor(34, 211, 238, 200), 0))
        for kind, v in self.guides:
            if kind == "v":
                p.drawLine(QPointF(v, -1e4), QPointF(v, h + 1e4))
            else:
                p.drawLine(QPointF(-1e4, v), QPointF(w + 1e4, v))
        if self.symmetry not in ("none", "wrap"):
            cx, cy = self.sym_centre or (w / 2, h / 2)
            p.setPen(QPen(QColor(232, 121, 249, 170), 0, Qt.PenStyle.DashLine))
            if self.symmetry in ("x", "xy"):
                p.drawLine(QPointF(cx, 0), QPointF(cx, h))
            if self.symmetry in ("y", "xy"):
                p.drawLine(QPointF(0, cy), QPointF(w, cy))
            if self.symmetry in ("radial", "mandala"):
                r = math.hypot(w, h)
                for k in range(max(2, self.spokes)):
                    a = 2 * math.pi * k / max(2, self.spokes)
                    p.drawLine(QPointF(cx, cy), QPointF(cx + r * math.cos(a), cy + r * math.sin(a)))
        col = QColor(251, 191, 36, 200) if self.assist_on else QColor(150, 150, 150, 120)
        p.setPen(QPen(col, 0))
        big = 4 * math.hypot(w, h)
        for a in self.assistants:
            P = a.points
            if a.kind in ("ruler", "parallel", "fisheye"):
                (ax, ay), (bx, by) = P[:2]
                n = math.hypot(bx - ax, by - ay) or 1
                ux, uy = (bx - ax) / n, (by - ay) / n
                p.drawLine(QPointF(ax - ux * big, ay - uy * big), QPointF(bx + ux * big, by + uy * big))
            elif a.kind in ("vanishing", "perspective"):
                for vx, vy in P:
                    for k in range(16):
                        t = 2 * math.pi * k / 16
                        p.drawLine(QPointF(vx, vy), QPointF(vx + big * math.cos(t), vy + big * math.sin(t)))
            elif a.kind in ("ellipse", "concentric"):
                (cx, cy), (ax, ay), (bx, by) = P[:3]
                rx = math.hypot(ax - cx, ay - cy)
                ang = math.atan2(ay - cy, ax - cx)
                ry = abs(-(bx - cx) * math.sin(ang) + (by - cy) * math.cos(ang)) or rx
                p.save()
                p.translate(cx, cy)
                p.rotate(math.degrees(ang))
                for f in ((1.0,) if a.kind == "ellipse" else (0.5, 1.0, 1.5)):
                    p.drawEllipse(QPointF(0, 0), rx * f, ry * f)
                p.restore()
            elif a.kind == "spline":
                path = QPainterPath(QPointF(*P[0]))
                path.cubicTo(QPointF(*P[1]), QPointF(*P[2]), QPointF(*P[3]))
                p.drawPath(path)
            for x, y in P:
                p.drawEllipse(QPointF(x, y), 4 / self.zoom, 4 / self.zoom)
        for x, y in self._assist_pts:
            p.drawEllipse(QPointF(x, y), 4 / self.zoom, 4 / self.zoom)

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
