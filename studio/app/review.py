"""Review (plan 24.7 A, 6.2's "Cull mode"): one photo large with the filmstrip below, rate with the keyboard, check
focus at 100%, compare two side by side with zoom and position locked together, without leaving the Library.

What is shown is the photographer's choice (rule 23.1), with plain text beside it:
  * Camera preview (amber): the JPEG the camera embedded in the raw. Instant, but it carries the camera's own
    sharpening and noise reduction, so it is not the raw.
  * From the raw (cyan): a quick render by the Studio's own engine with the photo's saved edit (or the defaults).
    Shows real noise and detail; slower, so the camera preview is shown until it is ready.
Where a camera's embedded preview is small (under about 2,000 px on the long side, e.g. many Sony bodies), 100% is
shown from the raw by itself, and the view says so.
"""
from __future__ import annotations

import os
from collections import OrderedDict

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFrame, QHBoxLayout, QListView, QPushButton,
                               QVBoxLayout, QWidget)

from . import theme as T
from .common import STARS, display8, label, log, run_bg, to_qimage

SMALL_PREVIEW = 2000                 # px: below this on the long side, 100% comes from the raw
FIT_SIDE = 2560                      # px: the reduced preview decoded for the fitted view
MODES = {
    "preview": ("Camera preview", T.AMBER,
                "Best for: speed, it is instant. Weak at: it carries the camera's own sharpening and noise "
                "reduction, so it is not the raw."),
    "raw": ("From the raw", T.CYAN,
            "Best for: judging noise and real detail (the Studio's own render, with the photo's edit or the "
            "defaults). Weak at: slower; the camera preview shows until it is ready."),
}


class PhotoView(QWidget):
    """One photo: fitted to the view, or at 100% around a centre point (drag to move). A rendered 100% window
    from the raw can be laid over the preview."""
    zoom_changed = Signal(bool, float, float)          # zoomed, centre u, v (to lock a compare partner)
    clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 150)
        self.setMouseTracking(True)
        self.img: QImage | None = None                 # what is shown (may be reduced)
        self.full_w = self.full_h = 0                  # the photo's size at 100%
        self.zoom = False
        self.cu = self.cv = 0.5
        self.tile: QImage | None = None                # a 100% window rendered from the raw
        self.tile_rect = None                          # (x, y, w, h) in 100% pixels
        self.chip, self.chip_colour, self.note = "", T.DIM, ""
        self.title = ""
        self.busy = False
        self._press = None
        self._moved = False
        self.click_zooms = True                        # a click toggles 100% (off where a click picks a place)

    def set_image(self, img: QImage | None, full_w: int = 0, full_h: int = 0) -> None:
        self.img = img
        if img is not None:
            self.full_w = full_w or img.width()
            self.full_h = full_h or img.height()
        self.update()

    def set_tile(self, img: QImage | None, rect=None) -> None:
        self.tile, self.tile_rect = img, rect
        self.update()

    # ------------------------------------------------------------------------------------ geometry
    def fit_rect(self) -> QRectF:
        r = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        if not self.full_w or not self.full_h:
            return r
        s = min(r.width() / self.full_w, r.height() / self.full_h)
        w, h = self.full_w * s, self.full_h * s
        return QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h)

    def window_100(self):
        """The 100% window shown: (x, y, w, h) in full-size pixels."""
        w, h = min(self.width(), self.full_w), min(self.height(), self.full_h)
        x = self.cu * self.full_w - w / 2
        y = self.cv * self.full_h - h / 2
        x = max(0.0, min(self.full_w - w, x))
        y = max(0.0, min(self.full_h - h, y))
        return x, y, w, h

    def set_zoom(self, on: bool, u: float | None = None, v: float | None = None, emit: bool = True) -> None:
        self.zoom = bool(on)
        if u is not None:
            self.cu, self.cv = min(1.0, max(0.0, u)), min(1.0, max(0.0, v))
        self.update()
        if emit:
            self.zoom_changed.emit(self.zoom, self.cu, self.cv)

    def uv_at(self, pos: QPointF):
        if not self.full_w:
            return 0.5, 0.5
        if self.zoom:
            x, y, w, h = self.window_100()
            ox = (self.width() - w) / 2
            oy = (self.height() - h) / 2
            return (x + pos.x() - ox) / self.full_w, (y + pos.y() - oy) / self.full_h
        r = self.fit_rect()
        return (pos.x() - r.left()) / r.width(), (pos.y() - r.top()) / r.height()

    # ------------------------------------------------------------------------------------ painting
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(T.BG))
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self.img is not None and self.full_w:
            if not self.zoom:
                p.drawImage(self.fit_rect(), self.img, QRectF(self.img.rect()))
            else:
                x, y, w, h = self.window_100()
                sx = self.img.width() / self.full_w
                src = QRectF(x * sx, y * sx, w * sx, h * sx)
                dst = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
                p.drawImage(dst, self.img, src)
                if self.tile is not None and self.tile_rect is not None:
                    tx, ty, tw, th = self.tile_rect
                    p.drawImage(QRectF(dst.left() + tx - x, dst.top() + ty - y, tw, th), self.tile,
                                QRectF(self.tile.rect()))
        elif self.title:
            p.setPen(QColor(T.DIM))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Loading...")
        # the chip: what is shown, coloured by how it was made
        if self.chip:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            f = QFont("Segoe UI", 9)
            f.setBold(True)
            p.setFont(f)
            text = self.chip + ("  ·  100%" if self.zoom else "") + ("  ·  rendering..." if self.busy else "")
            w = p.fontMetrics().horizontalAdvance(text) + 22
            r = QRectF(14, 12, w, 24)
            p.setPen(QPen(QColor(self.chip_colour), 1))
            p.setBrush(QColor(0, 0, 0, 175))
            p.drawRoundedRect(r, 12, 12)
            p.setPen(QColor(self.chip_colour))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
        if self.note:
            p.setFont(QFont("Segoe UI", 9))
            fm = p.fontMetrics()
            w = min(self.width() - 28, fm.horizontalAdvance(self.note) + 20)
            r = QRectF(14, self.height() - 34, w, 24)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 175))
            p.drawRoundedRect(r, 8, 8)
            p.setPen(QColor(T.AMBER))
            p.drawText(r.adjusted(10, 0, -10, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       fm.elidedText(self.note, Qt.TextElideMode.ElideRight, int(r.width() - 20)))
        if self.title:
            p.setFont(QFont("Segoe UI", 9))
            p.setPen(QColor(T.DIM))
            p.drawText(QRectF(0, 12, self.width() - 14, 24), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       self.title)
        p.end()

    # ------------------------------------------------------------------------------------ mouse
    def mousePressEvent(self, e):
        self._press = e.position()
        self._moved = False
        self._start_c = (self.cu, self.cv)
        self.clicked.emit()

    def mouseMoveEvent(self, e):
        if self._press is None or not (e.buttons() & Qt.MouseButton.LeftButton):
            return
        d = e.position() - self._press
        if abs(d.x()) + abs(d.y()) > 4:
            self._moved = True
        if self.zoom and self._moved and self.full_w:
            self.cu = min(1.0, max(0.0, self._start_c[0] - d.x() / self.full_w))
            self.cv = min(1.0, max(0.0, self._start_c[1] - d.y() / self.full_h))
            self.update()
            self.zoom_changed.emit(True, self.cu, self.cv)

    def mouseReleaseEvent(self, e):
        if self._press is not None and not self._moved and self.click_zooms:
            u, v = self.uv_at(e.position())
            self.set_zoom(not self.zoom, u, v)
        self._press = None

    def wheelEvent(self, e):
        if self.zoom:
            d = e.angleDelta()
            self.cu = min(1.0, max(0.0, self.cu - d.x() / max(1, self.full_w)))
            self.cv = min(1.0, max(0.0, self.cv - d.y() / max(1, self.full_h)))
            self.update()
            self.zoom_changed.emit(True, self.cu, self.cv)


class ReviewView(QWidget):
    """The Library's Review page. ``start(path)`` opens it on a photo."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.path = ""
        self.compare = False
        self.mode = str(page.settings.get("review_mode", "preview"))
        if self.mode not in MODES:
            self.mode = "preview"
        self.cache: OrderedDict = OrderedDict()        # path -> {"fit", "full", "info", "raw", "rawsize"}
        self.images: OrderedDict = OrderedDict()       # path -> opened engine Image (From the raw at 100%)
        self.wanted: set = set()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QFrame()
        bar.setObjectName("topbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(18, 6, 18, 6)
        back = QPushButton("◀ Grid")
        back.setToolTip("Back to the grid (Esc or Space)")
        back.clicked.connect(self.close)
        self.info = label("", None)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.mode_c = QComboBox()
        for k, (name, colour, text) in MODES.items():
            self.mode_c.addItem(name, k)
            self.mode_c.setItemData(self.mode_c.count() - 1, text, Qt.ItemDataRole.ToolTipRole)
        self.mode_c.setCurrentIndex(max(0, self.mode_c.findData(self.mode)))
        self.mode_c.currentIndexChanged.connect(self._mode_picked)
        self.advance = QCheckBox("Go to the next photo after rating")
        self.advance.setChecked(bool(page.settings.get("review_advance", True)))
        self.advance.toggled.connect(lambda on: page.settings.set("review_advance", bool(on)))
        self.cmp_b = QPushButton("Compare (C)")
        self.cmp_b.setCheckable(True)
        self.cmp_b.toggled.connect(self.set_compare)
        dev = QPushButton("Develop (Enter)")
        dev.clicked.connect(lambda: self.path and self.page.open_photo.emit(self.path))
        from PySide6.QtWidgets import QSizePolicy
        self.info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        bl.addWidget(back)
        bl.addSpacing(10)
        bl.addWidget(self.info, 1)
        bl.addWidget(self.cmp_b)
        bl.addWidget(dev)
        v.addWidget(bar)
        row2 = QHBoxLayout()
        row2.setContentsMargins(18, 4, 18, 4)
        self.mode_text = label(MODES[self.mode][2], "dim", wrap=True)
        self.mode_text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row2.addWidget(label("Show", "dim"))
        row2.addWidget(self.mode_c)
        row2.addWidget(self.mode_text, 1)
        row2.addWidget(self.advance)
        v.addLayout(row2)
        views = QHBoxLayout()
        views.setContentsMargins(0, 0, 0, 0)
        views.setSpacing(4)
        self.view_a, self.view_b = PhotoView(), PhotoView()
        self.view_b.hide()
        self.view_a.zoom_changed.connect(lambda z, u, vv: self._zoom_moved(self.view_a, z, u, vv))
        self.view_b.zoom_changed.connect(lambda z, u, vv: self._zoom_moved(self.view_b, z, u, vv))
        views.addWidget(self.view_a, 1)
        views.addWidget(self.view_b, 1)
        holder = QWidget()
        holder.setLayout(views)
        v.addWidget(holder, 1)
        from .library import PhotoDelegate
        self.strip = QListView()
        self.strip.setModel(page.model)
        self.strip.setItemDelegate(PhotoDelegate(page, strip=True))
        self.strip.setViewMode(QListView.ViewMode.IconMode)
        self.strip.setFlow(QListView.Flow.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setUniformItemSizes(True)
        self.strip.setMovement(QListView.Movement.Static)
        self.strip.setGridSize(QSize(page.strip_size + 16, int(page.strip_size * 0.72) + 16))
        self.strip.setFixedHeight(int(page.strip_size * 0.72) + 40)
        self.strip.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.strip.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.strip.clicked.connect(lambda ix: self.show_path(ix.data(Qt.ItemDataRole.UserRole)))
        self.strip.horizontalScrollBar().valueChanged.connect(lambda *_: page._schedule_thumbs())
        self.strip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        v.addWidget(self.strip)
        self._tile_timer = QTimer(self, singleShot=True, interval=200, timeout=self._render_tiles)

    # ------------------------------------------------------------------------------------ opening and moving
    def start(self, path: str) -> None:
        self.show_path(path)
        self.setFocus()

    def close(self):
        self.page.close_review(self.path)

    def rows(self) -> list:
        return self.page.model.rows

    def current_row(self) -> int:
        return self.page.model.row_of.get(self.path, -1)

    def show_path(self, path: str) -> None:
        if not path:
            return
        self.path = path
        r = self.current_row()
        if r >= 0:
            ix = self.page.model.index(r)
            self.strip.setCurrentIndex(ix)
            self.strip.scrollTo(ix, QAbstractItemView.ScrollHint.PositionAtCenter)
            self.page.select_path(path)
        self._update_info()
        self._show(self.view_a, path)
        if self.compare:
            self._show(self.view_b, self.partner())
        self._prefetch()
        self.page._schedule_thumbs()

    def partner(self) -> str:
        """The second photo in Compare: the other selected one if two are selected, else the next."""
        sel = [p for p in self.page.selected_files() if p != self.path]
        if len(self.page.selected_files()) == 2 and sel:
            return sel[0]
        r = self.current_row()
        rows = self.rows()
        return rows[r + 1]["path"] if 0 <= r < len(rows) - 1 else (rows[r - 1]["path"] if r > 0 else "")

    def move(self, step: int) -> None:
        r = self.current_row()
        rows = self.rows()
        if not rows:
            return
        n = min(len(rows) - 1, max(0, r + step)) if r >= 0 else 0
        if rows[n]["path"] != self.path:
            self.show_path(rows[n]["path"])

    def wanted_rows(self) -> list:
        """Rows whose thumbnails the filmstrip needs (around the current photo)."""
        r = self.current_row()
        if r < 0:
            return []
        vp = self.strip.viewport().width()
        n = max(4, vp // max(1, self.strip.gridSize().width()) + 2)
        lo = max(0, r - n // 2 - 1)
        return list(range(lo, min(len(self.rows()), lo + n + 2)))

    def thumb_ready(self, path: str) -> None:
        pass                                           # the strip repaints through the model

    def meta_ready(self) -> None:
        if self.path:
            self._update_info()

    def _update_info(self):
        from .library import entry_details
        e = self.page.model.by_path.get(self.path)
        if e is None:
            self.info.setText("")
            return
        stars = STARS[e.get("rating") or 0]
        flag = {1: f"<span style='color:{T.GREEN}'>✔ pick</span>",
                -1: f"<span style='color:{T.RED}'>✖ reject</span>"}.get(e.get("pick") or 0, "")
        r = self.current_row()
        self.info.setText(f"<b>{entry_details(e)}</b> &nbsp; <span style='color:{T.AMBER}'>{stars}</span> {flag}"
                          f" &nbsp; <span style='color:{T.DIM}'>{r + 1} of {len(self.rows())}</span>")

    # ------------------------------------------------------------------------------------ images
    def _entry(self, path: str) -> dict:
        c = self.cache.get(path)
        if c is None:
            c = {"fit": None, "full": None, "info": None, "raw": None, "rawsize": None}
            self.cache[path] = c
        self.cache.move_to_end(path)
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        fulls = [p for p, x in self.cache.items() if x.get("full") is not None]
        for p in fulls[:-2]:                           # full-size previews are big: keep two at most
            self.cache[p]["full"] = None
        return c

    def _show(self, view: PhotoView, path: str) -> None:
        view.title = os.path.basename(path) if view is self.view_b else ""
        if not path:
            view.set_image(None)
            view.chip = ""
            return
        c = self._entry(path)
        view.set_tile(None)
        view.note = ""
        img = c["full"] if view.zoom and c["full"] is not None else c["fit"]
        if self.mode == "raw" and c["raw"] is not None and not view.zoom:
            img = c["raw"]
        if img is not None:
            fw, fh = self._full_size(path, view)
            view.set_image(img, fw, fh)
        else:
            pm = self.page.pix.get(path)
            if pm is not None:
                view.set_image(pm.toImage(), *self._full_size(path, view, guess=(pm.width(), pm.height())))
            else:
                view.set_image(None)
        self._set_chip(view, path)
        self._load(path, full=view.zoom)
        if self.mode == "raw":
            self._load_raw(path)
        if view.zoom:
            self._tile_timer.start()

    def _full_size(self, path, view, guess=None):
        c = self.cache.get(path) or {}
        if self.mode == "raw" or self._small(path):
            if c.get("rawsize"):
                return c["rawsize"]
        info = c.get("info")
        if info:
            return info["width"], info["height"]
        if guess:
            return guess
        return 0, 0

    def _small(self, path: str) -> bool:
        info = (self.cache.get(path) or {}).get("info")
        return bool(info) and max(info["width"], info["height"]) < SMALL_PREVIEW

    def _uses_raw(self, view: PhotoView, path: str) -> bool:
        return self.mode == "raw" or (view.zoom and self._small(path))

    def _set_chip(self, view: PhotoView, path: str) -> None:
        c = self.cache.get(path) or {}
        raw = self._uses_raw(view, path)
        ready = (c.get("raw") is not None and not view.zoom) or (view.zoom and view.tile is not None)
        if raw and ready:
            view.chip, view.chip_colour = MODES["raw"][0], MODES["raw"][1]
        else:
            view.chip, view.chip_colour = MODES["preview"][0], MODES["preview"][1]
        view.busy = raw and not ready
        if view.zoom and self._small(path) and self.mode != "raw":
            info = c.get("info") or {}
            view.note = (f"This camera's preview is small ({max(info.get('width', 0), info.get('height', 0))} px): "
                         f"100% is shown from the raw.")
        view.update()

    def _load(self, path: str, full: bool = False) -> None:
        c = self._entry(path)
        key = (path, "full" if full else "fit")
        if key in self.wanted or (c["full"] is not None if full else c["fit"] is not None):
            return
        self.wanted.add(key)

        def work():
            from ..engine.io.thumbs import embedded_info, embedded_preview
            info = embedded_info(path)
            a = embedded_preview(path, None if full else FIT_SIDE)
            return info, to_qimage(a)

        def done(k, res):
            self.wanted.discard(k)
            info, img = res
            c = self._entry(path)
            c["info"] = info
            c["full" if full else "fit"] = img
            if full and c["fit"] is None:
                c["fit"] = img
            for view in (self.view_a, self.view_b):
                if self._view_path(view) == path:
                    self._refresh_view(view, path)

        def failed(k, err):
            self.wanted.discard(k)
            log.warning("review: no preview for %s: %s", path, err)
        run_bg(work, key, done, failed)

    def _load_raw(self, path: str) -> None:
        c = self._entry(path)
        key = (path, "raw")
        if c["raw"] is not None or key in self.wanted:
            return
        self.wanted.add(key)

        def work():
            from .. import engine as E
            img = E.open(path)
            edit = E.load_edit(path)
            ow, oh = E.output_size(img, edit)
            s = min(1.0, FIT_SIDE / max(ow, oh))
            a = display8(E.render(img, edit, scale=s))
            return img, (ow, oh), to_qimage(a)

        def done(k, res):
            self.wanted.discard(k)
            img, size, q = res
            c = self._entry(path)
            c["raw"], c["rawsize"] = q, size
            self._keep_image(path, img)
            for view in (self.view_a, self.view_b):
                if self._view_path(view) == path:
                    self._refresh_view(view, path)

        def failed(k, err):
            self.wanted.discard(k)
            log.warning("review: could not render %s from the raw: %s", path, err)
            for view in (self.view_a, self.view_b):
                if self._view_path(view) == path:
                    view.busy = False
                    view.note = f"Could not render from the raw: {err}"
                    view.update()
        run_bg(work, key, done, failed)

    def _keep_image(self, path, img):
        self.images[path] = img
        self.images.move_to_end(path)
        while len(self.images) > 2:
            _, old = self.images.popitem(last=False)
            try:
                old.drop_cache()
            except Exception:                          # noqa: BLE001
                pass

    def _view_path(self, view: PhotoView) -> str:
        if view is self.view_a:
            return self.path
        return self.partner() if self.compare else ""

    def _refresh_view(self, view: PhotoView, path: str) -> None:
        c = self.cache.get(path) or {}
        if view.zoom:
            img = c.get("full") or c.get("fit")
        else:
            img = c.get("raw") if (self.mode == "raw" and c.get("raw") is not None) else c.get("fit")
        if img is not None:
            view.set_image(img, *self._full_size(path, view, guess=(img.width(), img.height())))
        if view.zoom and not c.get("full"):
            self._load(path, full=True)
        if view.zoom and self._uses_raw(view, path):
            if c.get("rawsize") is None:
                self._load_raw(path)
            else:
                self._tile_timer.start()
        self._set_chip(view, path)

    def _render_tiles(self):
        for view in (self.view_a, self.view_b):
            path = self._view_path(view)
            if not path or not view.zoom or not view.isVisible() or not self._uses_raw(view, path):
                continue
            img = self.images.get(path)
            c = self.cache.get(path) or {}
            if img is None or not c.get("rawsize"):
                self._load_raw(path)
                continue
            x, y, w, h = view.window_100()
            u, v = (x + w / 2) / view.full_w, (y + h / 2) / view.full_h
            key = (path, "tile", round(u, 4), round(v, 4), int(w), int(h))

            def work(img=img, u=u, v=v, w=int(w), h=int(h)):
                from .. import engine as E
                return to_qimage(display8(E.render_view(img, E.load_edit(img.path), u, v, w, h)))

            def done(k, q, view=view, rect=(x, y, w, h), path=path):
                if self._view_path(view) != path or not view.zoom:
                    return
                if abs(view.window_100()[0] - rect[0]) > 1 or abs(view.window_100()[1] - rect[1]) > 1:
                    self._tile_timer.start()           # it moved while rendering: render where it is now
                    return
                view.set_tile(q, rect)
                self._set_chip(view, path)
            view.busy = True
            run_bg(work, key, done, lambda k, err: log.warning("review: 100%% render failed: %s", err))

    def _prefetch(self):
        r = self.current_row()
        rows = self.rows()
        for d in (1, 2, -1):
            i = r + d
            if 0 <= i < len(rows):
                p = rows[i]["path"]
                self._load(p, full=False)
                if self.view_a.zoom and d == 1:
                    self._load(p, full=True)
                if self.mode == "raw" and d == 1:
                    self._load_raw(p)

    # ------------------------------------------------------------------------------------ zoom, compare, mode
    def _zoom_moved(self, view, zoomed, u, v):
        other = self.view_b if view is self.view_a else self.view_a
        if self.compare:
            other.set_zoom(zoomed, u, v, emit=False)
            other.set_tile(None)
            self._refresh_view(other, self._view_path(other))
        view.set_tile(None)
        self._refresh_view(view, self._view_path(view))
        self._tile_timer.start()

    def toggle_zoom(self):
        pos = self.view_a.mapFromGlobal(self.cursor().pos())
        if self.view_a.rect().contains(pos) and not self.view_a.zoom:
            u, v = self.view_a.uv_at(QPointF(pos))
        else:
            u, v = self.view_a.cu, self.view_a.cv
        self.view_a.set_zoom(not self.view_a.zoom, u, v)

    def set_compare(self, on: bool):
        self.compare = bool(on)
        if self.cmp_b.isChecked() != self.compare:
            self.cmp_b.setChecked(self.compare)
        self.view_b.setVisible(self.compare)
        if self.compare:
            self.view_b.set_zoom(self.view_a.zoom, self.view_a.cu, self.view_a.cv, emit=False)
            self._show(self.view_b, self.partner())
        self.setFocus()

    def _mode_picked(self, _):
        self.mode = self.mode_c.currentData()
        self.page.settings.set("review_mode", self.mode)
        self.mode_text.setText(MODES[self.mode][2])
        self._show(self.view_a, self.path)
        if self.compare:
            self._show(self.view_b, self.partner())
        self.setFocus()

    # ------------------------------------------------------------------------------------ keys
    def keyPressEvent(self, e):
        k = e.key()
        t = e.text().lower()
        if k in (Qt.Key.Key_Escape, Qt.Key.Key_Space):
            self.close()
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self.move(1)
        elif k in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self.move(-1)
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.path:
                self.page.open_photo.emit(self.path)
        elif t == "z":
            self.toggle_zoom()
        elif t == "c":
            self.set_compare(not self.compare)
        elif t and t in "012345pxu":
            if self.path:
                self.page.set_flag([self.path], t)
                self._update_info()
                if self.advance.isChecked():
                    self.move(1)
        else:
            super().keyPressEvent(e)

