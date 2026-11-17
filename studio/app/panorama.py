"""The Panorama page (BUILD_PLAN_PANO_IR.md 2.7): overlapping frames in, one panorama out.

Find layout works out where every frame points (features matched between pairs, then one bundle adjustment) and draws
each frame's outline, coloured by how well it is tied in: cyan solved from many matches, amber few or a pair that does
not quite agree (parallax or something moving), red not placed. Preview stitches at 1/8 size in seconds; Build writes
the full panorama as FIRST_to_LAST_Pano_Studio.tif, which opens in Develop. Frames are left out with the ticks in the
strip. Views: Result, Layout, Seams (which frame each place came from), Ghosts (where frames disagreed: amber).
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QPushButton

from . import theme as T
from .common import hbox, label
from .stackui import MethodPicker, StackPage, honest_card, qimage_rgb8, scene_to_qimage

VIEWS = ("Result", "Layout", "Seams", "Ghosts (where frames disagreed)")
TIE = {"ok": T.CYAN, "warn": T.AMBER, "dropped": T.RED}


def label_colours(n: int) -> np.ndarray:
    from colorsys import hsv_to_rgb
    return np.array([hsv_to_rgb((i * 0.61803) % 1.0, 0.65, 0.95) for i in range(max(n, 1))]) * 255


class PanoramaPage(StackPage):
    SETTINGS_KEY = "panorama"
    MIN_FRAMES = 2

    def __init__(self, settings=None):
        from ..engine.stack import pano as PA
        self.PA = PA
        self.analysis = None
        self.akey = None
        self.last = None
        super().__init__("Panorama", PA.WHAT,
                         "Give it frames: select them in the Library, right-click, Send to Panorama; or drop files or "
                         "a folder here. Untick a frame in the strip to leave it out.",
                         "What to shoot: " + PA.SHOOT, settings)

    # ------------------------------------------------------------------------------------ panel
    def build_panel(self):
        from ..engine.stack import pano as PA
        from ..engine.stack import project as PJ
        g = self.group("Layout", "Where each frame points: found by matching details between overlapping frames.")
        self.find_b = QPushButton("Find layout")
        self.find_b.clicked.connect(self.find_layout)
        g.add(self.find_b)
        self.focal_mm = QDoubleSpinBox()
        self.focal_mm.setRange(0.0, 2000.0)
        self.focal_mm.setDecimals(1)
        self.focal_mm.setSuffix(" mm")
        self.focal_mm.setSpecialValueText("from the file, or estimated")
        self.focal_mm.setToolTip("Adapted and manual lenses record no focal length: the Studio estimates it from the "
                                 "overlaps (amber). Type it here if you know it; the camera's sensor size is used.")
        self.focal_mm.valueChanged.connect(lambda *_: self._layout_changed())
        g.add(hbox(label("Focal length"), self.focal_mm))
        self.focal_chip = g.add(label("", "dim", wrap=True))
        self.level = g.add(self.tick("Level the horizon", True, "Turns the whole panorama so its horizon is level "
                                     "(the up direction most perpendicular to every frame's sideways axis)."))
        self.level.toggled.connect(lambda *_: self._layout_changed())
        g = self.group("Projection")
        self.proj = g.add(MethodPicker(PJ.METHOD_INFO, ["auto"] + list(PJ.PROJECTIONS)))
        self.proj.changed.connect(lambda *_: self._options_changed())
        g = self.group("Seams and blending")
        self.seams = g.add(MethodPicker({k: PA.METHOD_INFO[k] for k in ("smart", "straight")}))
        self.seams.changed.connect(lambda *_: self._options_changed())
        self.flare = g.add(self.tick("Flare guard", True, "Flare and the infrared hot spot only add light and move "
                                     "with the camera: where two frames overlap and one is brighter in a soft patch, "
                                     "the darker one is used, so the flare is not repeated in every frame's place."))
        self.flare.toggled.connect(lambda *_: self._options_changed())
        self.expo = g.add(self.tick("Even out exposure and vignetting", True, "One gain per frame and colour, and "
                                    "the lens's falloff, measured from the overlaps so the panorama looks like one "
                                    "photo."))
        self.expo.toggled.connect(lambda *_: self._options_changed())
        g = self.group("Crop")
        self.crop = g.add(MethodPicker({k: PA.METHOD_INFO[k] for k in ("rectangle", "edge")}))
        self.crop.changed.connect(lambda *_: self._options_changed())
        self.size_pick = g.add(QComboBox())
        for text, k in (("Build at full size", 1.0), ("Build at 50%", 0.5), ("Build at 25%", 0.25)):
            self.size_pick.addItem(text, k)
        g = self.group("View")
        self.view_pick = g.add(QComboBox())
        self.view_pick.addItems(VIEWS)
        self.view_pick.currentIndexChanged.connect(self._show_view)
        self.report = g.add(label("Find layout or Preview to see where the frames go.", "dim", wrap=True))
        self.panel.addWidget(honest_card(PA.HONEST))
        self.panel.addStretch(1)
        self.add_actions()

    def params(self):
        PA = self.PA
        p = PA.PanoParams()
        p.projection = self.proj.key() or "auto"
        p.level = self.level.isChecked()
        p.focal_mm = float(self.focal_mm.value())
        p.seams = self.seams.key() or "smart"
        p.flare_guard = self.flare.isChecked()
        p.exposure = self.expo.isChecked()
        p.crop = self.crop.key() or "rectangle"
        return p

    def _key(self, p):
        return (tuple(self.frame_paths()), round(p.focal_mm, 2), p.level)

    def _layout_changed(self):
        self.analysis = None
        self.last = None
        self._refresh()

    def _options_changed(self):
        self.last = None
        if self.analysis is not None and self.view_pick.currentText() == "Layout":
            self._show_view()

    def frames_changed(self):
        """The strip changed (a frame ticked or unticked, added or removed): an old layout no longer applies. (Status
        text written into the strip changes nothing.)"""
        if self.analysis is not None and self.akey is not None and self.akey[0] == tuple(self.frame_paths()):
            return
        self.analysis = None
        self.last = None

    def _refresh(self):
        super()._refresh()
        if hasattr(self, "find_b"):
            self.find_b.setEnabled(len(self.frame_paths()) >= 2 and not self.jobbar.busy)

    # ------------------------------------------------------------------------------------ work
    def _analysed(self, p, progress, cancel, share=0.6):
        if self.analysis is not None and self.akey == self._key(p):
            return self.analysis
        return self.PA.analyse(self.frame_paths(), p, lambda f, m: progress(share * f, m), cancel)

    def find_layout(self):
        p = self.params()
        if len(self.frame_paths()) < 2:
            return

        def work(progress, cancel):
            return self._analysed(p, progress, cancel, 1.0)

        def done(an):
            self.analysis, self.akey = an, self._key(p)
            self._show_analysis()
            self.view_pick.setCurrentIndex(VIEWS.index("Layout"))
            self._show_view()
            self.jobbar.say(f"Layout found: {sum(an.placed)} of {an.n} frames placed. Preview to see the stitch.",
                            T.CYAN if all(an.placed) else T.AMBER)
            self._refresh()
        if self.jobbar.start(work, done, lambda e, c: self.failed("Find layout", e, c), text="Finding the layout..."):
            self._refresh()

    def preview(self):
        p = self.params()
        if len(self.frame_paths()) < 2:
            return

        def work(progress, cancel):
            an = self._analysed(p, progress, cancel)
            pv = self.PA.preview(an, p, lambda f, m: progress(0.6 + 0.4 * f, m), cancel)
            return an, pv

        def done(r):
            an, pv = r
            self.analysis, self.akey, self.last = an, self._key(p), pv
            self._show_analysis()
            if self.view_pick.currentText() == "Layout":
                self.view_pick.setCurrentIndex(0)
            self._show_view()
            self.jobbar.say("Preview ready (1/8 size). Build writes the full panorama.", T.CYAN)
            self._refresh()
        if self.jobbar.start(work, done, lambda e, c: self.failed("Preview", e, c), text="Previewing..."):
            self._refresh()

    def build_full(self):
        from ..engine.stack.frames import result_name
        p = self.params()
        paths = self.frame_paths()
        if len(paths) < 2 or self.jobbar.busy:
            return
        factor = float(self.size_pick.currentData() or 1.0)

        def work(progress, cancel):
            an = self._analysed(p, progress, cancel, 0.3)
            used = [q for i, q in enumerate(an.paths) if an.R[i] is not None]
            name = result_name([used[0], used[-1]], "Pano")
            out = self.target(name, used[0])
            rep = self.PA.build(an, p, out, lambda f, m: progress(0.3 + 0.7 * f, m), cancel, factor=factor)
            return an, rep

        def done(r):
            an, rep = r
            self.analysis, self.akey = an, self._key(p)
            self._show_analysis()
            self.last_result = rep["path"]
            self.result_written.emit(rep["path"])
            self.jobbar.say(rep["text"][:220] + "  Open in Develop to finish it.", T.GREEN)
            self._refresh()
        if self.jobbar.start(work, done, lambda e, c: self.failed("Build", e, c), keep=True, text="Building..."):
            self._refresh()

    # ------------------------------------------------------------------------------------ showing
    def _show_analysis(self):
        an = self.analysis
        if an is None:
            return
        for i, path in enumerate(an.paths):
            t = an.tie(i)
            mine = [q for q in an.pairs if i in (q["i"], q["j"])]
            tip = "; ".join(f"with {os.path.basename(an.paths[q['j'] if q['i'] == i else q['i']])}: "
                            f"{q['inliers']} matches, {q['rms']:.1f} px" for q in mine) or "no overlap found"
            self.tray.set_status(path, {"ok": "placed", "warn": "placed (check)", "dropped": "not placed"}[t],
                                 t, tip)
        for path, why in an.problems.items():
            self.tray.set_status(path, "left out", "dropped", why)
        src = {"file": "from the file (cyan)", "typed": "typed (cyan)", "estimated": "estimated from the overlaps "
               "(amber): type it if you know it", "guessed": "guessed (amber): type it if you know it"}
        mm = ""
        try:
            from ..engine.stack.pano import SENSOR_LONG_MM
            from .. import engine as E
            md = E.open(an.paths[0]).meta
            model = str(getattr(md, "model", "") or "").upper()
            sensor = next((v for k, v in sorted(SENSOR_LONG_MM.items(), key=lambda kv: -len(kv[0]))
                           if f" {k} " in f" {model} "), 0.0)
            if sensor:
                mm = f" = {an.f / max(an.shape) * sensor:.0f} mm"
        except Exception:                                # noqa: BLE001 - the mm figure is a courtesy
            mm = ""
        self.focal_chip.setText(f"Focal length {an.f:.0f} px{mm}, {src.get(an.focal_source, an.focal_source)}.")
        self.focal_chip.setStyleSheet(f"color:{T.CYAN if an.focal_source in ('file', 'typed') else T.AMBER};")
        rep = self.PA.report(an, None if self.last is None else {"gains": self.last["gains"], "vig": None})
        txt = rep["text"]
        if self.last is not None:
            txt = self.last["report"]["text"]
        self.report.setText(txt)
        self.report.setStyleSheet(f"color:{T.CYAN if rep['colour'] == 'cyan' else T.AMBER};")
        self.set_chips([(f"{sum(an.placed)} placed", T.CYAN if all(an.placed) else T.AMBER)])

    def _layout_image(self) -> QImage:
        lv = self.PA.layout_view(self.analysis, self.params(), 1100)
        w, h = lv["size"]
        pad = 30
        img = QImage(w + 2 * pad, h + 2 * pad, QImage.Format.Format_RGB888)
        img.fill(QColor(T.BG2))
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = QFont("Segoe UI", 10)
        f.setBold(True)
        p.setFont(f)
        for o in lv["outlines"]:
            if not o["pts"]:
                continue
            col = QColor(TIE[o["tie"]])
            poly = QPolygonF([QPointF(x + pad, y + pad) for x, y in o["pts"]])
            fillc = QColor(col)
            fillc.setAlpha(28)
            p.setBrush(fillc)
            p.setPen(QPen(col, 2))
            p.drawPolygon(poly)
            cx = sum(x for x, _ in o["pts"]) / len(o["pts"]) + pad
            cy = sum(y for _, y in o["pts"]) / len(o["pts"]) + pad
            p.drawText(QPointF(cx - 8, cy + 5), str(o["i"] + 1))
        p.setPen(QColor(T.DIM))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(QPointF(pad, img.height() - 10), f"{lv['projection']}: cyan placed, amber check the overlap, "
                   "red not placed")
        p.end()
        return img

    def _show_view(self, *_):
        name = self.view_pick.currentText()
        if self.analysis is None:
            return
        if name == "Layout":
            self.result.set_images(self._layout_image(), None, chip_text="Layout", chip_colour=T.VIOLET)
            return
        if self.last is None:
            self.result.note("Press Preview to see the stitch.")
            return
        pv = self.last
        y0, y1, x0, x1 = pv["crop"]
        if name == "Result":
            self.result.set_images(scene_to_qimage(pv["rgb"]), None, chip_text="Panorama preview (1/8 size)",
                                   chip_colour=T.CYAN)
        elif name == "Seams":
            lab = pv["labels"][y0:y1, x0:x1]
            cols = label_colours(int(lab.max()) + 1 if lab.size else 1)
            base = scene_to_qimage(pv["rgb"])
            from .common import to_qimage
            a8 = np.zeros(lab.shape + (3,), np.uint8)
            ok = lab >= 0
            a8[ok] = cols[lab[ok]].astype(np.uint8)
            q = to_qimage(a8)
            img = QImage(base)
            p = QPainter(img)
            p.setOpacity(0.45)
            p.drawImage(0, 0, q.scaled(img.width(), img.height()))
            p.end()
            self.result.set_images(img, base, chip_text="Seams: which frame each place came from", chip_colour=T.VIOLET)
        else:
            gmap = pv["ghosts"][y0:y1, x0:x1]
            L = pv["rgb"].mean(axis=2)
            g = np.sqrt(np.clip(L / max(float(np.percentile(L, 99.5)), 1e-6), 0, 1)) * 140
            a8 = np.stack([g, g, g], -1)
            a8 = a8 * (1 - gmap[..., None]) + gmap[..., None] * np.array([245, 158, 11])
            self.result.set_images(qimage_rgb8(np.clip(a8, 0, 255).astype(np.uint8)), scene_to_qimage(pv["rgb"]),
                                   chip_text="Ghosts: frames disagreed here (cut, not blended)", chip_colour=T.AMBER)
