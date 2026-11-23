"""The Panorama page (BUILD_PLAN_PANO_IR.md 2.7): overlapping frames in, one panorama out.

Find layout works out where every frame points (features matched between pairs, then one bundle adjustment) and draws
each frame's outline, coloured by how well it is tied in: cyan solved from many matches, amber few or a pair that does
not quite agree (parallax or something moving), red not placed. Preview stitches at 1/8 size in seconds; Build writes
the full panorama as FIRST_to_LAST_Pano_Studio.tif, which opens in Develop. Frames are left out with the ticks in the
strip. Views: Result, Layout, Seams (which frame each place came from), Ghosts (where frames disagreed: amber).

Kind (BUILD_PLAN_MOSAIC.md 5.1): **Panorama** turns the camera on one spot; **Flat mosaic** walks over a surface
(a road, a floor, a wall) with the camera pointing down, hand-held, at any height (mosaic.py, mosaicbuild.py,
mosaicio.py); **Auto** tells them apart from how neighbouring frames join and says which it chose. A mosaic's Build
writes game surfaces (colour, height, normal, confidence, agreement, misfit, coverage) and its Layout is the coverage
seen from above, each view coloured by how well it is tied in.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QLineEdit, QMessageBox, QPushButton

from . import theme as T
from .common import hbox, label
from .stackui import MethodPicker, StackPage, honest_card, qimage_rgb8, scene_to_qimage

VIEWS = ("Result", "Layout", "Seams", "Ghosts (where frames disagreed)")
M_VIEWS = VIEWS + ("Height", "Misfit (where views still disagree in position)")
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
        self.m = None               # a mosaic's views and solve: {"key", "views", "sol", "problems", "entries"}
        self.mlast = None           # a mosaic preview
        self.auto = None            # Auto's answer: {"key", "kind", "text"}
        super().__init__("Panorama", PA.WHAT,
                         "Give it frames: select them in the Library, right-click, Send to Panorama; or drop files or "
                         "a folder here. Untick a frame in the strip to leave it out.",
                         "What to shoot: " + PA.SHOOT, settings)

    # ------------------------------------------------------------------------------------ panel
    def build_panel(self):
        from ..engine.stack import mosaic as MO
        from ..engine.stack import mosaicbuild as MB
        from ..engine.stack import mosaicio as MI
        from ..engine.stack import pano as PA
        from ..engine.stack import project as PJ
        gk = self.group("Kind", "A panorama turns the camera on one spot; a flat mosaic walks over a surface.")
        self.kind = gk.add(MethodPicker(MI.KIND_INFO))
        self.kind.changed.connect(lambda *_: self._kind_changed())
        self.kind_chip = gk.add(label("", "dim", wrap=True))
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
        self.level_group = g
        g = self.group("Projection")
        self.proj_group = g
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
        self._build_mosaic_panel(MO, MB, MI)
        g = self.group("View")
        self.view_pick = g.add(QComboBox())
        self.view_pick.addItems(VIEWS)
        self.view_pick.currentIndexChanged.connect(self._show_view)
        self.report = g.add(label("Find layout or Preview to see where the frames go.", "dim", wrap=True))
        self.pano_honest = honest_card(PA.HONEST)
        self.panel.addWidget(self.pano_honest)
        self.mosaic_honest = honest_card(MI.HONEST, "Honest limits (flat mosaic)")
        self.panel.addWidget(self.mosaic_honest)
        self.panel.addStretch(1)
        self.add_actions()
        self._kind_changed()

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
        if self.m is not None and self.m["key"][0] != tuple(self.frame_paths()):
            self.m = None
            self.mlast = None
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

    def _pano_find_layout(self):
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

    def _pano_preview(self):
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

    def _pano_build_full(self):
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
        if self.kind_key() == "mosaic":
            if self.m is not None and name:
                self._show_mosaic_view(name)
            return
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

    # ------------------------------------------------------------------------------------ the flat mosaic
    def _build_mosaic_panel(self, MO, MB, MI):
        gm = self.group("Mosaic: cameras and ground", "Each view's camera found over the ground, hand-held at any "
                        "height; the whole set adjusted at once.")
        self.m_method = gm.add(MethodPicker(MO.METHOD_INFO))
        self.m_surface = gm.add(MethodPicker(MO.SURFACE_INFO))
        gs = self.group("Mosaic: size", "How big it is on the ground, and how many pixels per millimetre.")
        self.m_res = gs.add(MethodPicker(MB.RESOLUTION_INFO))
        self.m_mm = QDoubleSpinBox()
        self.m_mm.setRange(0.05, 50.0)
        self.m_mm.setDecimals(2)
        self.m_mm.setValue(1.0)
        self.m_mm.setSuffix(" mm per pixel")
        self.m_mm.setToolTip("Custom resolution (needs the size known: a ruler or focus distances).")
        gs.add(hbox(label("Custom"), self.m_mm))
        self.m_scale = gs.add(MethodPicker(MO.SCALE_INFO))
        self.m_ruler = QLineEdit()
        self.m_ruler.setPlaceholderText("Ruler: VIEW:U1,V1,U2,V2,MM  (e.g. 0:0.20,0.50,0.80,0.50,300)")
        self.m_ruler.setToolTip("Two points of a tape or ruler in one view (0..1 across and down that view's picture) "
                                "and the distance between them in mm: the mosaic's size becomes exact (cyan). The view "
                                "is its number in the strip, counting from 0, or its name.")
        gs.add(self.m_ruler)
        gh = self.group("Mosaic: height", "From each view's own depth (Dual Pixel RAW or a focus stack at each spot).")
        self.m_height = gh.add(self.tick("Height from the views' depth", True, "Each view's measured depth matched "
                                         "to its neighbours', the ground taken off, the relief's size measured from "
                                         "the parallax between overlapping views."))
        self.m_ortho = gh.add(self.tick("Orthorectify", True, "Each place lifted by its own height before it is "
                                        "looked up in a view, so kerbs, potholes and crack walls do not double at "
                                        "the seams."))
        self.m_nudge = gh.add(self.tick("Local nudge where no height is known", True, "Hides small leftover "
                                        "parallax in the overlaps. It hides it; it does not measure it (see Misfit)."))
        self.m_vig = gh.add(self.tick("Take out vignetting", True, "Lovely in a photograph; a texture repeats it as a "
                                      "dark blotch at every copy, so it is on here."))
        self.m_even = gh.add(self.tick("Even out brightness along the strip", True, "A cloud that passed while you "
                                       "walked does not leave a bright band."))
        go = self.group("Mosaic: out", "Game surfaces beside each other: colour, height, normal, confidence, "
                        "agreement, misfit and coverage, with a JSON that says how each was made.")
        self.m_tile = go.add(self.tick("Tileable", True, "The texture repeats without a visible join."))
        self.m_pieces = QLineEdit()
        self.m_pieces.setPlaceholderText("Pieces for decals: x0,y0,x1,y1; ... (0..1 of the result)")
        self.m_pieces.setToolTip("Boxes cut out of the finished mosaic with all their maps and a soft transparent "
                                 "edge: a pothole, a patch, a manhole, a stain.")
        go.add(self.m_pieces)
        self.card_b = QPushButton("Shooting card")
        self.card_b.clicked.connect(self._shooting_card)
        go.add(self.card_b)
        self.mosaic_groups = [gm, gs, gh, go]
        for w in (self.m_method, self.m_surface, self.m_res, self.m_scale):
            w.changed.connect(lambda *_: self._mosaic_changed())
        self.m_ruler.editingFinished.connect(self._mosaic_changed)

    def _shooting_card(self):
        from ..engine.stack import mosaicio as MI
        QMessageBox.information(self, "Shooting a flat mosaic", "\n".join(f"- {t}" for t in MI.SHOOTING_CARD))

    def kind_key(self) -> str:
        """panorama | mosaic | auto (Auto's answer once it has one for these frames)."""
        k = self.kind.key() if hasattr(self, "kind") else "panorama"
        if k == "auto" and self.auto is not None and self.auto["key"] == tuple(self.frame_paths()):
            return self.auto["kind"]
        return k or "panorama"

    def _kind_changed(self):
        k = self.kind_key()
        mosaic = k == "mosaic"
        pano = k == "panorama"
        for g in getattr(self, "mosaic_groups", []):
            g.setVisible(not pano)
        for w in (self.proj_group, self.level, self.flare, self.size_pick, self.pano_honest):
            w.setVisible(not mosaic)
        self.mosaic_honest.setVisible(not pano)
        cur = self.view_pick.currentText()
        self.view_pick.blockSignals(True)
        self.view_pick.clear()
        self.view_pick.addItems(M_VIEWS if mosaic else VIEWS)
        if cur in (M_VIEWS if mosaic else VIEWS):
            self.view_pick.setCurrentText(cur)
        self.view_pick.blockSignals(False)
        if self.kind.key() == "auto":
            self.kind_chip.setText(self.auto["text"] if (self.auto and self.auto["key"] == tuple(self.frame_paths()))
                                   else "Auto decides when you press Find layout, Preview or Build.")
        else:
            self.kind_chip.setText("")

    def _mosaic_changed(self):
        self.m = None
        self.mlast = None
        self._refresh()

    def mosaic_params(self):
        from ..engine.stack import mosaic as MO
        p = MO.MosaicParams()
        p.method = self.m_method.key() or "auto"
        p.surface = self.m_surface.key() or "plane"
        p.resolution = self.m_res.key() or "game"
        p.mm_per_px = float(self.m_mm.value())
        p.scale = self.m_scale.key() or "auto"
        p.ruler = self.m_ruler.text().strip()
        p.exposure = self.expo.isChecked()
        p.even_out = self.m_even.isChecked()
        p.vignette = self.m_vig.isChecked()
        p.seams = self.seams.key() or "smart"
        p.crop = self.crop.key() or "rectangle"
        p.height = self.m_height.isChecked()
        p.ortho = self.m_ortho.isChecked()
        p.nudge = self.m_nudge.isChecked()
        p.tileable = self.m_tile.isChecked()
        p.pieces = self.m_pieces.text().strip()
        p.focal_mm = float(self.focal_mm.value())
        return p

    def _mkey(self, p):
        return (tuple(self.frame_paths()), p.method, p.surface, p.scale, p.ruler, round(p.focal_mm, 2))

    def _decide(self, progress):
        """Auto's answer for these frames (in the worker)."""
        from ..engine.stack import mosaicio as MI
        key = tuple(self.frame_paths())
        if self.kind.key() != "auto":
            return self.kind.key()
        if self.auto is not None and self.auto["key"] == key:
            return self.auto["kind"]
        progress(0.01, "Panorama or flat mosaic?")
        a = MI.auto_kind(list(key))
        self.auto = dict(a, key=key)
        return a["kind"]

    def _solved(self, p, progress, cancel, share):
        from ..engine.stack import mosaic as MO
        from ..engine.stack import mosaicio as MI
        key = self._mkey(p)
        if self.m is not None and self.m["key"] == key:
            return self.m
        entries, left = MI.find_inputs(self.frame_paths())
        views, problems = MI.read_views(entries, None, lambda f, m: progress(share * 0.5 * f, m), cancel)
        for e in left:
            problems[e["name"]] = e["reason"]
        if len(views) < 2:
            raise MO.MosaicProblem("Fewer than two views could be read.")
        sol = MO.solve(views, p, lambda f, m: progress(share * (0.5 + 0.5 * f), m), cancel)
        return {"key": key, "views": views, "sol": sol, "problems": problems, "entries": entries}

    def find_layout(self):
        if len(self.frame_paths()) < 2:
            return
        if self.kind.key() == "panorama":
            return self._pano_find_layout()
        p = self.mosaic_params()

        def work(progress, cancel):
            kind = self._decide(progress)
            if kind != "mosaic":
                return ("panorama", None)
            return ("mosaic", self._solved(p, progress, cancel, 1.0))

        def done(r):
            kind, m = r
            self._kind_changed()
            if kind != "mosaic":
                self._pano_find_layout()
                return
            self.m = m
            self._show_mosaic()
            self.view_pick.setCurrentText("Layout")
            self._show_view()
            sol = m["sol"]
            self.jobbar.say(f"Layout found: {sum(sol.placed)} of {sol.n} views placed. Preview to see the mosaic.",
                            T.CYAN if all(sol.placed) else T.AMBER)
            self._refresh()
        if self.jobbar.start(work, done, lambda e, c: self.failed("Find layout", e, c), text="Finding the layout..."):
            self._refresh()

    def preview(self):
        if len(self.frame_paths()) < 2:
            return
        if self.kind.key() == "panorama":
            return self._pano_preview()
        from ..engine.stack import mosaicbuild as MB
        p = self.mosaic_params()

        def work(progress, cancel):
            kind = self._decide(progress)
            if kind != "mosaic":
                return ("panorama", None, None)
            m = self._solved(p, progress, cancel, 0.5)
            pv = MB.preview(m["views"], m["sol"], p, lambda f, msg: progress(0.5 + 0.5 * f, msg), cancel)
            return ("mosaic", m, pv)

        def done(r):
            kind, m, pv = r
            self._kind_changed()
            if kind != "mosaic":
                self._pano_preview()
                return
            self.m, self.mlast = m, pv
            self._show_mosaic()
            if self.view_pick.currentText() == "Layout":
                self.view_pick.setCurrentText("Result")
            self._show_view()
            self.jobbar.say("Preview ready (small). Build writes the full mosaic and its game surfaces.", T.CYAN)
            self._refresh()
        if self.jobbar.start(work, done, lambda e, c: self.failed("Preview", e, c), text="Previewing..."):
            self._refresh()

    def build_full(self):
        if len(self.frame_paths()) < 2 or self.jobbar.busy:
            return
        if self.kind.key() == "panorama":
            return self._pano_build_full()
        from ..engine.stack import mosaicbuild as MB
        from ..engine.stack import mosaicio as MI
        from ..engine.stack.frames import result_name
        p = self.mosaic_params()

        def work(progress, cancel):
            kind = self._decide(progress)
            if kind != "mosaic":
                return ("panorama", None, None)
            m = self._solved(p, progress, cancel, 0.3)
            r = MB.build(m["views"], m["sol"], p, None, lambda f, msg: progress(0.3 + 0.6 * f, msg), cancel)
            first = [q for e in m["entries"] for q in e["paths"]]
            name = result_name([first[0], first[-1]], "Mosaic")
            out = self.target(name, first[0])
            base = os.path.splitext(os.path.basename(out))[0]
            if base.endswith("_Studio"):
                base = base[:-len("_Studio")]
            progress(0.95, "Writing the surfaces")
            w = MI.write(r, m["sol"], m["views"], p, os.path.dirname(out), base, problems=m["problems"])
            return ("mosaic", m, w)

        def done(res):
            kind, m, w = res
            self._kind_changed()
            if kind != "mosaic":
                self._pano_build_full()
                return
            self.m = m
            self._show_mosaic()
            col = w["paths"].get("colour")
            self.last_result = col
            if col:
                self.result_written.emit(col)
            tip = (" " + w["coach"][0]) if w.get("coach") else ""
            self.jobbar.say(f"Saved {os.path.basename(col or '')} and its surfaces.{tip}"[:260], T.GREEN)
            self._refresh()
        if self.jobbar.start(work, done, lambda e, c: self.failed("Build", e, c), keep=True, text="Building..."):
            self._refresh()

    def _show_mosaic(self):
        from ..engine.stack import mosaicio as MI
        m = self.m
        if m is None:
            return
        sol, views = m["sol"], m["views"]
        names = {v.name: i for i, v in enumerate(views)}
        for e in m["entries"]:
            i = names.get(e["name"])
            if i is None:
                st, tie, tip = "left out", "dropped", m["problems"].get(e["name"], "could not be read")
            elif not sol.placed[i]:
                st, tie, tip = "not placed", "dropped", sol.problems.get(i, "no overlap found")
            else:
                mine = [q for q in sol.pairs if i in (q["i"], q["j"])]
                bad = [q for q in mine if q["rms"] > 1.5]
                st, tie = ("placed (check)", "warn") if bad or not mine else ("placed", "ok")
                tip = "; ".join(f"with {views[q['j'] if q['i'] == i else q['i']].name}: {q['inliers']} matches, "
                                f"{q['rms']:.1f} px" for q in mine) or "no overlap found"
            for path in e["paths"]:
                self.tray.set_status(path, st, tie, tip)
        tips = MI.coach(sol, views, self.mlast["plan"] if self.mlast else None)
        head = (f"{sum(sol.placed)} of {sol.n} views placed ({MO_name(sol.method)}), residual {sol.rms:.2f} px. "
                f"Size: {sol.scale_how}.")
        self.report.setText(head + (" " + " ".join(tips) if tips else ""))
        self.report.setStyleSheet(f"color:{T.CYAN if all(sol.placed) else T.AMBER};")
        chips = [(f"{sum(sol.placed)} placed", T.CYAN if all(sol.placed) else T.AMBER),
                 ("size " + {"cyan": "measured", "amber": "estimated", "violet": "relative"}[sol.scale_colour],
                  {"cyan": T.CYAN, "amber": T.AMBER, "violet": T.VIOLET}[sol.scale_colour])]
        if self.mlast is not None and self.mlast["plan"].get("hm") is not None:
            hm = self.mlast["plan"]["hm"]
            chips.append(("relief measured" if hm.alpha else "relief relative", T.CYAN if hm.alpha else T.AMBER))
        self.set_chips(chips)

    def _mosaic_layout_image(self) -> QImage:
        from ..engine.stack import mosaic as MO
        from ..engine.stack import mosaicbuild as MB
        m = self.m
        sol, views = m["sol"], m["views"]
        res, _, _ = MB.resolution(sol, views, MO.MosaicParams(resolution="full"))
        cv = MB.make_canvas(sol, views, res)
        s = 1100.0 / max(cv.W, cv.H)
        cv = cv.resized(s)
        pad = 30
        img = QImage(cv.W + 2 * pad, cv.H + 2 * pad, QImage.Format.Format_RGB888)
        img.fill(QColor(T.BG2))
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = QFont("Segoe UI", 10)
        f.setBold(True)
        p.setFont(f)
        for i in range(sol.n):
            if not sol.placed[i]:
                continue
            mine = [q for q in sol.pairs if i in (q["i"], q["j"])]
            tie = "warn" if (not mine or any(q["rms"] > 1.5 for q in mine)) else "ok"
            col = QColor(TIE[tie])
            fp = sol.footprint(i, views[i].shape, 10)
            u, v = cv.from_ground(fp[:, 0], fp[:, 1])
            poly = QPolygonF([QPointF(x + pad, y + pad) for x, y in zip(u.tolist(), v.tolist())])
            fillc = QColor(col)
            fillc.setAlpha(28)
            p.setBrush(fillc)
            p.setPen(QPen(col, 2))
            p.drawPolygon(poly)
            p.drawText(QPointF(float(u.mean()) + pad - 8, float(v.mean()) + pad + 5), str(i + 1))
        p.setPen(QColor(T.DIM))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(QPointF(pad, img.height() - 10), "Seen from above: cyan placed, amber check the overlap")
        p.end()
        return img

    def _show_mosaic_view(self, name):
        from ..engine.stack import mosaicbuild as MB
        if name == "Layout":
            self.result.set_images(self._mosaic_layout_image(), None, chip_text="Layout from above",
                                   chip_colour=T.VIOLET)
            return
        if self.mlast is None:
            self.result.note("Press Preview to see the mosaic.")
            return
        pv = self.mlast
        sp = pv["plan"]
        y0, y1, x0, x1 = pv["crop"]
        base = qimage_rgb8(MB.linear_to_srgb8(pv["rgb"]))
        if name == "Result":
            self.result.set_images(base, None, chip_text="Flat mosaic preview", chip_colour=T.CYAN)
        elif name == "Seams":
            lab = sp["labels"][y0:y1, x0:x1]
            cols = label_colours(int(lab.max()) + 1 if lab.size else 1)
            from .common import to_qimage
            a8 = np.zeros(lab.shape + (3,), np.uint8)
            ok = lab >= 0
            a8[ok] = cols[lab[ok]].astype(np.uint8)
            img = QImage(base)
            p = QPainter(img)
            p.setOpacity(0.45)
            p.drawImage(0, 0, to_qimage(a8).scaled(img.width(), img.height()))
            p.end()
            self.result.set_images(img, base, chip_text="Seams: which view each place came from", chip_colour=T.VIOLET)
        elif name.startswith("Ghosts"):
            gmap = sp["ghosts"][y0:y1, x0:x1]
            L = pv["rgb"].mean(axis=2)
            g = np.sqrt(np.clip(L / max(float(np.percentile(L, 99.5)), 1e-6), 0, 1)) * 140
            a8 = np.stack([g, g, g], -1) * (1 - gmap[..., None]) + gmap[..., None] * np.array([245, 158, 11])
            self.result.set_images(qimage_rgb8(np.clip(a8, 0, 255).astype(np.uint8)), base,
                                   chip_text="Ghosts: views disagreed here (left out or cut)", chip_colour=T.AMBER)
        elif name == "Height":
            if sp.get("relief") is None:
                self.result.note("No height: no view had measured depth (Dual Pixel RAW or a focus stack).")
                return
            h = sp["relief"][y0:y1, x0:x1]
            lo, hi = np.percentile(h, [1, 99])
            g = np.clip((h - lo) / max(hi - lo, 1e-9), 0, 1)
            gy, gx = np.gradient(g)
            shade = np.clip(0.75 + 6.0 * (gx - gy), 0, 1)
            a8 = (np.stack([g * 0.35 + 0.65 * shade] * 3, -1) * 255).astype(np.uint8)
            hm = sp["hm"]
            self.result.set_images(qimage_rgb8(a8), base, chip_text="Height: " + ("measured (relief by parallax)" if
                                   hm.alpha else "relative"), chip_colour=T.CYAN if hm.alpha else T.AMBER)
        else:
            mf = sp["misfit"][y0:y1, x0:x1]
            L = pv["rgb"].mean(axis=2)
            g = np.sqrt(np.clip(L / max(float(np.percentile(L, 99.5)), 1e-6), 0, 1)) * 140
            t = np.clip(mf / 2.0, 0, 1)[..., None]
            a8 = np.stack([g, g, g], -1) * (1 - t) + t * np.array([192, 132, 252])
            self.result.set_images(qimage_rgb8(np.clip(a8, 0, 255).astype(np.uint8)), base,
                                   chip_text="Misfit: violet where views still disagree in position (2 px = full)",
                                   chip_colour=T.VIOLET)


def MO_name(method: str) -> str:
    from ..engine.stack import mosaic as MO
    return MO.METHOD_INFO.get(method, {}).get("name", method).lower()
