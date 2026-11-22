"""The Dual Pixel Lab page (BUILD_PLAN_TABS 3.6): everything a Canon Dual Pixel RAW file can do.

One photo in (Open in Dual Pixel Lab from the Library, Open photo..., or drop it; or two DPRSplit DNGs). The Lab
reads both frames, estimates each half's shading, measures the offset between the halves (depth), how sure that is,
and where they agree (detail) or not (noise). Views: A+B, half 1, half 2, difference, depth, confidence, noise vs
detail; W flips between the two halves in place. It makes: the halves as files (what DPRSplit does), highlight
recovery (about one extra stop) as a linear intermediate for Develop, a depth PNG, a wigglegram, an anaglyph and a
side-by-side pair. A photo shot without Dual Pixel RAW says so, with how to turn it on.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QComboBox, QFileDialog, QPushButton

from ..engine.stack.focus import Cancelled
from . import theme as T
from .common import chip, hbox, label
from .stackui import StackPage, honest_card, qimage_rgb8, scene_to_qimage


class DualPixelPage(StackPage):
    SETTINGS_KEY = "dual_pixel"
    MIN_FRAMES = 1

    def __init__(self, settings=None):
        from ..engine.dualpixel import decode, depth, hdr, views
        self.D, self.DEP, self.HDR, self.V = decode, depth, hdr, views
        self.dp = None
        self.gains = None
        self.maps = {}
        self.cache = {}
        self.pair = None
        super().__init__(
            "Dual Pixel Lab",
            "Everything Canon Dual Pixel RAW can do: the two halves, about one extra stop in the highlights, depth "
            "and focus from the halves' offset, a noise-versus-detail map, and 3D.",
            "Give it one photo shot in Dual Pixel RAW: in the Library right-click it, Open in Dual Pixel Lab; or drop "
            "it here. Two DNGs from DPRSplit work too.",
            decode.HOW_TO, settings, add_text="Open photo...")
        self.pair_b = QPushButton("Use two DPRSplit files...")
        self.pair_b.clicked.connect(self.choose_pair)
        self.empty_extra.addWidget(self.pair_b)
        self.wiggle = QTimer(self, interval=160, timeout=self._wiggle_step)
        self._wig = 0
        QShortcut(QKeySequence("W"), self, activated=self.toggle_wiggle)
        self.tray.setFixedHeight(0)                  # one photo: the tray is not needed
        self.tray.hide()

    # ------------------------------------------------------------------------------------ panel
    def build_panel(self):
        from ..engine.dualpixel import hdr as HDR
        from ..engine.dualpixel import views as V
        g = self.group("View")
        self.view_pick = g.add(QComboBox())
        self.view_pick.addItems(V.VIEWS)
        self.view_pick.currentIndexChanged.connect(self._show_view)
        self.view_text = g.add(label(V.VIEW_INFO["A+B"], "dim", wrap=True))
        self.shade_chip = g.add(chip("Shading: not measured yet", T.DIM))
        g.add(label("W flips between half 1 and half 2 in place.", "dim", wrap=True))
        g = self.group("Split", "Each half as its own photo, developed like the original and brightened to match "
                       "(what DPRSplit does, built in).")
        b1, b2, bb = QPushButton("Half 1"), QPushButton("Half 2"), QPushButton("Both")
        b1.clicked.connect(lambda: self.export_halves((1,)))
        b2.clicked.connect(lambda: self.export_halves((2,)))
        bb.clicked.connect(lambda: self.export_halves((1, 2)))
        g.add(hbox(b1, b2, bb))
        g = self.group("Highlights", "Where the photo is blown, one half often is not: it got half the light.")
        self.strength = g.add(self.slider("Strength", 0.0, 1.0, 1.0, 0.05, "{:.0%}",
                                          "How much of the recovered highlight is used where the photo is near its "
                                          "clip."))
        self.hdr_report = g.add(chip("Not built yet", T.DIM))
        g.add(honest_card(HDR.HONEST, "What to expect", T.CYAN))
        self.hdr_b = QPushButton("Build highlight recovery")
        self.hdr_b.setObjectName("primary")
        self.hdr_b.clicked.connect(self.build_full)
        g.add(self.hdr_b)
        g = self.group("Depth and focus", "From the offset between the halves: things in focus line up, things in "
                       "front or behind do not.")
        self.focus_text = g.add(label("", "dim", wrap=True))
        self.swap = g.add(self.tick("Swap near / far", False, "Which side of the lens half 1 looks through is not "
                                    "assumed: if near and far look the wrong way round, tick this."))
        self.swap.toggled.connect(lambda *_: (self.cache.pop("Depth", None), self._update_text(), self._show_view()))
        db = QPushButton("Export depth map (16-bit PNG)")
        db.clicked.connect(self.export_depth)
        g.add(db)
        g = self.group("Teach the Studio which way is near", "The two halves cannot tell near from far on their own. "
                       "Pick photos of ground receding at an angle (a path, a lawn, a table top: the bottom of the "
                       "frame nearer), ideally from three or more different scenes. Each votes; the answer is saved "
                       "for your camera once three scenes agree and none disagrees.")
        self.near_text = g.add(label(self._near_state(), "dim", wrap=True))
        nb = QPushButton("Teach the Studio which way is near...")
        nb.clicked.connect(self.teach_near)
        g.add(nb)
        g = self.group("Noise vs detail")
        self.nvd_text = g.add(label("Real detail is the same in both halves; noise is not. Cyan in the Noise vs "
                                    "detail view is real detail, dark is noise.", "dim", wrap=True))
        g = self.group("3D")
        self.boost = g.add(self.slider("Exaggerate", 1.0, 4.0, 1.0, 0.1, "{:.1f}x",
                                       "Moves the two views further apart along their measured offset for a "
                                       "stronger 3D effect (wigglegram)."))
        wb, ab, sb = QPushButton("Wigglegram"), QPushButton("Anaglyph"), QPushButton("Side by side")
        wb.clicked.connect(lambda: self.export_3d("wiggle"))
        ab.clicked.connect(lambda: self.export_3d("anaglyph"))
        sb.clicked.connect(lambda: self.export_3d("pair"))
        g.add(hbox((wb, 1), (ab, 1), stretch=False))
        g.add(hbox((sb, 1), stretch=False))
        g = self.group("Lab (experimental)", "Air steadiness: heat shimmer bends the two views differently from place "
                       "to place. The patchy part of the offset map, as one number.")
        self.air_text = g.add(label("", "dim", wrap=True))
        self.panel.addStretch(1)
        self.open_b = QPushButton("Open in Develop")
        self.open_b.clicked.connect(lambda: self.last_result and self.open_result.emit(self.last_result))
        self.folder_b = QPushButton("Folder...")
        self.folder_b.clicked.connect(self.choose_folder)
        self.actions.addWidget(hbox((self.open_b, 1), self.folder_b, stretch=False))
        self.where = label("", "dim", wrap=True)
        self.actions.addWidget(self.where)

    def _near_state(self, model: str = "") -> str:
        from ..engine.dualpixel import depth as DEP
        m = model or (str(self.dp.image.meta.model) if self.dp is not None else "Canon EOS R5")
        sign, known, tier, how = DEP.near_source(m)
        if not known:
            return f"{m}: not taught yet (the usual reading is used, and every depth says so)."
        where = {"measured": "taught on your own frames (cyan)", "shared": "from a shared profile (violet)",
                 "table": "checked on receding scenes when the Studio was built"}.get(tier, tier)
        return f"{m}: known, {where}."

    def teach_near(self):
        from PySide6.QtWidgets import QFileDialog
        if self.jobbar.busy:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Photos of ground receding at an angle (Dual Pixel RAW)", "",
                                                "Dual Pixel RAW (*.CR3 *.cr3 *.CR2 *.cr2)")
        if not files:
            return

        def work(progress, cancel):
            from ..engine.dualpixel import nearcheck as NC
            res = NC.run(NC.groups_by_time(files), progress, cancel)
            res["saved"] = NC.save_to_profile(res)
            return res

        def done(res):
            dec = res["decision"]
            self.near_text.setText(dec["sentence"] + (" Saved for your camera." if res.get("saved") else "") +
                                   "\n" + self._near_state(res.get("camera") or ""))
            self.jobbar.say(dec["sentence"], T.CYAN if dec["locked"] else T.AMBER)

        self.jobbar.start(work, done, lambda e, c: self.failed("Which way is near", e, c),
                          text="Reading the halves of each photo...")

    # ------------------------------------------------------------------------------------ opening
    def add_files(self, files: list, replace: bool = True):
        files = self.expand(files)
        if not files:
            return
        dngs = [f for f in files if f.lower().endswith(".dng")]
        if len(dngs) == 2 and len(files) == 2:
            self.open_pair(dngs[0], dngs[1])
            return
        self.open_photo(files[0])

    def choose_files(self):
        from ..engine.io.rawfile import RAW_EXTENSIONS
        pats = " ".join(f"*{x}" for x in RAW_EXTENSIONS)
        f, _ = QFileDialog.getOpenFileName(self, "Open a Dual Pixel RAW photo", self.settings.get("folder") or "",
                                           f"Raw photos ({pats})")
        if f:
            self.open_photo(f)

    def choose_pair(self):
        files, _ = QFileDialog.getOpenFileNames(self, "The two DNGs DPRSplit made (A+B and one half)",
                                                self.settings.get("folder") or "", "DNG (*.dng *.DNG)")
        if len(files) == 2:
            self.open_pair(files[0], files[1])
        elif files:
            self.jobbar.say("Pick exactly two DNG files: the A+B frame and the half frame.", T.AMBER)

    def open_photo(self, path: str):
        self._load([path], lambda: self.D.open_dual(path))

    def open_pair(self, a: str, b: str):
        self._load([a, b], lambda: self.D.open_dual_pair(a, b))

    def _load(self, paths, opener):
        if self.jobbar.busy:
            return
        from ..engine.stack.frames import probe
        self.items = [(f.path, f.ok, f.problem) for f in probe(paths)]
        self.tray.set_frames(self.items)
        self.dp, self.gains, self.maps, self.cache = None, None, {}, {}
        self.last_result = ""
        self.out_dir = ""
        self._default_folder()
        self.result.set_images(None)
        self.centre.setCurrentIndex(1)
        D, DEP = self.D, self.DEP

        def work(progress, cancel):
            progress(0.05, "Reading both frames")
            dp = opener()
            progress(0.35, "Measuring each half's shading")
            gains = D.shading(dp)
            if cancel.is_set():
                raise Cancelled()
            progress(0.5, "Measuring the offset between the halves")
            la, lb = DEP.halves_luma(dp, gains)
            disp, conf = DEP.disparity(la, lb)
            ref = DEP.refine(disp, conf, la + lb)
            progress(0.8, "Noise versus detail")
            rho = DEP.detail_map(la, lb, ref)
            air = DEP.air_steadiness(ref, conf)
            progress(0.9, "Developing the photo")
            first = self.V.view(dp, gains, "A+B", 0.25, {})
            return dp, gains, {"disp": ref, "conf": conf, "rho": rho, "air": air}, first

        def done(r):
            self.dp, self.gains, self.maps, first = r
            self.cache = {"A+B": first}
            src = getattr(self.dp, "source", "")
            self.shade_chip.setText("Shading: estimated from this photo")
            self.shade_chip.setStyleSheet(f"color:{T.AMBER}; border-color:{T.alpha(T.AMBER, 0.33)};")
            self._update_text()
            self.view_pick.setCurrentIndex(0)
            self._show_view()
            notes = " ".join(self.dp.notes)
            self.jobbar.say(f"Dual Pixel photo read ({src}; halves {self.dp.level_ratio:.2f}x apart in brightness). "
                            + notes, T.CYAN if not notes else T.AMBER)
            self.set_chips([("Dual Pixel RAW", T.CYAN), (f"ratio {self.dp.level_ratio:.2f}", T.CYAN)])
            self._refresh()

        def fail(err, cancelled):
            if self.D.NOT_DUAL in err:
                self._not_dual(err)
                return
            self.failed("Opening", err, cancelled)

        self.jobbar.start(work, done, fail, text="Opening...")
        self._refresh()

    def _not_dual(self, err: str):
        self.result.set_images(None)
        self.result.note(self.D.NOT_DUAL)
        self.jobbar.say(err.split(": ", 1)[-1], T.AMBER)
        self.set_chips([("not Dual Pixel RAW", T.AMBER)])
        self._refresh()

    def _update_text(self):
        if self.dp is None:
            return
        rep = self.DEP.focus_report(self.maps["disp"], self.maps["conf"], self.swap.isChecked())
        self.focus_text.setText(rep["sentence"] + f" In focus: about {100 * rep['in_focus_share']:.0f}% of the "
                                f"frame.")
        air = self.maps.get("air", {})
        self.air_text.setText(f"Patchiness of the offset map: {air.get('spread_px', 0):.3f} px "
                              f"(lower is steadier air). Experimental.")

    # ------------------------------------------------------------------------------------ views
    def _show_view(self, *_):
        name = self.view_pick.currentText()
        self.view_text.setText(self.V.VIEW_INFO.get(name, ""))
        if self.dp is None:
            return
        if name in self.cache:
            self._put(name, self.cache[name])
            return
        maps = dict(self.maps)
        if name == "Depth" and self.swap.isChecked():
            maps["disp"] = -maps["disp"]
        dp, gains, V = self.dp, self.gains, self.V

        def work(progress, cancel):
            return V.view(dp, gains, name, 0.25, maps)

        def done(img):
            self.cache[name] = img
            if self.view_pick.currentText() == name:
                self._put(name, img)

        self.jobbar.start(work, done, lambda e, c: self.failed("View", e, c), text=f"Making the {name} view...")

    def _put(self, name, img):
        colour = T.CYAN if name in ("Depth", "Confidence", "Noise vs detail") else T.VIOLET
        ref = self.cache.get("A+B")
        self.result.set_images(qimage_rgb8(img), qimage_rgb8(ref) if ref is not None else None,
                               chip_text=name + (" · measured" if colour == T.CYAN else ""), chip_colour=colour)
        self.result.ref_chip = ("A+B (the normal photo)", T.AMBER)

    def toggle_wiggle(self):
        if self.dp is None:
            return
        if self.wiggle.isActive():
            self.wiggle.stop()
            self._show_view()
            return
        for n in ("Half 1", "Half 2"):
            if n not in self.cache:
                self.cache[n] = self.V.view(self.dp, self.gains, n, 0.25, self.maps)
        self.wiggle.start()

    def _wiggle_step(self):
        self._wig ^= 1
        n = "Half 1" if self._wig else "Half 2"
        self._put(n, self.cache[n])

    # ------------------------------------------------------------------------------------ outputs
    def _base(self) -> str:
        return self.dp.path if self.dp is not None else ""

    def export_halves(self, which):
        if self.dp is None or self.jobbar.busy:
            return
        dp, gains, V = self.dp, self.gains, self.V
        targets = [(w, self.target(V.out_name(self._base(), f"half{w}"), self._base())) for w in which]

        def work(progress, cancel):
            out = []
            for i, (w, t) in enumerate(targets):
                progress(i / len(targets), f"Writing half {w}")
                out.append(V.export_half(dp, gains, w, t))
            return out

        def done(paths):
            self.last_result = paths[-1]
            for p in paths:
                self.result_written.emit(p)
            self.jobbar.say("Saved " + ", ".join(os.path.basename(p) for p in paths), T.GREEN)
            self._refresh()

        self.jobbar.start(work, done, lambda e, c: self.failed("Split", e, c), keep=True, text="Writing...")

    def build_full(self):
        """Highlight recovery, written as <stem>_DP-HDR_Studio.tif."""
        if self.dp is None or self.jobbar.busy:
            return
        dp, gains, HDR = self.dp, self.gains, self.HDR
        disp = self.maps.get("disp")
        strength = self.strength.value()
        target = self.target(self.V.out_name(self._base(), "hdr"), self._base())

        def work(progress, cancel):
            from ..engine.io.intermediate import write_intermediate
            from ..engine.pipeline.pipeline import apply_orientation
            progress(0.1, "Developing the photo and half 1")
            rgb, rep = HDR.recover(dp, gains, disp, strength=strength)
            if cancel.is_set():
                raise Cancelled()
            progress(0.8, "Writing")
            md = dp.image.meta
            meta = {"make": md.make, "model": md.model, "iso": md.iso, "exposure_s": md.exposure_s,
                    "fnumber": md.fnumber, "focal_mm": md.focal_mm, "lens": md.lens, "datetime": md.datetime}
            write_intermediate(target, np.ascontiguousarray(apply_orientation(rgb, dp.flip)), meta,
                               {"kind": "dualpixel", "method": "hdr", "sources": [os.path.basename(dp.path)],
                                "scale": 1.0, "report": {k: v for k, v in rep.items() if k != "sentence"}})
            from ..engine.stack.focus import write_stack_block
            write_stack_block(target, "dualpixel", [dp.path], {"strength": strength})
            return target, rep

        def done(r):
            path, rep = r
            self.last_result = path
            self.result_written.emit(path)
            self.hdr_report.setText(rep["sentence"])
            col = T.CYAN if rep["stops"] > 0 else T.DIM
            self.hdr_report.setStyleSheet(f"color:{col}; border-color:{T.alpha(col, 0.33)};")
            self.jobbar.say(f"Saved {os.path.basename(path)}. Open in Develop: the Highlights slider brings the "
                            f"recovered range into view.", T.GREEN)
            self._refresh()

        self.jobbar.start(work, done, lambda e, c: self.failed("Highlight recovery", e, c), keep=True,
                          text="Recovering highlights...")

    def export_depth(self):
        if self.dp is None or self.jobbar.busy:
            return
        from ..engine.stack.focus import _up2
        dp, maps, V = self.dp, self.maps, self.V
        swap = self.swap.isChecked()
        target = self.target(V.out_name(self._base(), "depth"), self._base())

        def work(progress, cancel):
            return V.depth_png(_up2(maps["disp"], dp.shape), maps["conf"], target, dp.flip, swap)

        def done(p):
            self.result_written.emit(p)
            self.jobbar.say(f"Saved {os.path.basename(p)} (near is white).", T.GREEN)

        self.jobbar.start(work, done, lambda e, c: self.failed("Depth map", e, c), keep=True, text="Writing...")

    def export_3d(self, kind: str):
        if self.dp is None or self.jobbar.busy:
            return
        dp, gains, V, maps = self.dp, self.gains, self.V, self.maps
        boost = self.boost.value()
        target = self.target(V.out_name(self._base(), kind), self._base())

        def work(progress, cancel):
            if kind == "wiggle":
                return V.wigglegram(dp, gains, target, 1200, 120, boost, maps.get("disp"))
            arr = V.anaglyph(dp, gains, 0.5) if kind == "anaglyph" else V.side_by_side(dp, gains, 0.5)
            return V.save_jpeg(arr, target)

        def done(p):
            self.result_written.emit(p)
            self.jobbar.say(f"Saved {os.path.basename(p)}.", T.GREEN)

        self.jobbar.start(work, done, lambda e, c: self.failed("3D", e, c), keep=True, text="Writing...")

    def _refresh(self):
        super()._refresh()
        has = self.dp is not None
        for w in ("hdr_b",):
            if hasattr(self, w):
                getattr(self, w).setEnabled(has and not self.jobbar.busy)
