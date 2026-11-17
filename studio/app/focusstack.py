"""The Focus Stack page (BUILD_PLAN_TABS 4.2): a focus bracket in, one sharp photo out.

Preview analyses the frames at half size (line up, order, even out, measure sharpness) and merges them in seconds
to a minute; Build full size decodes every frame once more and writes <first>_to_<last>_Focus_Studio.tif, a linear
intermediate that opens in Develop with every tool working on it. Changing the method, smoothness or halo guard
re-merges from the analysis; changing alignment, detail size, brightness or "clean first" analyses again.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QPushButton

from . import theme as T
from .common import hbox, label
from .stackui import MethodPicker, StackPage, honest_card, qimage_rgb8, scene_to_qimage

VIEWS = ("Result", "Depth map", "Confidence", "Selected frame")


class FocusStackPage(StackPage):
    SETTINGS_KEY = "focus_stack"

    def __init__(self, settings=None):
        from ..engine.stack import focus as FS
        self.FS = FS
        self.analysis = None
        self.preview_img = None
        super().__init__(
            "Focus Stack",
            "Merges a focus bracket into one photo sharp from front to back, straight from the raws.",
            "Give it frames: select them in the Library, right-click, Send to Focus Stack; or drop files or a folder "
            "here.",
            FS.WHAT_TO_SHOOT, settings)

    # ------------------------------------------------------------------------------------ panel
    def build_panel(self):
        FS = self.FS
        from ..engine.stack.align import ALIGN_INFO
        g = self.group("Method")
        self.method = g.add(MethodPicker(FS.METHOD_INFO, list(FS.METHODS)))
        self.method.changed.connect(lambda *_: self._remerge())
        g = self.group("Line up", "Frames are lined up neighbour to neighbour and put in focus order.")
        self.align = g.add(MethodPicker({k: ALIGN_INFO[k] for k in ("similarity", "affine", "homography")}))
        self.local = g.add(self.tick("Line up locally too (hand-held)", True,
                                     "After the overall line-up, small local movements are measured and followed: "
                                     "a hand-held macro changes perspective a little between frames. Harmless on a "
                                     "tripod."))
        rev = QPushButton("Reverse order")
        rev.setToolTip("The tray shows the focus order (near to far). Reverse it if it came out the wrong way round.")
        rev.clicked.connect(self.tray.reverse)
        g.add(hbox(rev, label("near → far, as in the tray", "dim")))
        g = self.group("Detail and joins")
        self.radius = g.add(self.slider("Detail size", 0.5, 3.0, 1.0, 0.05, "{:.2f}",
                                        "How big the detail is that decides which frame is sharpest. Larger for "
                                        "noisy frames or soft lenses."))
        self.smooth = g.add(self.slider("Smoothness of joins", 0.0, 3.0, 1.0, 0.05, "{:.2f}",
                                        "How softly the places taken from different frames are joined."))
        self.smooth.changed.connect(lambda *_: self._remerge())
        self.equalise = g.add(self.tick("Even out brightness", True, "Flash or shutter flicker between frames is "
                                        "evened out (at most 0.3 stop per frame)."))
        self.halo = g.add(self.tick("Halo guard", True, "Nothing in the result may be brighter or darker than every "
                                    "frame was at that place: removes halos at depth edges."))
        self.halo.toggled.connect(lambda *_: self._remerge())
        self.denoise = g.add(self.tick("Clean each frame first", False, "Runs Clean (the denoiser) on every frame "
                                       "before merging. Slower; the weighted methods already average noise down."))
        g = self.group("View")
        self.view_pick = g.add(QComboBox())
        self.view_pick.addItems(VIEWS)
        self.view_pick.currentIndexChanged.connect(self._show_view)
        self.tray.selected.connect(lambda *_: self.view_pick.currentText() == "Selected frame" and self._show_view())
        self.report = g.add(label("", "dim", wrap=True))
        self.panel.addWidget(honest_card(FS.HONEST))
        self.panel.addStretch(1)
        self.add_actions()

    def params(self):
        FS = self.FS
        return FS.FocusParams(method=self.method.key() or "auto", denoise_first=self.denoise.isChecked(),
                              align=self.align.key() or "similarity", equalise=self.equalise.isChecked(),
                              radius=self.radius.value(), smooth=self.smooth.value(),
                              halo_guard=self.halo.isChecked(), order="capture", local=self.local.isChecked())

    # ------------------------------------------------------------------------------------ frames
    def frames_changed(self):
        if self.analysis is not None and self.analysis.paths != self._ordered_paths():
            self._drop_analysis()

    def _ordered_paths(self):
        return self.frame_paths()

    def _drop_analysis(self):
        if self.analysis is not None:
            try:
                self.analysis.close()
            except Exception:                        # noqa: BLE001
                pass
        self.analysis = None

    def _needs_analysis(self, p) -> bool:
        a = self.analysis
        return a is None or a.params.analysis_key() != p.analysis_key() or a.paths != self._ordered_paths()

    # ------------------------------------------------------------------------------------ preview / build
    def preview(self):
        p = self.params()
        paths = self._ordered_paths()
        if len(paths) < 2:
            return
        need = self._needs_analysis(p)
        if need:
            self._drop_analysis()
        FS = self.FS
        first = self.analysis is None and self.settings.get("focus_order_auto", True)
        p_an = FS.FocusParams.from_dict(p.to_dict())
        p_an.order = "auto" if first else "capture"
        old = self.analysis

        def work(progress, cancel):
            an = old if (old is not None and not need) else FS.analyse(paths, p_an, progress, cancel)
            try:
                img = FS.preview(an, p, progress, cancel)
            except BaseException:
                if an is not old:
                    an.close()
                raise
            return an, img

        def done(r):
            an, img = r
            self.analysis = an
            # the analysis used the frames' order it settled on; keep the tray in that order
            if an.paths != self._ordered_paths():
                self.tray.reorder(an.paths)
            self.analysis.params.order = "capture"
            self.preview_img = img
            self._show_analysis()
            self._show_view()
            self.jobbar.say(f"Preview ready ({img.shape[1]} x {img.shape[0]} px, half size). Press B to compare "
                            f"with the sharpest single frame.", T.CYAN)
            self._refresh()

        if self.jobbar.start(work, done, lambda e, c: self.failed("Preview", e, c), text="Previewing..."):
            self._refresh()

    def _show_analysis(self):
        an = self.analysis
        if an is None:
            return
        for r in an.reports:
            if r.status == "dropped":
                self.tray.set_status(r.path, "left out", "dropped", r.reason)
                continue
            parts = [f"{r.residual_px:.1f} px"]
            if abs(r.gain_stops) > 0.02:
                parts.append(f"{r.gain_stops:+.2f} st")
            self.tray.set_status(r.path, " · ".join(parts), r.status,
                                 (r.reason + "\n" if r.reason else "") +
                                 f"Line-up residual {r.residual_px:.2f} px, brightness {r.gain_stops:+.2f} stop, "
                                 f"scale {r.scale:.4f}")
        notes = list(an.notes)
        self.report.setText(" ".join(notes) if notes else f"{an.n} frames lined up and in focus order.")
        warn = sum(1 for r in an.reports if r.status == "warn")
        self.set_chips([(f"{warn} warning{'s' if warn != 1 else ''}", T.AMBER)] if warn else
                       [("lined up", T.CYAN)])

    def _remerge(self):
        if self.analysis is None or self.jobbar.busy:
            return
        self.preview()

    def _show_view(self, *_):
        FS = self.FS
        an = self.analysis
        name = self.view_pick.currentText()
        if an is None or self.preview_img is None:
            return
        ref = scene_to_qimage(FS.source_patch(an, an.ref), 2400)
        if name == "Result":
            self.result.set_images(scene_to_qimage(self.preview_img, 2400), ref, chip_text="Focus stack · preview",
                                   chip_colour=T.CYAN)
        elif name in ("Depth map", "Confidence"):
            m = FS.view_map(an, "depth" if name == "Depth map" else "confidence")
            self.result.set_images(qimage_rgb8(m), ref, chip_text=f"{name} · measured", chip_colour=T.CYAN)
        else:
            row = self.tray.currentRow()
            paths = self.tray.paths(False)
            j = an.paths.index(paths[row]) if 0 <= row < len(paths) and paths[row] in an.paths else an.ref
            self.result.set_images(scene_to_qimage(FS.source_patch(an, j), 2400), ref,
                                   chip_text=f"Frame {j + 1} lined up", chip_colour=T.VIOLET)

    def build_full(self):
        FS = self.FS
        p = self.params()
        paths = self._ordered_paths()
        if len(paths) < 2 or self.jobbar.busy:
            return
        from ..engine.stack.frames import result_name
        name = result_name(paths, "Focus")
        out = self.target(name, paths[0])
        need = self._needs_analysis(p)
        old = None if need else self.analysis
        if need:
            self._drop_analysis()
        p_an = FS.FocusParams.from_dict(p.to_dict())
        p_an.order = "capture"

        def work(progress, cancel):
            an = old if old is not None else FS.analyse(paths, p_an, lambda f, m: progress(0.4 * f, m), cancel)
            try:
                path = FS.build(an, p, out, lambda f, m: progress(0.4 + 0.6 * f, m), cancel)
            except BaseException:
                if an is not old:
                    an.close()
                raise
            return an, path

        def done(r):
            an, path = r
            self.analysis = an
            self._show_analysis()
            self.last_result = path
            self.result_written.emit(path)
            self.jobbar.say(f"Saved {os.path.basename(path)}. Open in Develop to finish it.", T.GREEN)
            self._refresh()

        def fail(err, cancelled):
            self.failed("Build", err, cancelled)     # a real fault has its error report from the job already

        if self.jobbar.start(work, done, fail, keep=True, text="Building the full-size stack..."):
            self._refresh()
