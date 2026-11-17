"""The Super-Res Stack page (BUILD_PLAN_TABS 5.7): a burst in, a larger or cleaner photo out.

Mode first (hand-held burst, cleaner same size, moon and planets, night sky), then the merge with what it is best
for and weak at. Preview builds a 100% window of the result the real way (about 768 px square, around the
sharpest place, or the place you click on the overview), so what you see is what Build will make; B shows the
best single frame enlarged the same way. The coverage diagram shows whether the frames landed on different
sub-pixel positions (that is where extra detail comes from). The size, disk space and time are said before a
build starts.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtWidgets import QButtonGroup, QComboBox, QGridLayout, QPushButton, QRadioButton, QWidget

from . import theme as T
from .common import hbox, label
from .stackui import CoverageDiagram, MethodPicker, StackPage, honest_card, qimage_rgb8, scene_to_qimage

VIEWS = ("Result", "Best single frame", "Rejected (left out)", "Overview (click to pick the place)")


def crop_box(an, window) -> tuple:
    """Crop at 2x: a third of the frame each way, around the chosen place (or the sharpest one)."""
    H, W = an.shape_full
    if window is not None:
        cy, cx = (window[0] + window[1]) / 2.0, (window[2] + window[3]) / 2.0
    else:
        cy, cx = an.sharp_centre
    h, w = H // 3, W // 3
    y0 = int(np.clip(cy - h / 2, 0, H - h)) // 2 * 2
    x0 = int(np.clip(cx - w / 2, 0, W - w)) // 2 * 2
    return (y0, y0 + h, x0, x0 + w)


class SuperResPage(StackPage):
    SETTINGS_KEY = "superres"

    def __init__(self, settings=None):
        from ..engine.stack import superres as SR
        self.SR = SR
        self.analysis = None
        self.window = None                           # preview window (input pixels of the reference frame)
        self.last = None
        super().__init__(
            "Super-Res Stack",
            "Lines up a burst and merges it: a larger photo with real extra detail from a hand-held burst, a "
            "cleaner one from any burst, the moon, or the night sky.",
            "Give it frames: select them in the Library, right-click, Send to Super-Res Stack; or drop files or a "
            "folder here.",
            " ".join(f"{SR.MODE_INFO[m]['name']}: {SR.WHAT_TO_SHOOT[m]}" for m in ("handheld", "moon")), settings)
        self.result.view.clicked.connect(self._clicked)

    # ------------------------------------------------------------------------------------ panel
    def build_panel(self):
        SR = self.SR
        g = self.group("Mode")
        grid = QWidget()
        gl = QGridLayout(grid)
        gl.setContentsMargins(0, 0, 0, 0)
        self.mode_group = QButtonGroup(self)
        for i, m in enumerate(SR.MODES):
            b = QRadioButton(SR.MODE_INFO[m]["name"])
            b.setToolTip(SR.MODE_INFO[m]["does"] + "\nBest for: " + SR.MODE_INFO[m]["best"] + "\nWeak at: " +
                         SR.MODE_INFO[m]["weak"])
            b.setProperty("mode", m)
            self.mode_group.addButton(b, i)
            gl.addWidget(b, i // 2, i % 2)
        self.mode_group.button(0).setChecked(True)
        g.add(grid)
        self.mode_text = g.add(label("", "dim", wrap=True))
        self.mode_group.idToggled.connect(lambda i, on: on and self._mode_changed())
        g = self.group("Merge")
        self.merge = g.add(MethodPicker(SR.MERGE_INFO, list(SR.MERGES_FOR["handheld"])))
        self.merge.changed.connect(lambda *_: self._options_changed())
        self.scale = g.add(QComboBox())
        for text, key in (("Whole frame at 2x", "2"), ("Whole frame at 1.5x", "1.5"), ("Crop at 2x (around the "
                                                                                       "chosen place)", "crop")):
            self.scale.addItem(text, key)
        self.scale.currentIndexChanged.connect(lambda *_: self._options_changed())
        self.size_line = g.add(label("", "dim", wrap=True))
        g = self.group("Options")
        self.keep = g.add(self.slider("Keep the sharpest", 0.05, 1.0, 1.0, 0.05, "{:.0%}",
                                      "Share of the sharp-enough frames used (the moon: the sharpest quarter)."))
        self.reject = g.add(self.slider("Leave out what moved", 1.5, 6.0, 3.0, 0.1, "{:.1f}",
                                        "How different from the reference a place may be and still count (in noise "
                                        "steps). Lower leaves out more."))
        self.pixfrac = g.add(self.slider("Drop size", 0.4, 1.0, 0.7, 0.05, "{:.2f}",
                                         "Drizzle: how big each photosite's drop is, in input pixels. Smaller is "
                                         "sharper but needs more frames."))
        self.sharpen = g.add(self.slider("Finish sharpening", 0.0, 2.0, 0.3, 0.05, "{:.2f}",
                                         "Hand-held: a light lift of the finest detail (drizzle is slightly soft). "
                                         "Moon: starlet sharpening. Noise is left alone."))
        self.local = g.add(self.tick("Follow local movement", True, "Hand shake, rolling shutter and air wobble, "
                                     "measured place by place after the overall line-up."))
        for w in (self.keep, self.reject, self.pixfrac, self.sharpen):
            w.changed.connect(lambda *_: self._options_changed())
        g = self.group("Sub-pixel positions", "Each dot is a frame. Spread out: there is detail to gain. One clump: "
                       "the camera did not move (that still cleans noise).")
        self.coverage = g.add(CoverageDiagram())
        g = self.group("View")
        self.view_pick = g.add(QComboBox())
        self.view_pick.addItems(VIEWS)
        self.view_pick.currentIndexChanged.connect(self._show_view)
        self.report = g.add(label("", "dim", wrap=True))
        self.panel.addWidget(honest_card(SR.BURST_ADVICE, "Burst exposure", T.CYAN))
        self.panel.addWidget(honest_card(SR.HONEST))
        self.panel.addStretch(1)
        self.add_actions()
        self._mode_changed()

    def mode(self) -> str:
        b = self.mode_group.checkedButton()
        return b.property("mode") if b is not None else "handheld"

    def _mode_changed(self):
        SR = self.SR
        m = self.mode()
        d = SR.SuperParams.for_mode(m)
        self.mode_text.setText(f"{SR.MODE_INFO[m]['does']} {SR.WHAT_TO_SHOOT[m]}")
        self.merge.set_keys(list(SR.MERGES_FOR[m]), d.merge)
        for w, v in ((self.keep, d.keep_best), (self.reject, d.reject), (self.sharpen, d.sharpen)):
            w.set_value(v, emit=False)
        self.local.setChecked(d.local)
        self.scale.setCurrentIndex(0 if d.scale >= 2 else (1 if d.scale > 1 else 0))
        self.scale.setEnabled(d.merge == "drizzle")
        if self.analysis is not None and self.analysis.params.mode != m:
            self._drop_analysis()
        self._options_changed()

    def params(self):
        SR = self.SR
        p = SR.SuperParams.for_mode(self.mode())
        p.merge = self.merge.key() or p.merge
        sc = self.scale.currentData()
        p.scale = 1.5 if sc == "1.5" else (2.0 if p.merge == "drizzle" else 1.0)
        if p.merge != "drizzle":
            p.scale = 1.0
        p.keep_best = self.keep.value()
        p.reject = self.reject.value()
        p.pixfrac = self.pixfrac.value()
        p.sharpen = self.sharpen.value()
        p.local = self.local.isChecked()
        if sc == "crop" and self.analysis is not None:
            p.crop = self._crop_box()
        return p

    def _crop_box(self):
        return crop_box(self.analysis, self.window)

    def _options_changed(self):
        SR = self.SR
        p = self.params()
        self.scale.setEnabled(p.merge == "drizzle")
        if self.analysis is not None:
            info = SR.output_info(self.analysis, p)
            warn = SR.frame_warnings(self.analysis, p)
            self.size_line.setText(SR.describe_output(info) + ("\n" + " ".join(warn) if warn else ""))
            self.size_line.setStyleSheet(f"color:{T.AMBER if warn else T.DIM};")
        else:
            self.size_line.setText("The size and time are shown after Preview.")

    # ------------------------------------------------------------------------------------ frames
    def frames_changed(self):
        if self.analysis is not None and self.analysis.paths != self.frame_paths():
            self._drop_analysis()

    def _drop_analysis(self):
        if self.analysis is not None:
            try:
                self.analysis.close()
            except Exception:                        # noqa: BLE001
                pass
        self.analysis = None
        self.window = None

    def _needs_analysis(self, p) -> bool:
        a = self.analysis
        return a is None or a.params.analysis_key() != p.analysis_key() or a.paths != self.frame_paths()

    # ------------------------------------------------------------------------------------ preview / build
    def preview(self):
        SR = self.SR
        p = self.params()
        paths = self.frame_paths()
        if len(paths) < 2:
            return
        need = self._needs_analysis(p)
        old = None if need else self.analysis
        if need:
            self._drop_analysis()
        win = self.window

        def work(progress, cancel):
            an = old if old is not None else SR.analyse(paths, p, lambda f, m: progress(0.6 * f, m), cancel)
            try:
                res, ref, extra = SR.preview(an, p, win, lambda f, m: progress(0.6 + 0.4 * f, m), cancel)
            except BaseException:
                if an is not old:
                    an.close()
                raise
            return an, res, ref, extra

        def done(r):
            an, res, ref, extra = r
            self.analysis = an
            self.last = (res, ref, extra)
            self._show_analysis()
            self._options_changed()
            self._show_view()
            self.jobbar.say("Preview ready: a 100% window of the result, built the real way. B shows the best "
                            "single frame enlarged the same way.", T.CYAN)
            self._refresh()

        if self.jobbar.start(work, done, lambda e, c: self.failed("Preview", e, c), text="Previewing..."):
            self._refresh()

    def _show_analysis(self):
        an = self.analysis
        kept = set(an.kept(self.params()))
        for i, r in enumerate(an.reports):
            if r.status == "dropped" or (i < an.n and i not in kept):
                why = r.reason or "not among the sharpest kept"
                self.tray.set_status(r.path, f"#{r.rank} left out" if r.rank else "left out", "dropped", why)
                continue
            off = f"{r.offset[0]:+.1f}, {r.offset[1]:+.1f} px"
            ref = " · reference" if i == an.ref else ""
            self.tray.set_status(r.path, f"#{r.rank}{ref} · {100 * r.rejected:.0f}% out", r.status,
                                 (r.reason + "\n" if r.reason else "") +
                                 f"Sharpness rank {r.rank}; moved {off}; {100 * r.rejected:.0f}% given little "
                                 f"weight (it differed from the reference there)")
        self.coverage.set_dots(an.frac, an.moved)
        self.report.setText(" ".join(an.notes) or f"{len(kept)} of {an.n} frames used; reference "
                                                   f"{os.path.basename(an.paths[an.ref])} (the sharpest).")
        self.set_chips([("moved: detail to gain", T.CYAN) if an.moved else ("did not move", T.AMBER)])

    def _show_view(self, *_):
        if self.analysis is None:
            return
        name = self.view_pick.currentText()
        an = self.analysis
        self.result.view.click_zooms = not name.startswith("Overview")
        if name.startswith("Overview"):
            from ..engine.pipeline.pipeline import apply_orientation
            L = an.ref_ana
            g = np.clip(np.sqrt(np.clip(L / max(float(np.percentile(L, 99.5)), 1e-6), 0, 1)) * 255, 0, 255)
            img = np.repeat(g.astype(np.uint8)[..., None], 3, 2)
            y0, y1, x0, x1 = self.window or self.SR.default_window(an, self.params())
            K = an.k_ana
            img[max(0, y0 // K):y1 // K, max(0, x0 // K):max(0, x0 // K) + 2] = (34, 211, 238)
            img[max(0, y0 // K):y1 // K, max(0, x1 // K - 2):x1 // K] = (34, 211, 238)
            img[max(0, y0 // K):max(0, y0 // K) + 2, x0 // K:x1 // K] = (34, 211, 238)
            img[max(0, y1 // K - 2):y1 // K, x0 // K:x1 // K] = (34, 211, 238)
            self.result.set_images(qimage_rgb8(np.ascontiguousarray(apply_orientation(img, an.flip))), None,
                                   chip_text="Overview · click to choose the place to preview", chip_colour=T.VIOLET)
            return
        if self.last is None:
            return
        res, ref, extra = self.last
        if name == "Result":
            self.result.set_images(scene_to_qimage(res), scene_to_qimage(ref), chip_text="Super-res · 100% window",
                                   chip_colour=T.CYAN)
        elif name == "Best single frame":
            self.result.set_images(scene_to_qimage(ref), scene_to_qimage(res), chip_text="Best single frame",
                                   chip_colour=T.AMBER)
        else:
            from ..engine.pipeline.pipeline import apply_orientation
            rv = extra.get("rejected") if isinstance(extra, dict) else None
            if rv is None:
                rv = an.rejected_view(self.params())
            rv = np.clip(np.asarray(rv, np.float32), 0, 1)
            img = np.stack([rv * 245, rv * 158, rv * 11], 2).astype(np.uint8)       # amber on black
            if rv.shape == an.ref_ana.shape:
                img = apply_orientation(img, an.flip)
            self.result.set_images(qimage_rgb8(np.ascontiguousarray(img)), None,
                                   chip_text="Left out (amber): moved, or did not line up", chip_colour=T.AMBER)

    def _clicked(self):
        """On the overview, a click chooses the place the preview (and Crop at 2x) are centred on."""
        if self.analysis is None or not self.view_pick.currentText().startswith("Overview"):
            return
        an = self.analysis
        v = self.result.view
        if v.zoom:
            return
        from PySide6.QtGui import QCursor
        pos = v.mapFromGlobal(QCursor.pos())
        u, w_ = v.uv_at(pos)
        u, w_ = min(max(u, 0.0), 1.0), min(max(w_, 0.0), 1.0)
        H, W = an.shape_full
        # the overview is upright: back to the frame's own (sensor) orientation
        if an.flip == 3:
            u, w_ = 1 - u, 1 - w_
        elif an.flip == 6:
            u, w_ = w_, 1 - u
        elif an.flip == 5:
            u, w_ = 1 - w_, u
        cy, cx = w_ * H, u * W
        y0, y1, x0, x1 = self.SR.default_window(an, self.params())
        hh, hw = (y1 - y0) // 2, (x1 - x0) // 2
        y0 = int(np.clip(cy - hh, 0, H - 2 * hh)) // 2 * 2
        x0 = int(np.clip(cx - hw, 0, W - 2 * hw)) // 2 * 2
        self.window = (y0, y0 + 2 * hh, x0, x0 + 2 * hw)
        self._show_view()
        self.jobbar.say("Place chosen. Press Preview to build that window.", T.CYAN)

    def build_full(self):
        SR = self.SR
        p = self.params()
        paths = self.frame_paths()
        if len(paths) < 2 or self.jobbar.busy:
            return
        from ..engine.stack.frames import result_name
        need = self._needs_analysis(p)
        old = None if need else self.analysis
        if need:
            self._drop_analysis()
        want_crop = self.scale.currentData() == "crop"
        win = self.window

        def work(progress, cancel):
            an = old if old is not None else SR.analyse(paths, p, lambda f, m: progress(0.3 * f, m), cancel)
            try:
                pp = SR.SuperParams.from_dict(p.to_dict())
                pp.crop = crop_box(an, win) if want_crop else None
                kept = an.kept(pp)
                name = result_name([an.paths[kept[0]], an.paths[kept[-1]]], SR.result_tag(pp, an.raw))
                out = self.target(name, an.paths[kept[0]])
                path = SR.build(an, pp, out, lambda f, m: progress(0.3 + 0.7 * f, m), cancel)
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

        if self.jobbar.start(work, done, lambda e, c: self.failed("Build", e, c), keep=True,
                             text="Building..."):
            self._refresh()
