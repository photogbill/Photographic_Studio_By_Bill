"""The Cameras workspace: teach the Studio your camera by dropping in one big pile of frames.

Bill, 2026-11-18: "I won't likely follow the wizard, but most people won't either. So it should be able to handle
a huge drop of raw photos at different ISO and temp. The key thing is the lens cap on." So there are no steps:
drop folders or files anywhere on the page (or Add a folder), every frame is measured one at a time and sorted by
itself, each camera in the pile gets its profile, and a second drop on another day adds to the first. The page
then shows what was measured (cyan), what is thin (amber) and what would make it better.
"""
from __future__ import annotations

import os
import time

import numpy as np
from PySide6.QtCore import QRectF, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from .. import log as slog, paths
from . import theme as T
from .common import card, chip, label, qalpha, report_problem, run_bg

log = slog.get("app.cameras")

HOW = ("<b>Lens cap on.</b> Shoot every ISO you use at your fastest shutter speed, at least twice each. Then drop "
       "the whole card or folder here: nothing needs sorting or renaming, and photos that are not calibration "
       "frames are left out by themselves.<br>"
       "<span style='color:{dim}'>Better still: do it once with the camera cool and once warm from use, and add a "
       "few long ones (1 s, 8 s, 30 s). Add frames on another day and they join the ones already measured."
       "</span>").format(dim=T.DIM)


def _temp_colour(i: int, n: int) -> QColor:
    """Cool to warm along the measured temperatures (a line per temperature in the chart)."""
    a, b = QColor(T.CYAN), QColor("#fb7185")
    f = 0.0 if n <= 1 else i / (n - 1)
    return QColor(int(a.red() + (b.red() - a.red()) * f), int(a.green() + (b.green() - a.green()) * f),
                  int(a.blue() + (b.blue() - a.blue()) * f))


class CoverageGrid(QWidget):
    """ISO down the side, camera temperature along the top, the number of dark frames in each cell.
    Cyan: two or more (noise can be told from fixed pattern). Amber: one. Empty: none."""
    CELL_H, CELL_W, LEFT, TOP = 22, 46, 64, 22

    def __init__(self):
        super().__init__()
        self.isos, self.temps, self.cells = [], [], {}
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_coverage(self, cov: dict):
        self.cells = cov.get("cells", {}) or {}
        self.isos = sorted(self.cells, key=float)
        self.temps = [f"{t:g}" for t in cov.get("temps", [])]
        self.setFixedSize(self.LEFT + self.CELL_W * max(len(self.temps), 1) + 8,
                          self.TOP + self.CELL_H * max(len(self.isos), 1) + 6)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = QFont(self.font())
        f.setPointSizeF(8.5)
        p.setFont(f)
        p.setPen(QColor(T.DIM))
        for j, t in enumerate(self.temps):
            p.drawText(QRectF(self.LEFT + j * self.CELL_W, 0, self.CELL_W, self.TOP),
                       Qt.AlignmentFlag.AlignCenter, f"{t} °C")
        for i, iso in enumerate(self.isos):
            y = self.TOP + i * self.CELL_H
            p.setPen(QColor(T.DIM))
            p.drawText(QRectF(0, y, self.LEFT - 8, self.CELL_H),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"ISO {iso}")
            for j, t in enumerate(self.temps):
                n = int(self.cells.get(iso, {}).get(t, 0))
                r = QRectF(self.LEFT + j * self.CELL_W + 2, y + 2, self.CELL_W - 4, self.CELL_H - 4)
                col = T.CYAN if n >= 2 else T.AMBER if n == 1 else T.LINE
                p.setPen(QPen(qalpha(col, 0.9 if n else 0.6), 1))
                p.setBrush(qalpha(col, min(0.16 + 0.05 * n, 0.5)) if n else Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(r, 5, 5)
                if n:
                    p.setPen(QColor(T.TEXT))
                    p.drawText(r, Qt.AlignmentFlag.AlignCenter, str(n))
        p.end()


class NoiseChart(QWidget):
    """Read noise against ISO: one line per measured camera temperature (cool cyan to warm rose)."""

    def __init__(self):
        super().__init__()
        self.lines = []                                   # [(name, colour, [(iso, dn)])]
        self.setMinimumSize(340, 190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(200)

    def set_profile(self, prof):
        temps = sorted({t["temp_c"] for e in prof.isos for t in e.by_temp if t.get("temp_c") is not None})
        self.lines = []
        if len(temps) >= 2:
            for k, tb in enumerate(temps):
                pts = [(e.iso, float(np.mean(t["read_noise_dn"]))) for e in prof.isos for t in e.by_temp
                       if t.get("temp_c") == tb]
                if pts:
                    self.lines.append((f"{tb:g} °C", _temp_colour(k, len(temps)), sorted(pts)))
        else:
            pts = [(e.iso, float(np.mean(e.read_noise_dn))) for e in prof.isos if e.read_noise_dn]
            self.lines.append(("read noise", QColor(T.CYAN), sorted(pts)))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        L, R, Tm, Bm = 46, 26, 30, 24
        p.setPen(QPen(QColor(T.LINE), 1))
        p.setBrush(QColor(T.BG2))
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 10, 10)
        pts = [pt for _, _, ln in self.lines for pt in ln if pt[1] > 0]
        if not pts:
            p.end()
            return
        x0, x1 = np.log2(min(i for i, _ in pts)), np.log2(max(i for i, _ in pts))
        if x1 - x0 < 1e-6:
            x0, x1 = x0 - 1, x1 + 1
        # noise on a log scale: it grows many times over from base ISO to the top, and the gap between a cool
        # and a warm camera must be readable at both ends
        y0, y1 = np.log2(min(d for _, d in pts) * 0.8), np.log2(max(d for _, d in pts) * 1.25)

        def X(iso):
            return L + (np.log2(iso) - x0) / (x1 - x0) * (w - L - R)

        def Y(dn):
            return h - Bm - (np.log2(max(dn, 1e-6)) - y0) / (y1 - y0) * (h - Tm - Bm)
        f = QFont(self.font())
        f.setPointSizeF(8)
        p.setFont(f)
        for k in range(5):
            v = 2.0 ** (y0 + (y1 - y0) * k / 4)
            p.setPen(QPen(qalpha(T.LINE, 0.7), 1))
            p.drawLine(int(L), int(Y(v)), int(w - R), int(Y(v)))
            p.setPen(QColor(T.DIM))
            p.drawText(QRectF(0, Y(v) - 8, L - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{v:.1f}" if v < 100 else f"{v:.0f}")
        last = -1e9
        for iso in sorted({i for i, _ in pts}):
            if X(iso) - last >= 48:
                p.drawText(QRectF(X(iso) - 30, h - Bm + 4, 60, 16), Qt.AlignmentFlag.AlignCenter, f"{iso:g}")
                last = X(iso)
        for k, (name, col, ln) in enumerate(self.lines):
            path = QPainterPath()
            for n, (iso, dn) in enumerate(ln):
                (path.moveTo if n == 0 else path.lineTo)(X(iso), Y(dn))
            p.setPen(QPen(col, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            p.setBrush(col)
            for iso, dn in ln:
                p.drawEllipse(QRectF(X(iso) - 2.5, Y(dn) - 2.5, 5, 5))
            if k < 7:
                p.setPen(col)
                p.drawText(QRectF(L + 6 + 62 * k, 8, 60, 14), Qt.AlignmentFlag.AlignLeft, name)
        p.setPen(QColor(T.DIM))
        p.drawText(QRectF(w - 190, 8, 180, 14), Qt.AlignmentFlag.AlignRight, "read noise (DN) by ISO")
        p.end()


class CameraCard(QFrame):
    start_over = Signal(str, str)

    def __init__(self, prof, fresh: bool = False):
        super().__init__()
        self.setObjectName("card")
        self.prof = prof
        q = prof.quality or {}
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        head = QHBoxLayout()
        head.addWidget(label(prof.model or "Camera", "h2"))
        head.addSpacing(8)
        head.addWidget(chip("measured on this PC", T.CYAN))
        head.addWidget(chip("Full: noise and gain" if prof.method == "full" else "Quick: dark frames", T.CYAN))
        score = q.get("score")
        if score is not None:
            head.addWidget(chip(f"quality {score:g} / 100", T.GREEN if score >= 70 else T.AMBER))
        if fresh:
            head.addWidget(chip("just updated", T.GREEN))
        head.addStretch(1)
        b = QPushButton("Start this camera over")
        b.setToolTip("Forgets the frames measured for this camera and its profile. Your raw files are not touched.")
        b.clicked.connect(lambda: self.start_over.emit(prof.make, prof.model))
        head.addWidget(b)
        v.addLayout(head)
        temps = ""
        if q.get("temp_min_c") is not None:
            temps = (f", camera at {q['temp_min_c']:g} °C" if q["temp_min_c"] == q["temp_max_c"]
                     else f", camera from {q['temp_min_c']:g} to {q['temp_max_c']:g} °C")
        hot = max((e.hot_pixels for e in prof.isos), default=0)
        hot_txt = (f"Up to {hot:,} hot pixels found (their map stays on this PC)." if hot
                   else "No hot pixels found in the short frames.")
        self.summary = label(f"{q.get('dark_frames', 0)} dark frames and {q.get('flat_frames', 0)} flat frames "
                             f"at {len(prof.isos)} ISO settings{temps}. {hot_txt}", "dim", wrap=True)
        v.addWidget(self.summary)
        v.addSpacing(8)
        row = QHBoxLayout()
        self.grid = CoverageGrid()
        self.grid.set_coverage(q.get("coverage") or {})
        left = QVBoxLayout()
        left.addWidget(label("Dark frames by ISO and camera temperature", "dim"))
        if (q.get("coverage") or {}).get("temps"):
            left.addWidget(self.grid)
        else:
            left.addWidget(label("These files do not record the camera's temperature.", "dim", wrap=True))
        left.addStretch(1)
        row.addLayout(left)
        row.addSpacing(18)
        self.chart = NoiseChart()
        self.chart.set_profile(prof)
        row.addWidget(self.chart, 1)
        v.addLayout(row)
        heat = self._heat_text()
        if heat:
            v.addWidget(label(heat, None, wrap=True))
        self.advice = list(q.get("advice") or [])
        if self.advice:
            v.addSpacing(6)
            v.addWidget(label("What would make it better", "dim"))
            for a in self.advice:
                v.addWidget(label(f"<span style='color:{T.AMBER}'>●</span> {a}", None, wrap=True))
        else:
            v.addWidget(label(f"<span style='color:{T.GREEN}'>●</span> Nothing is missing.", None))

    def _heat_text(self) -> str:
        ths = [(e.iso, e.thermal) for e in self.prof.isos if e.thermal]
        if not ths:
            return ""
        dbl = [t["doubling_c"] for _, t in ths if t.get("doubling_c")]
        txt = (f"<span style='color:{T.CYAN}'>●</span> Long exposures measured at {len(ths)} ISO setting(s): the "
               f"Studio now knows how much noise a long exposure adds on this camera")
        if dbl:
            txt += f", and that it doubles about every {float(np.median(dbl)):.0f} °C the camera warms"
        return txt + "."


class CamerasPage(QWidget):
    status = Signal(str)
    _progress = Signal(int, int, object)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.busy = False
        self.pending: list = []
        self._cancel = False
        self._t0 = 0.0
        self._measured = 0
        self.cards: list = []
        self.last = None                                  # the last DropResult
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("infopage")
        scroll.setWidget(content)
        outer.addWidget(scroll)
        v = QVBoxLayout(content)
        v.setContentsMargins(40, 34, 40, 34)
        head = QHBoxLayout()
        head.addWidget(label("Cameras", "h1"))
        head.addSpacing(12)
        head.addWidget(chip("Drop everything in one pile", T.CYAN))
        head.addStretch(1)
        v.addLayout(head)
        v.addWidget(label("Teach the Studio your camera. A profile is numbers only: noise, banding, black level and "
                          "hot-pixel counts per ISO and camera temperature. It stays on this PC unless you choose "
                          "to share it.", "dim", wrap=True))
        v.addSpacing(14)
        # ------------------------------------------------------------------------------------ the drop zone
        self.zone = QFrame()
        self.zone.setObjectName("dropzone")
        self._zone_style(False)
        z = QVBoxLayout(self.zone)
        z.setContentsMargins(26, 22, 26, 22)
        z.addWidget(label("Drop your calibration frames here", "h2"), 0, Qt.AlignmentFlag.AlignHCenter)
        how = label(HOW, None, wrap=True)
        how.setTextFormat(Qt.TextFormat.RichText)
        how.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        z.addWidget(how)
        z.addSpacing(8)
        row = QHBoxLayout()
        row.addStretch(1)
        self.add_folder_b = QPushButton("Add a folder or a card...")
        self.add_folder_b.setObjectName("primary")
        self.add_folder_b.clicked.connect(self.choose_folder)
        self.add_files_b = QPushButton("Add files...")
        self.add_files_b.clicked.connect(self.choose_files)
        self.cancel_b = QPushButton("Stop")
        self.cancel_b.setToolTip("Stops after the frames being measured now. What is measured is kept.")
        self.cancel_b.clicked.connect(self.cancel)
        self.cancel_b.hide()
        self.open_b = QPushButton("Open the calibration folder")
        self.open_b.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(paths.get("CALIBRATION_LOCAL")))))
        for b in (self.add_folder_b, self.add_files_b, self.cancel_b, self.open_b):
            row.addWidget(b)
        row.addStretch(1)
        z.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.hide()
        z.addSpacing(6)
        z.addWidget(self.bar)
        self.prog = label("", "dim", wrap=True)
        self.prog.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        z.addWidget(self.prog)
        v.addWidget(self.zone)
        v.addSpacing(10)
        self.out = label("", None, wrap=True)
        self.out.setTextFormat(Qt.TextFormat.RichText)
        self.out.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.out)
        v.addSpacing(8)
        self.cards_box = QVBoxLayout()
        self.cards_box.setSpacing(14)
        v.addLayout(self.cards_box)
        self.shared = label("", "dim", wrap=True)
        self.shared.setTextFormat(Qt.TextFormat.RichText)
        v.addSpacing(8)
        v.addWidget(self.shared)
        v.addStretch(1)
        self._progress.connect(self._on_progress)
        self.refresh()

    # ---------------------------------------------------------------------------------------- drop zone
    def _zone_style(self, hot: bool):
        col = T.CYAN if hot else T.LINE
        self.zone.setStyleSheet(f"QFrame#dropzone {{ border: 2px dashed {col}; border-radius: 16px; "
                                f"background: {T.alpha(T.CYAN, 0.07 if hot else 0.02)}; }}")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            self._zone_style(True)
            e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self._zone_style(False)

    def dropEvent(self, e):
        self._zone_style(False)
        items = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if items:
            self.add(items)
            e.acceptProposedAction()

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder (or card) of calibration frames")
        if d:
            self.add([d])

    def choose_files(self):
        from ..engine.io.rawfile import RAW_EXTENSIONS
        pats = " ".join(f"*{x}" for x in RAW_EXTENSIONS)
        files, _ = QFileDialog.getOpenFileNames(self, "Calibration frames", "", f"Raw files ({pats});;All files (*)")
        if files:
            self.add(files)

    def calibrate(self):                                  # the old button's name (Tools menu, tests)
        self.choose_folder()

    # ---------------------------------------------------------------------------------------- the job
    def add(self, items: list):
        """Folders and files, in any mix. A pile dropped while another is being measured waits its turn."""
        self.pending.append([str(i) for i in items])
        if not self.busy:
            self._next()

    def cancel(self):
        self._cancel = True
        self.pending.clear()
        self.prog.setText("Stopping after the frames being measured now...")

    def _next(self):
        if not self.pending:
            return
        items = self.pending.pop(0)
        self.busy, self._cancel, self._t0, self._measured = True, False, time.time(), 0
        self.bar.setRange(0, 0)
        self.bar.show()
        self.cancel_b.show()
        self.prog.setText("Looking for raw files...")
        log.info("calibration drop: %d item(s)", len(items))

        def work():
            from ..engine.calib.builder import calibrate_drop
            from ..engine.io.rawfile import RAW_EXTENSIONS
            files = []
            for it in items:
                if os.path.isdir(it):
                    for root, dirs, names in os.walk(it):
                        dirs[:] = [d for d in dirs if not d.startswith((".", "$"))]
                        files += [os.path.join(root, n) for n in sorted(names) if n.lower().endswith(RAW_EXTENSIONS)]
                elif os.path.isfile(it) and it.lower().endswith(RAW_EXTENSIONS):
                    files.append(it)
            return calibrate_drop(files, progress=self._emit_progress, cancelled=lambda: self._cancel)
        run_bg(work, "calibrating a camera", self._done, self._failed, keep=True)

    def _emit_progress(self, n, total, st):
        try:
            self._progress.emit(n, total, st)
        except RuntimeError:                              # the window is closing
            pass

    def _on_progress(self, n, total, st):
        self.bar.setRange(0, total)
        self.bar.setValue(n)
        if not st.cached:
            self._measured += 1
        what = {"dark": "dark frame", "flat": "flat frame"}.get(st.kind, "left out")
        bits = [what] + ([f"ISO {st.iso:g}"] if st.iso else []) + ([f"{st.temp_c:g} °C"] if st.temp_c is not None
                                                                   else [])
        left = ""
        if self._measured >= 3 and n < total:
            secs = (time.time() - self._t0) / max(n, 1) * (total - n)
            left = f"  ·  about {max(1, round(secs / 60))} min left" if secs >= 50 else "  ·  under a minute left"
        self.prog.setText(f"{n} of {total}  ·  {st.name}: {', '.join(bits)}{left}")

    def _failed(self, _, err):
        self._finish()
        self.out.setText(f"<span style='color:{T.RED}'>●</span> Calibration stopped on a problem: {err}. An error "
                         f"report was saved (Help has a button that opens the folder).")
        self._next()

    def _finish(self):
        self.busy = False
        self.bar.hide()
        self.cancel_b.hide()
        self.prog.setText("")

    def _done(self, _, res):
        self._finish()
        self.last = res
        try:
            self._show_result(res)
        except Exception as e:                            # the profiles are saved; only the page could not show them
            slog.log_exception("calibration result could not be shown", log)
            report_problem("Showing a calibration result", e)
        self._next()

    def _show_result(self, res):
        c = res.counts()
        took = time.time() - self._t0
        lines = []
        if not res.scanned:
            lines.append(f"<span style='color:{T.AMBER}'>●</span> No raw files were found in what you dropped.")
        else:
            lines.append(f"<b>This pile:</b> <span style='color:{T.CYAN}'>{c.get('dark', 0)} dark frames</span>, "
                         f"{c.get('flat', 0)} flat frames, {len(res.left_out)} left out"
                         f"{' (stopped early)' if res.cancelled else ''}, in {took / 60:.1f} min.")
        if res.scanned and not res.built:
            lines.append(f"<span style='color:{T.AMBER}'>●</span> No dark frames among them, so no profile was "
                         f"made. A dark frame is shot with the lens cap on.")
        left = res.left_out
        if left:
            by = {}
            for f in left:
                by.setdefault(f.reason or "not a calibration frame", []).append(f.name)
            for reason, names in sorted(by.items(), key=lambda kv: -len(kv[1])):
                shown = ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")
                lines.append(f"<span style='color:{T.DIM}'>Left out ({len(names)}): {reason}<br>"
                             f"&nbsp;&nbsp;&nbsp;{shown}</span>")
        notes = [w for _, rep, _ in res.built for w in rep.warnings]
        for w in notes[:12]:
            lines.append(f"<span style='color:{T.AMBER}'>note:</span> {w}")
        if len(notes) > 12:
            lines.append(f"<span style='color:{T.DIM}'>... and {len(notes) - 12} more notes (logs\\studio.log has "
                         f"them all).</span>")
        self.out.setText("<br>".join(lines))
        for w in notes:
            log.info("calibration note: %s", w)
        fresh = {(p.make, p.model) for p, _, _ in res.built}
        self.refresh(fresh)
        if res.built:
            self.status.emit("Camera profile saved: " + ", ".join(p.model for p, _, _ in res.built)
                             + ". The denoiser uses it from the next photo you open.")

    # ---------------------------------------------------------------------------------------- what exists
    def refresh(self, fresh=frozenset()):
        import glob
        import json
        from ..engine.calib.profile import CameraProfile
        while self.cards:
            w = self.cards.pop()
            self.cards_box.removeWidget(w)
            w.deleteLater()
        for f in sorted(glob.glob(str(paths.get("CALIBRATION_LOCAL") / "profiles" / "*.json"))):
            try:
                prof = CameraProfile.load(f, "local")
            except Exception:
                slog.log_exception(f"camera profile unreadable: {f}", log)
                continue
            cardw = CameraCard(prof, (prof.make, prof.model) in fresh)
            cardw.start_over.connect(self.start_over)
            self.cards_box.addWidget(cardw)
            self.cards.append(cardw)
        rows = []
        for f in sorted(glob.glob(str(paths.PROFILES / "cameras" / "*.json"))):
            try:
                p = json.load(open(f, encoding="utf-8"))
                rows.append(f"<span style='color:{T.VIOLET}'>●</span> <b>{p.get('model')}</b> - {p.get('method')} "
                            f"profile, {len(p.get('isos', []))} ISOs (shared)")
            except Exception:
                pass
        if not self.cards and not rows:
            rows.append(f"No camera profiles yet. Until there is one, the noise is estimated from each photo "
                        f"(<span style='color:{T.AMBER}'>amber</span>).")
        self.shared.setText("<br>".join(rows))

    def start_over(self, make: str, model: str, ask: bool = True):
        if self.busy:
            self.status.emit("Wait until the frames being measured are done.")
            return
        if ask and QMessageBox.question(
                self, "Start this camera over",
                f"Forget everything measured for the {model}? Its profile and the measurements of its calibration "
                f"frames are removed from this PC. Your raw files are not touched.") != QMessageBox.StandardButton.Yes:
            return
        from ..engine.calib import frames as F
        from ..engine.calib import profile as P
        n = F.forget_camera(make, model)
        try:
            os.remove(P.local_profile_path(make, model))
        except OSError:
            pass
        P._CACHE.clear()
        log.info("calibration: %s %s started over (%d measured frames forgotten)", make, model, n)
        self.out.setText(f"The {model} was started over ({n} measured frames forgotten).")
        self.refresh()
