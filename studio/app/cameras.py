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
from .common import Settings, card, chip, label, qalpha, report_problem, run_bg

log = slog.get("app.cameras")

HOW = ("Drop the whole card or folder here. Nothing needs sorting or renaming, ordinary photos in the pile are left "
       "out by themselves, and frames you add on another day join the ones already measured.")

# What to shoot, in the order it matters (Bill, 2026-11-18: "put a small list and description of what is needed at
# the top. Explain what a flat pair is"). (title, tag, tag colour, text)
NEEDED = [
    ("1. Dark frames", "needed", T.CYAN,
     "A photo of nothing: <b>lens cap on</b>, fastest shutter speed. Shoot every ISO you use, at least twice each "
     "(two frames let the Studio tell noise from the sensor's fixed pattern). Raw, not C-RAW. This alone gives a "
     "Quick profile: read noise, banding and hot pixels."),
    ("2. Long dark frames", "better", T.AMBER,
     "Cap still on, a few frames at about 1 s, 8 s and 30 s at ISOs you use for long exposures. They show the "
     "noise heat adds. Best of all: some with the camera cool and some with it warm from use."),
    ("3. Flat pairs", "for a Full profile", T.VIOLET,
     "A flat is a photo of an evenly lit blank: a white wall, paper, or a white T-shirt over the lens toward the "
     "sky, out of focus, bright but not blown out. A <b>pair</b> is two of them in a row with the same ISO and "
     "shutter speed. Dust and lens shading are the same in both and cancel; what is left is pure noise, which "
     "tells the Studio how noise grows with light (the camera's gain). One pair per ISO; daylight is steadier "
     "than lamps or screens."),
]


def _isos(n: int) -> str:
    return f"{n} ISO setting" + ("" if n == 1 else "s")


def shoot_status(prof) -> list:
    """For one camera's profile: how each of the three things to shoot stands. [(colour, text)] in NEEDED's order:
    green = covered, amber = partly, dim = not yet."""
    q = prof.quality or {}
    n = len(prof.isos)
    short = (q.get("coverage") or {}).get("short") or {}
    two = sum(1 for e in prof.isos if short.get(f"{e.iso:g}", 0) >= 2)
    if n and two == n:
        darks = (T.GREEN, f"✓ {_isos(n)}, two or more frames each")
    else:
        darks = (T.AMBER, f"{two} of {_isos(n)} have two short frames")
    heat = [e.thermal for e in prof.isos if e.thermal]
    if not heat:
        long_ = (T.DIM, "not yet")
    elif any(t.get("doubling_c") for t in heat):
        long_ = (T.GREEN, f"✓ measured at {_isos(len(heat))}, cool and warm")
    else:
        long_ = (T.AMBER, f"measured at {_isos(len(heat))}, all at one temperature: add some with the "
                          f"camera cold or warm")
    g = sum(1 for e in prof.isos if e.gain_e_per_dn)
    if n and g == n:
        flats = (T.GREEN, f"✓ gain measured at all {_isos(n)}")
    elif g:
        flats = (T.AMBER, f"gain measured at {g} of {_isos(n)}")
    else:
        flats = (T.DIM, "not yet")
    return [darks, long_, flats]


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


def teach_status(prof, clean: dict | None = None) -> list:
    """What stands between this camera and its own Clean network, in order. [(colour, text)]."""
    from .. import lab
    from ..engine.calib import deep as D
    from ..engine.raw import learned as LN
    out = []
    n = len(prof.isos)
    g = sum(1 for e in prof.isos if e.gain_e_per_dn)
    out.append((T.GREEN, f"✓ Full profile: gain at {g} of {_isos(n)}") if g else
               (T.AMBER, "Needs flat pairs first (card 3 at the top): a Quick profile has no gain to make photon "
                         "noise with"))
    bank = D.find_bank(prof.make, prof.model)
    if bank is None:
        out.append((T.AMBER, "Noise bank: not built yet. Press Deep calibration and give it the folder of this "
                             "camera's calibration raws"))
    else:
        out.append((T.GREEN, f"✓ Noise bank: {bank.meta.get('patches', 0):,} patches of real dark frame at "
                             f"{_isos(len(bank.isos()))}"))
    if clean and clean.get("folder"):
        tier, text = lab.photos_verdict(int(clean.get("usable", 0)))
        col = {"good": T.GREEN, "thin": T.AMBER, "few": T.RED}[tier]
        spread = (f" (from {clean.get('sessions', 0)} different hours on {clean.get('days', 0)} days)"
                  if clean.get("usable") else "")
        out.append((col, ("✓ " if tier == "good" else "") + text + spread))
    else:
        out.append((T.DIM, f"Clean photos: none chosen yet. Around {lab.PHOTOS_WANTED} varied ones is the aim"))
    st = LN.status(prof.make, prof.model)
    if st["model"]:
        c = st["card"]
        score = (f": noise left {100 * c['noise_left']:.0f}%, texture kept {100 * c['texture_kept']:.0f}% on "
                 f"photos it never saw" if "noise_left" in c else "")
        out.append((T.GREEN if st["runtime"] else T.AMBER,
                    f"✓ Trained {c.get('made', '')} on {c.get('clean_photos', '?')} photos{score}"
                    + ("" if st["runtime"] else ". The runtime that runs it is not installed: Train Camera "
                                               "Model.bat installs it")))
    else:
        out.append((T.DIM, "No network trained for this camera yet"))
    return out


class CameraCard(QFrame):
    start_over = Signal(str, str)
    deep = Signal(object)
    choose_clean = Signal()
    train = Signal(object)

    def __init__(self, prof, fresh: bool = False, clean: dict | None = None):
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
        warm = max((e.warm_pixels for e in prof.isos), default=0)
        if hot or warm:
            hot_txt = (f"{hot:,} hot pixels (a bright dot in any photo) and up to {warm:,} warm ones (they only "
                       f"stand out of the noise at some ISOs) found; their map stays on this PC.")
        else:
            hot_txt = "No hot pixels found in the short frames."
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
        self._teach(v, clean)

    def _teach(self, v, clean):
        """Clean, taught by this camera: the noise bank, the clean photos (how many, and why), the training."""
        from .. import lab
        v.addSpacing(12)
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background:{T.LINE};")
        v.addWidget(line)
        v.addSpacing(8)
        head = QHBoxLayout()
        head.addWidget(label("Clean, taught by this camera", "h2"))
        head.addSpacing(8)
        head.addWidget(chip("learned on this PC", T.VIOLET))
        head.addStretch(1)
        v.addLayout(head)
        v.addWidget(label("A denoising network trained here, on this camera's real noise and your own clean photos. "
                          "It is optional: Clean works without it. It never invents detail: whatever it removes is "
                          "checked against the measured noise afterwards (the detail guard).", "dim", wrap=True))
        self.teach = teach_status(self.prof, clean)
        for colour, text in self.teach:
            v.addWidget(label(f"<span style='color:{colour}'>●</span> {text}", None, wrap=True))
        v.addSpacing(4)
        self.why = label(f"<b>Clean photos: about {lab.PHOTOS_WANTED}, and varied.</b> {lab.PHOTOS_WHY}", "dim",
                         wrap=True)
        self.why.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self.why)
        v.addSpacing(6)
        row = QHBoxLayout()
        self.deep_b = QPushButton("Deep calibration...")
        self.deep_b.setToolTip("Reads this camera's dark frames once more and keeps patches of their real noise "
                               "(the noise bank) and the part of the banding that is the same in every photo. "
                               "Needs the calibration raws; a few minutes. Stays on this PC.")
        self.deep_b.clicked.connect(lambda: self.deep.emit(self.prof))
        self.clean_b = QPushButton("Choose clean photos...")
        self.clean_b.setToolTip("A folder (with its sub-folders) of sharp, well exposed raws at ISO 400 or lower. "
                                "They are only read.")
        self.clean_b.clicked.connect(self.choose_clean.emit)
        self.train_b = QPushButton("Train this camera's network")
        self.train_b.setObjectName("primary")
        self.train_b.setToolTip("Opens a window that trains overnight on your graphics card. The first time it "
                                "installs the training tools beside the Studio (about 2.5 GB). It can be stopped "
                                "and started again.")
        self.train_b.clicked.connect(lambda: self.train.emit(self.prof))
        for b in (self.deep_b, self.clean_b, self.train_b):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

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
    _plain = Signal(int, int, str)

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
        self.settings = Settings()
        self.clean_info = None                            # the survey of the chosen clean-photos folder
        self.deep_last = None                             # the last deep-calibration reports
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
        # ------------------------------------------------------------------------------------ what to shoot
        v.addWidget(label("What to shoot", "h2"))
        v.addSpacing(6)
        need = QHBoxLayout()
        need.setSpacing(14)
        self.need_cards, self.need_status = [], []
        for title, tag, colour, text in NEEDED:
            c = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(16, 14, 16, 14)
            top = QHBoxLayout()
            top.addWidget(label(title, "h2"))
            top.addSpacing(6)
            top.addWidget(chip(tag, colour))
            top.addStretch(1)
            cl.addLayout(top)
            body = label(text, "dim", wrap=True)
            body.setTextFormat(Qt.TextFormat.RichText)
            cl.addWidget(body)
            cl.addStretch(1)
            st = label("", None, wrap=True)                 # how each camera stands on this one (refresh)
            st.setTextFormat(Qt.TextFormat.RichText)
            st.hide()
            cl.addWidget(st)
            need.addWidget(c, 1)
            self.need_cards.append(c)
            self.need_status.append(st)
        v.addLayout(need)
        v.addSpacing(14)
        # ------------------------------------------------------------------------------------ the drop zone
        self.zone = QFrame()
        self.zone.setObjectName("dropzone")
        self._zone_style(False)
        z = QVBoxLayout(self.zone)
        z.setContentsMargins(26, 22, 26, 22)
        z.addWidget(label("Drop your calibration frames here", "h2"), 0, Qt.AlignmentFlag.AlignHCenter)
        how = label(HOW, "dim", wrap=True)
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
        self._plain.connect(self._on_plain)
        self.refresh()
        if self.settings.get("lab_clean_folder") and os.path.isdir(self.settings.get("lab_clean_folder")):
            self.survey_clean(self.settings.get("lab_clean_folder"))

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
            cardw = CameraCard(prof, (prof.make, prof.model) in fresh, self.clean_info)
            cardw.start_over.connect(self.start_over)
            cardw.deep.connect(self.deep_calibration)
            cardw.choose_clean.connect(self.choose_clean)
            cardw.train.connect(self.train_model)
            self.cards_box.addWidget(cardw)
            self.cards.append(cardw)
        lines = [[] for _ in self.need_status]             # the ticks on the three "what to shoot" cards
        for cw in self.cards:
            for k, (colour, text) in enumerate(shoot_status(cw.prof)):
                lines[k].append(f"<span style='color:{colour}'><b>{cw.prof.model}:</b> {text}</span>")
        for lb, ln in zip(self.need_status, lines):
            lb.setText("<br>".join(ln))
            lb.setVisible(bool(ln))
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

    # ---------------------------------------------------------------------------------------- teaching Clean
    def _emit_plain(self, n, total, name=""):
        try:
            self._plain.emit(int(n), int(total), str(name))
        except RuntimeError:
            pass

    def _on_plain(self, n, total, name):
        self.bar.setRange(0, max(total, 1))
        self.bar.setValue(n)
        self.prog.setText(f"{n} of {total}" + (f"  ·  {name}" if name else ""))

    def deep_calibration(self, prof, folder: str | None = None):
        """Build this camera's noise bank and fixed pattern from its calibration raws (engine/calib/deep.py)."""
        if self.busy:
            self.status.emit("Wait until the frames being measured are done.")
            return
        if folder is None:
            from ..engine.calib import deep as D
            bank = D.find_bank(prof.make, prof.model)
            start = (bank.meta.get("raws_folder") if bank else "") or str(paths.get("CALIBRATION_LOCAL"))
            folder = QFileDialog.getExistingDirectory(self, f"Folder of the {prof.model}'s calibration raws", start)
        if not folder:
            return
        self.busy, self._cancel, self._t0 = True, False, time.time()
        self.bar.setRange(0, 0)
        self.bar.show()
        self.cancel_b.show()
        self.prog.setText("Deep calibration: looking for this camera's dark frames...")

        def work():
            from ..engine.calib import deep as D
            from ..lab.data import list_raws
            return D.deep_calibrate(list_raws(folder), progress=self._emit_plain, cancelled=lambda: self._cancel)
        run_bg(work, "deep calibration", self._deep_done, self._failed, keep=True)

    def _deep_done(self, _, reports):
        self._finish()
        self.deep_last = reports
        if not reports:
            self.out.setText(f"<span style='color:{T.AMBER}'>●</span> Deep calibration found no dark frames in "
                             f"that folder. It needs the lens-cap raws the profile was made from.")
        else:
            lines = []
            for r in reports:
                fx = r.get("fixed") or {}
                share = [v["fixed_rows_dn"] ** 2 / max(v["one_frame_rows_dn"] ** 2, 1e-9) for v in fx.values()
                         if v.get("one_frame_rows_dn")]
                fixed = (f" About {100 * float(np.median(share)):.0f}% of the row banding is the same in every "
                         f"photo and can be subtracted exactly." if share else "")
                lines.append(f"<span style='color:{T.CYAN}'>●</span> <b>{r['model']}:</b> noise bank built, "
                             f"{r['patches']:,} patches of real dark frame in {len(r['groups'])} groups.{fixed}")
            self.out.setText("<br>".join(lines))
            self.status.emit("Noise bank built. The benchmark and the training now use this camera's real noise.")
        self.refresh()
        self._next()

    def choose_clean(self):
        start = self.settings.get("lab_clean_folder") or ""
        d = QFileDialog.getExistingDirectory(self, "Folder of clean photos (raws at ISO 400 or lower)", start)
        if d:
            self.survey_clean(d)

    def survey_clean(self, folder: str):
        """Count what the folder holds for training (in the background; nothing is decoded)."""
        self.settings.set("lab_clean_folder", folder)

        def work():
            from ..lab.data import survey
            sv = survey(folder)
            sv.pop("files", None)
            return sv
        run_bg(work, folder, self._clean_done, lambda *_: None)

    def _clean_done(self, folder, sv):
        if folder != self.settings.get("lab_clean_folder"):
            return
        self.clean_info = sv
        if not self.busy:
            self.refresh()

    def train_model(self, prof, launch=None):
        """Start the training window (Train Camera Model.bat) for this camera, or say what is missing first."""
        from .. import lab
        missing = [t for c, t in teach_status(prof, self.clean_info)[:2] if c == T.AMBER]
        folder = self.settings.get("lab_clean_folder")
        n = int((self.clean_info or {}).get("usable", 0))
        if not folder or not self.clean_info:
            missing.append(f"Choose the folder of clean photos first. About {lab.PHOTOS_WANTED} varied ones: "
                           + lab.PHOTOS_WHY)
        elif n < lab.PHOTOS_MINIMUM:
            missing.append(lab.photos_verdict(n)[1] + " " + lab.PHOTOS_WHY)
        if missing:
            self.out.setText("<br>".join(f"<span style='color:{T.AMBER}'>●</span> {m}" for m in missing))
            return False
        note = ""
        if n < lab.PHOTOS_WANTED:
            note = (f"<span style='color:{T.AMBER}'>●</span> {lab.photos_verdict(n)[1]} Training starts anyway; "
                    f"add photos and train again for a better network.<br>")
        bat = paths.APP_ROOT / "Train Camera Model.bat"
        args = [folder, "--camera", prof.model]
        try:
            if launch is not None:
                launch(str(bat), args)
            elif os.name == "nt" and bat.exists():
                import subprocess
                subprocess.Popen(["cmd", "/c", "start", "Train this camera's Clean", str(bat), *args],
                                 cwd=str(paths.APP_ROOT))
            else:
                self.out.setText(note + "Training runs from a console here: <code>python -m studio.lab.train --clean "
                                 f"\"{folder}\" --camera \"{prof.model}\"</code> (it needs PyTorch).")
                return False
        except OSError as e:
            report_problem("Starting the training window", e)
            self.out.setText(f"<span style='color:{T.RED}'>●</span> The training window could not be started: {e}")
            return False
        log.info("training started for %s on %s (%d clean photos)", prof.model, folder, n)
        self.out.setText(note + f"<span style='color:{T.CYAN}'>●</span> Training the {prof.model}'s network has "
                         f"started in its own window. Leave it overnight; it stops by itself and can be stopped "
                         f"and started again. When it is done, choose the method 'learned' in Develop &gt; Clean.")
        return True

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
