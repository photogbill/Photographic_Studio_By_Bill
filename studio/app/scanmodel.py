"""3D Scan's model view (BUILD_PLAN_DP3D H.3, H.4): the model turned on a software turntable (no OpenGL), coloured by
its texture or by how each part was made (cyan measured, violet filled), the cameras' quality in plain words, and the
coach's advice for the next shoot.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QComboBox, QLabel, QSlider, QVBoxLayout, QWidget

from . import theme as T
from .common import hbox, label

CYAN = np.array([0x22, 0xD3, 0xEE]) / 255.0
VIOLET = np.array([0xC0, 0x84, 0xFC]) / 255.0


class ModelView(QWidget):
    SIZE = 420

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 8, 16, 8)
        self.title = label("", "h2")
        v.addWidget(self.title)
        self.pic = QLabel()
        self.pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pic.setMinimumHeight(self.SIZE)
        self.turn = QSlider(Qt.Orientation.Horizontal)
        self.turn.setRange(0, 359)
        self.turn.setToolTip("Drag to turn the model")
        self.turn.valueChanged.connect(self._draw)
        self.mode = QComboBox()
        self.mode.addItem("Its colours", "colour")
        self.mode.addItem("How each part was made (cyan measured, violet filled)", "made")
        self.mode.currentIndexChanged.connect(self._draw)
        v.addWidget(self.pic, 1)
        v.addWidget(hbox(label("Turn", "dim"), (self.turn, 1), self.mode, stretch=False))
        self.text = label("", None, wrap=True)
        v.addWidget(self.text)
        self.mesh = None

    def show_model(self, result: dict):
        """``result``: what scan.model.build returned."""
        V, Tr, measured = result["mesh"]
        doc = result["json"]
        q = doc["quality"]
        self.title.setText("Model: " + ", ".join(f"{k}" for k in doc["files"] if k in ("glb", "dataset")))
        # colours per vertex: the texture sampled at the vertices' places is not kept here; the "made" colours are,
        # and the colour mode shows a neutral clay with the measured parts tinted
        made = np.zeros((len(V), 3))
        fc = np.where(measured[:, None], CYAN[None], VIOLET[None])
        for k in range(3):
            made[Tr[:, k]] = fc
        clay = np.full((len(V), 3), 0.78)
        self.mesh = (np.asarray(V), np.asarray(Tr), made, clay)
        lines = [f"{q['views_placed']} of {q['views_given']} views placed; the points land within "
                 f"{q['reprojection_px']} px of where the views saw them." if q.get("reprojection_px") is not None
                 else f"{q['views_placed']} of {q['views_given']} views placed.",
                 f"Measured: {100 * q['measured_share']:.0f}% of the surface; filled from the outline: "
                 f"{100 * q['filled_share']:.0f}%.",
                 ("Blender dataset written." if doc["gate"]["passed"] else
                  "No Blender dataset: " + "; ".join(doc["gate"]["reasons"]) + ".")]
        lines += doc["coach"]["lines"]
        colour = T.GREEN if doc["gate"]["passed"] else T.AMBER
        self.text.setText("\n".join(lines))
        self.text.setStyleSheet(f"color:{colour};" if not doc["gate"]["passed"] else "")
        self._draw()

    def _draw(self, *_):
        if self.mesh is None:
            return
        from ..engine.scan import raster as RA
        V, Tr, made, clay = self.mesh
        cols = made if self.mode.currentData() == "made" else clay
        img = RA.turntable(V, Tr, cols, float(self.turn.value()), 20.0, self.SIZE)
        h, w = img.shape[:2]
        qi = QImage(np.ascontiguousarray(img).data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.pic.setPixmap(QPixmap.fromImage(qi))
