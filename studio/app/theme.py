"""The look (rule 0.3.11): dark, cinematic, high contrast. Colours carry meaning everywhere:
cyan = measured / calibrated on this body, violet = shared profile, amber = estimated, green = good, red = problem."""
from __future__ import annotations

BG = "#07090f"
BG2 = "#0b101c"
PANEL = "#0e1320"
PANEL2 = "#121a2b"
LINE = "#1c2a4a"
TEXT = "#e6ecff"
DIM = "#8a97b8"
CYAN = "#22d3ee"
AMBER = "#f59e0b"
VIOLET = "#c084fc"
GREEN = "#22c55e"
RED = "#f87171"

TIER_COLOURS = {"calibrated": CYAN, "shared": VIOLET, "estimated": AMBER}


def alpha(colour: str, a: float) -> str:
    """'#rrggbb' with opacity as a Qt style-sheet colour. (Qt reads 8-digit hex as #AARRGGBB, not #RRGGBBAA,
    so '#f59e0b66' would come out magenta.)"""
    c = colour.lstrip("#")
    return f"rgba({int(c[0:2], 16)}, {int(c[2:4], 16)}, {int(c[4:6], 16)}, {int(round(a * 255))})"

QSS = f"""
* {{ font-family: "Segoe UI", "Inter", sans-serif; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget {{ background: transparent; }}
QToolTip {{ background: {PANEL2}; color: {TEXT}; border: 1px solid {LINE}; padding: 6px; border-radius: 6px; }}
QFrame#sidebar {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0d1424, stop:1 #070a12);
                 border-right: 1px solid {LINE}; }}
QListWidget#nav {{ border: none; outline: none; }}
QListWidget#nav::item {{ height: 46px; padding-left: 14px; border-radius: 10px; margin: 2px 8px; color: {DIM}; }}
QListWidget#nav::item:hover {{ background: #131c30; color: {TEXT}; }}
QListWidget#nav::item:selected {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #12324a, stop:1 #141a33);
                                 color: {TEXT}; border-left: 3px solid {CYAN}; }}
QLabel#brand {{ font-size: 13pt; font-weight: 600; padding: 18px 16px 2px 18px; }}
QLabel#brandsub {{ color: {DIM}; padding: 0 18px 14px 18px; font-size: 8.5pt; }}
QLabel#h1 {{ font-size: 18pt; font-weight: 600; }}
QLabel#h2 {{ font-size: 12pt; font-weight: 600; }}
QLabel#dim {{ color: {DIM}; }}
QLabel#chip {{ border: 1px solid {LINE}; border-radius: 11px; padding: 3px 10px; background: {BG2}; }}
QFrame#card {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0f1628, stop:1 #0b101c);
              border: 1px solid {LINE}; border-radius: 14px; }}
QFrame#panel {{ background: {PANEL}; border-left: 1px solid {LINE}; }}
QFrame#topbar, QFrame#bottombar {{ background: {BG2}; border-bottom: 1px solid {LINE}; }}
QFrame#bottombar {{ border-top: 1px solid {LINE}; border-bottom: none; }}
QPushButton {{ background: {PANEL2}; border: 1px solid {LINE}; border-radius: 8px; padding: 7px 14px; }}
QPushButton:hover {{ border-color: {CYAN}; }}
QPushButton:pressed {{ background: #0c2230; }}
QPushButton:checked {{ background: #0f2c3d; border-color: {CYAN}; }}
QPushButton:disabled {{ color: #4a5672; border-color: #151f36; }}
QPushButton#primary {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0891b2, stop:1 #7c3aed);
                      border: none; font-weight: 600; padding: 9px 18px; }}
QPushButton#primary:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #06b6d4, stop:1 #8b5cf6); }}
QPushButton#group {{ text-align: left; font-weight: 600; background: transparent; border: none;
                    border-bottom: 1px solid {LINE}; border-radius: 0; padding: 10px 4px; }}
QSlider::groove:horizontal {{ height: 4px; background: #1a2440; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0e7490, stop:1 {CYAN});
                               border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; background: {TEXT};
                             border: 2px solid {CYAN}; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {BG2}; border: 1px solid {LINE}; border-radius: 7px;
                                               padding: 5px 8px; }}
QComboBox QAbstractItemView {{ background: {PANEL2}; border: 1px solid {LINE}; selection-background-color: #12324a; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {LINE}; background: {BG2}; }}
QCheckBox::indicator:checked {{ background: {CYAN}; border-color: {CYAN}; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #1f2b49; border-radius: 5px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: #1f2b49; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QTreeView, QListWidget {{ background: {BG2}; border: 1px solid {LINE}; border-radius: 10px; outline: none; }}
QTreeView::item {{ padding: 4px; }}
QTreeView::item:selected, QListWidget::item:selected {{ background: #12324a; border-radius: 6px; }}
QListWidget#grid::item {{ border-radius: 10px; padding: 6px; margin: 4px; }}
QListWidget#grid::item:selected {{ background: #12324a; border: 1px solid {CYAN}; }}
QProgressBar {{ background: {BG2}; border: 1px solid {LINE}; border-radius: 6px; height: 10px; text-align: center; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0891b2, stop:1 #7c3aed);
                      border-radius: 6px; }}
QTabWidget::pane {{ border: none; }}
QDialog {{ background: {PANEL}; }}
QStatusBar {{ background: {BG2}; color: {DIM}; border-top: 1px solid {LINE}; }}
"""
