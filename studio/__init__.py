"""Photographic Studio By Bill.

A free, open-source raw studio. GPL-3.0 code, CC BY-SA 4.0 profiles.

Package layout (BUILD_PLAN.md section 2.2):
    studio.engine   no GUI imports, ever; usable from scripts, tests, the Photoshop bridge and ATK
    studio.app      the PySide6 interface (Phase 4)
    studio.cli      studio-cli, the command-line runner used by the .bat files and the tests
"""
from __future__ import annotations

__version__ = "0.1.9"
APP_NAME = "Photographic Studio By Bill"
