"""Photographic Studio By Bill.

A free, open-source raw studio. GPL-3.0 code, CC BY-SA 4.0 profiles.

Package layout (BUILD_PLAN.md section 2.2):
    studio.engine   no GUI imports, ever; usable from scripts, tests, the Photoshop bridge and ATK
    studio.app      the PySide6 interface (Phase 4)
    studio.cli      studio-cli, the command-line runner used by the .bat files and the tests
"""
from __future__ import annotations

import os as _os

# Must run before numpy is first imported anywhere in the Studio. numpy's bundled OpenBLAS crashed the Studio
# with a Windows access violation (0xC0000005) on Bill's 14-core Intel PC (Windows 11, numpy 2.2.6), inside an
# ordinary 3x3 colour-matrix multiply, on every raw; with one OpenBLAS thread the same raw develops normally
# (checked 2026-11-21). The Studio's own heavy work is spread over frames and strips, not inside BLAS, so one
# thread costs little. Anyone who sets OPENBLAS_NUM_THREADS themselves keeps their own value.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.1.15"
APP_NAME = "Photographic Studio By Bill"
