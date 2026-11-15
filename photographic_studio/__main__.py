"""``python -m photographic_studio``: what the MSI's Photographic Studio.exe runs."""
from __future__ import annotations

import sys

from studio.app.main import run

sys.exit(run())
