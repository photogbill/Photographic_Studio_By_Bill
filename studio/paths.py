"""Where the app keeps its own files.

Rule 0.3.3: nothing is written to C:\\ or the Windows user profile (AppData, Documents, Temp) by default.
Settings, caches, logs and outputs all live inside the app folder (portable install).

The app folder is the folder that contains the ``studio`` package. Every path below is derived from it,
so moving the folder moves everything with it.
"""
from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent


def is_msi_install(root: Path = APP_ROOT) -> bool:
    """The MSI installer's layout (Briefcase): the app in <install>/app, its packages in <install>/app_packages."""
    return root.name == "app" and (root.parent / "app_packages").is_dir()


def _data_root() -> Path:
    """Where the user's own files live (settings, logs, caches, outputs, calibrations). Beside the app, as rule
    0.3.3 says, in every copy but one: the MSI install's folder belongs to Windows Installer, which may clear it
    on an upgrade or uninstall, so that copy keeps them in Documents/Photographic Studio, where they survive
    (and are easy to find). STUDIO_DATA_DIR overrides both."""
    if os.environ.get("STUDIO_DATA_DIR"):
        return Path(os.environ["STUDIO_DATA_DIR"])
    if is_msi_install():
        return Path.home() / "Documents" / "Photographic Studio"
    return APP_ROOT


DATA_ROOT = _data_root()

LOGS = DATA_ROOT / "logs"
CACHE = DATA_ROOT / "cache"
SETTINGS = DATA_ROOT / "settings"
OUTPUTS = DATA_ROOT / "outputs"
PROFILES = APP_ROOT / "profiles"                       # shared profiles ship with the app
CALIBRATION_LOCAL = DATA_ROOT / "calibration_local"   # never shared: hot-pixel, PRNU, dust maps, darks
PHOTOS = DATA_ROOT / "Photos"                         # the suggested Photos folder for card imports (section 24.3)
JOBS = DATA_ROOT / "jobs"                             # the job queue (jobqueue.py): one folder per job
ERROR_REPORTS = DATA_ROOT / "error_reports"           # one dated file per unexpected error, written by itself (log.py)
PAINT = DATA_ROOT / "paint"                           # Paint: autosave / crash recovery, brushes (BUILD_PLAN_PAINT)

# Environment overrides exist for tests and for users who move the cache; the defaults stay beside the app.
_OVERRIDES = {
    "LOGS": "STUDIO_LOG_DIR",
    "CACHE": "STUDIO_CACHE_DIR",
    "OUTPUTS": "STUDIO_OUTPUT_DIR",
    "CALIBRATION_LOCAL": "STUDIO_CALIB_DIR",
    "SETTINGS": "STUDIO_SETTINGS_DIR",
    "PHOTOS": "STUDIO_PHOTOS_DIR",
    "ERROR_REPORTS": "STUDIO_ERROR_DIR",
    "JOBS": "STUDIO_JOBS_DIR",
    "PAINT": "STUDIO_PAINT_DIR",
}


def get(name: str) -> Path:
    """Return one of the app folders by name (LOGS, CACHE, SETTINGS, OUTPUTS, ...), creating it if needed."""
    env = _OVERRIDES.get(name)
    p = Path(os.environ[env]) if env and os.environ.get(env) else globals()[name]
    p.mkdir(parents=True, exist_ok=True)
    return p


def photos_dir() -> Path:
    """The suggested Photos folder (``Photos`` beside the app; STUDIO_PHOTOS_DIR overrides). Unlike ``get`` this
    never creates it: the photographer chooses and confirms where photos are kept first (section 24.3)."""
    env = os.environ.get(_OVERRIDES["PHOTOS"])
    return Path(env) if env else PHOTOS


def temp_dir() -> Path:
    """A scratch folder inside the app folder. Use this instead of tempfile's default (which is on C:)."""
    return get("CACHE").joinpath("tmp")
