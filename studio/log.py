"""Comprehensive logging (rule 0.3.5).

* Rotating log files in ``logs/`` beside the app.
* One line per pipeline stage with its timing (``with stage("demosaic"): ...``).
* Every exception with its traceback.
* ``diagnostic_report()`` builds the text behind the "Copy diagnostic report" button.
* Never log serial numbers or GPS: ``scrub()`` removes them from anything that is about to be logged.
"""
from __future__ import annotations

import contextlib
import logging
import logging.handlers
import os
import platform
import re
import sys
import threading
import time
import traceback
from pathlib import Path

from . import __version__, paths

LOG_NAME = "studio"
log = logging.getLogger(LOG_NAME)
_lock = threading.Lock()
_configured = False

_PRIVATE_KEYS = re.compile(r"(serial|gps|latitude|longitude|owner|body.?id|internal.?serial)", re.I)


def setup(console_level: int = logging.INFO, filename: str = "studio.log") -> Path:
    """Configure the 'studio' logger once. Returns the log file path."""
    global _configured
    with _lock:
        logdir = paths.get("LOGS")
        path = logdir / filename
        if _configured:
            return path
        log.setLevel(logging.DEBUG)
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"))
        log.addHandler(fh)
        if console_level is not None:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(console_level)
            ch.setFormatter(logging.Formatter("%(message)s"))
            log.addHandler(ch)
        log.propagate = False
        _configured = True
        log.debug("---- %s %s started: python %s on %s ----", "Photographic Studio By Bill", __version__,
                  platform.python_version(), platform.platform())
        return path


def get(name: str) -> logging.Logger:
    """Child logger, e.g. get('engine.demosaic')."""
    return logging.getLogger(f"{LOG_NAME}.{name}")


@contextlib.contextmanager
def stage(name: str, logger: logging.Logger | None = None, **info):
    """Time one pipeline stage and write one log line for it. Exceptions are logged with traceback and re-raised."""
    lg = logger or log
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        lg.error("stage %s FAILED after %.3f s %s\n%s", name, time.perf_counter() - t0,
                 _fmt_info(info), traceback.format_exc())
        raise
    else:
        lg.debug("stage %-22s %8.3f s %s", name, time.perf_counter() - t0, _fmt_info(info))


def _fmt_info(info: dict) -> str:
    if not info:
        return ""
    return " ".join(f"{k}={v}" for k, v in scrub(info).items())


def scrub(d):
    """Return a copy of a dict (recursively) with serial numbers, GPS and owner fields removed."""
    if isinstance(d, dict):
        return {k: ("<removed>" if _PRIVATE_KEYS.search(str(k)) else scrub(v)) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return type(d)(scrub(v) for v in d)
    return d


def log_exception(msg: str, logger: logging.Logger | None = None) -> None:
    (logger or log).error("%s\n%s", msg, traceback.format_exc())


def diagnostic_report(tail_lines: int = 200) -> str:
    """Plain-text report for bug reports: versions, platform, and the end of the log. No photos, no serials."""
    lines = [
        f"Photographic Studio By Bill {__version__}",
        f"Python {platform.python_version()} ({sys.executable})",
        f"Platform {platform.platform()}  machine={platform.machine()}  cpus={os.cpu_count()}",
        f"App folder {paths.APP_ROOT}",
    ]
    for mod in ("numpy", "scipy", "rawpy", "tifffile", "PIL", "numba", "PySide6"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            if mod == "rawpy":
                v = f"{v} (LibRaw {'.'.join(map(str, getattr(m, 'libraw_version', ())))})"
            lines.append(f"  {mod:9s} {v}")
        except Exception:
            lines.append(f"  {mod:9s} not installed")
    logfile = paths.get("LOGS") / "studio.log"
    lines.append("")
    lines.append(f"---- last {tail_lines} lines of {logfile.name} ----")
    try:
        with open(logfile, encoding="utf-8", errors="replace") as fh:
            tail = fh.readlines()[-tail_lines:]
        lines.extend(l.rstrip("\n") for l in tail)
    except OSError:
        lines.append("(no log yet)")
    text = "\n".join(lines)
    # belt and braces: strip anything that looks like a serial-number line
    return "\n".join(l for l in text.splitlines() if not _PRIVATE_KEYS.search(l.split(":")[0] if ":" in l else ""))
