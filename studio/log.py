"""Comprehensive logging (rule 0.3.5).

* Rotating log files in ``logs/`` beside the app.
* One line per pipeline stage with its timing (``with stage("demosaic"): ...``).
* Every exception with its traceback.
* ``diagnostic_report()`` builds the text behind the "Copy diagnostic report" button.
* ``save_error_report()`` writes that report, with the error on top, into ``error_reports/`` beside the app as
  ``error_YYYY-MM-DD_HH-MM-SS.txt``, by itself, whenever something goes wrong that should not (Bill, 2026-11-16:
  testers send the file; nobody has to find a button first).
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


# ------------------------------------------------------------------------------------------ error reports
REPORT_LIMIT = 30                 # new report files per run of the Studio (a fault that repeats must not fill a disk)
REPORT_KEEP = 200                 # the newest reports kept in the folder; older ones are removed
_report_lock = threading.Lock()
_reported: dict = {}              # signature of an error -> the report already written for it in this run


def error_reports_dir() -> Path:
    return paths.get("ERROR_REPORTS")


_EXPECTED = ("UnsupportedRaw", "UnsupportedScan", "ImportStopped", "Cancelled", "StackProblem")


def expected_problem(exc: BaseException) -> bool:
    """A problem with a photo, a drive or the internet, which is said to the person where it happens (a raw the
    decoder cannot read, a read-only card, no connection): not a fault in the Studio, so no error report."""
    return isinstance(exc, OSError) or type(exc).__name__ in _EXPECTED


def _signature(trace: str) -> str:
    """The same fault at the same place gives the same signature: its last code line and the kind of error."""
    lines = [ln for ln in trace.strip().splitlines() if ln.strip()]
    where = next((ln.strip() for ln in reversed(lines) if ln.lstrip().startswith("File ")), "")
    kind = lines[-1].split(":")[0].strip() if lines else ""
    return f"{where}|{kind}"


def save_error_report(what: str, trace: str | None = None, extra: str = "") -> Path | None:
    """Write an error report file and return its path: ``error_reports/error_YYYY-MM-DD_HH-MM-SS.txt`` (beside the
    app; in Documents/Photographic Studio for the MSI copy). ``what``: one plain line on what was being done.
    ``trace``: the error text (default: the exception being handled). The same fault is written once per run (the
    first report's path is returned again); at most REPORT_LIMIT files per run; the folder keeps the newest
    REPORT_KEEP. Holds no photos; lines naming serial numbers, GPS or the owner are removed. Never raises: a
    report that cannot be written returns None."""
    try:
        if trace is None:
            trace = traceback.format_exc()
        trace = trace or "(no error text)"
        sig = _signature(trace)
        with _report_lock:
            if sig in _reported:
                return _reported[sig]
            if len(_reported) >= REPORT_LIMIT:
                return None
            _reported[sig] = None                     # claimed: a second thread with the same fault writes nothing
        now = time.localtime()
        folder = error_reports_dir()
        stem = time.strftime("error_%Y-%m-%d_%H-%M-%S", now)
        path, n = folder / f"{stem}.txt", 1
        while path.exists():
            n += 1
            path = folder / f"{stem}_{n}.txt"
        lines = [
            "Photographic Studio By Bill - error report",
            f"When:  {time.strftime('%Y-%m-%d %H:%M:%S', now)}",
            f"What:  {what}",
            "",
            "This file holds no photos, serial numbers or locations. Please send it to Bill, with a line about what",
            "you were doing when it happened.",
            "",
            "---- the error ----",
            trace.rstrip(),
        ]
        if extra:
            lines += ["", extra.rstrip()]
        lines += ["", "---- the Studio and this PC ----", diagnostic_report()]
        text = "\n".join(lines)
        text = "\n".join(ln for ln in text.splitlines()
                         if not _PRIVATE_KEYS.search(ln.split(":")[0] if ":" in ln else ""))
        tmp = path.with_name(path.name + ".partial")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, path)
        with _report_lock:
            _reported[sig] = path
        log.error("error report written: %s (%s)", path, what)
        _prune_reports(folder)
        return path
    except Exception:                                 # noqa: BLE001  (reporting an error must never cause one)
        try:
            log.warning("an error report could not be written\n%s", traceback.format_exc())
        except Exception:                             # noqa: BLE001
            pass
        return None


def _prune_reports(folder: Path) -> None:
    try:
        files = sorted(folder.glob("error_*.txt"), key=lambda f: f.stat().st_mtime)
        for f in files[:-REPORT_KEEP]:
            f.unlink(missing_ok=True)
    except OSError:
        pass


def latest_error_report() -> Path | None:
    """The newest report in the folder (written in this run or an earlier one), or None."""
    try:
        files = sorted(error_reports_dir().glob("error_*.txt"), key=lambda f: f.stat().st_mtime)
        return files[-1] if files else None
    except OSError:
        return None


# ------------------------------------------------------------------------------------------ hard crashes
_crash_file = None


def watch_for_crashes() -> Path | None:
    """A crash inside Qt or the raw decoder ends the Studio with no Python error to report. Python's faulthandler
    writes where every thread was into logs/crash_trace.log as it dies; the next start finds that and turns it into
    an error report (returned; None when the last run ended normally)."""
    global _crash_file
    import faulthandler
    report = None
    try:
        trace_path = paths.get("LOGS") / "crash_trace.log"
        if trace_path.exists() and trace_path.stat().st_size > 0:
            old = trace_path.read_text(encoding="utf-8", errors="replace")
            report = save_error_report("The Studio stopped unexpectedly the last time it ran (found at this start).",
                                       old[-20000:])
        _crash_file = open(trace_path, "w", encoding="utf-8")      # kept open for as long as the Studio runs
        faulthandler.enable(file=_crash_file, all_threads=True)
    except Exception:                                 # noqa: BLE001  (never stops the Studio starting)
        log.debug("crash watch not started\n%s", traceback.format_exc())
    return report
