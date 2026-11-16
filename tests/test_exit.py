"""Closing the Studio. A friend on Linux closed it while a Clean render was running and got "The Studio stopped
with an error" (the render's thread outlived Python's shutdown, and the error handler tried to show a message box
with no application left). Now: previews are dropped at exit, export / import jobs are waited for, and the process
leaves without tearing down; one Studio at a time."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_closing_during_a_render_exits_cleanly_and_quickly(tmp_path):
    script = textwrap.dedent(f"""
        import os, sys, time
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["STUDIO_SETTINGS_DIR"] = {str(tmp_path / "s")!r}
        os.environ["STUDIO_LOG_DIR"] = {str(tmp_path / "l")!r}
        os.environ["STUDIO_CACHE_DIR"] = {str(tmp_path / "c")!r}
        sys.path.insert(0, {ROOT!r})
        from PySide6.QtCore import QTimer
        from studio.app import main as M
        from studio.app.common import run_bg
        real = M.MainWindow.__init__
        def init(self):
            real(self)
            def slow():                       # a render still running when the window closes
                time.sleep(20)
            QTimer.singleShot(100, lambda: run_bg(slow, "preview", lambda *a: None))
            QTimer.singleShot(400, self.close)
        M.MainWindow.__init__ = init
        sys.exit(M.run([]))
    """)
    t = time.time()
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    dt = time.time() - t
    assert r.returncode == 0, r.stderr[-2000:]
    assert dt < 12                                        # not held up by the 20 s preview
    assert "Traceback" not in r.stderr
    log = (tmp_path / "l" / "studio.log").read_text(encoding="utf-8")
    assert "app closed (exit 0" in log


def test_export_jobs_are_waited_for_and_late_work_is_refused(monkeypatch):
    from studio.app import common as C
    from studio.app import main as M
    done = []
    try:
        C.run_bg(lambda: (time.sleep(0.6), done.append("export")), "export", keep=True)
        C.run_bg(lambda: time.sleep(5), "preview")
        time.sleep(0.1)
        assert C.kept_jobs_running() == 1
        t = time.time()
        clean = M.finish_jobs(timeout_ms=200)
        assert done == ["export"]                         # the export finished before exit
        assert not clean and time.time() - t < 3          # the preview was not waited for
        C.run_bg(lambda: done.append("late"), "late")     # nothing new starts once closing
        time.sleep(0.2)
        assert done == ["export"]
    finally:
        C._SHUTTING_DOWN = False
        C.POOL.waitForDone(10000)


def test_one_studio_at_a_time(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from studio.app import main as M
    monkeypatch.setenv("STUDIO_SETTINGS_DIR", str(tmp_path))
    app = QApplication.instance() or QApplication([])
    first = M._single_instance(app)
    assert first is not None and not isinstance(first, M._NoLock)
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(self.text()) or 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: None)
    assert M._single_instance(app) is None                # a second start is told, and stops
    assert shown and "already open" in shown[0]
    first.unlock()
    again = M._single_instance(app)
    assert again is not None
    again.unlock()
