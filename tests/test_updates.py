"""Updates (Help > Check for updates): version maths, how a copy was installed, staging a download safely, and the
sidebar card. No network: GitHub's answers and the downloads are stood in for."""
from __future__ import annotations

import io
import json
import os
import time
import zipfile

import pytest

from studio import updates as U


def test_versions():
    assert U.parse_version("v1.2.3") == (1, 2, 3)
    assert U.parse_version("1.2") == (1, 2, 0)
    assert U.parse_version("0.1.0-beta") == (0, 1, 0)
    assert U.parse_version("") is None and U.parse_version("latest") is None
    assert U.is_newer("v0.2.0", "0.1.0") and U.is_newer("1.0.10", "1.0.9")
    assert not U.is_newer("v0.1.0", "0.1.0") and not U.is_newer("0.0.9", "0.1.0") and not U.is_newer("", "0.1.0")


def test_install_kind(tmp_path):
    git = tmp_path / "dev"
    (git / ".git").mkdir(parents=True)
    assert U.install_kind(git) == "git"
    bat = tmp_path / "bat"
    bat.mkdir()
    (bat / ".install_kind").write_text("bat\n")
    assert U.install_kind(bat) == "bat"
    msi = tmp_path / "Photographic Studio" / "app"
    msi.mkdir(parents=True)
    (msi.parent / "app_packages").mkdir()
    assert U.install_kind(msi) == "msi"
    plain = tmp_path / "plain"
    plain.mkdir()
    assert U.install_kind(plain) == "folder"
    sh = tmp_path / "sh"
    sh.mkdir()
    (sh / ".install_kind").write_text("sh\n")
    assert U.install_kind(sh) == "sh"


def test_restart_into_update_starts_the_right_launcher(tmp_path, monkeypatch):
    """Linux: start-studio.sh is started on its own (it puts the update in place); Windows: Start Studio.bat."""
    import subprocess
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kw: calls.append((args, kw)))
    monkeypatch.setattr(U.sys, "platform", "linux")
    assert not U.restart_into_update(tmp_path)                    # no launcher, nothing started
    (tmp_path / "start-studio.sh").write_text("#!/usr/bin/env bash\n")
    assert U.restart_into_update(tmp_path)
    args, kw = calls[-1]
    assert args == ["bash", str(tmp_path / "start-studio.sh")] and kw["start_new_session"]
    assert U.launcher_name() == "start-studio.sh"
    monkeypatch.setattr(U.sys, "platform", "win32")
    assert U.launcher_name() == "Start Studio.bat"


class _Resp(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _zip(files: dict) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return b.getvalue()


def test_check_and_stage(tmp_path, monkeypatch):
    release = {"tag_name": "v9.0.0", "name": "Studio 9", "body": "Faster.", "html_url": "https://example/rel",
               "assets": [{"name": "Install-Photographic-Studio.bat", "browser_download_url": "https://x/i.bat"},
                          {"name": "Photographic-Studio.msi", "browser_download_url": "https://x/s.msi"}]}
    good = _zip({"Photographic_Studio_By_Bill-9.0.0/studio/__init__.py": "__version__ = '9.0.0'\n",
                 "Photographic_Studio_By_Bill-9.0.0/Start Studio.bat": "@echo off\n"})
    seen = []

    def fake(url, timeout):
        seen.append(url)
        if url == U.API_LATEST:
            return _Resp(json.dumps(release).encode())
        if url.endswith(".zip"):
            return _Resp(good)
        return _Resp(b"MSI")
    monkeypatch.setattr(U, "_request", fake)
    rel = U.check("0.1.0")
    assert rel["newer"] and rel["tag"] == "v9.0.0" and rel["msi"] == "https://x/s.msi"
    assert rel["zip"].endswith("/archive/refs/tags/v9.0.0.zip")
    root = tmp_path / "Photographic Studio"
    (root / "settings").mkdir(parents=True)
    (root / "settings" / "app.json").write_text("{}")
    top = U.stage_folder_update(rel, root)
    assert (top / "studio" / "__init__.py").exists() and U.staged(root) == "v9.0.0"
    assert not (root / "_update" / "studio.zip").exists() and (root / "settings" / "app.json").read_text() == "{}"
    msi = U.stage_msi(rel, root)
    assert msi.read_bytes() == b"MSI" and U.downloaded_msi(rel, root) == msi
    assert not list((root / "_update").glob("*.partial"))
    # a git copy is never changed
    (root / ".git").mkdir()
    with pytest.raises(RuntimeError):
        U.stage_folder_update(rel, root)


def test_unsafe_or_wrong_downloads_are_refused(tmp_path, monkeypatch):
    rel = {"tag": "v9.0.0", "zip": "https://x/a.zip"}
    root = tmp_path / "s"
    root.mkdir()
    monkeypatch.setattr(U, "_request", lambda url, t: _Resp(_zip({"../evil.py": "x"})))
    with pytest.raises(RuntimeError):
        U.stage_folder_update(rel, root)
    assert not (tmp_path / "evil.py").exists() and U.staged(root) is None
    monkeypatch.setattr(U, "_request", lambda url, t: _Resp(_zip({"something/else.txt": "x"})))
    with pytest.raises(RuntimeError):
        U.stage_folder_update(rel, root)
    assert U.staged(root) is None


def test_sidebar_asks_once_then_offers_the_update(monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    store = {}
    monkeypatch.setattr(M.Settings, "set", lambda self, k, v: store.__setitem__(k, v) or self.d.__setitem__(k, v))
    monkeypatch.setattr(M.Settings, "__init__", lambda self: setattr(self, "d", {}))
    monkeypatch.setattr(U, "install_kind", lambda root=None: "bat")
    monkeypatch.setattr(U, "staged", lambda root=None: None)
    calls = []
    monkeypatch.setattr(U, "check", lambda current=None, timeout=10.0: calls.append(1) or
                        {"tag": "v9.0.0", "newer": True, "page": "https://example/rel", "current": "0.1.0",
                         "zip": "z", "msi": None})
    monkeypatch.setattr(U, "stage_folder_update", lambda rel, root=None, progress=None: "staged")
    jobs = []
    monkeypatch.setattr(M, "run_bg", lambda fn, tok=None, done=None, fail=None: jobs.append((fn, tok, done, fail)))
    app = QApplication.instance() or QApplication([])
    w = M.MainWindow()
    w._updates_start()                                    # first start of an installed copy: asked once
    assert not w.upd_card.isHidden() and "up to date" in w.upd_text.text()
    w._set_update_mode("notify")
    assert store["updates"] == "notify" and w.help.upd_combo.currentIndex() == 1
    fn, tok, done, fail = jobs[-1]
    done(tok, fn())                                       # GitHub says there is a newer version
    assert calls and "v9.0.0" in w.upd_text.text() and not w.upd_card.isHidden()
    assert time.time() - store["updates_last"] < 60
    w._upd_download(w.upd_rel)                            # Update now (without restarting, for the test)
    fn, tok, done, fail = jobs[-1]
    done(tok, fn())
    assert "ready" in w.upd_text.text()
    fail(tok, "offline")                                  # a failed download offers to try again
    assert "could not be downloaded" in w.upd_text.text()
    w._set_update_mode("off")
    assert store["updates"] == "off" and w.upd_card.isHidden()
    w.upd_msi = None
    app.processEvents()


def test_msi_copies_keep_user_files_outside_the_install(tmp_path):
    from studio import paths
    app = tmp_path / "Programs" / "Photographic Studio" / "app"
    app.mkdir(parents=True)
    assert not paths.is_msi_install(app)
    (app.parent / "app_packages").mkdir()
    assert paths.is_msi_install(app)
    assert not paths.is_msi_install(paths.APP_ROOT) and paths.DATA_ROOT == paths.APP_ROOT or os.environ.get(
        "STUDIO_DATA_DIR")


def test_version_numbers_for_releases(tmp_path, monkeypatch):
    """Release.bat (and push.bat, through it) set the version with tools/set_version.py: --bump raises the last
    number by one, a typed number must be higher, and both files change together."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("set_version", os.path.join(os.path.dirname(__file__), "..",
                                                                              "tools", "set_version.py"))
    sv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sv)
    init, toml = tmp_path / "__init__.py", tmp_path / "pyproject.toml"
    init.write_text('__version__ = "0.1.9"\nAPP_NAME = "x"\n')
    toml.write_text('[tool.briefcase]\nproject_name = "P"\nversion = "0.1.9"\n')
    monkeypatch.setattr(sv, "INIT", init)
    monkeypatch.setattr(sv, "PYPROJECT", toml)
    assert sv.main(["--bump"]) == 0 and sv.current() == "0.1.10" and 'version = "0.1.10"' in toml.read_text()
    assert sv.main(["0.1.5"]) == 1 and sv.current() == "0.1.10"           # lower: refused, nothing changed
    assert sv.main(["0.2"]) == 1                                           # not three numbers
    assert sv.main(["v0.2.0"]) == 0 and sv.current() == "0.2.0" and 'APP_NAME = "x"' in init.read_text()
