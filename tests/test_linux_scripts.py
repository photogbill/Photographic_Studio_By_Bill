"""The Linux installer and launcher (install-linux.sh, start-studio.sh, tools/ensure_env.sh): they parse, keep LF
line endings (a CR breaks bash on the first line), and the library-to-package table answers for each package
manager. The whole install was run end to end on Ubuntu 24.04 (see docs/PROGRESS.md decision 84); that needs the
network and minutes, so it is not repeated here."""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ["install-linux.sh", "start-studio.sh", "start-paint.sh", os.path.join("tools", "ensure_env.sh")]
BASH = shutil.which("bash")


def test_scripts_have_lf_line_endings_and_a_shebang():
    for name in SCRIPTS:
        with open(os.path.join(ROOT, name), "rb") as fh:
            data = fh.read()
        assert b"\r" not in data, f"{name} has CR characters (Windows line endings)"
        if not name.startswith("tools"):
            assert data.startswith(b"#!/usr/bin/env bash\n")
            assert data.rstrip().endswith(b'main "$@"; exit $?')   # read whole before running: safe to update


@pytest.mark.skipif(BASH is None or os.name == "nt", reason="needs bash (Linux, macOS)")
def test_scripts_parse():
    for name in SCRIPTS:
        r = subprocess.run([BASH, "-n", os.path.join(ROOT, name)], capture_output=True, text=True)
        assert r.returncode == 0, f"{name}: {r.stderr}"


@pytest.mark.skipif(BASH is None or os.name == "nt", reason="needs bash (Linux, macOS)")
def test_missing_libraries_map_to_packages():
    env_sh = os.path.join(ROOT, "tools", "ensure_env.sh")
    script = (f'. "{env_sh}"; for pm in apt dnf zypper pacman rpm-ostree; do '
              'printf "%s " "$(_studio_pkg_for libxcb-cursor.so.0 $pm)"; done; '
              '_studio_pkg_for libnothing.so.9 apt || echo none')
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    assert r.stdout.split() == ["libxcb-cursor0", "xcb-util-cursor", "libxcb-cursor0", "xcb-util-cursor",
                                "xcb-util-cursor", "none"]


@pytest.mark.skipif(BASH is None or os.name == "nt", reason="needs bash (Linux, macOS)")
def test_installer_help_and_unknown_options():
    r = subprocess.run([BASH, os.path.join(ROOT, "install-linux.sh"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--dir FOLDER" in r.stdout
    r = subprocess.run([BASH, os.path.join(ROOT, "install-linux.sh"), "--bogus"], capture_output=True, text=True)
    assert r.returncode == 1 and "Unknown option" in r.stdout
