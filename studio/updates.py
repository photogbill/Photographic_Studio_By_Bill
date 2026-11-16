"""Updates: Help > Check for updates. Off until the user turns it on (rule: nothing uses the internet by default).

What is sent: one HTTPS request to GitHub's public API asking for the latest release of this project, and, when
an update is wanted, the download itself. No photo, file name, setting, serial number or identifier is ever sent;
GitHub sees what any web page visit shows it (the PC's internet address and a User-Agent naming the Studio and its
version). Every check and download is logged in logs/studio.log.

How a copy is updated depends on how it was installed (``install_kind``):
    "bat"     Install Photographic Studio.bat (private Python in python/), "sh" install-linux.sh (Linux), or
              "folder": any other non-git copy. The new version is downloaded and unpacked into _update/pending;
              "Start Studio.bat" (tools/apply_update.bat) or start-studio.sh on Linux puts it in place before the
              Studio next starts, so no file is ever replaced while in use.
              Settings, calibrations, logs and photos are untouched: a release holds only software and shared
              profiles, and nothing is deleted.
    "msi"     the MSI installer (Briefcase layout: the app in <install>/app, packages in <install>/app_packages).
              The new MSI is downloaded into _update/ and Windows Installer replaces the old version.
    "git"     a developer's git copy: only told, never changed (update it with git).
Everything downloaded stays inside the app folder (rule 0.3.3); for the MSI copy, in its data folder.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from . import __version__, log as slog, paths

log = slog.get("updates")

REPO = "photogbill/Photographic_Studio_By_Bill"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
MODES = ("off", "notify", "auto")
CHECK_EVERY_S = 20 * 3600          # automatic checks at most about once a day
MSI_NAME = "Photographic-Studio.msi"


# ------------------------------------------------------------------------------------------ versions
def parse_version(v) -> tuple | None:
    """'v1.2.3' / '1.2' / '0.1.0-beta' -> (1, 2, 3) / (1, 2, 0) / (0, 1, 0); None if it holds no number."""
    s = str(v or "").strip().lstrip("vV")
    parts = []
    for p in s.split(".")[:3]:
        digits = ""
        for ch in p:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    if not parts:
        return None
    return tuple(parts + [0] * (3 - len(parts)))


def is_newer(remote, local=__version__) -> bool:
    r, l = parse_version(remote), parse_version(local)
    return r is not None and l is not None and r > l


# ------------------------------------------------------------------------------------------ this copy
def install_kind(root: Path | None = None) -> str:
    """How this copy was installed: "git", "msi", "bat", "sh" or "folder" (see the module docstring)."""
    root = Path(root or paths.APP_ROOT)
    if (root / ".git").exists():
        return "git"
    if root.name == "app" and (root.parent / "app_packages").is_dir():
        return "msi"
    try:
        k = (root / ".install_kind").read_text(encoding="utf-8").strip().lower()
        if k in ("bat", "msi", "sh"):
            return k
    except OSError:
        pass
    return "folder"


def update_dir(root: Path | None = None) -> Path:
    d = Path(root or paths.DATA_ROOT) / "_update"
    d.mkdir(parents=True, exist_ok=True)
    return d


def staged(root: Path | None = None) -> str | None:
    """The version already downloaded and waiting for the next start, if any."""
    p = Path(root or paths.DATA_ROOT) / "_update" / "ready.txt"
    try:
        return p.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# ------------------------------------------------------------------------------------------ GitHub
def _request(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": f"PhotographicStudio/{__version__}",
                                               "Accept": "application/vnd.github+json"})
    return urllib.request.urlopen(req, timeout=timeout)       # noqa: S310  (https only, fixed GitHub address)


def latest(timeout: float = 10.0) -> dict:
    """The newest release on GitHub: {"tag", "name", "notes", "page", "zip", "msi", "published"}."""
    with _request(API_LATEST, timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    tag = str(data.get("tag_name") or "")
    msi = None
    for a in data.get("assets") or []:
        name = str(a.get("name") or "")
        if name.lower().endswith(".msi"):
            msi = a.get("browser_download_url")
            if name == MSI_NAME:
                break
    return {"tag": tag, "name": str(data.get("name") or tag), "notes": str(data.get("body") or "")[:4000],
            "page": str(data.get("html_url") or RELEASES_PAGE),
            "zip": f"https://github.com/{REPO}/archive/refs/tags/{tag}.zip", "msi": msi,
            "published": str(data.get("published_at") or "")}


def check(current: str = __version__, timeout: float = 10.0) -> dict:
    """Ask GitHub for the newest release. Adds "newer" (bool) and "current". Raises on network errors."""
    t0 = time.perf_counter()
    rel = latest(timeout)
    rel["current"] = current
    rel["newer"] = is_newer(rel["tag"], current)
    log.info("update check: this copy %s, newest release %s (%s) in %.1f s", current, rel["tag"] or "-",
             "newer" if rel["newer"] else "nothing new", time.perf_counter() - t0)
    return rel


def _download(url: str, dest: Path, timeout: float = 60.0, progress=None) -> Path:
    """Stream a download to ``dest`` (via a .partial file, so a broken download never looks finished)."""
    part = dest.with_name(dest.name + ".partial")
    t0, got = time.perf_counter(), 0
    with _request(url, timeout) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if progress:
                progress(got, total)
    part.replace(dest)
    log.info("downloaded %s: %.1f MB in %.1f s", url, got / 1e6, time.perf_counter() - t0)
    return dest


# ------------------------------------------------------------------------------------------ staging
def stage_folder_update(rel: dict, root: Path | None = None, progress=None) -> Path:
    """Download a release's source and unpack it into _update/pending, ready for "Start Studio.bat" to put in
    place. Returns the unpacked folder. Never for git copies."""
    root = Path(root or paths.APP_ROOT)
    if install_kind(root) == "git":
        raise RuntimeError("this is a git copy: update it with git")
    up = update_dir(root)
    (up / "ready.txt").unlink(missing_ok=True)
    pending = up / "pending"
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir()
    z = _download(rel["zip"], up / "studio.zip", progress=progress)
    base = pending.resolve()
    with zipfile.ZipFile(z) as zf:
        for m in zf.infolist():
            target = (pending / m.filename).resolve()
            if base not in target.parents and target != base:      # a path escaping the folder: refuse the zip
                raise RuntimeError(f"unsafe path in the download: {m.filename}")
        zf.extractall(pending)
    z.unlink(missing_ok=True)
    tops = [p for p in pending.iterdir() if p.is_dir()]
    if len(tops) != 1 or not (tops[0] / "studio" / "__init__.py").exists():
        shutil.rmtree(pending, ignore_errors=True)
        raise RuntimeError("the download does not look like the Studio")
    (up / "ready.txt").write_text(rel["tag"] + "\n", encoding="utf-8")
    log.info("update %s unpacked into %s: it is put in place when the Studio next starts", rel["tag"], tops[0])
    return tops[0]


def stage_msi(rel: dict, root: Path | None = None, progress=None) -> Path:
    """Download a release's MSI into _update/. Returns its path."""
    if not rel.get("msi"):
        raise RuntimeError("this release has no MSI installer")
    up = update_dir(root)
    return _download(rel["msi"], up / f"Photographic-Studio-{rel['tag'].lstrip('vV') or 'new'}.msi",
                     progress=progress)


def downloaded_msi(rel: dict, root: Path | None = None) -> Path | None:
    p = Path(root or paths.DATA_ROOT) / "_update" / f"Photographic-Studio-{rel['tag'].lstrip('vV') or 'new'}.msi"
    return p if p.exists() else None


# ------------------------------------------------------------------------------------------ handing over
def launcher_name() -> str:
    """The file that starts the Studio on this system, as the person sees it."""
    return "Start Studio.bat" if sys.platform == "win32" else "start-studio.sh"


def restart_into_update(root: Path | None = None) -> bool:
    """Start the launcher on its own ("Start Studio.bat", or start-studio.sh on Linux): it puts the staged update in
    place, then opens the Studio. The caller closes the Studio straight after."""
    root = Path(root or paths.APP_ROOT)
    if sys.platform != "win32":
        sh = root / "start-studio.sh"
        if not sh.exists():
            return False
        subprocess.Popen(["bash", str(sh)], cwd=str(root), start_new_session=True, close_fds=True,  # noqa: S603,S607
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=_restart_env())
        log.info("restarting into the update (start-studio.sh)")
        return True
    bat = root / "Start Studio.bat"
    if not bat.exists():
        return False
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(["cmd", "/c", str(bat)], cwd=str(root), creationflags=flags, close_fds=True,   # noqa: S603
                     env=_restart_env())
    log.info("restarting into the update")
    return True


def _restart_env() -> dict:
    """The new Studio waits for this one to finish closing (it holds the one-window lock until then)."""
    import os
    env = dict(os.environ)
    env["STUDIO_AFTER_UPDATE"] = "1"
    return env


def run_msi(msi: Path, passive: bool = True) -> bool:
    """Hand a downloaded MSI to Windows Installer (/passive: a progress bar, no questions). Windows only."""
    if sys.platform != "win32" or not Path(msi).exists():
        return False
    args = ["msiexec", "/i", str(msi)] + (["/passive"] if passive else [])
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(args, creationflags=flags, close_fds=True)                                    # noqa: S603
    log.info("started the MSI update: %s", msi)
    return True
