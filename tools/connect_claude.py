"""Connect the Studio to the Claude desktop app as an MCP server (run by "Connect Studio to Claude.bat").

Adds one entry, "photographic-studio", to Claude's own settings file (claude_desktop_config.json) so Claude can start
the Studio's MCP server (studio/mcp_server.py). Every other entry in that file is kept as it is; the file as it was is
saved beside it first. This is the one thing the Studio writes outside its own folder, and only when you run it.

    python tools/connect_claude.py              add (or refresh) the entry
    python tools/connect_claude.py --remove     take it out again
    python tools/connect_claude.py --print      only show the entry, for any other MCP client
    python tools/connect_claude.py --config FILE    use this settings file (tests)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
NAME = "photographic-studio"


def entry(python: str | None = None) -> dict:
    """How any MCP client starts the Studio's server: this copy's own Python, the Studio folder on its path."""
    return {"command": python or sys.executable, "args": ["-m", "studio.mcp_server"],
            "env": {"PYTHONPATH": str(APP_ROOT), "PYTHONUTF8": "1"}}


def windows_candidates(local: str | None = None, roaming: str | None = None) -> list:
    """Where Claude's settings folder can be on Windows, the one the app really reads first. Claude is installed
    as a Windows package (MSIX), and Windows keeps a package's "AppData\\Roaming" files in a private place:
    AppData\\Local\\Packages\\Claude_<id>\\LocalCache\\Roaming\\Claude. The plain AppData\\Roaming\\Claude is
    only used by older, non-package installs (and may not exist at all)."""
    local = Path(local or os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    roaming = Path(roaming or os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    out = []
    try:
        for pkg in sorted((local / "Packages").glob("Claude_*")):
            out.append(pkg / "LocalCache" / "Roaming" / "Claude")
    except OSError:
        pass
    out.append(roaming / "Claude")
    return out


def config_path() -> Path:
    """Claude's settings file: in the first settings folder that exists (else where it would be made)."""
    if sys.platform == "win32":
        dirs = windows_candidates()
        return next((d for d in dirs if d.is_dir()), dirs[-1]) / "claude_desktop_config.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "Claude" / "claude_desktop_config.json"


def update(path: Path, remove: bool = False, python: str | None = None) -> str:
    """Add, refresh or remove the entry. Returns a sentence saying what happened. Raises ValueError when the file
    is there but is not readable JSON (it is then left exactly as it is)."""
    doc: dict = {}
    if path.is_file():
        text = path.read_text(encoding="utf-8-sig")
        if text.strip():
            try:
                doc = json.loads(text)
            except ValueError as e:
                raise ValueError(f"{path} is not valid JSON ({e}); it was left as it is") from None
            if not isinstance(doc, dict):
                raise ValueError(f"{path} does not hold a settings object; it was left as it is")
    servers = doc.get("mcpServers")
    if servers is None:
        servers = doc["mcpServers"] = {}
    if not isinstance(servers, dict):
        raise ValueError(f"'mcpServers' in {path} is not an object; the file was left as it is")
    if remove:
        if NAME not in servers:
            return "The Studio was not connected: nothing to remove."
        del servers[NAME]
        said = "The Studio was disconnected from Claude."
    else:
        new = entry(python)
        if servers.get(NAME) == new:
            return "The Studio is already connected (nothing changed)."
        said = "The Studio's connection was refreshed." if NAME in servers else "The Studio is now connected."
        servers[NAME] = new
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        shutil.copy2(path, path.with_name(path.stem + ".before_studio.json"))
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return said


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Connect the Studio to the Claude desktop app (MCP)")
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--print", dest="show", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    snippet = json.dumps({"mcpServers": {NAME: entry()}}, indent=2)
    if a.show:
        print(snippet)
        return 0
    if APP_ROOT.name == "app" and (APP_ROOT.parent / "app_packages").is_dir():
        print("  This copy of the Studio was installed with the MSI installer, which has no separate Python for")
        print("  Claude to start. Use a copy made by 'Install Photographic Studio.bat' (or a git copy) instead.")
        return 1
    path = Path(a.config) if a.config else config_path()
    if not a.config and not path.parent.is_dir():
        print("  The Claude desktop app's settings folder was not found. Looked in:")
        for d in (windows_candidates() if sys.platform == "win32" else [path.parent]):
            print(f"    {d}")
        print("  Install and start the Claude desktop app once, then run this again. For any other MCP client,")
        print("  give it this entry:")
        print(snippet)
        return 1
    try:
        said = update(path, a.remove)
    except (ValueError, OSError) as e:
        print(f"  Could not change Claude's settings: {e}")
        print("  Nothing was changed. You can add this entry by hand (Claude: Settings > Developer > Edit Config):")
        print(snippet)
        return 1
    print("  " + said)
    print(f"  Settings file: {path}")
    if not a.remove:
        print("  Now quit the Claude desktop app completely (also from the tray, near the clock) and start it again.")
        print("  Claude can then use the Studio's tools. Pictures Claude looks at are sent to Claude, like any")
        print("  picture you attach to a chat; everything else stays on this PC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
