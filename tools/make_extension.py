"""The Studio's MCP server as an extension file for the Claude desktop app (run by "Make Claude Extension.bat").

    python tools/make_extension.py                 writes extension/photographic-studio.mcpb and says how to install it
    python tools/make_extension.py --root D:\\Studio --python D:\\Studio\\.venv\\Scripts\\python.exe --win
                                                   made on one computer for another (a test, or a hand-over)

Why (Bill, 2026-11-21): "Connect Studio to Claude.bat" puts the server in Claude's settings file
(claude_desktop_config.json). On Bill's PC (the Microsoft Store install) an entry there was never started by the app
(found with the RPG Forge's connector, 2026-11-19), while connectors installed as extensions are (Blender, the
forge). The app has a button for installing one from a file; this makes that file.

An extension file (.mcpb) is a zip with a manifest.json. This one holds no Studio code: it only says how to start the
server that is already in this folder, the same command, arguments and environment as connect_claude.entry(). So
the paths in it belong to THIS computer; the file is written under extension/ (never committed) and is made again on
any other computer with the same command. Nothing outside the Studio folder is written.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from tools import connect_claude  # noqa: E402

NAME = connect_claude.NAME
TOOLS = [
    ("studio_schema", "Every setting, default, choice and command, as JSON (read this first)."),
    ("studio_edit", "Any photo or picture plus any settings into a finished file."),
    ("studio_view", "Look at a picture: returns it as a small JPEG."),
    ("studio_stack", "Focus, super-resolution or panorama stacks; focus stacks can make game surfaces."),
    ("studio_dualpixel", "What a Canon Dual Pixel RAW can make: measured depth, halves, 3-D views, a surface."),
    ("studio_foliage", "Leaf cards, an atlas and glb files for games, cut out by measured depth."),
    ("studio_blender", "A Blender scene from a surface, a model or a glb."),
    ("studio_queue_add", "Long work runs on its own: add jobs, they run one after another until done."),
    ("studio_queue_status", "The queue: progress of each job and, when finished, its result."),
]


def manifest(root: str, python: str, win: bool) -> dict:
    """The extension's manifest: literal paths, no settings to fill in (the same start as connect_claude.entry())."""
    try:
        from studio import __version__ as version
    except Exception:                                        # noqa: BLE001  (only the version number is lost)
        version = "0.1.0"
    return {
        "manifest_version": "0.3",
        "name": NAME,
        "display_name": "Photographic Studio By Bill",
        "version": version,
        "description": "Develop raws, clean, match colours, cut out, depth maps, focus stacks, game surfaces and leaf "
                       "cards with the Photographic Studio on this PC.",
        "long_description": ("The Photographic Studio as a connector. Claude can develop and edit pictures, stack focus "
                             "brackets, measure depth from Canon Dual Pixel RAWs, make game surfaces, leaf cards and "
                             "Blender scenes, and run long work through the Studio's job queue so nothing is cut off. "
                             "Results are written as NAME_Studio files and never replace a picture. It starts the "
                             "Studio that is in this computer's folder (" + root + ")."),
        "author": {"name": "Photographic Studio By Bill"},
        "license": "GPL-3.0",
        "server": {
            "type": "python",
            "entry_point": "server/launch.py",
            "mcp_config": {
                "command": python,
                "args": ["-m", "studio.mcp_server"],
                "env": {"PYTHONPATH": root, "PYTHONUTF8": "1"},
            },
        },
        "tools": [{"name": n, "description": d} for n, d in TOOLS],
        "compatibility": {"platforms": ["win32"] if win else ["darwin", "linux"]},
    }


def launcher(root: str) -> str:
    """The bundle's entry point: starts the same server by file, for an app that starts an extension that way."""
    return ('"""The Photographic Studio\'s MCP server, started from the Studio folder this extension was made for\n'
            '(tools/make_extension.py). The extension\'s own settings start it directly; this does the same."""\n'
            "import os\nimport runpy\nimport sys\n\n"
            f"ROOT = {root!r}\n"
            "sys.path.insert(0, ROOT)\n"
            'os.environ.setdefault("PYTHONPATH", ROOT)\n'
            'os.environ.setdefault("PYTHONUTF8", "1")\n'
            'runpy.run_module("studio.mcp_server", run_name="__main__", alter_sys=True)\n')


def build(out: Path, root: str, python: str, win: bool) -> Path:
    """Write the extension file: made whole beside its place, read back, then moved in."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest(root, python, win), indent=2))
        z.writestr("server/launch.py", launcher(root))
    with zipfile.ZipFile(tmp) as z:
        json.loads(z.read("manifest.json").decode("utf-8"))
        compile(z.read("server/launch.py").decode("utf-8"), "launch.py", "exec")
    tmp.replace(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Make the Studio's extension file for the Claude desktop app")
    ap.add_argument("--root", default=None, help="the Studio folder on the computer the file is for")
    ap.add_argument("--python", default=None, help="the Studio's own python.exe on that computer")
    ap.add_argument("--win", action="store_true", help="the file is for Windows (with --root)")
    ap.add_argument("--out", default=None, help="where to write the file (default extension/ in this folder)")
    a = ap.parse_args(argv)
    root = a.root or str(APP_ROOT)
    win = a.win or (a.root is None and sys.platform == "win32") or (a.root is not None and ":" in a.root[:3])
    python = a.python or connect_claude.entry()["command"]
    if a.root is None and not Path(python).exists():
        print(f"  The Studio's own Python was not found ({python}). Run 'Install Photographic Studio.bat' first.")
        return 1
    out = Path(a.out) if a.out else APP_ROOT / "extension" / f"{NAME}.mcpb"
    build(out, root, python, win)
    print(f"  Made: {out}")
    print(f"  It starts: {python} -m studio.mcp_server   (PYTHONPATH={root})")
    print()
    print("  To install it in the Claude desktop app:")
    print("    1. Settings -> Desktop app -> Manage (beside Extensions).   Older layout: Settings -> Extensions.")
    print("    2. Advanced settings -> Extension Developer -> Install Extension...")
    print(f"    3. Pick this file: {out}")
    print("    4. Install. If its tools do not show, quit Claude fully (also from the tray) and open it again.")
    print()
    print("  If you also ran 'Connect Studio to Claude.bat' before, run it again and press D: the settings-file")
    print("  entry and this extension share one name, and only one of them is needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
