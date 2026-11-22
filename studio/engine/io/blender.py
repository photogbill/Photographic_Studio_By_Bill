"""Open the Studio's files in Blender (BUILD_PLAN_DP3D F.3): the Blender program found, never guessed, and
studio/blender/build_material.py run inside it. No Qt here.

Where Blender is: the Studio's own setting (app.json "blender_program") first; else, read only, the RPG Forge's
setting when the Forge sits beside the Studio's folders (forge/threed/blender/profiles.json "blender_program", or
forge.json threed.blender_program); else not known (the button says so and asks for it once in Settings).
"""
from __future__ import annotations

import json
import os
import subprocess

from ... import log as slog, paths

log = slog.get("engine.io.blender")

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "blender",
                      "build_material.py")


def _forge_roots() -> list:
    env = os.environ.get("STUDIO_FORGE_DIR")
    if env:
        return [env]
    out = []
    root = os.path.abspath(str(paths.APP_ROOT))
    for up in (os.path.dirname(root), os.path.dirname(os.path.dirname(root))):
        out.append(os.path.join(up, "RPG_Forge"))
    out.append(os.path.join(os.path.splitdrive(root)[0] + os.sep, "RPG_Forge"))
    return out


def from_forge() -> str:
    """The Blender the RPG Forge uses (read only), or ''."""
    for r in _forge_roots():
        for rel, keys in ((os.path.join("forge", "threed", "blender", "profiles.json"), ("blender_program",)),
                          ("forge.json", ("threed", "blender_program"))):
            p = os.path.join(r, rel)
            try:
                with open(p, encoding="utf-8") as fh:
                    d = json.load(fh)
                for k in keys:
                    d = d.get(k, {}) if isinstance(d, dict) else {}
                if isinstance(d, str) and d:
                    return d
            except (OSError, ValueError):
                continue
    return ""


def program() -> tuple[str, str]:
    """(path to blender, where it came from) or ('', reason)."""
    try:
        d = json.loads((paths.get("SETTINGS") / "app.json").read_text(encoding="utf-8"))
        own = str(d.get("blender_program") or "")
    except (OSError, ValueError):
        own = ""
    if own:
        return (own, "your setting") if os.path.isfile(own) else ("", f"the Blender in your setting is gone: {own}")
    f = from_forge()
    if f:
        return (f, "the RPG Forge's setting") if os.path.isfile(f) else ("", f"the Forge's Blender is gone: {f}")
    return "", "Blender's place is not known: set it once in Help > Blender"


def command(blender: str, target: str, out_dir: str | None = None, background: bool = False) -> list:
    """The exact command line (pure: tested without a Blender)."""
    kind = "--surface" if target.lower().endswith("_surface_studio.json") else (
        "--model" if target.lower().endswith(".json") else "--glb")
    cmd = [blender]
    if background:
        cmd.append("--background")
    cmd += ["--python", SCRIPT, "--", kind, target]
    if out_dir:
        cmd += ["--out", out_dir]
    return cmd


def run(target: str, out_dir: str | None = None, background: bool = True, timeout: float = 900) -> dict:
    """Build the scene (background: wait and return {"blend", "built", "log"}; otherwise open Blender with it)."""
    exe, how = program()
    if not exe:
        raise FileNotFoundError(how)
    cmd = command(exe, os.path.abspath(target), out_dir, background)
    log.info("blender: %s", " ".join(cmd))
    if not background:
        subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(target)))
        return {"started": True, "blender": exe, "from": how}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    built = None
    for line in (r.stdout or "").splitlines():
        if line.startswith("STUDIO_RESULT "):
            built = json.loads(line[len("STUDIO_RESULT "):])
    if r.returncode != 0 or built is None:
        raise RuntimeError(f"Blender could not build it (exit {r.returncode}): " + (r.stderr or r.stdout or "")[-600:])
    return {"built": built, "blend": built.get("blend"), "blender": exe, "from": how}
