"""tools/make_extension.py: the Studio's MCP server as an extension file (.mcpb) for the Claude desktop app."""
from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path

from tools import make_extension as X

ROOT = Path(__file__).resolve().parent.parent


def test_extension_file_starts_this_studio():
    out = ROOT / "outputs" / "pytest_extension" / uuid.uuid4().hex[:8] / "photographic-studio.mcpb"
    root, py = r"D:\Studio", r"D:\Studio\.venv\Scripts\python.exe"
    assert X.main(["--root", root, "--python", py, "--out", str(out)]) == 0
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == ["manifest.json", "server/launch.py"]
        m = json.loads(z.read("manifest.json"))
        compile(z.read("server/launch.py").decode(), "launch.py", "exec")
    assert m["name"] == "photographic-studio" and m["compatibility"]["platforms"] == ["win32"]
    cfg = m["server"]["mcp_config"]
    assert cfg == {"command": py, "args": ["-m", "studio.mcp_server"], "env": {"PYTHONPATH": root, "PYTHONUTF8": "1"}}
    assert "studio_queue_add" in [t["name"] for t in m["tools"]]
    assert not out.with_name(out.name + ".part").exists()
