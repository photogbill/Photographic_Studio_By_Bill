"""The Studio's MCP server (studio/mcp_server.py) and the script that connects it to Claude (tools/connect_claude.py)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from studio import cli_tools, mcp_server as M

ROOT = Path(__file__).resolve().parent.parent
META = {M.META_VERSION: "2026-07-28", M.META_CAPS: {}}


def _req(method, params=None, mid=1, meta=META):
    p = dict(params or {})
    if meta is not None:
        p["_meta"] = dict(meta)
    return {"jsonrpc": "2.0", "id": mid, "method": method, "params": p}


def _picture(path):
    h, w = 120, 180
    img = np.full((h, w, 3), 0.3, np.float32)
    img[40:80, 60:120] = (0.9, 0.85, 0.2)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    return str(path)


def test_discover_and_version_rules():
    s = M.Server()
    r = s.handle(_req("server/discover"))["result"]
    assert r["resultType"] == "complete" and r["supportedVersions"] == ["2026-07-28"]
    assert r["capabilities"] == {"tools": {}} and r["_meta"][M.META_SERVER]["name"] == "photographic-studio"
    e = s.handle(_req("tools/list", meta={M.META_VERSION: "1900-01-01", M.META_CAPS: {}}))["error"]
    assert e["code"] == -32022 and e["data"] == {"supported": ["2026-07-28"], "requested": "1900-01-01"}
    assert s.handle(_req("tools/list", meta={M.META_VERSION: "2026-07-28"}))["error"]["code"] == -32602
    assert s.handle(_req("server/discover", meta=None))["error"]["code"] == -32602
    assert s.handle(_req("nothing/here"))["error"]["code"] == -32601
    assert s.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None       # never answered


def test_the_older_handshake_still_works():
    s = M.Server()
    r = s.handle(_req("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}}, meta=None))["result"]
    assert r["protocolVersion"] == "2025-06-18" and r["serverInfo"]["name"] == "photographic-studio"
    assert "resultType" not in r                                    # an older client is answered in its own terms
    r = s.handle(_req("initialize", {"protocolVersion": "2031-01-01"}, meta=None))["result"]
    assert r["protocolVersion"] == M.LEGACY[0]
    assert s.handle(_req("ping", meta=None))["result"] == {}
    assert len(s.handle(_req("tools/list", meta=None))["result"]["tools"]) == len(M.tools())


def test_every_command_is_a_tool_with_a_typed_schema():
    by = {t["name"]: t for t in M.tools()}
    want = ({"studio_" + c.replace("-", "_") for c in cli_tools.COMMANDS if c not in M.SKIP_COMMANDS}
            | {"studio_view"} | set(M.QUEUE_NAMES))
    assert set(by) == want
    assert [t["name"] for t in M.tools()] == [t["name"] for t in M.tools()]          # a fixed order
    p = by["studio_restore"]["inputSchema"]
    assert p["required"] == ["inputs"] and p["additionalProperties"] is False
    assert p["properties"]["inputs"]["type"] == "array" and p["properties"]["jpeg"]["type"] == "boolean"
    assert p["properties"]["scratches"]["type"] == ["number", "boolean"]
    assert p["properties"]["format"]["enum"] == ["tiff", "png", "jpeg", "fits"] and "settings" in p["properties"]
    assert by["studio_stack"]["inputSchema"]["required"] == ["kind", "inputs"]
    assert by["studio_look_profile"]["inputSchema"]["required"] == ["inputs", "save"]
    assert not ({"json", "result", "help"} & set(p["properties"]))
    for t in by.values():                                           # names a client will accept, a description each
        assert t["description"] and all(c.isalnum() or c == "_" for c in t["name"])


def test_arguments_become_the_command_line():
    M.tools()
    cmd, parser = M._PARSERS["studio_restore"]
    argv = M.to_argv(cmd, parser, {"inputs": ["a.png", "b.png"], "scratches": True, "stains": 80, "jpeg": True,
                                   "set": ["exposure_ev=-0.2"], "settings": {"light": {"shadows": 20}}})
    assert argv[:3] == ["restore", "a.png", "b.png"]
    assert "--scratches" in argv and "--stains=80" in argv and "--jpeg" in argv and "--set=exposure_ev=-0.2" in argv
    assert '--recipe={"light": {"shadows": 20}}' in argv
    cmd, parser = M._PARSERS["studio_stack"]
    assert M.to_argv(cmd, parser, {"inputs": ["f"], "kind": "focus"})[:3] == ["stack", "focus", "f"]
    with pytest.raises(M.BadArguments, match="settings"):
        M.to_argv("edit", M._PARSERS["studio_edit"][1], {"inputs": ["a.png"], "setings": {}})


def test_a_call_runs_the_command_and_can_show_its_preview(tmp_path):
    src = _picture(tmp_path / "a.png")
    s = M.Server()
    seen = []
    s.send = seen.append
    call = _req("tools/call", {"name": "studio_edit", "arguments": {
        "inputs": [src], "settings": {"exposure_ev": 0.5}, "preview": 96, "out": str(tmp_path / "o")}},
        meta=dict(META, progressToken="t1"))
    r = s.handle(call)["result"]
    assert r["isError"] is False and r["resultType"] == "complete"
    res = r["structuredContent"]
    assert res["ok"] and json.loads(r["content"][0]["text"]) == res
    assert os.path.isfile(res["outputs"][0]["output"]) and res["outputs"][0]["output"].endswith("_Preview_Studio.jpg")
    assert r["content"][1]["type"] == "image" and r["content"][1]["mimeType"] == "image/jpeg"
    assert seen and all(n["method"] == "notifications/progress" and n["params"]["progressToken"] == "t1" for n in seen)
    v = s.handle(_req("tools/call", {"name": "studio_view", "arguments": {"path": src, "long_edge": 64}}))["result"]
    assert v["isError"] is False and json.loads(v["content"][0]["text"])["shown"] == [64, 43]
    assert v["content"][1]["type"] == "image"


def test_mistakes_come_back_as_tool_errors_the_model_can_fix(tmp_path):
    src = _picture(tmp_path / "a.png")
    s = M.Server()
    r = s.handle(_req("tools/call", {"name": "studio_edit", "arguments": {"inputs": [src], "setings": {}}}))["result"]
    assert r["isError"] and "settings" in r["content"][0]["text"]
    r = s.handle(_req("tools/call", {"name": "studio_edit", "arguments": {
        "inputs": [src], "settings": {"light": {"shadow": 3}}, "out": str(tmp_path / "o")}}))["result"]
    assert r["isError"] and "shadows" in r["structuredContent"]["problems"][0]["text"]
    r = s.handle(_req("tools/call", {"name": "studio_view", "arguments": {"path": str(tmp_path / "none.png")}}))
    assert r["result"]["isError"]
    assert s.handle(_req("tools/call", {"name": "studio_nope", "arguments": {}}))["error"]["code"] == -32602


def test_standard_output_carries_only_protocol_messages(tmp_path):
    src = _picture(tmp_path / "a.png")
    lines = [b"this is not json", json.dumps(_req("server/discover", mid="d")).encode(),
             json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode(),
             json.dumps(_req("tools/call", {"name": "studio_find", "arguments": {"inputs": [src], "what": "size"}},
                             mid=7)).encode()]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT)] + sys.path), PYTHONUTF8="1")
    p = subprocess.run([sys.executable, "-m", "studio.mcp_server"], input=b"\n".join(lines) + b"\n",
                       capture_output=True, cwd=str(tmp_path), env=env, timeout=120)
    assert p.returncode == 0                                        # it stops when its input closes
    out = [json.loads(ln) for ln in p.stdout.decode("utf-8").splitlines()]         # every line is one JSON message
    assert [m.get("id") for m in out] == [None, "d", 7] and out[0]["error"]["code"] == -32700
    found = out[2]["result"]["structuredContent"]["outputs"][0]["found"]
    assert found["size"] == {"width": 180, "height": 120}


def test_connecting_to_claude_keeps_everything_else(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import connect_claude as C
    finally:
        sys.path.pop(0)
    cfg = tmp_path / "Claude" / "claude_desktop_config.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"mcpServers": {"blender": {"command": "x"}}, "other": 1}), encoding="utf-8")
    assert "now connected" in C.update(cfg, python="py.exe")
    doc = json.loads(cfg.read_text(encoding="utf-8"))
    assert doc["other"] == 1 and doc["mcpServers"]["blender"] == {"command": "x"}
    e = doc["mcpServers"]["photographic-studio"]
    assert e["command"] == "py.exe" and e["args"] == ["-m", "studio.mcp_server"] and e["env"]["PYTHONPATH"] == str(ROOT)
    assert json.loads((cfg.parent / "claude_desktop_config.before_studio.json").read_text())["other"] == 1
    assert "already connected" in C.update(cfg, python="py.exe")
    assert "disconnected" in C.update(cfg, remove=True)
    assert "photographic-studio" not in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]
    cfg.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError):
        C.update(cfg, python="py.exe")
    assert cfg.read_text(encoding="utf-8") == "{ not json"          # a file it cannot read is never touched
    assert C.main(["--print"]) == 0
    # Windows: the packaged app's own folder comes before the plain AppData\\Roaming one
    local, roaming = tmp_path / "Local", tmp_path / "Roaming"
    pkg = local / "Packages" / "Claude_pzs8sxrjxfjjc" / "LocalCache" / "Roaming" / "Claude"
    pkg.mkdir(parents=True)
    assert C.windows_candidates(str(local), str(roaming)) == [pkg, roaming / "Claude"]
    assert C.windows_candidates(str(tmp_path / "none"), str(roaming)) == [roaming / "Claude"]
