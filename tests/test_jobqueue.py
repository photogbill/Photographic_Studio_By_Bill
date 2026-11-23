"""The job queue (studio/jobqueue.py), its MCP tools (studio_queue_*) and the queue command.

The worker is run in this process with a stand-in child command (a tiny Python script with a known outcome), so the
tests check the queue's own logic: order, states, a crash failing one job only, cancel, a dead worker detected."""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from studio import cli, cli_tools, jobqueue as Q, mcp_server as M

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def jobs_dir(monkeypatch):
    d = ROOT / "outputs" / "pytest_jobs" / uuid.uuid4().hex[:8]
    monkeypatch.setenv("STUDIO_JOBS_DIR", str(d))
    monkeypatch.setattr(Q, "IDLE_EXIT_S", 0.5)
    monkeypatch.setattr(Q, "POLL_S", 0.05)
    return d


def _fake_child(monkeypatch):
    """Job argv ['ok'|'fail'|'crash'|'sleep', ...] -> a script with that outcome (the result JSON on stdout)."""
    script = (
        "import json, sys, time\n"
        "kind = sys.argv[1]\n"
        "sys.stderr.write('working on ' + kind + '\\r50%\\rdone\\n')\n"
        "if kind == 'sleep': time.sleep(30)\n"
        "if kind == 'crash': sys.exit(3221225477 if sys.platform == 'win32' else 139)\n"
        "ok = kind == 'ok'\n"
        "print(json.dumps({'ok': ok, 'outputs': [{'output': 'x_Studio.png'}] if ok else [],\n"
        "                  'problems': [] if ok else [{'text': 'it went wrong'}]}))\n"
        "sys.exit(0 if ok else 1)\n")
    monkeypatch.setattr(Q, "child_command", lambda argv: [sys.executable, "-c", script] + list(argv))


def test_add_keeps_order_and_states(jobs_dir):
    a = Q.add(["edit", "a.png"], title="first")
    b = Q.add(["edit", "b.png"], hold=True)
    ids = [j["id"] for j in Q.jobs()]
    assert ids == [a["id"], b["id"]]
    assert Q.get(a["id"])["state"] == "queued" and Q.get(b["id"])["state"] == "held"
    st = Q.status()
    assert st["counts"]["queued"] == 1 and st["counts"]["held"] == 1 and not st["worker"]["alive"]
    assert Q.cancel(b["id"]) == [b["id"]] and Q.get(b["id"])["state"] == "cancelled"
    with pytest.raises(ValueError):
        Q.cancel("nope")
    with pytest.raises(ValueError):
        Q.get("../escape")
    with pytest.raises(ValueError):
        Q.add([])


def test_worker_runs_every_job_and_a_crash_fails_only_that_job(jobs_dir, monkeypatch):
    _fake_child(monkeypatch)
    j1 = Q.add(["ok"])
    j2 = Q.add(["crash"])
    j3 = Q.add(["fail"])
    j4 = Q.add(["ok"])
    assert Q.run_worker() == 0
    s = {j: Q.get(j)["state"] for j in (j1["id"], j2["id"], j3["id"], j4["id"])}
    assert s == {j1["id"]: "done", j2["id"]: "failed", j3["id"]: "failed", j4["id"]: "done"}
    assert "stopped unexpectedly" in Q.get(j2["id"])["problem"] or "no result" in Q.get(j2["id"])["problem"]
    assert Q.get(j3["id"])["problem"] == "it went wrong"
    one = Q.status(j1["id"])
    assert one["result"]["ok"] is True and one["outputs"] == 1
    assert one["progress"][-1] == "done"
    assert not Q.worker_info()["alive"]                       # stopped once the queue was empty


def test_cancel_stops_a_running_job(jobs_dir, monkeypatch):
    _fake_child(monkeypatch)
    j = Q.add(["sleep"])
    t = threading.Thread(target=Q.run_worker)
    t.start()
    for _ in range(100):
        if Q.get(j["id"])["state"] == "running":
            break
        time.sleep(0.05)
    assert Q.worker_info()["alive"] and Q.worker_info()["job"] == j["id"]
    Q.cancel(j["id"])
    t.join(40)
    assert not t.is_alive()
    assert Q.get(j["id"])["state"] == "cancelled"


def test_a_dead_worker_is_detected_and_its_job_failed(jobs_dir, monkeypatch):
    _fake_child(monkeypatch)
    Q.root().joinpath("worker.json").write_text(json.dumps({"pid": 1, "beat": time.time() - 3600, "job": "x"}))
    assert not Q.worker_info()["alive"]
    j = Q.add(["ok"])
    job = Q.get(j["id"])
    job["state"] = "running"                                 # left running by a worker that died with the PC
    Q._save(job)
    Q.run_worker()
    assert Q.get(j["id"])["state"] == "failed" and "worker stopped" in Q.get(j["id"])["problem"]


def test_ensure_worker_starts_only_when_work_waits(jobs_dir, monkeypatch):
    started = []
    monkeypatch.setattr(Q.subprocess, "Popen", lambda *a, **k: started.append(a[0]))
    assert Q.ensure_worker() is False                        # nothing queued
    Q.add(["edit", "a.png"], hold=True)
    assert Q.ensure_worker() is False                        # held only
    r = Q.start()
    assert r["worker_started"] is True and len(r["released"]) == 1
    assert started and started[-1][-2:] == ["studio.jobqueue", "--worker"]


def test_mcp_queue_tools(jobs_dir, monkeypatch):
    monkeypatch.setattr(Q, "ensure_worker", lambda: False)
    names = [t["name"] for t in M.tools()]
    for n in M.QUEUE_NAMES:
        assert n in names
    assert "studio_queue" not in names                       # the command line's queue is not a second tool
    r = M.call_tool("studio_queue_add", {"jobs": [
        {"tool": "studio_edit", "arguments": {"inputs": ["a.png"], "out": "o"}, "title": "edit a"},
        {"tool": "studio_stack", "arguments": {"kind": "focus", "inputs": ["a.CR3", "b.CR3"], "make": "surface"}}],
        "hold": True})
    assert not r["isError"], r
    added = r["structuredContent"]["added"]
    assert [a["state"] for a in added] == ["held", "held"] and added[0]["title"] == "edit a"
    argv = Q.get(added[1]["id"])["argv"]
    assert argv[:2] == ["stack", "focus"] and "--make=surface" in argv
    bad = M.call_tool("studio_queue_add", {"tool": "studio_edit", "arguments": {"inptus": ["a.png"]}})
    assert bad["isError"] and "inputs" in bad["content"][0]["text"]
    assert M.call_tool("studio_queue_add", {"tool": "studio_nothing", "arguments": {}})["isError"]
    st = M.call_tool("studio_queue_status", {})["structuredContent"]
    assert st["counts"]["held"] == 2
    one = M.call_tool("studio_queue_status", {"id": added[0]["id"]})["structuredContent"]
    assert one["state"] == "held"
    c = M.call_tool("studio_queue_cancel", {"id": "all"})["structuredContent"]
    assert sorted(c["cancelled"]) == sorted(a["id"] for a in added)
    assert M.call_tool("studio_queue_status", {"id": "missing"})["isError"]


def test_mcp_server_routes_queue_tools(jobs_dir):
    s = M.Server()
    meta = {M.META_VERSION: "2026-07-28", M.META_CAPS: {}}
    r = s.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": "studio_queue_status", "arguments": {}, "_meta": meta}})
    assert r["result"]["isError"] is False


def test_queue_command_line(jobs_dir, monkeypatch):
    monkeypatch.setattr(Q, "ensure_worker", lambda: False)
    a = cli.build_parser().parse_args(["queue", "--json", "add", "--hold", "edit", "a.png", "--out", "o"])
    run = cli_tools.execute(a, quiet=True)
    res = cli_tools.plain(run.res, True)
    assert res["ok"], res
    jid = res["data"]["job"]["id"]
    assert Q.get(jid)["argv"] == ["edit", "a.png", "--out", "o"] and Q.get(jid)["state"] == "held"
    a = cli.build_parser().parse_args(["queue", "status", jid])
    assert cli_tools.plain(cli_tools.execute(a, quiet=True).res, True)["data"]["state"] == "held"
    a = cli.build_parser().parse_args(["queue", "add", "queue", "status"])
    assert not cli_tools.plain(cli_tools.execute(a, quiet=True).res, True)["ok"]


def test_tail_reads_progress_lines(jobs_dir):
    jobs_dir.mkdir(parents=True, exist_ok=True)
    p = jobs_dir / "progress.log"
    p.write_bytes(b"a\r10%\r20%\nb\n\n")
    assert Q.tail(p, 2) == ["20%", "b"]
    assert Q.tail(jobs_dir / "none.log") == []


def test_a_bare_out_file_goes_in_out_not_the_current_folder(jobs_dir):
    from types import SimpleNamespace
    from studio.engine.io.export import ExportSpec
    d = jobs_dir / "out_here"
    a = SimpleNamespace(out_file="cut.png", out=str(d), tag=None)
    run = SimpleNamespace(taken=set())
    p = cli_tools.target(run, a, str(jobs_dir / "src.jpg"), ExportSpec(format="png"))
    assert Path(p).parent == d and Path(p).name == "cut.png"
    p2 = cli_tools.target(run, a, str(jobs_dir / "src.jpg"), ExportSpec(format="png"), "Cutout", ".png")
    assert Path(p2).parent == d
