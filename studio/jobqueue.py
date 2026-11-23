"""The Studio's job queue: long work (stacks, models, leaves, Blender scenes) runs to its end on its own.

    python -m studio.jobqueue --worker       the worker (started by itself when a job is added; never by hand)

Why (Bill, 2026-11-21): a client such as Claude waits only so long for one tool call (about 80 s over the desktop
app's bridge), and a full-size focus stack takes 10-20 minutes, so a stack run as one call was cut off part way.
Now a caller adds jobs and gets an answer at once; a worker started in the background takes them one after another,
oldest first, until the queue is empty, and the caller looks in when it likes (``status``).

  * Every job is one command line of the Studio (``stack focus ... --make surface``), run in a fresh Studio
    process: memory is given back after each job, and a crash in one job (a native library, a damaged raw) fails
    that job only; the worker carries on with the next.
  * Everything lives in the ``jobs`` folder beside the app (``STUDIO_JOBS_DIR`` overrides; never C:): one folder per
    job with ``job.json`` (its state), ``progress.log`` (the command's progress lines), ``result.json`` (the
    command's own JSON result). ``worker.json`` is the worker's heartbeat: a worker whose heartbeat is older than
    ``STALE_S`` is taken as gone, so a worker killed with the PC is replaced by the next ``add`` or ``start``.
  * States: held (added with hold, waits for ``start``), queued, running, done, failed, cancelled.
  * The worker runs below normal priority on Windows, so the PC stays usable while it works.
  * Nothing here deletes anything; finished jobs stay listed (newest last) until their folders are moved away.

The MCP server's ``studio_queue_*`` tools and the ``queue`` command (docs/CLI.md, "The job queue") are the two
ways in. No Qt (rule 2.2).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import log as slog, paths

log = slog.get("queue")

STATES = ("held", "queued", "running", "done", "failed", "cancelled")
WAITING = ("held", "queued")
FINISHED = ("done", "failed", "cancelled")
HEARTBEAT_S = 3.0          # how often a working worker says it is alive
STALE_S = 45.0             # a heartbeat older than this: the worker is gone
IDLE_EXIT_S = 90.0         # an idle worker stops after this long with nothing queued
POLL_S = 2.0
CRASH_CODES = {3221225477: "access violation (0xC0000005)", 3221225725: "stack overflow (0xC00000FD)",
               3221226505: "fatal error (0xC0000409)", -11: "segmentation fault", -6: "aborted"}


# ------------------------------------------------------------------------------------------ files
def root() -> Path:
    return paths.get("JOBS")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write(path: Path, data: dict):
    """Written whole beside its place, then moved in. Windows refuses the move for a moment while another process
    (the MCP server reading the status) has the file open: tried again for up to two seconds."""
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    for i in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 39:
                raise
            time.sleep(0.05)


def _read(path: Path):
    """The file's JSON, or None when there is none. A read that meets the file being replaced is tried again."""
    for i in range(20):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            if i >= 3:                                # really not there (a brief gap while replaced: tried 4 times)
                return None
        except (OSError, ValueError):                 # locked for a moment, or caught half-replaced
            pass
        time.sleep(0.05)
    return None


def _job_file(jid: str) -> Path:
    if not jid or any(c in jid for c in "/\\:") or jid.startswith("."):
        raise ValueError(f"no job {jid!r}")
    return root() / jid / "job.json"


def tail(path: Path, n: int = 8) -> list:
    """The last ``n`` non-empty lines of a progress file (progress lines overwrite themselves with \\r)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 64 * 1024))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n")]
    return [ln for ln in lines if ln][-n:] if n > 0 else []


# ------------------------------------------------------------------------------------------ the queue
def add(argv: list, title: str = "", hold: bool = False) -> dict:
    """Put one Studio command line in the queue. Returns the job (its ``id`` is what ``status`` and ``cancel`` take)."""
    argv = [str(a) for a in argv]
    if not argv:
        raise ValueError("a job needs a command line, e.g. ['stack', 'focus', 'a.CR3', 'b.CR3', '--make', 'surface']")
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = root() / jid
    d.mkdir(parents=True)
    job = {"id": jid, "title": title or " ".join(argv[:2]), "argv": argv,
           "state": "held" if hold else "queued", "added": _now(), "seq": time.time()}
    _write(d / "job.json", job)
    log.info("queue: added %s %s (%s)", jid, job["title"], job["state"])
    return job


def jobs() -> list:
    """Every job, oldest first."""
    out = []
    try:
        names = os.listdir(root())
    except OSError:
        return out
    for n in names:
        j = _read(root() / n / "job.json")
        if isinstance(j, dict) and j.get("id") == n:
            out.append(j)
    out.sort(key=lambda j: j.get("seq", 0))
    return out


def get(jid: str) -> dict | None:
    return _read(_job_file(jid))


def _save(job: dict):
    _write(_job_file(job["id"]), job)


def start(ids=None) -> dict:
    """Release held jobs (all, or the ids given) and make sure a worker is working."""
    released = []
    for j in jobs():
        if j.get("state") == "held" and (ids is None or j["id"] in ids):
            j["state"] = "queued"
            _save(j)
            released.append(j["id"])
    started = ensure_worker()
    return {"released": released, "worker_started": started, "worker": worker_info()}


def cancel(jid: str) -> list:
    """Cancel one job, or ``all`` waiting and running jobs. A running job is stopped by the worker within seconds."""
    done = []
    for j in jobs():
        if jid != "all" and j["id"] != jid:
            continue
        if j.get("state") in WAITING:
            j.update(state="cancelled", finished=_now())
            _save(j)
            done.append(j["id"])
        elif j.get("state") == "running":
            (root() / j["id"] / "cancel").write_text(_now(), encoding="utf-8")
            done.append(j["id"])
    if jid != "all" and not done and get(jid) is None:
        raise ValueError(f"no job {jid!r}")
    return done


def summary(j: dict) -> dict:
    """A job in a few keys, for a list."""
    keys = ("id", "title", "state", "added", "started", "finished", "seconds", "exit", "ok", "outputs", "problem")
    return {k: j[k] for k in keys if k in j}


def status(jid: str | None = None, lines: int = 8, last: int = 40) -> dict:
    """The queue at a glance (counts, the worker, the last ``last`` jobs), or one job in full: its progress lines
    and, once finished, the command's own JSON result."""
    if jid:
        j = get(jid)
        if j is None:
            raise ValueError(f"no job {jid!r}")
        d = root() / jid
        out = dict(j, progress=tail(d / "progress.log", lines))
        if j.get("state") in FINISHED:
            res = _read(d / "result.json")
            if res is not None:
                out["result"] = res
        return out
    allj = jobs()
    counts = {s: sum(1 for j in allj if j.get("state") == s) for s in STATES}
    shown = []
    for j in allj[-last:]:
        s = summary(j)
        if j.get("state") == "running":
            s["progress"] = tail(root() / j["id"] / "progress.log", 2)
        shown.append(s)
    return {"counts": counts, "worker": worker_info(), "jobs": shown, "folder": str(root())}


# ------------------------------------------------------------------------------------------ the worker
def worker_info() -> dict:
    hb = _read(root() / "worker.json") or {}
    age = time.time() - float(hb.get("beat", 0) or 0)
    alive = bool(hb) and hb.get("state") != "stopped" and age < STALE_S
    return {"alive": alive, "pid": hb.get("pid"), "job": hb.get("job") if alive else None,
            "since": hb.get("since"), "last_beat_s": round(age, 1) if hb else None}


def _python() -> str:
    """The Studio's own Python, as a console-less program on Windows when there is one beside it."""
    exe = sys.executable
    if sys.platform == "win32":
        w = Path(exe).with_name("pythonw.exe")
        if w.exists():
            return str(w)
    return exe


def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(paths.APP_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONUTF8", "1")
    return env


def _spawn_flags(detached: bool) -> dict:
    if sys.platform != "win32":
        return {"start_new_session": True} if detached else {}
    f = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    if detached:
        f |= subprocess.CREATE_NEW_PROCESS_GROUP
    return {"creationflags": f}


def ensure_worker() -> bool:
    """Start a worker unless one is alive and nothing waits. True when one was started."""
    if worker_info()["alive"]:
        return False
    if not any(j.get("state") == "queued" for j in jobs()):
        return False
    cmd = [_python(), "-m", "studio.jobqueue", "--worker"]
    base = dict(cwd=str(paths.APP_ROOT), env=_env(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, close_fds=True)
    kw = _spawn_flags(True)
    if sys.platform == "win32":
        # Leave the caller's Windows job object when allowed, so the worker outlives the program that started it
        # (a client may close its connector); when the job object does not allow it, start inside it.
        try:
            subprocess.Popen(cmd, **base, creationflags=kw["creationflags"] | 0x01000000)  # CREATE_BREAKAWAY_FROM_JOB
            log.info("queue: worker started (outside the caller's job object)")
            return True
        except OSError as e:
            log.info("queue: breakaway not allowed (%s): worker started inside the caller's job object", e)
    subprocess.Popen(cmd, **base, **kw)
    log.info("queue: worker started")
    return True


class _Worker:
    def __init__(self):
        self.pid = os.getpid()
        self.since = _now()
        self.job = None
        self._last = 0.0

    def beat(self, force: bool = False, state: str = "working"):
        if force or time.time() - self._last >= HEARTBEAT_S:
            _write(root() / "worker.json", {"pid": self.pid, "since": self.since, "beat": time.time(),
                                            "job": self.job, "state": state})
            self._last = time.time()

    def mine(self) -> bool:
        hb = _read(root() / "worker.json") or {}
        return hb.get("pid") == self.pid


def child_command(argv: list) -> list:
    """The command line that runs one job (a fresh Studio process). Tests replace this."""
    return [sys.executable, "-m", "studio"] + list(argv)


def _result_of(stdout_path: Path):
    try:
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for ln in reversed(text.strip().splitlines()):
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                return json.loads(ln)
            except ValueError:
                return None
    return None


def run_one(w: _Worker, job: dict):
    d = root() / job["id"]
    argv = list(job["argv"])
    if "--json" not in argv:
        argv.append("--json")
    job.update(state="running", started=_now())
    _save(job)
    w.job = job["id"]
    w.beat(force=True)
    t0 = time.time()
    cancelled = False
    log.info("queue: running %s: %s", job["id"], " ".join(argv)[:400])
    with open(d / "stdout.txt", "wb") as out, open(d / "progress.log", "ab") as err:
        try:
            p = subprocess.Popen(child_command(argv), cwd=str(paths.APP_ROOT), env=_env(), stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err, **_spawn_flags(False))
        except OSError as e:
            job.update(state="failed", finished=_now(), problem=f"the job could not start: {e}")
            _save(job)
            slog.save_error_report(f"Job queue: starting job {job['id']}", str(e))
            return
        job["pid"] = p.pid
        _save(job)
        while p.poll() is None:
            w.beat()
            if (d / "cancel").exists() and not cancelled:
                cancelled = True
                log.info("queue: cancelling %s", job["id"])
                p.terminate()
                try:
                    p.wait(20)
                except subprocess.TimeoutExpired:
                    p.kill()
            time.sleep(0.5)
    code = p.returncode
    res = _result_of(d / "stdout.txt")
    if res is not None:
        _write(d / "result.json", res)
    job.update(finished=_now(), seconds=round(time.time() - t0, 1), exit=code)
    if res is not None:
        job["ok"] = bool(res.get("ok"))
        job["outputs"] = sum(1 for o in res.get("outputs", []) if isinstance(o, dict) and o.get("output"))
        probs = [p_.get("text", "") for p_ in res.get("problems", []) if isinstance(p_, dict)]
        if probs:
            job["problem"] = probs[0][-600:]
    if cancelled:
        job["state"] = "cancelled"
    elif code == 0 and res is not None and res.get("ok", False):
        job["state"] = "done"
    else:
        job["state"] = "failed"
        if code in CRASH_CODES:
            job["problem"] = (f"the Studio stopped unexpectedly: {CRASH_CODES[code]}. Last lines: "
                              + " | ".join(tail(d / "progress.log", 3)))
        elif res is None and "problem" not in job:
            job["problem"] = f"no result (exit {code}). Last lines: " + " | ".join(tail(d / "progress.log", 3))
    _save(job)
    w.job = None
    log.info("queue: %s %s in %.0f s (exit %s)", job["id"], job["state"], job["seconds"], code)


def run_worker() -> int:
    """Take queued jobs one after another, oldest first, until none has been waiting for ``IDLE_EXIT_S``."""
    slog.setup(console_level=None, filename="queue.log")
    if worker_info()["alive"]:
        log.info("queue: a worker is already working; this one stops")
        return 0
    w = _Worker()
    w.beat(force=True)
    time.sleep(0.3)
    if not w.mine():                                  # two started at once: the other one won
        return 0
    log.info("queue: worker %s started", w.pid)
    for j in jobs():                                  # a job left "running" by a worker that died with the PC
        if j.get("state") == "running":
            j.update(state="failed", finished=_now(), problem="the worker stopped while this job ran (PC shut down "
                     "or the worker was ended); add it again to redo it")
            _save(j)
    idle = time.time()
    try:
        while True:
            w.beat()
            if not w.mine():                          # replaced by another worker (this one was taken as gone)
                log.info("queue: worker %s replaced; stopping", w.pid)
                return 0
            nxt = next((j for j in jobs() if j.get("state") == "queued"), None)
            if nxt is None:
                if time.time() - idle > IDLE_EXIT_S:
                    break
                time.sleep(POLL_S)
                continue
            try:
                run_one(w, nxt)
            except Exception as e:                    # noqa: BLE001  (a fault of ours: that job fails, the queue goes on)
                slog.log_exception(f"queue: job {nxt.get('id')} failed in the worker", log)
                rep = slog.save_error_report(f"Job queue: job {nxt.get('id')}")
                nxt.update(state="failed", finished=_now(), problem=f"the worker failed on this job: {e}"
                           + (f" (error report: {rep})" if rep else ""))
                _save(nxt)
                w.job = None
            idle = time.time()
    finally:
        if w.mine():
            w.beat(force=True, state="stopped")
        log.info("queue: worker %s stopped", w.pid)
    return 0


# ------------------------------------------------------------------------------------------ the command line
def command(action: str, rest: list) -> dict:
    """``queue add|start|status|cancel|list ...`` (cli_tools.cmd_queue): the result's data."""
    rest = list(rest or [])
    if action == "add":
        hold = "--hold" in rest[:1]
        if hold:
            rest = rest[1:]
        if rest[:1] == ["--"]:
            rest = rest[1:]
        if rest and rest[0] == "queue":
            raise ValueError("a queue job cannot be another queue command")
        job = add(rest, hold=hold)
        started = False if hold else ensure_worker()
        return {"job": summary(job), "worker_started": started}
    if action == "start":
        return start(rest or None)
    if action in ("status", "list"):
        return status(rest[0] if rest else None)
    if action == "cancel":
        if not rest:
            raise ValueError("cancel needs a job id, or all")
        return {"cancelled": cancel(rest[0])}
    raise ValueError(f"no queue action {action!r}: add, start, status, cancel")


if __name__ == "__main__":
    if "--worker" in sys.argv[1:]:
        sys.exit(run_worker())
    print(__doc__)
    sys.exit(2)
