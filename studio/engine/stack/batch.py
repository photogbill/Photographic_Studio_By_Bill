"""Folders of stacks in one go (BUILD_PLAN_DP3D A.5).

Bill shoots dozens of stacks in an afternoon (shoot 2: 56 stacks, 382 Dual Pixel frames) and then wants surfaces,
then foliage, then models from the same stacks. This finds the stacks in a folder or on a card (Card to Stacks'
sorter, with its frame judging), and runs a list of jobs over them with a queue that resumes after a crash or a
Cancel: ``stacks_job.json`` in the output folder says what is done, what failed and why, so a second run carries on
where the first stopped (and the RPG Forge's queue can read it while it runs).

Jobs: ``surface`` and ``depth`` (the Depth Stack surfaces, BUILD_PLAN_DEPTH_STACK D) for every focus stack and
every single Dual Pixel photo (``surface`` also for one ordinary photo or a burst's sharpest frame: an ESTIMATED
surface, every map amber, for the RPG Forge's texture folders); ``foliage`` and ``model`` are added by their own builds (D, E) and are named here
so a job file can ask for them. Each focus stack is analysed once (the analysis cache, A.4) whatever jobs follow.
No Qt here.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ... import log as slog

log = slog.get("engine.stack.batch")

JOBS = ("surface", "depth", "foliage", "relief", "model")
JOB_FILE = "stacks_job.json"
DEFAULT_SUBFOLDER = "Studio surfaces"


@dataclass
class StackInfo:
    name: str                          # FIRST to LAST (or the one photo's name)
    kind: str                          # focus | burst | single
    paths: list = field(default_factory=list)          # frames used
    left_out: list = field(default_factory=list)       # [{"name", "reason"}]
    dual_pixel: int = 0                # frames that are Dual Pixel RAWs
    group: str | None = None           # the object it belongs to (set once views are matched: 3D Scan)
    note: str = ""
    best: str | None = None            # a burst's sharpest frame (an estimated surface is made from it)

    @property
    def base(self) -> str:
        from .frames import result_base
        return result_base(self.paths) if self.paths else self.name

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frames"] = [os.path.basename(p) for p in self.paths]
        d["base"] = self.base
        return d


def find_stacks(items, strictness: str = "normal", progress=None, cancel=None, dual_check: bool = True) -> list:
    """The stacks in files and folders (searched down through sub-folders), in card order: focus stacks with
    the frames Card to Stacks would use and those it leaves out (with reasons), bursts, and single shots."""
    from . import frames as F
    from . import sorter as SO
    paths = SO.find_paths(items)
    groups = SO.sort_frames(paths, strictness, progress=progress, cancel=cancel)
    out = []
    for g in groups:
        used = [f.path for f in g.used] if g.kind == "focus" else [f.path for f in g.frames]
        name = g.frames[0].name if len(g.frames) == 1 else f"{g.frames[0].name} to {g.frames[-1].name}"
        si = StackInfo(name=os.path.splitext(name)[0] if len(g.frames) == 1 else name, kind=g.kind, paths=used,
                       left_out=[{"name": f.name, "reason": f.reason} for f in g.left_out] if g.kind == "focus"
                       else [], note=g.note)
        if dual_check:
            si.dual_pixel = sum(1 for p in used if F.is_dual(p))
        if g.kind == "burst" and g.frames:
            sharp = [(float(np.nansum(f.energy)) if f.energy is not None else 0.0, f.path) for f in g.frames]
            si.best = max(sharp)[1]
        out.append(si)
    return out


# ------------------------------------------------------------------------------------------ the job file
def _load_job(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_job(path: str, doc: dict):
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)


def job_key(si: StackInfo, job: str) -> str:
    return f"{job}:{si.base}"


def run(stacks: list, jobs: list, out_dir: str, surface_params=None, focus_params=None, progress=None,
        cancel=None, resume: bool = True, workers: int | None = None, on_done=None) -> dict:
    """Every stack through every job, resumable. Returns the job document ({"jobs", "state": "done" | "stopped",
    "items": {key: {"status", "outputs", "error", "stack", "job"}}}); ``on_done(key, item)`` is called after each
    item (the command line reports it)."""
    from . import focus as FS
    from . import surface as SURF
    bad = [j for j in jobs if j not in JOBS]
    if bad:
        raise ValueError(f"no such job: {', '.join(bad)} (choose from {', '.join(JOBS)})")
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, JOB_FILE)
    doc = _load_job(jp) if resume else {}
    items = doc.get("items", {}) if doc.get("jobs") == list(jobs) or resume else {}
    doc = {"written_by": "Photographic Studio By Bill", "jobs": list(jobs), "out_dir": out_dir,
           "started": doc.get("started") or _dt.datetime.now().isoformat(timespec="seconds"),
           "state": "running", "items": items}
    _save_job(jp, doc)
    todo = []
    for si in stacks:
        for job in jobs:
            key = job_key(si, job)
            it = items.get(key)
            if it and it.get("status") == "done" and all(os.path.isfile(p) for p in it.get("outputs", [])):
                continue
            if not _applies(si, job):
                items[key] = {"status": "skipped", "stack": si.name, "job": job, "outputs": [],
                              "error": _why_not(si, job)}
                continue
            todo.append((si, job, key))
    _save_job(jp, doc)
    taken: set = set()
    n = len(todo)
    for k, (si, job, key) in enumerate(todo):
        if cancel is not None and cancel.is_set():
            doc["state"] = "stopped"
            _save_job(jp, doc)
            return doc
        if progress is not None:
            try:
                progress(k / max(n, 1), f"{job}: {si.name} ({k + 1} of {n})")
            except Exception:                        # noqa: BLE001  (a display must not stop the job)
                pass
        items[key] = {"status": "running", "stack": si.name, "job": job, "outputs": []}
        _save_job(jp, doc)
        try:
            outs, notes = _one(si, job, out_dir, surface_params, focus_params, taken, cancel, workers)
            items[key] = {"status": "done", "stack": si.name, "job": job, "outputs": outs, "notes": notes,
                          "finished": _dt.datetime.now().isoformat(timespec="seconds")}
        except FS.Cancelled:
            items[key] = {"status": "pending", "stack": si.name, "job": job, "outputs": []}
            doc["state"] = "stopped"
            _save_job(jp, doc)
            return doc
        except NotImplementedError as e:
            items[key] = {"status": "skipped", "stack": si.name, "job": job, "outputs": [], "error": str(e)}
        except Exception as e:                       # noqa: BLE001  (one stack fails; the rest go on)
            expected = slog.expected_problem(e) or isinstance(e, ValueError)
            if not expected:
                slog.log_exception(f"stacks job {key}", log)
                slog.save_error_report(f"Stacks: {job} for {si.name}")
            items[key] = {"status": "failed", "stack": si.name, "job": job, "outputs": [],
                          "error": str(e)[:300] or type(e).__name__}
        _save_job(jp, doc)
        if on_done is not None:
            on_done(key, items[key])
    doc["state"] = "done"
    doc["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    _save_job(jp, doc)
    return doc


def _applies(si: StackInfo, job: str) -> bool:
    if job == "surface":                             # one ordinary photo or a burst: an ESTIMATED surface (amber)
        return (si.kind == "focus" and len(si.paths) >= 2) or len(si.paths) == 1 or \
            (si.kind == "burst" and bool(si.best or si.paths))
    if job == "relief":                              # one view's measured depth as a front-only relief mesh
        return (si.kind == "focus" and len(si.paths) >= 2) or (len(si.paths) == 1 and si.dual_pixel == 1)
    if job == "depth":
        return (si.kind == "focus" and len(si.paths) >= 2) or (len(si.paths) == 1 and si.dual_pixel == 1)
    if job == "foliage":
        return (si.kind == "focus" and len(si.paths) >= 2) or (len(si.paths) == 1 and si.dual_pixel == 1)
    if job == "model":
        return si.kind == "focus" and len(si.paths) >= 2
    return False


def _why_not(si: StackInfo, job: str) -> str:
    if si.kind == "burst":
        return "a burst (the focus did not move): not a focus stack"
    if len(si.paths) == 1 and not si.dual_pixel:
        return "one ordinary photo: depth needs a focus stack or a Dual Pixel RAW"
    if si.kind == "focus" and len(si.paths) < 2:
        return "too few good frames left to stack"
    return "not for this job"


def _one(si: StackInfo, job: str, out_dir: str, sp, fp, taken: set, cancel, workers) -> tuple:
    from . import focus as FS
    from . import surface as SURF
    sp = sp or SURF.SurfaceParams()
    if job in ("surface", "depth"):
        kinds = SURF.kinds_from(sp) if job == "surface" else SURF.DEPTH_KINDS
        one = si.best if si.kind == "burst" else (si.paths[0] if len(si.paths) == 1 else None)
        if one is not None and (si.kind == "burst" or not si.dual_pixel):
            from .frames import result_base
            r = SURF.photo_surface(one, out_dir, result_base([one]), sp, kinds, taken)
            return [v for v in r["paths"].values()], list(r["json"].get("notes", []))
        if len(si.paths) == 1:
            from ..dualpixel import decode as DD
            dp = DD.open_dual(si.paths[0])
            try:
                r = SURF.dual_surface(dp, out_dir, si.base, sp, kinds, taken)
            finally:
                try:
                    dp.image.drop_cache()
                except Exception:                    # noqa: BLE001
                    pass
        else:
            p = FS.FocusParams.from_dict((fp or FS.FocusParams()).to_dict())
            an = FS.analyse(si.paths, p, None, cancel, cache=True, workers=workers)
            try:
                r = SURF.stack_surface(an, out_dir, si.base, sp, kinds, None, cancel, taken, p.method)
            finally:
                an.close()
        return [v for v in r["paths"].values()], list(r["json"].get("notes", []))
    if job == "foliage":
        from . import foliage as FOL                   # built in BUILD_PLAN_DP3D D
        return FOL.batch_job(si, out_dir, sp, fp, taken, cancel, workers)
    if job == "relief":
        from ..scan import relief as RL
        from ..scan import views as VW
        if len(si.paths) == 1:
            v = VW.from_dual(si.paths[0])
        else:
            v = VW.from_stack(si.paths, si.base, FS.FocusParams.from_dict((fp or FS.FocusParams()).to_dict()),
                              workers, True, None, cancel)
        r = RL.build(v, out_dir, si.base, getattr(sp, "relief_depth", None) or None, taken=taken)
        return [v_ for v_ in r["paths"].values()], []
    if job == "model":
        raise NotImplementedError("models are made per object from several stacks: use the 'model' command (3D "
                                  "Scan) on the object's stacks")
    raise ValueError(job)


def summary(doc: dict) -> dict:
    st = {}
    for it in doc.get("items", {}).values():
        st[it.get("status", "?")] = st.get(it.get("status", "?"), 0) + 1
    return st
