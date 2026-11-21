"""studio-cli, the tool commands: every editing tool of the Studio from the command line (docs/CLI.md).

Made for programs rather than people: the RPG Forge, Claude, and anybody's scripts. The photographer's own commands
(develop, import, calibrate ...) stay in cli.py; these are added to the same ``python -m studio`` runner.

    edit         any photo or picture + any settings -> a finished file (the recipe is the Edit, as JSON)
    clean, relight, colourize, restore, jpeg-repair, descreen, negative, infrared      one tool each (edit + a preset)
    match        give pictures the colour of a reference picture, or of a saved look profile
    look-profile measure a look profile from one or more reference pictures
    depth        the depth map, surface normals, a coloured view, or depth layers (RGBA) for parallax
    cutout       a subject cut out with a soft edge (RGBA), or its mask alone
    find         what the Studio can find in a picture, as data (faces, people, dust, flare, crops, colour statistics)
    stack        focus | superres | pano: frames in, one linear result out; focus --make surface: the surface
                 files for games (colour, height, normal, confidence, agreement, JSON)
    dualpixel    the files a Dual Pixel RAW can make (depth, halves, highlight recovery, 3-D views, a surface)
    schema       every setting, its default and what it does, as JSON (read this first)
    looks        the looks on this PC
    batch        many of the commands above in one run (no start-up cost per picture)

The contract (the same for every command here):
  * ``--json``: standard output carries exactly one JSON object, on one line; progress goes to standard error.
    ``--result FILE`` writes the same object to a file.  {"ok", "command", "version", "outputs": [...],
    "problems": [...], "notes": [...], "seconds"}.  Exit code 0 = everything worked, 1 = something failed,
    2 = the command line itself was wrong.
  * nothing is ever written under the name of a picture, and no file the Studio did not write is replaced
    (io/export.free_target): the path actually written is the one in the JSON.
  * settings are checked: a mistyped name is an error that names the nearest real ones, never silently ignored.
  * results are deterministic: the same picture and the same recipe give the same pixels.
No Qt here (rule 2.2).
"""
from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import logging
import os
import sys
import time

import numpy as np

from . import __version__, log as slog, paths

log = slog.get("cli.tools")

PROGRESS = None                 # set by the MCP server for one call: a function given each progress line
RECIPE_MARK = "studio_recipe"
PROFILE_MARK = "studio_look_profile"


class Usage(Exception):
    """The command line (or a recipe) asks for something that does not exist: said plainly, exit code 2."""


# ====================================================================================================== one run
class Run:
    """What one command did: collected as it goes, printed (or returned to ``batch``) at the end."""

    def __init__(self, a, command: str, quiet: bool = False):
        self.a = a
        self.json = bool(getattr(a, "json", False)) or quiet
        self.quiet = quiet
        self.t0 = time.perf_counter()
        self.taken: set = set()
        self.usage = False
        self.res = {"ok": True, "command": command, "version": __version__, "outputs": [], "problems": [],
                    "notes": []}

    def say(self, text: str) -> None:
        print(text, file=sys.stderr if self.json else sys.stdout, flush=True)
        if PROGRESS is not None:
            try:
                PROGRESS(text)
            except Exception:                              # noqa: BLE001  (progress is a courtesy, never a failure)
                pass

    def note(self, text: str) -> None:
        self.res["notes"].append(text)
        self.say("      " + text)

    def output(self, **d) -> dict:
        self.res["outputs"].append(d)
        if d.get("output"):
            self.say(f"      wrote {d['output']}")
        return d

    def problem(self, text: str, path: str | None = None, exc: BaseException | None = None) -> None:
        d = {"text": text}
        if path:
            d["input"] = path
        if exc is not None and not isinstance(exc, Usage) and not slog.expected_problem(exc):
            slog.log_exception(text, log)
            rep = slog.save_error_report(text)
            if rep:
                d["error_report"] = str(rep)
        self.res["problems"].append(d)
        self.say(f"      PROBLEM: {text}")

    def finish(self) -> dict:
        self.res["ok"] = not self.res["problems"]
        self.res["seconds"] = round(time.perf_counter() - self.t0, 2)
        return self.res

    @property
    def code(self) -> int:
        return 2 if self.usage else (0 if not self.res["problems"] else 1)


def console_to_stderr() -> None:
    """With --json, standard output is the result alone: the log's console lines move to standard error."""
    for h in logging.getLogger(slog.LOG_NAME).handlers:
        if type(h) is logging.StreamHandler:              # the console line only (not the log file, not a test capture)
            h.setStream(sys.stderr)


def plain(v, full: bool = False, depth: int = 0):
    """Anything -> JSON-ready. Without ``full``, bulky things (bitmaps, long point lists) become a short summary,
    so a result stays readable and small."""
    if isinstance(v, dict):
        return {str(k): plain(x, full, depth + 1) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        if full or v.size <= 64:
            return plain(v.tolist(), full, depth + 1)
        return {"array": list(v.shape), "min": float(v.min()), "max": float(v.max())}
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, (list, tuple)):
        if not full and len(v) > 64:
            return {"count": len(v), "first": plain(list(v[:3]), full, depth + 1)}
        return [plain(x, full, depth + 1) for x in v]
    if isinstance(v, str):
        return v if full or len(v) <= 2000 else f"<{len(v)} characters>"
    if isinstance(v, float):
        return round(v, 6)
    if v is None or isinstance(v, (bool, int)):
        return v
    if dataclasses.is_dataclass(v):
        return plain(dataclasses.asdict(v), full, depth + 1)
    return str(v)


# ====================================================================================================== pictures in
def collect(items, recursive: bool = False) -> list:
    """Files and folders -> the raws and pictures (TIFF / JPEG / PNG) in them, in name order, each once."""
    from .engine import RAW_EXTENSIONS, SCAN_EXTENSIONS
    from .cli import SKIP_DIRS
    ok = tuple(RAW_EXTENSIONS) + tuple(SCAN_EXTENSIONS)
    out = []
    for it in items or []:
        it = str(it).strip().strip('"')
        if os.path.isdir(it):
            if recursive:
                for root, dirs, files in os.walk(it):
                    dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                    out += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(ok)]
            else:
                out += [os.path.join(it, f) for f in sorted(os.listdir(it)) if f.lower().endswith(ok)]
        elif os.path.isfile(it):
            out.append(it)
        else:
            raise Usage(f"not found: {it}")
    seen, uniq = set(), []
    for p in out:
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


# ====================================================================================================== settings
def _nested_types() -> dict:
    """{field name: dataclass} for the Edit's grouped settings (light, colour, tone ...)."""
    from .engine.pipeline.edit import Edit
    e = Edit()
    return {f: type(getattr(e, f)) for f in Edit.__dataclass_fields__ if dataclasses.is_dataclass(getattr(e, f))}


def _near(name: str, names) -> str:
    m = difflib.get_close_matches(name, list(names), 3, 0.5)
    return f" Did you mean: {', '.join(m)}?" if m else ""


def check_keys(part: dict) -> None:
    """A recipe's names must exist: a typo is an error, never a setting that silently did nothing."""
    from .engine.pipeline.edit import Edit
    if not isinstance(part, dict):
        raise Usage("a recipe is a JSON object of settings, e.g. {\"light\": {\"shadows\": 30}}")
    nested = _nested_types()
    for k, v in part.items():
        if k not in Edit.__dataclass_fields__:
            raise Usage(f"no setting called {k!r}.{_near(k, Edit.__dataclass_fields__)} "
                        f"('schema' lists every setting.)")
        if k in nested:
            if not isinstance(v, dict):
                raise Usage(f"{k!r} is a group of settings: give it as {k}.NAME=value. Its settings: "
                            f"{', '.join(nested[k].__dataclass_fields__)}")
            for kk in v:
                if kk not in nested[k].__dataclass_fields__:
                    raise Usage(f"no setting called {k}.{kk}.{_near(kk, nested[k].__dataclass_fields__)} "
                                f"Settings of {k}: {', '.join(nested[k].__dataclass_fields__)}")


def merge(base: dict, part: dict) -> dict:
    for k, v in part.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v
    return base


def apply_partial(edit, part: dict):
    from .engine.pipeline.edit import Edit
    check_keys(part)
    return Edit.from_dict(merge(edit.to_dict(), json.loads(json.dumps(part))))


def diff(default, d):
    """What ``d`` changes from ``default`` (nested): a recipe holds only what was set, so it stays short and a
    later version's new defaults still apply."""
    if isinstance(d, dict) and isinstance(default, dict):
        out = {}
        for k, v in d.items():
            if k not in default:
                out[k] = v
            elif v != default[k]:
                out[k] = diff(default[k], v)
        return out
    return d


def recipe_of(edit) -> dict:
    from .engine.pipeline.edit import Edit
    d = diff(Edit().to_dict(), edit.to_dict())
    d.pop("stage_versions", None)
    return d


def parse_set(text: str) -> dict:
    """'light.shadows=30' -> {"light": {"shadows": 30}}. The value is JSON when it parses (numbers, true, lists,
    objects), else text."""
    key, eq, val = text.partition("=")
    if not eq or not key.strip():
        raise Usage(f"--set wants NAME=VALUE, e.g. --set light.shadows=30 (got {text!r})")
    try:
        v = json.loads(val)
    except ValueError:
        v = val
    out = v
    for part in reversed(key.strip().split(".")):
        out = {part: out}
    return out


def load_recipe(spec: str) -> tuple:
    """A recipe from a file or inline JSON -> (settings, export options). Reads a recipe written by --save-recipe,
    a bare JSON object of settings, or a photo's own .psb.json sidecar (its master edit)."""
    s = spec.strip()
    if s.startswith("{"):
        try:
            d = json.loads(s)
        except ValueError as e:
            raise Usage(f"--recipe: that JSON does not parse ({e})") from None
    else:
        if not os.path.isfile(s):
            raise Usage(f"--recipe: no file {s}")
        try:
            with open(s, encoding="utf-8") as fh:
                d = json.load(fh)
        except ValueError as e:
            raise Usage(f"--recipe: {s} is not JSON ({e})") from None
    if not isinstance(d, dict):
        raise Usage("--recipe: a recipe is a JSON object")
    if "edit" in d and (RECIPE_MARK in d or "app" in d or "format" in d):
        part = dict(d.get("edit") or {})
        if "app" in d:                                   # a sidecar holds the whole edit: keep only what it set
            from .engine.pipeline.edit import Edit
            part = recipe_of(Edit.from_dict(part))
        return part, dict(d.get("export") or {})
    return d, {}


EXPORT_KEYS = ("format", "space", "bits", "quality", "long_edge", "compression", "sharpen_for")


def make_spec(a, path: str, export: dict | None = None):
    """The export settings: what the command line says, else the recipe's, else by the kind of picture (a raw:
    16-bit Adobe RGB TIFF, Bill's rule for photographs; any other picture: 8-bit sRGB PNG, what a game wants)."""
    from .engine import ExportSpec, RAW_EXTENSIONS
    export = export or {}
    raw = path.lower().endswith(tuple(RAW_EXTENSIONS))
    d = {"format": "tiff", "space": "adobergb", "bits": 16} if raw else {"format": "png", "space": "srgb", "bits": 8}
    of = getattr(a, "out_file", None)
    if of:
        ext = os.path.splitext(of)[1].lower()
        fmt = {".tif": "tiff", ".tiff": "tiff", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}.get(ext)
        if fmt is None:
            raise Usage(f"--out-file: the name must end in .tif, .png or .jpg (got {of})")
        d["format"] = fmt
    for k in EXPORT_KEYS:
        if export.get(k) is not None:
            d[k] = export[k]
    for k in EXPORT_KEYS:
        v = getattr(a, k, None)
        if v is not None:
            d[k] = v
    if getattr(a, "preview", None):
        d.update(format="jpeg", space="srgb", bits=8, long_edge=None)
        d.setdefault("quality", 90)
    spec = ExportSpec(**{k: v for k, v in d.items() if k in ExportSpec.__dataclass_fields__})
    try:
        spec.validate()
    except ValueError as e:
        raise Usage(str(e)) from None
    return spec


def out_dir_of(a) -> str:
    d = getattr(a, "out", None) or str(paths.get("OUTPUTS") / "cli")
    os.makedirs(d, exist_ok=True)
    return d


def target(run: Run, a, src: str, spec, extra: str = "", ext: str | None = None) -> str:
    """Where a result goes: --out-file as given, else NAME[_extra]_Studio.ext in --out (default outputs/cli beside
    the app). Never the picture itself, never over a file the Studio did not write."""
    from .engine.io.export import EXT, export_name, free_target
    of = getattr(a, "out_file", None)
    tag = "".join(c if c.isalnum() or c in "-." else "_" for c in (getattr(a, "tag", None) or "")).strip("_")
    extra = "_".join(x for x in (tag, extra) if x) if not (of and not extra) else extra
    if of and not extra:
        os.makedirs(os.path.dirname(os.path.abspath(of)) or ".", exist_ok=True)
        p = os.path.abspath(of)
    else:
        name = export_name(src, spec, extra)
        if ext:
            name = os.path.splitext(name)[0] + ext
        elif of:
            name = os.path.splitext(os.path.basename(of))[0] + "_" + extra + EXT[spec.format]
        p = os.path.join(os.path.dirname(os.path.abspath(of)) if of else out_dir_of(a), name)
    return free_target(p, src, run.taken)


def scale_for(a, image, edit) -> float:
    """The render scale asked for: --scale, or --preview N (long edge N, the fast path). 1.0 = full size."""
    from . import engine
    if getattr(a, "preview", None):
        return min(1.0, float(a.preview) / max(engine.output_size(image, edit)))
    s = getattr(a, "scale", None)
    if s is None:
        return 1.0
    if not 0.02 <= s <= 1.0:
        raise Usage("--scale is between 0.02 and 1")
    return float(s)


def render_out(image, edit, spec, scale: float = 1.0) -> np.ndarray:
    """The finished picture as the export would write it (display-referred linear ProPhoto), resized and output
    sharpened as the export settings say."""
    from . import engine
    arr = engine.render(image, edit) if scale >= 1.0 else engine.render(image, edit, scale=scale)
    if spec.long_edge:
        h, w = arr.shape[:2]
        k = spec.long_edge / max(h, w)
        if k < 1:
            from .engine.pipeline.pipeline import resize
            arr = resize(np.ascontiguousarray(arr), max(1, round(w * k)), max(1, round(h * k)))
    if spec.sharpen_for != "none":
        from .engine.io.export import output_sharpen
        arr = output_sharpen(arr, spec)
    return arr


# ====================================================================================================== files out
def write_png_any(path: str, arr: np.ndarray, icc_bytes: bytes | None = None, meta: dict | None = None) -> str:
    """PNG with 1, 2, 3 or 4 channels, 8 or 16 bits, with the Studio's mark (engine/io/export.write_png_any)."""
    from .engine.io.export import write_png_any as w
    return w(path, arr, icc_bytes, meta)


def write_rgba(path: str, img_pp: np.ndarray, alpha: np.ndarray, spec, meta: dict | None = None) -> str:
    """Picture + alpha (0..1, straight, not premultiplied) -> RGBA PNG, or 16-bit RGBA TIFF when the name ends
    .tif."""
    from .engine.color import icc
    from .engine.io.export import to_output
    rgb = to_output(np.ascontiguousarray(img_pp, np.float32), spec)
    top = 65535.0 if rgb.dtype == np.uint16 else 255.0
    al = np.clip(np.rint(np.clip(alpha, 0.0, 1.0) * top), 0, top).astype(rgb.dtype)
    rgba = np.concatenate([rgb, al[..., None]], axis=2)
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        tmp = path + ".partial"
        tifffile.imwrite(tmp, rgba, photometric="rgb", extrasamples=["unassalpha"],
                         software=f"Photographic Studio By Bill {__version__}",
                         extratags=[(34675, 7, None, icc.profile_bytes(spec.space), True)])
        os.replace(tmp, path)
        return path
    return write_png_any(path, rgba, icc.profile_bytes(spec.space), meta)


# ====================================================================================================== the edit
def _match_target(a, run: Run):
    """The colour statistics --match / --to aims at: measured once from the reference picture, or read from a look
    profile (look-profile). Kept on the run, so a folder of pictures measures the reference once."""
    spec = getattr(a, "match", None)
    if not spec:
        return None
    hit = getattr(run, "_target", None)
    if hit is not None:
        return hit
    from . import engine
    from .engine.local import colourmatch as cm
    if spec.lower().endswith(".json"):
        try:
            with open(spec, encoding="utf-8") as fh:
                d = json.load(fh)
            t = np.asarray(d["stats"], float)
            assert t.shape == (len(cm.STATS),)
        except Exception:                                 # noqa: BLE001
            raise Usage(f"--match: {spec} is not a look profile (make one with: look-profile PICTURES --save "
                        f"FILE.json)") from None
        run.note(f"matching to the look profile {d.get('name') or os.path.basename(spec)}")
    else:
        if not os.path.isfile(spec):
            raise Usage(f"--match: no file {spec}")
        ref = engine.open(spec)
        t = picture_stats(ref, engine.load_edit(spec) if getattr(a, "match_sidecar", False) else engine.Edit())
        ref.drop_cache()
    run._target = t
    return t


def picture_stats(image, edit, size: int = 320) -> np.ndarray:
    """The nine colour statistics colour match works with (local/colourmatch.STATS), from a small render."""
    from . import engine
    from .engine.local import colourmatch as cm
    s = min(0.5, size / float(max(engine.output_size(image, edit))))
    return cm.stats(engine.render(image, edit, scale=s))


def match_to(image, edit, target, amount: float = 1.0, size: int = 320):
    """Colour match to a target's statistics (the engine's colour_match_solve, with the target given). Returns
    (edit, report)."""
    from . import engine
    from .engine.local import colourmatch as cm
    from .engine.pipeline.pipeline import wb_multipliers
    sc = min(0.5, size / float(max(engine.output_size(image, edit))))
    base = wb_multipliers(image, edit)
    p, rep = cm.solve(lambda e: engine.render(image, e, scale=sc), edit, np.asarray(target, float), base, log=log)
    rep["text"] = cm.describe(p)
    return engine.colour_match_apply(edit, p, base, amount, image), rep


def build_edit(image, path: str, a, run: Run, preset=None):
    """The settings for one picture, in this order (later wins): the photo's own sidecar (--use-sidecar), Make it
    good (--auto), a look (--look), recipes (--recipe, in the order given), the command's own tool, --set, then
    what has to be measured on the picture itself (lens fixes, a negative's film base, the infrared start, and
    last the colour match, which is solved on everything before it). Returns (Edit, export options)."""
    from . import engine
    edit = engine.load_edit(path) if getattr(a, "use_sidecar", False) else engine.Edit()
    export: dict = {}
    if getattr(a, "auto", False):
        edit, notes = engine.make_it_good(image, edit)
        run.note("make it good: " + "; ".join(notes))
    if getattr(a, "look", None):
        from .engine.presets import looks
        lk = looks.find_look(a.look)
        if lk is None:
            raise Usage(f"no look called {a.look!r}. Looks: {', '.join(x['name'] for x in looks.list_looks())}")
        edit = looks.apply_look(edit, lk)
    for r in getattr(a, "recipe", None) or []:
        part, ex = load_recipe(r)
        edit = apply_partial(edit, part)
        export.update(ex)
    if preset is not None:
        part = preset(a)
        if part:
            edit = apply_partial(edit, part)
    for s in getattr(a, "set", None) or []:
        edit = apply_partial(edit, parse_set(s))
    if getattr(a, "fix_lens", False):
        from .engine.lens.fix import fix_lens
        edit, rep = fix_lens(image, edit)
        for r in rep:
            run.note(f"{r['what']}: {r['text']}")
    kind = getattr(a, "negative_kind", None)
    if kind:
        edit.film = engine.film_measure(image, edit, kind)
        run.note(f"{kind} negative: film base and density range measured on this frame")
    if getattr(a, "cmd", "") == "infrared":
        from .engine.color import infrared as irm
        _need_raw(image, "infrared")
        edit, rep = irm.start(image, edit, None if a.filter in (None, "auto") else a.filter)
        run.note(f"infrared: {irm.sentence(rep['guess'])} {rep.get('wb', '')}".strip())
        if a.preset:
            edit = irm.apply_preset(image, edit, a.preset, a.strength)
        for s in getattr(a, "set", None) or []:            # what was typed still wins over the start
            edit = apply_partial(edit, parse_set(s))
    target = _match_target(a, run)
    if target is not None:
        edit, rep = match_to(image, edit, target, getattr(a, "match_amount", 1.0))
        run.note(f"colour match: {rep['text']} (distance {rep['start']:.3f} -> {rep['end']:.3f})")
    return edit, export


def _save_recipe(a, run: Run, edit, export: dict) -> None:
    """The recipe holds the settings, and only the export options that were asked for (so the same recipe still
    gives a raw its TIFF and a game picture its PNG)."""
    p = getattr(a, "save_recipe", None)
    if not p or getattr(run, "_recipe_saved", False):
        return
    ex = dict(export or {})
    ex.update({k: getattr(a, k) for k in EXPORT_KEYS if getattr(a, k, None) is not None})
    doc = {RECIPE_MARK: 1, "app_version": __version__, "edit": recipe_of(edit), "export": ex}
    os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    tmp = p + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    os.replace(tmp, p)
    run._recipe_saved = True
    run.res["recipe"] = os.path.abspath(p)
    run.say(f"      recipe saved: {p}")


def each_picture(a, run: Run, work) -> None:
    """Open every input in turn and hand it to ``work(image, path)``; one picture's failure does not stop the
    others, and no picture's caches outlive it."""
    from . import engine
    files = collect(a.inputs, getattr(a, "recursive", False))
    if not files:
        raise Usage("no pictures found in what was given")
    if getattr(a, "out_file", None) and len(files) > 1:
        raise Usage("--out-file names one result: give one picture, or use --out FOLDER")
    for i, f in enumerate(files, 1):
        run.say(f"[{i}/{len(files)}] {os.path.basename(f)}")
        img = None
        try:
            img = engine.open(f)
            work(img, f)
        except Usage:
            raise
        except Exception as e:                            # noqa: BLE001  (said, reported, and the batch goes on)
            run.problem(f"{os.path.basename(f)}: {e}", f, e)
        finally:
            if img is not None:
                img.drop_cache()
            del img


def cmd_edit(a, run: Run, preset=None) -> None:
    from . import engine

    def work(img, f):
        t0 = time.perf_counter()
        edit, export = build_edit(img, f, a, run, preset)
        spec = make_spec(a, f, export)
        _save_recipe(a, run, edit, export)
        if getattr(a, "save_sidecar", False):
            engine.save_edit(f, edit)
        if getattr(a, "dry_run", False):
            run.output(input=f, edit=recipe_of(edit), size=list(engine.output_size(img, edit)))
            return
        arr = render_out(img, edit, spec, scale_for(a, img, edit))
        p = target(run, a, f, spec, "Preview" if getattr(a, "preview", None) else "")
        from .engine.io.export import write
        write(p, arr, spec, engine.export_metadata(img, edit))
        run.output(input=f, output=p, width=int(arr.shape[1]), height=int(arr.shape[0]), format=spec.format,
                   space=spec.space, bits=spec.bits, seconds=round(time.perf_counter() - t0, 2))

    each_picture(a, run, work)


# ------------------------------------------------------------------------------------------ one-tool presets
def _p_clean(a):
    from .engine.pipeline.edit import Denoise
    d = {"enabled": True, "luma": round(Denoise().luma * a.strength, 3),
         "chroma": round((a.chroma if a.chroma is not None else Denoise().chroma) * a.strength, 3),
         "grain": a.keep_grain, "auto": bool(getattr(a, "auto_strength", False))}
    if d["auto"]:                                  # the photo chooses; --strength leans from its choice
        d["lean"] = round(float(a.strength), 3)
    if a.method:
        d["method"] = a.method
    return {"denoise": d}


def _p_relight(a):
    return {"relight": {"amount": a.amount, "azimuth": a.azimuth, "elevation": a.elevation, "soft": a.soft}}


def _p_colourize(a):
    return {"colourize": {"on": True, "amount": a.amount, "warm": a.warm, "tint": a.tint}}


def _p_restore(a):
    d = {}
    if a.scratches is not None:
        d["scratches"] = {"on": True, "amount": a.scratches, "size": a.scratch_size}
    if a.stains is not None:
        d["stains"] = {"on": True, "amount": a.stains}
    if a.descreen is not None:
        d["halftone"] = {"on": True, "amount": a.descreen}
    if a.jpeg:
        d["jpeg"] = {"on": True}
    if a.colourize is not None:
        d["colourize"] = {"on": True, "amount": a.colourize}
    if not d:
        raise Usage("restore: say what to repair: --scratches, --stains, --descreen, --jpeg and/or --colourize")
    return d


PRESETS = {
    "clean": _p_clean, "relight": _p_relight, "colourize": _p_colourize, "restore": _p_restore,
    "jpeg-repair": lambda a: {"jpeg": {"on": True}},
    "descreen": lambda a: {"halftone": {"on": True, "amount": a.amount}},
    "negative": lambda a: None, "infrared": lambda a: None, "match": lambda a: None,
}


# ------------------------------------------------------------------------------------------ look profiles
def cmd_look_profile(a, run: Run) -> None:
    from . import engine
    from .engine.local import colourmatch as cm
    files = collect(a.inputs, a.recursive)
    if not files:
        raise Usage("look-profile: no pictures found")
    rows = []
    for f in files:
        try:
            img = engine.open(f)
            rows.append(picture_stats(img, engine.load_edit(f) if a.use_sidecar else engine.Edit()))
            img.drop_cache()
        except Exception as e:                            # noqa: BLE001
            run.problem(f"{os.path.basename(f)}: {e}", f, e)
    if not rows:
        return
    stats = np.mean(np.asarray(rows, float), axis=0)
    doc = {PROFILE_MARK: 1, "name": a.name or os.path.splitext(os.path.basename(a.save))[0],
           "stats": [round(float(v), 6) for v in stats], "stat_names": list(cm.STATS),
           "sources": [os.path.basename(f) for f in files], "pictures": len(rows), "app_version": __version__}
    os.makedirs(os.path.dirname(os.path.abspath(a.save)) or ".", exist_ok=True)
    tmp = a.save + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, a.save)
    run.output(output=os.path.abspath(a.save), kind="look_profile", pictures=len(rows),
               stats=dict(zip(cm.STATS, doc["stats"])))


# ====================================================================================================== depth
class NotInstalled(OSError):
    """A tool's model file is not on this PC: said plainly (an OSError, so no error report is written)."""


class NotPossible(OSError):
    """The tool cannot work on this kind of picture (infrared needs a raw): said plainly, no error report."""


def _need_raw(image, what: str) -> None:
    from . import engine
    if engine.is_scan_image(image):
        raise NotPossible(f"{what} works on raw files only (it measures the sensor's own channels)")


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _resize(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    from .engine.pipeline.pipeline import resize
    a = np.ascontiguousarray(arr, np.float32)
    if a.ndim == 2:
        return resize(np.repeat(a[..., None], 3, axis=2), w, h)[..., 0]
    return resize(a, w, h)


def depth_display(image, edit, scale: float = 1.0) -> np.ndarray:
    """The depth map as the finished picture shows it (crop, rotation and geometry applied), 0..1, near = 1,
    the same size as render(scale=scale). Estimated from one picture (Depth Anything V2): amber, not measured."""
    from .engine.local import depth as dp
    from .engine.pipeline.edit import Edit
    from .engine.pipeline.pipeline import depth_shape, render_mask
    if not dp.available():
        raise NotInstalled("the depth model is not installed (models/depth/depth_anything_v2_small.npz: run "
                           "Convert Models.bat)")
    shp = depth_shape(image, edit)
    shp["raw"] = True
    e = Edit.from_dict(edit.to_dict())
    e.masks = [{"name": "depth", "enabled": True, "adjust": {}, "shapes": [shp]}]
    e.composite = None
    return np.clip(render_mask(image, e, 0, scale), 0.0, 1.0)


def depth_layers(D: np.ndarray, n: int, cuts=None, feather: float = 0.02) -> tuple:
    """Split a depth map into ``n`` soft layers, far to near. Returns (alphas, bounds): the alphas sum to 1 at
    every pixel, so the layers laid over each other in order give the picture back exactly."""
    if cuts:
        cuts = sorted(float(c) for c in cuts)
        if len(cuts) != n - 1 or not all(0.0 < c < 1.0 for c in cuts):
            raise Usage(f"--cuts: {n} layers need {n - 1} depth values between 0 and 1 (near = 1)")
    else:
        cuts = [float(np.quantile(D, k / n)) for k in range(1, n)]
    f = max(float(feather), 1e-4)
    steps = [np.ones_like(D)] + [_smooth((D - (c - f)) / (2 * f)) for c in cuts] + [np.zeros_like(D)]
    alphas = [steps[i] - steps[i + 1] if i == 0 else np.minimum(steps[i], 1.0) - steps[i + 1] for i in range(n)]
    alphas = [np.clip(a, 0.0, 1.0) for a in alphas]
    return alphas, [0.0] + cuts + [1.0]


def fill_behind(rgb: np.ndarray, hole: np.ndarray, size: int = 768, method: str = "auto") -> tuple:
    """A background plate: what is in ``hole`` (everything nearer) rebuilt from the surroundings with the Studio's
    Fill (LaMa when installed, else PatchMatch), on a small copy, then laid into the hole at full size. Invented
    picture (amber): it is only ever seen through the gaps parallax opens."""
    from scipy.ndimage import binary_dilation, gaussian_filter
    from .engine.local import fill as fl
    h, w = hole.shape
    k = min(1.0, size / float(max(h, w)))
    sw, sh = max(8, round(w * k)), max(8, round(h * k))
    small = _resize(rgb, sw, sh) if k < 1 else np.ascontiguousarray(rgb, np.float32)
    hs = (_resize(hole.astype(np.float32), sw, sh) if k < 1 else hole.astype(np.float32)) > 0.25
    hs = binary_dilation(hs, iterations=max(2, round(0.006 * max(sw, sh))))
    if hs.all():
        return rgb, {"method": "none", "text": "Everything is nearer than the background layer: nothing to fill from."}
    filled, rep = fl.fill(small, hs, method, 0)
    up = _resize(np.asarray(filled, np.float32), w, h) if k < 1 else np.asarray(filled, np.float32)
    soft = np.clip(gaussian_filter(binary_dilation(hole, iterations=max(2, round(0.004 * max(h, w))))
                                   .astype(np.float32), max(1.0, 0.002 * max(h, w))) * 1.5, 0.0, 1.0)[..., None]
    return (rgb * (1.0 - soft) + up * soft).astype(np.float32), rep


def _alpha_spec(a, f: str, export: dict):
    """Export settings for a result with transparency: PNG (or TIFF); JPEG cannot hold it."""
    spec = make_spec(a, f, export)
    if spec.format == "jpeg" or (f.lower().endswith(_raw_ext()) and getattr(a, "format", None) is None
                                 and not getattr(a, "out_file", None)):
        spec.format = "png"
        if getattr(a, "space", None) is None:
            spec.space = "srgb"
        if getattr(a, "bits", None) is None:
            spec.bits = 8
    return spec


def _raw_ext() -> tuple:
    from .engine import RAW_EXTENSIONS
    return tuple(RAW_EXTENSIONS)


def cmd_depth(a, run: Run) -> None:
    from . import engine
    from .engine.io.export import EXT
    makes = [m.strip().lower() for m in a.make.split(",") if m.strip()]
    bad = [m for m in makes if m not in ("map", "view", "normals", "layers")]
    if bad or not makes:
        raise Usage(f"depth --make: choose from map, view, normals, layers (got {a.make!r})")
    if "layers" in makes and not 2 <= a.layers <= 12:
        raise Usage("depth --layers: between 2 and 12")

    def work(img, f):
        t0 = time.perf_counter()
        edit, export = build_edit(img, f, a, run)
        spec = _alpha_spec(a, f, export)
        s = scale_for(a, img, edit)
        if spec.long_edge:
            s = min(s, spec.long_edge / float(max(engine.output_size(img, edit))))
            spec.long_edge = None
        D = depth_display(img, edit, s)
        h, w = D.shape
        common = dict(input=f, width=w, height=h, estimated=True)
        if "map" in makes:
            p = target(run, a, f, spec, "Depth", ".png")
            write_png_any(p, np.rint(D * 65535.0).astype(np.uint16))
            run.output(output=p, kind="depth_map", bits=16, near="white", **common)
        if "view" in makes:
            from .engine.local import depth as dp
            p = target(run, a, f, spec, "DepthView", ".png")
            write_png_any(p, np.ascontiguousarray(dp.colour_map(D)))
            run.output(output=p, kind="depth_view", **common)
        if "normals" in makes:
            from .engine.local import relight as rl
            n = rl.normals(D)
            n[..., 1] *= -1.0 if a.normals_y == "up" else 1.0          # image rows run down; OpenGL's green runs up
            p = target(run, a, f, spec, "Normals", ".png")
            write_png_any(p, np.rint((n * 0.5 + 0.5) * 255.0).astype(np.uint8))
            run.output(output=p, kind="normal_map", y=a.normals_y, **common)
        if "layers" in makes:
            rgb = render_out(img, edit, spec, s)
            if rgb.shape[:2] != D.shape:
                D2 = _resize(D, rgb.shape[1], rgb.shape[0])
            else:
                D2 = D
            alphas, bounds = depth_layers(D2, a.layers, a.cuts, a.feather)
            for i, al in enumerate(alphas):
                pic, extra = rgb, {}
                if i == 0 and a.fill:
                    pic, rep = fill_behind(rgb, alphas[0] < 0.5, a.fill_size, a.fill_method)
                    al = np.ones_like(al)
                    extra = {"filled": rep.get("method", ""), "fill_note": rep.get("text", "")}
                p = target(run, a, f, spec, f"Layer{i + 1}of{len(alphas)}", EXT[spec.format])
                write_rgba(p, pic, al, spec, engine.export_metadata(img, edit))
                run.output(output=p, kind="depth_layer", layer=i + 1, of=len(alphas), order="1 = farthest",
                           depth_range=[round(bounds[i], 4), round(bounds[i + 1], 4)],
                           coverage=round(float(al.mean()), 4), **extra, **common)
        run.say(f"      depth: {time.perf_counter() - t0:.1f} s")

    each_picture(a, run, work)


# ====================================================================================================== cut-out
def _pt(text: str) -> list:
    try:
        u, v = (float(x) for x in text.split(","))
    except ValueError:
        raise Usage(f"a point is U,V as fractions of the picture, e.g. 0.5,0.4 (got {text!r})") from None
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        raise Usage(f"a point's U and V are between 0 and 1 (got {text!r})")
    return [u, v]


def _nums(text: str, n: int, what: str) -> list:
    try:
        v = [float(x) for x in text.split(",")]
    except ValueError:
        v = []
    if len(v) != n:
        raise Usage(f"{what} wants {n} numbers separated by commas (got {text!r})")
    return v


def cutout_shapes(a, img, edit, run: Run) -> list:
    """The mask shapes the cut-out options ask for (local/masks.py). Several options combine with --combine."""
    from . import engine
    shapes = []
    if a.person:
        shp = engine.find_people(img, edit)
        if shp is None:
            raise NotPossible("nobody was found in this picture")
        shapes.append(dict(shp, part=a.person))
    if a.click:
        def to_s(p):
            xs, ys = engine.view_to_s(img, edit, p[0], p[1])
            return [float(xs), float(ys)]
        fg = [to_s(_pt(p)) for p in a.click]
        bg = [to_s(_pt(p)) for p in (a.avoid or [])]
        model = None if a.model == "classic" else a.model
        if model:
            from .engine.local import sam
            if not sam.available(model):
                raise NotInstalled(f"the {model} object model is not installed (models/sam: Convert Models.bat); "
                                   f"--model classic needs no model")
        shapes.append(engine.object_mask(img, edit, fg, bg, model=model))
    if a.depth_range:
        lo, hi = _nums(a.depth_range, 2, "--depth-range")
        from .engine.local import depth as dp
        if not dp.available():
            raise NotInstalled("the depth model is not installed (models/depth: run Convert Models.bat)")
        shapes.append(engine.depth_shape(img, edit, lo, hi, a.soft))
    if a.luminance:
        lo, hi = _nums(a.luminance, 2, "--luminance")
        shapes.append({"type": "luminance", "lo": lo, "hi": hi, "soft": a.soft})
    if a.colour:
        hue, width = _nums(a.colour, 2, "--colour")
        shapes.append({"type": "colour", "hue": hue, "width": width, "soft": max(1.0, a.soft * 100.0),
                       "chroma": a.chroma})
    if a.skin:
        shapes.append({"type": "skin"})
    for i, sh in enumerate(shapes):
        sh["op"] = "add" if i == 0 else a.combine
    for s in a.shape or []:
        try:
            sh = json.loads(s)
            assert isinstance(sh, dict) and sh.get("type")
        except Exception:                                 # noqa: BLE001
            raise Usage("--shape wants one mask shape as JSON, e.g. '{\"type\": \"radial\", \"cx\": 0.5, \"cy\": "
                        "0.5, \"rx\": 0.2, \"ry\": 0.3}' (see 'schema', masks)") from None
        sh.setdefault("op", "add" if not shapes else a.combine)
        shapes.append(sh)
    if not shapes:
        raise Usage("cutout: say what to cut out: --person, --click U,V, --depth-range LO,HI, --luminance LO,HI, "
                    "--colour HUE,WIDTH, --skin or --shape JSON")
    return shapes


def cmd_cutout(a, run: Run) -> None:
    from . import engine
    from .engine.io.export import EXT
    from .engine.pipeline.edit import Edit

    def work(img, f):
        edit, export = build_edit(img, f, a, run)
        spec = _alpha_spec(a, f, export)
        s = scale_for(a, img, edit)
        if spec.long_edge:
            s = min(s, spec.long_edge / float(max(engine.output_size(img, edit))))
            spec.long_edge = None
        e = Edit.from_dict(edit.to_dict())
        e.masks = list(e.masks or []) + [{"name": "cutout", "enabled": True, "invert": bool(a.invert),
                                          "opacity": 100, "adjust": {},
                                          "shapes": cutout_shapes(a, img, edit, run)}]
        e.composite = None
        alpha = np.clip(engine.render_mask(img, e, len(e.masks) - 1, s), 0.0, 1.0)
        if a.feather > 0:
            from scipy.ndimage import gaussian_filter
            alpha = gaussian_filter(alpha, a.feather * s)
        if a.hard is not None:
            alpha = _smooth((alpha - a.hard + 0.5 * a.hard_width) / max(a.hard_width, 1e-4))
        common = dict(input=f, width=int(alpha.shape[1]), height=int(alpha.shape[0]),
                      coverage=round(float(alpha.mean()), 4))
        if a.make in ("mask", "both"):
            p = target(run, a, f, spec, "Mask", ".png")
            write_png_any(p, np.rint(alpha * 65535.0).astype(np.uint16))
            run.output(output=p, kind="mask", bits=16, **common)
        if a.make in ("rgba", "both"):
            rgb = render_out(img, edit, spec, s)
            if rgb.shape[:2] != alpha.shape:
                alpha = _resize(alpha, rgb.shape[1], rgb.shape[0])
            p = target(run, a, f, spec, "Cutout", EXT[spec.format])
            write_rgba(p, rgb, alpha, spec, engine.export_metadata(img, edit))
            run.output(output=p, kind="cutout", alpha="straight", **common)
        if alpha.mean() < 0.002:
            run.note(f"{os.path.basename(f)}: the cut-out is almost empty (check the selection)")

    each_picture(a, run, work)


# ====================================================================================================== find
FINDERS = ("size", "info", "stats", "faces", "people", "dust", "flare", "crops", "subject", "red-eyes", "blemishes",
           "hdr", "jpeg", "infrared", "depth")


def cmd_find(a, run: Run) -> None:
    from . import engine
    whats = [w.strip().lower() for w in a.what.split(",") if w.strip()]
    bad = [w for w in whats if w not in FINDERS]
    if bad or not whats:
        raise Usage(f"find --what: choose from {', '.join(FINDERS)}" + (f" (got {', '.join(bad)})" if bad else ""))

    def one(w, img, f, edit):
        if w == "size":
            ow, oh = engine.output_size(img, edit)
            return {"width": int(ow), "height": int(oh)}
        if w == "info":
            return slog.scrub(img.meta.to_dict()) if hasattr(img.meta, "to_dict") else {}
        if w == "stats":
            from .engine.local import colourmatch as cm
            return dict(zip(cm.STATS, picture_stats(img, edit).tolist()))
        if w == "faces":
            out = []
            for fc in engine.find_faces(img, edit):
                P = np.asarray(fc.get("pts") or [], float)
                d = {k: v for k, v in fc.items() if k != "pts"}
                if P.size:
                    d["box"] = [float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max())]
                    if a.full:
                        d["pts"] = P.tolist()
                out.append(d)
            return out
        if w == "people":
            shp = engine.find_people(img, edit)
            return None if shp is None else {k: v for k, v in shp.items() if a.full or k not in ("data", "hair_maps")}
        if w == "dust":
            return engine.find_dust(img, edit)
        if w == "flare":
            return engine.find_flare(img, edit)
        if w == "crops":
            return engine.suggest_crops(img, edit, a.ratio, a.method, a.n)
        if w == "subject":
            return engine.composition_subject(img, edit)
        if w == "red-eyes":
            return engine.find_red_eyes(img, edit)
        if w == "blemishes":
            return engine.find_blemishes(img, edit)
        if w == "hdr":
            return engine.hdr_report(img, edit)
        if w == "jpeg":
            from .engine.local import jpegfix
            return jpegfix.info(f) if f.lower().endswith((".jpg", ".jpeg")) else None
        if w == "infrared":
            from .engine.color import infrared as irm
            _need_raw(img, "the infrared meter")
            meter = irm.colour_meter(img)
            return {"meter": meter, "guess": irm.guess_filter(img, meter.get("ratio"))}
        if w == "depth":
            D = depth_display(img, edit, min(1.0, 512.0 / max(engine.output_size(img, edit))))
            return {"estimated": True, "near_is": 1.0, "quartiles": [float(np.quantile(D, q)) for q in (0.25, 0.5, 0.75)],
                    "near_share": float((D > 0.66).mean()), "far_share": float((D < 0.33).mean())}
        return None

    def work(img, f):
        edit, _ = build_edit(img, f, a, run)
        found = {}
        for w in whats:
            try:
                found[w] = plain(one(w, img, f, edit), a.full)
            except Exception as e:                        # noqa: BLE001
                found[w] = None
                run.problem(f"{os.path.basename(f)}: {w}: {e}", f, e)
        run.output(input=f, found=found)

    each_picture(a, run, work)


# ====================================================================================================== stacks
def _set_params(p, sets, what: str):
    """--set NAME=VALUE on a stacking tool's own settings (a flat list; checked like the Edit's)."""
    fields = type(p).__dataclass_fields__
    for s in sets or []:
        d = parse_set(s)
        (k, v), = d.items()
        if k not in fields or isinstance(v, dict) and not isinstance(getattr(p, k), dict):
            raise Usage(f"{what} has no setting {k!r}.{_near(k, fields)} Its settings: {', '.join(fields)}")
        setattr(p, k, tuple(v) if isinstance(getattr(p, k), tuple) and isinstance(v, list) else v)
    return p


def cmd_stack(a, run: Run) -> None:
    from .engine.io.export import free_target
    from .engine.stack import frames as FR
    files = collect(a.inputs, a.recursive)
    if len(files) < 2:
        raise Usage("stack: give at least two frames (files, or a folder of them)")
    last = [""]

    def progress(frac, msg=""):
        if msg and msg != last[0]:
            last[0] = msg
            run.say(f"      {100 * float(frac):3.0f}%  {msg}")

    out_dir = out_dir_of(a) if a.out else FR.default_out_dir(files[0])[0]
    info: dict = {"kind": a.kind, "frames_given": len(files)}
    makes = [m.strip().lower() for m in (a.make or "merge").split(",") if m.strip()]
    bad = [m for m in makes if m not in STACK_MAKES]
    if bad or not makes:
        raise Usage(f"stack --make: choose from {', '.join(STACK_MAKES)} (got {a.make!r})")
    if a.kind != "focus" and makes != ["merge"]:
        raise Usage("stack --make depth / surface: focus stacks only")
    if a.kind == "focus":
        from .engine.stack import focus as FS
        p = _set_params(FS.FocusParams(), a.set, "Focus Stack")
        if p.method not in FS.METHODS:
            raise Usage(f"focus method: one of {', '.join(FS.METHODS)}")
        sp = surface_params(a) if ("depth" in makes or "surface" in makes) else None
        an = FS.analyse(files, p, progress)
        try:
            info.update(frames=an.n, method=p.method, dual_pixel=bool(an.dual_pixel))
            if "merge" in makes:
                out = free_target(os.path.join(out_dir, FR.result_name(files, "Focus")), None, run.taken)
                FS.build(an, p, out, progress)
                run.output(input=files[0], output=out, linear=True, **info)
                run.note("a stack's result is a linear file (32-bit): finish it with 'edit' (it opens like any photo)")
            if sp is not None:
                from .engine.stack import surface as SURF
                kinds = SURF.KINDS if "surface" in makes else SURF.DEPTH_KINDS
                r = SURF.stack_surface(an, out_dir, FR.result_base(files), sp, kinds, progress, None, run.taken,
                                       p.method)
                surface_outputs(run, files[0], r, info)
        finally:
            an.close()
        return
    elif a.kind == "superres":
        from .engine.stack import superres as SR
        if a.mode not in SR.MODES:
            raise Usage(f"superres --mode: one of {', '.join(SR.MODES)}")
        p = _set_params(SR.SuperParams.for_mode(a.mode), a.set, "Super-Res Stack")
        an = SR.analyse(files, p, progress)
        kept = an.kept(p)
        name = FR.result_name([an.paths[kept[0]], an.paths[kept[-1]]], SR.result_tag(p, an.raw))
        out = free_target(os.path.join(out_dir, name), None, run.taken)
        SR.build(an, p, out, progress)
        info.update(frames=len(kept), mode=p.mode, merge=p.merge, scale=p.scale,
                    warnings=plain(SR.frame_warnings(an, p)))
    else:
        from .engine.stack import pano as PA
        p = _set_params(PA.PanoParams(), a.set, "Panorama")
        an = PA.analyse(files, p, progress)
        used = [q for i, q in enumerate(an.paths) if an.R[i] is not None]
        out = free_target(os.path.join(out_dir, FR.result_name([used[0], used[-1]], "Pano")), None, run.taken)
        rep = PA.build(an, p, out, progress, None, factor=a.factor)
        out = rep.get("path", out)
        info.update(frames=len(used), projection=PA.resolved_projection(an, p),
                    report=plain({k: v for k, v in rep.items() if k != "path"}))
    run.output(input=files[0], output=out, linear=True, **info)
    run.note("a stack's result is a linear file (32-bit): finish it with 'edit' (it opens like any photo)")


# ====================================================================================================== dual pixel
DP_MAKES = ("report", "depth", "half1", "half2", "hdr", "anaglyph", "sbs", "wiggle", "surface")
STACK_MAKES = ("merge", "depth", "surface")


def surface_params(a):
    """The surface settings from the command line (checked: a wrong value is a usage error, never ignored)."""
    from .engine.stack import surface as SURF
    sp = SURF.SurfaceParams(ratio=getattr(a, "ratio", None) or "", step_mm=float(getattr(a, "step_mm", 0) or 0),
                            near=getattr(a, "near", "auto") or "auto", normals_y=getattr(a, "normals_y", "up"),
                            normal_strength=float(getattr(a, "normal_strength", 1.0)),
                            tileable=bool(getattr(a, "tileable", False)), swap=bool(getattr(a, "swap", False)))
    try:
        SURF.parse_ratio(sp.ratio)
    except ValueError as e:
        raise Usage(f"--ratio: {e}") from None
    if sp.step_mm < 0 or sp.step_mm > 100:
        raise Usage("--step-mm: the distance the focus moved between frames, in mm (for example 0.2)")
    if not 0.05 <= sp.normal_strength <= 20:
        raise Usage("--normal-strength: between 0.05 and 20")
    return sp


def surface_outputs(run: Run, src: str, r: dict, info: dict | None = None) -> None:
    """One result entry per surface file; the JSON's entry carries the summary a program needs first."""
    doc = r["json"]
    info = {("stack" if k == "kind" else k): v for k, v in (info or {}).items()}
    for k, path in r["paths"].items():
        if k == "json":
            continue
        run.output(input=src, output=path, kind=f"surface_{k}", width=doc["size"]["width"],
                   height=doc["size"]["height"], measured=k != "colour", **info)
    if "json" in r["paths"]:
        run.output(input=src, output=r["paths"]["json"], kind="surface_json", method=doc["method"],
                   height_units=doc["height"]["units"], near_end=doc["height"]["near_end"],
                   coverage=doc["coverage"], files=doc["files"])
    for t in doc.get("notes", []):
        run.note(t)


def cmd_dualpixel(a, run: Run) -> None:
    from .engine.dualpixel import decode as DD, depth as DEP, hdr as HDR, views as V
    from .engine.io.export import free_target
    makes = [m.strip().lower() for m in a.make.split(",") if m.strip()]
    bad = [m for m in makes if m not in DP_MAKES]
    if bad or not makes:
        raise Usage(f"dualpixel --make: choose from {', '.join(DP_MAKES)}")
    files = collect(a.inputs, a.recursive)
    if not files:
        raise Usage("dualpixel: no files found")
    for i, f in enumerate(files, 1):
        run.say(f"[{i}/{len(files)}] {os.path.basename(f)}")
        try:
            if not f.lower().endswith(".dng") and not DD.is_dual_pixel(f):
                run.problem(f"{os.path.basename(f)}: {DD.NOT_DUAL} {DD.HOW_TO}", f)
                continue
            sp = surface_params(a) if "surface" in makes else None
            dp = DD.open_dual(f)
            gains = DD.shading(dp)
            la, lb = DEP.halves_luma(dp, gains)
            disp, conf = DEP.disparity(la, lb)
            ref = DEP.refine(disp, conf, la + lb)
            out_dir = out_dir_of(a)

            def tgt(kind):
                return free_target(os.path.join(out_dir, V.out_name(f, kind)), f, run.taken)

            for m in makes:
                if m == "report":
                    run.output(input=f, kind="dp_report", focus=plain(DEP.focus_report(ref, conf, a.swap)),
                               air=plain(DEP.air_steadiness(ref, conf)), level_ratio=float(dp.level_ratio),
                               notes=list(dp.notes))
                elif m == "depth":
                    from .engine.stack.focus import _up2
                    p = V.depth_png(_up2(ref, dp.shape), conf, tgt("depth"), dp.flip, a.swap)
                    run.output(input=f, output=p, kind="dp_depth", measured=True, near="white", bits=16)
                elif m in ("half1", "half2"):
                    p = V.export_half(dp, gains, int(m[-1]), tgt(m))
                    run.output(input=f, output=p, kind="dp_" + m, linear=True)
                elif m == "hdr":
                    from .engine.io.intermediate import write_intermediate
                    from .engine.pipeline.pipeline import apply_orientation
                    from .engine.stack.focus import write_stack_block
                    rgb, rep = HDR.recover(dp, gains, ref, strength=a.strength)
                    md = dp.image.meta
                    meta = {"make": md.make, "model": md.model, "iso": md.iso, "exposure_s": md.exposure_s,
                            "fnumber": md.fnumber, "focal_mm": md.focal_mm, "lens": md.lens, "datetime": md.datetime}
                    p = tgt("hdr")
                    write_intermediate(p, np.ascontiguousarray(apply_orientation(rgb, dp.flip)), meta,
                                       {"kind": "dualpixel", "method": "hdr", "sources": [os.path.basename(dp.path)],
                                        "scale": 1.0, "report": {k: v for k, v in rep.items() if k != "sentence"}})
                    write_stack_block(p, "dualpixel", [dp.path], {"strength": a.strength})
                    run.output(input=f, output=p, kind="dp_hdr", linear=True, text=rep.get("sentence", ""),
                               stops=plain(rep.get("stops")))
                elif m == "anaglyph":
                    run.output(input=f, output=V.save_jpeg(V.anaglyph(dp, gains, a.view_scale), tgt("anaglyph")),
                               kind="dp_anaglyph")
                elif m == "sbs":
                    run.output(input=f, output=V.save_jpeg(V.side_by_side(dp, gains, a.view_scale), tgt("pair")),
                               kind="dp_side_by_side")
                elif m == "wiggle":
                    run.output(input=f, output=V.wigglegram(dp, gains, tgt("wiggle"), boost=a.boost, disp=ref),
                               kind="dp_wigglegram")
                elif m == "surface":
                    from .engine.stack import surface as SURF
                    r = SURF.dual_surface(dp, out_dir, os.path.splitext(os.path.basename(f))[0], sp, SURF.KINDS,
                                          run.taken, {"la": la, "lb": lb, "disp": disp, "conf": conf})
                    surface_outputs(run, f, r)
        except Usage:
            raise
        except Exception as e:                            # noqa: BLE001
            run.problem(f"{os.path.basename(f)}: {e}", f, e)


# ====================================================================================================== schema
def _field_docs(cls) -> dict:
    """{field: the comment written beside it in pipeline/edit.py}: the settings document themselves."""
    import inspect
    import re
    docs, cur = {}, None
    try:
        lines = inspect.getsource(cls).splitlines()
    except (OSError, TypeError):
        return docs
    for ln in lines:
        m = re.match(r"\s{4}(\w+)\s*:\s*[^#]*?(?:#\s?(.*))?$", ln)
        if m and m.group(1) in cls.__dataclass_fields__ and not ln.lstrip().startswith("#"):
            cur = m.group(1)
            docs[cur] = (m.group(2) or "").strip()
            continue
        m2 = re.match(r"\s{8,}#\s?(.*)$", ln)
        if m2 and cur:
            docs[cur] = (docs[cur] + " " + m2.group(1).strip()).strip()
        else:
            cur = None
    return docs


def schema() -> dict:
    """Everything a program needs to drive the Studio: every setting with its default and its own description,
    the tools' settings, the choices, the export options, the commands. Built from the code, so it cannot go out
    of date."""
    import inspect
    from .engine.color import hdr, infrared as irm
    from .engine.detail import scales
    from .engine.io.export import FORMATS, OUTPUT_SHARPEN
    from .engine.local import colourize, colourmatch, depth, halftone, masks, relight, scratches, stains
    from .engine.pipeline.edit import Edit
    from .engine.stack import depthfuse, focus, pano, project, superres, surface
    e = Edit()
    nested = _nested_types()
    top = _field_docs(Edit)
    settings, groups = [], {}
    for name in Edit.__dataclass_fields__:
        if name == "stage_versions":
            continue
        v = getattr(e, name)
        if name in nested:
            groups[name] = " ".join((inspect.getdoc(nested[name]) or "").split())
            docs = _field_docs(nested[name])
            for k in nested[name].__dataclass_fields__:
                d = getattr(v, k)
                settings.append({"name": f"{name}.{k}", "default": d, "type": type(d).__name__ if d is not None
                                 else "object", "about": docs.get(k, "")})
        else:
            settings.append({"name": name, "default": v, "type": type(v).__name__ if v is not None else "object",
                             "about": top.get(name, "")})
    return {
        "app": "Photographic Studio By Bill", "version": __version__,
        "how": "Settings go in with --set NAME=VALUE (repeatable) or --recipe FILE_OR_JSON; a recipe is a JSON "
               "object holding only what differs from the defaults, e.g. {\"exposure_ev\": 0.3, \"light\": "
               "{\"shadows\": 25}}. --save-recipe writes one from any command line. Points and boxes are "
               "fractions 0..1 of the picture (u right, v down).",
        "order": "sidecar (--use-sidecar), --auto, --look, --recipe (in order), the command's own tool, --set, "
                 "then what is measured on the picture (--fix-lens, negative, infrared), then --match.",
        "settings": settings,
        "groups": groups,
        "tools": {
            "relight": dict(relight.DEFAULTS, about="amount 0..100; azimuth degrees (0 = from the right, 90 = "
                            "from above); elevation degrees above the picture plane; soft 0..100"),
            "scratches": dict(scratches.DEFAULTS), "stains": dict(stains.DEFAULTS),
            "halftone": dict(halftone.DEFAULTS), "colourize": dict(colourize.DEFAULTS),
            "depth": dict(depth.DEFAULTS, about="haze by distance, -100..100"), "jpeg": {"on": False},
            "film": {"kind": "colour | bw", "about": "set by the 'negative' command, which measures the frame"},
            "infrared": {"filters": list(irm.FILTER_ORDER), "presets": list(irm.PRESETS),
                         "bw_presets": list(irm.BW_PRESETS), "palettes": list(irm.PALETTES),
                         "about": "set by the 'infrared' command, which measures the frame"},
        },
        "masks": {"about": " ".join((masks.__doc__ or "").split()), "adjustments": list(masks.ALL_KEYS)},
        "choices": {
            "demosaic": ["rcd", "menon", "lmmse", "dual", "mhc", "bilinear", "half"],
            "highlight_mode": ["clip", "rebuild", "glow", "none"],
            "white_balance.mode": ["as_shot", "daylight", "temp_tint", "multipliers"],
            "tone.mode": ["standard", "camera_match", "linear"],
            "denoise.method": ["hybrid", "fusion", "starlet", "nlm", "mppca", "collab", "learned"],
            "hdr.strength": list(hdr.STRENGTHS), "hdr.engine": list(hdr.ENGINES),
            "detail": {k: {"lo": v["lo"], "hi": v["hi"], "does": v.get("does", "")} for k, v in scales.METHODS.items()},
            "detail_presets": {k: dict(v) for k, v in scales.PRESETS.items()},
            "colour_match_stats": list(colourmatch.STATS),
        },
        "stack": {
            "focus": {"settings": plain(focus.FocusParams().to_dict()), "methods": plain(focus.METHOD_INFO),
                      "settings_about": _field_docs(focus.FocusParams)},
            "surface": {
                "about": "Depth measured two ways in a focus stack of Dual Pixel RAWs (which frame is sharpest, and "
                         "each frame's own Dual Pixel offset), fused into one height map with a map of where the two "
                         "agree; focus alone for ordinary raws; one Dual Pixel photo on its own. Made by: stack focus "
                         "FRAMES --make surface (or depth), dualpixel FILE --make surface.",
                "settings": {k: {"default": v, "about": _field_docs(surface.SurfaceParams).get(k, "")}
                             for k, v in surface.SurfaceParams().to_dict().items()},
                "methods": plain(depthfuse.METHOD_INFO),
                "files": surface.names("NAME"),
                "depth_files": surface.names("NAME", surface.DEPTH_KINDS),
                "colours": {"cyan": "measured both ways", "amber": "measured one way",
                            "violet": "filled in from the neighbours (confidence 0)",
                            "red": "the measures disagree (more than half a focus step)"}},
            "superres": {"settings": plain(superres.SuperParams().to_dict()), "modes": plain(superres.MODE_INFO),
                         "merges": plain(superres.MERGE_INFO)},
            "pano": {"settings": plain(dataclasses.asdict(pano.PanoParams())),
                     "projections": list(project.PROJECTIONS), "methods": plain(pano.METHOD_INFO)},
        },
        "export": {"format": list(FORMATS), "space": ["adobergb", "srgb", "prophoto", "displayp3"], "bits": [8, 16],
                   "sharpen_for": list(OUTPUT_SHARPEN), "compression": ["none", "zip"],
                   "defaults": "a raw: 16-bit Adobe RGB TIFF; any other picture: 8-bit sRGB PNG; results with "
                               "transparency: PNG (or .tif)",
                   "names": "NAME_Studio.ext, NAME_Cutout_Studio.png, NAME_Depth_Studio.png, "
                            "NAME_Layer1of3_Studio.png ...; never the picture's own name"},
        "find": list(FINDERS),
        "commands": {k: v for k, v in HELP.items()},
        "result": {"ok": "true when there are no problems", "outputs": "one entry per file written (or per "
                   "picture examined): input, output, kind, width, height ...", "problems": "[{text, input, "
                   "error_report}]", "notes": "what was measured or decided, in plain words",
                   "exit_codes": {"0": "all worked", "1": "something failed", "2": "the command line was wrong"}},
    }


def cmd_schema(a, run: Run) -> None:
    d = schema()
    if a.setting:
        hit = [s for s in d["settings"] if s["name"] == a.setting or s["name"].startswith(a.setting + ".")]
        if not hit:
            raise Usage(f"no setting called {a.setting!r}.{_near(a.setting, [s['name'] for s in d['settings']])}")
        d = {"settings": hit}
    run.res["data"] = plain(d, True)


def cmd_looks(a, run: Run) -> None:
    from .engine.presets import looks
    run.res["data"] = plain(looks.list_looks(), True)


# ====================================================================================================== batch
def cmd_batch(a, run: Run) -> None:
    """Many commands in one run: the Studio starts once, so a queue of small jobs costs no start-up each. The jobs
    file is a JSON list; each job is a list of arguments (["edit", "a.png", "--set", "exposure_ev=0.3"]), one
    command-line string, or {"args": [...], "id": anything}. Each job's own result comes back under "jobs"."""
    import shlex
    from .cli import build_parser
    try:
        text = sys.stdin.read() if a.jobs == "-" else open(a.jobs, encoding="utf-8").read()
        jobs = json.loads(text)
        assert isinstance(jobs, list)
    except OSError as e:
        raise Usage(f"batch: cannot read {a.jobs} ({e})") from None
    except (ValueError, AssertionError):
        raise Usage("batch: the jobs file is a JSON list of jobs") from None
    ap = build_parser()
    run.res["jobs"] = []
    for i, job in enumerate(jobs, 1):
        jid = job.get("id", i) if isinstance(job, dict) else i
        args = job.get("args") if isinstance(job, dict) else job
        if isinstance(args, str):
            args = shlex.split(args, posix=os.name != "nt")
        if not isinstance(args, list) or not args:
            run.problem(f"job {jid}: no arguments")
            run.res["jobs"].append({"id": jid, "ok": False, "problems": [{"text": "no arguments"}]})
            continue
        args = [str(x) for x in args]
        run.say(f"job {jid}: {' '.join(args[:6])}{' ...' if len(args) > 6 else ''}")
        if args[0] not in HANDLERS or args[0] == "batch":
            run.problem(f"job {jid}: {args[0]!r} is not a command batch can run ({', '.join(sorted(HANDLERS))})")
            run.res["jobs"].append({"id": jid, "ok": False, "problems": [{"text": f"unknown command {args[0]}"}]})
            continue
        try:
            ja = ap.parse_args(args)
        except SystemExit:
            run.problem(f"job {jid}: the arguments do not parse ({' '.join(args)})")
            run.res["jobs"].append({"id": jid, "ok": False, "problems": [{"text": "arguments do not parse"}]})
            if a.stop_on_error:
                break
            continue
        r = execute(ja, quiet=True)
        res = dict(r.res, id=jid)
        run.res["jobs"].append(res)
        run.res["outputs"] += [dict(o, job=jid) for o in res["outputs"]]
        if not res["ok"]:
            run.problem(f"job {jid}: " + "; ".join(p["text"] for p in res["problems"]))
            if a.stop_on_error:
                break


# ====================================================================================================== command line
HELP = {
    "edit": "any photo or picture + any settings -> a finished file",
    "clean": "remove noise (the Studio's Clean); best on raw files",
    "relight": "move the light, from the picture's depth (estimated)",
    "colourize": "black and white -> colour (a plausible guess)",
    "restore": "old photos and scans: dust and scratches, stains, printing dots, JPEG damage, colour",
    "jpeg-repair": "undo JPEG blocks and ringing inside what the file allows",
    "descreen": "remove the printing dots from a scan of a printed picture",
    "negative": "a film negative (scanned or photographed) -> a positive",
    "infrared": "infrared / full-spectrum raws: white balance, channel mix or black and white",
    "match": "give pictures the colour of a reference picture or of a look profile",
    "look-profile": "measure a look profile (colour statistics) from reference pictures",
    "depth": "depth map, normal map, coloured view, or depth layers (RGBA) for parallax",
    "cutout": "cut a subject out with a soft edge (RGBA), or write its mask",
    "find": "what the Studio finds in a picture, as data",
    "stack": "focus | superres | pano: frames in, one linear result out; focus --make surface: colour, height, "
             "normal, confidence and agreement maps for games",
    "dualpixel": "what a Dual Pixel RAW can make: depth, halves, highlight recovery, 3-D views, a surface for games",
    "schema": "every setting, default, choice and command, as JSON (read this first)",
    "looks": "the looks on this PC",
    "batch": "run many of these commands from a jobs file, in one start",
}
DATA_COMMANDS = ("schema", "looks", "find", "look-profile")


def _edit_cmd(name):
    return lambda a, run: cmd_edit(a, run, PRESETS.get(name))


HANDLERS = {
    "edit": _edit_cmd("edit"), "clean": _edit_cmd("clean"), "relight": _edit_cmd("relight"),
    "colourize": _edit_cmd("colourize"), "restore": _edit_cmd("restore"), "jpeg-repair": _edit_cmd("jpeg-repair"),
    "descreen": _edit_cmd("descreen"), "negative": _edit_cmd("negative"), "infrared": _edit_cmd("infrared"),
    "match": _edit_cmd("match"), "look-profile": cmd_look_profile, "depth": cmd_depth, "cutout": cmd_cutout,
    "find": cmd_find, "stack": cmd_stack, "dualpixel": cmd_dualpixel, "schema": cmd_schema, "looks": cmd_looks,
    "batch": cmd_batch,
}
COMMANDS = tuple(HANDLERS)


def _parents():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print one JSON result on standard output")
    common.add_argument("--result", default=None, metavar="FILE", help="also write the JSON result to this file")
    inp = argparse.ArgumentParser(add_help=False)
    inp.add_argument("inputs", nargs="+", help="pictures (raw, TIFF, PNG, JPEG) and/or folders of them")
    inp.add_argument("--recursive", action="store_true", help="include subfolders")
    ed = argparse.ArgumentParser(add_help=False)
    ed.add_argument("--recipe", action="append", default=None, metavar="FILE_OR_JSON",
                    help="settings as JSON: a recipe file, inline JSON, or a photo's .psb.json (repeatable)")
    ed.add_argument("--set", action="append", default=None, metavar="NAME=VALUE",
                    help="one setting, e.g. --set light.shadows=30 (repeatable; 'schema' lists them)")
    ed.add_argument("--look", default=None, help="apply a look by name ('looks' lists them)")
    ed.add_argument("--auto", action="store_true", help="start from Make it good")
    ed.add_argument("--use-sidecar", action="store_true", help="start from the photo's own saved edit (.psb.json)")
    ed.add_argument("--fix-lens", action="store_true", help="measure and fix the lens (fringing, distortion)")
    ed.add_argument("--match", default=None, metavar="PICTURE_OR_PROFILE",
                    help="colour match to a reference picture, or to a look profile (.json from look-profile)")
    ed.add_argument("--match-amount", type=float, default=1.0, help="how much of the match to apply, 0..1.5")
    ed.add_argument("--match-sidecar", action="store_true", help="render the reference with its own saved edit")
    out = argparse.ArgumentParser(add_help=False)
    out.add_argument("--out", default=None, metavar="FOLDER", help="where results go (default: outputs/cli)")
    out.add_argument("--out-file", default=None, metavar="FILE", help="the result's exact name (one picture)")
    out.add_argument("--format", default=None, choices=["tiff", "png", "jpeg"])
    out.add_argument("--space", default=None, choices=["adobergb", "srgb", "prophoto", "displayp3"])
    out.add_argument("--bits", type=int, default=None, choices=[8, 16])
    out.add_argument("--quality", type=int, default=None, help="JPEG quality (default 95)")
    out.add_argument("--long-edge", type=int, default=None, help="resize so the long edge is this many pixels")
    out.add_argument("--compression", default=None, choices=["none", "zip"])
    out.add_argument("--sharpen-for", default=None, choices=["none", "screen", "glossy", "matte"])
    out.add_argument("--tag", default=None, help="a word put in the result's name: NAME_TAG_Studio.ext (so several "
                     "versions of one picture can sit in one folder)")
    out.add_argument("--scale", type=float, default=None, help="render smaller, 0.02..1 (the fast preview path)")
    out.add_argument("--preview", type=int, default=None, metavar="N",
                     help="a quick look: sRGB JPEG, long edge N pixels, named NAME_Preview_Studio.jpg")
    return common, inp, ed, out


def _surface_args(p, stack: bool) -> None:
    """The settings of --make surface / depth (engine/stack/surface.SurfaceParams)."""
    p.add_argument("--ratio", default=None, metavar="M", help="surface: magnification as shot, e.g. 1:2 or 1:1 (gives "
                   "the size of a pixel on the subject in mm)")
    if stack:
        p.add_argument("--step-mm", type=float, default=0.0, help="surface: how far the focus moved between frames, "
                       "in mm (a rail): the height is then in mm")
        p.add_argument("--near", default="auto", choices=["auto", "first", "last"],
                       help="surface: which end of the stack is nearest (auto: measured when the camera's Dual Pixel "
                            "reading is known, else the first frame)")
    p.add_argument("--normals-y", default="up", choices=["up", "down"], help="surface: normal map green = up "
                   "(OpenGL, Godot, Blender) or down (DirectX)")
    p.add_argument("--normal-strength", type=float, default=1.0, help="surface: steepness of the normal map when "
                   "the height is relative (1 = the steep tenth tilts about 45 degrees)")
    p.add_argument("--tileable", action="store_true", help="surface: offset by half and heal the seams of every "
                   "file the same way (a repeating texture)")


def add_parsers(sub) -> None:
    """Add the tool commands to cli.py's sub-parsers."""
    common, inp, ed, out = _parents()
    full = [inp, ed, out, common]

    p = sub.add_parser("edit", parents=full, help=HELP["edit"])
    p.add_argument("--save-recipe", default=None, metavar="FILE", help="write the settings used as a recipe")
    p.add_argument("--save-sidecar", action="store_true", help="save the settings as the photo's own edit")
    p.add_argument("--dry-run", action="store_true", help="work the settings out and report them; write no picture")

    p = sub.add_parser("clean", parents=full, help=HELP["clean"])
    p.add_argument("--strength", type=float, default=1.0, help="overall strength, 1 = matched to the measured noise")
    p.add_argument("--chroma", type=float, default=None, help="colour-noise strength (default 1.5)")
    p.add_argument("--method", default=None,
                   choices=["hybrid", "fusion", "starlet", "nlm", "mppca", "collab", "learned"])
    p.add_argument("--keep-grain", type=float, default=0.0, help="share of the removed grain put back, 0..1")
    p.add_argument("--auto-strength", action="store_true",
                   help="let each photo choose its own strength, detail first (needs a calibrated camera; "
                        "--strength then leans lighter or heavier from that choice)")

    p = sub.add_parser("relight", parents=full, help=HELP["relight"])
    p.add_argument("--amount", type=float, default=60.0, help="0..100")
    p.add_argument("--azimuth", type=float, default=135.0, help="where the light comes from, degrees")
    p.add_argument("--elevation", type=float, default=35.0, help="how far in front of the picture, degrees")
    p.add_argument("--soft", type=float, default=40.0, help="0 hard .. 100 soft")

    p = sub.add_parser("colourize", parents=full, help=HELP["colourize"])
    p.add_argument("--amount", type=float, default=1.0, help="0..1")
    p.add_argument("--warm", type=float, default=0.0)
    p.add_argument("--tint", type=float, default=0.0)

    p = sub.add_parser("restore", parents=full, help=HELP["restore"])
    p.add_argument("--scratches", type=float, nargs="?", const=50.0, default=None, help="dust and scratches, 0..100")
    p.add_argument("--scratch-size", type=float, default=8.0, help="largest mark to remove, pixels")
    p.add_argument("--stains", type=float, nargs="?", const=100.0, default=None, help="stains / foxing, 0..150")
    p.add_argument("--descreen", type=float, nargs="?", const=100.0, default=None, help="printing dots, 0..100")
    p.add_argument("--jpeg", action="store_true", help="JPEG repair")
    p.add_argument("--colourize", type=float, nargs="?", const=1.0, default=None, help="add colour, 0..1")

    sub.add_parser("jpeg-repair", parents=full, help=HELP["jpeg-repair"])
    p = sub.add_parser("descreen", parents=full, help=HELP["descreen"])
    p.add_argument("--amount", type=float, default=100.0)

    p = sub.add_parser("negative", parents=full, help=HELP["negative"])
    p.add_argument("--kind", dest="negative_kind", default="colour", choices=["colour", "bw"])

    p = sub.add_parser("infrared", parents=full, help=HELP["infrared"])
    p.add_argument("--filter", default="auto", help="auto (guessed from the raw), full, 550, 590, 665, 720, 850")
    p.add_argument("--preset", default=None, help="a look: swap_rb, blue_sky_590, goldie, aerochrome_like, "
                   "mono_best, mono_red, candle_glow (default: the filter's own)")
    p.add_argument("--strength", type=float, default=1.0)

    p = sub.add_parser("match", parents=[inp, out, common], help=HELP["match"])
    p.add_argument("--to", dest="match", required=True, metavar="PICTURE_OR_PROFILE")
    p.add_argument("--amount", dest="match_amount", type=float, default=1.0)
    p.add_argument("--match-sidecar", action="store_true")
    p.add_argument("--recipe", action="append", default=None)
    p.add_argument("--set", action="append", default=None)
    p.add_argument("--save-recipe", default=None, metavar="FILE")

    p = sub.add_parser("look-profile", parents=[inp, common], help=HELP["look-profile"])
    p.add_argument("--save", required=True, metavar="FILE.json")
    p.add_argument("--name", default=None)
    p.add_argument("--use-sidecar", action="store_true")

    p = sub.add_parser("depth", parents=full, help=HELP["depth"])
    p.add_argument("--make", default="map", help="comma list of: map, view, normals, layers (default map)")
    p.add_argument("--layers", type=int, default=3, help="how many depth layers (2..12)")
    p.add_argument("--cuts", type=lambda s: [float(x) for x in s.split(",")], default=None,
                   help="where the layers divide, depth values 0..1 (near = 1), e.g. 0.35,0.7 (default: equal areas)")
    p.add_argument("--feather", type=float, default=0.02, help="softness of a layer's edge, in depth units")
    p.add_argument("--fill", action="store_true", help="make layer 1 a whole background plate (Fill behind the "
                   "nearer layers: invented picture)")
    p.add_argument("--fill-size", type=int, default=768, help="long edge the fill is worked out at")
    p.add_argument("--fill-method", default="auto", choices=["auto", "lama", "patchmatch"],
                   help="auto = LaMa for the layout when installed, PatchMatch for the texture")
    p.add_argument("--normals-y", default="up", choices=["up", "down"], help="green = up (OpenGL, Godot, Blender) "
                   "or down (DirectX)")

    p = sub.add_parser("cutout", parents=full, help=HELP["cutout"])
    p.add_argument("--person", nargs="?", const="person", default=None,
                   choices=["person", "background", "skin", "face", "hair", "clothes", "arms", "legs", "torso"])
    p.add_argument("--click", action="append", default=None, metavar="U,V", help="a point on the object (repeatable)")
    p.add_argument("--avoid", action="append", default=None, metavar="U,V", help="a point NOT on it (repeatable)")
    p.add_argument("--model", default="classic", choices=["classic", "tiny", "large"],
                   help="for --click: the Studio's own colour-and-edge method, or SAM 2.1")
    p.add_argument("--depth-range", default=None, metavar="LO,HI", help="by distance, as the Studio's depth mask")
    p.add_argument("--luminance", default=None, metavar="LO,HI", help="by screen brightness 0..1")
    p.add_argument("--colour", default=None, metavar="HUE,WIDTH", help="by colour: OKLCh hue and width in degrees")
    p.add_argument("--chroma", type=float, default=0.03, help="for --colour: least colourfulness 0..1")
    p.add_argument("--skin", action="store_true")
    p.add_argument("--shape", action="append", default=None, metavar="JSON", help="any mask shape (repeatable)")
    p.add_argument("--combine", default="intersect", choices=["add", "subtract", "intersect"],
                   help="how a second selection joins the first (default intersect)")
    p.add_argument("--soft", type=float, default=0.1, help="softness of a range's ends")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--feather", type=float, default=0.0, help="blur the edge, full-size pixels")
    p.add_argument("--hard", type=float, default=None, metavar="LEVEL", help="firm the edge up round this alpha "
                   "(e.g. 0.5): for leaves, fences and other see-through things")
    p.add_argument("--hard-width", type=float, default=0.2)
    p.add_argument("--make", default="rgba", choices=["rgba", "mask", "both"])

    p = sub.add_parser("find", parents=[inp, ed, common], help=HELP["find"])
    p.add_argument("--what", default="size,stats", help="comma list of: " + ", ".join(FINDERS))
    p.add_argument("--ratio", type=float, default=None, help="crops: width / height wanted")
    p.add_argument("--method", default=None, help="crops: a composition method")
    p.add_argument("--n", type=int, default=3, help="crops: how many")
    p.add_argument("--full", action="store_true", help="include bulky data (bitmaps, every face point)")

    p = sub.add_parser("stack", parents=[common], help=HELP["stack"])
    p.add_argument("kind", choices=["focus", "superres", "pano"])
    p.add_argument("inputs", nargs="+")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--out", default=None, metavar="FOLDER", help="default: beside the first frame")
    p.add_argument("--set", action="append", default=None, metavar="NAME=VALUE",
                   help="a setting of that tool ('schema' lists them under stack)")
    p.add_argument("--mode", default="handheld", help="superres: handheld, clean, moon or sky")
    p.add_argument("--factor", type=float, default=1.0, help="pano: build smaller (0.25 = quarter size)")
    p.add_argument("--make", default="merge", help="focus: comma list of merge (the linear result), depth (height, "
                   "confidence, agreement + JSON: fast) and surface (colour, height, normal, confidence, agreement "
                   "+ JSON, for games); default merge")
    _surface_args(p, stack=True)

    p = sub.add_parser("dualpixel", parents=[inp, common], help=HELP["dualpixel"])
    p.add_argument("--make", default="report,depth", help="comma list of: " + ", ".join(DP_MAKES))
    p.add_argument("--out", default=None, metavar="FOLDER")
    p.add_argument("--swap", action="store_true", help="the halves are the other way round (near / far swapped)")
    p.add_argument("--strength", type=float, default=1.0, help="hdr: highlight recovery strength")
    p.add_argument("--view-scale", type=float, default=0.5, help="anaglyph / sbs size")
    p.add_argument("--boost", type=float, default=1.0, help="wiggle: exaggerate the 3-D")
    _surface_args(p, stack=False)

    p = sub.add_parser("schema", parents=[common], help=HELP["schema"])
    p.add_argument("setting", nargs="?", default=None, help="one setting or group, e.g. light or light.shadows")
    sub.add_parser("looks", parents=[common], help=HELP["looks"])
    p = sub.add_parser("batch", parents=[common], help=HELP["batch"])
    p.add_argument("jobs", help="a JSON file of jobs ('-' reads standard input)")
    p.add_argument("--stop-on-error", action="store_true")


def execute(a, quiet: bool = False) -> Run:
    """Run one parsed command; every failure ends up in the result, never as a traceback on the screen."""
    run = Run(a, a.cmd, quiet)
    try:
        HANDLERS[a.cmd](a, run)
    except Usage as e:
        run.usage = True
        run.problem(str(e))
    except KeyboardInterrupt:
        raise
    except Exception as e:                                # noqa: BLE001
        run.problem(f"{a.cmd}: {e}", exc=e)
    run.finish()
    return run


def dispatch(a) -> int:
    if getattr(a, "json", False):
        console_to_stderr()
    run = execute(a)
    res = plain(run.res, True)
    if getattr(a, "result", None):
        os.makedirs(os.path.dirname(os.path.abspath(a.result)) or ".", exist_ok=True)
        tmp = a.result + ".partial"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        os.replace(tmp, a.result)
    if getattr(a, "json", False):
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()
    else:
        if a.cmd in DATA_COMMANDS and run.code != 2:
            print(json.dumps(res.get("data", res["outputs"]), indent=2))
        n = sum(1 for o in res["outputs"] if o.get("output"))
        print(f"Done: {n} file(s) written, {len(res['problems'])} problem(s), {res['seconds']} s.")
    return run.code
