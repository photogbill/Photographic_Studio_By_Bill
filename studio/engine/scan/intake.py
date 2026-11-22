"""One call per incoming folder (for the RPG Forge, and anyone): the Studio decides what each set of photos is for and
makes it, then lists every asset in one JSON (BUILD_PLAN_DP3D G; Bill, 2026-11-21: "build the rest of what I need to
allow forge to use it").

* Every focus stack, Dual Pixel photo and Focus Stack result (with its depth beside it) is a possible **view**. The
  camera solver is run on all of them together: views it places together (at least ``MIN_MODEL_VIEWS``) are one
  **object**, made into a model (glb with LODs, cameras, points, the Blender dataset behind its gate).
* Every view not in an object becomes a **texture** and a **front-only relief** (``relief``: its measured depth as a
  cut-out mesh, glb); every view not in an object becomes a **texture** (the measured surface and its game maps); every ordinary photo or
  burst becomes an **estimated texture** (amber: nothing measured); with ``leaves``, the stacks of leaves also become
  leaf cards.
* ``NAME_intake_Studio.json`` ("studio_intake": 1): the assets (kind, name, files, how they were made, quality), what
  was left out and why, and notes.

Nothing is written beside the photos except the output folder the caller names.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from ... import log as slog

log = slog.get("engine.scan.intake")

INTAKE_VERSION = 1
MIN_MODEL_VIEWS = 3


@dataclass
class IntakeParams:
    size: int = 2048                  # textures: the game size (square, power of two)
    tileable: bool = True
    leaves: bool = False
    triangles: int = 30000
    lods: bool = True
    affine: str = "auto"
    known_length_m: float = 0.0       # a model's longest side when the photographer knows it (else relative)
    reliefs: bool = True              # every view not in an object also becomes a front-only relief mesh
    relief_depth: float = 0.0         # the reliefs' depth against their width (0: estimated, half the width)
    min_model_views: int = MIN_MODEL_VIEWS

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


@dataclass
class Item:
    name: str
    kind: str                         # stack | dual | result | photo | burst
    paths: list
    base: str
    view: object = None
    note: str = ""
    group: int = -1
    assets: list = field(default_factory=list)


def _kind_of(path: str) -> str:
    """The kind a Studio file name says: NAME_<kind>_Studio.ext -> kind (with its extension when not a picture)."""
    stem, ext = os.path.splitext(os.path.basename(path))
    k = stem[:-len("_Studio")] if stem.endswith("_Studio") else stem
    k = k.rsplit("_", 1)[-1] if "_" in k else k
    return k if ext.lower() in (".png", ".tif", ".tiff") else f"{k}{ext.lower()}" if ext.lower() != ".json" else "json"


def find_items(folder: str) -> tuple[list, list]:
    """(items, left out) in a folder: Focus Stack results with their depth first (their frames, if also present, are
    not read again: the result is the stack), then raw stacks, Dual Pixel photos, ordinary photos and bursts."""
    from ..stack import batch as B
    from ..stack import focus as FS
    from ..stack.frames import result_base
    items, left = [], []
    names = sorted(os.listdir(folder))
    results = [os.path.join(folder, n) for n in names if os.path.splitext(n)[1].lower() in (".tif", ".tiff", ".png")
               and os.path.exists(FS.depth_sidecar_path(os.path.join(folder, n)))]
    for r in results:
        items.append(Item(os.path.basename(r), "result", [r], os.path.splitext(os.path.basename(r))[0]))
    photos = [os.path.join(folder, n) for n in names if os.path.splitext(n)[1].lower() in
              (".jpg", ".jpeg", ".png", ".tif", ".tiff") and os.path.join(folder, n) not in results
              and "_studio" not in n.lower()]
    for p in photos:
        items.append(Item(os.path.basename(p), "photo", [p], os.path.splitext(os.path.basename(p))[0]))
    raws = [n for n in names if os.path.splitext(n)[1].lower() in (".cr3", ".cr2", ".dng", ".nef", ".arw", ".raf",
                                                                    ".orf", ".rw2", ".pef")]
    if raws:
        for st in B.find_stacks([folder], "normal"):
            if st.kind == "focus" and len(st.paths) >= 2:
                items.append(Item(st.name, "stack", list(st.paths), st.base or result_base(st.paths)))
            elif len(st.paths) == 1 and st.dual_pixel:
                items.append(Item(st.name, "dual", list(st.paths), st.base or result_base(st.paths)))
            elif len(st.paths) == 1:
                items.append(Item(st.name, "photo", list(st.paths), st.base or result_base(st.paths)))
            elif st.kind == "burst":
                items.append(Item(st.name, "burst", list(st.paths), st.base or result_base(st.paths), note=st.best or ""))
            else:
                left.append({"name": st.name, "reason": st.note or f"a {st.kind} set: not a texture or a view"})
    return items, left


def _read_view(it: Item, fp, progress, cancel):
    from . import views as VW
    if it.kind == "result":
        return VW.from_result(it.paths[0])
    if it.kind == "dual":
        return VW.from_dual(it.paths[0])
    if it.kind == "stack":
        return VW.from_stack(it.paths, it.base, fp, None, True, progress, cancel)
    return None


def result_surface(path: str, out_dir: str, base: str, p, kinds, taken):
    """The surface of a Focus Stack result from its picture and the NAME.depth.npz beside it (the depth as Focus
    Stack measured it: 0..1, near = high, with its confidence and how-measured label)."""
    from PIL import Image as PILImage
    from ..stack import depthfuse as DF
    from ..stack import focus as FS
    from ..stack import surface as SURF
    z = np.load(FS.depth_sidecar_path(path), allow_pickle=False)
    d = np.asarray(z["depth"], np.float32)
    c = np.asarray(z["confidence"], np.float32)
    lab = np.asarray(z["label"], np.uint8)
    rng = float(z["range_steps"]) if "range_steps" in z.files else 1.0
    res = DF.DepthResult(depth=d * rng, confidence=c, label=lab, agreement=np.full(d.shape, np.nan, np.float32),
                         focus_depth=None, focus_quality=None, dp_depth=None, method="focus+dual_pixel",
                         units="steps")
    colour = np.asarray(PILImage.open(path).convert("RGB"))
    meta = {}
    try:
        from ..io.metadata import read_metadata
        md = read_metadata(path)
        meta = {"make": md.make, "model": md.model, "lens": md.lens, "focal_mm": md.focal_mm, "fnumber": md.fnumber,
                "focus_distance_m": getattr(md, "focus_distance_m", None)}
    except Exception:                                # noqa: BLE001 - no EXIF: relative scale
        pass
    source = {"kind": "Focus Stack result", "frames": [os.path.basename(path)], "frames_used": 1,
              "reference_frame": os.path.basename(path), "dual_pixel": True, "left_out": []}
    # the sidecar is upright already, near = high; full_maps upsamples from half size: give it the picture's size
    return SURF.write(out_dir, base, res, colour.shape[:2], 0, meta, source, p, colour, kinds, taken,
                      near_known="as Focus Stack measured it (its depth file keeps near = high)")


def run(folder: str, out_dir: str, p: IntakeParams | None = None, progress=None, cancel=None,
        taken: set | None = None, focus_params=None) -> dict:
    """Process one folder. Returns the intake document (also written as NAME_intake_Studio.json)."""
    from ... import __version__
    from . import model as MD
    from . import solve as SV
    from ..stack import batch as B
    from ..stack import focus as FS
    from ..stack import surface as SURF
    p = p or IntakeParams()
    fp = focus_params or FS.FocusParams()
    taken = set() if taken is None else taken

    def prog(f, msg):
        if progress:
            progress(float(f), msg)

    def check():
        if cancel is not None and cancel():
            raise InterruptedError("cancelled")
    items, left = find_items(folder)
    if not items:
        raise ValueError("nothing to make: no photos, stacks or Focus Stack results in the folder")
    os.makedirs(out_dir, exist_ok=True)
    notes = []
    # 1. views
    cand = [it for it in items if it.kind in ("result", "dual", "stack")]
    for k, it in enumerate(cand):
        check()
        prog(0.5 * k / max(len(cand), 1), f"reading {it.name}")
        try:
            it.view = _read_view(it, fp, lambda f, m: prog(0.5 * (k + f) / max(len(cand), 1), m), cancel)
        except InterruptedError:
            raise
        except Exception as exc:                     # noqa: BLE001 - one bad set: the others still go on
            it.note = f"could not be read: {exc}"
            from ... import log as L
            L.save_error_report(f"intake: reading {it.name}")
    views = [it for it in cand if it.view is not None]
    assets = []
    # 2. objects: views the solver places together
    groups = {}
    sol = None
    if len(views) >= p.min_model_views:
        check()
        prog(0.5, "placing the views (are some of them one object?)")
        try:
            sol = SV.solve([it.view for it in views], affine=p.affine, cancel=cancel,
                           progress=lambda f, m: prog(0.5 + 0.15 * f, m))
            for i, g in enumerate(sol.group):
                if g >= 0 and sol.placed[i]:
                    groups.setdefault(g, []).append(i)
            groups = {g: m for g, m in groups.items() if len(m) >= p.min_model_views}
            if not groups:
                notes.append("No object was found among the views (each is made as a texture): "
                             + SV.why_nothing_placed(sol) + ".")
        except InterruptedError:
            raise
        except Exception as exc:                     # noqa: BLE001 - no objects found: every view stays a texture
            from ... import log as L
            L.save_error_report("intake: placing the views")
            notes.append(f"The views could not be placed together ({exc}); each is made as a texture.")
    for gi, (g, members) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        check()
        name = f"{os.path.basename(os.path.normpath(folder))}_object{gi + 1}" if len(groups) > 1 else \
            os.path.basename(os.path.normpath(folder))
        prog(0.65, f"making the model of {name} ({len(members)} views)")
        mp = MD.ModelParams(triangles=p.triangles, lods=p.lods, affine=p.affine, known_length_m=p.known_length_m)
        try:
            r = MD.build([it.view for it in views], out_dir, name, mp, MD.MAKES, None, cancel, taken, solution=sol,
                         group=g)
        except InterruptedError:
            raise
        except Exception as exc:                     # noqa: BLE001
            from ... import log as L
            L.save_error_report(f"intake: model {name}")
            notes.append(f"The model of {len(members)} views could not be made ({exc}); they are made as textures.")
            continue
        for i in members:
            views[i].group = gi
        d = r["json"]
        assets.append({"kind": "model", "name": name, "files": {k: os.path.basename(v) for k, v in r["paths"].items()},
                       "json": os.path.basename(r["paths"]["json"]), "views": [views[i].name for i in members],
                       "units": d["units"], "scale": d["scale"], "quality": d["quality"], "gate": d["gate"],
                       "colour": "cyan" if d["quality"]["measured_share"] >= 0.5 else "amber",
                       "coach": d["coach"]["lines"]})
    # 3. textures: everything not in an object
    sp = SURF.SurfaceParams(maps="all", tileable=p.tileable, size=int(p.size))
    kinds = SURF.kinds_from(sp)
    tex_items = [it for it in items if it.group < 0 and it.kind in ("result", "dual", "stack", "photo", "burst")]
    for k, it in enumerate(tex_items):
        check()
        prog(0.75 + 0.2 * k / max(len(tex_items), 1), f"texture from {it.name}")
        try:
            if it.kind == "result":
                r = result_surface(it.paths[0], out_dir, it.base, sp, kinds, taken)
            else:
                # stacks, Dual Pixel photos, ordinary photos and bursts: the same path as 'stacks --run surface'
                si = B.StackInfo(it.name, "burst" if it.kind == "burst" else ("focus" if it.kind == "stack" else
                                                                              "single"), list(it.paths),
                                 dual_pixel=(len(it.paths) if it.kind in ("stack", "dual") else 0),
                                 best=(it.note or None) if it.kind == "burst" else None)
                paths_, _ = B._one(si, "surface", out_dir, sp, fp, taken, cancel, None)
                jp = next(q for q in paths_ if q.endswith("_surface_Studio.json"))
                r = {"paths": {_kind_of(q): q for q in paths_}, "json": json.load(open(jp, encoding="utf-8"))}
        except InterruptedError:
            raise
        except Exception as exc:                     # noqa: BLE001
            from ... import log as L
            L.save_error_report(f"intake: texture {it.name}")
            left.append({"name": it.name, "reason": f"the texture could not be made ({exc})"})
            continue
        doc = r["json"]
        measured = doc.get("method") != "estimated"
        assets.append({"kind": "texture", "name": it.base, "from": it.name,
                       "files": {k_: os.path.basename(v) for k_, v in r["paths"].items()},
                       "json": os.path.basename(r["paths"]["json"]), "method": doc.get("method"),
                       "measured": measured, "colour": "cyan" if measured else "amber",
                       "scale": doc.get("scale"), "coverage": doc.get("coverage"),
                       "about": "measured" if measured else "estimated from one ordinary photo (amber)"})
    # 4. front-only reliefs: every view that did not join an object (Bill: "simplifies the job in blender")
    if p.reliefs:
        from . import relief as RL
        for it in items:
            if it.view is None or it.group >= 0:
                continue
            check()
            try:
                r = RL.build(it.view, out_dir, it.base, p.relief_depth or None, taken=taken)
            except InterruptedError:
                raise
            except ValueError as exc:                # too little depth: not a fault
                left.append({"name": it.name, "reason": f"no relief ({exc})"})
                continue
            except Exception:                        # noqa: BLE001
                from ... import log as L
                L.save_error_report(f"intake: relief {it.name}")
                left.append({"name": it.name, "reason": "the relief could not be made (an error report was written)"})
                continue
            d = r["json"]
            assets.append({"kind": "relief", "name": it.base, "from": it.name,
                           "files": {k_: os.path.basename(v) for k_, v in r["paths"].items()},
                           "json": os.path.basename(r["paths"]["json"]), "depth": d["depth"],
                           "measured_share": d["measured_share"], "colour": d["depth"]["colour"],
                           "about": d["what"]})
    # 5. leaves
    if p.leaves:
        for it in items:
            if it.kind not in ("stack", "dual") or it.group >= 0:
                continue
            check()
            si = B.StackInfo(it.name, "focus" if it.kind == "stack" else "single", list(it.paths),
                             dual_pixel=len(it.paths))
            try:
                paths_, _ = B._one(si, "foliage", out_dir, sp, fp, taken, cancel, None)
            except InterruptedError:
                raise
            except Exception as exc:                 # noqa: BLE001 - no leaves found is not a fault
                left.append({"name": it.name, "reason": f"no leaf cards ({exc})"})
                continue
            assets.append({"kind": "foliage", "name": it.base, "from": it.name,
                           "files": {_kind_of(q): os.path.basename(q) for q in paths_}})
    for it in items:
        if it.note.startswith("could not"):
            left.append({"name": it.name, "reason": it.note})
    name = os.path.basename(os.path.normpath(folder)) or "intake"
    doc = {"studio_intake": INTAKE_VERSION, "written_by": f"Photographic Studio By Bill {__version__}",
           "folder": os.path.abspath(folder), "assets": assets, "left_out": left,
           "summary": {"models": sum(a["kind"] == "model" for a in assets),
                       "textures_measured": sum(a["kind"] == "texture" and a["measured"] for a in assets),
                       "textures_estimated": sum(a["kind"] == "texture" and not a["measured"] for a in assets),
                       "foliage": sum(a["kind"] == "foliage" for a in assets),
                       "reliefs": sum(a["kind"] == "relief" for a in assets)},
           "colours": {"cyan": "measured", "amber": "estimated or measured one way", "violet": "filled in",
                       "red": "the two measures disagree"},
           "settings": p.to_dict(), "notes": notes}
    path = os.path.join(out_dir, f"{name}_intake_Studio.json")
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(MD._plain(doc), fh, indent=1)
    os.replace(tmp, path)
    doc["path"] = path
    prog(1.0, "intake done")
    return doc
