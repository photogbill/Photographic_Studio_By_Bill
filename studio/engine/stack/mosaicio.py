"""Flat Mosaic, in and out (BUILD_PLAN_MOSAIC.md sections 4 and 5.2): what goes in, what comes out, and the coach.

In: any mix of Focus Stack results (with their measured depth beside them), raw focus stacks (one per spot: each is
merged into one all-in-focus picture with its fused depth, Card to Stacks' own grouping), single Dual Pixel photos,
and ordinary photos or raws (colour only). Bill's way (2026-11-23): close and wide open, a focus stack at each spot,
moving about 40% of a frame between spots.

Out: the surface files written by surface.py (colour, height, normal, confidence, agreement and any other map asked
for, tileable when asked), so the RPG Forge's intake takes them unchanged, plus:
    NAME_misfit_Studio.png     where the views still disagree in position (8-bit: 255 = MISFIT_FULL_MM or px)
    NAME_coverage_Studio.png   each view's footprint seen from above, coloured by its ground detail; gaps in red
    NAME_pieceNN_<map>_Studio.png   boxes cut out as decals (colour with a soft transparent edge, height, normal)
and the surface JSON gains a "mosaic" block: the views, how the size and the relief are known, each camera's height
and tilt, drift, the coach's advice, and ``repeat_m`` (the ground one copy of the texture covers).

No Qt here.
"""
from __future__ import annotations

import math
import os

import numpy as np

from ... import log as slog
from . import mosaic as MO
from . import mosaicbuild as MB

log = slog.get("engine.stack.mosaicio")

MISFIT_FULL_PX = 4.0             # misfit map: white at this many output pixels of disagreement
SHOOTING_CARD = [
    "Point the camera straight down, at roughly the same height each time.",
    "A 35-50 mm lens. For the most depth: close (about 0.6-0.8 m) and wide open, with a focus stack at each spot.",
    "Overlap each spot with the last by about 60% (move about 40% of a frame).",
    "Walk a second row, or back along the strip, so drift can be checked.",
    "Overcast light or full shade: no shadows of you or the camera.",
    "A polarizer if the surface has glare.",
    "Dual Pixel RAW on.",
    "Optional: a ruler or tape in the first frame (mark it for an exact size).",
]
WHAT = ("Stitches hand-held views of a flat or gently curved surface (a road, a path, a floor, a wall) into one "
        "texture, with its height: each camera's position found, the views' depth matched, the relief measured.")
HONEST = ("Things that stand well above the ground (a car, a post) are not flat and will smear. A place only two views "
          "saw cannot be decided by majority if something moved there. Without overlapping relief, or with plane "
          "warps, the height stays relative. The size from focus distances is amber: a ruler makes it exact.")


KIND_INFO = {
    "auto": {"name": "Auto", "does": "Looks at how neighbouring frames join: turned on one spot is a panorama, "
             "moved over a surface is a flat mosaic. Focus stacks and Dual Pixel views are always a mosaic.",
             "best": "Most sets: it says which it chose.", "weak": "Two frames are too few to tell: pick the kind."},
    "panorama": {"name": "Panorama: turn on one spot", "does": "Frames shot turning the camera around its lens.",
                 "best": "Landscapes, rooms, the sky.", "weak": "A camera that moved between frames (parallax)."},
    "mosaic": {"name": "Flat mosaic: move over a surface", "does": "Hand-held views of a flat or gently curved "
               "surface, each camera's height, tilt and turn found; with depth, a height mosaic too.",
               "best": "Roads, paths, floors, walls: game textures that do not repeat for metres.",
               "weak": "Things standing well above the ground (a car, a post) smear."},
}


def auto_kind(paths: list) -> dict:
    """Panorama or flat mosaic for the page's Auto (plan 5.1). Focus stacks, Focus Stack results and Dual Pixel views
    make a mosaic; otherwise neighbouring frames are compared (mosaic.kind_of)."""
    from scipy.ndimage import zoom
    from .sorter import proxy_grey
    entries, _ = find_inputs(paths)
    if any(e["kind"] in ("stack", "result", "dual") for e in entries):
        return {"kind": "mosaic", "text": "Focus stacks or Dual Pixel views: using Flat mosaic."}
    firsts = [e["paths"][0] for e in entries][:6]
    if len(firsts) < 2:
        return {"kind": "panorama", "text": "Too few frames to tell: using Panorama."}
    greys = []
    for p in firsts:
        try:
            g = np.asarray(proxy_grey(p), np.float32)
        except Exception as exc:                     # noqa: BLE001 - an unreadable frame: leave it out of the test
            if not slog.expected_problem(exc):
                slog.save_error_report("Mosaic: telling panorama from mosaic", exc)
            continue
        if greys and g.shape != greys[0].shape:
            g = zoom(g, (greys[0].shape[0] / g.shape[0], greys[0].shape[1] / g.shape[1]), order=1)[
                :greys[0].shape[0], :greys[0].shape[1]]
        greys.append(g)
    if len(greys) < 2:
        return {"kind": "panorama", "text": "The frames could not be compared: using Panorama."}
    return MO.kind_of(greys)


# ------------------------------------------------------------------------------------------- in
PICTURES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def _all_files(items) -> list:
    out = []
    for it in items:
        it = os.fspath(it)
        if os.path.isdir(it):
            for root, dirs, files in os.walk(it):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                out += [os.path.join(root, n) for n in sorted(files)]
        elif os.path.isfile(it):
            out.append(it)
    return out


def find_inputs(items) -> tuple:
    """(entries, left out) from files and folders: [{"name", "kind" (result | stack | dual | photo), "paths"}].
    Focus Stack results (a picture with NAME.depth.npz beside it, usually NAME_Studio.tif) come first."""
    from . import batch as B
    from . import focus as FS
    from .sorter import find_paths
    entries, left = [], []
    allf = _all_files(items)
    results = [p for p in allf if os.path.splitext(p)[1].lower() in PICTURES
               and os.path.exists(FS.depth_sidecar_path(p))]
    for r in results:
        entries.append({"name": os.path.splitext(os.path.basename(r))[0], "kind": "result", "paths": [r]})
    paths = [p for p in find_paths(items) if p not in results]
    pics = [p for p in paths if os.path.splitext(p)[1].lower() in PICTURES]
    for p in pics:
        entries.append({"name": os.path.splitext(os.path.basename(p))[0], "kind": "photo", "paths": [p]})
    raws = [p for p in paths if p not in pics]
    if raws:
        for st in B.find_stacks(raws, "normal"):
            if st.kind == "focus" and len(st.paths) >= 2:
                entries.append({"name": st.name, "kind": "stack", "paths": list(st.paths)})
            elif len(st.paths) == 1 and st.dual_pixel:
                entries.append({"name": st.name, "kind": "dual", "paths": list(st.paths)})
            elif len(st.paths) == 1:
                entries.append({"name": st.name, "kind": "photo", "paths": list(st.paths)})
            elif st.kind == "burst":
                best = st.best or st.paths[0]
                entries.append({"name": st.name, "kind": "photo", "paths": [best]})
            else:
                left.append({"name": st.name, "reason": st.note or f"a {st.kind} set"})
    return entries, left


def view_from_photo(path: str):
    """A view from an ordinary photo or raw: its colour with the default look, no depth."""
    from ..pipeline.edit import Edit
    from ..pipeline.pipeline import open_image, render
    from ..scan import views as VW
    from .surface import srgb8
    img = open_image(path)
    try:
        colour = srgb8(render(img, Edit()))
        md = img.meta
        meta = {"make": md.make, "model": md.model, "lens": md.lens, "focal_mm": md.focal_mm, "fnumber": md.fnumber,
                "focus_distance_m": getattr(md, "focus_distance_m", None)}
    finally:
        try:
            img.drop_cache()
        except Exception:                            # noqa: BLE001
            pass
    h, w = colour.shape[:2]
    z = np.zeros((h, w), np.float32)
    return VW.from_arrays(os.path.splitext(os.path.basename(path))[0], colour, z, z, None, meta.get("focal_mm"),
                          str(meta.get("model") or ""), path, meta)


def read_views(entries: list, focus_params=None, progress=None, cancel=None) -> tuple:
    """(views, problems {name: why}). One bad entry is named and left out; the others go on."""
    from ..scan import views as VW
    from .focus import Cancelled
    say = progress or (lambda *_: None)
    views, problems = [], {}
    for k, e in enumerate(entries):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        say(k / max(len(entries), 1), f"Reading {e['name']} ({k + 1} of {len(entries)})")
        try:
            if e["kind"] == "result":
                v = VW.from_result(e["paths"][0])
            elif e["kind"] == "dual":
                v = VW.from_dual(e["paths"][0])
            elif e["kind"] == "stack":
                v = VW.from_stack(e["paths"], e["name"], focus_params, None, True,
                                  lambda f, m: say((k + f) / max(len(entries), 1), m), cancel)
            else:
                v = view_from_photo(e["paths"][0])
            v.name = e["name"]
            v.meta = dict(v.meta or {}, kind=e["kind"])
            views.append(v)
        except Cancelled:
            raise
        except Exception as exc:                     # noqa: BLE001 - one bad entry: named, the rest go on
            if not slog.expected_problem(exc):
                slog.save_error_report(f"Mosaic: reading {e['name']}", exc)
            problems[e["name"]] = f"could not be read ({str(exc)[:80]})"
    return views, problems


# ------------------------------------------------------------------------------------------- the coach
def coach(sol: MO.Solution, views: list, sp: dict | None = None) -> list:
    """Plain sentences about how the set was shot and what to do next time (plan 5.2)."""
    out = []
    shapes = [v.shape for v in views]
    names = [v.name for v in views]
    ov = MO.footprint_overlaps(sol, shapes)
    for i in range(sol.n - 1):
        if not (sol.placed[i] and sol.placed[i + 1]):
            continue
        s = ov.get((i, i + 1), 0.0)
        if s < 0.35:
            out.append(f"{names[i]} and {names[i + 1]} overlap {100 * s:.0f}%; aim for about 60%.")
    for i, why in sol.problems.items():
        out.append(f"{names[i]} was left out: {why}.")
    if sol.method == "poses":
        hs = [(i, sol.cam_height(i)) for i in range(sol.n) if sol.placed[i]]
        med = float(np.median([h for _, h in hs]))
        mpu = sol.m_per_unit
        for i, h in hs:
            if h > 1.5 * med or h < med / 1.5:
                if mpu:
                    out.append(f"{names[i]} was shot at {h * mpu:.2f} m, the rest near {med * mpu:.2f} m; it is used, "
                               "but its detail is " + ("coarser." if h > med else "finer."))
                else:
                    out.append(f"{names[i]} was shot {h / med:.1f} times as high as the others; it is used, but its "
                               "detail is " + ("coarser." if h > med else "finer."))
            t = MO.tilt_deg(sol.R[i])
            if t > 20:
                out.append(f"{names[i]} was tilted {t:.0f} degrees; straight down keeps the detail even.")
    out.append(sol.drift.get("text", ""))
    if sol.scale_spread and sol.scale_spread > 0.08:
        out.append(f"The focus distances disagree by about {100 * sol.scale_spread:.0f}% between views; a ruler in "
                   "one frame would make the size exact.")
    if sp is not None:
        out += [n for n in sp.get("notes", [])]
        hm = sp.get("hm")
        if hm is not None and not hm.alpha:
            out.append("The relief's size could not be measured (" + hm.how.split(": ", 1)[-1] + "): the height is "
                       "relative. More overlap at the bumps and cracks, or a little more height between views, "
                       "would measure it.")
        elif hm is None and not any(MB.has_height(v) for v in views):
            out.append("No view had measured depth: Dual Pixel RAW or a focus stack at each spot gives the height.")
    return [s for s in out if s]


# ------------------------------------------------------------------------------------------- out
def _depth_result(r: dict, sp: dict):
    """The built mosaic's height as a depthfuse.DepthResult for surface.write (near = high: up from the ground)."""
    from . import depthfuse as DF
    hm = sp["hm"]
    H, W = r["rgb"].shape[:2]
    if r["height"] is None or hm is None:
        z = np.zeros((H, W), np.float32)
        return DF.DepthResult(depth=z, confidence=z, label=np.zeros((H, W), np.uint8),
                              agreement=np.full((H, W), np.nan, np.float32), focus_depth=None, focus_quality=None,
                              dp_depth=None, method="mosaic", units="relative"), None
    h = np.where(np.isfinite(r["height"]), r["height"], 0.0).astype(np.float32)
    c = np.clip(np.nan_to_num(r["hconf"]), 0, 1).astype(np.float32)
    c[~r["valid"]] = 0
    measured = c > 0.3
    lab = np.where(measured, DF.LABEL_BOTH if hm.alpha else DF.LABEL_ONE, DF.LABEL_FILLED).astype(np.uint8)
    lo, hi = np.percentile(h[measured], [1, 99]) if measured.any() else (0.0, 1.0)
    red = max(0.1 * float(hi - lo), 1e-9)
    if hm.alpha and sp.get("m_per_unit"):
        red = max(red, 0.0005 / sp["m_per_unit"])    # never under half a millimetre
    ag = r.get("agreement")
    if ag is not None:
        ag = (ag / red * DF.DISAGREE_STEPS).astype(np.float32)
    else:
        ag = np.full((H, W), np.nan, np.float32)
    res = DF.DepthResult(depth=h, confidence=c, label=lab, agreement=ag, focus_depth=None, focus_quality=None,
                         dp_depth=None, method="mosaic",
                         units="the mosaic's ground units (up from the fitted ground)" if hm.alpha else "relative")
    return res, red


def _png(path, arr, meta=None):
    from ..io.export import write_png_any
    write_png_any(path, arr, None, meta or {})


def coverage_image(sol: MO.Solution, views: list, sp: dict, crop_plan) -> np.ndarray:
    """RGBA, plan size: each view's footprint outlined and tinted by its ground detail (cyan finest .. amber
    coarsest), places no view covers inside the set's outline in red."""
    from PIL import Image, ImageDraw
    cv = sp["canvas"]
    y0, y1, x0, x1 = crop_plan
    img = Image.new("RGBA", (cv.W, cv.H), (16, 16, 20, 255))
    d = ImageDraw.Draw(img, "RGBA")
    g = {i: MB.native_gsd(sol, views, i) for i in sp["use"]}
    gmin, gmax = min(g.values()), max(g.values())
    cyan, amber = np.array([34, 211, 238]), np.array([245, 158, 11])
    for i in sp["use"]:
        fp = sol.footprint(i, views[i].shape, 12)
        u, v = cv.from_ground(fp[:, 0], fp[:, 1])
        t = 0.0 if gmax <= gmin else math.log(g[i] / gmin) / math.log(gmax / gmin)
        col = tuple(int(x) for x in (cyan * (1 - t) + amber * t))
        d.polygon(list(zip(u.tolist(), v.tolist())), fill=col + (40,), outline=col + (255,))
    a = np.array(img)
    valid = sp["valid"]
    from scipy.ndimage import binary_closing, binary_fill_holes
    hull = binary_fill_holes(binary_closing(valid, iterations=6))
    gap = hull & ~valid
    a[gap] = (248, 113, 113, 255)
    return a[y0:y1, x0:x1]


def cut_pieces(paths: dict, boxes: str, base_path: str, feather: float = 0.06) -> list:
    """Boxes "x0,y0,x1,y1;..." (0..1 of the result) cut out of the written maps as decals. Returns the files."""
    from PIL import Image
    out = []
    bx = []
    for part in [b for b in str(boxes or "").split(";") if b.strip()]:
        v = [float(x) for x in part.replace(" ", "").split(",")]
        if len(v) != 4 or not (0 <= v[0] < v[2] <= 1 and 0 <= v[1] < v[3] <= 1):
            raise ValueError("a piece is x0,y0,x1,y1 with 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
        bx.append(v)
    for k, (a0, b0, a1, b1) in enumerate(bx, 1):
        for kind in ("colour", "height", "normal"):
            src = paths.get(kind)
            if not src or not os.path.exists(src):
                continue
            im = np.array(Image.open(src))
            H, W = im.shape[:2]
            x0, x1 = int(a0 * W), max(int(a0 * W) + 1, int(round(a1 * W)))
            y0, y1 = int(b0 * H), max(int(b0 * H) + 1, int(round(b1 * H)))
            part = im[y0:y1, x0:x1]
            if kind == "colour":
                hh, ww = part.shape[:2]
                yy = np.minimum(np.arange(hh), np.arange(hh)[::-1]) / max(feather * hh, 1)
                xx = np.minimum(np.arange(ww), np.arange(ww)[::-1]) / max(feather * ww, 1)
                al = np.clip(np.minimum(yy[:, None], xx[None, :]), 0, 1)
                part = np.dstack([part[..., :3], np.rint(al * 255).astype(np.uint8)])
            p = f"{base_path}_piece{k:02d}_{kind}_Studio.png"
            _png(p, part)
            out.append(p)
    return out


def write(r: dict, sol: MO.Solution, views: list, params: MO.MosaicParams, out_dir: str, base: str,
          taken: set | None = None, problems: dict | None = None) -> dict:
    """The surface files, the misfit and coverage maps, any pieces, and the JSON's mosaic block."""
    from . import surface as SURF
    sp = r["plan"]
    sp["m_per_unit"] = sol.m_per_unit
    hm = sp["hm"]
    res, red = _depth_result(r, sp)
    mm_px = sp["mm_per_px"]
    p = SURF.SurfaceParams(tileable=bool(params.tileable), tile_method=params.tile_method,
                           normals_y=params.normals_y, maps=params.maps, size=int(params.size or 0))
    kinds = SURF.kinds_from(p)
    if r["height"] is None:
        kinds = tuple(k for k in kinds if k in ("colour",))
    colour = MB.linear_to_srgb8(r["rgb"])
    meta = {}
    for v in views:
        if v.meta:
            meta = {k: v.meta.get(k) for k in ("make", "model", "lens", "fnumber", "focal_mm", "iso")}
            break
    lateral = {"mm_per_px": mm_px, "how": sp["res_how"] + "; size " + sol.scale_how,
               "from": "the mosaic's own geometry",
               "info": {"colour": sol.scale_colour, "spread": sol.scale_spread}}
    unit_mm = (sol.m_per_unit * 1000.0) if (hm is not None and hm.alpha and sol.m_per_unit) else None
    hhow = None
    if hm is not None:
        hhow = hm.how + ("" if unit_mm or not hm.alpha else "; the size of the ground is not known, so the height is "
                         "in the mosaic's own units")
    used = [views[i].name for i in sp["use"]]
    tips = coach(sol, views, sp)
    mos = {
        "kind": "flat mosaic",
        "what": WHAT, "honest": HONEST,
        "method": dict(MO.METHOD_INFO[sol.method], key=sol.method),
        "surface": dict(MO.SURFACE_INFO[sol.surface], key=sol.surface),
        "solve": sol.summary(),
        "views": [{"name": views[i].name, "kind": (views[i].meta or {}).get("kind"),
                   "height_m": (round(sol.cam_height(i) * sol.m_per_unit, 4) if sol.cam_height(i) is not None and
                                sol.m_per_unit else None),
                   "height_units": (round(sol.cam_height(i), 5) if sol.cam_height(i) is not None else None),
                   "tilt_deg": (round(MO.tilt_deg(sol.R[i]), 2) if sol.method == "poses" and sol.R[i] is not None
                                else None),
                   "ground_mm_per_px": (round(MB.native_gsd(sol, views, i) * sol.m_per_unit * 1000, 4)
                                        if sol.m_per_unit and sol.placed[i] else None),
                   "used": i in sp["use"]} for i in range(sol.n)],
        "left_out": dict(problems or {}, **{views[i].name: w for i, w in sol.problems.items()}),
        "resolution": {"mode": params.resolution, "mm_per_px": mm_px, "how": sp["res_how"]},
        "relief": hm.to_dict() if hm is not None else None,
        "orthorectified": bool(sp["ortho"]),
        "nudged": bool(sp["nudges"]),
        "agreement_red_line": ({"units": "mm" if unit_mm else "height units",
                                "value": round(red * (unit_mm or 1.0), 5)} if red else None),
        "exposure": {"vignette_corner_ev": sp["corner_ev"], "evened_out": bool(sp["level"])},
        "coach": tips,
        "shooting_card": SHOOTING_CARD,
        "settings": params.to_dict(),
    }
    source = {"kind": "flat mosaic", "frames": used, "frames_used": len(used),
              "reference_frame": views[sol.ref].name, "dual_pixel": any(MB.has_height(v) for v in views),
              "left_out": list(mos["left_out"])}
    shape = colour.shape[:2]
    wr = SURF.write(out_dir, base, res, shape, 0, meta, source, p, colour, kinds, taken,
                    near_known="measured: up from the fitted ground", lateral=lateral, height_unit_mm=unit_mm,
                    height_how=hhow, extra={"mosaic": mos})
    paths = dict(wr["paths"])
    jpath = paths.get("json")
    b = jpath[:-len("_surface_Studio.json")] if jpath else os.path.join(out_dir, base)
    # the misfit and coverage maps beside them (same name, so the set stays a set)
    mf = r.get("misfit")
    k = sp["k"]
    if mf is not None:
        px = np.nan_to_num(mf) / max(k, 1e-9)             # plan pixels -> output pixels
        _png(f"{b}_misfit_Studio.png", np.rint(np.clip(px / MISFIT_FULL_PX, 0, 1) * 255).astype(np.uint8))
        paths["misfit"] = f"{b}_misfit_Studio.png"
    try:
        cy0, cy1, cx0, cx1 = r["crop"]
        crop_plan = (int(cy0 * k), max(int(cy0 * k) + 1, int(math.ceil(cy1 * k))),
                     int(cx0 * k), max(int(cx0 * k) + 1, int(math.ceil(cx1 * k))))
        _png(f"{b}_coverage_Studio.png", coverage_image(sol, views, sp, crop_plan))
        paths["coverage"] = f"{b}_coverage_Studio.png"
    except Exception as exc:                          # noqa: BLE001 - the coverage map is advice; the rest stands
        slog.save_error_report("Mosaic: the coverage map", exc)
    if params.pieces:
        paths["pieces"] = cut_pieces(paths, params.pieces, b)
    doc = wr["json"]
    doc["mosaic"]["files"] = {"misfit": os.path.basename(paths.get("misfit", "")) or None,
                              "coverage": os.path.basename(paths.get("coverage", "")) or None,
                              "pieces": [os.path.basename(x) for x in paths.get("pieces", [])],
                              "misfit_png": f"255 = {MISFIT_FULL_PX:g} output pixels of disagreement"}
    if jpath:
        import json
        tmp = jpath + ".partial"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(SURF._plain(doc), fh, indent=2)
        os.replace(tmp, jpath)
    return {"paths": paths, "json": doc, "coach": tips}


# ------------------------------------------------------------------------------------------- one run
def run(items, out_dir: str, base: str | None = None, params: MO.MosaicParams | None = None, progress=None,
        cancel=None, taken: set | None = None, focus_params=None) -> dict:
    """Files or folders in, the mosaic's files out (the command line's and the page's one call)."""
    from .frames import result_base
    params = params or MO.MosaicParams()
    say = progress or (lambda *_: None)
    entries, left = find_inputs(items)
    if len(entries) < 2:
        raise MO.MosaicProblem("A mosaic needs at least two views (photos, Dual Pixel RAWs or focus stacks).")
    views, problems = read_views(entries, focus_params, lambda f, m: say(0.25 * f, m), cancel)
    for e in left:
        problems[e["name"]] = e["reason"]
    if len(views) < 2:
        raise MO.MosaicProblem("Fewer than two views could be read.")
    sol = MO.solve(views, params, lambda f, m: say(0.25 + 0.2 * f, m), cancel)
    r = MB.build(views, sol, params, None, lambda f, m: say(0.45 + 0.5 * f, m), cancel)
    if base is None:
        allp = [p for e in entries for p in e["paths"]]
        base = result_base(allp) + "_Mosaic" if allp else "Mosaic"
    os.makedirs(out_dir, exist_ok=True)
    say(0.96, "Writing the surface")
    w = write(r, sol, views, params, out_dir, base, taken, problems)
    say(1.0, "Done")
    return {"paths": w["paths"], "json": w["json"], "coach": w["coach"], "size": r["rgb"].shape[1::-1],
            "views": len(views), "placed": int(sum(sol.placed))}
